//! Qwen3.6 vision encoder (`Qwen3_5VisionModel`, `model_type` `qwen3_5_vision`) — story sc-7633.
//!
//! The Qwen-VL ViT tower the multimodal `qwen3_5` checkpoint carries under `model.visual.*`. It turns
//! preprocessed image/video patches into decoder-width embeddings the text model splices in at the
//! image-token positions (the splice + M-RoPE land in a later slice; this slice is the encoder).
//!
//! Structurally it is a SigLIP-style pre-norm ViT (see [`super::siglip`]) — 27 blocks, bias-ful
//! attention + `gelu_pytorch_tanh` MLP — with four Qwen-VL specifics:
//!
//! - **Patch embed** is a `Conv3d` over `[temporal_patch_size, patch_size, patch_size]` with kernel ==
//!   stride == the patch, i.e. a per-patch linear projection — loaded as a reshaped `[hidden, C·T·P·P]`
//!   matmul ([`PatchEmbed`]).
//! - **Position embedding** is a learned `num_position_embeddings` (a `side×side`) grid **bilinearly
//!   resampled** to each image's patch grid ([`vision_bilinear`]) — not a fixed per-index add.
//! - **2-D rotary** in attention: each patch's `(row, col)` grid position drives a NeoX rotary table
//!   ([`vision_rotary`]), the row half over the first `head_dim/4` frequencies and the col half over
//!   the next, doubled into the full `head_dim` (`cat(freqs, freqs)`).
//! - **Patch merger**: a per-patch LayerNorm, then `spatial_merge_size²` adjacent patches are
//!   concatenated and projected (`fc1 → GELU(exact) → fc2`) to `out_hidden_size` — this `pooler`
//!   output (one token per merged 2×2 block) is what the decoder consumes.
//!
//! Attention is per **frame** (`grid_t × grid_h·grid_w` patches): a single image is one fully
//! bidirectional block; multiple images/frames get a block-diagonal mask from [`vision_cu_seqlens`].
//! Compute follows the SigLIP path — runs in the loaded weights' dtype against the f32 patches, which
//! MLX promotes to f32.

use mlx_rs::ops::{add, multiply, split, sum_axis};
use mlx_rs::Array;

use crate::error::{Error, Result};
use crate::primitives::attention::{sdpa, AttnMask};
use crate::primitives::nn::{gelu, gelu_tanh, layer_norm, linear};
use crate::primitives::rope::apply_rope;
use crate::primitives::Weights;

/// LayerNorm epsilon — fixed in the reference (`nn.LayerNorm(..., eps=1e-6)`), not a config field.
const LN_EPS: f32 = 1e-6;
/// Vision rotary base — fixed in the reference (`Qwen3_5VisionRotaryEmbedding(theta=10000.0)`).
const ROPE_THETA: f32 = 10000.0;
/// Disallowed-attention fill for the block-diagonal mask (a large finite negative; matches the
/// attention primitive's convention — avoids `-inf` through the softmax).
const MASK_NEG: f32 = -1e30;

/// Geometry of the Qwen3.6 vision tower (`vision_config`).
#[derive(Clone, Debug)]
pub struct Qwen35VisionConfig {
    pub depth: usize,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub intermediate_size: i32,
    pub in_channels: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub spatial_merge_size: i32,
    pub out_hidden_size: i32,
    pub num_position_embeddings: i32,
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Qwen35VisionConfig {
    /// Parse from a `config.json` value, descending into the `vision_config` sub-object.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let c = v
            .get("vision_config")
            .ok_or_else(|| Error::Config("qwen3_5 config.json missing `vision_config`".into()))?;
        let int = |k: &str| -> Option<i32> { c.get(k).and_then(|x| x.as_i64()).map(|x| x as i32) };
        let req = |k: &str| -> Result<i32> {
            int(k).ok_or_else(|| Error::Config(format!("qwen3_5 vision_config missing `{k}`")))
        };
        let deepstack_visual_indexes = c
            .get("deepstack_visual_indexes")
            .and_then(|x| x.as_array())
            .map(|xs| {
                xs.iter()
                    .filter_map(|x| x.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            depth: req("depth")? as usize,
            hidden_size: req("hidden_size")?,
            num_heads: req("num_heads")?,
            intermediate_size: req("intermediate_size")?,
            in_channels: int("in_channels").unwrap_or(3),
            patch_size: int("patch_size").unwrap_or(16),
            temporal_patch_size: int("temporal_patch_size").unwrap_or(2),
            spatial_merge_size: int("spatial_merge_size").unwrap_or(2),
            out_hidden_size: req("out_hidden_size")?,
            num_position_embeddings: req("num_position_embeddings")?,
            deepstack_visual_indexes,
        })
    }

    /// Per-head dimension (`hidden_size / num_heads`).
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }

    /// Learned position grid edge (`√num_position_embeddings`).
    pub fn grid_per_side(&self) -> i32 {
        (self.num_position_embeddings as f64).sqrt() as i32
    }

    /// Flattened patch-embed input width (`C · T · P · P`).
    pub fn patch_in(&self) -> i32 {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Merged-token width before the merger projection (`hidden · merge²`).
    pub fn merge_dim(&self) -> i32 {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }
}

/// The `Conv3d` patch embedding, loaded as a reshaped `[hidden, C·T·P·P]` linear (kernel == stride ==
/// patch ⇒ each patch is an independent dot product over its flattened pixels).
struct PatchEmbed {
    weight: Array,
    bias: Option<Array>,
}

impl PatchEmbed {
    /// `pixel_values` `[num_patches, C·T·P·P]` → `[num_patches, hidden]`.
    fn forward(&self, pixel_values: &Array) -> Result<Array> {
        linear(pixel_values, &self.weight, self.bias.as_ref())
    }
}

/// One pre-norm transformer block: `h += attn(norm1(h)); h += mlp(norm2(h))`.
struct VisionBlock {
    n1_w: Array,
    n1_b: Array,
    n2_w: Array,
    n2_b: Array,
    qkv_w: Array,
    qkv_b: Option<Array>,
    proj_w: Array,
    proj_b: Option<Array>,
    fc1_w: Array,
    fc1_b: Option<Array>,
    fc2_w: Array,
    fc2_b: Option<Array>,
}

impl VisionBlock {
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        num_heads: i32,
        head_dim: i32,
        mask: AttnMask<'_>,
    ) -> Result<Array> {
        let y = layer_norm(x, Some(&self.n1_w), Some(&self.n1_b), LN_EPS)?;
        let x = add(x, &self.attn(&y, cos, sin, num_heads, head_dim, mask)?)?;
        let y = layer_norm(&x, Some(&self.n2_w), Some(&self.n2_b), LN_EPS)?;
        Ok(add(&x, &self.mlp(&y)?)?)
    }

    fn attn(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        num_heads: i32,
        head_dim: i32,
        mask: AttnMask<'_>,
    ) -> Result<Array> {
        let n = x.shape()[0];
        let hidden = num_heads * head_dim;
        // Fused QKV → [n, 3, heads, head_dim] → three [n, heads, head_dim].
        let qkv =
            linear(x, &self.qkv_w, self.qkv_b.as_ref())?.reshape(&[n, 3, num_heads, head_dim])?;
        let parts = split(&qkv, 3, 1)?;
        // RoPE expects [batch, seq, heads, head_dim]; cos/sin [1, n, head_dim] broadcast over heads.
        let q = apply_rope(
            &parts[0].reshape(&[1, n, num_heads, head_dim])?,
            cos,
            sin,
            false,
        )?;
        let k = apply_rope(
            &parts[1].reshape(&[1, n, num_heads, head_dim])?,
            cos,
            sin,
            false,
        )?;
        let v = parts[2].reshape(&[1, n, num_heads, head_dim])?;
        // → [1, heads, n, head_dim] for SDPA.
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let scale = (head_dim as f32).powf(-0.5);
        let out = sdpa(&q, &k, &v, scale, mask)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[n, hidden])?;
        linear(&out, &self.proj_w, self.proj_b.as_ref())
    }

    fn mlp(&self, x: &Array) -> Result<Array> {
        let h = gelu_tanh(&linear(x, &self.fc1_w, self.fc1_b.as_ref())?)?;
        linear(&h, &self.fc2_w, self.fc2_b.as_ref())
    }
}

/// The `spatial_merge_size²` patch merger → `out_hidden_size` (the pooler output).
struct PatchMerger {
    norm_w: Array,
    norm_b: Array,
    fc1_w: Array,
    fc1_b: Option<Array>,
    fc2_w: Array,
    fc2_b: Option<Array>,
    merge_dim: i32,
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    fn forward(&self, x: &Array) -> Result<Array> {
        let m = if self.use_postshuffle_norm {
            let grouped = x.reshape(&[-1, self.merge_dim])?;
            layer_norm(&grouped, Some(&self.norm_w), Some(&self.norm_b), LN_EPS)?
        } else {
            layer_norm(x, Some(&self.norm_w), Some(&self.norm_b), LN_EPS)?
                .reshape(&[-1, self.merge_dim])?
        };
        let m = gelu(&linear(&m, &self.fc1_w, self.fc1_b.as_ref())?)?; // nn.GELU() default = exact erf
        linear(&m, &self.fc2_w, self.fc2_b.as_ref())
    }
}

pub struct Qwen35VisionOutput {
    pub last_hidden_state: Array,
    pub pooler_output: Array,
    pub deepstack_features: Vec<Array>,
}

pub type Qwen3VLVisionConfig = Qwen35VisionConfig;
pub type Qwen3VLVisionOutput = Qwen35VisionOutput;
pub type Qwen3VLVisionModel = Qwen35VisionModel;

/// A loaded Qwen3.6 vision tower.
pub struct Qwen35VisionModel {
    patch_embed: PatchEmbed,
    pos_embed: Array,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
    cfg: Qwen35VisionConfig,
}

impl Qwen35VisionModel {
    /// Load from a checkpoint. `prefix` points at the visual tower module — `model.visual` for the
    /// `qwen3_5` checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: Qwen35VisionConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let req = |k: String| -> Result<Array> { w.require(&k).cloned() };
        let opt = |k: String| -> Option<Array> { w.get(&k).cloned() };

        // Conv3d weight `[hidden, C, T, P, P]` → reshaped `[hidden, C·T·P·P]` linear.
        let patch_embed = PatchEmbed {
            weight: req(p("patch_embed.proj.weight"))?
                .reshape(&[cfg.hidden_size, cfg.patch_in()])?,
            bias: opt(p("patch_embed.proj.bias")),
        };
        let pos_embed = req(p("pos_embed.weight"))?;

        let blocks = (0..cfg.depth)
            .map(|i| {
                let b = |leaf: &str| p(&format!("blocks.{i}.{leaf}"));
                Ok(VisionBlock {
                    n1_w: req(b("norm1.weight"))?,
                    n1_b: req(b("norm1.bias"))?,
                    n2_w: req(b("norm2.weight"))?,
                    n2_b: req(b("norm2.bias"))?,
                    qkv_w: req(b("attn.qkv.weight"))?,
                    qkv_b: opt(b("attn.qkv.bias")),
                    proj_w: req(b("attn.proj.weight"))?,
                    proj_b: opt(b("attn.proj.bias")),
                    fc1_w: req(b("mlp.linear_fc1.weight"))?,
                    fc1_b: opt(b("mlp.linear_fc1.bias")),
                    fc2_w: req(b("mlp.linear_fc2.weight"))?,
                    fc2_b: opt(b("mlp.linear_fc2.bias")),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let load_merger = |stem: String, use_postshuffle_norm: bool| -> Result<PatchMerger> {
            let m = |leaf: &str| format!("{stem}.{leaf}");
            Ok(PatchMerger {
                norm_w: req(m("norm.weight"))?,
                norm_b: req(m("norm.bias"))?,
                fc1_w: req(m("linear_fc1.weight"))?,
                fc1_b: opt(m("linear_fc1.bias")),
                fc2_w: req(m("linear_fc2.weight"))?,
                fc2_b: opt(m("linear_fc2.bias")),
                merge_dim: cfg.merge_dim(),
                use_postshuffle_norm,
            })
        };

        let merger = load_merger(p("merger"), false)?;

        let deepstack_mergers = (0..cfg.deepstack_visual_indexes.len())
            .map(|i| load_merger(p(&format!("deepstack_merger_list.{i}")), true))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            deepstack_mergers,
            cfg,
        })
    }

    /// The tower geometry.
    pub fn config(&self) -> &Qwen35VisionConfig {
        &self.cfg
    }

    /// Encode preprocessed `pixel_values` `[total_patches, C·T·P·P]` for the images described by
    /// `grid_thw` (per image `[t, h, w]` in patch units) → `[total_patches / merge², out_hidden]`,
    /// one embedding per merged `merge×merge` block, in the same merge-block order the preprocessor
    /// emits patches.
    pub fn forward(&self, pixel_values: &Array, grid_thw: &[[i32; 3]]) -> Result<Array> {
        Ok(self
            .forward_with_deepstack(pixel_values, grid_thw)?
            .pooler_output)
    }

    pub fn forward_with_deepstack(
        &self,
        pixel_values: &Array,
        grid_thw: &[[i32; 3]],
    ) -> Result<Qwen35VisionOutput> {
        let cfg = &self.cfg;
        let merge = cfg.spatial_merge_size;
        let head_dim = cfg.head_dim();
        let n = pixel_values.shape()[0];

        let mut hs = self.patch_embed.forward(pixel_values)?;
        let pos = self.position_embeds(grid_thw, n)?;
        hs = add(&hs, &pos)?;

        let pos_ids = vision_position_ids(grid_thw, merge);
        let (cos, sin) = vision_rotary(&pos_ids, head_dim, ROPE_THETA)?;
        let cu = vision_cu_seqlens(grid_thw);
        let mask_arr = block_diag_mask(&cu, n);
        let mask = match &mask_arr {
            Some(a) => AttnMask::Additive(a),
            None => AttnMask::None,
        };

        let mut deepstack_features = Vec::with_capacity(self.deepstack_mergers.len());
        for (layer_num, blk) in self.blocks.iter().enumerate() {
            hs = blk.forward(&hs, &cos, &sin, cfg.num_heads, head_dim, mask)?;
            if let Some(tap) = cfg
                .deepstack_visual_indexes
                .iter()
                .position(|&idx| idx == layer_num)
            {
                deepstack_features.push(self.deepstack_mergers[tap].forward(&hs)?);
            }
        }
        let pooler_output = self.merger.forward(&hs)?;
        Ok(Qwen35VisionOutput {
            last_hidden_state: hs,
            pooler_output,
            deepstack_features,
        })
    }

    /// Gather + bilinearly weight the learned position table for the image grid → `[total_patches,
    /// hidden]` (the `(pos_embed(indices) * weights).sum(0)` of the reference).
    fn position_embeds(&self, grid_thw: &[[i32; 3]], n: i32) -> Result<Array> {
        let (idx, wts) = vision_bilinear(
            grid_thw,
            self.cfg.grid_per_side(),
            self.cfg.spatial_merge_size,
        );
        let idx_arr = Array::from_slice(&idx, &[4 * n]);
        let gathered =
            self.pos_embed
                .take_axis(&idx_arr, 0)?
                .reshape(&[4, n, self.cfg.hidden_size])?;
        let w_arr = Array::from_slice(&wts, &[4, n, 1]).as_dtype(gathered.dtype())?;
        Ok(sum_axis(&multiply(&gathered, &w_arr)?, 0, false)?)
    }
}

/// Per-frame cumulative sequence lengths (`get_vision_cu_seqlens`): each image contributes `t` frames
/// of `h·w` patches. Length `total_frames + 1`, starting at 0.
fn vision_cu_seqlens(grid_thw: &[[i32; 3]]) -> Vec<i32> {
    let mut cu = vec![0i32];
    let mut acc = 0;
    for &[t, h, w] in grid_thw {
        let frame = h * w;
        for _ in 0..t {
            acc += frame;
            cu.push(acc);
        }
    }
    cu
}

/// Per-patch `(row, col)` grid positions for the 2-D rotary (`get_vision_position_ids`), in
/// merge-block order (`merge×merge` adjacent patches grouped), repeated over `t` frames.
fn vision_position_ids(grid_thw: &[[i32; 3]], merge: i32) -> Vec<(i32, i32)> {
    let mut out = Vec::new();
    for &[t, h, w] in grid_thw {
        let (hm, wm) = (h / merge, w / merge);
        let mut per = Vec::with_capacity((h * w) as usize);
        for a in 0..hm {
            for c in 0..wm {
                for b in 0..merge {
                    for d in 0..merge {
                        per.push((a * merge + b, c * merge + d));
                    }
                }
            }
        }
        for _ in 0..t {
            out.extend_from_slice(&per);
        }
    }
    out
}

/// Bilinear interpolation indices/weights into the `side×side` learned position grid
/// (`get_vision_bilinear_indices_and_weights`). Returns `(indices, weights)` each laid out
/// `[corner(4)][patch]` row-major (corner-major), in merge-block order repeated over `t`.
fn vision_bilinear(grid_thw: &[[i32; 3]], side: i32, merge: i32) -> (Vec<i32>, Vec<f32>) {
    let mut idx: [Vec<i32>; 4] = [vec![], vec![], vec![], vec![]];
    let mut wts: [Vec<f32>; 4] = [vec![], vec![], vec![], vec![]];
    for &[t, h, w] in grid_thw {
        let lin = |len: i32| -> Vec<f32> {
            if len == 1 {
                vec![0.0]
            } else {
                (0..len)
                    .map(|i| (side - 1) as f32 * i as f32 / (len - 1) as f32)
                    .collect()
            }
        };
        let h_grid = lin(h);
        let w_grid = lin(w);
        // `.int()` truncates toward zero; the grid is non-negative, so this is floor.
        let h_floor: Vec<i32> = h_grid.iter().map(|&x| x as i32).collect();
        let w_floor: Vec<i32> = w_grid.iter().map(|&x| x as i32).collect();
        let h_ceil: Vec<i32> = h_floor.iter().map(|&f| (f + 1).min(side - 1)).collect();
        let w_ceil: Vec<i32> = w_floor.iter().map(|&f| (f + 1).min(side - 1)).collect();
        let h_frac: Vec<f32> = h_grid
            .iter()
            .zip(&h_floor)
            .map(|(&g, &f)| g - f as f32)
            .collect();
        let w_frac: Vec<f32> = w_grid
            .iter()
            .zip(&w_floor)
            .map(|(&g, &f)| g - f as f32)
            .collect();

        // Full row-major (h, w) corner index/weight arrays.
        let hw = (h * w) as usize;
        let mut c_idx: [Vec<i32>; 4] = [vec![0; hw], vec![0; hw], vec![0; hw], vec![0; hw]];
        let mut c_w: [Vec<f32>; 4] = [vec![0.0; hw], vec![0.0; hw], vec![0.0; hw], vec![0.0; hw]];
        for i in 0..h as usize {
            for j in 0..w as usize {
                let p = i * w as usize + j;
                let (hf, hc) = (h_floor[i] * side, h_ceil[i] * side);
                c_idx[0][p] = hf + w_floor[j];
                c_idx[1][p] = hf + w_ceil[j];
                c_idx[2][p] = hc + w_floor[j];
                c_idx[3][p] = hc + w_ceil[j];
                let (hfr, wfr) = (h_frac[i], w_frac[j]);
                c_w[0][p] = (1.0 - hfr) * (1.0 - wfr);
                c_w[1][p] = (1.0 - hfr) * wfr;
                c_w[2][p] = hfr * (1.0 - wfr);
                c_w[3][p] = hfr * wfr;
            }
        }

        // Reorder into merge-block order (same a,c,b,d transpose as the position ids), repeated `t`.
        let (hm, wm) = (h / merge, w / merge);
        let mut reorder = Vec::with_capacity(hw);
        for a in 0..hm {
            for c in 0..wm {
                for b in 0..merge {
                    for d in 0..merge {
                        reorder.push(((a * merge + b) * w + (c * merge + d)) as usize);
                    }
                }
            }
        }
        for _ in 0..t {
            for &r in &reorder {
                for corner in 0..4 {
                    idx[corner].push(c_idx[corner][r]);
                    wts[corner].push(c_w[corner][r]);
                }
            }
        }
    }
    let mut idx_flat = Vec::new();
    let mut w_flat = Vec::new();
    for corner in 0..4 {
        idx_flat.extend_from_slice(&idx[corner]);
        w_flat.extend_from_slice(&wts[corner]);
    }
    (idx_flat, w_flat)
}

/// Build the NeoX `(cos, sin)` rotary tables `[1, n, head_dim]` for the vision 2-D RoPE: each patch's
/// `(row, col)` drives the first/second `head_dim/4` frequencies, doubled into the full head
/// (`cat(freqs, freqs)`) so each rotate-half pair `(c, c + head_dim/2)` shares an angle.
fn vision_rotary(pos_ids: &[(i32, i32)], head_dim: i32, theta: f32) -> Result<(Array, Array)> {
    let half = (head_dim / 2) as usize; // the rotary sub-dim (Qwen3_5VisionRotaryEmbedding(head_dim/2))
    let nfreq = half / 2; // head_dim/4 frequencies per axis
    let inv: Vec<f32> = (0..nfreq)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / half as f32))
        .collect();
    let n = pos_ids.len();
    let mut cos = Vec::with_capacity(n * head_dim as usize);
    let mut sin = Vec::with_capacity(n * head_dim as usize);
    for &(row, col) in pos_ids {
        // First half: [row·inv ‖ col·inv] (length head_dim/2); then duplicated (cat(freqs, freqs)).
        let mut ang = Vec::with_capacity(head_dim as usize);
        for &f in &inv {
            ang.push(row as f32 * f);
        }
        for &f in &inv {
            ang.push(col as f32 * f);
        }
        ang.extend_from_within(..);
        for &a in &ang {
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    Ok((
        Array::from_slice(&cos, &[1, n as i32, head_dim]),
        Array::from_slice(&sin, &[1, n as i32, head_dim]),
    ))
}

/// Block-diagonal additive attention mask `[1, 1, n, n]` from per-frame `cu_seqlens` — `0` within a
/// frame, [`MASK_NEG`] across frames. `None` when there is a single frame (fully bidirectional, the
/// fast unmasked path).
fn block_diag_mask(cu: &[i32], n: i32) -> Option<Array> {
    if cu.len() <= 2 {
        return None;
    }
    let mut block_of = vec![0usize; n as usize];
    for b in 0..cu.len() - 1 {
        for j in cu[b]..cu[b + 1] {
            block_of[j as usize] = b;
        }
    }
    let mut data = vec![0f32; (n * n) as usize];
    for i in 0..n as usize {
        for j in 0..n as usize {
            if block_of[i] != block_of[j] {
                data[i * n as usize + j] = MASK_NEG;
            }
        }
    }
    Some(Array::from_slice(&data, &[1, 1, n, n]))
}

/// Join a weight-key `prefix` and `leaf` with `.` (no leading dot when empty).
fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_owned()
    } else {
        format!("{prefix}.{leaf}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;
    use std::collections::{BTreeSet, HashMap};
    use std::path::{Path, PathBuf};

    fn oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("testdata/qwen35_vision_oracle.json")).unwrap()
    }

    fn arr(j: &serde_json::Value, k: &str) -> Vec<f32> {
        j[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn arr_i32(j: &serde_json::Value, k: &str) -> Vec<i32> {
        j[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    }

    fn test_cfg() -> Qwen35VisionConfig {
        // Mirrors /tmp/gen_vision.py.
        Qwen35VisionConfig {
            depth: 2,
            hidden_size: 32,
            num_heads: 4,
            intermediate_size: 64,
            in_channels: 3,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 48,
            num_position_embeddings: 36,
            deepstack_visual_indexes: Vec::new(),
        }
    }

    #[test]
    fn config_geometry() {
        let cfg = test_cfg();
        assert_eq!(cfg.head_dim(), 8);
        assert_eq!(cfg.grid_per_side(), 6);
        assert_eq!(cfg.patch_in(), 24); // 3·2·2·2
        assert_eq!(cfg.merge_dim(), 128); // 32·4
    }

    /// The deterministic host helpers (`cu_seqlens`, `position_ids`, bilinear indices/weights) must
    /// reproduce the reference `vision_utils` outputs exactly — these are pure index math, the
    /// error-prone novelty of the encoder, and they drive the rotary tables and pos-embed gather.
    #[test]
    fn host_helpers_match_reference() {
        let j = oracle();
        let grid: Vec<[i32; 3]> = {
            let g = arr_i32(&j, "grid_thw");
            vec![[g[0], g[1], g[2]]]
        };

        let cu = vision_cu_seqlens(&grid);
        assert_eq!(cu, arr_i32(&j, "expect_cu"));

        let pos = vision_position_ids(&grid, 2);
        let pos_flat: Vec<i32> = pos.iter().flat_map(|&(r, c)| [r, c]).collect();
        assert_eq!(pos_flat, arr_i32(&j, "expect_pos_ids"));

        let (idx, wts) = vision_bilinear(&grid, 6, 2);
        assert_eq!(idx, arr_i32(&j, "expect_bi"));
        let exp_bw = arr(&j, "expect_bw");
        let md = wts
            .iter()
            .zip(&exp_bw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 1e-6,
            "bilinear weights vs reference: max abs diff {md}"
        );
    }

    /// End-to-end encoder vs the reference `Qwen3_5VisionModel.forward` (patch embed → bilinear pos
    /// embed → 2-D rotary blocks → merger). Built through the real [`Qwen35VisionModel::from_weights`]
    /// loader (incl. the Conv3d→linear patch-weight reshape). Gated on **relative** error (the engine's
    /// attention-primitive convention): patches are processed `N>1` rows at a time, so every matmul runs
    /// MLX's reduced-precision (bf16-class) f32 GEMM rather than the exact `M=1` GEMV, and the small
    /// per-op error compounds over 2 blocks + the merger to a ~0.2% floor. A structural error (a wrong
    /// qkv split, rotary layout, gelu variant, or merger grouping) diverges by `O(1)` — far above this.
    /// Regenerate the fixture with `/tmp/gen_vision.py` (sc-7633).
    #[test]
    fn encoder_matches_qwen3_5_vision_reference() {
        let j = oracle();
        let cfg = test_cfg();
        let grid: Vec<[i32; 3]> = {
            let g = arr_i32(&j, "grid_thw");
            vec![[g[0], g[1], g[2]]]
        };
        let n = (grid[0][0] * grid[0][1] * grid[0][2]) as i32;

        let mut m: HashMap<String, Array> = HashMap::new();
        let put = |m: &mut HashMap<String, Array>, key: &str, k: &str, shape: &[i32]| {
            m.insert(key.to_string(), Array::from_slice(&arr(&j, k), shape));
        };
        let (h, hid, inter, out_h) = (
            cfg.num_heads,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.out_hidden_size,
        );
        let _ = h;
        // Patch embed weight stored 5-D (as in the real checkpoint) → loader reshapes it.
        put(
            &mut m,
            "model.visual.patch_embed.proj.weight",
            "patch_w",
            &[
                hid,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ],
        );
        put(
            &mut m,
            "model.visual.patch_embed.proj.bias",
            "patch_b",
            &[hid],
        );
        put(
            &mut m,
            "model.visual.pos_embed.weight",
            "pos_embed",
            &[cfg.num_position_embeddings, hid],
        );
        for i in 0..cfg.depth {
            let pre = format!("model.visual.blocks.{i}");
            put(
                &mut m,
                &format!("{pre}.norm1.weight"),
                &format!("b{i}_n1w"),
                &[hid],
            );
            put(
                &mut m,
                &format!("{pre}.norm1.bias"),
                &format!("b{i}_n1b"),
                &[hid],
            );
            put(
                &mut m,
                &format!("{pre}.norm2.weight"),
                &format!("b{i}_n2w"),
                &[hid],
            );
            put(
                &mut m,
                &format!("{pre}.norm2.bias"),
                &format!("b{i}_n2b"),
                &[hid],
            );
            put(
                &mut m,
                &format!("{pre}.attn.qkv.weight"),
                &format!("b{i}_qkv_w"),
                &[3 * hid, hid],
            );
            put(
                &mut m,
                &format!("{pre}.attn.qkv.bias"),
                &format!("b{i}_qkv_b"),
                &[3 * hid],
            );
            put(
                &mut m,
                &format!("{pre}.attn.proj.weight"),
                &format!("b{i}_proj_w"),
                &[hid, hid],
            );
            put(
                &mut m,
                &format!("{pre}.attn.proj.bias"),
                &format!("b{i}_proj_b"),
                &[hid],
            );
            put(
                &mut m,
                &format!("{pre}.mlp.linear_fc1.weight"),
                &format!("b{i}_fc1_w"),
                &[inter, hid],
            );
            put(
                &mut m,
                &format!("{pre}.mlp.linear_fc1.bias"),
                &format!("b{i}_fc1_b"),
                &[inter],
            );
            put(
                &mut m,
                &format!("{pre}.mlp.linear_fc2.weight"),
                &format!("b{i}_fc2_w"),
                &[hid, inter],
            );
            put(
                &mut m,
                &format!("{pre}.mlp.linear_fc2.bias"),
                &format!("b{i}_fc2_b"),
                &[hid],
            );
        }
        put(
            &mut m,
            "model.visual.merger.norm.weight",
            "mg_norm_w",
            &[hid],
        );
        put(&mut m, "model.visual.merger.norm.bias", "mg_norm_b", &[hid]);
        put(
            &mut m,
            "model.visual.merger.linear_fc1.weight",
            "mg_fc1_w",
            &[cfg.merge_dim(), cfg.merge_dim()],
        );
        put(
            &mut m,
            "model.visual.merger.linear_fc1.bias",
            "mg_fc1_b",
            &[cfg.merge_dim()],
        );
        put(
            &mut m,
            "model.visual.merger.linear_fc2.weight",
            "mg_fc2_w",
            &[out_h, cfg.merge_dim()],
        );
        put(
            &mut m,
            "model.visual.merger.linear_fc2.bias",
            "mg_fc2_b",
            &[out_h],
        );

        let w = Weights::from_map(m);
        let model = Qwen35VisionModel::from_weights(&w, "model.visual", cfg.clone()).unwrap();

        let pixel = Array::from_slice(&arr(&j, "pixel"), &[n, cfg.patch_in()]);
        let out = model.forward(&pixel, &grid).unwrap();
        assert_eq!(out.shape(), &[n / 4, out_h]);

        let got = out
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let exp = arr(&j, "expected_output");
        let max_abs = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_mag = exp.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel = max_abs / (max_mag + 1e-20);
        assert!(
            rel < 3e-3,
            "vision encoder vs reference: rel err {rel} (max|Δ| {max_abs}, max|exp| {max_mag})\n got {got:?}\n exp {exp:?}"
        );
    }

    fn qwen3vl_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("testdata/qwen3vl_vision_oracle.json")).unwrap()
    }

    fn qwen3vl_cfg(j: &serde_json::Value) -> Qwen35VisionConfig {
        let d = &j["dims"];
        let get = |k: &str| d[k].as_i64().unwrap() as i32;
        Qwen35VisionConfig {
            depth: get("depth") as usize,
            hidden_size: get("hidden"),
            num_heads: get("num_heads"),
            intermediate_size: get("inter"),
            in_channels: get("in_ch"),
            patch_size: get("patch"),
            temporal_patch_size: get("tpatch"),
            spatial_merge_size: get("merge"),
            out_hidden_size: get("out_hidden"),
            num_position_embeddings: get("num_pos"),
            deepstack_visual_indexes: j["deepstack_visual_indexes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect(),
        }
    }

    /// Qwen3-VL parse: the `deepstack_visual_indexes` are read from `vision_config`.
    #[test]
    fn config_parses_deepstack_indexes() {
        let v = serde_json::json!({
            "vision_config": {
                "depth": 27, "hidden_size": 1152, "num_heads": 16, "intermediate_size": 4304,
                "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "out_hidden_size": 4096, "num_position_embeddings": 2304,
                "deepstack_visual_indexes": [8, 16, 24]
            }
        });
        let cfg = Qwen35VisionConfig::from_json(&v).unwrap();
        assert_eq!(cfg.deepstack_visual_indexes, vec![8, 16, 24]);
        assert_eq!(cfg.out_hidden_size, 4096);
        assert_eq!(cfg.head_dim(), 72);
    }

    fn qwen3vl_snapshot_dir() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("QWEN3VL_SNAPSHOT") {
            let path = PathBuf::from(path);
            return path.exists().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let path = PathBuf::from(home).join(
            ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        );
        path.exists().then_some(path)
    }

    fn qwen3vl_visual_weights(snapshot: &Path) -> Weights {
        let index_path = snapshot.join("model.safetensors.index.json");
        let index_raw = std::fs::read_to_string(&index_path).unwrap_or_else(|e| {
            panic!("qwen3vl oracle: reading index {}: {e}", index_path.display())
        });
        let index: serde_json::Value = serde_json::from_str(&index_raw)
            .unwrap_or_else(|e| panic!("qwen3vl oracle: parsing index {}: {e}", index_path.display()));
        let weight_map = index["weight_map"]
            .as_object()
            .expect("qwen3vl oracle: index missing `weight_map` object");
        let shards: BTreeSet<String> = weight_map
            .iter()
            .filter_map(|(k, v)| {
                k.starts_with("model.visual.")
                    .then(|| v.as_str().map(ToOwned::to_owned))
                    .flatten()
            })
            .collect();
        assert!(
            !shards.is_empty(),
            "qwen3vl oracle: no `model.visual.*` shards in index {}",
            index_path.display()
        );
        let mut tensors = HashMap::new();
        for shard in shards {
            let shard_path = snapshot.join(&shard);
            let weights = Weights::from_file(&shard_path)
                .unwrap_or_else(|e| panic!("qwen3vl oracle: loading shard {}: {e}", shard_path.display()));
            for (k, v) in weights.into_map() {
                if !k.starts_with("model.visual.") {
                    continue;
                }
                let v = v
                    .as_dtype(Dtype::Float32)
                    .unwrap_or_else(|e| panic!("qwen3vl oracle: casting `{k}` to f32: {e}"));
                tensors.insert(k, v);
            }
        }
        assert!(
            !tensors.is_empty(),
            "qwen3vl oracle: no `model.visual.*` tensors loaded from {}",
            snapshot.display()
        );
        Weights::from_map(tensors)
    }

    fn nested_arr(j: &serde_json::Value, k: &str, idx: usize) -> Vec<f32> {
        j[k].as_array().unwrap()[idx]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn max_rel(got: &[f32], exp: &[f32]) -> (f32, f32, f32) {
        let max_abs = got
            .iter()
            .zip(exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_mag = exp.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        (max_abs / (max_mag + 1e-20), max_abs, max_mag)
    }

    #[test]
    fn qwen3vl_oracle_matches_hf_reference() {
        let j = qwen3vl_oracle();
        let cfg = qwen3vl_cfg(&j);
        assert_eq!(cfg.depth, 27);
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.head_dim(), 72);
        assert_eq!(cfg.intermediate_size, 4304);
        assert_eq!(cfg.in_channels, 3);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.temporal_patch_size, 2);
        assert_eq!(cfg.spatial_merge_size, 2);
        assert_eq!(cfg.out_hidden_size, 4096);
        assert_eq!(cfg.num_position_embeddings, 2304);
        assert_eq!(cfg.grid_per_side(), 48);
        assert_eq!(cfg.patch_in(), 1536);
        assert_eq!(cfg.merge_dim(), 4608);
        assert_eq!(cfg.deepstack_visual_indexes, vec![8, 16, 24]);

        let grid: Vec<[i32; 3]> = {
            let g = arr_i32(&j, "grid_thw");
            vec![[g[0], g[1], g[2]]]
        };
        let n = (grid[0][0] * grid[0][1] * grid[0][2]) as i32;
        assert_eq!(n, j["dims"]["N"].as_i64().unwrap() as i32);
        assert_eq!(arr_i32(&j, "expected_output_shape"), vec![4, 4096]);
        let deepstack_shapes: Vec<Vec<i32>> = j["expected_deepstack_shapes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|shape| {
                shape
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap() as i32)
                    .collect()
            })
            .collect();
        assert_eq!(
            deepstack_shapes,
            vec![vec![4, 4096], vec![4, 4096], vec![4, 4096]]
        );

        let cu = vision_cu_seqlens(&grid);
        assert_eq!(cu, arr_i32(&j, "expect_cu"));
        let pos = vision_position_ids(&grid, cfg.spatial_merge_size);
        let pos_flat: Vec<i32> = pos.iter().flat_map(|&(r, c)| [r, c]).collect();
        assert_eq!(pos_flat, arr_i32(&j, "expect_pos_ids"));
        let (idx, wts) = vision_bilinear(&grid, cfg.grid_per_side(), cfg.spatial_merge_size);
        assert_eq!(idx, arr_i32(&j, "expect_bi"));
        let exp_bw = arr(&j, "expect_bw");
        let md = wts
            .iter()
            .zip(&exp_bw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            md < 1e-6,
            "qwen3-vl bilinear weights vs oracle: max abs diff {md}"
        );

        let Some(snapshot) = qwen3vl_snapshot_dir() else {
            eprintln!("skipping qwen3vl real-weight oracle: cached HF snapshot unavailable");
            return;
        };
        let weights = qwen3vl_visual_weights(&snapshot);
        let model = Qwen35VisionModel::from_weights(&weights, "model.visual", cfg.clone()).unwrap();
        let pixel = Array::from_slice(&arr(&j, "pixel"), &[n, cfg.patch_in()]);
        let out = model.forward_with_deepstack(&pixel, &grid).unwrap();
        assert_eq!(out.pooler_output.shape(), &[4, 4096]);
        assert_eq!(out.deepstack_features.len(), 3);

        let got = out
            .pooler_output
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let exp = arr(&j, "expected_output");
        let (rel, max_abs, max_mag) = max_rel(&got, &exp);
        assert!(
            rel < 7e-3,
            "qwen3-vl pooled output vs HF: rel err {rel} (max|Δ| {max_abs}, max|exp| {max_mag})"
        );

        for (tap, feature) in out.deepstack_features.iter().enumerate() {
            assert_eq!(feature.shape(), &[4, 4096]);
            let got = feature
                .as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            let exp = nested_arr(&j, "expected_deepstack", tap);
            let (rel, max_abs, max_mag) = max_rel(&got, &exp);
            assert!(
                rel < 7e-3,
                "qwen3-vl deepstack tap {tap} vs HF: rel err {rel} (max|Δ| {max_abs}, max|exp| {max_mag})"
            );
        }
    }
}
