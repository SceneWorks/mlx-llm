//! SigLIP vision tower (story 7157).
//!
//! JoyCaption's LLaVA stack uses `google/siglip2-so400m-patch14-384` as its image encoder: a patch
//! conv embedding + learned position embedding, a stack of pre-norm transformer encoder layers
//! (bias-ful QKV, `gelu_pytorch_tanh` MLP), and a final post-layernorm. The decoder reads a chosen
//! intermediate hidden state (JoyCaption: layer `-2`, all 729 patch tokens), so [`forward`] returns
//! the HF-style `hidden_states` list (embeddings + one per layer) in addition to the post-normed
//! `last_hidden_state`.
//!
//! Compute runs in the dtype the weights load at against the f32 preprocessed pixels — MLX promotes
//! to f32, matching the reference engine — so the vision features are dtype-stable before the
//! projector casts them into the decoder.
//!
//! [`SiglipVisionConfig::default`] is the so400m-patch14-384 geometry; nothing here is
//! JoyCaption-specific (the feature-layer choice lives with the VLM in [`super::joycaption`]).

use mlx_rs::ops::add;
use mlx_rs::Array;

use crate::error::{Error, Result};
use crate::primitives::nn::{conv2d, gelu_tanh, layer_norm, linear};
use crate::primitives::sdpa;
use crate::primitives::attention::AttnMask;
use crate::primitives::Weights;

/// Geometry of a SigLIP vision tower.
#[derive(Clone, Copy, Debug)]
pub struct SiglipVisionConfig {
    /// Square input edge in pixels.
    pub image_size: i32,
    /// Patch (and conv kernel/stride) size.
    pub patch_size: i32,
    /// Input channels (3 = RGB).
    pub num_channels: i32,
    /// Hidden width.
    pub hidden_size: i32,
    /// MLP inner width.
    pub intermediate_size: i32,
    /// Number of encoder layers.
    pub num_hidden_layers: i32,
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f32,
}

impl Default for SiglipVisionConfig {
    /// `siglip2-so400m-patch14-384`.
    fn default() -> Self {
        Self {
            image_size: 384,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            layer_norm_eps: 1e-6,
        }
    }
}

impl SiglipVisionConfig {
    /// Per-head dimension.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Patches per side (`image_size / patch_size`).
    pub fn grid(&self) -> i32 {
        self.image_size / self.patch_size
    }

    /// Total patch tokens (`grid^2`).
    pub fn num_patches(&self) -> i32 {
        self.grid() * self.grid()
    }
}

/// Output of the vision tower.
pub struct SiglipVisionOutput {
    /// `[b, num_patches, hidden]` after the final post-layernorm.
    pub last_hidden_state: Array,
    /// HF-style hidden states: the embeddings output followed by one output per encoder layer
    /// (before post-layernorm). Length is `num_hidden_layers + 1`.
    pub hidden_states: Vec<Array>,
}

/// A loaded SigLIP vision tower.
pub struct SiglipVisionTower {
    patch_embedding: Array,
    patch_bias: Option<Array>,
    position_embedding: Array,
    layers: Vec<SiglipEncoderLayer>,
    post_ln_w: Array,
    post_ln_b: Array,
    cfg: SiglipVisionConfig,
}

impl SiglipVisionTower {
    /// Load from a checkpoint. `prefix` points at the HF `vision_model` module — e.g.
    /// `vision_tower.vision_model` for a LLaVA checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: SiglipVisionConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        // HF stores the patch conv `[out, in, kH, kW]`; mlx wants `[out, kH, kW, in]`.
        let patch_embedding = w
            .require(&p("embeddings.patch_embedding.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?;
        let patch_bias = w.get(&p("embeddings.patch_embedding.bias")).cloned();
        let position_embedding = w.require(&p("embeddings.position_embedding.weight"))?.clone();
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| SiglipEncoderLayer::from_weights(w, &p(&format!("encoder.layers.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding,
            patch_bias,
            position_embedding,
            layers,
            post_ln_w: w.require(&p("post_layernorm.weight"))?.clone(),
            post_ln_b: w.require(&p("post_layernorm.bias"))?.clone(),
            cfg,
        })
    }

    /// Patch + position embeddings of preprocessed NHWC `pixel_values` → `[b, num_patches, hidden]`.
    pub fn embeddings(&self, pixel_values: &Array) -> Result<Array> {
        let b = pixel_values.shape()[0];
        let patches = conv2d(
            pixel_values,
            &self.patch_embedding,
            self.patch_bias.as_ref(),
            self.cfg.patch_size,
            0,
        )?;
        let patches = patches.reshape(&[b, self.cfg.num_patches(), self.cfg.hidden_size])?;
        let pos = self
            .position_embedding
            .reshape(&[1, self.cfg.num_patches(), self.cfg.hidden_size])?;
        Ok(add(&patches, &pos)?)
    }

    /// Run the tower over preprocessed NHWC `pixel_values`, collecting the per-layer hidden states.
    pub fn forward(&self, pixel_values: &Array) -> Result<SiglipVisionOutput> {
        let mut hidden = self.embeddings(pixel_values)?;
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(hidden.clone());
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
            hidden_states.push(hidden.clone());
        }
        let last_hidden_state = layer_norm(
            &hidden,
            Some(&self.post_ln_w),
            Some(&self.post_ln_b),
            self.cfg.layer_norm_eps,
        )?;
        Ok(SiglipVisionOutput {
            last_hidden_state,
            hidden_states,
        })
    }

    /// The tower geometry.
    pub fn config(&self) -> &SiglipVisionConfig {
        &self.cfg
    }
}

/// Select a hidden state from a [`SiglipVisionOutput`] by HF-style index (negatives count from the
/// end; `-2` = the penultimate state = the layer the LLaVA decoder reads).
pub fn select_vision_feature(output: &SiglipVisionOutput, layer: i32) -> Result<Array> {
    let len = output.hidden_states.len() as i32;
    let idx = if layer < 0 { len + layer } else { layer };
    if idx < 0 || idx >= len {
        return Err(Error::Msg(format!(
            "siglip: vision feature layer {layer} out of range for {len} hidden states"
        )));
    }
    Ok(output.hidden_states[idx as usize].clone())
}

struct SiglipEncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    ln2_w: Array,
    ln2_b: Array,
    attn: SiglipAttention,
    mlp: SiglipMlp,
    eps: f32,
}

impl SiglipEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &SiglipVisionConfig) -> Result<Self> {
        Ok(Self {
            ln1_w: w.require(&join(prefix, "layer_norm1.weight"))?.clone(),
            ln1_b: w.require(&join(prefix, "layer_norm1.bias"))?.clone(),
            ln2_w: w.require(&join(prefix, "layer_norm2.weight"))?.clone(),
            ln2_b: w.require(&join(prefix, "layer_norm2.bias"))?.clone(),
            attn: SiglipAttention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            mlp: SiglipMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.ln1_w), Some(&self.ln1_b), self.eps)?;
        let x = add(x, &self.attn.forward(&y)?)?;
        let y = layer_norm(&x, Some(&self.ln2_w), Some(&self.ln2_b), self.eps)?;
        Ok(add(&x, &self.mlp.forward(&y)?)?)
    }
}

struct SiglipAttention {
    q_w: Array,
    q_b: Option<Array>,
    k_w: Array,
    k_b: Option<Array>,
    v_w: Array,
    v_b: Option<Array>,
    out_w: Array,
    out_b: Option<Array>,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl SiglipAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &SiglipVisionConfig) -> Result<Self> {
        let bias = |leaf: &str| w.get(&join(prefix, leaf)).cloned();
        let head_dim = cfg.head_dim();
        Ok(Self {
            q_w: w.require(&join(prefix, "q_proj.weight"))?.clone(),
            q_b: bias("q_proj.bias"),
            k_w: w.require(&join(prefix, "k_proj.weight"))?.clone(),
            k_b: bias("k_proj.bias"),
            v_w: w.require(&join(prefix, "v_proj.weight"))?.clone(),
            v_b: bias("v_proj.bias"),
            out_w: w.require(&join(prefix, "out_proj.weight"))?.clone(),
            out_b: bias("out_proj.bias"),
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(linear(x, &self.q_w, self.q_b.as_ref())?)?;
        let k = to_heads(linear(x, &self.k_w, self.k_b.as_ref())?)?;
        let v = to_heads(linear(x, &self.v_w, self.v_b.as_ref())?)?;
        // SigLIP attention is fully bidirectional — no mask.
        let out = sdpa(&q, &k, &v, self.scale, AttnMask::None)?;
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, self.num_heads * self.head_dim])?;
        linear(&out, &self.out_w, self.out_b.as_ref())
    }
}

struct SiglipMlp {
    fc1_w: Array,
    fc1_b: Option<Array>,
    fc2_w: Array,
    fc2_b: Option<Array>,
}

impl SiglipMlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1_w: w.require(&join(prefix, "fc1.weight"))?.clone(),
            fc1_b: w.get(&join(prefix, "fc1.bias")).cloned(),
            fc2_w: w.require(&join(prefix, "fc2.weight"))?.clone(),
            fc2_b: w.get(&join(prefix, "fc2.bias")).cloned(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = linear(x, &self.fc1_w, self.fc1_b.as_ref())?;
        let x = gelu_tanh(&x)?;
        linear(&x, &self.fc2_w, self.fc2_b.as_ref())
    }
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

    #[test]
    fn so400m_config_geometry() {
        let cfg = SiglipVisionConfig::default();
        assert_eq!(cfg.num_patches(), 729); // 27*27
        assert_eq!(cfg.grid(), 27);
        assert_eq!(cfg.head_dim(), 72); // 1152/16
    }

    #[test]
    fn feature_layer_negative_index() {
        let hs = vec![
            Array::from_slice(&[1.0f32], &[1, 1, 1]),
            Array::from_slice(&[2.0f32], &[1, 1, 1]),
            Array::from_slice(&[3.0f32], &[1, 1, 1]),
        ];
        let out = SiglipVisionOutput {
            last_hidden_state: hs[2].clone(),
            hidden_states: hs,
        };
        assert_eq!(select_vision_feature(&out, -2).unwrap().as_slice::<f32>(), &[2.0]);
        assert!(select_vision_feature(&out, -4).is_err());
        assert_eq!(select_vision_feature(&out, 0).unwrap().as_slice::<f32>(), &[1.0]);
    }
}
