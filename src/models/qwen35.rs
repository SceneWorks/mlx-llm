//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) — the hybrid decoder (story sc-7628).
//!
//! Unlike the generic [`CausalLm`](super::llama::CausalLm) (all softmax attention over a growing KV
//! cache), this decoder interleaves two mixer types on a fixed schedule (`full_attention_interval`,
//! default 4 → **3 Gated DeltaNet linear-attention layers : 1 gated full-attention layer**):
//!
//! - [`GatedDeltaNet`] — linear attention carrying a fixed-size recurrent state (the verified
//!   primitives in [`crate::primitives::gated_delta`]): in-proj → short conv → q/k RMS-norm →
//!   gated delta recurrence → gated RMS-norm → out-proj.
//! - [`Qwen35Attention`] — grouped-query attention with **partial RoPE** (`partial_rotary_factor`,
//!   reusing the [`Rope::partial`] path), per-head q/k RMS-norm, and an **output gate** (the queries
//!   projection is doubled into `[queries ‖ gate]`, and the attended output is multiplied by
//!   `sigmoid(gate)` before the output projection).
//!
//! Each decoder layer is `input_layernorm → mixer → residual → post_attention_layernorm → MLP →
//! residual`. The MLP is a dense SwiGLU for the 27B; the 35B MoE bank is wired in sc-7630. The KV
//! cache (full-attn layers) and the recurrent [`DeltaNetCache`] (linear layers) live side by side in
//! a per-layer [`Qwen35Cache`]. RMSNorm weights follow the Qwen3-Next `(1 + weight)` convention.

use mlx_rs::ops::{add, concatenate_axis, multiply, rsqrt, sigmoid, split_sections, sum_axis};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::models::deepstack::deepstack_fused_decoder_layers;
use crate::primitives::attention::{sdpa_capped, AttnMask};
use crate::primitives::gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, linear, rms_norm, silu};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::Weights;

/// Cached decode runs in bf16 (matching the rest of the engine); the delta recurrence accumulates in
/// f32 (matching the reference GPU kernel) for stability.
const COMPUTE_DTYPE: Dtype = Dtype::Bfloat16;

/// Interleaved M-RoPE output of [`Qwen35Model::mrope_positions`]: the temporal / height / width
/// position rows (each length `S`) plus the `mrope_delta` (`max_position + 1 − len`) for continuing
/// positions after the prompt.
pub type MropePositions = (Vec<i32>, Vec<i32>, Vec<i32>, i32);

/// Mixture-of-Experts FFN parameters (`qwen3_5_moe`, the 35B-A3B). Every layer's dense MLP is
/// replaced by a sparse MoE block: a softmax router over `num_experts` experts (top-`experts_per_tok`
/// per token, weights renormalized to sum to 1) plus a sigmoid-gated always-on shared expert.
#[derive(Clone, Copy, Debug)]
pub struct MoeParams {
    pub num_experts: i32,
    pub experts_per_tok: usize,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
}

/// Parsed Qwen3.6 (`qwen3_5` / `qwen3_5_moe`) text-decoder configuration. Read from the nested
/// `text_config` of the VLM wrapper (or the top-level config if not wrapped).
#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub hidden_size: i32,
    pub num_layers: usize,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub max_position_embeddings: i32,
    pub tie_word_embeddings: bool,
    /// Every `full_attention_interval`-th layer (1-indexed) is full attention; the rest are linear.
    pub full_attention_interval: usize,
    // Linear (Gated DeltaNet) dims.
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    /// MoE FFN parameters when this is the MoE variant (`qwen3_5_moe`, 35B-A3B); `None` ⇒ dense MLP
    /// (`qwen3_5`, 27B).
    pub moe: Option<MoeParams>,
    /// Interleaved M-RoPE section `[t, h, w]` (`rope_parameters.mrope_section`, sums to
    /// `rotary_dim/2`); `None` ⇒ the even split from [`Qwen35Config::mrope_section_resolved`]. Drives
    /// the per-channel axis assignment for image (3-D) positions; irrelevant to the text path.
    pub mrope_section: Option<[i32; 3]>,
}

impl Qwen35Config {
    /// Parse from a `config.json` value, descending into `text_config` for the VLM wrapper.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let c = v.get("text_config").unwrap_or(v);
        let int = |k: &str| -> Option<i32> { c.get(k).and_then(|x| x.as_i64()).map(|x| x as i32) };
        let req = |k: &str| -> Result<i32> {
            int(k).ok_or_else(|| Error::Config(format!("qwen3_5 config.json missing `{k}`")))
        };
        let f32o = |k: &str| -> Option<f32> { c.get(k).and_then(|x| x.as_f64()).map(|x| x as f32) };
        // RoPE params moved into a `rope_parameters` sub-object in newer configs (Qwen3.6); read
        // there first, then a legacy top-level field, then the architecture default.
        let rope_f32 = |k: &str| -> Option<f32> {
            c.get("rope_parameters")
                .and_then(|rp| rp.get(k))
                .and_then(|x| x.as_f64())
                .or_else(|| c.get(k).and_then(|x| x.as_f64()))
                .map(|x| x as f32)
        };

        let hidden_size = req("hidden_size")?;
        let num_heads = req("num_attention_heads")?;
        // The MoE variant (`qwen3_5_moe`) has no dense `intermediate_size` — every layer is MoE —
        // so fall back to the per-expert width (unused on the MoE path, but keeps the field valid).
        let intermediate_size = int("intermediate_size")
            .or_else(|| int("moe_intermediate_size"))
            .unwrap_or(0);
        Ok(Self {
            hidden_size,
            num_layers: req("num_hidden_layers")? as usize,
            intermediate_size,
            num_heads,
            num_kv_heads: int("num_key_value_heads").unwrap_or(num_heads),
            head_dim: int("head_dim").unwrap_or(hidden_size / num_heads),
            vocab_size: req("vocab_size")?,
            rms_norm_eps: f32o("rms_norm_eps").unwrap_or(1e-6),
            rope_theta: rope_f32("rope_theta").unwrap_or(10_000_000.0),
            partial_rotary_factor: rope_f32("partial_rotary_factor").unwrap_or(0.25),
            max_position_embeddings: int("max_position_embeddings").unwrap_or(0),
            tie_word_embeddings: c
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            full_attention_interval: int("full_attention_interval").unwrap_or(4).max(1) as usize,
            linear_num_value_heads: req("linear_num_value_heads")?,
            linear_num_key_heads: req("linear_num_key_heads")?,
            linear_key_head_dim: req("linear_key_head_dim")?,
            linear_value_head_dim: req("linear_value_head_dim")?,
            linear_conv_kernel_dim: int("linear_conv_kernel_dim").unwrap_or(4),
            // MoE variant (`qwen3_5_moe`, 35B-A3B) iff `num_experts` is present.
            moe: int("num_experts").map(|num_experts| MoeParams {
                num_experts,
                experts_per_tok: int("num_experts_per_tok").unwrap_or(8).max(1) as usize,
                moe_intermediate_size: int("moe_intermediate_size").unwrap_or(intermediate_size),
                shared_expert_intermediate_size: int("shared_expert_intermediate_size")
                    .unwrap_or(intermediate_size),
            }),
            mrope_section: c
                .get("rope_parameters")
                .and_then(|rp| rp.get("mrope_section"))
                .and_then(|x| x.as_array())
                .filter(|a| a.len() == 3)
                .map(|a| {
                    let g = |i: usize| a[i].as_i64().unwrap_or(0) as i32;
                    [g(0), g(1), g(2)]
                }),
        })
    }

    /// The interleaved M-RoPE section `[t, h, w]`, defaulting to an even split of `rotary_dim/2` when
    /// the config omits it (e.g. text-only checkpoints — where the section is moot). The order biases
    /// the remainder toward `t` then `h` (matching the released `[11, 11, 10]` for `rotary_dim/2 = 32`).
    pub fn mrope_section_resolved(&self) -> [usize; 3] {
        if let Some(s) = self.mrope_section {
            return [s[0].max(0) as usize, s[1].max(0) as usize, s[2].max(0) as usize];
        }
        let half = (self.rotary_dim() / 2) as usize;
        let base = half / 3;
        let rem = half % 3;
        [base + (rem > 0) as usize, base + (rem > 1) as usize, base]
    }

    /// Whether layer `i` (0-indexed) is a linear (Gated DeltaNet) layer; otherwise full attention.
    pub fn is_linear(&self, i: usize) -> bool {
        !(i + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Number of head dimensions partial RoPE rotates (even).
    pub fn rotary_dim(&self) -> i32 {
        let rd = (self.head_dim as f32 * self.partial_rotary_factor).round() as i32;
        rd & !1
    }
}

/// Multiply `x` by an f32 scalar (cast to `x`'s dtype).
fn scale(x: &Array, c: f32) -> Result<Array> {
    Ok(multiply(x, &Array::from_f32(c).as_dtype(x.dtype())?)?)
}

/// Zeros `[b, len, c]` in `dtype`.
fn zeros3(b: i32, len: i32, c: i32, dtype: Dtype) -> Result<Array> {
    let n = (b * len * c) as usize;
    Ok(Array::from_slice(&vec![0.0f32; n], &[b, len, c]).as_dtype(dtype)?)
}

/// L2-normalize over the last axis: `x · rsqrt(Σ x² + eps)` (the FLA `use_qk_l2norm_in_kernel`
/// convention — `eps` is added to the **sum**, not the mean). Computed in `x`'s dtype, matching the
/// reference kernel which normalizes the projected q/k before the recurrence.
fn l2norm(x: &Array, eps: f32) -> Result<Array> {
    let ss = sum_axis(&multiply(x, x)?, -1, true)?; // Σ x²  → [.., 1]
    let inv = rsqrt(&add(&ss, &Array::from_f32(eps).as_dtype(ss.dtype())?)?)?;
    Ok(multiply(x, &inv)?)
}

/// The Gated DeltaNet linear-attention layer (`Qwen3_5GatedDeltaNet`).
///
/// The Qwen3.6 checkpoint splits the input projection **four ways** — `in_proj_qkv` (fused q‖k‖v,
/// the only part the short conv mixes), `in_proj_z` (the output gate), and the per-value-head
/// `in_proj_a` / `in_proj_b` (decay / delta-strength) — rather than the fused `in_proj_qkvz` /
/// `in_proj_ba` of the older Qwen3-Next. After the conv, q/k/v are a **contiguous** split of the
/// `[key_dim, key_dim, value_dim]` channels (no head interleaving).
#[derive(Debug)]
struct GatedDeltaNet {
    in_proj_qkv: Projection, // [key_dim·2 + value_dim, hidden]  → conv'd
    in_proj_z: Projection,   // [value_dim, hidden]              → output gate
    in_proj_a: Projection,   // [Hv, hidden]                     → decay input
    in_proj_b: Projection,   // [Hv, hidden]                     → delta-strength input
    conv_weight: Array,      // [conv_dim, K]
    a_log: Array,            // [Hv]
    dt_bias: Array,          // [Hv]
    norm_weight: Array,      // [Dv] (RMSNormGated; loaded directly, ones-centered)
    out_proj: Projection,
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    value_dim: i32,
    conv_dim: i32,
    conv_kernel: i32,
    eps: f32,
}

impl GatedDeltaNet {
    fn forward(&self, x: &Array, cache: &mut DeltaNetCache) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        // Four independent in-projections (dtype follows the projection weights).
        let mixed = self.in_proj_qkv.forward(x)?; // [B,S,conv_dim] = q‖k‖v channels
        let dt = mixed.dtype();
        let z = self
            .in_proj_z
            .forward(x)?
            .reshape(&[b, s, self.num_v_heads, self.head_v_dim])?; // output gate
        let a_in = self.in_proj_a.forward(x)?; // [B,S,Hv]  decay input
        let b_in = self.in_proj_b.forward(x)?; // [B,S,Hv]  delta-strength input

        // Short conv over the q‖k‖v channels (only these are convolved), seeded by the cache tail,
        // then a *contiguous* split into q [key_dim] ‖ k [key_dim] ‖ v [value_dim] and reshape to heads.
        let conv_state = match &cache.conv_state {
            Some(cs) => cs.clone(),
            None => zeros3(b, self.conv_kernel - 1, self.conv_dim, dt)?,
        };
        let (conv_out, new_conv) = causal_depthwise_conv(&mixed, &self.conv_weight, &conv_state)?;
        let cp = split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], 2)?;
        let qc = cp[0].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let kc = cp[1].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let vc = cp[2].reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;

        // L2-normalize q/k (eps 1e-6), then scale q by 1/√head_k_dim — `use_qk_l2norm_in_kernel`.
        let inv = (self.head_k_dim as f32).powf(-0.5);
        let qn = scale(&l2norm(&qc, 1e-6)?, inv)?;
        let kn = l2norm(&kc, 1e-6)?;

        // The gated delta recurrence, accumulated in f32 (matching the reference kernel). GQA
        // (q/k from Hk key heads → Hv value heads) is handled inside the recurrence primitive.
        let beta = sigmoid(&b_in)?;
        let g = compute_g(&a_in, &self.a_log, &self.dt_bias)?;
        let f32 = Dtype::Float32;
        let (y, new_ssm) = gated_delta_recurrence(
            &qn.as_dtype(f32)?,
            &kn.as_dtype(f32)?,
            &vc.as_dtype(f32)?,
            &g.as_dtype(f32)?,
            &beta.as_dtype(f32)?,
            cache.ssm_state.as_ref(),
        )?;

        // Gated RMS-norm with z (back in the layer dtype), then the output projection.
        let out = rms_norm_gated(&y.as_dtype(dt)?, &self.norm_weight, &z, self.eps)?;
        let result = self.out_proj.forward(&out.reshape(&[b, s, self.value_dim])?)?;
        cache.update(new_conv, new_ssm, s);
        Ok(result)
    }
}

/// The gated full-attention layer (`Qwen3NextAttention`).
#[derive(Debug)]
struct Qwen35Attention {
    q_proj: Projection, // out = num_heads · head_dim · 2 (queries ‖ gate)
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: Array,
    k_norm: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Qwen35Attention {
    fn forward(&self, x: &Array, cos: &Array, sin: &Array, cache: &mut AttnKv) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let qg = self
            .q_proj
            .forward(x)?
            .reshape(&[b, s, self.num_heads, 2 * self.head_dim])?;
        let qp = split_sections(&qg, &[self.head_dim], 3)?;
        let q = rms_norm(&qp[0], &self.q_norm, self.eps)?; // [B,S,H,hd]
        let gate = qp[1].reshape(&[b, s, self.num_heads * self.head_dim])?;

        let k = rms_norm(
            &self.k_proj.forward(x)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?,
            &self.k_norm,
            self.eps,
        )?;
        let v = self.v_proj.forward(x)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // Partial RoPE (NeoX), then transpose into head-major [B,H,S,hd].
        let q = apply_rope(&q, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, cos, sin, false)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let (k_all, v_all) = cache.update(&k, &v)?;
        let out = sdpa_capped(&q, &k_all, &v_all, self.scale, None, AttnMask::Causal)?; // [B,H,S,hd]
        let merged = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.num_heads * self.head_dim])?;
        // Output gate: multiply by sigmoid(gate) before the output projection.
        let gated = multiply(&merged, &sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }
}

/// Dense SwiGLU MLP (`Qwen3_5MLP`) — the 27B FFN and each 35B expert / shared expert.
#[derive(Debug)]
struct Mlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl Mlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&multiply(&gate, &up)?)
    }
}

/// Sparse Mixture-of-Experts FFN (`Qwen3_5MoeSparseMoeBlock`, the 35B-A3B): a softmax router over
/// `experts` (top-`experts_per_tok` per token, weights renormalized to sum to 1) plus an always-on
/// **sigmoid-gated** shared expert. Each expert runs only on its routed tokens (gathered, then
/// scatter-added back), so active compute scales with `experts_per_tok` (≈3B of 35B). The fused
/// checkpoint tensors (`experts.gate_up_proj` / `experts.down_proj`) are un-fused into per-expert
/// [`Mlp`]s at load.
#[derive(Debug)]
struct MoeFfn {
    router: Array, // [num_experts, hidden]
    experts: Vec<Mlp>,
    shared: Mlp,
    shared_gate: Array, // [1, hidden] sigmoid gate
    experts_per_tok: usize,
}

impl MoeFfn {
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s, h) = (sh[0], sh[1], sh[2]);
        let t = b * s;
        let dtype = x.dtype();
        let xf = x.reshape(&[t, h])?;
        let num_experts = self.experts.len();
        let k = self.experts_per_tok.min(num_experts).max(1);

        // Router probabilities (f32 softmax on the host, for a stable top-k), then invert the
        // per-token top-k into per-expert (token, weight) lists with the weights renormalized to 1.
        let logits = linear(&xf, &self.router, None)?; // [t, num_experts]
        let logits = logits.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let mut routed: Vec<Vec<(i32, f32)>> = vec![Vec::new(); num_experts];
        for ti in 0..t as usize {
            let row = &logits[ti * num_experts..(ti + 1) * num_experts];
            let m = row.iter().copied().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
            let mut idx: Vec<usize> = (0..num_experts).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            let top = &idx[..k];
            let denom = top.iter().map(|&e| probs[e]).sum::<f32>().max(f32::MIN_POSITIVE);
            for &e in top {
                routed[e].push((ti as i32, probs[e] / denom));
            }
        }

        // Each expert runs on just its tokens; scatter the weighted outputs back.
        let mut out = zeros3(t, 1, h, dtype)?.reshape(&[t, h])?;
        for (e, toks) in routed.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len() as i32;
            let idx_i: Vec<i32> = toks.iter().map(|&(ti, _)| ti).collect();
            let idx_u: Vec<u32> = toks.iter().map(|&(ti, _)| ti as u32).collect();
            let wts: Vec<f32> = toks.iter().map(|&(_, w)| w).collect();
            let idx = Array::from_slice(&idx_i, &[n]);
            let idx_u = Array::from_slice(&idx_u, &[n]);
            let wts = Array::from_slice(&wts, &[n, 1]).as_dtype(dtype)?;
            let xe = xf.take_axis(&idx, 0)?; // [n, h]
            let ye = multiply(&self.experts[e].forward(&xe)?, &wts)?.reshape(&[n, 1, h])?;
            out = mlx_rs::ops::indexing::scatter_add_single(&out, &idx_u, &ye, 0)?;
        }

        // Always-on shared expert, gated by sigmoid(x · shared_gateᵀ).
        let shared = self.shared.forward(&xf)?;
        let sg = sigmoid(&linear(&xf, &self.shared_gate, None)?)?; // [t, 1]
        let shared = multiply(&shared, &sg)?;
        Ok(add(&out, &shared)?.reshape(&[b, s, h])?)
    }
}

/// The per-layer FFN: a dense SwiGLU (27B) or a sparse MoE block (35B-A3B).
#[derive(Debug)]
enum Ffn {
    Dense(Mlp),
    Moe(MoeFfn),
}

impl Ffn {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

#[derive(Debug)]
enum Mixer {
    Delta(GatedDeltaNet),
    Attn(Qwen35Attention),
}

#[derive(Debug)]
struct DecoderLayer {
    input_ln: Array,
    post_ln: Array,
    mixer: Mixer,
    ffn: Ffn,
    eps: f32,
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut Qwen35LayerCache,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let r = match (&self.mixer, cache) {
            (Mixer::Delta(d), Qwen35LayerCache::Delta(c)) => d.forward(&normed, c)?,
            (Mixer::Attn(a), Qwen35LayerCache::Attn(c)) => a.forward(&normed, cos, sin, c)?,
            _ => return Err(Error::Msg("qwen3_5: cache/mixer type mismatch".into())),
        };
        let h = add(x, &r)?;
        let m = self.ffn.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
        Ok(add(&h, &m)?)
    }
}

/// A single full-attention layer's growing KV (the linear layers use [`DeltaNetCache`] instead).
#[derive(Clone, Debug, Default)]
pub struct AttnKv {
    kv: Option<(Array, Array)>,
}

impl AttnKv {
    fn update(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        let merged = match self.kv.take() {
            Some((pk, pv)) => (concatenate_axis(&[&pk, k], 2)?, concatenate_axis(&[&pv, v], 2)?),
            None => (k.clone(), v.clone()),
        };
        self.kv = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }

    fn offset(&self) -> i32 {
        self.kv.as_ref().map(|(k, _)| k.shape()[2]).unwrap_or(0)
    }
}

/// The per-layer cache slot — a recurrent [`DeltaNetCache`] for linear layers, growing KV for
/// full-attention layers.
#[derive(Clone, Debug)]
pub enum Qwen35LayerCache {
    Delta(DeltaNetCache),
    Attn(AttnKv),
}

/// The hybrid decoder's cache: one slot per decoder layer.
#[derive(Clone, Debug)]
pub struct Qwen35Cache {
    layers: Vec<Qwen35LayerCache>,
}

impl Qwen35Cache {
    /// Positions already cached — the RoPE offset for the next step (read from the first full-attn
    /// layer; all layers advance in lockstep).
    pub fn offset(&self) -> i32 {
        self.layers
            .iter()
            .find_map(|l| match l {
                Qwen35LayerCache::Attn(a) => Some(a.offset()),
                Qwen35LayerCache::Delta(_) => None,
            })
            .unwrap_or(0)
    }

    /// Drop all cached state.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            match l {
                Qwen35LayerCache::Delta(c) => c.reset(),
                Qwen35LayerCache::Attn(a) => a.kv = None,
            }
        }
    }
}

/// A loaded Qwen3.6 (`qwen3_5`) hybrid decoder.
#[derive(Debug)]
pub struct Qwen35Model {
    embed_tokens: Array,
    layers: Vec<DecoderLayer>,
    norm: Array,
    lm_head: Array,
    rope: Rope,
    cfg: Qwen35Config,
    eps: f32,
    quantized: bool,
}

impl Qwen35Model {
    /// The parsed config.
    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
    }

    /// Whether the large projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// A fresh per-layer cache (linear vs full-attn slot per the schedule).
    pub fn new_cache(&self) -> Qwen35Cache {
        let layers = (0..self.cfg.num_layers)
            .map(|i| {
                if self.cfg.is_linear(i) {
                    Qwen35LayerCache::Delta(DeltaNetCache::new())
                } else {
                    Qwen35LayerCache::Attn(AttnKv::default())
                }
            })
            .collect();
        Qwen35Cache { layers }
    }

    /// Run the decoder stack over `input_ids` `[B, S]` at sequence `offset`, returning the final
    /// hidden states `[B, S, hidden]` (before the final norm / lm_head).
    fn hidden(&self, input_ids: &Array, cache: &mut Qwen35Cache, offset: i32) -> Result<Array> {
        let h = embed(&self.embed_tokens, input_ids)?.as_dtype(COMPUTE_DTYPE)?;
        let s = h.shape()[1];
        let (cos, sin) = self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?;
        self.hidden_from_embeds(&h, &cos, &sin, cache)
    }

    /// Run the decoder stack over precomputed input `embeds` `[B, S, hidden]` with the given RoPE
    /// tables, returning the final hidden states `[B, S, hidden]`. The token-id path ([`Self::hidden`])
    /// and the multimodal embeds path ([`Self::decode_logits_from_embeds`]) share this.
    fn hidden_from_embeds(
        &self,
        embeds: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut Qwen35Cache,
    ) -> Result<Array> {
        let mut h = embeds.clone();
        for (layer, slot) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(&h, cos, sin, slot)?;
        }
        Ok(h)
    }

    /// Final RMSNorm + `lm_head` over hidden states `[B, n, hidden]` → logits `[B, n, vocab]`.
    fn project(&self, h: &Array) -> Result<Array> {
        let normed = rms_norm(h, &self.norm, self.eps)?;
        linear(&normed, &self.lm_head, None)
    }

    /// Project the **last** position of `h` `[B, S, hidden]` → logits `[B, vocab]`.
    fn project_last(&self, h: &Array) -> Result<Array> {
        let s = h.shape()[1];
        let last = h.take_axis(Array::from_slice(&[s - 1], &[1]), 1)?; // [B,1,hidden]
        let logits = self.project(&last)?; // [B,1,vocab]
        Ok(logits.reshape(&[logits.shape()[0], self.cfg.vocab_size])?)
    }

    /// Run the decoder over `input_ids` `[B, S]` at sequence `offset`, returning logits for **every**
    /// position `[B, S, vocab]`.
    pub fn forward(&self, input_ids: &Array, cache: &mut Qwen35Cache, offset: i32) -> Result<Array> {
        let h = self.hidden(input_ids, cache, offset)?;
        self.project(&h)
    }

    /// Run the decoder and return logits for the **last** position only, `[B, vocab]` — the
    /// [`crate::decode::Decode::step`] contract (prefill + single-token decode).
    pub fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut Qwen35Cache,
        offset: i32,
    ) -> Result<Array> {
        let h = self.hidden(input_ids, cache, offset)?;
        self.project_last(&h)
    }

    /// Embed token ids `[B, S]` → `[B, S, hidden]` in the compute dtype — the splice point where the
    /// multimodal path overwrites image-token rows with the encoder's projected patch features
    /// ([`Self::splice_image_features`]).
    pub fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        Ok(embed(&self.embed_tokens, input_ids)?.as_dtype(COMPUTE_DTYPE)?)
    }

    /// Replace the `image_token_id` rows of `embeds` `[1, S, hidden]` with `image_features`
    /// `[num_image_tokens, hidden]` (the vision encoder's projected, merged patch rows), in sequence
    /// order. The number of image-token positions must equal the feature-row count.
    pub fn splice_image_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        image_features: &Array,
        image_token_id: i32,
    ) -> Result<Array> {
        self.splice_vision_features(embeds, input_ids, image_features, &[image_token_id])
    }

    /// Replace every row whose id is **any** of `placeholder_tokens` (`<|image_pad|>` and/or
    /// `<|video_pad|>`) with the next `vision_features` row, in sequence order — the multimodal splice
    /// for a mixed image+video prompt. Features must be concatenated in the same order the
    /// placeholders appear. Reduces to [`Self::splice_image_features`] for a single token.
    pub fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        let hidden = self.cfg.hidden_size;
        let s = embeds.shape()[1] as usize;
        let feats = vision_features.as_dtype(COMPUTE_DTYPE)?;
        let is_vis = |id: i32| placeholder_tokens.contains(&id);
        let num_vis = input_ids.iter().filter(|&&x| is_vis(x)).count() as i32;
        if num_vis != feats.shape()[0] {
            return Err(Error::Msg(format!(
                "qwen3_5 splice: {num_vis} vision tokens != {} feature rows",
                feats.shape()[0]
            )));
        }
        if num_vis == 0 {
            return Ok(embeds.clone());
        }
        // Stitch text spans (from `embeds`) and vision spans (from `feats`) in order — no scatter.
        let mut pieces: Vec<Array> = Vec::new();
        let mut feat_off = 0i32;
        let mut i = 0usize;
        while i < s {
            let vis = is_vis(input_ids[i]);
            let mut j = i;
            while j < s && is_vis(input_ids[j]) == vis {
                j += 1;
            }
            let n = (j - i) as i32;
            if vis {
                let idx = Array::from_slice(&(feat_off..feat_off + n).collect::<Vec<_>>(), &[n]);
                pieces.push(feats.take_axis(&idx, 0)?.reshape(&[1, n, hidden])?);
                feat_off += n;
            } else {
                let idx = Array::from_slice(&(i as i32..j as i32).collect::<Vec<_>>(), &[n]);
                pieces.push(embeds.take_axis(&idx, 1)?);
            }
            i = j;
        }
        let refs: Vec<&Array> = pieces.iter().collect();
        Ok(concatenate_axis(&refs, 1)?)
    }

    /// Compute the interleaved M-RoPE 3-D position rows (`get_rope_index`, B=1) for `input_ids`
    /// containing `image_grid_thw`-described `image_token_id` runs, plus the `mrope_delta`
    /// (`max_position + 1 − len`) the decode loop adds to continue positions after the prompt.
    ///
    /// The image-only entry point — the multimodal prefill path. See
    /// [`Self::mrope_positions_mm`] for the general (image + video) port; this forwards to it with
    /// no video runs.
    pub fn mrope_positions(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        self.mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            &[],
            image_token_id, // unused: no video runs
            spatial_merge_size,
        )
    }

    /// The full Qwen3-VL `get_rope_index` port (B=1), covering **both** image and video vision runs.
    ///
    /// Text tokens advance all three axes (t,h,w) by 1. A vision run lays its tokens over the
    /// `(t, h/merge, w/merge)` grid — temporal index per frame, height = row, width = col — offset by
    /// the shared cursor `st_idx` (`= max(prev block) + 1`), then advances the cursor by
    /// `max(grid_t, h/merge, w/merge)` (the reference's `llm_positions.max() + 1`).
    ///
    /// **Qwen3-VL video is the synthetic time axis.** Unlike a single multi-`t` block, Qwen3-VL
    /// separates frames with timestamp text, so the processor emits one `video_token_id` run **per
    /// frame** and the model splits `video_grid_thw` via `repeat_interleave(t); t ← 1`. Each frame is
    /// thus its own `gt = 1` block, and the temporal index **resets to 0 each frame** (the frames are
    /// ordered only by the advancing cursor). We mirror that by expanding each `[t, h, w]` video grid
    /// into `t` per-frame `[1, h, w]` blocks, one per consecutive video-token run. Image grids are
    /// consumed one run per `image_grid_thw` entry (Qwen3-VL images are always `gt = 1`).
    ///
    /// `spatial_merge_size` comes from the vision config. Returns `(t_row, h_row, w_row, mrope_delta)`.
    pub fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        let merge = spatial_merge_size.max(1);
        // Per-frame video blocks: `[t, h, w]` → `t × [1, h, w]` (the timestamp frame-split). Image
        // blocks pass through unchanged (always `gt = 1`).
        let mut video_frames: Vec<[i32; 3]> = Vec::new();
        for &[t, h, w] in video_grid_thw {
            if t <= 0 {
                return Err(Error::Msg(format!("qwen3_5 mrope: bad video grid {:?}", [t, h, w])));
            }
            for _ in 0..t {
                video_frames.push([1, h, w]);
            }
        }

        let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let mut cur = 0i32;
        let (mut img_i, mut vid_i) = (0usize, 0usize);
        let mut i = 0usize;
        while i < input_ids.len() {
            let id = input_ids[i];
            let is_image = id == image_token_id;
            // `image_token_id == video_token_id` would be ambiguous; the configs always differ, and
            // an image run is matched first.
            let is_video = !is_image && id == video_token_id;
            if is_image || is_video {
                let (grid, label): ([i32; 3], &str) = if is_image {
                    let g = *image_grid_thw.get(img_i).ok_or_else(|| {
                        Error::Msg("qwen3_5 mrope: more image runs than image_grid_thw entries".into())
                    })?;
                    img_i += 1;
                    (g, "image")
                } else {
                    let g = *video_frames.get(vid_i).ok_or_else(|| {
                        Error::Msg("qwen3_5 mrope: more video frame runs than video_grid_thw frames".into())
                    })?;
                    vid_i += 1;
                    (g, "video")
                };
                let (gt, gh, gw) = (grid[0], grid[1] / merge, grid[2] / merge);
                if gt <= 0 || gh <= 0 || gw <= 0 {
                    return Err(Error::Msg(format!("qwen3_5 mrope: bad {label} grid {grid:?}")));
                }
                let count = (gt * gh * gw) as usize;
                let tok = id;
                let run = input_ids[i..].iter().take_while(|&&x| x == tok).count();
                if run != count {
                    return Err(Error::Msg(format!(
                        "qwen3_5 mrope: {label} run length {run} != grid tokens {count}"
                    )));
                }
                let frame = gh * gw;
                for k in 0..count as i32 {
                    t.push(k / frame + cur);
                    let rem = k % frame;
                    h.push(rem / gw + cur);
                    w.push(rem % gw + cur);
                }
                cur += gt.max(gh).max(gw);
                i += count;
            } else {
                t.push(cur);
                h.push(cur);
                w.push(cur);
                cur += 1;
                i += 1;
            }
        }
        let maxpos = t.iter().chain(h.iter()).chain(w.iter()).copied().max().unwrap_or(-1);
        let delta = maxpos + 1 - input_ids.len() as i32;
        Ok((t, h, w, delta))
    }

    /// Run the decoder over precomputed input `embeds` `[1, S, hidden]` (text embeds with image
    /// features spliced in) using **interleaved M-RoPE** from the explicit 3-D `positions`
    /// (temporal/height/width rows, each length `S`), returning last-position logits `[1, vocab]`.
    /// With all three rows equal (text-only) this is bit-identical to [`Self::decode_logits`].
    pub fn decode_logits_from_embeds(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
    ) -> Result<Array> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            COMPUTE_DTYPE,
        )?;
        let h = self.hidden_from_embeds(&embeds.as_dtype(COMPUTE_DTYPE)?, &cos, &sin, cache)?;
        self.project_last(&h)
    }

    /// Run the decoder over precomputed input `embeds` `[1, S, hidden]` with interleaved M-RoPE
    /// (the multimodal embeds path) **and DeepStack feature fusion**: after decoder layer `i`, for
    /// `i < deepstack.len()`, the `i`-th tapped/merged ViT feature set is added to the visual-token
    /// rows of the running hidden states (`visual_pos_mask[p]` marks an image-token position).
    ///
    /// This is the Qwen3-VL DeepStack seam (`Qwen3VLTextModel.forward` + `_deepstack_process`):
    /// `deepstack` carries the `deepstack_features` produced by the vision tower at its tap layers,
    /// projected to `out_hidden_size == hidden`, and is injected into the *first* `len(deepstack)`
    /// decoder layers (HF indexes by decoder position, not by the vision tap index). Returns
    /// last-position logits `[1, vocab]`. With an empty `deepstack` this equals
    /// [`Self::decode_logits_from_embeds`].
    pub fn decode_logits_from_embeds_with_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut Qwen35Cache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            COMPUTE_DTYPE,
        )?;
        let h0 = embeds.as_dtype(COMPUTE_DTYPE)?;
        let layers = &self.layers;
        let cache_layers = &mut cache.layers;
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            layers.len(),
            |i, h| layers[i].forward(h, &cos, &sin, &mut cache_layers[i]),
        )?;
        self.project_last(&h)
    }

    /// Build from a loaded checkpoint (dense). See [`Qwen35Model::from_weights_with`].
    pub fn from_weights(w: &Weights, prefix: &str, cfg: Qwen35Config) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the large projections on load.
    ///
    /// `prefix` is the **decoder root** path: keys are read as `{prefix}.embed_tokens.weight`,
    /// `{prefix}.norm.weight`, `{prefix}.layers.{i}.…`. For the VLM-wrapped Qwen3.6 checkpoint this
    /// is `model.language_model`; `lm_head.weight` lives at the **checkpoint root** (untied), not
    /// under the prefix. `quant` (Q4/Q8) is applied to the big matmuls (in/out projections,
    /// attention q/k/v/o, MLP); the per-head decay/delta projections, conv, `A_log`/`dt_bias`, and
    /// all norms stay dense.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: Qwen35Config,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        let eps = cfg.rms_norm_eps;
        let req = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?) };
        // Qwen3.6 RMSNorm weights are stored zero-centered → fold in the +1 (the (1 + weight)
        // convention). The gated DeltaNet norm is the exception: it is ones-centered, loaded raw.
        let norm_w = |key: String| -> Result<Array> {
            let t = req(key)?;
            Ok(add(&t, &Array::from_f32(1.0).as_dtype(t.dtype())?)?)
        };
        let proj_q = |key: String| -> Result<Projection> {
            Projection::load(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?, quant)
        };
        let proj_dense = |key: String| -> Result<Projection> {
            Projection::load(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?, None)
        };
        let dp = |s: &str| format!("{prefix}.{s}");

        let embed_tokens = req(dp("embed_tokens.weight"))?;
        let norm = norm_w(dp("norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req("lm_head.weight".to_string())?
        };

        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| dp(&format!("layers.{i}.{s}"));
            let mixer = if cfg.is_linear(i) {
                // conv1d.weight is [conv_dim, 1, K] (HF) → squeeze the singleton to [conv_dim, K].
                let conv_weight = req(lp("linear_attn.conv1d.weight"))?
                    .reshape(&[conv_dim, cfg.linear_conv_kernel_dim])?;
                Mixer::Delta(GatedDeltaNet {
                    in_proj_qkv: proj_q(lp("linear_attn.in_proj_qkv.weight"))?,
                    in_proj_z: proj_q(lp("linear_attn.in_proj_z.weight"))?,
                    in_proj_a: proj_dense(lp("linear_attn.in_proj_a.weight"))?,
                    in_proj_b: proj_dense(lp("linear_attn.in_proj_b.weight"))?,
                    conv_weight,
                    a_log: req(lp("linear_attn.A_log"))?,
                    dt_bias: req(lp("linear_attn.dt_bias"))?,
                    norm_weight: req(lp("linear_attn.norm.weight"))?,
                    out_proj: proj_q(lp("linear_attn.out_proj.weight"))?,
                    num_k_heads: cfg.linear_num_key_heads,
                    num_v_heads: cfg.linear_num_value_heads,
                    head_k_dim: cfg.linear_key_head_dim,
                    head_v_dim: cfg.linear_value_head_dim,
                    key_dim,
                    value_dim,
                    conv_dim,
                    conv_kernel: cfg.linear_conv_kernel_dim,
                    eps,
                })
            } else {
                Mixer::Attn(Qwen35Attention {
                    q_proj: proj_q(lp("self_attn.q_proj.weight"))?,
                    k_proj: proj_q(lp("self_attn.k_proj.weight"))?,
                    v_proj: proj_q(lp("self_attn.v_proj.weight"))?,
                    o_proj: proj_q(lp("self_attn.o_proj.weight"))?,
                    q_norm: norm_w(lp("self_attn.q_norm.weight"))?,
                    k_norm: norm_w(lp("self_attn.k_norm.weight"))?,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    scale: (cfg.head_dim as f32).powf(-0.5),
                    eps,
                })
            };
            let ffn = match &cfg.moe {
                // Dense SwiGLU (27B).
                None => Ffn::Dense(Mlp {
                    gate: proj_q(lp("mlp.gate_proj.weight"))?,
                    up: proj_q(lp("mlp.up_proj.weight"))?,
                    down: proj_q(lp("mlp.down_proj.weight"))?,
                }),
                // Sparse MoE (35B-A3B): un-fuse the stacked expert tensors into per-expert SwiGLUs.
                // `experts.gate_up_proj` is [E, 2·moe_inter, hidden] (gate rows ‖ up rows, matching
                // the reference `linear(x, gate_up_proj[e]).chunk(2, -1)`); `experts.down_proj` is
                // [E, hidden, moe_inter].
                Some(moe) => {
                    let h = cfg.hidden_size;
                    let mi = moe.moe_intermediate_size;
                    let gate_up = req(lp("mlp.experts.gate_up_proj"))?;
                    let down = req(lp("mlp.experts.down_proj"))?;
                    let mut experts = Vec::with_capacity(moe.num_experts as usize);
                    for e in 0..moe.num_experts {
                        let sel = Array::from_slice(&[e], &[1]);
                        let gu = gate_up.take_axis(&sel, 0)?.reshape(&[2 * mi, h])?;
                        let parts = split_sections(&gu, &[mi], 0)?; // [gate_w, up_w]
                        let dn = down.take_axis(&sel, 0)?.reshape(&[h, mi])?;
                        experts.push(Mlp {
                            gate: Projection::load(parts[0].clone(), quant)?,
                            up: Projection::load(parts[1].clone(), quant)?,
                            down: Projection::load(dn, quant)?,
                        });
                    }
                    Ffn::Moe(MoeFfn {
                        router: req(lp("mlp.gate.weight"))?,
                        experts,
                        shared: Mlp {
                            gate: proj_q(lp("mlp.shared_expert.gate_proj.weight"))?,
                            up: proj_q(lp("mlp.shared_expert.up_proj.weight"))?,
                            down: proj_q(lp("mlp.shared_expert.down_proj.weight"))?,
                        },
                        shared_gate: req(lp("mlp.shared_expert_gate.weight"))?,
                        experts_per_tok: moe.experts_per_tok,
                    })
                }
            };
            layers.push(DecoderLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp("post_attention_layernorm.weight"))?,
                mixer,
                ffn,
                eps,
            });
        }

        let rope = Rope::partial(cfg.rotary_dim(), cfg.rope_theta, false);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            eps,
            cfg,
            quantized: quant.is_some(),
        })
    }
}

impl KvCache for Qwen35Cache {
    fn offset(&self) -> i32 {
        Qwen35Cache::offset(self)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn batch_size(&self) -> i32 {
        self.layers
            .iter()
            .find_map(|l| match l {
                Qwen35LayerCache::Attn(a) => a.kv.as_ref().map(|(k, _)| k.shape()[0]),
                Qwen35LayerCache::Delta(_) => None,
            })
            .unwrap_or(0)
    }

    fn reset(&mut self) {
        Qwen35Cache::reset(self)
    }

    // The hybrid cache is driven natively by `Qwen35Model` (which downcasts via `as_any_mut`); the
    // softmax-only trait mutators below are never invoked through the trait object on this path.
    fn update(&mut self, _layer: usize, _keys: &Array, _values: &Array) -> Result<(Array, Array)> {
        Err(Error::Msg(
            "Qwen35Cache: generic KvCache::update is not supported (hybrid cache is driven natively)"
                .into(),
        ))
    }

    fn retain_sequences(&mut self, _keep: &[i32]) -> Result<()> {
        Err(Error::Msg("Qwen35Cache: retain_sequences not yet supported".into()))
    }

    fn truncate(&mut self, _len: i32) -> Result<()> {
        Err(Error::Msg("Qwen35Cache: truncate not yet supported".into()))
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl crate::decode::Decode for Qwen35Model {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array> {
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen35Cache>()
            .ok_or_else(|| Error::Msg("Qwen35Model::step: cache is not a Qwen35Cache".into()))?;
        self.decode_logits(input_ids, cache, offset)
    }
}

impl crate::models::VlmDecode for Qwen35Model {
    fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        Qwen35Model::embed_input_ids(self, input_ids)
    }

    fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        Qwen35Model::splice_vision_features(self, embeds, input_ids, vision_features, placeholder_tokens)
    }

    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions> {
        Qwen35Model::mrope_positions_mm(
            self,
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    fn prefill_with_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array> {
        // The hybrid decoder drives its own concrete cache; recover it (the same downcast
        // `Qwen35Model::step` performs) before the deepstack prefill.
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen35Cache>()
            .ok_or_else(|| {
                Error::Msg("Qwen35Model::prefill_with_deepstack: cache is not a Qwen35Cache".into())
            })?;
        self.decode_logits_from_embeds_with_deepstack(embeds, positions, cache, visual_pos_mask, deepstack)
    }
}

/// Expand a single vision placeholder token into `count` copies — the Qwen3-VL image/video
/// placeholder token expansion (`Qwen3VLProcessor`).
///
/// The chat template renders each visual as `<|vision_start|> <placeholder> <|vision_end|>` with a
/// **single** `placeholder_token` (image `151655` or video `151656`). The processor then replaces
/// that one placeholder with `grid_t · grid_h · grid_w / merge²` copies (`counts[k]` here — the
/// merged-patch count the vision tower emits for visual `k`), leaving the `vision_start` /
/// `vision_end` frame and all surrounding text untouched. The resulting ids line up one-to-one with
/// the spliced patch-feature rows ([`Qwen35Model::splice_image_features`]) and the M-RoPE layout
/// ([`Qwen35Model::mrope_positions_mm`]).
///
/// `counts` must have one entry per placeholder occurrence of `placeholder_token`, in sequence
/// order. Returns the expanded id stream.
pub fn expand_vision_placeholders(
    ids: &[i32],
    placeholder_token: i32,
    counts: &[usize],
) -> Result<Vec<i32>> {
    let n_placeholders = ids.iter().filter(|&&x| x == placeholder_token).count();
    if n_placeholders != counts.len() {
        return Err(Error::Msg(format!(
            "qwen3_5 token-expansion: {n_placeholders} placeholders for token {placeholder_token} \
             but {} counts supplied",
            counts.len()
        )));
    }
    let total: usize = counts.iter().sum();
    let mut out = Vec::with_capacity(ids.len() - n_placeholders + total);
    let mut ci = 0usize;
    for &id in ids {
        if id == placeholder_token {
            out.extend(std::iter::repeat_n(placeholder_token, counts[ci]));
            ci += 1;
        } else {
            out.push(id);
        }
    }
    Ok(out)
}

/// The merged-patch token count for a vision grid `[t, h, w]` (patch units): `t · h · w / merge²`
/// — the number of placeholder tokens the processor emits and the number of feature rows the vision
/// tower produces for that visual.
pub fn vision_merged_token_count(grid_thw: [i32; 3], spatial_merge_size: i32) -> usize {
    let merge = spatial_merge_size.max(1);
    let [t, h, w] = grid_thw;
    (t.max(0) * (h.max(0) / merge) * (w.max(0) / merge)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn cfg_json() -> serde_json::Value {
        // 4 layers → schedule (interval 4): layers 0,1,2 linear, layer 3 full attention.
        json!({
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "vocab_size": 50, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5, "max_position_embeddings": 128,
                "tie_word_embeddings": false, "full_attention_interval": 4,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4
            },
            "vision_config": { "model_type": "qwen3_5" }
        })
    }

    /// A deterministic small tensor `[shape]` (finite, non-degenerate).
    fn t(map: &mut HashMap<String, Array>, key: &str, shape: &[i32]) {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
        map.insert(key.to_string(), Array::from_slice(&data, shape));
    }

    fn synthetic_weights(cfg: &Qwen35Config) -> Weights {
        let h = cfg.hidden_size;
        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let mut m = HashMap::new();
        // Mirror the real VLM-wrapped layout: the text decoder nests under `model.language_model`,
        // with `lm_head.weight` at the checkpoint root.
        let pfx = "model.language_model";
        t(&mut m, &format!("{pfx}.embed_tokens.weight"), &[cfg.vocab_size, h]);
        t(&mut m, &format!("{pfx}.norm.weight"), &[h]);
        t(&mut m, "lm_head.weight", &[cfg.vocab_size, h]);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| format!("{pfx}.layers.{i}.{s}");
            t(&mut m, &lp("input_layernorm.weight"), &[h]);
            t(&mut m, &lp("post_attention_layernorm.weight"), &[h]);
            match &cfg.moe {
                None => {
                    t(&mut m, &lp("mlp.gate_proj.weight"), &[cfg.intermediate_size, h]);
                    t(&mut m, &lp("mlp.up_proj.weight"), &[cfg.intermediate_size, h]);
                    t(&mut m, &lp("mlp.down_proj.weight"), &[h, cfg.intermediate_size]);
                }
                Some(moe) => {
                    let mi = moe.moe_intermediate_size;
                    let si = moe.shared_expert_intermediate_size;
                    t(&mut m, &lp("mlp.experts.gate_up_proj"), &[moe.num_experts, 2 * mi, h]);
                    t(&mut m, &lp("mlp.experts.down_proj"), &[moe.num_experts, h, mi]);
                    t(&mut m, &lp("mlp.gate.weight"), &[moe.num_experts, h]);
                    t(&mut m, &lp("mlp.shared_expert.gate_proj.weight"), &[si, h]);
                    t(&mut m, &lp("mlp.shared_expert.up_proj.weight"), &[si, h]);
                    t(&mut m, &lp("mlp.shared_expert.down_proj.weight"), &[h, si]);
                    t(&mut m, &lp("mlp.shared_expert_gate.weight"), &[1, h]);
                }
            }
            if cfg.is_linear(i) {
                // 4-way split projections (real qwen3_5 layout).
                t(&mut m, &lp("linear_attn.in_proj_qkv.weight"), &[conv_dim, h]);
                t(&mut m, &lp("linear_attn.in_proj_z.weight"), &[value_dim, h]);
                t(&mut m, &lp("linear_attn.in_proj_a.weight"), &[cfg.linear_num_value_heads, h]);
                t(&mut m, &lp("linear_attn.in_proj_b.weight"), &[cfg.linear_num_value_heads, h]);
                t(&mut m, &lp("linear_attn.conv1d.weight"), &[conv_dim, 1, cfg.linear_conv_kernel_dim]);
                t(&mut m, &lp("linear_attn.A_log"), &[cfg.linear_num_value_heads]);
                t(&mut m, &lp("linear_attn.dt_bias"), &[cfg.linear_num_value_heads]);
                t(&mut m, &lp("linear_attn.norm.weight"), &[cfg.linear_value_head_dim]);
                t(&mut m, &lp("linear_attn.out_proj.weight"), &[h, value_dim]);
            } else {
                t(&mut m, &lp("self_attn.q_proj.weight"), &[cfg.num_heads * cfg.head_dim * 2, h]);
                t(&mut m, &lp("self_attn.k_proj.weight"), &[cfg.num_kv_heads * cfg.head_dim, h]);
                t(&mut m, &lp("self_attn.v_proj.weight"), &[cfg.num_kv_heads * cfg.head_dim, h]);
                t(&mut m, &lp("self_attn.o_proj.weight"), &[h, cfg.num_heads * cfg.head_dim]);
                t(&mut m, &lp("self_attn.q_norm.weight"), &[cfg.head_dim]);
                t(&mut m, &lp("self_attn.k_norm.weight"), &[cfg.head_dim]);
            }
        }
        Weights::from_map(m)
    }

    #[test]
    fn config_parses_and_schedules_3_linear_1_full() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.rotary_dim(), 4); // head_dim 8 * 0.5
        // 3 linear : 1 full.
        assert!(cfg.is_linear(0) && cfg.is_linear(1) && cfg.is_linear(2));
        assert!(!cfg.is_linear(3));
    }

    #[test]
    fn assembled_forward_produces_finite_logits() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let w = synthetic_weights(&cfg);
        let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
        let mut cache = model.new_cache();
        // A 5-token prefill.
        let ids = Array::from_slice(&[1i32, 7, 3, 42, 9], &[1, 5]);
        let logits = model.forward(&ids, &mut cache, 0).unwrap();
        assert_eq!(logits.shape(), &[1, 5, cfg.vocab_size]);
        for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
            assert!(x.is_finite(), "non-finite logit: {x}");
        }
        // The full-attention layer (layer 3) advanced the KV cache to 5 positions.
        assert_eq!(cache.offset(), 5);
    }

    #[test]
    fn decode_after_prefill_advances_cache() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let model =
            Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg.clone())
                .unwrap();
        let mut cache = model.new_cache();
        model.forward(&Array::from_slice(&[1i32, 2, 3], &[1, 3]), &mut cache, 0).unwrap();
        assert_eq!(cache.offset(), 3);
        // One decode step at offset 3.
        let logits = model.forward(&Array::from_slice(&[4i32], &[1, 1]), &mut cache, 3).unwrap();
        assert_eq!(logits.shape(), &[1, 1, cfg.vocab_size]);
        assert_eq!(cache.offset(), 4);
        for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
            assert!(x.is_finite());
        }
    }

    /// The whole Gated DeltaNet layer, validated against a numeric oracle from the exact
    /// `Qwen3_5GatedDeltaNet.forward` reference (4-way in-projection → short conv → contiguous q|k|v
    /// split → L2-norm + q-scale → GQA delta recurrence → gated RMS-norm(z) → out-proj).
    ///
    /// **Single token (S=1).** MLX routes f32 matmuls through an exact GEMV when M=1 but a
    /// reduced-precision (bf16-class) GEMM when M>1, so a multi-token f32 oracle floors at ~6e-3
    /// regardless of correctness. With S=1 every projection is a GEMV, so the whole layer runs in
    /// exact f32 and the match is tight (≈1e-5) — pinning the *structural* assembly precisely. The
    /// conv's multi-tap and the recurrence's multi-step carry are covered tightly by the primitive
    /// oracles (sc-7627); cross-token cache carry by `prefill_equals_stepwise_decode`. Regenerate the
    /// fixture with `/tmp/gen_s1.py` (see story sc-7629).
    #[test]
    fn deltanet_layer_matches_qwen3_5_reference() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("testdata/qwen35_deltanet_oracle.json")).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            json[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        // Dims mirror the generator (r = Hv/Hk = 2 GQA, single token).
        let (h, hk, hv, dk, dv) = (8i32, 2i32, 4i32, 4i32, 4i32);
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;
        let kk = 4i32;
        let (b, s) = (1i32, 1i32);

        let mk = |k: &str, shape: &[i32]| Array::from_slice(&arr(k), shape);
        let proj = |k: &str, shape: &[i32]| Projection::load(mk(k, shape), None).unwrap();
        let layer = GatedDeltaNet {
            in_proj_qkv: proj("in_proj_qkv", &[conv_dim, h]),
            in_proj_z: proj("in_proj_z", &[value_dim, h]),
            in_proj_a: proj("in_proj_a", &[hv, h]),
            in_proj_b: proj("in_proj_b", &[hv, h]),
            conv_weight: mk("conv_weight", &[conv_dim, kk]),
            a_log: mk("A_log", &[hv]),
            dt_bias: mk("dt_bias", &[hv]),
            norm_weight: mk("norm_weight", &[dv]),
            out_proj: proj("out_proj", &[h, value_dim]),
            num_k_heads: hk,
            num_v_heads: hv,
            head_k_dim: dk,
            head_v_dim: dv,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel: kk,
            eps: 1e-6,
        };

        let x = mk("x", &[b, s, h]);
        let mut cache = DeltaNetCache::new();
        let out = layer.forward(&x, &mut cache).unwrap();
        assert_eq!(out.shape(), &[b, s, h]);

        let got = out.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let exp = arr("expected_output");
        let md = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 2e-4, "deltanet layer vs reference: max abs diff {md}\n got {got:?}\n exp {exp:?}");

        // The cache advanced and holds both the recurrent and conv state for a follow-on decode step.
        assert_eq!(cache.offset(), s);
        assert!(cache.conv_state.is_some() && cache.ssm_state.is_some());
    }

    /// A model-level invariant the hybrid cache must satisfy: prefilling a sequence in one pass must
    /// produce the same final-token logits as feeding the tokens one at a time carrying the cache
    /// (conv tail + recurrent SSM state for linear layers, growing KV for full-attention). Run on the
    /// production bf16 path (MLX-vs-MLX, so no f32-GEMM-floor concern) — this is what guarantees the
    /// short conv and delta recurrence resume correctly at decode time on real weights.
    #[test]
    fn prefill_equals_stepwise_decode() {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let w = synthetic_weights(&cfg);
        let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
        let toks = [1i32, 7, 3, 42, 9, 2];

        // One-shot prefill of the whole sequence; keep the last-token logits.
        let mut c_pre = model.new_cache();
        let prefill = model
            .decode_logits(&Array::from_slice(&toks, &[1, toks.len() as i32]), &mut c_pre, 0)
            .unwrap();

        // Token-by-token with cache carry; keep the final step's logits.
        let mut c_step = model.new_cache();
        let mut last = None;
        for (i, &tok) in toks.iter().enumerate() {
            last = Some(
                model
                    .decode_logits(&Array::from_slice(&[tok], &[1, 1]), &mut c_step, i as i32)
                    .unwrap(),
            );
        }
        let step = last.unwrap();

        assert_eq!(c_pre.offset(), toks.len() as i32);
        assert_eq!(c_step.offset(), toks.len() as i32);
        let a = prefill.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let b = step.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        // Both bf16 paths; allow small per-op bf16 reorder noise, far below any structural error.
        assert!(md < 5e-2, "prefill vs stepwise last-token logits diverged: max abs diff {md}");
    }

    /// A MoE config (`qwen3_5_moe`, the 35B-A3B shape, scaled down): 6 experts, top-2, with a shared
    /// expert. Same 4-layer 3:1 mixer schedule as [`cfg_json`].
    fn cfg_json_moe() -> serde_json::Value {
        let mut v = cfg_json();
        let tc = v["text_config"].as_object_mut().unwrap();
        tc.insert("model_type".into(), json!("qwen3_5_moe_text"));
        tc.insert("num_experts".into(), json!(6));
        tc.insert("num_experts_per_tok".into(), json!(2));
        tc.insert("moe_intermediate_size".into(), json!(16));
        tc.insert("shared_expert_intermediate_size".into(), json!(16));
        v
    }

    /// The MoE FFN block, validated against a numeric oracle from the exact
    /// `Qwen3_5MoeSparseMoeBlock.forward` reference: softmax router → top-k → renormalize → per-expert
    /// SwiGLU (gathered/scattered) → sigmoid-gated shared expert. Builds the block via the same
    /// un-fuse path as the loader (`experts.gate_up_proj` → per-expert gate/up). Single token (S=1) so
    /// MLX runs the exact GEMV path and the match is tight; regenerate with `/tmp/gen_moe.py`.
    #[test]
    fn moe_ffn_matches_qwen3_5_moe_reference() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("testdata/qwen35_moe_oracle.json")).unwrap();
        let arr = |k: &str| -> Vec<f32> {
            json[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
        };
        let (h, e, k, mi) = (8i32, 6i32, 2usize, 4i32);
        let mk = |key: &str, shape: &[i32]| Array::from_slice(&arr(key), shape);
        let proj = |a: Array| Projection::load(a, None).unwrap();

        // Un-fuse experts.gate_up_proj / down_proj into per-expert SwiGLUs (mirrors the loader).
        let gate_up = mk("gate_up", &[e, 2 * mi, h]);
        let down = mk("down", &[e, h, mi]);
        let mut experts = Vec::new();
        for ei in 0..e {
            let sel = Array::from_slice(&[ei], &[1]);
            let gu = gate_up.take_axis(&sel, 0).unwrap().reshape(&[2 * mi, h]).unwrap();
            let parts = split_sections(&gu, &[mi], 0).unwrap();
            let dn = down.take_axis(&sel, 0).unwrap().reshape(&[h, mi]).unwrap();
            experts.push(Mlp { gate: proj(parts[0].clone()), up: proj(parts[1].clone()), down: proj(dn) });
        }
        let moe = MoeFfn {
            router: mk("router", &[e, h]),
            experts,
            shared: Mlp {
                gate: proj(mk("sh_gate", &[mi, h])),
                up: proj(mk("sh_up", &[mi, h])),
                down: proj(mk("sh_down", &[h, mi])),
            },
            shared_gate: mk("sh_gatew", &[1, h]),
            experts_per_tok: k,
        };

        let out = moe.forward(&mk("x", &[1, 1, h])).unwrap();
        assert_eq!(out.shape(), &[1, 1, h]);
        let got = out.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let exp = arr("expected_output");
        let md = got.iter().zip(&exp).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(md < 2e-4, "moe ffn vs reference: max abs diff {md}\n got {got:?}\n exp {exp:?}");
    }

    #[test]
    fn moe_model_forward_and_prefill_equals_stepwise() {
        let cfg = Qwen35Config::from_json(&cfg_json_moe()).unwrap();
        assert!(cfg.moe.is_some());
        assert_eq!(cfg.moe.unwrap().num_experts, 6);
        let model =
            Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg.clone())
                .unwrap();

        // Multi-token prefill exercises routing/scatter across tokens; logits are finite + shaped.
        let toks = [1i32, 7, 3, 42, 9];
        let mut c_pre = model.new_cache();
        let logits = model
            .forward(&Array::from_slice(&toks, &[1, toks.len() as i32]), &mut c_pre, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[1, toks.len() as i32, cfg.vocab_size]);
        for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
            assert!(x.is_finite(), "non-finite MoE logit");
        }

        // Prefill == stepwise decode over the hybrid cache, with the MoE FFN in the loop.
        let pre_last = model
            .decode_logits(&Array::from_slice(&toks, &[1, toks.len() as i32]), &mut model.new_cache(), 0)
            .unwrap();
        let mut c_step = model.new_cache();
        let mut last = None;
        for (i, &tok) in toks.iter().enumerate() {
            last = Some(
                model.decode_logits(&Array::from_slice(&[tok], &[1, 1]), &mut c_step, i as i32).unwrap(),
            );
        }
        let a = pre_last.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let b = last.unwrap().as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(md < 5e-2, "MoE prefill vs stepwise diverged: max abs diff {md}");
    }

    fn synthetic_model() -> (Qwen35Config, Qwen35Model) {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let w = synthetic_weights(&cfg);
        let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
        (cfg, model)
    }

    /// `mrope_positions` (the `get_rope_index` port) must reproduce the reference 3-D position rows +
    /// `mrope_delta` for an image+text sequence — exact integer index math (oracle /tmp/gen_mrope.py).
    #[test]
    fn mrope_positions_matches_reference() {
        let j: serde_json::Value =
            serde_json::from_str(include_str!("testdata/qwen35_mrope_oracle.json")).unwrap();
        let r = &j["rope_index"];
        let ints = |k: &str| -> Vec<i32> {
            r[k].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
        };
        let ids = ints("input_ids");
        let grid = {
            let g = &r["image_grid_thw"][0];
            vec![[g[0].as_i64().unwrap() as i32, g[1].as_i64().unwrap() as i32, g[2].as_i64().unwrap() as i32]]
        };
        let img_tok = r["image_token_id"].as_i64().unwrap() as i32;
        let merge = r["merge"].as_i64().unwrap() as i32;

        let (_cfg, model) = synthetic_model();
        let (t, h, w, delta) = model.mrope_positions(&ids, &grid, img_tok, merge).unwrap();
        assert_eq!(t, ints("t"));
        assert_eq!(h, ints("h"));
        assert_eq!(w, ints("w"));
        assert_eq!(delta, r["delta"].as_i64().unwrap() as i32);
    }

    // --- Qwen3-VL HF-backed oracles (tools/gen_qwen3vl_mrope_oracle.py) ----------------------------

    fn qwen3vl_mrope_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("testdata/qwen3vl_mrope_oracle.json")).unwrap()
    }

    fn ints_of(v: &serde_json::Value) -> Vec<i32> {
        v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
    }

    fn grids_of(v: &serde_json::Value) -> Vec<[i32; 3]> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|g| [g[0].as_i64().unwrap() as i32, g[1].as_i64().unwrap() as i32, g[2].as_i64().unwrap() as i32])
            .collect()
    }

    /// **Interleaved-MRoPE position ids — mixed text/image.** `mrope_positions_mm` must reproduce the
    /// real `Qwen3VLModel.get_rope_index` 3-D rows + `mrope_delta` for a text+image sequence, exact
    /// integer match. This is the Qwen3-VL placement (image block offset by the running cursor, cursor
    /// advanced by `max(t, h/merge, w/merge)`).
    #[test]
    fn qwen3vl_mrope_image_matches_hf_reference() {
        let j = qwen3vl_mrope_oracle();
        let r = &j["rope_index_image"];
        let ids = ints_of(&r["input_ids"]);
        let grid = grids_of(&r["image_grid_thw"]);
        let img = j["image_token_id"].as_i64().unwrap() as i32;
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (_cfg, model) = synthetic_model();
        let (t, h, w, delta) = model.mrope_positions(&ids, &grid, img, merge).unwrap();
        assert_eq!(t, ints_of(&r["t"]), "image t-row vs HF get_rope_index");
        assert_eq!(h, ints_of(&r["h"]), "image h-row vs HF get_rope_index");
        assert_eq!(w, ints_of(&r["w"]), "image w-row vs HF get_rope_index");
        assert_eq!(delta, r["delta"].as_i64().unwrap() as i32, "image mrope_delta");
    }

    /// **Interleaved-MRoPE position ids — synthetic time / multi-frame video axis.** Qwen3-VL splits a
    /// `[t, h, w]` video into `t` per-frame `gt = 1` blocks (timestamps separate frames), so the
    /// temporal index resets per frame and frames are ordered only by the advancing cursor.
    /// `mrope_positions_mm` must reproduce the HF rows + delta exactly for the 2-frame case — the
    /// Qwen3-VL-specific delta a single multi-`t` block would get wrong.
    #[test]
    fn qwen3vl_mrope_video_matches_hf_reference() {
        let j = qwen3vl_mrope_oracle();
        let r = &j["rope_index_video"];
        let ids = ints_of(&r["input_ids"]);
        let vgrid = grids_of(&r["video_grid_thw"]);
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let img = j["image_token_id"].as_i64().unwrap() as i32;
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (_cfg, model) = synthetic_model();
        let (t, h, w, delta) = model
            .mrope_positions_mm(&ids, &[], img, &vgrid, vid, merge)
            .unwrap();
        assert_eq!(t, ints_of(&r["t"]), "video t-row vs HF get_rope_index (per-frame reset)");
        assert_eq!(h, ints_of(&r["h"]), "video h-row vs HF get_rope_index");
        assert_eq!(w, ints_of(&r["w"]), "video w-row vs HF get_rope_index");
        assert_eq!(delta, r["delta"].as_i64().unwrap() as i32, "video mrope_delta");
    }

    /// **Interleaved-MRoPE table — end to end at Qwen3-VL config.** Build the partial-rotary RoPE at
    /// the real text config (head_dim 128, `rotary_dim` 128, theta 5e6, `mrope_section [24,20,20]`),
    /// feed it the HF rope rows for the image sequence, and require the interleaved cos/sin tables to
    /// match the reference `apply_interleaved_mrope` within 1e-5 (f32). Closes the loop from
    /// `get_rope_index` rows through the Qwen3-VL interleaving.
    #[test]
    fn qwen3vl_interleaved_cos_sin_matches_hf_reference() {
        let j = qwen3vl_mrope_oracle();
        let il = &j["interleaved"];
        let head_dim = j["head_dim"].as_i64().unwrap() as i32;
        let theta = j["rope_theta"].as_f64().unwrap() as f32;
        let section: Vec<usize> = j["mrope_section"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as usize)
            .collect();
        let sections = [section[0], section[1], section[2]];
        let (t, h, w) = (ints_of(&il["t"]), ints_of(&il["h"]), ints_of(&il["w"]));
        let expect = |k: &str| -> Vec<f32> {
            il[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
        };

        // Qwen3-VL text rope is full-rotary (partial_rotary_factor 1.0 ⇒ rotary_dim == head_dim),
        // NeoX half-split — exactly Rope::partial(head_dim, theta, false).
        let rope = crate::primitives::rope::Rope::partial(head_dim, theta, false);
        let (cos, sin) = rope
            .mrope_interleaved_cos_sin([&t, &h, &w], sections, Dtype::Float32)
            .unwrap();
        assert_eq!(cos.shape(), &[1, t.len() as i32, head_dim]);
        let cmp = |got: &[f32], exp: &[f32]| {
            assert_eq!(got.len(), exp.len(), "table length");
            got.iter().zip(exp).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
        };
        let dc = cmp(cos.as_slice::<f32>(), &expect("cos"));
        let ds = cmp(sin.as_slice::<f32>(), &expect("sin"));
        assert!(dc < 1e-5, "interleaved cos vs HF reference: max abs diff {dc}");
        assert!(ds < 1e-5, "interleaved sin vs HF reference: max abs diff {ds}");
    }

    /// **Image-placeholder token expansion matches the HF processor.** Given the raw chat ids with a
    /// single `<|image_pad|>` framed by `<|vision_start|>` / `<|vision_end|>`, expanding the
    /// placeholder to `grid.prod() / merge²` copies must reproduce the exact id stream
    /// `Qwen3VLProcessor` emits — same count, same vision framing, surrounding text untouched.
    #[test]
    fn qwen3vl_token_expansion_matches_hf_processor() {
        let j = qwen3vl_mrope_oracle();
        let ex = &j["expand"];
        let expanded_hf = ints_of(&ex["expanded_ids"]);
        let img = ex["image_token_id"].as_i64().unwrap() as i32;
        let vs = ex["vision_start_token_id"].as_i64().unwrap() as i32;
        let ve = ex["vision_end_token_id"].as_i64().unwrap() as i32;
        let g = ints_of(&ex["grid_thw"]); // single [t, h, w]
        let grid = [g[0], g[1], g[2]];
        let merge = ex["merge"].as_i64().unwrap() as i32;
        let expected_count = ex["expected_count"].as_i64().unwrap() as usize;

        // The count the vision tower / processor agree on.
        let count = vision_merged_token_count(grid, merge);
        assert_eq!(count, expected_count, "merged-token count formula vs HF processor");

        // Reconstruct the *raw* (pre-expansion) chat ids: the single image placeholder framed by
        // vision_start/vision_end, with all surrounding (non-image) tokens preserved in order. The HF
        // expanded ids are that with the placeholder repeated `count` times — so collapsing the image
        // run back to one token recovers the raw stream.
        let mut raw = Vec::new();
        let mut i = 0usize;
        while i < expanded_hf.len() {
            if expanded_hf[i] == img {
                raw.push(img); // one placeholder
                while i < expanded_hf.len() && expanded_hf[i] == img {
                    i += 1;
                }
            } else {
                raw.push(expanded_hf[i]);
                i += 1;
            }
        }

        let expanded = expand_vision_placeholders(&raw, img, &[count]).unwrap();
        assert_eq!(expanded, expanded_hf, "expanded ids vs HF processor");

        // The expansion is framed by exactly one vision_start … vision_end with `count` image tokens
        // between, and the count matches what the merger emits.
        let si = expanded.iter().position(|&x| x == vs).unwrap();
        let ei = expanded.iter().position(|&x| x == ve).unwrap();
        assert_eq!(ei - si - 1, count, "image tokens framed between vision_start/vision_end");
        assert_eq!(expanded[si + 1..ei].iter().filter(|&&x| x == img).count(), count);
    }

    /// **The text-path invariant.** Feeding token embeds + equal (text) 3-D positions through
    /// `decode_logits_from_embeds` must be **bit-identical** to the token-id `decode_logits` — the
    /// interleaved M-RoPE collapses to 1D and the embeds path is the same compute. This is the gate
    /// that the multimodal hook doesn't perturb the (verified) text decoder.
    #[test]
    fn decode_from_embeds_text_only_equals_decode_logits() {
        let (_cfg, model) = synthetic_model();
        let toks = [1i32, 7, 3, 42, 9, 2];
        let ids = Array::from_slice(&toks, &[1, toks.len() as i32]);

        let a = model.decode_logits(&ids, &mut model.new_cache(), 0).unwrap();
        let embeds = model.embed_input_ids(&ids).unwrap();
        let pos: Vec<i32> = (0..toks.len() as i32).collect();
        let b = model
            .decode_logits_from_embeds(&embeds, [&pos, &pos, &pos], &mut model.new_cache())
            .unwrap();

        assert_eq!(a.shape(), b.shape());
        let (av, bv) = (
            a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec(),
            b.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec(),
        );
        let md = av.iter().zip(&bv).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(md == 0.0, "embeds text path must equal token-id path bit-for-bit; max abs diff {md}");
    }

    /// The splice hook overwrites exactly the image-token rows (in order) with the feature rows and
    /// leaves text rows untouched.
    #[test]
    fn splice_image_features_replaces_image_rows() {
        let (cfg, model) = synthetic_model();
        let hidden = cfg.hidden_size as usize;
        let ids = [7i32, 49, 49, 8, 9]; // two image tokens (id 49) at positions 1,2
        // embeds[1,5,hidden]: row r filled with value r.
        let mut e = Vec::new();
        for r in 0..5 {
            e.extend(std::iter::repeat_n(r as f32, hidden));
        }
        let embeds = Array::from_slice(&e, &[1, 5, hidden as i32]).as_dtype(COMPUTE_DTYPE).unwrap();
        // feats[2,hidden]: row j filled with 100 + j.
        let mut f = Vec::new();
        for j in 0..2 {
            f.extend(std::iter::repeat_n(100.0f32 + j as f32, hidden));
        }
        let feats = Array::from_slice(&f, &[2, hidden as i32]);

        let out = model.splice_image_features(&embeds, &ids, &feats, 49).unwrap();
        assert_eq!(out.shape(), &[1, 5, hidden as i32]);
        let v = out.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let row = |r: usize| v[r * hidden]; // first element of each row (whole row is constant)
        assert_eq!([row(0), row(1), row(2), row(3), row(4)], [0.0, 100.0, 101.0, 3.0, 4.0]);
    }

    /// DeepStack fusion is actually wired through the decoder: feeding non-zero tapped features
    /// through `decode_logits_from_embeds_with_deepstack` must (a) run end to end to finite logits
    /// and (b) **differ** from the same prefill with the fusion disabled (empty taps) — proving the
    /// tapped features are consumed in the decoder layers, not computed and dropped.
    #[test]
    fn deepstack_fused_path_consumes_features_and_differs() {
        let (cfg, model) = synthetic_model();
        let img = 49i32;
        let ids = [1i32, 2, img, img, img, img, 3, 4];
        let grid = vec![[1i32, 4, 4]];
        let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);

        let embeds = model.embed_input_ids(&ids_arr).unwrap();
        let feats = Array::from_slice(
            &(0..4 * cfg.hidden_size).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect::<Vec<_>>(),
            &[4, cfg.hidden_size],
        );
        let spliced = model.splice_image_features(&embeds, &ids, &feats, img).unwrap();
        let (t, h, w, _delta) = model.mrope_positions(&ids, &grid, img, 2).unwrap();
        let visual_pos_mask: Vec<bool> = ids.iter().map(|&id| id == img).collect();

        let ds = |scale: f32| {
            Array::from_slice(
                &(0..4 * cfg.hidden_size).map(|i| (i % 5) as f32 * scale + 0.05).collect::<Vec<_>>(),
                &[4, cfg.hidden_size],
            )
        };
        let deepstack = [ds(0.2), ds(-0.15)];

        let fused = model
            .decode_logits_from_embeds_with_deepstack(
                &spliced,
                [&t, &h, &w],
                &mut model.new_cache(),
                &visual_pos_mask,
                &deepstack,
            )
            .unwrap();
        let unfused = model
            .decode_logits_from_embeds_with_deepstack(
                &spliced,
                [&t, &h, &w],
                &mut model.new_cache(),
                &visual_pos_mask,
                &[],
            )
            .unwrap();
        let baseline = model
            .decode_logits_from_embeds(&spliced, [&t, &h, &w], &mut model.new_cache())
            .unwrap();

        assert_eq!(fused.shape(), &[1, cfg.vocab_size]);
        let fv = fused.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let uv = unfused.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let bv = baseline.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        assert!(fv.iter().all(|x| x.is_finite()), "non-finite fused logit");

        let unfused_vs_baseline = uv.iter().zip(&bv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert_eq!(unfused_vs_baseline, 0.0, "empty-deepstack path must equal plain embeds path");

        let fused_vs_unfused = fv.iter().zip(&uv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(
            fused_vs_unfused > 1e-3,
            "DeepStack fusion did not change the logits (features dropped?): max abs diff {fused_vs_unfused}"
        );

        // The provider drives this decoder through the `VlmDecode` seam, which boxes the cache via
        // `Decode::make_cache` and downcasts it back to `Qwen35Cache` inside `prefill_with_deepstack`.
        // That trait path must reproduce the inherent fused logits bit-for-bit (same model, inputs,
        // and a fresh cache).
        let mut trait_cache = crate::decode::Decode::make_cache(&model);
        let via_trait = crate::models::VlmDecode::prefill_with_deepstack(
            &model,
            &spliced,
            [&t, &h, &w],
            trait_cache.as_mut(),
            &visual_pos_mask,
            &deepstack,
        )
        .unwrap();
        let tv = via_trait.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let trait_vs_inherent = fv.iter().zip(&tv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert_eq!(
            trait_vs_inherent, 0.0,
            "VlmDecode::prefill_with_deepstack must equal the inherent fused path (cache downcast seam)"
        );
    }

    #[test]
    fn deepstack_seam_is_decoder_agnostic_at_qwen3vl_shapes() {
        let hidden = 4096i32;
        let num_layers = 6usize;
        let taps = [8usize, 16, 24];
        let seq = 8i32;
        let visual_pos_mask: Vec<bool> = (0..seq).map(|i| (2..6).contains(&i)).collect();
        let num_visual = visual_pos_mask.iter().filter(|&&m| m).count() as i32;
        assert_eq!(num_visual, 4);

        let v0 = 0.5f32;
        let h0 = Array::from_slice(&vec![v0; (seq * hidden) as usize], &[1, seq, hidden])
            .as_dtype(COMPUTE_DTYPE)
            .unwrap();
        let feat_val = [0.25f32, 0.5, 0.75];
        let deepstack: Vec<Array> = (0..taps.len())
            .map(|t| {
                Array::from_slice(&vec![feat_val[t]; (num_visual * hidden) as usize], &[num_visual, hidden])
                    .as_dtype(COMPUTE_DTYPE)
                    .unwrap()
            })
            .collect();

        let two = Array::from_f32(2.0).as_dtype(COMPUTE_DTYPE).unwrap();
        let mut calls: Vec<usize> = Vec::new();
        let fused = deepstack_fused_decoder_layers(
            &h0,
            &visual_pos_mask,
            &deepstack,
            num_layers,
            |i, h| {
                calls.push(i);
                Ok(multiply(h, &two)?)
            },
        )
        .unwrap();
        assert_eq!(calls, (0..num_layers).collect::<Vec<_>>(), "every decoder layer must run once, in order");

        let unfused = deepstack_fused_decoder_layers(
            &h0,
            &visual_pos_mask,
            &[],
            num_layers,
            |_i, h| Ok(multiply(h, &two)?),
        )
        .unwrap();

        let scale = 2f32.powi(num_layers as i32);
        let text_expected = v0 * scale;
        let visual_expected: f32 = v0 * scale
            + (0..taps.len())
                .map(|t| feat_val[t] * 2f32.powi((num_layers - 1 - t) as i32))
                .sum::<f32>();

        let fv = fused.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let uv = unfused.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let row = |buf: &[f32], r: i32| buf[(r * hidden) as usize];
        for r in 0..seq {
            let got = row(&fv, r);
            let want = if visual_pos_mask[r as usize] { visual_expected } else { text_expected };
            assert!((got - want).abs() < 1e-1, "fused row {r}: got {got}, want {want}");
        }
        assert!((visual_expected - 54.0).abs() < 1e-3);
        assert!((text_expected - 32.0).abs() < 1e-3);

        for r in 0..seq {
            assert!((row(&uv, r) - text_expected).abs() < 1e-1, "unfused row {r} must skip injection");
        }
        let fused_vs_unfused = fv.iter().zip(&uv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(fused_vs_unfused > 1.0, "fused path must differ from non-fused: max abs diff {fused_vs_unfused}");
    }

    /// Smoke: the full image+text path (embed → splice features → M-RoPE positions →
    /// decode_logits_from_embeds) runs end to end and yields finite `[1, vocab]` logits.
    #[test]
    fn image_text_decode_from_embeds_runs() {
        let (cfg, model) = synthetic_model();
        let img = 49i32; // within the synthetic vocab (50) so embed gather is in-bounds
        let ids = [1i32, 2, img, img, img, img, 3, 4]; // 2x2 image (4 tokens) between text
        let grid = vec![[1i32, 4, 4]];
        let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);

        let embeds = model.embed_input_ids(&ids_arr).unwrap();
        let feats = Array::from_slice(
            &(0..4 * cfg.hidden_size).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect::<Vec<_>>(),
            &[4, cfg.hidden_size],
        );
        let spliced = model.splice_image_features(&embeds, &ids, &feats, img).unwrap();
        let (t, h, w, _delta) = model.mrope_positions(&ids, &grid, img, 2).unwrap();
        let logits = model
            .decode_logits_from_embeds(&spliced, [&t, &h, &w], &mut model.new_cache())
            .unwrap();
        assert_eq!(logits.shape(), &[1, cfg.vocab_size]);
        assert!(logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().iter().all(|x| x.is_finite()));
    }
}
