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

use mlx_rs::ops::{add, concatenate_axis, multiply, sigmoid, split_sections};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::primitives::attention::{sdpa_capped, AttnMask};
use crate::primitives::gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
use crate::primitives::nn::{embed, linear, rms_norm, silu};
use crate::primitives::projection::Projection;
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::Weights;

/// Cached decode runs in bf16 (matching the rest of the engine); the delta recurrence accumulates in
/// f32 (matching the reference GPU kernel) for stability.
const COMPUTE_DTYPE: Dtype = Dtype::Bfloat16;

/// Parsed Qwen3.6 (`qwen3_5`) text-decoder configuration. Read from the nested `text_config` of the
/// VLM wrapper (or the top-level config if not wrapped).
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

        let hidden_size = req("hidden_size")?;
        let num_heads = req("num_attention_heads")?;
        Ok(Self {
            hidden_size,
            num_layers: req("num_hidden_layers")? as usize,
            intermediate_size: req("intermediate_size")?,
            num_heads,
            num_kv_heads: int("num_key_value_heads").unwrap_or(num_heads),
            head_dim: int("head_dim").unwrap_or(hidden_size / num_heads),
            vocab_size: req("vocab_size")?,
            rms_norm_eps: f32o("rms_norm_eps").unwrap_or(1e-6),
            rope_theta: f32o("rope_theta").unwrap_or(10_000_000.0),
            partial_rotary_factor: f32o("partial_rotary_factor").unwrap_or(1.0),
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
        })
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

/// Zeros `[b, len, c]` in the compute dtype.
fn zeros_bf16(b: i32, len: i32, c: i32) -> Result<Array> {
    let n = (b * len * c) as usize;
    Ok(Array::from_slice(&vec![0.0f32; n], &[b, len, c]).as_dtype(COMPUTE_DTYPE)?)
}

/// The Gated DeltaNet linear-attention layer (`Qwen3NextGatedDeltaNet`).
#[derive(Debug)]
struct GatedDeltaNet {
    in_proj_qkvz: Projection,
    in_proj_ba: Projection,
    conv_weight: Array, // [conv_dim, K]
    a_log: Array,       // [Hv]
    dt_bias: Array,     // [Hv]
    norm_weight: Array, // [Dv] (RMSNormGated)
    out_proj: Projection,
    ones_k: Array, // [Dk] for the weightless q/k RMS-norm
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
        let r = self.num_v_heads / self.num_k_heads;

        // in-projections + fix_query_key_value_ordering: q,k [B,S,Hk,Dk]; v,z [B,S,Hv,Dv]; b,a [B,S,Hv].
        let qkvz = self
            .in_proj_qkvz
            .forward(x)?
            .reshape(&[b, s, self.num_k_heads, 2 * self.head_k_dim + 2 * r * self.head_v_dim])?;
        let cuts = [self.head_k_dim, 2 * self.head_k_dim, 2 * self.head_k_dim + r * self.head_v_dim];
        let p = split_sections(&qkvz, &cuts, 3)?;
        let (q, k) = (&p[0], &p[1]);
        let v = p[2].reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;
        let z = p[3].reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;
        let ba = self
            .in_proj_ba
            .forward(x)?
            .reshape(&[b, s, self.num_k_heads, 2 * r])?;
        let bap = split_sections(&ba, &[r], 3)?;
        let beta_in = bap[0].reshape(&[b, s, self.num_v_heads])?;
        let a_in = bap[1].reshape(&[b, s, self.num_v_heads])?;

        // Short conv over the concatenated q‖k‖v channels, seeded by the cache's conv tail.
        let mixed = concatenate_axis(
            &[
                &q.reshape(&[b, s, self.key_dim])?,
                &k.reshape(&[b, s, self.key_dim])?,
                &v.reshape(&[b, s, self.value_dim])?,
            ],
            2,
        )?;
        let conv_state = match &cache.conv_state {
            Some(cs) => cs.clone(),
            None => zeros_bf16(b, self.conv_kernel - 1, self.conv_dim)?,
        };
        let (conv_out, new_conv) = causal_depthwise_conv(&mixed, &self.conv_weight, &conv_state)?;
        let cp = split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], 2)?;
        let qc = cp[0].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let kc = cp[1].reshape(&[b, s, self.num_k_heads, self.head_k_dim])?;
        let vc = cp[2].reshape(&[b, s, self.num_v_heads, self.head_v_dim])?;

        // q = inv_scale² · rms_norm(q); k = inv_scale · rms_norm(k) (weightless, eps 1e-6).
        let inv = (self.head_k_dim as f32).powf(-0.5);
        let qn = scale(&rms_norm(&qc, &self.ones_k, 1e-6)?, inv * inv)?;
        let kn = scale(&rms_norm(&kc, &self.ones_k, 1e-6)?, inv)?;

        // The gated delta recurrence, accumulated in f32 (matching the reference kernel).
        let beta = sigmoid(&beta_in)?;
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

        // Gated RMS-norm with z, then the output projection.
        let y_c = y.as_dtype(COMPUTE_DTYPE)?;
        let out = rms_norm_gated(&y_c, &self.norm_weight, &z, self.eps)?;
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

/// Dense SwiGLU MLP (`Qwen3NextMLP`). The 35B MoE bank is wired in sc-7630.
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
    mlp: Mlp,
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
        let m = self.mlp.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
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
}

impl Qwen35Model {
    /// The parsed config.
    pub fn config(&self) -> &Qwen35Config {
        &self.cfg
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

    /// Run the decoder over `input_ids` `[B, S]` at sequence `offset`, returning logits
    /// `[B, S, vocab]`.
    pub fn forward(&self, input_ids: &Array, cache: &mut Qwen35Cache, offset: i32) -> Result<Array> {
        let mut h = embed(&self.embed_tokens, input_ids)?.as_dtype(COMPUTE_DTYPE)?;
        let s = h.shape()[1];
        let (cos, sin) = self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?;
        for (layer, slot) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward(&h, &cos, &sin, slot)?;
        }
        let normed = rms_norm(&h, &self.norm, self.eps)?;
        linear(&normed, &self.lm_head, None)
    }

    /// Build from a loaded checkpoint. `prefix` is the weight-key prefix (`""` for a plain decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: Qwen35Config) -> Result<Self> {
        let eps = cfg.rms_norm_eps;
        let req = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?) };
        // Qwen3-Next RMSNorm weights are stored zero-centered → fold in the +1.
        let norm_w = |key: String| -> Result<Array> {
            let t = req(key)?;
            Ok(add(&t, &Array::from_f32(1.0).as_dtype(t.dtype())?)?)
        };
        let proj = |key: String| -> Result<Projection> {
            Projection::load(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?, None)
        };
        let pfx = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };

        let embed_tokens = req(pfx("model.embed_tokens.weight"))?;
        let norm = norm_w(pfx("model.norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req(pfx("lm_head.weight"))?
        };

        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let ones_k =
            Array::from_slice(&vec![1.0f32; cfg.linear_key_head_dim as usize], &[cfg.linear_key_head_dim])
                .as_dtype(COMPUTE_DTYPE)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| pfx(&format!("model.layers.{i}.{s}"));
            let mixer = if cfg.is_linear(i) {
                let conv_weight = req(lp("linear_attn.conv1d.weight"))?
                    .reshape(&[conv_dim, cfg.linear_conv_kernel_dim])?;
                Mixer::Delta(GatedDeltaNet {
                    in_proj_qkvz: proj(lp("linear_attn.in_proj_qkvz.weight"))?,
                    in_proj_ba: proj(lp("linear_attn.in_proj_ba.weight"))?,
                    conv_weight,
                    a_log: req(lp("linear_attn.A_log"))?,
                    dt_bias: req(lp("linear_attn.dt_bias"))?,
                    norm_weight: req(lp("linear_attn.norm.weight"))?,
                    out_proj: proj(lp("linear_attn.out_proj.weight"))?,
                    ones_k: ones_k.clone(),
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
                    q_proj: proj(lp("self_attn.q_proj.weight"))?,
                    k_proj: proj(lp("self_attn.k_proj.weight"))?,
                    v_proj: proj(lp("self_attn.v_proj.weight"))?,
                    o_proj: proj(lp("self_attn.o_proj.weight"))?,
                    q_norm: norm_w(lp("self_attn.q_norm.weight"))?,
                    k_norm: norm_w(lp("self_attn.k_norm.weight"))?,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    scale: (cfg.head_dim as f32).powf(-0.5),
                    eps,
                })
            };
            layers.push(DecoderLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp("post_attention_layernorm.weight"))?,
                mixer,
                mlp: Mlp {
                    gate: proj(lp("mlp.gate_proj.weight"))?,
                    up: proj(lp("mlp.up_proj.weight"))?,
                    down: proj(lp("mlp.down_proj.weight"))?,
                },
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
        })
    }
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
        t(&mut m, "model.embed_tokens.weight", &[cfg.vocab_size, h]);
        t(&mut m, "model.norm.weight", &[h]);
        t(&mut m, "lm_head.weight", &[cfg.vocab_size, h]);
        for i in 0..cfg.num_layers {
            let lp = |s: &str| format!("model.layers.{i}.{s}");
            t(&mut m, &lp("input_layernorm.weight"), &[h]);
            t(&mut m, &lp("post_attention_layernorm.weight"), &[h]);
            t(&mut m, &lp("mlp.gate_proj.weight"), &[cfg.intermediate_size, h]);
            t(&mut m, &lp("mlp.up_proj.weight"), &[cfg.intermediate_size, h]);
            t(&mut m, &lp("mlp.down_proj.weight"), &[h, cfg.intermediate_size]);
            if cfg.is_linear(i) {
                t(&mut m, &lp("linear_attn.in_proj_qkvz.weight"), &[key_dim * 2 + value_dim * 2, h]);
                t(&mut m, &lp("linear_attn.in_proj_ba.weight"), &[cfg.linear_num_value_heads * 2, h]);
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
        let model = Qwen35Model::from_weights(&w, "", cfg.clone()).unwrap();
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
        let model = Qwen35Model::from_weights(&synthetic_weights(&cfg), "", cfg.clone()).unwrap();
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
}
