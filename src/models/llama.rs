//! Generic Llama-family decoder.
//!
//! Modelled on the working mlx-gen prompt-refine / JoyCaption Llama stacks, rebuilt on this crate's
//! own [`primitives`](crate::primitives). The forward is `&self`; the KV cache is the only mutable
//! state, threaded in as `&mut dyn KvCache` so any cache implementation (contiguous today, paged in
//! P4) works without the decoder changing.
//!
//! Shapes are batch-capable: the batch axis is real throughout (`[batch, seq, …]`), even though the
//! streaming driver in [`crate::decode`] runs batch-1.

use mlx_rs::ops::add;
use mlx_rs::{Array, Dtype};

use crate::config::LlamaConfig;
use crate::error::Result;
use crate::primitives::attention::sdpa_causal;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, linear, rms_norm, swiglu};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{ContiguousKvCache, Weights};

/// Cached decode runs in bf16 (matching the reference engines).
const COMPUTE_DTYPE: Dtype = Dtype::Bfloat16;

/// A loaded Llama decoder.
#[derive(Debug)]
pub struct LlamaModel {
    embed_tokens: Array,
    layers: Vec<LlamaLayer>,
    norm: Array,
    lm_head: Array,
    rope: Rope,
    cfg: LlamaConfig,
}

impl LlamaModel {
    /// Build from a loaded checkpoint. `prefix` is the weight-key prefix (`""` for a plain
    /// `LlamaForCausalLM`, e.g. `"language_model"` for a VLM-nested decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: LlamaConfig) -> Result<Self> {
        let p = |suffix: &str| join(prefix, suffix);
        let req_bf16 = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?) };

        let embed_tokens = req_bf16(p("model.embed_tokens.weight"))?;
        let norm = req_bf16(p("model.norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req_bf16(p("lm_head.weight"))?
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |suffix: &str| join(prefix, &format!("model.layers.{i}.{suffix}"));
            layers.push(LlamaLayer {
                input_ln: req_bf16(lp("input_layernorm.weight"))?,
                post_ln: req_bf16(lp("post_attention_layernorm.weight"))?,
                attn: LlamaAttention {
                    q_w: req_bf16(lp("self_attn.q_proj.weight"))?,
                    k_w: req_bf16(lp("self_attn.k_proj.weight"))?,
                    v_w: req_bf16(lp("self_attn.v_proj.weight"))?,
                    o_w: req_bf16(lp("self_attn.o_proj.weight"))?,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    scale: cfg.attn_scale(),
                    groups: cfg.groups(),
                },
                mlp: LlamaMlp {
                    gate_w: req_bf16(lp("mlp.gate_proj.weight"))?,
                    up_w: req_bf16(lp("mlp.up_proj.weight"))?,
                    down_w: req_bf16(lp("mlp.down_proj.weight"))?,
                },
                eps: cfg.rms_norm_eps,
            });
        }

        let rope = cfg.build_rope();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            cfg,
        })
    }

    /// The model config.
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// A fresh contiguous KV cache sized for this model.
    pub fn new_cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.cfg.num_layers)
    }

    /// Embed token ids `[batch, seq]` → `[batch, seq, hidden]` (bf16).
    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        embed(&self.embed_tokens, input_ids)
    }

    /// Run a forward step over token ids and return logits for the **last** position only,
    /// `[batch, vocab]`. `offset` is the position of the first input token (the RoPE offset =
    /// number of positions already cached).
    pub fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Like [`LlamaModel::decode_logits`] but starting from pre-computed input embeddings — the
    /// hook the VLM path (story 7157) uses to splice image features before the decoder.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let sh = input_embeds.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?;

        let mut h = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos, &sin, cache, i)?;
        }

        // Logits for the last position only (memory-efficient: never materialise full-seq logits).
        let last = s - 1;
        let last_idx = Array::from_slice(&[last], &[1]);
        let last_h = h.take_axis(&last_idx, 1)?.reshape(&[b, self.cfg.hidden_size])?;
        let normed = rms_norm(&last_h, &self.norm, self.cfg.rms_norm_eps)?;
        let logits = linear(&normed, &self.lm_head, None)?;
        Ok(logits.reshape(&[b, self.cfg.vocab_size])?)
    }
}

impl crate::decode::Decode for LlamaModel {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array> {
        self.decode_logits(input_ids, cache, offset)
    }
}

/// One pre-norm transformer block.
#[derive(Debug)]
struct LlamaLayer {
    input_ln: Array,
    post_ln: Array,
    attn: LlamaAttention,
    mlp: LlamaMlp,
    eps: f32,
}

impl LlamaLayer {
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, cache, layer_idx)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }
}

/// Grouped-query attention with RoPE.
#[derive(Debug)]
struct LlamaAttention {
    q_w: Array,
    k_w: Array,
    v_w: Array,
    o_w: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    groups: i32,
}

impl LlamaAttention {
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        // Project, then split into heads in [b, s, heads, head_dim] layout.
        let q = linear(x, &self.q_w, None)?.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k = linear(x, &self.k_w, None)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = linear(x, &self.v_w, None)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // RoPE on q,k in [b, s, heads, head_dim] (cos/sin broadcast over the head axis).
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // -> [b, heads, s, head_dim] for attention + caching.
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        let k_all = crate::primitives::repeat_kv(&k_all, self.groups)?;
        let v_all = crate::primitives::repeat_kv(&v_all, self.groups)?;

        let out = sdpa_causal(&q, &k_all, &v_all, self.scale)?; // [b, heads, s, head_dim]
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.num_heads * self.head_dim])?;
        linear(&out, &self.o_w, None)
    }
}

/// SwiGLU feed-forward.
#[derive(Debug)]
struct LlamaMlp {
    gate_w: Array,
    up_w: Array,
    down_w: Array,
}

impl LlamaMlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        swiglu(x, &self.gate_w, &self.up_w, &self.down_w)
    }
}

/// Join a key prefix and suffix (`""` prefix ⇒ the suffix verbatim).
fn join(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_handles_empty_prefix() {
        assert_eq!(join("", "model.norm.weight"), "model.norm.weight");
        assert_eq!(
            join("language_model", "model.norm.weight"),
            "language_model.model.norm.weight"
        );
    }
}
