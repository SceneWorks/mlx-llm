//! Generic Llama-family causal decoder (Llama / Mistral / Qwen3).
//!
//! Modelled on the working mlx-gen prompt-refine / JoyCaption Llama stacks, generalized for BYO
//! architecture dispatch (story 7163): attention optionally applies per-head q/k RMSNorm (Qwen3),
//! and projections are held behind [`Projection`] so a model can be quantized on load. The forward
//! is `&self`; the KV cache is the only mutable state, threaded in as `&mut dyn KvCache`.
//!
//! Shapes are batch-capable: the batch axis is real throughout (`[batch, seq, …]`), even though the
//! streaming driver in [`crate::decode`] runs batch-1. Note `head_dim` is taken from config and may
//! differ from `hidden_size / num_heads` (e.g. Qwen3-0.6B: hidden 1024, 16 heads, head_dim 128).

use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::{Array, Dtype};

use crate::config::LlamaConfig;
use crate::error::{Error, Result};
use crate::primitives::attention::{sdpa, AttnMask};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, linear, rms_norm};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{ContiguousKvCache, PagedKvCache, Weights};

/// Cached decode runs in bf16 (matching the reference engines).
const COMPUTE_DTYPE: Dtype = Dtype::Bfloat16;

/// A loaded causal decoder.
#[derive(Debug)]
pub struct LlamaModel {
    embed_tokens: Array,
    layers: Vec<LlamaLayer>,
    norm: Array,
    lm_head: Array,
    rope: Rope,
    cfg: LlamaConfig,
    quantized: bool,
}

impl LlamaModel {
    /// Build from a loaded checkpoint (dense). `prefix` is the weight-key prefix (`""` for a plain
    /// `*ForCausalLM`, e.g. `"language_model"` for a VLM-nested decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: LlamaConfig) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the attention/MLP projections on load.
    /// Embeddings, the LM head, and norms always stay dense.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: LlamaConfig,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        let p = |suffix: &str| join(prefix, suffix);
        let req_bf16 = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?) };
        // A snapshot that stores pre-quantized projections (the GGUF converter's MLX-requant output)
        // is loaded from its packed `weight`/`scales`/`biases` as-is; the group size / bits come from
        // the config's `quantization` block. Otherwise the dense weight is loaded (and quantized on
        // the fly if a load-time `quant` was requested).
        let stored_quant = cfg.quantization;
        let proj = |key: String| -> Result<Projection> {
            let base = key.strip_suffix(".weight").unwrap_or(&key);
            let scales_key = format!("{base}.scales");
            if w.contains(&scales_key) {
                let spec = stored_quant.ok_or_else(|| {
                    Error::Config(format!(
                        "snapshot stores quantized tensor `{scales_key}` but config.json has no \
                         `quantization` block"
                    ))
                })?;
                Ok(Projection::from_quantized(
                    w.require(&key)?.clone(),
                    w.require(&scales_key)?.clone(),
                    w.require(&format!("{base}.biases"))?.clone(),
                    spec,
                ))
            } else {
                Projection::load(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?, quant)
            }
        };

        let embed_tokens = req_bf16(p("model.embed_tokens.weight"))?;
        let norm = req_bf16(p("model.norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req_bf16(p("lm_head.weight"))?
        };

        let qk_norm = cfg.has_qk_norm();
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |suffix: &str| join(prefix, &format!("model.layers.{i}.{suffix}"));
            let (q_norm, k_norm) = if qk_norm {
                (
                    Some(req_bf16(lp("self_attn.q_norm.weight"))?),
                    Some(req_bf16(lp("self_attn.k_norm.weight"))?),
                )
            } else {
                (None, None)
            };
            layers.push(LlamaLayer {
                input_ln: req_bf16(lp("input_layernorm.weight"))?,
                post_ln: req_bf16(lp("post_attention_layernorm.weight"))?,
                attn: LlamaAttention {
                    q: proj(lp("self_attn.q_proj.weight"))?,
                    k: proj(lp("self_attn.k_proj.weight"))?,
                    v: proj(lp("self_attn.v_proj.weight"))?,
                    o: proj(lp("self_attn.o_proj.weight"))?,
                    q_norm,
                    k_norm,
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim,
                    scale: cfg.attn_scale(),
                    eps: cfg.rms_norm_eps,
                },
                mlp: LlamaMlp {
                    gate: proj(lp("mlp.gate_proj.weight"))?,
                    up: proj(lp("mlp.up_proj.weight"))?,
                    down: proj(lp("mlp.down_proj.weight"))?,
                },
                eps: cfg.rms_norm_eps,
            });
        }

        let rope = cfg.build_rope();
        let quantized = quant.is_some() || cfg.quantization.is_some();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            cfg,
            quantized,
        })
    }

    /// The model config.
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// Whether the projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// A fresh contiguous KV cache sized for this model.
    pub fn new_cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.cfg.num_layers)
    }

    /// A fresh single-sequence **paged** KV cache (story 7169) sized for this model, with
    /// `block_size`-token blocks. A drop-in for [`LlamaModel::new_cache`] behind the
    /// [`KvCache`] trait; pack concurrency as separate caches over one shared
    /// [`BlockPool`](crate::primitives::BlockPool).
    pub fn new_paged_cache(&self, block_size: usize) -> crate::primitives::PagedKvCache {
        crate::primitives::PagedKvCache::new(self.cfg.num_layers, block_size)
    }

    /// The engine's cached-decode compute dtype (bf16) — used by the batched decode to match the
    /// additive attention mask to the score dtype.
    pub const fn compute_dtype(&self) -> Dtype {
        COMPUTE_DTYPE
    }

    /// Build per-row RoPE `(cos, sin)` tables for a `[rows, cols]` grid of absolute positions
    /// (row-major flat `positions`, length `rows * cols`) — the **per-sequence** position tables the
    /// batched decode (story 7167) feeds [`LlamaModel::decode_logits_masked`]. Each is
    /// `[rows, cols, head_dim]` in the compute dtype.
    pub fn rope_tables(&self, positions: &[i32], rows: i32, cols: i32) -> Result<(Array, Array)> {
        let (cos, sin) = self.rope.cos_sin_at(positions, COMPUTE_DTYPE)?; // [1, rows*cols, head_dim]
        let hd = self.rope.dim();
        Ok((cos.reshape(&[rows, cols, hd])?, sin.reshape(&[rows, cols, hd])?))
    }

    /// Embed token ids `[batch, seq]` → `[batch, seq, hidden]` (bf16).
    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        embed(&self.embed_tokens, input_ids)
    }

    /// Run a forward step over token ids and return logits for the **last** position only,
    /// `[batch, vocab]`. `offset` is the position of the first input token (number of cached
    /// positions).
    pub fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Like [`LlamaModel::decode_logits`] but from pre-computed input embeddings — the hook the VLM
    /// path (story 7157) uses to splice image features before the decoder.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let s = input_embeds.shape()[1];
        // Single-sequence / uniform batch: positions [offset, offset+s) shared across the batch,
        // implicit bottom-right causal mask. cos/sin `[1, s, head_dim]` broadcast over the batch.
        let (cos, sin) = self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?;
        self.forward_to_last_logits(input_embeds, cache, &cos, &sin, AttnMask::Causal)
    }

    /// Run a forward step over token ids and return logits for **every** position, `[batch, seq,
    /// vocab]` — the all-position output speculative decoding (story 7171) needs to verify K proposed
    /// tokens in one pass. `offset` is the position of the first input token (cached positions);
    /// attention is implicit-causal, so position `i` attends `0..=offset+i`. Single-sequence / uniform
    /// batch.
    pub fn decode_logits_all(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        let s = embeds.shape()[1];
        let (cos, sin) = self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?;
        let mut h = embeds;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos, &sin, AttnMask::Causal, cache, i)?;
        }
        let normed = rms_norm(&h, &self.norm, self.cfg.rms_norm_eps)?;
        linear(&normed, &self.lm_head, None) // [batch, seq, vocab]
    }

    /// Batched forward over a **left-padded** `[batch, seq]` step with **per-sequence** RoPE
    /// positions and an explicit additive attention mask — the decode primitive the dynamic-batch
    /// scheduler (story 7167) runs each step.
    ///
    /// `input_ids` is `[batch, seq]`; `cos`/`sin` are `[batch, seq, head_dim]` (per-row positions,
    /// e.g. from [`Rope::cos_sin_at`] reshaped); `mask` is an additive `[batch, 1, seq, k_total]`
    /// score mask (`0` keep, `-inf` block) covering left-padding + causality. Returns logits for the
    /// **last column** `[batch, vocab]` — left-padding right-aligns every row's last real token to
    /// that column, so one slice serves the whole batch.
    pub fn decode_logits_masked(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        cos: &Array,
        sin: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        self.forward_to_last_logits(&embeds, cache, cos, sin, AttnMask::Additive(mask))
    }

    /// Throughput-mode batched decode forward for iteration-level continuous batching (story 7281):
    /// batched embed / projections / MLP / lm_head over a `[batch, seq]` step, but attention is
    /// **per-sequence** — each row attends only its own sequence's real KV in `caches[i]` (a
    /// [`PagedKvCache`]), gathered to its true length with no padding mask. `positions` are the
    /// per-row absolute RoPE positions (row-major, length `batch * seq`; ragged across rows by
    /// construction, since each sequence sits at its own cache offset). Returns the **last column**
    /// logits `[batch, vocab]`.
    ///
    /// Unlike [`LlamaModel::decode_logits`] run per row, the **batched** projections here are not
    /// bit-identical to the batch-1 logits: MLX's batched matmul is not row-invariant, so a row
    /// tracks its batch-1 run only to sub-ULP (the documented Throughput-mode tradeoff — it buys the
    /// weight-read amortization that scales throughput with occupancy). The bit-exact continuous path
    /// runs each sequence through [`LlamaModel::decode_logits`] on its own cache instead.
    pub fn decode_logits_per_seq(
        &self,
        input_ids: &Array,
        caches: &mut [&mut PagedKvCache],
        positions: &[i32],
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        if caches.len() != b as usize {
            return Err(Error::Msg(format!(
                "decode_logits_per_seq: {} caches for a batch of {b}",
                caches.len()
            )));
        }
        if positions.len() != (b * s) as usize {
            return Err(Error::Msg(format!(
                "decode_logits_per_seq: {} positions for a {b}x{s} step",
                positions.len()
            )));
        }
        let (cos, sin) = self.rope_tables(positions, b, s)?;
        let mut h = self.embed(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_per_seq(&h, &cos, &sin, caches, i)?;
        }
        let last = s - 1;
        let last_idx = Array::from_slice(&[last], &[1]);
        let last_h = h.take_axis(&last_idx, 1)?.reshape(&[b, self.cfg.hidden_size])?;
        let normed = rms_norm(&last_h, &self.norm, self.cfg.rms_norm_eps)?;
        let logits = linear(&normed, &self.lm_head, None)?;
        Ok(logits.reshape(&[b, self.cfg.vocab_size])?)
    }

    /// Run the decoder stack over `input_embeds` with the given RoPE tables and attention mask, and
    /// project the **last column** to logits `[batch, vocab]`. The shared core of the single and
    /// batched forwards: they differ only in how `cos`/`sin` and `mask` are built.
    fn forward_to_last_logits(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
    ) -> Result<Array> {
        let sh = input_embeds.shape();
        let (b, s) = (sh[0], sh[1]);

        let mut h = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, cos, sin, mask, cache, i)?;
        }

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
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, mask, cache, layer_idx)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }

    /// Per-sequence attention variant of [`LlamaLayer::forward`]: the MLP and norms are batched as
    /// usual, but attention runs row-by-row over each sequence's own paged cache (story 7281).
    fn forward_per_seq(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward_per_seq(&normed, cos, sin, caches, layer_idx)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }
}

/// Grouped-query attention with RoPE and optional per-head q/k RMSNorm (Qwen3).
#[derive(Debug)]
struct LlamaAttention {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Option<Array>,
    k_norm: Option<Array>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl LlamaAttention {
    /// Project `x` `[b, s, hidden]` into attention-layout `(q, k, v)` — `q` `[b, heads, s, head_dim]`,
    /// `k`/`v` `[b, kv_heads, s, head_dim]` — applying the qkv projections, optional Qwen3 per-head
    /// q/k RMSNorm, RoPE, and the transpose into the head-major layout. The shared front half of both
    /// the masked (batched) and per-sequence attention paths; nothing here touches the cache or
    /// attends, so the two paths differ only in how they gather K/V and call SDPA.
    fn project(&self, x: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array, Array)> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        // Project, then split into heads in [b, s, heads, head_dim] layout.
        let mut q = self.q.forward(x)?.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let mut k = self.k.forward(x)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = self.v.forward(x)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // Qwen3 per-head q/k RMSNorm over the head_dim axis, before RoPE.
        if let Some(qn) = &self.q_norm {
            q = rms_norm(&q, qn, self.eps)?;
        }
        if let Some(kn) = &self.k_norm {
            k = rms_norm(&k, kn, self.eps)?;
        }

        // RoPE on q,k (cos/sin broadcast over the head axis), then -> [b, heads, s, head_dim].
        let q = apply_rope(&q, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        Ok((q, k, v))
    }

    /// Project the attended output `[b, heads, s, head_dim]` back to `[b, s, hidden]` through `o`.
    /// The shared back half of both attention paths.
    fn output(&self, attn: &Array) -> Result<Array> {
        let sh = attn.shape(); // [b, heads, s, head_dim]
        let (b, s) = (sh[0], sh[2]);
        let merged = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o.forward(&merged)
    }

    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let (q, k, v) = self.project(x, cos, sin)?;

        // MLX's fused SDPA does grouped-query attention natively — it derives
        // `gqa_factor = q_heads / kv_heads` and reads K/V by head stride — so we hand it the
        // GQA-shaped cache K/V directly instead of materializing a full-head-count copy with
        // `repeat_kv` every step (which allocated+wrote a `groups`× larger K/V tensor MLX would
        // never need). MHA (kv_heads == heads) is unchanged. [sc-7307]
        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;

        let out = sdpa(&q, &k_all, &v_all, self.scale, mask)?; // [b, heads, s, head_dim]
        self.output(&out)
    }

    /// Per-sequence attention (story 7281): the projection is **batched** over all rows, but each
    /// row then attends only its own sequence's real KV in `caches[i]`, gathered to its true length
    /// and run through a stock causal SDPA — i.e. byte-identically to that sequence's batch-1
    /// attention, with no padding mask and no cross-row mixing. `caches.len()` must equal the batch.
    ///
    /// Attention is the only step that loops; the projections, MLP, and lm_head around it stay
    /// batched, which is where the throughput comes from. (The matmul batching is also why the
    /// surrounding logits only *track* batch-1 to sub-ULP — see [`LlamaModel::decode_logits_per_seq`].)
    fn forward_per_seq(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        let (q, k, v) = self.project(x, cos, sin)?; // [b,H,s,hd], [b,kvh,s,hd], [b,kvh,s,hd]
        let mut outs = Vec::with_capacity(caches.len());
        for (i, cache) in caches.iter_mut().enumerate() {
            let i = i as i32;
            let (qi, ki, vi) = (row_axis0(&q, i)?, row_axis0(&k, i)?, row_axis0(&v, i)?);
            let (k_all, v_all) = cache.update(layer_idx, &ki, &vi)?; // gather this seq's real KV
            outs.push(sdpa(&qi, &k_all, &v_all, self.scale, AttnMask::Causal)?); // [1,H,s,hd]
        }
        let refs: Vec<&Array> = outs.iter().collect();
        let out = concatenate_axis(&refs, 0)?; // [b, heads, s, head_dim]
        self.output(&out)
    }
}

/// Slice row `i` off the batch axis, keeping the axis (`[1, …]`).
fn row_axis0(a: &Array, i: i32) -> Result<Array> {
    let idx = Array::from_slice(&[i], &[1]);
    Ok(a.take_axis(&idx, 0)?)
}

/// SwiGLU feed-forward.
#[derive(Debug)]
struct LlamaMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl LlamaMlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = crate::primitives::nn::silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        let gated = mlx_rs::ops::multiply(&gate, &up)?;
        self.down.forward(&gated)
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
