//! Generic Llama-family causal decoder, config-dispatched across architectures.
//!
//! One block shape covers the family (Llama / Mistral / Qwen2 / Qwen3 / Phi-3 / Qwen2-MoE / Gemma-2 /
//! GLM-4 / DeepSeek-V2): self-attention is either grouped-query attention (with optional per-head q/k
//! RMSNorm for Qwen3, q/k/v bias for Qwen2 / GLM-4, a packed `qkv_proj` for Phi-3, Gemma-2 score
//! soft-cap) or Multi-head Latent Attention (DeepSeek's low-rank KV path); the FFN is a dense gated
//! MLP (SwiGLU, or GeGLU for Gemma) or a sparse Mixture-of-Experts bank; norms are the Llama pre-norm
//! or the Gemma-2 / GLM-4 4-norm "sandwich". Projections are held behind [`Projection`] so a model can
//! be quantized on load. The forward is `&self`; the KV cache is the only mutable state, threaded in as
//! `&mut dyn KvCache`. Ported alongside candle-llm's `models/llama.rs` (the cross-backend blueprint).
//!
//! Shapes are batch-capable (`[batch, seq, …]`). `head_dim` is taken from config and may differ from
//! `hidden_size / num_heads`. Cached decode runs in bf16.

use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, sigmoid, split_sections, zeros_dtype};
use mlx_rs::{Array, Dtype};

use crate::config::{Architecture, ModelConfig};
use crate::error::{Error, Result};
use crate::models::deepstack::deepstack_fused_decoder_layers;
use crate::primitives::attention::{sdpa_capped, AttnMask};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, gelu_tanh, linear, rms_norm, silu, soft_cap, to_f32_host};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::quant::QuantizedLinear;
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{ContiguousKvCache, PagedKvCache, Weights};

/// Cached decode runs in bf16 (matching the reference engines).
const COMPUTE_DTYPE: Dtype = Dtype::Bfloat16;

/// A loaded causal decoder.
#[derive(Debug)]
pub struct CausalLm {
    embed_tokens: Array,
    layers: Vec<LlamaLayer>,
    norm: Array,
    lm_head: Array,
    rope: Rope,
    cfg: ModelConfig,
    quantized: bool,
    /// Gemma scales token embeddings by √hidden; `None` ⇒ no scaling.
    embed_scale: Option<f32>,
    /// Gemma-2 final-logit soft-cap; `None` ⇒ no cap.
    final_softcap: Option<f32>,
}

impl CausalLm {
    /// Build from a loaded checkpoint (dense). `prefix` is the weight-key prefix (`""` for a plain
    /// `*ForCausalLM`, e.g. `"language_model"` for a VLM-nested decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ModelConfig) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the attention/MLP projections on load.
    /// Embeddings, the LM head, and norms always stay dense.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: ModelConfig,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        // The Qwen3-VL VLM wrapper nests the decoder under `model.language_model.*` (embeddings,
        // norm, and `layers.{i}.*`) — there is no second `model.` segment — while `lm_head.weight`
        // lives at the checkpoint root (untied). A plain `*ForCausalLM` keeps the historical
        // `[{prefix}.]model.*` / `[{prefix}.]lm_head.weight` layout.
        let vlm_nested = cfg.architecture.is_qwen3_vl();
        let decoder_root = if vlm_nested { "model.language_model".to_string() } else { join(prefix, "model") };
        let p = |suffix: &str| join(&decoder_root, suffix);
        let head_key = if vlm_nested { "lm_head.weight".to_string() } else { join(prefix, "lm_head.weight") };
        let req_bf16 = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?) };

        // A snapshot may store pre-quantized projections (the GGUF converter's MLX-requant output);
        // those are loaded from `weight`/`scales`/`biases` as-is. Otherwise the dense weight is loaded
        // (and quantized on the fly if a load-time `quant` was requested). `bias` is applied dense in
        // both cases (Qwen2 / GLM-4 attention bias).
        let stored_quant = cfg.quantization;
        let load_proj = |key: &str, bias: Option<Array>| -> Result<Projection> {
            let base = key.strip_suffix(".weight").unwrap_or(key);
            let scales_key = format!("{base}.scales");
            if w.contains(&scales_key) {
                let spec = stored_quant.ok_or_else(|| {
                    Error::Config(format!(
                        "snapshot stores quantized tensor `{scales_key}` but config.json has no \
                         `quantization` block"
                    ))
                })?;
                Ok(Projection::Quantized(QuantizedLinear {
                    weight: w.require(key)?.clone(),
                    scales: w.require(&scales_key)?.clone(),
                    biases: w.require(&format!("{base}.biases"))?.clone(),
                    group_size: spec.group_size,
                    bits: spec.bits,
                    bias,
                }))
            } else {
                Projection::load_with_bias(w.require(key)?.as_dtype(COMPUTE_DTYPE)?, bias, quant)
            }
        };
        let proj = |key: String| -> Result<Projection> { load_proj(&key, None) };
        // Like `proj`, but also loads a sibling `.bias` when present (Qwen2 / GLM-4 attention).
        let proj_b = |wkey: String| -> Result<Projection> {
            let base = wkey.strip_suffix(".weight").unwrap_or(&wkey);
            let bkey = format!("{base}.bias");
            let bias = if w.contains(&bkey) { Some(req_bf16(bkey)?) } else { None };
            load_proj(&wkey, bias)
        };
        // Gemma's norms are `(1 + weight)`; fold the +1 into the stored weight so the standard
        // `rms_norm` applies it. (Llama / Qwen3 / Qwen3-VL / GLM-4 norm weights are standard RMSNorm
        // — used verbatim, including Qwen3-VL's `Qwen3VLTextRMSNorm`, which is plain `weight · x`;
        // its small early-layer block-norm weights are genuine, verified by real-weights coherence.)
        let gemma = cfg.architecture.is_gemma2();
        let norm_w = |key: String| -> Result<Array> {
            let t = req_bf16(key)?;
            if gemma {
                Ok(add(&t, &Array::from_f32(1.0).as_dtype(t.dtype())?)?)
            } else {
                Ok(t)
            }
        };

        let embed_tokens = req_bf16(p("embed_tokens.weight"))?;
        let norm = norm_w(p("norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req_bf16(head_key)?
        };

        let qk_norm = cfg.has_qk_norm();
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let scale = cfg.attn_scale();
        let eps = cfg.rms_norm_eps;
        let qd = num_heads * head_dim;
        let kvd = num_kv_heads * head_dim;
        let inter = cfg.intermediate_size;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |suffix: &str| join(&decoder_root, &format!("layers.{i}.{suffix}"));

            // Attention: Multi-head Latent Attention (DeepSeek-V2) or grouped-query attention.
            let attn = if cfg.architecture.is_mla() {
                Attention::Mla(MlaAttention::load(w, &lp, &cfg, &load_proj, &req_bf16)?)
            } else {
                let (q_norm, k_norm) = if qk_norm {
                    (
                        Some(req_bf16(lp("self_attn.q_norm.weight"))?),
                        Some(req_bf16(lp("self_attn.k_norm.weight"))?),
                    )
                } else {
                    (None, None)
                };
                // A packed `qkv_proj` (Phi-3, no bias) is split into q/k/v along axis 0; otherwise the
                // separate q/k/v projections are loaded (with q/k/v bias for Qwen2 / GLM-4).
                let (q, k, v) = {
                    let packed = lp("self_attn.qkv_proj.weight");
                    if w.contains(&packed) {
                        let qkv = req_bf16(packed)?; // [qd + 2*kvd, hidden]
                        let parts = split_sections(&qkv, &[qd, qd + kvd], 0)?;
                        (
                            Projection::load(parts[0].clone(), quant)?,
                            Projection::load(parts[1].clone(), quant)?,
                            Projection::load(parts[2].clone(), quant)?,
                        )
                    } else {
                        (
                            proj_b(lp("self_attn.q_proj.weight"))?,
                            proj_b(lp("self_attn.k_proj.weight"))?,
                            proj_b(lp("self_attn.v_proj.weight"))?,
                        )
                    }
                };
                Attention::Gqa(LlamaAttention {
                    q,
                    k,
                    v,
                    o: proj_b(lp("self_attn.o_proj.weight"))?,
                    q_norm,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    eps,
                    softcap: cfg.attn_logit_softcap,
                    rope_interleaved: cfg.architecture.rope_interleaved(),
                })
            };

            // Feed-forward: a sparse Mixture-of-Experts bank or a dense MLP. DeepSeek keeps its leading
            // `first_k_dense_replace` layers dense even though the model is MoE. Gemma uses GeGLU.
            let moe_layer = cfg.moe.filter(|m| i >= m.first_k_dense_replace);
            let ffn = if let Some(moe) = moe_layer {
                let mut experts = Vec::with_capacity(moe.num_experts);
                for e in 0..moe.num_experts {
                    let ep = |s: &str| lp(&format!("mlp.experts.{e}.{s}"));
                    experts.push(LlamaMlp {
                        gate: proj(ep("gate_proj.weight"))?,
                        up: proj(ep("up_proj.weight"))?,
                        down: proj(ep("down_proj.weight"))?,
                        gelu: false,
                    });
                }
                // Shared-expert key stem: DeepSeek packs `n_shared_experts` into `mlp.shared_experts`
                // (plural, ungated); Qwen2-MoE has a single `mlp.shared_expert` gated by a sigmoid.
                let shared_stem = if w.contains(&lp("mlp.shared_experts.gate_proj.weight")) {
                    "mlp.shared_experts"
                } else {
                    "mlp.shared_expert"
                };
                let shared_gate_key = lp("mlp.shared_expert_gate.weight");
                Ffn::Moe(MoeMlp {
                    router: req_bf16(lp("mlp.gate.weight"))?, // [num_experts, hidden]
                    experts,
                    shared: LlamaMlp {
                        gate: proj(lp(&format!("{shared_stem}.gate_proj.weight")))?,
                        up: proj(lp(&format!("{shared_stem}.up_proj.weight")))?,
                        down: proj(lp(&format!("{shared_stem}.down_proj.weight")))?,
                        gelu: false,
                    },
                    shared_gate: if w.contains(&shared_gate_key) {
                        Some(req_bf16(shared_gate_key)?) // [1, hidden]
                    } else {
                        None
                    },
                    experts_per_tok: moe.num_experts_per_tok,
                    norm_topk_prob: moe.norm_topk_prob,
                    routed_scaling_factor: moe.routed_scaling_factor,
                })
            } else {
                // Dense MLP; Phi-3 fuses gate‖up into one weight, split along axis 0.
                let (gate, up) = {
                    let packed = lp("mlp.gate_up_proj.weight");
                    if w.contains(&packed) {
                        let gu = req_bf16(packed)?; // [2*inter, hidden]
                        let parts = split_sections(&gu, &[inter], 0)?;
                        (
                            Projection::load(parts[0].clone(), quant)?,
                            Projection::load(parts[1].clone(), quant)?,
                        )
                    } else {
                        (proj(lp("mlp.gate_proj.weight"))?, proj(lp("mlp.up_proj.weight"))?)
                    }
                };
                Ffn::Dense(LlamaMlp {
                    gate,
                    up,
                    down: proj(lp("mlp.down_proj.weight"))?,
                    gelu: gemma,
                })
            };

            // Gemma-2 / GLM-4 wrap the block in a 4-norm "sandwich" (pre+post for both attn and MLP);
            // the Llama shape has only the two pre-norms. The norm key names differ per family.
            let (post_attn_key, pre_ff_key, post_ff_key) = match cfg.architecture {
                Architecture::Glm4 => (
                    "post_self_attn_layernorm",
                    "post_attention_layernorm",
                    "post_mlp_layernorm",
                ),
                _ => (
                    "post_attention_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                ),
            };
            let (pre_ff_ln, post_ff_ln) = if cfg.architecture.is_sandwich() {
                (
                    Some(norm_w(lp(&format!("{pre_ff_key}.weight")))?),
                    Some(norm_w(lp(&format!("{post_ff_key}.weight")))?),
                )
            } else {
                (None, None)
            };

            layers.push(LlamaLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp(&format!("{post_attn_key}.weight")))?,
                pre_ff_ln,
                post_ff_ln,
                attn,
                ffn,
                eps,
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
            quantized,
            embed_scale: gemma.then(|| (cfg.hidden_size as f32).sqrt()),
            final_softcap: cfg.final_logit_softcap,
            cfg,
        })
    }

    /// The model config.
    pub fn config(&self) -> &ModelConfig {
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
    /// `block_size`-token blocks.
    pub fn new_paged_cache(&self, block_size: usize) -> PagedKvCache {
        PagedKvCache::new(self.cfg.num_layers, block_size)
    }

    /// The engine's cached-decode compute dtype (bf16).
    pub const fn compute_dtype(&self) -> Dtype {
        COMPUTE_DTYPE
    }

    /// Build per-row RoPE `(cos, sin)` tables for a `[rows, cols]` grid of absolute positions
    /// (row-major flat `positions`, length `rows * cols`). Each is `[rows, cols, rope_dim]` in bf16.
    pub fn rope_tables(&self, positions: &[i32], rows: i32, cols: i32) -> Result<(Array, Array)> {
        let (cos, sin) = self.rope.cos_sin_at(positions, COMPUTE_DTYPE)?; // [1, rows*cols, rope_dim]
        let hd = self.rope.dim();
        Ok((cos.reshape(&[rows, cols, hd])?, sin.reshape(&[rows, cols, hd])?))
    }

    /// Embed token ids `[batch, seq]` → `[batch, seq, hidden]` (bf16). Gemma scales by √hidden.
    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        let e = embed(&self.embed_tokens, input_ids)?;
        match self.embed_scale {
            Some(s) => Ok(multiply(&e, &Array::from_f32(s).as_dtype(e.dtype())?)?),
            None => Ok(e),
        }
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

    /// Like [`CausalLm::decode_logits`] but from pre-computed input embeddings — the VLM splice hook.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let s = input_embeds.shape()[1];
        let (cos, sin) = self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?;
        self.forward_to_last_logits(input_embeds, cache, &cos, &sin, AttnMask::Causal)
    }

    /// Embed token ids `[1, S]` → `[1, S, hidden]` in the compute dtype — the Qwen3-VL multimodal
    /// splice point (image-token rows are overwritten with the vision tower's merged patch features).
    pub fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        Ok(self.embed(input_ids)?.as_dtype(COMPUTE_DTYPE)?)
    }

    /// Replace the `image_token_id` rows of `embeds` `[1, S, hidden]` with the vision tower's merged
    /// patch features `[num_image_tokens, hidden]`, in sequence order (the Qwen3-VL splice).
    pub fn splice_image_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        image_features: &Array,
        image_token_id: i32,
    ) -> Result<Array> {
        crate::models::deepstack::splice_image_features(
            embeds,
            input_ids,
            image_features,
            image_token_id,
            self.cfg.hidden_size,
            COMPUTE_DTYPE,
        )
    }

    /// Replace every row whose id is any of `placeholder_tokens` (`<|image_pad|>` and/or
    /// `<|video_pad|>`) with the next vision-feature row, in sequence order — the multimodal splice for
    /// a mixed image+video prompt. Reduces to [`Self::splice_image_features`] for a single token.
    pub fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        crate::models::deepstack::splice_vision_features(
            embeds,
            input_ids,
            vision_features,
            placeholder_tokens,
            self.cfg.hidden_size,
            COMPUTE_DTYPE,
        )
    }

    /// Compute the interleaved M-RoPE 3-D position rows (`get_rope_index`, B=1) for `input_ids`
    /// containing `image_grid_thw`-described `image_token_id` runs, plus the `mrope_delta`. The
    /// image-only entry point; see [`crate::models::deepstack::mrope_positions_mm`] for image+video.
    pub fn mrope_positions(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<crate::models::deepstack::MropePositions> {
        crate::models::deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            &[],
            image_token_id,
            spatial_merge_size,
        )
    }

    /// The full image **and** video interleaved-M-RoPE entry: `input_ids` with `image_token_id` runs
    /// (one per `image_grid_thw` entry) and `video_token_id` runs (one per frame; each `[t, h, w]`
    /// video grid is split into `t` per-frame `[1, h, w]` blocks by the synthetic time axis). See
    /// [`crate::models::deepstack::mrope_positions_mm`].
    #[allow(clippy::too_many_arguments)]
    pub fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<crate::models::deepstack::MropePositions> {
        crate::models::deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    /// Run the decoder over precomputed input `embeds` `[1, S, hidden]` (text embeds with image
    /// features spliced in) using **interleaved multimodal RoPE** from explicit 3-D `positions`
    /// (temporal/height/width rows, each length `S`) **and DeepStack feature fusion**: after decoder
    /// layer `i`, for `i < deepstack.len()`, the `i`-th tapped/merged ViT feature set is added to the
    /// visual-token rows (`visual_pos_mask[p]` marks an image-token position). Returns last-position
    /// logits `[1, vocab]`. This is the Qwen3-VL prefill seam (`Qwen3VLTextModel.forward` +
    /// `_deepstack_process`); with all three position rows equal and an empty `deepstack` it is
    /// bit-identical to a plain 1-D-RoPE prefill.
    pub fn decode_logits_from_embeds_mrope_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array> {
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            COMPUTE_DTYPE,
        )?;
        let h0 = embeds.as_dtype(COMPUTE_DTYPE)?;
        let s = h0.shape()[1];
        let layers = &self.layers;
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            layers.len(),
            |i, h| layers[i].forward(h, &cos, &sin, AttnMask::Causal, cache, i),
        )?;
        let last_h = take_last(&h, s)?;
        let logits = self.project_logits(&last_h)?;
        Ok(logits.reshape(&[logits.shape()[0], self.cfg.vocab_size])?)
    }

    /// Run a forward step over token ids and return logits for **every** position,
    /// `[batch, seq, vocab]` — the all-position output speculative decoding (story 7171) verifies K
    /// proposed tokens in one pass.
    pub fn decode_logits_all(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        let s = embeds.shape()[1];
        let (cos, sin) = self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?;
        let h = self.run_decoder_stack(&embeds, cache, &cos, &sin, AttnMask::Causal)?;
        self.project_logits(&h) // [batch, seq, vocab]
    }

    /// Batched forward over a **left-padded** `[batch, seq]` step with **per-sequence** RoPE positions
    /// and an explicit additive attention mask — the dynamic-batch scheduler decode primitive.
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
    /// batched embed / projections / MLP / lm_head over a `[batch, seq]` step, attention per-sequence.
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
        let last_h = take_last(&h, s)?; // [b, 1, hidden]
        let logits = self.project_logits(&last_h)?; // [b, 1, vocab]
        Ok(logits.reshape(&[b, self.cfg.vocab_size])?)
    }

    /// Run the decoder stack over `input_embeds` with the given RoPE tables and attention mask, and
    /// project the **last column** to logits `[batch, vocab]`.
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
        let h = self.run_decoder_stack(input_embeds, cache, cos, sin, mask)?;
        let last_h = take_last(&h, s)?; // [b, 1, hidden]
        let logits = self.project_logits(&last_h)?; // [b, 1, vocab]
        Ok(logits.reshape(&[b, self.cfg.vocab_size])?)
    }

    /// Run every decoder layer, returning the final hidden states `[batch, seq, hidden]`.
    fn run_decoder_stack(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
    ) -> Result<Array> {
        let mut h = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, cos, sin, mask, cache, i)?;
        }
        Ok(h)
    }

    /// Final RMSNorm + `lm_head` (+ Gemma-2 logit soft-cap) over hidden states `[batch, n, hidden]`.
    fn project_logits(&self, h: &Array) -> Result<Array> {
        let normed = rms_norm(h, &self.norm, self.cfg.rms_norm_eps)?;
        let logits = linear(&normed, &self.lm_head, None)?;
        match self.final_softcap {
            // Soft-cap in f32 for precision (the cap denominator matters near the extremes).
            Some(c) => soft_cap(&logits.as_dtype(Dtype::Float32)?, c),
            None => Ok(logits),
        }
    }
}

impl crate::decode::Decode for CausalLm {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array> {
        self.decode_logits(input_ids, cache, offset)
    }
}

impl crate::models::VlmDecode for CausalLm {
    fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        CausalLm::embed_input_ids(self, input_ids)
    }

    fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        CausalLm::splice_vision_features(self, embeds, input_ids, vision_features, placeholder_tokens)
    }

    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<crate::models::deepstack::MropePositions> {
        CausalLm::mrope_positions_mm(
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
        // The generic decoder's cache is already the trait-object form — no downcast needed.
        self.decode_logits_from_embeds_mrope_deepstack(embeds, positions, cache, visual_pos_mask, deepstack)
    }
}

/// Slice the last position off the seq axis, keeping the axis (`[b, 1, hidden]`).
fn take_last(h: &Array, s: i32) -> Result<Array> {
    let last_idx = Array::from_slice(&[s - 1], &[1]);
    Ok(h.take_axis(&last_idx, 1)?)
}

/// One transformer block. Pre-norm by default (Llama / Qwen / Phi); Gemma-2 / GLM-4 add the
/// post-attention and post-feedforward norms (`pre_ff_ln` / `post_ff_ln` are `Some`) for the
/// 4-norm "sandwich" residual.
#[derive(Debug)]
struct LlamaLayer {
    /// Pre-attention norm.
    input_ln: Array,
    /// Llama: the MLP pre-norm. Sandwich: the post-attention norm.
    post_ln: Array,
    /// Sandwich only: the MLP pre-norm.
    pre_ff_ln: Option<Array>,
    /// Sandwich only: the post-feedforward norm applied to the MLP output before the residual add.
    post_ff_ln: Option<Array>,
    attn: Attention,
    ffn: Ffn,
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
        let attn = self.attn.forward(&normed, cos, sin, mask, cache, layer_idx)?;
        self.combine_ffn(x, &attn)
    }

    fn forward_per_seq(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let attn = self.attn.forward_per_seq(&normed, cos, sin, caches, layer_idx)?;
        self.combine_ffn(x, &attn)
    }

    /// The residual + MLP half shared by both forwards: the Llama pre-norm, or the 4-norm sandwich
    /// when `pre_ff_ln`/`post_ff_ln` are set. `x` is the block input, `attn` the attention output.
    fn combine_ffn(&self, x: &Array, attn: &Array) -> Result<Array> {
        match (&self.pre_ff_ln, &self.post_ff_ln) {
            // Sandwich (Gemma-2 / GLM-4): post-norm the attention and MLP outputs before each add.
            (Some(pre_ff), Some(post_ff)) => {
                let attn = rms_norm(attn, &self.post_ln, self.eps)?;
                let h = add(x, &attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, pre_ff, self.eps)?)?;
                let ffn = rms_norm(&ffn, post_ff, self.eps)?;
                Ok(add(&h, &ffn)?)
            }
            // Llama pre-norm: `post_ln` is the MLP pre-norm.
            _ => {
                let h = add(x, attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
                Ok(add(&h, &ffn)?)
            }
        }
    }
}

/// A layer's self-attention: grouped-query attention (Llama family) or Multi-head Latent Attention
/// (DeepSeek-V2).
#[derive(Debug)]
enum Attention {
    Gqa(LlamaAttention),
    Mla(MlaAttention),
}

impl Attention {
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        match self {
            Attention::Gqa(a) => a.forward(x, cos, sin, mask, cache, layer_idx),
            Attention::Mla(a) => a.forward(x, cos, sin, mask, cache, layer_idx),
        }
    }

    fn forward_per_seq(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        match self {
            Attention::Gqa(a) => a.forward_per_seq(x, cos, sin, caches, layer_idx),
            Attention::Mla(_) => Err(Error::Msg(
                "continuous-batching Throughput mode is not supported for MLA (DeepSeek-V2); use the \
                 Exact mode"
                    .into(),
            )),
        }
    }
}

/// Grouped-query attention with RoPE, optional per-head q/k RMSNorm (Qwen3), optional q/k/v bias
/// (Qwen2 / GLM-4), interleaved RoPE (GLM-4), and Gemma-2 score soft-cap.
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
    /// Gemma-2 attention-score soft-cap; `None` ⇒ no cap.
    softcap: Option<f32>,
    /// Whether RoPE uses the interleaved (GPT-J) pairing (GLM-4).
    rope_interleaved: bool,
}

impl LlamaAttention {
    /// Project `x` `[b, s, hidden]` into attention-layout `(q, k, v)` — `q` `[b, heads, s, head_dim]`,
    /// `k`/`v` `[b, kv_heads, s, head_dim]` — applying the qkv projections, optional q/k RMSNorm, RoPE,
    /// and the transpose into head-major layout.
    fn project(&self, x: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array, Array)> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let mut q = self.q.forward(x)?.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let mut k = self.k.forward(x)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v = self.v.forward(x)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        if let Some(qn) = &self.q_norm {
            q = rms_norm(&q, qn, self.eps)?;
        }
        if let Some(kn) = &self.k_norm {
            k = rms_norm(&k, kn, self.eps)?;
        }

        let q = apply_rope(&q, cos, sin, self.rope_interleaved)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = apply_rope(&k, cos, sin, self.rope_interleaved)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        Ok((q, k, v))
    }

    /// Project the attended output `[b, heads, s, head_dim]` back to `[b, s, hidden]` through `o`.
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
        // GQA-shaped K/V go straight to the fused kernel (native GQA); Gemma-2's score soft-cap forces
        // the eager path. `sdpa_capped` dispatches on `softcap` + head dims.
        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        let out = sdpa_capped(&q, &k_all, &v_all, self.scale, self.softcap, mask)?;
        self.output(&out)
    }

    fn forward_per_seq(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        let (q, k, v) = self.project(x, cos, sin)?;
        let mut outs = Vec::with_capacity(caches.len());
        for (i, cache) in caches.iter_mut().enumerate() {
            let i = i as i32;
            let (qi, ki, vi) = (row_axis0(&q, i)?, row_axis0(&k, i)?, row_axis0(&v, i)?);
            let (k_all, v_all) = cache.update(layer_idx, &ki, &vi)?;
            outs.push(sdpa_capped(&qi, &k_all, &v_all, self.scale, self.softcap, AttnMask::Causal)?);
        }
        let refs: Vec<&Array> = outs.iter().collect();
        let out = concatenate_axis(&refs, 0)?; // [b, heads, s, head_dim]
        self.output(&out)
    }
}

/// Multi-head Latent Attention (DeepSeek-V2).
///
/// MLA down-projects the input to a small shared KV latent (`kv_a_proj_with_mqa` → `kv_a_layernorm`,
/// width `kv_lora_rank`) plus a single shared rotary key sub-vector (`k_pe`, MQA-style). The latent is
/// up-projected (`kv_b_proj`) to per-head content keys (`k_nope`) and values. Queries split the same
/// way — a content part (`q_nope`) and a rotary part (`q_pe`), from a full `q_proj` or a low-rank
/// `q_a → norm → q_b`. RoPE rotates only the `qk_rope_head_dim` sub-vectors; the per-head key is
/// `[k_nope ‖ k_pe]` and the query `[q_nope ‖ q_pe]`, attended at `q_head_dim = qk_nope + qk_rope`.
///
/// This is the **correctness-first** materialized form: it reconstructs full per-head K (`q_head_dim`)
/// and V (`v_head_dim`) and caches them like ordinary attention, so the existing [`KvCache`] and
/// [`sdpa_capped`] are reused (the latent-caching "absorbed" optimization is a later throughput
/// concern). Heads are full MHA (no GQA expansion).
#[derive(Debug)]
struct MlaAttention {
    q_proj: Option<Projection>,
    q_a_proj: Option<Projection>,
    q_a_layernorm: Option<Array>,
    q_b_proj: Option<Projection>,
    kv_a_proj: Projection,
    kv_a_layernorm: Array,
    kv_b_proj: Projection,
    o_proj: Projection,
    num_heads: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
    scale: f32,
    eps: f32,
}

impl MlaAttention {
    fn load(
        w: &Weights,
        lp: &dyn Fn(&str) -> String,
        cfg: &ModelConfig,
        load_proj: &dyn Fn(&str, Option<Array>) -> Result<Projection>,
        req_bf16: &dyn Fn(String) -> Result<Array>,
    ) -> Result<Self> {
        let mla = cfg.mla.expect("MLA config present for a DeepSeek-V2 decoder");
        // Query: a low-rank `q_a → norm → q_b` when the model has a query LoRA, else a full `q_proj`.
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) =
            if w.contains(&lp("self_attn.q_a_proj.weight")) {
                (
                    None,
                    Some(load_proj(&lp("self_attn.q_a_proj.weight"), None)?),
                    Some(req_bf16(lp("self_attn.q_a_layernorm.weight"))?),
                    Some(load_proj(&lp("self_attn.q_b_proj.weight"), None)?),
                )
            } else {
                (Some(load_proj(&lp("self_attn.q_proj.weight"), None)?), None, None, None)
            };
        Ok(Self {
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj: load_proj(&lp("self_attn.kv_a_proj_with_mqa.weight"), None)?,
            kv_a_layernorm: req_bf16(lp("self_attn.kv_a_layernorm.weight"))?,
            kv_b_proj: load_proj(&lp("self_attn.kv_b_proj.weight"), None)?,
            o_proj: load_proj(&lp("self_attn.o_proj.weight"), None)?,
            num_heads: cfg.num_heads,
            qk_nope_head_dim: mla.qk_nope_head_dim,
            qk_rope_head_dim: mla.qk_rope_head_dim,
            v_head_dim: mla.v_head_dim,
            kv_lora_rank: mla.kv_lora_rank,
            scale: cfg.attn_scale(),
            eps: cfg.rms_norm_eps,
        })
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
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let nh = self.num_heads;
        let (nope, rope, vhd) = (self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim);
        let qhd = nope + rope; // per-head q/k dim attended over

        // Query → [b, s, nh, qhd], split into content (nope) and rotary (rope) parts.
        let q = match (&self.q_proj, &self.q_a_proj) {
            (Some(qp), _) => qp.forward(x)?,
            (None, Some(qa)) => {
                let c = qa.forward(x)?;
                let c = rms_norm(&c, self.q_a_layernorm.as_ref().unwrap(), self.eps)?;
                self.q_b_proj.as_ref().unwrap().forward(&c)?
            }
            _ => unreachable!("MLA query has either q_proj or q_a/q_b"),
        };
        let q = q.reshape(&[b, s, nh, qhd])?;
        let q_parts = split_sections(&q, &[nope], 3)?; // [q_nope, q_pe]
        let q_nope = &q_parts[0];
        let q_pe = &q_parts[1];

        // Shared KV latent + the single MQA rotary key.
        let kv = self.kv_a_proj.forward(x)?; // [b, s, kv_lora_rank + rope]
        let kv_parts = split_sections(&kv, &[self.kv_lora_rank], 2)?; // [compressed, k_pe_flat]
        let compressed = rms_norm(&kv_parts[0], &self.kv_a_layernorm, self.eps)?;
        let k_pe = kv_parts[1].reshape(&[b, s, 1, rope])?; // shared across heads

        // Up-project to per-head content keys and values: [b, s, nh, nope + vhd].
        let kv_b = self.kv_b_proj.forward(&compressed)?.reshape(&[b, s, nh, nope + vhd])?;
        let kv_b_parts = split_sections(&kv_b, &[nope], 3)?; // [k_nope, value]
        let k_nope = &kv_b_parts[0];
        let value = &kv_b_parts[1];

        // RoPE the rotary sub-vectors (interleaved); broadcast the shared key over heads.
        let q_pe = apply_rope(q_pe, cos, sin, true)?;
        let k_pe = apply_rope(&k_pe, cos, sin, true)?;
        let k_pe = broadcast_to(&k_pe, &[b, s, nh, rope])?;

        // Assemble full per-head q/k, then [b, nh, s, *] for attention.
        let q = concatenate_axis(&[q_nope, &q_pe], 3)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = concatenate_axis(&[k_nope, &k_pe], 3)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = value.transpose_axes(&[0, 2, 1, 3])?;

        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        // q/k head dim (qhd) ≠ v head dim (vhd) → `sdpa_capped` takes the eager path.
        let out = sdpa_capped(&q, &k_all, &v_all, self.scale, None, mask)?; // [b, nh, s, vhd]
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, nh * vhd])?;
        self.o_proj.forward(&out)
    }
}

/// Slice row `i` off the batch axis, keeping the axis (`[1, …]`).
fn row_axis0(a: &Array, i: i32) -> Result<Array> {
    let idx = Array::from_slice(&[i], &[1]);
    Ok(a.take_axis(&idx, 0)?)
}

/// A layer's feed-forward network: a dense gated MLP, or a sparse Mixture-of-Experts bank.
#[derive(Debug)]
enum Ffn {
    Dense(LlamaMlp),
    Moe(MoeMlp),
}

impl Ffn {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

/// A gated MLP: SwiGLU (`silu`) by default, or GeGLU (tanh GELU, the Gemma activation) when `gelu`.
#[derive(Debug)]
struct LlamaMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
    gelu: bool,
}

impl LlamaMlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let g = self.gate.forward(x)?;
        let g = if self.gelu { gelu_tanh(&g)? } else { silu(&g)? };
        let up = self.up.forward(x)?;
        self.down.forward(&multiply(&g, &up)?)
    }
}

/// A sparse Mixture-of-Experts feed-forward (Qwen2-MoE, DeepSeek-V2): a softmax router over `experts`
/// (top-k per token) plus an always-on `shared` expert. Correctness-first — each expert runs only on
/// its routed tokens (gathered, then scatter-added back), so active compute scales with
/// `experts_per_tok`. Top-k selection is done on the host. `n_group`/`topk_group` group-limited
/// routing (DeepSeek-V2-236B / V3) is not modelled — V2-Lite uses plain greedy top-k.
#[derive(Debug)]
struct MoeMlp {
    /// Router weight `[num_experts, hidden]`.
    router: Array,
    experts: Vec<LlamaMlp>,
    shared: LlamaMlp,
    /// Shared-expert sigmoid gate `[1, hidden]` (Qwen2-MoE); `None` ⇒ added ungated (DeepSeek-V2).
    shared_gate: Option<Array>,
    experts_per_tok: usize,
    norm_topk_prob: bool,
    /// Multiplier on the (un-normalized) routed weights — DeepSeek's `routed_scaling_factor`; `1.0`
    /// for Qwen2-MoE. Ignored when `norm_topk_prob` (the weights are renormalized instead).
    routed_scaling_factor: f32,
}

impl MoeMlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s, h) = (sh[0], sh[1], sh[2]);
        let t = b * s;
        let dtype = x.dtype();
        let xf = x.reshape(&[t, h])?;
        let num_experts = self.experts.len();
        let k = self.experts_per_tok.min(num_experts).max(1);

        // Router probabilities (f32 softmax on the host, for a stable top-k).
        let logits = linear(&xf, &self.router, None)?; // [t, num_experts]
        let logits = to_f32_host(&logits)?; // row-major [t * num_experts]

        // Invert the per-token top-k into per-expert (token, weight) lists.
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
            // Renormalize the top-k weights to sum to 1, or apply the routed scaling factor.
            let (denom, post_scale) = if self.norm_topk_prob {
                (top.iter().map(|&e| probs[e]).sum::<f32>().max(f32::MIN_POSITIVE), 1.0)
            } else {
                (1.0, self.routed_scaling_factor)
            };
            for &e in top {
                routed[e].push((ti as i32, probs[e] / denom * post_scale));
            }
        }

        // Each expert runs on just its tokens; scatter the weighted outputs back.
        let mut out = zeros_dtype(&[t, h], dtype)?;
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

        // Always-on shared expert: Qwen2 gates it by sigmoid(x · shared_gateᵀ); DeepSeek adds it
        // ungated.
        let shared = self.shared.forward(&xf)?;
        let shared = match &self.shared_gate {
            Some(g) => {
                let sg = sigmoid(&linear(&xf, g, None)?)?; // [t, 1]
                multiply(&shared, &sg)?
            }
            None => shared,
        };
        Ok(add(&out, &shared)?.reshape(&[b, s, h])?)
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
