//! Model configuration parsed from a Hugging Face `config.json`.
//!
//! Value-based parsing (no `serde` derive) matching the mlx-gen provider convention, so config keys
//! can vary and default gracefully. Story 7156 covers the Llama family; BYO architecture dispatch
//! (story 7163) layers Qwen3 on top; model breadth (story 7173) adds Phi-3, Qwen2-MoE, Gemma-2,
//! GLM-4, and DeepSeek-V2 (MLA) behind the same single generic decoder. Mirrors candle-llm's
//! `config.rs` (the cross-backend blueprint) so the two backends dispatch identically.

use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::primitives::{QuantSpec, Rope};

/// The decoder architecture, dispatched from `config.json` (`architectures` / `model_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// Llama family (also Mistral, and dense Qwen2 — same decoder shape: no q/k norm; Qwen2 only
    /// adds q/k/v bias, auto-detected by weight-key presence).
    Llama,
    /// Qwen3 family: adds per-head q/k RMSNorm in attention.
    Qwen3,
    /// Phi-3 family: the Llama decoder shape, but with a **packed** `qkv_proj` (q‖k‖v) and a packed
    /// `gate_up_proj` (gate‖up) — split at load into the standard projections.
    Phi3,
    /// Qwen2-MoE family: Qwen2 attention (GQA **with q/k/v bias**, no q/k norm) + a sparse
    /// mixture-of-experts FFN (router + top-k experts + a shared expert).
    Qwen2Moe,
    /// Gemma-2 family: `(1 + weight)` RMSNorm, embedding ×√hidden, GeGLU, a `query_pre_attn_scalar`
    /// attention scale, attention- and final-logit soft-capping, and a 4-norm "sandwich" block.
    Gemma2,
    /// GLM-4 family: a 4-norm sandwich block (standard RMSNorm), q/k/v bias, a packed `gate_up_proj`,
    /// and **partial, interleaved** RoPE (only `head_dim · partial_rotary_factor` dims are rotated).
    Glm4,
    /// DeepSeek-V2 family: **Multi-head Latent Attention (MLA)** — a low-rank compressed KV path
    /// (`kv_a`/`kv_b` projections, a decoupled RoPE key sub-vector) that is a distinct attention
    /// implementation rather than GQA — plus a fine-grained MoE FFN (many routed experts, several
    /// shared experts, the first layers dense) and YaRN RoPE on the `qk_rope_head_dim` sub-vector.
    DeepseekV2,
    /// Qwen3.5/3.6 family (`model_type` `qwen3_5` / `qwen3_5_text`): a VLM-wrapped **hybrid
    /// linear-attention** decoder — 3-of-4 layers are Gated DeltaNet (linear attention) and 1-of-4 is
    /// gated full attention with partial RoPE; the 35B variant is MoE. **Dispatch + routing only
    /// today** (sc-7626) so it no longer misroutes to the JoyCaption vision provider; the hybrid
    /// decoder itself is built in sc-7627 (DeltaNet primitive + recurrent cache), sc-7628 (decoder),
    /// sc-7629 (27B), and sc-7630 (35B MoE). The decoder-shape predicates below stay at their
    /// defaults until then.
    Qwen35,
    /// Qwen3-VL family (`model_type` `qwen3_vl` / `text_config.model_type` `qwen3_vl_text`): a
    /// VLM-wrapped **standard full-attention** Qwen3 decoder (GQA + per-head q/k RMSNorm + SwiGLU)
    /// — NOT the Qwen3.6 hybrid — with standard (verbatim-weight) RMSNorm (`Qwen3VLTextRMSNorm` is
    /// plain `weight · x`; no `(1 + weight)` fold — that fold applies only to Gemma), a 256K
    /// context (`rope_theta` 5e6, `max_position_embeddings` 262144), and **interleaved multimodal
    /// RoPE** (`rope_scaling.mrope_interleaved` + `mrope_section`). The decoder weights nest under
    /// `model.language_model.*` (the ViT tower under `model.visual.*`, loaded by the vision path).
    /// The text path is plain 1-D RoPE (interleaved M-RoPE with equal t/h/w rows is bit-identical);
    /// the image path uses the interleaved-M-RoPE cos/sin table from story B (sc-8075).
    Qwen3Vl,
}

impl Architecture {
    /// Determine the architecture from a parsed `config.json`. A config with no `architectures` /
    /// `model_type` (e.g. a minimal synthetic config) defaults to [`Architecture::Llama`]; a config
    /// that names an unrecognized architecture is rejected.
    pub fn from_config(v: &Value) -> Result<Self> {
        // A VLM wrapper (LLaVA / Qwen-VL) nests the language decoder under `text_config`; dispatch on
        // that nested decoder when present, else the top-level config.
        let cfg = v.get("text_config").unwrap_or(v);
        let arch = cfg
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|s| s.as_str());
        let model_type = cfg.get("model_type").and_then(|s| s.as_str());
        let hay = format!(
            "{} {}",
            arch.unwrap_or("").to_lowercase(),
            model_type.unwrap_or("").to_lowercase()
        );
        // `qwen3_vl` / `qwen3_5` must each be tested before the bare `qwen3` (a substring of both),
        // and they are mutually distinct (`qwen3_vl` ⊄ `qwen3_5` and vice versa) so their order is
        // free.
        if hay.contains("qwen3_vl") || hay.contains("qwen3vl") {
            Ok(Architecture::Qwen3Vl)
        } else if hay.contains("qwen3_5") || hay.contains("qwen35") {
            Ok(Architecture::Qwen35)
        } else if hay.contains("qwen3") {
            Ok(Architecture::Qwen3)
        } else if hay.contains("qwen2_moe") || hay.contains("qwen2moe") {
            Ok(Architecture::Qwen2Moe)
        } else if hay.contains("gemma2") {
            Ok(Architecture::Gemma2)
        } else if hay.contains("glm4") {
            Ok(Architecture::Glm4)
        } else if hay.contains("deepseek") {
            Ok(Architecture::DeepseekV2)
        } else if hay.contains("phi3") {
            Ok(Architecture::Phi3)
        } else if hay.contains("llama")
            || hay.contains("mistral")
            || hay.contains("qwen2")
            || (arch.is_none() && model_type.is_none())
        {
            // Llama / Mistral / dense Qwen2 share the decoder shape — Qwen2 only adds q/k/v bias,
            // which the projection loader auto-detects by weight-key presence (no q/k norm, standard
            // RoPE, SwiGLU). A minimal config (no arch fields) also defaults here.
            Ok(Architecture::Llama)
        } else {
            Err(Error::Unsupported(format!(
                "unsupported architecture (architectures={arch:?}, model_type={model_type:?})"
            )))
        }
    }

    /// The model-family tag.
    pub fn family(self) -> &'static str {
        match self {
            Architecture::Llama => "llama",
            Architecture::Qwen3 => "qwen3",
            Architecture::Phi3 => "phi3",
            Architecture::Qwen2Moe => "qwen2_moe",
            Architecture::Gemma2 => "gemma2",
            Architecture::Glm4 => "glm4",
            Architecture::DeepseekV2 => "deepseek_v2",
            Architecture::Qwen35 => "qwen3_5",
            Architecture::Qwen3Vl => "qwen3_vl",
        }
    }

    /// Whether this is the Qwen3-VL decoder (drives the `text_config` descent, standard
    /// (verbatim-weight) RMSNorm — `Qwen3VLTextRMSNorm` is plain `weight · x`, no `(1 + weight)`
    /// fold — the `model.language_model.*` weight prefix, and the interleaved multimodal RoPE).
    pub fn is_qwen3_vl(self) -> bool {
        matches!(self, Architecture::Qwen3Vl)
    }

    /// Whether this is a Gemma-2 decoder (drives `(1+weight)` norms, embedding scaling, GeGLU, the
    /// sandwich-norm block, and logit soft-capping).
    pub fn is_gemma2(self) -> bool {
        matches!(self, Architecture::Gemma2)
    }

    /// Whether the block uses the 4-norm "sandwich" residual (Gemma-2, GLM-4) rather than the plain
    /// Llama pre-norm.
    pub fn is_sandwich(self) -> bool {
        matches!(self, Architecture::Gemma2 | Architecture::Glm4)
    }

    /// Whether RoPE uses the interleaved (GPT-J-style) pairing rather than NeoX half-split (GLM-4,
    /// and the DeepSeek MLA rope sub-vector — DeepSeek's de-interleave-then-rotate is equivalent to
    /// the interleaved convention applied to the raw projection).
    pub fn rope_interleaved(self) -> bool {
        matches!(self, Architecture::Glm4 | Architecture::DeepseekV2)
    }

    /// Whether attention uses Multi-head Latent Attention (DeepSeek-V2) rather than GQA.
    pub fn is_mla(self) -> bool {
        matches!(self, Architecture::DeepseekV2)
    }

    /// Whether attention applies per-head q/k RMSNorm (Qwen3 / Qwen3-VL).
    pub fn has_qk_norm(self) -> bool {
        matches!(self, Architecture::Qwen3 | Architecture::Qwen3Vl)
    }
}

/// `rope_scaling` parameters for the Llama-3 NTK-by-parts schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeScaling {
    /// Scaling factor (e.g. 8.0 for Llama-3.1).
    pub factor: f32,
    /// Low-frequency factor.
    pub low_freq_factor: f32,
    /// High-frequency factor.
    pub high_freq_factor: f32,
    /// Original (pre-scaling) max context.
    pub original_context: f32,
}

/// Mixture-of-Experts FFN parameters (Qwen2-MoE, DeepSeek-V2). Present when a layer's dense MLP is
/// replaced by a router + routed expert bank + shared expert(s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeConfig {
    /// Total number of routed experts (`num_experts` / `n_routed_experts`).
    pub num_experts: usize,
    /// Experts activated per token (top-k routing).
    pub num_experts_per_tok: usize,
    /// Inner width of each routed expert's SwiGLU.
    pub moe_intermediate_size: i32,
    /// Inner width of the always-on shared expert's SwiGLU (DeepSeek packs `n_shared_experts` shared
    /// experts into one MLP, so this is `n_shared_experts · moe_intermediate_size`).
    pub shared_expert_intermediate_size: i32,
    /// Whether the top-k routing weights are renormalized to sum to 1.
    pub norm_topk_prob: bool,
    /// Multiplier applied to the routed-expert weights when they are *not* renormalized (DeepSeek's
    /// `routed_scaling_factor`; `1.0` for Qwen2-MoE).
    pub routed_scaling_factor: f32,
    /// Number of leading layers that stay a dense MLP before the MoE layers begin (DeepSeek's
    /// `first_k_dense_replace`; `0` for Qwen2-MoE — every layer is MoE).
    pub first_k_dense_replace: usize,
}

/// Multi-head Latent Attention parameters (DeepSeek-V2). Present when attention uses a low-rank
/// compressed KV path instead of GQA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaConfig {
    /// Low-rank dimension of the query down-projection (`q_a_proj`); `None` ⇒ a full `q_proj`
    /// (DeepSeek-V2-Lite has no query LoRA).
    pub q_lora_rank: Option<i32>,
    /// Low-rank dimension of the shared KV down-projection (`kv_a_proj_with_mqa` → `kv_a_layernorm`).
    pub kv_lora_rank: i32,
    /// Per-head non-rotary (content) query/key dimension.
    pub qk_nope_head_dim: i32,
    /// Per-head rotary (decoupled-RoPE) query/key dimension.
    pub qk_rope_head_dim: i32,
    /// Per-head value dimension (may differ from the q/k head dim).
    pub v_head_dim: i32,
}

impl MlaConfig {
    /// The full per-head query/key dimension attended over: `qk_nope_head_dim + qk_rope_head_dim`.
    pub fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// YaRN RoPE-scaling parameters (DeepSeek-V2). A wavelength-ramped blend of extrapolated
/// (high-frequency) and interpolated (low-frequency, divided by `factor`) inverse frequencies, plus
/// an attention-softmax magnitude scale (`mscale`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnConfig {
    /// Context-extension factor (interpolated frequencies are divided by this).
    pub factor: f32,
    /// Fast-rotation boundary (dims with shorter wavelength are extrapolated unchanged).
    pub beta_fast: f32,
    /// Slow-rotation boundary (dims with longer wavelength are interpolated).
    pub beta_slow: f32,
    /// Original (pre-extension) max context.
    pub original_context: f32,
    /// Magnitude scale for the cos/sin tables.
    pub mscale: f32,
    /// Magnitude scale applied to the attention softmax scale (`mscale_all_dim`).
    pub mscale_all_dim: f32,
}

impl YarnConfig {
    /// YaRN attention-softmax magnitude scale: `0.1 · mscale_all_dim · ln(factor) + 1` (`1.0` when
    /// `factor ≤ 1`). The attention scale is multiplied by this **squared**.
    fn softmax_mscale(&self) -> f32 {
        if self.factor > 1.0 {
            0.1 * self.mscale_all_dim * self.factor.ln() + 1.0
        } else {
            1.0
        }
    }
}

/// Configuration for a generic causal decoder (Llama family + the breadth architectures).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// Model/residual width.
    pub hidden_size: i32,
    /// MLP inner width.
    pub intermediate_size: i32,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention (query) heads.
    pub num_heads: i32,
    /// Number of key/value heads (GQA; equals `num_heads` for MHA).
    pub num_kv_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Vocabulary size.
    pub vocab_size: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Optional Llama-3 RoPE scaling; `None` ⇒ standard RoPE.
    pub rope_scaling: Option<RopeScaling>,
    /// Whether `lm_head` is tied to the input embeddings.
    pub tie_word_embeddings: bool,
    /// The decoder architecture (drives q/k norm, the family tag, and the breadth-specific paths).
    pub architecture: Architecture,
    /// Max context length (`max_position_embeddings`); `0` if unspecified.
    pub max_position_embeddings: i32,
    /// Present when the snapshot stores **pre-quantized** projections (a `quantization` block in
    /// `config.json`, e.g. from the GGUF converter): the group size / bit width the stored
    /// `weight`/`scales`/`biases` tensors were packed with. `None` ⇒ a dense snapshot.
    pub quantization: Option<QuantSpec>,
    /// Mixture-of-Experts parameters, present for an MoE decoder (Qwen2-MoE, DeepSeek-V2); `None` ⇒
    /// dense MLP.
    pub moe: Option<MoeConfig>,
    /// Gemma-2 attention-score soft-cap (`attn_logit_softcapping`); `None` ⇒ no cap.
    pub attn_logit_softcap: Option<f32>,
    /// Gemma-2 final-logit soft-cap (`final_logit_softcapping`); `None` ⇒ no cap.
    pub final_logit_softcap: Option<f32>,
    /// Gemma-2 attention scale denominator (`query_pre_attn_scalar`); `None` ⇒ `head_dim`.
    pub query_pre_attn_scalar: Option<i32>,
    /// Fraction of `head_dim` that RoPE rotates (`partial_rotary_factor`); `1.0` ⇒ full rotary.
    pub partial_rotary_factor: f32,
    /// Multi-head Latent Attention parameters, present for a DeepSeek-V2 decoder; `None` ⇒ GQA.
    pub mla: Option<MlaConfig>,
    /// YaRN RoPE-scaling parameters (DeepSeek-V2); `None` ⇒ no YaRN.
    pub yarn: Option<YarnConfig>,
    /// Interleaved multimodal-RoPE section `[t, h, w]` (`rope_scaling.mrope_section`, sums to
    /// `rotary_dim/2`), present for Qwen3-VL; `None` ⇒ a plain 1-D RoPE model. Drives the per-channel
    /// axis assignment for image (3-D) positions; irrelevant to the text path (where all three rows
    /// are equal, so the interleaved table is bit-identical to plain 1-D RoPE).
    pub mrope_section: Option<[i32; 3]>,
}

impl ModelConfig {
    /// Parse from an already-decoded `config.json` value.
    pub fn from_json(v: &Value) -> Result<Self> {
        let architecture = Architecture::from_config(v)?;
        // Recognized for routing (so it no longer misroutes to the JoyCaption vision provider —
        // sc-7626), but the hybrid Gated-DeltaNet / gated-full-attention decoder is not built yet.
        // Fail clearly rather than mis-parsing the VLM-wrapped config as a dense decoder.
        if architecture == Architecture::Qwen35 {
            return Err(Error::Unsupported(
                "Qwen3.6 (model_type `qwen3_5`) is a hybrid linear-attention (Gated DeltaNet + gated \
                 full attention) decoder; text generation is not yet implemented (tracked: sc-7627 \
                 DeltaNet primitive, sc-7628 decoder, sc-7629 27B, sc-7630 35B MoE)"
                    .to_string(),
            ));
        }
        // The Qwen3-VL VLM wrapper nests the standard Qwen3 decoder fields under `text_config`
        // (`hidden_size`, `num_hidden_layers`, `rope_theta`, …). Read the decoder fields from there,
        // falling back to the top level for a non-wrapped config. Other architectures read top-level.
        let v = if architecture.is_qwen3_vl() {
            v.get("text_config").unwrap_or(v)
        } else {
            v
        };
        let int = |key: &str| -> Option<i32> { v.get(key).and_then(|x| x.as_i64()).map(|x| x as i32) };
        let req_int = |key: &str| -> Result<i32> {
            int(key).ok_or_else(|| Error::Config(format!("config.json missing integer `{key}`")))
        };
        let f32_opt =
            |key: &str| -> Option<f32> { v.get(key).and_then(|x| x.as_f64()).map(|x| x as f32) };

        let hidden_size = req_int("hidden_size")?;
        let num_heads = req_int("num_attention_heads")?;
        // Multi-head Latent Attention (DeepSeek-V2): present iff the KV low-rank dim is configured.
        // `q_lora_rank` is `null` for DeepSeek-V2-Lite (a full `q_proj`), an int for the larger models.
        let mla = int("kv_lora_rank").map(|kv_lora_rank| MlaConfig {
            q_lora_rank: int("q_lora_rank"),
            kv_lora_rank,
            qk_nope_head_dim: int("qk_nope_head_dim").unwrap_or(0),
            qk_rope_head_dim: int("qk_rope_head_dim").unwrap_or(0),
            v_head_dim: int("v_head_dim").unwrap_or(hidden_size / num_heads),
        });
        // MLA attends over `qk_nope_head_dim + qk_rope_head_dim` per head (no `head_dim` config key).
        let head_dim = match &mla {
            Some(m) => m.q_head_dim(),
            None => int("head_dim").unwrap_or(hidden_size / num_heads),
        };
        let num_kv_heads = int("num_key_value_heads").unwrap_or(num_heads);
        let num_layers = req_int("num_hidden_layers")? as usize;
        let intermediate_size = req_int("intermediate_size")?;
        let vocab_size = req_int("vocab_size")?;
        let rms_norm_eps = v
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(1e-5);
        let rope_theta = v
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(500_000.0);
        let tie_word_embeddings = v
            .get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            // Gemma always ties its (huge) embedding to the LM head; the config often omits the key.
            .unwrap_or(architecture.is_gemma2());
        let max_position_embeddings = int("max_position_embeddings").unwrap_or(0);

        let attn_logit_softcap = f32_opt("attn_logit_softcapping");
        let final_logit_softcap = f32_opt("final_logit_softcapping");
        let query_pre_attn_scalar = int("query_pre_attn_scalar");
        let partial_rotary_factor = f32_opt("partial_rotary_factor").unwrap_or(1.0);

        // A `quantization` block marks a pre-quantized snapshot (the GGUF converter writes it).
        let quantization = v.get("quantization").and_then(|q| {
            let group_size = q.get("group_size").and_then(|x| x.as_i64())? as i32;
            let bits = q.get("bits").and_then(|x| x.as_i64())? as i32;
            Some(QuantSpec { group_size, bits })
        });

        // Mixture-of-Experts FFN params: present iff a routed-expert count is configured
        // (`num_experts` for Qwen2-MoE, `n_routed_experts` for DeepSeek-V2).
        let moe = int("num_experts")
            .or_else(|| int("n_routed_experts"))
            .map(|num_experts| {
                let moe_inter = int("moe_intermediate_size").unwrap_or(intermediate_size);
                MoeConfig {
                    num_experts: num_experts as usize,
                    num_experts_per_tok: int("num_experts_per_tok").unwrap_or(num_experts).max(1)
                        as usize,
                    moe_intermediate_size: moe_inter,
                    // Qwen2 gives the shared width directly; DeepSeek packs `n_shared_experts` of them.
                    shared_expert_intermediate_size: int("shared_expert_intermediate_size")
                        .or_else(|| int("n_shared_experts").map(|n| n * moe_inter))
                        .unwrap_or(intermediate_size),
                    norm_topk_prob: v
                        .get("norm_topk_prob")
                        .and_then(|x| x.as_bool())
                        .unwrap_or(false),
                    routed_scaling_factor: f32_opt("routed_scaling_factor").unwrap_or(1.0),
                    first_k_dense_replace: int("first_k_dense_replace").unwrap_or(0).max(0) as usize,
                }
            });

        // YaRN RoPE scaling (DeepSeek-V2): a separate `rope_scaling` block of `type: "yarn"`.
        let yarn = v.get("rope_scaling").and_then(|rs| {
            let ty = rs
                .get("type")
                .or_else(|| rs.get("rope_type"))
                .and_then(|x| x.as_str());
            if ty != Some("yarn") {
                return None;
            }
            let f = |key: &str, default: f32| {
                rs.get(key)
                    .and_then(|x| x.as_f64())
                    .map(|x| x as f32)
                    .unwrap_or(default)
            };
            Some(YarnConfig {
                factor: f("factor", 1.0),
                beta_fast: f("beta_fast", 32.0),
                beta_slow: f("beta_slow", 1.0),
                original_context: f("original_max_position_embeddings", 4096.0),
                mscale: f("mscale", 1.0),
                mscale_all_dim: f("mscale_all_dim", 0.0),
            })
        });

        let rope_scaling = v.get("rope_scaling").and_then(|rs| {
            // Only the "llama3" schedule is parsed here; absent / other types (e.g. yarn) fall back to
            // standard / the dedicated yarn path.
            let f = |key: &str, default: f32| {
                rs.get(key)
                    .and_then(|x| x.as_f64())
                    .map(|x| x as f32)
                    .unwrap_or(default)
            };
            let is_llama3 = rs
                .get("rope_type")
                .or_else(|| rs.get("type"))
                .and_then(|x| x.as_str())
                .map(|s| s == "llama3")
                .unwrap_or(true); // a bare factor block is treated as llama3
            if !is_llama3 {
                return None;
            }
            Some(RopeScaling {
                factor: f("factor", 1.0),
                low_freq_factor: f("low_freq_factor", 1.0),
                high_freq_factor: f("high_freq_factor", 4.0),
                original_context: f("original_max_position_embeddings", 8192.0),
            })
        });

        // Interleaved multimodal-RoPE section `[t, h, w]` (Qwen3-VL `rope_scaling.mrope_section`).
        let mrope_section = v
            .get("rope_scaling")
            .and_then(|rs| rs.get("mrope_section"))
            .and_then(|x| x.as_array())
            .filter(|a| a.len() == 3)
            .map(|a| {
                let g = |i: usize| a[i].as_i64().unwrap_or(0) as i32;
                [g(0), g(1), g(2)]
            });

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            rope_scaling,
            tie_word_embeddings,
            architecture,
            max_position_embeddings,
            quantization,
            moe,
            attn_logit_softcap,
            final_logit_softcap,
            query_pre_attn_scalar,
            partial_rotary_factor,
            mla,
            yarn,
            mrope_section,
        })
    }

    /// Number of head dimensions RoPE rotates (`round(head_dim · partial_rotary_factor)`, even).
    pub fn rotary_dim(&self) -> i32 {
        let rd = (self.head_dim as f32 * self.partial_rotary_factor).round() as i32;
        rd & !1 // force even (RoPE rotates in pairs)
    }

    /// The interleaved multimodal-RoPE section `[t, h, w]` (Qwen3-VL), defaulting to an even split
    /// of `rotary_dim/2` (biased toward `t` then `h`) when the config omits `mrope_section`. Used by
    /// the image (3-D) RoPE path; the text path (all rows equal) is independent of the split.
    pub fn mrope_section_resolved(&self) -> [usize; 3] {
        if let Some(s) = self.mrope_section {
            return [s[0].max(0) as usize, s[1].max(0) as usize, s[2].max(0) as usize];
        }
        let half = (self.rotary_dim() / 2) as usize;
        let base = half / 3;
        let rem = half % 3;
        [base + (rem > 0) as usize, base + (rem > 1) as usize, base]
    }

    /// Whether the decoder uses a Mixture-of-Experts FFN.
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Whether the decoder uses Multi-head Latent Attention (DeepSeek-V2).
    pub fn is_mla(&self) -> bool {
        self.mla.is_some()
    }

    /// Whether attention applies per-head q/k RMSNorm (Qwen3).
    pub fn has_qk_norm(&self) -> bool {
        self.architecture.has_qk_norm()
    }

    /// Read and parse `config.json` from a snapshot directory (or a file path).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = if dir.is_dir() { dir.join("config.json") } else { dir.to_path_buf() };
        let text = std::fs::read_to_string(&path)?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("parse {}: {e}", path.display())))?;
        Self::from_json(&v)
    }

    /// Build the RoPE for this config. DeepSeek-V2 (MLA) rotates only the `qk_rope_head_dim`
    /// sub-vector with YaRN-scaled, interleaved frequencies; GLM-4 uses partial/interleaved RoPE;
    /// Llama-3 scaling applies the NTK-by-parts schedule; otherwise standard full-width NeoX RoPE.
    pub fn build_rope(&self) -> Rope {
        if let Some(mla) = self.mla {
            let rope_dim = mla.qk_rope_head_dim;
            return match self.yarn {
                Some(y) => Rope::yarn(
                    rope_dim,
                    self.rope_theta,
                    y.factor,
                    y.beta_fast,
                    y.beta_slow,
                    y.original_context,
                ),
                None => Rope::partial(rope_dim, self.rope_theta, true),
            };
        }
        let rotary_dim = self.rotary_dim();
        if rotary_dim < self.head_dim || self.architecture.rope_interleaved() {
            return Rope::partial(rotary_dim, self.rope_theta, self.architecture.rope_interleaved());
        }
        match self.rope_scaling {
            Some(rs) => Rope::llama3(
                self.head_dim,
                self.rope_theta,
                rs.factor,
                rs.low_freq_factor,
                rs.high_freq_factor,
                rs.original_context,
            ),
            None => Rope::standard(self.head_dim, self.rope_theta),
        }
    }

    /// Number of GQA groups (`num_heads / num_kv_heads`).
    pub fn groups(&self) -> i32 {
        self.num_heads / self.num_kv_heads
    }

    /// Attention scale. MLA (DeepSeek-V2) scales by `q_head_dim^(-0.5)` multiplied by the YaRN
    /// magnitude `mscale²`; Gemma-2 scales by `query_pre_attn_scalar^(-0.5)` (a denominator that may
    /// differ from `head_dim`); otherwise the usual `head_dim^(-0.5)`.
    pub fn attn_scale(&self) -> f32 {
        if let Some(mla) = self.mla {
            let base = (mla.q_head_dim() as f32).powf(-0.5);
            let mscale = self.yarn.map_or(1.0, |y| y.softmax_mscale());
            return base * mscale * mscale;
        }
        let denom = self.query_pre_attn_scalar.unwrap_or(self.head_dim) as f32;
        denom.powf(-0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_llama31_8b_style_config() {
        let v = json!({
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            }
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.head_dim, 128); // 4096/32
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.groups(), 4);
        assert_eq!(cfg.rope_scaling.unwrap().factor, 8.0);
    }

    #[test]
    fn defaults_kv_heads_and_head_dim() {
        let v = json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 32
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.head_dim, 16); // 64/4
        assert_eq!(cfg.num_kv_heads, 4); // defaults to num_heads (MHA)
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn missing_required_field_errors() {
        let v = json!({ "hidden_size": 64 });
        assert!(matches!(ModelConfig::from_json(&v), Err(Error::Config(_))));
    }

    #[test]
    fn architecture_dispatch() {
        let qwen3 = json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" });
        assert_eq!(Architecture::from_config(&qwen3).unwrap(), Architecture::Qwen3);

        let llama = json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" });
        assert_eq!(Architecture::from_config(&llama).unwrap(), Architecture::Llama);

        let mistral = json!({ "architectures": ["MistralForCausalLM"] });
        assert_eq!(Architecture::from_config(&mistral).unwrap(), Architecture::Llama);

        // Dense Qwen2 shares the Llama decoder shape (q/k/v bias auto-detected); routed to Llama.
        let qwen2 = json!({ "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2" });
        let a = Architecture::from_config(&qwen2).unwrap();
        assert_eq!(a, Architecture::Llama);
        assert!(!a.has_qk_norm());

        // Minimal config (no arch fields) defaults to Llama.
        let minimal = json!({ "hidden_size": 8 });
        assert_eq!(Architecture::from_config(&minimal).unwrap(), Architecture::Llama);

        // A named-but-unsupported arch is rejected.
        let unknown = json!({ "architectures": ["MambaForCausalLM"], "model_type": "mamba" });
        assert!(matches!(Architecture::from_config(&unknown), Err(Error::Unsupported(_))));

        // Phi-3 (packed qkv / gate_up; otherwise the Llama shape).
        let phi3 = json!({ "architectures": ["Phi3ForCausalLM"], "model_type": "phi3" });
        let a = Architecture::from_config(&phi3).unwrap();
        assert_eq!(a, Architecture::Phi3);
        assert_eq!(a.family(), "phi3");
        assert!(!a.has_qk_norm());

        // Qwen2-MoE (sparse FFN + q/k/v bias; no q/k norm).
        let qwen2_moe = json!({ "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe" });
        let a = Architecture::from_config(&qwen2_moe).unwrap();
        assert_eq!(a, Architecture::Qwen2Moe);
        assert_eq!(a.family(), "qwen2_moe");
        assert!(!a.has_qk_norm());

        // Gemma-2 (soft-caps, sandwich norms; ties embeddings by default).
        let gemma2 = json!({ "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2" });
        let a = Architecture::from_config(&gemma2).unwrap();
        assert_eq!(a, Architecture::Gemma2);
        assert_eq!(a.family(), "gemma2");
        assert!(a.is_gemma2());
        assert!(a.is_sandwich());
        assert!(!a.has_qk_norm());

        // GLM-4 (sandwich norms, partial+interleaved RoPE; standard RMSNorm).
        let glm4 = json!({ "architectures": ["Glm4ForCausalLM"], "model_type": "glm4" });
        let a = Architecture::from_config(&glm4).unwrap();
        assert_eq!(a, Architecture::Glm4);
        assert_eq!(a.family(), "glm4");
        assert!(a.is_sandwich());
        assert!(a.rope_interleaved());
        assert!(!a.is_gemma2());
    }

    #[test]
    fn parses_glm4_config_partial_rotary() {
        let v = json!({
            "architectures": ["Glm4ForCausalLM"], "model_type": "glm4",
            "hidden_size": 4096, "intermediate_size": 13696, "num_hidden_layers": 40,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "vocab_size": 151552, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Glm4);
        assert_eq!(cfg.partial_rotary_factor, 0.5);
        assert_eq!(cfg.rotary_dim(), 64); // 128 * 0.5
        let rope = cfg.build_rope();
        assert_eq!(rope.dim(), 64);
        assert!(rope.interleaved());
        // A model with no partial_rotary_factor rotates the full head_dim.
        let full = ModelConfig::from_json(&json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 32
        }))
        .unwrap();
        assert_eq!(full.partial_rotary_factor, 1.0);
        assert_eq!(full.rotary_dim(), full.head_dim);
    }

    #[test]
    fn parses_gemma2_config() {
        let v = json!({
            "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2",
            "hidden_size": 2304, "intermediate_size": 9216, "num_hidden_layers": 26,
            "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 256000,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "query_pre_attn_scalar": 256,
            "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0
            // tie_word_embeddings intentionally omitted — Gemma ties by default.
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Gemma2);
        assert_eq!(cfg.head_dim, 256); // explicit, != 2304/8
        assert!(cfg.tie_word_embeddings, "Gemma ties by default");
        assert_eq!(cfg.attn_logit_softcap, Some(50.0));
        assert_eq!(cfg.final_logit_softcap, Some(30.0));
        assert_eq!(cfg.query_pre_attn_scalar, Some(256));
        // Attention scale uses query_pre_attn_scalar (256), not head_dim — equal here, but the path
        // is exercised.
        assert!((cfg.attn_scale() - (256f32).powf(-0.5)).abs() < 1e-9);
    }

    #[test]
    fn parses_qwen2_moe_config() {
        let v = json!({
            "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
            "hidden_size": 2048, "intermediate_size": 5632, "num_hidden_layers": 24,
            "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 151936,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "num_experts": 60, "num_experts_per_tok": 4, "norm_topk_prob": false,
            "moe_intermediate_size": 1408, "shared_expert_intermediate_size": 5632
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen2Moe);
        assert!(cfg.is_moe());
        let moe = cfg.moe.unwrap();
        assert_eq!(moe.num_experts, 60);
        assert_eq!(moe.num_experts_per_tok, 4);
        assert_eq!(moe.moe_intermediate_size, 1408);
        assert_eq!(moe.shared_expert_intermediate_size, 5632);
        assert!(!moe.norm_topk_prob);
        // A dense (non-MoE) config has no MoE block.
        assert!(!ModelConfig::from_json(&json!({
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "vocab_size": 32
        }))
        .unwrap()
        .is_moe());
    }

    #[test]
    fn parses_deepseek_v2_lite_config() {
        // DeepSeek-V2-Lite-Chat: MLA (no query LoRA), fine-grained MoE with a leading dense layer,
        // YaRN RoPE. The numbers are the real config's.
        let v = json!({
            "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
            "hidden_size": 2048, "intermediate_size": 10944, "num_hidden_layers": 27,
            "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 102400,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 163840,
            "tie_word_embeddings": false,
            "q_lora_rank": null, "kv_lora_rank": 512,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
            "n_routed_experts": 64, "num_experts_per_tok": 6, "n_shared_experts": 2,
            "moe_intermediate_size": 1408, "first_k_dense_replace": 1, "norm_topk_prob": false,
            "routed_scaling_factor": 1.0,
            "rope_scaling": {
                "type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1,
                "mscale": 0.707, "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 4096
            }
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::DeepseekV2);
        assert!(cfg.is_mla());
        assert!(cfg.architecture.rope_interleaved());

        let mla = cfg.mla.unwrap();
        assert_eq!(mla.q_lora_rank, None); // V2-Lite has a full q_proj
        assert_eq!(mla.kv_lora_rank, 512);
        assert_eq!(mla.qk_nope_head_dim, 128);
        assert_eq!(mla.qk_rope_head_dim, 64);
        assert_eq!(mla.v_head_dim, 128);
        assert_eq!(mla.q_head_dim(), 192);
        // head_dim is overridden to the MLA q/k head dim (no `head_dim` config key).
        assert_eq!(cfg.head_dim, 192);

        let moe = cfg.moe.unwrap();
        assert_eq!(moe.num_experts, 64);
        assert_eq!(moe.num_experts_per_tok, 6);
        assert_eq!(moe.moe_intermediate_size, 1408);
        assert_eq!(moe.shared_expert_intermediate_size, 2 * 1408); // n_shared_experts · moe_inter
        assert_eq!(moe.first_k_dense_replace, 1);
        assert!(!moe.norm_topk_prob);
        assert_eq!(moe.routed_scaling_factor, 1.0);

        let yarn = cfg.yarn.unwrap();
        assert_eq!(yarn.factor, 40.0);
        assert_eq!(yarn.original_context, 4096.0);
        // The legacy llama3 `rope_scaling` path must not also fire for a yarn block.
        assert!(cfg.rope_scaling.is_none());

        // Attention scale folds in the YaRN mscale²: q_head_dim^-0.5 · (0.1·0.707·ln40 + 1)².
        let mscale = 0.1 * 0.707 * 40f32.ln() + 1.0;
        let expected = (192f32).powf(-0.5) * mscale * mscale;
        assert!((cfg.attn_scale() - expected).abs() < 1e-6, "{}", cfg.attn_scale());

        // The RoPE rotates only the 64-dim sub-vector, interleaved.
        let rope = cfg.build_rope();
        assert_eq!(rope.dim(), 64);
        assert!(rope.interleaved());
    }

    #[test]
    fn parses_quantization_block() {
        let v = json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 32,
            "quantization": { "group_size": 64, "bits": 4 }
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.quantization, Some(QuantSpec { group_size: 64, bits: 4 }));

        // Absent block ⇒ dense snapshot.
        let dense = json!({
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 32
        });
        assert_eq!(ModelConfig::from_json(&dense).unwrap().quantization, None);
    }

    #[test]
    fn qwen3_config_has_qk_norm_and_explicit_head_dim() {
        let v = json!({
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "tie_word_embeddings": true, "max_position_embeddings": 40960
        });
        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen3);
        assert!(cfg.has_qk_norm());
        assert_eq!(cfg.head_dim, 128); // explicit, != 1024/16
        assert_eq!(cfg.max_position_embeddings, 40960);
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn qwen35_dispatches_via_text_config_and_is_recognized_for_routing() {
        // Qwen3.6 self-identifies as `qwen3_5`, wrapped as a VLM with the real decoder under
        // `text_config` (`qwen3_5_text`). Dispatch descends into text_config and must NOT mistake
        // `qwen3_5` for plain `qwen3` (a substring).
        let v = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5", "depth": 27 }
        });
        assert_eq!(Architecture::from_config(&v).unwrap(), Architecture::Qwen35);
        assert_eq!(Architecture::Qwen35.family(), "qwen3_5");

        // Plain Qwen3 is unaffected (the `qwen3_5`-before-`qwen3` ordering).
        let q3 = json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" });
        assert_eq!(Architecture::from_config(&q3).unwrap(), Architecture::Qwen3);
    }

    #[test]
    fn qwen3vl_dispatches_and_parses_nested_text_config() {
        // Qwen3-VL self-identifies as `qwen3_vl`, wrapped as a VLM with the standard full-attention
        // Qwen3 text decoder under `text_config` (`qwen3_vl_text`). The dispatch must pick Qwen3-VL
        // (NOT the bare `qwen3` substring, NOT the hybrid `qwen3_5`), and the config must parse the
        // decoder fields out of `text_config` along with the 256K RoPE and interleaved M-RoPE section.
        // These mirror the cached Qwen/Qwen3-VL-8B-Instruct rev 0c351dd0 `text_config`.
        let v = json!({
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "image_token_id": 151655,
            "model_type": "qwen3_vl",
            "tie_word_embeddings": false,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "text_config": {
                "model_type": "qwen3_vl_text",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000,
                "max_position_embeddings": 262144,
                "rope_scaling": { "mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default" }
            },
            "vision_config": { "model_type": "qwen3_vl", "depth": 27 }
        });
        assert_eq!(Architecture::from_config(&v).unwrap(), Architecture::Qwen3Vl);
        assert_eq!(Architecture::Qwen3Vl.family(), "qwen3_vl");
        assert!(Architecture::Qwen3Vl.is_qwen3_vl());
        assert!(Architecture::Qwen3Vl.has_qk_norm());

        let cfg = ModelConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
        // Decoder shape read from text_config.
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.groups(), 4); // GQA 32/8
        assert_eq!(cfg.head_dim, 128); // explicit, != 4096/32
        assert_eq!(cfg.vocab_size, 151936);
        assert!(cfg.has_qk_norm());
        assert!(!cfg.tie_word_embeddings, "Qwen3-VL lm_head is untied");
        // RoPE / 256K context.
        assert_eq!(cfg.rope_theta, 5_000_000.0);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert_eq!(cfg.partial_rotary_factor, 1.0); // full rotary (rope_type "default")
        assert_eq!(cfg.rotary_dim(), 128);
        // Interleaved M-RoPE section parsed from rope_scaling.
        assert_eq!(cfg.mrope_section, Some([24, 20, 20]));
        assert_eq!(cfg.mrope_section_resolved(), [24, 20, 20]); // sums to dim/2 = 64
        assert_eq!(cfg.mrope_section_resolved().iter().sum::<usize>(), (cfg.rotary_dim() / 2) as usize);
        // The text-path RoPE is plain full-width NeoX (theta 5e6); not interleaved, not partial, no yarn.
        let rope = cfg.build_rope();
        assert_eq!(rope.dim(), 128);
        assert!(!rope.interleaved());
        assert!(cfg.yarn.is_none());
        assert!(cfg.rope_scaling.is_none()); // rope_type "default" is neither llama3 nor yarn
    }

    #[test]
    fn qwen35_modelconfig_fails_clearly_until_decoder_lands() {
        // Recognized for routing, but generation isn't implemented yet — the parse must fail with a
        // clear Unsupported message (NOT a cryptic missing-field error, NOT the JoyCaption error).
        let v = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text", "hidden_size": 5120 },
            "vision_config": { "model_type": "qwen3_5" }
        });
        match ModelConfig::from_json(&v) {
            Err(Error::Unsupported(m)) => {
                assert!(m.contains("qwen3_5"), "{m}");
                assert!(m.contains("sc-7627") || m.contains("not yet implemented"), "{m}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
