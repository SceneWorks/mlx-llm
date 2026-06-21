//! Model configuration parsed from a Hugging Face `config.json`.
//!
//! Value-based parsing (no `serde` derive) matching the mlx-gen provider convention, so config keys
//! can vary and default gracefully. Story 7156 covers the Llama family; BYO architecture dispatch
//! (story 7163) layers on top.

use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::primitives::Rope;

/// The decoder architecture, dispatched from `config.json` (`architectures` / `model_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// Llama family (also Mistral — same decoder shape: no q/k norm, no QKV bias).
    Llama,
    /// Qwen3 family: adds per-head q/k RMSNorm in attention.
    Qwen3,
}

impl Architecture {
    /// Determine the architecture from a parsed `config.json`. A config with no `architectures` /
    /// `model_type` (e.g. a minimal synthetic config) defaults to [`Architecture::Llama`]; a config
    /// that names an unrecognized architecture is rejected.
    pub fn from_config(v: &Value) -> Result<Self> {
        let arch = v
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|s| s.as_str());
        let model_type = v.get("model_type").and_then(|s| s.as_str());
        let hay = format!(
            "{} {}",
            arch.unwrap_or("").to_lowercase(),
            model_type.unwrap_or("").to_lowercase()
        );
        if hay.contains("qwen3") {
            Ok(Architecture::Qwen3)
        } else if hay.contains("llama")
            || hay.contains("mistral")
            || (arch.is_none() && model_type.is_none())
        {
            // Llama/Mistral share the decoder shape; a minimal config (no arch fields) defaults here.
            Ok(Architecture::Llama)
        } else {
            Err(Error::Unsupported(format!(
                "unsupported architecture (architectures={arch:?}, model_type={model_type:?})"
            )))
        }
    }

    /// The model-family tag (`"llama"` / `"qwen3"`).
    pub fn family(self) -> &'static str {
        match self {
            Architecture::Llama => "llama",
            Architecture::Qwen3 => "qwen3",
        }
    }

    /// Whether attention applies per-head q/k RMSNorm (Qwen3).
    pub fn has_qk_norm(self) -> bool {
        matches!(self, Architecture::Qwen3)
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

/// Configuration for a Llama-family decoder.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
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
    /// The decoder architecture (drives q/k norm and the family tag).
    pub architecture: Architecture,
    /// Max context length (`max_position_embeddings`); `0` if unspecified.
    pub max_position_embeddings: i32,
}

impl LlamaConfig {
    /// Parse from an already-decoded `config.json` value.
    pub fn from_json(v: &Value) -> Result<Self> {
        let int = |key: &str| -> Option<i32> { v.get(key).and_then(|x| x.as_i64()).map(|x| x as i32) };
        let req_int = |key: &str| -> Result<i32> {
            int(key).ok_or_else(|| Error::Config(format!("config.json missing integer `{key}`")))
        };

        let hidden_size = req_int("hidden_size")?;
        let num_heads = req_int("num_attention_heads")?;
        let head_dim = int("head_dim").unwrap_or(hidden_size / num_heads);
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
            .unwrap_or(false);
        let architecture = Architecture::from_config(v)?;
        let max_position_embeddings = int("max_position_embeddings").unwrap_or(0);

        let rope_scaling = v.get("rope_scaling").and_then(|rs| {
            // Only the "llama3" schedule is parsed; absent / other types fall back to standard RoPE.
            let f = |key: &str, default: f32| rs.get(key).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(default);
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
        })
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

    /// Build the RoPE for this config (Llama-3 scaled when `rope_scaling` is present, else standard).
    pub fn build_rope(&self) -> Rope {
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

    /// Attention scale, `head_dim^(-0.5)`.
    pub fn attn_scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
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
        let cfg = LlamaConfig::from_json(&v).unwrap();
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
        let cfg = LlamaConfig::from_json(&v).unwrap();
        assert_eq!(cfg.head_dim, 16); // 64/4
        assert_eq!(cfg.num_kv_heads, 4); // defaults to num_heads (MHA)
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn missing_required_field_errors() {
        let v = json!({ "hidden_size": 64 });
        assert!(matches!(LlamaConfig::from_json(&v), Err(Error::Config(_))));
    }

    #[test]
    fn architecture_dispatch() {
        let qwen3 = json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" });
        assert_eq!(Architecture::from_config(&qwen3).unwrap(), Architecture::Qwen3);

        let llama = json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" });
        assert_eq!(Architecture::from_config(&llama).unwrap(), Architecture::Llama);

        let mistral = json!({ "architectures": ["MistralForCausalLM"] });
        assert_eq!(Architecture::from_config(&mistral).unwrap(), Architecture::Llama);

        // Minimal config (no arch fields) defaults to Llama.
        let minimal = json!({ "hidden_size": 8 });
        assert_eq!(Architecture::from_config(&minimal).unwrap(), Architecture::Llama);

        // A named-but-unsupported arch is rejected.
        let unknown = json!({ "architectures": ["MambaForCausalLM"], "model_type": "mamba" });
        assert!(matches!(Architecture::from_config(&unknown), Err(Error::Unsupported(_))));
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
        let cfg = LlamaConfig::from_json(&v).unwrap();
        assert_eq!(cfg.architecture, Architecture::Qwen3);
        assert!(cfg.has_qk_norm());
        assert_eq!(cfg.head_dim, 128); // explicit, != 1024/16
        assert_eq!(cfg.max_position_embeddings, 40960);
        assert!(cfg.rope_scaling.is_none());
    }
}
