//! Persisted MLX snapshot writer, shared by the GGUF and HF-safetensors ingest paths (epic 7153,
//! story 7660).
//!
//! A *snapshot* is the `{config.json, model.safetensors, tokenizer.json, tokenizer_config.json}`
//! directory the engine loads ([`crate::models::CausalLm::from_weights`]). Two producers feed it:
//! the GGUF converter ([`crate::gguf::convert`], story 7165) and — added here — a Hugging Face
//! safetensors directory. Both funnel their dense tensor set through one sink, [`write_snapshot`],
//! so the requant + write logic lives in a single place.
//!
//! [`write_snapshot`] optionally re-quantizes the attention/MLP **projection** weights to MLX
//! group-wise Q4/Q8 ([`QuantizedLinear::quantize`]) and stores them as packed
//! `weight`/`scales`/`biases`, keeping embeddings, the LM head, and norms dense — the engine's
//! quant invariant. It writes a `config.json` carrying a matching `quantization` block (so the
//! loader reads the projections through its existing pre-quantized branch, llama.rs:69, with no
//! loader change) and drops the tokenizer files through verbatim.
//!
//! [`write_hf_snapshot`] is the HF leaf: load a dense HF model directory via [`Weights`] and persist
//! it as such a snapshot. With `quantize: None` the weights are written through unchanged, so the
//! snapshot reloads bit-identically to loading the source directly. Quantization operates at the
//! tensor level (projection weights selected by name), so it is architecture-agnostic — not
//! Llama-specific — and covers any decoder whose projections use the HF key layout.

use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::primitives::quant::QuantizedLinear;
use crate::primitives::QuantSpec;
use crate::primitives::Weights;

/// The engine's bf16 compute/storage dtype. Projection weights are cast to it before requant so a
/// quantized snapshot matches the loader's compute path (and the GGUF converter's behavior).
const STORE_DTYPE: Dtype = Dtype::Bfloat16;

/// The attention/MLP projection weight suffixes quantization targets. Selection is at the tensor
/// level on the HF key layout, so it is architecture-agnostic (Llama / Qwen3 / … and VLM decoders);
/// embeddings (`model.embed_tokens.weight`), the LM head (`lm_head.weight`), and norms never match
/// and so stay dense — the engine's quant invariant ([`crate::models::CausalLm::from_weights_with`]).
pub const PROJECTION_SUFFIXES: [&str; 7] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// Whether a weight key is a quantization-eligible attention/MLP projection.
pub fn is_projection(key: &str) -> bool {
    PROJECTION_SUFFIXES.iter().any(|s| key.ends_with(s))
}

/// Tokenizer files to drop into a snapshot, written verbatim. The GGUF path supplies its
/// reconstructed `tokenizer.json` / `tokenizer_config.json` (serialized); the HF path supplies the
/// source files read through byte-for-byte. Either may be `None` (no file written).
#[derive(Clone, Debug, Default)]
pub struct SnapshotTokenizer {
    /// `tokenizer.json` contents.
    pub tokenizer_json: Option<String>,
    /// `tokenizer_config.json` contents.
    pub tokenizer_config_json: Option<String>,
}

/// What writing a snapshot produced.
#[derive(Clone, Debug)]
pub struct SnapshotReport {
    /// Number of weight tensors written (a quantized projection contributes three:
    /// `weight`/`scales`/`biases`).
    pub num_tensors: usize,
    /// The requant scheme applied to the projections, if any (`None` ⇒ dense).
    pub quantized: Option<QuantSpec>,
    /// Directory the snapshot was written to.
    pub out_dir: PathBuf,
}

/// Write a loadable MLX snapshot to `out_dir` from a dense, HF-keyed tensor set.
///
/// When `quantize` is `Some`, each attention/MLP projection weight ([`is_projection`]) is cast to
/// bf16 and re-quantized to MLX group-wise Q4/Q8, stored as packed `weight`/`scales`/`biases`;
/// every other tensor (embeddings, LM head, norms, anything else) is written through unchanged, and
/// a matching `quantization` block is added to `config`. When `quantize` is `None` every tensor is
/// written through unchanged — a dense snapshot.
pub fn write_snapshot(
    out_dir: &Path,
    tensors: impl IntoIterator<Item = (String, Array)>,
    mut config: Value,
    tokenizer: &SnapshotTokenizer,
    quantize: Option<QuantSpec>,
) -> Result<SnapshotReport> {
    std::fs::create_dir_all(out_dir)?;

    // Build the safetensors set: projections optionally requantized (cast to bf16 first), the rest
    // written through unchanged.
    let mut out: Vec<(String, Array)> = Vec::new();
    for (key, arr) in tensors {
        match quantize {
            Some(spec) if is_projection(&key) => {
                let w = arr.as_dtype(STORE_DTYPE)?;
                let q = QuantizedLinear::quantize(&w, spec.group_size, spec.bits, None)?;
                let base = key.strip_suffix(".weight").unwrap_or(&key);
                out.push((format!("{base}.weight"), q.weight));
                out.push((format!("{base}.scales"), q.scales));
                out.push((format!("{base}.biases"), q.biases));
            }
            _ => out.push((key, arr)),
        }
    }
    let num_tensors = out.len();

    // A `quantization` block marks the snapshot pre-quantized so the loader reads the stored
    // projections as-is (its `stored_quant` branch) rather than re-quantizing on load.
    if let Some(spec) = quantize {
        if let Value::Object(map) = &mut config {
            map.insert(
                "quantization".into(),
                json!({ "group_size": spec.group_size, "bits": spec.bits }),
            );
        } else {
            return Err(Error::Config(
                "snapshot config.json is not a JSON object".into(),
            ));
        }
    }
    write_json_string(&out_dir.join("config.json"), &config)?;

    Array::save_safetensors(
        out.iter().map(|(k, v)| (k.as_str(), v)),
        None,
        out_dir.join("model.safetensors"),
    )
    .map_err(|e| Error::Msg(format!("write model.safetensors: {e}")))?;

    if let Some(t) = &tokenizer.tokenizer_json {
        std::fs::write(out_dir.join("tokenizer.json"), t)?;
    }
    if let Some(t) = &tokenizer.tokenizer_config_json {
        std::fs::write(out_dir.join("tokenizer_config.json"), t)?;
    }

    Ok(SnapshotReport {
        num_tensors,
        quantized: quantize,
        out_dir: out_dir.to_path_buf(),
    })
}

/// Persist a dense Hugging Face safetensors model directory as an MLX snapshot, optionally
/// quantizing the projections to Q4/Q8.
///
/// The dense tensor set is loaded via [`Weights`] (single file or sharded) and handed to
/// [`write_snapshot`]; `config.json` is read through (with a `quantization` block added when
/// quantizing — every other key preserved) and `tokenizer.json` / `tokenizer_config.json` are
/// copied verbatim when present. With `quantize: None` the weights are written unchanged, so the
/// snapshot reloads bit-identically to loading the source directly.
pub fn write_hf_snapshot(
    source_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<QuantSpec>,
) -> Result<SnapshotReport> {
    let source = source_dir.as_ref();
    let out_dir = out_dir.as_ref();

    // config.json is required — it carries the architecture + shapes the loader dispatches on. Read
    // it as a Value so the writer can add the quantization block; all other keys pass through.
    let config_path = source.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::Config(format!("read {}: {e}", config_path.display())))?;
    let config: Value = serde_json::from_str(&config_text)
        .map_err(|e| Error::Config(format!("parse {}: {e}", config_path.display())))?;

    // Tokenizer files pass through verbatim (byte-identical) when present.
    let tokenizer = SnapshotTokenizer {
        tokenizer_json: read_to_string_if_exists(&source.join("tokenizer.json"))?,
        tokenizer_config_json: read_to_string_if_exists(&source.join("tokenizer_config.json"))?,
    };

    let weights = Weights::from_dir(source)?;
    write_snapshot(out_dir, weights.into_map(), config, &tokenizer, quantize)
}

/// Write a JSON value to `path`, pretty-printed (the snapshot's `config.json`).
fn write_json_string(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(path, text)
        .map_err(|e| Error::Msg(format!("write {}: {e}", path.display())))?;
    Ok(())
}

/// Read a file to a string, returning `None` if it does not exist (other IO errors propagate).
fn read_to_string_if_exists(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::Msg(format!("read {}: {e}", path.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::models::CausalLm;
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    use std::collections::HashMap;

    fn unique_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mlx-llm-snapshot-{label}-{}", std::process::id()))
    }

    fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Array::from_slice(&data, shape)
    }

    /// A complete tiny Llama tensor set + matching config.json. Widths are multiples of the Q4/Q8
    /// group size (64) so the projections' input dim is quantizable: hidden 64 (head_dim 32 ×
    /// 2 heads), intermediate 128.
    fn tiny_model() -> (Vec<(String, Array)>, Value) {
        let (h, v, inter, qd, kvd, layers) = (64i32, 4i32, 128i32, 64i32, 32i32, 2usize);
        let mut rng = SplitMix64::new(0xABCDEF);
        let mut t: Vec<(String, Array)> = Vec::new();
        t.push(("model.embed_tokens.weight".into(), randn(&[v, h], &mut rng)));
        t.push(("model.norm.weight".into(), Array::ones::<f32>(&[h]).unwrap()));
        t.push(("lm_head.weight".into(), randn(&[v, h], &mut rng)));
        for i in 0..layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            t.push((p("input_layernorm.weight"), Array::ones::<f32>(&[h]).unwrap()));
            t.push((p("post_attention_layernorm.weight"), Array::ones::<f32>(&[h]).unwrap()));
            t.push((p("self_attn.q_proj.weight"), randn(&[qd, h], &mut rng)));
            t.push((p("self_attn.k_proj.weight"), randn(&[kvd, h], &mut rng)));
            t.push((p("self_attn.v_proj.weight"), randn(&[kvd, h], &mut rng)));
            t.push((p("self_attn.o_proj.weight"), randn(&[h, qd], &mut rng)));
            t.push((p("mlp.gate_proj.weight"), randn(&[inter, h], &mut rng)));
            t.push((p("mlp.up_proj.weight"), randn(&[inter, h], &mut rng)));
            t.push((p("mlp.down_proj.weight"), randn(&[h, inter], &mut rng)));
        }
        let config = json!({
            "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": layers,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": v,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            "eos_token_id": 99
        });
        (t, config)
    }

    #[test]
    fn projection_predicate_selects_only_attn_mlp_projections() {
        for k in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.7.self_attn.o_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
        ] {
            assert!(is_projection(k), "{k} should be a projection");
        }
        for k in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight", // Qwen3 q/k norm stays dense
        ] {
            assert!(!is_projection(k), "{k} should NOT be a projection");
        }
    }

    /// Dense write: every tensor reloads bit-identical and config carries no quantization block.
    #[test]
    fn dense_write_round_trips_bit_identical() {
        let dir = unique_dir("dense");
        let (tensors, config) = tiny_model();
        let original: HashMap<String, Vec<u8>> = tensors
            .iter()
            .map(|(k, a)| (k.clone(), bytes_of(a)))
            .collect();

        let report = write_snapshot(&dir, tensors, config, &SnapshotTokenizer::default(), None).unwrap();
        assert_eq!(report.quantized, None);

        let reloaded = Weights::from_dir(&dir).unwrap();
        assert_eq!(reloaded.len(), original.len(), "tensor count preserved");
        for (k, want) in &original {
            let got = bytes_of(reloaded.require(k).unwrap());
            assert_eq!(&got, want, "tensor {k} must reload bit-identical");
        }
        // No quantization block was added.
        assert_eq!(ModelConfig::from_dir(&dir).unwrap().quantization, None);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Quantized write: projections expand to weight/scales/biases, config gains the quantization
    /// block, and the snapshot loads through the loader's pre-quantized branch and runs a forward.
    #[test]
    fn quantized_write_loads_through_prequantized_branch() {
        let dir = unique_dir("q8");
        let (tensors, config) = tiny_model();
        let spec = QuantSpec::q8();

        let report = write_snapshot(&dir, tensors, config, &SnapshotTokenizer::default(), Some(spec)).unwrap();
        assert_eq!(report.quantized, Some(spec));

        let w = Weights::from_dir(&dir).unwrap();
        // A projection was stored as packed parts; a dense tensor was not.
        let base = "model.layers.0.self_attn.q_proj";
        assert!(w.contains(&format!("{base}.weight")));
        assert!(w.contains(&format!("{base}.scales")));
        assert!(w.contains(&format!("{base}.biases")));
        assert!(!w.contains("model.embed_tokens.scales"), "embeddings stay dense");

        let cfg = ModelConfig::from_dir(&dir).unwrap();
        assert_eq!(cfg.quantization, Some(spec), "config carries quantization block");

        // Loads through `from_weights` (no load-time quant) as a quantized model and runs.
        let model = CausalLm::from_weights(&w, "", cfg).unwrap();
        assert!(model.is_quantized(), "snapshot must load as quantized");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The HF leaf: a dense source dir written through with `None` reloads bit-identical, and the
    /// tokenizer files pass through verbatim.
    #[test]
    fn hf_dense_passthrough_round_trips() {
        let src = unique_dir("hf-src");
        let out = unique_dir("hf-out");
        std::fs::create_dir_all(&src).unwrap();

        let (tensors, config) = tiny_model();
        let original: HashMap<String, Vec<u8>> = tensors
            .iter()
            .map(|(k, a)| (k.clone(), bytes_of(a)))
            .collect();
        std::fs::write(src.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
        std::fs::write(src.join("tokenizer.json"), "{\"tok\":true}").unwrap();
        std::fs::write(src.join("tokenizer_config.json"), "{\"cfg\":true}").unwrap();
        let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, src.join("model.safetensors")).unwrap();

        let report = write_hf_snapshot(&src, &out, None).unwrap();
        assert_eq!(report.quantized, None);

        let reloaded = Weights::from_dir(&out).unwrap();
        for (k, want) in &original {
            assert_eq!(&bytes_of(reloaded.require(k).unwrap()), want, "tensor {k} bit-identical");
        }
        // Tokenizer files copied verbatim.
        assert_eq!(std::fs::read_to_string(out.join("tokenizer.json")).unwrap(), "{\"tok\":true}");
        assert_eq!(
            std::fs::read_to_string(out.join("tokenizer_config.json")).unwrap(),
            "{\"cfg\":true}"
        );
        assert_eq!(ModelConfig::from_dir(&out).unwrap().quantization, None);

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&out).ok();
    }

    /// The HF leaf with quantization: source dir → quantized snapshot loads as quantized and runs.
    #[test]
    fn hf_quantized_loads_as_quantized() {
        let src = unique_dir("hfq-src");
        let out = unique_dir("hfq-out");
        std::fs::create_dir_all(&src).unwrap();

        let (tensors, config) = tiny_model();
        std::fs::write(src.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
        let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, src.join("model.safetensors")).unwrap();

        let report = write_hf_snapshot(&src, &out, Some(QuantSpec::q4())).unwrap();
        assert_eq!(report.quantized, Some(QuantSpec::q4()));

        let cfg = ModelConfig::from_dir(&out).unwrap();
        assert_eq!(cfg.quantization, Some(QuantSpec::q4()));
        let model = CausalLm::from_weights(&Weights::from_dir(&out).unwrap(), "", cfg).unwrap();
        assert!(model.is_quantized());

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&out).ok();
    }

    /// Raw little-endian bytes of an array's values, for bit-identity checks (dtype preserved).
    fn bytes_of(a: &Array) -> Vec<u8> {
        // Compare in the stored dtype without converting: read the f32 view is lossy for bf16, so
        // round-trip through the array's own element bytes via safetensors-equivalent f32 cast only
        // when float — here all tiny-model tensors are f32, so a direct f32 slice is exact.
        a.as_slice::<f32>().iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}
