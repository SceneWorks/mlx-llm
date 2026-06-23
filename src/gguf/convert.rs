//! GGUF → MLX snapshot conversion: remap keys, reconstruct config, write `{config.json,
//! model.safetensors}`.
//!
//! The engine loads a Hugging Face-shaped snapshot ([`crate::models::CausalLm::from_weights`]):
//! transformer-named weights in safetensors plus a `config.json`. A GGUF instead uses llama.cpp's
//! `blk.{i}.attn_q.weight` naming, packs hyperparameters in its metadata table, and stores weights
//! in GGML quant blocks. This module bridges the two: each tensor is dequantized to dense
//! ([`super::dequant`]), its key remapped to the transformer layout, the config rebuilt from GGUF
//! metadata, and the result written as a snapshot the engine loads unchanged.
//!
//! Two output modes:
//! - **dense** (`quantize: None`) — every weight stored bf16; loads at parity with the source.
//! - **requantized** (`quantize: Some`) — attention/MLP projections re-quantized to MLX group-wise
//!   Q4/Q8 and stored as packed `weight`/`scales`/`biases` with a `quantization` block in
//!   `config.json`; embeddings, the LM head, and norms stay dense (the engine's quant invariant).
//!
//! The tokenizer is reconstructed too (stories 7251/7334): the GGUF's `tokenizer.ggml.*` metadata is
//! encoded into a `tokenizer.json` + `tokenizer_config.json` ([`super::tokenizer`]) for both the
//! byte-level BPE family (SmolLM2/Qwen/Llama-3) and the SentencePiece BPE family (Llama-2/Mistral),
//! and the special-token ids are stamped into `config.json` so the snapshot runs end-to-end with no
//! external files. A tokenizer kind we can't faithfully rebuild from GGUF (Unigram/T5-style
//! SentencePiece, whose normalizer charsmap isn't stored) is reported in [`ConvertReport::tokenizer`]
//! rather than guessed at; pass the source repo's `tokenizer.json` via the `convert_gguf` example's
//! `--tokenizer` in that case (which also overrides a reconstructed one).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype};
use serde_json::{json, Map, Value};

use crate::error::{Error, Result};
use crate::gguf::reader::GgufFile;
use crate::gguf::tokenizer::{self, TokenizerOutcome};
use crate::primitives::QuantSpec;
use crate::snapshot::{write_snapshot, SnapshotTokenizer};

/// Compute/storage dtype for dense tensors — bf16, the engine's load dtype, so a dense conversion
/// reloads with no extra rounding.
const STORE_DTYPE: Dtype = Dtype::Bfloat16;

/// Options controlling the conversion.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConvertOptions {
    /// `Some` ⇒ re-quantize the attention/MLP projections to this MLX group-wise scheme; `None` ⇒
    /// dense bf16 snapshot.
    pub quantize: Option<QuantSpec>,
}

/// What a conversion produced.
#[derive(Clone, Debug)]
pub struct ConvertReport {
    /// GGUF `general.architecture` (e.g. `"llama"`, `"qwen3"`).
    pub architecture: String,
    /// HF `model_type` written to `config.json`.
    pub model_type: String,
    /// Number of weight tensors written.
    pub num_tensors: usize,
    /// The requant scheme applied to projections, if any.
    pub quantized: Option<QuantSpec>,
    /// Whether a `tokenizer.json` was reconstructed from the GGUF metadata.
    pub tokenizer: TokenizerStatus,
    /// Directory the snapshot was written to.
    pub out_dir: PathBuf,
}

/// Outcome of reconstructing a `tokenizer.json` from the GGUF metadata.
#[derive(Clone, Debug)]
pub enum TokenizerStatus {
    /// `tokenizer.json` + `tokenizer_config.json` were written (with a short description of the kind).
    Reconstructed(String),
    /// The GGUF carries tokenizer metadata we can't faithfully rebuild (the reason); pass
    /// `--tokenizer`.
    Unsupported(String),
    /// The GGUF carries no tokenizer metadata.
    Absent,
}

/// Convert a GGUF file on disk into an MLX snapshot directory.
pub fn convert_file(
    gguf_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    opts: ConvertOptions,
) -> Result<ConvertReport> {
    let g = GgufFile::open(gguf_path)?;
    convert(&g, out_dir, opts)
}

/// Convert an already-parsed GGUF file into an MLX snapshot directory.
pub fn convert(g: &GgufFile, out_dir: impl AsRef<Path>, opts: ConvertOptions) -> Result<ConvertReport> {
    let out_dir = out_dir.as_ref();

    let arch = g
        .meta_str("general.architecture")
        .ok_or_else(|| Error::Config("gguf: missing general.architecture".into()))?
        .to_string();
    let (model_type, hf_arch) = match arch.as_str() {
        "llama" => ("llama", "LlamaForCausalLM"),
        "qwen3" => ("qwen3", "Qwen3ForCausalLM"),
        other => {
            return Err(Error::Unsupported(format!(
                "GGUF architecture {other:?} (engine supports llama/mistral and qwen3; \
                 mistral GGUFs are labelled \"llama\")"
            )))
        }
    };

    // Head counts + whether the q/k projections need un-permuting (see `permute_inverse_qk`).
    let mkey = |s: &str| format!("{arch}.{s}");
    let num_heads = g
        .meta_u64(&mkey("attention.head_count"))
        .ok_or_else(|| Error::Config(format!("gguf: missing metadata {}", mkey("attention.head_count"))))?
        as usize;
    let num_kv_heads = g
        .meta_u64(&mkey("attention.head_count_kv"))
        .unwrap_or(num_heads as u64) as usize;
    // llama.cpp interleaves q/k rows for the Llama/Mistral RoPE (`rope_type=NORM`); Qwen3 keeps the
    // HF half-split layout (`rope_type=NEOX`) and so is not permuted.
    let permute_qk = arch == "llama";

    // --- remap + dequantize every tensor to dense f32 (with its torch-order shape) ---
    let mut dense: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut unmapped: Vec<String> = Vec::new();
    for info in &g.tensors {
        let Some(hf_key) = remap_key(&info.name) else {
            if !is_ignorable(&info.name) {
                unmapped.push(info.name.clone());
            }
            continue;
        };
        let raw = g.tensor_data(info)?;
        let mut data = super::dequant::dequantize(info.ggml_type, raw, info.num_elements())?;
        if permute_qk && hf_key.ends_with("self_attn.q_proj.weight") {
            data = permute_inverse_qk(&data, &info.shape, num_heads)?;
        } else if permute_qk && hf_key.ends_with("self_attn.k_proj.weight") {
            data = permute_inverse_qk(&data, &info.shape, num_kv_heads)?;
        }
        dense.insert(hf_key, (data, info.shape.clone()));
    }
    if !unmapped.is_empty() {
        return Err(Error::Unsupported(format!(
            "gguf: {} tensor(s) with no transformer-key mapping (would be silently dropped): {}",
            unmapped.len(),
            unmapped.join(", ")
        )));
    }

    // --- reconstruct config.json from GGUF metadata (the writer adds any `quantization` block) ---
    let config = reconstruct_config(g, &arch, model_type, hf_arch, &dense)?;

    // --- dense tensor set (bf16), remapped to the transformer key layout; the shared writer does
    // any projection requant ---
    let mut tensors: Vec<(String, Array)> = Vec::with_capacity(dense.len());
    for (key, (data, shape)) in &dense {
        let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let arr = Array::from_slice(data, &shape_i32).as_dtype(STORE_DTYPE)?;
        tensors.push((key.clone(), arr));
    }

    // --- reconstruct tokenizer.json / tokenizer_config.json from the GGUF tokenizer metadata ---
    let (snapshot_tokenizer, tokenizer) = match tokenizer::reconstruct(g)? {
        TokenizerOutcome::Reconstructed(t) => (
            SnapshotTokenizer {
                tokenizer_json: Some(to_pretty(&t.tokenizer_json)?),
                tokenizer_config_json: Some(to_pretty(&t.tokenizer_config_json)?),
            },
            TokenizerStatus::Reconstructed(t.kind),
        ),
        TokenizerOutcome::Unsupported(reason) => {
            (SnapshotTokenizer::default(), TokenizerStatus::Unsupported(reason))
        }
        TokenizerOutcome::Absent => (SnapshotTokenizer::default(), TokenizerStatus::Absent),
    };

    // --- write the snapshot through the shared writer (requant + config + safetensors + tokenizer) ---
    let report = write_snapshot(out_dir, tensors, config, &snapshot_tokenizer, opts.quantize)?;

    Ok(ConvertReport {
        architecture: arch,
        model_type: model_type.to_string(),
        num_tensors: report.num_tensors,
        quantized: report.quantized,
        tokenizer,
        out_dir: report.out_dir,
    })
}

/// Serialize a reconstructed tokenizer JSON value to a pretty string for the snapshot writer.
fn to_pretty(value: &Value) -> Result<String> {
    serde_json::to_string_pretty(value)
        .map_err(|e| Error::Msg(format!("gguf: serialize tokenizer json: {e}")))
}

/// Map a GGML tensor name to the transformer (HF) key the engine loads, or `None` if it is not a
/// weight the engine consumes.
pub fn remap_key(name: &str) -> Option<String> {
    // Non-layer tensors.
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    // Per-layer: blk.{i}.{ggml_suffix} -> model.layers.{i}.{hf_suffix}
    let rest = name.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    idx.parse::<usize>().ok()?;
    let hf_suffix = match suffix {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight", // Qwen3
        "attn_k_norm.weight" => "self_attn.k_norm.weight", // Qwen3
        _ => return None,
    };
    Some(format!("model.layers.{idx}.{hf_suffix}"))
}

/// Undo llama.cpp's q/k row permutation, mapping a GGUF projection weight back to the HF layout.
///
/// `convert_hf_to_gguf.py` reorders each head's `head_dim` rows
/// (`reshape(n_head, 2, hd/2).swapaxes(1, 2)`) so llama.cpp's interleaved RoPE matches HF's
/// half-split RoPE. The inverse gather here restores the HF order: for HF row
/// `r = k·(hd/2) + j` in a head, the source GGUF row is `2·j + k`. Applied per head to a
/// `[out, in]` weight where `out = n_head · head_dim`.
fn permute_inverse_qk(data: &[f32], shape: &[usize], n_head: usize) -> Result<Vec<f32>> {
    let out = shape[0];
    let in_dim = if shape.len() == 2 { shape[1] } else { data.len() / out };
    if n_head == 0 || !out.is_multiple_of(n_head) {
        return Err(Error::Msg(format!(
            "gguf: q/k permute: out {out} not divisible by n_head {n_head}"
        )));
    }
    let head_dim = out / n_head;
    if !head_dim.is_multiple_of(2) {
        return Err(Error::Msg(format!("gguf: q/k permute: odd head_dim {head_dim}")));
    }
    let half = head_dim / 2;
    let mut res = vec![0f32; data.len()];
    for h in 0..n_head {
        for r in 0..head_dim {
            let (k, j) = (r / half, r % half);
            let src = h * head_dim + (2 * j + k);
            let dst = h * head_dim + r;
            res[dst * in_dim..dst * in_dim + in_dim]
                .copy_from_slice(&data[src * in_dim..src * in_dim + in_dim]);
        }
    }
    Ok(res)
}

/// GGML tensors the engine recomputes itself and so can ignore (rather than reject as unmapped).
fn is_ignorable(name: &str) -> bool {
    // Llama-3 ships precomputed RoPE frequencies; the engine derives RoPE from theta/scaling.
    name == "rope_freqs.weight" || name.ends_with(".rope_freqs.weight")
}

/// Rebuild a HF-style `config.json` value from the GGUF metadata table.
fn reconstruct_config(
    g: &GgufFile,
    arch: &str,
    model_type: &str,
    hf_arch: &str,
    dense: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<Value> {
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let req_u64 = |suffix: &str| -> Result<u64> {
        g.meta_u64(&key(suffix))
            .ok_or_else(|| Error::Config(format!("gguf: missing metadata {}", key(suffix))))
    };

    let hidden = req_u64("embedding_length")? as i64;
    let blocks = req_u64("block_count")? as i64;
    let heads = req_u64("attention.head_count")? as i64;
    let kv_heads = g.meta_u64(&key("attention.head_count_kv")).unwrap_or(heads as u64) as i64;
    let ffn = req_u64("feed_forward_length")? as i64;
    let head_dim = g
        .meta_u64(&key("attention.key_length"))
        .map(|v| v as i64)
        .unwrap_or(hidden / heads);
    let rms_eps = g
        .meta_f64(&key("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    let rope_theta = g.meta_f64(&key("rope.freq_base")).unwrap_or(10000.0);
    let context = g.meta_u64(&key("context_length")).unwrap_or(0) as i64;

    // vocab from the embedding rows (torch [vocab, hidden]) — the most reliable source.
    let vocab = dense
        .get("model.embed_tokens.weight")
        .map(|(_, shape)| shape[0] as i64)
        .ok_or_else(|| Error::Config("gguf: no token embedding tensor".into()))?;

    // lm_head tied iff the GGUF has no separate output projection.
    let tied = !dense.contains_key("lm_head.weight");

    let mut cfg = Map::new();
    cfg.insert("architectures".into(), json!([hf_arch]));
    cfg.insert("model_type".into(), json!(model_type));
    cfg.insert("hidden_size".into(), json!(hidden));
    cfg.insert("intermediate_size".into(), json!(ffn));
    cfg.insert("num_hidden_layers".into(), json!(blocks));
    cfg.insert("num_attention_heads".into(), json!(heads));
    cfg.insert("num_key_value_heads".into(), json!(kv_heads));
    cfg.insert("head_dim".into(), json!(head_dim));
    cfg.insert("vocab_size".into(), json!(vocab));
    cfg.insert("rms_norm_eps".into(), json!(rms_eps));
    cfg.insert("rope_theta".into(), json!(rope_theta));
    cfg.insert("tie_word_embeddings".into(), json!(tied));
    if context > 0 {
        cfg.insert("max_position_embeddings".into(), json!(context));
    }
    // Special-token ids so the engine resolves stop tokens (`provider::eos_token_ids` reads
    // `config.json`) and BOS without an external file.
    if let Some(eos) = g.meta_u64("tokenizer.ggml.eos_token_id") {
        cfg.insert("eos_token_id".into(), json!(eos as i64));
    }
    if let Some(bos) = g.meta_u64("tokenizer.ggml.bos_token_id") {
        cfg.insert("bos_token_id".into(), json!(bos as i64));
    }
    if let Some(scaling) = reconstruct_rope_scaling(g, arch) {
        cfg.insert("rope_scaling".into(), scaling);
    }
    Ok(Value::Object(cfg))
}

/// Best-effort llama3 RoPE-scaling reconstruction. The verification models (SmolLM2, Qwen3) carry no
/// scaling, so this is exercised only by llama3-scaled GGUFs; absent keys ⇒ standard RoPE.
fn reconstruct_rope_scaling(g: &GgufFile, arch: &str) -> Option<Value> {
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let low = g.meta_f64(&key("rope.scaling.low_freq_factor"));
    let high = g.meta_f64(&key("rope.scaling.high_freq_factor"));
    let scaling_type = g.meta_str(&key("rope.scaling.type"));
    // Only the llama3 NTK-by-parts schedule is modelled by the engine.
    if scaling_type != Some("llama3") && low.is_none() && high.is_none() {
        return None;
    }
    let factor = g.meta_f64(&key("rope.scaling.factor")).unwrap_or(8.0);
    let orig = g
        .meta_u64(&key("rope.scaling.original_context_length"))
        .unwrap_or(8192) as f64;
    Some(json!({
        "rope_type": "llama3",
        "factor": factor,
        "low_freq_factor": low.unwrap_or(1.0),
        "high_freq_factor": high.unwrap_or(4.0),
        "original_max_position_embeddings": orig,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_non_layer_keys() {
        assert_eq!(remap_key("token_embd.weight").unwrap(), "model.embed_tokens.weight");
        assert_eq!(remap_key("output_norm.weight").unwrap(), "model.norm.weight");
        assert_eq!(remap_key("output.weight").unwrap(), "lm_head.weight");
    }

    #[test]
    fn remaps_layer_keys() {
        assert_eq!(
            remap_key("blk.0.attn_q.weight").unwrap(),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            remap_key("blk.13.ffn_down.weight").unwrap(),
            "model.layers.13.mlp.down_proj.weight"
        );
        assert_eq!(
            remap_key("blk.5.attn_norm.weight").unwrap(),
            "model.layers.5.input_layernorm.weight"
        );
        assert_eq!(
            remap_key("blk.5.ffn_norm.weight").unwrap(),
            "model.layers.5.post_attention_layernorm.weight"
        );
        assert_eq!(
            remap_key("blk.2.attn_q_norm.weight").unwrap(),
            "model.layers.2.self_attn.q_norm.weight"
        );
    }

    #[test]
    fn qk_permute_inverts_llama_cpp_forward() {
        // Forward (HF -> GGUF) permute per head: reshape(n_head, 2, hd/2).swapaxes(1,2).
        // For one head, hd=4: GGUF_row(2j+k) = HF_row(k*2+j).
        fn forward(data: &[f32], shape: &[usize], n_head: usize) -> Vec<f32> {
            let out = shape[0];
            let in_dim = shape[1];
            let head_dim = out / n_head;
            let half = head_dim / 2;
            let mut res = vec![0f32; data.len()];
            for h in 0..n_head {
                for k in 0..2 {
                    for j in 0..half {
                        let dst = h * head_dim + (2 * j + k);
                        let src = h * head_dim + (k * half + j);
                        res[dst * in_dim..dst * in_dim + in_dim]
                            .copy_from_slice(&data[src * in_dim..src * in_dim + in_dim]);
                    }
                }
            }
            res
        }
        // 2 heads, head_dim 4, in 3 — distinct per-row values so the gather is checkable.
        let (n_head, head_dim, in_dim) = (2usize, 4usize, 3usize);
        let out = n_head * head_dim;
        let hf: Vec<f32> = (0..(out * in_dim) as i32).map(|x| x as f32).collect();
        let shape = vec![out, in_dim];
        let gguf = forward(&hf, &shape, n_head);
        let back = permute_inverse_qk(&gguf, &shape, n_head).unwrap();
        assert_eq!(back, hf, "inverse permute must recover the HF layout");
        // And it is a non-trivial permutation (not identity).
        assert_ne!(gguf, hf);
    }

    #[test]
    fn unknown_keys_and_ignorables() {
        assert!(remap_key("blk.0.some_future_tensor.weight").is_none());
        assert!(remap_key("rope_freqs.weight").is_none());
        assert!(is_ignorable("rope_freqs.weight"));
        assert!(is_ignorable("blk.0.rope_freqs.weight"));
        assert!(!is_ignorable("blk.0.attn_q.weight"));
    }
}
