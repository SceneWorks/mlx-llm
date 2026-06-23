//! Real-weights verification of the HF-safetensors → persisted (optionally quantized) snapshot
//! writer (`#[ignore]` — needs a model on disk), story 7660.
//!
//! Point `MLX_LLM_TEST_MODEL` at a Hugging Face snapshot directory (config.json + tokenizer.json +
//! *.safetensors) — the acceptance models are SmolLM2-135M and Qwen3-0.6B — and run:
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/SmolLM2-135M cargo test --test snapshot_hf -- --ignored --nocapture
//! ```
//!
//! Asserts (1) `write_hf_snapshot` produces dense / Q4 / Q8 snapshots that load through the provider
//! and generate coherent, reproducible text — the Q4/Q8 snapshots loading through the loader's
//! existing pre-quantized branch with no load-time requant; (2) a dense passthrough snapshot reloads
//! bit-identically to loading the source directly (same keys, dtypes, and values).

use core_llm::{LoadSpec, Message, Sampling, TextLlm, TextLlmRequest};

use mlx_llm::primitives::projection::QuantSpec;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::PROVIDER_ID;
use mlx_llm::{write_hf_snapshot, LlamaProvider};

use mlx_rs::Dtype;

const PROMPT: &str = "The capital of France is";

fn source_dir() -> Option<String> {
    match std::env::var("MLX_LLM_TEST_MODEL") {
        Ok(d) => Some(d),
        Err(_) => {
            eprintln!("skip: set MLX_LLM_TEST_MODEL to an HF snapshot directory");
            None
        }
    }
}

fn tmp_out(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("mlx-llm-hfsnap-{label}-{}", std::process::id()))
}

fn greedy_request() -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(PROMPT)],
        sampling: Sampling::greedy(),
        // Generous budget so a reasoning model (Qwen3) finishes its `<think>` block and emits a
        // visible answer; a non-reasoning model (SmolLM2) hits EOS well before this.
        max_new_tokens: 256,
        seed: Some(0),
        ..Default::default()
    }
}

/// dense / Q4 / Q8: `write_hf_snapshot` → load through the provider → coherent, reproducible text.
/// Quantized snapshots must load as quantized (the loader's pre-quantized branch, no load-time
/// requant), and the dense snapshot must generate exactly what loading the source directly does.
#[test]
#[ignore = "needs an HF snapshot via MLX_LLM_TEST_MODEL"]
fn hf_prepare_dense_q4_q8_loads_and_generates() {
    let Some(source) = source_dir() else { return };

    // Reference: the source loaded directly through the provider.
    let reference = LlamaProvider::load(&LoadSpec::dense(&source)).unwrap();
    assert!(!reference.is_quantized(), "source is dense");
    let ref_text = reference.complete(&greedy_request()).unwrap().text;
    println!("source ({PROMPT:?}) => {ref_text:?}");
    assert!(!ref_text.trim().is_empty(), "source generation must be non-empty");

    for (label, quantize) in [
        ("dense", None),
        ("q4", Some(QuantSpec::q4())),
        ("q8", Some(QuantSpec::q8())),
    ] {
        let out = tmp_out(label);
        let report = write_hf_snapshot(&source, &out, quantize).unwrap();
        assert_eq!(report.quantized, quantize, "{label}: report records the scheme");

        // Load via the provider (the snapshot IS the loader's input contract — no loader change).
        let provider = LlamaProvider::load(&LoadSpec::dense(out.to_str().unwrap())).unwrap();
        assert_eq!(
            provider.is_quantized(),
            quantize.is_some(),
            "{label}: quantized snapshot must load as quantized (and dense as dense)"
        );

        let a = provider.complete(&greedy_request()).unwrap();
        let b = provider.complete(&greedy_request()).unwrap();
        println!("{label:>5}: {:?}", a.text);
        assert!(!a.text.trim().is_empty(), "{label}: produced no text");
        assert!(a.usage.generated_tokens > 0, "{label}: no generated tokens");
        assert_eq!(a.text, b.text, "{label}: greedy generation must be reproducible");

        // The dense snapshot is a faithful passthrough — it must generate exactly what the source
        // does. (Q4/Q8 are lossy, so only coherence is asserted for them.)
        if quantize.is_none() {
            assert_eq!(a.text, ref_text, "dense snapshot must match the source's generation");
        }

        // Also confirm the registry route resolves to the same provider id.
        let routed = core_llm::load_textllm(PROVIDER_ID, &LoadSpec::dense(out.to_str().unwrap())).unwrap();
        assert_eq!(routed.descriptor().id, PROVIDER_ID, "{label}: registry routes to mlx provider");

        std::fs::remove_dir_all(&out).ok();
    }
}

/// A dense passthrough snapshot reloads bit-identically to loading the source directly: same key
/// set, same dtypes, same values (the f32 cast of bf16/f16 is exact, so equality is bitwise).
#[test]
#[ignore = "needs an HF snapshot via MLX_LLM_TEST_MODEL"]
fn hf_dense_passthrough_is_bit_identical() {
    let Some(source) = source_dir() else { return };
    let out = tmp_out("parity");
    write_hf_snapshot(&source, &out, None).unwrap();

    let src = Weights::from_dir(&source).unwrap();
    let snap = Weights::from_dir(&out).unwrap();
    assert_eq!(snap.len(), src.len(), "tensor count preserved");

    let mut keys: Vec<&str> = src.keys().collect();
    keys.sort_unstable();
    for key in keys {
        let s = src.require(key).unwrap();
        let d = snap.require(key).unwrap();
        assert_eq!(d.shape(), s.shape(), "{key}: shape preserved");
        assert_eq!(d.dtype(), s.dtype(), "{key}: dtype preserved (no cast on dense passthrough)");
        let sv = s.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let dv = d.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        assert_eq!(dv, sv, "{key}: values must reload bit-identical");
    }
    println!("dense passthrough bit-identical across {} tensors", src.len());

    std::fs::remove_dir_all(&out).ok();
}
