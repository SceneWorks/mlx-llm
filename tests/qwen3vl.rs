//! Real-weights acceptance for the Qwen3-VL (`qwen3_vl`) **text decoder** (sc-8075).
//!
//! Loads the standard full-attention Qwen3 text decoder from a real Qwen3-VL-8B-Instruct snapshot
//! (the VLM-wrapped `model.language_model.*` weights, 256K-context interleaved-M-RoPE config) and
//! asserts that text-only greedy generation is coherent and reproducible. The snapshot is resolved
//! from `QWEN3VL_SNAPSHOT` or the default HF cache; the tests **self-skip cleanly** when it is
//! absent, but run fully when present (it is present in this env).
//!
//! ```text
//! QWEN3VL_SNAPSHOT=/path/to/Qwen3-VL-8B-Instruct cargo test --test qwen3vl -- --nocapture
//! ```

use std::path::PathBuf;

use core_llm::{
    load_textllm, Channel, LoadSpec, Message, Sampling, StreamEvent as CoreEvent, TextLlmRequest,
    Tokenizer,
};

use mlx_llm::config::{Architecture, ModelConfig};
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig};
use mlx_llm::decode::StreamEvent;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::{eos_token_ids, PROVIDER_ID};

/// Resolve the cached Qwen3-VL-8B-Instruct snapshot (rev 0c351dd0). `QWEN3VL_SNAPSHOT` overrides.
fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QWEN3VL_SNAPSHOT") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    );
    path.exists().then_some(path)
}

/// Direct `CausalLm` load + greedy text-only generation from the real snapshot: the decoder must
/// load (36 layers, GQA 32/8, head_dim 128, vocab 151936, 256K RoPE), and a factual prompt must
/// generate coherent, reproducible text. "The capital of France is" → must mention Paris — a tight
/// end-to-end grounding check (if the weight prefix, the norm convention (standard RMSNorm), or the
/// RoPE were wrong, the model could not answer).
#[test]
fn qwen3vl_text_decoder_loads_and_generates_coherently() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };

    let cfg = ModelConfig::from_dir(&dir).expect("parse Qwen3-VL config");
    assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
    assert_eq!(cfg.num_layers, 36);
    assert_eq!(cfg.num_heads, 32);
    assert_eq!(cfg.num_kv_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.rotary_dim(), 128);
    assert_eq!(cfg.mrope_section_resolved().iter().sum::<usize>(), 64);

    let weights = Weights::from_dir(&dir).expect("load shards");
    // Prefix is ignored on the Qwen3-VL path (the decoder roots at `model.language_model`).
    let model = CausalLm::from_weights(&weights, "", cfg).expect("load Qwen3-VL text decoder");

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    // Render via the model's own chat template so the prompt is in-distribution.
    let prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids: Vec<i32> =
        tok.encode(prompt, false).unwrap().into_iter().map(|id| id as i32).collect();

    let config = GenerationConfig {
        max_new_tokens: 16,
        sampling: SamplingParams::default(), // greedy ⇒ reproducible
        seed: Some(0),
        stop_tokens: eos_token_ids(&dir),
    };

    let run = || {
        generate(&model, &prompt_ids, &config, &CancelFlag::new(), &mut |_| {})
            .unwrap()
            .tokens
    };
    let a = run();
    let b = run();
    assert!(!a.is_empty(), "expected non-empty generation");
    assert_eq!(a, b, "greedy generation must be reproducible");

    let text = tok.decode(&a.iter().map(|&i| i as u32).collect::<Vec<_>>(), true).unwrap();
    println!("\n=== Qwen3-VL text-only generation ===\n{text}\n=====================================");
    assert!(!text.trim().is_empty());
    assert!(
        text.to_lowercase().contains("paris"),
        "coherent answer must name Paris, got: {text:?}"
    );
}

/// The full `core-llm` contract path: load via the registry by id (`mlx-llama`), confirm the loaded
/// descriptor reports the Qwen3-VL family + 256K context, and stream a coherent reply.
#[test]
fn qwen3vl_round_trips_through_core_llm_contract() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    assert_eq!(provider.descriptor().id, PROVIDER_ID);
    assert_eq!(provider.descriptor().family, "qwen3_vl");
    assert_eq!(provider.descriptor().capabilities.max_context_tokens, 262144);

    let req = TextLlmRequest {
        messages: vec![Message::user("What is the capital of France? Answer in one word.")],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        ..Default::default()
    };

    let mut streamed = String::new();
    let out = provider
        .generate(&req, &mut |ev| {
            if let CoreEvent::Token { text, channel, .. } = ev {
                if channel == Channel::Content {
                    streamed.push_str(&text);
                }
            }
        })
        .unwrap();
    println!("\n=== contract output ===\n{}\n=======================", out.text);
    assert!(out.usage.generated_tokens > 0);
    assert!(out.usage.prompt_tokens > 0);
    assert!(
        out.text.to_lowercase().contains("paris"),
        "coherent contract answer must name Paris, got: {:?}",
        out.text
    );
}

/// A mid-stream cancel on the real decoder stops promptly with a partial result.
#[test]
fn qwen3vl_mid_stream_cancel() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let weights = Weights::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&weights, "", cfg).unwrap();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prompt = "<|im_start|>user\nWrite a long story about a robot.<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids: Vec<i32> =
        tok.encode(prompt, false).unwrap().into_iter().map(|id| id as i32).collect();

    let cancel = CancelFlag::new();
    let mut count = 0;
    let out = generate(
        &model,
        &prompt_ids,
        &GenerationConfig {
            max_new_tokens: 200,
            sampling: SamplingParams::default(),
            seed: Some(0),
            stop_tokens: eos_token_ids(&dir),
        },
        &cancel,
        &mut |e| {
            if let StreamEvent::Token { .. } = e {
                count += 1;
                if count == 5 {
                    cancel.cancel();
                }
            }
        },
    )
    .unwrap();
    assert_eq!(out.finish_reason, mlx_llm::decode::FinishReason::Cancelled);
    assert!(out.tokens.len() <= 6, "cancel should stop promptly");
}
