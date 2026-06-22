//! Real-weights end-to-end streaming test (`#[ignore]` — needs a model on disk).
//!
//! Point `MLX_LLM_TEST_MODEL` at a Hugging Face Llama snapshot directory (config.json +
//! tokenizer.json + *.safetensors) and run:
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/Llama-3.2-1B-Instruct cargo test --test real_weights -- --ignored --nocapture
//! ```
//!
//! Asserts the engine loads the snapshot and streams non-empty text, that greedy decoding is
//! reproducible, and that a mid-stream cancel stops promptly with a partial result.

use core_llm::{
    load_textllm, LoadSpec, Message, Sampling, StreamEvent as CoreEvent, TextLlmRequest, Tokenizer,
};

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, CancelFlag, FinishReason, GenerationConfig, StreamEvent};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::PROVIDER_ID;

fn model_dir() -> Option<String> {
    std::env::var("MLX_LLM_TEST_MODEL").ok()
}

#[test]
#[ignore = "needs a real Llama snapshot via MLX_LLM_TEST_MODEL"]
fn streams_text_from_snapshot() {
    let dir = model_dir().expect("set MLX_LLM_TEST_MODEL");
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let weights = Weights::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&weights, "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();

    let prompt_ids: Vec<i32> = tok
        .encode("The capital of France is", true)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect();

    let config = GenerationConfig {
        max_new_tokens: 24,
        sampling: SamplingParams::default(), // greedy => reproducible
        seed: Some(0),
        stop_tokens: vec![128001, 128008, 128009],
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

    let text = tok
        .decode(&a.iter().map(|&i| i as u32).collect::<Vec<_>>(), true)
        .unwrap();
    println!("\n=== generated ===\n{text}\n=================");
    assert!(!text.trim().is_empty());
}

#[test]
#[ignore = "needs a real Llama snapshot via MLX_LLM_TEST_MODEL"]
fn mid_stream_cancel_on_real_model() {
    let dir = model_dir().expect("set MLX_LLM_TEST_MODEL");
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let weights = Weights::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&weights, "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let prompt_ids: Vec<i32> = tok
        .encode("Write a long story about a robot:", true)
        .unwrap()
        .into_iter()
        .map(|id| id as i32)
        .collect();

    let cancel = CancelFlag::new();
    let mut count = 0;
    let out = generate(
        &model,
        &prompt_ids,
        &GenerationConfig {
            max_new_tokens: 200,
            sampling: SamplingParams::default(),
            seed: Some(0),
            stop_tokens: vec![128001, 128008, 128009],
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

    assert_eq!(out.finish_reason, FinishReason::Cancelled);
    assert!(out.tokens.len() <= 6, "cancel should stop promptly");
}

#[test]
#[ignore = "needs a real Llama snapshot via MLX_LLM_TEST_MODEL"]
fn round_trips_a_generation_through_the_core_llm_contract() {
    // The full contract path: load via the registry by id, then stream through `core_llm::TextLlm`.
    let dir = model_dir().expect("set MLX_LLM_TEST_MODEL");
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(&dir)).unwrap();
    assert_eq!(provider.descriptor().id, PROVIDER_ID);

    let req = TextLlmRequest {
        messages: vec![Message::user("Name three primary colors.")],
        sampling: Sampling::greedy(),
        max_new_tokens: 24,
        seed: Some(0),
        ..Default::default()
    };

    let mut streamed = String::new();
    let mut token_events = 0usize;
    let out = provider
        .generate(&req, &mut |ev| {
            if let CoreEvent::Token { text, .. } = ev {
                token_events += 1;
                streamed.push_str(&text);
            }
        })
        .unwrap();

    println!("\n=== contract output ===\n{}\n=======================", out.text);
    assert!(token_events > 0, "expected streamed tokens");
    assert!(!out.text.trim().is_empty(), "expected non-empty output");
    assert!(out.usage.generated_tokens > 0);
    assert!(out.usage.prompt_tokens > 0);
    assert_eq!(streamed, out.text, "streamed deltas must reconstruct the output");
}
