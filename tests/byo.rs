//! BYO architecture-dispatch + quantize-on-load tests (story 7163), gated on real snapshots.
//!
//! ```text
//! MLX_LLM_QWEN3_MODEL=/path/to/Qwen3-0.6B \
//! MLX_LLM_TEST_MODEL=/path/to/Llama-snapshot \
//!   cargo test --test byo -- --ignored --nocapture
//! ```

use core_llm::{
    LoadSpec, Message, Quantize, Sampling, StreamEvent, TextLlm, TextLlmOutput, TextLlmRequest,
};

use mlx_llm::provider::PROVIDER_ID;
use mlx_llm::LlamaProvider;

fn greedy_req(prompt: &str, n: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(prompt)],
        sampling: Sampling::greedy(),
        max_new_tokens: n,
        seed: Some(0),
        ..Default::default()
    }
}

fn run(provider: &LlamaProvider, prompt: &str, n: u32) -> (TextLlmOutput, usize) {
    let mut tokens = 0usize;
    let out = provider
        .generate(&greedy_req(prompt, n), &mut |ev| {
            if let StreamEvent::Token { .. } = ev {
                tokens += 1;
            }
        })
        .unwrap();
    (out, tokens)
}

#[test]
#[ignore = "needs a Qwen3 snapshot via MLX_LLM_QWEN3_MODEL"]
fn qwen3_dense_dispatch_and_stream() {
    let dir = std::env::var("MLX_LLM_QWEN3_MODEL").expect("set MLX_LLM_QWEN3_MODEL");
    let provider = LlamaProvider::load(&LoadSpec::dense(dir)).unwrap();

    // Architecture dispatch: family + capabilities reflect the loaded Qwen3 model.
    assert_eq!(provider.descriptor().id, PROVIDER_ID);
    assert_eq!(provider.descriptor().family, "qwen3");
    assert!(provider.descriptor().capabilities.max_context_tokens > 0); // context length reported
    assert!(!provider.is_quantized());

    // Qwen3 is a thinking model (sc-7585): in its default (Auto) mode a short budget is still inside
    // the <think> block, so the answer (out.text) may be empty while reasoning streams. Assert the
    // model streamed coherent text on *either* channel — the reasoning/answer split itself is
    // covered by tests/thinking.rs.
    let (out, n) = run(&provider, "The capital of France is", 16);
    println!("\n[qwen3 dense] answer={:?} thinking={:?}\n", out.text, out.thinking);
    let produced =
        !out.text.trim().is_empty() || out.thinking.as_deref().is_some_and(|t| !t.trim().is_empty());
    assert!(n > 0 && produced);
}

#[test]
#[ignore = "needs a Llama snapshot via MLX_LLM_TEST_MODEL"]
fn llama_quantize_on_load_q8() {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");

    let dense = LlamaProvider::load(&LoadSpec::dense(&dir)).unwrap();
    assert!(!dense.is_quantized());

    let q8 = LlamaProvider::load(&LoadSpec {
        source: dir.clone(),
        quantize: Some(Quantize::Q8),
    })
    .unwrap();
    assert!(q8.is_quantized());

    let (qout, qn) = run(&q8, "The capital of France is", 16);
    println!("\n[llama Q8] {}\n", qout.text);
    assert!(qn > 0 && !qout.text.trim().is_empty());
}

#[test]
#[ignore = "needs a Llama snapshot via MLX_LLM_TEST_MODEL"]
fn llama_quantize_on_load_q4() {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let q4 = LlamaProvider::load(&LoadSpec {
        source: dir,
        quantize: Some(Quantize::Q4),
    })
    .unwrap();
    assert!(q4.is_quantized());
    let (out, n) = run(&q4, "The capital of France is", 16);
    println!("\n[llama Q4] {}\n", out.text);
    assert!(n > 0 && !out.text.trim().is_empty());
}
