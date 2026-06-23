//! Real-weights end-to-end tests for Qwen3.6 (`qwen3_5` / `qwen3_5_moe`), the hybrid Gated-DeltaNet /
//! gated-full-attention decoder (stories sc-7627…sc-7630). The same tests cover both variants — point
//! `MLX_LLM_QWEN35_MODEL` at the 27B (dense) or the 35B-A3B (MoE) snapshot:
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B-or-35B-A3B \
//!   cargo test --test qwen35 -- --ignored --nocapture
//! ```
//!
//! These are the acceptance gate for sc-7629 (27B dense) and sc-7630 (35B-A3B MoE): dispatch (family
//! `qwen3_5`), coherent greedy text on real weights, and the thinking / no-think split driven by the
//! model's own chat template.

use core_llm::{
    Channel, LoadSpec, Message, Quantize, Sampling, StreamEvent, TextLlm, TextLlmOutput,
    TextLlmRequest, ThinkingMode,
};
use mlx_llm::provider::PROVIDER_ID;
use mlx_llm::LlamaProvider;

fn req(prompt: &str, mode: ThinkingMode, max_new_tokens: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(prompt)],
        sampling: Sampling::greedy(),
        max_new_tokens,
        seed: Some(0),
        thinking: mode,
        ..Default::default()
    }
}

/// Run a request, reconstructing the per-channel text from the streamed deltas.
fn run(p: &dyn TextLlm, r: &TextLlmRequest) -> (TextLlmOutput, String, String) {
    let (mut think, mut content) = (String::new(), String::new());
    let out = p
        .generate(r, &mut |ev| {
            if let StreamEvent::Token { text, channel, .. } = ev {
                match channel {
                    Channel::Thinking => think.push_str(&text),
                    Channel::Content => content.push_str(&text),
                }
            }
        })
        .expect("generate");
    (out, think, content)
}

fn model_dir() -> String {
    std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL")
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot (27B dense or 35B-A3B MoE) via MLX_LLM_QWEN35_MODEL"]
fn qwen35_dispatch_and_coherent_text() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load qwen3.6");

    // Architecture dispatch: the hybrid decoder loads and reports itself as `qwen3_5`.
    assert_eq!(p.descriptor().id, PROVIDER_ID);
    assert_eq!(p.descriptor().family, "qwen3_5", "must dispatch to the qwen3_5 hybrid decoder");
    assert!(p.descriptor().capabilities.max_context_tokens > 0);
    assert!(!p.is_quantized());

    // Coherence gate: greedy, no-think → a direct factual answer. A wrong architecture (split,
    // l2-norm, schedule, RoPE…) produces token soup, not "Paris".
    let (out, _think, content) = run(
        &p,
        &req("What is the capital of France? Answer with just the city name.", ThinkingMode::Disabled, 24),
    );
    println!("\n=== qwen3.6 NO-THINK ===\n[answer] {:?}\n", out.text);
    assert!(!content.trim().is_empty(), "must produce a direct answer");
    assert!(
        content.to_lowercase().contains("paris"),
        "greedy answer should be coherent and name Paris, got: {content:?}"
    );
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot (27B dense or 35B-A3B MoE) via MLX_LLM_QWEN35_MODEL"]
fn qwen35_thinking_and_nothink() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load qwen3.6");
    assert!(
        p.descriptor().capabilities.supports_thinking,
        "Qwen3.6's chat template gates enable_thinking → supports_thinking must be on"
    );
    for mode in [ThinkingMode::Auto, ThinkingMode::Enabled, ThinkingMode::Disabled] {
        p.validate(&req("hi", mode, 8)).unwrap_or_else(|e| panic!("validate {mode:?}: {e}"));
    }

    // Thinking: a <think>…</think> block is emitted and split into output.thinking; the answer
    // excludes the reasoning and the markers.
    let (out, think, content) = run(&p, &req("What is 2+2? Reply briefly.", ThinkingMode::Enabled, 512));
    println!("\n=== qwen3.6 THINK ===\n[reasoning]\n{think}\n[answer]\n{}\n", out.text);
    assert!(
        out.thinking.as_deref().is_some_and(|t| !t.trim().is_empty()),
        "thinking run must produce a reasoning block"
    );
    assert!(
        !out.text.contains("<think>") && !out.text.contains("</think>"),
        "markers must be stripped from the answer: {:?}",
        out.text
    );
    assert_eq!(content, out.text, "content-channel deltas reconstruct output.text");
    assert_eq!(think, out.thinking.clone().unwrap_or_default());

    // No-think: the empty <think></think> echo is injected, so the model answers directly.
    let (nout, nthink, ncontent) = run(&p, &req("What is 2+2? Reply briefly.", ThinkingMode::Disabled, 64));
    println!("=== qwen3.6 NO-THINK ===\n[answer]\n{}\n", nout.text);
    assert!(nthink.is_empty(), "no-think must emit no Thinking-channel tokens");
    assert!(nout.thinking.is_none(), "no-think output.thinking must be None");
    assert!(!ncontent.trim().is_empty(), "no-think must produce a direct answer");
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot (27B dense or 35B-A3B MoE) via MLX_LLM_QWEN35_MODEL"]
fn qwen35_quantize_on_load_q8() {
    let dir = model_dir();
    let q8 = LlamaProvider::load(&LoadSpec { source: dir, quantize: Some(Quantize::Q8) })
        .expect("load q8");
    assert!(q8.is_quantized(), "Q8 load must report quantized");
    let (_out, _think, content) =
        run(&q8, &req("Name a primary color. One word.", ThinkingMode::Disabled, 16));
    println!("\n=== qwen3.6 Q8 ===\n[answer] {content:?}\n");
    assert!(!content.trim().is_empty(), "quantized model must still generate text");
}
