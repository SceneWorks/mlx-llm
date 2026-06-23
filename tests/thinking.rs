//! Real-weights thinking / no-think tests (sc-7585; `#[ignore]` — need models on disk).
//!
//! Qwen3 (`MLX_LLM_QWEN3_MODEL`) is a controllable-reasoning model: verifies `supports_thinking`,
//! the no-think request path, and reasoning/answer channel separation. SmolLM2
//! (`MLX_LLM_TEST_MODEL`) is a non-thinking model: verifies the capability is off and an explicit
//! enable-thinking request is rejected.
//!
//! ```text
//! MLX_LLM_QWEN3_MODEL=/tmp/qwen3-0.6b MLX_LLM_TEST_MODEL=/tmp/smollm2-135m \
//!   cargo test --test thinking -- --ignored --nocapture
//! ```

use core_llm::{
    Channel, LoadSpec, Message, Sampling, StreamEvent, TextLlm, TextLlmOutput, TextLlmRequest,
    ThinkingMode,
};
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

/// Run a request, returning the output plus the per-channel reconstructed text and the token indices.
fn run(p: &dyn TextLlm, r: &TextLlmRequest) -> (TextLlmOutput, String, String, Vec<usize>) {
    let (mut think, mut content, mut indices) = (String::new(), String::new(), Vec::new());
    let out = p
        .generate(r, &mut |ev| {
            if let StreamEvent::Token { text, index, channel, .. } = ev {
                indices.push(index);
                match channel {
                    Channel::Thinking => think.push_str(&text),
                    Channel::Content => content.push_str(&text),
                }
            }
        })
        .expect("generate");
    (out, think, content, indices)
}

#[test]
#[ignore = "needs Qwen3 via MLX_LLM_QWEN3_MODEL"]
fn qwen3_thinking_and_nothink() {
    let dir = std::env::var("MLX_LLM_QWEN3_MODEL").expect("set MLX_LLM_QWEN3_MODEL");
    let p = LlamaProvider::load(&LoadSpec::dense(dir)).expect("load qwen3");
    assert!(
        p.descriptor().capabilities.supports_thinking,
        "Qwen3's template gates enable_thinking → supports_thinking must be on"
    );

    // validate accepts every mode on a thinking model.
    for mode in [ThinkingMode::Auto, ThinkingMode::Enabled, ThinkingMode::Disabled] {
        p.validate(&req("hi", mode, 8))
            .unwrap_or_else(|e| panic!("validate {mode:?}: {e}"));
    }

    // Thinking (Auto = Qwen3's template default = on): the model emits a <think>…</think> block,
    // which is split into output.thinking; the answer excludes the reasoning and the markers.
    let (out, think, content, indices) =
        run(&p, &req("What is 2+2? Reply briefly.", ThinkingMode::Auto, 256));
    println!("\n=== Qwen3 THINK ===\n[reasoning]\n{think}\n[answer]\n{}\n", out.text);
    assert!(
        out.thinking.as_deref().is_some_and(|t| !t.trim().is_empty()),
        "thinking run must produce a reasoning block"
    );
    assert!(
        !out.text.contains("<think>") && !out.text.contains("</think>"),
        "markers must be stripped from the answer: {:?}",
        out.text
    );
    assert!(
        !think.contains("<think>") && !think.contains("</think>"),
        "markers must be stripped from reasoning"
    );
    assert_eq!(content, out.text, "Content-channel deltas reconstruct output.text");
    assert_eq!(
        think,
        out.thinking.clone().unwrap_or_default(),
        "Thinking-channel deltas reconstruct output.thinking"
    );
    assert!(!indices.is_empty(), "expected emitted tokens");
    assert!(
        indices.iter().enumerate().all(|(i, &x)| i == x),
        "token indices must be contiguous 0..n (marker tokens emit nothing): {indices:?}"
    );

    // No-think (Disabled): the empty <think></think> echo is injected into the prompt, so the model
    // answers directly — no reasoning channel at all.
    let (nout, nthink, ncontent, _) =
        run(&p, &req("What is 2+2? Reply briefly.", ThinkingMode::Disabled, 64));
    println!("=== Qwen3 NO-THINK ===\n[answer]\n{}\n", nout.text);
    assert!(nthink.is_empty(), "no-think must emit no Thinking-channel tokens");
    assert!(nout.thinking.is_none(), "no-think output.thinking must be None");
    assert!(!nout.text.contains("<think>"), "no marker leak in the no-think answer");
    assert!(!ncontent.trim().is_empty(), "no-think must produce a direct answer");
}

#[test]
#[ignore = "needs Qwen3 via MLX_LLM_QWEN3_MODEL"]
fn qwen3_passes_check_thinking_conformance() {
    let dir = std::env::var("MLX_LLM_QWEN3_MODEL").expect("set MLX_LLM_QWEN3_MODEL");
    let p = LlamaProvider::load(&LoadSpec::dense(dir)).expect("load qwen3");
    // The contract's thinking conformance check (sc-7584) against the real thinking provider:
    // every mode validates, the no-think path yields no reasoning, and the streamed reasoning
    // channel stays in sync with output.thinking.
    core_llm_testkit::check_thinking(&p, &core_llm_testkit::TextLlmProfile::cheap())
        .expect("Qwen3 must pass check_thinking");
}

#[test]
#[ignore = "needs Qwen3 via MLX_LLM_QWEN3_MODEL"]
fn qwen3_multi_turn_carries_reasoning() {
    // End-to-end multi-turn round-trip (sc-7586): generate a turn, feed the assistant turn back
    // carrying its reasoning via Message::with_thinking, then continue. The prior reasoning renders
    // into the prompt (Qwen3's template strips it for the now-earlier turn) without breaking the
    // conversation — the next turn still produces a coherent answer.
    let dir = std::env::var("MLX_LLM_QWEN3_MODEL").expect("set MLX_LLM_QWEN3_MODEL");
    let p = LlamaProvider::load(&LoadSpec::dense(dir)).expect("load qwen3");

    let q1 = "What is 2+2? Reply briefly.";
    let (out1, _t, _c, _) = run(&p, &req(q1, ThinkingMode::Auto, 256));
    assert!(
        out1.thinking.as_deref().is_some_and(|t| !t.trim().is_empty()),
        "turn 1 should reason"
    );

    let mut assistant = Message::assistant(out1.text.clone());
    if let Some(t) = &out1.thinking {
        assistant = assistant.with_thinking(t.clone());
    }
    let req2 = TextLlmRequest {
        messages: vec![
            Message::user(q1),
            assistant,
            Message::user("And 3+3? Reply briefly."),
        ],
        sampling: Sampling::greedy(),
        max_new_tokens: 256,
        seed: Some(0),
        thinking: ThinkingMode::Auto,
        ..Default::default()
    };
    let (out2, _t2, _c2, indices2) = run(&p, &req2);
    println!("\n=== Qwen3 turn 2 ===\n[answer]\n{}\n", out2.text);
    assert!(
        !out2.text.trim().is_empty() || out2.thinking.as_deref().is_some_and(|t| !t.trim().is_empty()),
        "turn 2 must produce an answer or reasoning"
    );
    assert!(
        indices2.iter().enumerate().all(|(i, &x)| i == x),
        "turn 2 token indices must be contiguous: {indices2:?}"
    );
}

#[test]
#[ignore = "needs a non-thinking model (e.g. SmolLM2) via MLX_LLM_TEST_MODEL"]
fn non_thinking_model_rejects_enable() {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let p = LlamaProvider::load(&LoadSpec::dense(dir)).expect("load model");
    assert!(
        !p.descriptor().capabilities.supports_thinking,
        "a model whose template has no enable_thinking must report supports_thinking=false"
    );

    // Explicit enable is rejected as Unsupported; Auto / no-think are accepted as no-ops.
    match p.validate(&req("hi", ThinkingMode::Enabled, 8)) {
        Err(core_llm::Error::Unsupported(_)) => {}
        other => panic!("expected Unsupported for enable-thinking, got {other:?}"),
    }
    p.validate(&req("hi", ThinkingMode::Auto, 8)).unwrap();
    p.validate(&req("hi", ThinkingMode::Disabled, 8)).unwrap();

    // It still generates normally, with no reasoning channel.
    let (out, think, _content, _) =
        run(&p, &req("The capital of France is", ThinkingMode::Auto, 8));
    assert!(think.is_empty() && out.thinking.is_none(), "non-thinking model emits no reasoning");
    assert!(!out.text.is_empty());
}
