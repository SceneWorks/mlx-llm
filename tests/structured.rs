//! JSON-constrained decode (story 7166), gated on a real snapshot.
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/snapshot cargo test --test structured -- --ignored --nocapture
//! ```

use core_llm::{Constraint, JsonState, LoadSpec, Message, Sampling, TextLlm, TextLlmRequest};

use mlx_llm::LlamaProvider;

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn json_constrained_output_is_valid_json() {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let provider = LlamaProvider::load(&LoadSpec::dense(dir)).unwrap();

    // The provider advertises the JSON constraint.
    assert!(provider
        .descriptor()
        .capabilities
        .supports_constraint(Constraint::Json));

    let req = TextLlmRequest {
        messages: vec![Message::user(
            "Give a JSON object describing a person with a name and an age.",
        )],
        sampling: Sampling::greedy(),
        max_new_tokens: 80,
        seed: Some(0),
        constraint: Some(Constraint::Json),
        ..Default::default()
    };
    let out = provider.complete(&req).unwrap();
    println!("\n[json-constrained] {:?}\n  finish={:?}", out.text, out.finish_reason);

    // Core guarantee: every emitted token kept the output a valid JSON prefix.
    assert!(
        JsonState::start().advance(out.text.trim()).is_some(),
        "constrained output is not a valid JSON prefix: {:?}",
        out.text
    );
    // If generation stopped on its own, the value is complete and parses with serde.
    if out.finish_reason == Some(core_llm::FinishReason::Stop) {
        assert!(
            serde_json::from_str::<serde_json::Value>(out.text.trim()).is_ok(),
            "stopped output should be complete valid JSON: {:?}",
            out.text
        );
    }
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn unconstrained_streams_arbitrary_text() {
    // Control: without the constraint the same prompt is free to produce non-JSON.
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let provider = LlamaProvider::load(&LoadSpec::dense(dir)).unwrap();
    let req = TextLlmRequest {
        messages: vec![Message::user("Say hello.")],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        ..Default::default()
    };
    let out = provider.complete(&req).unwrap();
    assert!(!out.text.trim().is_empty());
}
