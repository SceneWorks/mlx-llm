//! Real-weights end-to-end test for Qwen3.6 **tool calling** (sc-7756) — the acceptance gate for the
//! tool-calling slice. Point `MLX_LLM_QWEN35_MODEL` at a Qwen3.6 snapshot (the 27B works):
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B \
//!   cargo test --test qwen35_tools -- --ignored --nocapture
//! ```
//!
//! This exercises the whole tool path on real weights: a request offering a `get_weather` function
//! renders the `<tools>` section (via the model's own chat template) → greedy decode → the model
//! emits a `<tool_call>` block in the Qwen3.6 `<function=…>/<parameter=…>` XML → the provider's
//! `ToolCallSegmenter` lifts it out of the answer text and parses it into a structured
//! `output.tool_calls`. A grounded call (the right function + a Paris-ish location) is the proof.

use core_llm::{
    LoadSpec, Message, Sampling, TextLlm, TextLlmRequest, ToolSpec,
};
use mlx_llm::LlamaProvider;

fn model_dir() -> String {
    std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL")
}

fn weather_tool() -> ToolSpec {
    ToolSpec::new(
        "get_weather",
        "Get the current weather for a given city.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city to get the weather for, e.g. 'Paris'."
                }
            },
            "required": ["location"]
        }),
    )
}

#[test]
#[ignore = "needs a Qwen3.6 snapshot via MLX_LLM_QWEN35_MODEL"]
fn qwen35_emits_and_parses_a_tool_call() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load qwen3.6");
    assert!(
        p.descriptor().capabilities.supports_tools,
        "a qwen3_5 checkpoint whose template renders tool calls must advertise supports_tools"
    );

    let req = TextLlmRequest {
        messages: vec![Message::user(
            "What is the weather in Paris right now? Use the available tool to find out.",
        )],
        sampling: Sampling::greedy(),
        max_new_tokens: 256,
        seed: Some(0),
        tools: vec![weather_tool()],
        ..Default::default()
    };

    let out = p.complete(&req).expect("generate");
    println!(
        "\n=== qwen3.6 TOOLS ===\n[text] {:?}\n[thinking] {:?}\n[tool_calls] {:?}\n",
        out.text, out.thinking, out.tool_calls
    );

    // The raw call markup must never leak into the answer text — it is structure, parsed out.
    assert!(
        !out.text.contains("<tool_call>") && !out.text.contains("<function="),
        "raw tool-call markup leaked into output.text: {:?}",
        out.text
    );

    // The model must have emitted a parseable call to the offered function, grounded on Paris.
    let call = out
        .tool_calls
        .iter()
        .find(|c| c.name == "get_weather")
        .unwrap_or_else(|| {
            panic!(
                "expected a get_weather tool call, got tool_calls={:?} text={:?}",
                out.tool_calls, out.text
            )
        });
    let location = call
        .arguments
        .get("location")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("get_weather call missing a string `location`: {:?}", call.arguments));
    assert!(
        location.to_lowercase().contains("paris"),
        "the parsed tool call must ground on Paris, got location={location:?}"
    );
}
