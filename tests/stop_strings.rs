//! Real-weights test for request `stop` strings (story 7349, `#[ignore]` — needs a model on disk).
//!
//! Point `MLX_LLM_TEST_MODEL` at a Hugging Face Llama/Qwen snapshot directory and run:
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/SmolLM2-135M-Instruct \
//!   cargo test --test stop_strings -- --ignored --nocapture
//! ```
//!
//! Acceptance (sc-7349): a request with `stop` halts at the first occurrence, the stop sequence is
//! not in the returned/streamed text, and `finish_reason` is `Stop`. The stop string is derived
//! from the model's own deterministic (greedy) baseline so it is guaranteed to be generated and to
//! sit mid-output — proving generation would have continued past it.

use core_llm::{
    load_textllm, FinishReason as CoreFinish, LoadSpec, Message, Sampling,
    StreamEvent as CoreEvent, TextLlmRequest,
};
use mlx_llm::provider::PROVIDER_ID;

fn model_dir() -> Option<String> {
    std::env::var("MLX_LLM_TEST_MODEL").ok()
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn honors_request_stop_strings_on_real_model() {
    let dir = model_dir().expect("set MLX_LLM_TEST_MODEL");
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(&dir)).unwrap();

    let req = |stop: Vec<String>| TextLlmRequest {
        messages: vec![Message::user("List the eight planets of the solar system in order.")],
        sampling: Sampling::greedy(), // deterministic => the baseline is reproducible
        max_new_tokens: 48,
        seed: Some(0),
        stop,
        ..Default::default()
    };

    // Baseline: the deterministic full output with no stop strings.
    let baseline = provider.complete(&req(vec![])).unwrap();
    assert!(
        baseline.text.chars().count() > 12,
        "need a non-trivial baseline to derive a mid-text stop string; got {:?}",
        baseline.text
    );

    // Derive a stop string from the middle of the baseline: guaranteed to be generated, with text
    // both before it (so the trim is observable) and after it (so generation truly continued past).
    let offs: Vec<usize> = baseline.text.char_indices().map(|(i, _)| i).collect();
    let start = offs[offs.len() / 3];
    let end = offs[offs.len() / 3 + 6];
    let stop = baseline.text[start..end].to_string();
    let first = baseline.text.find(&stop).unwrap();
    assert!(first > 0, "derived stop unexpectedly at the very start: {stop:?}");
    assert!(
        first + stop.len() < baseline.text.len(),
        "derived stop must have generated text after it: {stop:?}"
    );

    // Streamed run with the stop string active.
    let mut streamed = String::new();
    let out = provider
        .generate(&req(vec![stop.clone()]), &mut |ev| {
            if let CoreEvent::Token { text, .. } = ev {
                streamed.push_str(&text);
            }
        })
        .unwrap();

    println!(
        "\n=== stop = {stop:?} ===\nbaseline : {:?}\nstopped  : {:?}\n",
        baseline.text, out.text
    );

    // The output is the baseline truncated at the first occurrence of the stop string...
    assert_eq!(
        out.text,
        baseline.text[..first],
        "output must be the baseline truncated at the first stop occurrence"
    );
    // ...the stop string itself is not emitted...
    assert!(!out.text.contains(&stop), "the stop string must not appear in the output");
    // ...the streamed deltas reconstruct that same trimmed text...
    assert_eq!(streamed, out.text, "streamed deltas must reconstruct the trimmed output");
    // ...the finish reason is Stop...
    assert_eq!(out.finish_reason, Some(CoreFinish::Stop), "finish_reason must be Stop");
    // ...and it genuinely stopped early.
    assert!(out.text.len() < baseline.text.len(), "stopping must shorten the output");
    assert!(
        out.usage.generated_tokens <= baseline.usage.generated_tokens,
        "an early stop cannot generate more tokens than the baseline ({} vs {})",
        out.usage.generated_tokens,
        baseline.usage.generated_tokens
    );

    // A stop string that never appears must not alter the output — exercises the held-tail flush
    // path (no false trimming).
    let unmatched = provider
        .complete(&req(vec!["QZX-never-emitted-sequence".to_string()]))
        .unwrap();
    assert_eq!(
        unmatched.text, baseline.text,
        "an unmatched stop string must leave the output identical to the baseline"
    );
}
