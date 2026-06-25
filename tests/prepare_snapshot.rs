use core_llm::{
    load_for_model, prepare_snapshot, LoadSpec, Message, PrepareSpec, Quantize, Sampling,
    TextLlmRequest,
};
use mlx_llm::provider::PROVIDER_ID;

const PROMPT: &str = "The capital of France is";

fn tmp_out(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mlx-llm-prepare-e2e-{label}-{}",
        std::process::id()
    ))
}

fn greedy_request() -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::user(PROMPT)],
        sampling: Sampling::greedy(),
        max_new_tokens: 256,
        seed: Some(0),
        ..Default::default()
    }
}

fn assert_loads_and_generates(out_dir: &std::path::Path) {
    let llm = load_for_model(&LoadSpec::dense(out_dir.to_string_lossy().to_string())).unwrap();
    assert_eq!(llm.descriptor().id, PROVIDER_ID);
    let out = llm.complete(&greedy_request()).unwrap();
    assert!(out.usage.generated_tokens > 0);
    let thinking = out.thinking.unwrap_or_default();
    assert!(!out.text.trim().is_empty() || !thinking.trim().is_empty());
}

#[test]
fn unknown_input_is_unsupported() {
    let src = tmp_out("unknown-src");
    let out = tmp_out("unknown-out");
    std::fs::create_dir_all(&src).unwrap();
    match prepare_snapshot(&PrepareSpec::dense(&src, &out)) {
        Err(core_llm::Error::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
    std::fs::remove_dir_all(&src).ok();
    std::fs::remove_dir_all(&out).ok();
}

#[test]
#[ignore = "needs an HF snapshot via MLX_LLM_TEST_MODEL"]
fn hf_q4_prepare_loads_and_generates() {
    let source = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let out = tmp_out("hf-q4");
    std::fs::remove_dir_all(&out).ok();
    let report = prepare_snapshot(&PrepareSpec::quantized(&source, &out, Quantize::Q4)).unwrap();
    assert_eq!(report.quantized, Some(Quantize::Q4));
    assert!(!report.passthrough);
    assert_eq!(report.out_dir, out);
    assert_loads_and_generates(&report.out_dir);
    std::fs::remove_dir_all(&report.out_dir).ok();
}

#[test]
#[ignore = "needs an HF snapshot via MLX_LLM_TEST_MODEL"]
fn hf_dense_passthrough_returns_source_without_rewrite() {
    let source = std::path::PathBuf::from(
        std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL"),
    );
    let out = tmp_out("hf-passthrough-out");
    std::fs::remove_dir_all(&out).ok();
    let report = prepare_snapshot(&PrepareSpec::dense(&source, &out)).unwrap();
    assert!(report.passthrough);
    assert_eq!(report.quantized, None);
    assert_eq!(report.out_dir, source);
    assert!(!out.exists());
    assert_loads_and_generates(&report.out_dir);
}

#[test]
#[ignore = "needs a GGUF file or dir via MLX_LLM_GGUF_SOURCE"]
fn gguf_dense_and_q4_prepare_load_and_generate() {
    let source = std::env::var("MLX_LLM_GGUF_SOURCE").expect("set MLX_LLM_GGUF_SOURCE");
    for (label, quantize) in [("dense", None), ("q4", Some(Quantize::Q4))] {
        let out = tmp_out(&format!("gguf-{label}"));
        std::fs::remove_dir_all(&out).ok();
        let spec = PrepareSpec {
            source: source.clone().into(),
            out_dir: out.clone(),
            quantize,
        };
        let report = prepare_snapshot(&spec).unwrap();
        assert_eq!(report.quantized, quantize);
        assert!(!report.passthrough);
        assert_eq!(report.out_dir, out);
        assert_loads_and_generates(&report.out_dir);
        std::fs::remove_dir_all(&report.out_dir).ok();
    }
}
