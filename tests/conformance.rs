//! Runs the registered `LlamaProvider` through the `core-llm` conformance suite (story 7201).
//!
//! Builds a tiny synthetic snapshot (no model weights needed, runs in CI) whose vocabulary is fully
//! covered by the tokenizer, so the suite's seed-determinism check sees genuinely distinct text for
//! distinct seeds. A separate gated test runs the suite against a real model.

use std::path::PathBuf;

use mlx_rs::Array;

use core_llm::{load_for_model, LoadSpec, Message, TextLlmRequest};
use core_llm_testkit::{
    check_snapshot_preparer, textllm_conformance, SnapshotPreparerProfile, TextLlmProfile,
};
use mlx_llm::primitives::sampler::{SplitMix64, TokenRng};
use mlx_llm::provider::PROVIDER_ID;
use mlx_llm::LlamaProvider;

const VOCAB: usize = 32;

fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Array::from_slice(&data, shape)
}

/// A tokenizer.json whose vocab is `t0..t{VOCAB-1}` (whitespace WordLevel), so every model token id
/// decodes to a distinct, non-empty piece — distinct seeds therefore yield distinct text.
fn tokenizer_json() -> String {
    let entries: Vec<String> = (0..VOCAB).map(|i| format!("\"t{i}\": {i}")).collect();
    format!(
        r#"{{
            "version": "1.0",
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }},
            "post_processor": null,
            "decoder": null,
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }}
        }}"#,
        entries.join(", ")
    )
}

fn write_snapshot() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mlx-llm-conformance-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // eos_token_id outside the vocab so generation always runs to the token budget.
    let config = format!(
        r#"{{
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": {VOCAB},
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
            "eos_token_id": 999
        }}"#
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();

    let (h, v, inter, qd, kvd) = (8, VOCAB as i32, 16, 8, 4);
    let mut rng = SplitMix64::new(0xBEEF);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    arrays.push(("model.embed_tokens.weight".into(), randn(&[v, h], &mut rng)));
    arrays.push(("model.norm.weight".into(), Array::ones::<f32>(&[h]).unwrap()));
    arrays.push(("lm_head.weight".into(), randn(&[v, h], &mut rng)));
    for i in 0..2 {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        arrays.push((p("input_layernorm.weight"), Array::ones::<f32>(&[h]).unwrap()));
        arrays.push((
            p("post_attention_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        ));
        arrays.push((p("self_attn.q_proj.weight"), randn(&[qd, h], &mut rng)));
        arrays.push((p("self_attn.k_proj.weight"), randn(&[kvd, h], &mut rng)));
        arrays.push((p("self_attn.v_proj.weight"), randn(&[kvd, h], &mut rng)));
        arrays.push((p("self_attn.o_proj.weight"), randn(&[h, qd], &mut rng)));
        arrays.push((p("mlp.gate_proj.weight"), randn(&[inter, h], &mut rng)));
        arrays.push((p("mlp.up_proj.weight"), randn(&[inter, h], &mut rng)));
        arrays.push((p("mlp.down_proj.weight"), randn(&[h, inter], &mut rng)));
    }
    let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, a)| (k.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
    dir
}

#[test]
fn llama_provider_passes_core_llm_conformance() {
    let dir = write_snapshot();
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());

    // The closure loads a fresh provider; the suite drives it through every contract guarantee and
    // panics with an aggregated message on any failure.
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load synthetic provider")),
        &TextLlmProfile::cheap(),
    );

    // Sanity: the provider id the suite checked is the registered one.
    assert_eq!(PROVIDER_ID, "mlx-llama");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs an HF/GGUF model source via MLX_LLM_PREPARE_SOURCE"]
fn real_snapshot_preparer_passes_core_llm_conformance() {
    let source = std::env::var("MLX_LLM_PREPARE_SOURCE").expect("set MLX_LLM_PREPARE_SOURCE");
    let out_dir = std::env::temp_dir().join(format!("mlx-llm-prepare-conformance-{}", std::process::id()));
    std::fs::remove_dir_all(&out_dir).ok();
    check_snapshot_preparer(&SnapshotPreparerProfile {
        source: PathBuf::from(source),
        out_dir: out_dir.clone(),
        quantize: Some(core_llm::Quantize::Q4),
    })
    .expect("snapshot preparer conformance");
    std::fs::remove_dir_all(&out_dir).ok();
}

#[test]
#[ignore = "needs a real Llama snapshot via MLX_LLM_TEST_MODEL"]
fn real_model_passes_core_llm_conformance() {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let spec = LoadSpec::dense(dir);
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load real provider")),
        &TextLlmProfile::cheap(),
    );
}

#[test]
#[ignore = "needs a real Qwen3 (thinking) snapshot via MLX_LLM_QWEN3_MODEL"]
fn real_qwen3_passes_core_llm_conformance() {
    // The full generic suite against a *thinking* model: exercises the thinking-aware
    // check_streaming / check_seed_determinism / check_thinking (sc-7595) on real reasoning output —
    // a short greedy run stays inside <think>, so the answer is empty and the reasoning is split
    // into output.thinking. Regression lock for the engine being fully verifiable on a thinking model.
    let dir = std::env::var("MLX_LLM_QWEN3_MODEL").expect("set MLX_LLM_QWEN3_MODEL");
    let spec = LoadSpec::dense(dir);
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load real Qwen3 provider")),
        &TextLlmProfile::cheap(),
    );
}

#[test]
#[ignore = "needs the Qwen3.6-27B (hybrid, thinking) snapshot via MLX_LLM_QWEN35_MODEL"]
fn real_qwen35_passes_core_llm_conformance() {
    // The full generic suite against the hybrid Gated-DeltaNet / gated-full-attention decoder
    // (sc-7629): streaming, seed determinism, and thinking conformance on real Qwen3.6-27B weights —
    // the contract-level acceptance that the hybrid decoder behaves as a first-class text provider.
    let dir = std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL");
    let spec = LoadSpec::dense(dir);
    // The `cheap()` profile ("Hello", 16 tokens) is too short and predictable for a 27B: its opening
    // reasoning tokens are near-argmax, so two seeds sample identically and the determinism check
    // false-flags the seed as ignored (the sampler is the same one Qwen3-0.6B passes with). Tune the
    // profile — an open-ended prompt and a larger budget — so seed divergence is actually observable.
    let mut profile = TextLlmProfile::cheap();
    profile.prompt = "Tell me a short story about a robot who learns to paint.".to_string();
    profile.max_new_tokens = 96;
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load real Qwen3.6 provider")),
        &profile,
    );
}

// --- story 7406: model-first resolution (core_llm::load_for_model) over the weightless probe ---

/// A `config.json`-only snapshot (no safetensors, no tokenizer) used to prove the `can_load` probe
/// is weightless and architecture-aware.
fn write_config_only(name: &str, config: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mlx-llm-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    dir
}

#[test]
fn can_load_is_weightless_and_architecture_aware() {
    // A directory with ONLY config.json (no shards): if the probe read weights this would fail.
    let llama = write_config_only(
        "canload-llama",
        r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":8}"#,
    );
    let lspec = LoadSpec::dense(llama.to_str().unwrap().to_string());
    assert!(mlx_llm::provider::can_load(&lspec), "text provider must claim a Llama snapshot");
    assert!(!mlx_llm::joycaption::can_load(&lspec), "vision provider must decline a text snapshot");
    let _ = std::fs::remove_dir_all(&llama);

    // An unsupported architecture is declined (no panic, no silent default).
    let unknown = write_config_only(
        "canload-unknown",
        r#"{"architectures":["BertModel"],"model_type":"bert"}"#,
    );
    let uspec = LoadSpec::dense(unknown.to_str().unwrap().to_string());
    assert!(!mlx_llm::provider::can_load(&uspec));
    let _ = std::fs::remove_dir_all(&unknown);

    // A multimodal snapshot: the text provider declines (a `vision_config` is present even though
    // the nested text arch is llama), the vision provider claims it.
    let vlm = write_config_only(
        "canload-vlm",
        r#"{"architectures":["LlavaForConditionalGeneration"],"model_type":"llava",
            "text_config":{"architectures":["LlamaForCausalLM"],"model_type":"llama"},
            "vision_config":{"hidden_size":16}}"#,
    );
    let vspec = LoadSpec::dense(vlm.to_str().unwrap().to_string());
    assert!(!mlx_llm::provider::can_load(&vspec), "text provider must decline a VLM");
    assert!(mlx_llm::joycaption::can_load(&vspec), "vision provider must claim a VLM");
    let _ = std::fs::remove_dir_all(&vlm);

    // A nonexistent path is declined gracefully.
    assert!(!mlx_llm::provider::can_load(&LoadSpec::dense("/no/such/dir")));
}

#[test]
fn load_for_model_resolves_synthetic_snapshot_without_naming_a_provider() {
    let dir = write_snapshot();
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());

    // No provider id named: the resolver reads config.json, picks the mlx text provider via its
    // can_load probe, and loads it.
    let llm = load_for_model(&spec).expect("load_for_model resolves the synthetic snapshot");
    assert_eq!(llm.descriptor().id, PROVIDER_ID);
    assert_eq!(llm.descriptor().backend, "mlx");

    // And it actually generates.
    let req = TextLlmRequest::new(vec![Message::user("t1 t2 t3")], 4);
    let out = llm.complete(&req).expect("generate");
    assert!(!out.text.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_for_model_unknown_architecture_is_a_typed_error() {
    let dir = write_config_only(
        "lfm-unknown",
        r#"{"architectures":["BertModel"],"model_type":"bert"}"#,
    );
    let spec = LoadSpec::dense(dir.to_str().unwrap().to_string());
    match load_for_model(&spec) {
        Err(core_llm::Error::Unsupported(m)) => {
            assert!(m.contains("no registered provider can serve"), "{m}");
            assert!(m.contains("bert"), "error should surface the model arch: {m}");
        }
        Err(e) => panic!("expected Unsupported, got error: {e}"),
        Ok(_) => panic!("expected Unsupported, got a loaded provider"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs a real Llama snapshot via MLX_LLM_TEST_MODEL"]
fn load_for_model_round_trips_real_llama() {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let spec = LoadSpec::dense(dir);
    assert!(mlx_llm::provider::can_load(&spec));
    let llm = load_for_model(&spec).expect("resolve + load real Llama by model");
    assert_eq!(llm.descriptor().id, PROVIDER_ID);
    let req = TextLlmRequest::new(vec![Message::user("The capital of France is")], 8);
    let out = llm.complete(&req).expect("generate");
    assert!(!out.text.is_empty());
}

#[test]
#[ignore = "needs a real Qwen3 snapshot via MLX_LLM_QWEN3_MODEL"]
fn load_for_model_round_trips_real_qwen3() {
    let dir = std::env::var("MLX_LLM_QWEN3_MODEL").expect("set MLX_LLM_QWEN3_MODEL");
    let spec = LoadSpec::dense(dir);
    let llm = load_for_model(&spec).expect("resolve + load real Qwen3 by model");
    assert_eq!(llm.descriptor().id, PROVIDER_ID);
    // The true family resolves post-load to qwen3 even though the static descriptor.family is
    // "llama" — the exact case `can_load`-based resolution exists to handle.
    assert_eq!(llm.descriptor().family, "qwen3");
    // Qwen3 is a thinking model (sc-7585): its template gates enable_thinking → supports_thinking on.
    assert!(llm.descriptor().capabilities.supports_thinking);
    let req = TextLlmRequest::new(vec![Message::user("The capital of France is")], 8);
    let out = llm.complete(&req).expect("generate");
    // A short greedy run is still inside the <think> block, so the answer (out.text) may be empty
    // while reasoning streams — assert it produced text on either channel.
    assert!(
        !out.text.is_empty() || out.thinking.as_deref().is_some_and(|t| !t.trim().is_empty()),
        "expected text on the answer or reasoning channel"
    );
}
