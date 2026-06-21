//! Runs the registered `LlamaProvider` through the `core-llm` conformance suite (story 7201).
//!
//! Builds a tiny synthetic snapshot (no model weights needed, runs in CI) whose vocabulary is fully
//! covered by the tokenizer, so the suite's seed-determinism check sees genuinely distinct text for
//! distinct seeds. A separate gated test runs the suite against a real model.

use std::path::PathBuf;

use mlx_rs::Array;

use core_llm::LoadSpec;
use core_llm_testkit::{textllm_conformance, TextLlmProfile};
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
#[ignore = "needs a real Llama snapshot via MLX_LLM_TEST_MODEL"]
fn real_model_passes_core_llm_conformance() {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").expect("set MLX_LLM_TEST_MODEL");
    let spec = LoadSpec::dense(dir);
    textllm_conformance(
        || Box::new(LlamaProvider::load(&spec).expect("load real provider")),
        &TextLlmProfile::cheap(),
    );
}
