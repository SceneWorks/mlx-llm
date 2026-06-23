//! Real-weights model-breadth tests (`#[ignore]` — need a snapshot on disk), story 7173.
//!
//! Each non-Llama architecture must load from a real HF snapshot and stream coherent text through the
//! backend-neutral `core_llm::TextLlm`, reporting the correct family tag (dispatched from
//! `config.json`). Point the per-family env var at a snapshot dir and run:
//!
//! ```text
//! MLX_LLM_GEMMA2_MODEL=/path/to/gemma-2-2b-it \
//!   cargo test --test real_breadth -- --ignored --nocapture
//! ```
//!
//! The synthetic-weights wiring gate (no download) is `tests/breadth.rs`; this is the parity-vs-real
//! check. Loading goes through the registered `mlx-llama` provider, whose descriptor family reflects
//! the architecture `config.json` dispatched to (the breadth lives behind one generic provider).

use core_llm::{load_textllm, LoadSpec, Message, Sampling, StreamEvent, TextLlmRequest};
use mlx_llm::provider::PROVIDER_ID;

/// Load the snapshot at `$env` through the `mlx-llama` provider, check its reported family tag, and
/// assert it streams coherent, word-bearing text (the streamed deltas reconstructing the output).
fn assert_streams_coherent(env: &str, family: &str) {
    let Some(dir) = std::env::var(env).ok().filter(|v| !v.is_empty()) else {
        eprintln!("skip: set {env}");
        return;
    };
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(&dir)).expect("load provider");
    assert_eq!(provider.descriptor().id, PROVIDER_ID, "resolved provider id");
    assert_eq!(provider.descriptor().family, family, "reported family tag");

    let req = TextLlmRequest {
        messages: vec![Message::user("The capital of France is")],
        sampling: Sampling::greedy(),
        max_new_tokens: 24,
        seed: Some(0),
        ..Default::default()
    };

    let mut streamed = String::new();
    let out = provider
        .generate(&req, &mut |ev| {
            if let StreamEvent::Token { text, .. } = ev {
                streamed.push_str(&text);
            }
        })
        .expect("generate");

    println!("[{family}] {}", out.text.replace('\n', " "));
    assert!(!out.text.trim().is_empty(), "{family}: produced no text");
    assert_eq!(streamed, out.text, "{family}: streamed deltas must reconstruct the final text");
    assert!(
        out.text.chars().any(|c| c.is_alphabetic()),
        "{family}: output should contain words, not just punctuation"
    );
}

/// Phi-3: the Llama decoder shape with a packed `qkv_proj` + `gate_up_proj` (split at load).
#[test]
#[ignore = "needs a Phi-3 snapshot via MLX_LLM_PHI3_MODEL"]
fn phi3_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_PHI3_MODEL", "phi3");
}

/// Qwen2-MoE: Qwen2 attention (q/k/v bias) + a sparse MoE FFN (router + top-k experts + shared).
#[test]
#[ignore = "needs a Qwen2-MoE snapshot via MLX_LLM_QWEN2MOE_MODEL"]
fn qwen2_moe_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_QWEN2MOE_MODEL", "qwen2_moe");
}

/// Gemma-2: `(1+weight)` norms, embedding ×√hidden, GeGLU, soft-capped attention + final logits,
/// the 4-norm sandwich block.
#[test]
#[ignore = "needs a Gemma-2 snapshot via MLX_LLM_GEMMA2_MODEL"]
fn gemma2_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_GEMMA2_MODEL", "gemma2");
}

/// GLM-4: 4-norm sandwich (standard RMSNorm), q/k/v bias, packed gate_up, partial + interleaved RoPE.
#[test]
#[ignore = "needs a GLM-4 snapshot via MLX_LLM_GLM4_MODEL"]
fn glm4_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_GLM4_MODEL", "glm4");
}

/// DeepSeek-V2: Multi-head Latent Attention (low-rank KV path + decoupled YaRN RoPE) and a
/// fine-grained MoE FFN (many routed experts + shared experts, a leading dense layer). Verified on
/// `deepseek-ai/DeepSeek-V2-Lite-Chat`.
#[test]
#[ignore = "needs a DeepSeek-V2 snapshot via MLX_LLM_DEEPSEEK_MODEL"]
fn deepseek_v2_streams_coherent_text() {
    assert_streams_coherent("MLX_LLM_DEEPSEEK_MODEL", "deepseek_v2");
}
