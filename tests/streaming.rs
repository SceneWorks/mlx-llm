//! End-to-end streaming-decode tests on a tiny synthetic Llama model.
//!
//! These run in CI without any model weights: a small model is built from deterministic random
//! tensors, exercising the full path (`from_weights` → batch-1 contiguous cache → streaming decode
//! loop) including mid-stream cancellation and stop-token handling. A separate real-weights test
//! (`tests/real_weights.rs`, `#[ignore]`) streams from an actual Llama snapshot.

use std::collections::HashMap;

use mlx_rs::Array;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, CancelFlag, Decode, FinishReason, GenerationConfig, StreamEvent};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::{SamplingParams, SplitMix64, TokenRng};
use mlx_llm::primitives::Weights;

/// Tiny but structurally complete config: hidden = num_heads * head_dim, GQA (2 q heads, 1 kv head).
fn tiny_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        tie_word_embeddings: false,
        architecture: mlx_llm::config::Architecture::Llama,
        max_position_embeddings: 0,
        quantization: None,
        moe: None,
        attn_logit_softcap: None,
        final_logit_softcap: None,
        query_pre_attn_scalar: None,
        partial_rotary_factor: 1.0,
        mla: None,
        yarn: None,
        mrope_section: None,
    }
}

fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Array::from_slice(&data, shape)
}

/// Build a complete set of weights for `tiny_config` — also exercises `from_weights` key handling.
fn tiny_model(cfg: &ModelConfig) -> CausalLm {
    let mut rng = SplitMix64::new(0xC0FFEE);
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let inter = cfg.intermediate_size;
    let qd = cfg.num_heads * cfg.head_dim;
    let kvd = cfg.num_kv_heads * cfg.head_dim;

    let mut m: HashMap<String, Array> = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[v, h], &mut rng));
    m.insert("model.norm.weight".into(), Array::ones::<f32>(&[h]).unwrap());
    m.insert("lm_head.weight".into(), randn(&[v, h], &mut rng));
    for i in 0..cfg.num_layers {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), Array::ones::<f32>(&[h]).unwrap());
        m.insert(
            p("post_attention_layernorm.weight"),
            Array::ones::<f32>(&[h]).unwrap(),
        );
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, h], &mut rng));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, h], &mut rng));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, h], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[h, qd], &mut rng));
        m.insert(p("mlp.gate_proj.weight"), randn(&[inter, h], &mut rng));
        m.insert(p("mlp.up_proj.weight"), randn(&[inter, h], &mut rng));
        m.insert(p("mlp.down_proj.weight"), randn(&[h, inter], &mut rng));
    }

    let w = Weights::from_map(m);
    CausalLm::from_weights(&w, "", cfg.clone()).unwrap()
}

fn greedy(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(), // greedy
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

#[test]
fn streams_up_to_max_tokens() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);

    let mut events = Vec::new();
    let out = generate(
        &model,
        &[1, 2, 3],
        &greedy(5),
        &CancelFlag::new(),
        &mut |e| events.push(e),
    )
    .unwrap();

    assert_eq!(out.finish_reason, FinishReason::MaxTokens);
    assert_eq!(out.tokens.len(), 5);

    let token_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Token { .. }))
        .collect();
    assert_eq!(token_events.len(), 5);
    // Last event is Done with the right count.
    assert_eq!(
        events.last(),
        Some(&StreamEvent::Done {
            reason: FinishReason::MaxTokens,
            generated: 5
        })
    );
}

#[test]
fn generation_is_deterministic_for_fixed_seed() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let run = || {
        generate(&model, &[1, 2, 3], &greedy(8), &CancelFlag::new(), &mut |_| {})
            .unwrap()
            .tokens
    };
    assert_eq!(run(), run());
}

#[test]
fn mid_stream_cancel_stops_early_and_is_partial() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let cancel = CancelFlag::new();

    let mut count = 0;
    let out = generate(&model, &[1, 2, 3], &greedy(100), &cancel, &mut |e| {
        if let StreamEvent::Token { .. } = e {
            count += 1;
            if count == 2 {
                cancel.cancel(); // trip after the 2nd token
            }
        }
    })
    .unwrap();

    assert_eq!(out.finish_reason, FinishReason::Cancelled);
    // Stopped promptly: the loop breaks on the next iteration's cancel check.
    assert!(out.tokens.len() <= 3, "got {} tokens", out.tokens.len());
    assert!(out.tokens.len() >= 2);
}

#[test]
fn already_cancelled_returns_typed_error() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let cancel = CancelFlag::new();
    cancel.cancel();
    let res = generate(&model, &[1, 2, 3], &greedy(5), &cancel, &mut |_| {});
    assert!(matches!(res, Err(mlx_llm::Error::Canceled)));
}

#[test]
fn stop_token_halts_before_emitting_it() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);

    // Find the first greedy token, then make it a stop token: generation must halt immediately.
    let first = generate(&model, &[1, 2, 3], &greedy(1), &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .tokens[0];

    let mut cfg2 = greedy(10);
    cfg2.stop_tokens = vec![first];
    let mut events = Vec::new();
    let out = generate(&model, &[1, 2, 3], &cfg2, &CancelFlag::new(), &mut |e| {
        events.push(e)
    })
    .unwrap();

    assert_eq!(out.finish_reason, FinishReason::StopToken);
    assert!(out.tokens.is_empty()); // stop token excluded, nothing before it
    assert_eq!(
        events.last(),
        Some(&StreamEvent::Done {
            reason: FinishReason::StopToken,
            generated: 0
        })
    );
}

#[test]
fn empty_prompt_errors() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let res = generate(&model, &[], &greedy(5), &CancelFlag::new(), &mut |_| {});
    assert!(res.is_err());
}

#[test]
fn decode_trait_object_works() {
    // The loop drives `&dyn Decode`, not a concrete type.
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let decoder: &dyn Decode = &model;
    let out = generate(decoder, &[5, 6], &greedy(3), &CancelFlag::new(), &mut |_| {}).unwrap();
    assert_eq!(out.tokens.len(), 3);
}
