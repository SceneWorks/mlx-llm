//! Model-breadth smoke tests on tiny synthetic models (story 7173).
//!
//! Each non-Llama architecture added to the `config.json` dispatch — Phi-3, Qwen2-MoE, Gemma-2,
//! GLM-4, DeepSeek-V2 (MLA) — is built here from deterministic random weights and driven through a
//! prefill + several cached decode steps, asserting every step yields finite, correctly-shaped
//! `[1, vocab]` logits and that the KV cache grows one position per step. These run in CI on the
//! Metal device with **no model download**, guarding the per-family decoder wiring (packed qkv /
//! gate_up split, q/k/v bias, the 4-norm sandwich, GeGLU + soft-cap, the MoE router/expert/shared
//! bank, and the MLA low-rank KV path + decoupled YaRN RoPE). The real-weights parity check is the
//! env-gated `real_breadth` test (see `tests/real_weights_breadth.rs`).

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};
use serde_json::{json, Value};

use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::{SplitMix64, TokenRng};
use mlx_llm::primitives::{input_ids, KvCache, Weights};

const HIDDEN: i32 = 32;
const VOCAB: i32 = 48;
const HEAD_DIM: i32 = 8;
const LAYERS: usize = 2;

/// Small deterministic random `[out, in]` weight.
fn randn(shape: &[i32], rng: &mut SplitMix64) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
    Array::from_slice(&data, shape)
}

fn ones(d: i32) -> Array {
    Array::ones::<f32>(&[d]).unwrap()
}

/// Build the model from a config value + weight map, then drive prefill + 4 cached decode steps,
/// asserting finite `[1, vocab]` logits at every step and that the cache grows one position per step.
fn assert_prefills_and_decodes(cfg: Value, weights: HashMap<String, Array>, family: &str) {
    let cfg = ModelConfig::from_json(&cfg).unwrap();
    assert_eq!(cfg.architecture.family(), family);
    let vocab = cfg.vocab_size;
    let model = CausalLm::from_weights(&Weights::from_map(weights), "", cfg).unwrap();

    let prompt = [1i32, 2, 3, 4, 5];
    let ids = input_ids(&prompt);
    let mut cache = model.new_cache();

    let logits = model.decode_logits(&ids, &mut cache, 0).unwrap();
    assert_eq!(logits.shape(), &[1, vocab], "{family}: prefill logits shape");
    let mut next = assert_finite_argmax(&logits, family);

    for step in 0..4 {
        let offset = prompt.len() as i32 + step;
        let step_ids = input_ids(&[next]);
        let logits = model.decode_logits(&step_ids, &mut cache, offset).unwrap();
        assert_eq!(logits.shape(), &[1, vocab], "{family}: decode logits shape");
        next = assert_finite_argmax(&logits, family);
    }
    assert_eq!(cache.offset(), prompt.len() as i32 + 4, "{family}: cache grew per step");
}

fn assert_finite_argmax(logits: &Array, family: &str) -> i32 {
    let host = logits.as_dtype(Dtype::Float32).unwrap();
    let v = host.as_slice::<f32>();
    assert!(v.iter().all(|x| x.is_finite()), "{family}: non-finite logits");
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

/// Insert a dense gated MLP (`gate`/`up` `[inter, hidden]`, `down` `[hidden, inter]`).
fn dense_mlp(m: &mut HashMap<String, Array>, p: &dyn Fn(&str) -> String, inter: i32, rng: &mut SplitMix64) {
    m.insert(p("mlp.gate_proj.weight"), randn(&[inter, HIDDEN], rng));
    m.insert(p("mlp.up_proj.weight"), randn(&[inter, HIDDEN], rng));
    m.insert(p("mlp.down_proj.weight"), randn(&[HIDDEN, inter], rng));
}

/// Phi-3: the Llama shape with a packed `qkv_proj` + `gate_up_proj`, split at load.
#[test]
fn phi3_prefills_and_decodes() {
    let (heads, kv, inter) = (4, 4, 64);
    let (qd, kvd) = (heads * HEAD_DIM, kv * HEAD_DIM);
    let cfg = json!({
        "architectures": ["Phi3ForCausalLM"], "model_type": "phi3",
        "hidden_size": HIDDEN, "intermediate_size": inter, "num_hidden_layers": LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": kv, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false
    });
    let mut rng = SplitMix64::new(0x9117_3001);
    let mut m = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    m.insert("model.norm.weight".into(), ones(HIDDEN));
    m.insert("lm_head.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        m.insert(p("self_attn.qkv_proj.weight"), randn(&[qd + 2 * kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[HIDDEN, qd], &mut rng));
        m.insert(p("mlp.gate_up_proj.weight"), randn(&[2 * inter, HIDDEN], &mut rng));
        m.insert(p("mlp.down_proj.weight"), randn(&[HIDDEN, inter], &mut rng));
    }
    assert_prefills_and_decodes(cfg, m, "phi3");
}

/// Qwen2-MoE: Qwen2 attention (q/k/v bias) + a sparse MoE FFN with a sigmoid-gated shared expert.
#[test]
fn qwen2_moe_prefills_and_decodes() {
    let (heads, kv, inter) = (4, 2, 64);
    let (qd, kvd) = (heads * HEAD_DIM, kv * HEAD_DIM);
    let (n_exp, moe_inter, shared_inter) = (4, 16, 32);
    let cfg = json!({
        "architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe",
        "hidden_size": HIDDEN, "intermediate_size": inter, "num_hidden_layers": LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": kv, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": false,
        "num_experts": n_exp, "num_experts_per_tok": 2, "norm_topk_prob": false,
        "moe_intermediate_size": moe_inter, "shared_expert_intermediate_size": shared_inter
    });
    let mut rng = SplitMix64::new(0x9217_3002);
    let mut m = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    m.insert("model.norm.weight".into(), ones(HIDDEN));
    m.insert("lm_head.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        // q/k/v carry bias (Qwen2); o_proj has none.
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, HIDDEN], &mut rng));
        m.insert(p("self_attn.q_proj.bias"), randn(&[qd], &mut rng));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.k_proj.bias"), randn(&[kvd], &mut rng));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.v_proj.bias"), randn(&[kvd], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[HIDDEN, qd], &mut rng));
        // MoE FFN: router + routed experts + a sigmoid-gated shared expert.
        m.insert(p("mlp.gate.weight"), randn(&[n_exp, HIDDEN], &mut rng));
        for e in 0..n_exp {
            let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
            m.insert(ep("gate_proj.weight"), randn(&[moe_inter, HIDDEN], &mut rng));
            m.insert(ep("up_proj.weight"), randn(&[moe_inter, HIDDEN], &mut rng));
            m.insert(ep("down_proj.weight"), randn(&[HIDDEN, moe_inter], &mut rng));
        }
        m.insert(p("mlp.shared_expert.gate_proj.weight"), randn(&[shared_inter, HIDDEN], &mut rng));
        m.insert(p("mlp.shared_expert.up_proj.weight"), randn(&[shared_inter, HIDDEN], &mut rng));
        m.insert(p("mlp.shared_expert.down_proj.weight"), randn(&[HIDDEN, shared_inter], &mut rng));
        m.insert(p("mlp.shared_expert_gate.weight"), randn(&[1, HIDDEN], &mut rng));
    }
    assert_prefills_and_decodes(cfg, m, "qwen2_moe");
}

/// Gemma-2: `(1+weight)` norms, embedding ×√hidden, GeGLU, soft-capped attention + final logits,
/// the 4-norm sandwich block, and a `query_pre_attn_scalar` attention scale.
#[test]
fn gemma2_prefills_and_decodes() {
    let (heads, kv, inter) = (4, 2, 64);
    let (qd, kvd) = (heads * HEAD_DIM, kv * HEAD_DIM);
    let cfg = json!({
        "architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2",
        "hidden_size": HIDDEN, "intermediate_size": inter, "num_hidden_layers": LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": HEAD_DIM,
        "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "query_pre_attn_scalar": HEAD_DIM, "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0
        // tie_word_embeddings omitted — Gemma ties by default (no lm_head needed).
    });
    let mut rng = SplitMix64::new(0x9317_3003);
    let mut m = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    m.insert("model.norm.weight".into(), ones(HIDDEN));
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        // Four-norm sandwich.
        m.insert(p("input_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        m.insert(p("pre_feedforward_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_feedforward_layernorm.weight"), ones(HIDDEN));
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, HIDDEN], &mut rng));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[HIDDEN, qd], &mut rng));
        dense_mlp(&mut m, &p, inter, &mut rng);
    }
    assert_prefills_and_decodes(cfg, m, "gemma2");
}

/// GLM-4: a 4-norm sandwich (standard RMSNorm, GLM key names), q/k/v bias, a packed `gate_up_proj`,
/// and partial + interleaved RoPE.
#[test]
fn glm4_prefills_and_decodes() {
    let (heads, kv, inter) = (4, 2, 64);
    let (qd, kvd) = (heads * HEAD_DIM, kv * HEAD_DIM);
    let cfg = json!({
        "architectures": ["Glm4ForCausalLM"], "model_type": "glm4",
        "hidden_size": HIDDEN, "intermediate_size": inter, "num_hidden_layers": LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": HEAD_DIM,
        "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5, "tie_word_embeddings": false
    });
    let mut rng = SplitMix64::new(0x9417_3004);
    let mut m = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    m.insert("model.norm.weight".into(), ones(HIDDEN));
    m.insert("lm_head.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        // GLM-4 sandwich norm names.
        m.insert(p("input_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_self_attn_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_mlp_layernorm.weight"), ones(HIDDEN));
        // q/k/v carry bias (GLM-4); o_proj has none.
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, HIDDEN], &mut rng));
        m.insert(p("self_attn.q_proj.bias"), randn(&[qd], &mut rng));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.k_proj.bias"), randn(&[kvd], &mut rng));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.v_proj.bias"), randn(&[kvd], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[HIDDEN, qd], &mut rng));
        // Packed gate_up.
        m.insert(p("mlp.gate_up_proj.weight"), randn(&[2 * inter, HIDDEN], &mut rng));
        m.insert(p("mlp.down_proj.weight"), randn(&[HIDDEN, inter], &mut rng));
    }
    assert_prefills_and_decodes(cfg, m, "glm4");
}

/// Qwen3-VL (`qwen3_vl`): the VLM-wrapped standard full-attention Qwen3 decoder. Exercises the
/// VLM-nested weight prefix (`model.language_model.*` with `lm_head.weight` at the root), the
/// per-head q/k norm, the 256K interleaved-M-RoPE config, the multimodal splice, and the
/// interleaved-M-RoPE + DeepStack prefill seam — all on tiny synthetic weights (no snapshot).
#[test]
fn qwen3vl_prefills_decodes_and_fuses_deepstack() {
    let (heads, kv, inter) = (4, 2, 64);
    let (qd, kvd) = (heads * HEAD_DIM, kv * HEAD_DIM);
    let cfg_v = json!({
        "architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl",
        "image_token_id": 40, "video_token_id": 41,
        "vision_start_token_id": 42, "vision_end_token_id": 43,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": HIDDEN, "intermediate_size": inter, "num_hidden_layers": LAYERS,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": HEAD_DIM,
            "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 5000000.0,
            "max_position_embeddings": 262144,
            "rope_scaling": { "mrope_interleaved": true, "mrope_section": [2, 1, 1], "rope_type": "default" }
        },
        "vision_config": { "model_type": "qwen3_vl", "depth": 2 }
    });
    let cfg = ModelConfig::from_json(&cfg_v).unwrap();
    assert_eq!(cfg.architecture.family(), "qwen3_vl");
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.mrope_section, Some([2, 1, 1])); // sums to head_dim/2 = 4

    let mut rng = SplitMix64::new(0x9617_3006);
    let mut m = HashMap::new();
    // VLM-nested layout: decoder under `model.language_model.*`, `lm_head.weight` at the root.
    m.insert("model.language_model.embed_tokens.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    m.insert("model.language_model.norm.weight".into(), ones(HIDDEN));
    m.insert("lm_head.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.language_model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        m.insert(p("self_attn.q_proj.weight"), randn(&[qd, HIDDEN], &mut rng));
        m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, HIDDEN], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[HIDDEN, qd], &mut rng));
        m.insert(p("self_attn.q_norm.weight"), ones(HEAD_DIM)); // per-head q/k norm (Qwen3)
        m.insert(p("self_attn.k_norm.weight"), ones(HEAD_DIM));
        m.insert(p("mlp.gate_proj.weight"), randn(&[inter, HIDDEN], &mut rng));
        m.insert(p("mlp.up_proj.weight"), randn(&[inter, HIDDEN], &mut rng));
        m.insert(p("mlp.down_proj.weight"), randn(&[HIDDEN, inter], &mut rng));
    }
    let model = CausalLm::from_weights(&Weights::from_map(m), "", cfg).unwrap();

    // (a) Plain text prefill + cached decode works through the VLM-nested decoder.
    let prompt = [1i32, 2, 3, 4, 5];
    let ids = input_ids(&prompt);
    let mut cache = model.new_cache();
    let logits = model.decode_logits(&ids, &mut cache, 0).unwrap();
    assert_eq!(logits.shape(), &[1, VOCAB], "qwen3_vl prefill logits shape");
    assert!(logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().iter().all(|x| x.is_finite()));

    // (b) Text-only invariant: the interleaved-M-RoPE + DeepStack prefill with **equal** t/h/w rows
    // and an **empty** deepstack must be bit-identical to the plain 1-D-RoPE prefill.
    let s = prompt.len() as i32;
    let pos: Vec<i32> = (0..s).collect();
    let embeds = model.embed_input_ids(&ids).unwrap();
    let mut cache_a = model.new_cache();
    let plain = model.decode_logits_from_embeds(&embeds, &mut cache_a, 0).unwrap();
    let mut cache_b = model.new_cache();
    let visual_mask = vec![false; s as usize];
    let mrope = model
        .decode_logits_from_embeds_mrope_deepstack(
            &embeds,
            [pos.as_slice(), pos.as_slice(), pos.as_slice()],
            &mut cache_b,
            &visual_mask,
            &[],
        )
        .unwrap();
    let pa = plain.as_dtype(Dtype::Float32).unwrap();
    let mb = mrope.as_dtype(Dtype::Float32).unwrap();
    let max_diff = pa
        .as_slice::<f32>()
        .iter()
        .zip(mb.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "text-only M-RoPE must equal plain RoPE, max diff {max_diff}");

    // (c) Image path: expand a single image placeholder to its merged-token count, splice synthetic
    // features, compute interleaved-M-RoPE 3-D positions, and run the DeepStack-fused prefill. A 2×2
    // patch grid (merge 2 ⇒ 1 token... use a 4×4 grid ⇒ 4 tokens) checked end to end.
    let img_id = 40;
    let grid = [1i32, 4, 4]; // t,h,w in patch units; merge 2 ⇒ 1·2·2 = 4 merged tokens
    let merge = 2;
    let count = mlx_llm::models::qwen35::vision_merged_token_count(grid, merge);
    assert_eq!(count, 4);
    // Prompt: [text, <vision_start>, <img>, <vision_end>, text]; expand the single <img> to `count`.
    let raw = [1i32, 42, img_id, 43, 9];
    let expanded =
        mlx_llm::models::qwen35::expand_vision_placeholders(&raw, img_id, &[count]).unwrap();
    assert_eq!(expanded.len(), raw.len() - 1 + count);
    let visual_pos_mask: Vec<bool> = expanded.iter().map(|&id| id == img_id).collect();

    let features = randn(&[count as i32, HIDDEN], &mut rng).as_dtype(Dtype::Bfloat16).unwrap();
    let exp_ids = input_ids(&expanded);
    let img_embeds = model.embed_input_ids(&exp_ids).unwrap();
    let spliced = model.splice_image_features(&img_embeds, &expanded, &features, img_id).unwrap();
    let (t, h, w, delta) = model.mrope_positions(&expanded, &[grid], img_id, merge).unwrap();
    assert_eq!(t.len(), expanded.len());

    // Two synthetic DeepStack taps (one per decoder layer), each [count, hidden].
    let deepstack = vec![
        randn(&[count as i32, HIDDEN], &mut rng).as_dtype(Dtype::Bfloat16).unwrap(),
        randn(&[count as i32, HIDDEN], &mut rng).as_dtype(Dtype::Bfloat16).unwrap(),
    ];
    let mut cache_img = model.new_cache();
    let img_logits = model
        .decode_logits_from_embeds_mrope_deepstack(
            &spliced,
            [t.as_slice(), h.as_slice(), w.as_slice()],
            &mut cache_img,
            &visual_pos_mask,
            &deepstack,
        )
        .unwrap();
    assert_eq!(img_logits.shape(), &[1, VOCAB], "qwen3_vl image prefill logits shape");
    assert!(img_logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().iter().all(|x| x.is_finite()));

    // DeepStack fusion must change the output vs. an empty deepstack (the features are added in).
    let mut cache_nods = model.new_cache();
    let no_ds = model
        .decode_logits_from_embeds_mrope_deepstack(
            &spliced,
            [t.as_slice(), h.as_slice(), w.as_slice()],
            &mut cache_nods,
            &visual_pos_mask,
            &[],
        )
        .unwrap();
    let diff = img_logits
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .zip(no_ds.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff > 1e-3, "DeepStack fusion must change the logits, max diff {diff}");

    // The continuation decode uses `cache_len + mrope_delta` as the M-RoPE position (image tokens
    // compress the cursor, so `delta` may be negative). It must run and yield finite logits.
    let next = input_ids(&[7]);
    let cont = model
        .decode_logits(&next, &mut cache_img, expanded.len() as i32 + delta)
        .unwrap();
    assert_eq!(cont.shape(), &[1, VOCAB], "qwen3_vl continuation logits shape");
    assert!(cont.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().iter().all(|x| x.is_finite()));
}

/// DeepSeek-V2: Multi-head Latent Attention (full `q_proj`, low-rank KV path, decoupled YaRN RoPE)
/// and a fine-grained MoE FFN with a leading dense layer and an ungated shared expert.
#[test]
fn deepseek_v2_prefills_and_decodes() {
    let (heads, qk_nope, qk_rope, v_head, kv_lora) = (2, 16, 8, 16, 24);
    let q_head = qk_nope + qk_rope;
    let (n_routed, moe_inter, n_shared, dense_inter) = (4, 16, 1, 32);
    let cfg = json!({
        "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
        "hidden_size": HIDDEN, "intermediate_size": dense_inter, "num_hidden_layers": LAYERS,
        "num_attention_heads": heads, "num_key_value_heads": heads, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false,
        "q_lora_rank": null, "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
        "n_routed_experts": n_routed, "num_experts_per_tok": 2, "n_shared_experts": n_shared,
        "moe_intermediate_size": moe_inter, "first_k_dense_replace": 1, "norm_topk_prob": false,
        "routed_scaling_factor": 1.0,
        "rope_scaling": { "type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1,
            "mscale": 0.707, "mscale_all_dim": 0.707, "original_max_position_embeddings": 64 }
    });
    let mut rng = SplitMix64::new(0x9517_3005);
    let mut m = HashMap::new();
    m.insert("model.embed_tokens.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    m.insert("model.norm.weight".into(), ones(HIDDEN));
    m.insert("lm_head.weight".into(), randn(&[VOCAB, HIDDEN], &mut rng));
    for i in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        m.insert(p("input_layernorm.weight"), ones(HIDDEN));
        m.insert(p("post_attention_layernorm.weight"), ones(HIDDEN));
        // MLA projections (full q_proj — DeepSeek-V2-Lite shape).
        m.insert(p("self_attn.q_proj.weight"), randn(&[heads * q_head, HIDDEN], &mut rng));
        m.insert(p("self_attn.kv_a_proj_with_mqa.weight"), randn(&[kv_lora + qk_rope, HIDDEN], &mut rng));
        m.insert(p("self_attn.kv_a_layernorm.weight"), ones(kv_lora));
        m.insert(p("self_attn.kv_b_proj.weight"), randn(&[heads * (qk_nope + v_head), kv_lora], &mut rng));
        m.insert(p("self_attn.o_proj.weight"), randn(&[HIDDEN, heads * v_head], &mut rng));
        if i == 0 {
            // Leading dense layer (first_k_dense_replace = 1).
            dense_mlp(&mut m, &p, dense_inter, &mut rng);
        } else {
            // Fine-grained MoE with an ungated shared expert (plural key).
            m.insert(p("mlp.gate.weight"), randn(&[n_routed, HIDDEN], &mut rng));
            for e in 0..n_routed {
                let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                m.insert(ep("gate_proj.weight"), randn(&[moe_inter, HIDDEN], &mut rng));
                m.insert(ep("up_proj.weight"), randn(&[moe_inter, HIDDEN], &mut rng));
                m.insert(ep("down_proj.weight"), randn(&[HIDDEN, moe_inter], &mut rng));
            }
            let shared_inter = n_shared * moe_inter;
            m.insert(p("mlp.shared_experts.gate_proj.weight"), randn(&[shared_inter, HIDDEN], &mut rng));
            m.insert(p("mlp.shared_experts.up_proj.weight"), randn(&[shared_inter, HIDDEN], &mut rng));
            m.insert(p("mlp.shared_experts.down_proj.weight"), randn(&[HIDDEN, shared_inter], &mut rng));
        }
    }
    assert_prefills_and_decodes(cfg, m, "deepseek_v2");
}
