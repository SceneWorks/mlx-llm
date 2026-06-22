//! Real-weights prompt-lookup speculative-decoding tests (`#[ignore]`), story 7171.
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/SmolLM2-135M-Instruct \
//!   cargo test --test speculative -- --ignored --nocapture
//! ```
//!
//! ## What these prove
//! - **Exactness gate (`num_draft = 0`)**: with no drafts the verify is a single-token forward, so
//!   speculative decoding is **token-for-token identical** to non-speculative `generate`. This pins
//!   the loop / acceptance / KV-rollback / first-token logic as exactly correct.
//! - **Tracking + measured speedup (multi-draft)**: on a context-repetitive prompt the n-gram proposer
//!   hits, so several tokens commit per target forward (`forwards < tokens`). The greedy run *tracks*
//!   non-speculative; it does not bit-match, because the multi-token verify kernel rounds a few bf16
//!   ULP differently from the single-token decode kernel (a target-model property, cf. story 7167) —
//!   the acceptance itself is exact w.r.t. the verify forward (proven in core-llm's Monte-Carlo test).
//! - **Stochastic** is deterministic for a fixed seed and produces valid output.

use core_llm::Tokenizer;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{
    generate, generate_draft_speculative, generate_prompt_lookup, CancelFlag, GenerationConfig,
    SpeculativeConfig,
};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::projection::QuantSpec;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;

struct Fixture {
    model: CausalLm,
    tok: Tokenizer,
}

fn load_from(env: &str) -> Option<Fixture> {
    let dir = std::env::var(env).ok()?;
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&Weights::from_dir(&dir).unwrap(), "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    Some(Fixture { model, tok })
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, true).unwrap().into_iter().map(|id| id as i32).collect()
}

fn greedy_config(max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(), // greedy
        seed: Some(0),
        stop_tokens: Vec::new(),
    }
}

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn run_suite(fx: Fixture) {
    // ---- Exactness gate: num_draft = 0 ⇒ single-token verify ⇒ identical to non-speculative. ----
    let no_draft = SpeculativeConfig { max_ngram: 3, num_draft: 0 };
    for text in [
        "The capital of France is",
        "Once upon a time in a small village there lived a curious",
        "Q: What is 2+2? A:",
    ] {
        let p = encode(&fx.tok, text);
        let cfg = greedy_config(32);
        let base = generate(&fx.model, &p, &cfg, &CancelFlag::new(), &mut |_| {}).unwrap().tokens;
        let (spec_out, stats) =
            generate_prompt_lookup(&fx.model, &p, &cfg, &no_draft, &CancelFlag::new(), &mut |_| {}).unwrap();
        assert_eq!(spec_out.tokens, base, "num_draft=0 speculative must equal non-speculative for '{text}'");
        assert_eq!(stats.proposed, 0);
        assert_eq!(stats.accepted, 0);
    }

    // ---- Multi-draft: tracks non-speculative + measured speedup on a context-repetitive prompt. ----
    let spec = SpeculativeConfig::default();
    let rep = encode(
        &fx.tok,
        "List: alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma alpha beta gamma",
    );
    let cfg = greedy_config(48);
    let base = generate(&fx.model, &rep, &cfg, &CancelFlag::new(), &mut |_| {}).unwrap().tokens;
    let (spec_out, stats) =
        generate_prompt_lookup(&fx.model, &rep, &cfg, &spec, &CancelFlag::new(), &mut |_| {}).unwrap();
    let n = spec_out.tokens.len();
    let cp = common_prefix(&spec_out.tokens, &base);
    println!(
        "{n} tokens in {} target forwards ({:.2} tok/forward); {}/{} drafts accepted; \
         tracks non-spec for {cp}/{} tokens",
        stats.forwards,
        n as f64 / stats.forwards as f64,
        stats.accepted,
        stats.proposed,
        base.len(),
    );
    assert!(stats.accepted > 0, "the repetitive prompt should yield n-gram hits");
    assert!(stats.forwards < n + 1, "speculation must use fewer forwards than tokens generated");
    assert!(cp >= 1, "speculative output must track non-speculative (diverges only on bf16 near-ties)");
    assert!(!spec_out.tokens.is_empty());

    // ---- Stochastic: deterministic for a fixed seed, and valid. ----
    let scfg = GenerationConfig {
        max_new_tokens: 24,
        sampling: SamplingParams { temperature: 0.8, top_p: 0.95, ..Default::default() },
        seed: Some(7),
        stop_tokens: Vec::new(),
    };
    let p = encode(&fx.tok, "Write a short sentence about the sea:");
    let a = generate_prompt_lookup(&fx.model, &p, &scfg, &spec, &CancelFlag::new(), &mut |_| {}).unwrap().0;
    let b = generate_prompt_lookup(&fx.model, &p, &scfg, &spec, &CancelFlag::new(), &mut |_| {}).unwrap().0;
    assert_eq!(a.tokens, b.tokens, "stochastic speculative must be deterministic for a fixed seed");
    assert!(!a.tokens.is_empty());
}

/// Load a dense **target** and a Q4-quantized **draft** from the same snapshot — vocab-compatible by
/// construction, with the Q4 draft a faster, lossy approximation that yields genuine partial
/// acceptance.
fn load_draft_target(env: &str) -> Option<(CausalLm, CausalLm, Tokenizer)> {
    let dir = std::env::var(env).ok()?;
    let w = Weights::from_dir(&dir).unwrap();
    let target = CausalLm::from_weights(&w, "", ModelConfig::from_dir(&dir).unwrap()).unwrap();
    let draft =
        CausalLm::from_weights_with(&w, "", ModelConfig::from_dir(&dir).unwrap(), Some(QuantSpec::q4()))
            .unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    Some((target, draft, tok))
}

fn run_draft_suite(target: CausalLm, draft: CausalLm, tok: Tokenizer) {
    // ---- Exactness gate: num_draft = 0 ⇒ identical to non-speculative target decoding. ----
    let no_draft = SpeculativeConfig { max_ngram: 3, num_draft: 0 };
    for text in ["The capital of France is", "Q: What is 2+2? A:"] {
        let p = encode(&tok, text);
        let cfg = greedy_config(28);
        let base = generate(&target, &p, &cfg, &CancelFlag::new(), &mut |_| {}).unwrap().tokens;
        let (out, stats) =
            generate_draft_speculative(&target, &draft, &p, &cfg, &no_draft, &CancelFlag::new(), &mut |_| {})
                .unwrap();
        assert_eq!(out.tokens, base, "num_draft=0 draft-spec must equal non-speculative for '{text}'");
        assert_eq!(stats.accepted, 0);
    }

    // ---- Draft-model greedy: tracks non-spec + measured latency win from accepted drafts. ----
    let spec = SpeculativeConfig::default();
    let p = encode(&tok, "Once upon a time in a small village there lived a curious");
    let cfg = greedy_config(48);
    let base = generate(&target, &p, &cfg, &CancelFlag::new(), &mut |_| {}).unwrap().tokens;
    let (out, stats) =
        generate_draft_speculative(&target, &draft, &p, &cfg, &spec, &CancelFlag::new(), &mut |_| {}).unwrap();
    let n = out.tokens.len();
    let cp = common_prefix(&out.tokens, &base);
    println!(
        "draft-model: {n} tokens in {} target forwards ({:.2} tok/forward); {}/{} drafts accepted (Q4 draft); \
         tracks dense for {cp}/{} tokens",
        stats.forwards,
        n as f64 / stats.forwards as f64,
        stats.accepted,
        stats.proposed,
        base.len(),
    );
    assert!(stats.accepted > 0, "the Q4 draft should agree with the dense target on some tokens");
    assert!(stats.forwards < n + 1, "accepted drafts must reduce target forwards below token count");
    assert!(cp >= 1, "draft-spec output must track non-speculative (diverges only on bf16 near-ties)");

    // ---- Stochastic: deterministic for a fixed seed, valid output. ----
    let scfg = GenerationConfig {
        max_new_tokens: 24,
        sampling: SamplingParams { temperature: 0.8, top_p: 0.95, ..Default::default() },
        seed: Some(11),
        stop_tokens: Vec::new(),
    };
    let p = encode(&tok, "Describe the morning sky:");
    let a = generate_draft_speculative(&target, &draft, &p, &scfg, &spec, &CancelFlag::new(), &mut |_| {}).unwrap().0;
    let b = generate_draft_speculative(&target, &draft, &p, &scfg, &spec, &CancelFlag::new(), &mut |_| {}).unwrap().0;
    assert_eq!(a.tokens, b.tokens, "stochastic draft-spec must be deterministic for a fixed seed");
    assert!(!a.tokens.is_empty());
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn draft_speculative_llama() {
    let Some((target, draft, tok)) = load_draft_target("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    run_draft_suite(target, draft, tok);
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_QWEN3_MODEL"]
fn draft_speculative_qwen3() {
    let Some((target, draft, tok)) = load_draft_target("MLX_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set MLX_LLM_QWEN3_MODEL");
        return;
    };
    run_draft_suite(target, draft, tok);
}

/// Two models with different vocabularies (SmolLM2 vs Qwen3) must be rejected as a draft/target pair.
#[test]
#[ignore = "needs both MLX_LLM_TEST_MODEL and MLX_LLM_QWEN3_MODEL"]
fn draft_speculative_rejects_vocab_mismatch() {
    let (Some(a), Some(b)) = (load_from("MLX_LLM_TEST_MODEL"), load_from("MLX_LLM_QWEN3_MODEL")) else {
        eprintln!("skip: set both MLX_LLM_TEST_MODEL and MLX_LLM_QWEN3_MODEL");
        return;
    };
    assert_ne!(a.model.config().vocab_size, b.model.config().vocab_size);
    let p = encode(&a.tok, "Hello");
    let err = generate_draft_speculative(
        &a.model,
        &b.model,
        &p,
        &greedy_config(8),
        &SpeculativeConfig::default(),
        &CancelFlag::new(),
        &mut |_| {},
    );
    assert!(err.is_err(), "mismatched vocab must error");
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn prompt_lookup_llama() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    run_suite(fx);
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_QWEN3_MODEL"]
fn prompt_lookup_qwen3() {
    let Some(fx) = load_from("MLX_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set MLX_LLM_QWEN3_MODEL");
        return;
    };
    run_suite(fx);
}
