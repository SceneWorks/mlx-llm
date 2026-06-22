//! Real-weights paged-KV-cache tests (`#[ignore]` — needs a model on disk), story 7169.
//!
//! Point `MLX_LLM_TEST_MODEL` (Llama-family) and/or `MLX_LLM_QWEN3_MODEL` (Qwen3) at a Hugging Face
//! snapshot and run:
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/SmolLM2-135M-Instruct \
//!   cargo test --test paged -- --ignored --nocapture
//! ```
//!
//! ## What these prove (the story-7169 acceptance, on real weights)
//! - **Outputs unchanged**: swapping the paged cache in for the contiguous one is token-for-token
//!   identical — gather returns the same per-position KV, in order, attended over by the same kernels.
//! - **Near-zero reservation waste**: the paged cache reserves `ceil(len/block_size)` blocks, never a
//!   `max_position_embeddings` slab — the ratio is printed and asserted.
//! - **Shared-prefix sequences share blocks**: two *divergent* requests sharing a system prefix point
//!   at the same physical blocks (copy-on-write), each still bit-exact vs its cold run, and the pooled
//!   reservation is far below two independent caches.

use core_llm::Tokenizer;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, generate_with_cache, CancelFlag, GenerationConfig};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::{BlockPool, PagedKvCache, Weights};

const BLOCK_SIZE: usize = 8;

struct Fixture {
    model: CausalLm,
    tok: Tokenizer,
    max_ctx: i32,
}

fn load_from(env: &str) -> Option<Fixture> {
    let dir = std::env::var(env).ok()?;
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let max_ctx = cfg.max_position_embeddings;
    let model = CausalLm::from_weights(&Weights::from_dir(&dir).unwrap(), "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    Some(Fixture { model, tok, max_ctx })
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, true).unwrap().into_iter().map(|id| id as i32).collect()
}

fn encode_no_bos(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, false).unwrap().into_iter().map(|id| id as i32).collect()
}

fn config(max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(), // greedy ⇒ deterministic
        seed: Some(0),
        stop_tokens: Vec::new(), // run the full budget (model-agnostic; bit-exactness is the point)
    }
}

fn cold(fx: &Fixture, prompt: &[i32], max_new: usize) -> Vec<i32> {
    generate(&fx.model, prompt, &config(max_new), &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .tokens
}

fn run_suite(fx: Fixture) {
    let max_new = 24;

    // ---- Outputs unchanged: paged cache is a token-for-token drop-in ----
    let prompt = encode(&fx.tok, "The capital of France is");
    let base = cold(&fx, &prompt, max_new);
    assert!(!base.is_empty());

    let mut paged = fx.model.new_paged_cache(BLOCK_SIZE);
    let out = generate_with_cache(&fx.model, &prompt, &mut paged, &config(max_new), &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .tokens;
    assert_eq!(out, base, "paged cache must be token-for-token identical to the contiguous cache");

    // ---- Near-zero reservation waste vs naive max-context ----
    let len = prompt.len() + out.len();
    let reserved = paged.reserved_tokens();
    let naive = fx.max_ctx.max(2048) as usize; // a naive cache reserves a full max_position slab/seq
    assert!(reserved <= len + BLOCK_SIZE, "paged reserves ~len, not a fixed max");
    assert!(reserved * 8 < naive, "paged reservation must be a small fraction of naive max-context");
    println!(
        "reservation: paged {reserved} tokens for a {len}-token sequence vs naive {naive} \
         ({:.1}x less)",
        naive as f64 / reserved as f64
    );

    // ---- Shared-prefix sequences share blocks (copy-on-write), still bit-exact ----
    // A long system preamble so the shared prefix spans several whole blocks.
    let sys = encode(
        &fx.tok,
        "You are a helpful, knowledgeable, and meticulous assistant. Always answer accurately and \
         concisely, reason step by step when a question requires it, cite concrete facts, and avoid \
         unnecessary repetition, filler, or hedging in your responses.\n\n",
    );
    let q1 = encode_no_bos(&fx.tok, "What is the capital of France?");
    let q2 = encode_no_bos(&fx.tok, "Name three primary colors.");
    assert_ne!(q1.first(), q2.first());
    let p1: Vec<i32> = sys.iter().chain(&q1).copied().collect();
    let p2: Vec<i32> = sys.iter().chain(&q2).copied().collect();

    let cold1 = cold(&fx, &p1, max_new);
    let cold2 = cold(&fx, &p2, max_new);

    let pool = BlockPool::new(BLOCK_SIZE);
    // Sequence 1 (cold) populates the shared system-prefix blocks.
    let mut c1 = PagedKvCache::with_pool(pool.clone(), fx.model.config().num_layers);
    let out1 = generate_with_cache(&fx.model, &p1, &mut c1, &config(max_new), &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .tokens;
    assert_eq!(out1, cold1, "paged seq 1 must match its cold run");

    // Sequence 2 adopts seq 1's whole system-prefix blocks (no recompute, no copy).
    let shared = c1.shareable_prefix_blocks(sys.len()).unwrap();
    let shared_tokens = shared.len() * BLOCK_SIZE;
    assert!(!shared.is_empty(), "the system prefix should span at least one block");
    let mut c2 = PagedKvCache::new_seeded(pool.clone(), fx.model.config().num_layers, &shared);
    let out2 = generate_with_cache(&fx.model, &p2, &mut c2, &config(max_new), &CancelFlag::new(), &mut |_| {})
        .unwrap()
        .tokens;
    assert_eq!(out2, cold2, "paged seq 2 (sharing seq 1's prefix blocks) must match its cold run");

    // The shared blocks are physically shared (refcount > 1) and counted once: the pool holds the
    // union of both sequences' blocks minus the de-duplicated shared prefix.
    {
        let p = pool.borrow();
        assert_eq!(p.shared_blocks(), shared.len(), "the whole system prefix is shared");
        assert_eq!(
            p.live_blocks(),
            c1.blocks() + c2.blocks() - shared.len(),
            "shared prefix blocks are referenced, not duplicated"
        );
        println!(
            "sharing: {} blocks ({} tokens) shared between the two requests; pool holds {} live \
             blocks (vs {} if independent)",
            shared.len(),
            shared_tokens,
            p.live_blocks(),
            c1.blocks() + c2.blocks(),
        );
        assert!(p.live_blocks() < c1.blocks() + c2.blocks(), "sharing reduces the block count");
    }

    drop(c2);
    assert_eq!(pool.borrow().shared_blocks(), 0, "dropping seq 2 releases the shared references");
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn paged_cache_llama() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    run_suite(fx);
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_QWEN3_MODEL"]
fn paged_cache_qwen3() {
    let Some(fx) = load_from("MLX_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set MLX_LLM_QWEN3_MODEL");
        return;
    };
    run_suite(fx);
}
