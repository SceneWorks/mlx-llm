//! Real-weights dynamic-batch tests (`#[ignore]` — needs a model on disk), story 7167.
//!
//! Point `MLX_LLM_TEST_MODEL` at a Hugging Face Llama-family snapshot and run:
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/path/to/SmolLM2-135M-Instruct \
//!   cargo test --test batch -- --ignored --nocapture
//! ```
//!
//! ## What "outputs identical to batch-1" means for a floating-point engine
//! The batched decode is **exactly** the single-sequence computation per row — left-padding +
//! per-row RoPE positions + a per-row additive mask reproduce each sequence's attention. We prove the
//! per-row math is right by running a request through the batched path at **batch size 1** (zero
//! padding) and asserting it is *token-for-token identical* to the single-sequence loop
//! ([`batch_of_one_matches_single`]).
//!
//! At **batch size > 1**, bit-exact identity to the batch-1 run is *not* attainable on MLX (or any
//! GPU backend): the batched matmul/attention kernels reduce in a different order than the unbatched
//! ones, so results differ at **sub-ULP** in bf16 — a known "batch-invariance" limitation, made
//! slightly larger for short rows by the masked left-pad keys. That is enough to flip an occasional
//! greedy *near-tie*, so a row may match batch-1 for many tokens and then diverge on one tie. It is
//! numerical, not a logic error: the batched decode is **deterministic** (same batch ⇒ same output,
//! [`batched_decode_is_reproducible`]), identical input rows produce identical outputs
//! ([`identical_rows_are_identical`]), and retirement/compaction is structurally exact
//! ([`retirement_and_compaction_respects_budgets`]). Bit-exact differing-length batching is what
//! paged attention (story 7169) buys — each sequence attends only its real keys, no padding mask.

use core_llm::Tokenizer;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{
    generate, generate_batch, generate_continuous, BatchExactness, BatchRequest, CancelFlag,
    ContinuousConfig, FinishReason, GenerationConfig,
};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::eos_token_ids;

struct Fixture {
    model: CausalLm,
    tok: Tokenizer,
    stop: Vec<i32>,
}

fn load() -> Option<Fixture> {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").ok()?;
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&Weights::from_dir(&dir).unwrap(), "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let stop = eos_token_ids(std::path::Path::new(&dir));
    Some(Fixture { model, tok, stop })
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, true).unwrap().into_iter().map(|id| id as i32).collect()
}

fn request(fx: &Fixture, prompt: &[i32], max_new: usize) -> BatchRequest {
    BatchRequest {
        prompt_ids: prompt.to_vec(),
        sampling: SamplingParams::default(), // greedy ⇒ deterministic
        seed: Some(0),
        max_new_tokens: max_new,
        stop_tokens: fx.stop.clone(),
    }
}

fn run_single(fx: &Fixture, prompt: &[i32], max_new: usize) -> (Vec<i32>, FinishReason) {
    let config = GenerationConfig {
        max_new_tokens: max_new,
        sampling: SamplingParams::default(),
        seed: Some(0),
        stop_tokens: fx.stop.clone(),
    };
    let out = generate(&fx.model, prompt, &config, &CancelFlag::new(), &mut |_| {}).unwrap();
    (out.tokens, out.finish_reason)
}

fn run_batch(fx: &Fixture, reqs: &[BatchRequest]) -> Vec<Vec<i32>> {
    generate_batch(&fx.model, reqs, &CancelFlag::new(), &mut |_, _| {})
        .unwrap()
        .into_iter()
        .map(|o| o.tokens)
        .collect()
}

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A request run through the batched path at batch size 1 (zero padding) is **bit-identical** to the
/// single-sequence loop — proof that the per-row masked-additive + per-row-position math is exact,
/// across differing prompt lengths.
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn batch_of_one_matches_single() {
    let Some(fx) = load() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    for text in [
        "Hi",
        "The capital of France is",
        "Once upon a time in a small village there lived a curious",
    ] {
        let p = encode(&fx.tok, text);
        let (single, single_fin) = run_single(&fx, &p, 24);
        let batched = generate_batch(&fx.model, &[request(&fx, &p, 24)], &CancelFlag::new(), &mut |ri, ev| {
            assert_eq!(ri, 0);
            let _ = ev;
        })
        .unwrap();
        assert!(!single.is_empty(), "'{text}' should generate");
        assert_eq!(batched[0].tokens, single, "batch-of-one must equal single for '{text}'");
        assert_eq!(batched[0].finish_reason, single_fin);
    }
}

/// Identical input rows in one batch produce identical outputs — adding rows does not corrupt a row,
/// and a row's output does not depend on its batch position.
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn identical_rows_are_identical() {
    let Some(fx) = load() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let p = encode(&fx.tok, "The capital of France is");
    let mut streamed: Vec<Vec<i32>> = vec![Vec::new(); 4];
    let batched = generate_batch(
        &fx.model,
        &[request(&fx, &p, 24), request(&fx, &p, 24), request(&fx, &p, 24), request(&fx, &p, 24)],
        &CancelFlag::new(),
        &mut |ri, ev| {
            if let mlx_llm::StreamEvent::Token { id, .. } = ev {
                streamed[ri].push(id);
            }
        },
    )
    .unwrap();
    for i in 1..batched.len() {
        assert_eq!(batched[i].tokens, batched[0].tokens, "row {i} must equal row 0 (identical inputs)");
    }
    for (i, out) in batched.iter().enumerate() {
        assert!(!out.tokens.is_empty());
        assert_eq!(streamed[i], out.tokens, "row {i} streamed ids must reconstruct the output");
    }
}

/// The whole batched run is deterministic: the same batch produces the same outputs every time.
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn batched_decode_is_reproducible() {
    let Some(fx) = load() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let reqs = vec![
        request(&fx, &encode(&fx.tok, "Hi"), 24),
        request(&fx, &encode(&fx.tok, "The capital of France is"), 24),
        request(&fx, &encode(&fx.tok, "Once upon a time in a small village there lived a curious"), 24),
    ];
    let a = run_batch(&fx, &reqs);
    let b = run_batch(&fx, &reqs);
    assert_eq!(a, b, "batched decode must be reproducible");
    assert!(a.iter().all(|r| !r.is_empty()));
}

/// N differing-length requests, two ways:
///   - through the **bit-exact** continuous path ([`generate_continuous`] / [`BatchExactness::Exact`])
///     each row is **token-for-token identical** to its batch-1 run — the equality assertion that
///     retires the 7167 sub-ULP left-pad caveat (per-sequence paged attention, no padding mask, no
///     batched matmul);
///   - through `generate_batch` (the synchronous left-pad path) each row only **tracks** batch-1,
///     agreeing on the prefill argmax but free to diverge at sub-ULP — the documented limitation the
///     exact path now supersedes.
///
/// (Full continuous-batching coverage — admit-on-retire, Throughput mode, cancel — lives in
/// `tests/continuous.rs`; this test pins the headline equality alongside the `generate_batch`
/// contrast.)
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn differing_lengths_complete_and_track_batch1() {
    let Some(fx) = load() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let prompts = [
        encode(&fx.tok, "Hi"),
        encode(&fx.tok, "The capital of France is"),
        encode(&fx.tok, "Once upon a time in a small village there lived a curious"),
    ];
    let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    println!("prompt token lengths: {lens:?}");

    let singles: Vec<(Vec<i32>, FinishReason)> =
        prompts.iter().map(|p| run_single(&fx, p, 24)).collect();
    let reqs: Vec<BatchRequest> = prompts.iter().map(|p| request(&fx, p, 24)).collect();

    // Bit-exact: the continuous Exact path must equal batch-1 token-for-token.
    let exact = generate_continuous(
        &fx.model,
        &reqs,
        &ContinuousConfig { max_batch: reqs.len(), block_size: 8, exactness: BatchExactness::Exact },
        &CancelFlag::new(),
        &mut |_, _| {},
    )
    .unwrap();

    // Tracking: generate_batch agrees on the prefill argmax, may diverge at sub-ULP afterward.
    let batched = generate_batch(&fx.model, &reqs, &CancelFlag::new(), &mut |_, _| {}).unwrap();

    for (i, p) in prompts.iter().enumerate() {
        let (single, single_fin) = &singles[i];
        assert!(!single.is_empty(), "row {i} should generate");

        assert_eq!(&exact[i].tokens, single, "row {i} (len {}) Exact must equal batch-1", p.len());
        assert_eq!(exact[i].finish_reason, *single_fin, "row {i} Exact finish reason");

        let got = &batched[i].tokens;
        let cp = common_prefix(got, single);
        let text = fx.tok.decode(&got.iter().map(|&x| x as u32).collect::<Vec<_>>(), true).unwrap();
        println!("row {i} (len {}): generate_batch common prefix {cp}/{} :: {text}", p.len(), single.len());
        assert!(!got.is_empty(), "row {i} produced no tokens");
        assert_eq!(batched[i].finish_reason, *single_fin, "row {i} generate_batch finish reason");
        assert!(cp >= 1, "row {i} must agree with batch-1 on at least the prefill argmax");
    }
}

/// Differing `max_new_tokens` retire sequences at different steps, exercising mid-batch retirement
/// and cache compaction. Retirement is structurally exact: a length-bounded row produces exactly its
/// budget, and rows track their batch-1 run.
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn retirement_and_compaction_respects_budgets() {
    let Some(fx) = load() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let prompts = [
        encode(&fx.tok, "Count: one, two,"),
        encode(&fx.tok, "List three colors:"),
        encode(&fx.tok, "The quick brown fox"),
    ];
    let budgets = [3usize, 8, 16]; // staggered ⇒ row 0 retires first, then row 1, then row 2

    let reqs: Vec<BatchRequest> = prompts.iter().zip(budgets).map(|(p, b)| request(&fx, p, b)).collect();
    let batched = generate_batch(&fx.model, &reqs, &CancelFlag::new(), &mut |_, _| {}).unwrap();

    for (i, p) in prompts.iter().enumerate() {
        let (single, single_fin) = run_single(&fx, p, budgets[i]);
        assert_eq!(batched[i].finish_reason, single_fin, "row {i} finish reason");
        if single_fin == FinishReason::MaxTokens {
            // Structural: a length-bounded row generates exactly its budget regardless of batching.
            assert_eq!(batched[i].tokens.len(), budgets[i], "row {i} should fill its budget");
        }
        let cp = common_prefix(&batched[i].tokens, &single);
        println!("row {i} (budget {}): common prefix {cp}/{}", budgets[i], single.len());
        assert!(cp >= 1, "row {i} must track its batch-1 run");
    }
}

/// A mid-stream cancel stops the whole batch promptly; rows still running finish `Cancelled` with
/// their partial output.
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn mid_stream_cancel_stops_the_batch() {
    let Some(fx) = load() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let prompts = [
        encode(&fx.tok, "Tell me a long story about a robot:"),
        encode(&fx.tok, "Describe a city at night in detail:"),
    ];
    let reqs: Vec<BatchRequest> = prompts.iter().map(|p| request(&fx, p, 200)).collect();

    let cancel = CancelFlag::new();
    let mut token_events = 0usize;
    let batched = generate_batch(&fx.model, &reqs, &cancel, &mut |_, ev| {
        if let mlx_llm::StreamEvent::Token { .. } = ev {
            token_events += 1;
            if token_events == 6 {
                cancel.cancel();
            }
        }
    })
    .unwrap();

    for out in &batched {
        assert_eq!(out.finish_reason, FinishReason::Cancelled, "cancelled rows finish Cancelled");
        assert!(out.tokens.len() < 200, "cancel should stop well before the budget");
    }
}
