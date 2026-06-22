//! Real-weights iteration-level continuous-batching tests (`#[ignore]` — needs a model), story 7281.
//!
//! Point `MLX_LLM_TEST_MODEL` (Llama-family) and/or `MLX_LLM_QWEN3_MODEL` (Qwen3) at a Hugging Face
//! snapshot and run:
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/tmp/smollm2-135m MLX_LLM_QWEN3_MODEL=/tmp/qwen3-0.6b \
//!   cargo test --test continuous -- --ignored --nocapture
//! ```
//!
//! ## What these prove (the story-7281 acceptance, on real weights)
//! - **Bit-exact to batch-1** ([`BatchExactness::Exact`]): a differing-length batch is **token-for-
//!   token identical** to each row's batch-1 run — the equality assertion that retires the 7167
//!   sub-ULP left-pad caveat. Each sequence is its own batch-1 forward over its own paged cache, so
//!   there is no batched matmul and no padding mask to perturb the result.
//! - **Admit-on-retire**: with more requests than `max_batch`, a retiring sequence's slot is
//!   immediately refilled; every request — including ones admitted mid-flight — still equals its
//!   batch-1 run.
//! - **Throughput mode tracks batch-1** ([`BatchExactness::Throughput`]): the batched-projection path
//!   completes coherently and tracks each row's batch-1 run (diverging only at the documented
//!   sub-ULP, like `generate_batch`). Its throughput scaling is measured in `tests/throughput`.

use core_llm::Tokenizer;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{
    generate, generate_continuous, BatchExactness, BatchRequest, CancelFlag, ContinuousConfig,
    FinishReason, GenerationConfig, GenerationOutput,
};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::eos_token_ids;

const BLOCK: usize = 8;

struct Fixture {
    model: CausalLm,
    tok: Tokenizer,
    stop: Vec<i32>,
}

fn load_from(env: &str) -> Option<Fixture> {
    let dir = std::env::var(env).ok()?;
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

fn cfg(max_batch: usize, exactness: BatchExactness) -> ContinuousConfig {
    ContinuousConfig { max_batch, block_size: BLOCK, exactness }
}

fn run_continuous(fx: &Fixture, reqs: &[BatchRequest], config: &ContinuousConfig) -> Vec<GenerationOutput> {
    generate_continuous(&fx.model, reqs, config, &CancelFlag::new(), &mut |_, _| {}).unwrap()
}

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The headline acceptance: a differing-length batch run through the **Exact** continuous path is
/// **token-for-token identical** to each row's batch-1 run — the equality assertion that retires the
/// 7167 sub-ULP left-pad caveat. Run at `max_batch >= n` (all admitted at once).
fn exact_equals_batch1(fx: &Fixture) {
    let prompts = [
        encode(&fx.tok, "Hi"),
        encode(&fx.tok, "The capital of France is"),
        encode(&fx.tok, "Once upon a time in a small village there lived a curious"),
        // A long prompt (25+ tokens) so prefill spans several whole paged blocks (BLOCK = 8),
        // exercising freeze-on-full + multi-block gather inside the continuous path.
        encode(
            &fx.tok,
            "In the year twenty fifty, a team of meticulous engineers gathered in a quiet \
             laboratory to carefully design a new kind of machine that could reason about",
        ),
    ];
    let reqs: Vec<BatchRequest> = prompts.iter().map(|p| request(fx, p, 24)).collect();
    let batched = run_continuous(fx, &reqs, &cfg(prompts.len(), BatchExactness::Exact));

    for (i, p) in prompts.iter().enumerate() {
        let (single, single_fin) = run_single(fx, p, 24);
        assert!(!single.is_empty(), "row {i} should generate");
        assert_eq!(
            batched[i].tokens, single,
            "row {i} (len {}) must be BIT-EXACT to its batch-1 run under Exact mode",
            p.len()
        );
        assert_eq!(batched[i].finish_reason, single_fin, "row {i} finish reason");
    }
}

/// Admit-on-retire: with more requests than `max_batch`, retiring sequences free slots that the
/// waiting requests fill mid-flight. Every request — including ones admitted only after others
/// retired — is still bit-exact to its batch-1 run, and they all complete.
fn admit_on_retire_equals_batch1(fx: &Fixture) {
    // Staggered budgets so the first slots retire at different steps, forcing several mid-flight
    // admissions through max_batch = 2 over 5 requests.
    let specs: [(&str, usize); 5] = [
        ("Hi", 4),
        ("The capital of France is", 8),
        ("Count: one, two,", 6),
        ("List three colors:", 10),
        ("The quick brown fox", 12),
    ];
    let prompts: Vec<Vec<i32>> = specs.iter().map(|(t, _)| encode(&fx.tok, t)).collect();
    let reqs: Vec<BatchRequest> =
        prompts.iter().zip(specs).map(|(p, (_, b))| request(fx, p, b)).collect();

    let batched = run_continuous(fx, &reqs, &cfg(2, BatchExactness::Exact));
    assert_eq!(batched.len(), specs.len());

    for (i, p) in prompts.iter().enumerate() {
        let (single, single_fin) = run_single(fx, p, specs[i].1);
        assert!(!single.is_empty(), "row {i} should generate");
        assert_eq!(
            batched[i].tokens, single,
            "row {i} (admitted into a freed slot) must be bit-exact to its batch-1 run"
        );
        assert_eq!(batched[i].finish_reason, single_fin, "row {i} finish reason");
    }
}

/// Throughput mode (batched projections + per-sequence attention) completes coherently and **tracks**
/// each row's batch-1 run — agreeing on at least the prefill argmax, with later tokens free to differ
/// only by the documented sub-ULP batched-matmul rounding. The common prefix is printed.
fn throughput_tracks_batch1(fx: &Fixture) {
    let prompts = [
        encode(&fx.tok, "Hi"),
        encode(&fx.tok, "The capital of France is"),
        encode(&fx.tok, "Once upon a time in a small village there lived a curious"),
    ];
    let reqs: Vec<BatchRequest> = prompts.iter().map(|p| request(fx, p, 24)).collect();
    let batched = run_continuous(fx, &reqs, &cfg(prompts.len(), BatchExactness::Throughput));

    for (i, p) in prompts.iter().enumerate() {
        let (single, single_fin) = run_single(fx, p, 24);
        let got = &batched[i].tokens;
        let cp = common_prefix(got, &single);
        let text = fx.tok.decode(&got.iter().map(|&x| x as u32).collect::<Vec<_>>(), true).unwrap();
        println!("throughput row {i} (len {}): common prefix {cp}/{} :: {text}", p.len(), single.len());
        assert!(!got.is_empty(), "row {i} produced no tokens");
        assert!(!text.trim().is_empty(), "row {i} should decode to text");
        assert_eq!(batched[i].finish_reason, single_fin, "row {i} finish reason");
        assert!(cp >= 1, "row {i} must track batch-1 on at least the prefill argmax");
    }
}

fn run_suite(fx: Fixture) {
    exact_equals_batch1(&fx);
    admit_on_retire_equals_batch1(&fx);
    throughput_tracks_batch1(&fx);
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn continuous_llama() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    run_suite(fx);
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_QWEN3_MODEL"]
fn continuous_qwen3() {
    let Some(fx) = load_from("MLX_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set MLX_LLM_QWEN3_MODEL");
        return;
    };
    run_suite(fx);
}

/// Acceptance #2 — throughput scales with batch occupancy. Aggregate decode tok/s (total generated
/// tokens / wall time) for N identical fixed-budget requests at `max_batch = N`: **Throughput** mode
/// rises with N (the projections/MLP/lm_head are batched, amortizing weight reads), while **Exact**
/// mode stays flat (per-sequence forwards serialize on MLX). The table is printed; the assertion is
/// loose so hardware/thermal variance doesn't make it flaky (measured ~2.6x at N=8 on M-series).
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL; perf"]
fn throughput_scales_with_occupancy() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let prompt = encode(&fx.tok, "Count slowly and describe the number:");
    let budget = 48usize;
    let agg = |n: usize, mode: BatchExactness| -> f64 {
        let mut r = request(&fx, &prompt, budget);
        r.stop_tokens = Vec::new(); // run the full budget so the token count is fixed
        let reqs: Vec<BatchRequest> = (0..n).map(|_| r.clone()).collect();
        let config = ContinuousConfig { max_batch: n, block_size: BLOCK, exactness: mode };
        let _ = run_continuous(&fx, &reqs, &config); // warm
        let t = std::time::Instant::now();
        let out = run_continuous(&fx, &reqs, &config);
        let total: usize = out.iter().map(|o| o.tokens.len()).sum();
        total as f64 / t.elapsed().as_secs_f64()
    };

    println!("occupancy | Exact tok/s | Throughput tok/s | speedup");
    let mut thru_by_n = Vec::new();
    for &n in &[1usize, 2, 4, 8] {
        let (e, tp) = (agg(n, BatchExactness::Exact), agg(n, BatchExactness::Throughput));
        println!("{n:>9} | {e:>11.1} | {tp:>16.1} | {:>5.2}x", tp / e);
        thru_by_n.push((n, e, tp));
    }
    let (_, exact8, thru8) = *thru_by_n.last().unwrap();
    let (_, _, thru1) = thru_by_n[0];
    assert!(thru8 > thru1 * 1.5, "Throughput@8 ({thru8:.0}) should scale well past Throughput@1 ({thru1:.0})");
    assert!(thru8 > exact8 * 1.3, "Throughput@8 ({thru8:.0}) should beat Exact@8 ({exact8:.0}) via batched matmul");
}

/// A mid-stream cancel stops the whole continuous run promptly; every request finishes `Cancelled`
/// with its partial output.
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn cancel_stops_continuous() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let prompts = [
        encode(&fx.tok, "Tell me a long story about a robot:"),
        encode(&fx.tok, "Describe a city at night in detail:"),
        encode(&fx.tok, "Explain how a rainbow forms:"),
    ];
    let reqs: Vec<BatchRequest> = prompts.iter().map(|p| request(&fx, p, 200)).collect();

    let cancel = CancelFlag::new();
    let mut token_events = 0usize;
    let batched = generate_continuous(
        &fx.model,
        &reqs,
        &cfg(3, BatchExactness::Exact),
        &cancel,
        &mut |_, ev| {
            if let mlx_llm::StreamEvent::Token { .. } = ev {
                token_events += 1;
                if token_events == 6 {
                    cancel.cancel();
                }
            }
        },
    )
    .unwrap();

    for out in &batched {
        assert_eq!(out.finish_reason, FinishReason::Cancelled, "cancelled rows finish Cancelled");
        assert!(out.tokens.len() < 200, "cancel should stop well before the budget");
    }
}

/// The streaming contract under cancel with a **queue**: with `max_batch < n`, some requests are
/// still waiting (never prefilled) when the cancel hits. Every request — decoding *or* queued — must
/// emit exactly one terminal `Done` and finish `Cancelled` (a consumer waiting on `Done` per request
/// would otherwise hang on the never-admitted ones).
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn cancel_emits_one_done_per_request() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let texts = [
        "Tell me a long story about a robot:",
        "Describe a city at night in detail:",
        "Explain how a rainbow forms:",
        "Write a poem about the ocean:",
        "List the planets and describe each:",
    ];
    // Empty stop tokens + a large budget so nothing retires normally before the cancel ⇒ every Done
    // must be Cancelled.
    let reqs: Vec<BatchRequest> = texts
        .iter()
        .map(|t| BatchRequest {
            prompt_ids: encode(&fx.tok, t),
            sampling: SamplingParams::default(),
            seed: Some(0),
            max_new_tokens: 200,
            stop_tokens: Vec::new(),
        })
        .collect();

    let cancel = CancelFlag::new();
    let mut done_count = vec![0usize; texts.len()];
    let mut token_events = 0usize;
    let out = generate_continuous(
        &fx.model,
        &reqs,
        &cfg(2, BatchExactness::Exact), // max_batch 2 < 5 ⇒ requests 2..5 are queued at cancel time
        &cancel,
        &mut |ri, ev| match ev {
            mlx_llm::StreamEvent::Token { .. } => {
                token_events += 1;
                if token_events == 4 {
                    cancel.cancel();
                }
            }
            mlx_llm::StreamEvent::Done { reason, .. } => {
                done_count[ri] += 1;
                assert_eq!(reason, FinishReason::Cancelled, "request {ri} Done must be Cancelled");
            }
        },
    )
    .unwrap();

    for (i, c) in done_count.iter().enumerate() {
        assert_eq!(*c, 1, "request {i} must emit exactly one Done (incl. never-admitted queued rows)");
    }
    for (i, o) in out.iter().enumerate() {
        assert_eq!(o.finish_reason, FinishReason::Cancelled, "request {i} output Cancelled");
    }
}

/// A zero-budget request (`max_new_tokens == 0`) returns empty with `MaxTokens` and emits exactly one
/// `Done`, while its batch-mate generates normally — the zero-budget path through the continuous loop.
#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL"]
fn zero_budget_request_completes_empty() {
    let Some(fx) = load_from("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let reqs = vec![
        request(&fx, &encode(&fx.tok, "Hi"), 0), // zero budget
        request(&fx, &encode(&fx.tok, "The capital of France is"), 8),
    ];
    let mut done_count = [0usize; 2];
    let out = generate_continuous(
        &fx.model,
        &reqs,
        &cfg(2, BatchExactness::Exact),
        &CancelFlag::new(),
        &mut |ri, ev| {
            if let mlx_llm::StreamEvent::Done { .. } = ev {
                done_count[ri] += 1;
            }
        },
    )
    .unwrap();

    assert!(out[0].tokens.is_empty(), "zero-budget request generates nothing");
    assert_eq!(out[0].finish_reason, FinishReason::MaxTokens);
    assert_eq!(done_count[0], 1, "zero-budget request emits exactly one Done");
    assert!(!out[1].tokens.is_empty(), "the budgeted request still generates");
    assert_eq!(done_count[1], 1);
}
