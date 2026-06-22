//! Iteration-level continuous batching (epic 7153, story 7281).
//!
//! [`generate_batch`](crate::decode::generate_batch) (story 7167) is *synchronous*: a batch is
//! assembled, decoded in lockstep to completion, and only then does the next batch start.
//! [`generate_continuous`] is *iteration-level*: it keeps up to `max_batch` sequences decoding at
//! once over **per-sequence** [`PagedKvCache`]s on one shared [`BlockPool`], and the moment a
//! sequence retires it prefills a waiting request into the freed slot — the batch never drains.
//! The host-side admission / retirement policy is the backend-neutral [`core_llm::Scheduler`]; this
//! module owns only the MLX tensors.
//!
//! ## Two modes — an irreducible tradeoff (see [`BatchExactness`])
//! On MLX (and any GPU backend) bit-exactness to a batch-1 run and throughput-scaling-with-occupancy
//! are **mutually exclusive**, because the throughput win of batching *is* the restructured matmul
//! reduction (amortizing the weight reads across the batch), which is exactly what perturbs the
//! floating-point result. Measured on this engine: MLX's bf16 matmul is not row-invariant (a batched
//! projection's row diverges from its batch-1 value by up to ~1 bf16 ULP on the logits — enough to
//! flip a greedy near-tie), and conversely per-sequence forwards do not overlap on the device (decode
//! tok/s is flat across occupancy). So:
//! - [`BatchExactness::Exact`] runs each sequence as its own batch-1 forward: **byte-identical** to
//!   running that request alone, but no throughput scaling.
//! - [`BatchExactness::Throughput`] batches the projections / MLP / lm_head and runs only attention
//!   per-sequence ([`CausalLm::decode_logits_per_seq`]): throughput scales with occupancy, at the
//!   cost of sub-ULP divergence (a row *tracks* its batch-1 run, like `generate_batch`).
//!
//! Both modes get iteration-level admission (admit-on-retire) and per-sequence paged attention (no
//! padding mask, no max-context reservation). The bit-exact equality assertion in `tests/batch.rs`
//! runs against `Exact`.

use std::cell::RefCell;
use std::rc::Rc;

use mlx_rs::transforms::eval;
use mlx_rs::Array;

use core_llm::schedule::{Scheduler, SeqId, SeqSpec};
use core_llm::FinishReason as CoreFinish;

use crate::decode::batch::BatchRequest;
use crate::decode::cancel::CancelFlag;
use crate::decode::stream::{default_seed, FinishReason, GenerationOutput, StreamEvent};
use crate::error::{Error, Result};
use crate::models::CausalLm;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::input_ids;
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};
use crate::primitives::{BlockPool, PagedKvCache};

/// Numerical mode for [`generate_continuous`]'s decode forward — a throughput/exactness tradeoff
/// that is fundamental, not an implementation choice (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchExactness {
    /// Each sequence is decoded as its own batch-1 forward, so a row's output is **byte-identical**
    /// to running that request alone. Throughput does **not** scale with occupancy (MLX serializes
    /// the independent per-sequence forwards). The default, and the mode the bit-exact equality
    /// assertion runs against.
    #[default]
    Exact,
    /// The projections, MLP, and lm_head are **batched** over the active sequences and only attention
    /// runs per-sequence. Throughput scales with occupancy (weight reads amortized across the batch),
    /// but a row only **tracks** its batch-1 run — MLX's batched matmul is not row-invariant, so the
    /// logits diverge at sub-ULP (the same class as the `generate_batch` caveat).
    Throughput,
}

/// Configuration for [`generate_continuous`].
#[derive(Clone, Debug)]
pub struct ContinuousConfig {
    /// Maximum number of sequences decoding concurrently. Requests beyond this wait in admission
    /// order and fill a slot the instant one retires.
    pub max_batch: usize,
    /// Tokens per block in the shared paged-KV [`BlockPool`].
    pub block_size: usize,
    /// Numerical mode (see [`BatchExactness`]).
    pub exactness: BatchExactness,
}

impl Default for ContinuousConfig {
    fn default() -> Self {
        Self {
            max_batch: 8,
            block_size: 16,
            exactness: BatchExactness::Exact,
        }
    }
}

/// Per-sequence host state for one in-flight slot, alongside its own paged cache.
struct Lane {
    seq: SeqId,
    /// Request index == admission index == `seq.0`; kept explicit for the `on_event` callback.
    req_index: usize,
    cache: PagedKvCache,
    rng: SplitMix64,
    params: SamplingParams,
    /// Prompt + generated tokens (the repetition-penalty window the sampler reads).
    history: Vec<i32>,
    /// The token to feed at the next decode step (the most recently sampled token).
    next_token: i32,
}

/// Whether a lane keeps decoding after a token, or has retired.
enum LaneStep {
    Continue,
    Done,
}

/// Generate for many requests with **iteration-level continuous batching**: up to
/// `config.max_batch` sequences decode at once over per-sequence paged caches, and a retiring
/// sequence's slot is immediately refilled from the waiting requests. Returns a
/// [`GenerationOutput`] per request, in request order. `on_event` receives `(request_index, event)`
/// as each row streams.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference. A mid-stream cancel
/// stops promptly: every request still decoding **or still waiting in the queue** finishes
/// [`FinishReason::Cancelled`] with whatever partial output it had, and each request emits exactly
/// one terminal [`StreamEvent::Done`].
pub fn generate_continuous(
    model: &CausalLm,
    requests: &[BatchRequest],
    config: &ContinuousConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<Vec<GenerationOutput>> {
    if requests.is_empty() {
        return Err(Error::Msg("generate_continuous: no requests".into()));
    }
    if config.max_batch == 0 {
        return Err(Error::Msg("generate_continuous: max_batch must be > 0".into()));
    }
    if config.block_size == 0 {
        return Err(Error::Msg("generate_continuous: block_size must be > 0".into()));
    }
    for (i, r) in requests.iter().enumerate() {
        if r.prompt_ids.is_empty() {
            return Err(Error::Msg(format!("generate_continuous: request {i} has an empty prompt")));
        }
    }
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }

    let pool = BlockPool::new(config.block_size);
    let num_layers = model.config().num_layers;

    // Admit every request to the scheduler up front (stable SeqId == request index); only `max_batch`
    // are prefilled into a live lane at a time, the rest wait at `next_req`.
    let mut sched = Scheduler::new();
    let seq_ids: Vec<SeqId> = requests
        .iter()
        .map(|r| sched.admit(SeqSpec::new(r.prompt_ids.clone(), r.max_new_tokens, r.stop_tokens.clone())))
        .collect();

    let mut lanes: Vec<Lane> = Vec::new();
    let mut next_req = 0usize;

    // Fill the initial slots (prefill is per-sequence: each prompt at its own length, no left-pad).
    while lanes.len() < config.max_batch && next_req < requests.len() {
        if let Some(lane) = admit_lane(model, &pool, num_layers, requests, &seq_ids, next_req, &mut sched, on_event)? {
            lanes.push(lane);
        }
        next_req += 1;
    }

    // Decode loop: step every live lane, retire finished ones, refill freed slots from the queue.
    // Cancel is checked once at the top of each step (before another forward), so a mid-stream cancel
    // stops promptly without abandoning a forward that has already been computed.
    while !lanes.is_empty() {
        if cancel.is_cancelled() {
            break;
        }

        let per_lane = step_logits(model, &mut lanes, config.exactness)?;

        let mut survivors: Vec<Lane> = Vec::with_capacity(lanes.len());
        for (mut lane, logits) in std::mem::take(&mut lanes).into_iter().zip(per_lane) {
            let tok = sample(&logits, &lane.history, &lane.params, &mut lane.rng, None)?;
            if let LaneStep::Continue = record_token(&mut sched, &mut lane, tok, on_event) {
                survivors.push(lane);
            }
        }
        lanes = survivors;

        // Admit-on-retire: refill every freed slot from the waiting requests.
        while lanes.len() < config.max_batch && next_req < requests.len() {
            if let Some(lane) = admit_lane(model, &pool, num_layers, requests, &seq_ids, next_req, &mut sched, on_event)? {
                lanes.push(lane);
            }
            next_req += 1;
        }
    }

    // A cancel breaks the loop with live lanes still decoding and requests still waiting in the
    // queue; neither has emitted a `Done` yet (a request that retired normally emitted its `Done` as
    // it finished). Emit exactly one `Done` for each so every request signals completion. The final
    // tokens / reason are already recorded in `sched`; these two ranges are disjoint from the
    // already-retired requests and from each other (lane indices are `< next_req`, the queue is
    // `next_req..`), so no request is signalled twice.
    if cancel.is_cancelled() {
        for lane in &lanes {
            on_event(lane.req_index, StreamEvent::Done {
                reason: FinishReason::Cancelled,
                generated: sched.generated(lane.seq).len(),
            });
        }
        for (ri, &seq) in seq_ids.iter().enumerate().skip(next_req) {
            // Never admitted, so never generated: Cancelled — unless it was a zero-budget request the
            // scheduler retired at admission, whose own `MaxTokens` (empty output) we keep.
            let reason = match sched.finish_reason(seq) {
                Some(CoreFinish::Length) => FinishReason::MaxTokens,
                _ => FinishReason::Cancelled,
            };
            on_event(ri, StreamEvent::Done { reason, generated: 0 });
        }
    }

    // Assemble per-request outputs in request order from the scheduler's record.
    Ok(seq_ids
        .iter()
        .map(|&seq| GenerationOutput {
            tokens: sched.generated(seq).to_vec(),
            finish_reason: match sched.finish_reason(seq) {
                Some(CoreFinish::Stop) => FinishReason::StopToken,
                Some(CoreFinish::Length) => FinishReason::MaxTokens,
                // `None` ⇒ still active when a cancel broke the loop (or never admitted).
                _ => FinishReason::Cancelled,
            },
        })
        .collect())
}

/// Prefill request `ri` into a fresh lane and sample its first token. Returns `None` if the request
/// retired immediately (zero budget, or a stop/length on the first token), in which case the
/// scheduler already holds its final state.
#[allow(clippy::too_many_arguments)]
fn admit_lane(
    model: &CausalLm,
    pool: &Rc<RefCell<BlockPool>>,
    num_layers: usize,
    requests: &[BatchRequest],
    seq_ids: &[SeqId],
    ri: usize,
    sched: &mut Scheduler,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<Option<Lane>> {
    let r = &requests[ri];
    let seq = seq_ids[ri];
    if !sched.is_active(seq) {
        // The scheduler retires a zero-budget request (`max_new_tokens == 0`) at admission, so it is
        // already inactive here: emit its terminal event and skip the prefill entirely.
        on_event(ri, StreamEvent::Done { reason: FinishReason::MaxTokens, generated: 0 });
        return Ok(None);
    }

    let mut cache = PagedKvCache::with_pool(pool.clone(), num_layers);
    let logits = model.decode_logits(&input_ids(&r.prompt_ids), &mut cache, 0)?; // batch-1 prefill
    let mut lane = Lane {
        seq,
        req_index: ri,
        cache,
        rng: SplitMix64::new(r.seed.unwrap_or_else(default_seed)),
        params: r.sampling,
        history: r.prompt_ids.clone(),
        next_token: 0,
    };
    let tok = sample(&logits, &lane.history, &lane.params, &mut lane.rng, None)?;
    Ok(match record_token(sched, &mut lane, tok, on_event) {
        LaneStep::Continue => Some(lane),
        LaneStep::Done => None,
    })
}

/// One decode step's logits, one `[1, vocab]` per live lane (in lane order). `Exact` builds each
/// sequence's own batch-1 forward and evaluates them together (a single device round-trip for the
/// step); `Throughput` runs one batched forward with per-sequence attention and splits the rows.
fn step_logits(model: &CausalLm, lanes: &mut [Lane], exactness: BatchExactness) -> Result<Vec<Array>> {
    match exactness {
        BatchExactness::Exact => {
            let mut logits = Vec::with_capacity(lanes.len());
            for lane in lanes.iter_mut() {
                let off = lane.cache.offset();
                logits.push(model.decode_logits(&input_ids(&[lane.next_token]), &mut lane.cache, off)?);
            }
            eval(logits.iter())?; // force the whole step in one go
            Ok(logits)
        }
        BatchExactness::Throughput => {
            let b = lanes.len();
            let feed: Vec<i32> = lanes.iter().map(|l| l.next_token).collect();
            let ids = Array::from_slice(&feed, &[b as i32, 1]);
            let positions: Vec<i32> = lanes.iter().map(|l| l.cache.offset()).collect();
            let mut caches: Vec<&mut PagedKvCache> = lanes.iter_mut().map(|l| &mut l.cache).collect();
            let logits = model.decode_logits_per_seq(&ids, &mut caches, &positions)?; // [b, vocab]
            (0..b as i32)
                .map(|i| {
                    let idx = Array::from_slice(&[i], &[1]);
                    Ok(logits.take_axis(&idx, 0)?) // [1, vocab]
                })
                .collect()
        }
    }
}

/// Record `tok` for `lane` through the scheduler and emit its stream events, identically to the
/// single-sequence loop: a stop token retires the lane (excluded, no `Token` event); otherwise the
/// token is emitted + kept, and a filled budget retires the lane after including it.
fn record_token(
    sched: &mut Scheduler,
    lane: &mut Lane,
    tok: i32,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> LaneStep {
    let ri = lane.req_index;
    match sched.record(lane.seq, tok) {
        Some(CoreFinish::Stop) => {
            on_event(ri, StreamEvent::Done {
                reason: FinishReason::StopToken,
                generated: sched.generated(lane.seq).len(),
            });
            LaneStep::Done
        }
        other => {
            let step = sched.generated(lane.seq).len() - 1;
            on_event(ri, StreamEvent::Token { id: tok, step });
            lane.history.push(tok);
            match other {
                Some(CoreFinish::Length) => {
                    on_event(ri, StreamEvent::Done {
                        reason: FinishReason::MaxTokens,
                        generated: sched.generated(lane.seq).len(),
                    });
                    LaneStep::Done
                }
                _ => {
                    lane.next_token = tok;
                    LaneStep::Continue
                }
            }
        }
    }
}
