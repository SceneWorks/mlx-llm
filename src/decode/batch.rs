//! Synchronous dynamic batching (epic 7153, story 7167).
//!
//! Concurrency on an MLX device lives in the **batch dimension**, not in threads (the default Metal
//! device is single-threaded). [`generate_batch`] packs N requests into one forward step so a batch
//! of differing-length prompts completes under a single eval thread with throughput that scales with
//! batch occupancy — while each row's output is **identical to running that request alone**.
//!
//! The host-side policy (admission, per-sequence stop/length retirement) is the backend-neutral
//! [`core_llm::Scheduler`]; this module owns only the MLX tensors that policy drives. The batch is
//! assembled, **left-padded** to a common width, prefilled together, and decoded in lockstep;
//! finished sequences are retired and the cache compacted ([`KvCache::retain_sequences`]) so the next
//! step runs a smaller batch.
//!
//! ## Why outputs match batch-1
//! Left-padding right-aligns every sequence's real tokens, and each row carries its **own** absolute
//! RoPE positions (`prompt token i → position i`) plus an additive mask that blocks the left-pad
//! region. RoPE attention scores depend only on the *relative* offset between query and key, so
//! shifting a sequence within the padded width leaves its intra-sequence attention unchanged, and the
//! masked pad keys never contribute — the per-row logits equal the batch-1 logits, hence identical
//! sampling.
//!
//! ## Scope
//! This is *synchronous* dynamic batching: a batch decodes in lockstep to completion, with
//! retirement. *Iteration-level* continuous batching — admitting a new request into a running batch
//! mid-flight, which needs ragged query lengths / chunked prefill — rides on the paged cache (story
//! 7169); the [`core_llm::Scheduler`] already carries per-sequence offsets so that lands without
//! reworking the policy.

use mlx_rs::Array;

use core_llm::schedule::{Scheduler, SeqId, SeqSpec};
use core_llm::FinishReason as CoreFinish;

use crate::decode::cancel::CancelFlag;
use crate::decode::stream::{default_seed, Decode, FinishReason, GenerationOutput, StreamEvent};
use crate::error::{Error, Result};
use crate::models::CausalLm;
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};

/// The pad token id stuffed into the left-pad region. Any in-vocabulary id works — the attention
/// mask blocks these positions and their outputs are never read — so `0` is a safe choice.
const PAD_ID: i32 = 0;

/// One request in a batch. Mirrors the per-request knobs of a single
/// [`GenerationConfig`](crate::decode::GenerationConfig); each sequence carries its own sampling,
/// seed, budget, and stop tokens.
#[derive(Clone, Debug)]
pub struct BatchRequest {
    /// Prompt token ids (already rendered + tokenized). Must be non-empty.
    pub prompt_ids: Vec<i32>,
    /// Sampling knobs for this sequence.
    pub sampling: SamplingParams,
    /// RNG seed; `None` ⇒ a fresh per-call seed. An explicit seed reproduces the batch-1 run.
    pub seed: Option<u64>,
    /// Maximum new tokens for this sequence.
    pub max_new_tokens: usize,
    /// Token ids that stop this sequence when sampled (excluded from its output).
    pub stop_tokens: Vec<i32>,
}

/// Per-sequence host state living alongside its row in the batch.
struct Lane {
    seq: SeqId,
    /// Left-pad width for this row (constant for the sequence's lifetime).
    pad_len: i32,
    rng: SplitMix64,
    params: SamplingParams,
    /// Prompt + generated tokens — the repetition-penalty window the sampler reads.
    history: Vec<i32>,
    /// The most recently sampled token, awaiting its feed into the next decode step.
    next_token: i32,
}

/// Whether a lane keeps decoding after a token, or has retired.
enum LaneStep {
    Continue,
    Done,
}

/// Generate for a batch of requests in one continuous-batched run, returning a
/// [`GenerationOutput`] per request (in request order). `on_event` receives `(request_index, event)`
/// as each row streams — the per-sequence analog of the single [`generate`](crate::decode::generate)
/// callback.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference; a mid-stream cancel
/// stops promptly, and any still-running rows finish [`FinishReason::Cancelled`] with their partial
/// output.
pub fn generate_batch(
    model: &CausalLm,
    requests: &[BatchRequest],
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<Vec<GenerationOutput>> {
    if requests.is_empty() {
        return Err(Error::Msg("generate_batch: no requests".into()));
    }
    for (i, r) in requests.iter().enumerate() {
        if r.prompt_ids.is_empty() {
            return Err(Error::Msg(format!("generate_batch: request {i} has an empty prompt")));
        }
    }
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }

    let n = requests.len();
    let dtype = model.compute_dtype();
    let max_prompt = requests.iter().map(|r| r.prompt_ids.len()).max().unwrap() as i32;

    // Admit every request; SeqId(i) lines up with request index i.
    let mut sched = Scheduler::new();
    let seq_ids: Vec<SeqId> = requests
        .iter()
        .map(|r| sched.admit(SeqSpec::new(r.prompt_ids.clone(), r.max_new_tokens, r.stop_tokens.clone())))
        .collect();

    let mut lanes: Vec<Lane> = requests
        .iter()
        .enumerate()
        .map(|(ri, r)| Lane {
            seq: seq_ids[ri],
            pad_len: max_prompt - r.prompt_ids.len() as i32,
            rng: SplitMix64::new(r.seed.unwrap_or_else(default_seed)),
            params: r.sampling,
            history: r.prompt_ids.clone(),
            next_token: 0,
        })
        .collect();

    let mut cache = model.make_cache();

    // ---- Prefill: left-pad every prompt to `max_prompt` and run one batched forward. ----
    let (ids, positions, mask_h) = build_prefill(requests, max_prompt);
    let ids = Array::from_slice(&ids, &[n as i32, max_prompt]);
    let (cos, sin) = model.rope_tables(&positions, n as i32, max_prompt)?;
    let mask = Array::from_slice(&mask_h, &[n as i32, 1, max_prompt, max_prompt]).as_dtype(dtype)?;
    let logits = model.decode_logits_masked(&ids, cache.as_mut(), &cos, &sin, &mask)?;

    // Sample the first token per sequence; keep the lanes that did not immediately retire.
    let mut active: Vec<Lane> = Vec::new();
    let mut keep: Vec<i32> = Vec::new();
    for (ri, mut lane) in std::mem::take(&mut lanes).into_iter().enumerate() {
        if !sched.is_active(lane.seq) {
            // Zero-budget sequence: admitted already finished, never generates.
            on_event(ri, StreamEvent::Done { reason: FinishReason::MaxTokens, generated: 0 });
            continue;
        }
        let tok = sample_row(&logits, ri, &mut lane)?;
        if let LaneStep::Continue = apply_token(&mut sched, &mut lane, tok, on_event) {
            keep.push(ri as i32);
            active.push(lane);
        }
    }
    // Compact away any sequence that finished during prefill before the first decode step.
    if !keep.is_empty() && keep.len() < n {
        cache.retain_sequences(&keep)?;
    }

    // ---- Decode: lockstep, one token per active sequence per step, retiring + compacting. ----
    while !active.is_empty() {
        if cancel.is_cancelled() {
            for lane in &active {
                let generated = sched.generated(lane.seq).len();
                on_event(lane.seq.0, StreamEvent::Done { reason: FinishReason::Cancelled, generated });
            }
            break;
        }

        let b = active.len();
        let width = cache.offset(); // cached columns so far; the new token lands at `width`
        let k_total = width + 1;

        let feed: Vec<i32> = active.iter().map(|l| l.next_token).collect();
        let feed = Array::from_slice(&feed, &[b as i32, 1]);
        // Per-row absolute position of the token being fed: prompt_len + (#tokens already cached).
        let positions: Vec<i32> = active.iter().map(|l| sched.offset(l.seq) as i32 - 1).collect();
        let (cos, sin) = model.rope_tables(&positions, b as i32, 1)?;
        let mask = decode_mask(&active, k_total, dtype)?;

        let logits = model.decode_logits_masked(&feed, cache.as_mut(), &cos, &sin, &mask)?;

        let mut next_keep: Vec<i32> = Vec::new();
        let mut next_active: Vec<Lane> = Vec::new();
        for (row, mut lane) in std::mem::take(&mut active).into_iter().enumerate() {
            let tok = sample_row(&logits, row, &mut lane)?;
            if let LaneStep::Continue = apply_token(&mut sched, &mut lane, tok, on_event) {
                next_keep.push(row as i32);
                next_active.push(lane);
            }
        }
        if next_active.is_empty() {
            break;
        }
        if next_active.len() < b {
            cache.retain_sequences(&next_keep)?;
        }
        active = next_active;
    }

    // ---- Assemble per-request outputs in request order. ----
    Ok(seq_ids
        .iter()
        .map(|&seq| GenerationOutput {
            tokens: sched.generated(seq).to_vec(),
            finish_reason: match sched.finish_reason(seq) {
                Some(CoreFinish::Stop) => FinishReason::StopToken,
                Some(CoreFinish::Length) => FinishReason::MaxTokens,
                // `None` ⇒ still active when a cancel broke the loop; ContentFilter is not produced.
                _ => FinishReason::Cancelled,
            },
        })
        .collect())
}

/// Build the left-padded prefill inputs: token ids, per-row RoPE positions, and the additive
/// `[n, 1, max_prompt, max_prompt]` attention mask (host f32, row-major).
fn build_prefill(requests: &[BatchRequest], max_prompt: i32) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n = requests.len();
    let mp = max_prompt as usize;
    let mut ids = Vec::with_capacity(n * mp);
    let mut positions = Vec::with_capacity(n * mp);
    let mut mask = Vec::with_capacity(n * mp * mp);
    for r in requests {
        let pad = mp - r.prompt_ids.len();
        for c in 0..mp {
            if c < pad {
                ids.push(PAD_ID);
                positions.push(0);
            } else {
                ids.push(r.prompt_ids[c - pad]);
                positions.push((c - pad) as i32);
            }
        }
        let pad = pad as i32;
        for i in 0..max_prompt {
            for j in 0..max_prompt {
                // Causal, and a key is attendable only if it is a real (non-pad) token. The diagonal
                // is always allowed so a pure-padding query row never attends an all-masked set
                // (which would yield NaN); such rows are discarded anyway.
                let ok = j <= i && (j >= pad || j == i);
                mask.push(if ok { 0.0 } else { f32::NEG_INFINITY });
            }
        }
    }
    (ids, positions, mask)
}

/// Build the decode-step additive mask `[b, 1, 1, k_total]`: the single new (real) query attends
/// every cached key except this row's left-pad region.
fn decode_mask(active: &[Lane], k_total: i32, dtype: mlx_rs::Dtype) -> Result<Array> {
    let b = active.len();
    let mut mask = Vec::with_capacity(b * k_total as usize);
    for lane in active {
        for j in 0..k_total {
            mask.push(if j >= lane.pad_len { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    Ok(Array::from_slice(&mask, &[b as i32, 1, 1, k_total]).as_dtype(dtype)?)
}

/// Sample the token for batch row `row` of a `[b, vocab]` logits tensor, using the lane's own
/// sampler state (identical call to the single-sequence path ⇒ identical token for identical
/// logits).
fn sample_row(logits: &Array, row: usize, lane: &mut Lane) -> Result<i32> {
    let idx = Array::from_slice(&[row as i32], &[1]);
    let lg = logits.take_axis(&idx, 0)?; // [1, vocab]
    sample(&lg, &lane.history, &lane.params, &mut lane.rng, None)
}

/// Record `tok` for `lane` through the scheduler and emit its stream events, mirroring the
/// single-sequence loop: a stop token retires the lane (excluded, no `Token` event); otherwise the
/// token is emitted + kept, and a filled budget retires the lane after including it.
fn apply_token(
    sched: &mut Scheduler,
    lane: &mut Lane,
    tok: i32,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> LaneStep {
    let ri = lane.seq.0;
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
