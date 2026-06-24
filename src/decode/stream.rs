//! The streaming, cancellable decode loop.
//!
//! This is the internal streaming API the engine is built around (story 7156). The backend-neutral
//! `core-llm` contract (story 7154) is **extracted from this working loop** — built concrete first,
//! not designed in a vacuum. The loop is model-agnostic: it drives anything implementing [`Decode`]
//! (the Llama decoder today, Qwen3 / BYO architectures later), emitting a [`StreamEvent`] per token
//! through a callback.
//!
//! Cancellation follows the established contract: a request that is *already cancelled* before any
//! work returns the typed [`Error::Canceled`](crate::error::Error::Canceled); a cancel that trips
//! *mid-stream* stops promptly and returns the partial output marked
//! [`FinishReason::Cancelled`].

use mlx_rs::Array;

use crate::error::{Error, Result};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};
use crate::primitives::input_ids;

/// A decoder the streaming loop can drive: it makes its own cache and produces last-position logits.
pub trait Decode {
    /// A fresh KV cache sized for this decoder.
    fn make_cache(&self) -> Box<dyn KvCache>;

    /// One forward step over `input_ids` (`[batch, seq]`) returning last-position logits
    /// `[batch, vocab]`. `offset` is the RoPE offset (positions already cached).
    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array>;
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop / EOS token was sampled.
    StopToken,
    /// The `max_new_tokens` budget was reached.
    MaxTokens,
    /// Cancellation tripped mid-stream.
    Cancelled,
    /// A host stop condition halted generation after emitting a token — e.g. a request `stop`
    /// string was detected in the decoded text (see the `should_stop` predicate on
    /// [`generate_with`]). Distinct from [`StopToken`](FinishReason::StopToken) (an EOS *id*) but
    /// maps to the same contract `Stop` finish reason.
    Stopped,
}

/// An event emitted as decoding proceeds.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// A newly generated token. `step` is 0-based over generated tokens.
    Token {
        /// The sampled token id.
        id: i32,
        /// Index of this token among the generated tokens.
        step: usize,
    },
    /// Terminal event: generation finished.
    Done {
        /// Why it stopped.
        reason: FinishReason,
        /// How many tokens were generated.
        generated: usize,
    },
}

/// Generation parameters.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling knobs.
    pub sampling: SamplingParams,
    /// RNG seed; `None` ⇒ a fresh per-call seed (non-reproducible).
    pub seed: Option<u64>,
    /// Token ids that stop generation when sampled (EOS / EOT / …). The stop token is excluded
    /// from the output.
    pub stop_tokens: Vec<i32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingParams::default(),
            seed: None,
            stop_tokens: Vec::new(),
        }
    }
}

/// The result of a generation run.
#[derive(Clone, Debug)]
pub struct GenerationOutput {
    /// Generated token ids (excludes the prompt and any stop token).
    pub tokens: Vec<i32>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// A per-step logit constraint (e.g. JSON grammar). Before each token the loop asks for the
/// [`ConstraintMask::allowed`] mask (passed to the sampler so disallowed ids are forced to `-inf`),
/// and after a token is chosen it calls [`ConstraintMask::accept`]. The engine owns no grammar
/// policy — `core_llm::JsonConstraint` is one implementation behind this seam.
pub trait ConstraintMask {
    /// The per-vocab allow mask for the current step.
    fn allowed(&mut self) -> &[bool];
    /// Advance the constraint after `token` is chosen.
    fn accept(&mut self, token: i32);
}

/// Stream tokens from `decoder`, starting from `prompt_ids`, emitting a [`StreamEvent`] per token
/// through `on_event`. Unconstrained convenience wrapper over [`generate_with`].
pub fn generate(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<GenerationOutput> {
    generate_with(decoder, prompt_ids, config, cancel, on_event, None, None)
}

/// Like [`generate`], with an optional per-step [`ConstraintMask`] (structured-output decoding) and
/// an optional `should_stop` predicate.
///
/// `should_stop` is checked after each token is emitted and counted; returning `true` halts
/// generation with [`FinishReason::Stopped`]. It is the seam the provider uses to honor request
/// `stop` strings — the predicate inspects state the provider's detokenizing `on_event` maintains
/// (a `core_llm::StopMatcher`), which is why stop-string matching lives in the text/detok layer and
/// not in this token-id loop.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference runs; otherwise runs
/// to a stop token, the token budget, a mid-stream cancel, or a tripped `should_stop`, returning the
/// generated tokens.
pub fn generate_with(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate: empty prompt".into()));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let mut cache = decoder.make_cache();

    // Prefill the whole prompt at offset 0; logits are for the last prompt position.
    let prompt = input_ids(prompt_ids);
    let logits = decoder.step(&prompt, cache.as_mut(), 0)?;

    decode_loop(
        decoder,
        cache.as_mut(),
        logits,
        rng,
        prompt_ids.to_vec(),
        config,
        cancel,
        on_event,
        constraint,
        should_stop,
    )
}

/// Like [`generate`], but driving a **caller-provided** KV cache that may already hold a prefix
/// (e.g. a [`PagedKvCache`](crate::primitives::PagedKvCache) seeded with shared blocks). Prefills
/// only `prompt_ids[cache.offset()..]` at that offset, then decodes. The cache is borrowed (not
/// consumed) so the caller can inspect or seed siblings from it afterward.
///
/// `cache.offset()` must be `< prompt_ids.len()` (there must be at least one token to prefill).
/// Returns [`Error::Canceled`] on an already-set cancel.
pub fn generate_with_cache(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    cache: &mut dyn KvCache,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_with_cache: empty prompt".into()));
    }
    let offset = cache.offset() as usize;
    if offset >= prompt_ids.len() {
        return Err(Error::Msg(format!(
            "generate_with_cache: cache offset {offset} leaves no prompt suffix to prefill \
             (prompt len {})",
            prompt_ids.len()
        )));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let suffix = input_ids(&prompt_ids[offset..]);
    let logits = decoder.step(&suffix, cache, offset as i32)?;

    decode_loop(
        decoder,
        cache,
        logits,
        rng,
        prompt_ids.to_vec(),
        config,
        cancel,
        on_event,
        None,
        None,
    )
}

/// Drive generation from a **caller-supplied prefill**: a `cache` already advanced over the prompt
/// and its last-position `first_logits`. The multimodal path prefills outside this loop — it embeds
/// the prompt, splices the encoder's image features at the image-token rows, and runs the decoder
/// with interleaved M-RoPE — then hands the result here to sample + stream the continuation. `history`
/// seeds the repetition-penalty window (the full, image-token-expanded prompt ids).
///
/// The `decoder` drives the **decode steps** only (the prompt is already cached); for the Qwen3.6
/// multimodal path that decoder shifts the RoPE offset by `mrope_delta` so post-image text positions
/// continue correctly. Returns [`Error::Canceled`] on an already-set cancel.
#[allow(clippy::too_many_arguments)]
pub fn generate_from_prefill(
    decoder: &dyn Decode,
    cache: &mut dyn KvCache,
    first_logits: Array,
    history: Vec<i32>,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    decode_loop(
        decoder, cache, first_logits, rng, history, config, cancel, on_event, constraint, should_stop,
    )
}

/// The token-by-token decode loop shared by [`generate_with`] and the prefix-cached path
/// ([`crate::decode::generate_cached`]): given the prefill `logits`, an RNG, the seeded `history`
/// (the prompt — the repetition-penalty window), and a `cache` already positioned past the prompt,
/// sample, emit, and step until a stop token, the budget, or a mid-stream cancel.
///
/// The two entry points differ only in how the cache + first `logits` are produced (cold prefill vs.
/// shared-prefix reuse); the loop is identical, so a cached run is token-for-token the same as a cold
/// one for the same prompt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_loop(
    decoder: &dyn Decode,
    cache: &mut dyn KvCache,
    mut logits: Array,
    mut rng: SplitMix64,
    mut history: Vec<i32>,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    mut constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<GenerationOutput> {
    let mut generated: Vec<i32> = Vec::new();
    let mut finish = FinishReason::MaxTokens;

    for step in 0..config.max_new_tokens {
        // Pulling logits to host for sampling forces a graph eval each step, so this check is
        // genuinely effective despite MLX's lazy evaluation.
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }

        // Apply the constraint mask (if any) for this step, then sample. The mask borrow is scoped
        // so the constraint is free to be advanced again below.
        let next = {
            let mask = constraint.as_mut().map(|c| c.allowed());
            sample(&logits, &history, &config.sampling, &mut rng, mask)?
        };

        if config.stop_tokens.contains(&next) {
            finish = FinishReason::StopToken;
            break;
        }

        if let Some(c) = &mut constraint {
            c.accept(next);
        }

        on_event(StreamEvent::Token { id: next, step });
        generated.push(next);
        history.push(next);

        // A host stop condition (e.g. a request `stop` string detected by `on_event`'s detokenizer)
        // halts here, after the triggering token is counted. The next decode step is skipped.
        if should_stop.is_some_and(|f| f()) {
            finish = FinishReason::Stopped;
            break;
        }

        if step + 1 == config.max_new_tokens {
            break; // budget reached; finish stays MaxTokens
        }

        // Feed the new token back; its absolute position is the current cache length.
        let offset = cache.offset();
        let tok = input_ids(&[next]);
        logits = decoder.step(&tok, cache, offset)?;
    }

    on_event(StreamEvent::Done {
        reason: finish,
        generated: generated.len(),
    });
    Ok(GenerationOutput {
        tokens: generated,
        finish_reason: finish,
    })
}

/// A non-reproducible seed for `GenerationConfig::seed == None`.
pub(crate) fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}

pub use super::cancel::CancelFlag;
