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
    generate_with(decoder, prompt_ids, config, cancel, on_event, None)
}

/// Like [`generate`], with an optional per-step [`ConstraintMask`] (structured-output decoding).
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference runs; otherwise runs
/// to a stop token, the token budget, or a mid-stream cancel, returning the generated tokens.
pub fn generate_with(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    mut constraint: Option<&mut dyn ConstraintMask>,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate: empty prompt".into()));
    }

    let mut rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let mut cache = decoder.make_cache();

    // Prefill the whole prompt at offset 0; logits are for the last prompt position.
    let prompt = input_ids(prompt_ids);
    let mut logits = decoder.step(&prompt, cache.as_mut(), 0)?;

    let mut history: Vec<i32> = prompt_ids.to_vec();
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

        if step + 1 == config.max_new_tokens {
            break; // budget reached; finish stays MaxTokens
        }

        // Feed the new token back; its absolute position is the current cache length.
        let offset = cache.offset();
        let tok = input_ids(&[next]);
        logits = decoder.step(&tok, cache.as_mut(), offset)?;
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
fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}

pub use super::cancel::CancelFlag;
