//! Shared-prefix KV reuse (epic 7153, story 7168).
//!
//! Requests routinely share a leading run of tokens — a common system prompt, a few-shot preamble,
//! the growing history of a multi-turn chat. A causal decoder's keys/values at position `i` depend
//! only on tokens `0..=i`, so that shared run has **bit-identical** KV across the requests; the
//! [`PrefixCache`] caches it and reuses it instead of recomputing it.
//!
//! The policy — which stored token sequence shares the longest prefix, and LRU eviction — is the
//! backend-neutral [`core_llm::prefix::PrefixIndex`]; this module owns only the MLX tensors that
//! policy points at. [`generate_cached`] is the single-sequence decode loop with reuse spliced into
//! prefill: longest-match → seed a [`ContiguousKvCache`] to the matched length → prefill only the
//! suffix → decode → store the request's full `prompt + generated` KV for next time.
//!
//! Reuse is exact, not approximate: a cached run is **token-for-token identical** to a cold run for
//! the same prompt (same kernels, just KV that was already computed). Concurrent in-batch prefix
//! sharing with copy-on-write is the paged cache's job (story 7169); this is the simpler
//! contiguous-friendly cousin that lands first.

use std::collections::HashMap;

use mlx_rs::Array;

use core_llm::prefix::{PrefixId, PrefixIndex};

use crate::decode::cancel::CancelFlag;
use crate::decode::stream::{
    decode_loop, default_seed, GenerationConfig, GenerationOutput, StreamEvent,
};
use crate::error::Result;
use crate::models::CausalLm;
use crate::primitives::input_ids;
use crate::primitives::kv_cache::{ContiguousKvCache, SEQ_AXIS};
use crate::primitives::sampler::SplitMix64;

/// Cumulative reuse accounting for a [`PrefixCache`] — the measurable payoff of story 7168.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixStats {
    /// Number of [`generate_cached`] lookups performed.
    pub lookups: usize,
    /// Lookups that found and reused a shared prefix.
    pub hits: usize,
    /// Total prompt positions whose KV was reused (prefill skipped) across all hits — the saved span.
    pub reused_prefix_tokens: usize,
    /// Total prompt positions actually run through the model during prefill (cold + post-match
    /// suffixes). With reuse this is smaller than the sum of prompt lengths by exactly
    /// `reused_prefix_tokens`.
    pub computed_prefill_tokens: usize,
}

/// A bounded, LRU shared-prefix KV cache for the single-sequence decode path.
///
/// Pair one with repeated [`generate_cached`] calls that share a prefix (e.g. a fixed system prompt,
/// or successive turns of one conversation) to skip recomputing the shared span. Holds at most
/// `capacity` stored sequences; least-recently-used entries (and their KV tensors) are evicted past
/// that. Single-threaded, like the rest of the engine.
pub struct PrefixCache {
    index: PrefixIndex,
    /// Full-sequence per-layer `(keys, values)` for each live entry, keyed by the index's handle.
    kv: HashMap<PrefixId, Vec<(Array, Array)>>,
    stats: PrefixStats,
}

impl PrefixCache {
    /// A cache retaining at most `capacity` stored sequences (LRU eviction past that).
    pub fn new(capacity: usize) -> Self {
        Self {
            index: PrefixIndex::new(capacity),
            kv: HashMap::new(),
            stats: PrefixStats::default(),
        }
    }

    /// Cumulative reuse accounting since construction.
    pub fn stats(&self) -> PrefixStats {
        self.stats
    }

    /// Number of stored sequences currently held.
    pub fn len(&self) -> usize {
        self.kv.len()
    }

    /// Whether the cache holds no sequences.
    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }

    /// Find the longest cached prefix of `prompt` and, on a hit, build a [`ContiguousKvCache`] seeded
    /// with its KV plus the number of prompt tokens it covers — always `< prompt.len()`, so the
    /// suffix prefill always has at least one token (a whole-prompt match recomputes only the final
    /// token). Returns `None` on a miss (the caller prefills cold). Updates [`PrefixStats`].
    fn seed_for(&mut self, prompt: &[i32]) -> Result<Option<(ContiguousKvCache, usize)>> {
        self.stats.lookups += 1;
        let prompt_len = prompt.len();
        let hit = self
            .index
            .longest_match(prompt)
            .map(|m| (m.id, m.matched_len.min(prompt_len.saturating_sub(1))))
            .filter(|&(id, len)| len > 0 && self.kv.contains_key(&id));

        match hit {
            Some((id, len)) => {
                let layers = slice_layers(&self.kv[&id], len)?;
                self.stats.hits += 1;
                self.stats.reused_prefix_tokens += len;
                self.stats.computed_prefill_tokens += prompt_len - len;
                Ok(Some((ContiguousKvCache::seeded(layers), len)))
            }
            None => {
                self.stats.computed_prefill_tokens += prompt_len;
                Ok(None)
            }
        }
    }

    /// Store `tokens`' full per-layer KV (from the just-finished `cache`) for future reuse, freeing
    /// any LRU entries the insertion evicts. A no-op if the cache has no exportable state.
    fn store(&mut self, tokens: Vec<i32>, cache: &ContiguousKvCache) {
        let Some(layers) = cache.export() else {
            return;
        };
        let out = self.index.insert(tokens);
        for evicted in &out.evicted {
            self.kv.remove(evicted);
        }
        // `contains` guards the degenerate `capacity == 0` case, where the insert immediately evicts
        // its own entry (so we must not leave an orphan in `kv`).
        if self.index.contains(out.id) {
            self.kv.insert(out.id, layers);
        }
    }
}

/// Like [`generate`](crate::decode::generate), but reusing shared-prefix KV through `prefix_cache`.
///
/// On each call: look up the longest cached prefix of `prompt_ids`, seed the KV cache with it and
/// prefill only the remaining suffix (a miss prefills the whole prompt cold), decode to a stop token
/// / the budget / a mid-stream cancel, then store the request's full `prompt + generated` KV for
/// future reuse. The output is **token-for-token identical** to a cold [`generate`](crate::decode::generate)
/// of the same prompt.
///
/// Returns [`Error::Canceled`](crate::error::Error::Canceled) if `cancel` is already set before any
/// inference.
pub fn generate_cached(
    model: &CausalLm,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    prefix_cache: &mut PrefixCache,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(crate::error::Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(crate::error::Error::Msg("generate_cached: empty prompt".into()));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));

    // Reuse the longest cached prefix (or start cold), then prefill only the uncached suffix.
    let (mut cache, matched_len) = match prefix_cache.seed_for(prompt_ids)? {
        Some((cache, len)) => (cache, len),
        None => (model.new_cache(), 0),
    };
    let suffix = input_ids(&prompt_ids[matched_len..]);
    let logits = model.decode_logits(&suffix, &mut cache, matched_len as i32)?;

    let out = decode_loop(
        model,
        &mut cache,
        logits,
        rng,
        prompt_ids.to_vec(),
        config,
        cancel,
        on_event,
        None,
        None,
    )?;

    // Store the full sequence (prompt + generated) so the next shared-prefix request reuses it.
    let mut full = prompt_ids.to_vec();
    full.extend_from_slice(&out.tokens);
    prefix_cache.store(full, &cache);

    Ok(out)
}

/// Slice each layer's `(keys, values)` to the first `len` sequence positions (axis [`SEQ_AXIS`]).
/// When `len` already equals the stored length the tensors are cloned as-is (no gather).
fn slice_layers(stored: &[(Array, Array)], len: usize) -> Result<Vec<(Array, Array)>> {
    let mut out = Vec::with_capacity(stored.len());
    let idx = Array::from_slice(&(0..len as i32).collect::<Vec<_>>(), &[len as i32]);
    for (k, v) in stored {
        if k.shape()[SEQ_AXIS as usize] == len as i32 {
            out.push((k.clone(), v.clone()));
        } else {
            out.push((k.take_axis(&idx, SEQ_AXIS)?, v.take_axis(&idx, SEQ_AXIS)?));
        }
    }
    Ok(out)
}
