//! Key/value cache.
//!
//! The cache is the seam the throughput work (P4) plugs into: the dynamic-batch scheduler
//! (story 7167), the prefix cache (7168) and the paged cache (7169/7170) all implement the
//! [`KvCache`] trait so they swap in without the decoder changing. The decoder only ever talks to
//! the trait.
//!
//! [`ContiguousKvCache`] is the day-one implementation: a per-layer growing concat along the
//! sequence axis, modelled on the working mlx-gen caches (prompt-refine `LlamaKvCache`,
//! sensenova/flux2 `Qwen3KvCache`). It is **batch-capable** — the batch axis is real, not
//! hardcoded to 1, so an N-sequence batch with a uniform length works today. Ragged per-sequence
//! offsets (sequences of differing lengths in one batch) need the paged cache, which is exactly
//! why the trait exists.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use crate::error::Result;

/// Layout, per layer, of the cached keys/values: `[batch, n_kv_heads, seq, head_dim]`. Keys are
/// stored already-RoPE'd; values raw. The sequence axis (2) is the one that grows each step.
pub const SEQ_AXIS: i32 = 2;

/// The decoder-facing cache contract.
///
/// A decoder, for each layer, hands the cache this step's keys/values and gets back the full
/// keys/values to attend over. Positional offset bookkeeping is the cache's job — [`KvCache::offset`]
/// reports how many positions are already cached (the RoPE offset for the next step), so the
/// decoder reads it once before the step rather than threading an `index_pos` through every call.
pub trait KvCache {
    /// Append `keys`/`values` for `layer` (each `[batch, n_kv_heads, step, head_dim]`) and return
    /// the full cached `(keys, values)` to attend over, same layout with the sequence axis grown.
    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)>;

    /// Number of sequence positions currently cached — i.e. the RoPE offset for the next step.
    /// `0` before the first update. Inferred from layer 0 (all layers advance in lockstep).
    fn offset(&self) -> i32;

    /// Batch size of the cached tensors, or `0` before the first update.
    fn batch_size(&self) -> i32;

    /// Number of decoder layers this cache holds slots for.
    fn num_layers(&self) -> usize;

    /// Compact the batch to keep only the rows in `keep` (indices into the current batch axis),
    /// in the given order — the seam the dynamic-batch scheduler (story 7167) retires a finished
    /// sequence through, so the next step runs a smaller batch. A contiguous cache gathers the kept
    /// rows along the batch axis; the paged cache (7169) frees the dropped sequences' pages. `keep`
    /// must be a subset of `0..batch_size`; an empty cache is a no-op.
    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()>;

    /// Drop cached positions past `len`, keeping positions `0..len` along the sequence axis — the
    /// seam speculative decoding (story 7171) rolls back rejected draft tokens through. `len` must be
    /// `<= offset()`; `len == offset()` is a no-op and an empty cache ignores it.
    fn truncate(&mut self, len: i32) -> Result<()>;

    /// Drop all cached state, returning the cache to its freshly-constructed (empty) condition.
    fn reset(&mut self);

    /// Downcast hook so a decoder can recover its concrete cache from a `&mut dyn KvCache` — the
    /// hybrid Qwen3.6 cache (recurrent linear-attention state + KV) is driven natively rather than
    /// through the softmax-only [`KvCache::update`] path.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Growing-concat KV cache: one `Option<(K, V)>` slot per layer, concatenated along the sequence
/// axis each step. Correctness-first; the paged cache (P4) is the throughput replacement behind
/// the same trait.
#[derive(Debug)]
pub struct ContiguousKvCache {
    layers: Vec<Option<(Array, Array)>>,
}

impl ContiguousKvCache {
    /// A fresh cache with `num_layers` empty slots.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
        }
    }

    /// Borrow the currently-cached `(keys, values)` for `layer`, if any.
    pub fn peek(&self, layer: usize) -> Option<&(Array, Array)> {
        self.layers.get(layer).and_then(|s| s.as_ref())
    }

    /// Construct a cache pre-populated with per-layer `(keys, values)` — the seam the prefix cache
    /// (story 7168) reuses a shared prefix's KV through. Each entry is `[batch, n_kv_heads, seq,
    /// head_dim]` (keys already-RoPE'd); the cache then reports [`KvCache::offset`] equal to that
    /// seq length, so a decoder prefills only the suffix at that offset and attends over the seeded
    /// keys. Layout/length consistency across layers is the caller's responsibility.
    pub fn seeded(layers: Vec<(Array, Array)>) -> Self {
        Self {
            layers: layers.into_iter().map(Some).collect(),
        }
    }

    /// Snapshot every layer's cached `(keys, values)` as clones (MLX arrays are refcounted, so this
    /// shares buffers rather than copying), or `None` if any layer is still empty. The prefix cache
    /// stores this after a generation so a later shared-prefix request can be [`seeded`] from it.
    ///
    /// [`seeded`]: ContiguousKvCache::seeded
    pub fn export(&self) -> Option<Vec<(Array, Array)>> {
        self.layers.iter().cloned().collect()
    }
}

impl KvCache for ContiguousKvCache {
    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        let merged = match self.layers[layer].take() {
            Some((pk, pv)) => (
                concatenate_axis(&[&pk, keys], SEQ_AXIS)?,
                concatenate_axis(&[&pv, values], SEQ_AXIS)?,
            ),
            None => (keys.clone(), values.clone()),
        };
        self.layers[layer] = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }

    fn offset(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map(|(k, _)| k.shape()[SEQ_AXIS as usize])
            .unwrap_or(0)
    }

    fn batch_size(&self) -> i32 {
        self.layers
            .first()
            .and_then(|s| s.as_ref())
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        let idx = Array::from_slice(keep, &[keep.len() as i32]);
        for slot in &mut self.layers {
            if let Some((k, v)) = slot.take() {
                *slot = Some((k.take_axis(&idx, 0)?, v.take_axis(&idx, 0)?));
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if len < 0 {
            return Err(crate::error::Error::Msg(format!(
                "truncate: negative len {len}"
            )));
        }
        let idx = Array::from_slice(&(0..len).collect::<Vec<_>>(), &[len]);
        for slot in &mut self.layers {
            if let Some((k, v)) = slot.take() {
                if k.shape()[SEQ_AXIS as usize] <= len {
                    *slot = Some((k, v)); // already at/under the target length
                } else {
                    *slot = Some((k.take_axis(&idx, SEQ_AXIS)?, v.take_axis(&idx, SEQ_AXIS)?));
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// The compress/dequantize seam a [`QuantizedKvCache`] plugs a real method (RVQ, VecInfer, …) into
/// without touching the cache shell.
///
/// A quantizer turns one `update()`'s worth of keys/values — each `[batch, n_kv_heads, step,
/// head_dim]`, the [`SEQ_AXIS`] layout — into an opaque compressed [`Block`](Quantizer::Block), and
/// turns that block back into the same-layout dense `(keys, values)` on read. The cache shell knows
/// nothing about the representation: it only stores [`Block`](Quantizer::Block)s and asks the
/// quantizer to round-trip them. Compaction (`retain_sequences`) and rollback (`truncate`) operate on
/// the compressed representation through the small set of block-level operations below, so a method
/// implements them once and never reaches into the cache.
///
/// The reference implementation is [`IdentityQuantizer`], a pass-through that stores the arrays
/// verbatim; with it a [`QuantizedKvCache`] is behaviorally identical to [`ContiguousKvCache`].
pub trait Quantizer {
    /// The method's compressed representation of one block of keys/values. Opaque to the cache.
    type Block;

    /// Compress one `update()`'s `keys`/`values` (each `[batch, n_kv_heads, step, head_dim]`).
    fn encode(&self, keys: &Array, values: &Array) -> Result<Self::Block>;

    /// Dequantize a block back to dense `(keys, values)`, same layout the [`encode`](Quantizer::encode)
    /// inputs had.
    fn decode(&self, block: &Self::Block) -> Result<(Array, Array)>;

    /// Number of token positions (the [`SEQ_AXIS`] extent) a block holds. Used for offset
    /// bookkeeping without a full decode.
    fn seq_len(&self, block: &Self::Block) -> i32;

    /// Batch size (axis 0 extent) a block holds. Used for [`KvCache::batch_size`] without a decode.
    fn batch_size(&self, block: &Self::Block) -> i32;

    /// Compact a block's batch axis to keep only the rows in `keep` (indices into the current batch
    /// axis), in order — the compressed-path equivalent of [`KvCache::retain_sequences`].
    fn retain_sequences(&self, block: &Self::Block, keep: &Array) -> Result<Self::Block>;

    /// Keep only the first `len` token positions of a block along [`SEQ_AXIS`], returning the
    /// truncated block — the compressed-path equivalent of [`KvCache::truncate`] applied within a
    /// block. `len` is `<= self.seq_len(block)`.
    fn truncate(&self, block: &Self::Block, len: i32) -> Result<Self::Block>;
}

/// Pass-through quantizer: stores keys/values verbatim. The reference seam implementation — a
/// [`QuantizedKvCache<IdentityQuantizer>`] is behaviorally identical (array-for-array) to a
/// [`ContiguousKvCache`], which is exactly what the parametrized kv_cache tests assert.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityQuantizer;

impl Quantizer for IdentityQuantizer {
    type Block = (Array, Array);

    fn encode(&self, keys: &Array, values: &Array) -> Result<Self::Block> {
        Ok((keys.clone(), values.clone()))
    }

    fn decode(&self, block: &Self::Block) -> Result<(Array, Array)> {
        Ok((block.0.clone(), block.1.clone()))
    }

    fn seq_len(&self, block: &Self::Block) -> i32 {
        block.0.shape()[SEQ_AXIS as usize]
    }

    fn batch_size(&self, block: &Self::Block) -> i32 {
        block.0.shape()[0]
    }

    fn retain_sequences(&self, block: &Self::Block, keep: &Array) -> Result<Self::Block> {
        Ok((block.0.take_axis(keep, 0)?, block.1.take_axis(keep, 0)?))
    }

    fn truncate(&self, block: &Self::Block, len: i32) -> Result<Self::Block> {
        let idx = Array::from_slice(&(0..len).collect::<Vec<_>>(), &[len]);
        Ok((
            block.0.take_axis(&idx, SEQ_AXIS)?,
            block.1.take_axis(&idx, SEQ_AXIS)?,
        ))
    }
}

/// How the first tokens of a sequence are handled. Many compression methods keep a small dense
/// prefix (attention "sinks"): the first `sink_tokens` positions are stored verbatim and never
/// compressed, mirroring VeloxQuant's `sink_cache.py`/`sliding_window_cache.py` semantics. Carried as
/// a config knob and honored by `update`/read even though only the dense path is fully exercised by
/// the shipped [`IdentityQuantizer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SinkConfig {
    /// Number of leading token positions kept dense (uncompressed). `0` (the default) compresses
    /// every position — the configuration under which a [`QuantizedKvCache`] is array-identical to a
    /// [`ContiguousKvCache`].
    pub sink_tokens: i32,

    /// Sliding-window bound, in token positions, on the **non-sink** (compressed) tail: when set,
    /// only the most recent `window` positions after the dense sink are retained; older positions
    /// are eligible for eviction. This mirrors the sliding-window half of VeloxQuant's
    /// `sliding_window_cache.py` semantics and is the companion knob to [`sink_tokens`](Self::sink_tokens)
    /// — sink keeps the *oldest* positions dense, window bounds the *newest* compressed positions.
    ///
    /// **Config-only knob (reserved for a future eviction policy).** `None` (the default) means no
    /// window bound — every compressed position is retained, which is the configuration under which a
    /// [`QuantizedKvCache`] stays array-identical to a [`ContiguousKvCache`]. A future eviction
    /// implementation (or a real [`Quantizer`]) reads this field to drop blocks past the window; the
    /// shipped [`IdentityQuantizer`] path does not evict, so a `Some(_)` window currently has no
    /// effect on the decoded view. Stored as `i32` to match [`sink_tokens`](Self::sink_tokens); a
    /// non-positive value is treated the same as `None` by any consumer.
    pub window: Option<i32>,
}

impl SinkConfig {
    /// No dense sink and no window bound — every position is compressed and retained.
    pub const NONE: SinkConfig = SinkConfig {
        sink_tokens: 0,
        window: None,
    };

    /// Keep the first `n` token positions dense, with no sliding-window bound on the tail.
    pub fn keep_first(n: i32) -> Self {
        Self {
            sink_tokens: n,
            window: None,
        }
    }

    /// Set the sliding-window bound (in token positions) on the compressed tail. A non-positive
    /// `window` clears the bound (equivalent to `None`). Config-only: reserved for a future eviction
    /// policy; see [`window`](Self::window).
    pub fn with_window(mut self, window: i32) -> Self {
        self.window = (window > 0).then_some(window);
        self
    }

    /// The configured sliding-window bound, normalized: `None` when unset or non-positive.
    pub fn window(&self) -> Option<i32> {
        self.window.filter(|&w| w > 0)
    }
}

/// One layer's quantized store: an optional dense **sink** (the first [`SinkConfig::sink_tokens`]
/// positions, kept verbatim) followed by an ordered run of **compressed blocks**, each one block's
/// worth of later positions. The decoder-visible keys/values are `concat(sink, decode(blocks…))`
/// along [`SEQ_AXIS`] — same order, same per-position values a contiguous cache holds, so the
/// identity quantizer reproduces [`ContiguousKvCache`] exactly.
struct LayerStore<B> {
    /// First `sink_tokens` positions kept dense, or `None` when no sink is configured / filled yet.
    sink: Option<(Array, Array)>,
    /// Compressed blocks for the remaining positions, in sequence order.
    blocks: Vec<B>,
}

impl<B> LayerStore<B> {
    fn new() -> Self {
        Self {
            sink: None,
            blocks: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.sink.is_none() && self.blocks.is_empty()
    }
}

impl<B> std::fmt::Debug for LayerStore<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerStore")
            .field("has_sink", &self.sink.is_some())
            .field("blocks", &self.blocks.len())
            .finish()
    }
}

/// Compressed KV cache: a third [`KvCache`] implementation holding per-layer keys/values behind a
/// pluggable [`Quantizer`]. On [`update`](KvCache::update) it compresses the new positions into the
/// quantizer's block representation (after first filling the configured dense [`SinkConfig`] sink);
/// on read it dequantizes and concatenates back to the dense `(keys, values)` the decoder attends
/// over. The decoder is untouched — it still only sees `&mut dyn KvCache`.
///
/// With [`IdentityQuantizer`] and the default ([`SinkConfig::NONE`]) sink, this is array-for-array
/// identical to [`ContiguousKvCache`]; that is the seam test. A real method (RVQ, VecInfer, …)
/// implements [`Quantizer`] and drops in without touching this shell.
#[derive(Debug)]
pub struct QuantizedKvCache<Q: Quantizer> {
    quantizer: Q,
    sink: SinkConfig,
    layers: Vec<LayerStore<Q::Block>>,
}

impl<Q: Quantizer> QuantizedKvCache<Q> {
    /// A fresh cache with `num_layers` empty slots, the given `quantizer`, and no dense sink
    /// ([`SinkConfig::NONE`]) — the configuration under which it matches [`ContiguousKvCache`].
    pub fn new(num_layers: usize, quantizer: Q) -> Self {
        Self::with_sink(num_layers, quantizer, SinkConfig::NONE)
    }

    /// A fresh cache with an explicit dense-sink configuration.
    pub fn with_sink(num_layers: usize, quantizer: Q, sink: SinkConfig) -> Self {
        Self {
            quantizer,
            sink,
            layers: (0..num_layers).map(|_| LayerStore::new()).collect(),
        }
    }

    /// The configured dense-sink policy.
    pub fn sink_config(&self) -> SinkConfig {
        self.sink
    }

    /// Dequantize `layer`'s full cached `(keys, values)` — `concat(sink, decode(blocks…))` along
    /// [`SEQ_AXIS`] — or `None` if the layer is still empty.
    pub fn peek(&self, layer: usize) -> Option<Result<(Array, Array)>> {
        let store = self.layers.get(layer)?;
        if store.is_empty() {
            return None;
        }
        Some(self.materialize(store))
    }

    /// Sequence length of a layer's store (dense sink + every block) without a full decode.
    fn layer_offset(store: &LayerStore<Q::Block>, q: &Q) -> i32 {
        let sink = store
            .sink
            .as_ref()
            .map(|(k, _)| k.shape()[SEQ_AXIS as usize])
            .unwrap_or(0);
        sink + store.blocks.iter().map(|b| q.seq_len(b)).sum::<i32>()
    }

    /// Batch size of a layer's store without a full decode.
    fn layer_batch(store: &LayerStore<Q::Block>, q: &Q) -> i32 {
        if let Some((k, _)) = store.sink.as_ref() {
            k.shape()[0]
        } else if let Some(b) = store.blocks.first() {
            q.batch_size(b)
        } else {
            0
        }
    }

    fn materialize(&self, store: &LayerStore<Q::Block>) -> Result<(Array, Array)> {
        let mut ks: Vec<Array> = Vec::with_capacity(store.blocks.len() + 1);
        let mut vs: Vec<Array> = Vec::with_capacity(store.blocks.len() + 1);
        if let Some((k, v)) = store.sink.as_ref() {
            ks.push(k.clone());
            vs.push(v.clone());
        }
        for b in &store.blocks {
            let (k, v) = self.quantizer.decode(b)?;
            ks.push(k);
            vs.push(v);
        }
        let kr: Vec<&Array> = ks.iter().collect();
        let vr: Vec<&Array> = vs.iter().collect();
        Ok((
            concatenate_axis(&kr, SEQ_AXIS)?,
            concatenate_axis(&vr, SEQ_AXIS)?,
        ))
    }

    /// Split `[..., step, ...]` keys/values into the leading `n` positions and the rest along
    /// [`SEQ_AXIS`]. Either side may be empty (length 0).
    fn split_seq(x: &Array, n: i32) -> Result<(Array, Array)> {
        let total = x.shape()[SEQ_AXIS as usize];
        let head_idx = Array::from_slice(&(0..n).collect::<Vec<_>>(), &[n]);
        let tail_idx = Array::from_slice(&(n..total).collect::<Vec<_>>(), &[total - n]);
        Ok((
            x.take_axis(&head_idx, SEQ_AXIS)?,
            x.take_axis(&tail_idx, SEQ_AXIS)?,
        ))
    }
}

impl<Q: Quantizer + 'static> KvCache for QuantizedKvCache<Q> {
    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        let store = &mut self.layers[layer];

        // Route incoming positions: first fill the dense sink up to `sink_tokens`, compress the rest.
        let sink_have = store
            .sink
            .as_ref()
            .map(|(k, _)| k.shape()[SEQ_AXIS as usize])
            .unwrap_or(0);
        let sink_want = (self.sink.sink_tokens - sink_have).max(0);
        let step = keys.shape()[SEQ_AXIS as usize];
        let to_sink = sink_want.min(step);

        if to_sink > 0 {
            let (k_head, k_tail) = Self::split_seq(keys, to_sink)?;
            let (v_head, v_tail) = Self::split_seq(values, to_sink)?;
            store.sink = match store.sink.take() {
                Some((sk, sv)) => Some((
                    concatenate_axis(&[&sk, &k_head], SEQ_AXIS)?,
                    concatenate_axis(&[&sv, &v_head], SEQ_AXIS)?,
                )),
                None => Some((k_head, v_head)),
            };
            if k_tail.shape()[SEQ_AXIS as usize] > 0 {
                store.blocks.push(self.quantizer.encode(&k_tail, &v_tail)?);
            }
        } else {
            store.blocks.push(self.quantizer.encode(keys, values)?);
        }

        self.materialize(&self.layers[layer])
    }

    fn offset(&self) -> i32 {
        self.layers
            .first()
            .map(|s| Self::layer_offset(s, &self.quantizer))
            .unwrap_or(0)
    }

    fn batch_size(&self) -> i32 {
        self.layers
            .first()
            .map(|s| Self::layer_batch(s, &self.quantizer))
            .unwrap_or(0)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        let idx = Array::from_slice(keep, &[keep.len() as i32]);
        for store in &mut self.layers {
            if let Some((k, v)) = store.sink.take() {
                store.sink = Some((k.take_axis(&idx, 0)?, v.take_axis(&idx, 0)?));
            }
            for b in &mut store.blocks {
                *b = self.quantizer.retain_sequences(b, &idx)?;
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if len < 0 {
            return Err(crate::error::Error::Msg(format!(
                "truncate: negative len {len}"
            )));
        }
        for store in &mut self.layers {
            if store.is_empty() {
                continue;
            }
            // A populated layer that truncates to length 0 must stay non-empty with an *empty* dense
            // slot — `peek` returns `Some((empty, empty))`, matching `ContiguousKvCache::truncate`,
            // which keeps `Some((empty, empty))` rather than collapsing the slot to `None`. Capture a
            // zero-length `(keys, values)` donor (correct batch/heads/head_dim, seq extent 0) from the
            // current contents before we drain them.
            let empty_slot: Option<(Array, Array)> = if len == 0 {
                Some(if let Some((k, v)) = store.sink.as_ref() {
                    IdentityQuantizer.truncate(&(k.clone(), v.clone()), 0)?
                } else {
                    let b = store
                        .blocks
                        .first()
                        .expect("non-empty store with no sink has at least one block");
                    let (k, v) = self.quantizer.decode(b)?;
                    IdentityQuantizer.truncate(&(k, v), 0)?
                })
            } else {
                None
            };

            // Walk sink then blocks, keeping positions until `len` is exhausted; drop the remainder.
            let mut remaining = len;
            if let Some((k, v)) = store.sink.take() {
                let sink_len = k.shape()[SEQ_AXIS as usize];
                if remaining >= sink_len {
                    store.sink = Some((k, v));
                    remaining -= sink_len;
                } else if remaining > 0 {
                    let kept = IdentityQuantizer.truncate(&(k, v), remaining)?;
                    store.sink = Some(kept);
                    remaining = 0;
                } else {
                    // remaining == 0: sink fully dropped (sink stays None).
                }
            }
            let mut kept_blocks: Vec<Q::Block> = Vec::with_capacity(store.blocks.len());
            for b in store.blocks.drain(..) {
                if remaining <= 0 {
                    break; // drop this and all following blocks
                }
                let bl = self.quantizer.seq_len(&b);
                if remaining >= bl {
                    kept_blocks.push(b);
                    remaining -= bl;
                } else {
                    kept_blocks.push(self.quantizer.truncate(&b, remaining)?);
                    remaining = 0;
                }
            }
            store.blocks = kept_blocks;

            // Retain the empty dense slot so a truncated-to-empty layer still peeks as `Some(empty)`.
            if store.is_empty() {
                store.sink = empty_slot;
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        for store in &mut self.layers {
            store.sink = None;
            store.blocks.clear();
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// `[b, h, s, d]` of sequential f32 values, for shape/equality checks.
    fn arange4(b: i32, h: i32, s: i32, d: i32) -> Array {
        let n = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Array::from_slice(&data, &[b, h, s, d])
    }

    fn host(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// Read a layer's decoded keys back through whichever cache is under test. Both implementations
    /// expose a `peek` returning the full dense `(keys, values)`, but with different signatures
    /// (`ContiguousKvCache` borrows; `QuantizedKvCache` re-materializes), so the harness passes the
    /// reader in.
    type PeekKeys<C> = fn(&C, usize) -> Option<Array>;

    fn contiguous_peek(c: &ContiguousKvCache, layer: usize) -> Option<Array> {
        c.peek(layer).map(|(k, _)| k.clone())
    }

    fn quantized_peek(c: &QuantizedKvCache<IdentityQuantizer>, layer: usize) -> Option<Array> {
        c.peek(layer).map(|r| r.unwrap().0)
    }

    // ---- Parametrized trait-level test bodies (run against every `impl KvCache`) ----------------

    fn body_first_update_stores_and_returns_input(cache: &mut dyn KvCache) {
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.batch_size(), 0);

        let k = arange4(1, 2, 3, 4);
        let v = arange4(1, 2, 3, 4);
        let (ka, va) = cache.update(0, &k, &v).unwrap();
        assert_eq!(ka.shape(), &[1, 2, 3, 4]);
        assert_eq!(va.shape(), &[1, 2, 3, 4]);
        assert_eq!(host(&ka), host(&k)); // first store returns the input verbatim
        assert_eq!(cache.offset(), 3);
        assert_eq!(cache.num_layers(), 2);
    }

    fn body_second_update_concatenates_on_seq_axis(cache: &mut dyn KvCache) {
        let k0 = arange4(1, 2, 3, 4);
        cache.update(0, &k0, &k0).unwrap();
        let k1 = arange4(1, 2, 1, 4); // one new token
        let (ka, _) = cache.update(0, &k1, &k1).unwrap();
        assert_eq!(ka.shape(), &[1, 2, 4, 4]); // 3 + 1 along seq
        assert_eq!(cache.offset(), 4);
    }

    fn body_supports_batch_greater_than_one(cache: &mut dyn KvCache) {
        let k0 = arange4(4, 8, 5, 16); // batch = 4
        cache.update(0, &k0, &k0).unwrap();
        let k1 = arange4(4, 8, 2, 16);
        let (ka, va) = cache.update(0, &k1, &k1).unwrap();
        assert_eq!(ka.shape(), &[4, 8, 7, 16]);
        assert_eq!(va.shape(), &[4, 8, 7, 16]);
        assert_eq!(cache.batch_size(), 4);
        assert_eq!(cache.offset(), 7);
    }

    fn body_concatenated_values_are_in_order(cache: &mut dyn KvCache) {
        // [1,1,2,2] = [[0,1],[2,3]]
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]);
        // [1,1,1,2] = [[10,11]]
        let b = Array::from_slice(&[10.0f32, 11.0], &[1, 1, 1, 2]);
        cache.update(0, &a, &a).unwrap();
        let (ka, _) = cache.update(0, &b, &b).unwrap();
        assert_eq!(host(&ka), vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0]);
    }

    fn body_retain_sequences_compacts_batch_rows<C: KvCache>(cache: &mut C, peek: PeekKeys<C>) {
        // Batch of 3 rows; drop the middle one, keep [0, 2] in order.
        let row = |r: f32| vec![r; 2]; // [1, hkv=2, s=1, hd=1] flattened (hd=1) => 2 values/row
        let mut data = Vec::new();
        for r in 0..3 {
            data.extend(row(r as f32));
        }
        let k = Array::from_slice(&data, &[3, 2, 1, 1]);
        cache.update(0, &k, &k).unwrap();
        assert_eq!(cache.batch_size(), 3);

        cache.retain_sequences(&[0, 2]).unwrap();
        assert_eq!(cache.batch_size(), 2);
        assert_eq!(cache.offset(), 1);
        let ka = peek(cache, 0).unwrap();
        // Kept rows 0 and 2 (each 2 heads * 1 * 1 = 2 values): all 0.0 then all 2.0.
        assert_eq!(host(&ka), vec![0.0, 0.0, 2.0, 2.0]);
    }

    fn body_retain_sequences_on_empty_cache_is_noop<C: KvCache>(cache: &mut C, peek: PeekKeys<C>) {
        cache.retain_sequences(&[0]).unwrap();
        assert_eq!(cache.batch_size(), 0);
        assert!(peek(cache, 0).is_none());
    }

    fn body_truncate_slices_sequence_axis<C: KvCache>(cache: &mut C, peek: PeekKeys<C>) {
        // [1,1,5,1] = values 0..4 along the seq axis.
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
        cache.update(0, &a, &a).unwrap();
        assert_eq!(cache.offset(), 5);
        cache.truncate(3).unwrap();
        assert_eq!(cache.offset(), 3);
        let k = peek(cache, 0).unwrap();
        assert_eq!(host(&k), vec![0.0, 1.0, 2.0]);
        cache.truncate(10).unwrap(); // no-op past the end
        assert_eq!(cache.offset(), 3);
    }

    /// `truncate(0)` on a populated layer keeps an **empty** dense slot (offset 0, `peek` returns an
    /// empty array), rather than collapsing the layer to `None`. Both implementations must agree —
    /// this is the contiguous-reference parity the seam guarantees.
    fn body_truncate_to_zero_keeps_empty_slot<C: KvCache>(cache: &mut C, peek: PeekKeys<C>) {
        let a = arange4(1, 2, 5, 4);
        cache.update(0, &a, &a).unwrap();
        assert_eq!(cache.batch_size(), 1);

        cache.truncate(0).unwrap();
        assert_eq!(cache.offset(), 0);
        let k = peek(cache, 0).expect("truncate(0) retains an empty slot (Some), not None");
        assert_eq!(k.shape(), &[1, 2, 0, 4]); // empty along seq, batch/heads/head_dim preserved
        assert_eq!(k.size(), 0); // zero elements
        assert_eq!(cache.batch_size(), 1); // batch dim still readable from the empty slot
    }

    fn body_reset_clears_state<C: KvCache>(cache: &mut C, peek: PeekKeys<C>) {
        let k = arange4(1, 2, 3, 4);
        cache.update(0, &k, &k).unwrap();
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert!(peek(cache, 0).is_none());
    }

    // ---- Run every body against both `ContiguousKvCache` and `QuantizedKvCache<Identity>` -------

    macro_rules! kv_cache_suite {
        ($modname:ident, $new2:expr, $new1:expr, $peek:expr) => {
            mod $modname {
                use super::*;

                #[test]
                fn first_update_stores_and_returns_input() {
                    body_first_update_stores_and_returns_input(&mut $new2);
                }
                #[test]
                fn second_update_concatenates_on_seq_axis() {
                    body_second_update_concatenates_on_seq_axis(&mut $new1);
                }
                #[test]
                fn supports_batch_greater_than_one() {
                    body_supports_batch_greater_than_one(&mut $new1);
                }
                #[test]
                fn concatenated_values_are_in_order() {
                    body_concatenated_values_are_in_order(&mut $new1);
                }
                #[test]
                fn retain_sequences_compacts_batch_rows() {
                    body_retain_sequences_compacts_batch_rows(&mut $new1, $peek);
                }
                #[test]
                fn retain_sequences_on_empty_cache_is_noop() {
                    body_retain_sequences_on_empty_cache_is_noop(&mut $new2, $peek);
                }
                #[test]
                fn truncate_slices_sequence_axis() {
                    body_truncate_slices_sequence_axis(&mut $new1, $peek);
                }
                #[test]
                fn truncate_to_zero_keeps_empty_slot() {
                    body_truncate_to_zero_keeps_empty_slot(&mut $new1, $peek);
                }
                #[test]
                fn reset_clears_state() {
                    body_reset_clears_state(&mut $new2, $peek);
                }
            }
        };
    }

    kv_cache_suite!(
        contiguous,
        ContiguousKvCache::new(2),
        ContiguousKvCache::new(1),
        contiguous_peek as PeekKeys<ContiguousKvCache>
    );

    kv_cache_suite!(
        quantized_identity,
        QuantizedKvCache::new(2, IdentityQuantizer),
        QuantizedKvCache::new(1, IdentityQuantizer),
        quantized_peek as PeekKeys<QuantizedKvCache<IdentityQuantizer>>
    );

    // ---- Quantized-path-specific tests ----------------------------------------------------------

    /// The identity quantizer must reproduce `ContiguousKvCache` array-for-array across a multi-step
    /// interleaving of updates on multiple layers.
    #[test]
    fn identity_matches_contiguous_step_for_step() {
        let mut contig = ContiguousKvCache::new(2);
        let mut quant = QuantizedKvCache::new(2, IdentityQuantizer);
        let steps = [
            arange4(2, 4, 3, 8),
            arange4(2, 4, 1, 8),
            arange4(2, 4, 5, 8),
        ];
        for layer in 0..2 {
            for s in &steps {
                let (ck, cv) = contig.update(layer, s, s).unwrap();
                let (qk, qv) = quant.update(layer, s, s).unwrap();
                assert_eq!(qk.shape(), ck.shape());
                assert_eq!(host(&qk), host(&ck));
                assert_eq!(host(&qv), host(&cv));
            }
        }
        assert_eq!(quant.offset(), contig.offset());
        assert_eq!(quant.batch_size(), contig.batch_size());
    }

    /// `retain_sequences` must compact the compressed (multi-block) path correctly.
    #[test]
    fn quantized_retain_sequences_across_blocks() {
        let mut cache = QuantizedKvCache::new(1, IdentityQuantizer);
        // Two separate updates => two blocks; 3 rows each, row r filled with r.
        let mk = || {
            let mut data = Vec::new();
            for r in 0..3 {
                data.extend(vec![r as f32; 2]); // [3, hkv=2, s=1, hd=1]
            }
            Array::from_slice(&data, &[3, 2, 1, 1])
        };
        cache.update(0, &mk(), &mk()).unwrap();
        cache.update(0, &mk(), &mk()).unwrap();
        assert_eq!(cache.batch_size(), 3);
        assert_eq!(cache.offset(), 2); // two single-token blocks

        cache.retain_sequences(&[2, 0]).unwrap(); // keep rows 2,0 in that order
        assert_eq!(cache.batch_size(), 2);
        let (k, _) = cache.peek(0).unwrap().unwrap();
        // shape [batch=2, heads=2, seq=2, hd=1], batch-major: kept row 2 (all heads, both blocks
        // along seq => 2.0 x4) then kept row 0 (0.0 x4).
        assert_eq!(k.shape(), &[2, 2, 2, 1]);
        assert_eq!(host(&k), vec![2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// `truncate` must roll back across the compressed multi-block path, including cutting inside a
    /// block.
    #[test]
    fn quantized_truncate_across_blocks() {
        let mut cache = QuantizedKvCache::new(1, IdentityQuantizer);
        let blk = |start: f32, n: i32| {
            let data: Vec<f32> = (0..n).map(|i| start + i as f32).collect();
            Array::from_slice(&data, &[1, 1, n, 1])
        };
        cache.update(0, &blk(0.0, 3), &blk(0.0, 3)).unwrap(); // positions 0,1,2
        cache.update(0, &blk(3.0, 4), &blk(3.0, 4)).unwrap(); // positions 3,4,5,6
        assert_eq!(cache.offset(), 7);

        cache.truncate(5).unwrap(); // keep 0..5 — cuts inside the second block
        assert_eq!(cache.offset(), 5);
        let (k, _) = cache.peek(0).unwrap().unwrap();
        assert_eq!(host(&k), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        cache.truncate(0).unwrap(); // drop everything
        assert_eq!(cache.offset(), 0);
        // Parity with `ContiguousKvCache::truncate(0)`: a populated layer stays non-empty with an
        // empty dense slot — `peek` returns `Some((empty, empty))`, not `None`.
        let (k, _) = cache.peek(0).unwrap().unwrap();
        assert_eq!(k.shape()[SEQ_AXIS as usize], 0);
        assert_eq!(k.size(), 0);
    }

    #[test]
    fn quantized_reset_clears_blocks_and_sink() {
        let mut cache =
            QuantizedKvCache::with_sink(1, IdentityQuantizer, SinkConfig::keep_first(2));
        cache
            .update(0, &arange4(1, 1, 4, 1), &arange4(1, 1, 4, 1))
            .unwrap();
        assert_eq!(cache.offset(), 4);
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert!(cache.peek(0).is_none());
    }

    /// The sink config keeps the first N positions dense, but the decoded view is unchanged — it
    /// still matches the contiguous cache exactly (sink is a storage-layout knob, not a behavior
    /// change for the identity quantizer).
    #[test]
    fn sink_config_keeps_first_n_dense_but_view_is_identical() {
        let mut sinkful =
            QuantizedKvCache::with_sink(1, IdentityQuantizer, SinkConfig::keep_first(2));
        let mut contig = ContiguousKvCache::new(1);
        // Prefill 1 token, then 1, then 5 — sink fills across the first two updates, rest compresses.
        let steps = [
            arange4(1, 2, 1, 4),
            arange4(1, 2, 1, 4),
            arange4(1, 2, 5, 4),
        ];
        for s in &steps {
            let (qk, _) = sinkful.update(0, s, s).unwrap();
            let (ck, _) = contig.update(0, s, s).unwrap();
            assert_eq!(host(&qk), host(&ck));
        }
        assert_eq!(sinkful.sink_config().sink_tokens, 2);
        assert_eq!(sinkful.offset(), 7);
        // truncate below the sink boundary still works (drops blocks and trims the sink).
        sinkful.truncate(1).unwrap();
        contig.truncate(1).unwrap();
        let (qk, _) = sinkful.peek(0).unwrap().unwrap();
        let (ck, _) = contig.peek(0).unwrap();
        assert_eq!(host(&qk), host(ck));
        assert_eq!(sinkful.offset(), 1);
    }

    /// The sliding-window knob is config-only on the identity path: it is stored, normalized
    /// (non-positive == unset), and reachable, but does not evict — so the decoded view stays
    /// array-identical to a `ContiguousKvCache`.
    #[test]
    fn window_config_is_stored_and_view_is_identical() {
        let cfg = SinkConfig::keep_first(2).with_window(3);
        assert_eq!(cfg.window, Some(3));
        assert_eq!(cfg.window(), Some(3));
        // Non-positive window normalizes to None.
        assert_eq!(SinkConfig::keep_first(0).with_window(0).window(), None);
        assert_eq!(SinkConfig::NONE.window(), None);

        let mut windowed = QuantizedKvCache::with_sink(1, IdentityQuantizer, cfg);
        let mut contig = ContiguousKvCache::new(1);
        let steps = [
            arange4(1, 2, 1, 4),
            arange4(1, 2, 1, 4),
            arange4(1, 2, 5, 4), // total 7 positions > window of 3
        ];
        for s in &steps {
            let (qk, _) = windowed.update(0, s, s).unwrap();
            let (ck, _) = contig.update(0, s, s).unwrap();
            assert_eq!(host(&qk), host(&ck)); // identity path does not evict
        }
        assert_eq!(windowed.sink_config().window(), Some(3));
        assert_eq!(windowed.offset(), contig.offset()); // no eviction: all 7 retained
    }

    /// The cache is usable purely through `&mut dyn KvCache`, including the `as_any_mut` downcast.
    #[test]
    fn quantized_usable_as_dyn_and_downcasts() {
        let mut cache = QuantizedKvCache::new(1, IdentityQuantizer);
        {
            let dynref: &mut dyn KvCache = &mut cache;
            let k = arange4(1, 1, 2, 1);
            dynref.update(0, &k, &k).unwrap();
            assert_eq!(dynref.offset(), 2);
            let downcast = dynref
                .as_any_mut()
                .downcast_mut::<QuantizedKvCache<IdentityQuantizer>>()
                .expect("downcast to concrete quantized cache");
            assert_eq!(downcast.sink_config(), SinkConfig::NONE);
        }
    }
}
