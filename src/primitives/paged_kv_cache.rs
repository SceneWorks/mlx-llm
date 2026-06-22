//! Paged KV cache — strategy A (gather-then-SDPA), epic 7153 story 7169.
//!
//! PagedAttention-style KV management without a custom kernel. Each sequence's keys/values live in
//! fixed-size **blocks** drawn from a shared [`BlockPool`]; a per-sequence **block table** records
//! which physical blocks hold its tokens, and blocks are allocated on demand. Before attention the
//! sequence's blocks are **gathered** back into a contiguous tensor and fed to the stock
//! [`scaled_dot_product_attention`](mlx_rs::fast::scaled_dot_product_attention) — so the cache is a
//! drop-in behind the [`KvCache`](crate::primitives::KvCache) trait and the decoder never changes.
//! The custom Metal kernel that reads scattered blocks directly (removing the gather) is the perf
//! follow-up, story 7170.
//!
//! ## Why paging
//! A growing-concat cache reserves nothing it does not use, but it also cannot **share** storage. The
//! pool's two wins:
//! - **No max-context reservation**: a sequence holds `ceil(len / block_size)` blocks, never a
//!   pre-reserved `max_context` slab — [`PagedKvCache::reserved_tokens`] vs a naive allocator is the
//!   measured saving.
//! - **Copy-on-write prefix sharing**: a full block is immutable once frozen, so sequences sharing a
//!   prompt prefix point at the **same physical blocks** ([`PagedKvCache::new_seeded`]); only each
//!   sequence's private partial *tail* ever diverges, so no block is ever copied mid-write.
//!
//! ## Correctness
//! Gather returns `concat(frozen blocks, tail)` — exactly the same per-position keys/values a
//! contiguous cache holds, in the same order — so a sequence decoded with a paged cache is
//! **token-for-token identical** to one decoded with [`ContiguousKvCache`](super::ContiguousKvCache).
//! Per-sequence caches mean each sequence attends only its own real keys (no padding mask), so
//! differing-length sequences are handled bit-exactly.
//!
//! Block id lifetimes — allocation, recycling, and the copy-on-write reference counts — are the
//! backend-neutral [`core_llm::paging::BlockAllocator`] policy; this module adds only the per-id MLX
//! tensor storage and the gather. The pool is `Rc<RefCell<…>>`-shared and single-threaded,
//! consistent with the engine's MLX device (instances are neither `Send` nor `Sync`).

use std::cell::RefCell;
use std::rc::Rc;

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use core_llm::paging::BlockAllocator;

use crate::error::{Error, Result};
use crate::primitives::kv_cache::{KvCache, SEQ_AXIS};

/// One physical block: per-layer keys and values for a contiguous run of up to `block_size` token
/// positions of a single sequence. Frozen blocks are exactly `block_size` long and never mutated
/// (which is what makes copy-on-write sharing free).
#[derive(Debug)]
struct PhysBlock {
    /// Per layer, `[1, n_kv_heads, block_size, head_dim]` (keys already-RoPE'd).
    k: Vec<Array>,
    /// Per layer, `[1, n_kv_heads, block_size, head_dim]`.
    v: Vec<Array>,
}

/// A pool of fixed-size physical KV blocks, shared by the [`PagedKvCache`]s that draw from it. Block
/// id lifetimes (allocation, recycling, copy-on-write reference counts) are the backend-neutral
/// [`core_llm::paging::BlockAllocator`] policy; this pool adds the per-id MLX tensor storage.
#[derive(Debug)]
pub struct BlockPool {
    block_size: usize,
    /// Per-block tensors, indexed by allocator id (`None` when the id is free).
    blocks: Vec<Option<PhysBlock>>,
    alloc: BlockAllocator,
}

impl BlockPool {
    /// A pool handing out `block_size`-token blocks.
    pub fn new(block_size: usize) -> Rc<RefCell<Self>> {
        assert!(block_size > 0, "block_size must be > 0");
        Rc::new(RefCell::new(Self {
            block_size,
            blocks: Vec::new(),
            alloc: BlockAllocator::new(),
        }))
    }

    /// Token capacity of one block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks currently live (refcount > 0).
    pub fn live_blocks(&self) -> usize {
        self.alloc.live_blocks()
    }

    /// Number of blocks shared by more than one sequence (refcount > 1) — the copy-on-write win.
    pub fn shared_blocks(&self) -> usize {
        self.alloc.shared_blocks()
    }

    /// High-water mark of simultaneously-live blocks since construction.
    pub fn peak_live_blocks(&self) -> usize {
        self.alloc.peak_live_blocks()
    }

    /// Token slots reserved across all live blocks (`live_blocks * block_size`) — the apples-to-apples
    /// figure to compare against a naive `sequences * max_context` reservation.
    pub fn reserved_tokens(&self) -> usize {
        self.live_blocks() * self.block_size
    }

    /// Allocate a fresh block holding `k`/`v` (per layer), refcount 1, returning its id. The
    /// allocator reuses a freed id when available.
    fn alloc(&mut self, k: Vec<Array>, v: Vec<Array>) -> usize {
        let id = self.alloc.alloc();
        let block = Some(PhysBlock { k, v });
        if id == self.blocks.len() {
            self.blocks.push(block);
        } else {
            self.blocks[id] = block; // recycled id: overwrite the freed slot
        }
        id
    }

    /// Add a reference to `id` (a sequence adopting a shared block).
    fn retain(&mut self, id: usize) {
        self.alloc.retain(id);
    }

    /// Drop a reference to `id`, freeing the block tensors when the last reference goes.
    fn release(&mut self, id: usize) {
        if self.alloc.release(id) {
            self.blocks[id] = None;
        }
    }

    fn block(&self, id: usize) -> &PhysBlock {
        self.blocks[id].as_ref().expect("live block")
    }
}

/// A single sequence's paged KV cache: a block table into a [`BlockPool`] plus a private partial
/// tail of the not-yet-full positions.
///
/// One cache holds one sequence (`batch_size == 1`); pack concurrency as separate caches over a
/// shared pool. Implements [`KvCache`] so it drops into the streaming decode loop unchanged.
#[derive(Debug)]
pub struct PagedKvCache {
    num_layers: usize,
    block_size: usize,
    pool: Rc<RefCell<BlockPool>>,
    /// Physical block ids holding this sequence's full (frozen) blocks, in position order.
    block_ids: Vec<usize>,
    /// Per-layer partial tail (`Some([1, n_kv_heads, tail_len, head_dim])`), positions after the last
    /// full block. `None` per layer until the first token arrives.
    tail_k: Vec<Option<Array>>,
    tail_v: Vec<Option<Array>>,
    /// Tokens in the tail (same across layers); authoritative after a full step (all layers updated).
    tail_len: usize,
    /// Per-layer cached concat of **all frozen blocks** (`concat(block_ids' k/v[layer])`), maintained
    /// incrementally so the per-step [`PagedKvCache::gather`] is a 2-array concat (`frozen + tail`)
    /// instead of an O(blocks) concat. The per-step gather was ~85% of the decode step before this
    /// (sc-7325). `Some` ⟺ current for `block_ids`; `None` means "rebuild on next use" (empty, freshly
    /// seeded, or invalidated by truncate/reset). It duplicates the frozen KV (a contiguous copy
    /// alongside the block tensors) — the throughput-for-memory tradeoff the gather-free paged kernel
    /// (sc-7301) would avoid.
    frozen_k: Vec<Option<Array>>,
    frozen_v: Vec<Option<Array>>,
}

impl PagedKvCache {
    /// A fresh single-sequence paged cache backed by its own pool.
    pub fn new(num_layers: usize, block_size: usize) -> Self {
        Self::with_pool(BlockPool::new(block_size), num_layers)
    }

    /// A fresh single-sequence paged cache drawing from an existing (shared) pool.
    pub fn with_pool(pool: Rc<RefCell<BlockPool>>, num_layers: usize) -> Self {
        let block_size = pool.borrow().block_size;
        Self {
            num_layers,
            block_size,
            pool,
            block_ids: Vec::new(),
            tail_k: vec![None; num_layers],
            tail_v: vec![None; num_layers],
            tail_len: 0,
            frozen_k: vec![None; num_layers],
            frozen_v: vec![None; num_layers],
        }
    }

    /// A cache that **shares** `shared_block_ids` (a prior sequence's frozen prefix blocks) from
    /// `pool`, adopting a reference to each. The new sequence starts positioned at
    /// `shared_block_ids.len() * block_size` and recomputes only its suffix — copy-on-write prefix
    /// reuse with zero block copies.
    pub fn new_seeded(
        pool: Rc<RefCell<BlockPool>>,
        num_layers: usize,
        shared_block_ids: &[usize],
    ) -> Self {
        {
            let mut p = pool.borrow_mut();
            for &id in shared_block_ids {
                p.retain(id);
            }
        }
        let mut cache = Self::with_pool(pool, num_layers);
        cache.block_ids = shared_block_ids.to_vec();
        cache
    }

    /// The pool this cache draws from (for accounting / seeding sibling sequences).
    pub fn pool(&self) -> &Rc<RefCell<BlockPool>> {
        &self.pool
    }

    /// The frozen block ids covering this sequence's first `tokens` positions — the shareable prefix
    /// for [`PagedKvCache::new_seeded`]. Rounded **down** to a whole number of blocks (a partial
    /// block is private and not shareable).
    pub fn shareable_prefix_blocks(&self, tokens: usize) -> Vec<usize> {
        let n = tokens / self.block_size;
        self.block_ids[..n.min(self.block_ids.len())].to_vec()
    }

    /// Number of frozen (full) blocks this sequence holds.
    pub fn blocks(&self) -> usize {
        self.block_ids.len()
    }

    /// Token slots this sequence reserves: full blocks plus (if non-empty) one block for the tail —
    /// i.e. real paged allocation, at most `block_size - 1` over its true length.
    pub fn reserved_tokens(&self) -> usize {
        (self.block_ids.len() + usize::from(self.tail_len > 0)) * self.block_size
    }

    /// Logical token length of the sequence.
    fn len(&self) -> usize {
        self.block_ids.len() * self.block_size + self.tail_len
    }

    /// Append this step's `keys`/`values` (`[1, n_kv_heads, step, head_dim]`) to `layer`'s tail.
    fn append_tail(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<()> {
        self.tail_k[layer] = Some(match self.tail_k[layer].take() {
            Some(t) => concatenate_axis(&[&t, keys], SEQ_AXIS)?,
            None => keys.clone(),
        });
        self.tail_v[layer] = Some(match self.tail_v[layer].take() {
            Some(t) => concatenate_axis(&[&t, values], SEQ_AXIS)?,
            None => values.clone(),
        });
        Ok(())
    }

    /// Freeze whole blocks off the front of every layer's tail until the tail is under `block_size`.
    /// Called once per step (after the last layer appended), so all layers carry the same tokens.
    fn freeze_full_blocks(&mut self, new_tokens: usize) -> Result<()> {
        self.tail_len += new_tokens;
        let bs = self.block_size as i32;
        while self.tail_len >= self.block_size {
            let mut bk = Vec::with_capacity(self.num_layers);
            let mut bv = Vec::with_capacity(self.num_layers);
            for l in 0..self.num_layers {
                let tk = self.tail_k[l].take().expect("tail present at freeze");
                let tv = self.tail_v[l].take().expect("tail present at freeze");
                let (hk, rk) = split_seq(&tk, bs)?;
                let (hv, rv) = split_seq(&tv, bs)?;
                // Maintain the frozen-blocks concat: ensure it reflects the prior blocks, then append
                // this newly frozen block (a 2-array concat) so gather never re-concats every block.
                self.ensure_frozen(l)?;
                self.frozen_k[l] = Some(append_seq(self.frozen_k[l].take(), &hk)?);
                self.frozen_v[l] = Some(append_seq(self.frozen_v[l].take(), &hv)?);
                bk.push(hk);
                bv.push(hv);
                self.tail_k[l] = keep_nonempty(rk);
                self.tail_v[l] = keep_nonempty(rv);
            }
            let id = self.pool.borrow_mut().alloc(bk, bv);
            self.block_ids.push(id);
            self.tail_len -= self.block_size;
        }
        Ok(())
    }

    /// Gather `layer`'s full keys/values — `concat(frozen blocks, tail)` — into one contiguous
    /// `[1, n_kv_heads, len, head_dim]` pair to attend over. The frozen-blocks half is the cached
    /// [`frozen_k`]/[`frozen_v`] concat (built/maintained incrementally), so this is a **2-array**
    /// concat per step, not an O(blocks) one (sc-7325).
    ///
    /// [`frozen_k`]: PagedKvCache::frozen_k
    /// [`frozen_v`]: PagedKvCache::frozen_v
    fn gather(&mut self, layer: usize) -> Result<(Array, Array)> {
        self.ensure_frozen(layer)?;
        let k = combine_seq(self.frozen_k[layer].as_ref(), self.tail_k[layer].as_ref())?;
        let v = combine_seq(self.frozen_v[layer].as_ref(), self.tail_v[layer].as_ref())?;
        Ok((k, v))
    }

    /// Materialize `layer`'s frozen-blocks concat from the block table when it is missing (a freshly
    /// [`new_seeded`](PagedKvCache::new_seeded) prefix, or after a `truncate`/`reset` invalidation).
    /// A no-op once cached or when there are no frozen blocks. This is the only O(blocks) concat, and
    /// it runs at most once per invalidation — the steady state appends one block at a time.
    fn ensure_frozen(&mut self, layer: usize) -> Result<()> {
        if self.frozen_k[layer].is_some() || self.block_ids.is_empty() {
            return Ok(());
        }
        let pool = self.pool.borrow();
        let ks: Vec<&Array> = self.block_ids.iter().map(|&id| &pool.block(id).k[layer]).collect();
        let vs: Vec<&Array> = self.block_ids.iter().map(|&id| &pool.block(id).v[layer]).collect();
        self.frozen_k[layer] = Some(concat_or_clone(&ks)?);
        self.frozen_v[layer] = Some(concat_or_clone(&vs)?);
        Ok(())
    }

    /// Invalidate every layer's cached frozen-blocks concat (after the block table changes in a way
    /// that is not a simple append — `truncate`/`reset`). The next [`gather`](PagedKvCache::gather)
    /// rebuilds it from the current blocks.
    fn invalidate_frozen(&mut self) {
        for l in 0..self.num_layers {
            self.frozen_k[l] = None;
            self.frozen_v[l] = None;
        }
    }
}

impl KvCache for PagedKvCache {
    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        if keys.shape()[0] != 1 {
            return Err(Error::Msg(format!(
                "PagedKvCache is single-sequence (batch 1); got batch {}",
                keys.shape()[0]
            )));
        }
        let step = keys.shape()[SEQ_AXIS as usize] as usize;
        self.append_tail(layer, keys, values)?;
        // The block layout advances once per step; do it after the final layer, when every layer has
        // this step's tokens, so a frozen block carries all layers consistently.
        if layer + 1 == self.num_layers {
            self.freeze_full_blocks(step)?;
        }
        self.gather(layer)
    }

    fn offset(&self) -> i32 {
        self.len() as i32
    }

    fn batch_size(&self) -> i32 {
        i32::from(!self.block_ids.is_empty() || self.tail_len > 0 || self.tail_k.iter().any(Option::is_some))
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        // Single-sequence: the only valid non-empty keep is `[0]` (a no-op); an empty keep drops it.
        match keep {
            [] => self.reset(),
            [0] => {}
            other => {
                return Err(Error::Msg(format!(
                    "PagedKvCache is single-sequence; retain_sequences expects [] or [0], got {other:?}"
                )))
            }
        }
        Ok(())
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        if len < 0 {
            return Err(Error::Msg(format!("truncate: negative len {len}")));
        }
        let len = len as usize;
        if len >= self.len() {
            return Ok(()); // already at/under the target length
        }
        let full = self.block_ids.len() * self.block_size;
        if len >= full {
            // Truncation falls within the partial tail: slice every layer's tail to the remainder
            // (tail is present here, since len < total ⇒ tail_len > 0).
            let new_tail = len - full;
            for l in 0..self.num_layers {
                if new_tail == 0 {
                    self.tail_k[l] = None;
                    self.tail_v[l] = None;
                } else {
                    let k = slice_prefix(self.tail_k[l].as_ref().expect("tail present"), new_tail as i32)?;
                    let v = slice_prefix(self.tail_v[l].as_ref().expect("tail present"), new_tail as i32)?;
                    self.tail_k[l] = Some(k);
                    self.tail_v[l] = Some(v);
                }
            }
            self.tail_len = new_tail;
            return Ok(());
        }
        // Truncation drops whole blocks (and may unfreeze part of one into a fresh private tail).
        let keep_full = len / self.block_size;
        let rem = len % self.block_size;
        // Unfreeze the remainder of the boundary block into the tail *before* releasing anything.
        if rem > 0 {
            let id = self.block_ids[keep_full];
            let pool = self.pool.borrow();
            let block = pool.block(id);
            let mut nk = Vec::with_capacity(self.num_layers);
            let mut nv = Vec::with_capacity(self.num_layers);
            for l in 0..self.num_layers {
                nk.push(slice_prefix(&block.k[l], rem as i32)?);
                nv.push(slice_prefix(&block.v[l], rem as i32)?);
            }
            drop(pool);
            for l in 0..self.num_layers {
                self.tail_k[l] = Some(nk[l].clone());
                self.tail_v[l] = Some(nv[l].clone());
            }
            self.tail_len = rem;
        } else {
            self.tail_k = vec![None; self.num_layers];
            self.tail_v = vec![None; self.num_layers];
            self.tail_len = 0;
        }
        // Release the dropped blocks (everything from keep_full on) and shrink the table.
        {
            let mut pool = self.pool.borrow_mut();
            for &id in &self.block_ids[keep_full..] {
                pool.release(id);
            }
        }
        self.block_ids.truncate(keep_full);
        // The block table changed by more than an append; rebuild the frozen concat on next gather.
        self.invalidate_frozen();
        Ok(())
    }

    fn reset(&mut self) {
        {
            let mut pool = self.pool.borrow_mut();
            for &id in &self.block_ids {
                pool.release(id);
            }
        }
        self.block_ids.clear();
        self.tail_k = vec![None; self.num_layers];
        self.tail_v = vec![None; self.num_layers];
        self.tail_len = 0;
        self.invalidate_frozen();
    }
}

impl Drop for PagedKvCache {
    fn drop(&mut self) {
        // Release this sequence's blocks so a shared pool reclaims them (and shared prefixes survive
        // until their last referent drops).
        let mut pool = self.pool.borrow_mut();
        for &id in &self.block_ids {
            pool.release(id);
        }
    }
}

/// Split `a` along the sequence axis at `at`: `(a[..at], a[at..])`.
fn split_seq(a: &Array, at: i32) -> Result<(Array, Array)> {
    let total = a.shape()[SEQ_AXIS as usize];
    let head = Array::from_slice(&(0..at).collect::<Vec<_>>(), &[at]);
    let tail = Array::from_slice(&(at..total).collect::<Vec<_>>(), &[total - at]);
    Ok((a.take_axis(&head, SEQ_AXIS)?, a.take_axis(&tail, SEQ_AXIS)?))
}

/// The first `n` positions of `a` along the sequence axis (`a[..n]`). `n` must be `> 0` and `<=` the
/// sequence length.
fn slice_prefix(a: &Array, n: i32) -> Result<Array> {
    let idx = Array::from_slice(&(0..n).collect::<Vec<_>>(), &[n]);
    Ok(a.take_axis(&idx, SEQ_AXIS)?)
}

/// `None` if the array is empty along the sequence axis, else `Some(array)`.
fn keep_nonempty(a: Array) -> Option<Array> {
    if a.shape()[SEQ_AXIS as usize] == 0 {
        None
    } else {
        Some(a)
    }
}

/// Concatenate one-or-more arrays along the sequence axis, cloning (no copy — MLX is refcounted) when
/// there is a single one.
fn concat_or_clone(parts: &[&Array]) -> Result<Array> {
    Ok(match parts {
        [single] => (*single).clone(),
        many => concatenate_axis(many, SEQ_AXIS)?,
    })
}

/// Append `new` to `base` along the sequence axis (`base` absent ⇒ just `new`).
fn append_seq(base: Option<Array>, new: &Array) -> Result<Array> {
    Ok(match base {
        Some(f) => concatenate_axis(&[&f, new], SEQ_AXIS)?,
        None => new.clone(),
    })
}

/// Join the frozen-blocks concat and the partial tail into the full sequence (`[1, n_kv_heads, len,
/// head_dim]`). At least one side is present whenever a sequence holds any tokens.
fn combine_seq(frozen: Option<&Array>, tail: Option<&Array>) -> Result<Array> {
    match (frozen, tail) {
        (Some(f), Some(t)) => Ok(concatenate_axis(&[f, t], SEQ_AXIS)?),
        (Some(f), None) => Ok(f.clone()),
        (None, Some(t)) => Ok(t.clone()),
        (None, None) => Err(Error::Msg("gather on an empty paged cache".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    /// `[1, h, s, d]` of sequential f32 values starting at `base`, for order/equality checks.
    fn seq(h: i32, s: i32, d: i32, base: f32) -> Array {
        let n = (h * s * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| base + i as f32).collect();
        Array::from_slice(&data, &[1, h, s, d])
    }

    fn host(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec()
    }

    #[test]
    fn single_step_under_block_stays_in_tail() {
        let mut c = PagedKvCache::new(1, 4);
        let k = seq(2, 3, 2, 0.0); // 3 tokens, block_size 4 -> no freeze
        let (ka, _) = c.update(0, &k, &k).unwrap();
        assert_eq!(ka.shape(), &[1, 2, 3, 2]);
        assert_eq!(c.offset(), 3);
        assert_eq!(c.block_ids.len(), 0, "nothing frozen yet");
        assert_eq!(c.pool().borrow().live_blocks(), 0);
    }

    #[test]
    fn crossing_block_boundary_freezes_and_gather_preserves_order() {
        let mut c = PagedKvCache::new(1, 2);
        // First step: 2 tokens -> exactly one full block, empty tail.
        let k0 = seq(1, 2, 1, 0.0); // values [0, 1]
        let (g0, _) = c.update(0, &k0, &k0).unwrap();
        assert_eq!(host(&g0), vec![0.0, 1.0]);
        assert_eq!(c.block_ids.len(), 1);
        assert_eq!(c.tail_len, 0);
        // Second step: 3 tokens -> one more full block + a 1-token tail; total 5.
        let k1 = seq(1, 3, 1, 2.0); // values [2, 3, 4]
        let (g1, _) = c.update(0, &k1, &k1).unwrap();
        assert_eq!(c.offset(), 5);
        assert_eq!(c.block_ids.len(), 2);
        assert_eq!(c.tail_len, 1);
        assert_eq!(host(&g1), vec![0.0, 1.0, 2.0, 3.0, 4.0], "gather is in position order");
    }

    #[test]
    fn matches_contiguous_cache_step_by_step() {
        use crate::primitives::ContiguousKvCache;
        let mut paged = PagedKvCache::new(2, 4);
        let mut contig = ContiguousKvCache::new(2);
        let mut off = 0.0;
        for step in [5, 1, 1, 4, 1] {
            for layer in 0..2 {
                let k = seq(2, step, 3, off + layer as f32 * 100.0);
                let v = seq(2, step, 3, off + 50.0 + layer as f32 * 100.0);
                let (pk, pv) = paged.update(layer, &k, &v).unwrap();
                let (ck, cv) = contig.update(layer, &k, &v).unwrap();
                assert_eq!(host(&pk), host(&ck), "step {step} layer {layer} keys");
                assert_eq!(host(&pv), host(&cv), "step {step} layer {layer} values");
            }
            off += (step * 6) as f32;
        }
        assert_eq!(paged.offset(), contig.offset());
    }

    #[test]
    fn seeded_prefix_then_freeze_matches_contiguous() {
        use crate::primitives::ContiguousKvCache;
        // Sequence A: 4 tokens => 2 frozen blocks (block_size 2), 2 layers, distinct per-layer values.
        let pool = BlockPool::new(2);
        let mut a = PagedKvCache::with_pool(pool.clone(), 2);
        for layer in 0..2 {
            let k = seq(1, 4, 1, layer as f32 * 100.0);
            a.update(layer, &k, &k).unwrap();
        }
        let shared = a.shareable_prefix_blocks(4);
        assert_eq!(shared.len(), 2);

        // Seed B from A's two prefix blocks, then decode a 3-token suffix per layer — crossing a block
        // boundary so a new block freezes while the seeded blocks' frozen concat is still unbuilt
        // (the ensure_frozen-inside-freeze path). Must equal a contiguous cache holding prefix+suffix.
        let mut b = PagedKvCache::new_seeded(pool.clone(), 2, &shared);
        let mut contig = ContiguousKvCache::new(2);
        for layer in 0..2 {
            let prefix = seq(1, 4, 1, layer as f32 * 100.0); // identical to A's stored values
            let suffix = seq(1, 3, 1, layer as f32 * 100.0 + 4.0);
            contig.update(layer, &prefix, &prefix).unwrap();
            let (ck, cv) = contig.update(layer, &suffix, &suffix).unwrap();
            let (bk, bv) = b.update(layer, &suffix, &suffix).unwrap();
            assert_eq!(host(&bk), host(&ck), "layer {layer}: seeded prefix + frozen suffix keys");
            assert_eq!(host(&bv), host(&cv), "layer {layer}: seeded prefix + frozen suffix values");
        }
        assert_eq!(b.offset(), 7, "4 seeded + 3 suffix");
        assert_eq!(b.block_ids.len(), 3, "2 seeded + 1 newly frozen block");
    }

    #[test]
    fn reserved_tokens_tracks_blocks_not_max_context() {
        let mut c = PagedKvCache::new(1, 16);
        let k = seq(1, 20, 1, 0.0); // 20 tokens -> 1 full block + 4-token tail
        c.update(0, &k, &k).unwrap();
        assert_eq!(c.block_ids.len(), 1);
        assert_eq!(c.tail_len, 4);
        // Reserved = (1 full + 1 tail block) * 16 = 32; far below a naive max_context (e.g. 2048).
        assert_eq!(c.reserved_tokens(), 32);
        assert!(c.reserved_tokens() < 2048);
    }

    #[test]
    fn shared_prefix_blocks_are_refcounted_not_copied() {
        let pool = BlockPool::new(2);
        let mut a = PagedKvCache::with_pool(pool.clone(), 1);
        // 4 tokens -> 2 full shared-able blocks.
        let k = seq(1, 4, 1, 0.0);
        a.update(0, &k, &k).unwrap();
        assert_eq!(pool.borrow().live_blocks(), 2);
        assert_eq!(pool.borrow().shared_blocks(), 0);

        // A sibling sequence adopts both prefix blocks (copy-on-write share).
        let shared = a.shareable_prefix_blocks(4);
        assert_eq!(shared.len(), 2);
        let mut b = PagedKvCache::new_seeded(pool.clone(), 1, &shared);
        assert_eq!(b.offset(), 4, "seeded sequence starts past the shared prefix");
        assert_eq!(pool.borrow().live_blocks(), 2, "no new blocks: prefix is shared");
        assert_eq!(pool.borrow().shared_blocks(), 2);

        // B diverges in its own private tail; the shared full blocks are untouched (refcount stays 2).
        let bk = seq(1, 1, 1, 99.0);
        b.update(0, &bk, &bk).unwrap();
        assert_eq!(pool.borrow().shared_blocks(), 2, "divergence touches only the private tail");
        let (bg, _) = b.gather(0).unwrap();
        assert_eq!(host(&bg), vec![0.0, 1.0, 2.0, 3.0, 99.0], "shared prefix + private suffix");

        // Dropping B releases its references; the shared blocks return to refcount 1 (still A's).
        drop(b);
        assert_eq!(pool.borrow().shared_blocks(), 0);
        assert_eq!(pool.borrow().live_blocks(), 2);
    }

    #[test]
    fn truncate_within_tail_and_across_blocks() {
        let mut c = PagedKvCache::new(1, 4);
        let k = seq(1, 10, 1, 0.0); // values 0..9 -> blocks [0..3][4..7] + tail [8,9]
        c.update(0, &k, &k).unwrap();
        assert_eq!(c.offset(), 10);

        // Case A: within the tail.
        c.truncate(9).unwrap();
        assert_eq!(c.offset(), 9);
        assert_eq!(host(&c.gather(0).unwrap().0), (0..9).map(|x| x as f32).collect::<Vec<_>>());

        // Case B: drop into a block, unfreezing its remainder into a fresh tail.
        c.truncate(5).unwrap();
        assert_eq!(c.offset(), 5);
        assert_eq!(host(&c.gather(0).unwrap().0), (0..5).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(c.pool().borrow().live_blocks(), 1, "the dropped block is freed");

        // Case A again, landing exactly on a block boundary (empty tail).
        c.truncate(4).unwrap();
        assert_eq!(c.offset(), 4);
        assert_eq!(host(&c.gather(0).unwrap().0), (0..4).map(|x| x as f32).collect::<Vec<_>>());

        // No-op for len >= current length.
        c.truncate(100).unwrap();
        assert_eq!(c.offset(), 4);
    }

    #[test]
    fn reset_frees_blocks_back_to_the_pool() {
        let pool = BlockPool::new(2);
        let mut c = PagedKvCache::with_pool(pool.clone(), 1);
        let k = seq(1, 4, 1, 0.0);
        c.update(0, &k, &k).unwrap();
        assert_eq!(pool.borrow().live_blocks(), 2);
        c.reset();
        assert_eq!(pool.borrow().live_blocks(), 0);
        assert_eq!(c.offset(), 0);
    }

    #[test]
    fn single_sequence_rejects_batched_update() {
        let mut c = PagedKvCache::new(1, 4);
        let k = Array::from_slice(&[0.0f32; 8], &[2, 2, 1, 2]); // batch 2
        assert!(c.update(0, &k, &k).is_err());
    }
}
