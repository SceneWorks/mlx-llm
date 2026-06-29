//! KV-cache quantization benchmark + observer harness (story sc-8534, epic sc-8528).
//!
//! Ports VeloxQuant-MLX's `observers/` (distortion, latency, memory) into a **method-agnostic**
//! Rust harness so a KV-compression method's quality/memory/throughput are **measured**, not
//! asserted (the epic's success criterion). The harness wraps the [`KvCache`](crate::primitives::KvCache)
//! path: it drives a fixed synthetic prompt set at several context lengths through a registered set
//! of cache builders, measures three things against a dense baseline, and emits a comparison table
//! (method × {KV-memory, quality-delta, tok/s}).
//!
//! ## What is measured
//! - **Distortion** ([`DistortionObserver`]): per-method reconstruction error of the cache's decoded
//!   keys/values vs. the dense baseline's, accumulated as mean-squared error (MSE) over **both K and
//!   V across every layer** at every update — the mean over the full KV path, so a lossy method's
//!   true error is captured. A lossless method (the
//!   [`IdentityQuantizer`](crate::primitives::IdentityQuantizer)) reports ~0.
//! - **Resident KV-memory** ([`KvCache::resident_bytes`](crate::primitives::KvCache::resident_bytes)):
//!   the bytes the cache **actually stores** (summed `nbytes` of its compressed-or-dense
//!   representation), sampled at a quiescent point after the fill loop completes. This is faithful to
//!   a method's compressed footprint and immune to the transient full-context concats + allocator
//!   fragmentation that dominate an active-memory high-water mark (which masked a compressing method's
//!   benefit). The dense baseline's footprint is the reference; a method that stores fewer bytes
//!   reports a lower footprint (and a >1× compression ratio); identity reports ~1.0× (exactly equal).
//! - **Throughput** ([`LatencyObserver`]): wall-clock tokens/sec of the update path (the per-step
//!   `update` + decode-and-materialize that a decoder pays), averaged over the swept steps after a
//!   discarded warm-up step so the column is reproducible.
//!
//! ## Method-agnostic by construction
//! The driver takes a `Vec<Method>` where each [`Method`] is `{ name, builder }` and `builder` is a
//! closure `Fn(num_layers) -> Box<dyn KvCache>`. The dense baseline is just another method
//! (`ContiguousKvCache`). Registering RVQ later (story D) is one line:
//! `Method::new("rvq", |n| Box::new(QuantizedKvCache::new(n, RvqQuantizer::new(...))))`. The harness
//! never names a concrete quantizer.
//!
//! At this point the only available cache impls are
//! [`ContiguousKvCache`](crate::primitives::ContiguousKvCache) (the dense baseline) and
//! `QuantizedKvCache<IdentityQuantizer>` (story B); the `#[ignore]`d end-to-end driver test below
//! validates the harness with exactly those two — identity must show ~0 quality delta and ~1.0×
//! resident memory vs dense.
//!
//! ## Memory safety
//! Unbounded MLX sweeps crash the machine, so the harness is bounded by default
//! ([`BenchConfig::small`]): few short contexts, batch 1, small head geometry. It installs an MLX
//! memory limit + cache-cap ([`BenchConfig::apply_mlx_limits`]), and **clears the MLX buffer cache
//! between every (method, context) run** so freed KV returns to the OS instead of accumulating.
//! Larger sweeps are opt-in by constructing a wider [`BenchConfig`].

use std::collections::BTreeMap;
use std::time::Instant;

use mlx_rs::ops::{mean, square, subtract};
use mlx_rs::random::{key, normal};
use mlx_rs::transforms::eval;
use mlx_rs::{memory, Array, Dtype};

use crate::error::Result;
use crate::primitives::KvCache;

// ---------------------------------------------------------------------------------------------
// Observers (ported from VeloxQuant-MLX `veloxquant_mlx/observers/`)
// ---------------------------------------------------------------------------------------------

/// Accumulates mean-squared reconstruction error of a method's decoded KV vs. the dense baseline's,
/// the Rust analogue of VeloxQuant's `DistortionObserver`. Each [`observe`](Self::observe) call adds
/// the per-element MSE between the reference (dense) tensor and the candidate (method) tensor for one
/// update; [`report`](Self::report) returns the mean over all observed updates.
///
/// Both arrays must have identical shape — they are the same logical keys/values, one routed through
/// the dense cache and one through the method's cache. A lossless method yields `~0`.
#[derive(Debug, Default)]
pub struct DistortionObserver {
    mse_sum: f64,
    n: usize,
}

impl DistortionObserver {
    /// A fresh observer with no samples.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate the mean-squared error between a `reference` (dense) tensor and a `candidate`
    /// (method) tensor for one update. Shapes must match. The MSE is computed in f32 on-device and
    /// read back as a single scalar (cheap: one reduction per update).
    pub fn observe(&mut self, reference: &Array, candidate: &Array) -> Result<()> {
        let r = reference.as_dtype(Dtype::Float32)?;
        let c = candidate.as_dtype(Dtype::Float32)?;
        let diff = subtract(&r, &c)?;
        let mse = mean(square(&diff)?, None)?;
        eval([&mse])?;
        self.mse_sum += mse.item::<f32>() as f64;
        self.n += 1;
        Ok(())
    }

    /// Mean MSE over all observed updates (`0.0` if nothing was observed).
    pub fn mean_mse(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.mse_sum / self.n as f64
        }
    }

    /// Number of updates observed.
    pub fn samples(&self) -> usize {
        self.n
    }
}

/// Records per-step wall-clock samples and reports tokens/sec, the Rust analogue of VeloxQuant's
/// `LatencyObserver`. Each sample is `(tokens_processed, elapsed)`; [`tokens_per_sec`](Self::tokens_per_sec)
/// is total tokens over total elapsed.
#[derive(Debug, Default)]
pub struct LatencyObserver {
    tokens: u64,
    elapsed_secs: f64,
    samples: usize,
}

impl LatencyObserver {
    /// A fresh observer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one timed step that processed `tokens` positions in `elapsed_secs` seconds.
    pub fn record(&mut self, tokens: u64, elapsed_secs: f64) {
        self.tokens += tokens;
        self.elapsed_secs += elapsed_secs;
        self.samples += 1;
    }

    /// Aggregate throughput in tokens/sec (`0.0` if no time elapsed).
    pub fn tokens_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.tokens as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    /// Number of timed steps recorded.
    pub fn samples(&self) -> usize {
        self.samples
    }
}

/// Tracks the MLX active-memory high-water mark across a run, the Rust analogue of VeloxQuant's
/// `MemoryObserver`. Construct it with [`start`](Self::start) (which resets the MLX peak counter and
/// records the active-memory baseline), drive the workload, then call [`peak_delta_bytes`](Self::peak_delta_bytes)
/// for the peak attributable to this run (MLX peak minus the entry baseline).
#[derive(Debug)]
pub struct MemoryObserver {
    baseline: usize,
}

impl MemoryObserver {
    /// Reset the MLX peak counter and snapshot the active-memory baseline. Call immediately before the
    /// workload whose peak you want to attribute.
    pub fn start() -> Self {
        memory::reset_peak_memory();
        Self {
            baseline: memory::get_active_memory(),
        }
    }

    /// Peak MLX-resident bytes attributable to the workload: the MLX peak high-water mark since
    /// [`start`](Self::start) minus the active-memory baseline at start (saturating at 0).
    pub fn peak_delta_bytes(&self) -> usize {
        memory::get_peak_memory().saturating_sub(self.baseline)
    }
}

// ---------------------------------------------------------------------------------------------
// Method registry (the method-agnostic seam)
// ---------------------------------------------------------------------------------------------

/// A cache builder: given the number of decoder layers, produce a boxed [`KvCache`]. This is the only
/// thing the harness knows about a "method" — it never names a concrete [`Quantizer`](crate::primitives::Quantizer).
pub type CacheBuilder = Box<dyn Fn(usize) -> Box<dyn KvCache>>;

/// A registered benchmark method: a display `name` and a [`CacheBuilder`]. Add a method by
/// constructing one of these; the dense baseline is itself a `Method`.
pub struct Method {
    /// Display name shown in the comparison table (e.g. `"dense"`, `"identity"`, `"rvq-b2"`).
    pub name: String,
    /// Builds a fresh cache of this method for the configured layer count.
    pub builder: CacheBuilder,
}

impl Method {
    /// Register a method from a name and a builder closure.
    ///
    /// ```ignore
    /// Method::new("identity", |n| {
    ///     Box::new(QuantizedKvCache::new(n, IdentityQuantizer))
    /// });
    /// ```
    pub fn new(
        name: impl Into<String>,
        builder: impl Fn(usize) -> Box<dyn KvCache> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            builder: Box::new(builder),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Bench configuration (bounded by default for memory safety)
// ---------------------------------------------------------------------------------------------

/// A bounded benchmark configuration. The defaults ([`small`](Self::small)) are deliberately tiny so
/// a run cannot exhaust machine memory; widen explicitly for an opt-in larger sweep.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Number of decoder layers simulated (each gets its own per-layer store).
    pub num_layers: usize,
    /// Batch size (axis 0). Keep at 1 for the safe default.
    pub batch: i32,
    /// Number of KV heads (axis 1).
    pub n_kv_heads: i32,
    /// Per-head dimension (axis 3).
    pub head_dim: i32,
    /// Number of prompt tokens prefilled in one shot before the decode steps.
    pub prefill_tokens: i32,
    /// Context lengths to sweep — total positions reached (prefill + decode steps). Must have >= 2
    /// entries to satisfy the acceptance criterion. Each entry `>= prefill_tokens`.
    pub context_lengths: Vec<i32>,
    /// MLX backpressure memory limit (bytes) installed by [`apply_mlx_limits`](Self::apply_mlx_limits).
    pub mlx_memory_limit: usize,
    /// MLX buffer-cache cap (bytes) installed by [`apply_mlx_limits`](Self::apply_mlx_limits).
    pub mlx_cache_limit: usize,
}

impl BenchConfig {
    /// The default **small, safe** sweep: 4 layers, batch 1, 4 KV heads, head_dim 64, 16 prefill
    /// tokens, contexts {64, 128}. This is well under any memory risk and still exercises prefill +
    /// multi-step decode across two context lengths.
    pub fn small() -> Self {
        Self {
            num_layers: 4,
            batch: 1,
            n_kv_heads: 4,
            head_dim: 64,
            prefill_tokens: 16,
            context_lengths: vec![64, 128],
            mlx_memory_limit: 8 * 1024 * 1024 * 1024,
            mlx_cache_limit: 1024 * 1024 * 1024,
        }
    }

    /// Install the MLX backpressure + buffer-cache limits from this config. Idempotent; safe to call
    /// once at the start of a run. Returns the previous `(memory_limit, cache_limit)`.
    pub fn apply_mlx_limits(&self) -> (usize, usize) {
        let prev_mem = memory::set_memory_limit(self.mlx_memory_limit);
        let prev_cache = memory::set_cache_limit(self.mlx_cache_limit);
        (prev_mem, prev_cache)
    }

    /// Rough estimate of the dense resident KV bytes for one context length, for the pre-run guard:
    /// `2 (k+v) × layers × batch × heads × ctx × head_dim × 2 bytes (bf16)`.
    pub fn dense_kv_bytes(&self, ctx: i32) -> u64 {
        2u64 * self.num_layers as u64
            * self.batch as u64
            * self.n_kv_heads as u64
            * ctx as u64
            * self.head_dim as u64
            * 2
    }
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

/// One `(method, context)` measurement row.
#[derive(Debug, Clone)]
pub struct BenchRow {
    /// Method display name.
    pub method: String,
    /// Context length (total positions reached) this row was measured at.
    pub context: i32,
    /// Resident KV bytes the cache actually stores at this context (its faithful compressed-or-dense
    /// footprint, [`KvCache::resident_bytes`]) — not a transient/high-water active-memory mark.
    pub resident_memory_bytes: usize,
    /// Mean reconstruction MSE vs. the dense baseline at this context (`0.0` for the baseline itself).
    pub quality_delta_mse: f64,
    /// Update-path throughput in tokens/sec.
    pub tokens_per_sec: f64,
}

/// The full comparison result: every `(method, context)` row plus the configuration it was produced
/// under.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// The config the sweep ran under.
    pub config: BenchConfig,
    /// All measured rows, method-major then context-ascending.
    pub rows: Vec<BenchRow>,
}

/// A synthetic `[batch, n_kv_heads, step, head_dim]` bf16 tensor of normal noise — values are
/// irrelevant to memory/latency, and distortion is measured *relative* to the dense path so any fixed
/// content works. A fixed `seed` makes the dense and method paths see identical inputs.
fn synth(cfg: &BenchConfig, step: i32, seed: u64) -> Result<Array> {
    let shape = [cfg.batch, cfg.n_kv_heads, step, cfg.head_dim];
    let k = key(seed)?;
    let a = normal::<f32>(&shape, None, None, &k)?.as_dtype(Dtype::Bfloat16)?;
    Ok(a)
}

/// The fixed prompt set, as a list of `(prefill, then per-step decode tokens...)` segment lengths for
/// one context length: a single one-shot prefill of `prefill_tokens` then `(ctx - prefill_tokens)`
/// single-token decode steps. Returning the segment plan (rather than driving inline) keeps the dense
/// and method runs perfectly aligned.
fn segment_plan(cfg: &BenchConfig, ctx: i32) -> Vec<i32> {
    let prefill = cfg.prefill_tokens.min(ctx);
    let decode_steps = (ctx - prefill).max(0) as usize;
    let mut segs = Vec::with_capacity(decode_steps + 1);
    segs.push(prefill);
    segs.extend(std::iter::repeat_n(1, decode_steps));
    segs
}

/// Per-segment reference snapshot for the distortion comparison: the dense baseline's decoded
/// `(keys, values)` for **every layer** at one segment. A method run diffs its own decoded K and V,
/// layer-by-layer, against these — so distortion is the mean reconstruction error over the full KV
/// path (both tensors, all layers), not just one slice.
type RefSegment = Vec<(Array, Array)>;

/// Drive one cache through the segment plan for `ctx`, measuring resident KV footprint, throughput
/// and (when matching `reference` segments are supplied) reconstruction error against the dense
/// baseline. Returns `(resident_bytes, tokens_per_sec, mean_mse, decoded_reference)`.
///
/// `reference` is `Some(per-segment dense decoded (K,V) for every layer)` for a method, `None` for
/// the baseline run itself (which *produces* the reference). The same fixed-seed [`synth`] inputs are
/// fed to both, so the only difference in decoded output is the method's compression error.
///
/// **Memory methodology (sc-8534 fix).** The reported number is the cache's *resident* footprint —
/// [`KvCache::resident_bytes`], the summed `nbytes` of the arrays the cache actually stores — sampled
/// at a **quiescent** point: after every update for this config completes, we drop all transient
/// full-context `update()` return arrays and force one MLX `eval` so nothing is left lazily pending,
/// then read the footprint. This is faithful to a method's compressed representation and is immune to
/// the transient full-context concats + allocator fragmentation that dominate an active-memory
/// high-water mark (which masked a compressing method's benefit and made identity read a nonsensical
/// 0.85×).
#[allow(clippy::type_complexity)]
fn run_one(
    cfg: &BenchConfig,
    method: &Method,
    ctx: i32,
    reference: Option<&[RefSegment]>,
) -> Result<(usize, f64, f64, Vec<RefSegment>)> {
    // Fresh cache + clean slate. Clearing the buffer cache BEFORE we start keeps successive runs from
    // accumulating freed KV in the MLX allocator.
    memory::clear_cache();
    let mut cache = (method.builder)(cfg.num_layers);
    let mut lat = LatencyObserver::new();
    let mut dist = DistortionObserver::new();

    let plan = segment_plan(cfg, ctx);
    // Decoded (K, V) per layer per segment — this becomes the reference for later methods, and is
    // what a method run diffs against `reference`.
    let mut decoded: Vec<RefSegment> = Vec::with_capacity(plan.len());

    // Warm-up: run the first segment once and discard its timing so throughput is reproducible (the
    // first update pays one-time graph-build / allocator-warm costs that swing tok/s run-to-run).
    if let Some(&first) = plan.first() {
        let seed = (ctx as u64) << 8;
        let wk = synth(cfg, first, seed)?;
        let wv = synth(cfg, first, seed.wrapping_add(1))?;
        for layer in 0..cfg.num_layers {
            let (kk, vv) = cache.update(layer, &wk, &wv)?;
            eval([&kk, &vv])?;
        }
        cache.reset();
        memory::clear_cache();
    }

    let mut seed = (ctx as u64) << 8;
    for (seg_idx, &step) in plan.iter().enumerate() {
        // Identical synthetic input across dense + method runs (seed depends only on ctx + position).
        let k = synth(cfg, step, seed)?;
        let v = synth(cfg, step, seed.wrapping_add(1))?;
        seed = seed.wrapping_add(2);

        let t0 = Instant::now();
        let mut seg: RefSegment = Vec::with_capacity(cfg.num_layers);
        for layer in 0..cfg.num_layers {
            let (kk, vv) = cache.update(layer, &k, &v)?;
            // Force materialization so timing reflects the real decode-path work, not lazy deferral.
            eval([&kk, &vv])?;
            seg.push((kk, vv));
        }
        let elapsed = t0.elapsed().as_secs_f64();
        lat.record(step as u64, elapsed);

        // Distortion: compare this method's decoded K and V against the dense reference's, over every
        // layer. The mean over the full KV path is the faithful reconstruction error.
        if let Some(refs) = reference {
            if let Some(ref_seg) = refs.get(seg_idx) {
                for ((rk, rv), (ck, cv)) in ref_seg.iter().zip(seg.iter()) {
                    dist.observe(rk, ck)?;
                    dist.observe(rv, cv)?;
                }
            }
        }
        decoded.push(seg);
    }

    let tps = lat.tokens_per_sec();
    let mse = dist.mean_mse();

    // Quiescent memory sample. `resident_bytes` sums `nbytes` of exactly the arrays the cache stores
    // (its compressed representation + any dense sink), so it is inherently free of the transient
    // full-context `update()` returns and allocator slack that inflate an active-memory high-water
    // mark. We read it after the fill loop has fully run.
    let resident = cache.resident_bytes();

    // Method runs don't reuse `decoded` as a reference, so drop it; baseline runs return it.
    if reference.is_some() {
        drop(decoded);
        drop(cache);
        memory::clear_cache();
        return Ok((resident, tps, mse, Vec::new()));
    }

    drop(cache);
    memory::clear_cache();

    Ok((resident, tps, mse, decoded))
}

/// Run the full method × context sweep and return every measured row.
///
/// The **first** registered method is treated as the dense baseline: it is run first at each context
/// to produce the reference decoded KV, and every subsequent method's quality delta is measured
/// against it. (Pass `ContiguousKvCache` first.)
///
/// Memory safety: [`BenchConfig::apply_mlx_limits`] is installed once up front, every context is
/// guarded against the config's estimated dense KV budget, and the buffer cache is cleared between
/// every run inside [`run_one`].
pub fn run_bench(cfg: &BenchConfig, methods: &[Method]) -> Result<BenchResult> {
    assert!(
        cfg.context_lengths.len() >= 2,
        "acceptance requires >= 2 context lengths"
    );
    assert!(
        !methods.is_empty(),
        "need at least the dense baseline method"
    );
    cfg.apply_mlx_limits();

    let mut rows = Vec::new();

    for &ctx in &cfg.context_lengths {
        // Pre-run guard: never start a context whose estimated dense KV exceeds the cache cap.
        let est = cfg.dense_kv_bytes(ctx);
        if est as usize > cfg.mlx_cache_limit {
            eprintln!(
                "skip ctx={ctx}: estimated dense KV {:.1} MB exceeds cache cap {:.1} MB",
                est as f64 / 1e6,
                cfg.mlx_cache_limit as f64 / 1e6
            );
            continue;
        }

        // Baseline first → produces the reference decoded KV for this context.
        let (base_resident, base_tps, _base_mse, reference) = run_one(cfg, &methods[0], ctx, None)?;
        rows.push(BenchRow {
            method: methods[0].name.clone(),
            context: ctx,
            resident_memory_bytes: base_resident,
            quality_delta_mse: 0.0,
            tokens_per_sec: base_tps,
        });

        for method in &methods[1..] {
            let (resident, tps, mse, _decoded) = run_one(cfg, method, ctx, Some(&reference))?;
            rows.push(BenchRow {
                method: method.name.clone(),
                context: ctx,
                resident_memory_bytes: resident,
                quality_delta_mse: mse,
                tokens_per_sec: tps,
            });
        }
    }

    Ok(BenchResult {
        config: cfg.clone(),
        rows,
    })
}

/// Render the comparison table (method × {KV-memory, quality-delta, tok/s}) as text. Memory is shown
/// in MB and (for non-baseline methods) as a compression ratio vs. the dense baseline at the same
/// context.
pub fn format_table(result: &BenchResult) -> String {
    // Index baseline (first method) resident bytes per context for the ratio column.
    let baseline = result.rows.first().map(|r| r.method.clone());
    let mut base_resident: BTreeMap<i32, usize> = BTreeMap::new();
    if let Some(ref base) = baseline {
        for r in &result.rows {
            if &r.method == base {
                base_resident.insert(r.context, r.resident_memory_bytes);
            }
        }
    }

    let mut out = String::new();
    out.push_str("KV-quant bench — method × {KV-memory, quality-delta (MSE vs dense), tok/s}\n");
    out.push_str(&format!(
        "config: layers={} batch={} kv_heads={} head_dim={} prefill={}\n",
        result.config.num_layers,
        result.config.batch,
        result.config.n_kv_heads,
        result.config.head_dim,
        result.config.prefill_tokens,
    ));
    out.push_str(&format!(
        "{:<14} {:>7} {:>12} {:>9} {:>16} {:>12}\n",
        "method", "ctx", "KV-mem(MB)", "vs-dense", "qual-delta(MSE)", "tok/s"
    ));
    out.push_str(&"-".repeat(74));
    out.push('\n');
    for r in &result.rows {
        let ratio = base_resident
            .get(&r.context)
            .filter(|_| r.resident_memory_bytes > 0)
            .map(|&b| format!("{:.2}x", b as f64 / r.resident_memory_bytes as f64))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "{:<14} {:>7} {:>12.2} {:>9} {:>16.3e} {:>12.1}\n",
            r.method,
            r.context,
            r.resident_memory_bytes as f64 / 1e6,
            ratio,
            r.quality_delta_mse,
            r.tokens_per_sec,
        ));
    }
    out
}

/// Build the default method set for the harness: the dense baseline first, then the identity-quantized
/// cache (lossless seam check), then the **RVQ** methods (story D) at `b=1` and `b=2`. The RVQ methods
/// are what produce the epic's memory/quality/throughput numbers; the driver and observers do not
/// change to add them — registering a method is one `Method::new(...)` line.
///
/// The RVQ methods are built for the bench's configured `head_dim` via [`rvq_methods`]; this default
/// set assumes the [`BenchConfig::small`] head_dim of 64 (Hadamard-compatible). For a different
/// head_dim, build the method list with [`rvq_methods`] at the call site.
pub fn default_methods() -> Vec<Method> {
    use crate::primitives::{ContiguousKvCache, IdentityQuantizer, QuantizedKvCache};
    let mut methods = vec![
        Method::new("dense", |n| Box::new(ContiguousKvCache::new(n))),
        Method::new("identity", |n| {
            Box::new(QuantizedKvCache::new(n, IdentityQuantizer))
        }),
    ];
    methods.extend(rvq_methods(BenchConfig::small().head_dim));
    methods
}

/// Build the RVQ [`Method`]s (`rvq-b1`, `rvq-b2`) for a given `head_dim`, with a dense first-token
/// sink (attention-sink semantics) so the first position stays lossless. Each method plugs an
/// [`RvqQuantizer`](crate::primitives::RvqQuantizer) into a
/// [`QuantizedKvCache`](crate::primitives::QuantizedKvCache). Returns an empty list (with a warning to
/// stderr) if `head_dim` is not Hadamard-compatible, so a sweep at an incompatible geometry still runs
/// the dense/identity baselines rather than panicking.
pub fn rvq_methods(head_dim: i32) -> Vec<Method> {
    use crate::primitives::{QuantizedKvCache, RvqQuantizer, SinkConfig};

    let mut out = Vec::new();
    for b in [1i32, 2] {
        // Probe constructibility once; if the geometry is unsupported, skip (don't panic the sweep).
        if RvqQuantizer::new(head_dim, b, 42).is_err() {
            eprintln!(
                "skip rvq-b{b}: head_dim {head_dim} unsupported (not Hadamard-compatible / bad bits)"
            );
            continue;
        }
        out.push(Method::new(format!("rvq-b{b}"), move |n| {
            let q = RvqQuantizer::new(head_dim, b, 42)
                .expect("rvq quantizer constructible (probed at registration)");
            Box::new(QuantizedKvCache::with_sink(n, q, SinkConfig::keep_first(1)))
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSE of identical arrays is exactly 0; MSE of arrays differing by a constant `c` is `c²`.
    #[test]
    fn distortion_observer_computes_known_mse() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let mut d = DistortionObserver::new();
        d.observe(&a, &b).unwrap();
        assert_eq!(d.mean_mse(), 0.0);
        assert_eq!(d.samples(), 1);

        // Shift every element by 3 → per-element squared error 9, mean 9.
        let c = Array::from_slice(&[4.0f32, 5.0, 6.0, 7.0], &[1, 1, 2, 2]);
        let mut d2 = DistortionObserver::new();
        d2.observe(&a, &c).unwrap();
        assert!((d2.mean_mse() - 9.0).abs() < 1e-5, "got {}", d2.mean_mse());
    }

    /// The distortion observer averages MSE across multiple updates.
    #[test]
    fn distortion_observer_averages_across_samples() {
        let a = Array::from_slice(&[0.0f32, 0.0], &[1, 1, 1, 2]);
        let zero = Array::from_slice(&[0.0f32, 0.0], &[1, 1, 1, 2]);
        let two = Array::from_slice(&[2.0f32, 2.0], &[1, 1, 1, 2]);
        let mut d = DistortionObserver::new();
        d.observe(&a, &zero).unwrap(); // mse 0
        d.observe(&a, &two).unwrap(); // mse 4
        assert!((d.mean_mse() - 2.0).abs() < 1e-5, "got {}", d.mean_mse());
        assert_eq!(d.samples(), 2);
    }

    /// The latency observer reports total-tokens / total-time.
    #[test]
    fn latency_observer_aggregates_tokens_per_sec() {
        let mut l = LatencyObserver::new();
        l.record(10, 1.0);
        l.record(10, 1.0);
        assert!((l.tokens_per_sec() - 10.0).abs() < 1e-9);
        assert_eq!(l.samples(), 2);

        let empty = LatencyObserver::new();
        assert_eq!(empty.tokens_per_sec(), 0.0);
    }

    /// The memory observer reports a non-negative delta and never underflows.
    #[test]
    fn memory_observer_delta_is_non_negative() {
        let m = MemoryObserver::start();
        // Allocate + evaluate something so the peak moves.
        let a = normal::<f32>(&[256, 256], None, None, None).unwrap();
        eval([&a]).unwrap();
        let _ = &a;
        // peak_delta_bytes is usize and saturating — always valid, may be 0 on a quiet allocator.
        let _ = m.peak_delta_bytes();
    }

    /// The segment plan is one prefill segment then single-token decode steps summing to `ctx`.
    #[test]
    fn segment_plan_sums_to_context() {
        let cfg = BenchConfig::small();
        for &ctx in &[64, 128] {
            let plan = segment_plan(&cfg, ctx);
            assert_eq!(plan.iter().sum::<i32>(), ctx);
            assert_eq!(plan[0], cfg.prefill_tokens.min(ctx));
        }
    }

    /// Lightweight `run_one` check (stays in the default suite): a tiny 2-layer, ctx-4 sweep proves
    /// the two fixes faithfully — identity's resident footprint equals dense's exactly, and the
    /// distortion (now accumulated over **both K and V across every layer**) is exactly 0 for the
    /// lossless identity path. This exercises the real measurement code (not just the observers) but
    /// is cheap enough to run by default.
    #[test]
    fn run_one_identity_resident_and_distortion_faithful() {
        let cfg = BenchConfig {
            num_layers: 2,
            batch: 1,
            n_kv_heads: 2,
            head_dim: 8,
            prefill_tokens: 2,
            context_lengths: vec![4],
            ..BenchConfig::small()
        };
        cfg.apply_mlx_limits();

        let dense = Method::new("dense", |n| {
            Box::new(crate::primitives::ContiguousKvCache::new(n))
        });
        let ident = Method::new("identity", |n| {
            use crate::primitives::{IdentityQuantizer, QuantizedKvCache};
            Box::new(QuantizedKvCache::new(n, IdentityQuantizer))
        });

        // Baseline produces the per-layer (K,V) reference for every segment.
        let (dense_resident, dense_tps, _mse, reference) = run_one(&cfg, &dense, 4, None).unwrap();
        // Reference must carry every layer's K and V for every segment (proves the path is full-KV).
        assert!(reference
            .iter()
            .all(|seg| seg.len() == cfg.num_layers as usize));

        let (ident_resident, ident_tps, ident_mse, _) =
            run_one(&cfg, &ident, 4, Some(&reference)).unwrap();

        // Faithful resident: identity stores the exact same bytes as dense → equal, non-zero.
        assert!(dense_resident > 0, "dense must store bytes");
        assert_eq!(
            ident_resident, dense_resident,
            "identity resident must equal dense resident byte-for-byte"
        );
        // Distortion over K+V, all layers, is exactly 0 for the lossless identity round-trip.
        assert_eq!(ident_mse, 0.0, "identity (K+V all layers) must be lossless");
        assert!(dense_tps > 0.0 && ident_tps > 0.0, "tok/s must be measured");
    }

    /// End-to-end harness validation (the acceptance check): identity-vs-dense over the small safe
    /// sweep must produce a table with identity showing ~0 quality delta and ~1.0× resident memory.
    ///
    /// Gated `#[ignore]`: it spins up the full `run_bench` driver (the heavy path) and must not run in
    /// the default `cargo test` unit suite. Run explicitly with
    /// `cargo test -p mlx-llm identity_matches_dense_end_to_end -- --ignored`. The lightweight
    /// observer-unit tests above stay in the default suite.
    #[test]
    #[ignore = "heavy run_bench driver; run with --ignored"]
    fn identity_matches_dense_end_to_end() {
        let cfg = BenchConfig::small();
        let result = run_bench(&cfg, &default_methods()).unwrap();

        // Two methods × two contexts = four rows.
        assert_eq!(result.rows.len(), 4);

        for r in &result.rows {
            if r.method == "identity" {
                // Identity is lossless → quality delta is exactly 0 (same bf16 values round-tripped).
                assert_eq!(
                    r.quality_delta_mse, 0.0,
                    "identity must be lossless vs dense at ctx={}",
                    r.context
                );
            }
            assert!(r.tokens_per_sec > 0.0, "tok/s must be measured");
        }

        // Identity resident memory == dense resident memory at each context: identity stores the exact
        // same arrays, so its faithful footprint is byte-for-byte equal (ratio exactly 1.0). The
        // resident measure is deterministic (summed `nbytes`), not a noisy allocator high-water mark,
        // so the tolerance is tight.
        for &ctx in &cfg.context_lengths {
            let dense = result
                .rows
                .iter()
                .find(|r| r.method == "dense" && r.context == ctx)
                .unwrap();
            let ident = result
                .rows
                .iter()
                .find(|r| r.method == "identity" && r.context == ctx)
                .unwrap();
            assert!(dense.resident_memory_bytes > 0, "dense must store bytes");
            let ratio = ident.resident_memory_bytes as f64 / dense.resident_memory_bytes as f64;
            assert!(
                (0.98..=1.02).contains(&ratio),
                "identity resident {} should equal dense resident {} at ctx={ctx} (ratio {ratio:.3})",
                ident.resident_memory_bytes,
                dense.resident_memory_bytes
            );
        }

        // The table renders and mentions both methods.
        let table = format_table(&result);
        assert!(table.contains("dense"));
        assert!(table.contains("identity"));
    }
}
