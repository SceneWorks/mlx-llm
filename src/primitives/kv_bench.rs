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
//!   keys/values vs. the dense baseline's, accumulated as mean-squared error (MSE) over every
//!   update. A lossless method (the [`IdentityQuantizer`](crate::primitives::IdentityQuantizer))
//!   reports ~0.
//! - **Peak KV-memory** ([`MemoryObserver`]): MLX active-memory high-water mark while the cache for a
//!   method is resident, via [`mlx_rs::memory`]. The dense baseline's peak is the reference; a method
//!   that compresses reports a lower peak (and a compression ratio).
//! - **Throughput** ([`LatencyObserver`]): wall-clock tokens/sec of the update path (the per-step
//!   `update` + decode-and-materialize that a decoder pays), averaged over the swept steps.
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
//! `QuantizedKvCache<IdentityQuantizer>` (story B); the end-to-end test below validates the harness
//! with exactly those two — identity must show ~0 quality delta and equal memory vs dense.
//!
//! ## Memory safety
//! Unbounded MLX sweeps crash the machine, so the harness is bounded by default
//! ([`BenchConfig::small`]): few short contexts, batch 1, small head geometry. It installs an MLX
//! memory limit + cache-cap ([`BenchConfig::apply_mlx_limits`]), and **clears the MLX buffer cache
//! and resets the peak counter between every (method, context) run** so freed KV returns to the OS
//! instead of accumulating. Larger sweeps are opt-in by constructing a wider [`BenchConfig`].

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
    /// Peak MLX-resident bytes attributable to this run.
    pub peak_memory_bytes: usize,
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

/// Drive one cache through the segment plan for `ctx`, measuring throughput and (when a
/// `distortion` observer + matching `reference` segments are supplied) reconstruction error against
/// the dense baseline. Returns `(peak_memory_bytes, tokens_per_sec, mean_mse)`.
///
/// `reference` is `Some(per-segment dense decoded keys)` for a method, `None` for the baseline run
/// itself (which *produces* the reference). The same fixed-seed [`synth`] inputs are fed to both, so
/// the only difference in decoded output is the method's compression error.
#[allow(clippy::type_complexity)]
fn run_one(
    cfg: &BenchConfig,
    method: &Method,
    ctx: i32,
    reference: Option<&[Array]>,
) -> Result<(usize, f64, f64, Vec<Array>)> {
    // Fresh cache + clean slate. Clearing the buffer cache and resetting peak BEFORE we start is what
    // keeps successive runs from accumulating freed KV in the MLX allocator.
    memory::clear_cache();
    let mut cache = (method.builder)(cfg.num_layers);
    let mem = MemoryObserver::start();
    let mut lat = LatencyObserver::new();
    let mut dist = DistortionObserver::new();

    let plan = segment_plan(cfg, ctx);
    // Decoded keys captured per segment (layer 0) — this becomes the reference for later methods, and
    // is what we diff against `reference` for a method run.
    let mut decoded_keys: Vec<Array> = Vec::with_capacity(plan.len());

    let mut seed = (ctx as u64) << 8;
    for (seg_idx, &step) in plan.iter().enumerate() {
        // Identical synthetic input across dense + method runs (seed depends only on ctx + position).
        let k = synth(cfg, step, seed)?;
        let v = synth(cfg, step, seed.wrapping_add(1))?;
        seed = seed.wrapping_add(2);

        let t0 = Instant::now();
        let mut last_k = k.clone();
        for layer in 0..cfg.num_layers {
            let (kk, vv) = cache.update(layer, &k, &v)?;
            if layer == 0 {
                last_k = kk.clone();
            }
            // Force materialization so timing/memory reflect the real decode-path work, not lazy
            // graph deferral.
            eval([&kk, &vv])?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        lat.record(step as u64, elapsed);

        // Distortion: compare this method's full decoded layer-0 keys against the dense reference's.
        if let Some(refs) = reference {
            if let Some(r) = refs.get(seg_idx) {
                dist.observe(r, &last_k)?;
            }
        }
        decoded_keys.push(last_k);
    }

    let peak = mem.peak_delta_bytes();
    let tps = lat.tokens_per_sec();
    let mse = dist.mean_mse();

    // Drop the cache and clear the buffer cache so the next run starts from a clean allocator.
    drop(cache);
    memory::clear_cache();

    Ok((peak, tps, mse, decoded_keys))
}

/// Run the full method × context sweep and return every measured row.
///
/// The **first** registered method is treated as the dense baseline: it is run first at each context
/// to produce the reference decoded KV, and every subsequent method's quality delta is measured
/// against it. (Pass `ContiguousKvCache` first.)
///
/// Memory safety: [`BenchConfig::apply_mlx_limits`] is installed once up front, every context is
/// guarded against the config's estimated dense KV budget, and the buffer cache is cleared + the peak
/// counter reset between every run inside [`run_one`].
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
        let (base_peak, base_tps, _base_mse, reference) = run_one(cfg, &methods[0], ctx, None)?;
        rows.push(BenchRow {
            method: methods[0].name.clone(),
            context: ctx,
            peak_memory_bytes: base_peak,
            quality_delta_mse: 0.0,
            tokens_per_sec: base_tps,
        });

        for method in &methods[1..] {
            let (peak, tps, mse, _decoded) = run_one(cfg, method, ctx, Some(&reference))?;
            rows.push(BenchRow {
                method: method.name.clone(),
                context: ctx,
                peak_memory_bytes: peak,
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
    // Index baseline (first method) peak per context for the ratio column.
    let baseline = result.rows.first().map(|r| r.method.clone());
    let mut base_peak: BTreeMap<i32, usize> = BTreeMap::new();
    if let Some(ref base) = baseline {
        for r in &result.rows {
            if &r.method == base {
                base_peak.insert(r.context, r.peak_memory_bytes);
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
        let ratio = base_peak
            .get(&r.context)
            .filter(|_| r.peak_memory_bytes > 0)
            .map(|&b| format!("{:.2}x", b as f64 / r.peak_memory_bytes as f64))
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!(
            "{:<14} {:>7} {:>12.2} {:>9} {:>16.3e} {:>12.1}\n",
            r.method,
            r.context,
            r.peak_memory_bytes as f64 / 1e6,
            ratio,
            r.quality_delta_mse,
            r.tokens_per_sec,
        ));
    }
    out
}

/// Build the default method set for the harness as it stands today: the dense baseline first, then
/// the identity-quantized cache. Registering RVQ later (story D) appends one more [`Method`] here (or
/// at the call site) — the driver and observers do not change.
pub fn default_methods() -> Vec<Method> {
    use crate::primitives::{ContiguousKvCache, IdentityQuantizer, QuantizedKvCache};
    vec![
        Method::new("dense", |n| Box::new(ContiguousKvCache::new(n))),
        Method::new("identity", |n| {
            Box::new(QuantizedKvCache::new(n, IdentityQuantizer))
        }),
    ]
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

    /// End-to-end harness validation (the acceptance check): identity-vs-dense over the small safe
    /// sweep must produce a table with identity showing ~0 quality delta and ~equal peak memory. This
    /// is bounded (no model, tiny synthetic KV) so it runs in the default suite.
    #[test]
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

        // Identity peak memory ≈ dense peak memory at each context (same stored tensors). Allow a
        // generous tolerance: the MLX allocator's peak is noisy, but identity stores the exact same
        // arrays as dense so it must not balloon.
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
            if dense.peak_memory_bytes > 0 {
                let ratio = ident.peak_memory_bytes as f64 / dense.peak_memory_bytes as f64;
                assert!(
                    (0.5..=2.0).contains(&ratio),
                    "identity peak {} should be ~equal to dense peak {} at ctx={ctx} (ratio {ratio:.2})",
                    ident.peak_memory_bytes,
                    dense.peak_memory_bytes
                );
            }
        }

        // The table renders and mentions both methods.
        let table = format_table(&result);
        assert!(table.contains("dense"));
        assert!(table.contains("identity"));
    }
}
