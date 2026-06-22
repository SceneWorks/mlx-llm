//! Throughput-mode "where does the step time go" breakdown (story 7325, `#[ignore]` — needs a model).
//!
//! Answers the sc-7325 measurement deliverable: at occupancy `N` × context `L`, how is a
//! **Throughput-mode** decode step ([`CausalLm::decode_logits_per_seq`]) split between the batched
//! matmuls (projections / MLP / lm_head) and the **per-sequence attention loop** (`N × (gather +
//! SDPA)` per layer, [`LlamaAttention::forward_per_seq`]) — and, within the attention loop, how much
//! is the **real per-step paged gather** vs SDPA.
//!
//! The real per-step gather is a [`PagedKvCache::gather`] — `concatenate_axis` over a *growing*
//! `Vec<Array>` of per-block tensors. sc-7301 flagged it as never measured (its harness reused a
//! static pool); here the caches are **actually grown to `L`** step by step, so the concat is over a
//! real block list.
//!
//! ## Method (faithful but cheap)
//! Step timing depends only on cache **lengths** and tensor **shapes**, not on KV *values*, so the
//! caches are grown to `L` with synthetic KV (avoiding an O(L²) real prefill) while every
//! matmul / gather / SDPA still runs at true production shapes (bf16, the model's real head counts).
//! Two cross-checking views:
//! - **Direct isolation** at each `(N, L)`: `C_gather` (the real per-step paged gather, summed over
//!   all layers × N) and `C_sdpa` (decode-shape SDPA, summed over all layers × N), against the real
//!   full step `T_full` (`decode_logits_per_seq`). The isolated components carry per-eval sync
//!   overhead the single-eval full step does not, so `C_gather + C_sdpa` is an **upper bound** on the
//!   attention loop's true in-graph share (⇒ the matmul residual is a lower bound). Directionally
//!   safe for the decision either way.
//! - **L-sweep fit**: decode is `s = 1`, so only attention (gather + SDPA, both O(L)) grows with `L`
//!   while the batched matmuls are L-independent. A least-squares fit of `T_full` vs `L` gives the
//!   **batched-matmul floor** (intercept) and the **attention marginal cost** (slope/token) under the
//!   real in-graph MLX scheduling — independent of the isolation overhead above.
//!
//! ```text
//! MLX_LLM_TEST_MODEL=/tmp/smollm2-135m MLX_LLM_QWEN3_MODEL=/tmp/qwen3-0.6b \
//!   cargo test --test throughput_breakdown -- --ignored --nocapture
//! ```

use std::time::Instant;

use mlx_rs::memory;
use mlx_rs::random::normal;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::attention::{sdpa, AttnMask};
use mlx_llm::primitives::{BlockPool, KvCache, PagedKvCache, Weights};

/// Realistic paged block size (the `ContinuousConfig` default).
const BLOCK: usize = 16;
/// Occupancies and contexts swept. **Bounded hard for memory safety**: an earlier unbounded grid
/// (N=32 × L=4096, two cache sets, MLX's buffer cache never evicted) climbed system RAM until the
/// machine fell over. These N≤16 / L≤2048 cells already gave a decisive, internally-consistent
/// breakdown; do not widen without the per-cell guard + cache eviction below.
const NS: &[usize] = &[4, 8, 16];
const LS: &[usize] = &[256, 1024, 2048];
/// Timed iterations per measurement (after warmup). Each step advances the cache by one token, so the
/// effective context drifts `L → L + ITERS`; kept small so the drift is negligible at `L ≥ 256`.
const WARMUP: usize = 3;
const ITERS: usize = 10;
/// Hard MLX backpressure limit and buffer-cache ceiling — freed KV returns to the OS between cells
/// instead of accumulating.
const MLX_MEMORY_LIMIT: usize = 24 * 1024 * 1024 * 1024;
const MLX_CACHE_LIMIT: usize = 2 * 1024 * 1024 * 1024;
/// Skip (and log) any `(N, L)` cell whose estimated resident KV exceeds this, rather than risk the OS.
const KV_BUDGET_BYTES: usize = 8 * 1024 * 1024 * 1024;

fn load(env: &str) -> Option<CausalLm> {
    let dir = std::env::var(env).ok()?;
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    Some(CausalLm::from_weights(&Weights::from_dir(&dir).unwrap(), "", cfg).unwrap())
}

/// A bf16 random tensor of the given shape (values are irrelevant — timing is shape/length driven).
fn synth(shape: &[i32]) -> Array {
    normal::<f32>(shape, None, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// `N` fresh per-sequence paged caches over one shared pool, each grown to length `l` with synthetic
/// KV. Growth uses one `update` per layer with an `l`-token chunk (same block-freeze structure `l`
/// real decode steps would produce), and evals each cache's gather so the block tensors are
/// materialized now — not charged to the first timed step.
fn grown_caches(cfg: &ModelConfig, n: usize, l: usize) -> Vec<PagedKvCache> {
    let pool = BlockPool::new(BLOCK);
    let (kvh, hd) = (cfg.num_kv_heads, cfg.head_dim);
    let mut caches: Vec<PagedKvCache> =
        (0..n).map(|_| PagedKvCache::with_pool(pool.clone(), cfg.num_layers)).collect();
    for c in caches.iter_mut() {
        let mut outs = Vec::with_capacity(cfg.num_layers * 2);
        for layer in 0..cfg.num_layers {
            let (k, v) = (synth(&[1, kvh, l as i32, hd]), synth(&[1, kvh, l as i32, hd]));
            let (k_all, v_all) = c.update(layer, &k, &v).unwrap();
            outs.push(k_all);
            outs.push(v_all);
        }
        eval(outs.iter()).unwrap(); // force this cache's blocks before timing
    }
    caches
}

/// Mean ms over `ITERS` calls of `f`, after `WARMUP` untimed calls.
fn timed(mut f: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / ITERS as f64
}

/// One real Throughput decode step over all `caches` (`decode_logits_per_seq`, evaluated).
fn full_step(model: &CausalLm, caches: &mut [PagedKvCache], ids: &Array) {
    let positions: Vec<i32> = caches.iter().map(|c| c.offset()).collect();
    let mut refs: Vec<&mut PagedKvCache> = caches.iter_mut().collect();
    let logits = model.decode_logits_per_seq(ids, &mut refs, &positions).unwrap();
    eval(std::iter::once(&logits)).unwrap();
}

/// The cache work of one step: `update` (append + the real paged gather) for every layer × sequence,
/// mirroring `forward_per_seq`'s layer-outer / sequence-inner loop. The gathered K/V are evaluated.
fn gather_step(caches: &mut [PagedKvCache], cfg: &ModelConfig, k1: &Array, v1: &Array) {
    let mut outs = Vec::with_capacity(cfg.num_layers * caches.len() * 2);
    for layer in 0..cfg.num_layers {
        for c in caches.iter_mut() {
            let (k_all, v_all) = c.update(layer, k1, v1).unwrap();
            outs.push(k_all);
            outs.push(v_all);
        }
    }
    eval(outs.iter()).unwrap();
}

/// Decode-shape SDPA for every layer × sequence: `q [1,H,1,hd]` over `k/v [1,kvh,L,hd]`, causal.
fn sdpa_step(q: &Array, k: &Array, v: &Array, scale: f32, layers: usize, n: usize) {
    let mut outs = Vec::with_capacity(layers * n);
    for _ in 0..layers {
        for _ in 0..n {
            outs.push(sdpa(q, k, v, scale, AttnMask::Causal).unwrap());
        }
    }
    eval(outs.iter()).unwrap();
}

/// Estimated resident KV bytes for `N` per-sequence bf16 caches at length `l` (k+v, all layers).
fn kv_bytes(cfg: &ModelConfig, n: usize, l: usize) -> usize {
    n * cfg.num_layers * 2 * cfg.num_kv_heads as usize * l * cfg.head_dim as usize * 2
}

/// Least-squares fit `y ≈ a + b·x`; returns `(a, b)`.
fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let (sx, sy) = (xs.iter().sum::<f64>(), ys.iter().sum::<f64>());
    let sxx = xs.iter().map(|x| x * x).sum::<f64>();
    let sxy = xs.iter().zip(ys).map(|(x, y)| x * y).sum::<f64>();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;
    (a, b)
}

/// Run and print the full breakdown for one loaded model.
fn breakdown(name: &str, model: &CausalLm) {
    // Memory safety: cap MLX's backpressure limit and keep the buffer cache small so freed KV
    // returns to the OS between cells (the unbounded version let it climb until the Mac fell over).
    memory::set_memory_limit(MLX_MEMORY_LIMIT);
    memory::set_cache_limit(MLX_CACHE_LIMIT);
    let cfg = model.config();
    let (h, kvh, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    println!(
        "\n================ {name}: {} layers, {h} heads / {kvh} kv (groups {}), head_dim {hd} ================",
        cfg.num_layers,
        h / kvh
    );
    println!("Throughput-mode decode step (ms), and where it goes. attn = gather + sdpa (isolated, upper bound).");

    for &n in NS {
        println!(
            "\n  N = {n} occupancy{}",
            if n == NS[0] { "   [L=context tokens, tok/s = N·1000/T_full]" } else { "" }
        );
        println!(
            "    {:>5} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>6} | {:>8}",
            "L", "T_full", "gather", "sdpa", "attn", "matmul", "attn%", "tok/s"
        );
        let (mut ls, mut ts) = (Vec::new(), Vec::new());
        for &l in LS {
            // Footprint guard: never allocate a cell that could threaten the OS. ×2 as a safe ceiling
            // for the ~1× resident frozen KV (sc-7363 dropped the sc-7325 duplicate) plus the
            // transient per-step gather it sits beside.
            let est = 2 * kv_bytes(cfg, n, l);
            if est > KV_BUDGET_BYTES {
                println!("    {l:>5} | SKIPPED — est resident {:.1} GB > {:.0} GB budget", est as f64 / 1e9, KV_BUDGET_BYTES as f64 / 1e9);
                continue;
            }
            // T_full on its own caches; C_gather on a fresh set at the same L; C_sdpa from shapes.
            let mut full_caches = grown_caches(cfg, n, l);
            let ids = Array::from_slice(&vec![1i32; n], &[n as i32, 1]);
            let t_full = timed(|| full_step(model, &mut full_caches, &ids));
            drop(full_caches);

            let mut gcaches = grown_caches(cfg, n, l);
            let (k1, v1) = (synth(&[1, kvh, 1, hd]), synth(&[1, kvh, 1, hd]));
            let c_gather = timed(|| gather_step(&mut gcaches, cfg, &k1, &v1));
            drop(gcaches);

            let (q, k, v) = (synth(&[1, h, 1, hd]), synth(&[1, kvh, l as i32, hd]), synth(&[1, kvh, l as i32, hd]));
            let c_sdpa = timed(|| sdpa_step(&q, &k, &v, cfg.attn_scale(), cfg.num_layers, n));

            let attn = c_gather + c_sdpa;
            let matmul = (t_full - attn).max(0.0);
            let attn_pct = 100.0 * attn / t_full;
            let toks = n as f64 * 1000.0 / t_full;
            drop((q, k, v)); // release this cell's SDPA tensors before evicting
            // Evict freed KV back to the OS before the next cell so resident memory cannot climb.
            memory::clear_cache();
            println!(
                "    {l:>5} | {t_full:>8.3} | {c_gather:>8.3} | {c_sdpa:>8.3} | {attn:>8.3} | {matmul:>8.3} | {attn_pct:>5.0}% | {toks:>8.0}  (active {:.1} GB)",
                memory::get_active_memory() as f64 / 1e9
            );
            ls.push(l as f64);
            ts.push(t_full);
        }
        // L-sweep fit: intercept = batched-matmul floor, slope = attention marginal cost / token.
        let (a, b) = fit(&ls, &ts);
        let at2k = b * 2048.0;
        println!(
            "    fit: T_full ≈ {a:.3} ms (matmul floor) + {:.4} ms/tok · L   ⇒ attention ≈ {at2k:.3} ms ({:.0}%) at L=2048",
            b,
            100.0 * at2k / (a + at2k)
        );
    }
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_TEST_MODEL; perf"]
fn throughput_breakdown_llama() {
    let Some(model) = load("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    breakdown("SmolLM2-135M (GQA g=3)", &model);
}

#[test]
#[ignore = "needs a real snapshot via MLX_LLM_QWEN3_MODEL; perf"]
fn throughput_breakdown_qwen3() {
    let Some(model) = load("MLX_LLM_QWEN3_MODEL") else {
        eprintln!("skip: set MLX_LLM_QWEN3_MODEL");
        return;
    };
    breakdown("Qwen3-0.6B (GQA g=2, hd=128)", &model);
}
