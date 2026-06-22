//! Paged attention — strategy B (custom Metal kernel), epic 7153 story 7170.
//!
//! The perf upgrade of the gather-then-SDPA path ([`PagedKvCache`](super::PagedKvCache), story 7169):
//! a custom Metal kernel reads a sequence's scattered KV **blocks directly** through its block table,
//! removing the per-step gather copy. The kernel is JIT-compiled through mlx-c's
//! `mlx_fast_metal_kernel` (see [`metal_kernel`](super::metal_kernel)) — no fork patch or MLX-core
//! rebuild.
//!
//! This is the **decode** kernel: one query token per sequence (the common serving case) attending
//! over all cached keys. Each thread owns one query head, walks the block table, and runs an online
//! (flash-style) softmax in f32, with grouped-query head mapping. Strategy A is the correctness
//! oracle — [`paged_attention_decode`] is bit-comparable to `sdpa` over the equivalent contiguous KV.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use crate::error::Result;
use crate::primitives::kv_cache::SEQ_AXIS;
use crate::primitives::metal_kernel::{MetalKernel, TemplateArg};

/// Shape parameters for [`paged_attention_decode`].
#[derive(Clone, Copy, Debug)]
pub struct PagedAttnParams {
    /// Query heads.
    pub n_heads: i32,
    /// Key/value heads (`n_heads / groups`).
    pub n_kv_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// Tokens per physical block.
    pub block_size: i32,
}

impl PagedAttnParams {
    fn groups(&self) -> i32 {
        self.n_heads / self.n_kv_heads
    }
}

/// The kernel body. Inputs are typed by their array dtype (f32 data, i32 indices). The template ints
/// are model-constant (HEAD_DIM, N_HEADS, N_KV_HEADS, BLOCK_SIZE, GROUPS), so MLX compiles it once per
/// model; the per-step `seq_len` and `scale` are runtime buffers, so a new length never recompiles.
const KERNEL_SRC: &str = r#"
    uint h = thread_position_in_grid.x;
    if (h >= (uint)N_HEADS) return;
    uint kvh = h / (uint)GROUPS;
    int L = seq_len[0];
    float s = scale[0];

    float qreg[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) qreg[d] = q[h * HEAD_DIM + d];

    float m = -INFINITY;
    float denom = 0.0f;
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

    for (int j = 0; j < L; ++j) {
        int logical = j / BLOCK_SIZE;
        int within = j - logical * BLOCK_SIZE;
        int phys = block_table[logical];
        uint base = (((uint)phys * (uint)N_KV_HEADS + kvh) * (uint)BLOCK_SIZE + (uint)within) * (uint)HEAD_DIM;
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) score += qreg[d] * pool_k[base + d];
        score *= s;
        float m_new = metal::max(m, score);
        float corr = metal::exp(m - m_new);
        float p = metal::exp(score - m_new);
        denom = denom * corr + p;
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * corr + p * pool_v[base + d];
        m = m_new;
    }
    for (int d = 0; d < HEAD_DIM; ++d) out[h * HEAD_DIM + d] = acc[d] / denom;
"#;

/// Build the paged-attention decode kernel for a model's fixed shape (compile once, reuse).
pub fn build_kernel() -> Result<MetalKernel> {
    MetalKernel::new(
        "paged_attention_decode",
        &["q", "pool_k", "pool_v", "block_table", "seq_len", "scale"],
        &["out"],
        KERNEL_SRC,
    )
}

/// One decode step of paged attention via the custom kernel.
///
/// - `q`: `[n_heads, head_dim]` (f32) — the single query token's projected heads.
/// - `pool_k`/`pool_v`: `[num_blocks, n_kv_heads, block_size, head_dim]` (f32) — the physical block pool.
/// - `block_table`: `[num_logical_blocks]` (i32) — logical→physical block ids for this sequence.
/// - `seq_len`: number of valid cached keys (≤ `num_logical_blocks * block_size`).
/// - `scale`: the usual `head_dim^(-0.5)`.
///
/// Returns `[n_heads, head_dim]` (f32). Reads the blocks in place — no gather.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode(
    kernel: &MetalKernel,
    q: &Array,
    pool_k: &Array,
    pool_v: &Array,
    block_table: &Array,
    seq_len: i32,
    scale: f32,
    params: PagedAttnParams,
) -> Result<Array> {
    let seq_len_arr = Array::from_slice(&[seq_len], &[1]);
    let scale_arr = Array::from_slice(&[scale], &[1]);
    let template = [
        TemplateArg::Int("HEAD_DIM", params.head_dim),
        TemplateArg::Int("N_HEADS", params.n_heads),
        TemplateArg::Int("N_KV_HEADS", params.n_kv_heads),
        TemplateArg::Int("BLOCK_SIZE", params.block_size),
        TemplateArg::Int("GROUPS", params.groups()),
    ];
    kernel.run(
        &[q, pool_k, pool_v, block_table, &seq_len_arr, &scale_arr],
        &[params.n_heads, params.head_dim],
        Dtype::Float32,
        (params.n_heads, 1, 1),
        (params.n_heads, 1, 1),
        &template,
    )
}

/// Lay contiguous per-head KV (`[1, n_kv_heads, seq, head_dim]`) into the block pool layout
/// `[num_blocks, n_kv_heads, block_size, head_dim]` plus an identity block table — the helper a
/// pool-backed cache and the parity tests use. The trailing partial block is zero-padded (the padded
/// keys are never read, since `seq_len` bounds the kernel loop).
pub fn build_pool(k: &Array, v: &Array, block_size: i32) -> Result<(Array, Array, Array, i32)> {
    let sh = k.shape();
    let (n_kv_heads, seq, head_dim) = (sh[1], sh[2], sh[3]);
    let num_blocks = (seq + block_size - 1) / block_size;
    let padded = num_blocks * block_size;

    let pool = |x: &Array| -> Result<Array> {
        let x = if padded > seq {
            let pad = Array::from_slice(
                &vec![0.0f32; (n_kv_heads * (padded - seq) * head_dim) as usize],
                &[1, n_kv_heads, padded - seq, head_dim],
            );
            concatenate_axis(&[&x.as_dtype(Dtype::Float32)?, &pad], SEQ_AXIS)?
        } else {
            x.as_dtype(Dtype::Float32)?
        };
        // [1, n_kv_heads, padded, hd] -> [n_kv_heads, num_blocks, block_size, hd] -> [num_blocks, n_kv_heads, block_size, hd]
        Ok(x.reshape(&[n_kv_heads, num_blocks, block_size, head_dim])?
            .transpose_axes(&[1, 0, 2, 3])?)
    };
    let pool_k = pool(k)?;
    let pool_v = pool(v)?;
    let block_table = Array::from_slice(&(0..num_blocks).collect::<Vec<_>>(), &[num_blocks]);
    Ok((pool_k, pool_v, block_table, num_blocks))
}

// ======================================================================================
// Flash (threadgroup-parallel) decode kernel — story 7301 (perf).
//
// The naive [`paged_attention_decode`] kernel above launches one thread per head (a single
// 32-thread threadgroup = one SIMD-group on one GPU core), each thread scanning all L keys ×
// head_dim serially. That under-fills the GPU by ~3 orders of magnitude and is ~25× slower than
// MLX's SDPA. This kernel mirrors MLX's own decode kernel (`sdpa_vector`): a **1024-thread
// threadgroup per head** (BN=32 SIMD-groups × BD=32 lanes) that splits BOTH axes of the work —
// the head_dim dot is reduced across 32 lanes via `simd_sum`, and the key range is striped across
// the 32 SIMD-groups — then a cross-SIMD-group online-softmax reduction combines the partials. The
// only difference from `sdpa_vector` is the **paged** K/V read: each key's address comes from the
// block table instead of a contiguous stride. Correctness is parity-gated against `sdpa`.
//
// Requires `head_dim % 32 == 0` (true for 64/128/256). BD/BN are fixed at 32 (Apple SIMD width).
const KERNEL_SRC_FLASH: &str = r#"
    const int BN = 32;             // SIMD-groups per threadgroup (key-range stripes)
    const int BD = 32;             // lanes per SIMD-group (head_dim split)
    const int QK = HEAD_DIM / BD;  // head_dim elements per lane

    uint h = threadgroup_position_in_grid.y;
    if (h >= (uint)N_HEADS) return;
    uint sg = simdgroup_index_in_threadgroup;   // 0..BN-1
    uint sl = thread_index_in_simdgroup;        // 0..BD-1
    uint kvh = h / (uint)GROUPS;
    int L = seq_len[0];
    float s = scale[0];

    // This lane owns head_dim elements [sl*QK, sl*QK+QK); fold the scale into q once.
    float qreg[QK];
    for (int j = 0; j < QK; ++j) qreg[j] = s * q[h * (uint)HEAD_DIM + sl * (uint)QK + j];

    float acc[QK];
    for (int j = 0; j < QK; ++j) acc[j] = 0.0f;
    float m = -INFINITY;
    float denom = 0.0f;

    // SIMD-group `sg` processes keys sg, sg+BN, sg+2BN, ...; lanes cooperate on each key's dot.
    for (int i = (int)sg; i < L; i += BN) {
        int logical = i / BLOCK_SIZE;
        int within = i - logical * BLOCK_SIZE;
        int phys = block_table[logical];
        uint base = (((uint)phys * (uint)N_KV_HEADS + kvh) * (uint)BLOCK_SIZE + (uint)within)
                        * (uint)HEAD_DIM + sl * (uint)QK;
        float part = 0.0f;
        for (int j = 0; j < QK; ++j) part += qreg[j] * pool_k[base + j];
        float score = simd_sum(part);   // full head_dim dot, reduced across the 32 lanes
        float m_new = metal::max(m, score);
        float corr = metal::exp(m - m_new);
        float p = metal::exp(score - m_new);
        m = m_new;
        denom = denom * corr + p;
        for (int j = 0; j < QK; ++j) acc[j] = acc[j] * corr + p * pool_v[base + j];
    }

    // Combine the BN SIMD-groups' partial (max, denom, acc) — the same transpose-reduce MLX uses.
    threadgroup float red_out[BN * BD];
    threadgroup float red_max[BN];
    threadgroup float red_den[BN];
    if (sl == 0) { red_max[sg] = m; red_den[sg] = denom; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lane_max = red_max[sl];
    float gmax = simd_max(lane_max);
    float factor = metal::exp(lane_max - gmax);
    float gden = simd_sum(red_den[sl] * factor);

    for (int j = 0; j < QK; ++j) {
        red_out[sl * BD + sg] = acc[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float v = simd_sum(red_out[sg * BD + sl] * factor);
        acc[j] = (gden == 0.0f) ? v : (v / gden);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // SIMD-group `sg` writes output dims [sg*QK, sg*QK+QK).
    if (sl == 0) {
        for (int j = 0; j < QK; ++j) out[h * (uint)HEAD_DIM + sg * (uint)QK + j] = acc[j];
    }
"#;

/// Build the flash (threadgroup-parallel) paged-attention decode kernel. Same I/O contract as
/// [`build_kernel`]; needs `#include <metal_simdgroup>` for the `simd_sum`/`simd_max` intrinsics.
pub fn build_kernel_flash() -> Result<MetalKernel> {
    MetalKernel::new_with_header(
        "paged_attention_decode_flash",
        &["q", "pool_k", "pool_v", "block_table", "seq_len", "scale"],
        &["out"],
        "#include <metal_simdgroup>\n",
        KERNEL_SRC_FLASH,
    )
}

/// One decode step via the flash kernel. Identical inputs/outputs to [`paged_attention_decode`],
/// but launches a 1024-thread threadgroup per head (`grid=(1024, n_heads, 1)`), so the GPU is
/// actually filled. `head_dim` must be a multiple of 32.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode_flash(
    kernel: &MetalKernel,
    q: &Array,
    pool_k: &Array,
    pool_v: &Array,
    block_table: &Array,
    seq_len: i32,
    scale: f32,
    params: PagedAttnParams,
) -> Result<Array> {
    debug_assert_eq!(params.head_dim % 32, 0, "flash kernel needs head_dim % 32 == 0");
    let seq_len_arr = Array::from_slice(&[seq_len], &[1]);
    let scale_arr = Array::from_slice(&[scale], &[1]);
    let template = [
        TemplateArg::Int("HEAD_DIM", params.head_dim),
        TemplateArg::Int("N_HEADS", params.n_heads),
        TemplateArg::Int("N_KV_HEADS", params.n_kv_heads),
        TemplateArg::Int("BLOCK_SIZE", params.block_size),
        TemplateArg::Int("GROUPS", params.groups()),
    ];
    kernel.run(
        &[q, pool_k, pool_v, block_table, &seq_len_arr, &scale_arr],
        &[params.n_heads, params.head_dim],
        Dtype::Float32,
        (32 * 32, params.n_heads, 1), // total threads: one 1024-thread threadgroup per head
        (32 * 32, 1, 1),
        &template,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::attention::{repeat_kv, sdpa, AttnMask};

    fn randf(shape: &[i32], seed: u64) -> Array {
        // Deterministic pseudo-random f32 in ~[-1, 1].
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let mut s = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        Array::from_slice(&data, shape)
    }

    fn host(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec()
    }

    /// The kernel matches `sdpa` (strategy A's oracle) over the equivalent contiguous KV, across MHA
    /// and GQA shapes, head_dim 64/128, and block boundaries.
    #[test]
    fn paged_kernel_matches_sdpa() {
        let kernel = build_kernel().unwrap();
        // (n_heads, n_kv_heads, head_dim, block_size, seq)
        let cases = [
            (4, 4, 64, 16, 40),  // MHA, crosses blocks, partial last
            (8, 2, 64, 16, 33),  // GQA groups=4
            (16, 16, 128, 8, 20), // Qwen3-like head_dim, small blocks
            (6, 3, 64, 16, 16),  // exactly one... two blocks, no padding
        ];
        for (nh, nkv, hd, bs, seq) in cases {
            let params = PagedAttnParams { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: bs };
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[nh, hd], 1);
            let k = randf(&[1, nkv, seq, hd], 2);
            let v = randf(&[1, nkv, seq, hd], 3);

            // Reference: sdpa over contiguous KV, single causal query attends all `seq` keys.
            let q4 = q.reshape(&[1, nh, 1, hd]).unwrap();
            let k_all = repeat_kv(&k, nh / nkv).unwrap();
            let v_all = repeat_kv(&v, nh / nkv).unwrap();
            let reference = sdpa(&q4, &k_all, &v_all, scale, AttnMask::Causal)
                .unwrap()
                .reshape(&[nh, hd])
                .unwrap();

            // Kernel: build the pool, attend in place.
            let (pool_k, pool_v, block_table, _) = build_pool(&k, &v, bs).unwrap();
            let got = paged_attention_decode(&kernel, &q, &pool_k, &pool_v, &block_table, seq, scale, params)
                .unwrap();

            let (g, r) = (host(&got), host(&reference));
            let maxdiff = g.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(maxdiff < 1e-4, "case {nh}/{nkv}/{hd}/{bs}/{seq}: max|Δ| = {maxdiff}");
        }
    }

    /// The flash (threadgroup-parallel) kernel matches `sdpa` over the same shapes as the naive
    /// kernel — proving the occupancy rewrite preserves the attention math.
    #[test]
    fn paged_flash_kernel_matches_sdpa() {
        let kernel = build_kernel_flash().unwrap();
        // (n_heads, n_kv_heads, head_dim, block_size, seq); head_dim must be a multiple of 32.
        let cases = [
            (4, 4, 64, 16, 40),   // MHA, crosses blocks, partial last
            (8, 2, 64, 16, 33),   // GQA groups=4
            (16, 16, 128, 8, 20), // Qwen3-like head_dim, small blocks
            (6, 3, 64, 16, 16),   // two exact blocks, no padding
            (32, 8, 128, 16, 2048), // serving-scale GQA, long context
            (4, 4, 64, 16, 7),    // fewer keys than the 32 SIMD-groups (empty-stripe path)
        ];
        for (nh, nkv, hd, bs, seq) in cases {
            let params = PagedAttnParams { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: bs };
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[nh, hd], 1);
            let k = randf(&[1, nkv, seq, hd], 2);
            let v = randf(&[1, nkv, seq, hd], 3);

            let q4 = q.reshape(&[1, nh, 1, hd]).unwrap();
            let k_all = repeat_kv(&k, nh / nkv).unwrap();
            let v_all = repeat_kv(&v, nh / nkv).unwrap();
            let reference = sdpa(&q4, &k_all, &v_all, scale, AttnMask::Causal)
                .unwrap()
                .reshape(&[nh, hd])
                .unwrap();

            let (pool_k, pool_v, block_table, _) = build_pool(&k, &v, bs).unwrap();
            let got = paged_attention_decode_flash(&kernel, &q, &pool_k, &pool_v, &block_table, seq, scale, params)
                .unwrap();

            let (g, r) = (host(&got), host(&reference));
            let maxdiff = g.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(maxdiff < 1e-4, "case {nh}/{nkv}/{hd}/{bs}/{seq}: max|Δ| = {maxdiff}");
        }
    }

    /// Throughput sweep across context lengths, isolating every factor so the comparison is honest:
    ///   naive  — current one-thread-per-head kernel (the 25× baseline)
    ///   flash  — our threadgroup-parallel paged kernel (no gather, GQA done in-kernel)
    ///   sdpaGR — gather(pool→contiguous) + repeat_kv(8→32) + SDPA = strategy A *as benched in 7170*
    ///   sdpaG  — gather + SDPA with MLX's native GQA (no repeat_kv) = the *optimal* strategy A
    ///   sdpaC  — SDPA on already-contiguous KV, native GQA, NO gather = MLX's pure-compute floor
    /// The two questions: is the flash kernel competitive with MLX's own compute (flash vs sdpaC),
    /// and does the no-gather paged path beat strategy A end-to-end (flash vs sdpaG)?
    #[test]
    #[ignore = "throughput micro-benchmark (run with --nocapture)"]
    fn paged_flash_throughput() {
        use std::time::Instant;
        let naive = build_kernel().unwrap();
        let flash = build_kernel_flash().unwrap();
        let (hd, bs) = (128, 16);
        // Two shapes to probe the *residual* flash-vs-SDPA gap (the part left after occupancy is
        // fixed). The gap is multi-factor and the GQA/MHA contrast does NOT cleanly isolate one
        // cause — flash MHA runs ~2.7× slower than flash GQA at long context at *identical* launch
        // geometry (32 threadgroups either way), so a large share of the tail is KV cache-footprint
        // bound (MHA reads 4× more unique KV), confounded with MLX's 2-pass split-K oversubscription
        // (engaged at k_seq≥1024) and its register-level GQA packing. See the story write-up.
        for &(nh, nkv) in &[(32, 8), (32, 32)] {
        let params = PagedAttnParams { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: bs };
        let scale = 1.0 / (hd as f32).sqrt();

        println!("\nshape: n_heads={nh} n_kv_heads={nkv} head_dim={hd} block_size={bs}  groups={}  (M5 Max, f32)", nh / nkv);
        println!(
            "{:>6} | {:>9} {:>9} {:>9} {:>9} {:>9} | {:>10} {:>10} {:>10}",
            "seq", "naive", "flash", "sdpaGR", "sdpaG", "sdpaC", "flash/sdpaC", "flash/sdpaG", "naive/sdpaGR",
        );
        for &seq in &[128, 512, 1024, 2048, 4096, 8192] {
            let q = randf(&[nh, hd], 1);
            let k = randf(&[1, nkv, seq, hd], 2);
            let v = randf(&[1, nkv, seq, hd], 3);
            let (pool_k, pool_v, bt, nb) = build_pool(&k, &v, bs).unwrap();
            let q4 = q.reshape(&[1, nh, 1, hd]).unwrap();
            // Contiguous [1, nkv, seq, hd] KV already on-device (the no-gather floor for SDPA).
            let kc = k.as_dtype(Dtype::Float32).unwrap();
            let vc = v.as_dtype(Dtype::Float32).unwrap();
            let to_contig = |pool: &Array| {
                pool.transpose_axes(&[1, 0, 2, 3]).unwrap().reshape(&[1, nkv, nb * bs, hd]).unwrap()
            };

            let naive_step = || paged_attention_decode(&naive, &q, &pool_k, &pool_v, &bt, seq, scale, params).unwrap();
            let flash_step = || paged_attention_decode_flash(&flash, &q, &pool_k, &pool_v, &bt, seq, scale, params).unwrap();
            let sdpa_gr = || {
                let k_all = repeat_kv(&to_contig(&pool_k), nh / nkv).unwrap();
                let v_all = repeat_kv(&to_contig(&pool_v), nh / nkv).unwrap();
                sdpa(&q4, &k_all, &v_all, scale, AttnMask::Causal).unwrap()
            };
            let sdpa_g = || sdpa(&q4, &to_contig(&pool_k), &to_contig(&pool_v), scale, AttnMask::Causal).unwrap();
            let sdpa_c = || sdpa(&q4, &kc, &vc, scale, AttnMask::Causal).unwrap();

            for _ in 0..5 {
                host(&naive_step()); host(&flash_step()); host(&sdpa_gr()); host(&sdpa_g()); host(&sdpa_c());
            }
            let iters = 100;
            let bench = |f: &dyn Fn()| {
                let t = Instant::now();
                for _ in 0..iters { f(); }
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            };
            let nt = bench(&|| { host(&naive_step()); });
            let ft = bench(&|| { host(&flash_step()); });
            let grt = bench(&|| { host(&sdpa_gr()); });
            let gt = bench(&|| { host(&sdpa_g()); });
            let ct = bench(&|| { host(&sdpa_c()); });
            println!(
                "{seq:>6} | {nt:>9.3} {ft:>9.3} {grt:>9.3} {gt:>9.3} {ct:>9.3} | {:>10.2} {:>10.2} {:>10.1}",
                ft / ct, ft / gt, nt / grt,
            );
        }
        }
    }

    /// Honest throughput comparison of the in-place kernel vs the gather-then-SDPA path at a large
    /// context. Printed, not asserted — the win depends on kernel optimization (see story notes).
    #[test]
    #[ignore = "throughput micro-benchmark (run with --nocapture)"]
    fn paged_kernel_throughput() {
        use std::time::Instant;
        let kernel = build_kernel().unwrap();
        let (nh, nkv, hd, bs, seq) = (32, 8, 128, 16, 2048);
        let params = PagedAttnParams { n_heads: nh, n_kv_heads: nkv, head_dim: hd, block_size: bs };
        let scale = 1.0 / (hd as f32).sqrt();
        let q = randf(&[nh, hd], 1);
        let k = randf(&[1, nkv, seq, hd], 2);
        let v = randf(&[1, nkv, seq, hd], 3);
        let (pool_k, pool_v, bt, nb) = build_pool(&k, &v, bs).unwrap();
        let q4 = q.reshape(&[1, nh, 1, hd]).unwrap();

        let kernel_step = || {
            paged_attention_decode(&kernel, &q, &pool_k, &pool_v, &bt, seq, scale, params).unwrap()
        };
        // Strategy A: reconstruct contiguous KV from the pool (the gather copy) + repeat_kv + sdpa.
        let gather_step = || {
            let to_contig = |pool: &Array| {
                pool.transpose_axes(&[1, 0, 2, 3])
                    .unwrap()
                    .reshape(&[1, nkv, nb * bs, hd])
                    .unwrap()
            };
            let k_all = repeat_kv(&to_contig(&pool_k), nh / nkv).unwrap();
            let v_all = repeat_kv(&to_contig(&pool_v), nh / nkv).unwrap();
            sdpa(&q4, &k_all, &v_all, scale, AttnMask::Causal).unwrap()
        };

        for _ in 0..5 {
            host(&kernel_step());
            host(&gather_step());
        }
        let iters = 200;
        let t = Instant::now();
        for _ in 0..iters {
            host(&kernel_step());
        }
        let kernel_t = t.elapsed();
        let t = Instant::now();
        for _ in 0..iters {
            host(&gather_step());
        }
        let gather_t = t.elapsed();
        println!(
            "seq={seq}: kernel {:.3}ms/step vs gather+sdpa {:.3}ms/step",
            kernel_t.as_secs_f64() * 1e3 / iters as f64,
            gather_t.as_secs_f64() * 1e3 / iters as f64,
        );
    }
}
