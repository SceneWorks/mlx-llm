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
