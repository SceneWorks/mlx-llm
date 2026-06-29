//! Calibration-free 2-stage residual scalar VQ (TurboQuant-RVQ) as a KV-cache [`Quantizer`].
//!
//! This is the story-D method that plugs into [`QuantizedKvCache`](crate::primitives::QuantizedKvCache):
//! a faithful pure-MLX port of VeloxQuant-MLX's `quantizers/turboquant_rvq.py` +
//! `cache/turboquant_rvq_cache.py` (read, and run as the numeric oracle — see `tests/rvq_oracle.rs`).
//!
//! ## Algorithm (per key vector, head_dim `d`, `b` bits/stage)
//!
//! The cache wrapper unit-normalizes each key vector first (preserving direction, which is what
//! attention scores depend on) and stores the scalar norm separately, exactly as
//! `turboquant_rvq_cache.py` does:
//!
//! ```text
//!   norm   = ||k||₂            (fp32, clamped at 1e-4)
//!   u      = k / norm          (unit vector, fp16)
//!   y      = Hadamard(D ⊙ u)   (randomized Hadamard rotation; D ∈ {±1}^d seeded by `seed`)
//!   idx1   = codebook1.quantize(y)            (stage 1: N(0, 1/d) Lloyd-Max, b bits)
//!   ŷ1     = codebook1.dequantize(idx1)
//!   r1     = y - ŷ1                           (residual)
//!   idx2   = codebook2.quantize(r1)           (stage 2: Laplacian Lloyd-Max, b bits)
//!   decode: û = D ⊙ Hadamard(ŷ1 + ŷ2);  k̂ = û · norm
//! ```
//!
//! Total key storage per coordinate is `2·b` bits (two index sets) plus one fp16 norm per vector.
//! Values pass through **dense** (uncompressed) — mirroring the upstream cache, which compresses keys
//! only. At `d=128, b=1` the upstream realizes ~7.5× on keys; this port reports the same compressed
//! footprint through [`block_bytes`](RvqQuantizer::block_bytes) so the bench memory column is faithful.
//!
//! ## Reuse of story-C primitives
//!
//! - [`lloyd_max`](crate::primitives::quant::lloyd_max) + [`ScalarCodebook`] fit & apply both stage
//!   codebooks (no reimplementation of the codebook math).
//! - [`bit_pack`](crate::primitives::quant::bit_pack)/[`bit_unpack`] losslessly pack the two `b`-bit
//!   index sets (`b ∈ {1, 2, 4}`). RVQ feeds group-aligned counts: the packed length per index set is
//!   `batch·heads·seq·d` codes, padded up to a multiple of `8/b` (the pad codes are sliced off on
//!   unpack), so the bit-packer's multiple-of-`8/b` requirement is always satisfied.
//!
//! ## Sink / first-block semantics
//!
//! The dense **sink** (first-N tokens kept verbatim) is owned by [`QuantizedKvCache`] via
//! [`SinkConfig`](crate::primitives::SinkConfig) — those positions never reach this quantizer, so the
//! attention-sink tokens stay lossless exactly as the upstream `sink_cache.py` keeps them. This
//! quantizer compresses every position it is handed.
//!
//! ## Phase-2 follow-on
//!
//! All hot loops here are the same pure-MLX/host paths story C marked `TODO(sc-8529/Phase2)`; the
//! Metal-kernel speedup (story A) is gated and does not change correctness.

use std::f64::consts::PI;

use mlx_rs::ops::{matmul, maximum, multiply, sqrt};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::primitives::kv_cache::{Quantizer, SEQ_AXIS};
use crate::primitives::quant::{bit_pack, bit_unpack, lloyd_max, ScalarCodebook};

/// Number of quadrature points for the Lloyd-Max codebook fit (matches VeloxQuant `LLOYD_MAX_N_QUAD`).
const N_QUAD: usize = 10_000;
/// Max Lloyd-Max iterations (matches VeloxQuant `LLOYD_MAX_N_ITER`).
const N_ITER: usize = 500;
/// Lloyd-Max convergence tolerance (matches VeloxQuant `LLOYD_MAX_TOL`).
const TOL: f64 = 1e-9;

/// Build a dense `(support, pdf)` grid over `[lo, hi]` for a PDF closure, for [`lloyd_max`].
fn grid(lo: f64, hi: f64, pdf: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut support = Vec::with_capacity(N_QUAD);
    let mut p = Vec::with_capacity(N_QUAD);
    for i in 0..N_QUAD {
        let x = lo + (hi - lo) * (i as f64) / ((N_QUAD - 1) as f64);
        support.push(x);
        p.push(pdf(x));
    }
    (support, p)
}

/// Fit the stage-1 codebook: Lloyd-Max on the `N(0, 1/d)` coordinate distribution (the TurboQuant
/// rotated-coordinate model), `2^b` levels, support `±6σ`.
fn fit_gaussian_codebook(d: i32, b: i32) -> Result<ScalarCodebook> {
    let sigma = 1.0 / (d as f64).sqrt();
    let (support, pdf) = grid(-6.0 * sigma, 6.0 * sigma, |x| {
        let z = x / sigma;
        (1.0 / (sigma * (2.0 * PI).sqrt())) * (-0.5 * z * z).exp()
    });
    let lm = lloyd_max(&pdf, &support, 1usize << b, N_ITER, TOL)?;
    let c: Vec<f32> = lm.centroids.iter().map(|&v| v as f32).collect();
    ScalarCodebook::new(&c)
}

/// The Laplacian scale matching the std of stage-1 quantization error on a unit-variance Gaussian
/// source — VeloxQuant's `residual_scale` default.
fn residual_scale(d: i32, b: i32) -> f64 {
    let sigma_q = (1.0 / d as f64).sqrt() * ((3.0 * PI).sqrt() / 2.0) * 4.0f64.powi(-b);
    (sigma_q / 2.0f64.sqrt()).max(1e-6)
}

/// Fit the stage-2 codebook: Lloyd-Max on a zero-mean Laplacian of the given `scale`, `2^b` levels,
/// support `±8·scale`.
fn fit_laplacian_codebook(scale: f64, b: i32) -> Result<ScalarCodebook> {
    let inv = 1.0 / (2.0 * scale);
    let inv_scale = 1.0 / scale;
    let hi = 8.0 * scale;
    let (support, pdf) = grid(-hi, hi, |x| inv * (-x.abs() * inv_scale).exp());
    let lm = lloyd_max(&pdf, &support, 1usize << b, N_ITER, TOL)?;
    let c: Vec<f32> = lm.centroids.iter().map(|&v| v as f32).collect();
    ScalarCodebook::new(&c)
}

/// Immutable RVQ configuration + fitted codebooks. Cloned cheaply — every field is either `Copy` or a
/// refcounted [`Array`]/[`ScalarCodebook`] (no deep copy), so cloning [`RvqQuantizer`] is cheap without
/// an `Arc` wrapper (an `Arc` over `!Send`+`!Sync` `Array`s trips `clippy::arc_with_non_send_sync`).
#[derive(Debug, Clone)]
struct RvqInner {
    /// Head dimension (the [`SEQ_AXIS`]+1 trailing axis the rotation/codebooks operate on).
    d: i32,
    /// Bits per stage; total key bits/coordinate is `2·b`. `b ∈ {1, 2, 4}` (bit-packing widths).
    b: i32,
    /// Stage-1 (`N(0,1/d)` Gaussian) codebook.
    cb1: ScalarCodebook,
    /// Stage-2 (Laplacian residual) codebook.
    cb2: ScalarCodebook,
    /// Randomized-Hadamard ±1 diagonal of length `d`, seeded by `seed`.
    diag: Array,
}

/// Calibration-free 2-stage residual scalar VQ quantizer (TurboQuant-RVQ) for the KV cache.
///
/// Construct with [`RvqQuantizer::new`] (the head dimension `d` and bits/stage `b` are fixed at
/// construction — they pin the rotation length and codebook tables). Plug into
/// [`QuantizedKvCache`](crate::primitives::QuantizedKvCache) like any [`Quantizer`].
#[derive(Debug, Clone)]
pub struct RvqQuantizer {
    inner: RvqInner,
}

/// One RVQ-compressed block: the two index sets (bit-packed `u8`), the per-vector key norms (fp16),
/// the dense value tensor, and the geometry needed to round-trip. Opaque to the cache shell.
#[derive(Debug, Clone)]
pub struct RvqBlock {
    /// Stage-1 indices, bit-packed `b`-bit codes, flat `u8`. `n_codes = batch·heads·seq·d`.
    idx1: Array,
    /// Stage-2 indices, bit-packed `b`-bit codes, flat `u8`.
    idx2: Array,
    /// Per-key-vector L2 norms, shape `[batch, heads, seq, 1]`, fp16.
    norms: Array,
    /// Dense values `[batch, heads, seq, d]` (passed through uncompressed, like the upstream cache).
    values: Array,
    /// Batch size (axis 0).
    batch: i32,
    /// KV heads (axis 1).
    heads: i32,
    /// Sequence positions (axis 2).
    seq: i32,
}

impl RvqQuantizer {
    /// Build an RVQ quantizer for head dimension `d` and `b` bits/stage.
    ///
    /// `b` must be in `{1, 2, 4}` (the lossless bit-packing widths). `d` must be ≥ 1 and
    /// Hadamard-compatible (`d = m·2^k`, `m ∈ {1,12,20,28}`) — every power-of-two head_dim (64, 128,
    /// …) qualifies. `seed` selects the randomized-Hadamard ±1 diagonal (deterministic per seed,
    /// matching the upstream `make_hadamard_diagonal`).
    pub fn new(d: i32, b: i32, seed: u64) -> Result<Self> {
        // Validate `d` up front so we never call `hadamard_diagonal` with a nonsensical length;
        // `with_diagonal` re-checks `b`/Hadamard-compatibility (and the diagonal length).
        if d < 1 {
            return Err(Error::Msg(format!("RvqQuantizer: d must be >= 1, got {d}")));
        }
        Self::with_diagonal(d, b, hadamard_diagonal(d, seed))
    }

    /// Build an RVQ quantizer with an explicit ±1 Hadamard `diag` (length `d`) instead of a
    /// seed-derived one. The codebooks are still fitted from `(d, b)`. This is the seam the
    /// VeloxQuant **oracle-parity** test uses to inject upstream's exact diagonal so encode→decode
    /// reproduces the upstream reconstruction element-for-element; production callers use
    /// [`new`](RvqQuantizer::new).
    pub fn with_diagonal(d: i32, b: i32, diag: Array) -> Result<Self> {
        if d < 1 {
            return Err(Error::Msg(format!("RvqQuantizer: d must be >= 1, got {d}")));
        }
        if !matches!(b, 1 | 2 | 4) {
            return Err(Error::Unsupported(format!(
                "RvqQuantizer: b must be 1, 2, or 4 (bit-packing widths), got {b}"
            )));
        }
        if !is_hadamard_compatible(d) {
            return Err(Error::Unsupported(format!(
                "RvqQuantizer: head_dim {d} is not Hadamard-compatible (need d = m·2^k, m ∈ {{1,12,20,28}})"
            )));
        }
        if diag.size() != d as usize {
            return Err(Error::Msg(format!(
                "RvqQuantizer::with_diagonal: diag len {} != d {d}",
                diag.size()
            )));
        }
        let cb1 = fit_gaussian_codebook(d, b)?;
        let cb2 = fit_laplacian_codebook(residual_scale(d, b), b)?;
        Ok(Self {
            inner: RvqInner {
                d,
                b,
                cb1,
                cb2,
                diag: diag.as_dtype(Dtype::Float32)?,
            },
        })
    }

    /// Bits per stage (total key bits/coordinate is `2·b`).
    pub fn bits(&self) -> i32 {
        self.inner.b
    }

    /// Head dimension the quantizer is fitted for.
    pub fn head_dim(&self) -> i32 {
        self.inner.d
    }

    /// Stage-1 (`N(0,1/d)` Gaussian) codebook centroids, ascending — exposed for oracle-parity checks.
    pub fn stage1_centroids(&self) -> Vec<f32> {
        self.inner.cb1.centroids().to_vec()
    }

    /// Stage-2 (Laplacian residual) codebook centroids, ascending — exposed for oracle-parity checks.
    pub fn stage2_centroids(&self) -> Vec<f32> {
        self.inner.cb2.centroids().to_vec()
    }

    /// Forward randomized-Hadamard rotation: `y = Hadamard(D ⊙ x)`. `x` is `[.., d]`. MLX's
    /// `hadamard_transform` is normalized (self-inverse), matching the upstream preconditioner.
    fn rotate(&self, x: &Array) -> Result<Array> {
        let xd = multiply(&x.as_dtype(Dtype::Float32)?, &self.inner.diag)?;
        Ok(xd.hadamard_transform(None)?)
    }

    /// Inverse rotation: `x = D ⊙ Hadamard(y)` (self-inverse Hadamard, `D² = I`).
    fn unrotate(&self, y: &Array) -> Result<Array> {
        let h = y.as_dtype(Dtype::Float32)?.hadamard_transform(None)?;
        Ok(multiply(&h, &self.inner.diag)?)
    }
}

/// MLX `hadamard_transform` constraint: `d = m·2^k`, `m ∈ {1, 12, 20, 28}` (every power of two works).
fn is_hadamard_compatible(d: i32) -> bool {
    if d < 1 {
        return false;
    }
    for m in [1i32, 12, 20, 28] {
        if d % m == 0 {
            let r = d / m;
            if r & (r - 1) == 0 {
                return true;
            }
        }
    }
    false
}

/// Deterministic ±1 Hadamard diagonal of length `d`, seeded by `seed`. Uses a SplitMix64 stream so the
/// sign pattern is reproducible per `(d, seed)` (the Rust analogue of the upstream
/// `np.random.default_rng(seed).choice([-1, 1], d)` — pattern differs from numpy's but is fixed and
/// the rotation is orthogonal for any ±1 diagonal, so reconstruction quality is identical).
fn hadamard_diagonal(d: i32, seed: u64) -> Array {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let signs: Vec<f32> = (0..d)
        .map(|_| {
            // SplitMix64 step.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if z & 1 == 0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    Array::from_slice(&signs, &[d])
}

impl Quantizer for RvqQuantizer {
    type Block = RvqBlock;

    fn encode(&self, keys: &Array, values: &Array) -> Result<Self::Block> {
        let shape = keys.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "RvqQuantizer::encode: expected 4-D [batch, heads, seq, head_dim], got {shape:?}"
            )));
        }
        let (batch, heads, seq, d) = (shape[0], shape[1], shape[2], shape[3]);
        if d != self.inner.d {
            return Err(Error::Msg(format!(
                "RvqQuantizer::encode: head_dim {d} != fitted d {}",
                self.inner.d
            )));
        }

        // Flatten the per-vector axis: [batch·heads·seq, d].
        let n_vec = batch * heads * seq;
        let flat = keys.as_dtype(Dtype::Float32)?.reshape(&[n_vec, d])?;

        // Per-vector L2 norm (fp32), clamped at 1e-4, then unit-normalize.
        let sq = multiply(&flat, &flat)?;
        let sumsq = mlx_rs::ops::sum_axis(&sq, -1, true)?; // [n_vec, 1]
        let norm = sqrt(&sumsq)?;
        let safe = maximum(&norm, Array::from_f32(1e-4))?;
        let unit = mlx_rs::ops::divide(&flat, &safe)?; // [n_vec, d]

        // Stage 1: rotate + quantize.
        let y = self.rotate(&unit)?;
        let idx1 = self.inner.cb1.quantize(&y)?; // u8 [n_vec, d]
        let yhat1 = self.inner.cb1.dequantize(&idx1)?;
        // Stage 2: quantize residual.
        let r1 = mlx_rs::ops::subtract(&y, &yhat1)?;
        let idx2 = self.inner.cb2.quantize(&r1)?;

        // Bit-pack both index sets (group-aligned: pad up to a multiple of 8/b, sliced off on unpack).
        let packed1 = pack_codes(&idx1, self.inner.b)?;
        let packed2 = pack_codes(&idx2, self.inner.b)?;

        // Store norms as fp16 (the upstream stores one fp16 norm/vector).
        let norms = safe
            .reshape(&[batch, heads, seq, 1])?
            .as_dtype(Dtype::Float16)?;

        Ok(RvqBlock {
            idx1: packed1,
            idx2: packed2,
            norms,
            values: values.clone(),
            batch,
            heads,
            seq,
        })
    }

    fn decode(&self, block: &Self::Block) -> Result<(Array, Array)> {
        let d = self.inner.d;
        let n_vec = block.batch * block.heads * block.seq;
        let n_codes = (n_vec * d) as usize;

        let idx1 = unpack_codes(&block.idx1, n_codes, self.inner.b)?.reshape(&[n_vec, d])?;
        let idx2 = unpack_codes(&block.idx2, n_codes, self.inner.b)?.reshape(&[n_vec, d])?;

        let yhat1 = self.inner.cb1.dequantize(&idx1)?;
        let yhat2 = self.inner.cb2.dequantize(&idx2)?;
        let yhat = mlx_rs::ops::add(&yhat1, &yhat2)?;
        let unit = self.unrotate(&yhat)?; // [n_vec, d] fp32

        // Rescale by the per-vector norm.
        let norms = block.norms.as_dtype(Dtype::Float32)?.reshape(&[n_vec, 1])?;
        let k = multiply(&unit, &norms)?.reshape(&[block.batch, block.heads, block.seq, d])?;
        // Match the values dtype so the decoded K/V layout is uniform.
        let k = k.as_dtype(block.values.dtype())?;
        Ok((k, block.values.clone()))
    }

    fn seq_len(&self, block: &Self::Block) -> i32 {
        block.seq
    }

    fn batch_size(&self, block: &Self::Block) -> i32 {
        block.batch
    }

    fn block_bytes(&self, block: &Self::Block) -> usize {
        // Faithful compressed footprint: the two packed index sets + the fp16 norms + the dense
        // values. Keys cost only 2·b bits/coordinate (the packed sets) plus one fp16 norm/vector —
        // exactly what the upstream cache charges; values are dense (uncompressed) just like upstream.
        block.idx1.nbytes() + block.idx2.nbytes() + block.norms.nbytes() + block.values.nbytes()
    }

    fn retain_sequences(&self, block: &Self::Block, keep: &Array) -> Result<Self::Block> {
        // Reconstruct the per-vector code grids, gather the kept batch rows, re-pack. Norms and values
        // gather directly along the batch axis.
        let d = self.inner.d;
        let n_vec = block.batch * block.heads * block.seq;
        let n_codes = (n_vec * d) as usize;

        let g1 = unpack_codes(&block.idx1, n_codes, self.inner.b)?.reshape(&[
            block.batch,
            block.heads,
            block.seq,
            d,
        ])?;
        let g2 = unpack_codes(&block.idx2, n_codes, self.inner.b)?.reshape(&[
            block.batch,
            block.heads,
            block.seq,
            d,
        ])?;

        let g1 = g1.take_axis(keep, 0)?;
        let g2 = g2.take_axis(keep, 0)?;
        let new_batch = g1.shape()[0];

        let idx1 = pack_codes(&g1, self.inner.b)?;
        let idx2 = pack_codes(&g2, self.inner.b)?;
        let norms = block.norms.take_axis(keep, 0)?;
        let values = block.values.take_axis(keep, 0)?;

        Ok(RvqBlock {
            idx1,
            idx2,
            norms,
            values,
            batch: new_batch,
            heads: block.heads,
            seq: block.seq,
        })
    }

    fn truncate(&self, block: &Self::Block, len: i32) -> Result<Self::Block> {
        // Keep the first `len` positions along the sequence axis. Reconstruct grids, slice, re-pack.
        let d = self.inner.d;
        let n_vec = block.batch * block.heads * block.seq;
        let n_codes = (n_vec * d) as usize;
        let idx = Array::from_slice(&(0..len).collect::<Vec<_>>(), &[len]);

        let g1 = unpack_codes(&block.idx1, n_codes, self.inner.b)?
            .reshape(&[block.batch, block.heads, block.seq, d])?
            .take_axis(&idx, SEQ_AXIS)?;
        let g2 = unpack_codes(&block.idx2, n_codes, self.inner.b)?
            .reshape(&[block.batch, block.heads, block.seq, d])?
            .take_axis(&idx, SEQ_AXIS)?;

        let idx1 = pack_codes(&g1, self.inner.b)?;
        let idx2 = pack_codes(&g2, self.inner.b)?;
        let norms = block.norms.take_axis(&idx, SEQ_AXIS)?;
        let values = block.values.take_axis(&idx, SEQ_AXIS)?;

        Ok(RvqBlock {
            idx1,
            idx2,
            norms,
            values,
            batch: block.batch,
            heads: block.heads,
            seq: len,
        })
    }
}

/// Pack a `u8` code array (any shape) into a tight `b`-bit `u8` buffer, padding the flat code count up
/// to a multiple of `8/b` with zeros (the pad codes are sliced off on [`unpack_codes`]). This keeps
/// the bit-packer's group-alignment requirement satisfied for any per-vector geometry.
fn pack_codes(codes: &Array, b: i32) -> Result<Array> {
    let flat: Vec<u8> = codes
        .as_dtype(Dtype::Uint8)?
        .reshape(&[codes.size() as i32])?
        .as_slice::<u8>()
        .to_vec();
    let per_byte = (8 / b) as usize;
    let pad = (per_byte - flat.len() % per_byte) % per_byte;
    let padded: Vec<u8> = if pad == 0 {
        flat
    } else {
        let mut v = flat;
        v.extend(std::iter::repeat_n(0u8, pad));
        v
    };
    let arr = Array::from_slice(&padded, &[padded.len() as i32]);
    bit_pack(&arr, b)
}

/// Inverse of [`pack_codes`]: unpack `n_codes` codes (the unpadded count) from a packed buffer. Unpacks
/// the padded length, then slices back to `n_codes`.
fn unpack_codes(packed: &Array, n_codes: usize, b: i32) -> Result<Array> {
    let per_byte = (8 / b) as usize;
    let padded_n = n_codes.div_ceil(per_byte) * per_byte;
    let full = bit_unpack(packed, padded_n, b)?;
    if padded_n == n_codes {
        Ok(full)
    } else {
        let idx = Array::from_slice(&(0..n_codes as i32).collect::<Vec<_>>(), &[n_codes as i32]);
        Ok(full.take_axis(&idx, 0)?)
    }
}

/// Estimate `⟨q, k⟩` from an RVQ-encoded key block without a full dense decode — the rotated-query
/// trick the upstream `estimate_inner_product` uses, exposed so a future attention kernel (story A)
/// can score against compressed keys directly. Returns `[batch, heads, seq]` inner products.
///
/// Currently unused by the cache decode path (which materializes dense K), but kept as the documented
/// Phase-2 seam so the capability surface is complete rather than a happy-path slice.
pub fn estimate_inner_product(q: &Array, quant: &RvqQuantizer, block: &RvqBlock) -> Result<Array> {
    let d = quant.inner.d;
    let n_vec = block.batch * block.heads * block.seq;
    let n_codes = (n_vec * d) as usize;

    let idx1 = unpack_codes(&block.idx1, n_codes, quant.inner.b)?.reshape(&[n_vec, d])?;
    let idx2 = unpack_codes(&block.idx2, n_codes, quant.inner.b)?.reshape(&[n_vec, d])?;
    let yhat = mlx_rs::ops::add(
        &quant.inner.cb1.dequantize(&idx1)?,
        &quant.inner.cb2.dequantize(&idx2)?,
    )?; // [n_vec, d], rotated space

    // Rotate the query into the same space and dot. q is [d] or [.., d]; flatten to [d].
    let q_rot = quant
        .rotate(&q.as_dtype(Dtype::Float32)?.reshape(&[1, d])?)?
        .reshape(&[d, 1])?;
    let ip = matmul(&yhat, &q_rot)?.reshape(&[block.batch, block.heads, block.seq])?;
    // Re-scale by the stored per-vector norm (we dotted the unit vector).
    let norms =
        block
            .norms
            .as_dtype(Dtype::Float32)?
            .reshape(&[block.batch, block.heads, block.seq])?;
    Ok(multiply(&ip, &norms)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{mean, square, subtract};
    use mlx_rs::random::{key, normal};

    fn f32s(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// A `[batch, heads, seq, d]` tensor of seeded normal noise.
    fn noise(batch: i32, heads: i32, seq: i32, d: i32, seed: u64) -> Array {
        let k = key(seed).unwrap();
        normal::<f32>(&[batch, heads, seq, d], None, None, &k).unwrap()
    }

    /// Mean cosine similarity between rows of two `[.., d]` tensors flattened to `[n, d]`.
    fn mean_cosine(a: &Array, b: &Array, d: i32) -> f32 {
        let n = a.size() as i32 / d;
        let av = f32s(&a.reshape(&[n, d]).unwrap());
        let bv = f32s(&b.reshape(&[n, d]).unwrap());
        let mut acc = 0.0f32;
        for r in 0..n as usize {
            let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for j in 0..d as usize {
                let x = av[r * d as usize + j];
                let y = bv[r * d as usize + j];
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            acc += dot / (na.sqrt() * nb.sqrt() + 1e-9);
        }
        acc / n as f32
    }

    /// Encode→decode round-trip reconstructs key direction within the documented cosine threshold
    /// (the upstream quality target: ~0.92 cosine at d=128 b=1, ~0.98 at b=2), and values pass through
    /// losslessly.
    #[test]
    fn rvq_roundtrip_cosine_within_threshold() {
        for (d, b, min_cos) in [
            (64, 1, 0.85f32),
            (64, 2, 0.95),
            (128, 1, 0.88),
            (128, 2, 0.96),
        ] {
            let q = RvqQuantizer::new(d, b, 42).unwrap();
            let keys = noise(2, 3, 8, d, 7);
            let vals = noise(2, 3, 8, d, 9);
            let block = q.encode(&keys, &vals).unwrap();
            let (kd, vd) = q.decode(&block).unwrap();
            let cos = mean_cosine(&keys, &kd, d);
            assert!(
                cos >= min_cos,
                "d={d} b={b}: cosine {cos} below threshold {min_cos}"
            );
            // Values are passed through dense → exactly equal.
            assert_eq!(f32s(&vals), f32s(&vd));
        }
    }

    /// `block_bytes` reflects the real compressed size and shows >1× key compression vs dense bf16.
    #[test]
    fn rvq_block_bytes_compresses_keys() {
        let d = 128;
        let b = 1;
        let q = RvqQuantizer::new(d, b, 42).unwrap();
        let (batch, heads, seq) = (1, 4, 64);
        let keys = noise(batch, heads, seq, d, 1)
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let vals = noise(batch, heads, seq, d, 2)
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let block = q.encode(&keys, &vals).unwrap();

        // Key-only compressed bytes = two packed index sets + fp16 norms.
        let key_compressed = block.idx1.nbytes() + block.idx2.nbytes() + block.norms.nbytes();
        let key_dense = keys.nbytes(); // bf16 dense keys
        let ratio = key_dense as f64 / key_compressed as f64;
        // At d=128, b=1: 2 packed bits/coord (2/8 byte) + 2 bytes norm/vector vs 256 bytes fp16/vector.
        // 34 bytes vs 256 → ~7.5×. Allow a margin for the fp16 (not fp16-vs-bf16) accounting.
        assert!(
            ratio > 5.0,
            "key compression {ratio:.2}× should exceed 5× at d=128 b=1"
        );

        // block_bytes is the faithful total (keys compressed + dense values).
        let total = q.block_bytes(&block);
        assert_eq!(total, key_compressed + vals.nbytes());
    }

    /// `retain_sequences` compacts the batch axis and keeps the kept rows' reconstruction intact.
    #[test]
    fn rvq_retain_sequences_compacts_batch() {
        let d = 64;
        let q = RvqQuantizer::new(d, 2, 42).unwrap();
        let keys = noise(3, 2, 4, d, 11);
        let vals = noise(3, 2, 4, d, 13);
        let block = q.encode(&keys, &vals).unwrap();

        let keep = Array::from_slice(&[2i32, 0], &[2]);
        let compact = q.retain_sequences(&block, &keep).unwrap();
        assert_eq!(q.batch_size(&compact), 2);
        assert_eq!(q.seq_len(&compact), 4);

        // Decoded values for the kept rows match the originals (values are lossless).
        let (_, vd) = q.decode(&compact).unwrap();
        let orig = f32s(&vals.reshape(&[3, 2, 4, d]).unwrap());
        let got = f32s(&vd);
        // Row 2 then row 0 of the original batch.
        let per_row = (2 * 4 * d) as usize;
        assert_eq!(&got[..per_row], &orig[2 * per_row..3 * per_row]);
        assert_eq!(&got[per_row..2 * per_row], &orig[..per_row]);
    }

    /// `truncate` keeps the first `len` sequence positions; the kept positions reconstruct unchanged.
    #[test]
    fn rvq_truncate_slices_sequence() {
        let d = 64;
        let q = RvqQuantizer::new(d, 2, 42).unwrap();
        let keys = noise(1, 1, 6, d, 21);
        let vals = noise(1, 1, 6, d, 23);
        let block = q.encode(&keys, &vals).unwrap();

        let cut = q.truncate(&block, 4).unwrap();
        assert_eq!(q.seq_len(&cut), 4);
        let (kd, _) = q.decode(&cut).unwrap();
        assert_eq!(kd.shape(), &[1, 1, 4, d]);
    }

    /// Stage-2 residual quantization strictly improves reconstruction over stage-1 alone (the whole
    /// point of the second pass) — measured as lower MSE on the unit-normalized rotated keys.
    #[test]
    fn rvq_second_stage_reduces_error() {
        let d = 128;
        let b = 2;
        let q = RvqQuantizer::new(d, b, 42).unwrap();
        let keys = noise(1, 2, 16, d, 31);

        // Replicate stage-1-only vs two-stage reconstruction in rotated space.
        let n = 2 * 16;
        let flat = keys.reshape(&[n, d]).unwrap();
        let sumsq = mlx_rs::ops::sum_axis(multiply(&flat, &flat).unwrap(), -1, true).unwrap();
        let safe = maximum(sqrt(&sumsq).unwrap(), Array::from_f32(1e-4)).unwrap();
        let unit = mlx_rs::ops::divide(&flat, &safe).unwrap();
        let y = q.rotate(&unit).unwrap();
        let idx1 = q.inner.cb1.quantize(&y).unwrap();
        let yhat1 = q.inner.cb1.dequantize(&idx1).unwrap();
        let r1 = subtract(&y, &yhat1).unwrap();
        let idx2 = q.inner.cb2.quantize(&r1).unwrap();
        let yhat2 = q.inner.cb2.dequantize(&idx2).unwrap();
        let yhat = mlx_rs::ops::add(&yhat1, &yhat2).unwrap();

        let mse1 = mean(square(subtract(&y, &yhat1).unwrap()).unwrap(), None).unwrap();
        let mse2 = mean(square(subtract(&y, &yhat).unwrap()).unwrap(), None).unwrap();
        let (e1, e2) = (mse1.item::<f32>(), mse2.item::<f32>());
        assert!(e2 < e1, "two-stage MSE {e2} should beat stage-1 {e1}");
    }

    #[test]
    fn rvq_rejects_bad_config() {
        assert!(RvqQuantizer::new(64, 3, 42).is_err()); // b=3 not a packing width
        assert!(RvqQuantizer::new(100, 1, 42).is_err()); // 100 = 4·25 not Hadamard-compatible (no m·2^k)
        assert!(RvqQuantizer::new(0, 1, 42).is_err());
        // Power-of-two and m·2^k dims ARE accepted (e.g. 192 = 12·16).
        assert!(RvqQuantizer::new(192, 1, 42).is_ok());
    }
}
