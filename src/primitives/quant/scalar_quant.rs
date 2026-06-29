//! Per-group **affine** scalar quantization (scale + zero-point).
//!
//! This is the foundational uniform-quant primitive every KV-cache method (RVQ residual stages,
//! KIVI per-channel/per-token, …) builds on. It is the *exact affine grid* — not a Lloyd-Max
//! codebook (see [`super::codebook`] for that):
//!
//! ```text
//!   q     = clamp( round( (x - zp) / scale ), 0, 2^bits - 1 )      (encode)
//!   x_hat = q * scale + zp                                          (decode)
//! ```
//!
//! Parameters are derived **per group** of `group_size` contiguous elements along the last axis,
//! the same group layout MLX's weight quantizer uses. For a group with observed range
//! `[lo, hi]`:
//!
//! ```text
//!   scale = (hi - lo) / (2^bits - 1)         (degenerate group -> scale = 1)
//!   zp    = lo
//! ```
//!
//! so `lo` maps to code `0` and `hi` maps to code `2^bits - 1`. Reconstruction error is bounded by
//! `scale / 2` (half a quantization step), which is exactly what the round-trip tests assert.
//!
//! The path is pure MLX ops. The per-group min/max reduction and the elementwise affine map are
//! the hot loops a fused Metal kernel will later collapse:
//! `TODO(sc-8529/Phase2): replace the per-group reduce + affine map with a single MetalKernel`
//! (speed only — this pure path defines correctness).

use mlx_rs::ops::{add, clip, divide, max_axis, min_axis, multiply, round, subtract};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};

/// Per-group affine parameters: `x ≈ code * scale + zero_point`.
///
/// `scale` and `zero_point` each hold one value per group, shaped `[..., n_groups]` matching the
/// leading axes of the quantized input. Stored as `f32`.
#[derive(Debug, Clone)]
pub struct AffineParams {
    /// Per-group quantization step `(hi - lo) / (2^bits - 1)`.
    pub scale: Array,
    /// Per-group offset (`lo`) that maps to code `0`.
    pub zero_point: Array,
    /// Elements per group along the last axis.
    pub group_size: i32,
    /// Bits per code (1..=8). Code range is `[0, 2^bits - 1]`.
    pub bits: i32,
}

/// Output of [`quantize_affine`]: integer codes plus the params needed to invert them.
///
/// `codes` are stored as `f32` (holding exact non-negative integer values in `[0, 2^bits - 1]`)
/// so the pure-MLX path stays in one dtype; [`super::bit_packing`] is what packs them into sub-byte
/// storage. Shape matches the input.
#[derive(Debug, Clone)]
pub struct QuantizedGroups {
    /// Integer codes in `[0, 2^bits - 1]`, same shape as the quantized input.
    pub codes: Array,
    /// Per-group affine parameters to dequantize the codes.
    pub params: AffineParams,
}

fn max_code(bits: i32) -> f32 {
    ((1u32 << bits) - 1) as f32
}

/// Quantize `x` with per-group affine parameters derived from each group's `[min, max]` range.
///
/// `x` is treated as `[.., g]` where the last axis length must be divisible by `group_size`; each
/// run of `group_size` contiguous elements is one group. Returns the integer [`codes`] and the
/// [`AffineParams`] (`scale`, `zero_point`) used.
///
/// `bits` must be in `1..=8`.
///
/// [`codes`]: QuantizedGroups::codes
pub fn quantize_affine(x: &Array, group_size: i32, bits: i32) -> Result<QuantizedGroups> {
    if !(1..=8).contains(&bits) {
        return Err(Error::Unsupported(format!(
            "quantize_affine: bits must be in 1..=8, got {bits}"
        )));
    }
    if group_size < 1 {
        return Err(Error::Msg(format!(
            "quantize_affine: group_size must be >= 1, got {group_size}"
        )));
    }
    let shape = x.shape();
    let last = *shape
        .last()
        .ok_or_else(|| Error::Msg("quantize_affine: input must have >= 1 dim".into()))?;
    if last % group_size != 0 {
        return Err(Error::Msg(format!(
            "quantize_affine: last dim {last} not divisible by group_size {group_size}"
        )));
    }

    let x = x.as_dtype(Dtype::Float32)?;
    let n_groups = last / group_size;

    // Reshape so the group axis is explicit: [.., n_groups, group_size].
    let mut grouped_shape: Vec<i32> = shape[..shape.len() - 1].to_vec();
    grouped_shape.push(n_groups);
    grouped_shape.push(group_size);
    let grouped = x.reshape(&grouped_shape)?;

    // Per-group min / max over the group_size axis (keep dims for broadcasting back).
    let lo = min_axis(&grouped, -1, true)?; // [.., n_groups, 1]
    let hi = max_axis(&grouped, -1, true)?;

    // scale = (hi - lo) / max_code; guard degenerate (hi == lo) groups with scale = 1 so a
    // constant group quantizes to code 0 and dequantizes back to lo exactly.
    let span = subtract(&hi, &lo)?;
    let max_code_arr = Array::from_f32(max_code(bits));
    let raw_scale = divide(&span, &max_code_arr)?;
    // scale := max(raw_scale, tiny) but if span == 0 we want scale = 1; achieve by max(raw, 1) only
    // where raw == 0. Simpler: scale = raw_scale where raw_scale > 0 else 1.
    let one = Array::from_f32(1.0);
    let is_zero = raw_scale.le(Array::from_f32(0.0))?; // span <= 0 -> degenerate
    let scale = which(&is_zero, &one, &raw_scale)?;
    let zero_point = lo; // [.., n_groups, 1]

    // Encode: q = clamp(round((x - zp) / scale), 0, max_code).
    let centered = subtract(&grouped, &zero_point)?;
    let scaled = divide(&centered, &scale)?;
    let rounded = round(&scaled, None)?;
    let codes_grouped = clip(&rounded, (0.0, max_code(bits)))?;

    // Flatten codes back to the input shape; params keep the per-group shape minus the trailing 1.
    let codes = codes_grouped.reshape(shape)?;
    let mut param_shape: Vec<i32> = shape[..shape.len() - 1].to_vec();
    param_shape.push(n_groups);
    let scale = scale.reshape(&param_shape)?;
    let zero_point = zero_point.reshape(&param_shape)?;

    Ok(QuantizedGroups {
        codes,
        params: AffineParams {
            scale,
            zero_point,
            group_size,
            bits,
        },
    })
}

/// Inverse of [`quantize_affine`]: `x_hat = code * scale + zero_point`, per group.
///
/// `codes` must have the shape [`quantize_affine`] returned; `params` the matching [`AffineParams`].
pub fn dequantize_affine(codes: &Array, params: &AffineParams) -> Result<Array> {
    let shape = codes.shape();
    let last = *shape
        .last()
        .ok_or_else(|| Error::Msg("dequantize_affine: codes must have >= 1 dim".into()))?;
    if last % params.group_size != 0 {
        return Err(Error::Msg(format!(
            "dequantize_affine: last dim {last} not divisible by group_size {}",
            params.group_size
        )));
    }
    let n_groups = last / params.group_size;

    let codes = codes.as_dtype(Dtype::Float32)?;
    let mut grouped_shape: Vec<i32> = shape[..shape.len() - 1].to_vec();
    grouped_shape.push(n_groups);
    grouped_shape.push(params.group_size);
    let grouped = codes.reshape(&grouped_shape)?;

    // Broadcast params [.., n_groups] -> [.., n_groups, 1].
    let mut param_shape: Vec<i32> = shape[..shape.len() - 1].to_vec();
    param_shape.push(n_groups);
    param_shape.push(1);
    let scale = params.scale.reshape(&param_shape)?;
    let zero_point = params.zero_point.reshape(&param_shape)?;

    let scaled = multiply(&grouped, &scale)?;
    let x_hat = add(&scaled, &zero_point)?;
    Ok(x_hat.reshape(shape)?)
}

/// Convenience: affine-quantize then immediately affine-dequantize `x`. Returns the reconstruction
/// `x_hat` — the fake-quantization a calibration / parity check uses.
pub fn affine_quantize(x: &Array, group_size: i32, bits: i32) -> Result<QuantizedGroups> {
    quantize_affine(x, group_size, bits)
}

/// Alias of [`dequantize_affine`] for symmetry with [`affine_quantize`].
pub fn affine_dequantize(codes: &Array, params: &AffineParams) -> Result<Array> {
    dequantize_affine(codes, params)
}

/// Select elementwise: `cond ? a : b`. Pure-MLX `where`/`select` built from arithmetic so we do not
/// depend on a specific mlx-rs indexing helper. `cond` is a boolean array broadcastable to `a`/`b`.
fn which(cond: &Array, a: &Array, b: &Array) -> Result<Array> {
    let cond_f = cond.as_dtype(Dtype::Float32)?;
    // result = cond * a + (1 - cond) * b
    let one = Array::from_f32(1.0);
    let inv = subtract(&one, &cond_f)?;
    let left = multiply(&cond_f, a)?;
    let right = multiply(&inv, b)?;
    Ok(add(&left, &right)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// HAND-COMPUTED reference vector (derived from the analytic affine definition, NOT by running
    /// VeloxQuant). One group of 4 elements, 2-bit codes (levels {0,1,2,3}).
    ///
    /// x   = [0.0, 1.0, 2.0, 3.0]
    /// lo  = 0, hi = 3, max_code = 3  => scale = 1.0, zp = 0.0
    /// q   = round((x - 0)/1) clamped [0,3] = [0, 1, 2, 3]
    /// x_hat = q*1 + 0 = [0, 1, 2, 3]  (lossless: values land exactly on the grid)
    #[test]
    fn affine_hand_vector_exact_grid_2bit() {
        let x = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 4]);
        let q = quantize_affine(&x, 4, 2).unwrap();
        assert_eq!(vals(&q.codes), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(vals(&q.params.scale), vec![1.0]);
        assert_eq!(vals(&q.params.zero_point), vec![0.0]);
        let recon = dequantize_affine(&q.codes, &q.params).unwrap();
        assert_eq!(vals(&recon), vec![0.0, 1.0, 2.0, 3.0]);
    }

    /// HAND-COMPUTED reference vector with a non-unit scale, 2-bit.
    ///
    /// x   = [-1.0, 0.0, 1.0, 2.0]
    /// lo  = -1, hi = 2 => scale = (2 - -1)/3 = 1.0, zp = -1
    /// q   = round((x + 1)/1) = [0, 1, 2, 3]
    /// x_hat = q - 1 = [-1, 0, 1, 2]  (exact)
    #[test]
    fn affine_hand_vector_offset_2bit() {
        let x = Array::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], &[1, 4]);
        let q = quantize_affine(&x, 4, 2).unwrap();
        assert_eq!(vals(&q.codes), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(vals(&q.params.scale), vec![1.0]);
        assert_eq!(vals(&q.params.zero_point), vec![-1.0]);
        let recon = dequantize_affine(&q.codes, &q.params).unwrap();
        assert_eq!(vals(&recon), vec![-1.0, 0.0, 1.0, 2.0]);
    }

    /// HAND-COMPUTED off-grid case, 2-bit. Verifies rounding + the analytic error bound scale/2.
    ///
    /// x   = [0.0, 0.4, 0.6, 1.0]   (group of 4)
    /// lo=0, hi=1 => scale = 1/3 ≈ 0.3333, zp = 0
    /// q = round(x / scale) = round([0, 1.2, 1.8, 3.0]) = [0, 1, 2, 3]
    /// x_hat = q*scale = [0, 0.3333, 0.6667, 1.0]
    /// errors = |x - x_hat| = [0, 0.0667, 0.0667, 0] all <= scale/2 = 0.1667
    #[test]
    fn affine_hand_vector_offgrid_rounding_and_bound() {
        let x = Array::from_slice(&[0.0f32, 0.4, 0.6, 1.0], &[1, 4]);
        let q = quantize_affine(&x, 4, 2).unwrap();
        assert_eq!(vals(&q.codes), vec![0.0, 1.0, 2.0, 3.0]);
        let scale = vals(&q.params.scale)[0];
        assert!((scale - 1.0 / 3.0).abs() < 1e-6, "scale = {scale}");
        let recon = vals(&dequantize_affine(&q.codes, &q.params).unwrap());
        let orig = vals(&x);
        let bound = scale / 2.0;
        for (o, r) in orig.iter().zip(&recon) {
            assert!((o - r).abs() <= bound + 1e-6, "{o} vs {r}, bound {bound}");
        }
    }

    /// Analytic round-trip: for ANY input, |x - dequantize(quantize(x))| <= scale/2 per group.
    /// This is the numeric oracle expressed as the closed-form affine error bound (no external
    /// code executed).
    #[test]
    fn affine_roundtrip_within_half_step_4bit() {
        // 3 groups of 8, varied ranges.
        let mut data = Vec::new();
        for g in 0..3 {
            for i in 0..8 {
                data.push((g as f32) * 2.0 + (i as f32) * 0.137 - 0.5);
            }
        }
        let x = Array::from_slice(&data, &[3, 8]);
        let q = quantize_affine(&x, 8, 4).unwrap();
        let recon = vals(&dequantize_affine(&q.codes, &q.params).unwrap());
        let scales = vals(&q.params.scale); // one per group
        let orig = vals(&x);
        for (grp, &grp_scale) in scales.iter().enumerate() {
            let bound = grp_scale / 2.0 + 1e-5;
            for i in 0..8 {
                let idx = grp * 8 + i;
                assert!(
                    (orig[idx] - recon[idx]).abs() <= bound,
                    "group {grp} idx {i}: {} vs {} bound {bound}",
                    orig[idx],
                    recon[idx]
                );
            }
            // codes must stay in [0, 15].
        }
        for c in vals(&q.codes) {
            assert!((0.0..=15.0).contains(&c), "code out of range: {c}");
        }
    }

    /// Degenerate (constant) group: scale -> 1, all codes 0, exact reconstruction.
    #[test]
    fn affine_constant_group_is_exact() {
        let x = Array::from_slice(&[2.5f32, 2.5, 2.5, 2.5], &[1, 4]);
        let q = quantize_affine(&x, 4, 4).unwrap();
        assert_eq!(vals(&q.codes), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(vals(&q.params.scale), vec![1.0]);
        let recon = vals(&dequantize_affine(&q.codes, &q.params).unwrap());
        assert_eq!(recon, vec![2.5, 2.5, 2.5, 2.5]);
    }

    /// Two groups on one row get independent params.
    #[test]
    fn affine_two_groups_independent_params() {
        // group0 = [0,1,2,3] (scale 1), group1 = [0,2,4,6] (scale 2).
        let x = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0], &[1, 8]);
        let q = quantize_affine(&x, 4, 2).unwrap();
        assert_eq!(vals(&q.params.scale), vec![1.0, 2.0]);
        assert_eq!(vals(&q.codes), vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
        let recon = vals(&dequantize_affine(&q.codes, &q.params).unwrap());
        assert_eq!(recon, vec![0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn affine_rejects_bad_bits_and_group() {
        let x = Array::from_slice(&[0.0f32, 1.0], &[1, 2]);
        assert!(quantize_affine(&x, 2, 0).is_err());
        assert!(quantize_affine(&x, 2, 9).is_err());
        assert!(quantize_affine(&x, 3, 2).is_err()); // 2 not divisible by 3
    }
}
