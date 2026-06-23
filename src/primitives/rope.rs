//! Rotary position embeddings.
//!
//! Modelled byte-for-byte on the working mlx-gen RoPE (prompt-refine / JoyCaption `Llama3Rope`,
//! sensenova/flux2 `apply_rope`): inverse frequencies and the cos/sin tables are built on the host
//! and lifted to the device. Rotation supports both the GPT-NeoX / HF **rotate-half**
//! (split-the-head-in-half) convention and the GPT-J **interleaved** (adjacent even/odd pairs)
//! convention, over the full head dim or a leading **partial** slice. We deliberately keep this
//! hand-rolled rather than calling `mlx_rs::fast::rope` so the Llama-3 / YaRN frequency schedules stay
//! explicit and bit-comparable to the reference engines the generating stacks are validated against.
//!
//! The family covered: **standard** RoPE (also Qwen3 — same rotation, config theta), **Llama-3
//! scaled** RoPE (NTK-by-parts wavelength smoothing), **partial + interleaved** RoPE (GLM-4, the
//! DeepSeek MLA rope sub-vector; story 7399), and **YaRN-scaled** RoPE (DeepSeek-V2; story 7398).
//! [`Rope::cos_sin_at`] takes explicit positions, which is what a 3-axis multimodal RoPE (sensenova
//! MRoPE) is built from when that consolidation lands.

use mlx_rs::ops::{add, concatenate_axis, cos, multiply, sin, split, split_sections};
use mlx_rs::{Array, Dtype};

use crate::error::Result;

/// A rotary embedding: the host-side inverse-frequency table, the head dimension it rotates, and the
/// pairing convention.
#[derive(Clone, Debug)]
pub struct Rope {
    /// `inv_freq[i]`, length `dim / 2`.
    inv_freq: Vec<f32>,
    /// The number of (last-axis) dimensions RoPE rotates — `head_dim` for full rotary, less for a
    /// partial schedule (GLM-4, DeepSeek MLA). Equals `inv_freq.len() * 2`.
    dim: i32,
    /// Pairing convention: `false` ⇒ NeoX half-split (the default); `true` ⇒ GPT-J interleaved
    /// (adjacent even/odd dims form a pair) — GLM-4 and the DeepSeek MLA rope sub-vector.
    interleaved: bool,
}

impl Rope {
    /// Standard RoPE: `inv_freq[i] = theta^(-2i / head_dim)`.
    ///
    /// This also covers **Qwen3** (Qwen3 uses standard RoPE — its distinctive per-head q/k norm
    /// lives in attention, not here) by passing the model's `rope_theta` (e.g. `1_000_000`).
    pub fn standard(head_dim: i32, theta: f32) -> Self {
        let half = (head_dim / 2) as usize;
        let inv_freq = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        Self {
            inv_freq,
            dim: head_dim,
            interleaved: false,
        }
    }

    /// Partial RoPE over the first `rotary_dim` dimensions (`inv_freq[i] = theta^(-2i / rotary_dim)`),
    /// leaving the remaining `head_dim − rotary_dim` dims unrotated. `interleaved` selects the GPT-J
    /// pairing (GLM-4) instead of NeoX half-split. With `rotary_dim == head_dim` and `interleaved ==
    /// false` this is exactly [`Rope::standard`].
    pub fn partial(rotary_dim: i32, theta: f32, interleaved: bool) -> Self {
        let half = (rotary_dim / 2) as usize;
        let inv_freq = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        Self {
            inv_freq,
            dim: rotary_dim,
            interleaved,
        }
    }

    /// YaRN-scaled RoPE (DeepSeek-V2's `rope_scaling` "yarn" schedule), over `rope_dim` dimensions.
    ///
    /// Each inverse frequency is a wavelength-ramped blend of the **extrapolated** frequency (the
    /// plain `theta^(-2i/rope_dim)`, kept for short-wavelength / high-frequency dims) and the
    /// **interpolated** frequency (divided by `factor`, for long-wavelength / low-frequency dims).
    /// The ramp boundaries are the dimensions whose rotation count crosses `beta_fast` / `beta_slow`
    /// over the `original_context`. Always interleaved (the DeepSeek pairing). The cos/sin magnitude
    /// scale (`mscale`) is `1.0` in the symmetric `mscale == mscale_all_dim` case the released models
    /// use, and is therefore not applied here; the attention-softmax `mscale²` lives in the attention
    /// scale ([`crate::config::ModelConfig::attn_scale`]).
    pub fn yarn(
        rope_dim: i32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_context: f32,
    ) -> Self {
        let dim = rope_dim as f32;
        let half = (rope_dim / 2) as usize;
        // The (fractional) dimension index whose wavelength completes `num_rotations` cycles over the
        // original context — the YaRN correction-range endpoints.
        let correction_dim = |num_rotations: f32| -> f32 {
            (dim * (original_context / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * theta.ln())
        };
        let low = correction_dim(beta_fast).floor().max(0.0);
        let high = correction_dim(beta_slow).ceil().min(dim - 1.0);
        // Guard a degenerate (zero-width) ramp, matching the reference's `min == max` nudge.
        let span = if (high - low).abs() < 1e-3 {
            1e-3
        } else {
            high - low
        };
        let inv_freq = (0..half)
            .map(|i| {
                let freq_extra = 1.0 / theta.powf((2 * i) as f32 / dim);
                let freq_inter = freq_extra / factor;
                // ramp 0 → extrapolate (high freq), 1 → interpolate (low freq).
                let ramp = ((i as f32 - low) / span).clamp(0.0, 1.0);
                freq_inter * ramp + freq_extra * (1.0 - ramp)
            })
            .collect();
        Self {
            inv_freq,
            dim: rope_dim,
            interleaved: true,
        }
    }

    /// Llama-3 scaled RoPE (the `rope_scaling` "llama3" NTK-by-parts schedule).
    ///
    /// Low-frequency components (long wavelength) are divided by `factor`; high-frequency
    /// components pass through unchanged; the band between is smoothly interpolated. With
    /// `factor == 1.0` this collapses to [`Rope::standard`].
    pub fn llama3(
        head_dim: i32,
        theta: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_context: f32,
    ) -> Self {
        let half = (head_dim / 2) as usize;
        let low_freq_wavelen = original_context / low_freq_factor;
        let high_freq_wavelen = original_context / high_freq_factor;
        let inv_freq = (0..half)
            .map(|i| {
                let inv = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let wavelen = 2.0 * std::f32::consts::PI / inv;
                if wavelen > low_freq_wavelen {
                    inv / factor
                } else if wavelen < high_freq_wavelen {
                    inv
                } else {
                    let smooth = (original_context / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    (1.0 - smooth) * inv / factor + smooth * inv
                }
            })
            .collect();
        Self {
            inv_freq,
            dim: head_dim,
            interleaved: false,
        }
    }

    /// The head dimension this RoPE rotates.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// Whether this RoPE uses the interleaved (GPT-J) pairing.
    pub fn interleaved(&self) -> bool {
        self.interleaved
    }

    /// Inverse frequencies (length `dim / 2`).
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// Build `(cos, sin)` tables for `seq_len` contiguous positions starting at `offset`. Each is
    /// `[1, seq_len, dim]` in `dtype` (pass `Dtype::Bfloat16` to match the bf16 decoders, or
    /// `Dtype::Float32` for the f32 vision path).
    pub fn cos_sin(&self, seq_len: i32, offset: i32, dtype: Dtype) -> Result<(Array, Array)> {
        let positions: Vec<i32> = (0..seq_len).map(|s| offset + s).collect();
        self.cos_sin_at(&positions, dtype)
    }

    /// Build `(cos, sin)` tables for an explicit list of positions — the building block for packed
    /// / paged batches and for multi-axis (3D) RoPE, which is three of these concatenated.
    pub fn cos_sin_at(&self, positions: &[i32], dtype: Dtype) -> Result<(Array, Array)> {
        let n = positions.len() as i32;
        // The per-position angle table, laid out to match the rotation convention:
        //   NeoX        → cat(freqs, freqs)         (`apply_rope` pairs dim i with i + dim/2)
        //   interleaved → each freq repeated twice  (`apply_rope` pairs dim 2i with 2i+1)
        let mut emb = Vec::with_capacity(positions.len() * self.dim as usize);
        for &pos in positions {
            if self.interleaved {
                for &f in &self.inv_freq {
                    emb.push(pos as f32 * f);
                    emb.push(pos as f32 * f);
                }
            } else {
                for &f in &self.inv_freq {
                    emb.push(pos as f32 * f);
                }
                for &f in &self.inv_freq {
                    emb.push(pos as f32 * f);
                }
            }
        }
        let emb = Array::from_slice(&emb, &[1, n, self.dim]);
        let cos_t = cos(&emb)?.as_dtype(dtype)?;
        let sin_t = sin(&emb)?.as_dtype(dtype)?;
        Ok((cos_t, sin_t))
    }
}

/// Apply rotary embeddings to `x`.
///
/// `x` is `[batch, seq, heads, head_dim]` (RoPE is applied before the transpose into
/// `[batch, heads, seq, head_dim]`); `cos`/`sin` are `[*, seq, rotary_dim]` and broadcast over heads.
/// Only the first `rotary_dim = cos.last_dim` dims are rotated (the rest pass through — partial RoPE,
/// GLM-4 / DeepSeek MLA); `interleaved` selects the GPT-J even/odd pairing instead of NeoX half-split.
pub fn apply_rope(x: &Array, cos: &Array, sin: &Array, interleaved: bool) -> Result<Array> {
    let head_dim = x.shape()[3];
    let rd = *cos.shape().last().unwrap(); // rotary_dim
    let cos = cos.expand_dims(2)?; // [*, seq, 1, rotary_dim]
    let sin = sin.expand_dims(2)?;

    // Split off the rotated prefix from the (optional) passed-through tail.
    let (x_rot, x_pass) = if rd < head_dim {
        let parts = split_sections(x, &[rd], 3)?;
        (parts[0].clone(), Some(parts[1].clone()))
    } else {
        (x.clone(), None)
    };

    let rotated = if interleaved {
        // Pairs (x[2i], x[2i+1]); rotate_half = interleave(-x_odd, x_even).
        let sh = x_rot.shape();
        let (b, s, h) = (sh[0], sh[1], sh[2]);
        let pairs = x_rot.reshape(&[b, s, h, rd / 2, 2])?;
        let halves = split(&pairs, 2, 4)?; // even = [..,0], odd = [..,1]
        let rot = concatenate_axis(&[&halves[1].negative()?, &halves[0]], 4)?.reshape(&[b, s, h, rd])?;
        add(&multiply(&x_rot, &cos)?, &multiply(&rot, &sin)?)?
    } else {
        // NeoX half-split: pairs (x[i], x[i + rd/2]); rotate_half = cat(-x2, x1).
        let halves = split(&x_rot, 2, 3)?;
        let rot = concatenate_axis(&[&halves[1].negative()?, &halves[0]], 3)?;
        add(&multiply(&x_rot, &cos)?, &multiply(&rot, &sin)?)?
    };

    match x_pass {
        Some(tail) => Ok(concatenate_axis(&[&rotated, &tail], 3)?),
        None => Ok(rotated),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_inv_freq_matches_formula() {
        let rope = Rope::standard(8, 10000.0);
        assert_eq!(rope.inv_freq().len(), 4);
        // inv_freq[0] is always 1.0 (theta^0); inv_freq[i] = theta^(-2i/dim).
        assert!((rope.inv_freq()[0] - 1.0).abs() < 1e-6);
        let expected1 = 1.0f32 / 10000.0f32.powf(2.0 / 8.0);
        assert!((rope.inv_freq()[1] - expected1).abs() < 1e-6);
        assert!(!rope.interleaved());
    }

    #[test]
    fn llama3_with_unit_factor_equals_standard() {
        let std_rope = Rope::standard(128, 500000.0);
        let l3 = Rope::llama3(128, 500000.0, 1.0, 1.0, 4.0, 8192.0);
        for (a, b) in std_rope.inv_freq().iter().zip(l3.inv_freq()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn llama3_scales_low_frequencies_down() {
        // With factor 8, the lowest-frequency (i.e. largest i) components get divided by ~8.
        let std_rope = Rope::standard(128, 500000.0);
        let l3 = Rope::llama3(128, 500000.0, 8.0, 1.0, 4.0, 8192.0);
        let last = std_rope.inv_freq().len() - 1;
        // The very lowest frequency is long-wavelength -> divided by factor.
        assert!(l3.inv_freq()[last] < std_rope.inv_freq()[last]);
        assert!((l3.inv_freq()[last] - std_rope.inv_freq()[last] / 8.0).abs() < 1e-9);
    }

    #[test]
    fn yarn_blends_extrapolated_and_interpolated_frequencies() {
        // DeepSeek-V2-Lite params: rope_dim 64, theta 1e4, factor 40, beta 32/1, orig ctx 4096.
        // Correction range is dims [10, 23]: below it the frequency is extrapolated (plain), above it
        // interpolated (plain / factor), with a linear ramp between.
        let rope = Rope::yarn(64, 10000.0, 40.0, 32.0, 1.0, 4096.0);
        assert!(rope.interleaved());
        assert_eq!(rope.dim(), 64);
        assert_eq!(rope.inv_freq().len(), 32);

        let extra = |i: usize| 1.0f32 / 10000f32.powf((2 * i) as f32 / 64.0);
        // High-frequency dims (i ≤ low=10) are extrapolated unchanged.
        assert!((rope.inv_freq()[0] - extra(0)).abs() < 1e-9); // == 1.0
        assert!((rope.inv_freq()[10] - extra(10)).abs() < 1e-6);
        // Low-frequency dims (i ≥ high=23) are interpolated: plain / factor.
        assert!((rope.inv_freq()[31] - extra(31) / 40.0).abs() < 1e-9);
        assert!((rope.inv_freq()[23] - extra(23) / 40.0).abs() < 1e-6);
        // A mid-band dim is a strict blend, between the two endpoints.
        let mid = rope.inv_freq()[16];
        assert!(mid < extra(16) && mid > extra(16) / 40.0, "{mid}");
        // Frequencies are monotonically decreasing across the head.
        for w in rope.inv_freq().windows(2) {
            assert!(w[0] > w[1], "not decreasing: {} !> {}", w[0], w[1]);
        }
    }

    #[test]
    fn cos_sin_shapes() {
        let rope = Rope::standard(16, 10000.0);
        let (c, s) = rope.cos_sin(5, 0, Dtype::Float32).unwrap();
        assert_eq!(c.shape(), &[1, 5, 16]);
        assert_eq!(s.shape(), &[1, 5, 16]);
    }

    #[test]
    fn position_zero_is_identity() {
        // At position 0, cos = 1 and sin = 0, so apply_rope is a no-op.
        let rope = Rope::standard(8, 10000.0);
        let (c, s) = rope.cos_sin(1, 0, Dtype::Float32).unwrap();
        // x: [b=1, seq=1, heads=1, head_dim=8]
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 1, 8]);
        let y = apply_rope(&x, &c, &s, false).unwrap();
        let yh = y.as_slice::<f32>().to_vec();
        let xh = x.as_slice::<f32>().to_vec();
        for (a, b) in xh.iter().zip(&yh) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn partial_rope_passes_through_unrotated_tail() {
        // rotary_dim 2 over a head_dim of 4: dims [2,4) must pass through unchanged at any position.
        let rope = Rope::partial(2, 10000.0, false);
        let (c, s) = rope.cos_sin(1, 3, Dtype::Float32).unwrap();
        assert_eq!(c.shape(), &[1, 1, 2]); // table width == rotary_dim
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        let y = apply_rope(&x, &c, &s, false).unwrap().as_slice::<f32>().to_vec();
        // Tail (indices 2,3) untouched.
        assert!((y[2] - 3.0).abs() < 1e-5 && (y[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn interleaved_rope_rotates_even_odd_pairs() {
        // One pair (a, b) at the base frequency (inv_freq[0] = 1), position p: the interleaved
        // convention rotates (a, b) -> (a·cosp − b·sinp, a·sinp + b·cosp).
        let rope = Rope::partial(2, 10000.0, true);
        let p = 1.0f32;
        let (c, s) = rope.cos_sin(1, 1, Dtype::Float32).unwrap();
        let (a, b) = (0.7f32, -0.3f32);
        let x = Array::from_slice(&[a, b], &[1, 1, 1, 2]);
        let y = apply_rope(&x, &c, &s, true).unwrap().as_slice::<f32>().to_vec();
        let (cp, sp) = (p.cos(), p.sin());
        assert!((y[0] - (a * cp - b * sp)).abs() < 1e-5, "{y:?}");
        assert!((y[1] - (a * sp + b * cp)).abs() < 1e-5, "{y:?}");
    }

    #[test]
    fn rotation_preserves_norm() {
        // RoPE is a rotation, so it preserves the L2 norm of each rotated pair.
        let rope = Rope::standard(4, 10000.0);
        let (c, s) = rope.cos_sin(3, 0, Dtype::Float32).unwrap();
        let x = Array::from_slice(
            &[1.0f32, 0.5, -0.5, 2.0, 0.3, 1.0, -1.0, 0.7, 2.0, -0.2, 0.1, 0.9],
            &[1, 3, 1, 4],
        );
        let y = apply_rope(&x, &c, &s, false).unwrap();
        let xh = x.as_slice::<f32>().to_vec();
        let yh = y.as_slice::<f32>().to_vec();
        let norm = |v: &[f32]| -> f32 { v.iter().map(|a| a * a).sum::<f32>().sqrt() };
        // each position is a 4-vector
        for pos in 0..3 {
            let xs = &xh[pos * 4..pos * 4 + 4];
            let ys = &yh[pos * 4..pos * 4 + 4];
            assert!((norm(xs) - norm(ys)).abs() < 1e-4, "pos {pos}");
        }
    }
}
