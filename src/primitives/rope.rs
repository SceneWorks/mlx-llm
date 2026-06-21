//! Rotary position embeddings.
//!
//! Modelled byte-for-byte on the working mlx-gen RoPE (prompt-refine / JoyCaption `Llama3Rope`,
//! sensenova/flux2 `apply_rope`): inverse frequencies and the cos/sin tables are built on the host
//! and lifted to the device; rotation uses the GPT-NeoX / HF **rotate-half** (split-the-head-in-half)
//! convention, not the interleaved-pairs convention. We deliberately keep this hand-rolled rather
//! than calling `mlx_rs::fast::rope` so the Llama-3 NTK-by-parts frequency schedule stays explicit
//! and bit-comparable to the reference engines the generating stacks will be validated against.
//!
//! The family covered (story 7155): **standard** RoPE (also Qwen3 — same rotation, config theta),
//! and **Llama-3 scaled** RoPE (the NTK-by-parts wavelength smoothing). [`Rope::cos_sin_at`] takes
//! explicit positions, which is what a 3-axis multimodal RoPE (sensenova MRoPE) is built from when
//! that consolidation lands.

use mlx_rs::ops::{add, concatenate_axis, cos, multiply, sin, split};
use mlx_rs::{Array, Dtype};

use crate::error::Result;

/// A rotary embedding: the host-side inverse-frequency table plus the head dimension it rotates.
#[derive(Clone, Debug)]
pub struct Rope {
    /// `inv_freq[i]`, length `head_dim / 2`.
    inv_freq: Vec<f32>,
    /// Head dimension (the size of the last axis RoPE rotates).
    dim: i32,
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
        }
    }

    /// The head dimension this RoPE rotates.
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// Inverse frequencies (length `head_dim / 2`).
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// Build `(cos, sin)` tables for `seq_len` contiguous positions starting at `offset`. Each is
    /// `[1, seq_len, head_dim]` in `dtype` (pass `Dtype::Bfloat16` to match the bf16 decoders, or
    /// `Dtype::Float32` for the f32 vision path).
    pub fn cos_sin(&self, seq_len: i32, offset: i32, dtype: Dtype) -> Result<(Array, Array)> {
        let positions: Vec<i32> = (0..seq_len).map(|s| offset + s).collect();
        self.cos_sin_at(&positions, dtype)
    }

    /// Build `(cos, sin)` tables for an explicit list of positions — the building block for packed
    /// / paged batches and for multi-axis (3D) RoPE, which is three of these concatenated.
    pub fn cos_sin_at(&self, positions: &[i32], dtype: Dtype) -> Result<(Array, Array)> {
        let half = self.inv_freq.len();
        let n = positions.len() as i32;
        let mut freqs = Vec::with_capacity(positions.len() * half);
        for &pos in positions {
            for &f in &self.inv_freq {
                freqs.push(pos as f32 * f);
            }
        }
        let freqs = Array::from_slice(&freqs, &[n, half as i32]);
        // Duplicate the half-table to the full head dim: [f0..f_{h-1}, f0..f_{h-1}].
        let emb = concatenate_axis(&[&freqs, &freqs], 1)?;
        let cos_t = cos(&emb)?.reshape(&[1, n, self.dim])?.as_dtype(dtype)?;
        let sin_t = sin(&emb)?.reshape(&[1, n, self.dim])?.as_dtype(dtype)?;
        Ok((cos_t, sin_t))
    }
}

/// Apply rotary embeddings to `x` (rotate-half / GPT-NeoX convention).
///
/// `x` is `[batch, seq, heads, head_dim]` (RoPE is applied before the transpose into
/// `[batch, heads, seq, head_dim]`); `cos`/`sin` are `[1, seq, head_dim]` and broadcast over heads.
pub fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let cos = cos.expand_dims(2)?; // [1, seq, 1, head_dim]
    let sin = sin.expand_dims(2)?;
    let parts = split(x, 2, 3)?; // two halves of the head dim
    let rot = concatenate_axis(&[&parts[1].negative()?, &parts[0]], 3)?;
    Ok(add(&multiply(x, &cos)?, &multiply(&rot, &sin)?)?)
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
        let y = apply_rope(&x, &c, &s).unwrap();
        let yh = y.as_slice::<f32>().to_vec();
        let xh = x.as_slice::<f32>().to_vec();
        for (a, b) in xh.iter().zip(&yh) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
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
        let y = apply_rope(&x, &c, &s).unwrap();
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
