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
//! [`Rope::cos_sin_at`] takes explicit positions, which is what a 3-axis multimodal RoPE is built
//! from: [`Rope::mrope_cos_sin`] composes three of those into the Qwen2-VL / Qwen2.5-VL M-RoPE
//! tables (and serves the sensenova Qwen3 MRoPE consolidation).

use mlx_rs::ops::{add, concatenate_axis, cos, multiply, sin, split, split_sections};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};

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

    /// Build the 3-axis multimodal-RoPE `(cos, sin)` tables — Qwen2-VL / Qwen2.5-VL **M-RoPE**.
    ///
    /// `position_ids` holds the three position rows (temporal / height / width), each of length `L`
    /// (the sequence length); a text-only sequence sets all three rows equal. `sections` is the
    /// (un-doubled) `mrope_section` `[t, h, w]`, which must sum to `dim / 2`. The doubled sections
    /// `[t, h, w, t, h, w]` carve the `dim` channels so that each NeoX rotate-half pair `(c, c + dim/2)`
    /// draws its frequency from a single axis.
    ///
    /// Each axis gets its own [`Rope::cos_sin_at`] table (built in f32), the six channel chunks are
    /// stitched taking chunk `i` from axis `i % 3` (`apply_multimodal_rotary_pos_emb`), and the result
    /// is cast to `dtype`. Each returned table is `[1, L, dim]`. With all three rows equal this is
    /// bit-identical to a plain 1D [`Rope::cos_sin_at`] over that row.
    ///
    /// Requires the NeoX (half-split) convention: the doubled-section layout assumes the
    /// `cat(freqs, freqs)` channel order. The rotation is plain [`apply_rope`] (M-RoPE == RoPE once
    /// the tables are assembled). All released M-RoPE models are NeoX.
    pub fn mrope_cos_sin(
        &self,
        position_ids: [&[i32]; 3],
        sections: [usize; 3],
        dtype: Dtype,
    ) -> Result<(Array, Array)> {
        if self.interleaved {
            return Err(Error::Unsupported(
                "mrope_cos_sin requires the NeoX (half-split) convention".into(),
            ));
        }
        let half = self.inv_freq.len();
        let sum: usize = sections.iter().sum();
        if sum != half {
            return Err(Error::Msg(format!(
                "mrope sections {sections:?} sum to {sum}, expected dim/2 = {half}"
            )));
        }
        let l = position_ids[0].len();
        if position_ids.iter().any(|row| row.len() != l) {
            return Err(Error::Msg(format!(
                "mrope position_ids rows must share one length, got {:?}",
                position_ids.map(<[i32]>::len)
            )));
        }

        // Doubled-section cut points: e.g. [16,24,24,16,24,24] → exclusive cuts [16,40,64,80,104],
        // splitting the `dim` channels into the six chunks (the last width is implied by the end).
        let doubled = [
            sections[0],
            sections[1],
            sections[2],
            sections[0],
            sections[1],
            sections[2],
        ];
        let mut cuts = Vec::with_capacity(5);
        let mut acc = 0i32;
        for &d in doubled.iter().take(5) {
            acc += d as i32;
            cuts.push(acc);
        }

        // One rotary table per axis (over its own position row), each split into the six chunks.
        let mut cos_pieces: Vec<Vec<Array>> = Vec::with_capacity(3);
        let mut sin_pieces: Vec<Vec<Array>> = Vec::with_capacity(3);
        for row in position_ids {
            let (cos_t, sin_t) = self.cos_sin_at(row, Dtype::Float32)?;
            cos_pieces.push(split_sections(&cos_t, &cuts, 2)?);
            sin_pieces.push(split_sections(&sin_t, &cuts, 2)?);
        }

        // Stitch: channel chunk `i` is taken from axis `i % 3`, then cast to the request dtype.
        let cos_sel: Vec<&Array> = (0..6).map(|i| &cos_pieces[i % 3][i]).collect();
        let sin_sel: Vec<&Array> = (0..6).map(|i| &sin_pieces[i % 3][i]).collect();
        let cos_t = concatenate_axis(&cos_sel, 2)?.as_dtype(dtype)?;
        let sin_t = concatenate_axis(&sin_sel, 2)?.as_dtype(dtype)?;
        Ok((cos_t, sin_t))
    }

    /// Build the **interleaved** 3-axis multimodal-RoPE `(cos, sin)` tables — Qwen3.6's
    /// `mrope_interleaved` (`Qwen3_5TextRotaryEmbedding` + `apply_interleaved_mrope`).
    ///
    /// Unlike the Qwen2.5-VL chunked layout ([`Rope::mrope_cos_sin`], where each axis owns a
    /// contiguous block `[TTT…HHH…WWW]`), Qwen3.6 **interleaves** the axes across the `dim/2`
    /// frequency channels — round-robin `T,H,W,T,H,W,…` truncated by `sections`: channel `c` draws its
    /// position from axis `T` by default, from `H` for `c ∈ {1,4,7,…} ∩ [0, sections[1]·3)`, and from
    /// `W` for `c ∈ {2,5,8,…} ∩ [0, sections[2]·3)` (`sections` sums to `dim/2`). The frequency stays
    /// `inv_freq[c]`; only which position row feeds it changes. `emb = cat(freqs, freqs)` (NeoX), then
    /// cos/sin — so [`apply_rope`] (half-split) rotates it.
    ///
    /// `position_ids` holds the temporal / height / width rows, each length `L`. With all three rows
    /// equal (text-only) the axis choice is irrelevant and this is **bit-identical** to a plain 1D
    /// [`Rope::cos_sin_at`] over that row — the invariant that keeps the text path unchanged.
    pub fn mrope_interleaved_cos_sin(
        &self,
        position_ids: [&[i32]; 3],
        sections: [usize; 3],
        dtype: Dtype,
    ) -> Result<(Array, Array)> {
        if self.interleaved {
            return Err(Error::Unsupported(
                "mrope_interleaved_cos_sin requires the NeoX (half-split) convention".into(),
            ));
        }
        let half = self.inv_freq.len(); // dim/2
        let sum: usize = sections.iter().sum();
        if sum != half {
            return Err(Error::Msg(format!(
                "mrope sections {sections:?} sum to {sum}, expected dim/2 = {half}"
            )));
        }
        let l = position_ids[0].len();
        if position_ids.iter().any(|row| row.len() != l) {
            return Err(Error::Msg(format!(
                "mrope position_ids rows must share one length, got {:?}",
                position_ids.map(<[i32]>::len)
            )));
        }

        // Per-channel axis assignment: T (0) by default; H (1) / W (2) on their interleaved indices.
        let mut axis_of = vec![0usize; half];
        for (axis, offset) in [(1usize, 1usize), (2usize, 2usize)] {
            let length = (sections[axis] * 3).min(half);
            let mut c = offset;
            while c < length {
                axis_of[c] = axis;
                c += 3;
            }
        }

        // emb row = cat(freqs, freqs), freqs[c] = position_ids[axis_of[c]] · inv_freq[c].
        let mut emb = Vec::with_capacity(l * self.dim as usize);
        #[allow(clippy::needless_range_loop)] // p cross-indexes all three position rows
        for p in 0..l {
            let freqs: Vec<f32> = (0..half)
                .map(|c| position_ids[axis_of[c]][p] as f32 * self.inv_freq[c])
                .collect();
            emb.extend_from_slice(&freqs);
            emb.extend_from_slice(&freqs);
        }
        let emb = Array::from_slice(&emb, &[1, l as i32, self.dim]);
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

    // Qwen2.5-VL M-RoPE: head_dim 128, theta 1e6, mrope_section [16,24,24] (sums to head_dim/2).
    const QWEN_HEAD_DIM: i32 = 128;
    const QWEN_THETA: f32 = 1_000_000.0;
    const QWEN_SECTIONS: [usize; 3] = [16, 24, 24];

    #[test]
    fn mrope_text_equals_1d() {
        // Acceptance gate: text-only positions (all three axis rows equal) must reproduce a plain 1D
        // cos_sin_at over that row, bit-for-bit (split → re-concatenate moves data, never arithmetic).
        let rope = Rope::standard(QWEN_HEAD_DIM, QWEN_THETA);
        let positions: Vec<i32> = (0..7).collect();
        let (c1, s1) = rope.cos_sin_at(&positions, Dtype::Float32).unwrap();
        let (cm, sm) = rope
            .mrope_cos_sin(
                [&positions, &positions, &positions],
                QWEN_SECTIONS,
                Dtype::Float32,
            )
            .unwrap();
        assert_eq!(cm.shape(), &[1, 7, QWEN_HEAD_DIM]);
        assert_eq!(cm.shape(), c1.shape());
        assert_eq!(cm.as_slice::<f32>(), c1.as_slice::<f32>());
        assert_eq!(sm.as_slice::<f32>(), s1.as_slice::<f32>());
    }

    #[test]
    fn mrope_mixed_matches_reference_assembly() {
        // A genuinely 3-axis case: channel `c` must take its angle from axis `axis(c)` at frequency
        // inv_freq[c % half] (NeoX `cat(freqs, freqs)`), where axis(c) is the i%3 of the doubled
        // section chunk c falls in: chunks [0,16) [16,40) [40,64) [80? ...] → axes 0,1,2,0,1,2.
        let rope = Rope::standard(QWEN_HEAD_DIM, QWEN_THETA);
        let half = (QWEN_HEAD_DIM / 2) as usize; // 64
        let rows: [Vec<i32>; 3] = [vec![0, 5, 11], vec![2, 3, 9], vec![7, 1, 4]];
        let l = rows[0].len();
        let (cm, sm) = rope
            .mrope_cos_sin(
                [&rows[0], &rows[1], &rows[2]],
                QWEN_SECTIONS,
                Dtype::Float32,
            )
            .unwrap();
        assert_eq!(cm.shape(), &[1, l as i32, QWEN_HEAD_DIM]);

        // Map each channel to its source axis via the doubled-section chunk boundaries.
        let doubled = [
            QWEN_SECTIONS[0],
            QWEN_SECTIONS[1],
            QWEN_SECTIONS[2],
            QWEN_SECTIONS[0],
            QWEN_SECTIONS[1],
            QWEN_SECTIONS[2],
        ];
        let mut axis_of = vec![0usize; QWEN_HEAD_DIM as usize];
        let mut c = 0usize;
        for (chunk, &w) in doubled.iter().enumerate() {
            for _ in 0..w {
                axis_of[c] = chunk % 3;
                c += 1;
            }
        }

        let inv = rope.inv_freq();
        let cos_h = cm.as_slice::<f32>().to_vec();
        let sin_h = sm.as_slice::<f32>().to_vec();
        let hd = QWEN_HEAD_DIM as usize;
        #[allow(clippy::needless_range_loop)] // pos cross-indexes rows[axis][pos] and the flat table
        for pos in 0..l {
            for ch in 0..hd {
                let p = rows[axis_of[ch]][pos] as f32;
                let angle = p * inv[ch % half];
                let idx = pos * hd + ch;
                assert!((cos_h[idx] - angle.cos()).abs() < 1e-5, "cos[{pos},{ch}]");
                assert!((sin_h[idx] - angle.sin()).abs() < 1e-5, "sin[{pos},{ch}]");
            }
        }
    }

    #[test]
    fn mrope_rejects_bad_sections() {
        // Sections must sum to dim/2 (64 here); [16,24,16] sums to 56 → typed error, no panic.
        let rope = Rope::standard(QWEN_HEAD_DIM, QWEN_THETA);
        let positions = [0, 1, 2];
        let err = rope
            .mrope_cos_sin([&positions, &positions, &positions], [16, 24, 16], Dtype::Float32)
            .unwrap_err();
        assert!(matches!(err, Error::Msg(_)), "{err:?}");
    }

    #[test]
    fn mrope_rejects_interleaved() {
        // The doubled-section split is meaningless for the GPT-J interleaved layout.
        let rope = Rope::partial(QWEN_HEAD_DIM, QWEN_THETA, true);
        let positions = [0, 1, 2];
        let err = rope
            .mrope_cos_sin([&positions, &positions, &positions], QWEN_SECTIONS, Dtype::Float32)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }

    // Qwen3.6 interleaved M-RoPE: partial rotary_dim 64, theta 1e7, mrope_section [11,11,10].
    const Q36_ROT: i32 = 64;
    const Q36_THETA: f32 = 10_000_000.0;
    const Q36_SECTIONS: [usize; 3] = [11, 11, 10];

    #[test]
    fn mrope_interleaved_text_equals_1d() {
        // Acceptance gate: text-only (all three rows equal) must reproduce a plain 1D cos_sin_at over
        // that row, bit-for-bit — the invariant that keeps the Qwen3.6 text path unchanged.
        let rope = Rope::partial(Q36_ROT, Q36_THETA, false);
        let positions: Vec<i32> = (0..6).collect();
        let (c1, s1) = rope.cos_sin_at(&positions, Dtype::Float32).unwrap();
        let (cm, sm) = rope
            .mrope_interleaved_cos_sin([&positions, &positions, &positions], Q36_SECTIONS, Dtype::Float32)
            .unwrap();
        assert_eq!(cm.shape(), &[1, 6, Q36_ROT]);
        assert_eq!(cm.as_slice::<f32>(), c1.as_slice::<f32>());
        assert_eq!(sm.as_slice::<f32>(), s1.as_slice::<f32>());
    }

    #[test]
    fn mrope_interleaved_matches_reference() {
        // A genuinely-3D case vs the reference `apply_interleaved_mrope` (oracle from /tmp/gen_mrope.py).
        let j: serde_json::Value =
            serde_json::from_str(include_str!("../models/testdata/qwen35_mrope_oracle.json")).unwrap();
        let m = &j["mrope"];
        let row = |k: &str| -> Vec<i32> {
            m[k].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
        };
        let exp = |k: &str| -> Vec<f32> {
            m[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
        };
        let (t, h, w) = (row("t"), row("h"), row("w"));
        let rope = Rope::partial(Q36_ROT, Q36_THETA, false);
        let (cos, sin) = rope
            .mrope_interleaved_cos_sin([&t, &h, &w], Q36_SECTIONS, Dtype::Float32)
            .unwrap();
        let cmp = |got: &[f32], exp: &[f32]| {
            got.iter().zip(exp).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
        };
        assert!(cmp(cos.as_slice::<f32>(), &exp("cos")) < 1e-6, "interleaved mrope cos vs reference");
        assert!(cmp(sin.as_slice::<f32>(), &exp("sin")) < 1e-6, "interleaved mrope sin vs reference");
    }
}
