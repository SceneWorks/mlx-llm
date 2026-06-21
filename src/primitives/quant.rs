//! Group-wise affine quantization (Q4 / Q8) for linear projections.
//!
//! This is greenfield for the engine — the mlx-gen LLM stacks (prompt-refine, JoyCaption) reject
//! quantization at load. We build on MLX's native group-wise affine quantization
//! (`ops::quantize` / `ops::quantized_matmul`), which packs an `[out, in]` weight into a quantized
//! tensor plus per-group `scales`/`biases`. This backs quantize-on-load (story 7163) and the GGUF
//! path (7165).

use mlx_rs::ops::{add, dequantize, quantize, quantized_matmul};
use mlx_rs::Array;

use crate::error::Result;

/// A linear projection whose weight is stored group-wise quantized.
///
/// Forward is `quantized_matmul(x, weight, scales, biases, transpose = true, ...)`, which computes
/// `x @ weight.t()` against the dequantized weight — the quantized analogue of
/// [`super::nn::linear`]. `transpose = true` matches the HF `[out, in]` weight layout.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Packed quantized weight.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group biases (zero-points).
    pub biases: Array,
    /// Elements per quantization group (e.g. 64).
    pub group_size: i32,
    /// Bits per weight (4 or 8).
    pub bits: i32,
    /// Optional additive bias applied after the matmul.
    pub bias: Option<Array>,
}

impl QuantizedLinear {
    /// Quantize a dense `[out, in]` weight into a `QuantizedLinear`. `group_size` must divide the
    /// input dimension; `bits` is typically 4 or 8.
    pub fn quantize(
        weight: &Array,
        group_size: i32,
        bits: i32,
        bias: Option<Array>,
    ) -> Result<Self> {
        let (w, scales, biases) = quantize(weight, group_size, bits)?;
        Ok(Self {
            weight: w,
            scales,
            biases,
            group_size,
            bits,
            bias,
        })
    }

    /// Forward pass: `x @ dequant(weight).t() (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true, // transpose: weight is [out, in]
            self.group_size,
            self.bits,
        )?;
        match &self.bias {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }

    /// Reconstruct the dense weight (mostly for parity tests / inspection).
    pub fn dequantize_weight(&self) -> Result<Array> {
        Ok(dequantize(
            &self.weight,
            &self.scales,
            Some(&self.biases),
            self.group_size,
            self.bits,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::nn::linear;

    /// Quantize→dequantize should round-trip within the affine grid's tolerance.
    #[test]
    fn quantize_dequantize_roundtrip_q8() {
        // [out=2, in=64] so a single group of 64 covers the input dim.
        let n = 2 * 64;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) - 0.5).collect();
        let w = Array::from_slice(&data, &[2, 64]);
        let q = QuantizedLinear::quantize(&w, 64, 8, None).unwrap();
        let recon = q.dequantize_weight().unwrap();
        let orig = w.as_slice::<f32>().to_vec();
        let back = recon.as_slice::<f32>().to_vec();
        let max_err = orig
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // 8-bit affine over a ~1.0 range: error well under 1%.
        assert!(max_err < 0.01, "max_err = {max_err}");
    }

    /// Quantized matmul should approximate the dense linear it replaces.
    #[test]
    fn quantized_matmul_approximates_linear_q8() {
        let n = 4 * 64;
        let wdata: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5).collect();
        let w = Array::from_slice(&wdata, &[4, 64]); // [out=4, in=64]
        let x = Array::from_slice(
            &(0..64).map(|i| (i as f32 / 64.0) - 0.5).collect::<Vec<_>>(),
            &[1, 64],
        );
        let dense = linear(&x, &w, None).unwrap().as_slice::<f32>().to_vec();
        let q = QuantizedLinear::quantize(&w, 64, 8, None).unwrap();
        let quant = q.forward(&x).unwrap().as_slice::<f32>().to_vec();
        for (a, b) in dense.iter().zip(&quant) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }
}
