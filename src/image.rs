//! Host-side image preprocessing for the vision path (story 7157).
//!
//! [`resize_bicubic_u8`] bit-matches PIL's `ImagingResample` 8-bit BICUBIC path (float coefficients
//! quantized to `PRECISION_BITS` fixed-point, an integer multiply-accumulate seeded with the
//! rounding bias, then `clip8`), so a resized conditioning image is pixel-identical to the Python
//! reference — reproducing PIL's *fixed-point* arithmetic, not merely "a bicubic", is what avoids
//! the ±1–2 ULP divergence at gradient cliffs. [`SiglipImageProcessor`] resizes to the model's
//! square input and applies the mean/std normalization, producing the NHWC tensor the
//! [`crate::models::SiglipVisionTower`] consumes. Pure host code (no gen-ai deps); the only tensor
//! is the final `Array`.

use mlx_rs::Array;

use crate::error::{Error, Result};

/// PIL `bicubic_filter` (Keys cubic, a = -0.5), support 2.0.
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-pixel 1-D resampling coefficients, matching PIL `precompute_coeffs`: antialias by
/// scaling the filter support when downscaling, clamp the window to the input bounds, renormalize.
fn precompute_coeffs(in_size: usize, out_size: usize, support_radius: f64) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support_radius * filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as i64).min(in_size as i64) as usize;
        let mut weights = Vec::with_capacity(xmax - xmin);
        let mut total = 0.0;
        for x in xmin..xmax {
            let w = cubic((x as f64 - center + 0.5) / filterscale);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }
        out.push((xmin, weights));
    }
    out
}

/// PIL `PRECISION_BITS` for the 8-bit resample path (`32 - 8 - 2`).
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// PIL `clip8` for the resample accumulator (which carries the `1<<(PRECISION_BITS-1)` rounding
/// bias): shift down by `PRECISION_BITS` and clamp to `[0,255]`.
#[inline]
fn clip8(acc: i64) -> f32 {
    if acc <= 0 {
        return 0.0;
    }
    let v = acc >> PRECISION_BITS;
    if v >= 255 {
        255.0
    } else {
        v as f32
    }
}

/// Quantize PIL float coefficients to fixed-point ints (`normalize_coeffs_8bpc`: round half away
/// from zero at `1<<PRECISION_BITS`).
fn quantize_coeffs(coeffs: &[(usize, Vec<f64>)]) -> Vec<(usize, Vec<i64>)> {
    let scale = (1i64 << PRECISION_BITS) as f64;
    coeffs
        .iter()
        .map(|(xmin, w)| {
            let ik = w
                .iter()
                .map(|&c| if c < 0.0 { (c * scale - 0.5) as i64 } else { (c * scale + 0.5) as i64 })
                .collect();
            (*xmin, ik)
        })
        .collect()
}

/// PIL `Image.BICUBIC` resize of a uint8 RGB HWC image (two separable passes, fixed-point). Returns
/// f32 HWC, integer-valued in `[0, 255]`. Assumes 3 channels.
pub fn resize_bicubic_u8(src: &[u8], in_h: usize, in_w: usize, out_h: usize, out_w: usize) -> Result<Vec<f32>> {
    const C: usize = 3;
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(Error::Msg(format!(
            "resize: zero dimension {in_w}x{in_h} -> {out_w}x{out_h}"
        )));
    }
    if src.len() < in_h * in_w * C {
        return Err(Error::Msg(format!(
            "resize: pixel buffer too small ({} bytes for {in_w}x{in_h} RGB, need {})",
            src.len(),
            in_h * in_w * C
        )));
    }
    let bias = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass: (in_h, in_w) -> (in_h, out_w).
    let hcoeffs = quantize_coeffs(&precompute_coeffs(in_w, out_w, 2.0));
    let mut horiz = vec![0f32; in_h * out_w * C];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hcoeffs.iter().enumerate() {
            for ch in 0..C {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * C + ch] as i64 * wk;
                }
                horiz[(y * out_w + xx) * C + ch] = clip8(acc);
            }
        }
    }

    // Vertical pass: (in_h, out_w) -> (out_h, out_w).
    let vcoeffs = quantize_coeffs(&precompute_coeffs(in_h, out_h, 2.0));
    let mut out = vec![0f32; out_h * out_w * C];
    for (yy, (ymin, w)) in vcoeffs.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..C {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * C + ch] as i64 * wk;
                }
                out[(yy * out_w + x) * C + ch] = clip8(acc);
            }
        }
    }
    Ok(out)
}

/// RGB uint8 → SigLIP input tensor: resize (BICUBIC) to a square `size`, rescale to `[0,1]`, and
/// normalize per channel. Output is NHWC `[1, size, size, 3]` f32.
#[derive(Clone, Debug)]
pub struct SiglipImageProcessor {
    /// Target square edge (384 for so400m-patch14-384).
    pub size: usize,
    /// Per-channel mean (SigLIP: 0.5).
    pub mean: [f32; 3],
    /// Per-channel std (SigLIP: 0.5).
    pub std: [f32; 3],
}

impl Default for SiglipImageProcessor {
    fn default() -> Self {
        Self {
            size: 384,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

impl SiglipImageProcessor {
    /// Preprocess raw interleaved RGB8 `pixels` (`width*height*3` bytes) into the SigLIP NHWC tensor.
    pub fn preprocess(&self, pixels: &[u8], width: usize, height: usize) -> Result<Array> {
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(Error::Msg(format!(
                "siglip preprocess: expected {expected} RGB bytes for {width}x{height}, got {}",
                pixels.len()
            )));
        }
        let resized: Vec<f32> = if width == self.size && height == self.size {
            pixels.iter().map(|&p| p as f32).collect()
        } else {
            resize_bicubic_u8(pixels, height, width, self.size, self.size)?
        };
        let mut normalized = Vec::with_capacity(self.size * self.size * 3);
        for px in resized.chunks_exact(3) {
            for (ch, &v) in px.iter().enumerate() {
                normalized.push((v / 255.0 - self.mean[ch]) / self.std[ch]);
            }
        }
        Ok(Array::from_slice(
            &normalized,
            &[1, self.size as i32, self.size as i32, 3],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_f32(a: &Array) -> Vec<f32> {
        a.as_slice::<f32>().to_vec()
    }

    #[test]
    fn preprocess_normalizes_to_minus_one_one() {
        // 384x384 so no resize; mean/std 0.5 maps 0->-1, 255->1, 128->~0.004.
        let pixels = [0u8, 128, 255].repeat(384 * 384);
        let out = SiglipImageProcessor::default().preprocess(&pixels, 384, 384).unwrap();
        assert_eq!(out.shape(), &[1, 384, 384, 3]);
        let v = host_f32(&out);
        assert_eq!(v[0], -1.0);
        assert!((v[1] - 0.003_921_628).abs() < 1e-6);
        assert_eq!(v[2], 1.0);
    }

    #[test]
    fn preprocess_rejects_bad_buffer() {
        assert!(SiglipImageProcessor::default().preprocess(&[0u8; 3], 2, 2).is_err());
    }

    #[test]
    fn resize_uniform_image_is_uniform() {
        // A solid color resized stays that exact color everywhere (renormalized weights sum to 1).
        let src = [200u8, 100, 50].repeat(64 * 48); // 48x64 (wxh) wait: width=64,height=48
        let out = resize_bicubic_u8(&src, 48, 64, 384, 384).unwrap();
        assert_eq!(out.len(), 384 * 384 * 3);
        for px in out.chunks_exact(3) {
            assert_eq!(px, &[200.0, 100.0, 50.0]);
        }
    }

    #[test]
    fn resize_rejects_zero_and_undersized() {
        assert!(resize_bicubic_u8(&[0u8; 12], 0, 2, 4, 4).is_err());
        assert!(resize_bicubic_u8(&[0u8; 3], 2, 2, 4, 4).is_err());
    }
}
