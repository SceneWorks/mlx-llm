//! Lloyd-Max scalar codebooks for **non-uniform** quantization.
//!
//! Where [`super::scalar_quant`] lays a *uniform* affine grid, a Lloyd-Max codebook places `k = 2^b`
//! centroids to minimize MSE for a given source distribution `f(x)` — the right tool when the data
//! is far from uniform (e.g. Gaussian-ish post-rotation KV activations). It iterates the two
//! Lloyd-Max optimality conditions to convergence (algorithm read from, but NOT executed against,
//! the VeloxQuant `math/lloyd_max.py` reference):
//!
//! 1. **Centroid → boundary**: Voronoi boundaries are midpoints of adjacent centroids.
//! 2. **Boundary → centroid**: each centroid moves to the conditional mean of its cell,
//!    `c_i = ∫_{b_{i-1}}^{b_i} x f(x) dx / ∫ f(x) dx`, estimated by trapezoid quadrature on a dense
//!    grid.
//!
//! The fitted distortion `C = Σ_i ∫ (x - c_i)² f(x) dx` is **non-increasing** every iteration — a
//! property the tests assert directly (the monotonic-MSE oracle).
//!
//! Codebook *fitting* runs once on the host (not a per-step hot loop). The per-element
//! quantize/dequantize that runs at inference time is the gather a kernel will accelerate:
//! `TODO(sc-8529/Phase2): replace nearest-centroid argmin + gather with a MetalKernel`
//! (speed only — this pure path defines correctness).

use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};

/// Result of a Lloyd-Max fit: sorted centroids and the MSE distortion at convergence.
#[derive(Debug, Clone)]
pub struct LloydMax {
    /// `k = 2^b` centroid values, ascending.
    pub centroids: Vec<f64>,
    /// `k + 1` Voronoi boundaries, ascending, with `±∞` at the ends.
    pub boundaries: Vec<f64>,
    /// Final MSE distortion `Σ_i ∫ (x - c_i)² f(x) dx` over the support.
    pub mse: f64,
}

/// Solve the 1-D Lloyd-Max scalar quantization problem for a sampled PDF.
///
/// * `pdf` — non-negative density evaluated at `support` sample points (need not be normalized).
/// * `support` — strictly increasing grid the `pdf` samples correspond to (`support.len() == pdf.len()`,
///   length ≥ 2). Defines the effective support and the quadrature grid.
/// * `n_levels` — number of centroids `k` (≥ 1; for a bit-width `b` pass `1 << b`).
/// * `n_iter` — max Lloyd-Max iterations.
/// * `tol` — convergence threshold on the max centroid shift.
///
/// Returns the fitted [`LloydMax`]. Centroids and boundaries are ascending.
pub fn lloyd_max(
    pdf: &[f64],
    support: &[f64],
    n_levels: usize,
    n_iter: usize,
    tol: f64,
) -> Result<LloydMax> {
    if n_levels < 1 {
        return Err(Error::Msg(format!(
            "lloyd_max: n_levels must be >= 1, got {n_levels}"
        )));
    }
    if support.len() < 2 || support.len() != pdf.len() {
        return Err(Error::Msg(format!(
            "lloyd_max: need len(support) == len(pdf) >= 2, got {} and {}",
            support.len(),
            pdf.len()
        )));
    }
    for w in support.windows(2) {
        if w[1] <= w[0] {
            return Err(Error::Msg(
                "lloyd_max: support must be strictly increasing".into(),
            ));
        }
    }

    let lo = support[0];
    let hi = support[support.len() - 1];

    // Initialise centroids uniformly over the support.
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| {
            if n_levels == 1 {
                0.5 * (lo + hi)
            } else {
                lo + (hi - lo) * (i as f64) / ((n_levels - 1) as f64)
            }
        })
        .collect();

    for _ in 0..n_iter {
        let boundaries = boundaries_of(&centroids);
        let new_centroids = update_centroids(&centroids, &boundaries, support, pdf);
        let shift = new_centroids
            .iter()
            .zip(&centroids)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        centroids = new_centroids;
        if shift < tol {
            break;
        }
    }

    let boundaries = boundaries_of(&centroids);
    let mse = mse_cost(&centroids, &boundaries, support, pdf);
    Ok(LloydMax {
        centroids,
        boundaries,
        mse,
    })
}

/// Voronoi boundaries = midpoints between adjacent centroids, with `±∞` at the edges.
fn boundaries_of(centroids: &[f64]) -> Vec<f64> {
    let mut b = Vec::with_capacity(centroids.len() + 1);
    b.push(f64::NEG_INFINITY);
    for w in centroids.windows(2) {
        b.push(0.5 * (w[0] + w[1]));
    }
    b.push(f64::INFINITY);
    b
}

/// Trapezoid integral of `y` over the sub-grid where `lo <= x <= hi`. Returns `(∫ y dx, points)`.
fn trapz_masked(x: &[f64], y: &[f64], lo: f64, hi: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let (x0, x1) = (x[i], x[i + 1]);
        // Include the interval if its midpoint falls in the cell — keeps adjacent cells from
        // double-counting the shared boundary.
        let mid = 0.5 * (x0 + x1);
        if mid >= lo && mid <= hi {
            acc += 0.5 * (y[i] + y[i + 1]) * (x1 - x0);
        }
    }
    acc
}

fn update_centroids(centroids: &[f64], boundaries: &[f64], x: &[f64], p: &[f64]) -> Vec<f64> {
    let xp: Vec<f64> = x.iter().zip(p).map(|(a, b)| a * b).collect();
    (0..centroids.len())
        .map(|i| {
            let b_lo = boundaries[i];
            let b_hi = boundaries[i + 1];
            let mass = trapz_masked(x, p, b_lo, b_hi);
            if mass < 1e-12 {
                centroids[i]
            } else {
                trapz_masked(x, &xp, b_lo, b_hi) / mass
            }
        })
        .collect()
}

fn mse_cost(centroids: &[f64], boundaries: &[f64], x: &[f64], p: &[f64]) -> f64 {
    let mut cost = 0.0;
    for (i, &c) in centroids.iter().enumerate() {
        let b_lo = boundaries[i];
        let b_hi = boundaries[i + 1];
        let sq: Vec<f64> = x
            .iter()
            .zip(p)
            .map(|(xi, pi)| (xi - c).powi(2) * pi)
            .collect();
        cost += trapz_masked(x, &sq, b_lo, b_hi);
    }
    cost
}

/// A fitted scalar codebook: `k = 2^b` sorted centroids, with nearest-centroid encode and
/// gather decode over MLX arrays.
#[derive(Debug, Clone)]
pub struct ScalarCodebook {
    centroids: Vec<f32>,
    bits: i32,
}

impl ScalarCodebook {
    /// Build from a centroid list whose length is a power of two (`2^b`, `1..=8` bits). Centroids
    /// are sorted ascending.
    pub fn new(centroids: &[f32]) -> Result<Self> {
        let k = centroids.len();
        if k == 0 || (k & (k - 1)) != 0 {
            return Err(Error::Msg(format!(
                "ScalarCodebook: centroid count must be a power of 2, got {k}"
            )));
        }
        let bits = k.trailing_zeros() as i32;
        if !(1..=8).contains(&bits) {
            return Err(Error::Unsupported(format!(
                "ScalarCodebook: bits must be in 1..=8, got {bits}"
            )));
        }
        let mut c = centroids.to_vec();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(Self { centroids: c, bits })
    }

    /// Fit a codebook from a sampled PDF via [`lloyd_max`] for the given bit-width `b`.
    pub fn fit(pdf: &[f64], support: &[f64], bits: i32, n_iter: usize, tol: f64) -> Result<Self> {
        if !(1..=8).contains(&bits) {
            return Err(Error::Unsupported(format!(
                "ScalarCodebook::fit: bits must be in 1..=8, got {bits}"
            )));
        }
        let lm = lloyd_max(pdf, support, 1usize << bits, n_iter, tol)?;
        let c: Vec<f32> = lm.centroids.iter().map(|&v| v as f32).collect();
        Self::new(&c)
    }

    /// Number of centroids (`2^bits`).
    pub fn k(&self) -> usize {
        self.centroids.len()
    }

    /// Bit-width of the codebook.
    pub fn bits(&self) -> i32 {
        self.bits
    }

    /// Centroid values, ascending, as `f32`.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Encode `x` to nearest-centroid `u8` indices (same shape as `x`).
    ///
    /// Pure-MLX-via-host nearest-centroid scan. Ties go to the lower index.
    pub fn quantize(&self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let flat: Vec<f32> = x.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let cents = &self.centroids;
        // TODO(sc-8529/Phase2): replace nearest-centroid argmin with a MetalKernel (one thread/elem).
        let idx: Vec<u8> = flat
            .iter()
            .map(|&v| {
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for (j, &c) in cents.iter().enumerate() {
                    let d = (v - c).abs();
                    if d < best_d {
                        best_d = d;
                        best = j;
                    }
                }
                best as u8
            })
            .collect();
        Ok(Array::from_slice(&idx, &shape))
    }

    /// Decode `u8` indices back to centroid values (same shape as `idx`).
    pub fn dequantize(&self, idx: &Array) -> Result<Array> {
        let shape = idx.shape().to_vec();
        let flat: Vec<u8> = idx.as_dtype(Dtype::Uint8)?.as_slice::<u8>().to_vec();
        let cents = &self.centroids;
        // TODO(sc-8529/Phase2): replace centroid gather with a MetalKernel (one thread/elem).
        let out: Vec<f32> = flat
            .iter()
            .map(|&i| cents[(i as usize).min(cents.len() - 1)])
            .collect();
        Ok(Array::from_slice(&out, &shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform density on [0, 1]: the Lloyd-Max optimum for k=2 is centroids at the cell means
    /// 1/4 and 3/4 (HAND-COMPUTED from the optimality conditions). Boundary at 1/2.
    #[test]
    fn lloyd_uniform_k2_known_centroids() {
        let n = 4001;
        let support: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let pdf = vec![1.0f64; n];
        let lm = lloyd_max(&pdf, &support, 2, 200, 1e-9).unwrap();
        assert!((lm.centroids[0] - 0.25).abs() < 2e-3, "{:?}", lm.centroids);
        assert!((lm.centroids[1] - 0.75).abs() < 2e-3, "{:?}", lm.centroids);
        // Analytic distortion for uniform k-level quant on unit interval = 1/(12 k^2) = 1/48.
        assert!((lm.mse - 1.0 / 48.0).abs() < 1e-3, "mse = {}", lm.mse);
    }

    /// Uniform density, k=4: centroids at 1/8, 3/8, 5/8, 7/8; distortion 1/(12·16) = 1/192.
    #[test]
    fn lloyd_uniform_k4_known_centroids() {
        let n = 8001;
        let support: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let pdf = vec![1.0f64; n];
        let lm = lloyd_max(&pdf, &support, 4, 300, 1e-10).unwrap();
        let expected = [0.125, 0.375, 0.625, 0.875];
        for (c, e) in lm.centroids.iter().zip(&expected) {
            assert!((c - e).abs() < 3e-3, "got {:?}", lm.centroids);
        }
        assert!((lm.mse - 1.0 / 192.0).abs() < 5e-4, "mse = {}", lm.mse);
    }

    /// MSE must be MONOTONICALLY NON-INCREASING across iterations (the Lloyd-Max descent oracle).
    /// We run the fit at increasing iteration caps and check distortion never goes up.
    #[test]
    fn lloyd_mse_monotonic_non_increasing() {
        // A skewed (triangular) density so the iterations actually move.
        let n = 2001;
        let support: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
            .collect();
        // Triangular peaked at 0: f(x) = 1 - |x|.
        let pdf: Vec<f64> = support.iter().map(|&x| (1.0 - x.abs()).max(0.0)).collect();
        let mut prev = f64::INFINITY;
        for iters in 1..=12 {
            let lm = lloyd_max(&pdf, &support, 4, iters, 0.0).unwrap();
            assert!(
                lm.mse <= prev + 1e-9,
                "MSE increased at iters={iters}: {} > {}",
                lm.mse,
                prev
            );
            prev = lm.mse;
        }
    }

    /// More levels => lower (or equal) distortion for the same distribution.
    #[test]
    fn lloyd_more_levels_lower_distortion() {
        let n = 2001;
        let support: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64)
            .collect();
        let pdf: Vec<f64> = support.iter().map(|&x| (1.0 - x.abs()).max(0.0)).collect();
        let m2 = lloyd_max(&pdf, &support, 2, 300, 1e-10).unwrap().mse;
        let m4 = lloyd_max(&pdf, &support, 4, 300, 1e-10).unwrap().mse;
        let m8 = lloyd_max(&pdf, &support, 8, 300, 1e-10).unwrap().mse;
        assert!(m4 <= m2 + 1e-9, "{m4} !<= {m2}");
        assert!(m8 <= m4 + 1e-9, "{m8} !<= {m4}");
    }

    /// Codebook encode/decode round-trip reconstructs the nearest centroid exactly.
    #[test]
    fn codebook_quantize_dequantize_nearest() {
        let cb = ScalarCodebook::new(&[-1.0, 0.0, 1.0, 2.0]).unwrap(); // k=4, b=2
        assert_eq!(cb.k(), 4);
        assert_eq!(cb.bits(), 2);
        // values -> nearest centroid index: -0.9->0, 0.1->1, 0.6->? (|0.6-0|=0.6,|0.6-1|=0.4)->2
        let x = Array::from_slice(&[-0.9f32, 0.1, 0.6, 1.9], &[4]);
        let idx = cb.quantize(&x).unwrap();
        assert_eq!(
            idx.as_dtype(Dtype::Uint8)
                .unwrap()
                .as_slice::<u8>()
                .to_vec(),
            vec![0u8, 1, 2, 3]
        );
        let recon = cb.dequantize(&idx).unwrap();
        assert_eq!(
            recon.as_slice::<f32>().to_vec(),
            vec![-1.0f32, 0.0, 1.0, 2.0]
        );
    }

    /// fit() produces a usable, power-of-two codebook with sorted centroids.
    #[test]
    fn codebook_fit_from_pdf() {
        let n = 1001;
        let support: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let pdf = vec![1.0f64; n];
        let cb = ScalarCodebook::fit(&pdf, &support, 2, 200, 1e-9).unwrap();
        assert_eq!(cb.k(), 4);
        // sorted ascending
        let c = cb.centroids();
        assert!(c.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn codebook_rejects_non_power_of_two() {
        assert!(ScalarCodebook::new(&[0.0, 1.0, 2.0]).is_err());
        assert!(lloyd_max(&[1.0, 1.0], &[0.0], 2, 10, 1e-6).is_err()); // mismatched lengths
    }
}
