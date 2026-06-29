//! RVQ ⇄ VeloxQuant-MLX oracle parity (story sc-8532, epic sc-8528).
//!
//! `tests/fixtures/rvq_oracle.json` holds reference vectors captured by RUNNING the upstream
//! VeloxQuant-MLX RVQ reference (`quantizers/turboquant_rvq.py` + the cache-side unit-norm semantics
//! from `cache/turboquant_rvq_cache.py`) in a Python venv on fixed, seeded inputs — see the generator
//! `gen_oracle.py` archived alongside the clone in scratch (NOT committed). Each `cache_*` case records,
//! for head_dim `d` and bits `b`:
//!   - `x`        : the raw input keys `[batch, d]` (fed to the cache, which unit-normalizes),
//!   - `diag`     : the exact ±1 randomized-Hadamard diagonal upstream used,
//!   - `centroids1` / `centroids2` : the upstream Lloyd-Max stage-1 (Gaussian N(0,1/d)) and stage-2
//!     (Laplacian residual) centroid tables,
//!   - `xhat`     : upstream's reconstructed keys.
//!
//! This crate's [`RvqQuantizer`] is then built with the upstream diagonal injected
//! ([`RvqQuantizer::with_diagonal`]) and run on the same inputs; we assert:
//!   1. Our independently-fitted Lloyd-Max centroid tables match upstream's within tolerance
//!      (proves the codebook math is faithful), and
//!   2. Our encode→decode reconstruction matches upstream's `xhat` within tolerance
//!      (the end-to-end numeric-parity acceptance criterion).
//!
//! TOLERANCE (documented): centroids `< 5e-3` abs (Lloyd-Max quadrature + fp64→fp32). Reconstruction
//! `xhat` mean-abs-error `< 6e-2` of the per-vector scale and cosine within `2e-2` of upstream's own
//! cosine — the residual gap is fp16-vs-fp32 codebook-apply + a possible boundary-tie flip in the
//! fp16 `searchsorted` quantizer (upstream computes the index comparison in fp16; we compute nearest
//! centroid in fp32, which can disagree only for a coordinate sitting exactly on a Voronoi boundary).

use std::path::Path;

use mlx_rs::{Array, Dtype};
use serde_json::Value;

use mlx_llm::primitives::quant::RvqQuantizer;
use mlx_llm::primitives::Quantizer;

fn load_oracle() -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rvq_oracle.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("parse rvq_oracle.json")
}

fn f32_vec(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

fn cosine_rows(a: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let n = a.len() / d;
    (0..n)
        .map(|r| {
            let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for j in 0..d {
                let (x, y) = (a[r * d + j], b[r * d + j]);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            dot / (na.sqrt() * nb.sqrt() + 1e-9)
        })
        .collect()
}

/// Our independently-fitted stage-1/stage-2 Lloyd-Max codebooks reproduce upstream's centroids
/// within tolerance (proves the codebook math, not just the end-to-end reconstruction).
#[test]
fn rvq_oracle_centroid_tables_match() {
    const CENTROID_TOL: f32 = 5e-3;
    for case in load_oracle() {
        if !case["unit_norm"].as_bool().unwrap() {
            continue; // centroid tables are config-only; check once per (d,b) via the cache cases.
        }
        let d = case["d"].as_i64().unwrap() as i32;
        let b = case["b"].as_i64().unwrap() as i32;
        let diag = Array::from_slice(&f32_vec(&case["diag"]), &[d]);
        let q = RvqQuantizer::with_diagonal(d, b, diag).unwrap();

        let oc1 = f32_vec(&case["centroids1"]);
        let oc2 = f32_vec(&case["centroids2"]);
        let mc1 = q.stage1_centroids();
        let mc2 = q.stage2_centroids();
        assert_eq!(mc1.len(), oc1.len());
        for (m, o) in mc1.iter().zip(&oc1) {
            assert!(
                (m - o).abs() < CENTROID_TOL,
                "d={d} b={b} stage-1 centroid {m} vs upstream {o}"
            );
        }
        for (m, o) in mc2.iter().zip(&oc2) {
            assert!(
                (m - o).abs() < CENTROID_TOL,
                "d={d} b={b} stage-2 centroid {m} vs upstream {o}"
            );
        }
    }
}

/// End-to-end numeric parity: our RVQ encode→decode (cache-side unit-norm path) reconstructs the
/// keys to within tolerance of the upstream VeloxQuant RVQ reference on the same seeded inputs and
/// the same Hadamard diagonal. This is the acceptance criterion "numeric parity vs the VeloxQuant RVQ
/// oracle within tolerance".
#[test]
fn rvq_oracle_reconstruction_parity() {
    // Documented tolerances (see module doc).
    const MAE_TOL: f32 = 6e-2;
    const COS_TOL: f32 = 2e-2;
    let mut worst_mae = 0.0f32;
    let mut worst_cos_gap = 0.0f32;

    for case in load_oracle() {
        if !case["unit_norm"].as_bool().unwrap() {
            continue; // parity is measured on the real cache (unit-normalized) path.
        }
        let d = case["d"].as_i64().unwrap() as i32;
        let b = case["b"].as_i64().unwrap() as i32;
        let batch = case["batch"].as_i64().unwrap() as i32;

        let x = f32_vec(&case["x"]);
        let xhat_oracle = f32_vec(&case["xhat"]);
        let diag = Array::from_slice(&f32_vec(&case["diag"]), &[d]);
        let q = RvqQuantizer::with_diagonal(d, b, diag).unwrap();

        // Feed `x` as KV keys `[batch, heads=1, seq=1, d]`; the quantizer unit-normalizes per vector
        // exactly as the upstream cache does. Values are irrelevant here (dense pass-through).
        let keys = Array::from_slice(&x, &[batch, 1, 1, d]);
        let vals = keys.clone();
        let block = q.encode(&keys, &vals).unwrap();
        let (kd, _) = q.decode(&block).unwrap();
        let xhat_ours = kd
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();

        // Per-vector scale to normalize the MAE (keys have ~unit-Gaussian magnitude × sqrt(d)).
        let scale = {
            let n = (batch as usize) * d as usize;
            (x.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt().max(1e-6)
        };
        let mae = mean_abs(&xhat_ours, &xhat_oracle) / scale;
        worst_mae = worst_mae.max(mae);

        let cos_ours = cosine_rows(&x, &xhat_ours, d as usize);
        let cos_oracle = cosine_rows(&x, &xhat_oracle, d as usize);
        for (co, cor) in cos_ours.iter().zip(&cos_oracle) {
            worst_cos_gap = worst_cos_gap.max((co - cor).abs());
        }

        assert!(
            mae < MAE_TOL,
            "d={d} b={b}: reconstruction MAE/scale {mae:.4} exceeds tol {MAE_TOL}"
        );
        for (co, cor) in cos_ours.iter().zip(&cos_oracle) {
            assert!(
                (co - cor).abs() < COS_TOL,
                "d={d} b={b}: cosine {co:.4} vs upstream {cor:.4} (gap exceeds {COS_TOL})"
            );
        }
    }

    eprintln!(
        "RVQ oracle parity: worst MAE/scale = {worst_mae:.4} (tol {MAE_TOL}), \
         worst cosine gap = {worst_cos_gap:.4} (tol {COS_TOL})"
    );
}
