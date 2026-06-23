//! Neural-net leaves the decoders compose: linear projection, RMS/layer norm, activations, and the
//! embedding gather (plus the host token-id → `Array` lift).
//!
//! Weights follow the HF convention — stored `[out, in]` — so [`linear`] is `matmul(x, w.t())`
//! (no bias on Llama/Qwen; bias is optional for e.g. SigLIP). These wrap MLX ops so the decoders
//! read cleanly and so a single place documents the layout contract.

use mlx_rs::ops::{add, conv2d as conv2d_op, matmul};
use mlx_rs::{Array, Dtype};

use crate::error::Result;

/// 2-D convolution over NHWC `x` with an mlx `[out, kH, kW, in]` weight (+ optional bias), square
/// `stride`/`padding`, no dilation, groups = 1 — the patch-embedding conv the SigLIP vision tower
/// uses (story 7157). HF stores the conv weight `[out, in, kH, kW]`; transpose to `[out, kH, kW, in]`
/// before calling.
pub fn conv2d(x: &Array, weight: &Array, bias: Option<&Array>, stride: i32, padding: i32) -> Result<Array> {
    let y = conv2d_op(x, weight, (stride, stride), (padding, padding), (1, 1), 1)?;
    match bias {
        Some(b) => Ok(add(&y, b)?),
        None => Ok(y),
    }
}

/// `x @ weight.t() (+ bias)`. `weight` is `[out, in]` (HF layout); `x` is `[..., in]`.
pub fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
    let y = matmul(x, weight.t())?;
    match bias {
        Some(b) => Ok(add(&y, b)?),
        None => Ok(y),
    }
}

/// RMSNorm via MLX's fused kernel: `x / rms(x) * weight`.
pub fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    Ok(mlx_rs::fast::rms_norm(x, weight, eps)?)
}

/// LayerNorm via MLX's fused kernel (optional affine weight/bias).
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
) -> Result<Array> {
    Ok(mlx_rs::fast::layer_norm(x, weight, bias, eps)?)
}

/// SiLU / swish activation.
pub fn silu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::silu(x)?)
}

/// Exact (erf) GELU — the LLaVA projector variant.
pub fn gelu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::gelu(x)?)
}

/// Tanh-approximate GELU — the SigLIP MLP variant. (Do not unify with [`gelu`]: the two are
/// numerically distinct and both appear in the JoyCaption VLM.)
pub fn gelu_tanh(x: &Array) -> Result<Array> {
    Ok(mlx_rs::nn::gelu_approximate(x)?)
}

/// Logit soft-cap `cap · tanh(x / cap)` (Gemma-2 caps attention scores and final logits). A no-op as
/// `cap → ∞`; it squashes extremes toward `±cap` while staying ~linear near 0. Dtype-preserving (the
/// scalars are cast to `x`'s dtype), so the caller controls precision — pass f32 where it matters.
pub fn soft_cap(x: &Array, cap: f32) -> Result<Array> {
    use mlx_rs::ops::{multiply, tanh};
    let inv = Array::from_f32(1.0 / cap).as_dtype(x.dtype())?;
    let c = Array::from_f32(cap).as_dtype(x.dtype())?;
    let capped = tanh(&multiply(x, &inv)?)?;
    Ok(multiply(&capped, &c)?)
}

/// SwiGLU MLP block: `down( silu(gate(x)) * up(x) )`. All weights `[out, in]`, bias-free.
pub fn swiglu(x: &Array, gate_w: &Array, up_w: &Array, down_w: &Array) -> Result<Array> {
    let gate = silu(&linear(x, gate_w, None)?)?;
    let up = linear(x, up_w, None)?;
    let gated = mlx_rs::ops::multiply(&gate, &up)?;
    linear(&gated, down_w, None)
}

/// Embedding gather: rows of `weight` (`[vocab, hidden]`) selected by `ids` (`[batch, seq]` i32),
/// returning `[batch, seq, hidden]`. The result keeps `weight`'s dtype.
pub fn embed(weight: &Array, ids: &Array) -> Result<Array> {
    let sh = ids.shape();
    let (b, s) = (sh[0], sh[1]);
    let hidden = weight.shape()[1];
    let flat = ids.reshape(&[-1])?;
    let gathered = weight.take_axis(&flat, 0)?;
    Ok(gathered.reshape(&[b, s, hidden])?)
}

/// Lift a host token-id slice into a batch-1 `[1, len]` i32 `Array`.
pub fn input_ids(ids: &[i32]) -> Array {
    Array::from_slice(ids, &[1, ids.len() as i32])
}

/// Lift equal-length host token-id rows into a `[batch, len]` i32 `Array`.
pub fn input_ids_batch(rows: &[&[i32]]) -> Result<Array> {
    let batch = rows.len();
    if batch == 0 {
        return Err(crate::error::Error::Msg("input_ids_batch: no rows".into()));
    }
    let len = rows[0].len();
    let mut flat = Vec::with_capacity(batch * len);
    for (i, r) in rows.iter().enumerate() {
        if r.len() != len {
            return Err(crate::error::Error::Msg(format!(
                "input_ids_batch: row {i} has length {} != {len}",
                r.len()
            )));
        }
        flat.extend_from_slice(r);
    }
    Ok(Array::from_slice(&flat, &[batch as i32, len as i32]))
}

/// Convert a logits/last-position `Array` to a host `f32` vector (e.g. for host-side sampling).
pub fn to_f32_host(x: &Array) -> Result<Vec<f32>> {
    Ok(x.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_manual_matmul() {
        // x: [1,2], w: [3,2] (out=3, in=2). y = x @ w.t() -> [1,3].
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let w = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
        let h = y.as_slice::<f32>().to_vec();
        assert_eq!(h, vec![1.0, 2.0, 3.0]); // [x0, x1, x0+x1]
    }

    #[test]
    fn linear_adds_bias() {
        let x = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);
        let w = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let b = Array::from_slice(&[10.0f32, 20.0], &[2]);
        let y = linear(&x, &w, Some(&b)).unwrap();
        assert_eq!(y.as_slice::<f32>().to_vec(), vec![11.0, 21.0]);
    }

    #[test]
    fn embed_gathers_rows() {
        // vocab 3, hidden 2: row0=[0,1] row1=[2,3] row2=[4,5]
        let w = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2]);
        let ids = input_ids(&[2, 0]);
        let e = embed(&w, &ids).unwrap();
        assert_eq!(e.shape(), &[1, 2, 2]);
        assert_eq!(e.as_slice::<f32>().to_vec(), vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn input_ids_batch_shapes() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let arr = input_ids_batch(&[&a, &b]).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
    }

    #[test]
    fn input_ids_batch_rejects_ragged() {
        let a = [1, 2, 3];
        let b = [4, 5];
        assert!(input_ids_batch(&[&a, &b]).is_err());
    }

    #[test]
    fn swiglu_runs() {
        let x = Array::from_slice(&[0.5f32, -0.5], &[1, 2]);
        let gate = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let up = gate.clone();
        let down = Array::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]);
        let y = swiglu(&x, &gate, &up, &down).unwrap();
        assert_eq!(y.shape(), &[1, 2]);
    }
}
