//! Gated DeltaNet linear attention — the recurrence (story sc-7627).
//!
//! Qwen3.6 (`model_type` `qwen3_5`, the Qwen3-Next architecture) interleaves 3 **Gated DeltaNet**
//! linear-attention layers with 1 gated full-attention layer. Unlike softmax attention over a
//! growing KV cache, a linear layer carries a **fixed-size recurrent state** `S ∈ [Dv, Dk]` per head
//! and updates it with the gated delta rule each step — so it costs O(1) memory in sequence length.
//!
//! This module ports the **ops path** of `mlx_lm.models.gated_delta` (`gated_delta_ops` /
//! `_gated_delta_step_ops`) — the sequential reference the MLX engine itself falls back to off-GPU —
//! into `mlx-rs`. The recurrence is validated bit-for-bit against that reference via an embedded
//! numeric fixture (see the tests). The per-step update, for head state `S` (decayed by the gate `g`,
//! `β` the delta strength):
//!
//! ```text
//!   S      = S · g                          # forget (per-head scalar decay)
//!   kv_mem = (S · kᵀ) summed over Dk         # what the current key already recalls  → [Dv]
//!   Δ      = (v − kv_mem) · β                # the correction to write               → [Dv]
//!   S      = S + Δ ⊗ k                        # delta-rule outer-product write         → [Dv, Dk]
//!   y      = (S · qᵀ) summed over Dk          # read out with the query                → [Dv]
//! ```
//!
//! The gate and delta strength come from the layer's learned projections via [`compute_g`] (`g =
//! exp(−exp(A_log) · softplus(a + dt_bias))`) and `β = sigmoid(b)`; the surrounding short-conv,
//! normalisation, and in/out projections (the full layer) build on this in the layer story (sc-7628).
//! GQA is handled by repeating each of the `Hk` key/query heads to the `Hv` value heads.

use mlx_rs::nn::{silu, softplus};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, exp, multiply, subtract, sum_axis};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::error::Result;

/// The per-step gate `g = exp(−exp(A_log) · softplus(a + dt_bias))` (a faithful port of
/// `mlx_lm.models.gated_delta.compute_g`). `a` is `[B, T, Hv]` (the gating projection), `A_log` and
/// `dt_bias` are per-value-head `[Hv]`. The inner exponentials are evaluated in f32 (matching the
/// reference's `.astype(float32)`) and the result is cast back to `a`'s dtype.
pub fn compute_g(a: &Array, a_log: &Array, dt_bias: &Array) -> Result<Array> {
    let a32 = a.as_dtype(Dtype::Float32)?;
    let dt32 = dt_bias.as_dtype(Dtype::Float32)?;
    let al32 = a_log.as_dtype(Dtype::Float32)?;
    let sp = softplus(&add(&a32, &dt32)?)?; // softplus(a + dt_bias)            [B,T,Hv]
    let coeff = multiply(&exp(&al32)?, &sp)?; // exp(A_log) · softplus(...)      [B,T,Hv]
    let g = exp(&coeff.negative()?)?; // exp(−coeff)
    Ok(g.as_dtype(a.dtype())?)
}

/// Run the gated delta recurrence over a `[B, T, ·]` chunk, a faithful port of the
/// `mlx_lm.models.gated_delta` ops path.
///
/// Shapes: `q`, `k` are `[B, T, Hk, Dk]`; `v` is `[B, T, Hv, Dv]`; `g` (the per-step gate from
/// [`compute_g`]) and `beta` are `[B, T, Hv]`; `state` (the carried recurrent state, or `None` to
/// start from zeros) is `[B, Hv, Dv, Dk]`. Returns the per-step output `y` `[B, T, Hv, Dv]` and the
/// final `state` `[B, Hv, Dv, Dk]` — feed `state` back in for the next chunk / decode step (T = 1).
///
/// GQA: when `Hv > Hk` each key/query head is repeated `Hv / Hk` times so it pairs with the value
/// heads (`Hv` must be a multiple of `Hk`). The math runs in the inputs' dtype, matching the ops
/// reference (the layer story can lift accumulation to f32 to match the GPU kernel where it matters).
pub fn gated_delta_recurrence(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
) -> Result<(Array, Array)> {
    let qs = q.shape();
    let (b, t, hk, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let vs = v.shape();
    let (hv, dv) = (vs[2], vs[3]);

    // GQA: repeat each of the Hk key/query heads to the Hv value heads (contiguous, matching
    // `mx.repeat(q, Hv/Hk, axis=-2)`): expand → broadcast → reshape, the engine's GQA idiom.
    let (q, k) = if hv != hk {
        let r = hv / hk;
        (repeat_heads(q, r)?, repeat_heads(k, r)?)
    } else {
        (q.clone(), k.clone())
    };

    let mut state = match state {
        Some(s) => s.clone(),
        None => zeros_state(b, hv, dv, dk, q.dtype())?,
    };

    // The recurrence is sequential, so the whole `t`-step graph is built before the caller's single
    // `eval` at the end of the forward. Every step eagerly allocates index arrays (one Metal buffer
    // each, via `slice_time`'s gather) that stay live until that eval — so on a long sequence the live
    // buffer count grows without bound and trips the Metal allocator's resource limit, surfacing as a
    // spurious "expected a non-empty mlx_array" from the next gather (a high-res image expands to
    // thousands of vision tokens × the decoder's many linear layers). Force the outputs + carried
    // state every `EVAL_CHUNK` steps so MLX materializes and frees the chunk's transients, bounding
    // peak buffers to one chunk. Decode (`t == 1`) and short prefills never reach the cadence, so their
    // graphs — and the per-step sync cost `sc-7469` warned against — are unchanged.
    const EVAL_CHUNK: i32 = 256;
    let mut ys: Vec<Array> = Vec::with_capacity(t as usize);
    let mut flushed = 0usize;
    for ti in 0..t {
        let qt = slice_time(&q, ti)?; // [B,Hv,Dk]
        let kt = slice_time(&k, ti)?; // [B,Hv,Dk]
        let vt = slice_time(v, ti)?; // [B,Hv,Dv]
        let gt = slice_time(g, ti)?; // [B,Hv]
        let bt = slice_time(beta, ti)?; // [B,Hv]
        let (y, next) = delta_step(&qt, &kt, &vt, &gt, &bt, &state, b, hv, dk, dv)?;
        state = next;
        ys.push(y.expand_dims(1)?); // [B,1,Hv,Dv]
        if (ti + 1) % EVAL_CHUNK == 0 {
            // Force this chunk's outputs + the carried state (which transitively pins every step's
            // index/intermediate arrays) so they free before the next chunk.
            eval(ys[flushed..].iter().chain(std::iter::once(&state)))?;
            flushed = ys.len();
        }
    }
    let refs: Vec<&Array> = ys.iter().collect();
    let y = concatenate_axis(&refs, 1)?; // [B,T,Hv,Dv]
    Ok((y, state))
}

/// Causal depthwise short convolution over `[B, S, C]` with per-channel kernel `weight` `[C, K]`
/// (the HF/MLX depthwise `Conv1d`, no bias), left-seeded by `conv_state` `[B, K-1, C]` (the previous
/// step's tail). Returns `(silu(conv) [B,S,C], new_conv_state [B,K-1,C])` — a port of the Qwen3-Next
/// short-conv path: `out[b,s,c] = silu(Σ_j weight[c,j] · concat(conv_state, x)[b, s+j, c])`. Mixing
/// q/k/v through this 1-D conv before the recurrence is what gives Gated DeltaNet its local context.
pub fn causal_depthwise_conv(x: &Array, weight: &Array, conv_state: &Array) -> Result<(Array, Array)> {
    let xs = x.shape();
    let (s, c) = (xs[1], xs[2]);
    let kk = weight.shape()[1]; // kernel size K (weight is [C, K])
    let cat = concatenate_axis(&[conv_state, x], 1)?; // [B, S+K-1, C]
    let mut acc: Option<Array> = None;
    for j in 0..kk {
        let window = slice_seq(&cat, j, j + s)?; // cat[:, j:j+S, :] → [B,S,C]
        let wj = weight
            .take_axis(Array::from_slice(&[j], &[1]), 1)? // weight[:, j] → [C,1]
            .reshape(&[1, 1, c])?; // → [1,1,C] (broadcast over B,S)
        let term = multiply(&window, &wj)?;
        acc = Some(match acc {
            None => term,
            Some(a) => add(&a, &term)?,
        });
    }
    let out = silu(acc.expect("conv kernel size must be >= 1"))?; // [B,S,C]
    let new_state = slice_seq(&cat, s, s + kk - 1)?; // last K-1 of conv_in → [B,K-1,C]
    Ok((out, new_state))
}

/// Gated RMSNorm (`Qwen3NextRMSNormGated`): `rms_norm(x, weight, eps) · silu(gate)`. Applied to the
/// delta-net output before the out-projection (`x`, `gate`, and `weight` share the head-value dim).
pub fn rms_norm_gated(x: &Array, weight: &Array, gate: &Array, eps: f32) -> Result<Array> {
    let normed = mlx_rs::fast::rms_norm(x, weight, eps)?;
    Ok(multiply(&normed, &silu(gate)?)?)
}

/// The recurrent state of one Gated DeltaNet layer — the linear-attention analog of a KV-cache slot
/// (the Mamba/SSM cache). It holds the short-conv tail `conv_state` `[B, K-1, conv_dim]` and the
/// delta-rule `ssm_state` `[B, Hv, Dv, Dk]`, both **fixed size** in sequence length (unlike the
/// growing KV cache). A hybrid decoder keeps one of these per linear layer alongside a
/// [`KvCache`](super::KvCache) per full-attention layer (the decoder assembles the mixed list).
#[derive(Clone, Debug, Default)]
pub struct DeltaNetCache {
    /// The short-conv history (previous `K-1` tokens), or `None` before the first step.
    pub conv_state: Option<Array>,
    /// The delta-rule recurrent state, or `None` before the first step.
    pub ssm_state: Option<Array>,
    offset: i32,
}

impl DeltaNetCache {
    /// An empty cache (no conv history, zero recurrent state).
    pub fn new() -> Self {
        Self::default()
    }

    /// Positions consumed so far (the linear-layer analog of [`KvCache::offset`](super::KvCache::offset)).
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Store the post-step `(conv_state, ssm_state)` and advance the position by `step` tokens.
    pub fn update(&mut self, conv_state: Array, ssm_state: Array, step: i32) {
        self.conv_state = Some(conv_state);
        self.ssm_state = Some(ssm_state);
        self.offset += step;
    }

    /// Drop all state, returning the cache to its freshly-constructed condition.
    pub fn reset(&mut self) {
        self.conv_state = None;
        self.ssm_state = None;
        self.offset = 0;
    }
}

/// Slice `x` `[B, L, ...]` to `x[:, start:end, ...]` (a contiguous range along the sequence axis).
fn slice_seq(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    let arr = Array::from_slice(&idx, &[idx.len() as i32]);
    Ok(x.take_axis(&arr, 1)?)
}

/// One recurrent step (`_gated_delta_step_ops`). `q`,`k` `[B,Hv,Dk]`; `v` `[B,Hv,Dv]`; `g`,`beta`
/// `[B,Hv]`; `state` `[B,Hv,Dv,Dk]`. Returns `(y [B,Hv,Dv], new_state [B,Hv,Dv,Dk])`.
#[allow(clippy::too_many_arguments)]
fn delta_step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    b: i32,
    hv: i32,
    dk: i32,
    dv: i32,
) -> Result<(Array, Array)> {
    let decay = g.reshape(&[b, hv, 1, 1])?; // [B,Hv,1,1]
    let state = multiply(state, &decay)?; // S · g
    let k_r = k.reshape(&[b, hv, 1, dk])?; // [B,Hv,1,Dk]
    let kv_mem = sum_axis(&multiply(&state, &k_r)?, -1, false)?; // (S·k).sum(Dk) → [B,Hv,Dv]
    let delta = multiply(&subtract(v, &kv_mem)?, &beta.reshape(&[b, hv, 1])?)?; // (v−kv)·β → [B,Hv,Dv]
    let state = add(&state, &multiply(&k_r, &delta.reshape(&[b, hv, dv, 1])?)?)?; // S + Δ⊗k
    let q_r = q.reshape(&[b, hv, 1, dk])?;
    let y = sum_axis(&multiply(&state, &q_r)?, -1, false)?; // (S·q).sum(Dk) → [B,Hv,Dv]
    Ok((y, state))
}

/// Repeat each head of `x` `[B,T,H,D]` `r` times along the head axis (contiguous), giving
/// `[B,T,H·r,D]` — the GQA expansion (`mx.repeat(x, r, axis=-2)`).
fn repeat_heads(x: &Array, r: i32) -> Result<Array> {
    let s = x.shape();
    let (b, t, h, d) = (s[0], s[1], s[2], s[3]);
    let expanded = x.expand_dims(3)?; // [B,T,H,1,D]
    let broad = broadcast_to(&expanded, &[b, t, h, r, d])?; // [B,T,H,r,D]
    Ok(broad.reshape(&[b, t, h * r, d])?) // [B,T,H·r,D]
}

/// Take time index `t` along axis 1 of `[B, T, ...]`, dropping that axis.
fn slice_time(x: &Array, t: i32) -> Result<Array> {
    let idx = Array::from_slice(&[t], &[1]);
    let picked = x.take_axis(&idx, 1)?; // [B,1,...]
    let mut shape: Vec<i32> = picked.shape().to_vec();
    shape.remove(1);
    Ok(picked.reshape(&shape)?)
}

/// A zero recurrent state `[B, Hv, Dv, Dk]` in `dtype`.
fn zeros_state(b: i32, hv: i32, dv: i32, dk: i32, dtype: Dtype) -> Result<Array> {
    let n = (b * hv * dv * dk) as usize;
    Ok(Array::from_slice(&vec![0.0f32; n], &[b, hv, dv, dk]).as_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::sigmoid;

    // Numeric oracle generated from `mlx_lm.models.gated_delta.gated_delta_update(..., use_kernel=
    // False)` (the ops reference) on seeded inputs — B=1, T=3, Hk=2, Hv=4 (GQA ×2), Dk=2, Dv=2.
    // Regenerate with the venv MLX: see story sc-7627.
    const Q: &[f32] = &[
        -0.8805123, -0.2520431, 0.6974789, -0.8069373, 0.5367976, -0.5163696, 0.4404979, -0.721925,
        0.7900037, -0.5997525, 0.0088437, -0.2685511,
    ];
    const K: &[f32] = &[
        -0.5530653, -0.0336156, 0.34984, -0.9507952, 0.3385786, 0.7130073, 0.4130355, 0.7005113,
        -0.9424242, -0.8679754, 0.0911343, -0.1635016,
    ];
    const V: &[f32] = &[
        0.9033704, 0.9242766, -0.5594231, 0.5815381, -0.9665546, 0.8363392, 0.8484675, -0.1261052,
        -0.8719618, 0.0562458, 0.8790255, 0.7971791, -0.8360307, -0.2071904, -0.6701862, -0.7691332,
        -0.2697922, -0.7519733, 0.0098643, -0.2587476, 0.5248392, 0.9719371, -0.0304079, -0.2898675,
    ];
    const A: &[f32] = &[
        -0.6942793, -1.9834223, 1.6954608, 1.8882596, 1.5180013, 1.9647338, -0.7497661, -1.9337821,
        -1.3342639, 1.7648504, -0.3198055, -1.4922521,
    ];
    const B: &[f32] = &[
        -1.6042936, 1.7927692, 0.1977825, 0.2890182, 0.152185, -0.4371433, 0.8649859, 0.2619474,
        -1.2190499, -1.3681375, -1.4745429, 1.3650055,
    ];
    const A_LOG: &[f32] = &[2.0919125, 1.5201275, 2.7469416, 0.104467];
    const DT_BIAS: &[f32] = &[-0.2865368, -0.2390987, 0.5489618, 0.1692053];
    const EXP_Y: &[f32] = &[
        0.0749167, 0.0766504, -0.2376069, 0.2469999, -0.5368805, 0.4645513, 0.490568, -0.0729116,
        0.0874512, -0.0056413, -0.0642828, -0.0583461, 0.1904479, 0.0472361, 0.4258981, 0.0956538,
        0.0364119, 0.0369535, -0.0004748, 0.0117344, 0.0043713, 0.0080946, 0.1139456, 0.041226,
    ];
    const EXP_STATE: &[f32] = &[
        0.0431024, -0.0039364, 0.1626127, 0.1525815, -0.0018656, -0.0016657, 0.0495011, 0.0456383,
        0.0089079, -0.015984, 0.0164975, -0.0295984, 0.0197504, -0.4236471, -0.1824342, -0.1595202,
    ];

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn recurrence_matches_python_ops_reference() {
        let q = Array::from_slice(Q, &[1, 3, 2, 2]);
        let k = Array::from_slice(K, &[1, 3, 2, 2]);
        let v = Array::from_slice(V, &[1, 3, 4, 2]);
        let a = Array::from_slice(A, &[1, 3, 4]);
        let b_raw = Array::from_slice(B, &[1, 3, 4]);
        let a_log = Array::from_slice(A_LOG, &[4]);
        let dt_bias = Array::from_slice(DT_BIAS, &[4]);

        // beta = sigmoid(b); g = compute_g(a, A_log, dt_bias) — exactly what gated_delta_update does.
        let beta = sigmoid(&b_raw).unwrap();
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        let (y, state) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        assert_eq!(y.shape(), &[1, 3, 4, 2]);
        assert_eq!(state.shape(), &[1, 4, 2, 2]);

        let yh = y.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        let sh = state.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec();
        assert!(max_abs_diff(&yh, EXP_Y) < 1e-4, "y diff {}", max_abs_diff(&yh, EXP_Y));
        assert!(
            max_abs_diff(&sh, EXP_STATE) < 1e-4,
            "state diff {}",
            max_abs_diff(&sh, EXP_STATE)
        );
    }

    #[test]
    fn decode_step_matches_chunked_prefill() {
        // Feeding the sequence one token at a time (carrying state) must equal one T-step call —
        // the prefill/decode equivalence the hybrid cache relies on.
        let q = Array::from_slice(Q, &[1, 3, 2, 2]);
        let k = Array::from_slice(K, &[1, 3, 2, 2]);
        let v = Array::from_slice(V, &[1, 3, 4, 2]);
        let g = Array::from_slice(
            &[0.6f32, 0.7, 0.8, 0.9, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 0.4, 0.45],
            &[1, 3, 4],
        );
        let beta = Array::from_slice(
            &[0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.15, 0.25, 0.35],
            &[1, 3, 4],
        );

        let (y_full, s_full) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None).unwrap();

        // Step one token at a time, carrying the state.
        let pick = |x: &Array, t: i32| -> Array { slice_time(x, t).unwrap().expand_dims(1).unwrap() };
        let mut state: Option<Array> = None;
        let mut ys = Vec::new();
        for t in 0..3 {
            let (y, s) = gated_delta_recurrence(
                &pick(&q, t),
                &pick(&k, t),
                &pick(&v, t),
                &pick(&g, t),
                &pick(&beta, t),
                state.as_ref(),
            )
            .unwrap();
            ys.push(y);
            state = Some(s);
        }
        let refs: Vec<&Array> = ys.iter().collect();
        let y_step = concatenate_axis(&refs, 1).unwrap();

        let yf = y_full.as_slice::<f32>().to_vec();
        let ys = y_step.as_slice::<f32>().to_vec();
        let sf = s_full.as_slice::<f32>().to_vec();
        let ss = state.unwrap().as_slice::<f32>().to_vec();
        assert!(max_abs_diff(&yf, &ys) < 1e-5, "prefill vs step y: {}", max_abs_diff(&yf, &ys));
        assert!(max_abs_diff(&sf, &ss) < 1e-5, "prefill vs step state");
    }

    #[test]
    fn compute_g_is_in_unit_interval_and_shaped() {
        // g = exp(−positive) ∈ (0, 1]: a per-head forget gate.
        let a = Array::from_slice(A, &[1, 3, 4]);
        let a_log = Array::from_slice(A_LOG, &[4]);
        let dt_bias = Array::from_slice(DT_BIAS, &[4]);
        let g = compute_g(&a, &a_log, &dt_bias).unwrap();
        assert_eq!(g.shape(), &[1, 3, 4]);
        for x in g.as_slice::<f32>() {
            assert!(*x > 0.0 && *x <= 1.0 + 1e-6, "gate out of (0,1]: {x}");
        }
    }

    // Conv oracle from MLX's depthwise `nn.Conv1d` + silu — C=3, K=4, S=2, conv_state K-1=3.
    const CW: &[f32] = &[
        0.2422637, 0.3207079, 0.3157361, 0.493552, 0.6888506, 0.2758986, -0.2604986, 0.5719915,
        -0.677193, -0.1786878, -0.1721832, 0.024104,
    ];
    const CX: &[f32] = &[-0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096, 0.7106992];
    const CSTATE: &[f32] = &[
        0.2526282, 0.8012137, 0.0075967, -0.7815045, -0.8113322, -0.8019857, 0.0879031, 0.8514872,
        0.00599,
    ];
    const CEXP_OUT: &[f32] = &[-0.1465172, 0.0095216, 0.0727914, -0.1432332, -0.2082713, 0.3602719];
    const CEXP_STATE: &[f32] = &[
        0.0879031, 0.8514872, 0.00599, -0.3929182, -0.15279, -0.0340949, -0.1223795, -0.5179096,
        0.7106992,
    ];

    #[test]
    fn causal_conv_matches_mlx_conv1d() {
        // weight stored [C,K,1] in the checkpoint; the helper takes [C,K] (squeezed).
        let weight = Array::from_slice(CW, &[3, 4]);
        let x = Array::from_slice(CX, &[1, 2, 3]);
        let state = Array::from_slice(CSTATE, &[1, 3, 3]);
        let (out, new_state) = causal_depthwise_conv(&x, &weight, &state).unwrap();
        assert_eq!(out.shape(), &[1, 2, 3]);
        assert_eq!(new_state.shape(), &[1, 3, 3]);
        assert!(max_abs_diff(out.as_slice::<f32>(), CEXP_OUT) < 1e-5);
        // new conv_state is the last K-1 tokens of [conv_state ++ x] — exact (a slice, no arithmetic).
        assert_eq!(new_state.as_slice::<f32>().to_vec(), CEXP_STATE.to_vec());
    }

    #[test]
    fn delta_cache_tracks_state_and_offset() {
        let mut cache = DeltaNetCache::new();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state.is_none() && cache.ssm_state.is_none());
        let conv = Array::from_slice(&[0.0f32; 9], &[1, 3, 3]);
        let ssm = Array::from_slice(&[0.0f32; 16], &[1, 4, 2, 2]);
        cache.update(conv, ssm, 5);
        assert_eq!(cache.offset(), 5);
        assert!(cache.conv_state.is_some() && cache.ssm_state.is_some());
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert!(cache.conv_state.is_none());
    }

    /// A long prefill must not exhaust the Metal allocator. The recurrence builds the whole `t`-step
    /// graph before a single eval, eagerly allocating per-step index buffers; without periodic flushing
    /// a sequence past ~100k steps trips the allocator's resource limit and the next gather fails with a
    /// spurious "expected a non-empty mlx_array". `t = 130_000` clears that limit (~5 buffers/step >
    /// 499_000) and must now complete because the recurrence flushes every `EVAL_CHUNK` steps. Marked
    /// `ignore` only for runtime (the loop is long), not flakiness — it is the direct regression guard
    /// for the high-res-image vision crash.
    #[test]
    #[ignore = "slow: 130k-step recurrence; guards the long-prefill Metal buffer-limit regression"]
    fn long_prefill_does_not_exhaust_buffer_limit() {
        let t = 130_000i32;
        let q = Array::from_slice(&vec![0.01f32; t as usize], &[1, t, 1, 1]);
        let k = Array::from_slice(&vec![0.02f32; t as usize], &[1, t, 1, 1]);
        let v = Array::from_slice(&vec![0.03f32; t as usize], &[1, t, 1, 1]);
        let g = Array::from_slice(&vec![0.9f32; t as usize], &[1, t, 1]);
        let beta = Array::from_slice(&vec![0.5f32; t as usize], &[1, t, 1]);
        let (y, state) = gated_delta_recurrence(&q, &k, &v, &g, &beta, None)
            .expect("long recurrence must not exhaust the Metal allocator");
        assert_eq!(y.shape(), &[1, t, 1, 1]);
        assert_eq!(state.shape(), &[1, 1, 1, 1]);
        // Force a final materialization so a deferred allocator failure can't hide behind laziness.
        eval([&y, &state]).unwrap();
    }
}
