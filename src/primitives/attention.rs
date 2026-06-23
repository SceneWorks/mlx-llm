//! Attention leaves: grouped-query KV expansion and a scaled-dot-product-attention wrapper.
//!
//! The decoders run GQA — fewer KV heads than query heads — so cached K/V must be expanded to the
//! query head count before attention. [`repeat_kv`] is the `[b, hkv, s, hd] -> [b, hkv*groups, s, hd]`
//! broadcast the mlx-gen stacks use. [`sdpa`] wraps MLX's fused `scaled_dot_product_attention`,
//! exposing the two masking modes the references need: implicit bottom-right [`AttnMask::Causal`]
//! (decode) and an explicit [`AttnMask::Additive`] mask (the block-causal / bidirectional paths).

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, matmul, multiply, softmax_axis};
use mlx_rs::{Array, Dtype};

use crate::error::Result;
use crate::primitives::nn::soft_cap;

/// Disallowed-attention fill for the eager additive mask: a large finite negative (matching the
/// reference slices — avoids `-inf` propagating through the softmax).
const MASK_NEG: f32 = -1e30;

/// Max query rows per fused SDPA call (sc-7430 / sc-7455). The pinned pmetal mlx-rs fork's fused
/// `scaled_dot_product_attention` **GPU** kernel miscompiles its `q_len > 8` "full/steel" path for
/// multi-head, power-of-2 `head_dim` (64, 128, …) — returning numerically wrong attention. The
/// `q_len <= 8` kernel is correct for ANY cache length, so [`sdpa`] never hands the fused kernel more
/// than this many query rows on the affected shapes; it chunks prefill instead. When the
/// `sc7430_*` tripwire test starts failing, the fork is fixed and this whole mitigation can go.
const SDPA_MAX_FUSED_QLEN: i32 = 8;

/// How attention should be masked.
#[derive(Debug, Clone, Copy)]
pub enum AttnMask<'a> {
    /// No mask (fully bidirectional) — e.g. a vision tower.
    None,
    /// Implicit causal mask. MLX aligns the `q_len` queries to the bottom-right of the cached
    /// keys, so query `r` attends keys `0..=offset+r` — exactly right for cached decode.
    Causal,
    /// An explicit additive mask broadcast over the score tensor (`0` keep, `-inf` block).
    Additive(&'a Array),
}

/// Expand grouped-query KV heads to the full query head count.
///
/// `x` is `[batch, n_kv_heads, seq, head_dim]`; the result is `[batch, n_kv_heads * groups, seq,
/// head_dim]` where `groups = n_query_heads / n_kv_heads`. `groups == 1` (MHA) is a no-op clone.
pub fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, hkv, s, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let expanded = x.expand_dims(2)?; // [b, hkv, 1, s, hd]
    let broad = broadcast_to(&expanded, &[b, hkv, groups, s, hd])?;
    Ok(broad.reshape(&[b, hkv * groups, s, hd])?)
}

/// Scaled-dot-product attention over `[batch, heads, seq, head_dim]` tensors.
///
/// `scale` is the usual `head_dim^(-0.5)`. Grouped-query attention is handled **natively** by MLX —
/// pass K/V with `n_kv_heads` (fewer than the query heads) and the fused kernel derives
/// `gqa_factor = q_heads / kv_heads`, reading K/V by head stride with no head-count materialization.
/// (Pre-expanding with [`repeat_kv`] is equivalent but allocates a `groups`× larger K/V — avoid it
/// on the hot path; see sc-7307.)
///
/// On the pinned pmetal fork the fused kernel is wrong for `q_len > 8` × multi-head × power-of-2
/// `head_dim` (sc-7430), so for those shapes this **chunks the queries** into `≤ 8`-row pieces and
/// runs the correct kernel on each (sc-7455). Decode (`q_len = 1`) and short/odd-head-dim prefill go
/// straight to the fused kernel unchanged.
pub fn sdpa(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let (heads, q_len, head_dim) = (queries.shape()[1], queries.shape()[2], queries.shape()[3]);
    let hits_broken_kernel = q_len > SDPA_MAX_FUSED_QLEN
        && heads >= 2
        && head_dim >= 64
        && (head_dim as u32).is_power_of_two();
    if hits_broken_kernel {
        return sdpa_chunked_prefill(queries, keys, values, scale, mask);
    }
    sdpa_fused(queries, keys, values, scale, mask)
}

/// The raw fused MLX kernel — correct for `q_len <= 8` (and any `q_len` at non-power-of-2 head dims).
fn sdpa_fused(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let m: Option<ScaledDotProductAttentionMask> = match mask {
        AttnMask::None => None,
        AttnMask::Causal => Some(ScaledDotProductAttentionMask::Causal),
        AttnMask::Additive(a) => Some(ScaledDotProductAttentionMask::Array(a)),
    };
    Ok(scaled_dot_product_attention(
        queries, keys, values, scale, m, None,
    )?)
}

/// A contiguous `[start, end)` index vector for [`Array::take_axis`].
fn range_index(start: i32, end: i32) -> Array {
    Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start])
}

/// sc-7455 mitigation for the broken `q_len > 8` fused kernel: split the queries into `≤ 8`-row
/// chunks so every fused call lands on the correct path, attend each chunk against exactly the keys
/// it should, and concatenate. Mathematically identical to one correct attention call (gated against
/// a host reference by `fused_sdpa_correct_vs_host_for_short_qlen` + the chunked-path test).
///
/// - **Causal**: chunk rows `[c0, c1)` (with `offset = k_len − q_len`) attend keys `0..(offset+c1)`;
///   slice K/V to that prefix and use the implicit causal mask, which then bottom-right-aligns the
///   chunk correctly (`offset' = (offset+c1) − (c1−c0) = offset+c0`). Caps the score matrix at
///   `[·, ·, 8, k_len]` — no full `[·, ·, q_len, k_len]` materialization. The prefix is taken with a
///   strided **slice view** (`index`, `mlx_slice`), not a `take_axis` gather — no index-vector alloc
///   and no eager copy of the growing prefix per chunk (sc-7469).
/// - **None**: every query attends all keys — pass the full K/V.
/// - **Additive**: the mask already encodes visibility, so pass full K/V and slice the mask's query
///   axis to `[c0, c1)` (when that axis isn't broadcast).
///
/// The chunk outputs stay **lazy** and are concatenated into one graph for the caller to `eval` — the
/// same contract as [`sdpa_fused`]. sc-7469 measured the original per-chunk `eval` (a GPU sync every
/// chunk × every layer, ~1900 syncs for a 512-token 30-layer prefill) at **~95% of total prefill
/// time**, collapsing a 135M prefill from ~57k to ~1.1k tok/s. It bounded nothing the single forward
/// `eval` doesn't already bound: MLX frees each chunk's K/V gather as it streams the graph, so peak
/// transient is one chunk's key prefix, not the sum — exactly as the pre-mitigation fused prefill
/// (which never OOM'd) behaved ([[mlx-benchmarks-must-bound-memory]] is about unbounded *multi-eval*
/// sweeps, not one forward's intermediates).
fn sdpa_chunked_prefill(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let q_len = queries.shape()[2];
    let k_len = keys.shape()[2];
    let offset = k_len - q_len; // cached prefix before the new queries; ≥ 0 for cached decode/prefill

    let mut outs: Vec<Array> = Vec::with_capacity(((q_len + SDPA_MAX_FUSED_QLEN - 1) / SDPA_MAX_FUSED_QLEN) as usize);
    let mut c0 = 0;
    while c0 < q_len {
        let c1 = (c0 + SDPA_MAX_FUSED_QLEN).min(q_len);
        let q_chunk = queries.try_index((.., .., c0..c1, ..))?; // [·, ·, c1−c0, hd] view
        let out = match mask {
            AttnMask::None => sdpa_fused(&q_chunk, keys, values, scale, AttnMask::None)?,
            AttnMask::Causal => {
                let end = offset + c1;
                let k_chunk = keys.try_index((.., .., 0..end, ..))?; // [·, ·, end, hd] prefix view
                let v_chunk = values.try_index((.., .., 0..end, ..))?;
                sdpa_fused(&q_chunk, &k_chunk, &v_chunk, scale, AttnMask::Causal)?
            }
            AttnMask::Additive(a) => {
                let q_axis = a.ndim() as i32 - 2;
                if a.shape()[q_axis as usize] == q_len {
                    let a_chunk = a.take_axis(range_index(c0, c1), q_axis)?;
                    sdpa_fused(&q_chunk, keys, values, scale, AttnMask::Additive(&a_chunk))?
                } else {
                    sdpa_fused(&q_chunk, keys, values, scale, AttnMask::Additive(a))?
                }
            }
        };
        outs.push(out); // stay lazy: the caller's single forward `eval` streams + frees each chunk's
                        // K/V prefix slice, so peak transient is one chunk (sc-7469 — no per-chunk sync)
        c0 = c1;
    }
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(concatenate_axis(&refs, 2)?)
}

/// Convenience: causal attention (the decode default).
pub fn sdpa_causal(queries: &Array, keys: &Array, values: &Array, scale: f32) -> Result<Array> {
    sdpa(queries, keys, values, scale, AttnMask::Causal)
}

/// Scaled-dot-product attention with optional Gemma-2 score soft-cap, dispatching to the fused MLX
/// kernel when it can serve the case and an explicit eager path otherwise.
///
/// The fused [`sdpa`] cannot express two things the breadth architectures need: Gemma-2's
/// attention-score soft-cap (`c·tanh(scores/c)` before the softmax), and DeepSeek-V2 MLA's mismatched
/// query/key head dim (192) vs value head dim (128). When `softcap` is `None` **and** q/v share a
/// head dim, this is exactly the fused native-GQA hot path (unchanged — `softcap.is_none()` callers
/// pay nothing). Otherwise it runs the f32-precise eager `softmax(scale·QKᵀ [+softcap] +mask)·V`, with
/// K/V GQA-expanded inside (see [`repeat_kv`]).
pub fn sdpa_capped(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let q_hd = queries.shape()[3];
    let v_hd = values.shape()[3];
    if softcap.is_none() && q_hd == v_hd {
        return sdpa(queries, keys, values, scale, mask);
    }
    sdpa_eager(queries, keys, values, scale, softcap, mask)
}

/// The eager `softmax(scale · QKᵀ [+ softcap] + mask) · V` path — the portable fallback
/// [`sdpa_capped`] runs when the fused kernel can't serve the case. Computed in f32 (the scores,
/// soft-cap, and softmax all upcast), then cast back to the queries' dtype, so bf16 decoders stay
/// numerically stable through the score soft-cap. K/V are GQA-expanded here (`groups = q_heads /
/// kv_heads`); the value head dim may differ from the query/key head dim (MLA).
fn sdpa_eager(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    softcap: Option<f32>,
    mask: AttnMask<'_>,
) -> Result<Array> {
    let out_dtype = queries.dtype();
    let groups = queries.shape()[1] / keys.shape()[1];
    let q = queries.as_dtype(Dtype::Float32)?;
    let k = repeat_kv(&keys.as_dtype(Dtype::Float32)?, groups)?;
    let v = repeat_kv(&values.as_dtype(Dtype::Float32)?, groups)?;
    let (b, nh) = (q.shape()[0], q.shape()[1]);
    let q_len = q.shape()[2];
    let k_len = k.shape()[2];

    // scores = (q @ kᵀ) * scale → [b, heads, q_len, k_len]. MLX's batched 4-D `matmul` misreads the
    // `[b, heads]` leading dims on this fork (wrong results for multi-head/large shapes — the fused
    // SDPA avoids raw matmul), so fold `[b, heads]` into one batch axis and run 3-D batched matmuls.
    let kt = k.transpose_axes(&[0, 1, 3, 2])?;
    let scores = bmm(&q, &kt, b * nh)?; // [b, heads, q_len, k_len]
    let mut scores = multiply(&scores, Array::from_f32(scale))?;
    if let Some(c) = softcap {
        scores = soft_cap(&scores, c)?;
    }
    let scores = match mask {
        AttnMask::None => scores,
        AttnMask::Causal => add(&scores, &causal_mask(q_len, k_len)?)?,
        AttnMask::Additive(a) => add(&scores, &a.as_dtype(Dtype::Float32)?)?,
    };
    let last_axis = scores.ndim() as i32 - 1;
    let weights = softmax_axis(&scores, last_axis, None)?;
    let out = bmm(&weights, &v, b * nh)?; // [b, heads, q_len, v_head_dim]
    Ok(out.as_dtype(out_dtype)?)
}

/// Batched matrix multiply `a @ b` over 4-D `[lead0, lead1, m, k] @ [lead0, lead1, k, n]` tensors,
/// run as a **3-D** batched matmul over a single folded batch axis (`batch = lead0 · lead1`). MLX's
/// 4-D batched `matmul` returns wrong results on this fork for multi-batch/large shapes; folding to
/// 3-D sidesteps it. Reshape materializes any transposed/broadcast operand into row-major storage.
fn bmm(a: &Array, b: &Array, batch: i32) -> Result<Array> {
    let sa = a.shape();
    let sb = b.shape();
    let (lead0, lead1, m, k) = (sa[0], sa[1], sa[2], sa[3]);
    let n = sb[3];
    let a3 = a.reshape(&[batch, m, k])?;
    let b3 = b.reshape(&[batch, k, n])?;
    Ok(matmul(&a3, &b3)?.reshape(&[lead0, lead1, m, n])?)
}

/// The additive causal mask `[1, 1, q_len, k_len]` (`0` keep / [`MASK_NEG`] block) for keys that
/// include `offset = k_len - q_len` cached positions before the new queries — bottom-right aligned, so
/// query row `r` attends keys `0..=offset+r` (matching the fused kernel's implicit causal convention).
fn causal_mask(q_len: i32, k_len: i32) -> Result<Array> {
    let offset = k_len - q_len;
    let mut data = vec![0f32; (q_len * k_len) as usize];
    for r in 0..q_len {
        for j in 0..k_len {
            if j > offset + r {
                data[(r * k_len + j) as usize] = MASK_NEG;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, 1, q_len, k_len]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_kv_noop_for_one_group() {
        let x = Array::from_slice(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[1, 2, 3, 4]);
        let y = repeat_kv(&x, 1).unwrap();
        assert_eq!(y.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn repeat_kv_expands_head_axis() {
        let x = Array::from_slice(&(0..16).map(|i| i as f32).collect::<Vec<_>>(), &[1, 2, 2, 4]);
        let y = repeat_kv(&x, 4).unwrap();
        assert_eq!(y.shape(), &[1, 8, 2, 4]);
    }

    #[test]
    fn repeat_kv_duplicates_each_head() {
        // Two KV heads, head_dim 2, seq 1: head0 = [0,1], head1 = [2,3].
        let x = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 1, 2]);
        let y = repeat_kv(&x, 2).unwrap(); // [1, 4, 1, 2]
        let h = y.as_slice::<f32>().to_vec();
        // groups are adjacent: head0,head0,head1,head1
        assert_eq!(h, vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn sdpa_causal_runs_and_shapes() {
        // [b=1, heads=1, seq=2, hd=4]
        let q = Array::from_slice(&(0..8).map(|i| i as f32 * 0.1).collect::<Vec<_>>(), &[1, 1, 2, 4]);
        let out = sdpa_causal(&q, &q, &q, 0.5).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 4]);
    }

    fn randf(shape: &[i32], seed: u64) -> Array {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let mut s = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        Array::from_slice(&data, shape)
    }

    /// Passing GQA-shaped K/V straight to `sdpa` (MLX native GQA) is numerically identical to the
    /// old `repeat_kv`-then-`sdpa` path — across decode (q_len=1), prefill (q_len>8), batched, and
    /// MHA shapes, for both the implicit-causal and no-mask paths. This is the correctness gate for
    /// dropping the per-step `repeat_kv` from the model (sc-7307).
    #[test]
    fn sdpa_native_gqa_matches_repeat_kv() {
        // (b, n_heads, n_kv_heads, seq, head_dim)
        let cases = [
            (1, 8, 2, 1, 64),    // decode, groups=4 (vector path)
            (1, 32, 8, 1, 128),  // decode, Qwen3-like hd=128, groups=4
            (1, 8, 2, 16, 64),   // prefill q_len=16>8 (full/steel path), groups=4
            (2, 6, 3, 5, 64),    // batched, groups=2
            (1, 4, 4, 7, 64),    // MHA groups=1 (repeat_kv is a no-op — must be unchanged)
        ];
        for (b, nh, nkv, s, hd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[b, nh, s, hd], 1);
            let k = randf(&[b, nkv, s, hd], 2);
            let v = randf(&[b, nkv, s, hd], 3);
            let groups = nh / nkv;

            for mask in [AttnMask::None, AttnMask::Causal] {
                let native = sdpa(&q, &k, &v, scale, mask).unwrap();
                let expanded = sdpa(
                    &q,
                    &repeat_kv(&k, groups).unwrap(),
                    &repeat_kv(&v, groups).unwrap(),
                    scale,
                    mask,
                )
                .unwrap();
                assert_eq!(native.shape(), expanded.shape());
                let (a, e) = (
                    native.as_slice::<f32>().to_vec(),
                    expanded.as_slice::<f32>().to_vec(),
                );
                let maxdiff = a.iter().zip(&e).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                assert!(
                    maxdiff < 1e-4,
                    "shape {b}/{nh}/{nkv}/{s}/{hd} mask {mask:?}: max|Δ| = {maxdiff}"
                );
            }
        }
    }

    /// From-scratch host attention `softmax(scale·QKᵀ [+causal])·V` over `[1, h, ·, hd]` GQA tensors
    /// (kv expanded to `h`), f64 accumulation — the ground truth the chunked-prefill mitigation is
    /// gated against. Causal aligns the `ql` queries to the bottom-right of the `kl` keys
    /// (offset = kl − ql), so it also covers a cached prefix.
    fn host_attn_gqa(q: &Array, k: &Array, v: &Array, groups: i32, scale: f32, causal: bool) -> Vec<f32> {
        let kx = repeat_kv(k, groups).unwrap();
        let vx = repeat_kv(v, groups).unwrap();
        let (qs, ks) = (q.shape(), kx.shape());
        let (h, ql, hd) = (qs[1] as usize, qs[2] as usize, qs[3] as usize);
        let kl = ks[2] as usize;
        let (qh, kh, vh) = (q.as_slice::<f32>(), kx.as_slice::<f32>(), vx.as_slice::<f32>());
        let offset = kl as isize - ql as isize;
        let mut out = vec![0f32; h * ql * hd];
        for head in 0..h {
            let (qb, kb) = (head * ql * hd, head * kl * hd);
            for i in 0..ql {
                let jmax = if causal { (offset + i as isize) as usize } else { kl - 1 };
                let mut logits = vec![0f64; kl];
                let mut m = f64::MIN;
                for j in 0..=jmax {
                    let dot: f64 = (0..hd).map(|d| qh[qb + i * hd + d] as f64 * kh[kb + j * hd + d] as f64).sum();
                    logits[j] = dot * scale as f64;
                    m = m.max(logits[j]);
                }
                let mut denom = 0f64;
                for lj in logits.iter_mut().take(jmax + 1) {
                    *lj = (*lj - m).exp();
                    denom += *lj;
                }
                for d in 0..hd {
                    let acc: f64 = (0..=jmax).map(|j| logits[j] / denom * vh[kb + j * hd + d] as f64).sum();
                    out[(head * ql + i) * hd + d] = acc as f32;
                }
            }
        }
        out
    }

    /// The eager path (no soft-cap, equal head dims) must match the fused kernel within a small
    /// tolerance — across decode/prefill/GQA shapes and the causal + no-mask cases. This is the
    /// correctness gate that lets Gemma-2 (which forces eager via soft-cap) and MLA (mismatched dims)
    /// trust the eager softmax against the same reference the rest of the engine uses.
    /// Host reference: per-head softmax(scale·q·kᵀ [+causal])·v over `[1, h, s, *]` MHA tensors. The
    /// value head dim (`vhd`) may differ from the q/k head dim (`hd`) — the DeepSeek MLA case.
    fn host_attn(q: &Array, k: &Array, v: &Array, scale: f32, causal: bool) -> Vec<f32> {
        let (qs, vs) = (q.shape(), v.shape());
        let (h, s, hd) = (qs[1] as usize, qs[2] as usize, qs[3] as usize);
        let vhd = vs[3] as usize;
        let (qh, kh, vh) = (q.as_slice::<f32>(), k.as_slice::<f32>(), v.as_slice::<f32>());
        let qk_at = |t: &[f32], head: usize, i: usize, d: usize| t[(head * s + i) * hd + d];
        let v_at = |head: usize, j: usize, d: usize| vh[(head * s + j) * vhd + d];
        let mut out = vec![0f32; h * s * vhd];
        for head in 0..h {
            for i in 0..s {
                let mut logits = vec![0f32; s];
                for (j, lj) in logits.iter_mut().enumerate() {
                    let dot: f32 = (0..hd).map(|d| qk_at(qh, head, i, d) * qk_at(kh, head, j, d)).sum();
                    *lj = dot * scale;
                }
                let jmax = if causal { i } else { s - 1 };
                let m = (0..=jmax).map(|j| logits[j]).fold(f32::MIN, f32::max);
                let mut denom = 0f32;
                let mut w = vec![0f32; s];
                for j in 0..=jmax {
                    w[j] = (logits[j] - m).exp();
                    denom += w[j];
                }
                for d in 0..vhd {
                    let acc: f32 = (0..=jmax).map(|j| w[j] / denom * v_at(head, j, d)).sum();
                    out[(head * s + i) * vhd + d] = acc;
                }
            }
        }
        out
    }

    fn rel_err(a: &[f32], host: &[f32]) -> f32 {
        let maxd = a.iter().zip(host).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let maxh = host.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        maxd / (maxh + 1e-20)
    }

    /// **The safe-path gate (sc-7430).** Fused `sdpa` is numerically correct vs a host reference for
    /// `q_len <= 8` at ALL the shapes the engine actually relies on: decode (`q_len = 1`, any cache
    /// length) and short / chunked prefill, multi-head GQA, the power-of-2 head dims 64 & 128, both
    /// masks. On the pinned pmetal fork the fused steel kernel miscompiles `q_len > 8` × multi-head ×
    /// head_dim ∈ {64,128} (see the tripwire below); this gate pins the regime that stays correct.
    #[test]
    fn fused_sdpa_correct_vs_host_for_short_qlen() {
        // (n_heads, n_kv_heads, q_len, k_len, head_dim)
        let cases = [
            (8, 2, 1, 64, 64),   // decode, long cache, hd=64
            (32, 8, 1, 256, 128), // decode, long cache, hd=128 (Qwen3-like)
            (8, 2, 8, 8, 64),    // short prefill q_len=8 (chunk boundary)
            (8, 2, 8, 256, 64),  // chunked prefill: 8 new queries into a 256-key cache
            (8, 2, 8, 8, 128),   // q_len=8 at hd=128 (chunk boundary for the Qwen3 head dim)
            (4, 4, 4, 64, 128),  // MHA, hd=128
        ];
        for (nh, nkv, ql, kl, hd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, nh, ql, hd], 1);
            let k = randf(&[1, nkv, kl, hd], 2);
            let v = randf(&[1, nkv, kl, hd], 3);
            let groups = nh / nkv;
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let out = sdpa(&q, &k, &v, scale, mask).unwrap();
                let host = host_attn_gqa(&q, &k, &v, groups, scale, causal);
                let e = rel_err(out.as_slice::<f32>(), &host);
                assert!(e < 2e-3, "fused sdpa wrong on a SAFE shape {nh}/{nkv}/{ql}/{kl}/{hd} causal={causal}: rel={e}");
            }
        }
    }

    /// The eager path must match a from-scratch host attention (the ground truth) — across GQA / MHA
    /// shapes, both masks, GQA expansion, and a mismatched value head dim (MLA). This is the
    /// correctness gate for Gemma-2 (forced eager by its score soft-cap) and DeepSeek-V2 MLA, pinning
    /// the eager f32 path directly to truth.
    #[test]
    fn sdpa_eager_matches_host_reference() {
        // (b, n_heads, n_kv_heads, seq, qk_head_dim, v_head_dim)
        let cases = [
            (1, 8, 2, 16, 64, 64), // prefill, groups=4
            (1, 2, 2, 16, 64, 64), // MHA big s/hd
            (1, 4, 4, 7, 32, 32),  // MHA small
            (1, 2, 2, 5, 6, 4),    // MLA-style: q/k head_dim 6, v head_dim 4
        ];
        let maxdiff = |a: &[f32], b: &[f32]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        for (b, nh, nkv, s, hd, vhd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[b, nh, s, hd], 1);
            let k = randf(&[b, nkv, s, hd], 2);
            let v = randf(&[b, nkv, s, vhd], 3);
            let groups = nh / nkv;
            let kx = repeat_kv(&k, groups).unwrap(); // [1, nh, s, hd] for the host reference
            let vx = repeat_kv(&v, groups).unwrap();
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let href = host_attn(&q, &kx, &vx, scale, causal);
                let eager = sdpa_eager(&q, &k, &v, scale, None, mask).unwrap();
                let de = maxdiff(&href, eager.as_slice::<f32>());
                assert!(de < 2e-3, "eager {b}/{nh}/{nkv}/{s}/{hd}/{vhd} causal={causal}: max|Δ| vs host = {de}");
            }
        }
    }

    /// **Characterization tripwire (sc-7430).** Documents the known pmetal-fork bug by calling the
    /// RAW fused kernel (`scaled_dot_product_attention`) directly — NOT the [`sdpa`] wrapper, which now
    /// chunks around it. The raw steel kernel returns numerically WRONG results for `q_len > 8` ×
    /// multi-head × head_dim ∈ {64,128} — diverging from the host reference by ~O(1) (rel ≈ 1.4),
    /// identically in f32/bf16/f16, GPU only. When this test FAILS, the fork/MLX has FIXED the kernel:
    /// drop the [`sdpa`] chunked-prefill mitigation (sc-7455) and delete this tripwire.
    #[test]
    fn sc7430_raw_fused_sdpa_still_broken_for_long_qlen_pow2_headdim() {
        let mut broken = 0;
        for hd in [64, 128] {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, 8, 16, hd], 1); // q_len=16 > 8, 8 heads
            let k = randf(&[1, 8, 16, hd], 2);
            let v = randf(&[1, 8, 16, hd], 3);
            let raw = scaled_dot_product_attention(
                &q, &k, &v, scale, Some(ScaledDotProductAttentionMask::Causal), None,
            )
            .unwrap();
            let host = host_attn_gqa(&q, &k, &v, 1, scale, true);
            if rel_err(raw.as_slice::<f32>(), &host) > 0.1 {
                broken += 1;
            }
        }
        assert_eq!(
            broken, 2,
            "raw fused SDPA is no longer broken for q_len>8 × multi-head × hd∈{{64,128}} — \
             the pmetal-fork steel kernel was fixed; remove the sdpa chunked-prefill mitigation"
        );
    }

    /// **The mitigation gate (sc-7455).** The [`sdpa`] wrapper — which chunks queries into ≤8-row
    /// pieces on the broken envelope — is numerically correct vs the host reference exactly where the
    /// raw kernel is wrong: `q_len > 8` × multi-head × head_dim ∈ {64,128}, across GQA/MHA, causal &
    /// no-mask, square prefill and a cached-prefix offset, incl. the real SmolLM2-135M prefill shape.
    #[test]
    fn sdpa_chunked_prefill_matches_host_for_long_qlen() {
        // (n_heads, n_kv_heads, q_len, k_len, head_dim)
        let cases = [
            (8, 2, 16, 16, 64),   // prefill square, GQA, hd64
            (8, 8, 64, 64, 128),  // prefill square, MHA, hd128, well past the chunk size
            (32, 8, 40, 40, 128), // Qwen3-like
            (8, 2, 16, 80, 64),   // 16 new queries into a 64-key cache (offset = 64)
            (9, 3, 26, 26, 64),   // SmolLM2-135M prefill shape (the sc-7430 real-model case)
        ];
        for (nh, nkv, ql, kl, hd) in cases {
            let scale = 1.0 / (hd as f32).sqrt();
            let q = randf(&[1, nh, ql, hd], 1);
            let k = randf(&[1, nkv, kl, hd], 2);
            let v = randf(&[1, nkv, kl, hd], 3);
            let groups = nh / nkv;
            for (causal, mask) in [(false, AttnMask::None), (true, AttnMask::Causal)] {
                let out = sdpa(&q, &k, &v, scale, mask).unwrap();
                assert_eq!(out.shape(), &[1, nh, ql, hd]);
                let host = host_attn_gqa(&q, &k, &v, groups, scale, causal);
                let e = rel_err(out.as_slice::<f32>(), &host);
                assert!(e < 2e-3, "chunked sdpa wrong {nh}/{nkv}/{ql}/{kl}/{hd} causal={causal}: rel={e}");
            }
        }
    }

    /// sc-7455: the chunked path also handles an explicit **Additive** mask (the batched-prefill
    /// block-causal mask, shape `[b,1,q_len,k_len]`) by slicing the mask's query axis per chunk — so
    /// the result matches a causal host reference. Guards the `decode/batch.rs` prefill path.
    #[test]
    fn sdpa_chunked_prefill_additive_mask_matches_host() {
        let (nh, ql, hd) = (8, 20, 64); // q_len=20 > 8, hd=64 → chunked
        let scale = 1.0 / (hd as f32).sqrt();
        let q = randf(&[1, nh, ql, hd], 1);
        let k = randf(&[1, nh, ql, hd], 2);
        let v = randf(&[1, nh, ql, hd], 3);
        // Additive causal mask [1,1,ql,ql]: 0 on/below the diagonal, -inf above.
        let mut md = vec![0f32; (ql * ql) as usize];
        for r in 0..ql {
            for j in 0..ql {
                if j > r {
                    md[(r * ql + j) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let m = Array::from_slice(&md, &[1, 1, ql, ql]);
        let out = sdpa(&q, &k, &v, scale, AttnMask::Additive(&m)).unwrap();
        let host = host_attn_gqa(&q, &k, &v, 1, scale, true);
        let e = rel_err(out.as_slice::<f32>(), &host);
        assert!(e < 2e-3, "chunked additive-mask sdpa wrong: rel={e}");
    }

    /// `sdpa_capped` routes a mismatched query/value head dim (DeepSeek-V2 MLA: q/k=6, v=4) through the
    /// eager path and produces finite `[b, heads, s, v_head_dim]` output.
    #[test]
    fn sdpa_capped_handles_mla_head_dims() {
        let (b, h, s, qk_hd, v_hd) = (1, 2, 3, 6, 4);
        let scale = 1.0 / (qk_hd as f32).sqrt();
        let q = randf(&[b, h, s, qk_hd], 1);
        let k = randf(&[b, h, s, qk_hd], 2);
        let v = randf(&[b, h, s, v_hd], 3);
        let out = sdpa_capped(&q, &k, &v, scale, None, AttnMask::Causal).unwrap();
        assert_eq!(out.shape(), &[b, h, s, v_hd]);
        assert!(out.as_slice::<f32>().iter().all(|x| x.is_finite()));
    }

    /// A dominant soft-cap pulls extreme scores toward `±cap`, so the attention distribution over a
    /// peaked key set is flatter than the uncapped one (the Gemma-2 effect).
    #[test]
    fn softcap_flattens_attention() {
        // One query, three keys with very different alignments → uncapped attention is peaky.
        let (b, h, s) = (1, 1, 1);
        let q = Array::from_slice(&[1.0f32, 0.0], &[b, h, s, 2]);
        let k = Array::from_slice(&[10.0f32, 0.0, 0.0, 10.0, 5.0, 5.0], &[b, h, 3, 2]);
        let v = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[b, h, 3, 2]);
        let uncapped = sdpa_capped(&q, &k, &v, 1.0, None, AttnMask::None).unwrap();
        let capped = sdpa_capped(&q, &k, &v, 1.0, Some(2.0), AttnMask::None).unwrap();
        // With a tight cap the output moves toward the mean of the values (a flatter mix).
        let mean = 3.0f32; // mean of v[:,0] = (1+3+5)/3
        let u = uncapped.as_slice::<f32>()[0];
        let c = capped.as_slice::<f32>()[0];
        assert!((c - mean).abs() < (u - mean).abs(), "capped {c} should be nearer mean {mean} than uncapped {u}");
    }
}
