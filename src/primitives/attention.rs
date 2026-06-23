//! Attention leaves: grouped-query KV expansion and a scaled-dot-product-attention wrapper.
//!
//! The decoders run GQA — fewer KV heads than query heads — so cached K/V must be expanded to the
//! query head count before attention. [`repeat_kv`] is the `[b, hkv, s, hd] -> [b, hkv*groups, s, hd]`
//! broadcast the mlx-gen stacks use. [`sdpa`] wraps MLX's fused `scaled_dot_product_attention`,
//! exposing the two masking modes the references need: implicit bottom-right [`AttnMask::Causal`]
//! (decode) and an explicit [`AttnMask::Additive`] mask (the block-causal / bidirectional paths).

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{broadcast_to, concatenate_axis};
use mlx_rs::Array;

use crate::error::Result;

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
///   `[·, ·, 8, k_len]` — no full `[·, ·, q_len, k_len]` materialization.
/// - **None**: every query attends all keys — pass the full K/V.
/// - **Additive**: the mask already encodes visibility, so pass full K/V and slice the mask's query
///   axis to `[c0, c1)` (when that axis isn't broadcast).
///
/// Each chunk output is `eval`'d before the next so the per-chunk K/V gathers don't accumulate across
/// the prefill/layer graph into an out-of-memory (the very failure mode `mlx-benchmarks` warns of).
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
        let q_chunk = queries.take_axis(range_index(c0, c1), 2)?;
        let out = match mask {
            AttnMask::None => sdpa_fused(&q_chunk, keys, values, scale, AttnMask::None)?,
            AttnMask::Causal => {
                let key_prefix = range_index(0, offset + c1);
                let k_chunk = keys.take_axis(&key_prefix, 2)?;
                let v_chunk = values.take_axis(&key_prefix, 2)?;
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
        out.eval()?; // bound memory: don't let chunk K/V gathers pile up in the lazy graph
        outs.push(out);
        c0 = c1;
    }
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(concatenate_axis(&refs, 2)?)
}

/// Convenience: causal attention (the decode default).
pub fn sdpa_causal(queries: &Array, keys: &Array, values: &Array, scale: f32) -> Result<Array> {
    sdpa(queries, keys, values, scale, AttnMask::Causal)
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
    /// (kv expanded to `h`), f64 accumulation — the ground truth the fused kernel is gated against.
    /// Causal aligns the `ql` queries to the bottom-right of the `kl` keys (offset = kl − ql).
    fn host_attn(q: &Array, k: &Array, v: &Array, groups: i32, scale: f32, causal: bool) -> Vec<f32> {
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
                let host = host_attn(&q, &k, &v, groups, scale, causal);
                let e = rel_err(out.as_slice::<f32>(), &host);
                assert!(e < 2e-3, "fused sdpa wrong on a SAFE shape {nh}/{nkv}/{ql}/{kl}/{hd} causal={causal}: rel={e}");
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
            let host = host_attn(&q, &k, &v, 1, scale, true);
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
                let host = host_attn(&q, &k, &v, groups, scale, causal);
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
        let host = host_attn(&q, &k, &v, 1, scale, true);
        let e = rel_err(out.as_slice::<f32>(), &host);
        assert!(e < 2e-3, "chunked additive-mask sdpa wrong: rel={e}");
    }
}
