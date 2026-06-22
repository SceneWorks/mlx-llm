//! Attention leaves: grouped-query KV expansion and a scaled-dot-product-attention wrapper.
//!
//! The decoders run GQA — fewer KV heads than query heads — so cached K/V must be expanded to the
//! query head count before attention. [`repeat_kv`] is the `[b, hkv, s, hd] -> [b, hkv*groups, s, hd]`
//! broadcast the mlx-gen stacks use. [`sdpa`] wraps MLX's fused `scaled_dot_product_attention`,
//! exposing the two masking modes the references need: implicit bottom-right [`AttnMask::Causal`]
//! (decode) and an explicit [`AttnMask::Additive`] mask (the block-causal / bidirectional paths).

use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::broadcast_to;
use mlx_rs::Array;

use crate::error::Result;

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

/// Fused scaled-dot-product attention over `[batch, heads, seq, head_dim]` tensors.
///
/// `scale` is the usual `head_dim^(-0.5)`. Grouped-query attention is handled **natively** by MLX —
/// pass K/V with `n_kv_heads` (fewer than the query heads) and the fused kernel derives
/// `gqa_factor = q_heads / kv_heads`, reading K/V by head stride with no head-count materialization.
/// (Pre-expanding with [`repeat_kv`] is equivalent but allocates a `groups`× larger K/V — avoid it
/// on the hot path; see sc-7307.)
pub fn sdpa(
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
}
