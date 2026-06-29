//! Residual vector quantization (RVQ) for the KV cache (sc-8533).
//!
//! A concrete [`Quantizer`](super::kv_cache::Quantizer) the [`QuantizedKvCache`](super::kv_cache::QuantizedKvCache)
//! plugs in to compress per-step keys/values. Story D (sc-8530) landed the pluggable seam with only
//! the pass-through [`IdentityQuantizer`](super::kv_cache::IdentityQuantizer); this is the first
//! *lossy* method, selectable from the backend-neutral contract via
//! [`core_llm::KvCacheQuant`] with [`core_llm::KvCacheQuantMethod::Rvq`].
//!
//! # Method
//! RVQ quantizes the values in **stages**: stage 0 affine-quantizes each `[batch, n_kv_heads, step,
//! head_dim]` block group-wise along the `head_dim` axis to `bits`-bit integer codes (one scale +
//! zero-point per `(batch, head, position)` row); stage 1 quantizes the *residual* (original minus
//! the stage-0 reconstruction) the same way. The dequantized value is the sum of the two stages'
//! reconstructions, so two stages of `bits`-bit codes recover far more precision than a single
//! `bits`-bit pass at the same nominal width — the residual stage corrects the first stage's error.
//!
//! Shape-agnostic by construction: every op preserves the `[batch, n_kv_heads, step, head_dim]`
//! layout (the reduction is along the last axis with `keep_dims`), so the cache shell's
//! [`retain_sequences`](super::kv_cache::Quantizer::retain_sequences) (batch axis 0) and
//! [`truncate`](super::kv_cache::Quantizer::truncate) (sequence axis [`SEQ_AXIS`](super::kv_cache::SEQ_AXIS))
//! are plain `take_axis` slices over the stored codes — no method-specific bookkeeping.

use mlx_rs::ops::{add, clip, divide, max_axis, min_axis, multiply, round, subtract};
use mlx_rs::Array;

use super::kv_cache::{Quantizer, SEQ_AXIS};
use crate::error::{Error, Result};

/// Number of residual stages. Two stages (a coarse pass + a residual-correction pass) is the
/// standard RVQ depth and gives a good accuracy/footprint trade for a KV cache.
const STAGES: usize = 2;

/// Residual-vector-quantization quantizer for the KV cache.
///
/// Stores `bits` (per-stage code width). Built from the contract's [`core_llm::KvCacheQuant`] via
/// [`RvqQuantizer::new`].
#[derive(Debug, Clone, Copy)]
pub struct RvqQuantizer {
    /// Bits per quantized value, per stage (1..=8).
    bits: u8,
}

impl RvqQuantizer {
    /// A new RVQ quantizer at `bits` per-stage code width. `bits` must be in `1..=8`.
    pub fn new(bits: u8) -> Result<Self> {
        if !(1..=8).contains(&bits) {
            return Err(Error::Unsupported(format!(
                "RVQ KV-cache quant: unsupported bit-width {bits} (supported: 1..=8)"
            )));
        }
        Ok(Self { bits })
    }

    /// The number of discrete levels for the configured bit-width (`2^bits - 1`, the max integer
    /// code; e.g. 4 bits ⇒ levels 0..=15).
    fn max_level(self) -> f32 {
        ((1u32 << self.bits) - 1) as f32
    }

    /// Affine-quantize `x` group-wise along the last axis: returns `(codes, scale, zero)` where
    /// `codes` are integer-valued floats in `[0, max_level]`, and `x ≈ codes * scale + zero`. `scale`
    /// and `zero` keep the reduced last axis as length 1 (broadcastable back over `x`).
    fn quantize_stage(self, x: &Array) -> Result<QStage> {
        let last = (x.ndim() as i32) - 1;
        let lo = min_axis(x, last, true)?; // [..., 1]
        let hi = max_axis(x, last, true)?; // [..., 1]
        let levels = Array::from_f32(self.max_level());
        // scale = (hi - lo) / max_level, guarded against a zero range (constant row) so we never
        // divide by zero — a zero scale reconstructs every code to `zero == lo`, which is exact for a
        // constant row.
        let span = subtract(&hi, &lo)?;
        let scale = divide(&span, &levels)?;
        // codes = round(clip((x - lo) / scale_safe, 0, max_level)); scale_safe avoids div-by-zero.
        let one = Array::from_f32(1.0);
        let scale_is_zero = scale.eq(Array::from_f32(0.0))?;
        let scale_safe = mlx_rs::ops::r#where(&scale_is_zero, &one, &scale)?;
        let shifted = subtract(x, &lo)?;
        let raw = divide(&shifted, &scale_safe)?;
        let clipped = clip(&raw, (0.0f32, self.max_level()))?;
        let codes = round(&clipped, None)?;
        Ok(QStage {
            codes,
            scale,
            zero: lo,
        })
    }

    /// Reconstruct `x ≈ codes * scale + zero` from one stage.
    fn dequantize_stage(stage: &QStage) -> Result<Array> {
        let scaled = multiply(&stage.codes, &stage.scale)?;
        Ok(add(&scaled, &stage.zero)?)
    }
}

/// One affine-quantized stage: integer `codes` plus the per-row `scale`/`zero` to reconstruct.
#[derive(Debug, Clone)]
struct QStage {
    codes: Array,
    scale: Array,
    zero: Array,
}

impl QStage {
    /// Keep only the batch rows in `keep` (axis 0) across all three arrays.
    fn retain(&self, keep: &Array) -> Result<Self> {
        Ok(Self {
            codes: self.codes.take_axis(keep, 0)?,
            scale: self.scale.take_axis(keep, 0)?,
            zero: self.zero.take_axis(keep, 0)?,
        })
    }

    /// Keep only the first `len` sequence positions ([`SEQ_AXIS`]) across all three arrays.
    fn truncate(&self, len: i32) -> Result<Self> {
        let idx = Array::from_slice(&(0..len).collect::<Vec<_>>(), &[len]);
        Ok(Self {
            codes: self.codes.take_axis(&idx, SEQ_AXIS)?,
            scale: self.scale.take_axis(&idx, SEQ_AXIS)?,
            zero: self.zero.take_axis(&idx, SEQ_AXIS)?,
        })
    }
}

/// One RVQ-compressed block: the two quantized stages for keys and for values, plus the block's
/// `[batch, n_kv_heads, step, head_dim]` shape (so `seq_len`/`batch_size` need no decode).
#[derive(Debug, Clone)]
pub struct RvqBlock {
    key_stages: Vec<QStage>,
    value_stages: Vec<QStage>,
    batch: i32,
    seq: i32,
}

impl RvqQuantizer {
    /// Quantize one tensor (keys or values) into `STAGES` residual stages.
    fn encode_tensor(self, x: &Array) -> Result<Vec<QStage>> {
        let mut stages = Vec::with_capacity(STAGES);
        let mut residual = x.clone();
        for _ in 0..STAGES {
            let stage = self.quantize_stage(&residual)?;
            let recon = Self::dequantize_stage(&stage)?;
            residual = subtract(&residual, &recon)?;
            stages.push(stage);
        }
        Ok(stages)
    }

    /// Sum every stage's reconstruction back into the dense tensor.
    fn decode_tensor(stages: &[QStage]) -> Result<Array> {
        let mut acc: Option<Array> = None;
        for stage in stages {
            let recon = Self::dequantize_stage(stage)?;
            acc = Some(match acc {
                Some(a) => add(&a, &recon)?,
                None => recon,
            });
        }
        // A block always has at least one stage, so `acc` is always `Some`.
        acc.ok_or_else(|| Error::Msg("RVQ decode_tensor: empty stage list".into()))
    }
}

impl Quantizer for RvqQuantizer {
    type Block = RvqBlock;

    fn encode(&self, keys: &Array, values: &Array) -> Result<Self::Block> {
        let shape = keys.shape();
        Ok(RvqBlock {
            key_stages: self.encode_tensor(keys)?,
            value_stages: self.encode_tensor(values)?,
            batch: shape[0],
            seq: shape[SEQ_AXIS as usize],
        })
    }

    fn decode(&self, block: &Self::Block) -> Result<(Array, Array)> {
        Ok((
            Self::decode_tensor(&block.key_stages)?,
            Self::decode_tensor(&block.value_stages)?,
        ))
    }

    fn seq_len(&self, block: &Self::Block) -> i32 {
        block.seq
    }

    fn batch_size(&self, block: &Self::Block) -> i32 {
        block.batch
    }

    fn retain_sequences(&self, block: &Self::Block, keep: &Array) -> Result<Self::Block> {
        Ok(RvqBlock {
            key_stages: block
                .key_stages
                .iter()
                .map(|s| s.retain(keep))
                .collect::<Result<_>>()?,
            value_stages: block
                .value_stages
                .iter()
                .map(|s| s.retain(keep))
                .collect::<Result<_>>()?,
            batch: keep.size() as i32,
            seq: block.seq,
        })
    }

    fn truncate(&self, block: &Self::Block, len: i32) -> Result<Self::Block> {
        Ok(RvqBlock {
            key_stages: block
                .key_stages
                .iter()
                .map(|s| s.truncate(len))
                .collect::<Result<_>>()?,
            value_stages: block
                .value_stages
                .iter()
                .map(|s| s.truncate(len))
                .collect::<Result<_>>()?,
            batch: block.batch,
            seq: len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::kv_cache::{KvCache, QuantizedKvCache};
    use mlx_rs::Dtype;

    fn arange4(b: i32, h: i32, s: i32, d: i32) -> Array {
        let n = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
        Array::from_slice(&data, &[b, h, s, d])
    }

    fn host(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    fn max_abs_err(a: &Array, b: &Array) -> f32 {
        host(a)
            .iter()
            .zip(host(b).iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(RvqQuantizer::new(0).is_err());
        assert!(RvqQuantizer::new(9).is_err());
        assert!(RvqQuantizer::new(4).is_ok());
        assert!(RvqQuantizer::new(8).is_ok());
    }

    /// Round-trip through the quantizer reconstructs the input within the quantization error, and a
    /// higher bit-width is at least as accurate (the residual stage shrinks the error).
    #[test]
    fn round_trip_is_lossy_but_bounded_and_improves_with_bits() {
        let k = arange4(2, 3, 4, 8);
        let v = multiply(&k, Array::from_f32(-1.0)).unwrap();

        let q4 = RvqQuantizer::new(4).unwrap();
        let b4 = q4.encode(&k, &v).unwrap();
        let (k4, v4) = q4.decode(&b4).unwrap();
        let err4 = max_abs_err(&k, &k4).max(max_abs_err(&v, &v4));

        let q8 = RvqQuantizer::new(8).unwrap();
        let b8 = q8.encode(&k, &v).unwrap();
        let (k8, _) = q8.decode(&b8).unwrap();
        let err8 = max_abs_err(&k, &k8);

        assert_eq!(k4.shape(), k.shape());
        // Range of arange4(2,3,4,8) is ~96 wide; two 4-bit stages keep error well under a coarse
        // single-pass bound.
        assert!(err4 < 1.0, "4-bit RVQ error too large: {err4}");
        assert!(
            err8 <= err4 + 1e-4,
            "more bits should not be worse: {err8} vs {err4}"
        );
    }

    /// A constant row (zero range) round-trips exactly — the zero-scale guard reconstructs it to the
    /// shared value with no NaNs.
    #[test]
    fn constant_row_is_exact() {
        let k = Array::from_slice(&[7.0f32; 16], &[1, 2, 1, 8]);
        let q = RvqQuantizer::new(4).unwrap();
        let block = q.encode(&k, &k).unwrap();
        let (kd, _) = q.decode(&block).unwrap();
        assert_eq!(host(&kd), vec![7.0f32; 16]);
    }

    /// Driven through the `QuantizedKvCache` shell: offset/batch bookkeeping is correct across two
    /// updates, and `truncate` + `retain_sequences` roll back/compact the compressed path.
    #[test]
    fn through_quantized_cache_offset_truncate_retain() {
        let q = RvqQuantizer::new(8).unwrap();
        let mut cache = QuantizedKvCache::new(1, q);
        let a = arange4(2, 2, 3, 8);
        let b = arange4(2, 2, 2, 8);
        cache.update(0, &a, &a).unwrap();
        cache.update(0, &b, &b).unwrap();
        assert_eq!(cache.offset(), 5);
        assert_eq!(cache.batch_size(), 2);

        cache.truncate(4).unwrap(); // cuts inside the second block
        assert_eq!(cache.offset(), 4);

        cache.retain_sequences(&[1]).unwrap(); // keep batch row 1
        assert_eq!(cache.batch_size(), 1);
        assert_eq!(cache.offset(), 4);
        let (k, _) = cache.peek(0).unwrap().unwrap();
        assert_eq!(k.shape(), &[1, 2, 4, 8]);
    }
}
