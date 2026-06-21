//! A linear projection that is either dense or group-wise quantized.
//!
//! The decoders hold their attention/MLP projections behind this so quantize-on-load (story 7163)
//! is a load-time choice with no decoder changes: a dense `[out, in]` weight either stays dense
//! (`matmul(x, wᵀ)`) or is quantized to Q4/Q8 ([`QuantizedLinear`]).

use mlx_rs::Array;

use crate::error::Result;
use crate::primitives::nn::linear;
use crate::primitives::quant::QuantizedLinear;

/// Group-wise affine quantization parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    /// Elements per quantization group.
    pub group_size: i32,
    /// Bits per weight (4 or 8).
    pub bits: i32,
}

impl QuantSpec {
    /// 4-bit, group size 64.
    pub fn q4() -> Self {
        Self { group_size: 64, bits: 4 }
    }

    /// 8-bit, group size 64.
    pub fn q8() -> Self {
        Self { group_size: 64, bits: 8 }
    }
}

/// A linear projection weight, dense or quantized.
#[derive(Debug)]
pub enum Projection {
    /// A dense `[out, in]` weight.
    Dense(Array),
    /// A group-wise quantized weight.
    Quantized(QuantizedLinear),
}

impl Projection {
    /// Load from a dense `[out, in]` weight, quantizing it if `quant` is set.
    pub fn load(weight: Array, quant: Option<QuantSpec>) -> Result<Self> {
        match quant {
            None => Ok(Projection::Dense(weight)),
            Some(q) => Ok(Projection::Quantized(QuantizedLinear::quantize(
                &weight,
                q.group_size,
                q.bits,
                None,
            )?)),
        }
    }

    /// `x @ weightᵀ`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Projection::Dense(w) => linear(x, w, None),
            Projection::Quantized(q) => q.forward(x),
        }
    }

    /// Whether this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Projection::Quantized(_))
    }
}
