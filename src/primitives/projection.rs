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
    /// A dense `[out, in]` weight with an optional `[out]` bias.
    Dense {
        /// The `[out, in]` weight (HF layout).
        weight: Array,
        /// Optional additive `[out]` bias (Qwen2 / GLM-4 attention carry q/k/v bias; Llama / Qwen3 /
        /// Phi-3 do not).
        bias: Option<Array>,
    },
    /// A group-wise quantized weight.
    Quantized(QuantizedLinear),
}

impl Projection {
    /// Load from a dense `[out, in]` weight, quantizing it if `quant` is set.
    pub fn load(weight: Array, quant: Option<QuantSpec>) -> Result<Self> {
        Self::load_with_bias(weight, None, quant)
    }

    /// Load from a dense `[out, in]` weight plus an optional `[out]` bias (Qwen2 / GLM-4 attention
    /// carry q/k/v bias), quantizing the weight if `quant` is set. The bias is always applied dense.
    pub fn load_with_bias(
        weight: Array,
        bias: Option<Array>,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        match quant {
            None => Ok(Projection::Dense { weight, bias }),
            Some(q) => Ok(Projection::Quantized(QuantizedLinear::quantize(
                &weight,
                q.group_size,
                q.bits,
                bias,
            )?)),
        }
    }

    /// Load from **already-quantized** parts stored in a snapshot (the packed `weight`, per-group
    /// `scales`/`biases`) — the read side of the GGUF converter's optional MLX requant. No
    /// quantization happens here; the parts are used as-is.
    pub fn from_quantized(
        weight: Array,
        scales: Array,
        biases: Array,
        spec: QuantSpec,
    ) -> Self {
        Projection::Quantized(QuantizedLinear {
            weight,
            scales,
            biases,
            group_size: spec.group_size,
            bits: spec.bits,
            bias: None,
        })
    }

    /// `x @ weightᵀ (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Projection::Dense { weight, bias } => linear(x, weight, bias.as_ref()),
            Projection::Quantized(q) => q.forward(x),
        }
    }

    /// Whether this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Projection::Quantized(_))
    }
}
