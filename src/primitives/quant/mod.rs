//! Quantization primitives.
//!
//! Two layers live here:
//!
//! * [`linear`] — group-wise affine quantization for *linear projection weights*, built on MLX's
//!   native [`quantize`](mlx_rs::ops::quantize)/[`quantized_matmul`](mlx_rs::ops::quantized_matmul)
//!   (story 7163 / GGUF 7165).
//! * The **foundational KV-cache quant primitives** (sc-8531, epic sc-8528) — the shared,
//!   method-agnostic building blocks every KV-cache compression method (RVQ, KIVI, VecInfer, …)
//!   composes:
//!     - [`scalar_quant`] — per-group affine scalar quant/dequant (scale + zero-point).
//!     - [`bit_packing`] — lossless sub-byte (1/2/4-bit) pack/unpack of integer codes.
//!     - [`codebook`] — Lloyd-Max + adaptive scalar codebooks for non-uniform quantization.
//!
//! These are standalone reusable functions/types, **not** a [`Quantizer`](super::Quantizer)
//! implementation — story D is what consumes them to implement a real KV-cache `Quantizer`.
//!
//! All paths are **pure MLX ops / pure Rust** for correctness. Hot loops carry
//! `TODO(sc-8529/Phase2)` markers where story A's `MetalKernel`s will later replace them for speed
//! (never for correctness).

pub mod bit_packing;
pub mod codebook;
pub mod linear;
pub mod rvq;
pub mod scalar_quant;

pub use bit_packing::{bit_pack, bit_unpack, packed_len};
pub use codebook::{lloyd_max, ScalarCodebook};
pub use linear::QuantizedLinear;
pub use rvq::{estimate_inner_product, RvqBlock, RvqQuantizer};
pub use scalar_quant::{
    affine_dequantize, affine_quantize, dequantize_affine, quantize_affine, AffineParams,
    QuantizedGroups,
};
