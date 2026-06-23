//! Backend-owned tensor primitives (epic 7153, story 7155).
//!
//! These are the decode leaves `mlx-llm` owns because it must not pull them from the gen-ai core:
//! a batch-capable KV cache, the sampler, the RoPE family, GQA attention helpers, group-wise
//! quantization, the `nn` leaves (linear / norms / activations / embedding), and a safetensors
//! weights loader. They are modelled faithfully on the working mlx-gen implementations
//! (prompt-refine's Llama decoder, JoyCaption's VLM, sensenova/flux2's Qwen3 stacks) so that the
//! generating stacks can later adopt them at parity — the accepted small leaf-duplication that
//! buys this crate its independence. The Candle backend (`candle-llm`) reimplements the
//! equivalents on candle.
//!
//! Shapes are **batch-capable from day one**: the batch axis is a real dimension everywhere, even
//! though the first decoders run batch-1. The [`KvCache`] trait is the seam the P4 paged cache
//! (story 7169) slots in behind without touching decoders.

pub mod attention;
pub mod gated_delta;
pub mod kv_cache;
pub mod nn;
pub mod paged_kv_cache;
pub mod projection;
pub mod quant;
pub mod rope;
pub mod sampler;
pub mod weights;

pub use attention::{repeat_kv, sdpa, sdpa_capped, sdpa_causal, AttnMask};
pub use gated_delta::{
    causal_depthwise_conv, compute_g, gated_delta_recurrence, rms_norm_gated, DeltaNetCache,
};
pub use kv_cache::{ContiguousKvCache, KvCache};
pub use nn::{conv2d, embed, input_ids, input_ids_batch, layer_norm, linear, rms_norm, soft_cap};
pub use paged_kv_cache::{BlockPool, PagedKvCache};
pub use projection::{Projection, QuantSpec};
pub use quant::QuantizedLinear;
pub use rope::{apply_rope, Rope};
pub use sampler::{sample, shaped_candidates, SamplingParams, SplitMix64, TokenRng};
pub use weights::Weights;
