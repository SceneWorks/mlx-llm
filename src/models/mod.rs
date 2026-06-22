//! Concrete model decoders built on the [`crate::primitives`].
//!
//! Story 7156 ships the generic Llama decoder. Each decoder uses an **immutable `&self` forward**
//! and a `from_weights` constructor — deliberately *not* mlx-rs's `&mut self` `Module` trait, so a
//! single loaded model can be shared and driven concurrently in the batch dimension later. The only
//! mutable state in a forward pass is the KV cache, threaded in as `&mut dyn KvCache`.

pub mod llama;
pub mod siglip;

pub use llama::CausalLm;
pub use siglip::{SiglipVisionConfig, SiglipVisionTower};
