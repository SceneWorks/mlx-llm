//! Concrete model decoders built on the [`crate::primitives`].
//!
//! Story 7156 ships the generic Llama decoder. Each decoder uses an **immutable `&self` forward**
//! and a `from_weights` constructor — deliberately *not* mlx-rs's `&mut self` `Module` trait, so a
//! single loaded model can be shared and driven concurrently in the batch dimension later. The only
//! mutable state in a forward pass is the KV cache, threaded in as `&mut dyn KvCache`.

pub mod llama;
pub mod qwen35;
pub mod qwen35_vision;
pub mod siglip;

pub use llama::CausalLm;
pub use qwen35::{Qwen35Cache, Qwen35Config, Qwen35Model};
pub use qwen35_vision::{Qwen35VisionConfig, Qwen35VisionModel};
pub use siglip::{SiglipVisionConfig, SiglipVisionTower};
