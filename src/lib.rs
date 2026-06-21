//! `mlx-llm` — on-device text + vision LLM serving engine for Apple MLX.
//!
//! The crate is built bottom-up, mirroring the dependency order of epic 7153:
//!
//! 1. [`primitives`] — the backend-owned tensor leaves the engine needs: a batch-capable
//!    [`KvCache`](primitives::KvCache), a pluggable [`Sampler`](primitives::sampler), the
//!    [`Rope`](primitives::Rope) family, GQA attention helpers, quantization, the `nn` leaves,
//!    and a safetensors [`Weights`](primitives::Weights) loader. These own MLX `Array`s directly
//!    and deliberately depend on nothing from the gen-ai side.
//! 2. [`config`] + [`models`] — model configuration ([`LlamaConfig`](config::LlamaConfig)) and the
//!    generic Llama decoder ([`LlamaModel`](models::LlamaModel)), `&self` forward + `from_weights`.
//! 3. [`decode`] — the streaming, cancellable decode loop ([`generate`](decode::generate)) that
//!    drives any [`Decode`](decode::Decode) model, emitting a [`StreamEvent`](decode::StreamEvent)
//!    per token. This internal streaming API is what the backend-neutral `core-llm` contract
//!    (story 7154) is extracted from.
//! 4. [`provider`] — implements the backend-neutral [`core_llm::TextLlm`] contract over the engine
//!    and registers it (`mlx-llama`), so consumers stream a generation entirely through `core-llm`.
//!
//! MLX's default Metal device is single-threaded; engine instances hold MLX `Array`s and are
//! therefore neither `Send` nor `Sync`. Drive one engine from one thread (or behind a mutex).

pub mod config;
pub mod decode;
pub mod error;
pub mod models;
pub mod primitives;
pub mod provider;

// Re-export the contract crate so consumers can reach it as `mlx_llm::core_llm::…`.
pub use core_llm;

pub use config::LlamaConfig;
pub use decode::{generate, CancelFlag, FinishReason, GenerationConfig, GenerationOutput, StreamEvent};
pub use error::{Error, Result};
pub use models::LlamaModel;
pub use provider::LlamaProvider;
