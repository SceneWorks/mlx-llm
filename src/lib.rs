//! `mlx-llm` — on-device text + vision LLM serving engine for Apple MLX.
//!
//! The crate is built bottom-up, mirroring the dependency order of epic 7153:
//!
//! 1. [`primitives`] — the backend-owned tensor leaves the engine needs: a batch-capable
//!    [`KvCache`](primitives::KvCache), a pluggable [`Sampler`](primitives::sampler), the
//!    [`Rope`](primitives::Rope) family, GQA attention helpers, quantization, the `nn` leaves,
//!    and a safetensors [`Weights`](primitives::Weights) loader. These own MLX `Array`s directly
//!    and deliberately depend on nothing from the gen-ai side.
//! 2. [`config`] + [`models`] — model configuration ([`ModelConfig`](config::ModelConfig)) and the
//!    generic Llama decoder ([`CausalLm`](models::CausalLm)), `&self` forward + `from_weights`.
//! 3. [`decode`] — the streaming, cancellable decode loop ([`generate`](decode::generate)) that
//!    drives any [`Decode`](decode::Decode) model, emitting a [`StreamEvent`](decode::StreamEvent)
//!    per token. This internal streaming API is what the backend-neutral `core-llm` contract
//!    (story 7154) is extracted from. [`generate_batch`](decode::generate_batch) packs many requests
//!    into one forward step (story 7167), driven by `core_llm`'s backend-neutral scheduler policy;
//!    [`generate_cached`](decode::generate_cached) reuses a shared prompt prefix's KV across requests
//!    (story 7168), driven by `core_llm`'s backend-neutral prefix-index policy;
//!    [`generate_prompt_lookup`](decode::generate_prompt_lookup) is n-gram speculative decoding
//!    (story 7171) and [`generate_draft_speculative`](decode::generate_draft_speculative) is
//!    draft-model speculative decoding (story 7172), both driven by `core_llm`'s backend-neutral
//!    proposer + distribution-preserving acceptance sampler.
//! 4. [`provider`] — implements the backend-neutral [`core_llm::TextLlm`] contract over the engine
//!    and registers it (`mlx-llama`), so consumers stream a generation entirely through `core-llm`.
//!
//! [`gguf`] is a side door into step 2: it converts a llama.cpp `*.gguf` (incl. k-quants) into the
//! `{config.json, model.safetensors}` snapshot the loader consumes (story 7165), optionally
//! re-quantizing to MLX Q4/Q8.
//!
//! [`snapshot`] is the shared persisted-snapshot writer both ingest paths funnel through
//! ([`snapshot::write_snapshot`], story 7660): it (optionally) re-quantizes the attention/MLP
//! projections and writes the loadable snapshot directory. [`snapshot::write_hf_snapshot`] is the
//! Hugging Face leaf — load a dense HF safetensors directory and persist it as a snapshot, dense or
//! Q4/Q8 (the backend half of `core_llm::SnapshotPreparer`'s HF case).
//!
//! [`joycaption`] is the **text + vision** path (story 7157): a SigLIP vision tower
//! ([`models::SiglipVisionTower`]) + LLaVA projector + image splice in front of the reused
//! [`CausalLm`] decode, served as a multimodal [`core_llm::TextLlm`] provider (`mlx-joycaption`).
//!
//! MLX's default Metal device is single-threaded; engine instances hold MLX `Array`s and are
//! therefore neither `Send` nor `Sync`. Drive one engine from one thread (or behind a mutex).

pub mod config;
pub mod decode;
pub mod error;
pub mod gguf;
pub mod image;
pub mod joycaption;
pub mod models;
pub mod prepare;
pub mod primitives;
pub mod provider;
pub mod snapshot;

// Re-export the contract crate so consumers can reach it as `mlx_llm::core_llm::…`.
pub use core_llm;

pub use config::ModelConfig;
pub use decode::{
    generate, generate_batch, generate_cached, generate_continuous, generate_draft_speculative,
    generate_prompt_lookup, generate_with_cache, BatchExactness, BatchRequest, CancelFlag,
    ContinuousConfig, FinishReason, GenerationConfig, GenerationOutput, PrefixCache, PrefixStats,
    SpeculativeConfig, SpeculativeStats, StreamEvent,
};
pub use error::{Error, Result};
pub use joycaption::{JoyCaptionModel, JoyCaptionProvider};
pub use models::CausalLm;
pub use provider::LlamaProvider;
pub use snapshot::{write_hf_snapshot, write_snapshot, SnapshotReport, SnapshotTokenizer};
