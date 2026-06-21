//! GGUF model ingestion (story 7165): bring a llama.cpp `*.gguf` and convert it to an MLX snapshot
//! the engine loads.
//!
//! The pipeline is three layers:
//! 1. [`reader`] — parse the GGUF container (header, metadata key/values, tensor infos, aligned
//!    data section).
//! 2. [`dequant`] — turn each tensor's GGML quant blocks into dense `f32` (F16/BF16, the legacy
//!    `Q*_0/_1` types, and the `Q2_K…Q6_K` k-quants).
//! 3. [`convert`] — remap llama.cpp tensor names to the transformer layout, rebuild `config.json`
//!    from the metadata, and write `{config.json, model.safetensors}` — optionally re-quantizing
//!    the projections to MLX Q4/Q8.
//!
//! Entry points: [`convert_file`] / [`convert`] do the whole thing; [`GgufFile`] exposes the parse
//! if a caller wants to inspect metadata first. The `convert_gguf` example is the CLI wrapper.

pub mod convert;
pub mod dequant;
pub mod reader;

pub use convert::{convert, convert_file, ConvertOptions, ConvertReport};
pub use dequant::{dequantize, GgmlType};
pub use reader::{GgufFile, MetaValue, TensorInfo};
