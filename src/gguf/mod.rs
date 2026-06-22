//! GGUF model ingestion (story 7165): bring a llama.cpp `*.gguf` and convert it to an MLX snapshot
//! the engine loads.
//!
//! The pipeline is three layers:
//! 1. [`reader`] — parse the GGUF container (header, metadata key/values, tensor infos, aligned
//!    data section).
//! 2. [`dequant`] — turn each tensor's GGML quant blocks into dense `f32` (F16/BF16, the legacy
//!    `Q*_0/_1` types, the `Q2_K…Q6_K` k-quants, `IQ4_NL`/`IQ4_XS`, and the sub-4-bit importance-matrix
//!    grid quants `IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`).
//! 3. [`convert`] — remap llama.cpp tensor names to the transformer layout, rebuild `config.json`
//!    from the metadata, reconstruct `tokenizer.json`/`tokenizer_config.json` from the embedded
//!    tokenizer metadata ([`tokenizer`]), and write `{config.json, model.safetensors,
//!    tokenizer.json, tokenizer_config.json}` — optionally re-quantizing the projections to MLX
//!    Q4/Q8.
//!
//! Entry points: [`convert_file`] / [`convert`] do the whole thing; [`GgufFile`] exposes the parse
//! if a caller wants to inspect metadata first. The `convert_gguf` example is the CLI wrapper.

pub mod convert;
pub mod dequant;
mod iq_grids;
pub mod reader;
pub mod tokenizer;

pub use convert::{convert, convert_file, ConvertOptions, ConvertReport, TokenizerStatus};
pub use dequant::{dequantize, GgmlType};
pub use reader::{GgufFile, MetaValue, TensorInfo};
