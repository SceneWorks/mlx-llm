//! Convert a llama.cpp `*.gguf` model into an MLX snapshot the engine loads.
//!
//! ```text
//! cargo run --release --example convert_gguf -- <model.gguf> <out_dir> [--quant q4|q8] [--tokenizer tokenizer.json]
//! ```
//!
//! Writes `<out_dir>/config.json`, `<out_dir>/model.safetensors`, and — for the byte-level BPE
//! tokenizer family — `<out_dir>/tokenizer.json` + `<out_dir>/tokenizer_config.json` reconstructed
//! from the GGUF metadata, so the snapshot runs end-to-end with no external files. The weights are
//! dequantized from the GGUF (F16/BF16, legacy `Q*_0/_1`, the `Q2_K…Q6_K` k-quants, `IQ4_NL`/`IQ4_XS`,
//! and the sub-4-bit importance-matrix grid quants `IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/
//! `IQ3_XXS`/`IQ3_S`) and remapped to the transformer key layout. `--quant q4|q8` re-quantizes the
//! attention/MLP projections to MLX group-wise quantization (embeddings, LM head, and norms stay
//! dense).
//!
//! Byte-level BPE (SmolLM2/Qwen/Llama-3) and SentencePiece BPE (Llama-2/Mistral) tokenizers are
//! reconstructed. If the GGUF uses a kind that can't be rebuilt from its metadata (Unigram/T5-style
//! SentencePiece, whose normalizer charsmap isn't stored), the converter reports that and writes no
//! `tokenizer.json`. Pass `--tokenizer <path>` to drop in a `tokenizer.json` from the source model's
//! HF repo (it also overrides a reconstructed one).

use std::path::Path;

use mlx_llm::gguf::{convert_file, ConvertOptions, TokenizerStatus};
use mlx_llm::primitives::QuantSpec;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let gguf = args.next().ok_or(
        "usage: convert_gguf <model.gguf> <out_dir> [--quant q4|q8] [--tokenizer tokenizer.json]",
    )?;
    let out_dir = args.next().ok_or("missing <out_dir> argument")?;

    let mut quantize: Option<QuantSpec> = None;
    let mut tokenizer: Option<String> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--quant" => {
                let v = args.next().ok_or("--quant needs a value (q4|q8)")?;
                quantize = Some(match v.as_str() {
                    "q4" => QuantSpec::q4(),
                    "q8" => QuantSpec::q8(),
                    other => return Err(format!("unknown --quant {other:?} (expected q4|q8)").into()),
                });
            }
            "--tokenizer" => tokenizer = Some(args.next().ok_or("--tokenizer needs a path")?),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let report = convert_file(&gguf, &out_dir, ConvertOptions { quantize })?;

    match &report.tokenizer {
        TokenizerStatus::Reconstructed(kind) => println!("tokenizer: reconstructed {kind}"),
        TokenizerStatus::Unsupported(reason) => println!("tokenizer: not reconstructed — {reason}"),
        TokenizerStatus::Absent => println!("tokenizer: no tokenizer metadata in GGUF"),
    }

    if let Some(tok) = tokenizer {
        let dst = Path::new(&out_dir).join("tokenizer.json");
        std::fs::copy(&tok, &dst)?;
        println!("copied tokenizer (override) -> {}", dst.display());
    }

    println!(
        "converted {} ({}) -> {}: {} tensors{}",
        gguf,
        report.architecture,
        report.out_dir.display(),
        report.num_tensors,
        match report.quantized {
            Some(q) => format!(", projections quantized to {}-bit (group {})", q.bits, q.group_size),
            None => ", dense bf16".into(),
        },
    );
    Ok(())
}
