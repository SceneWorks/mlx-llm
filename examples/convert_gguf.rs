//! Convert a llama.cpp `*.gguf` model into an MLX snapshot the engine loads.
//!
//! ```text
//! cargo run --release --example convert_gguf -- <model.gguf> <out_dir> [--quant q4|q8] [--tokenizer tokenizer.json]
//! ```
//!
//! Writes `<out_dir>/config.json` and `<out_dir>/model.safetensors`. The weights are dequantized
//! from the GGUF (F16/BF16, legacy `Q*_0/_1`, and the `Q2_K…Q6_K` k-quants) and remapped to the
//! transformer key layout. `--quant q4|q8` re-quantizes the attention/MLP projections to MLX
//! group-wise quantization (embeddings, LM head, and norms stay dense).
//!
//! A GGUF carries a llama.cpp tokenizer in its metadata, not a HF `tokenizer.json`, so the
//! tokenizer is not reconstructed here. Pass `--tokenizer <path>` to copy a `tokenizer.json` (from
//! the source model's HF repo) into the output directory so the snapshot runs end-to-end; without
//! it the snapshot is weights + config only.

use std::path::Path;

use mlx_llm::gguf::{convert_file, ConvertOptions};
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

    if let Some(tok) = tokenizer {
        let dst = Path::new(&out_dir).join("tokenizer.json");
        std::fs::copy(&tok, &dst)?;
        println!("copied tokenizer -> {}", dst.display());
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
