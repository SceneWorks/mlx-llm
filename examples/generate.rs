//! Stream a completion from a local Llama snapshot.
//!
//! ```text
//! cargo run --release --example generate -- <model_dir> "Your prompt here" [max_new_tokens]
//! ```
//!
//! `<model_dir>` is a Hugging Face snapshot directory containing `config.json`, `tokenizer.json`,
//! and the `*.safetensors` shards. Tokens are detokenized and printed incrementally as they stream.
//!
//! This demonstrates the low-level streaming loop directly (raw completion, no chat template) using
//! the shared `core_llm::Tokenizer`. For the full contract path (chat template + `TextLlm`) see the
//! `LlamaProvider` in `tests/contract_roundtrip.rs`.

use std::io::Write;
use std::path::Path;

use core_llm::Tokenizer;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig, StreamEvent};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::eos_token_ids;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: generate <model_dir> <prompt> [max_new_tokens]")?;
    let prompt = args.next().ok_or("missing <prompt> argument")?;
    let max_new_tokens: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(128);

    let dir = Path::new(&dir);
    eprintln!("loading config + weights from {} …", dir.display());
    let cfg = ModelConfig::from_dir(dir)?;
    let stop_tokens = eos_token_ids(dir);
    let weights = Weights::from_dir(dir)?;
    let model = CausalLm::from_weights(&weights, "", cfg)?;

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let prompt_ids: Vec<i32> = tokenizer
        .encode(&prompt, true)?
        .into_iter()
        .map(|id| id as i32)
        .collect();
    eprintln!("prompt: {} tokens; streaming up to {max_new_tokens} …\n", prompt_ids.len());

    let config = GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            ..Default::default()
        },
        seed: None,
        stop_tokens,
    };

    // Incremental detokenization: re-decode the running sequence and print the new suffix.
    let mut acc: Vec<u32> = Vec::new();
    let mut shown = 0usize;
    let out = generate(&model, &prompt_ids, &config, &CancelFlag::new(), &mut |event| {
        if let StreamEvent::Token { id, .. } = event {
            acc.push(id as u32);
            if let Ok(full) = tokenizer.decode(&acc, true) {
                if full.len() > shown {
                    print!("{}", &full[shown..]);
                    let _ = std::io::stdout().flush();
                    shown = full.len();
                }
            }
        }
    })?;

    println!(
        "\n\n[{} tokens, finish: {:?}]",
        out.tokens.len(),
        out.finish_reason
    );
    Ok(())
}
