//! Stream a completion from a local Llama snapshot.
//!
//! ```text
//! cargo run --release --example generate -- <model_dir> "Your prompt here" [max_new_tokens]
//! ```
//!
//! `<model_dir>` is a Hugging Face snapshot directory containing `config.json`, `tokenizer.json`,
//! and the `*.safetensors` shards. Tokens are detokenized and printed incrementally as they stream.
//!
//! This is the thin end-to-end vertical for story 7156. Prompt chat-templating and a first-class
//! host tokenizer land in `core-llm` (stories 7164 / 7154); this example tokenizes the raw prompt
//! directly via the HF `tokenizers` crate to keep the demonstration self-contained.

use std::io::Write;
use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_llm::config::LlamaConfig;
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig, StreamEvent};
use mlx_llm::models::LlamaModel;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;

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
    let cfg = LlamaConfig::from_dir(dir)?;
    let stop_tokens = read_eos_ids(dir);
    let weights = Weights::from_dir(dir)?;
    let model = LlamaModel::from_weights(&weights, "", cfg)?;

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let prompt_ids: Vec<i32> = tokenizer
        .encode(prompt.as_str(), true)?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
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

/// Read `eos_token_id` (int or array) from `config.json`; falls back to the Llama-3 stop ids.
fn read_eos_ids(dir: &Path) -> Vec<i32> {
    let fallback = vec![128001, 128008, 128009]; // <|end_of_text|>, <|eom_id|>, <|eot_id|>
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return fallback;
    };
    let Ok(v): Result<Value, _> = serde_json::from_str(&text) else {
        return fallback;
    };
    match v.get("eos_token_id") {
        Some(Value::Number(n)) => n.as_i64().map(|x| vec![x as i32]).unwrap_or(fallback),
        Some(Value::Array(a)) => {
            let ids: Vec<i32> = a.iter().filter_map(|x| x.as_i64().map(|x| x as i32)).collect();
            if ids.is_empty() {
                fallback
            } else {
                ids
            }
        }
        _ => fallback,
    }
}
