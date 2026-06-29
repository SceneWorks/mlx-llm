//! KV-cache quantization benchmark — one command that emits a method × {KV-memory, quality-delta,
//! tok/s} comparison table at multiple context lengths (story sc-8534, epic sc-8528).
//!
//! ```text
//! cargo run --release --example kv_quant_bench
//! ```
//!
//! The default sweep is the bounded, memory-safe [`BenchConfig::small`] one (4 layers, batch 1, 4 KV
//! heads, head_dim 64, contexts {64, 128}). Optional CLI args widen it (opt-in only — larger sweeps
//! can stress machine memory; the harness still installs an MLX memory limit + buffer-cache cap and
//! clears the cache between runs):
//!
//! ```text
//! cargo run --release --example kv_quant_bench -- --contexts 128,256,512 --layers 8 --kv-heads 8
//! ```
//!
//! The harness is **method-agnostic**: it takes any list of [`Method`]s (a name + a
//! `Fn(num_layers) -> Box<dyn KvCache>` builder) with the dense baseline first. Today it registers
//! `dense` ([`ContiguousKvCache`]) and `identity` (`QuantizedKvCache<IdentityQuantizer>`); registering
//! RVQ (story D) is one more `Method::new(...)` line — see [`mlx_llm::primitives::kv_bench`].

use mlx_llm::primitives::kv_bench::{default_methods, format_table, run_bench, BenchConfig};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut cfg = BenchConfig::small();

    // Minimal flag parsing — every flag is optional and keeps the default safe sweep unless given.
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--contexts" => {
                let val = args.next().ok_or("--contexts needs a value, e.g. 64,128")?;
                cfg.context_lengths = val
                    .split(',')
                    .map(|s| s.trim().parse::<i32>())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--layers" => {
                cfg.num_layers = args.next().ok_or("--layers needs a value")?.parse()?;
            }
            "--kv-heads" => {
                cfg.n_kv_heads = args.next().ok_or("--kv-heads needs a value")?.parse()?;
            }
            "--head-dim" => {
                cfg.head_dim = args.next().ok_or("--head-dim needs a value")?.parse()?;
            }
            "--prefill" => {
                cfg.prefill_tokens = args.next().ok_or("--prefill needs a value")?.parse()?;
            }
            "--batch" => {
                cfg.batch = args.next().ok_or("--batch needs a value")?.parse()?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: kv_quant_bench [--contexts a,b,..] [--layers N] [--kv-heads N] \
                     [--head-dim N] [--prefill N] [--batch N]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown flag {other:?}").into()),
        }
    }

    if cfg.context_lengths.len() < 2 {
        return Err("need >= 2 context lengths (acceptance criterion)".into());
    }

    eprintln!(
        "running bounded KV-quant bench: contexts={:?} layers={} kv_heads={} head_dim={}",
        cfg.context_lengths, cfg.num_layers, cfg.n_kv_heads, cfg.head_dim
    );

    let result = run_bench(&cfg, &default_methods())?;
    print!("{}", format_table(&result));
    Ok(())
}
