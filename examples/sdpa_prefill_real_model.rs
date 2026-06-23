//! sc-7430 real-model context demo: shows that generation stays FLUENT on prompts over 8 tokens
//! even though the fused-SDPA prefill kernel (q_len > 8, multi-head, head_dim ∈ {64,128}) is
//! numerically broken on the pinned pmetal fork — which is exactly why the bug went unnoticed and
//! why short-prompt smoke tests passed.
//!
//! Point `MLX_LLM_TEST_MODEL` at a HF snapshot (config.json + tokenizer.json + *.safetensors), e.g.
//! SmolLM2-135M (hidden 576 / 9 heads / head_dim 64 — in the broken envelope), and run:
//!   MLX_LLM_TEST_MODEL=/path/to/smollm2-135m cargo run --release --example sdpa_prefill_real_model
//!
//! The numerical bug itself is proven by `examples/sdpa_f32_repro.rs` (mlx-rs only) and the
//! `sc7430_*` unit tripwire in `src/primitives/attention.rs`. The end-to-end IMPACT (the corrupted
//! prefill changing the greedy continuation on some >8-token prompts) was confirmed with a temporary
//! fused-vs-eager toggle in `sdpa` (see sc-7430); generations here look fine because the corruption
//! stays fluent and the last-position argmax is usually robust.

use core_llm::Tokenizer;
use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;

fn main() {
    let Ok(dir) = std::env::var("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL to a HF snapshot dir (config.json + tokenizer.json + *.safetensors)");
        return;
    };
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let weights = Weights::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&weights, "", cfg).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();

    let gen = |prompt: &str| {
        let ids: Vec<i32> = tok
            .encode(prompt, true)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let n = ids.len();
        let config = GenerationConfig {
            max_new_tokens: 30,
            sampling: SamplingParams::default(), // greedy
            seed: Some(0),
            stop_tokens: vec![2], // SmolLM2 EOS
        };
        let out = generate(&model, &ids, &config, &CancelFlag::new(), &mut |_| {}).unwrap();
        let text = tok
            .decode(&out.tokens.iter().map(|&i| i as u32).collect::<Vec<_>>(), true)
            .unwrap();
        (n, text)
    };

    // A short (<=8 token) prompt that uses the correct kernel, plus several LONG (>8 token) prompts
    // whose prefill runs through the broken steel kernel — yet all generate fluent, plausible text.
    let prompts = [
        "The capital of France is",
        "The capital of France is Paris. The capital of Germany is Berlin. \
         The capital of Italy is Rome. The capital of Spain is",
        "Once upon a time, in a small village nestled between two tall mountains, there lived a young",
        "Here is a short list of three healthy breakfast ideas that are quick to prepare:",
        "Question: If a train travels 60 miles in 1.5 hours, what is its average speed? Answer:",
        "The most important thing to remember when learning to play the guitar for the first time is",
    ];

    println!("\n######## greedy generation (fused steel kernel; all outputs are fluent) ########");
    for p in prompts {
        let (n, t) = gen(p);
        println!("\n[{n} tok] prompt: {p:?}");
        println!("output: {t:?}");
    }
}
