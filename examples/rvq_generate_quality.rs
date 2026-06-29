//! End-to-end RVQ-vs-dense generation quality (story sc-8532, epic sc-8528, acceptance criterion a).
//!
//! Runs on a small Llama-architecture MLX model (e.g. SmolLM2-135M) with two KV caches over a fixed
//! prompt set and reports two complementary signals of how well the RVQ KV cache preserves the model's
//! behaviour against the dense baseline:
//!
//! * dense  : [`ContiguousKvCache`] (the baseline)
//! * rvq-bN : `QuantizedKvCache<RvqQuantizer>` with a 1-token attention-sink kept dense
//!
//! ## Metric 1 — teacher-forced next-token agreement (the GATED acceptance signal)
//!
//! Both caches are driven down the **same fixed token trajectory** (the dense run's greedy tokens). At
//! every decode position we compare the RVQ cache's argmax against dense's *given identical history*.
//! This is the right end-to-end quality signal: it isolates "does the RVQ-compressed KV preserve the
//! model's next-token decision" from trajectory divergence. (Free-running greedy — metric 2 — cascades:
//! the instant the two runs pick a different token their *contexts* diverge and every later token
//! differs by construction, so free-running agreement collapses even when each step is individually
//! faithful. We report it as a diagnostic but do **not** gate on it.)
//!
//! ## Metric 2 — free-running greedy agreement + per-token perplexity (reported, not gated)
//!
//! Each cache greedily generates independently; we report the (cascade-prone) exact-match agreement and
//! the mean per-token perplexity of each run, so fluency is quantified beyond exact match.
//!
//! ```text
//! cargo run --release --example rvq_generate_quality -- <model_dir> [max_new_tokens] [bits]
//! ```
//!
//! `<model_dir>` is a Hugging Face snapshot dir (config.json + tokenizer.json + *.safetensors).
//! Agreed quality threshold (documented in the PR): RVQ at b=2 must reach **≥ 0.80 teacher-forced
//! token-agreement** with dense over the prompt set; b=1 is reported but not gated. The harness prints
//! PASS/FAIL on the teacher-forced metric.

use std::path::Path;

use core_llm::Tokenizer;
use mlx_rs::Dtype;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::Decode;
use mlx_llm::error::Result;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::input_ids;
use mlx_llm::primitives::{
    ContiguousKvCache, KvCache, QuantizedKvCache, RvqQuantizer, SinkConfig, Weights,
};

/// The fixed prompt set (criterion: "a fixed prompt set").
const PROMPTS: &[&str] = &[
    "The capital of France is",
    "Water boils at a temperature of",
    "In a shocking turn of events, the scientists discovered that",
    "Once upon a time, in a land far away, there lived",
    "The three primary colors are red, blue, and",
    "To make a cup of tea, first you need to",
];

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// One run's outcome over a single prompt: the greedy token ids and their per-token logprobs.
struct RunOut {
    tokens: Vec<i32>,
    logprobs: Vec<f32>,
}

/// Greedy-decode `max_new` tokens from `prompt_ids` through `cache`, recording each token's logprob
/// under the model's own distribution (for perplexity). Pure greedy: argmax each step.
fn greedy_run(
    model: &CausalLm,
    prompt_ids: &[i32],
    cache: &mut dyn KvCache,
    max_new: usize,
) -> Result<RunOut> {
    let prompt = input_ids(prompt_ids);
    let mut logits = model.step(&prompt, cache, 0)?; // [1, vocab]
    let mut tokens = Vec::with_capacity(max_new);
    let mut logprobs = Vec::with_capacity(max_new);

    for _ in 0..max_new {
        // Pull the [1, vocab] logits row to host f32 and do greedy argmax + logprob there.
        let row: Vec<f32> = logits
            .as_dtype(Dtype::Float32)?
            .reshape(&[logits.size() as i32])?
            .as_slice::<f32>()
            .to_vec();
        let (next, &max_logit) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, v)| (i as i32, v))
            .unwrap();
        // logprob = logit - logsumexp(row) computed stably around max_logit.
        let lse = max_logit + row.iter().map(|&l| (l - max_logit).exp()).sum::<f32>().ln();
        logprobs.push(max_logit - lse);
        tokens.push(next);

        let offset = cache.offset();
        let step_in = input_ids(&[next]);
        logits = model.step(&step_in, cache, offset)?;
    }
    Ok(RunOut { tokens, logprobs })
}

/// Argmax token id of a `[1, vocab]` logits row, read to host f32.
fn argmax_row(logits: &mlx_rs::Array) -> Result<i32> {
    let row: Vec<f32> = logits
        .as_dtype(Dtype::Float32)?
        .reshape(&[logits.size() as i32])?
        .as_slice::<f32>()
        .to_vec();
    Ok(row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap())
}

/// Teacher-forced next-token agreement: drive **both** caches down the same fixed `trajectory` (the
/// dense run's tokens) and count the positions where the RVQ cache's argmax equals the dense cache's
/// argmax *given identical history*. Returns `(matched, total)`.
///
/// Both caches are seeded with the prompt, then fed `trajectory[0..len-1]` one token at a time; at each
/// step the two caches see the **same** input token (so their histories never diverge), and we compare
/// the next-token argmax each produces. This is the non-cascading quality signal acceptance criterion
/// (a) is gated on — unlike free-running greedy, a single disagreement does not poison later positions.
fn teacher_forced_agreement(
    model: &CausalLm,
    prompt_ids: &[i32],
    trajectory: &[i32],
    quant: &RvqQuantizer,
    num_layers: usize,
    sink_tokens: i32,
) -> Result<(usize, usize)> {
    let mut dense = ContiguousKvCache::new(num_layers);
    let mut rvq = QuantizedKvCache::with_sink(
        num_layers,
        quant.clone(),
        SinkConfig::keep_first(sink_tokens),
    );

    // Seed both caches with the prompt; compare the first next-token decision too.
    let prompt = input_ids(prompt_ids);
    let mut dense_logits = model.step(&prompt, &mut dense, 0)?;
    let mut rvq_logits = model.step(&prompt, &mut rvq, 0)?;

    let mut matched = 0usize;
    let mut total = 0usize;
    for &forced in trajectory {
        // Compare the two caches' next-token argmax for the identical current history.
        if argmax_row(&dense_logits)? == argmax_row(&rvq_logits)? {
            matched += 1;
        }
        total += 1;

        // Advance both caches with the SAME forced token (the dense trajectory) so histories stay
        // aligned — this is what makes the metric non-cascading.
        let step_in = input_ids(&[forced]);
        let d_off = dense.offset();
        let r_off = rvq.offset();
        dense_logits = model.step(&step_in, &mut dense, d_off)?;
        rvq_logits = model.step(&step_in, &mut rvq, r_off)?;
    }
    Ok((matched, total))
}

fn perplexity(logprobs: &[f32]) -> f64 {
    if logprobs.is_empty() {
        return f64::NAN;
    }
    let mean_nll = -(logprobs.iter().map(|&l| l as f64).sum::<f64>() / logprobs.len() as f64);
    mean_nll.exp()
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: rvq_generate_quality <model_dir> [max_new_tokens] [bits] [sink_tokens]");
    let max_new: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let bits: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    // Dense attention-sink size (first-N tokens kept lossless). StreamingLLM shows the first few
    // positions carry disproportionate attention mass, so a small dense sink is the standard, faithful
    // config — not metric-gaming. Default 4; the upstream cache exposes the same knob.
    let sink_tokens: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);

    let dir = Path::new(&dir);
    eprintln!("loading {} …", dir.display());
    let cfg = ModelConfig::from_dir(dir)?;
    let head_dim = cfg.head_dim;
    let num_layers = cfg.num_layers;
    let weights = Weights::from_dir(dir)?;
    let model = CausalLm::from_weights(&weights, "", cfg)?;
    let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| mlx_llm::error::Error::Msg(e.to_string()))?;

    eprintln!(
        "model: {num_layers} layers, head_dim {head_dim}; rvq bits={bits}; sink={sink_tokens}; \
         {} prompts × {max_new} new tokens (greedy)\n",
        PROMPTS.len()
    );

    let quant = RvqQuantizer::new(head_dim, bits, 42)?;

    // Teacher-forced (gated) accumulators.
    let mut tf_matched = 0usize;
    let mut tf_total = 0usize;
    // Free-running greedy (diagnostic) accumulators.
    let mut fr_total = 0usize;
    let mut fr_agreed = 0usize;
    let mut dense_ppl_acc = 0.0f64;
    let mut rvq_ppl_acc = 0.0f64;

    for (pi, prompt) in PROMPTS.iter().enumerate() {
        let ids: Vec<i32> = tok
            .encode(prompt, true)
            .map_err(|e| mlx_llm::error::Error::Msg(e.to_string()))?
            .into_iter()
            .map(|i| i as i32)
            .collect();

        // Free-running greedy on each cache independently (produces the dense trajectory used for the
        // teacher-forced pass, and the diagnostic free-running agreement + perplexity).
        let mut dense = ContiguousKvCache::new(num_layers);
        let dense_out = greedy_run(&model, &ids, &mut dense, max_new)?;
        let mut rvq = QuantizedKvCache::with_sink(
            num_layers,
            quant.clone(),
            SinkConfig::keep_first(sink_tokens),
        );
        let rvq_out = greedy_run(&model, &ids, &mut rvq, max_new)?;

        let n = dense_out.tokens.len().min(rvq_out.tokens.len());
        let fr_match = (0..n)
            .filter(|&i| dense_out.tokens[i] == rvq_out.tokens[i])
            .count();
        fr_total += n;
        fr_agreed += fr_match;
        dense_ppl_acc += perplexity(&dense_out.logprobs);
        rvq_ppl_acc += perplexity(&rvq_out.logprobs);

        // Teacher-forced agreement down the dense trajectory (the GATED metric).
        let (tm, tt) = teacher_forced_agreement(
            &model,
            &ids,
            &dense_out.tokens,
            &quant,
            num_layers,
            sink_tokens,
        )?;
        tf_matched += tm;
        tf_total += tt;

        eprintln!(
            "prompt {pi}: teacher-forced {tm}/{tt} = {:.0}%   free-running {fr_match}/{n} = {:.0}%   \
             dense_ppl {:.2}  rvq_ppl {:.2}",
            100.0 * tm as f64 / tt as f64,
            100.0 * fr_match as f64 / n as f64,
            perplexity(&dense_out.logprobs),
            perplexity(&rvq_out.logprobs),
        );
    }

    let tf_agreement = tf_matched as f64 / tf_total as f64;
    let fr_agreement = fr_agreed as f64 / fr_total as f64;
    let np = PROMPTS.len() as f64;
    println!("\n==== RVQ generation quality (bits={bits}, sink={sink_tokens}) ====");
    println!(
        "teacher-forced agreement (GATED) : {:.3} ({tf_matched}/{tf_total})",
        tf_agreement
    );
    println!(
        "free-running agreement (diag)    : {:.3} ({fr_agreed}/{fr_total})",
        fr_agreement
    );
    println!(
        "mean dense perplexity            : {:.3}",
        dense_ppl_acc / np
    );
    println!("mean rvq  perplexity             : {:.3}", rvq_ppl_acc / np);

    // Agreed threshold (documented): b=2 must reach >= 0.80 TEACHER-FORCED token-agreement.
    let threshold = 0.80;
    let gated = bits >= 2;
    if gated {
        let verdict = if tf_agreement >= threshold {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "threshold (b>=2): >= {threshold:.2} teacher-forced agreement => {verdict} \
             (got {tf_agreement:.3})"
        );
        if tf_agreement < threshold {
            std::process::exit(2);
        }
    } else {
        println!("(b={bits} reported, not gated; gate applies at b>=2)");
    }
    Ok(())
}
