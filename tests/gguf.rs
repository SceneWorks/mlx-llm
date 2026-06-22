//! Real-weights GGUF conversion parity tests (`#[ignore]` — need models on disk), story 7165.
//!
//! Point `MLX_LLM_GGUF_DIR` at a directory of `*.gguf` files for one model and `MLX_LLM_TEST_MODEL`
//! at that model's HF snapshot (for the reference weights + tokenizer), then:
//!
//! ```text
//! MLX_LLM_GGUF_DIR=/path/to/ggufs MLX_LLM_TEST_MODEL=/path/to/HF-snapshot \
//!   cargo test --test gguf -- --ignored --nocapture
//! ```
//!
//! ## What parity means here, and how it's measured
//! Conversion is **lossless apart from the GGUF's own quantization**, so a converted model should
//! reproduce the HF safetensors load. Two complementary checks, because raw greedy-token equality is
//! brittle (a single near-tie diverges and cascades — the same effect the dynamic-batch work
//! documented), and bit-equality is unattainable for a quantized GGUF:
//!
//! - [`gguf_dequant_matches_hf_weights`] — the **direct** dequant proof: every converted tensor is
//!   compared to the HF tensor (per-tensor cosine). A correct block decode reproduces each weight up
//!   to that type's quantization error (cosine ≈ 1, scaling with bit-width); a mis-read layout or a
//!   mis-permuted q/k projection collapses the cosine toward 0 on exactly the affected tensors.
//! - [`gguf_dense_conversion_tracks_hf`] — the **behavioral** check: each type loads and generates
//!   coherent text with a next-token distribution far from random (softmax-probability cosine vs HF).
//!   The lossless `F16`/`BF16` conversion additionally reproduces HF's greedy continuation exactly.
//!
//! [`gguf_requant_snapshot_loads_quantized`] covers the optional MLX-requant output + load round-trip.

use core_llm::Tokenizer;
use mlx_rs::Dtype;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig};
use mlx_llm::gguf::convert::remap_key;
use mlx_llm::gguf::{convert_file, ConvertOptions, GgufFile};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::{input_ids, QuantSpec, Weights};
use mlx_llm::provider::eos_token_ids;

const PROMPT: &str = "The capital of France is";

struct Ref {
    model: CausalLm,
    cfg: ModelConfig,
    tok: Tokenizer,
    stop: Vec<i32>,
}

fn load_ref() -> Option<Ref> {
    let dir = std::env::var("MLX_LLM_TEST_MODEL").ok()?;
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&Weights::from_dir(&dir).unwrap(), "", cfg.clone()).unwrap();
    let tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let stop = eos_token_ids(std::path::Path::new(&dir));
    Some(Ref { model, cfg, tok, stop })
}

fn gguf_files() -> Vec<std::path::PathBuf> {
    let Ok(dir) = std::env::var("MLX_LLM_GGUF_DIR") else {
        return Vec::new();
    };
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    v.sort();
    v
}

fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, true).unwrap().into_iter().map(|id| id as i32).collect()
}

/// Last-position prefill logits as host `f32`.
fn prefill_logits(model: &CausalLm, ids: &[i32]) -> Vec<f32> {
    let mut cache = model.new_cache();
    let arr = input_ids(ids);
    let logits = model.decode_logits(&arr, &mut cache, 0).unwrap();
    logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec()
}

/// Softmax of a logit vector (numerically stabilized) — the behavioral distribution the engine
/// samples from. Comparing these, not raw logits, measures whether two models *behave* the same:
/// raw-logit cosine over a 49k vocab is dominated by tens of thousands of tiny bf16-noisy
/// background logits and stays low even for behaviorally identical models.
fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn greedy_tokens(model: &CausalLm, ids: &[i32], stop: &[i32], n: usize) -> Vec<i32> {
    let cfg = GenerationConfig {
        max_new_tokens: n,
        sampling: SamplingParams::default(),
        seed: Some(0),
        stop_tokens: stop.to_vec(),
    };
    generate(model, ids, &cfg, &CancelFlag::new(), &mut |_| {}).unwrap().tokens
}

fn tmp_out(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("mlx-llm-gguf-test-{}-{label}", std::process::id()))
}

/// Every GGUF in the directory converts to a dense snapshot whose prefill logits track the HF load
/// (high cosine for all types; top-1 agreement for the near-lossless ones), and generates coherent
/// text. This is the cross-type proof that each block layout — including the k-quants — matches
/// llama.cpp.
#[test]
#[ignore = "needs MLX_LLM_GGUF_DIR + MLX_LLM_TEST_MODEL"]
fn gguf_dense_conversion_tracks_hf() {
    let Some(r) = load_ref() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let files = gguf_files();
    if files.is_empty() {
        eprintln!("skip: set MLX_LLM_GGUF_DIR to a directory of *.gguf");
        return;
    }

    let ids = encode(&r.tok, PROMPT);
    let hf_logits = prefill_logits(&r.model, &ids);
    let hf_probs = softmax(&hf_logits);
    let hf_top1 = argmax(&hf_logits);
    let hf_tokens = greedy_tokens(&r.model, &ids, &r.stop, 24);
    println!("HF top-1 next-token id for {PROMPT:?} = {hf_top1}; greedy = {hf_tokens:?}");
    println!("{:>34}  {:>7}  {:>5}  {:>9}  text", "gguf", "probcos", "top1", "prefix/24");

    // Collect every file's result first (so one run reports the whole matrix), then assert — a
    // failing type does not hide the others.
    let mut failures: Vec<String> = Vec::new();
    let mut covered = 0usize;
    for path in &files {
        let label = path.file_stem().unwrap().to_string_lossy().to_string();
        let out = tmp_out(&label);
        // A GGUF using a genuinely-unsupported type (sub-4-bit IQ) is reported and skipped, not
        // silently passed over — the suite covers every type it claims to and names the rest.
        let report = match convert_file(path, &out, ConvertOptions::default()) {
            Ok(r) => r,
            Err(mlx_llm::error::Error::Unsupported(msg)) => {
                println!("{label:>34}  SKIP — {msg}");
                continue;
            }
            Err(e) => panic!("{label}: convert failed: {e}"),
        };

        let cfg = ModelConfig::from_dir(&out).unwrap();
        // Config reconstructed from GGUF metadata must equal the HF config's load-relevant fields.
        assert_eq!(cfg.hidden_size, r.cfg.hidden_size, "{label}: hidden_size");
        assert_eq!(cfg.num_layers, r.cfg.num_layers, "{label}: num_layers");
        assert_eq!(cfg.num_heads, r.cfg.num_heads, "{label}: num_heads");
        assert_eq!(cfg.num_kv_heads, r.cfg.num_kv_heads, "{label}: num_kv_heads");
        assert_eq!(cfg.head_dim, r.cfg.head_dim, "{label}: head_dim");
        assert_eq!(cfg.vocab_size, r.cfg.vocab_size, "{label}: vocab_size");
        assert_eq!(cfg.tie_word_embeddings, r.cfg.tie_word_embeddings, "{label}: tie");
        assert!(report.quantized.is_none(), "{label}: dense expected");

        let model = CausalLm::from_weights(&Weights::from_dir(&out).unwrap(), "", cfg).unwrap();
        let logits = prefill_logits(&model, &ids);
        let probcos = cosine(&softmax(&logits), &hf_probs);
        let top1 = argmax(&logits);
        let tokens = greedy_tokens(&model, &ids, &r.stop, 24);
        let prefix = common_prefix(&tokens, &hf_tokens);
        let text = r.tok.decode(&tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(), true).unwrap();
        println!(
            "{label:>34}  {probcos:>7.4}  {:>5}  {prefix:>4}/24   {}",
            top1 == hf_top1,
            text.replace('\n', " ")
        );

        // Every type must convert to a model that runs and produces coherent text whose next-token
        // distribution is far from random (a broken dequant gives probcos ≈ 0 / gibberish). Exact
        // *behavioral* parity is only attainable from the lossless F16 conversion: lower-bit greedy
        // continuations diverge from HF at the first near-tie and cascade (the same brittleness the
        // dynamic-batch work documented), so per-type dequant correctness is proven at the weight
        // level in `gguf_dequant_matches_hf_weights`, not by greedy-token equality here.
        let lossless = label.to_uppercase().contains("F16");
        if text.trim().is_empty() {
            failures.push(format!("{label}: produced no text"));
        }
        if probcos < 0.80 {
            failures.push(format!("{label}: softmax cosine {probcos:.4} < 0.80 (sanity floor)"));
        }
        if lossless {
            if top1 != hf_top1 {
                failures.push(format!("{label}: lossless next-token {top1} != HF {hf_top1}"));
            }
            if probcos < 0.999 {
                failures.push(format!("{label}: lossless softmax cosine {probcos:.4} < 0.999"));
            }
            if prefix < 20 {
                failures.push(format!("{label}: lossless greedy prefix {prefix}/24 — expected ≈ exact"));
            }
        }
        covered += 1;
        std::fs::remove_dir_all(&out).ok();
    }

    assert!(covered > 0, "no convertible GGUF found in MLX_LLM_GGUF_DIR");
    assert!(failures.is_empty(), "parity failures:\n  {}", failures.join("\n  "));
}

/// The optional MLX requant path: convert with `--quant q8`/`q4`, and the resulting snapshot loads
/// as **quantized** (no on-load quantization) and still generates coherent text tracking HF. Proves
/// the converter's quantized-snapshot output and the loader's pre-quantized read path round-trip.
#[test]
#[ignore = "needs MLX_LLM_GGUF_DIR + MLX_LLM_TEST_MODEL"]
fn gguf_requant_snapshot_loads_quantized() {
    let Some(r) = load_ref() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    // Prefer a high-precision source so the only meaningful loss is the MLX requant itself.
    let files = gguf_files();
    let Some(src) = files
        .iter()
        .find(|p| p.to_string_lossy().to_uppercase().contains("Q8_0"))
        .or_else(|| files.first())
    else {
        eprintln!("skip: set MLX_LLM_GGUF_DIR");
        return;
    };

    let ids = encode(&r.tok, PROMPT);
    let hf_probs = softmax(&prefill_logits(&r.model, &ids));

    // q8 is near-lossless even on a 135M model (the round-trip must preserve it); q4 group-64 on a
    // 135M model is genuinely lossy — the point of that case is the load path round-trips and runs,
    // not q4 quality. A *broken* pre-quantized load (mis-paired scales/biases) gives ~0, so the q4
    // floor still cleanly separates "works" from "broken".
    for (tag, spec, floor) in [("q8", QuantSpec::q8(), 0.99f32), ("q4", QuantSpec::q4(), 0.6)] {
        let out = tmp_out(&format!("requant-{tag}"));
        let report = convert_file(src, &out, ConvertOptions { quantize: Some(spec) }).unwrap();
        assert_eq!(report.quantized, Some(spec));

        let cfg = ModelConfig::from_dir(&out).unwrap();
        assert_eq!(cfg.quantization, Some(spec), "{tag}: config carries quantization block");

        let model = CausalLm::from_weights(&Weights::from_dir(&out).unwrap(), "", cfg).unwrap();
        assert!(model.is_quantized(), "{tag}: model loaded as quantized");

        let probcos = cosine(&softmax(&prefill_logits(&model, &ids)), &hf_probs);
        let tokens = greedy_tokens(&model, &ids, &r.stop, 16);
        let text = r.tok.decode(&tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(), true).unwrap();
        println!("requant {tag}: softmax-cosine {probcos:.4}  :: {}", text.replace('\n', " "));

        assert!(probcos >= floor, "{tag}: requantized softmax cosine {probcos} below {floor}");
        assert!(!text.trim().is_empty(), "{tag}: produced no text");
        std::fs::remove_dir_all(&out).ok();
    }
}

/// Per-tensor weight parity vs HF — the direct proof every block layout is decoded correctly,
/// independent of the behavioral brittleness above. For each GGUF type we convert and compare every
/// tensor against the HF safetensors: a correct dequant reproduces each weight up to that type's
/// quantization error (cosine close to 1, scaling with bit-width), while a mis-read block layout or
/// a mis-permuted q/k projection collapses the cosine toward 0 on exactly the affected tensors. The
/// F16 conversion (lossless apart from f16↔bf16) must be ≈ 1 on every tensor — the tight guard for
/// the key remap and the q/k RoPE un-permute.
#[test]
#[ignore = "needs MLX_LLM_GGUF_DIR + MLX_LLM_TEST_MODEL"]
fn gguf_dequant_matches_hf_weights() {
    let Ok(hf_dir) = std::env::var("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let files = gguf_files();
    if files.is_empty() {
        eprintln!("skip: set MLX_LLM_GGUF_DIR");
        return;
    }
    let hf = Weights::from_dir(&hf_dir).unwrap();
    // Cache HF tensors as host f32 once, keyed for lookup. (A tied model may carry a redundant
    // `lm_head.weight` here that the converter legitimately omits — we iterate converted keys, so
    // that asymmetry is harmless; load-time completeness is enforced by `from_weights`.)
    let hf_f32: std::collections::HashMap<String, Vec<f32>> = hf
        .keys()
        .map(|k| (k.to_string(), hf.require(k).unwrap().as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec()))
        .collect();

    println!("{:>34}  {:>8}  {:>8}  worst-tensor", "gguf", "min", "mean");
    let mut failures: Vec<String> = Vec::new();
    let mut covered = 0usize;
    for path in &files {
        let label = path.file_stem().unwrap().to_string_lossy().to_string();
        let out = tmp_out(&format!("wcos-{label}"));
        let report = match convert_file(path, &out, ConvertOptions::default()) {
            Ok(r) => r,
            Err(mlx_llm::error::Error::Unsupported(_)) => continue, // covered by the behavioral test's skip log
            Err(e) => panic!("{label}: convert failed: {e}"),
        };
        let _ = report;
        let conv = Weights::from_dir(&out).unwrap();
        let mut conv_keys: Vec<String> = conv.keys().map(|k| k.to_string()).collect();
        conv_keys.sort();
        let mut min = (f32::INFINITY, String::new());
        let mut sum = 0.0f32;
        for k in &conv_keys {
            let av = hf_f32
                .get(k)
                .unwrap_or_else(|| panic!("{label}: converted tensor {k} absent from HF snapshot"));
            let b = conv.require(k).unwrap().as_dtype(Dtype::Float32).unwrap();
            let cos = cosine(av, b.as_slice::<f32>());
            sum += cos;
            if cos < min.0 {
                min = (cos, k.clone());
            }
        }
        let mean = sum / conv_keys.len() as f32;
        println!("{label:>34}  {:>8.5}  {mean:>8.5}  {}", min.0, min.1);

        // Floors that a *correct* dequant clears for any quant (a broken one collapses to ≈ 0):
        // F16 is lossless; the most aggressive 2/3-bit recipe still keeps every tensor well
        // correlated with HF.
        let (min_floor, mean_floor) = if label.to_uppercase().contains("F16") {
            (0.999, 0.9999)
        } else {
            (0.85, 0.95)
        };
        if min.0 < min_floor {
            failures.push(format!("{label}: worst tensor {} cosine {:.5} < {min_floor}", min.1, min.0));
        }
        if mean < mean_floor {
            failures.push(format!("{label}: mean tensor cosine {mean:.5} < {mean_floor}"));
        }
        covered += 1;
        std::fs::remove_dir_all(&out).ok();
    }
    assert!(covered > 0, "no convertible GGUF found");
    assert!(failures.is_empty(), "dequant weight-parity failures:\n  {}", failures.join("\n  "));
}

/// Human name for a GGML quant tag (the sub-4-bit IQ grid types this story added, plus the
/// neighbours that the unsloth dynamic mixes interleave).
fn ggml_type_name(tag: u32) -> &'static str {
    match tag {
        0 => "F32",
        1 => "F16",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "other",
    }
}

/// Per-GGML-type weight-parity for the sub-4-bit IQ grid codebooks (story 7250). Point
/// `MLX_LLM_IQ_GGUF_DIR` at a directory of IQ `*.gguf` files (e.g. unsloth's `UD-IQ1_S/IQ1_M/
/// IQ2_XXS/IQ2_M/IQ3_XXS` dynamic quants) for the model whose HF snapshot is `MLX_LLM_TEST_MODEL`:
///
/// ```text
/// MLX_LLM_IQ_GGUF_DIR=/path/to/iq-ggufs MLX_LLM_TEST_MODEL=/tmp/qwen3-0.6b \
///   cargo test --test gguf -- --ignored gguf_iq_dequant_matches_hf_by_type --nocapture
/// ```
///
/// Unlike the per-file check above, this buckets every converted tensor's cosine-vs-HF by the
/// tensor's *actual* GGML type (the dynamic quants interleave many IQ types per file), so each of
/// the seven new grid types is validated directly. A correct grid decode reproduces each weight up
/// to that type's (genuinely lossy, for the 1–3 bit types) quantization error — cosine well clear of
/// zero; a wrong grid/sign unpack collapses the cosine toward 0 on exactly the tensors of that type.
/// The floors are set per bit-width to cleanly separate "correctly lossy" from "broken".
#[test]
#[ignore = "needs MLX_LLM_IQ_GGUF_DIR + MLX_LLM_TEST_MODEL"]
fn gguf_iq_dequant_matches_hf_by_type() {
    let Ok(hf_dir) = std::env::var("MLX_LLM_TEST_MODEL") else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let Ok(gguf_dir) = std::env::var("MLX_LLM_IQ_GGUF_DIR") else {
        eprintln!("skip: set MLX_LLM_IQ_GGUF_DIR to a directory of IQ *.gguf files");
        return;
    };
    let mut files: Vec<_> = std::fs::read_dir(&gguf_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no *.gguf in MLX_LLM_IQ_GGUF_DIR");

    let hf = Weights::from_dir(&hf_dir).unwrap();
    let hf_f32: std::collections::HashMap<String, Vec<f32>> = hf
        .keys()
        .map(|k| (k.to_string(), hf.require(k).unwrap().as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().to_vec()))
        .collect();

    // type -> (count, sum_cos, min_cos, worst_key)
    let mut by_type: std::collections::BTreeMap<&'static str, (usize, f32, f32, String)> =
        std::collections::BTreeMap::new();

    for path in &files {
        let label = path.file_stem().unwrap().to_string_lossy().to_string();
        // GGUF tensor name -> its GGML type, so a converted (HF-keyed) tensor can be bucketed.
        let g = GgufFile::open(path).unwrap();
        let mut hf_key_type: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for t in &g.tensors {
            if let Some(hf_key) = remap_key(&t.name) {
                hf_key_type.insert(hf_key, t.ggml_type);
            }
        }

        let out = tmp_out(&format!("iqcos-{label}"));
        let _report = convert_file(path, &out, ConvertOptions::default())
            .unwrap_or_else(|e| panic!("{label}: convert failed: {e}"));
        let conv = Weights::from_dir(&out).unwrap();
        for k in conv.keys() {
            let Some(&tag) = hf_key_type.get(k) else { continue };
            let av = hf_f32.get(k).unwrap_or_else(|| panic!("{label}: {k} absent from HF"));
            let b = conv.require(k).unwrap().as_dtype(Dtype::Float32).unwrap();
            let cos = cosine(av, b.as_slice::<f32>());
            let name = ggml_type_name(tag);
            let e = by_type.entry(name).or_insert((0, 0.0, f32::INFINITY, String::new()));
            e.0 += 1;
            e.1 += cos;
            if cos < e.2 {
                e.2 = cos;
                e.3 = format!("{label}:{k}");
            }
        }
        std::fs::remove_dir_all(&out).ok();
    }

    println!("{:>9}  {:>5}  {:>8}  {:>8}  worst-tensor", "type", "n", "min", "mean");
    for (name, (n, sum, min, worst)) in &by_type {
        println!("{name:>9}  {n:>5}  {min:>8.5}  {:>8.5}  {worst}", sum / *n as f32);
    }

    // Floors a *correct* (lossy) decode clears for each bit-width; a broken grid/sign unpack lands
    // near 0 and trips them. The seven grid types this story added must all be present and pass.
    // Calibrated on unsloth's Qwen3-0.6B dynamic quants (observed per-type worst-tensor cosine:
    // IQ1_S 0.878, IQ1_M 0.892, IQ2_XXS 0.936, IQ2_XS 0.954, IQ2_S 0.964, IQ3_XXS 0.982, IQ3_S
    // 0.980), with margin for model/recipe variation. A broken grid/sign unpack lands near 0.
    let floor = |name: &str| -> f32 {
        match name {
            "IQ1_S" | "IQ1_M" => 0.60,
            "IQ2_XXS" | "IQ2_XS" | "IQ2_S" => 0.80,
            "IQ3_XXS" | "IQ3_S" => 0.88,
            _ => 0.90, // Q2_K..Q6_K / IQ4_* neighbours in the dynamic mix
        }
    };
    let required = ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M"];
    let mut failures: Vec<String> = Vec::new();
    for t in required {
        match by_type.get(t) {
            None => failures.push(format!("{t}: no tensors of this type found in the GGUF dir")),
            Some((n, sum, min, worst)) => {
                let f = floor(t);
                if *min < f {
                    failures.push(format!("{t}: worst tensor {worst} cosine {min:.5} < {f} (broken decode?)"));
                }
                let mean = sum / *n as f32;
                if mean < f {
                    failures.push(format!("{t}: mean cosine {mean:.5} < {f}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "IQ per-type dequant parity failures:\n  {}", failures.join("\n  "));
}

/// End-to-end: each IQ-quantized GGUF converts to a snapshot that loads and *runs* — the prefill
/// produces all-finite logits (no NaN/Inf) and greedy generation yields non-empty text. This proves
/// the IQ grids decode into a numerically-sound, runnable model end to end (convert → load → forward
/// → sample), not just correlated weights.
///
/// It deliberately does **not** gate on a next-token-distribution match with HF. On a 0.6B base these
/// 1.5–3 bpw quants are genuinely lossy, and softmax-cosine over the near-one-hot next-token
/// distribution essentially measures top-1 agreement — which a correct-but-lossy decode legitimately
/// shifts (e.g. the IQ3_XXS snapshot still generates "...the city of Paris..." while its first-token
/// softmax-cosine sits near zero). So that metric can't separate "lossy" from "broken" here; the
/// rigorous per-type correctness proof is `gguf_iq_dequant_matches_hf_by_type` (weight cosine ≈ 0.88
/// for IQ1 up to ≈ 0.98 for IQ3 — a broken grid/sign unpack would instead collapse to ≈ 0). The
/// probabilities-vs-HF cosine is still printed per file for human inspection.
#[test]
#[ignore = "needs MLX_LLM_IQ_GGUF_DIR + MLX_LLM_TEST_MODEL"]
fn gguf_iq_snapshot_generates() {
    let Some(r) = load_ref() else {
        eprintln!("skip: set MLX_LLM_TEST_MODEL");
        return;
    };
    let Ok(gguf_dir) = std::env::var("MLX_LLM_IQ_GGUF_DIR") else {
        eprintln!("skip: set MLX_LLM_IQ_GGUF_DIR");
        return;
    };
    let mut files: Vec<_> = std::fs::read_dir(&gguf_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .filter(|p| {
            let s = p.to_string_lossy().to_uppercase();
            s.contains("IQ") && !s.contains("BF16") && !s.contains("F16") && !s.contains("IQ4")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        eprintln!("skip: no IQ *.gguf in MLX_LLM_IQ_GGUF_DIR");
        return;
    }

    let ids = encode(&r.tok, PROMPT);
    let hf_probs = softmax(&prefill_logits(&r.model, &ids));

    let mut covered = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for path in &files {
        let label = path.file_stem().unwrap().to_string_lossy().to_string();
        let out = tmp_out(&format!("iqgen-{label}"));
        convert_file(path, &out, ConvertOptions::default()).unwrap();
        let cfg = ModelConfig::from_dir(&out).unwrap();
        let model = CausalLm::from_weights(&Weights::from_dir(&out).unwrap(), "", cfg).unwrap();
        let logits = prefill_logits(&model, &ids);
        let probcos = cosine(&softmax(&logits), &hf_probs);
        let tokens = greedy_tokens(&model, &ids, &r.stop, 16);
        let text = r.tok.decode(&tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(), true).unwrap();
        println!("{label:>34}: probcos {probcos:.4}  :: {}", text.replace('\n', " "));
        // Runnability gates: finite logits (a numerically-broken decode would NaN/Inf) and non-empty
        // generation. Correctness vs HF is asserted at the weight level in the companion test.
        if !logits.iter().all(|v| v.is_finite()) {
            failures.push(format!("{label}: prefill logits contain NaN/Inf"));
        }
        if text.trim().is_empty() {
            failures.push(format!("{label}: produced no text"));
        }
        covered += 1;
        std::fs::remove_dir_all(&out).ok();
    }
    assert!(covered > 0, "no IQ *.gguf found in MLX_LLM_IQ_GGUF_DIR");
    assert!(failures.is_empty(), "IQ snapshot run failures:\n  {}", failures.join("\n  "));
}
