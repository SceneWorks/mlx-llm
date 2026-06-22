//! GGUF → `tokenizer.json` reconstruction parity (`#[ignore]` — needs models on disk), stories
//! 7251 (byte-level BPE) + 7334 (SentencePiece BPE).
//!
//! Point `MLX_LLM_BPE_GGUF` at a single `*.gguf` — byte-level BPE (SmolLM2/Qwen/Llama-3) or
//! SentencePiece BPE (Llama-2/Mistral) — and `MLX_LLM_TEST_MODEL` at that model's HF snapshot (for
//! the reference `tokenizer.json`), then:
//!
//! ```text
//! MLX_LLM_BPE_GGUF=/tmp/SmolLM2-135M-Instruct-F16.gguf MLX_LLM_TEST_MODEL=/tmp/smollm2-135m \
//!   cargo test --test gguf_tokenizer -- --ignored --nocapture
//! # or a SentencePiece model:
//! MLX_LLM_BPE_GGUF=/tmp/tinyllama-q4km.gguf MLX_LLM_TEST_MODEL=/tmp/tinyllama-hf \
//!   cargo test --test gguf_tokenizer -- --ignored --nocapture
//! ```
//!
//! The converter reconstructs `tokenizer.json` from the GGUF metadata with no external files; the
//! acceptance bar is **token-id identical** to the source model's HF tokenizer on a varied corpus
//! (so the reconstructed normalizer / pre-tokenizer / merges all match, not just the vocab), and a
//! self-contained snapshot that loads + generates through the full provider path.

use core_llm::Tokenizer;

use mlx_llm::config::ModelConfig;
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig};
use mlx_llm::gguf::{convert_file, ConvertOptions, TokenizerStatus};
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::eos_token_ids;

/// A varied corpus exercising the pieces that distinguish a correct reconstruction from a vocab-only
/// one: leading/internal spaces, digits (the digit-split families), punctuation, casing, unicode,
/// newlines, code, and special-token strings.
const CORPUS: &[&str] = &[
    "The capital of France is Paris.",
    " leading space and  double  spaces",
    "Numbers: 0 1 42 100 2024 1234567890",
    "Punctuation!!! Really??? (yes) — em-dash, \"quotes\", it's fine.",
    "MixedCASE CamelCase snake_case kebab-case",
    "Unicode: café naïve Москва 東京 🤖🚀",
    "line one\nline two\n\tindented\ttabs",
    "def f(x):\n    return x ** 2 + 1  # code",
    "<|im_start|>system\nYou are helpful.<|im_end|>",
    "Mixing<|im_start|>special tokens inline",
];

fn tmp_out(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("mlx-llm-tok-test-{}-{label}", std::process::id()))
}

fn convert_self_contained(gguf: &str) -> (std::path::PathBuf, TokenizerStatus) {
    let out = tmp_out("snap");
    // No --tokenizer: the snapshot must stand on its own.
    let report = convert_file(gguf, &out, ConvertOptions::default()).expect("convert failed");
    (out, report.tokenizer)
}

/// The reconstructed `tokenizer.json` is **token-id identical** to the source HF tokenizer across the
/// corpus (encode), and decodes back to the same text (round-trip). A wrong normalizer/pre-tokenizer
/// or a mis-split merge table would diverge on at least one of the varied lines even with an
/// identical vocab.
#[test]
#[ignore = "needs MLX_LLM_BPE_GGUF + MLX_LLM_TEST_MODEL"]
fn gguf_bpe_tokenizer_roundtrip_matches_hf() {
    let (Ok(gguf), Ok(hf_dir)) = (
        std::env::var("MLX_LLM_BPE_GGUF"),
        std::env::var("MLX_LLM_TEST_MODEL"),
    ) else {
        eprintln!("skip: set MLX_LLM_BPE_GGUF and MLX_LLM_TEST_MODEL");
        return;
    };

    let (out, status) = convert_self_contained(&gguf);
    match &status {
        TokenizerStatus::Reconstructed(kind) => println!("reconstructed: {kind}"),
        other => panic!("expected a reconstructed tokenizer, got {other:?}"),
    }
    assert!(out.join("tokenizer.json").exists(), "tokenizer.json not written");
    assert!(out.join("tokenizer_config.json").exists(), "tokenizer_config.json not written");

    let recon = Tokenizer::from_file(out.join("tokenizer.json")).expect("load reconstructed tokenizer");
    let hf = Tokenizer::from_file(format!("{hf_dir}/tokenizer.json")).expect("load HF tokenizer");

    let mut mismatches: Vec<String> = Vec::new();
    for text in CORPUS {
        // add_special=false isolates the BPE + pre-tokenizer behaviour from any post-processor.
        let a = hf.encode(text, false).unwrap();
        let b = recon.encode(text, false).unwrap();
        if a != b {
            mismatches.push(format!("encode mismatch on {text:?}:\n    hf   ={a:?}\n    recon={b:?}"));
            continue;
        }
        // Decode the ids back; the reconstructed decoder must reproduce the same surface text the HF
        // decoder does (byte-level decoders are deterministic).
        let da = hf.decode(&a, false).unwrap();
        let db = recon.decode(&b, false).unwrap();
        if da != db {
            mismatches.push(format!("decode mismatch on {text:?}: hf={da:?} recon={db:?}"));
        }
    }
    std::fs::remove_dir_all(&out).ok();
    assert!(
        mismatches.is_empty(),
        "{} / {} corpus lines diverged:\n  {}",
        mismatches.len(),
        CORPUS.len(),
        mismatches.join("\n  ")
    );
    println!("all {} corpus lines token-id identical to HF", CORPUS.len());
}

/// The converted snapshot is **self-contained**: with no external files it loads through the full
/// provider path (config + reconstructed tokenizer + weights) and generates non-empty text, and
/// `config.json` carries the GGUF's EOS id so stop-token resolution works.
#[test]
#[ignore = "needs MLX_LLM_BPE_GGUF + MLX_LLM_TEST_MODEL"]
fn gguf_bpe_snapshot_runs_self_contained() {
    let Ok(gguf) = std::env::var("MLX_LLM_BPE_GGUF") else {
        eprintln!("skip: set MLX_LLM_BPE_GGUF");
        return;
    };

    let (out, status) = convert_self_contained(&gguf);
    assert!(matches!(status, TokenizerStatus::Reconstructed(_)), "tokenizer not reconstructed");

    // EOS id stamped into config.json so the engine resolves stop tokens from the snapshot alone.
    let eos = eos_token_ids(&out);
    let fallback = vec![128001, 128008, 128009];
    assert_ne!(eos, fallback, "config.json did not carry an eos_token_id (got the llama3 fallback)");

    // Full self-contained load: tokenizer + config + weights all from the converted directory.
    let tok = Tokenizer::from_file(out.join("tokenizer.json")).unwrap();
    let cfg = ModelConfig::from_dir(&out).unwrap();
    let model = CausalLm::from_weights(&Weights::from_dir(&out).unwrap(), "", cfg).unwrap();

    let prompt = "The capital of France is";
    let ids: Vec<i32> = tok.encode(prompt, false).unwrap().into_iter().map(|x| x as i32).collect();
    let gen = GenerationConfig {
        max_new_tokens: 16,
        sampling: SamplingParams::default(),
        seed: Some(0),
        stop_tokens: eos.clone(),
    };
    let tokens = generate(&model, &ids, &gen, &CancelFlag::new(), &mut |_| {}).unwrap().tokens;
    let text = tok.decode(&tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(), true).unwrap();
    println!("self-contained generation :: {}", text.replace('\n', " "));
    std::fs::remove_dir_all(&out).ok();
    assert!(!text.trim().is_empty(), "self-contained snapshot produced no text");
}
