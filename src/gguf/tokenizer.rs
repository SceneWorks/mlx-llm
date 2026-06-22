//! Reconstruct a Hugging Face `tokenizer.json` (+ `tokenizer_config.json`) from a GGUF's embedded
//! llama.cpp tokenizer metadata (story 7251), so a converted snapshot is self-contained and runs
//! through [`crate::provider::LlamaProvider::load`] with no external `tokenizer.json`.
//!
//! A GGUF stores its tokenizer as `tokenizer.ggml.*` metadata — a token list, BPE merges, per-token
//! types, and special-token ids — *not* a HF `tokenizer.json`. The engine loads
//! `core_llm::Tokenizer::from_file("tokenizer.json")` (a `tokenizers`-crate model), so this module is
//! the encoder from the former to the latter.
//!
//! ## Scope: byte-level BPE (`tokenizer.ggml.model == "gpt2"`)
//! Covers the GPT-2 byte-level BPE family — SmolLM2, Qwen, Llama-3, and friends. The vocabulary,
//! merges, and added/special tokens come straight from the metadata; the `normalizer` /
//! `pre_tokenizer` / `decoder` come from a small per-`tokenizer.ggml.pre` template table (these
//! differ materially between families — e.g. Qwen2's NFC + split regex vs SmolLM2's digit split), so
//! the reconstructed tokenizer matches the source's tokenization, not just its vocab.
//!
//! The SentencePiece/Unigram `llama` model is **not** reconstructed: GGUF does not carry the
//! precompiled normalizer charsmap a faithful SPM `tokenizer.json` needs, so a reconstruction would
//! silently mis-tokenize. That case is reported as [`TokenizerOutcome::Unsupported`] (convert with
//! `--tokenizer`), not guessed at.

use serde_json::{json, Map, Value};

use crate::error::Result;
use crate::gguf::reader::GgufFile;

/// GGUF token-type tags (`tokenizer.ggml.token_type`), mirroring llama.cpp `llama_token_type`.
const TOKEN_TYPE_UNKNOWN: i64 = 2;
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

/// What [`reconstruct`] could do with a GGUF's tokenizer metadata.
pub enum TokenizerOutcome {
    /// A `tokenizer.json` (+ `tokenizer_config.json`) was reconstructed.
    Reconstructed(Box<ReconstructedTokenizer>),
    /// The GGUF carries tokenizer metadata, but of a kind we can't faithfully rebuild from GGUF
    /// alone (the reason explains; convert with `--tokenizer`).
    Unsupported(String),
    /// The GGUF carries no tokenizer metadata.
    Absent,
}

/// A reconstructed tokenizer ready to write next to the converted weights.
pub struct ReconstructedTokenizer {
    /// The `tokenizer.json` document (a `tokenizers`-crate model).
    pub tokenizer_json: Value,
    /// The `tokenizer_config.json` document (`chat_template` + `bos_token`/`eos_token` for the
    /// engine's Jinja chat template).
    pub tokenizer_config_json: Value,
    /// Short human description of what was built (e.g. `"gpt2 byte-level BPE (pre=qwen2)"`).
    pub kind: String,
}

/// Reconstruct a tokenizer from the GGUF's `tokenizer.ggml.*` metadata.
pub fn reconstruct(g: &GgufFile) -> Result<TokenizerOutcome> {
    let Some(model) = g.meta_str("tokenizer.ggml.model") else {
        return Ok(TokenizerOutcome::Absent);
    };
    match model {
        // GPT-2 byte-level BPE (SmolLM2 / Qwen / Llama-3 / …).
        "gpt2" | "bpe" => reconstruct_bpe(g),
        // SentencePiece / Unigram — GGUF lacks the precompiled normalizer charsmap; don't guess.
        "llama" | "spm" | "t5" | "unigram" => Ok(TokenizerOutcome::Unsupported(format!(
            "GGUF tokenizer model {model:?} is SentencePiece/Unigram, which cannot be faithfully \
             reconstructed from GGUF metadata alone (the precompiled normalizer is not stored); \
             pass --tokenizer <tokenizer.json> from the source model"
        ))),
        other => Ok(TokenizerOutcome::Unsupported(format!(
            "unrecognized GGUF tokenizer model {other:?}; pass --tokenizer <tokenizer.json>"
        ))),
    }
}

/// Build a byte-level BPE `tokenizer.json` from the GGUF metadata.
fn reconstruct_bpe(g: &GgufFile) -> Result<TokenizerOutcome> {
    let Some(tokens) = str_array(g, "tokenizer.ggml.tokens") else {
        return Ok(TokenizerOutcome::Unsupported(
            "GGUF has tokenizer.ggml.model but no tokenizer.ggml.tokens".into(),
        ));
    };
    let Some(merges) = str_array(g, "tokenizer.ggml.merges") else {
        return Ok(TokenizerOutcome::Unsupported(
            "GGUF byte-level BPE tokenizer has no tokenizer.ggml.merges".into(),
        ));
    };
    let token_types = i64_array(g, "tokenizer.ggml.token_type");
    let pre = g.meta_str("tokenizer.ggml.pre").unwrap_or("default");

    // model.vocab: every token by its id. The GGUF token list has no duplicate strings (verified on
    // the SmolLM2/Qwen vocabs, including reserved/unused slots), so a flat map is exact and lossless.
    let mut vocab = Map::with_capacity(tokens.len());
    for (id, tok) in tokens.iter().enumerate() {
        if vocab.insert((*tok).to_string(), json!(id)).is_some() {
            return Ok(TokenizerOutcome::Unsupported(format!(
                "duplicate token {tok:?} in GGUF vocab — cannot build a 1:1 BPE vocab"
            )));
        }
    }

    // model.merges: each GGUF merge is "<left> <right>"; byte-level encoding never puts a literal
    // space inside a piece, so splitting on the first space recovers the pair exactly.
    let mut merge_pairs = Vec::with_capacity(merges.len());
    for m in &merges {
        match m.split_once(' ') {
            Some((a, b)) => merge_pairs.push(json!([a, b])),
            None => {
                return Ok(TokenizerOutcome::Unsupported(format!(
                    "malformed GGUF merge {m:?} (no space separator)"
                )))
            }
        }
    }

    // added_tokens: control / user-defined / unknown tokens become added tokens (matched atomically,
    // ahead of BPE). `special` follows HF: control & unknown are special; user-defined are not (e.g.
    // Qwen's `<think>`/`<tool_call>` are non-special added tokens).
    let mut added = Vec::new();
    if let Some(types) = &token_types {
        for (id, tok) in tokens.iter().enumerate() {
            let ty = types.get(id).copied().unwrap_or(0);
            let special = match ty {
                TOKEN_TYPE_CONTROL | TOKEN_TYPE_UNKNOWN => true,
                TOKEN_TYPE_USER_DEFINED => false,
                _ => continue,
            };
            added.push(json!({
                "id": id,
                "content": tok,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": special,
            }));
        }
    }

    let (normalizer, pre_tokenizer, post_processor, decoder) = bpe_components(pre);

    let model = json!({
        "type": "BPE",
        "dropout": Value::Null,
        "unk_token": Value::Null,
        "continuing_subword_prefix": Value::Null,
        "end_of_word_suffix": Value::Null,
        "fuse_unk": false,
        "byte_fallback": false,
        "ignore_merges": false,
        "vocab": Value::Object(vocab),
        "merges": Value::Array(merge_pairs),
    });

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": decoder,
        "model": model,
    });

    let tokenizer_config_json = build_tokenizer_config(g, &tokens);

    Ok(TokenizerOutcome::Reconstructed(Box::new(ReconstructedTokenizer {
        tokenizer_json,
        tokenizer_config_json,
        kind: format!("gpt2 byte-level BPE (pre={pre})"),
    })))
}

/// The Qwen2 pre-tokenizer split regex (verbatim from a Qwen `tokenizer.json`).
const QWEN2_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// The Llama-3 pre-tokenizer split regex — like Qwen2 but with `\p{N}{1,3}` (digit groups of ≤3).
const LLAMA3_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// `(normalizer, pre_tokenizer, post_processor, decoder)` for a `tokenizer.ggml.pre` family.
///
/// The byte-level families share a `ByteLevel` decoder/post-processor but differ in normalization
/// and pre-tokenization; these templates are taken verbatim from each family's reference
/// `tokenizer.json`. `qwen2` and `smollm` are validated against SmolLM2-135M / Qwen3-0.6B; the others
/// are best-effort from the published configs, and an unknown `pre` falls back to the plain GPT-2
/// byte-level setup (correct for most byte-level BPE, exact for the GPT-2 regex families).
fn bpe_components(pre: &str) -> (Value, Value, Value, Value) {
    // A `Split` pre-tokenizer over `regex`, isolated, then byte-level (regex off — the split already
    // segmented). Used by the families whose regex differs from GPT-2's built-in one.
    let split_then_bytelevel = |regex: &str| {
        json!({
            "type": "Sequence",
            "pretokenizers": [
                { "type": "Split", "pattern": { "Regex": regex }, "behavior": "Isolated", "invert": false },
                { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false }
            ]
        })
    };
    let bytelevel = |apsf: bool, trim: bool, use_regex: bool| {
        json!({ "type": "ByteLevel", "add_prefix_space": apsf, "trim_offsets": trim, "use_regex": use_regex })
    };

    match pre {
        "qwen2" => (
            json!({ "type": "NFC" }),
            split_then_bytelevel(QWEN2_SPLIT_REGEX),
            bytelevel(false, false, false),
            bytelevel(false, false, false),
        ),
        "llama-bpe" | "llama3" | "llama-v3" => (
            Value::Null,
            split_then_bytelevel(LLAMA3_SPLIT_REGEX),
            bytelevel(false, false, false),
            bytelevel(false, false, false),
        ),
        "smollm" => (
            Value::Null,
            json!({
                "type": "Sequence",
                "pretokenizers": [
                    { "type": "Digits", "individual_digits": true },
                    { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true }
                ]
            }),
            Value::Null,
            bytelevel(true, true, true),
        ),
        // "gpt-2", "default", and any unrecognized byte-level family: the plain GPT-2 byte-level
        // setup (the ByteLevel pre-tokenizer applies GPT-2's built-in split regex itself).
        _ => (
            Value::Null,
            bytelevel(false, true, true),
            Value::Null,
            bytelevel(true, true, true),
        ),
    }
}

/// Build `tokenizer_config.json` — the chat template + BOS/EOS strings the engine's Jinja template
/// reads (`core_llm::JinjaChatTemplate::from_tokenizer_config_file`).
fn build_tokenizer_config(g: &GgufFile, tokens: &[&str]) -> Value {
    let mut cfg = Map::new();
    if let Some(tmpl) = g.meta_str("tokenizer.chat_template") {
        cfg.insert("chat_template".into(), json!(tmpl));
    }
    let token_str = |id_key: &str| -> Option<String> {
        let id = g.meta_u64(id_key)? as usize;
        tokens.get(id).map(|s| (*s).to_string())
    };
    if let Some(bos) = token_str("tokenizer.ggml.bos_token_id") {
        cfg.insert("bos_token".into(), json!(bos));
    }
    if let Some(eos) = token_str("tokenizer.ggml.eos_token_id") {
        cfg.insert("eos_token".into(), json!(eos));
    }
    if let Some(unk) = token_str("tokenizer.ggml.unknown_token_id") {
        cfg.insert("unk_token".into(), json!(unk));
    }
    if let Some(pad) = token_str("tokenizer.ggml.padding_token_id") {
        cfg.insert("pad_token".into(), json!(pad));
    }
    if let Some(add_bos) = g.meta("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()) {
        cfg.insert("add_bos_token".into(), json!(add_bos));
    }
    if let Some(add_eos) = g.meta("tokenizer.ggml.add_eos_token").and_then(|v| v.as_bool()) {
        cfg.insert("add_eos_token".into(), json!(add_eos));
    }
    Value::Object(cfg)
}

/// Read a GGUF metadata array of strings, borrowing the underlying `String`s.
fn str_array<'a>(g: &'a GgufFile, key: &str) -> Option<Vec<&'a str>> {
    let arr = g.meta(key)?.as_array()?;
    arr.iter().map(|v| v.as_str()).collect()
}

/// Read a GGUF metadata array of integers as `i64`.
fn i64_array(g: &GgufFile, key: &str) -> Option<Vec<i64>> {
    let arr = g.meta(key)?.as_array()?;
    arr.iter().map(|v| v.as_i64()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GGUF_MAGIC: u32 = 0x4655_4747;

    // Minimal in-memory GGUF builder (tensor-free) for exercising the tokenizer encoder without a
    // model on disk. Mirrors the container format in `reader.rs`.
    struct Builder {
        meta: Vec<u8>,
        count: u64,
    }
    impl Builder {
        fn new() -> Self {
            Self { meta: Vec::new(), count: 0 }
        }
        fn key(&mut self, k: &str) {
            self.meta.extend_from_slice(&(k.len() as u64).to_le_bytes());
            self.meta.extend_from_slice(k.as_bytes());
            self.count += 1;
        }
        fn str_val(&mut self, s: &str) {
            self.meta.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.meta.extend_from_slice(s.as_bytes());
        }
        fn string(mut self, k: &str, v: &str) -> Self {
            self.key(k);
            self.meta.extend_from_slice(&8u32.to_le_bytes()); // T_STRING
            self.str_val(v);
            self
        }
        fn u32(mut self, k: &str, v: u32) -> Self {
            self.key(k);
            self.meta.extend_from_slice(&4u32.to_le_bytes()); // T_UINT32
            self.meta.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn str_array(mut self, k: &str, vals: &[&str]) -> Self {
            self.key(k);
            self.meta.extend_from_slice(&9u32.to_le_bytes()); // T_ARRAY
            self.meta.extend_from_slice(&8u32.to_le_bytes()); // elem T_STRING
            self.meta.extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                self.str_val(v);
            }
            self
        }
        fn i32_array(mut self, k: &str, vals: &[i32]) -> Self {
            self.key(k);
            self.meta.extend_from_slice(&9u32.to_le_bytes()); // T_ARRAY
            self.meta.extend_from_slice(&5u32.to_le_bytes()); // elem T_INT32
            self.meta.extend_from_slice(&(vals.len() as u64).to_le_bytes());
            for v in vals {
                self.meta.extend_from_slice(&v.to_le_bytes());
            }
            self
        }
        fn build(self) -> GgufFile {
            let mut b = Vec::new();
            b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            b.extend_from_slice(&3u32.to_le_bytes()); // version
            b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            b.extend_from_slice(&self.count.to_le_bytes()); // metadata_count
            b.extend_from_slice(&self.meta);
            GgufFile::parse(b).unwrap()
        }
    }

    fn reconstructed(g: &GgufFile) -> ReconstructedTokenizer {
        match reconstruct(g).unwrap() {
            TokenizerOutcome::Reconstructed(t) => *t,
            other => panic!("expected Reconstructed, got {}", outcome_name(&other)),
        }
    }
    fn outcome_name(o: &TokenizerOutcome) -> &'static str {
        match o {
            TokenizerOutcome::Reconstructed(_) => "Reconstructed",
            TokenizerOutcome::Unsupported(_) => "Unsupported",
            TokenizerOutcome::Absent => "Absent",
        }
    }

    #[test]
    fn absent_without_tokenizer_metadata() {
        let g = Builder::new().string("general.architecture", "llama").build();
        assert!(matches!(reconstruct(&g).unwrap(), TokenizerOutcome::Absent));
    }

    #[test]
    fn spm_model_is_reported_unsupported_not_guessed() {
        let g = Builder::new().string("tokenizer.ggml.model", "llama").build();
        match reconstruct(&g).unwrap() {
            TokenizerOutcome::Unsupported(reason) => assert!(reason.contains("SentencePiece")),
            other => panic!("expected Unsupported, got {}", outcome_name(&other)),
        }
    }

    #[test]
    fn bpe_vocab_merges_and_added_tokens() {
        // tokens: "a","b","ab" (normal), "<s>" (control), "<tool>" (user-defined).
        let g = Builder::new()
            .string("tokenizer.ggml.model", "gpt2")
            .string("tokenizer.ggml.pre", "qwen2")
            .str_array("tokenizer.ggml.tokens", &["a", "b", "ab", "<s>", "<tool>"])
            .i32_array("tokenizer.ggml.token_type", &[1, 1, 1, 3, 4])
            .str_array("tokenizer.ggml.merges", &["a b"])
            .u32("tokenizer.ggml.eos_token_id", 3)
            .string("tokenizer.chat_template", "TEMPLATE")
            .build();
        let t = reconstructed(&g);
        let tj = &t.tokenizer_json;

        // vocab: every token by id.
        let vocab = tj["model"]["vocab"].as_object().unwrap();
        assert_eq!(vocab["a"], 0);
        assert_eq!(vocab["ab"], 2);
        assert_eq!(vocab["<s>"], 3);
        assert_eq!(vocab.len(), 5);

        // merges: "a b" -> ["a","b"].
        assert_eq!(tj["model"]["merges"], serde_json::json!([["a", "b"]]));

        // added_tokens: control "<s>" (special) + user-defined "<tool>" (non-special). Normal tokens
        // are not added.
        let added = tj["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 2);
        let s = added.iter().find(|a| a["content"] == "<s>").unwrap();
        assert_eq!(s["id"], 3);
        assert_eq!(s["special"], true);
        let tool = added.iter().find(|a| a["content"] == "<tool>").unwrap();
        assert_eq!(tool["special"], false);

        // qwen2 pre => NFC normalizer + Split(regex)+ByteLevel pre-tokenizer.
        assert_eq!(tj["normalizer"]["type"], "NFC");
        assert_eq!(tj["pre_tokenizer"]["type"], "Sequence");
        assert_eq!(tj["pre_tokenizer"]["pretokenizers"][0]["type"], "Split");

        // tokenizer_config: chat template + eos token *string* (id 3 -> "<s>").
        assert_eq!(t.tokenizer_config_json["chat_template"], "TEMPLATE");
        assert_eq!(t.tokenizer_config_json["eos_token"], "<s>");
    }

    #[test]
    fn pre_templates_differ_by_family() {
        let base = |pre: &str| {
            Builder::new()
                .string("tokenizer.ggml.model", "gpt2")
                .string("tokenizer.ggml.pre", pre)
                .str_array("tokenizer.ggml.tokens", &["a", "b"])
                .str_array("tokenizer.ggml.merges", &["a b"])
                .build()
        };
        // smollm: no normalizer, Digits-led pre-tokenizer.
        let sm = reconstructed(&base("smollm"));
        assert!(sm.tokenizer_json["normalizer"].is_null());
        assert_eq!(sm.tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["type"], "Digits");
        // unknown pre: plain GPT-2 byte-level (single ByteLevel pre-tokenizer, no normalizer).
        let def = reconstructed(&base("totally-unknown-pre"));
        assert!(def.tokenizer_json["normalizer"].is_null());
        assert_eq!(def.tokenizer_json["pre_tokenizer"]["type"], "ByteLevel");
    }

    #[test]
    fn duplicate_tokens_are_rejected() {
        let g = Builder::new()
            .string("tokenizer.ggml.model", "gpt2")
            .str_array("tokenizer.ggml.tokens", &["a", "a"])
            .str_array("tokenizer.ggml.merges", &["a a"])
            .build();
        assert!(matches!(reconstruct(&g).unwrap(), TokenizerOutcome::Unsupported(_)));
    }
}
