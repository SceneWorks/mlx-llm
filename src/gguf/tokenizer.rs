//! Reconstruct a Hugging Face `tokenizer.json` (+ `tokenizer_config.json`) from a GGUF's embedded
//! llama.cpp tokenizer metadata (story 7251), so a converted snapshot is self-contained and runs
//! through [`crate::provider::LlamaProvider::load`] with no external `tokenizer.json`.
//!
//! A GGUF stores its tokenizer as `tokenizer.ggml.*` metadata — a token list, BPE merges, per-token
//! types, and special-token ids — *not* a HF `tokenizer.json`. The engine loads
//! `core_llm::Tokenizer::from_file("tokenizer.json")` (a `tokenizers`-crate model), so this module is
//! the encoder from the former to the latter.
//!
//! ## Byte-level BPE (`tokenizer.ggml.model == "gpt2"`)
//! Covers the GPT-2 byte-level BPE family — SmolLM2, Qwen, Llama-3, and friends. The vocabulary,
//! merges, and added/special tokens come straight from the metadata; the `normalizer` /
//! `pre_tokenizer` / `decoder` come from a small per-`tokenizer.ggml.pre` template table (these
//! differ materially between families — e.g. Qwen2's NFC + split regex vs SmolLM2's digit split), so
//! the reconstructed tokenizer matches the source's tokenization, not just its vocab.
//!
//! ## SentencePiece BPE (`tokenizer.ggml.model == "llama"`/`spm`)
//! The Llama-2 / Mistral family (story 7334). Despite the SentencePiece origin, HF's *fast*
//! tokenizer for these is a **BPE** model with `byte_fallback`, a `Prepend("▁") + Replace(" "→"▁")`
//! normalizer (no precompiled charsmap — that only applies to T5/XLNet-style SPM), and a
//! `Replace/ByteFallback/Fuse/Strip` decoder. A modern `convert_hf_to_gguf.py` GGUF already carries
//! `tokenizer.ggml.merges` (copied from the HF tokenizer), so reconstruction is **byte-exact**; if a
//! GGUF lacks merges, they are derived from `tokenizer.ggml.scores` via HF's `SentencePieceExtractor`
//! algorithm (same merge set, occasionally a different tie order — flagged as derived).

use serde_json::{json, Map, Value};

use crate::error::Result;
use crate::gguf::reader::GgufFile;

/// GGUF token-type tags (`tokenizer.ggml.token_type`), mirroring llama.cpp `llama_token_type`.
const TOKEN_TYPE_UNKNOWN: i64 = 2;
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

/// The SentencePiece space marker (`▁`, U+2581) used by the SPM normalizer/decoder.
const SPM_SPACE: &str = "\u{2581}";

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
        "gpt2" | "bpe" => reconstruct_byte_level_bpe(g),
        // SentencePiece BPE (Llama-2 / Mistral / …) — HF's fast tokenizer is byte_fallback BPE.
        "llama" | "spm" => reconstruct_spm(g),
        // Unigram/T5-style SPM genuinely needs the precompiled normalizer charsmap (not in GGUF).
        "t5" | "unigram" => Ok(TokenizerOutcome::Unsupported(format!(
            "GGUF tokenizer model {model:?} is a Unigram/T5-style SentencePiece tokenizer whose \
             precompiled normalizer is not stored in GGUF; pass --tokenizer <tokenizer.json>"
        ))),
        other => Ok(TokenizerOutcome::Unsupported(format!(
            "unrecognized GGUF tokenizer model {other:?}; pass --tokenizer <tokenizer.json>"
        ))),
    }
}

/// Build a byte-level BPE `tokenizer.json` from the GGUF metadata (gpt2 family).
fn reconstruct_byte_level_bpe(g: &GgufFile) -> Result<TokenizerOutcome> {
    let Some(tokens) = str_array(g, "tokenizer.ggml.tokens") else {
        return Ok(unsupported("GGUF has tokenizer.ggml.model but no tokenizer.ggml.tokens"));
    };
    let Some(merges) = str_array(g, "tokenizer.ggml.merges") else {
        return Ok(unsupported("GGUF byte-level BPE tokenizer has no tokenizer.ggml.merges"));
    };
    let token_types = i64_array(g, "tokenizer.ggml.token_type");
    let pre = g.meta_str("tokenizer.ggml.pre").unwrap_or("default");

    let vocab = match build_vocab(&tokens) {
        Ok(v) => v,
        Err(e) => return Ok(TokenizerOutcome::Unsupported(e)),
    };
    let merge_pairs = match parse_merges(&merges) {
        Ok(m) => m,
        Err(e) => return Ok(TokenizerOutcome::Unsupported(e)),
    };
    let added = build_added_tokens(&tokens, &token_types);
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
    let tokenizer_json = assemble(added, normalizer, pre_tokenizer, post_processor, decoder, model);
    let tokenizer_config_json = build_tokenizer_config(g, &tokens);

    Ok(TokenizerOutcome::Reconstructed(Box::new(ReconstructedTokenizer {
        tokenizer_json,
        tokenizer_config_json,
        kind: format!("gpt2 byte-level BPE (pre={pre})"),
    })))
}

/// Build a SentencePiece BPE `tokenizer.json` (Llama-2 / Mistral family) from the GGUF metadata.
fn reconstruct_spm(g: &GgufFile) -> Result<TokenizerOutcome> {
    let Some(tokens) = str_array(g, "tokenizer.ggml.tokens") else {
        return Ok(unsupported("GGUF has tokenizer.ggml.model but no tokenizer.ggml.tokens"));
    };
    let token_types = i64_array(g, "tokenizer.ggml.token_type");

    let vocab = match build_vocab(&tokens) {
        Ok(v) => v,
        Err(e) => return Ok(TokenizerOutcome::Unsupported(e)),
    };

    // Merges: prefer the GGUF's own list (copied verbatim from the HF tokenizer ⇒ byte-exact). If a
    // GGUF omits them, derive from the per-token scores via HF's `SentencePieceExtractor` (same merge
    // set; tie order can differ slightly — hence "derived").
    let (merge_pairs, merge_source) = if let Some(merges) = str_array(g, "tokenizer.ggml.merges") {
        match parse_merges(&merges) {
            Ok(m) => (m, "merges"),
            Err(e) => return Ok(TokenizerOutcome::Unsupported(e)),
        }
    } else {
        let Some(scores) = f32_array(g, "tokenizer.ggml.scores") else {
            return Ok(unsupported(
                "GGUF SentencePiece tokenizer has neither tokenizer.ggml.merges nor \
                 tokenizer.ggml.scores — cannot reconstruct BPE merges",
            ));
        };
        (derive_merges_from_scores(&tokens, &scores), "derived")
    };

    let added = build_added_tokens(&tokens, &token_types);
    // unk_token: the SPM `<unk>` piece, needed by `fuse_unk` + `byte_fallback`.
    let unk = g
        .meta_u64("tokenizer.ggml.unknown_token_id")
        .and_then(|id| tokens.get(id as usize).copied());
    // SPM prepends a `▁` to the start of the text unless add_space_prefix is explicitly false.
    let add_space_prefix = g
        .meta("tokenizer.ggml.add_space_prefix")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let (normalizer, decoder) = spm_components(add_space_prefix);
    let model = json!({
        "type": "BPE",
        "dropout": Value::Null,
        "unk_token": unk.map(Value::from).unwrap_or(Value::Null),
        "continuing_subword_prefix": Value::Null,
        "end_of_word_suffix": Value::Null,
        "fuse_unk": true,
        "byte_fallback": true,
        "vocab": Value::Object(vocab),
        "merges": Value::Array(merge_pairs),
    });
    // pre_tokenizer / post_processor are null: the engine encodes with add_special=false and renders
    // BOS via the chat template, so HF's TemplateProcessing(BOS) is not needed for the engine path.
    let tokenizer_json = assemble(added, normalizer, Value::Null, Value::Null, decoder, model);
    let tokenizer_config_json = build_tokenizer_config(g, &tokens);

    Ok(TokenizerOutcome::Reconstructed(Box::new(ReconstructedTokenizer {
        tokenizer_json,
        tokenizer_config_json,
        kind: format!("llama SentencePiece BPE (merges={merge_source})"),
    })))
}

/// `model.vocab`: every token by its id. The GGUF token list has no duplicate strings (verified on
/// the SmolLM2/Qwen/Llama vocabs, including reserved slots and byte-fallback tokens), so a flat map
/// is exact and lossless; a duplicate would silently collapse two ids, so it's rejected.
fn build_vocab(tokens: &[&str]) -> std::result::Result<Map<String, Value>, String> {
    let mut vocab = Map::with_capacity(tokens.len());
    for (id, tok) in tokens.iter().enumerate() {
        if vocab.insert((*tok).to_string(), json!(id)).is_some() {
            return Err(format!("duplicate token {tok:?} in GGUF vocab — cannot build a 1:1 vocab"));
        }
    }
    Ok(vocab)
}

/// `model.merges`: each GGUF merge is `"<left> <right>"`; neither byte-level nor SPM pieces contain a
/// literal space (space is encoded as `Ġ`/`▁`), so splitting on the first space recovers the pair.
fn parse_merges(merges: &[&str]) -> std::result::Result<Vec<Value>, String> {
    merges
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| json!([a, b]))
                .ok_or_else(|| format!("malformed GGUF merge {m:?} (no space separator)"))
        })
        .collect()
}

/// `added_tokens`: control / user-defined / unknown tokens become added tokens (matched atomically,
/// ahead of BPE). `special` follows HF: control & unknown are special; user-defined are not (e.g.
/// Qwen's `<think>`/`<tool_call>`). Byte-fallback (`BYTE`) and normal tokens stay in the plain vocab.
fn build_added_tokens(tokens: &[&str], token_types: &Option<Vec<i64>>) -> Vec<Value> {
    let mut added = Vec::new();
    if let Some(types) = token_types {
        for (id, tok) in tokens.iter().enumerate() {
            let special = match types.get(id).copied().unwrap_or(0) {
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
    added
}

/// Assemble the top-level `tokenizer.json` document from its parts.
fn assemble(
    added: Vec<Value>,
    normalizer: Value,
    pre_tokenizer: Value,
    post_processor: Value,
    decoder: Value,
    model: Value,
) -> Value {
    json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": decoder,
        "model": model,
    })
}

/// Shorthand for a non-fatal "couldn't reconstruct" outcome.
fn unsupported(reason: &str) -> TokenizerOutcome {
    TokenizerOutcome::Unsupported(reason.to_string())
}

/// `(normalizer, decoder)` for the SentencePiece BPE family — taken verbatim from a Llama-2/Mistral
/// `tokenizer.json`. The normalizer turns spaces into `▁` (and prepends one if `add_space_prefix`);
/// the decoder reverses it, with `ByteFallback`+`Fuse` reassembling `<0xNN>` byte tokens.
fn spm_components(add_space_prefix: bool) -> (Value, Value) {
    let mut normalizers = Vec::new();
    if add_space_prefix {
        normalizers.push(json!({ "type": "Prepend", "prepend": SPM_SPACE }));
    }
    normalizers.push(json!({ "type": "Replace", "pattern": { "String": " " }, "content": SPM_SPACE }));
    let normalizer = json!({ "type": "Sequence", "normalizers": normalizers });
    let decoder = json!({
        "type": "Sequence",
        "decoders": [
            { "type": "Replace", "pattern": { "String": SPM_SPACE }, "content": " " },
            { "type": "ByteFallback" },
            { "type": "Fuse" },
            { "type": "Strip", "content": " ", "start": 1, "stop": 0 }
        ]
    });
    (normalizer, decoder)
}

/// Derive BPE merges from per-token scores when a GGUF omits `tokenizer.ggml.merges`, porting HF's
/// `SentencePieceExtractor.extract(vocab_scores)`: for each piece, every split into two in-vocab
/// pieces is a candidate merge (local-sorted by the halves' ids); all candidates are then globally
/// stable-sorted by the produced piece's score, descending. Yields the same merge *set* HF would; the
/// tie order can differ from a GGUF's stored list (which is why stored merges are preferred).
fn derive_merges_from_scores(tokens: &[&str], scores: &[f32]) -> Vec<Value> {
    use std::collections::HashMap;
    let vocab: HashMap<&str, usize> = tokens.iter().enumerate().map(|(i, t)| (*t, i)).collect();
    // (score, left, right) — score drives the global order, ties keep insertion order.
    let mut merges: Vec<(f32, &str, &str)> = Vec::new();
    for (id, piece) in tokens.iter().enumerate() {
        let score = scores.get(id).copied().unwrap_or(0.0);
        let mut local: Vec<(usize, usize, &str, &str)> = Vec::new();
        // Split at each internal char boundary; both halves must be in the vocab.
        for (b, _) in piece.char_indices().skip(1) {
            let (l, r) = piece.split_at(b);
            if let (Some(&il), Some(&ir)) = (vocab.get(l), vocab.get(r)) {
                local.push((il, ir, l, r));
            }
        }
        local.sort_by_key(|&(il, ir, _, _)| (il, ir));
        merges.extend(local.into_iter().map(|(_, _, l, r)| (score, l, r)));
    }
    // Stable sort by score descending (matches HF's `reverse=True` stable sort).
    merges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    merges.into_iter().map(|(_, l, r)| json!([l, r])).collect()
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

/// Read a GGUF metadata array of floats as `f32` (e.g. `tokenizer.ggml.scores`).
fn f32_array(g: &GgufFile, key: &str) -> Option<Vec<f32>> {
    let arr = g.meta(key)?.as_array()?;
    arr.iter().map(|v| v.as_f64().map(|f| f as f32)).collect()
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
        fn f32_array(mut self, k: &str, vals: &[f32]) -> Self {
            self.key(k);
            self.meta.extend_from_slice(&9u32.to_le_bytes()); // T_ARRAY
            self.meta.extend_from_slice(&6u32.to_le_bytes()); // elem T_FLOAT32
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
    fn unigram_t5_model_is_reported_unsupported_not_guessed() {
        // Unigram/T5-style SPM genuinely needs the precompiled charsmap (not in GGUF) — don't guess.
        let g = Builder::new().string("tokenizer.ggml.model", "t5").build();
        match reconstruct(&g).unwrap() {
            TokenizerOutcome::Unsupported(reason) => assert!(reason.contains("Unigram")),
            other => panic!("expected Unsupported, got {}", outcome_name(&other)),
        }
    }

    #[test]
    fn spm_reconstructs_byte_fallback_bpe_from_merges() {
        // Llama-style SPM: <unk>(unknown), <s>/</s>(control), normal pieces, a byte token, + merges.
        let g = Builder::new()
            .string("tokenizer.ggml.model", "llama")
            .str_array(
                "tokenizer.ggml.tokens",
                &["<unk>", "<s>", "</s>", "\u{2581}", "h", "i", "\u{2581}h", "<0x0A>"],
            )
            .i32_array("tokenizer.ggml.token_type", &[2, 3, 3, 1, 1, 1, 1, 6])
            .str_array("tokenizer.ggml.merges", &["\u{2581} h"])
            .u32("tokenizer.ggml.unknown_token_id", 0)
            .u32("tokenizer.ggml.bos_token_id", 1)
            .u32("tokenizer.ggml.eos_token_id", 2)
            .build();
        let t = reconstructed(&g);
        let tj = &t.tokenizer_json;
        assert!(t.kind.contains("merges=merges"));

        // BPE model with SPM flags: byte_fallback + fuse_unk + unk_token.
        assert_eq!(tj["model"]["type"], "BPE");
        assert_eq!(tj["model"]["byte_fallback"], true);
        assert_eq!(tj["model"]["fuse_unk"], true);
        assert_eq!(tj["model"]["unk_token"], "<unk>");
        assert_eq!(tj["model"]["merges"], serde_json::json!([["\u{2581}", "h"]]));

        // The byte token stays in the plain vocab (byte_fallback handles it), NOT added_tokens.
        assert_eq!(tj["model"]["vocab"]["<0x0A>"], 7);
        let added = tj["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 3, "only <unk>/<s>/</s> are added tokens");
        assert!(added.iter().all(|a| a["content"] != "<0x0A>"));

        // SPM normalizer (Prepend ▁ + Replace space→▁) and ByteFallback decoder.
        assert_eq!(tj["normalizer"]["normalizers"][0]["type"], "Prepend");
        assert_eq!(tj["normalizer"]["normalizers"][0]["prepend"], "\u{2581}");
        let decoders = tj["decoder"]["decoders"].as_array().unwrap();
        assert!(decoders.iter().any(|d| d["type"] == "ByteFallback"));
        assert!(tj["pre_tokenizer"].is_null());

        // tokenizer_config carries the eos string (id 2 -> "</s>") + unk.
        assert_eq!(t.tokenizer_config_json["eos_token"], "</s>");
        assert_eq!(t.tokenizer_config_json["unk_token"], "<unk>");
    }

    #[test]
    fn spm_derives_merges_from_scores_when_absent() {
        // No `tokenizer.ggml.merges` -> derive from scores. Hand vocab where the merges are
        // determinable: scores descending so "ab" (-4) and "bc" (-5) precede "abc"'s two splits (-6).
        let g = Builder::new()
            .string("tokenizer.ggml.model", "llama")
            .str_array("tokenizer.ggml.tokens", &["<unk>", "a", "b", "c", "ab", "bc", "abc"])
            .i32_array("tokenizer.ggml.token_type", &[2, 1, 1, 1, 1, 1, 1])
            .f32_array("tokenizer.ggml.scores", &[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
            .u32("tokenizer.ggml.unknown_token_id", 0)
            .build();
        let t = reconstructed(&g);
        assert!(t.kind.contains("merges=derived"));
        // Global stable sort by score desc; "abc"'s splits local-sorted by (left_id,right_id):
        // ("a"=1,"bc"=5) before ("ab"=4,"c"=3).
        assert_eq!(
            t.tokenizer_json["model"]["merges"],
            serde_json::json!([["a", "b"], ["b", "c"], ["a", "bc"], ["ab", "c"]])
        );
    }

    #[test]
    fn spm_without_merges_or_scores_is_unsupported() {
        let g = Builder::new()
            .string("tokenizer.ggml.model", "llama")
            .str_array("tokenizer.ggml.tokens", &["<unk>", "a", "b"])
            .build();
        assert!(matches!(reconstruct(&g).unwrap(), TokenizerOutcome::Unsupported(_)));
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
