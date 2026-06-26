//! JoyCaption — the LLaVA vision-language model, served through the engine's multimodal contract
//! (story 7157).
//!
//! `fancyfeast/llama-joycaption-beta-one-hf-llava` is a `LlavaForConditionalGeneration`: a SigLIP2
//! vision tower ([`crate::models::SiglipVisionTower`]) encodes the image, a two-layer GELU MLP
//! projector lifts the penultimate-layer patch features into the language hidden size, those 729
//! projected rows replace the expanded image-token placeholders in the prompt embeddings, and a
//! Llama-3.1 8B decoder ([`crate::models::CausalLm`], reused as-is) generates the caption.
//!
//! Only the **model** lives here — the vision tower, projector, image splice, the model's LLaVA
//! chat-input format, and generation (repetition penalty + stop tokens + cancellation). The
//! caption-type product templates stay with the consumer; this crate depends on nothing gen-ai.
//!
//! Numerics mirror the reference engine for output parity: the vision tower + projector run in f32
//! against the f32-preprocessed pixels (MLX promotes the bf16 weights), then the projected features
//! are cast to bf16 and spliced into the bf16 token embeddings before the bf16 Llama decode.

use std::path::Path;

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};
use serde_json::Value;

use core_llm::{
    Channel, Content, Error as CoreError, FinishReason as CoreFinish, LoadSpec, Result as CoreResult,
    Sampling, StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor,
    TextLlmOutput, TextLlmRequest, Tokenizer, Usage,
};

use crate::config::ModelConfig;
use crate::decode::{CancelFlag, FinishReason};
use crate::error::{Error, Result};
use crate::image::SiglipImageProcessor;
use crate::models::siglip::{select_vision_feature, SiglipVisionConfig, SiglipVisionTower};
use crate::models::CausalLm;
use crate::primitives::nn::{gelu, linear};
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};
use crate::primitives::{input_ids, Weights};

/// The registry id of the JoyCaption provider.
pub const PROVIDER_ID: &str = "mlx-joycaption";

/// HF `image_token_index` — the single placeholder token expanded to the image rows.
pub const IMAGE_TOKEN_ID: i32 = 128077;
/// Number of SigLIP patch tokens spliced in for one image (27×27).
pub const IMAGE_SEQ_LENGTH: usize = 729;
/// The SigLIP hidden-state layer the projector reads (HF `vision_feature_layer`).
pub const VISION_FEATURE_LAYER: i32 = -2;

/// Generation stops on any of Llama-3's end tokens (`<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`).
pub const STOP_TOKENS: &[i32] = &[128001, 128008, 128009];

/// JoyCaption's default system prompt.
pub const SYSTEM_PROMPT: &str = "You are a helpful image captioner.";
const CUTTING_KNOWLEDGE_DATE: &str = "December 2023";
const DEFAULT_DATE_STRING: &str = "26 July 2024";
const IMAGE_TOKEN: &str = "<|reserved_special_token_69|>";
const IMAGE_START_TOKEN: &str = "<|reserved_special_token_70|>";
const IMAGE_END_TOKEN: &str = "<|reserved_special_token_71|>";

/// Build the model's single-turn LLaVA chat input with the default system prompt.
pub fn build_chat_text(prompt: &str) -> String {
    build_chat_text_with_system(prompt, SYSTEM_PROMPT, DEFAULT_DATE_STRING, true)
}

/// Build the model's LLaVA chat input: a Llama-3 header-wrapped system + user turn, with the image
/// markers (`start | image | end`) prepended to the user text. Mirrors the model's processor format
/// exactly (it is the model's input contract, not product policy).
pub fn build_chat_text_with_system(
    prompt: &str,
    system_prompt: &str,
    date_string: &str,
    add_generation_prompt: bool,
) -> String {
    let user_prompt = prompt.replace(IMAGE_TOKEN, "");
    let user_prompt = user_prompt.trim_start();
    let mut text = format!(
        "<|start_header_id|>system<|end_header_id|>\n\n\
         Cutting Knowledge Date: {CUTTING_KNOWLEDGE_DATE}\n\
         Today Date: {date_string}\n\n\
         {system_prompt}<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\n\
         {IMAGE_START_TOKEN}{IMAGE_TOKEN}{IMAGE_END_TOKEN}{user_prompt}<|eot_id|>"
    );
    if add_generation_prompt {
        text.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    text
}

/// The LLaVA multimodal projector: `linear_2(gelu(linear_1(x)))`, both layers with bias.
pub struct LlavaProjector {
    linear1_w: Array,
    linear1_b: Array,
    linear2_w: Array,
    linear2_b: Array,
}

impl LlavaProjector {
    /// Load HF `multi_modal_projector.{linear_1,linear_2}.{weight,bias}` (cast to bf16).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let bf16 = |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(Dtype::Bfloat16)?) };
        Ok(Self {
            linear1_w: bf16(format!("{prefix}.linear_1.weight"))?,
            linear1_b: bf16(format!("{prefix}.linear_1.bias"))?,
            linear2_w: bf16(format!("{prefix}.linear_2.weight"))?,
            linear2_b: bf16(format!("{prefix}.linear_2.bias"))?,
        })
    }

    /// Project SigLIP features `[b, seq, 1152]` to language features `[b, seq, hidden]`. The f32
    /// input promotes the bf16 weights to f32, matching the reference.
    pub fn forward(&self, features: &Array) -> Result<Array> {
        let h = gelu(&linear(features, &self.linear1_w, Some(&self.linear1_b))?)?;
        linear(&h, &self.linear2_w, Some(&self.linear2_b))
    }
}

/// HF LLaVA prompt expansion: each `IMAGE_TOKEN_ID` becomes `IMAGE_SEQ_LENGTH` placeholders so the
/// projected image rows replace them one-for-one.
pub fn expand_image_tokens(ids: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(ids.len() + IMAGE_SEQ_LENGTH.saturating_sub(1));
    for &id in ids {
        if id == IMAGE_TOKEN_ID {
            out.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, IMAGE_SEQ_LENGTH));
        } else {
            out.push(id);
        }
    }
    out
}

/// Gather index that replaces each image-token row with the next projected image row: text position
/// `p` keeps row `p`; the `k`-th image token maps to row `n_text + k` (the appended features).
fn image_gather_index(ids: &[i32], n_vis: usize, n_text: usize) -> Result<Vec<i32>> {
    if ids.len() != n_text {
        return Err(Error::Msg(format!(
            "joycaption splice: ids length {} != embedding rows {n_text}",
            ids.len()
        )));
    }
    let count = ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    if count != n_vis {
        return Err(Error::Msg(format!(
            "joycaption splice: {count} image tokens != {n_vis} projected image rows"
        )));
    }
    let mut out = Vec::with_capacity(n_text);
    let mut vi = 0i32;
    for (p, &id) in ids.iter().enumerate() {
        if id == IMAGE_TOKEN_ID {
            out.push(n_text as i32 + vi);
            vi += 1;
        } else {
            out.push(p as i32);
        }
    }
    Ok(out)
}

/// Replace the image-token rows of `embeds` (`[b, s, h]`) with `features` (`[b, n_vis, h]` or
/// `[n_vis, h]`), keeping all other rows. `expanded_ids` must already be image-token-expanded.
fn splice_image_features(embeds: &Array, expanded_ids: &[i32], features: &Array) -> Result<Array> {
    let sh = embeds.shape();
    let (b, s, h) = (sh[0], sh[1], sh[2]);
    let n_text = b * s;
    let fsh = features.shape();
    let feat = match *fsh {
        [fb, fs, fh] if fb == b && fh == h => features.reshape(&[fb * fs, h])?,
        [fs, fh] if fh == h => features.reshape(&[fs, h])?,
        _ => {
            return Err(Error::Msg(format!(
                "joycaption splice: features must be [b, n_vis, {h}] or [n_vis, {h}], got {fsh:?}"
            )))
        }
    };
    let n_vis = feat.shape()[0];
    let gather = image_gather_index(expanded_ids, n_vis as usize, n_text as usize)?;
    let embeds_flat = embeds.reshape(&[n_text, h])?;
    let src = concatenate_axis(&[&embeds_flat, &feat], 0)?;
    let idx = Array::from_slice(&gather, &[n_text]);
    Ok(src.take_axis(&idx, 0)?.reshape(&[b, s, h])?)
}

/// The result of a caption generation.
#[derive(Clone, Debug)]
pub struct JoyGeneration {
    /// Generated token ids (excludes the prompt and any stop token).
    pub tokens: Vec<i32>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// The loaded JoyCaption VLM: vision tower, projector, language decoder, and image preprocessor.
pub struct JoyCaptionModel {
    vision: SiglipVisionTower,
    projector: LlavaProjector,
    language: CausalLm,
    processor: SiglipImageProcessor,
}

impl JoyCaptionModel {
    /// Load from a `LlavaForConditionalGeneration` snapshot directory. Parses the nested
    /// `text_config` for the Llama decoder and loads the LLaVA-prefixed weight tree.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("joycaption config.json: {e}")))?;
        let tc = v
            .get("text_config")
            .ok_or_else(|| Error::Config("joycaption: config.json has no text_config".into()))?;
        let llama_cfg = ModelConfig::from_json(tc)?;

        let w = Weights::from_dir(dir)?;
        let language = CausalLm::from_weights(&w, "language_model", llama_cfg)?;
        let vision =
            SiglipVisionTower::from_weights(&w, "vision_tower.vision_model", SiglipVisionConfig::default())?;
        let projector = LlavaProjector::from_weights(&w, "multi_modal_projector")?;
        Ok(Self {
            vision,
            projector,
            language,
            processor: SiglipImageProcessor::default(),
        })
    }

    /// The language config.
    pub fn language_config(&self) -> &ModelConfig {
        self.language.config()
    }

    /// Encode `pixels` (interleaved RGB8, `width*height*3` bytes) into projected image features
    /// `[1, IMAGE_SEQ_LENGTH, hidden]` (f32).
    pub fn image_features(&self, pixels: &[u8], width: usize, height: usize) -> Result<Array> {
        let pix = self.processor.preprocess(pixels, width, height)?;
        let out = self.vision.forward(&pix)?;
        let feat = select_vision_feature(&out, VISION_FEATURE_LAYER)?;
        self.projector.forward(&feat)
    }

    /// Generate a caption from a tokenized prompt (containing a single [`IMAGE_TOKEN_ID`]) and the
    /// projected image features. Emits each token through `on_token(id, step)`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt_ids: &[i32],
        image_features: &Array,
        params: &SamplingParams,
        max_new_tokens: usize,
        seed: Option<u64>,
        stop_tokens: &[i32],
        cancel: &CancelFlag,
        on_token: &mut dyn FnMut(i32, usize),
    ) -> Result<JoyGeneration> {
        if prompt_ids.is_empty() {
            return Err(Error::Msg("joycaption: empty prompt".into()));
        }
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // Splice the (bf16) image rows into the (bf16) token embeddings, then decode in bf16.
        let expanded = expand_image_tokens(prompt_ids);
        let ids_arr = input_ids(&expanded);
        let embeds = self.language.embed(&ids_arr)?;
        let feat_bf16 = image_features.as_dtype(Dtype::Bfloat16)?;
        let spliced = splice_image_features(&embeds, &expanded, &feat_bf16)?;

        let mut cache = self.language.new_cache();
        let mut rng = SplitMix64::new(seed.unwrap_or_else(crate::decode::stream::default_seed));
        let mut history = expanded.clone();
        let mut generated: Vec<i32> = Vec::new();
        let prompt_len = expanded.len() as i32;
        let mut logits = self.language.decode_logits_from_embeds(&spliced, &mut cache, 0)?;
        let mut finish = FinishReason::MaxTokens;

        for step in 0..max_new_tokens {
            if cancel.is_cancelled() {
                finish = FinishReason::Cancelled;
                break;
            }
            let next = sample(&logits, &history, params, &mut rng, None)?;
            if stop_tokens.contains(&next) {
                finish = FinishReason::StopToken;
                break;
            }
            on_token(next, step);
            generated.push(next);
            history.push(next);
            if step + 1 == max_new_tokens {
                break;
            }
            let tok = input_ids(&[next]);
            logits = self.language.decode_logits(&tok, &mut cache, prompt_len + step as i32)?;
        }

        Ok(JoyGeneration {
            tokens: generated,
            finish_reason: finish,
        })
    }
}

/// JoyCaption served as a multimodal [`core_llm::TextLlm`] provider.
pub struct JoyCaptionProvider {
    descriptor: TextLlmDescriptor,
    model: JoyCaptionModel,
    tokenizer: Tokenizer,
}

impl JoyCaptionProvider {
    /// Load from a snapshot directory (config.json + tokenizer.json + shards).
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.quantize.is_some() {
            return Err(CoreError::Unsupported(
                "joycaption: load-time quantization is not supported".into(),
            ));
        }
        let dir = Path::new(&spec.source);
        let model = JoyCaptionModel::from_dir(dir).map_err(to_core)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        Ok(Self {
            descriptor: descriptor(),
            model,
            tokenizer,
        })
    }

    /// Render the request's messages into the model's LLaVA chat prompt + the single image.
    fn build_inputs<'a>(&self, req: &'a TextLlmRequest) -> CoreResult<(String, &'a core_llm::ImageRef)> {
        let mut image = None;
        let mut user_text = String::new();
        let mut system = None;
        for msg in &req.messages {
            for c in &msg.content {
                if let Content::Image(img) = c {
                    if image.is_some() {
                        return Err(CoreError::Unsupported(
                            "joycaption: exactly one image is supported".into(),
                        ));
                    }
                    image = Some(img);
                }
            }
            match msg.role {
                core_llm::Role::System => system = Some(msg.text_content()),
                _ => {
                    let t = msg.text_content();
                    if !t.is_empty() {
                        if !user_text.is_empty() {
                            user_text.push(' ');
                        }
                        user_text.push_str(&t);
                    }
                }
            }
        }
        let image = image.ok_or_else(|| CoreError::InvalidRequest("joycaption: request has no image".into()))?;
        let system = system.unwrap_or_else(|| SYSTEM_PROMPT.to_string());
        Ok((build_chat_text_with_system(&user_text, &system, DEFAULT_DATE_STRING, true), image))
    }
}

impl TextLlm for JoyCaptionProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
        self.descriptor.capabilities.validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
    ) -> CoreResult<TextLlmOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled);
        }

        let (chat_text, image) = self.build_inputs(req)?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&chat_text, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        // Prompt length the engine sees = expanded (image token → IMAGE_SEQ_LENGTH rows).
        let prompt_len = expand_image_tokens(&prompt_ids).len() as u32;

        let features = self
            .model
            .image_features(&image.pixels, image.width as usize, image.height as usize)
            .map_err(to_core)?;

        let params = map_sampling(&req.sampling);
        let max_new = req.max_new_tokens as usize;

        // Stream contract token events via incremental detokenization (re-decode, emit new suffix).
        let tokenizer = &self.tokenizer;
        let mut acc: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        let mut on_token = |id: i32, step: usize| {
            acc.push(id as u32);
            if let Ok(text) = tokenizer.decode(&acc, true) {
                if text.len() > shown {
                    let delta = text[shown..].to_string();
                    shown = text.len();
                    on_event(CoreEvent::Token {
                        id: id as u32,
                        text: delta,
                        index: step,
                        channel: Channel::Content, // captioner: no reasoning mode
                    });
                }
            }
        };
        let gen = self
            .model
            .generate(
                &prompt_ids,
                &features,
                &params,
                max_new,
                req.seed,
                STOP_TOKENS,
                &req.cancel,
                &mut on_token,
            )
            .map_err(to_core)?;

        let gen_u32: Vec<u32> = gen.tokens.iter().map(|&i| i as u32).collect();
        let text = tokenizer.decode(&gen_u32, true)?;
        let finish = map_finish(gen.finish_reason);
        let usage = Usage {
            prompt_tokens: prompt_len,
            generated_tokens: gen.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            text,
            thinking: None,
            tool_calls: Vec::new(),
            usage,
            finish_reason: Some(finish),
        })
    }
}

/// The JoyCaption provider descriptor (constructible without weights; used for registration).
pub fn descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.to_string(),
        family: "joycaption".to_string(),
        backend: "mlx".to_string(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            supports_vision: true,
            supports_video: false,
            supports_thinking: false,
            supports_tools: false,
            supported_constraints: Vec::new(),
        },
    }
}

fn map_sampling(s: &Sampling) -> SamplingParams {
    SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        repetition_penalty: s.repetition_penalty,
        repetition_context: s.repetition_context,
    }
}

fn map_finish(f: FinishReason) -> CoreFinish {
    match f {
        // `Stopped` (a host stop condition) maps to `Stop` like an EOS id; JoyCaption's loop never
        // produces it, but the mapping stays total.
        FinishReason::StopToken | FinishReason::Stopped => CoreFinish::Stop,
        FinishReason::MaxTokens => CoreFinish::Length,
        FinishReason::Cancelled => CoreFinish::Cancelled,
    }
}

fn to_core(e: Error) -> CoreError {
    match e {
        Error::Canceled => CoreError::Canceled,
        Error::Unsupported(m) => CoreError::Unsupported(m),
        Error::MissingTensor(m) => CoreError::Load(format!("missing tensor: {m}")),
        Error::Config(m) => CoreError::Load(m),
        Error::Io(e) => CoreError::Io(e),
        other => CoreError::backend(other),
    }
}

// Register `mlx-joycaption` into core-llm's provider registry at link time.
inventory::submit! {
    core_llm::TextLlmRegistration {
        descriptor,
        load: load_registered,
        can_load,
        // No per-snapshot vision distinction: this dedicated vision provider's static descriptor
        // already declares `supports_vision=true` for every (LLaVA) snapshot its `can_load` claims,
        // so the core-llm gate's static fallback is correct. (`None` ⇒ unchanged prior behavior.)
        weightless_vision: None,
    }
}

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(JoyCaptionProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `mlx-joycaption` vision provider serve the
/// snapshot at `spec.source`? Reads **only** `config.json` and keys on the LLaVA structural
/// signature — a nested `text_config` (the language decoder) plus a `vision_config` (the SigLIP
/// tower) — which `JoyCaptionModel::from_dir` requires. Never opens a safetensors shard.
pub fn can_load(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() { dir.join("config.json") } else { dir.to_path_buf() };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    can_load_value(&v)
}

/// Pure LLaVA-signature decision over a parsed `config.json` (split out for unit testing). Requires a
/// nested `text_config` (Llama decoder) + a `vision_config` (SigLIP tower) **and** the LLaVA / SigLIP
/// signature, so a *non*-LLaVA VLM (e.g. a Qwen-VL `qwen3_5` wrapper) is NOT claimed here and is
/// served text-only by `mlx-llama` instead (sc-7626).
pub(crate) fn can_load_value(v: &Value) -> bool {
    if v.get("text_config").is_none() || v.get("vision_config").is_none() {
        return false;
    }
    let arch_is_llava = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .map(|s| s.to_lowercase().contains("llava"))
        .unwrap_or(false);
    let vision_is_siglip = v
        .get("vision_config")
        .and_then(|vc| vc.get("model_type"))
        .and_then(|s| s.as_str())
        .map(|s| s.to_lowercase().contains("siglip"))
        .unwrap_or(false);
    arch_is_llava || vision_is_siglip
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn can_load_claims_llava_not_qwen_vl() {
        // Real LLaVA (JoyCaption) signature: claimed.
        let llava = json!({
            "architectures": ["LlavaForConditionalGeneration"], "model_type": "llava",
            "text_config": { "model_type": "llama" },
            "vision_config": { "model_type": "siglip_vision_model" }
        });
        assert!(can_load_value(&llava));
        // Qwen-VL wrapper (Qwen3.6): NOT a LLaVA snapshot — declined so it routes to mlx-llama.
        let qwen_vl = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5", "depth": 27 }
        });
        assert!(!can_load_value(&qwen_vl));
        // A plain text model (no vision_config) is never a JoyCaption candidate.
        assert!(!can_load_value(&json!({ "model_type": "qwen3" })));
    }

    #[test]
    fn chat_text_matches_llava_format() {
        let text = build_chat_text("Write a caption.");
        assert_eq!(
            text,
            "<|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\n\
             Today Date: 26 July 2024\n\n\
             You are a helpful image captioner.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n\
             <|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>Write a caption.<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn expand_replaces_image_token() {
        let ids = [1, IMAGE_TOKEN_ID, 2];
        let expanded = expand_image_tokens(&ids);
        assert_eq!(expanded.len(), 2 + IMAGE_SEQ_LENGTH);
        assert_eq!(expanded[0], 1);
        assert!(expanded[1..1 + IMAGE_SEQ_LENGTH].iter().all(|&t| t == IMAGE_TOKEN_ID));
        assert_eq!(*expanded.last().unwrap(), 2);
    }

    #[test]
    fn gather_index_maps_image_rows_to_appended_features() {
        // ids [10, IMG, IMG, 11], 4 text rows, 2 image rows appended at 4,5.
        let got = image_gather_index(&[10, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 11], 2, 4).unwrap();
        assert_eq!(got, vec![0, 4, 5, 3]);
    }

    #[test]
    fn gather_index_rejects_count_mismatch() {
        assert!(image_gather_index(&[IMAGE_TOKEN_ID, 7], 2, 2).is_err());
    }

    #[test]
    fn splice_replaces_only_image_rows() {
        // [1,4,2]: rows for ids [5, IMG, IMG, 6]; features [1,2,2].
        let embeds = Array::from_slice(&[1.0f32, 1.0, 10.0, 10.0, 20.0, 20.0, 2.0, 2.0], &[1, 4, 2]);
        let ids = [5, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 6];
        let features = Array::from_slice(&[100.0f32, 101.0, 200.0, 201.0], &[1, 2, 2]);
        let got = splice_image_features(&embeds, &ids, &features).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[1.0, 1.0, 100.0, 101.0, 200.0, 201.0, 2.0, 2.0]);
    }

    #[test]
    fn descriptor_declares_vision() {
        let d = descriptor();
        assert_eq!(d.id, PROVIDER_ID);
        assert!(d.capabilities.supports_vision);
    }
}
