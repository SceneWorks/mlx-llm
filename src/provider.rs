//! The `core-llm` provider: a generic Llama model exposed through the backend-neutral contract.
//!
//! This is the mlx-llm half of story 7154 — it implements [`core_llm::TextLlm`] by wrapping the
//! [`CausalLm`] decoder, a [`core_llm::Tokenizer`], and a chat template, driving the internal
//! streaming decode loop and translating its token events into contract [`StreamEvent`]s (with
//! incremental detokenization). It registers into [`core_llm::registry`] under the id `mlx-llama`.
//!
//! The chat template is the model's own `chat_template` from `tokenizer_config.json` (rendered via
//! `core_llm::JinjaChatTemplate`, story 7164), falling back to the typed [`Llama3Template`] when a
//! snapshot ships no `tokenizer_config.json`.

use std::cell::OnceCell;
use std::path::Path;

use core_llm::{
    Channel, ChatTemplate, Constraint, ConstraintDecodeTable, Content, Error as CoreError,
    FinishReason as CoreFinish, ImageRef, JinjaChatTemplate, JsonConstraint, Llama3Template, LoadSpec,
    Message, Quantize, RenderOptions, Result as CoreResult, Sampling, StopMatcher,
    StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput,
    TextLlmRequest, ThinkingSegmenter, Tokenizer, ToolCallSegmenter, Usage, VideoRef,
};

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_from_prefill, generate_with, ConstraintMask, Decode, FinishReason, GenerationConfig,
    StreamEvent,
};
use crate::image::Qwen35ImageProcessor;
use crate::models::{
    CausalLm, Qwen35Config, Qwen35Model, Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{input_ids, Weights};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

/// The registry id of this provider.
pub const PROVIDER_ID: &str = "mlx-llama";

/// The loaded decoder, dispatched by architecture. The generic softmax-attention decoders share
/// [`CausalLm`]; Qwen3.6 (`qwen3_5`) is the hybrid linear-attention/full-attention decoder. Both
/// implement [`Decode`], so the generation loop is identical.
enum Decoder {
    Causal(CausalLm),
    Qwen35(Qwen35Model),
}

impl Decode for Decoder {
    fn make_cache(&self) -> Box<dyn KvCache> {
        match self {
            Decoder::Causal(m) => m.make_cache(),
            Decoder::Qwen35(m) => m.make_cache(),
        }
    }

    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> crate::error::Result<Array> {
        match self {
            Decoder::Causal(m) => m.step(input_ids, cache, offset),
            Decoder::Qwen35(m) => m.step(input_ids, cache, offset),
        }
    }
}

impl Decoder {
    fn is_quantized(&self) -> bool {
        match self {
            Decoder::Causal(m) => m.is_quantized(),
            Decoder::Qwen35(m) => m.is_quantized(),
        }
    }

    /// The decoder as the backend-neutral multimodal seam. Both backbones implement [`VlmDecode`]
    /// (the Qwen3.6 hybrid and the generic Qwen3-VL causal decoder), so the provider drives the
    /// image/video prefill + decode through this one trait object rather than forking on the concrete
    /// decoder type.
    fn as_vlm(&self) -> &dyn VlmDecode {
        match self {
            Decoder::Causal(m) => m,
            Decoder::Qwen35(m) => m,
        }
    }
}

/// The Qwen-VL vision side of the provider: the ViT tower, the image preprocessor, and the
/// multimodal token ids needed to expand placeholders and assign M-RoPE positions. Present when the
/// loaded `qwen3_5` (Qwen3.6) or `qwen3_vl` (Qwen3-VL) checkpoint carries `model.visual.*`. The two
/// share the identical Qwen3-VL ViT tower (`Qwen3VLVisionModel == Qwen35VisionModel`); only the
/// decoder prefill differs ([`Decoder::Qwen35`] vs [`Decoder::Causal`]).
struct Qwen35Vision {
    tower: Qwen35VisionModel,
    processor: Qwen35ImageProcessor,
    image_token_id: i32,
    /// The `<|video_pad|>` placeholder token id (151656 for Qwen3-VL) — the per-frame video
    /// placeholder the processor expands to `frame_seqlen` copies.
    video_token_id: i32,
    spatial_merge_size: i32,
}

impl Qwen35Vision {
    /// Encode one image to its merged patch rows `[n_tokens, hidden]` (the merger output is already
    /// the language hidden size — no separate projector), the per-tap **DeepStack** feature sets
    /// (each `[n_tokens, hidden]`, one per `deepstack_visual_indexes` tap), plus the image's
    /// `grid_thw` (`[1, h, w]` in patch units). `n_tokens = (grid_h/merge)·(grid_w/merge)` is the
    /// placeholder expansion count.
    fn encode(&self, img: &ImageRef) -> CoreResult<(Array, Vec<Array>, [i32; 3])> {
        let (pixels, grid) = self
            .processor
            .preprocess(&img.pixels, img.width as usize, img.height as usize)
            .map_err(to_core)?;
        let out = self.tower.forward_with_deepstack(&pixels, &grid).map_err(to_core)?;
        Ok((out.pooler_output, out.deepstack_features, grid[0]))
    }

    /// Encode one **video** (sampled frames) to its merged patch rows `[grid_t·n_per_frame, hidden]`,
    /// the per-tap DeepStack features, and the `video_grid_thw` (`[grid_t, h, w]`). The ViT tower is
    /// modality-agnostic — it processes the `grid_t` temporal patches as a block-diagonal-masked
    /// frame sequence exactly like multiple images — so this reuses `forward_with_deepstack`. The
    /// per-frame timestamp tokens are rendered separately (Text–Timestamp Alignment); here we only
    /// produce the visual features and the grid.
    fn encode_video(&self, video: &VideoRef) -> CoreResult<(Array, Vec<Array>, [i32; 3])> {
        let frames: Vec<(&[u8], usize, usize)> = video
            .frames
            .iter()
            .map(|f| (f.pixels.as_slice(), f.width as usize, f.height as usize))
            .collect();
        let (pixels, grid) = self.processor.preprocess_video(&frames).map_err(to_core)?;
        let out = self.tower.forward_with_deepstack(&pixels, &[grid]).map_err(to_core)?;
        Ok((out.pooler_output, out.deepstack_features, grid))
    }
}

/// The prepared multimodal prefill: the image-token-expanded prompt ids, the decoder input embeds
/// with image features spliced in, the interleaved M-RoPE position rows + delta, the per-position
/// visual mask (`true` at image-token positions), and the per-tap DeepStack feature sets fused into
/// the first decoder layers.
struct MultimodalPrefill {
    expanded_ids: Vec<i32>,
    embeds: Array,
    positions: (Vec<i32>, Vec<i32>, Vec<i32>, i32),
    visual_pos_mask: Vec<bool>,
    deepstack: Vec<Array>,
}

/// A [`Decode`] wrapper that shifts the RoPE offset by a constant `delta` for the post-prompt
/// continuation of a multimodal decode. Image tokens compress the position cursor, so the text
/// positions that follow the prompt are `cache_len + mrope_delta`, not `cache_len`; the new tokens
/// are text, so a single shifted 1-D position is the correct (interleaved-)M-RoPE position. Drives
/// either backbone through [`VlmDecode`]'s [`Decode`] supertrait — each decoder downcasts its own
/// cache inside `step`, so no concrete-type fork is needed here.
struct Shifted<'a> {
    model: &'a dyn VlmDecode,
    delta: i32,
}

impl Decode for Shifted<'_> {
    fn make_cache(&self) -> Box<dyn KvCache> {
        self.model.make_cache()
    }

    fn step(&self, ids: &Array, cache: &mut dyn KvCache, offset: i32) -> crate::error::Result<Array> {
        self.model.step(ids, cache, offset + self.delta)
    }
}

/// Collect the image blocks of a conversation, in order.
fn collect_images(messages: &[Message]) -> Vec<&ImageRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Image(img) => Some(img),
                Content::Text(_) | Content::Video(_) => None,
            })
        })
        .collect()
}

/// Collect the video blocks of a conversation, in order.
fn collect_videos(messages: &[Message]) -> Vec<&VideoRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Video(v) => Some(v),
                Content::Text(_) | Content::Image(_) => None,
            })
        })
        .collect()
}

/// Compute the Qwen3-VL **merged per-frame timestamps** for a sampled video, mirroring
/// `Qwen3VLProcessor._calculate_timestamps`. The vision encoder folds `temporal_patch_size` frames
/// into one temporal patch, so the per-sample timestamps are padded up to a multiple of
/// `temporal_patch_size` (repeating the last) and then **averaged within each temporal patch**,
/// yielding one timestamp per emitted vision frame (`grid_t = padded / temporal_patch_size`). These
/// are the `<{t:.1f} seconds>` values the Text–Timestamp-Alignment placeholder uses.
fn merged_frame_timestamps(timestamps: &[f32], temporal_patch_size: usize) -> Vec<f32> {
    let tps = temporal_patch_size.max(1);
    let mut ts: Vec<f32> = timestamps.to_vec();
    if ts.is_empty() {
        return ts;
    }
    while !ts.len().is_multiple_of(tps) {
        ts.push(*ts.last().unwrap());
    }
    (0..ts.len())
        .step_by(tps)
        .map(|i| (ts[i] + ts[i + tps - 1]) / 2.0)
        .collect()
}

/// The Text–Timestamp-Alignment placeholder text for one video: per merged frame, a
/// `<{t:.1f} seconds>` timestamp tag followed by `<|vision_start|><|video_pad|><|vision_end|>`
/// (exactly `Qwen3VLProcessor.replace_video_token`). The single `<|video_pad|>` per frame is expanded
/// to `frame_seqlen` copies after tokenizing.
fn video_placeholder_text(video: &VideoRef, temporal_patch_size: usize) -> String {
    let merged = merged_frame_timestamps(&video.timestamps, temporal_patch_size);
    let mut out = String::new();
    for t in merged {
        out.push_str(&format!("<{t:.1} seconds>"));
        out.push_str("<|vision_start|><|video_pad|><|vision_end|>");
    }
    out
}

/// Replace each image/video block with its Qwen-VL placeholder text so the (text-only) chat template
/// renders the vision framing verbatim. Images become `<|vision_start|><|image_pad|><|vision_end|>`
/// (one `image_pad`, expanded to the per-image patch count after tokenizing); videos become the
/// per-frame Text–Timestamp-Alignment string `<{t} seconds><|vision_start|><|video_pad|><|vision_end|>`
/// (one `video_pad` per frame, each expanded to `frame_seqlen` after tokenizing). Keeps the core-llm
/// template contract image/video-free.
fn substitute_vision_placeholders(messages: &[Message], temporal_patch_size: usize) -> Vec<Message> {
    const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    messages
        .iter()
        .map(|m| Message {
            role: m.role,
            content: m
                .content
                .iter()
                .map(|c| match c {
                    Content::Image(_) => Content::text(IMAGE_PLACEHOLDER),
                    Content::Video(v) => Content::text(video_placeholder_text(v, temporal_patch_size)),
                    Content::Text(t) => Content::Text(t.clone()),
                })
                .collect(),
            thinking: m.thinking.clone(),
            tool_calls: m.tool_calls.clone(),
        })
        .collect()
}

/// A generic Llama provider implementing [`core_llm::TextLlm`].
pub struct LlamaProvider {
    descriptor: TextLlmDescriptor,
    model: Decoder,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    stop_tokens: Vec<i32>,
    /// Cached per-vocab decode table for constrained decoding — built once (it decodes the whole
    /// vocabulary) on the first JSON-constrained request, then reused.
    constraint_table: OnceCell<ConstraintDecodeTable>,
    /// The Qwen3.6 vision tower + preprocessor, present iff this is a `qwen3_5` checkpoint carrying
    /// `model.visual.*`. Drives the image path in [`LlamaProvider::generate`].
    vision: Option<Qwen35Vision>,
}

impl LlamaProvider {
    /// Load a provider from a snapshot directory (config.json + tokenizer.json + shards). Dispatches
    /// the decoder architecture from `config.json` (Llama / Mistral / Qwen3) and optionally
    /// quantizes the projections on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        let dir = Path::new(&spec.source);
        let quant = spec.quantize.map(|q| match q {
            Quantize::Q4 => QuantSpec::q4(),
            Quantize::Q8 => QuantSpec::q8(),
        });
        // Read config.json once to dispatch the architecture: the hybrid Qwen3.6 (`qwen3_5`) decoder
        // has its own config/weights path (and `ModelConfig` deliberately rejects it).
        let cfg_value = read_config_value(dir)?;
        let arch = Architecture::from_config(&cfg_value).map_err(to_core)?;
        let weights = Weights::from_dir(dir).map_err(to_core)?;

        let (model, mut descriptor) = if arch == Architecture::Qwen35 {
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for_qwen35(&qcfg);
            // The text decoder nests under `model.language_model` in the VLM-wrapped checkpoint.
            let m = Qwen35Model::from_weights_with(&weights, "model.language_model", qcfg, quant)
                .map_err(to_core)?;
            (Decoder::Qwen35(m), descriptor)
        } else {
            let cfg = ModelConfig::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let m = CausalLm::from_weights_with(&weights, "", cfg, quant).map_err(to_core)?;
            (Decoder::Causal(m), descriptor)
        };

        // Qwen-VL vision: load the ViT tower when the checkpoint carries `model.visual.*` (a wrapped
        // VLM) and the config exposes a `vision_config`. Covers Qwen3.6 (`qwen3_5`) and Qwen3-VL
        // (`qwen3_vl`), which share the identical Qwen3-VL ViT tower. Absent → a text-only checkpoint.
        let vision = if (arch == Architecture::Qwen35 || arch == Architecture::Qwen3Vl)
            && cfg_value.get("vision_config").is_some()
            && weights.get("model.visual.patch_embed.proj.weight").is_some()
        {
            let vcfg = Qwen35VisionConfig::from_json(&cfg_value).map_err(to_core)?;
            let tower = Qwen35VisionModel::from_weights(&weights, "model.visual", vcfg.clone())
                .map_err(to_core)?;
            let image_token_id = cfg_value
                .get("image_token_id")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .unwrap_or(248056);
            // Video tokens (Qwen3-VL): `video_token_id` (`<|video_pad|>`, 151656) plus the
            // vision_start/end framing the Text–Timestamp-Alignment substitution emits per frame. A
            // checkpoint without `video_token_id` in its config does not advertise video.
            let video_token_id = cfg_value
                .get("video_token_id")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32);
            descriptor.capabilities.supports_vision = true;
            descriptor.capabilities.supports_video = video_token_id.is_some();
            Some(Qwen35Vision {
                tower,
                processor: Qwen35ImageProcessor::default(),
                image_token_id,
                // Fall back to the canonical Qwen3-VL id when absent so the field is always valid;
                // `supports_video` already gates whether the video path is reachable.
                video_token_id: video_token_id.unwrap_or(151656),
                spatial_merge_size: vcfg.spatial_merge_size,
            })
        } else {
            None
        };

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let stop_tokens = eos_token_ids(dir);
        let (template, supports_thinking, supports_tools) = load_chat_template(dir);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        Ok(Self {
            descriptor,
            model,
            tokenizer,
            template,
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision,
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// Assemble a provider from already-loaded parts with a default Llama-3 template (used by tests
    /// and converters that don't have a `tokenizer_config.json`).
    pub fn from_parts(model: CausalLm, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        Self {
            descriptor: provider_descriptor(),
            model: Decoder::Causal(model),
            tokenizer,
            template: Box::new(Llama3Template),
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision: None,
        }
    }

    /// Build the multimodal prefill: encode each visual (image or video) in **document order**
    /// (preprocess → ViT → merged rows), expand the rendered `image_pad` / `video_pad` placeholders to
    /// the per-visual / per-frame token counts, splice the features into the token embeds, and compute
    /// the interleaved M-RoPE 3-D positions over the image **and** video grids. `prompt_ids` is the
    /// tokenized prompt (one `image_token_id` per image; one `video_token_id` per frame from the
    /// Text–Timestamp-Alignment placeholders). `messages` is the *original* (un-substituted)
    /// conversation, walked to recover the visual order.
    fn prepare_multimodal(
        &self,
        prompt_ids: &[i32],
        messages: &[Message],
    ) -> CoreResult<MultimodalPrefill> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| CoreError::Load("qwen-vl vision: provider has no vision tower".into()))?;

        // Walk the conversation in document order; encode each visual once, in order, so the
        // concatenated feature buffer lines up one-to-one with the visual placeholder spans of the
        // (image+video) prompt. Image placeholders expand to one count; a video expands to `grid_t`
        // per-frame counts (`frame_seqlen` each), in frame order.
        let mut feats: Vec<Array> = Vec::new();
        let mut image_counts: Vec<usize> = Vec::new();
        let mut video_counts: Vec<usize> = Vec::new();
        let mut image_grids: Vec<[i32; 3]> = Vec::new();
        let mut video_grids: Vec<[i32; 3]> = Vec::new();
        let mut deepstack_by_tap: Vec<Vec<Array>> = Vec::new();
        let merge = vision.spatial_merge_size;

        let mut push_deepstack = |deepstack: Vec<Array>| -> CoreResult<()> {
            if deepstack_by_tap.is_empty() {
                deepstack_by_tap.resize_with(deepstack.len(), Vec::new);
            }
            if deepstack.len() != deepstack_by_tap.len() {
                return Err(CoreError::Load(format!(
                    "qwen-vl vision: inconsistent DeepStack tap count {} != {}",
                    deepstack.len(),
                    deepstack_by_tap.len()
                )));
            }
            for (tap, feature) in deepstack.into_iter().enumerate() {
                deepstack_by_tap[tap].push(feature);
            }
            Ok(())
        };

        for m in messages {
            for c in &m.content {
                match c {
                    Content::Image(img) => {
                        let (f, deepstack, grid) = vision.encode(img)?;
                        image_counts.push(f.shape()[0] as usize);
                        image_grids.push(grid);
                        feats.push(f);
                        push_deepstack(deepstack)?;
                    }
                    Content::Video(video) => {
                        let (f, deepstack, grid) = vision.encode_video(video)?;
                        // One placeholder count per frame: `frame_seqlen = (h/merge)·(w/merge)`.
                        let [gt, gh, gw] = grid;
                        let frame_seqlen = ((gh / merge) * (gw / merge)) as usize;
                        for _ in 0..gt {
                            video_counts.push(frame_seqlen);
                        }
                        video_grids.push(grid);
                        feats.push(f);
                        push_deepstack(deepstack)?;
                    }
                    Content::Text(_) => {}
                }
            }
        }

        // Expand both placeholder tokens to their per-occurrence counts. Each call only touches its
        // own token, so order across the two is preserved and the result interleaves correctly.
        let img_id = vision.image_token_id;
        let vid_id = vision.video_token_id;
        let expanded = crate::models::qwen35::expand_vision_placeholders(prompt_ids, img_id, &image_counts)
            .map_err(to_core)?;
        let expanded =
            crate::models::qwen35::expand_vision_placeholders(&expanded, vid_id, &video_counts)
                .map_err(to_core)?;
        let visual_pos_mask: Vec<bool> =
            expanded.iter().map(|&id| id == img_id || id == vid_id).collect();

        let refs: Vec<&Array> = feats.iter().collect();
        let all_features = match refs.as_slice() {
            [one] => (*one).clone(),
            many => concatenate_axis(many, 0).map_err(|e| to_core(e.into()))?,
        };
        let mut deepstack = Vec::with_capacity(deepstack_by_tap.len());
        for by_visual in deepstack_by_tap {
            let refs: Vec<&Array> = by_visual.iter().collect();
            deepstack.push(match refs.as_slice() {
                [one] => (*one).clone(),
                many => concatenate_axis(many, 0).map_err(|e| to_core(e.into()))?,
            });
        }

        // Embed the expanded ids, splice in the vision features (image+video placeholder rows), and
        // compute interleaved-M-RoPE positions over both image and video grids — through the shared
        // `VlmDecode` seam, identical for whichever decoder powers this VLM (Qwen3.6 hybrid or
        // Qwen3-VL generic-causal).
        let placeholders = [img_id, vid_id];
        let model = self.model.as_vlm();
        let embeds = model.embed_input_ids(&input_ids(&expanded)).map_err(to_core)?;
        let spliced = model
            .splice_vision_features(&embeds, &expanded, &all_features, &placeholders)
            .map_err(to_core)?;
        let positions = model
            .mrope_positions_mm(&expanded, &image_grids, img_id, &video_grids, vid_id, merge)
            .map_err(to_core)?;

        Ok(MultimodalPrefill {
            expanded_ids: expanded,
            embeds: spliced,
            positions,
            visual_pos_mask,
            deepstack,
        })
    }
}

/// Adapts a `core_llm::JsonConstraint` to the engine's [`ConstraintMask`] decode seam.
struct JsonMask<'a>(JsonConstraint<'a>);

impl ConstraintMask for JsonMask<'_> {
    fn allowed(&mut self) -> &[bool] {
        self.0.allowed()
    }
    fn accept(&mut self, token: i32) {
        self.0.accept(token as u32);
    }
}

/// Use the model's own Jinja `chat_template` (from `tokenizer_config.json`, story 7164) when
/// present; otherwise fall back to the typed Llama-3 template. Also reports two template-gated
/// capabilities, detected from the source (not the family, matching the transformers convention):
/// - **thinking** — the template gates an `enable_thinking` kwarg (sc-7585).
/// - **tools** — the template renders tool calls (it mentions `tool_call`), so it has a `tools`
///   section and the model emits parseable `<tool_call>` blocks (sc-7636). Covers the Qwen3.6 XML and
///   the Qwen2.5/Hermes JSON tool templates alike.
fn load_chat_template(dir: &Path) -> (Box<dyn ChatTemplate>, bool, bool) {
    match JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json")) {
        Ok(t) => {
            let supports_thinking = t.source().contains("enable_thinking");
            let supports_tools = t.source().contains("tool_call");
            (Box::new(t), supports_thinking, supports_tools)
        }
        Err(_) => (Box::new(Llama3Template), false, false),
    }
}

/// Run a piece of answer-channel text through the tool-call segmenter when active, returning the
/// plain-content runs to stream (tool-call blocks lifted out + parsed). With no segmenter the text
/// passes straight through, so the non-tools path is byte-identical to before.
fn tool_pieces(seg: &mut Option<ToolCallSegmenter>, text: &str) -> Vec<String> {
    match seg {
        Some(ts) => ts.push(text),
        None => vec![text.to_string()],
    }
}

/// Push one content piece through the stop matcher and emit the released text as a Content token
/// event. Shared by the streaming loop and the end-of-generation tails. `*last_id` / `*emit_index`
/// advance only when text is actually emitted, so the contract's token index stays gap-free across
/// stripped markers and lifted-out tool-call blocks.
#[allow(clippy::too_many_arguments)]
fn emit_content(
    piece: &str,
    id: u32,
    stop_matcher: &mut StopMatcher,
    streamed: &mut String,
    emit_index: &mut usize,
    last_id: &mut u32,
    halt: &std::cell::Cell<bool>,
    on_event: &mut dyn FnMut(CoreEvent),
) {
    let chunk = stop_matcher.push(piece);
    if !chunk.emit.is_empty() {
        streamed.push_str(&chunk.emit);
        *last_id = id;
        on_event(CoreEvent::Token {
            id,
            text: chunk.emit,
            index: *emit_index,
            channel: Channel::Content,
        });
        *emit_index += 1;
    }
    if chunk.stop {
        halt.set(true);
    }
}

impl TextLlm for LlamaProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
    ) -> CoreResult<TextLlmOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled); // typed pre-inference cancel
        }

        // Multimodal (Qwen-VL + image/video content): replace image blocks with the Qwen-VL image
        // placeholder (`<|vision_start|><|image_pad|><|vision_end|>`) and video blocks with the
        // per-frame Text–Timestamp-Alignment placeholder string
        // (`<{t} seconds><|vision_start|><|video_pad|><|vision_end|>` ×frames), so the (vision-free)
        // chat template renders the framing verbatim. The visuals are encoded + spliced after
        // tokenizing, in document order. Text-only requests are unchanged.
        let temporal_patch = self
            .vision
            .as_ref()
            .map(|v| v.processor.temporal_patch_size)
            .unwrap_or(2);
        let (images, videos): (Vec<&ImageRef>, Vec<&VideoRef>) = match &self.vision {
            Some(_) => (collect_images(&req.messages), collect_videos(&req.messages)),
            None => (Vec::new(), Vec::new()),
        };
        let multimodal = !images.is_empty() || !videos.is_empty();
        let substituted;
        let messages: &[Message] = if multimodal {
            substituted = substitute_vision_placeholders(&req.messages, temporal_patch);
            &substituted
        } else {
            &req.messages
        };

        // Render the conversation and tokenize. The template already includes BOS, so encode
        // without auto special tokens. `enable_thinking` (sc-7585) flows into the template kwarg so
        // a no-think (Disabled) request injects the model's empty `<think></think>` generation
        // prompt; Auto omits the kwarg (template default).
        let prompt = self.template.render_with(
            messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.enable_thinking_kwarg(),
                tools: &req.tools,
            },
        )?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();

        // Encode + splice the visuals and compute M-RoPE positions (the placeholder-expanded prompt
        // becomes the effective sequence). `None` on the text-only path.
        let mm = if multimodal {
            Some(self.prepare_multimodal(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let prompt_len = mm.as_ref().map(|m| m.expanded_ids.len()).unwrap_or(prompt_ids.len());

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: map_sampling(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };

        // Structured-output constraint (story 7166): build a JSON mask over the cached decode table.
        let mut json_mask = match req.constraint {
            Some(Constraint::Json) => {
                let table = self
                    .constraint_table
                    .get_or_init(|| self.tokenizer.constraint_decode_table());
                Some(JsonMask(JsonConstraint::new(
                    table,
                    self.stop_tokens.iter().map(|&i| i as u32),
                )))
            }
            None => None,
        };

        // Request `stop` strings (story 7349): a backend-neutral matcher over the decoded text.
        // Matching is in the detokenization seam below (not the token-id loop) because a stop string
        // need not align to a token boundary. When no stops are requested the matcher is a
        // transparent pass-through, so streaming output stays byte-identical to before.
        let mut stop_matcher = StopMatcher::new(req.stop.iter().cloned());
        let stop_active = !stop_matcher.is_empty();
        // A single-threaded latch the detok sink trips on a stop hit; the decode loop reads it after
        // each token via `should_stop` and halts with `FinishReason::Stopped`.
        let halt = std::cell::Cell::new(false);
        // Content emitted as the matchers release it — the result text when stop strings or thinking
        // are active (the plain path still decodes all tokens, byte-identical to before).
        let mut streamed = String::new();
        // Reasoning text, accumulated from the Thinking channel (sc-7585).
        let mut thinking_buf = String::new();
        // Contract token index: a running counter over *emitted* events, not the raw decode step —
        // detok hold-backs and stripped `<think>`/`</think>` marker tokens produce no event, so this
        // stays gap-free (and equals the step in the common one-delta-per-token case).
        let mut emit_index = 0usize;
        let mut last_id = 0u32; // id of the last emitted token, for flushed-tail events

        // A reasoning segmenter when the model advertises a thinking mode: it splits the decoded
        // stream into `<think>…</think>` reasoning vs answer (markers stripped). `None` otherwise, so
        // a non-thinking provider stays on the original single-channel path.
        let thinking_active = self.descriptor.capabilities.supports_thinking;
        let mut segmenter = thinking_active.then(ThinkingSegmenter::default);
        // Some chat templates open the reasoning block *in the prompt* — e.g. Qwen3.6 ends the
        // generation prompt with `<|im_start|>assistant\n<think>\n`, so the model generates inside
        // the block and only emits the closing `</think>`. Prime the segmenter into the Thinking
        // channel by feeding it that already-rendered opening marker (stripped, so it emits nothing);
        // otherwise the reasoning would be misclassified as answer. Disabled mode renders a *closed*
        // `<think>\n\n</think>`, so this correctly does not prime.
        if let Some(seg) = segmenter.as_mut() {
            if prompt_opens_thinking(&prompt) {
                let _ = seg.push("<think>");
                debug_assert!(seg.in_thinking());
            }
        }

        // A tool-call segmenter when the request offers tools and the model's template renders them:
        // it lifts `<tool_call>` blocks out of the answer channel (markup excluded from the streamed
        // text) and parses them into structured calls (sc-7636). `None` otherwise, so a no-tools
        // request flows straight through `tool_pieces` unchanged.
        let tools_active = self.descriptor.capabilities.supports_tools && !req.tools.is_empty();
        let mut tool_seg = tools_active.then(|| ToolCallSegmenter::new(&req.tools));

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix). The
        // segmenter (when active) splits each delta into reasoning vs answer; answer text then feeds
        // the stop matcher so a stop string is trimmed and halts generation.
        let tokenizer = &self.tokenizer;
        let out = {
            let mut acc: Vec<u32> = Vec::new();
            let mut shown = 0usize;
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, .. } = ev {
                    let id = id as u32;
                    acc.push(id);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if text.len() > shown {
                            let delta = text[shown..].to_string();
                            shown = text.len();
                            match segmenter.as_mut() {
                                Some(seg) => {
                                    for span in seg.push(&delta) {
                                        match span.channel {
                                            Channel::Thinking => {
                                                thinking_buf.push_str(&span.text);
                                                last_id = id;
                                                on_event(CoreEvent::Token {
                                                    id,
                                                    text: span.text,
                                                    index: emit_index,
                                                    channel: Channel::Thinking,
                                                });
                                                emit_index += 1;
                                            }
                                            Channel::Content => {
                                                // Answer text → tool segmenter (lifts out tool-call
                                                // blocks) → stop matcher → emit.
                                                for piece in tool_pieces(&mut tool_seg, &span.text) {
                                                    emit_content(
                                                        &piece,
                                                        id,
                                                        &mut stop_matcher,
                                                        &mut streamed,
                                                        &mut emit_index,
                                                        &mut last_id,
                                                        &halt,
                                                        &mut *on_event,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {
                                    for piece in tool_pieces(&mut tool_seg, &delta) {
                                        emit_content(
                                            &piece,
                                            id,
                                            &mut stop_matcher,
                                            &mut streamed,
                                            &mut emit_index,
                                            &mut last_id,
                                            &halt,
                                            &mut *on_event,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            };
            let constraint = json_mask
                .as_mut()
                .map(|m| m as &mut dyn ConstraintMask);
            let should_stop = || halt.get();
            let should_stop_opt = stop_active.then_some(&should_stop as &dyn Fn() -> bool);
            match &mm {
                // Multimodal: prefill the spliced embeds with interleaved M-RoPE + DeepStack fusion,
                // then decode the continuation (text positions shifted by `mrope_delta`) through the
                // shared loop — one path for either backbone via the `VlmDecode` seam.
                Some(m) => {
                    let (t, h, w, delta) = &m.positions;
                    let pos = [t.as_slice(), h.as_slice(), w.as_slice()];
                    let model = self.model.as_vlm();
                    let mut cache = model.make_cache();
                    let first = model
                        .prefill_with_deepstack(
                            &m.embeds,
                            pos,
                            cache.as_mut(),
                            &m.visual_pos_mask,
                            &m.deepstack,
                        )
                        .map_err(to_core)?;
                    let shifted = Shifted { model, delta: *delta };
                    generate_from_prefill(
                        &shifted,
                        cache.as_mut(),
                        first,
                        m.expanded_ids.clone(),
                        &config,
                        &req.cancel,
                        &mut sink,
                        constraint,
                        should_stop_opt,
                    )
                    .map_err(to_core)?
                }
                None => generate_with(
                    &self.model,
                    &prompt_ids,
                    &config,
                    &req.cancel,
                    &mut sink,
                    constraint,
                    should_stop_opt,
                )
                .map_err(to_core)?,
            }
        };

        // End-of-generation tails, in pipeline order. First the thinking segmenter's held-back
        // partial marker (it turned out not to begin a marker) as current-channel text — reasoning
        // straight out, answer through the tool segmenter; then the tool segmenter's own tail (held
        // partial `<tool_call>` / an unterminated block surfaced as content); then the stop matcher's
        // held-back partial stop.
        if let Some(seg) = segmenter.as_mut() {
            for span in seg.flush() {
                match span.channel {
                    Channel::Thinking => {
                        thinking_buf.push_str(&span.text);
                        on_event(CoreEvent::Token {
                            id: last_id,
                            text: span.text,
                            index: emit_index,
                            channel: Channel::Thinking,
                        });
                        emit_index += 1;
                    }
                    Channel::Content => {
                        for piece in tool_pieces(&mut tool_seg, &span.text) {
                            emit_content(
                                &piece,
                                last_id,
                                &mut stop_matcher,
                                &mut streamed,
                                &mut emit_index,
                                &mut last_id,
                                &halt,
                                &mut *on_event,
                            );
                        }
                    }
                }
            }
        }
        if let Some(ts) = tool_seg.as_mut() {
            for piece in ts.flush() {
                emit_content(
                    &piece,
                    last_id,
                    &mut stop_matcher,
                    &mut streamed,
                    &mut emit_index,
                    &mut last_id,
                    &halt,
                    &mut *on_event,
                );
            }
        }
        // If generation ended for any reason other than a stop string, flush the stop matcher's
        // held-back tail (a partial stop-prefix that never completed) — real output to stream/return.
        if stop_active && !halt.get() {
            let tail = stop_matcher.flush();
            if !tail.is_empty() {
                streamed.push_str(&tail);
                on_event(CoreEvent::Token {
                    id: last_id,
                    text: tail,
                    index: emit_index,
                    channel: Channel::Content,
                });
            }
        }

        // Result text: the streamed answer when stop strings, thinking, or tools are active (any of
        // which means the streamed channel is the authoritative answer with markup removed);
        // otherwise the original decode-all-tokens path (byte-identical to before). Reasoning and
        // tool calls, if the model produced any, are reported separately (markup excluded from text).
        let text = if stop_active || thinking_active || tools_active {
            streamed
        } else {
            let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&gen_u32, true)?
        };
        let thinking = (!thinking_buf.is_empty()).then_some(thinking_buf);
        let tool_calls = tool_seg.map(|mut ts| ts.take_calls()).unwrap_or_default();
        let finish = map_finish(out.finish_reason);
        let usage = Usage {
            prompt_tokens: prompt_len as u32,
            generated_tokens: out.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            text,
            thinking,
            tool_calls,
            usage,
            finish_reason: Some(finish),
        })
    }
}

/// The descriptor for the `mlx-llama` provider (constructible without loading weights; used for
/// link-time registration and registry discovery).
pub fn provider_descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.to_string(),
        family: "llama".to_string(),
        backend: "mlx".to_string(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            // Text-only today; the VLM path (sc-7157) flips this on for a vision provider.
            supports_vision: false,
            // Weightless default: conservative. The load path flips this on for a Qwen3-VL checkpoint
            // whose config carries a `video_token_id` (sc-8081).
            supports_video: false,
            // Weightless default: conservative. The load path (descriptor_for + load) flips this on
            // when the loaded model's own chat template gates an `enable_thinking` kwarg (sc-7585).
            supports_thinking: false,
            // Weightless default: conservative. The load path flips this on when the loaded model's
            // chat template renders tool calls (sc-7636).
            supports_tools: false,
            // JSON-constrained decoding (sc-7166).
            supported_constraints: vec![Constraint::Json],
        },
    }
}

/// A descriptor reflecting a *loaded* model: family from the dispatched architecture and the context
/// length from `config.json`. (Quantization state is reported via [`LlamaProvider::is_quantized`].)
fn descriptor_for(cfg: &ModelConfig) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = cfg.architecture.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d
}

/// The loaded-model descriptor for the hybrid Qwen3.6 (`qwen3_5`) decoder (parsed via
/// [`Qwen35Config`], which `ModelConfig` does not represent).
fn descriptor_for_qwen35(cfg: &Qwen35Config) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = Architecture::Qwen35.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d
}

/// Read and parse `config.json` from a snapshot directory into a JSON value (used to dispatch the
/// architecture before constructing the architecture-specific config).
fn read_config_value(dir: &Path) -> CoreResult<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| CoreError::Load(format!("read {}: {e}", dir.join("config.json").display())))?;
    serde_json::from_str(&text)
        .map_err(|e| CoreError::Load(format!("parse {}: {e}", dir.join("config.json").display())))
}

/// Read `eos_token_id` (int or array) from `config.json`; falls back to the Llama-3 stop ids.
pub fn eos_token_ids(dir: &Path) -> Vec<i32> {
    let fallback = vec![128001, 128008, 128009]; // <|end_of_text|>, <|eom_id|>, <|eot_id|>

    // Prefer `generation_config.json` — HF's canonical "how to generate" source, where models put
    // the *generation* EOS set (Qwen3.6's `<|im_end|>` turn-end lives only here, not in config.json).
    if let Some(ids) = read_json(dir, "generation_config.json")
        .as_ref()
        .and_then(|v| parse_token_ids(v.get("eos_token_id")))
    {
        return ids;
    }
    // Otherwise fall back to `config.json` — top-level, then the VLM-nested `text_config`.
    if let Some(v) = read_json(dir, "config.json") {
        if let Some(ids) = parse_token_ids(v.get("eos_token_id"))
            .or_else(|| parse_token_ids(v.get("text_config").and_then(|t| t.get("eos_token_id"))))
        {
            return ids;
        }
    }
    fallback
}

/// Whether a rendered prompt ends with an **unclosed** `<think>` block — i.e. the chat template
/// opened reasoning in the prompt (Qwen3.6's thinking/auto generation prompt) so the model generates
/// inside it. True iff the last `<think>` occurs after the last `</think>` (or there is no close).
fn prompt_opens_thinking(prompt: &str) -> bool {
    match prompt.rfind("<think>") {
        None => false,
        Some(open) => prompt.rfind("</think>").is_none_or(|close| open > close),
    }
}

/// Read and parse a JSON file from a snapshot dir, or `None` if missing/invalid.
fn read_json(dir: &Path, name: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join(name)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Parse an `eos_token_id`-style field — a single int or an array of ints — into a non-empty id list.
fn parse_token_ids(v: Option<&serde_json::Value>) -> Option<Vec<i32>> {
    let ids = match v? {
        serde_json::Value::Number(n) => vec![n.as_i64()? as i32],
        serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_i64().map(|x| x as i32)).collect(),
        _ => return None,
    };
    (!ids.is_empty()).then_some(ids)
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
        // An EOS *id* and a host stop condition (a request `stop` string) are both `Stop` per the
        // contract / OpenAI semantics.
        FinishReason::StopToken | FinishReason::Stopped => CoreFinish::Stop,
        FinishReason::MaxTokens => CoreFinish::Length,
        FinishReason::Cancelled => CoreFinish::Cancelled,
    }
}

/// Bridge an engine error into the contract error, preserving the typed cancellation / capability
/// variants (do not stringify those).
fn to_core(e: crate::Error) -> CoreError {
    match e {
        crate::Error::Canceled => CoreError::Canceled,
        crate::Error::Unsupported(m) => CoreError::Unsupported(m),
        crate::Error::MissingTensor(m) => CoreError::Load(format!("missing tensor: {m}")),
        crate::Error::Config(m) => CoreError::Load(m),
        crate::Error::Io(e) => CoreError::Io(e),
        other => CoreError::backend(other),
    }
}

// Register `mlx-llama` into core-llm's provider registry at link time.
inventory::submit! {
    core_llm::TextLlmRegistration {
        descriptor: provider_descriptor,
        load: load_registered,
        can_load,
        weightless_vision: Some(weightless_vision),
    }
}

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `mlx-llama` provider serve the snapshot at
/// `spec.source`? Reads **only** `config.json` and runs the same [`Architecture::from_config`]
/// dispatch the loader uses — it never opens a safetensors shard, so `core-llm`'s `load_for_model`
/// can resolve a provider by model without loading weights. A VLM wrapper (one carrying a
/// `vision_config`) is served **text-only** here unless a dedicated vision provider claims it: a
/// LLaVA snapshot is ceded to `mlx-joycaption`, while a Qwen-VL (`qwen3_5`) wrapper is claimed here
/// (its nested text decoder) so it no longer misroutes to JoyCaption (sc-7626).
pub fn can_load(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() { dir.join("config.json") } else { dir.to_path_buf() };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    can_load_value(&v)
}

/// Pure architecture/routing decision over a parsed `config.json` (split out for unit testing without
/// a snapshot on disk).
fn can_load_value(v: &serde_json::Value) -> bool {
    // A VLM wrapper carries a `vision_config`. Cede it to the dedicated vision provider when that
    // provider claims it (LLaVA → `mlx-joycaption`); otherwise serve the nested text decoder
    // text-only (a Qwen-VL `qwen3_5` wrapper).
    if v.get("vision_config").is_some() && crate::joycaption::can_load_value(v) {
        return false;
    }
    Architecture::from_config(v).is_ok()
}

/// **Weightless** per-snapshot vision probe (sc-8077): does `mlx-llama` serve the snapshot at
/// `spec.source` *with* vision? Reads **only** `config.json` (never a weight shard), mirroring
/// [`can_load`]. It drives core-llm's pre-load capability gate so a *model-first* vision-required
/// load (`load_for_model_with(spec, with_vision())`) resolves a Qwen-VL wrapper here.
///
/// This is necessary because the provider's STATIC [`provider_descriptor`] must report
/// `supports_vision=false` (the one registration also serves plain text-only checkpoints; vision is
/// only flipped on post-load when the loaded checkpoint carries `model.visual.*`). Without a
/// per-snapshot probe a genuine Qwen3-VL snapshot would be rejected at the gate (the story-D gap).
pub fn weightless_vision(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() { dir.join("config.json") } else { dir.to_path_buf() };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    weightless_vision_value(&v)
}

/// Pure per-snapshot vision decision over a parsed `config.json` (split out for unit testing). True
/// iff this provider both *claims* the snapshot ([`can_load_value`]) AND the snapshot is a Qwen-VL
/// wrapper that loads its vision tower: a `vision_config` is present and the architecture dispatches
/// to a Qwen-VL decoder (`qwen3_vl` or the Qwen3.6 `qwen3_5` hybrid) — exactly the post-load
/// condition that flips `supports_vision` on in [`LlamaProvider::load`]. A plain text checkpoint (no
/// `vision_config`) or a ceded LLaVA snapshot returns false.
fn weightless_vision_value(v: &serde_json::Value) -> bool {
    if !can_load_value(v) || v.get("vision_config").is_none() {
        return false;
    }
    matches!(
        Architecture::from_config(v),
        Ok(Architecture::Qwen3Vl) | Ok(Architecture::Qwen35)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qwen36_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5", "depth": 27 }
        })
    }
    fn llava_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
            "text_config": { "model_type": "llama" },
            "vision_config": { "model_type": "siglip_vision_model" }
        })
    }
    fn qwen3vl_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": { "model_type": "qwen3_vl_text" },
            "vision_config": { "model_type": "qwen3_vl", "depth": 27 }
        })
    }

    #[test]
    fn text_provider_claims_qwen36_but_cedes_llava() {
        // Qwen3.6 (Qwen-VL wrapper): claimed by the text provider, served text-only.
        assert!(can_load_value(&qwen36_wrapper()));
        // Qwen3-VL (Qwen-VL wrapper): also claimed by `mlx-llama` (its nested standard Qwen3 decoder).
        assert!(can_load_value(&qwen3vl_wrapper()));
        // LLaVA: ceded to the JoyCaption vision provider.
        assert!(!can_load_value(&llava_wrapper()));
        // Plain text models (no vision_config) are unaffected.
        assert!(can_load_value(&json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })));
        assert!(can_load_value(&json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })));
    }

    #[test]
    fn weightless_vision_advertises_qwen_vl_wrappers_only() {
        // The sc-8077 weightless vision gate: `mlx-llama` advertises vision (pre-load, config.json
        // only) for a Qwen-VL wrapper so a model-first vision-required load resolves here.
        // Qwen3-VL: a vision_config + qwen3_vl arch ⇒ vision-capable.
        assert!(weightless_vision_value(&qwen3vl_wrapper()));
        // Qwen3.6 hybrid (qwen3_5) Qwen-VL wrapper: likewise vision-capable.
        assert!(weightless_vision_value(&qwen36_wrapper()));
        // A LLaVA snapshot is ceded to JoyCaption (can_load=false here) ⇒ NOT advertised, so a
        // Qwen3-VL load can never be mistaken for / misrouted via this provider's vision path.
        assert!(!weightless_vision_value(&llava_wrapper()));
        // Plain text checkpoints (no vision_config) ⇒ NOT vision-capable (the static descriptor
        // already reports supports_vision=false; the probe must not flip it on).
        assert!(!weightless_vision_value(&json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })));
        assert!(!weightless_vision_value(&json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })));
    }

    /// Locate the cached Qwen3-VL-8B-Instruct snapshot (rev 0c351dd0) for the chat-template oracle.
    /// `QWEN3VL_SNAPSHOT` overrides; otherwise the default HF cache path. `None` ⇒ the gated tests
    /// self-skip cleanly (the snapshot is present in CI for this story).
    fn qwen3vl_snapshot_dir() -> Option<std::path::PathBuf> {
        if let Ok(path) = std::env::var("QWEN3VL_SNAPSHOT") {
            let path = std::path::PathBuf::from(path);
            return path.exists().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let path = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        );
        path.exists().then_some(path)
    }

    /// Build the representative chat-message sets the oracle pins (text-only, system+user, single
    /// image, multi-image/mixed, multi-turn). Image content is a `Content::Image` (a 1×1 black pixel
    /// placeholder); the byte-match exercises the same `substitute_image_placeholders` path the
    /// provider uses, so the rendered single `<|image_pad|>` per image must match HF.
    fn oracle_messages(case: &str) -> Vec<Message> {
        let img = || Content::Image(ImageRef::new(1, 1, vec![0, 0, 0]).unwrap());
        let user = |content: Vec<Content>| Message {
            role: core_llm::Role::User,
            content,
            thinking: None,
            tool_calls: Vec::new(),
        };
        let sys = |t: &str| Message {
            role: core_llm::Role::System,
            content: vec![Content::text(t)],
            thinking: None,
            tool_calls: Vec::new(),
        };
        let asst = |t: &str| Message {
            role: core_llm::Role::Assistant,
            content: vec![Content::text(t)],
            thinking: None,
            tool_calls: Vec::new(),
        };
        match case {
            "text_only" => vec![user(vec![Content::text("What is the capital of France?")])],
            "system_user_text" => {
                vec![sys("You are a helpful assistant."), user(vec![Content::text("Hello!")])]
            }
            "single_image" => vec![user(vec![img(), Content::text("Describe this image.")])],
            "multi_image_mixed" => vec![user(vec![
                Content::text("Compare:"),
                img(),
                Content::text("and"),
                img(),
                Content::text("please."),
            ])],
            "multi_turn" => vec![
                user(vec![Content::text("Hi")]),
                asst("Hello there!"),
                user(vec![Content::text("How are you?")]),
            ],
            other => panic!("unknown oracle case {other}"),
        }
    }

    /// Byte-match oracle (sc-8075 AC #1): the engine's chat-template + image-placeholder + tokenize
    /// path must reproduce the pinned HF `apply_chat_template` prompt token ids **exactly** for the
    /// representative messages — text-only, single-image, and multi-image/mixed. The single
    /// `<|image_pad|>` (151655) per image is what the processor emits *before* patch-count expansion;
    /// these fixtures pin that pre-expansion prompt. Self-skips cleanly when the snapshot is absent.
    #[test]
    fn chat_template_byte_matches_hf_processor() {
        let Some(dir) = qwen3vl_snapshot_dir() else {
            eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
            return;
        };
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("models/testdata/qwen3vl_chat_template_oracle.json"))
                .expect("parse chat-template oracle");

        // Sanity: the dispatch and config see Qwen3-VL (and parse the nested 256K-context text decoder).
        let cfg_value = read_config_value(&dir).expect("read config.json");
        assert_eq!(
            Architecture::from_config(&cfg_value).unwrap(),
            Architecture::Qwen3Vl,
            "snapshot must dispatch to Qwen3-VL"
        );
        let cfg = ModelConfig::from_json(&cfg_value).expect("parse Qwen3-VL text config");
        assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
        assert_eq!(cfg.max_position_embeddings, 262144, "256K context");

        let (template, _, _) = load_chat_template(&dir);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        for (case, expected) in oracle["cases"].as_object().unwrap() {
            let want: Vec<i32> = expected["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            // Drive the exact provider path: substitute image content with the Qwen-VL placeholder,
            // render the model's own Jinja template (add_generation_prompt), then tokenize without
            // auto special tokens.
            let messages = oracle_messages(case);
            let substituted = substitute_vision_placeholders(&messages, 2);
            let prompt = template
                .render_with(
                    &substituted,
                    &RenderOptions {
                        add_generation_prompt: true,
                        enable_thinking: None,
                        tools: &[],
                    },
                )
                .expect("render");
            assert_eq!(
                prompt,
                expected["text"].as_str().unwrap(),
                "case {case}: rendered prompt string must byte-match HF"
            );
            let got: Vec<i32> = tokenizer
                .encode(&prompt, false)
                .expect("encode")
                .into_iter()
                .map(|id| id as i32)
                .collect();
            assert_eq!(got, want, "case {case}: tokenized prompt ids must byte-match HF processor");
        }
    }

    #[test]
    fn prompt_opens_thinking_matches_template_modes() {
        // Qwen3.6 thinking/auto generation prompt: opens the block, leaves it unclosed.
        assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
        // Disabled mode renders a *closed* empty block: must not prime.
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        // A prior closed reasoning turn followed by a fresh open block still opens.
        assert!(prompt_opens_thinking("<think>\nold\n</think>\n\nq<|im_start|>assistant\n<think>\n"));
        // No reasoning markers at all (non-thinking template).
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
    }

    // --- Qwen3-VL video Text–Timestamp-Alignment oracle (tools/gen_qwen3vl_video_oracle.py) --------

    fn qwen3vl_video_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("models/testdata/qwen3vl_video_oracle.json")).unwrap()
    }

    /// **Merged per-frame timestamps match `Qwen3VLProcessor._calculate_timestamps`.** Given the
    /// sampled `frames_indices` + `fps`, the per-temporal-patch averaged timestamps must equal the HF
    /// reference exactly — the values that feed the `<{t:.1f} seconds>` Text–Timestamp-Alignment tags.
    #[test]
    fn video_merged_timestamps_match_hf_reference() {
        let j = qwen3vl_video_oracle();
        let fps = j["fps"].as_f64().unwrap() as f32;
        let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
        let indices: Vec<f32> = j["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        // Per-sample timestamps are `idx / fps` (matching the reference, which averages these).
        let per_sample: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
        let got = merged_frame_timestamps(&per_sample, temporal);
        let want: Vec<f32> = j["merged_timestamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(got.len(), want.len(), "merged timestamp count vs HF");
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "merged timestamp {g} vs HF {w}");
        }
    }

    /// **The per-frame placeholder string matches `Qwen3VLProcessor.replace_video_token` (collapsed
    /// form).** The engine emits **one** `<|video_pad|>` per frame and expands it to `frame_seqlen`
    /// copies after tokenizing (exactly the image path's pattern, where the chat template renders a
    /// single `<|image_pad|>`). The reference `replace_video_token` writes the *already-expanded*
    /// string (`frame_seqlen` `<|video_pad|>` per frame). Collapsing each consecutive `<|video_pad|>`
    /// run of the reference string to one token must yield exactly [`video_placeholder_text`]: same
    /// `<{t:.1f} seconds>` Text–Timestamp-Alignment tags, same per-frame vision framing.
    #[test]
    fn video_placeholder_string_matches_hf_reference() {
        let j = qwen3vl_video_oracle();
        let fps = j["fps"].as_f64().unwrap() as f32;
        let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
        let indices: Vec<f32> = j["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        // Build a synthetic VideoRef with one 1x1 frame per sampled index carrying its `idx/fps`
        // timestamp (the frame pixels are irrelevant to the placeholder string).
        let frames: Vec<ImageRef> = indices
            .iter()
            .map(|_| ImageRef::new(1, 1, vec![0, 0, 0]).unwrap())
            .collect();
        let timestamps: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
        let video = VideoRef::new(frames, timestamps).unwrap();
        let got = video_placeholder_text(&video, temporal);

        // Collapse the reference string's `<|video_pad|>` runs to a single token per frame.
        let pad = "<|video_pad|>";
        let mut collapsed = j["placeholder_text"].as_str().unwrap().to_string();
        while collapsed.contains(&format!("{pad}{pad}")) {
            collapsed = collapsed.replace(&format!("{pad}{pad}"), pad);
        }
        assert_eq!(
            got, collapsed,
            "collapsed Text–Timestamp-Alignment placeholder string must byte-match HF replace_video_token"
        );
        // The timestamp tags themselves must appear verbatim (the core of Text–Timestamp Alignment).
        for t in j["merged_timestamps"].as_array().unwrap() {
            let tag = format!("<{:.1} seconds>", t.as_f64().unwrap());
            assert!(got.contains(&tag), "placeholder must carry the `{tag}` timestamp tag: {got}");
        }
    }

    /// **The per-frame `<|video_pad|>` expansion matches the HF id stream.** Tokenizing the reference
    /// placeholder string yields one `<|video_pad|>` per frame; expanding each to `frame_seqlen`
    /// copies (the merged patch count the ViT emits per frame) must reproduce the exact id stream the
    /// processor produces — same per-frame vision framing, same timestamp tokens, same counts. This is
    /// the video analogue of `expand_vision_placeholders` for images, but with `grid_t` runs.
    #[test]
    fn video_token_expansion_matches_hf_reference() {
        let j = qwen3vl_video_oracle();
        let expanded_hf: Vec<i32> =
            j["expanded_ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let grid_t = j["grid_t"].as_u64().unwrap() as usize;
        let frame_seqlen = j["frame_seqlen"].as_u64().unwrap() as usize;
        let expected_video_tokens = j["expected_video_tokens"].as_u64().unwrap() as usize;

        // The merged-token count per frame, and total, must agree with the HF processor.
        assert_eq!(grid_t * frame_seqlen, expected_video_tokens, "total video tokens vs HF");
        assert_eq!(
            expanded_hf.iter().filter(|&&x| x == vid).count(),
            expected_video_tokens,
            "video tokens in HF id stream"
        );

        // Reconstruct the *raw* (pre-expansion) ids: collapse each consecutive `<|video_pad|>` run
        // back to a single placeholder. The HF stream has `grid_t` such runs (one per frame), each of
        // `frame_seqlen` tokens; collapsing recovers one `<|video_pad|>` per frame.
        let mut raw = Vec::new();
        let mut i = 0usize;
        let mut runs = 0usize;
        while i < expanded_hf.len() {
            if expanded_hf[i] == vid {
                raw.push(vid);
                runs += 1;
                while i < expanded_hf.len() && expanded_hf[i] == vid {
                    i += 1;
                }
            } else {
                raw.push(expanded_hf[i]);
                i += 1;
            }
        }
        assert_eq!(runs, grid_t, "one <|video_pad|> run per frame (grid_t)");

        // Expanding each per-frame placeholder to `frame_seqlen` reproduces the HF id stream exactly.
        let counts = vec![frame_seqlen; grid_t];
        let expanded = crate::models::qwen35::expand_vision_placeholders(&raw, vid, &counts).unwrap();
        assert_eq!(expanded, expanded_hf, "expanded video ids vs HF processor");
    }

    /// **The video M-RoPE positions over the oracle grid are well-formed and per-frame-reset.** Feed
    /// the expanded video id stream + the `video_grid_thw` through `mrope_positions_mm`: the temporal
    /// row must reset to the frame's cursor at each frame (Qwen3-VL's synthetic time axis splits each
    /// `[t,h,w]` into `t` per-frame `[1,h,w]` blocks). The HF-pinned exact-row check lives in
    /// `qwen35::qwen3vl_mrope_video_matches_hf_reference`; here we confirm the provider's video grid
    /// drives the same path consistently.
    #[test]
    fn video_mrope_positions_split_frames() {
        let j = qwen3vl_video_oracle();
        let expanded_hf: Vec<i32> =
            j["expanded_ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let img = j["video_token_id"].as_i64().unwrap() as i32 - 1; // a distinct unused image id
        let g = j["video_grid_thw"].as_array().unwrap()[0].as_array().unwrap();
        let grid = [g[0].as_i64().unwrap() as i32, g[1].as_i64().unwrap() as i32, g[2].as_i64().unwrap() as i32];
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (t, h, w, _delta) = crate::models::deepstack::mrope_positions_mm(
            &expanded_hf,
            &[],
            img,
            &[grid],
            vid,
            merge,
        )
        .unwrap();
        assert_eq!(t.len(), expanded_hf.len());
        // Each frame's video tokens share one temporal index (gt = 1 per frame after the split), and
        // the two frames sit at *different* temporal positions (the cursor advances between them).
        let frame_temporals: Vec<i32> = expanded_hf
            .iter()
            .zip(&t)
            .filter_map(|(&id, &tt)| (id == vid).then_some(tt))
            .collect();
        let distinct: std::collections::BTreeSet<i32> = frame_temporals.iter().copied().collect();
        assert_eq!(distinct.len(), grid[0] as usize, "one distinct temporal index per frame");
        // h/w spans are bounded by the per-frame grid (h/merge, w/merge).
        let max_w = (grid[2] / merge) - 1;
        let frame_ws: Vec<i32> = expanded_hf
            .iter()
            .zip(&w)
            .zip(&t)
            .filter_map(|((&id, &ww), &tt)| (id == vid).then_some(ww - tt))
            .collect();
        assert!(frame_ws.iter().all(|&rel| (0..=max_w).contains(&rel)), "w within per-frame grid");
        let _ = h;
    }
}
