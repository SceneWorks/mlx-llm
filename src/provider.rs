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
    TextLlmRequest, ThinkingSegmenter, Tokenizer, ToolCallSegmenter, Usage,
};

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_from_prefill, generate_with, ConstraintMask, Decode, FinishReason, GenerationConfig,
    StreamEvent,
};
use crate::image::Qwen35ImageProcessor;
use crate::models::{
    CausalLm, Qwen35Cache, Qwen35Config, Qwen35Model, Qwen35VisionConfig, Qwen35VisionModel,
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

    /// The concrete hybrid decoder, when this is the Qwen3.6 path (the multimodal embeds/M-RoPE hooks
    /// live on `Qwen35Model`, not the generic `Decode` trait).
    fn as_qwen35(&self) -> Option<&Qwen35Model> {
        match self {
            Decoder::Qwen35(m) => Some(m),
            Decoder::Causal(_) => None,
        }
    }
}

/// The Qwen3.6 vision side of the provider: the ViT tower, the image preprocessor, and the
/// multimodal token ids needed to expand placeholders and assign M-RoPE positions. Present only when
/// the loaded `qwen3_5` checkpoint carries `model.visual.*`.
struct Qwen35Vision {
    tower: Qwen35VisionModel,
    processor: Qwen35ImageProcessor,
    image_token_id: i32,
    spatial_merge_size: i32,
}

impl Qwen35Vision {
    /// Encode one image to its merged patch rows `[n_tokens, hidden]` (the merger output is already
    /// the language hidden size — no separate projector) plus the image's `grid_thw` (`[1, h, w]` in
    /// patch units). `n_tokens = (grid_h/merge)·(grid_w/merge)` is the placeholder expansion count.
    fn encode(&self, img: &ImageRef) -> CoreResult<(Array, [i32; 3])> {
        let (pixels, grid) = self
            .processor
            .preprocess(&img.pixels, img.width as usize, img.height as usize)
            .map_err(to_core)?;
        let features = self.tower.forward(&pixels, &grid).map_err(to_core)?;
        Ok((features, grid[0]))
    }
}

/// The prepared multimodal prefill: the image-token-expanded prompt ids, the decoder input embeds
/// with image features spliced in, and the interleaved M-RoPE position rows + delta.
struct MultimodalPrefill {
    expanded_ids: Vec<i32>,
    embeds: Array,
    positions: (Vec<i32>, Vec<i32>, Vec<i32>, i32),
}

/// A [`Decode`] wrapper that shifts the RoPE offset by a constant `delta` — the Qwen3.6 multimodal
/// decode steps continue from `mrope_delta` past the cached length (image tokens compress the
/// position cursor, so post-prompt text positions are `cache_len + mrope_delta`, not `cache_len`).
/// The new tokens are text, so a single shifted 1-D position is the correct M-RoPE position.
struct ShiftedQwen35<'a> {
    model: &'a Qwen35Model,
    delta: i32,
}

impl Decode for ShiftedQwen35<'_> {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.model.new_cache())
    }

    fn step(&self, ids: &Array, cache: &mut dyn KvCache, offset: i32) -> crate::error::Result<Array> {
        let c = cache
            .as_any_mut()
            .downcast_mut::<Qwen35Cache>()
            .ok_or_else(|| crate::error::Error::Msg("ShiftedQwen35: cache is not a Qwen35Cache".into()))?;
        self.model.decode_logits(ids, c, offset + self.delta)
    }
}

/// Collect the image blocks of a conversation, in order.
fn collect_images(messages: &[Message]) -> Vec<&ImageRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Image(img) => Some(img),
                Content::Text(_) => None,
            })
        })
        .collect()
}

/// Replace each image block with the Qwen-VL placeholder text so the (text-only) chat template
/// renders `<|vision_start|><|image_pad|><|vision_end|>`; the single `image_pad` token is expanded to
/// the per-image token count after tokenizing. Keeps the core-llm template contract image-free.
fn substitute_image_placeholders(messages: &[Message]) -> Vec<Message> {
    const PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    messages
        .iter()
        .map(|m| Message {
            role: m.role,
            content: m
                .content
                .iter()
                .map(|c| match c {
                    Content::Image(_) => Content::text(PLACEHOLDER),
                    Content::Text(t) => Content::Text(t.clone()),
                })
                .collect(),
            thinking: m.thinking.clone(),
            tool_calls: m.tool_calls.clone(),
        })
        .collect()
}

/// Expand each `image_token_id` placeholder in `ids` into `counts[i]` image tokens (the i-th image's
/// merged-patch count), in order. Errors if the placeholder count and image count disagree.
fn expand_image_placeholders(
    ids: &[i32],
    image_token_id: i32,
    counts: &[usize],
) -> crate::error::Result<Vec<i32>> {
    use crate::error::Error;
    let mut out = Vec::with_capacity(ids.len());
    let mut ci = 0usize;
    for &id in ids {
        if id == image_token_id {
            let n = *counts.get(ci).ok_or_else(|| {
                Error::Msg(format!(
                    "qwen3.6 vision: {} image placeholders but only {} images supplied",
                    ci + 1,
                    counts.len()
                ))
            })?;
            ci += 1;
            out.extend(std::iter::repeat_n(image_token_id, n));
        } else {
            out.push(id);
        }
    }
    if ci != counts.len() {
        return Err(Error::Msg(format!(
            "qwen3.6 vision: {ci} image placeholders rendered but {} images supplied",
            counts.len()
        )));
    }
    Ok(out)
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

        // Qwen3.6 vision: load the ViT tower when the checkpoint carries `model.visual.*` (a wrapped
        // VLM) and the config exposes a `vision_config`. Absent → a text-only `qwen3_5` checkpoint.
        let vision = if arch == Architecture::Qwen35
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
            descriptor.capabilities.supports_vision = true;
            Some(Qwen35Vision {
                tower,
                processor: Qwen35ImageProcessor::default(),
                image_token_id,
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

    /// Build the multimodal prefill: encode each image (preprocess → ViT → merged rows), expand the
    /// rendered `image_pad` placeholders to the per-image token counts, splice the features into the
    /// token embeds, and compute the interleaved M-RoPE 3-D positions. `prompt_ids` is the tokenized
    /// prompt (one `image_token_id` per image, from the rendered placeholders).
    fn prepare_multimodal(
        &self,
        prompt_ids: &[i32],
        images: &[&ImageRef],
    ) -> CoreResult<MultimodalPrefill> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| CoreError::Load("qwen3.6 vision: provider has no vision tower".into()))?;
        let model = self
            .model
            .as_qwen35()
            .ok_or_else(|| CoreError::Load("qwen3.6 vision requires the qwen3_5 decoder".into()))?;

        let mut feats: Vec<Array> = Vec::with_capacity(images.len());
        let mut counts: Vec<usize> = Vec::with_capacity(images.len());
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(images.len());
        for img in images {
            let (f, grid) = vision.encode(img)?;
            counts.push(f.shape()[0] as usize);
            grids.push(grid);
            feats.push(f);
        }

        let expanded = expand_image_placeholders(prompt_ids, vision.image_token_id, &counts).map_err(to_core)?;
        let refs: Vec<&Array> = feats.iter().collect();
        let all_features = match refs.as_slice() {
            [one] => (*one).clone(),
            many => concatenate_axis(many, 0).map_err(|e| to_core(e.into()))?,
        };

        let embeds = model.embed_input_ids(&input_ids(&expanded)).map_err(to_core)?;
        let spliced = model
            .splice_image_features(&embeds, &expanded, &all_features, vision.image_token_id)
            .map_err(to_core)?;
        let positions = model
            .mrope_positions(&expanded, &grids, vision.image_token_id, vision.spatial_merge_size)
            .map_err(to_core)?;

        Ok(MultimodalPrefill {
            expanded_ids: expanded,
            embeds: spliced,
            positions,
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

        // Multimodal (Qwen3.6 + image content): replace image blocks with the Qwen-VL placeholder so
        // the (image-free) chat template renders `<|vision_start|><|image_pad|><|vision_end|>`. The
        // images are encoded + spliced after tokenizing. Text-only requests are unchanged.
        let images: Vec<&ImageRef> = match &self.vision {
            Some(_) => collect_images(&req.messages),
            None => Vec::new(),
        };
        let multimodal = !images.is_empty();
        let substituted;
        let messages: &[Message] = if multimodal {
            substituted = substitute_image_placeholders(&req.messages);
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

        // Encode + splice the images and compute M-RoPE positions (the image-token-expanded prompt
        // becomes the effective sequence). `None` on the text-only path.
        let mm = if multimodal {
            Some(self.prepare_multimodal(&prompt_ids, &images)?)
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
                // Multimodal: prefill the spliced embeds with interleaved M-RoPE, then decode the
                // continuation (text positions shifted by `mrope_delta`) through the shared loop.
                Some(m) => {
                    let model = self
                        .model
                        .as_qwen35()
                        .ok_or_else(|| CoreError::Load("qwen3.6 vision requires the qwen3_5 decoder".into()))?;
                    let mut cache = model.new_cache();
                    let (t, h, w, delta) = &m.positions;
                    let first = model
                        .decode_logits_from_embeds(&m.embeds, [t.as_slice(), h.as_slice(), w.as_slice()], &mut cache)
                        .map_err(to_core)?;
                    let shifted = ShiftedQwen35 { model, delta: *delta };
                    generate_from_prefill(
                        &shifted,
                        &mut cache,
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

    #[test]
    fn text_provider_claims_qwen36_but_cedes_llava() {
        // Qwen3.6 (Qwen-VL wrapper): claimed by the text provider, served text-only.
        assert!(can_load_value(&qwen36_wrapper()));
        // LLaVA: ceded to the JoyCaption vision provider.
        assert!(!can_load_value(&llava_wrapper()));
        // Plain text models (no vision_config) are unaffected.
        assert!(can_load_value(&json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })));
        assert!(can_load_value(&json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })));
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
}
