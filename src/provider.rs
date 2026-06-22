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
    ChatTemplate, Constraint, ConstraintDecodeTable, Error as CoreError, FinishReason as CoreFinish,
    JinjaChatTemplate, JsonConstraint, Llama3Template, LoadSpec, Quantize, Result as CoreResult,
    Sampling, StopMatcher, StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities,
    TextLlmDescriptor, TextLlmOutput, TextLlmRequest, Tokenizer, Usage,
};

use crate::config::ModelConfig;
use crate::decode::{generate_with, ConstraintMask, FinishReason, GenerationConfig, StreamEvent};
use crate::models::CausalLm;
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::Weights;

/// The registry id of this provider.
pub const PROVIDER_ID: &str = "mlx-llama";

/// A generic Llama provider implementing [`core_llm::TextLlm`].
pub struct LlamaProvider {
    descriptor: TextLlmDescriptor,
    model: CausalLm,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    stop_tokens: Vec<i32>,
    /// Cached per-vocab decode table for constrained decoding — built once (it decodes the whole
    /// vocabulary) on the first JSON-constrained request, then reused.
    constraint_table: OnceCell<ConstraintDecodeTable>,
}

impl LlamaProvider {
    /// Load a provider from a snapshot directory (config.json + tokenizer.json + shards). Dispatches
    /// the decoder architecture from `config.json` (Llama / Mistral / Qwen3) and optionally
    /// quantizes the projections on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        let dir = Path::new(&spec.source);
        let cfg = ModelConfig::from_dir(dir).map_err(to_core)?;
        let quant = spec.quantize.map(|q| match q {
            Quantize::Q4 => QuantSpec::q4(),
            Quantize::Q8 => QuantSpec::q8(),
        });
        let descriptor = descriptor_for(&cfg);
        let weights = Weights::from_dir(dir).map_err(to_core)?;
        let model = CausalLm::from_weights_with(&weights, "", cfg, quant).map_err(to_core)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let stop_tokens = eos_token_ids(dir);
        Ok(Self {
            descriptor,
            model,
            tokenizer,
            template: load_chat_template(dir),
            stop_tokens,
            constraint_table: OnceCell::new(),
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
            model,
            tokenizer,
            template: Box::new(Llama3Template),
            stop_tokens,
            constraint_table: OnceCell::new(),
        }
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
/// present; otherwise fall back to the typed Llama-3 template.
fn load_chat_template(dir: &Path) -> Box<dyn ChatTemplate> {
    match JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json")) {
        Ok(t) => Box::new(t),
        Err(_) => Box::new(Llama3Template),
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

        // Render the conversation and tokenize. The template already includes BOS, so encode
        // without auto special tokens.
        let prompt = self.template.render(&req.messages, true)?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let prompt_len = prompt_ids.len();

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
        // The emitted text, accumulated as the matcher releases it — the truncated result when stop
        // strings are active (the no-stop path keeps decoding all tokens, exactly as before).
        let mut streamed = String::new();
        let mut last_emit: Option<(u32, usize)> = None; // (id, index) of the last emitted token

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix), feeding
        // each delta through the stop matcher so a stop string is trimmed and halts generation.
        let tokenizer = &self.tokenizer;
        let out = {
            let mut acc: Vec<u32> = Vec::new();
            let mut shown = 0usize;
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, step } = ev {
                    acc.push(id as u32);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if text.len() > shown {
                            let delta = text[shown..].to_string();
                            shown = text.len();
                            let chunk = stop_matcher.push(&delta);
                            if !chunk.emit.is_empty() {
                                streamed.push_str(&chunk.emit);
                                last_emit = Some((id as u32, step));
                                on_event(CoreEvent::Token {
                                    id: id as u32,
                                    text: chunk.emit,
                                    index: step,
                                });
                            }
                            if chunk.stop {
                                halt.set(true);
                            }
                        }
                    }
                }
            };
            let constraint = json_mask
                .as_mut()
                .map(|m| m as &mut dyn ConstraintMask);
            let should_stop = || halt.get();
            generate_with(
                &self.model,
                &prompt_ids,
                &config,
                &req.cancel,
                &mut sink,
                constraint,
                stop_active.then_some(&should_stop as &dyn Fn() -> bool),
            )
            .map_err(to_core)?
        };

        // If generation ended for any reason other than a stop string, flush the matcher's
        // held-back tail (a partial stop-prefix that never completed) — it is real output that must
        // still be streamed and returned.
        if stop_active && !halt.get() {
            let tail = stop_matcher.flush();
            if !tail.is_empty() {
                let (id, index) = last_emit.unwrap_or((0, out.tokens.len().saturating_sub(1)));
                streamed.push_str(&tail);
                on_event(CoreEvent::Token { id, text: tail, index });
            }
        }

        // With stop strings active the result is the trimmed streamed text; otherwise keep the
        // existing decode-the-tokens path (byte-identical to before this change).
        let text = if stop_active {
            streamed
        } else {
            let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&gen_u32, true)?
        };
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

/// Read `eos_token_id` (int or array) from `config.json`; falls back to the Llama-3 stop ids.
pub fn eos_token_ids(dir: &Path) -> Vec<i32> {
    let fallback = vec![128001, 128008, 128009]; // <|end_of_text|>, <|eom_id|>, <|eot_id|>
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return fallback;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return fallback;
    };
    match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_i64().map(|x| vec![x as i32]).unwrap_or(fallback),
        Some(serde_json::Value::Array(a)) => {
            let ids: Vec<i32> = a.iter().filter_map(|x| x.as_i64().map(|x| x as i32)).collect();
            if ids.is_empty() {
                fallback
            } else {
                ids
            }
        }
        _ => fallback,
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
    }
}

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}
