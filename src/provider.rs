//! The `core-llm` provider: a generic Llama model exposed through the backend-neutral contract.
//!
//! This is the mlx-llm half of story 7154 — it implements [`core_llm::TextLlm`] by wrapping the
//! [`LlamaModel`] decoder, a [`core_llm::Tokenizer`], and a chat template, driving the internal
//! streaming decode loop and translating its token events into contract [`StreamEvent`]s (with
//! incremental detokenization). It registers into [`core_llm::registry`] under the id `mlx-llama`.
//!
//! The chat template is the typed [`Llama3Template`] for now; reading a model's own `chat_template`
//! (minijinja) lands in story 7164 behind the same seam.

use std::path::Path;

use core_llm::{
    ChatTemplate, Error as CoreError, FinishReason as CoreFinish, Llama3Template, LoadSpec,
    Result as CoreResult, Sampling, StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities,
    TextLlmDescriptor, TextLlmOutput, TextLlmRequest, Tokenizer, Usage,
};

use crate::config::LlamaConfig;
use crate::decode::{generate, FinishReason, GenerationConfig, StreamEvent};
use crate::models::LlamaModel;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::Weights;

/// The registry id of this provider.
pub const PROVIDER_ID: &str = "mlx-llama";

/// A generic Llama provider implementing [`core_llm::TextLlm`].
pub struct LlamaProvider {
    descriptor: TextLlmDescriptor,
    model: LlamaModel,
    tokenizer: Tokenizer,
    template: Llama3Template,
    stop_tokens: Vec<i32>,
}

impl LlamaProvider {
    /// Load a Llama provider from a snapshot directory (config.json + tokenizer.json + shards).
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.quantize.is_some() {
            return Err(CoreError::Unsupported(
                "quantize-on-load is not yet implemented (sc-7163)".into(),
            ));
        }
        let dir = Path::new(&spec.source);
        let cfg = LlamaConfig::from_dir(dir).map_err(to_core)?;
        let weights = Weights::from_dir(dir).map_err(to_core)?;
        let model = LlamaModel::from_weights(&weights, "", cfg).map_err(to_core)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let stop_tokens = eos_token_ids(dir);
        Ok(Self::from_parts(model, tokenizer, stop_tokens))
    }

    /// Assemble a provider from already-loaded parts (used by tests and converters).
    pub fn from_parts(model: LlamaModel, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        Self {
            descriptor: provider_descriptor(),
            model,
            tokenizer,
            template: Llama3Template,
            stop_tokens,
        }
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

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix).
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
                            on_event(CoreEvent::Token {
                                id: id as u32,
                                text: delta,
                                index: step,
                            });
                        }
                    }
                }
            };
            generate(&self.model, &prompt_ids, &config, &req.cancel, &mut sink).map_err(to_core)?
        };

        let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
        let text = tokenizer.decode(&gen_u32, true)?;
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

/// The descriptor for the `mlx-llama` provider (constructible without loading weights).
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
            // Constrained decoding wiring lands in sc-7166.
            supported_constraints: Vec::new(),
        },
    }
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
        FinishReason::StopToken => CoreFinish::Stop,
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
