//! The OpenAI chat-completions wire format: request parsing → the backend-neutral
//! [`core_llm::TextLlmRequest`], and response/SSE-chunk construction from the contract's events.
//!
//! These are pure data transforms (no model, no I/O), so they're unit-tested directly. The server
//! ([`crate::main`]) wires them to a TCP socket + a loaded `core_llm::TextLlm` provider.

use mlx_llm::core_llm::{Constraint, Content, Message, Role, Sampling, TextLlmRequest};
use serde::Deserialize;
use serde_json::{json, Value};

/// Default `max_tokens` when a request omits it (OpenAI has no implicit cap; we pick a sane one).
const DEFAULT_MAX_TOKENS: u32 = 512;

/// An OpenAI `POST /v1/chat/completions` request body (the subset this example serves, plus the
/// common `top_k`/`repetition_penalty` extensions other on-device servers accept).
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Requested model id (informational — this server hosts a single loaded model; echoed back).
    #[serde(default)]
    pub model: Option<String>,
    /// The conversation.
    pub messages: Vec<ChatMessage>,
    /// Stream the response as Server-Sent Events.
    #[serde(default)]
    pub stream: bool,
    /// Max new tokens to generate.
    pub max_tokens: Option<u32>,
    /// Sampling temperature (`0` ⇒ greedy).
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Top-k cutoff (non-OpenAI extension).
    pub top_k: Option<usize>,
    /// Repetition penalty (non-OpenAI extension).
    pub repetition_penalty: Option<f32>,
    /// RNG seed for reproducible sampling.
    pub seed: Option<u64>,
    /// Stop strings (string or array).
    pub stop: Option<StringOrVec>,
    /// `{"type":"json_object"}` ⇒ constrain output to valid JSON.
    pub response_format: Option<ResponseFormat>,
}

/// One chat turn. `content` is a string or an array of typed parts (the vision wire form); this
/// text-only server uses the text parts.
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

/// OpenAI message content: a plain string, or an array of `{type, text}` parts.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One content part (only the `text` kind is consumed by this text-only server).
#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

impl MessageContent {
    /// Flatten to plain text (concatenating text parts; non-text parts are dropped — the provider's
    /// capabilities reject vision input up front anyway).
    fn into_text(self) -> String {
        match self {
            MessageContent::Text(s) => s,
            MessageContent::Parts(parts) => parts
                .into_iter()
                .filter(|p| p.kind == "text")
                .filter_map(|p| p.text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// A JSON value that may be a single string or a list of strings (e.g. OpenAI `stop`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    fn into_vec(self) -> Vec<String> {
        match self {
            StringOrVec::One(s) => vec![s],
            StringOrVec::Many(v) => v,
        }
    }
}

/// `response_format` — only `{"type":"json_object"}` is acted on (JSON-constrained decode).
#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
}

/// Map an OpenAI role string to the contract [`Role`] (unknown ⇒ treated as a user turn).
fn role_of(s: &str) -> Role {
    match s {
        "system" | "developer" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

impl ChatRequest {
    /// Build the backend-neutral [`TextLlmRequest`]. Returns an error string for an empty
    /// conversation. Sampling starts from the engine's chat defaults; provided fields override.
    pub fn into_text_llm_request(self) -> Result<TextLlmRequest, String> {
        if self.messages.is_empty() {
            return Err("`messages` must not be empty".into());
        }
        let messages = self
            .messages
            .into_iter()
            .map(|m| Message {
                role: role_of(&m.role),
                content: vec![Content::Text(m.content.into_text())],
                // This OpenAI shim does not yet accept prior-turn reasoning or assistant tool calls
                // on input; default both to the contract's "absent" values (no behavior change).
                thinking: None,
                tool_calls: Vec::new(),
            })
            .collect();

        let mut sampling = Sampling::default();
        if let Some(t) = self.temperature {
            sampling.temperature = t;
        }
        if let Some(p) = self.top_p {
            sampling.top_p = p;
        }
        if let Some(k) = self.top_k {
            sampling.top_k = k;
        }
        if let Some(rp) = self.repetition_penalty {
            sampling.repetition_penalty = rp;
        }

        let constraint = self
            .response_format
            .as_ref()
            .filter(|rf| rf.kind == "json_object")
            .map(|_| Constraint::Json);

        Ok(TextLlmRequest {
            messages,
            sampling,
            max_new_tokens: self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            seed: self.seed,
            constraint,
            // This shim does not yet surface request-level thinking/tools controls; leave thinking
            // at the template default (Auto) and offer no tools (no behavior change).
            thinking: Default::default(),
            tools: Vec::new(),
            stop: self.stop.map(StringOrVec::into_vec).unwrap_or_default(),
            cancel: Default::default(),
        })
    }
}

/// Map a contract finish reason to the OpenAI `finish_reason` string.
pub fn finish_reason_str(f: mlx_llm::core_llm::FinishReason) -> &'static str {
    use mlx_llm::core_llm::FinishReason::*;
    match f {
        Stop | Cancelled => "stop",
        Length => "length",
        ContentFilter => "content_filter",
    }
}

/// The first SSE chunk: an empty assistant-role delta (matches OpenAI clients' expectations).
pub fn role_chunk(id: &str, model: &str, created: u64) -> String {
    chunk(id, model, created, json!({ "role": "assistant" }), Value::Null)
}

/// A content SSE chunk carrying the next text delta.
pub fn content_chunk(id: &str, model: &str, created: u64, delta: &str) -> String {
    chunk(id, model, created, json!({ "content": delta }), Value::Null)
}

/// The terminal SSE chunk: an empty delta plus the finish reason.
pub fn final_chunk(id: &str, model: &str, created: u64, finish: &str) -> String {
    chunk(id, model, created, json!({}), json!(finish))
}

fn chunk(id: &str, model: &str, created: u64, delta: Value, finish_reason: Value) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    })
    .to_string()
}

/// A non-streaming `chat.completion` response body.
pub fn completion(
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    finish: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> String {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
    .to_string()
}

/// The `GET /v1/models` body listing the single hosted model.
pub fn models_list(model: &str, created: u64) -> String {
    json!({
        "object": "list",
        "data": [{ "id": model, "object": "model", "created": created, "owned_by": "mlx-llm" }],
    })
    .to_string()
}

/// An OpenAI-style error body.
pub fn error_body(message: &str, kind: &str) -> String {
    json!({ "error": { "message": message, "type": kind } }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> ChatRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn maps_messages_sampling_and_max_tokens() {
        let req = parse(
            r#"{"model":"m","messages":[
                {"role":"system","content":"be brief"},
                {"role":"user","content":"hi"}
            ],"temperature":0.0,"max_tokens":32,"seed":7,"stream":true}"#,
        );
        assert!(req.stream);
        assert_eq!(req.model.as_deref(), Some("m"));
        let r = req.into_text_llm_request().unwrap();
        assert_eq!(r.messages.len(), 2);
        assert_eq!(r.messages[0].role, Role::System);
        assert_eq!(r.messages[1].role, Role::User);
        assert_eq!(r.messages[1].content, vec![Content::Text("hi".into())]);
        assert_eq!(r.sampling.temperature, 0.0); // explicit override (greedy)
        assert_eq!(r.sampling.top_p, Sampling::default().top_p); // untouched -> engine default
        assert_eq!(r.max_new_tokens, 32);
        assert_eq!(r.seed, Some(7));
        assert!(r.constraint.is_none());
    }

    #[test]
    fn defaults_when_fields_omitted() {
        let r = parse(r#"{"messages":[{"role":"user","content":"x"}]}"#)
            .into_text_llm_request()
            .unwrap();
        assert_eq!(r.max_new_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(r.sampling, Sampling::default());
        assert!(r.seed.is_none());
    }

    #[test]
    fn json_object_response_format_sets_constraint() {
        let r = parse(
            r#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}"#,
        )
        .into_text_llm_request()
        .unwrap();
        assert_eq!(r.constraint, Some(Constraint::Json));
    }

    #[test]
    fn content_parts_and_stop_string_or_array() {
        let r = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"a"},{"type":"text","text":"b"}
            ]}],"stop":"END"}"#,
        )
        .into_text_llm_request()
        .unwrap();
        assert_eq!(r.messages[0].content, vec![Content::Text("ab".into())]);
        assert_eq!(r.stop, vec!["END".to_string()]);

        let r2 = parse(r#"{"messages":[{"role":"user","content":"x"}],"stop":["a","b"]}"#)
            .into_text_llm_request()
            .unwrap();
        assert_eq!(r2.stop, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn empty_messages_rejected() {
        assert!(parse(r#"{"messages":[]}"#).into_text_llm_request().is_err());
    }

    #[test]
    fn sse_chunks_have_openai_shape() {
        let role = serde_json::from_str::<Value>(&role_chunk("id1", "m", 100)).unwrap();
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(role["choices"][0]["finish_reason"], Value::Null);

        let content = serde_json::from_str::<Value>(&content_chunk("id1", "m", 100, "hello")).unwrap();
        assert_eq!(content["choices"][0]["delta"]["content"], "hello");

        let fin = serde_json::from_str::<Value>(&final_chunk("id1", "m", 100, "length")).unwrap();
        assert_eq!(fin["choices"][0]["finish_reason"], "length");
        assert_eq!(fin["choices"][0]["delta"], json!({}));
    }

    #[test]
    fn completion_body_carries_usage_and_message() {
        let v = serde_json::from_str::<Value>(&completion("id", "m", 1, "hi there", "stop", 3, 2)).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi there");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 3);
        assert_eq!(v["usage"]["completion_tokens"], 2);
        assert_eq!(v["usage"]["total_tokens"], 5);
    }
}
