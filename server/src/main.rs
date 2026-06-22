//! Example OpenAI-compatible chat server on the mlx-llm engine (story 7174).
//!
//! ```text
//! cargo run --release -p mlx-llm-server -- --model <snapshot_dir> [--port 8080] [--quant q4|q8]
//! ```
//!
//! Serves `POST /v1/chat/completions` (streaming SSE or buffered JSON), `GET /v1/models`, and a
//! health check, for a single model loaded through the **backend-neutral** `core_llm` contract
//! (`load_textllm` routing → a `TextLlm` provider). It speaks only that contract, so it would serve
//! a `candle-llm` provider unchanged — the dependency on `mlx-llm` is just to link its provider
//! registration in.
//!
//! This is a *reference*, deliberately minimal: one model, one request at a time (MLX's Metal device
//! is single-threaded — see the engine's `.cargo/config.toml`), `Connection: close`, no auth. A
//! production gateway (multi-model, auth, batching across requests, Anthropic/Ollama compat) is the
//! separate server-app project, not this example.
//!
//! ```text
//! curl -N http://localhost:8080/v1/chat/completions \
//!   -H 'content-type: application/json' \
//!   -d '{"model":"local","stream":true,"messages":[{"role":"user","content":"Hi!"}]}'
//! ```

mod http;
mod openai;

use std::io::{self, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use mlx_llm::core_llm::{self, CancelFlag, Error as CoreError, LoadSpec, Quantize, StreamEvent, TextLlm};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Parsed CLI configuration.
struct Args {
    model: String,
    host: String,
    port: u16,
    quantize: Option<Quantize>,
    provider: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut quantize = None;
    let mut provider = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut next = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" | "-m" => model = Some(next()?),
            "--host" => host = next()?,
            "--port" | "-p" => port = next()?.parse().map_err(|_| "invalid --port".to_string())?,
            "--provider" => provider = Some(next()?),
            "--quant" => {
                quantize = Some(match next()?.as_str() {
                    "q4" => Quantize::Q4,
                    "q8" => Quantize::Q8,
                    other => return Err(format!("unknown --quant {other:?} (expected q4|q8)")),
                })
            }
            "-h" | "--help" => {
                println!("usage: mlx-llm-server --model <dir> [--host 127.0.0.1] [--port 8080] [--quant q4|q8] [--provider <id>]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("missing required --model <snapshot_dir>")?,
        host,
        port,
        quantize,
        provider,
    })
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    // Backend-neutral routing: use the requested provider id, else default to a registered *text*
    // (non-vision) provider — several may be registered (e.g. a VLM captioner alongside the generic
    // text model), so don't just grab the first. Stays backend-agnostic: no hard-coded id.
    let provider_id = match args.provider {
        Some(id) => id,
        None => {
            let descriptors = || core_llm::textllms().map(|r| (r.descriptor)());
            descriptors()
                .find(|d| !d.capabilities.supports_vision)
                .or_else(|| descriptors().next())
                .ok_or("no TextLlm provider registered")?
                .id
        }
    };
    eprintln!("loading model from {} via provider '{provider_id}' …", args.model);
    let spec = LoadSpec { source: args.model.clone(), quantize: args.quantize };
    let provider = core_llm::load_textllm(&provider_id, &spec)?;

    // A friendly default model name for responses (the snapshot dir's basename).
    let default_model = std::path::Path::new(&args.model)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| provider_id.clone());

    let listener = TcpListener::bind((args.host.as_str(), args.port))?;
    let addr = listener.local_addr()?;
    eprintln!("mlx-llm-server listening on http://{addr}  (model: {default_model})");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(e) = handle_connection(stream, provider.as_ref(), &default_model) {
                    eprintln!("connection error: {e}");
                }
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

/// Serve one request on a connection, then close it (`Connection: close`).
fn handle_connection(
    mut stream: TcpStream,
    provider: &dyn TextLlm,
    default_model: &str,
) -> io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let req = match http::read_request(&mut reader) {
        Ok(Some(req)) => req,
        Ok(None) => return Ok(()), // idle disconnect
        Err(e) => {
            return write_json(&mut stream, 400, &openai::error_body(&e.to_string(), "invalid_request"));
        }
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/v1/chat/completions") => handle_chat(&mut stream, provider, &req.body, default_model),
        ("GET", "/v1/models") => {
            write_json(&mut stream, 200, &openai::models_list(default_model, unix_secs()))
        }
        ("GET", "/" | "/health") => write_text(&mut stream, 200, "ok"),
        _ => write_json(&mut stream, 404, &openai::error_body("not found", "not_found")),
    }
}

/// Handle a chat completion: parse → validate → stream SSE or return one JSON body.
fn handle_chat(
    stream: &mut TcpStream,
    provider: &dyn TextLlm,
    body: &[u8],
    default_model: &str,
) -> io::Result<()> {
    let chat: openai::ChatRequest = match serde_json::from_slice(body) {
        Ok(c) => c,
        Err(e) => {
            return write_json(stream, 400, &openai::error_body(&e.to_string(), "invalid_request"))
        }
    };
    let model = chat.model.clone().unwrap_or_else(|| default_model.to_string());
    let want_stream = chat.stream;

    let mut req = match chat.into_text_llm_request() {
        Ok(r) => r,
        Err(msg) => return write_json(stream, 400, &openai::error_body(&msg, "invalid_request")),
    };
    // Reject anything outside the provider's declared surface before sending any 200.
    if let Err(e) = provider.validate(&req) {
        return write_json(stream, 400, &openai::error_body(&e.to_string(), "invalid_request"));
    }

    let cancel = CancelFlag::new();
    req.cancel = cancel.clone();
    let id = completion_id();
    let created = unix_secs();

    if want_stream {
        stream_chat(stream, provider, &req, &cancel, &id, &model, created)
    } else {
        match provider.complete(&req) {
            Ok(out) => {
                let finish = out.finish_reason.map(openai::finish_reason_str).unwrap_or("stop");
                let body = openai::completion(
                    &id,
                    &model,
                    created,
                    &out.text,
                    finish,
                    out.usage.prompt_tokens,
                    out.usage.generated_tokens,
                );
                write_json(stream, 200, &body)
            }
            Err(CoreError::Canceled) => Ok(()), // client vanished mid-generation
            Err(e) => write_json(stream, 500, &openai::error_body(&e.to_string(), "server_error")),
        }
    }
}

/// Stream a chat completion as Server-Sent Events. A failed write (client disconnected) trips the
/// request's [`CancelFlag`], so the decode loop stops promptly — i.e. **cancel disconnects the
/// stream** and frees the engine.
fn stream_chat(
    stream: &mut TcpStream,
    provider: &dyn TextLlm,
    req: &core_llm::TextLlmRequest,
    cancel: &CancelFlag,
    id: &str,
    model: &str,
    created: u64,
) -> io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: close\r\n\
          X-Accel-Buffering: no\r\n\r\n",
    )?;
    // If even the role chunk can't be written, the client is already gone.
    if sse(stream, &openai::role_chunk(id, model, created)).is_err() {
        cancel.cancel();
        return Ok(());
    }

    let mut disconnected = false;
    let result = {
        let mut sink = |ev: StreamEvent| {
            if disconnected {
                return;
            }
            if let StreamEvent::Token { text, .. } = ev {
                if !text.is_empty()
                    && sse(stream, &openai::content_chunk(id, model, created, &text)).is_err()
                {
                    cancel.cancel();
                    disconnected = true;
                }
            }
        };
        provider.generate(req, &mut sink)
    };

    if disconnected {
        return Ok(()); // socket is dead; nothing more to send
    }
    match result {
        Ok(out) => {
            let finish = out.finish_reason.map(openai::finish_reason_str).unwrap_or("stop");
            let _ = sse(stream, &openai::final_chunk(id, model, created, finish));
        }
        Err(CoreError::Canceled) => return Ok(()),
        Err(e) => {
            let _ = sse(stream, &openai::error_body(&e.to_string(), "server_error"));
        }
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
    Ok(())
}

/// Write one SSE event (`data: <payload>\n\n`) and flush it so the client sees it immediately.
fn sse(w: &mut impl Write, data: &str) -> io::Result<()> {
    write!(w, "data: {data}\n\n")?;
    w.flush()
}

/// Write a fixed-length JSON response with the given status.
fn write_json(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
    write_response(stream, status, "application/json", body.as_bytes())
}

/// Write a fixed-length plain-text response.
fn write_text(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
    write_response(stream, status, "text/plain; charset=utf-8", body.as_bytes())
}

fn write_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

/// Seconds since the Unix epoch (the OpenAI `created` field).
fn unix_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A per-process-monotonic completion id (`chatcmpl-…`).
fn completion_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!("chatcmpl-{:012}", N.fetch_add(1, Ordering::Relaxed))
}
