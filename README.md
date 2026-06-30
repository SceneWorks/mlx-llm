# mlx-llm

An **on-device text + vision LLM serving engine** for Apple [MLX](https://github.com/ml-explore/mlx).

`mlx-llm` is the MLX backend of an independent, backend-neutral LLM serving library. It owns its
own tensor primitives (depending only on `mlx-rs`) and implements the backend-neutral
[`core-llm`](https://github.com/SceneWorks/core-llm) contract; a sibling
[`candle-llm`](https://github.com/SceneWorks/candle-llm) crate provides the Candle backend behind
the same contract.

> Work in progress. The engine streams text from real Llama snapshots today; broader architecture
> coverage, batching/throughput, and the published API are landing incrementally.

## What works today

- **Tensor primitives** — batch-capable KV cache (behind a `KvCache` trait), a unified sampler
  (greedy / temperature / top-p / top-k / repetition-penalty over a pluggable RNG), the RoPE family
  (standard / Qwen3 / Llama-3 scaled), GQA attention helpers, group-wise Q4/Q8 quantization, and a
  safetensors weights loader.
- **Generic causal decoder** — `&self` forward + `from_weights`, GQA + SwiGLU + RMSNorm + RoPE.
  **Architecture dispatch** from `config.json` covers Llama / Mistral and Qwen3 (per-head q/k
  RMSNorm); loaded from any Hugging Face snapshot directory.
- **Quantize-on-load** — group-wise affine Q4/Q8 of the attention/MLP projections.
- **GGUF conversion** — bring a llama.cpp `*.gguf` and convert it to a loadable MLX snapshot:
  dequantizes F16/BF16, the legacy `Q*_0/_1`, the `Q2_K…Q6_K` k-quants, and the non-linear
  `IQ4_NL`/`IQ4_XS`; remaps llama.cpp tensor names + un-permutes q/k RoPE; rebuilds `config.json`;
  optionally re-quantizes to MLX Q4/Q8. Verified at parity with the HF safetensors load on Llama and
  Qwen3.
- **Streaming, cancellable decode loop** — drives any `Decode` model, emitting a `StreamEvent` per
  token, with cooperative mid-stream cancellation.
- **`core-llm` contract** — `LlamaProvider` implements `core_llm::TextLlm`, renders the model's own
  chat template, and registers as `mlx-llama`.
- **Text + vision (VLM)** — JoyCaption (`LlavaForConditionalGeneration`): a SigLIP2 vision tower +
  LLaVA projector + image-token splice in front of the reused Llama decode, served as a multimodal
  `TextLlm` provider (`mlx-joycaption`, `supports_vision`). Verified token-for-token against the
  reference engine, including the bicubic resize of non-square images.

## Example

Stream a completion from a local snapshot (config.json + tokenizer.json + `*.safetensors`):

```sh
cargo run --release --example generate -- /path/to/Llama-snapshot "Once upon a time" 64
```

Convert a llama.cpp GGUF to a loadable snapshot (add `--quant q4|q8` to re-quantize, `--tokenizer
tokenizer.json` to copy a tokenizer in for end-to-end use):

```sh
cargo run --release --example convert_gguf -- model.gguf /path/to/out --tokenizer tokenizer.json
```

Serve an **OpenAI-compatible** chat endpoint (`server/`, a separate bin so its HTTP deps stay out of
the engine — streaming SSE, single model, backend-neutral over the `core_llm::TextLlm` contract):

```sh
cargo run --release -p mlx-llm-server -- --model /path/to/snapshot --port 8080
curl -N http://localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"Hi!"}]}'
```

```rust
use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig, StreamEvent};
use mlx_llm::primitives::Weights;

let cfg = ModelConfig::from_dir(dir)?;
let model = CausalLm::from_weights(&Weights::from_dir(dir)?, "", cfg)?;

let mut out_ids = Vec::new();
generate(&model, &prompt_ids, &GenerationConfig::default(), &CancelFlag::new(), &mut |ev| {
    if let StreamEvent::Token { id, .. } = ev {
        out_ids.push(id);
    }
})?;
```

## Building

`mlx-llm` builds MLX from source via `mlx-rs` (pinned to the pmetal fork, matching `mlx-gen` so the
two share one `Array` type). The first build is slow; subsequent builds are incremental. macOS with
Apple Silicon required.

```sh
cargo test          # unit + synthetic streaming tests (no model weights needed)
# Real-weights end-to-end test (downloads/points at a Llama snapshot):
MLX_LLM_TEST_MODEL=/path/to/snapshot cargo test --test real_weights -- --ignored --nocapture
```

## License

Apache-2.0.
