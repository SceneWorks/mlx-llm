//! Real-weights acceptance for the Qwen3-VL (`qwen3_vl`) decoder (sc-8075) **and the end-to-end
//! image VLM provider** wired into `core-llm` (sc-8076).
//!
//! The first group (sc-8075) loads the standard full-attention Qwen3 text decoder from a real
//! Qwen3-VL-8B-Instruct snapshot (the VLM-wrapped `model.language_model.*` weights, 256K-context
//! interleaved-M-RoPE config) and asserts that text-only greedy generation is coherent and
//! reproducible.
//!
//! The second group (sc-8076) is the keystone end-to-end gate: it assembles stories A (ViT +
//! DeepStack), B (merger + interleaved-M-RoPE + token expansion), and C (text decoder + chat
//! template + multimodal prefill dispatch) into a working `qwen3_vl` **vision** provider, served
//! through the `core-llm` contract (`load_textllm` / the `mlx-llama` registration). It feeds
//! interleaved image+text and asserts a **grounded** answer: a solid-color image asked for its
//! dominant color is a tight check — if any stage is wrong (preprocess, ViT numerics, merger, splice,
//! interleaved M-RoPE, DeepStack fusion) the model cannot reliably name the color. It covers a
//! single image, a 2-image prompt, the `supports_vision` descriptor flag, and **on-load q4/q8**
//! quantization (the decoder projections quantize; the ViT tower stays dense — a mixed-precision
//! VLM, see the q4/q8 test doc).
//!
//! The snapshot is resolved from `QWEN3VL_SNAPSHOT` or the default HF cache; the tests **self-skip
//! cleanly** when it is absent, but run fully when present (it is present in this env).
//!
//! ```text
//! QWEN3VL_SNAPSHOT=/path/to/Qwen3-VL-8B-Instruct cargo test --test qwen3vl -- --nocapture
//! ```

use std::path::PathBuf;

use core_llm::{
    load_for_model_with, load_textllm, Channel, Content, ImageRef, LoadSpec, Message,
    ModelRequirements, Quantize, Role, Sampling, StreamEvent as CoreEvent, TextLlm, TextLlmRequest,
    ThinkingMode, Tokenizer, VideoRef,
};

use mlx_llm::config::{Architecture, ModelConfig};
use mlx_llm::decode::{generate, CancelFlag, GenerationConfig};
use mlx_llm::decode::StreamEvent;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::sampler::SamplingParams;
use mlx_llm::primitives::Weights;
use mlx_llm::provider::{eos_token_ids, PROVIDER_ID};

/// Resolve the cached Qwen3-VL-8B-Instruct snapshot (rev 0c351dd0). `QWEN3VL_SNAPSHOT` overrides.
fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QWEN3VL_SNAPSHOT") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    );
    path.exists().then_some(path)
}

/// Direct `CausalLm` load + greedy text-only generation from the real snapshot: the decoder must
/// load (36 layers, GQA 32/8, head_dim 128, vocab 151936, 256K RoPE), and a factual prompt must
/// generate coherent, reproducible text. "The capital of France is" → must mention Paris — a tight
/// end-to-end grounding check (if the weight prefix, the norm convention (standard RMSNorm), or the
/// RoPE were wrong, the model could not answer).
#[test]
fn qwen3vl_text_decoder_loads_and_generates_coherently() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };

    let cfg = ModelConfig::from_dir(&dir).expect("parse Qwen3-VL config");
    assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
    assert_eq!(cfg.num_layers, 36);
    assert_eq!(cfg.num_heads, 32);
    assert_eq!(cfg.num_kv_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.rotary_dim(), 128);
    assert_eq!(cfg.mrope_section_resolved().iter().sum::<usize>(), 64);

    let weights = Weights::from_dir(&dir).expect("load shards");
    // Prefix is ignored on the Qwen3-VL path (the decoder roots at `model.language_model`).
    let model = CausalLm::from_weights(&weights, "", cfg).expect("load Qwen3-VL text decoder");

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    // Render via the model's own chat template so the prompt is in-distribution.
    let prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids: Vec<i32> =
        tok.encode(prompt, false).unwrap().into_iter().map(|id| id as i32).collect();

    let config = GenerationConfig {
        max_new_tokens: 16,
        sampling: SamplingParams::default(), // greedy ⇒ reproducible
        seed: Some(0),
        stop_tokens: eos_token_ids(&dir),
    };

    let run = || {
        generate(&model, &prompt_ids, &config, &CancelFlag::new(), &mut |_| {})
            .unwrap()
            .tokens
    };
    let a = run();
    let b = run();
    assert!(!a.is_empty(), "expected non-empty generation");
    assert_eq!(a, b, "greedy generation must be reproducible");

    let text = tok.decode(&a.iter().map(|&i| i as u32).collect::<Vec<_>>(), true).unwrap();
    println!("\n=== Qwen3-VL text-only generation ===\n{text}\n=====================================");
    assert!(!text.trim().is_empty());
    assert!(
        text.to_lowercase().contains("paris"),
        "coherent answer must name Paris, got: {text:?}"
    );
}

/// The full `core-llm` contract path: load via the registry by id (`mlx-llama`), confirm the loaded
/// descriptor reports the Qwen3-VL family + 256K context, and stream a coherent reply.
#[test]
fn qwen3vl_round_trips_through_core_llm_contract() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    assert_eq!(provider.descriptor().id, PROVIDER_ID);
    assert_eq!(provider.descriptor().family, "qwen3_vl");
    assert_eq!(provider.descriptor().capabilities.max_context_tokens, 262144);

    let req = TextLlmRequest {
        messages: vec![Message::user("What is the capital of France? Answer in one word.")],
        sampling: Sampling::greedy(),
        max_new_tokens: 16,
        seed: Some(0),
        ..Default::default()
    };

    let mut streamed = String::new();
    let out = provider
        .generate(&req, &mut |ev| {
            if let CoreEvent::Token { text, channel, .. } = ev {
                if channel == Channel::Content {
                    streamed.push_str(&text);
                }
            }
        })
        .unwrap();
    println!("\n=== contract output ===\n{}\n=======================", out.text);
    assert!(out.usage.generated_tokens > 0);
    assert!(out.usage.prompt_tokens > 0);
    assert!(
        out.text.to_lowercase().contains("paris"),
        "coherent contract answer must name Paris, got: {:?}",
        out.text
    );
}

/// A mid-stream cancel on the real decoder stops promptly with a partial result.
#[test]
fn qwen3vl_mid_stream_cancel() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    let weights = Weights::from_dir(&dir).unwrap();
    let model = CausalLm::from_weights(&weights, "", cfg).unwrap();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prompt = "<|im_start|>user\nWrite a long story about a robot.<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids: Vec<i32> =
        tok.encode(prompt, false).unwrap().into_iter().map(|id| id as i32).collect();

    let cancel = CancelFlag::new();
    let mut count = 0;
    let out = generate(
        &model,
        &prompt_ids,
        &GenerationConfig {
            max_new_tokens: 200,
            sampling: SamplingParams::default(),
            seed: Some(0),
            stop_tokens: eos_token_ids(&dir),
        },
        &cancel,
        &mut |e| {
            if let StreamEvent::Token { .. } = e {
                count += 1;
                if count == 5 {
                    cancel.cancel();
                }
            }
        },
    )
    .unwrap();
    assert_eq!(out.finish_reason, mlx_llm::decode::FinishReason::Cancelled);
    assert!(out.tokens.len() <= 6, "cancel should stop promptly");
}

// =============================================================================================
// sc-8076 — Qwen3-VL end-to-end **image** VLM provider, served through the core-llm contract.
// =============================================================================================

/// A solid-color RGB image (`w·h` pixels, every pixel `rgb`).
fn solid_image(w: u32, h: u32, rgb: [u8; 3]) -> ImageRef {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        px.extend_from_slice(&rgb);
    }
    ImageRef::new(w, h, px).expect("image bytes")
}

/// A user turn carrying interleaved image+text content (`contents` in order).
fn user_turn(contents: Vec<Content>) -> Message {
    Message {
        role: Role::User,
        content: contents,
        thinking: None,
        tool_calls: Vec::new(),
    }
}

/// Greedy, seeded, no-think request for `messages` (reproducible grounding checks).
fn vision_request(messages: Vec<Message>, max_new_tokens: u32) -> TextLlmRequest {
    TextLlmRequest {
        messages,
        sampling: Sampling::greedy(),
        max_new_tokens,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    }
}

/// Drive a provider to completion, collecting the streamed Content-channel text.
fn run_vision(p: &dyn TextLlm, req: &TextLlmRequest) -> (String, core_llm::Usage) {
    let mut content = String::new();
    let out = p
        .generate(req, &mut |ev| {
            if let CoreEvent::Token { text, channel, .. } = ev {
                if channel == Channel::Content {
                    content.push_str(&text);
                }
            }
        })
        .expect("generate");
    (content, out.usage)
}

/// **Keystone acceptance (sc-8076 AC #1, #3):** load the real Qwen3-VL-8B snapshot end-to-end via the
/// `core-llm` provider registry (model-first `load_for_model`, the path a host uses to resolve a
/// provider by snapshot alone), confirm the loaded descriptor advertises **vision**, then send a
/// single image + a grounded question and assert the answer names the image's color.
///
/// `load_for_model` (default requirements) must route this snapshot to `mlx-llama` and **not**
/// misroute it to a different vision provider (JoyCaption / LLaVA) — `mlx-llama` is the only provider
/// whose weightless `can_load` claims a `qwen3_vl` wrapper. A solid-color image asked for its
/// dominant color exercises the whole fused stack (preprocess → ViT → DeepStack → merger → splice →
/// interleaved M-RoPE → decode); a break anywhere keeps it from naming the color reliably.
#[test]
fn qwen3vl_vision_grounds_single_image_via_core_llm() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };

    // Model-first resolution: by snapshot alone, with no vision requirement asserted at the weightless
    // probe (the descriptor's vision flag is only known post-load — the worker's vision-required
    // routing gate is hardened separately in sc-8077). This must land on `mlx-llama`, family
    // `qwen3_vl`, and report vision once loaded.
    let provider = load_for_model_with(
        &LoadSpec::dense(dir.to_str().unwrap()),
        &ModelRequirements::default(),
    )
    .expect("load_for_model must resolve the Qwen3-VL snapshot to a provider");
    assert_eq!(provider.descriptor().id, PROVIDER_ID, "must route to mlx-llama, not a misrouted provider");
    assert_eq!(provider.descriptor().family, "qwen3_vl");
    assert!(
        provider.descriptor().capabilities.supports_vision,
        "a loaded Qwen3-VL checkpoint (model.visual.*) must advertise vision"
    );

    // Two solid colors: a correct vision path names each; a broken one cannot ground on both.
    for (rgb, want, label) in [([205u8, 35, 35], "red", "red"), ([35u8, 70, 200], "blue", "blue")] {
        let req = vision_request(
            vec![user_turn(vec![
                Content::Image(solid_image(256, 256, rgb)),
                Content::text("What is the dominant color of this image? Answer with one word."),
            ])],
            16,
        );
        let (content, usage) = run_vision(provider.as_ref(), &req);
        println!("\n=== Qwen3-VL VISION single ({label}) ===\n[answer] {content:?}\n");
        assert!(usage.prompt_tokens > 0 && usage.generated_tokens > 0);
        assert!(!content.trim().is_empty(), "{label}: must produce an answer");
        assert!(
            content.to_lowercase().contains(want),
            "{label}: greedy answer must ground on the image and name '{want}', got: {content:?}"
        );
    }
}

/// **sc-8077 (the story-D gap, closed):** model-first **vision-required** routing must resolve the
/// Qwen3-VL snapshot to `mlx-llama` at the pre-load capability gate — `load_for_model_with(spec,
/// ModelRequirements::with_vision())`, the path a worker uses when a request carries an image.
///
/// Before sc-8077 this REJECTED a genuine Qwen3-VL snapshot: the provider's weightless descriptor
/// reports `supports_vision=false` (it also serves text-only checkpoints), so core-llm's gate
/// (`meets`) filtered `mlx-llama` out and returned `Error::Unsupported`. Only the id-based
/// (`load_textllm`) / default-requirements path worked. The weightless per-snapshot vision probe
/// (`provider::weightless_vision`, wired into the registration) closes the gap. This asserts the
/// vision-required load now resolves to `mlx-llama` (NOT a misrouted JoyCaption/LLaVA provider) and,
/// loaded, grounds on an image.
#[test]
fn qwen3vl_vision_required_routing_resolves_via_core_llm() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let spec = LoadSpec::dense(dir.to_str().unwrap());

    // The pre-load weightless gate (no weights loaded): the provider's probe advertises vision for
    // this snapshot from config.json alone, so a vision-required model-first load resolves it.
    assert!(
        mlx_llm::provider::weightless_vision(&spec),
        "the weightless vision probe must advertise vision for a real Qwen3-VL snapshot"
    );

    // The gap closed: with_vision() requirement no longer rejects the snapshot at the gate.
    let provider = load_for_model_with(&spec, &ModelRequirements::default().with_vision())
        .expect("vision-required model-first load must resolve the Qwen3-VL snapshot (sc-8077 gap)");
    assert_eq!(
        provider.descriptor().id, PROVIDER_ID,
        "vision-required routing must land on mlx-llama, not a misrouted vision provider"
    );
    assert_eq!(provider.descriptor().family, "qwen3_vl");
    assert!(
        provider.descriptor().capabilities.supports_vision,
        "the loaded Qwen3-VL checkpoint must advertise vision"
    );

    // And it actually grounds on an image through the vision-required-resolved provider.
    let req = vision_request(
        vec![user_turn(vec![
            Content::Image(solid_image(256, 256, [205, 35, 35])),
            Content::text("What is the dominant color of this image? Answer with one word."),
        ])],
        16,
    );
    let (content, usage) = run_vision(provider.as_ref(), &req);
    println!("\n=== Qwen3-VL VISION-REQUIRED routing ===\n[answer] {content:?}\n");
    assert!(usage.generated_tokens > 0);
    assert!(
        content.to_lowercase().contains("red"),
        "vision-required-resolved provider must ground on the image, got: {content:?}"
    );
}

/// **sc-8076 AC #2:** a 2-image prompt works end-to-end. Two solid-color images are interleaved with
/// text in a single user turn; the model must name the first image's color and the second's (in
/// order). This exercises multi-image preprocessing, per-image patch-count placeholder expansion,
/// feature concatenation, and the interleaved M-RoPE position assignment across two images.
#[test]
fn qwen3vl_vision_grounds_two_images_via_core_llm() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(dir.to_str().unwrap())).unwrap();

    let req = vision_request(
        vec![user_turn(vec![
            Content::text("Here are two images. The first:"),
            Content::Image(solid_image(256, 256, [205, 35, 35])),
            Content::text("The second:"),
            Content::Image(solid_image(256, 256, [35, 70, 200])),
            Content::text("Name the dominant color of the first image, then the second, as two words."),
        ])],
        32,
    );
    let (content, usage) = run_vision(provider.as_ref(), &req);
    println!("\n=== Qwen3-VL VISION multi-image ===\n[answer] {content:?}\n");
    assert!(usage.generated_tokens > 0);
    let lc = content.to_lowercase();
    assert!(lc.contains("red"), "multi-image answer must name the first image (red), got: {content:?}");
    assert!(lc.contains("blue"), "multi-image answer must name the second image (blue), got: {content:?}");
    assert!(
        lc.find("red") < lc.find("blue"),
        "multi-image answer must name the images in order (red before blue), got: {content:?}"
    );
}

/// **sc-8076 AC #4 — q4/q8 import path verified.** Load the Qwen3-VL snapshot with on-load `Q4` then
/// `Q8` quantization and assert each (a) actually quantizes the decoder projections and (b) still
/// grounds on an image.
///
/// **Scope note (mixed-precision VLM):** mlx-llm's load-time quantization applies to the **text
/// decoder** projections (`CausalLm::from_weights_with(.., quant)`); the **ViT tower**
/// (`Qwen35VisionModel::from_weights`) takes no quant spec and stays dense — its `linear()` is a
/// plain float matmul with no quantized seam. So a q4/q8 Qwen3-VL load is a *mixed-precision* VLM:
/// quantized decoder + dense vision encoder. That is sufficient for the import/serve path this AC
/// requires and is verified here against the real snapshot. A fully-quantized ViT (and loading a
/// snapshot whose ViT weights are pre-quantized) is a separate enhancement, tracked as a follow-up.
#[test]
fn qwen3vl_vision_q4_q8_import_path() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let source = dir.to_str().unwrap().to_string();

    for (q, rgb, want, label) in [
        (Quantize::Q4, [205u8, 35, 35], "red", "q4"),
        (Quantize::Q8, [35u8, 70, 200], "blue", "q8"),
    ] {
        let spec = LoadSpec { source: source.clone(), quantize: Some(q) };

        // The concrete provider exposes `is_quantized()` — assert the decoder actually quantized
        // (not silently fell back to dense).
        let concrete = mlx_llm::LlamaProvider::load(&spec).expect("load quantized Qwen3-VL");
        assert!(concrete.is_quantized(), "{label}: decoder projections must be quantized on load");
        assert!(
            concrete.descriptor().capabilities.supports_vision,
            "{label}: quantized Qwen3-VL must still advertise vision"
        );

        // And serve a grounded image through the contract registry, quantized.
        let provider = load_textllm(PROVIDER_ID, &spec).unwrap();
        let req = vision_request(
            vec![user_turn(vec![
                Content::Image(solid_image(256, 256, rgb)),
                Content::text("What is the dominant color of this image? Answer with one word."),
            ])],
            16,
        );
        let (content, _) = run_vision(provider.as_ref(), &req);
        println!("\n=== Qwen3-VL VISION {label} ({want}) ===\n[answer] {content:?}\n");
        assert!(
            content.to_lowercase().contains(want),
            "{label}: quantized greedy answer must ground on the image and name '{want}', got: {content:?}"
        );
    }
}

// =============================================================================================
// sc-8081 — Qwen3-VL VIDEO modality + Text–Timestamp Alignment, end-to-end via the core-llm contract.
// =============================================================================================

/// A solid-color RGB video **frame** of `w·h` pixels.
fn solid_frame(w: u32, h: u32, rgb: [u8; 3]) -> ImageRef {
    solid_image(w, h, rgb)
}

/// Build a sampled video from solid-color frames, at `fps`, with the frame timestamps the host would
/// derive from the sampled frame indices (`idx / fps`). 64×64 keeps the video preprocess resize a
/// no-op (a clean, fast grid) while still exercising the full multi-frame ViT path.
fn solid_video(colors: &[[u8; 3]], fps: f32) -> VideoRef {
    let frames: Vec<ImageRef> = colors.iter().map(|&c| solid_frame(64, 64, c)).collect();
    let timestamps: Vec<f32> = (0..colors.len()).map(|i| i as f32 / fps).collect();
    VideoRef::new(frames, timestamps).expect("video frames + timestamps")
}

/// **sc-8081 AC #1 — a short video prompt produces a temporally-grounded answer.** Load the real
/// Qwen3-VL-8B snapshot end-to-end via the core-llm provider; confirm it advertises **video**; then
/// feed a short synthetic video whose color changes over time (red → blue) and ask which color comes
/// **first**. A correct end-to-end video path (frame sampling layout → multi-frame ViT → per-frame
/// `<|video_pad|>` expansion → Text–Timestamp-Alignment timestamps → interleaved-M-RoPE per-frame time
/// axis → DeepStack fusion → decode) is what lets the model order the frames; a break anywhere makes
/// the temporal answer unreliable. The model's exact phrasing is logged.
#[test]
fn qwen3vl_video_grounds_temporal_order_via_core_llm() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let provider = load_for_model_with(
        &LoadSpec::dense(dir.to_str().unwrap()),
        &ModelRequirements::default(),
    )
    .expect("load_for_model must resolve the Qwen3-VL snapshot");
    assert_eq!(provider.descriptor().id, PROVIDER_ID, "must route to mlx-llama");
    assert_eq!(provider.descriptor().family, "qwen3_vl");
    assert!(
        provider.descriptor().capabilities.supports_video,
        "a loaded Qwen3-VL checkpoint must advertise video (config carries video_token_id)"
    );

    // A video that is solid RED for the first half then solid BLUE for the second half. Four sampled
    // frames at 1 fps fold (temporal_patch_size=2) into **two** merged temporal patches — patch 0 =
    // red+red (≈0.5s), patch 1 = blue+blue (≈2.5s) — so the model sees two distinct timestamped
    // vision frames in order. (Two frames would fold into a single patch with no temporal extent.)
    let video = solid_video(&[[205, 35, 35], [205, 35, 35], [35, 70, 200], [35, 70, 200]], 1.0);
    let req = vision_request(
        vec![user_turn(vec![
            Content::Video(video),
            Content::text(
                "This is a short video. What color is shown at the start, and what color is shown \
                 at the end? Answer with two color words: first the starting color, then the ending color.",
            ),
        ])],
        48,
    );
    let (content, usage) = run_vision(provider.as_ref(), &req);
    println!("\n=== Qwen3-VL VIDEO temporal-order ===\n[answer] {content:?}\n");
    assert!(usage.prompt_tokens > 0 && usage.generated_tokens > 0, "must run the video prefill");
    assert!(!content.trim().is_empty(), "must produce an answer");
    let lc = content.to_lowercase();
    // The video path must ground on *both* frames' colors. Temporal ordering (red before blue) is the
    // strong claim; if the 8B model names them out of order on synthetic input we still require both
    // colors to appear (proving both frames reached the model with distinct content).
    assert!(
        lc.contains("red") && lc.contains("blue"),
        "video answer must ground on both frames (red and blue), got: {content:?}"
    );
    if let (Some(r), Some(b)) = (lc.find("red"), lc.find("blue")) {
        // Log whether temporal order was correct; this is the grounding signal we care about.
        println!(
            "temporal order {}: red@{r} blue@{b}",
            if r < b { "CORRECT (red before blue)" } else { "out-of-order" }
        );
        assert!(r < b, "temporally-grounded answer must name the starting color (red) before the ending color (blue), got: {content:?}");
    }
}

/// **sc-8081 — temporal grounding on a second, independent video (order reversed).** A
/// complementary grounding check with the colors in the *opposite* order (blue first, then green) so
/// a model that simply always emits a fixed pair cannot pass both this and the red→blue test. The
/// four frames fold into two merged temporal patches (blue@~0.5s, green@~2.5s); a temporally-grounded
/// answer names blue before green. Exercises the same Text–Timestamp-Alignment path on different
/// content. The model's exact phrasing is logged.
#[test]
fn qwen3vl_video_temporal_order_reversed_via_core_llm() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(dir.to_str().unwrap())).unwrap();

    // Solid BLUE for the first half, then solid GREEN for the second half.
    let video = solid_video(&[[35, 70, 200], [35, 70, 200], [40, 180, 60], [40, 180, 60]], 1.0);
    let req = vision_request(
        vec![user_turn(vec![
            Content::Video(video),
            Content::text(
                "This is a short video. What color is shown at the start, and what color is shown \
                 at the end? Answer with two color words: first the starting color, then the ending color.",
            ),
        ])],
        48,
    );
    let (content, usage) = run_vision(provider.as_ref(), &req);
    println!("\n=== Qwen3-VL VIDEO temporal-order (reversed) ===\n[answer] {content:?}\n");
    assert!(usage.generated_tokens > 0);
    let lc = content.to_lowercase();
    assert!(!lc.trim().is_empty(), "must produce an answer");
    assert!(
        lc.contains("blue") && lc.contains("green"),
        "video answer must ground on both frames (blue and green), got: {content:?}"
    );
    if let (Some(b), Some(g)) = (lc.find("blue"), lc.find("green")) {
        println!(
            "temporal order {}: blue@{b} green@{g}",
            if b < g { "CORRECT (blue before green)" } else { "out-of-order" }
        );
        assert!(b < g, "temporally-grounded answer must name the starting color (blue) before the ending color (green), got: {content:?}");
    }
}

/// **sc-8081 AC #2 (representation sanity) — the descriptor advertises video and a video request is
/// accepted.** A provider-level check that the video capability flag is wired and a `Content::Video`
/// request validates (the contract gate from core-llm) on the real snapshot.
#[test]
fn qwen3vl_video_capability_and_validate() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
        return;
    };
    let provider = load_textllm(PROVIDER_ID, &LoadSpec::dense(dir.to_str().unwrap())).unwrap();
    assert!(provider.descriptor().capabilities.supports_video);
    let req = vision_request(
        vec![user_turn(vec![
            Content::Video(solid_video(&[[0, 0, 0], [255, 255, 255]], 2.0)),
            Content::text("Describe this video."),
        ])],
        8,
    );
    provider.validate(&req).expect("a video request must validate on a video-capable provider");
}
