//! Real-weights end-to-end test for Qwen3.6 **vision** (`qwen3_5` VLM, story sc-7701) — the epic
//! acceptance gate. Point `MLX_LLM_QWEN35_MODEL` at the 27B snapshot (the one carrying `model.visual.*`):
//!
//! ```text
//! MLX_LLM_QWEN35_MODEL=/path/to/Qwen3.6-27B \
//!   cargo test --test qwen35_vision -- --ignored --nocapture
//! ```
//!
//! This exercises the whole vision stack on real weights: image → preprocess → ViT encoder → merger
//! → splice into the decoder embeds → interleaved M-RoPE → greedy decode. A solid-color image asked
//! for its dominant color is a tight grounding check: if any stage is wrong (preprocess layout,
//! encoder numerics, splice, M-RoPE positions), the model cannot reliably name the color.

use core_llm::{
    Channel, Content, ImageRef, LoadSpec, Message, Role, Sampling, StreamEvent, TextLlm,
    TextLlmOutput, TextLlmRequest, ThinkingMode,
};
use mlx_llm::LlamaProvider;

fn model_dir() -> String {
    std::env::var("MLX_LLM_QWEN35_MODEL").expect("set MLX_LLM_QWEN35_MODEL")
}

/// A solid-color RGB image.
fn solid_image(w: u32, h: u32, rgb: [u8; 3]) -> ImageRef {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        px.extend_from_slice(&rgb);
    }
    ImageRef::new(w, h, px).expect("image bytes")
}

/// An image + text user turn.
fn image_request(img: ImageRef, prompt: &str) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content: vec![Content::Image(img), Content::text(prompt)],
            thinking: None,
            tool_calls: Vec::new(),
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 32,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    }
}

fn run(p: &dyn TextLlm, r: &TextLlmRequest) -> (TextLlmOutput, String) {
    let mut content = String::new();
    let out = p
        .generate(r, &mut |ev| {
            if let StreamEvent::Token { text, channel, .. } = ev {
                if channel == Channel::Content {
                    content.push_str(&text);
                }
            }
        })
        .expect("generate");
    (out, content)
}

#[test]
#[ignore = "needs a Qwen3.6 27B VLM snapshot (model.visual.*) via MLX_LLM_QWEN35_MODEL"]
fn qwen35_vision_grounds_on_image() {
    let p = LlamaProvider::load(&LoadSpec::dense(model_dir())).expect("load qwen3.6 VLM");
    assert!(
        p.descriptor().capabilities.supports_vision,
        "a qwen3_5 checkpoint carrying model.visual.* must advertise vision"
    );

    // Two solid colors: a correct vision path names each; a broken one (preprocess / encoder / splice
    // / M-RoPE) cannot ground reliably, and certainly not on both.
    for (rgb, want, label) in [([205u8, 35, 35], "red", "red"), ([35u8, 70, 200], "blue", "blue")] {
        let img = solid_image(256, 256, rgb);
        let (out, content) = run(
            &p,
            &image_request(img, "What is the dominant color of this image? Answer with one word."),
        );
        println!("\n=== qwen3.6 VISION ({label}) ===\n[answer] {:?}\n", out.text);
        assert!(!content.trim().is_empty(), "{label}: must produce an answer");
        assert!(
            content.to_lowercase().contains(want),
            "{label}: greedy answer must ground on the image and name '{want}', got: {content:?}"
        );
    }
}
