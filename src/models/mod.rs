//! Concrete model decoders built on the [`crate::primitives`].
//!
//! Story 7156 ships the generic Llama decoder. Each decoder uses an **immutable `&self` forward**
//! and a `from_weights` constructor — deliberately *not* mlx-rs's `&mut self` `Module` trait, so a
//! single loaded model can be shared and driven concurrently in the batch dimension later. The only
//! mutable state in a forward pass is the KV cache, threaded in as `&mut dyn KvCache`.

pub(crate) mod deepstack;
pub mod llama;
pub mod qwen35;
pub mod qwen35_vision;
pub mod siglip;

pub use llama::CausalLm;
pub use qwen35::{Qwen35Cache, Qwen35Config, Qwen35Model};
pub use qwen35_vision::{
    Qwen35VisionConfig, Qwen35VisionModel, Qwen35VisionOutput, Qwen3VLVisionConfig,
    Qwen3VLVisionModel, Qwen3VLVisionOutput,
};
pub use siglip::{SiglipVisionConfig, SiglipVisionTower};

use mlx_rs::Array;

use crate::decode::Decode;
use crate::error::Result;
use crate::models::deepstack::MropePositions;
use crate::primitives::kv_cache::KvCache;

/// The multimodal (Qwen-VL) decoder seam: everything the provider needs to build an image/video
/// prefill and decode, independent of which backbone powers the VLM — the Qwen3.6 hybrid
/// ([`Qwen35Model`], Gated-DeltaNet linear attention) or the generic full-attention decoder
/// ([`CausalLm`], which serves Qwen3-VL).
///
/// Both backbones share the *same* vision tower (`Qwen3VLVisionModel == Qwen35VisionModel`) and the
/// *same* multimodal conventions (interleaved M-RoPE + DeepStack); only the decoder math differs.
/// This trait lets the provider drive vision **once** through `&dyn VlmDecode` instead of forking on
/// the concrete decoder type. The method bodies remain the decoders' own (backend- and
/// architecture-specific inherent methods); the trait just unifies the dispatch.
pub trait VlmDecode: Decode {
    /// Embed token ids `[1, S]` → `[1, S, hidden]` in the compute dtype — the splice point where the
    /// multimodal path overwrites placeholder rows with the vision tower's merged patch features.
    fn embed_input_ids(&self, input_ids: &Array) -> Result<Array>;

    /// Replace every row whose id is any of `placeholder_tokens` (`<|image_pad|>` / `<|video_pad|>`)
    /// with the next `vision_features` row, in sequence order — the mixed image+video splice.
    fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array>;

    /// Interleaved-M-RoPE 3-D position rows (temporal/height/width) + the `mrope_delta`, computed
    /// over the image **and** video grids.
    #[allow(clippy::too_many_arguments)]
    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<MropePositions>;

    /// Prefill precomputed `embeds` `[1, S, hidden]` with interleaved M-RoPE from the explicit 3-D
    /// `positions` **and DeepStack feature fusion**, returning last-position logits `[1, vocab]`.
    /// `cache` is the decoder's own cache (from [`Decode::make_cache`]); each impl downcasts it
    /// internally. Unifies the two backbones' deepstack-prefill methods under one name.
    fn prefill_with_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array>;
}
