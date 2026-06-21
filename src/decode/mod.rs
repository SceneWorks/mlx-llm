//! Streaming, cancellable decoding (story 7156).
//!
//! [`generate`] is the model-agnostic decode loop; [`Decode`] is the seam any model implements to
//! be driven by it. [`StreamEvent`]s are emitted per token through a callback. This is the internal
//! streaming API the backend-neutral `core-llm` contract (story 7154) is later extracted from.

pub mod cancel;
pub mod stream;

pub use cancel::CancelFlag;
pub use stream::{
    generate, generate_with, ConstraintMask, Decode, FinishReason, GenerationConfig,
    GenerationOutput, StreamEvent,
};
