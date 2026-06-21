//! Cooperative cancellation.
//!
//! Re-exported from the backend-neutral contract crate so the engine and the contract share a
//! single cancellation type — a provider can hand a request's `CancelFlag` straight to the decode
//! loop with no bridging.

pub use core_llm::CancelFlag;
