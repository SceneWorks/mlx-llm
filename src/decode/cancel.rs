//! Cooperative cancellation.
//!
//! A cheap clonable flag the decode loop polls between tokens. This is the concrete-first version;
//! the backend-neutral `core-llm` contract (story 7154) defines the shared cancellation type the
//! engine will ultimately expose, and this bridges to it at the trait boundary.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A shared, thread-safe "please stop" flag.
#[derive(Clone, Default, Debug)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    /// A fresh, un-cancelled flag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Clear the flag (reuse across generations).
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_round_trips_and_shares() {
        let a = CancelFlag::new();
        let b = a.clone();
        assert!(!a.is_cancelled());
        b.cancel();
        assert!(a.is_cancelled()); // clone shares state
        a.reset();
        assert!(!b.is_cancelled());
    }
}
