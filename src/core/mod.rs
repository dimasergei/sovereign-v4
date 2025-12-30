//! Core trading logic module
//! 
//! Contains:
//! - Lossless algorithms (support/resistance, trend, momentum)
//! - Agent implementation
//! - Coordinator
//! - Risk guardian

pub mod lossless;
pub mod agent;
pub mod coordinator;
pub mod guardian;
pub mod types;

pub use types::*;
