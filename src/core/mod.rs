//! Core trading logic
//!
//! This module implements the "lossless" trading algorithm based on
//! pftq's Tech Trader philosophy:
//!
//! - No parameters, no thresholds
//! - Pure counting-based S/R detection
//! - Volume capitulation for entry signals
//! - One independent agent per symbol

pub mod types;
pub mod sr;
pub mod capitulation;
pub mod agent;
pub mod health;
pub mod regime;
pub mod learner;
pub mod transfer;
pub mod moe;

// Re-export commonly used types
pub use agent::{SymbolAgent, AgentSignal, Signal, Side, Position, EntryContext};
pub use health::HealthMonitor;
pub use regime::{Regime, RegimeDetector};
pub use learner::{ConfidenceCalibrator, TradeOutcome};
pub use transfer::{TransferManager, AssetCluster, get_cluster};
pub use moe::{MixtureOfExperts, Expert};
