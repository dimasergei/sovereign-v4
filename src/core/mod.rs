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

// Re-export commonly used types
pub use types::{Candle, Tick};
pub use sr::{SRLevels, PriceRange};
pub use capitulation::{VolumeTracker, CapitulationSignal};
pub use agent::{SymbolAgent, AgentSignal, Signal, Side, Position};
pub use health::HealthMonitor;

// Keep guardian for risk management (this is NOT strategy parameters)
pub mod guardian;
pub use guardian::{RiskGuardian, RiskConfig};
