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
pub mod metalearner;
pub mod learner;
pub mod transfer;
pub mod moe;
pub mod weakness;
pub mod causality;
pub mod worldmodel;
pub mod counterfactual;
pub mod monitor;

// Re-export commonly used types
pub use agent::{SymbolAgent, AgentSignal, Signal, Side, Position, EntryContext};
pub use health::HealthMonitor;
pub use regime::{Regime, RegimeDetector};
pub use metalearner::{MetaLearner, AdaptationResult};
pub use learner::{Calibrator, ConfidenceCalibrator, TradeOutcome};
pub use transfer::{TransferManager, AssetCluster, get_cluster};
pub use moe::{MixtureOfExperts, Expert};
pub use weakness::{WeaknessAnalyzer, Weakness, WeaknessType};
pub use causality::{CausalAnalyzer, CausalGraph, CausalRelationship, CausalDirection};
pub use worldmodel::{WorldModel, MarketState, TransitionModel, Action, SimPosition, SimulationResult, PriceForecast, PositionDirection};
pub use counterfactual::{CounterfactualAnalyzer, CounterfactualResult, TradingInsight, InsightType, TradeForAnalysis, Direction};
pub use monitor::{AGIMonitor, AGIMetrics, AGIReport, PerformanceMetrics, SystemHealth, ExpertStats};
