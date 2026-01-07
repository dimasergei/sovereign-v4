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
pub mod sequence;
pub mod embeddings;
pub mod consolidation;
pub mod transferability;
pub mod selfmod;
pub mod codegen;
pub mod foundation;

// Re-export commonly used types
pub use agent::{SymbolAgent, AgentSignal, Signal, Side, Position, EntryContext};
pub use health::HealthMonitor;
pub use regime::{Regime, RegimeDetector};
pub use metalearner::{MetaLearner, AdaptationResult};
pub use learner::{Calibrator, ConfidenceCalibrator, TradeOutcome};
pub use transfer::{TransferManager, AssetCluster, get_cluster};
pub use moe::{MixtureOfExperts, Expert};
pub use weakness::{WeaknessAnalyzer, Weakness, WeaknessType};
pub use causality::{
    CausalAnalyzer, CausalGraph, CausalRelationship, CausalDirection,
    // Do-calculus types
    InterventionType, CausalQuery, QueryResult, EstimationMethod,
    CausalModel, StructuralEquation, DoCalculusEngine, DoRule,
    blocks_backdoor, find_minimal_adjustment,
};
pub use worldmodel::{WorldModel, MarketState, TransitionModel, Action, SimPosition, SimulationResult, PriceForecast, PositionDirection};
pub use counterfactual::{CounterfactualAnalyzer, CounterfactualResult, TradingInsight, InsightType, TradeForAnalysis, Direction};
pub use monitor::{AGIMonitor, AGIMetrics, AGIReport, PerformanceMetrics, SystemHealth, ExpertStats};
pub use sequence::{LSTMCell, SequenceEncoder, MarketFeatures, RegimePredictor};
pub use embeddings::{VectorIndex, TradeEmbedding, TradeContext, EmbeddingModel, IndexType, cosine_similarity};
pub use consolidation::{MemoryConsolidator, MemoryTier, Episode, EpisodeContext, Pattern, MemoryStats, ImportanceScorer, PatternExtractor};
pub use transferability::{TransferabilityPredictor, TransferabilityScore, SymbolProfile, TransferOutcome};
pub use selfmod::{
    SelfModificationEngine, Constitution, ConstitutionalGuard, RuleEngine,
    TradingRule, RuleCondition, RuleAction, RuleContext, RulePerformance,
    ModificationType, Creator, ApprovalStatus, PendingModification, AppliedModification,
};
pub use codegen::{
    CodeDeployer, CodeGenerator, GeneratedCode, CodeType, Expression, EvalContext,
    Sandbox, SafetyViolation, CompareOp, evaluate, evaluate_bool,
    TestResults, BacktestResults, CodePerformance, DeploymentStatus, DeploymentRecord,
};
pub use foundation::{
    TimeSeriesFoundation, FoundationModelType, TimeSeriesTokenizer,
    FoundationWeights, TransformerLayer, MultiHeadAttention, FeedForward, LayerNorm,
    ForecastDistribution, FoundationTransfer,
};
