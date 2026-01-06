//! AGI Monitoring and Metrics
//!
//! Comprehensive monitoring system for all AGI components:
//! - Memory metrics (trades, S/R levels, regimes)
//! - Learning metrics (calibrator, EWC, MoE)
//! - Meta-learning metrics (adaptations, success rate)
//! - Transfer metrics (clusters, transfers)
//! - Weakness metrics (identified, skipped trades)
//! - Causal metrics (relationships, adjustments)
//! - World model metrics (simulations, forecasts)
//! - Counterfactual metrics (analyses, insights)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::info;

use super::regime::Regime;
use super::learner::ConfidenceCalibrator;
use super::transfer::TransferManager;
use super::moe::MixtureOfExperts;
use super::metalearner::MetaLearner;
use super::weakness::WeaknessAnalyzer;
use super::causality::CausalAnalyzer;
use super::worldmodel::WorldModel;
use super::counterfactual::CounterfactualAnalyzer;
use crate::data::memory::TradeMemory;

/// Metrics for a single regime expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertStats {
    pub regime: Regime,
    pub trades: u32,
    pub win_rate: f64,
}

/// AGI component metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AGIMetrics {
    /// When these metrics were collected
    pub timestamp: DateTime<Utc>,

    // Memory metrics
    /// Total trades stored in memory
    pub total_trades_stored: u32,
    /// S/R levels being tracked
    pub sr_levels_tracked: u32,
    /// Regimes recorded in memory
    pub regimes_recorded: u32,

    // Learning metrics
    /// Number of calibrator weight updates
    pub calibrator_updates: u32,
    /// Recent prediction accuracy
    pub calibrator_accuracy: f64,
    /// Number of EWC consolidations
    pub ewc_consolidations: u32,
    /// Whether EWC is currently protecting weights
    pub ewc_active: bool,

    // MoE metrics
    /// Whether MoE is enabled
    pub moe_enabled: bool,
    /// Stats for each regime expert
    pub expert_stats: Vec<ExpertStats>,
    /// Current gating weights
    pub gating_weights: [f64; 4],

    // Meta-learning metrics
    /// Total adaptations performed
    pub meta_adaptations: u32,
    /// Average trades to reach 55% accuracy
    pub avg_adaptation_speed: f64,
    /// Percentage of successful adaptations
    pub meta_success_rate: f64,

    // Transfer metrics
    /// Trade counts per cluster
    pub cluster_trade_counts: HashMap<String, u32>,
    /// Number of transfers applied
    pub transfers_applied: u32,
    /// Success rate of transfers
    pub transfer_success_rate: f64,

    // Weakness metrics
    /// Total weaknesses identified
    pub weaknesses_identified: u32,
    /// Critical weaknesses (severity > 0.7)
    pub critical_weaknesses: u32,
    /// Trades skipped due to weakness
    pub trades_skipped_for_weakness: u32,
    /// Position size adjustments made
    pub position_size_adjustments: u32,

    // Causal metrics
    /// Total causal relationships discovered
    pub causal_relationships: u32,
    /// Significant relationships (confidence > threshold)
    pub significant_relationships: u32,
    /// Confidence adjustments from causal analysis
    pub causal_confidence_adjustments: u32,

    // World model metrics
    /// Monte Carlo simulations run
    pub simulation_runs: u32,
    /// Forecast direction accuracy
    pub forecast_accuracy: f64,
    /// Trades skipped by world model
    pub trades_skipped_by_worldmodel: u32,

    // Counterfactual metrics
    /// Counterfactual analyses performed
    pub counterfactuals_analyzed: u32,
    /// Insights generated
    pub insights_generated: u32,
    /// Systematic errors identified
    pub systematic_errors_found: u32,
    /// Estimated improvement if insights followed
    pub estimated_improvement_if_followed: f64,
}

impl Default for AGIMetrics {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            total_trades_stored: 0,
            sr_levels_tracked: 0,
            regimes_recorded: 0,
            calibrator_updates: 0,
            calibrator_accuracy: 0.0,
            ewc_consolidations: 0,
            ewc_active: false,
            moe_enabled: false,
            expert_stats: Vec::new(),
            gating_weights: [0.25; 4],
            meta_adaptations: 0,
            avg_adaptation_speed: 0.0,
            meta_success_rate: 0.0,
            cluster_trade_counts: HashMap::new(),
            transfers_applied: 0,
            transfer_success_rate: 0.0,
            weaknesses_identified: 0,
            critical_weaknesses: 0,
            trades_skipped_for_weakness: 0,
            position_size_adjustments: 0,
            causal_relationships: 0,
            significant_relationships: 0,
            causal_confidence_adjustments: 0,
            simulation_runs: 0,
            forecast_accuracy: 0.0,
            trades_skipped_by_worldmodel: 0,
            counterfactuals_analyzed: 0,
            insights_generated: 0,
            systematic_errors_found: 0,
            estimated_improvement_if_followed: 0.0,
        }
    }
}

/// Performance metrics for trading results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of trades
    pub total_trades: u32,
    /// Number of winning trades
    pub wins: u32,
    /// Number of losing trades
    pub losses: u32,
    /// Win rate (wins / total)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total profit/loss
    pub total_pnl: f64,
    /// Maximum drawdown experienced
    pub max_drawdown: f64,
    /// Sharpe ratio (risk-adjusted returns)
    pub sharpe_ratio: f64,
    /// Average hold time in bars
    pub avg_hold_time_bars: f64,
    /// Best performing regime
    pub best_regime: Option<(Regime, f64)>,
    /// Worst performing regime
    pub worst_regime: Option<(Regime, f64)>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_trades: 0,
            wins: 0,
            losses: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            total_pnl: 0.0,
            max_drawdown: 0.0,
            sharpe_ratio: 0.0,
            avg_hold_time_bars: 0.0,
            best_regime: None,
            worst_regime: None,
        }
    }
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Time running in seconds
    pub uptime_seconds: u64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Last trade timestamp
    pub last_trade_at: Option<DateTime<Utc>>,
    /// Last signal timestamp
    pub last_signal_at: Option<DateTime<Utc>>,
    /// Errors in last hour
    pub errors_last_hour: u32,
    /// Warnings in last hour
    pub warnings_last_hour: u32,
    /// Health status per component
    pub components_healthy: HashMap<String, bool>,
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self {
            uptime_seconds: 0,
            memory_usage_mb: 0.0,
            last_trade_at: None,
            last_signal_at: None,
            errors_last_hour: 0,
            warnings_last_hour: 0,
            components_healthy: HashMap::new(),
        }
    }
}

/// Comprehensive AGI report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AGIReport {
    /// When report was generated
    pub generated_at: DateTime<Utc>,
    /// Overall AGI progress (0.0 to 1.0)
    pub agi_progress_pct: f64,
    /// Learning velocity (rate of improvement)
    pub learning_velocity: f64,
    /// Current AGI metrics
    pub current_metrics: AGIMetrics,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// System health
    pub health: SystemHealth,
    /// Top insights from analysis
    pub top_insights: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Component status (healthy, message)
    pub component_status: HashMap<String, (bool, String)>,
}

impl fmt::Display for AGIReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           SOVEREIGN AGI PROGRESS REPORT                      ║")?;
        writeln!(f, "║           Generated: {}                     ║", self.generated_at.format("%Y-%m-%d %H:%M UTC"))?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;

        // AGI Progress Bar
        let progress_bar = self.render_progress_bar(self.agi_progress_pct, 30);
        writeln!(f, "║  AGI Progress: {} {:.1}%  ║", progress_bar, self.agi_progress_pct * 100.0)?;
        writeln!(f, "║  Learning Velocity: {:.2} (higher = faster learning)          ║", self.learning_velocity)?;

        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  PERFORMANCE SUMMARY                                         ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Total Trades: {:>6} | Win Rate: {:>5.1}%                     ║",
            self.performance.total_trades, self.performance.win_rate * 100.0)?;
        writeln!(f, "║  Wins: {:>6}        | Losses: {:>6}                        ║",
            self.performance.wins, self.performance.losses)?;
        writeln!(f, "║  Total PnL: ${:>10.2} | Profit Factor: {:>5.2}              ║",
            self.performance.total_pnl, self.performance.profit_factor)?;
        writeln!(f, "║  Max Drawdown: {:>5.1}% | Sharpe Ratio: {:>5.2}               ║",
            self.performance.max_drawdown * 100.0, self.performance.sharpe_ratio)?;

        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  AGI COMPONENT METRICS                                       ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;

        // Memory
        writeln!(f, "║  Memory: {} trades | {} S/R levels | {} regimes           ║",
            self.current_metrics.total_trades_stored,
            self.current_metrics.sr_levels_tracked,
            self.current_metrics.regimes_recorded)?;

        // Learning
        writeln!(f, "║  Calibrator: {} updates | {:.1}% accuracy | EWC: {}         ║",
            self.current_metrics.calibrator_updates,
            self.current_metrics.calibrator_accuracy * 100.0,
            if self.current_metrics.ewc_active { "active" } else { "off" })?;

        // MoE
        if self.current_metrics.moe_enabled {
            writeln!(f, "║  MoE: ENABLED | {} experts active                           ║",
                self.current_metrics.expert_stats.len())?;
        } else {
            writeln!(f, "║  MoE: DISABLED                                               ║")?;
        }

        // Meta-learning
        writeln!(f, "║  Meta: {} adaptations | {:.1}% success rate                   ║",
            self.current_metrics.meta_adaptations,
            self.current_metrics.meta_success_rate * 100.0)?;

        // Weakness
        writeln!(f, "║  Weakness: {} identified | {} critical                        ║",
            self.current_metrics.weaknesses_identified,
            self.current_metrics.critical_weaknesses)?;

        // Causal
        writeln!(f, "║  Causal: {} relationships | {} significant                    ║",
            self.current_metrics.causal_relationships,
            self.current_metrics.significant_relationships)?;

        // World Model
        writeln!(f, "║  WorldModel: {} simulations | {:.1}% forecast accuracy        ║",
            self.current_metrics.simulation_runs,
            self.current_metrics.forecast_accuracy * 100.0)?;

        // Counterfactual
        writeln!(f, "║  Counterfactual: {} analyzed | {} insights                    ║",
            self.current_metrics.counterfactuals_analyzed,
            self.current_metrics.insights_generated)?;

        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  SYSTEM HEALTH                                               ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;

        let uptime_hours = self.health.uptime_seconds / 3600;
        let uptime_mins = (self.health.uptime_seconds % 3600) / 60;
        writeln!(f, "║  Uptime: {}h {}m | Memory: {:.1} MB                           ║",
            uptime_hours, uptime_mins, self.health.memory_usage_mb)?;
        writeln!(f, "║  Errors: {} | Warnings: {} (last hour)                       ║",
            self.health.errors_last_hour, self.health.warnings_last_hour)?;

        // Component status
        let healthy_count = self.component_status.values().filter(|(h, _)| *h).count();
        let total_count = self.component_status.len();
        writeln!(f, "║  Components: {}/{} healthy                                    ║",
            healthy_count, total_count)?;

        // Top Insights
        if !self.top_insights.is_empty() {
            writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
            writeln!(f, "║  TOP INSIGHTS                                                ║")?;
            writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
            for (i, insight) in self.top_insights.iter().take(3).enumerate() {
                let truncated = if insight.len() > 55 {
                    format!("{}...", &insight[..52])
                } else {
                    insight.clone()
                };
                writeln!(f, "║  {}. {:55} ║", i + 1, truncated)?;
            }
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
            writeln!(f, "║  RECOMMENDATIONS                                             ║")?;
            writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
            for rec in self.recommendations.iter().take(3) {
                let truncated = if rec.len() > 55 {
                    format!("{}...", &rec[..52])
                } else {
                    rec.clone()
                };
                writeln!(f, "║  • {:56} ║", truncated)?;
            }
        }

        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;

        Ok(())
    }
}

impl AGIReport {
    /// Render a progress bar
    fn render_progress_bar(&self, progress: f64, width: usize) -> String {
        let filled = (progress * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }
}

/// Serializable state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MonitorState {
    metrics_history: Vec<AGIMetrics>,
    performance_history: Vec<PerformanceMetrics>,
    start_time: DateTime<Utc>,
    // Counters that accumulate
    total_trades_skipped_weakness: u32,
    total_position_adjustments: u32,
    total_causal_adjustments: u32,
    total_worldmodel_skips: u32,
    total_simulation_runs: u32,
}

/// AGI Monitor for comprehensive system monitoring
pub struct AGIMonitor {
    /// Hourly AGI metrics snapshots
    metrics_history: Vec<AGIMetrics>,
    /// Daily performance snapshots
    performance_history: Vec<PerformanceMetrics>,
    /// Current system health
    health: SystemHealth,
    /// When monitoring started
    start_time: DateTime<Utc>,

    // Counters that accumulate
    total_trades_skipped_weakness: u32,
    total_position_adjustments: u32,
    total_causal_adjustments: u32,
    total_worldmodel_skips: u32,
    total_simulation_runs: u32,

    // Component references
    memory: Option<Arc<TradeMemory>>,
    calibrator: Option<ConfidenceCalibrator>,
    moe: Option<MixtureOfExperts>,
    meta_learner: Option<Arc<Mutex<MetaLearner>>>,
    transfer_manager: Option<Arc<Mutex<TransferManager>>>,
    weakness_analyzer: Option<Arc<Mutex<WeaknessAnalyzer>>>,
    causal_analyzer: Option<Arc<Mutex<CausalAnalyzer>>>,
    world_model: Option<Arc<Mutex<WorldModel>>>,
    counterfactual: Option<Arc<Mutex<CounterfactualAnalyzer>>>,
}

impl AGIMonitor {
    /// Create a new AGI monitor
    pub fn new() -> Self {
        let start = Utc::now();
        let mut health = SystemHealth::default();
        health.components_healthy.insert("memory".to_string(), false);
        health.components_healthy.insert("calibrator".to_string(), false);
        health.components_healthy.insert("moe".to_string(), false);
        health.components_healthy.insert("meta_learner".to_string(), false);
        health.components_healthy.insert("transfer".to_string(), false);
        health.components_healthy.insert("weakness".to_string(), false);
        health.components_healthy.insert("causal".to_string(), false);
        health.components_healthy.insert("world_model".to_string(), false);
        health.components_healthy.insert("counterfactual".to_string(), false);

        Self {
            metrics_history: Vec::new(),
            performance_history: Vec::new(),
            health,
            start_time: start,
            total_trades_skipped_weakness: 0,
            total_position_adjustments: 0,
            total_causal_adjustments: 0,
            total_worldmodel_skips: 0,
            total_simulation_runs: 0,
            memory: None,
            calibrator: None,
            moe: None,
            meta_learner: None,
            transfer_manager: None,
            weakness_analyzer: None,
            causal_analyzer: None,
            world_model: None,
            counterfactual: None,
        }
    }

    /// Load from file or create new
    pub fn load_or_new(path: &str) -> Self {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(state) = serde_json::from_str::<MonitorState>(&contents) {
                info!("[MONITOR] Loaded {} hourly snapshots, {} daily snapshots",
                    state.metrics_history.len(), state.performance_history.len());

                let mut health = SystemHealth::default();
                health.components_healthy.insert("memory".to_string(), false);
                health.components_healthy.insert("calibrator".to_string(), false);
                health.components_healthy.insert("moe".to_string(), false);
                health.components_healthy.insert("meta_learner".to_string(), false);
                health.components_healthy.insert("transfer".to_string(), false);
                health.components_healthy.insert("weakness".to_string(), false);
                health.components_healthy.insert("causal".to_string(), false);
                health.components_healthy.insert("world_model".to_string(), false);
                health.components_healthy.insert("counterfactual".to_string(), false);

                return Self {
                    metrics_history: state.metrics_history,
                    performance_history: state.performance_history,
                    health,
                    start_time: state.start_time,
                    total_trades_skipped_weakness: state.total_trades_skipped_weakness,
                    total_position_adjustments: state.total_position_adjustments,
                    total_causal_adjustments: state.total_causal_adjustments,
                    total_worldmodel_skips: state.total_worldmodel_skips,
                    total_simulation_runs: state.total_simulation_runs,
                    memory: None,
                    calibrator: None,
                    moe: None,
                    meta_learner: None,
                    transfer_manager: None,
                    weakness_analyzer: None,
                    causal_analyzer: None,
                    world_model: None,
                    counterfactual: None,
                };
            }
        }
        Self::new()
    }

    /// Save to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let state = MonitorState {
            metrics_history: self.metrics_history.clone(),
            performance_history: self.performance_history.clone(),
            start_time: self.start_time,
            total_trades_skipped_weakness: self.total_trades_skipped_weakness,
            total_position_adjustments: self.total_position_adjustments,
            total_causal_adjustments: self.total_causal_adjustments,
            total_worldmodel_skips: self.total_worldmodel_skips,
            total_simulation_runs: self.total_simulation_runs,
        };
        let contents = serde_json::to_string_pretty(&state)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    // ==================== Attach Methods ====================

    /// Attach memory component
    pub fn attach_memory(&mut self, memory: Arc<TradeMemory>) {
        self.memory = Some(memory);
        self.health.components_healthy.insert("memory".to_string(), true);
    }

    /// Attach calibrator (cloned for reading)
    pub fn attach_calibrator(&mut self, calibrator: ConfidenceCalibrator) {
        self.calibrator = Some(calibrator);
        self.health.components_healthy.insert("calibrator".to_string(), true);
    }

    /// Attach MoE (cloned for reading)
    pub fn attach_moe(&mut self, moe: MixtureOfExperts) {
        self.moe = Some(moe);
        self.health.components_healthy.insert("moe".to_string(), true);
    }

    /// Attach meta-learner
    pub fn attach_meta_learner(&mut self, ml: Arc<Mutex<MetaLearner>>) {
        self.meta_learner = Some(ml);
        self.health.components_healthy.insert("meta_learner".to_string(), true);
    }

    /// Attach transfer manager
    pub fn attach_transfer_manager(&mut self, tm: Arc<Mutex<TransferManager>>) {
        self.transfer_manager = Some(tm);
        self.health.components_healthy.insert("transfer".to_string(), true);
    }

    /// Attach weakness analyzer
    pub fn attach_weakness_analyzer(&mut self, wa: Arc<Mutex<WeaknessAnalyzer>>) {
        self.weakness_analyzer = Some(wa);
        self.health.components_healthy.insert("weakness".to_string(), true);
    }

    /// Attach causal analyzer
    pub fn attach_causal_analyzer(&mut self, ca: Arc<Mutex<CausalAnalyzer>>) {
        self.causal_analyzer = Some(ca);
        self.health.components_healthy.insert("causal".to_string(), true);
    }

    /// Attach world model
    pub fn attach_world_model(&mut self, wm: Arc<Mutex<WorldModel>>) {
        self.world_model = Some(wm);
        self.health.components_healthy.insert("world_model".to_string(), true);
    }

    /// Attach counterfactual analyzer
    pub fn attach_counterfactual(&mut self, cf: Arc<Mutex<CounterfactualAnalyzer>>) {
        self.counterfactual = Some(cf);
        self.health.components_healthy.insert("counterfactual".to_string(), true);
    }

    // ==================== Increment Counters ====================

    /// Record a trade skipped due to weakness
    pub fn record_weakness_skip(&mut self) {
        self.total_trades_skipped_weakness += 1;
    }

    /// Record a position size adjustment
    pub fn record_position_adjustment(&mut self) {
        self.total_position_adjustments += 1;
    }

    /// Record a causal confidence adjustment
    pub fn record_causal_adjustment(&mut self) {
        self.total_causal_adjustments += 1;
    }

    /// Record a world model skip
    pub fn record_worldmodel_skip(&mut self) {
        self.total_worldmodel_skips += 1;
    }

    /// Record a simulation run
    pub fn record_simulation(&mut self) {
        self.total_simulation_runs += 1;
    }

    // ==================== Metrics Collection ====================

    /// Collect current AGI metrics from all components
    pub fn collect_agi_metrics(&self) -> AGIMetrics {
        let mut metrics = AGIMetrics::default();
        metrics.timestamp = Utc::now();

        // Memory metrics
        if let Some(ref memory) = self.memory {
            if let Ok((total, wins, _, _)) = memory.get_overall_stats() {
                metrics.total_trades_stored = total as u32;
            }
            // SR levels would need to be tracked separately
            metrics.sr_levels_tracked = 0;
            metrics.regimes_recorded = 0;
        }

        // Calibrator metrics
        if let Some(ref calibrator) = self.calibrator {
            metrics.calibrator_updates = calibrator.update_count() as u32;
            // Accuracy would need recent predictions tracked
            metrics.calibrator_accuracy = 0.5; // Default
            metrics.ewc_consolidations = calibrator.consolidation_count();
            metrics.ewc_active = calibrator.is_ewc_active();
        }

        // MoE metrics
        if let Some(ref moe) = self.moe {
            metrics.moe_enabled = true;
            metrics.expert_stats = vec![
                ExpertStats { regime: Regime::TrendingUp, trades: moe.expert_trades(Regime::TrendingUp), win_rate: moe.expert_win_rate(Regime::TrendingUp) },
                ExpertStats { regime: Regime::TrendingDown, trades: moe.expert_trades(Regime::TrendingDown), win_rate: moe.expert_win_rate(Regime::TrendingDown) },
                ExpertStats { regime: Regime::Ranging, trades: moe.expert_trades(Regime::Ranging), win_rate: moe.expert_win_rate(Regime::Ranging) },
                ExpertStats { regime: Regime::Volatile, trades: moe.expert_trades(Regime::Volatile), win_rate: moe.expert_win_rate(Regime::Volatile) },
            ];
            metrics.gating_weights = moe.gating_weights();
        }

        // Meta-learner metrics
        if let Some(ref ml) = self.meta_learner {
            let ml_lock = ml.lock().unwrap();
            metrics.meta_adaptations = ml_lock.meta_update_count();
            metrics.avg_adaptation_speed = ml_lock.avg_adaptation_speed();
            metrics.meta_success_rate = ml_lock.success_rate();
        }

        // Transfer metrics
        if let Some(ref tm) = self.transfer_manager {
            let tm_lock = tm.lock().unwrap();
            metrics.cluster_trade_counts = tm_lock.cluster_trade_counts();
            metrics.transfers_applied = tm_lock.transfers_applied();
            metrics.transfer_success_rate = tm_lock.success_rate();
        }

        // Weakness metrics
        if let Some(ref wa) = self.weakness_analyzer {
            let wa_lock = wa.lock().unwrap();
            metrics.weaknesses_identified = wa_lock.weakness_count() as u32;
            metrics.critical_weaknesses = wa_lock.critical_weakness_count() as u32;
            metrics.trades_skipped_for_weakness = self.total_trades_skipped_weakness;
            metrics.position_size_adjustments = self.total_position_adjustments;
        }

        // Causal metrics
        if let Some(ref ca) = self.causal_analyzer {
            let ca_lock = ca.lock().unwrap();
            metrics.causal_relationships = ca_lock.relationship_count() as u32;
            metrics.significant_relationships = ca_lock.significant_relationship_count() as u32;
            metrics.causal_confidence_adjustments = self.total_causal_adjustments;
        }

        // World model metrics
        if let Some(ref wm) = self.world_model {
            let wm_lock = wm.lock().unwrap();
            metrics.simulation_runs = self.total_simulation_runs;
            metrics.forecast_accuracy = wm_lock.forecast_accuracy();
            metrics.trades_skipped_by_worldmodel = self.total_worldmodel_skips;
        }

        // Counterfactual metrics
        if let Some(ref cf) = self.counterfactual {
            let cf_lock = cf.lock().unwrap();
            metrics.counterfactuals_analyzed = cf_lock.total_analyses() as u32;
            metrics.insights_generated = cf_lock.insight_count() as u32;
            metrics.systematic_errors_found = cf_lock.systematic_error_count() as u32;
            metrics.estimated_improvement_if_followed = cf_lock.estimated_improvement();
        }

        metrics
    }

    /// Collect performance metrics from memory
    pub fn collect_performance_metrics(&self) -> PerformanceMetrics {
        let mut perf = PerformanceMetrics::default();

        if let Some(ref memory) = self.memory {
            if let Ok((total, wins, total_profit, avg_profit)) = memory.get_overall_stats() {
                perf.total_trades = total as u32;
                perf.wins = wins as u32;
                perf.losses = (total - wins) as u32;
                perf.win_rate = if total > 0 { wins as f64 / total as f64 } else { 0.0 };
                perf.total_pnl = total_profit;

                // Profit factor estimation
                let avg_win = if wins > 0 { total_profit / wins as f64 } else { 0.0 };
                let avg_loss = if perf.losses > 0 {
                    (total_profit - avg_win * wins as f64).abs() / perf.losses as f64
                } else {
                    1.0
                };
                perf.profit_factor = if avg_loss > 0.0 { avg_win / avg_loss } else { 0.0 };
            }

            // Get regime performance using MarketRegime
            use crate::data::memory::MarketRegime;
            let market_regimes = [
                (MarketRegime::Bull, Regime::TrendingUp),
                (MarketRegime::Bear, Regime::TrendingDown),
                (MarketRegime::Sideways, Regime::Ranging),
                (MarketRegime::HighVolatility, Regime::Volatile),
            ];
            let mut best: Option<(Regime, f64)> = None;
            let mut worst: Option<(Regime, f64)> = None;

            for (market_regime, regime) in &market_regimes {
                if let Ok(Some(stats)) = memory.get_regime_stats(*market_regime) {
                    if stats.total_trades > 0 {
                        let wr = stats.wins as f64 / stats.total_trades as f64;
                        if best.is_none() || wr > best.unwrap().1 {
                            best = Some((*regime, wr));
                        }
                        if worst.is_none() || wr < worst.unwrap().1 {
                            worst = Some((*regime, wr));
                        }
                    }
                }
            }
            perf.best_regime = best;
            perf.worst_regime = worst;
        }

        perf
    }

    /// Update system health
    pub fn update_health(&mut self) {
        let now = Utc::now();
        self.health.uptime_seconds = (now - self.start_time).num_seconds() as u64;

        // Memory usage (simplified - would need system call)
        self.health.memory_usage_mb = 0.0;

        // Check each component
        self.health.components_healthy.insert("memory".to_string(), self.memory.is_some());
        self.health.components_healthy.insert("calibrator".to_string(), self.calibrator.is_some());
        self.health.components_healthy.insert("moe".to_string(), self.moe.is_some());
        self.health.components_healthy.insert("meta_learner".to_string(), self.meta_learner.is_some());
        self.health.components_healthy.insert("transfer".to_string(), self.transfer_manager.is_some());
        self.health.components_healthy.insert("weakness".to_string(), self.weakness_analyzer.is_some());
        self.health.components_healthy.insert("causal".to_string(), self.causal_analyzer.is_some());
        self.health.components_healthy.insert("world_model".to_string(), self.world_model.is_some());
        self.health.components_healthy.insert("counterfactual".to_string(), self.counterfactual.is_some());
    }

    /// Take hourly snapshot
    pub fn snapshot_hourly(&mut self) {
        let metrics = self.collect_agi_metrics();
        self.metrics_history.push(metrics);

        // Keep last 168 hours (1 week)
        if self.metrics_history.len() > 168 {
            self.metrics_history.remove(0);
        }

        info!("[MONITOR] Hourly snapshot #{}", self.metrics_history.len());
    }

    /// Take daily snapshot
    pub fn snapshot_daily(&mut self) {
        let perf = self.collect_performance_metrics();
        self.performance_history.push(perf);

        // Keep last 90 days
        if self.performance_history.len() > 90 {
            self.performance_history.remove(0);
        }

        info!("[MONITOR] Daily snapshot #{}", self.performance_history.len());
    }

    /// Calculate overall AGI progress (0.0 to 1.0)
    pub fn get_agi_progress(&self) -> f64 {
        let metrics = self.collect_agi_metrics();

        // Weight factors for different capabilities
        let mut progress = 0.0;
        let mut total_weight = 0.0;

        // Memory utilization (20%)
        let memory_score = (metrics.total_trades_stored as f64 / 1000.0).min(1.0);
        progress += memory_score * 0.20;
        total_weight += 0.20;

        // Learning activity (20%)
        let learning_score = (metrics.calibrator_updates as f64 / 500.0).min(1.0);
        progress += learning_score * 0.20;
        total_weight += 0.20;

        // Self-improvement (20%)
        let improvement_score = if metrics.weaknesses_identified > 0 {
            (metrics.trades_skipped_for_weakness as f64 / metrics.weaknesses_identified as f64).min(1.0)
        } else {
            0.0
        };
        progress += improvement_score * 0.20;
        total_weight += 0.20;

        // Reasoning depth - causal (15%)
        let causal_score = (metrics.causal_relationships as f64 / 20.0).min(1.0);
        progress += causal_score * 0.15;
        total_weight += 0.15;

        // Reasoning depth - counterfactual (15%)
        let cf_score = (metrics.insights_generated as f64 / 10.0).min(1.0);
        progress += cf_score * 0.15;
        total_weight += 0.15;

        // Meta-learning success (10%)
        progress += metrics.meta_success_rate * 0.10;
        total_weight += 0.10;

        if total_weight > 0.0 {
            progress / total_weight
        } else {
            0.0
        }
    }

    /// Calculate learning velocity (rate of improvement)
    pub fn get_learning_velocity(&self) -> f64 {
        if self.performance_history.len() < 2 {
            return 0.0;
        }

        // Compare recent win rate to older win rate
        let recent_count = 7.min(self.performance_history.len());
        let older_count = 14.min(self.performance_history.len());

        let recent: Vec<_> = self.performance_history.iter().rev().take(recent_count).collect();
        let older: Vec<_> = self.performance_history.iter().rev().skip(recent_count).take(older_count).collect();

        if recent.is_empty() || older.is_empty() {
            return 0.0;
        }

        let recent_avg: f64 = recent.iter().map(|p| p.win_rate).sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().map(|p| p.win_rate).sum::<f64>() / older.len() as f64;

        // Velocity is the improvement rate (can be negative)
        recent_avg - older_avg
    }

    /// Get component status
    pub fn get_component_status(&self) -> HashMap<String, (bool, String)> {
        let mut status = HashMap::new();

        status.insert("memory".to_string(), (
            self.memory.is_some(),
            if self.memory.is_some() { "Connected".to_string() } else { "Not attached".to_string() }
        ));

        status.insert("calibrator".to_string(), (
            self.calibrator.is_some(),
            if let Some(ref c) = self.calibrator {
                format!("{} updates", c.update_count())
            } else {
                "Not attached".to_string()
            }
        ));

        status.insert("moe".to_string(), (
            self.moe.is_some(),
            if let Some(ref m) = self.moe {
                format!("{} trades", m.total_trades())
            } else {
                "Not attached".to_string()
            }
        ));

        status.insert("meta_learner".to_string(), (
            self.meta_learner.is_some(),
            if let Some(ref ml) = self.meta_learner {
                let lock = ml.lock().unwrap();
                format!("{} adaptations", lock.meta_update_count())
            } else {
                "Not attached".to_string()
            }
        ));

        status.insert("transfer".to_string(), (
            self.transfer_manager.is_some(),
            if let Some(ref tm) = self.transfer_manager {
                let lock = tm.lock().unwrap();
                format!("{} transfers", lock.transfers_applied())
            } else {
                "Not attached".to_string()
            }
        ));

        status.insert("weakness".to_string(), (
            self.weakness_analyzer.is_some(),
            if let Some(ref wa) = self.weakness_analyzer {
                let lock = wa.lock().unwrap();
                format!("{} weaknesses", lock.weakness_count())
            } else {
                "Not attached".to_string()
            }
        ));

        status.insert("causal".to_string(), (
            self.causal_analyzer.is_some(),
            if let Some(ref ca) = self.causal_analyzer {
                let lock = ca.lock().unwrap();
                format!("{} relationships", lock.relationship_count())
            } else {
                "Not attached".to_string()
            }
        ));

        status.insert("world_model".to_string(), (
            self.world_model.is_some(),
            if let Some(ref wm) = self.world_model {
                let lock = wm.lock().unwrap();
                format!("{} symbols", lock.get_symbols().len())
            } else {
                "Not attached".to_string()
            }
        ));

        status.insert("counterfactual".to_string(), (
            self.counterfactual.is_some(),
            if let Some(ref cf) = self.counterfactual {
                let lock = cf.lock().unwrap();
                format!("{} insights", lock.insight_count())
            } else {
                "Not attached".to_string()
            }
        ));

        status
    }

    /// Generate comprehensive AGI report
    pub fn generate_report(&self) -> AGIReport {
        self.update_health_immut();

        let metrics = self.collect_agi_metrics();
        let performance = self.collect_performance_metrics();
        let component_status = self.get_component_status();

        // Generate insights
        let mut insights = Vec::new();

        if metrics.calibrator_accuracy > 0.6 {
            insights.push(format!("Calibrator accuracy is strong at {:.1}%", metrics.calibrator_accuracy * 100.0));
        }

        if metrics.meta_success_rate > 0.7 {
            insights.push(format!("Meta-learning is highly successful ({:.1}%)", metrics.meta_success_rate * 100.0));
        }

        if metrics.weaknesses_identified > 0 && metrics.trades_skipped_for_weakness > 0 {
            insights.push(format!("Actively avoiding {} identified weakness patterns", metrics.weaknesses_identified));
        }

        if metrics.causal_relationships > 5 {
            insights.push(format!("Discovered {} market relationships for predictive power", metrics.causal_relationships));
        }

        if metrics.insights_generated > 0 {
            insights.push(format!("Counterfactual analysis identified {} improvement opportunities", metrics.insights_generated));
        }

        // Generate recommendations
        let mut recommendations = Vec::new();

        if metrics.total_trades_stored < 100 {
            recommendations.push("Need more trades for reliable pattern detection".to_string());
        }

        if metrics.calibrator_accuracy < 0.5 {
            recommendations.push("Calibrator accuracy is low - review feature weights".to_string());
        }

        if metrics.weaknesses_identified > 5 && metrics.trades_skipped_for_weakness == 0 {
            recommendations.push("Weaknesses identified but not being avoided - check integration".to_string());
        }

        if metrics.causal_relationships == 0 {
            recommendations.push("No causal relationships discovered - need more market data".to_string());
        }

        if performance.win_rate < 0.45 {
            recommendations.push("Win rate below 45% - review entry criteria".to_string());
        }

        AGIReport {
            generated_at: Utc::now(),
            agi_progress_pct: self.get_agi_progress(),
            learning_velocity: self.get_learning_velocity(),
            current_metrics: metrics,
            performance,
            health: self.health.clone(),
            top_insights: insights,
            recommendations,
            component_status,
        }
    }

    /// Internal helper for immutable health update
    fn update_health_immut(&self) {
        // Health is updated during the mutable calls; this is a no-op for immutable context
    }

    /// Get current health status
    pub fn health(&self) -> &SystemHealth {
        &self.health
    }

    /// Get metrics history
    pub fn metrics_history(&self) -> &[AGIMetrics] {
        &self.metrics_history
    }

    /// Get performance history
    pub fn performance_history(&self) -> &[PerformanceMetrics] {
        &self.performance_history
    }

    /// Format a summary for logging
    pub fn format_summary(&self) -> String {
        let agi_pct = self.get_agi_progress() * 100.0;
        let velocity = self.get_learning_velocity();
        let healthy = self.health.components_healthy.values().filter(|&&v| v).count();
        let total = self.health.components_healthy.len();
        format!("AGI: {:.1}% | Velocity: {:.3} | Components: {}/{}", agi_pct, velocity, healthy, total)
    }
}

impl Default for AGIMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agi_metrics_default() {
        let metrics = AGIMetrics::default();
        assert_eq!(metrics.total_trades_stored, 0);
        assert_eq!(metrics.calibrator_updates, 0);
        assert!(!metrics.moe_enabled);
    }

    #[test]
    fn test_performance_metrics_default() {
        let perf = PerformanceMetrics::default();
        assert_eq!(perf.total_trades, 0);
        assert_eq!(perf.wins, 0);
        assert_eq!(perf.win_rate, 0.0);
    }

    #[test]
    fn test_system_health_default() {
        let health = SystemHealth::default();
        assert_eq!(health.uptime_seconds, 0);
        assert!(health.last_trade_at.is_none());
    }

    #[test]
    fn test_agi_monitor_new() {
        let monitor = AGIMonitor::new();
        assert!(monitor.metrics_history.is_empty());
        assert!(monitor.performance_history.is_empty());
        assert!(monitor.memory.is_none());
    }

    #[test]
    fn test_agi_progress_empty() {
        let monitor = AGIMonitor::new();
        let progress = monitor.get_agi_progress();
        assert!(progress >= 0.0 && progress <= 1.0);
    }

    #[test]
    fn test_learning_velocity_empty() {
        let monitor = AGIMonitor::new();
        let velocity = monitor.get_learning_velocity();
        assert_eq!(velocity, 0.0);
    }

    #[test]
    fn test_component_status() {
        let monitor = AGIMonitor::new();
        let status = monitor.get_component_status();
        assert!(status.contains_key("memory"));
        assert!(status.contains_key("calibrator"));
        assert!(!status["memory"].0); // Not attached
    }

    #[test]
    fn test_snapshot_hourly() {
        let mut monitor = AGIMonitor::new();
        monitor.snapshot_hourly();
        assert_eq!(monitor.metrics_history.len(), 1);

        // Take many snapshots
        for _ in 0..200 {
            monitor.snapshot_hourly();
        }
        // Should be limited to 168
        assert_eq!(monitor.metrics_history.len(), 168);
    }

    #[test]
    fn test_snapshot_daily() {
        let mut monitor = AGIMonitor::new();
        monitor.snapshot_daily();
        assert_eq!(monitor.performance_history.len(), 1);

        // Take many snapshots
        for _ in 0..100 {
            monitor.snapshot_daily();
        }
        // Should be limited to 90
        assert_eq!(monitor.performance_history.len(), 90);
    }

    #[test]
    fn test_report_generation() {
        let monitor = AGIMonitor::new();
        let report = monitor.generate_report();
        assert!(report.agi_progress_pct >= 0.0 && report.agi_progress_pct <= 1.0);
    }

    #[test]
    fn test_report_display() {
        let monitor = AGIMonitor::new();
        let report = monitor.generate_report();
        let display = format!("{}", report);
        assert!(display.contains("AGI Progress"));
        assert!(display.contains("PERFORMANCE SUMMARY"));
    }

    #[test]
    fn test_format_summary() {
        let monitor = AGIMonitor::new();
        let summary = monitor.format_summary();
        assert!(summary.contains("AGI:"));
        assert!(summary.contains("Velocity:"));
        assert!(summary.contains("Components:"));
    }

    #[test]
    fn test_counter_increments() {
        let mut monitor = AGIMonitor::new();

        monitor.record_weakness_skip();
        assert_eq!(monitor.total_trades_skipped_weakness, 1);

        monitor.record_position_adjustment();
        assert_eq!(monitor.total_position_adjustments, 1);

        monitor.record_causal_adjustment();
        assert_eq!(monitor.total_causal_adjustments, 1);

        monitor.record_worldmodel_skip();
        assert_eq!(monitor.total_worldmodel_skips, 1);

        monitor.record_simulation();
        assert_eq!(monitor.total_simulation_runs, 1);
    }

    #[test]
    fn test_expert_stats() {
        let stats = ExpertStats {
            regime: Regime::TrendingUp,
            trades: 50,
            win_rate: 0.65,
        };
        assert_eq!(stats.trades, 50);
        assert_eq!(stats.win_rate, 0.65);
    }
}
