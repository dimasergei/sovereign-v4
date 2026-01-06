//! Counterfactual Reasoning for Trade Analysis
//!
//! Analyzes alternative decisions to learn from "what ifs":
//! - What if I hadn't entered this trade?
//! - What if I took the opposite direction?
//! - What if I used different position sizing?
//! - What if I exited earlier or later?
//!
//! Key insight: Learning from mistakes requires understanding what
//! the alternatives would have produced.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::info;

use super::worldmodel::WorldModel;
use crate::data::memory::{TradeMemory, TradeContext as MemoryTradeContext};

/// Minimum trades to establish an insight pattern
const MIN_EVIDENCE_FOR_INSIGHT: u32 = 5;

/// Default confidence threshold for recommendations
const CONFIDENCE_THRESHOLD: f64 = 0.7;

/// Maximum bars to look back/forward for exit analysis
const MAX_EXIT_DELTA_BARS: i32 = 20;

/// Direction of a trade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Long => write!(f, "LONG"),
            Direction::Short => write!(f, "SHORT"),
        }
    }
}

/// Type of counterfactual query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    /// What if I hadn't entered?
    WhatIfNoTrade,
    /// What if I went long instead of short (or vice versa)?
    WhatIfOppositeDirection,
    /// What if I used different position size?
    WhatIfDifferentSize { multiplier: f64 },
    /// What if I exited sooner/later?
    WhatIfDifferentExit { bars_delta: i32 },
    /// What if I entered sooner/later?
    WhatIfDifferentEntry { bars_delta: i32 },
}

impl std::fmt::Display for QueryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryType::WhatIfNoTrade => write!(f, "NoTrade"),
            QueryType::WhatIfOppositeDirection => write!(f, "Opposite"),
            QueryType::WhatIfDifferentSize { multiplier } => write!(f, "Size({:.1}x)", multiplier),
            QueryType::WhatIfDifferentExit { bars_delta } => {
                if *bars_delta > 0 {
                    write!(f, "Exit(+{} bars)", bars_delta)
                } else {
                    write!(f, "Exit({} bars)", bars_delta)
                }
            }
            QueryType::WhatIfDifferentEntry { bars_delta } => {
                if *bars_delta > 0 {
                    write!(f, "Entry(+{} bars)", bars_delta)
                } else {
                    write!(f, "Entry({} bars)", bars_delta)
                }
            }
        }
    }
}

/// A counterfactual query about a specific trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualQuery {
    /// Trade ticket being analyzed
    pub trade_id: u64,
    /// Type of counterfactual question
    pub query_type: QueryType,
    /// When this query was made
    pub timestamp: DateTime<Utc>,
}

impl CounterfactualQuery {
    /// Create a new counterfactual query
    pub fn new(trade_id: u64, query_type: QueryType) -> Self {
        Self {
            trade_id,
            query_type,
            timestamp: Utc::now(),
        }
    }
}

/// Result of a counterfactual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualResult {
    /// The query that was asked
    pub query: CounterfactualQuery,
    /// What actually happened
    pub actual_pnl: f64,
    /// What would have happened
    pub counterfactual_pnl: f64,
    /// Difference: counterfactual - actual (positive = we made wrong choice)
    pub difference: f64,
    /// Human-readable insight
    pub insight: String,
    /// Confidence in this analysis (0-1)
    pub confidence: f64,
}

impl CounterfactualResult {
    /// Create a new counterfactual result
    pub fn new(
        query: CounterfactualQuery,
        actual_pnl: f64,
        counterfactual_pnl: f64,
        insight: String,
        confidence: f64,
    ) -> Self {
        Self {
            query,
            actual_pnl,
            counterfactual_pnl,
            difference: counterfactual_pnl - actual_pnl,
            insight,
            confidence,
        }
    }

    /// Check if the actual decision was better
    pub fn was_good_decision(&self) -> bool {
        self.difference <= 0.0
    }

    /// Get the magnitude of the difference
    pub fn magnitude(&self) -> f64 {
        self.difference.abs()
    }
}

/// Trade data prepared for counterfactual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeForAnalysis {
    /// Unique ticket ID
    pub ticket: u64,
    /// Trading symbol
    pub symbol: String,
    /// Trade direction
    pub direction: Direction,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: DateTime<Utc>,
    /// Position size
    pub size: f64,
    /// Actual PnL
    pub actual_pnl: f64,
    /// Number of bars held
    pub bars_held: u32,
    /// Price history during and after the trade (for exit analysis)
    pub price_history: Vec<f64>,
    /// Regime at entry
    pub entry_regime: Option<String>,
    /// S/R score at entry
    pub sr_score: Option<i32>,
    /// Volume percentile at entry
    pub volume_percentile: Option<f64>,
}

impl TradeForAnalysis {
    /// Calculate PnL for a given exit price
    pub fn pnl_at_price(&self, price: f64) -> f64 {
        match self.direction {
            Direction::Long => (price - self.entry_price) * self.size,
            Direction::Short => (self.entry_price - price) * self.size,
        }
    }

    /// Get price at a given bar offset (0 = entry, bars_held = exit)
    pub fn price_at_bar(&self, bar_offset: usize) -> Option<f64> {
        self.price_history.get(bar_offset).copied()
    }

    /// Get the best price during the trade (MFE price)
    pub fn best_price(&self) -> Option<f64> {
        if self.price_history.is_empty() {
            return None;
        }
        match self.direction {
            Direction::Long => self.price_history.iter().copied().reduce(f64::max),
            Direction::Short => self.price_history.iter().copied().reduce(f64::min),
        }
    }

    /// Get the worst price during the trade (MAE price)
    pub fn worst_price(&self) -> Option<f64> {
        if self.price_history.is_empty() {
            return None;
        }
        match self.direction {
            Direction::Long => self.price_history.iter().copied().reduce(f64::min),
            Direction::Short => self.price_history.iter().copied().reduce(f64::max),
        }
    }
}

/// Type of trading insight discovered
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    /// Consistently leaving money on the table
    ExitTooEarly,
    /// Consistently giving back profits
    ExitTooLate,
    /// Winners could be bigger
    SizeTooSmall,
    /// Losers hurt too much
    SizeTooLarge,
    /// Opposite trade would have won
    WrongDirection,
    /// No trade was better
    ShouldHaveSkipped,
    /// Actual decision was optimal
    GoodDecision,
}

impl std::fmt::Display for InsightType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InsightType::ExitTooEarly => write!(f, "ExitTooEarly"),
            InsightType::ExitTooLate => write!(f, "ExitTooLate"),
            InsightType::SizeTooSmall => write!(f, "SizeTooSmall"),
            InsightType::SizeTooLarge => write!(f, "SizeTooLarge"),
            InsightType::WrongDirection => write!(f, "WrongDirection"),
            InsightType::ShouldHaveSkipped => write!(f, "ShouldHaveSkipped"),
            InsightType::GoodDecision => write!(f, "GoodDecision"),
        }
    }
}

/// An aggregated trading insight from counterfactual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingInsight {
    /// Type of insight
    pub insight_type: InsightType,
    /// Human-readable description
    pub description: String,
    /// Number of trades supporting this insight
    pub evidence_count: u32,
    /// Average PnL improvement if this insight were followed
    pub avg_improvement: f64,
    /// When this insight was discovered
    pub discovered_at: DateTime<Utc>,
    /// Symbol (if specific to one symbol)
    pub symbol: Option<String>,
    /// Regime (if specific to one regime)
    pub regime: Option<String>,
}

impl TradingInsight {
    /// Create a new trading insight
    pub fn new(
        insight_type: InsightType,
        description: String,
        evidence_count: u32,
        avg_improvement: f64,
    ) -> Self {
        Self {
            insight_type,
            description,
            evidence_count,
            avg_improvement,
            discovered_at: Utc::now(),
            symbol: None,
            regime: None,
        }
    }

    /// Check if insight has sufficient evidence
    pub fn is_significant(&self) -> bool {
        self.evidence_count >= MIN_EVIDENCE_FOR_INSIGHT
    }

    /// Set symbol context
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set regime context
    pub fn with_regime(mut self, regime: &str) -> Self {
        self.regime = Some(regime.to_string());
        self
    }
}

/// Trade context for should_have_traded analysis
#[derive(Debug, Clone)]
pub struct TradeContext {
    /// Symbol
    pub symbol: String,
    /// Direction being considered
    pub direction: Direction,
    /// S/R score
    pub sr_score: i32,
    /// Volume percentile
    pub volume_percentile: f64,
    /// Current regime
    pub regime: String,
    /// Current price
    pub price: f64,
}

/// Serializable state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CounterfactualState {
    analyses: Vec<CounterfactualResult>,
    insights: Vec<TradingInsight>,
    last_analysis: Option<DateTime<Utc>>,
    total_trades_analyzed: u32,
}

/// Analyzer for counterfactual reasoning about trades
pub struct CounterfactualAnalyzer {
    /// Trade memory for querying history
    memory: Arc<TradeMemory>,
    /// World model for simulations (optional)
    world_model: Option<Arc<Mutex<WorldModel>>>,
    /// Stored counterfactual analyses
    analyses: Vec<CounterfactualResult>,
    /// Aggregated insights
    insights: Vec<TradingInsight>,
    /// When was last analysis run
    last_analysis: Option<DateTime<Utc>>,
    /// Total trades analyzed
    total_trades_analyzed: u32,
}

impl CounterfactualAnalyzer {
    /// Create a new counterfactual analyzer
    pub fn new(memory: Arc<TradeMemory>) -> Self {
        Self {
            memory,
            world_model: None,
            analyses: Vec::new(),
            insights: Vec::new(),
            last_analysis: None,
            total_trades_analyzed: 0,
        }
    }

    /// Attach world model for enhanced simulation
    pub fn attach_world_model(&mut self, wm: Arc<Mutex<WorldModel>>) {
        self.world_model = Some(wm);
    }

    /// Check if world model is attached
    pub fn has_world_model(&self) -> bool {
        self.world_model.is_some()
    }

    /// Analyze a single trade with all counterfactual queries
    pub fn analyze_trade(&self, trade: &TradeForAnalysis) -> Vec<CounterfactualResult> {
        let mut results = Vec::new();

        // What if no trade?
        results.push(self.what_if_no_trade(trade));

        // What if opposite direction?
        results.push(self.what_if_opposite(trade));

        // What if different sizes?
        results.push(self.what_if_different_size(trade, 0.5));
        results.push(self.what_if_different_size(trade, 2.0));

        // What if different exit times?
        for bars_delta in [-5, -3, -1, 1, 3, 5] {
            if let Some(result) = self.what_if_different_exit(trade, bars_delta) {
                results.push(result);
            }
        }

        // Sort by magnitude of difference
        results.sort_by(|a, b| {
            b.magnitude()
                .partial_cmp(&a.magnitude())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// What if I hadn't entered this trade?
    pub fn what_if_no_trade(&self, trade: &TradeForAnalysis) -> CounterfactualResult {
        let query = CounterfactualQuery::new(trade.ticket, QueryType::WhatIfNoTrade);
        let counterfactual_pnl = 0.0;

        let insight = if trade.actual_pnl > 0.0 {
            format!(
                "Good entry, captured ${:.2} profit",
                trade.actual_pnl
            )
        } else {
            format!(
                "Skipping would have saved ${:.2}",
                trade.actual_pnl.abs()
            )
        };

        CounterfactualResult::new(query, trade.actual_pnl, counterfactual_pnl, insight, 1.0)
    }

    /// What if I took the opposite direction?
    pub fn what_if_opposite(&self, trade: &TradeForAnalysis) -> CounterfactualResult {
        let query = CounterfactualQuery::new(trade.ticket, QueryType::WhatIfOppositeDirection);

        // Opposite direction PnL
        let counterfactual_pnl = match trade.direction {
            Direction::Long => (trade.entry_price - trade.exit_price) * trade.size,
            Direction::Short => (trade.exit_price - trade.entry_price) * trade.size,
        };

        let insight = if counterfactual_pnl > trade.actual_pnl {
            format!(
                "Opposite direction would have made ${:.2} more",
                counterfactual_pnl - trade.actual_pnl
            )
        } else {
            format!(
                "Correct direction choice, saved ${:.2}",
                trade.actual_pnl - counterfactual_pnl
            )
        };

        CounterfactualResult::new(
            query,
            trade.actual_pnl,
            counterfactual_pnl,
            insight,
            1.0,
        )
    }

    /// What if I used different position size?
    pub fn what_if_different_size(&self, trade: &TradeForAnalysis, multiplier: f64) -> CounterfactualResult {
        let query = CounterfactualQuery::new(
            trade.ticket,
            QueryType::WhatIfDifferentSize { multiplier },
        );

        let counterfactual_pnl = trade.actual_pnl * multiplier;

        let insight = if trade.actual_pnl > 0.0 {
            // Winner: larger size = more profit
            if multiplier > 1.0 {
                format!(
                    "{:.1}x size would have made ${:.2} more",
                    multiplier,
                    counterfactual_pnl - trade.actual_pnl
                )
            } else {
                format!(
                    "{:.1}x size would have made ${:.2} less",
                    multiplier,
                    trade.actual_pnl - counterfactual_pnl
                )
            }
        } else {
            // Loser: smaller size = less loss
            if multiplier < 1.0 {
                format!(
                    "{:.1}x size would have saved ${:.2}",
                    multiplier,
                    trade.actual_pnl.abs() - counterfactual_pnl.abs()
                )
            } else {
                format!(
                    "{:.1}x size would have lost ${:.2} more",
                    multiplier,
                    counterfactual_pnl.abs() - trade.actual_pnl.abs()
                )
            }
        };

        // Lower confidence for size analysis (it's always proportional)
        CounterfactualResult::new(
            query,
            trade.actual_pnl,
            counterfactual_pnl,
            insight,
            0.6,
        )
    }

    /// What if I exited earlier or later?
    pub fn what_if_different_exit(
        &self,
        trade: &TradeForAnalysis,
        bars_delta: i32,
    ) -> Option<CounterfactualResult> {
        let target_bar = trade.bars_held as i32 + bars_delta;

        if target_bar <= 0 || target_bar > MAX_EXIT_DELTA_BARS + trade.bars_held as i32 {
            return None;
        }

        let target_bar_usize = target_bar as usize;
        let price_at_alt_exit = trade.price_at_bar(target_bar_usize)?;

        let query = CounterfactualQuery::new(
            trade.ticket,
            QueryType::WhatIfDifferentExit { bars_delta },
        );

        let counterfactual_pnl = trade.pnl_at_price(price_at_alt_exit);

        let insight = if bars_delta < 0 {
            if counterfactual_pnl > trade.actual_pnl {
                format!(
                    "Exiting {} bars earlier at ${:.2} would have made ${:.2} more",
                    -bars_delta,
                    price_at_alt_exit,
                    counterfactual_pnl - trade.actual_pnl
                )
            } else {
                format!(
                    "Exiting {} bars earlier would have lost ${:.2}",
                    -bars_delta,
                    trade.actual_pnl - counterfactual_pnl
                )
            }
        } else {
            if counterfactual_pnl > trade.actual_pnl {
                format!(
                    "Holding {} bars longer at ${:.2} would have made ${:.2} more",
                    bars_delta,
                    price_at_alt_exit,
                    counterfactual_pnl - trade.actual_pnl
                )
            } else {
                format!(
                    "Holding {} bars longer would have lost ${:.2}",
                    bars_delta,
                    counterfactual_pnl - trade.actual_pnl
                )
            }
        };

        // Confidence decreases with larger bar deltas
        let confidence = 1.0 - (bars_delta.abs() as f64 * 0.05).min(0.5);

        Some(CounterfactualResult::new(
            query,
            trade.actual_pnl,
            counterfactual_pnl,
            insight,
            confidence,
        ))
    }

    /// Analyze all recent trades and aggregate insights
    pub fn analyze_all_recent(&mut self, limit: u32) -> Vec<TradingInsight> {
        // Get recent closed trades from memory
        let trades = match self.get_recent_trades_for_analysis(limit) {
            Some(t) => t,
            None => return Vec::new(),
        };

        if trades.is_empty() {
            return Vec::new();
        }

        // Analyze each trade
        let mut all_results: Vec<CounterfactualResult> = Vec::new();
        for trade in &trades {
            let results = self.analyze_trade(trade);
            all_results.extend(results);
        }

        // Store results
        self.analyses.extend(all_results.clone());
        self.total_trades_analyzed += trades.len() as u32;
        self.last_analysis = Some(Utc::now());

        // Keep only recent analyses (last 1000)
        if self.analyses.len() > 1000 {
            self.analyses = self.analyses.split_off(self.analyses.len() - 1000);
        }

        // Aggregate into insights
        let insights = self.aggregate_insights(&all_results, &trades);
        self.insights = insights.clone();

        insights
    }

    /// Get recent trades formatted for analysis
    fn get_recent_trades_for_analysis(&self, limit: u32) -> Option<Vec<TradeForAnalysis>> {
        // Get all symbols from recent trade contexts
        // We'll aggregate across all symbols
        let mut all_trades = Vec::new();

        // Try to get trades for known symbols (we'll use a broad query)
        // In practice, we'd track symbols or query differently
        if let Ok(trades) = self.memory.get_trade_contexts("*", limit as i32) {
            for trade in trades {
                // Only include closed trades (have exit_price)
                if trade.exit_price.is_none() {
                    continue;
                }

                // Build price history (simplified - would need bar data)
                let price_history = self.build_price_history_from_context(&trade);

                let direction = if trade.direction == "LONG" {
                    Direction::Long
                } else {
                    Direction::Short
                };

                let trade_analysis = TradeForAnalysis {
                    ticket: trade.ticket,
                    symbol: trade.symbol.clone(),
                    direction,
                    entry_price: trade.entry_price,
                    exit_price: trade.exit_price.unwrap_or(trade.entry_price),
                    entry_time: trade.opened_at,
                    exit_time: trade.closed_at.unwrap_or_else(Utc::now),
                    size: 1.0, // Size not stored in TradeContext
                    actual_pnl: trade.profit.unwrap_or(0.0),
                    bars_held: trade.hold_bars.unwrap_or(1) as u32,
                    price_history,
                    entry_regime: Some(trade.regime.clone()),
                    sr_score: Some(trade.sr_score),
                    volume_percentile: Some(trade.volume_percentile),
                };

                all_trades.push(trade_analysis);
            }
        }

        if all_trades.is_empty() {
            return None;
        }

        // Sort by close time, most recent first
        all_trades.sort_by(|a, b| b.exit_time.cmp(&a.exit_time));

        // Limit to requested number
        all_trades.truncate(limit as usize);

        Some(all_trades)
    }

    /// Build price history for a trade (simplified - would need bar data)
    fn build_price_history_from_context(&self, trade: &MemoryTradeContext) -> Vec<f64> {
        let bars = trade.hold_bars.unwrap_or(1) as usize;
        let entry = trade.entry_price;
        let exit = trade.exit_price.unwrap_or(entry);

        // Linear interpolation as placeholder
        // In production, we'd query actual bar data
        let mut history = Vec::with_capacity(bars + 10);
        for i in 0..=bars {
            let t = i as f64 / bars.max(1) as f64;
            let price = entry + (exit - entry) * t;
            // Add some noise to make it more realistic
            let noise = (i as f64 * 0.1).sin() * 0.001 * entry;
            history.push(price + noise);
        }

        // Add a few bars after exit for "what if held longer" analysis
        for i in 1..=10 {
            let t = i as f64 / 10.0;
            let drift = (exit - entry) * 0.1 * t; // Continue trend slightly
            let noise = (i as f64 * 0.2).sin() * 0.002 * exit;
            history.push(exit + drift + noise);
        }

        history
    }

    /// Aggregate individual results into insights
    fn aggregate_insights(
        &self,
        results: &[CounterfactualResult],
        trades: &[TradeForAnalysis],
    ) -> Vec<TradingInsight> {
        let mut insights = Vec::new();

        // Analyze exit timing patterns
        let exit_insights = self.analyze_exit_patterns(results);
        insights.extend(exit_insights);

        // Analyze direction patterns
        let direction_insights = self.analyze_direction_patterns(results, trades);
        insights.extend(direction_insights);

        // Analyze sizing patterns
        let sizing_insights = self.analyze_sizing_patterns(results, trades);
        insights.extend(sizing_insights);

        // Analyze no-trade patterns
        let skip_insights = self.analyze_skip_patterns(results, trades);
        insights.extend(skip_insights);

        // Sort by improvement magnitude
        insights.sort_by(|a, b| {
            b.avg_improvement
                .partial_cmp(&a.avg_improvement)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        insights
    }

    /// Analyze exit timing patterns
    fn analyze_exit_patterns(&self, results: &[CounterfactualResult]) -> Vec<TradingInsight> {
        let mut insights = Vec::new();

        // Group exit results
        let mut earlier_better: Vec<f64> = Vec::new();
        let mut later_better: Vec<f64> = Vec::new();

        for result in results {
            if let QueryType::WhatIfDifferentExit { bars_delta } = result.query.query_type {
                if result.difference > 0.0 {
                    // Counterfactual was better
                    if bars_delta < 0 {
                        earlier_better.push(result.difference);
                    } else {
                        later_better.push(result.difference);
                    }
                }
            }
        }

        // Exit too early pattern
        if later_better.len() >= MIN_EVIDENCE_FOR_INSIGHT as usize {
            let avg_improvement: f64 = later_better.iter().sum::<f64>() / later_better.len() as f64;
            insights.push(TradingInsight::new(
                InsightType::ExitTooEarly,
                format!(
                    "Consistently exiting too early, losing avg ${:.2} per trade",
                    avg_improvement
                ),
                later_better.len() as u32,
                avg_improvement,
            ));
        }

        // Exit too late pattern
        if earlier_better.len() >= MIN_EVIDENCE_FOR_INSIGHT as usize {
            let avg_improvement: f64 = earlier_better.iter().sum::<f64>() / earlier_better.len() as f64;
            insights.push(TradingInsight::new(
                InsightType::ExitTooLate,
                format!(
                    "Consistently exiting too late, losing avg ${:.2} per trade",
                    avg_improvement
                ),
                earlier_better.len() as u32,
                avg_improvement,
            ));
        }

        insights
    }

    /// Analyze direction patterns
    fn analyze_direction_patterns(
        &self,
        results: &[CounterfactualResult],
        trades: &[TradeForAnalysis],
    ) -> Vec<TradingInsight> {
        let mut insights = Vec::new();

        let mut wrong_direction_count = 0;
        let mut wrong_direction_loss: f64 = 0.0;

        for result in results {
            if let QueryType::WhatIfOppositeDirection = result.query.query_type {
                if result.difference > 0.0 && result.difference > result.actual_pnl.abs() * 0.5 {
                    // Opposite direction significantly better
                    wrong_direction_count += 1;
                    wrong_direction_loss += result.difference;
                }
            }
        }

        if wrong_direction_count >= MIN_EVIDENCE_FOR_INSIGHT {
            let avg_improvement = wrong_direction_loss / wrong_direction_count as f64;
            insights.push(TradingInsight::new(
                InsightType::WrongDirection,
                format!(
                    "Wrong direction in {} trades, opposite would have made avg ${:.2} more",
                    wrong_direction_count, avg_improvement
                ),
                wrong_direction_count,
                avg_improvement,
            ));
        }

        // Check for good decisions too
        let good_direction: usize = results
            .iter()
            .filter(|r| matches!(r.query.query_type, QueryType::WhatIfOppositeDirection))
            .filter(|r| r.difference <= 0.0)
            .count();

        if good_direction >= MIN_EVIDENCE_FOR_INSIGHT as usize {
            insights.push(TradingInsight::new(
                InsightType::GoodDecision,
                format!("Direction choice correct in {} trades", good_direction),
                good_direction as u32,
                0.0,
            ));
        }

        insights
    }

    /// Analyze sizing patterns
    fn analyze_sizing_patterns(
        &self,
        results: &[CounterfactualResult],
        trades: &[TradeForAnalysis],
    ) -> Vec<TradingInsight> {
        let mut insights = Vec::new();

        // Find winners where 2x would have been better
        let mut size_too_small: Vec<f64> = Vec::new();
        // Find losers where 0.5x would have been better
        let mut size_too_large: Vec<f64> = Vec::new();

        for result in results {
            if let QueryType::WhatIfDifferentSize { multiplier } = result.query.query_type {
                if multiplier > 1.0 && result.difference > 0.0 && result.actual_pnl > 0.0 {
                    // Winner that would have been better with larger size
                    size_too_small.push(result.difference);
                } else if multiplier < 1.0 && result.difference > 0.0 && result.actual_pnl < 0.0 {
                    // Loser that would have been better with smaller size
                    size_too_large.push(result.difference);
                }
            }
        }

        if size_too_small.len() >= MIN_EVIDENCE_FOR_INSIGHT as usize {
            let avg_improvement: f64 = size_too_small.iter().sum::<f64>() / size_too_small.len() as f64;
            insights.push(TradingInsight::new(
                InsightType::SizeTooSmall,
                format!(
                    "Size too conservative on {} winners, avg ${:.2} left on table",
                    size_too_small.len(),
                    avg_improvement
                ),
                size_too_small.len() as u32,
                avg_improvement,
            ));
        }

        if size_too_large.len() >= MIN_EVIDENCE_FOR_INSIGHT as usize {
            let avg_improvement: f64 = size_too_large.iter().sum::<f64>() / size_too_large.len() as f64;
            insights.push(TradingInsight::new(
                InsightType::SizeTooLarge,
                format!(
                    "Size too aggressive on {} losers, avg ${:.2} excess loss",
                    size_too_large.len(),
                    avg_improvement
                ),
                size_too_large.len() as u32,
                avg_improvement,
            ));
        }

        insights
    }

    /// Analyze patterns where skipping would have been better
    fn analyze_skip_patterns(
        &self,
        results: &[CounterfactualResult],
        trades: &[TradeForAnalysis],
    ) -> Vec<TradingInsight> {
        let mut insights = Vec::new();

        let mut should_have_skipped: Vec<f64> = Vec::new();

        for result in results {
            if let QueryType::WhatIfNoTrade = result.query.query_type {
                if result.difference > 0.0 {
                    // No trade would have been better
                    should_have_skipped.push(result.difference);
                }
            }
        }

        if should_have_skipped.len() >= MIN_EVIDENCE_FOR_INSIGHT as usize {
            let avg_improvement: f64 = should_have_skipped.iter().sum::<f64>() / should_have_skipped.len() as f64;
            insights.push(TradingInsight::new(
                InsightType::ShouldHaveSkipped,
                format!(
                    "{} trades should have been skipped, avg ${:.2} saved",
                    should_have_skipped.len(),
                    avg_improvement
                ),
                should_have_skipped.len() as u32,
                avg_improvement,
            ));
        }

        insights
    }

    /// Get systematic errors (insights with sufficient evidence)
    pub fn get_systematic_errors(&self) -> Vec<TradingInsight> {
        self.insights
            .iter()
            .filter(|i| i.is_significant() && !matches!(i.insight_type, InsightType::GoodDecision))
            .cloned()
            .collect()
    }

    /// Get actionable recommendations
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        for insight in &self.insights {
            if !insight.is_significant() {
                continue;
            }

            let rec = match insight.insight_type {
                InsightType::ExitTooEarly => {
                    format!(
                        "Consider extending hold period by 2-3 bars (avg ${:.2} improvement)",
                        insight.avg_improvement
                    )
                }
                InsightType::ExitTooLate => {
                    "Consider tightening trailing stop or taking profits earlier".to_string()
                }
                InsightType::SizeTooSmall => {
                    format!(
                        "Consider increasing position size by 50% on high-confidence setups (potential ${:.2}/trade)",
                        insight.avg_improvement
                    )
                }
                InsightType::SizeTooLarge => {
                    format!(
                        "Consider reducing position size by 50% to limit downside (save ${:.2}/trade)",
                        insight.avg_improvement
                    )
                }
                InsightType::WrongDirection => {
                    "Review entry criteria - direction selection needs improvement".to_string()
                }
                InsightType::ShouldHaveSkipped => {
                    format!(
                        "Review entry filters - {} trades should have been skipped",
                        insight.evidence_count
                    )
                }
                InsightType::GoodDecision => continue,
            };

            recommendations.push(rec);
        }

        recommendations
    }

    /// Evaluate if a trade setup should be taken
    pub fn should_have_traded(&self, setup: &TradeContext) -> (bool, f64, String) {
        // Look for similar historical trades
        let similar_trades = self.find_similar_trades(setup);

        if similar_trades.is_empty() {
            return (true, 0.0, "No similar historical trades found".to_string());
        }

        // Calculate win rate and average PnL for similar trades
        let winners: Vec<_> = similar_trades.iter().filter(|t| t.actual_pnl > 0.0).collect();
        let win_rate = winners.len() as f64 / similar_trades.len() as f64;
        let avg_pnl: f64 = similar_trades.iter().map(|t| t.actual_pnl).sum::<f64>()
            / similar_trades.len() as f64;

        let should_trade = win_rate >= 0.45 && avg_pnl > 0.0;
        let expected_value = avg_pnl;

        let reason = if should_trade {
            format!(
                "Similar setups have {:.1}% win rate and ${:.2} avg PnL",
                win_rate * 100.0,
                avg_pnl
            )
        } else {
            format!(
                "Similar setups have only {:.1}% win rate and ${:.2} avg PnL - consider skipping",
                win_rate * 100.0,
                avg_pnl
            )
        };

        (should_trade, expected_value, reason)
    }

    /// Find similar historical trades
    fn find_similar_trades(&self, setup: &TradeContext) -> Vec<&TradeForAnalysis> {
        // We'd search our analyzed trades for similar conditions
        // Simplified: return empty for now since we don't store TradeForAnalysis persistently
        Vec::new()
    }

    /// Get recent analyses
    pub fn get_recent_analyses(&self, limit: usize) -> &[CounterfactualResult] {
        let start = self.analyses.len().saturating_sub(limit);
        &self.analyses[start..]
    }

    /// Get all insights
    pub fn get_insights(&self) -> &[TradingInsight] {
        &self.insights
    }

    /// Get total trades analyzed
    pub fn total_analyzed(&self) -> u32 {
        self.total_trades_analyzed
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        let errors = self.get_systematic_errors();
        if errors.is_empty() {
            format!(
                "{} trades analyzed, no systematic errors found",
                self.total_trades_analyzed
            )
        } else {
            format!(
                "{} trades analyzed, {} systematic errors",
                self.total_trades_analyzed,
                errors.len()
            )
        }
    }

    /// Save to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let state = CounterfactualState {
            analyses: self.analyses.clone(),
            insights: self.insights.clone(),
            last_analysis: self.last_analysis,
            total_trades_analyzed: self.total_trades_analyzed,
        };
        let contents = serde_json::to_string_pretty(&state)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Load from file or create new
    pub fn load_or_new(path: &str, memory: Arc<TradeMemory>) -> Self {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(state) = serde_json::from_str::<CounterfactualState>(&contents) {
                info!(
                    "[COUNTERFACTUAL] Loaded {} analyses from {}",
                    state.analyses.len(),
                    path
                );
                return Self {
                    memory,
                    world_model: None,
                    analyses: state.analyses,
                    insights: state.insights,
                    last_analysis: state.last_analysis,
                    total_trades_analyzed: state.total_trades_analyzed,
                };
            }
        }
        Self::new(memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trade(pnl: f64, direction: Direction) -> TradeForAnalysis {
        let entry = 100.0;
        let exit = if direction == Direction::Long {
            entry + pnl / 10.0 // Size of 10
        } else {
            entry - pnl / 10.0
        };

        TradeForAnalysis {
            ticket: 1,
            symbol: "TEST".to_string(),
            direction,
            entry_price: entry,
            exit_price: exit,
            entry_time: Utc::now(),
            exit_time: Utc::now(),
            size: 10.0,
            actual_pnl: pnl,
            bars_held: 5,
            price_history: vec![
                entry,
                entry + 0.5,
                entry + 1.0,
                entry + 0.8,
                entry + 1.2,
                exit,
                exit + 0.2,
                exit + 0.5,
                exit + 0.3,
                exit + 0.1,
                exit,
            ],
            entry_regime: Some("TRENDING_UP".to_string()),
            sr_score: Some(-2),
            volume_percentile: Some(85.0),
        }
    }

    #[test]
    fn test_direction_display() {
        assert_eq!(format!("{}", Direction::Long), "LONG");
        assert_eq!(format!("{}", Direction::Short), "SHORT");
    }

    #[test]
    fn test_query_type_display() {
        assert_eq!(format!("{}", QueryType::WhatIfNoTrade), "NoTrade");
        assert_eq!(format!("{}", QueryType::WhatIfOppositeDirection), "Opposite");
        assert_eq!(
            format!("{}", QueryType::WhatIfDifferentSize { multiplier: 2.0 }),
            "Size(2.0x)"
        );
        assert_eq!(
            format!("{}", QueryType::WhatIfDifferentExit { bars_delta: -3 }),
            "Exit(-3 bars)"
        );
    }

    #[test]
    fn test_trade_pnl_at_price() {
        let trade = create_test_trade(100.0, Direction::Long);
        let pnl_entry = trade.pnl_at_price(100.0);
        assert!((pnl_entry - 0.0).abs() < 0.01);

        let pnl_above = trade.pnl_at_price(105.0);
        assert!((pnl_above - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_what_if_no_trade() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let analyzer = CounterfactualAnalyzer::new(memory);

        // Winning trade
        let winner = create_test_trade(100.0, Direction::Long);
        let result = analyzer.what_if_no_trade(&winner);
        assert!(result.was_good_decision());
        assert!(result.insight.contains("captured"));

        // Losing trade
        let loser = create_test_trade(-50.0, Direction::Long);
        let result = analyzer.what_if_no_trade(&loser);
        assert!(!result.was_good_decision());
        assert!(result.insight.contains("Skipping"));
    }

    #[test]
    fn test_what_if_opposite() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let analyzer = CounterfactualAnalyzer::new(memory);

        let trade = create_test_trade(100.0, Direction::Long);
        let result = analyzer.what_if_opposite(&trade);

        // Opposite direction would have lost money
        assert!(result.counterfactual_pnl < 0.0);
        assert!(result.was_good_decision());
    }

    #[test]
    fn test_what_if_different_size() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let analyzer = CounterfactualAnalyzer::new(memory);

        let winner = create_test_trade(100.0, Direction::Long);

        // 2x size on winner = more profit
        let result_2x = analyzer.what_if_different_size(&winner, 2.0);
        assert!((result_2x.counterfactual_pnl - 200.0).abs() < 0.01);
        assert!(result_2x.difference > 0.0); // Could have made more

        // 0.5x size on winner = less profit
        let result_half = analyzer.what_if_different_size(&winner, 0.5);
        assert!((result_half.counterfactual_pnl - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_what_if_different_exit() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let analyzer = CounterfactualAnalyzer::new(memory);

        let trade = create_test_trade(100.0, Direction::Long);

        // Exit earlier
        let result_earlier = analyzer.what_if_different_exit(&trade, -2);
        assert!(result_earlier.is_some());
        let r = result_earlier.unwrap();
        assert!(r.confidence > 0.8); // Close bar delta = high confidence

        // Exit later
        let result_later = analyzer.what_if_different_exit(&trade, 3);
        assert!(result_later.is_some());
    }

    #[test]
    fn test_analyze_trade() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let analyzer = CounterfactualAnalyzer::new(memory);

        let trade = create_test_trade(100.0, Direction::Long);
        let results = analyzer.analyze_trade(&trade);

        // Should have multiple results
        assert!(results.len() >= 4); // NoTrade, Opposite, Size x2, Size x0.5 + exit variations

        // Sorted by magnitude
        for i in 1..results.len() {
            assert!(results[i - 1].magnitude() >= results[i].magnitude());
        }
    }

    #[test]
    fn test_insight_significance() {
        let insight = TradingInsight::new(
            InsightType::ExitTooEarly,
            "Test".to_string(),
            MIN_EVIDENCE_FOR_INSIGHT,
            50.0,
        );
        assert!(insight.is_significant());

        let weak_insight = TradingInsight::new(
            InsightType::ExitTooEarly,
            "Test".to_string(),
            MIN_EVIDENCE_FOR_INSIGHT - 1,
            50.0,
        );
        assert!(!weak_insight.is_significant());
    }

    #[test]
    fn test_insight_type_display() {
        assert_eq!(format!("{}", InsightType::ExitTooEarly), "ExitTooEarly");
        assert_eq!(format!("{}", InsightType::WrongDirection), "WrongDirection");
    }

    #[test]
    fn test_counterfactual_result() {
        let query = CounterfactualQuery::new(1, QueryType::WhatIfNoTrade);
        let result = CounterfactualResult::new(query, -100.0, 0.0, "Test".to_string(), 0.9);

        assert!(!result.was_good_decision()); // Counterfactual (no trade) was better
        assert!((result.difference - 100.0).abs() < 0.01);
        assert!((result.magnitude() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_recommendations() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let mut analyzer = CounterfactualAnalyzer::new(memory);

        // Add significant insight
        analyzer.insights.push(TradingInsight::new(
            InsightType::ExitTooEarly,
            "Test".to_string(),
            10,
            100.0,
        ));

        let recommendations = analyzer.get_recommendations();
        assert!(!recommendations.is_empty());
        assert!(recommendations[0].contains("extending hold period"));
    }

    #[test]
    fn test_format_summary() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let analyzer = CounterfactualAnalyzer::new(memory);

        let summary = analyzer.format_summary();
        assert!(summary.contains("0 trades analyzed"));
    }

    #[test]
    fn test_best_worst_price() {
        let trade = create_test_trade(100.0, Direction::Long);

        let best = trade.best_price();
        assert!(best.is_some());
        let best_val = best.unwrap();
        assert!(best_val >= trade.entry_price);

        let worst = trade.worst_price();
        assert!(worst.is_some());
        let worst_val = worst.unwrap();
        assert!(worst_val <= best_val);
    }

    #[test]
    fn test_should_have_traded() {
        let memory = Arc::new(TradeMemory::new(":memory:").unwrap());
        let analyzer = CounterfactualAnalyzer::new(memory);

        let context = TradeContext {
            symbol: "TEST".to_string(),
            direction: Direction::Long,
            sr_score: -2,
            volume_percentile: 85.0,
            regime: "TRENDING_UP".to_string(),
            price: 100.0,
        };

        let (should_trade, _ev, reason) = analyzer.should_have_traded(&context);
        // No historical data, so defaults to yes
        assert!(should_trade);
        assert!(reason.contains("No similar"));
    }
}
