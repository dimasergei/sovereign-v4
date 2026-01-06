//! Self-Directed Weakness Identification and Targeted Improvement
//!
//! Analyzes trade history to identify patterns where the system underperforms
//! and uses that information to improve future trading decisions.
//!
//! Features:
//! - Identifies weaknesses by regime, symbol, S/R score, volume threshold
//! - Calculates severity based on win rate and trade count
//! - Suggests actionable improvements
//! - Can skip trades matching critical weakness patterns
//! - Adjusts position sizing based on historical performance

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

use crate::data::memory::TradeMemory;
use super::regime::Regime;
use super::transfer::AssetCluster;

/// Default minimum trades required for analysis
const DEFAULT_MIN_TRADES: u32 = 10;

/// Default weakness threshold (win rate below this = weakness)
const DEFAULT_WEAKNESS_THRESHOLD: f64 = 0.45;

/// Critical severity threshold (skip trades matching this)
const CRITICAL_SEVERITY_THRESHOLD: f64 = 0.7;

/// Types of weaknesses that can be identified
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WeaknessType {
    /// Poor performance in a specific market regime
    RegimeWeakness {
        regime: Regime,
        win_rate: f64,
        trade_count: u32,
    },
    /// Poor performance trading a specific symbol
    SymbolWeakness {
        symbol: String,
        win_rate: f64,
        trade_count: u32,
    },
    /// Poor performance in an asset cluster
    ClusterWeakness {
        cluster: AssetCluster,
        win_rate: f64,
        trade_count: u32,
    },
    /// Poor performance at certain S/R score ranges
    SRScoreWeakness {
        score_range: (i32, i32),
        win_rate: f64,
        trade_count: u32,
    },
    /// Poor performance at certain volume thresholds
    VolumeThresholdWeakness {
        threshold_range: (f64, f64),
        win_rate: f64,
        trade_count: u32,
    },
    /// Poor performance at certain hours
    TimeOfDayWeakness {
        hour_utc: u8,
        win_rate: f64,
        trade_count: u32,
    },
}

impl WeaknessType {
    /// Get a short description of the weakness
    pub fn description(&self) -> String {
        match self {
            WeaknessType::RegimeWeakness { regime, win_rate, trade_count } => {
                format!(
                    "{} regime: {:.1}% win rate ({} trades)",
                    regime, win_rate * 100.0, trade_count
                )
            }
            WeaknessType::SymbolWeakness { symbol, win_rate, trade_count } => {
                format!(
                    "{}: {:.1}% win rate ({} trades)",
                    symbol, win_rate * 100.0, trade_count
                )
            }
            WeaknessType::ClusterWeakness { cluster, win_rate, trade_count } => {
                format!(
                    "{} cluster: {:.1}% win rate ({} trades)",
                    cluster.name(), win_rate * 100.0, trade_count
                )
            }
            WeaknessType::SRScoreWeakness { score_range, win_rate, trade_count } => {
                format!(
                    "S/R score [{},{}]: {:.1}% win rate ({} trades)",
                    score_range.0, score_range.1, win_rate * 100.0, trade_count
                )
            }
            WeaknessType::VolumeThresholdWeakness { threshold_range, win_rate, trade_count } => {
                format!(
                    "Volume {:.0}-{:.0}%: {:.1}% win rate ({} trades)",
                    threshold_range.0, threshold_range.1, win_rate * 100.0, trade_count
                )
            }
            WeaknessType::TimeOfDayWeakness { hour_utc, win_rate, trade_count } => {
                format!(
                    "Hour {} UTC: {:.1}% win rate ({} trades)",
                    hour_utc, win_rate * 100.0, trade_count
                )
            }
        }
    }

    /// Get win rate from any weakness type
    pub fn win_rate(&self) -> f64 {
        match self {
            WeaknessType::RegimeWeakness { win_rate, .. } => *win_rate,
            WeaknessType::SymbolWeakness { win_rate, .. } => *win_rate,
            WeaknessType::ClusterWeakness { win_rate, .. } => *win_rate,
            WeaknessType::SRScoreWeakness { win_rate, .. } => *win_rate,
            WeaknessType::VolumeThresholdWeakness { win_rate, .. } => *win_rate,
            WeaknessType::TimeOfDayWeakness { win_rate, .. } => *win_rate,
        }
    }

    /// Get trade count from any weakness type
    pub fn trade_count(&self) -> u32 {
        match self {
            WeaknessType::RegimeWeakness { trade_count, .. } => *trade_count,
            WeaknessType::SymbolWeakness { trade_count, .. } => *trade_count,
            WeaknessType::ClusterWeakness { trade_count, .. } => *trade_count,
            WeaknessType::SRScoreWeakness { trade_count, .. } => *trade_count,
            WeaknessType::VolumeThresholdWeakness { trade_count, .. } => *trade_count,
            WeaknessType::TimeOfDayWeakness { trade_count, .. } => *trade_count,
        }
    }
}

/// An identified weakness with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weakness {
    /// The type and details of the weakness
    pub weakness_type: WeaknessType,
    /// Severity score (0.0 = minor, 1.0 = critical)
    pub severity: f64,
    /// When this weakness was identified
    pub identified_at: DateTime<Utc>,
    /// Number of trades analyzed
    pub trades_analyzed: u32,
    /// Suggested action to address the weakness
    pub suggested_action: String,
}

impl Weakness {
    /// Create a new weakness with calculated severity
    pub fn new(
        weakness_type: WeaknessType,
        threshold: f64,
    ) -> Self {
        let severity = Self::calculate_severity(
            weakness_type.win_rate(),
            weakness_type.trade_count(),
            threshold,
        );
        let suggested_action = Self::suggest_action(&weakness_type);

        Self {
            trades_analyzed: weakness_type.trade_count(),
            weakness_type,
            severity,
            identified_at: Utc::now(),
            suggested_action,
        }
    }

    /// Calculate severity based on win rate and trade count
    ///
    /// Lower win rate = higher severity
    /// More trades = higher confidence in severity
    fn calculate_severity(win_rate: f64, trade_count: u32, threshold: f64) -> f64 {
        if win_rate >= threshold || trade_count == 0 {
            return 0.0;
        }

        // Base severity from win rate gap
        let gap = threshold - win_rate;

        // Confidence multiplier from trade count (log scale)
        let confidence = (trade_count as f64 + 1.0).ln() / 100_f64.ln();

        // Severity = gap * confidence, capped at 1.0
        (gap * confidence * 2.0).min(1.0)
    }

    /// Suggest an action to address the weakness
    fn suggest_action(weakness_type: &WeaknessType) -> String {
        match weakness_type {
            WeaknessType::RegimeWeakness { regime, win_rate, .. } => {
                if *win_rate < 0.35 {
                    format!("Skip {} regime trades entirely", regime)
                } else {
                    format!("Reduce position size in {} by 50%", regime)
                }
            }
            WeaknessType::SymbolWeakness { symbol, win_rate, .. } => {
                if *win_rate < 0.35 {
                    format!("Remove {} from trading universe", symbol)
                } else {
                    format!("Reduce {} position size by 50%", symbol)
                }
            }
            WeaknessType::ClusterWeakness { cluster, win_rate, .. } => {
                if *win_rate < 0.35 {
                    format!("Avoid {} assets", cluster.name())
                } else {
                    format!("Reduce {} cluster exposure by 50%", cluster.name())
                }
            }
            WeaknessType::SRScoreWeakness { score_range, .. } => {
                format!(
                    "Require stronger S/R (score >= {}) for entry",
                    score_range.0.min(score_range.1) + 1
                )
            }
            WeaknessType::VolumeThresholdWeakness { threshold_range, .. } => {
                format!(
                    "Raise volume threshold to {}th percentile",
                    threshold_range.1 as i32
                )
            }
            WeaknessType::TimeOfDayWeakness { hour_utc, .. } => {
                format!("Avoid trading during hour {} UTC", hour_utc)
            }
        }
    }

    /// Check if this weakness is critical (high severity)
    pub fn is_critical(&self) -> bool {
        self.severity >= CRITICAL_SEVERITY_THRESHOLD
    }
}

/// Serializable state for WeaknessAnalyzer persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeaknessState {
    weaknesses: Vec<Weakness>,
    min_trades_for_analysis: u32,
    weakness_threshold: f64,
    last_analysis: Option<DateTime<Utc>>,
    symbols: Vec<String>,
}

/// Analyzer for identifying trading weaknesses
pub struct WeaknessAnalyzer {
    /// Trade memory for querying history
    memory: Arc<TradeMemory>,
    /// Minimum trades required for analysis
    min_trades_for_analysis: u32,
    /// Win rate below this is considered weak
    weakness_threshold: f64,
    /// Currently identified weaknesses
    weaknesses: Vec<Weakness>,
    /// Last time analysis was run
    last_analysis: Option<DateTime<Utc>>,
    /// Symbols to analyze
    symbols: Vec<String>,
}

impl WeaknessAnalyzer {
    /// Create a new weakness analyzer
    pub fn new(memory: Arc<TradeMemory>) -> Self {
        Self {
            memory,
            min_trades_for_analysis: DEFAULT_MIN_TRADES,
            weakness_threshold: DEFAULT_WEAKNESS_THRESHOLD,
            weaknesses: Vec::new(),
            last_analysis: None,
            symbols: Vec::new(),
        }
    }

    /// Create analyzer with custom thresholds
    pub fn with_thresholds(
        memory: Arc<TradeMemory>,
        min_trades: u32,
        weakness_threshold: f64,
    ) -> Self {
        Self {
            memory,
            min_trades_for_analysis: min_trades,
            weakness_threshold,
            weaknesses: Vec::new(),
            last_analysis: None,
            symbols: Vec::new(),
        }
    }

    /// Set symbols to analyze
    pub fn set_symbols(&mut self, symbols: Vec<String>) {
        self.symbols = symbols;
    }

    /// Get the current weakness threshold
    pub fn weakness_threshold(&self) -> f64 {
        self.weakness_threshold
    }

    /// Get minimum trades for analysis
    pub fn min_trades(&self) -> u32 {
        self.min_trades_for_analysis
    }

    /// Run all analysis methods and return top weaknesses
    pub fn analyze_all(&mut self) -> Vec<Weakness> {
        let mut all_weaknesses = Vec::new();

        // Analyze by regime
        all_weaknesses.extend(self.analyze_by_regime());

        // Analyze by symbol
        all_weaknesses.extend(self.analyze_by_symbol());

        // Analyze by S/R score
        all_weaknesses.extend(self.analyze_by_sr_score());

        // Analyze by volume threshold
        all_weaknesses.extend(self.analyze_by_volume_threshold());

        // Sort by severity descending
        all_weaknesses.sort_by(|a, b| {
            b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top 10
        all_weaknesses.truncate(10);

        // Store and update timestamp
        self.weaknesses = all_weaknesses.clone();
        self.last_analysis = Some(Utc::now());

        info!(
            "[WEAKNESS] Analyzed: found {} weaknesses, {} critical",
            self.weaknesses.len(),
            self.weaknesses.iter().filter(|w| w.is_critical()).count()
        );

        all_weaknesses
    }

    /// Analyze weaknesses by market regime
    pub fn analyze_by_regime(&self) -> Vec<Weakness> {
        let mut weaknesses = Vec::new();

        if let Ok(regime_stats) = self.memory.get_win_rate_by_regime() {
            for (regime_str, win_rate, trade_count) in regime_stats {
                if (trade_count as u32) < self.min_trades_for_analysis {
                    continue;
                }

                if win_rate < self.weakness_threshold {
                    // Convert string to Regime enum
                    let regime = match regime_str.as_str() {
                        "TrendingUp" | "Bull" => Regime::TrendingUp,
                        "TrendingDown" | "Bear" => Regime::TrendingDown,
                        "Ranging" | "Sideways" => Regime::Ranging,
                        "Volatile" | "HighVolatility" | "LowVolatility" => Regime::Volatile,
                        _ => continue,
                    };

                    let weakness_type = WeaknessType::RegimeWeakness {
                        regime,
                        win_rate,
                        trade_count: trade_count as u32,
                    };
                    weaknesses.push(Weakness::new(weakness_type, self.weakness_threshold));
                }
            }
        }

        weaknesses
    }

    /// Analyze weaknesses by symbol
    pub fn analyze_by_symbol(&self) -> Vec<Weakness> {
        let mut weaknesses = Vec::new();

        for symbol in &self.symbols {
            if let Ok(trades) = self.memory.get_trade_contexts(symbol, 100) {
                if trades.len() < self.min_trades_for_analysis as usize {
                    continue;
                }

                // Calculate win rate for this symbol
                let wins = trades.iter()
                    .filter(|t| t.profit.map(|p| p > 0.0).unwrap_or(false))
                    .count();
                let win_rate = wins as f64 / trades.len() as f64;

                if win_rate < self.weakness_threshold {
                    let weakness_type = WeaknessType::SymbolWeakness {
                        symbol: symbol.clone(),
                        win_rate,
                        trade_count: trades.len() as u32,
                    };
                    weaknesses.push(Weakness::new(weakness_type, self.weakness_threshold));
                }
            }
        }

        weaknesses
    }

    /// Analyze weaknesses by S/R score ranges
    pub fn analyze_by_sr_score(&self) -> Vec<Weakness> {
        let mut weaknesses = Vec::new();

        // S/R score buckets: [0], [-1,-2], [-3,-4], [-5 and below]
        let buckets: [(i32, i32); 4] = [
            (0, 0),
            (-2, -1),
            (-4, -3),
            (-10, -5),
        ];

        // Aggregate across all symbols
        let mut bucket_stats: Vec<(u32, u32)> = vec![(0, 0); 4]; // (wins, total)

        for symbol in &self.symbols {
            if let Ok(score_stats) = self.memory.get_sr_score_effectiveness(symbol) {
                for (score, win_rate, count) in score_stats {
                    // Find which bucket this score belongs to
                    for (idx, (low, high)) in buckets.iter().enumerate() {
                        if score >= *low && score <= *high {
                            let wins = (win_rate * count as f64).round() as u32;
                            bucket_stats[idx].0 += wins;
                            bucket_stats[idx].1 += count as u32;
                            break;
                        }
                    }
                }
            }
        }

        // Create weaknesses for underperforming buckets
        for (idx, (wins, total)) in bucket_stats.iter().enumerate() {
            if *total < self.min_trades_for_analysis {
                continue;
            }

            let win_rate = *wins as f64 / *total as f64;
            if win_rate < self.weakness_threshold {
                let weakness_type = WeaknessType::SRScoreWeakness {
                    score_range: buckets[idx],
                    win_rate,
                    trade_count: *total,
                };
                weaknesses.push(Weakness::new(weakness_type, self.weakness_threshold));
            }
        }

        weaknesses
    }

    /// Analyze weaknesses by volume threshold ranges
    pub fn analyze_by_volume_threshold(&self) -> Vec<Weakness> {
        let mut weaknesses = Vec::new();

        // Volume percentile buckets
        let buckets: [(f64, f64); 4] = [
            (50.0, 65.0),
            (65.0, 80.0),
            (80.0, 90.0),
            (90.0, 100.0),
        ];

        // Aggregate across all symbols
        let mut bucket_stats: Vec<(u32, u32)> = vec![(0, 0); 4]; // (wins, total)

        for symbol in &self.symbols {
            if let Ok(vol_stats) = self.memory.get_volume_percentile_effectiveness(symbol) {
                for (percentile_bucket, win_rate, count) in vol_stats {
                    // Find which bucket this belongs to (percentile_bucket is typically 0-10 scale)
                    let pct = (percentile_bucket * 10) as f64;
                    for (idx, (low, high)) in buckets.iter().enumerate() {
                        if pct >= *low && pct < *high {
                            let wins = (win_rate * count as f64).round() as u32;
                            bucket_stats[idx].0 += wins;
                            bucket_stats[idx].1 += count as u32;
                            break;
                        }
                    }
                }
            }
        }

        // Create weaknesses for underperforming buckets
        for (idx, (wins, total)) in bucket_stats.iter().enumerate() {
            if *total < self.min_trades_for_analysis {
                continue;
            }

            let win_rate = *wins as f64 / *total as f64;
            if win_rate < self.weakness_threshold {
                let weakness_type = WeaknessType::VolumeThresholdWeakness {
                    threshold_range: buckets[idx],
                    win_rate,
                    trade_count: *total,
                };
                weaknesses.push(Weakness::new(weakness_type, self.weakness_threshold));
            }
        }

        weaknesses
    }

    /// Get current weaknesses
    pub fn get_weaknesses(&self) -> &[Weakness] {
        &self.weaknesses
    }

    /// Get critical weaknesses only
    pub fn get_critical_weaknesses(&self) -> Vec<&Weakness> {
        self.weaknesses.iter().filter(|w| w.is_critical()).collect()
    }

    /// Get improvement priorities (top 3 weaknesses with actions)
    pub fn get_improvement_priorities(&self) -> Vec<(String, String)> {
        self.weaknesses
            .iter()
            .take(3)
            .map(|w| (w.weakness_type.description(), w.suggested_action.clone()))
            .collect()
    }

    /// Check if a trade setup matches a critical weakness
    ///
    /// Returns Some(reason) to skip the trade, None to proceed
    pub fn should_skip_trade(
        &self,
        regime: &Regime,
        sr_score: i32,
        vol_pct: f64,
        symbol: &str,
    ) -> Option<String> {
        for weakness in &self.weaknesses {
            if !weakness.is_critical() {
                continue;
            }

            match &weakness.weakness_type {
                WeaknessType::RegimeWeakness { regime: weak_regime, .. } => {
                    if regime == weak_regime {
                        return Some(format!(
                            "Critical weakness: {} regime ({:.1}% win rate)",
                            regime, weakness.weakness_type.win_rate() * 100.0
                        ));
                    }
                }
                WeaknessType::SymbolWeakness { symbol: weak_symbol, .. } => {
                    if symbol == weak_symbol {
                        return Some(format!(
                            "Critical weakness: {} ({:.1}% win rate)",
                            symbol, weakness.weakness_type.win_rate() * 100.0
                        ));
                    }
                }
                WeaknessType::SRScoreWeakness { score_range, .. } => {
                    if sr_score >= score_range.0 && sr_score <= score_range.1 {
                        return Some(format!(
                            "Critical weakness: S/R score {} in weak range [{},{}]",
                            sr_score, score_range.0, score_range.1
                        ));
                    }
                }
                WeaknessType::VolumeThresholdWeakness { threshold_range, .. } => {
                    if vol_pct >= threshold_range.0 && vol_pct < threshold_range.1 {
                        return Some(format!(
                            "Critical weakness: volume {:.0}% in weak range [{:.0},{:.0}]",
                            vol_pct, threshold_range.0, threshold_range.1
                        ));
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Get position size multiplier based on weakness analysis
    ///
    /// Returns:
    /// - 0.5 for weak regime/symbol combinations
    /// - 1.0 for normal
    /// - 1.25 for strong historical performance
    pub fn get_position_size_multiplier(&self, regime: &Regime, symbol: &str) -> f64 {
        // Check for regime weakness
        for weakness in &self.weaknesses {
            match &weakness.weakness_type {
                WeaknessType::RegimeWeakness { regime: weak_regime, .. } => {
                    if regime == weak_regime {
                        return if weakness.is_critical() { 0.0 } else { 0.5 };
                    }
                }
                WeaknessType::SymbolWeakness { symbol: weak_symbol, .. } => {
                    if symbol == weak_symbol {
                        return if weakness.is_critical() { 0.0 } else { 0.5 };
                    }
                }
                _ => {}
            }
        }

        // Check for strong performance (inverse analysis)
        // If win rate in this regime/symbol is notably above threshold, boost size
        if let Ok(regime_stats) = self.memory.get_win_rate_by_regime() {
            let regime_str = regime.as_str();
            for (r_str, win_rate, count) in regime_stats {
                if r_str == regime_str && count >= self.min_trades_for_analysis as i32 {
                    if win_rate >= 0.60 {
                        return 1.25;
                    }
                }
            }
        }

        1.0
    }

    /// Get last analysis timestamp
    pub fn last_analysis(&self) -> Option<DateTime<Utc>> {
        self.last_analysis
    }

    /// Get weakness count
    pub fn weakness_count(&self) -> usize {
        self.weaknesses.len()
    }

    /// Get count of critical weaknesses (severity > 0.7)
    pub fn critical_weakness_count(&self) -> usize {
        self.weaknesses.iter().filter(|w| w.is_critical()).count()
    }

    /// Load from file or create new (with dummy memory - weaknesses are persisted separately)
    pub fn load_or_new(path: &str) -> Self {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(state) = serde_json::from_str::<WeaknessState>(&contents) {
                info!("[WEAKNESS] Loaded {} weaknesses from {}", state.weaknesses.len(), path);
                // We need memory to be set separately via set_memory()
                return Self {
                    memory: Arc::new(TradeMemory::new(":memory:").unwrap()),
                    min_trades_for_analysis: state.min_trades_for_analysis,
                    weakness_threshold: state.weakness_threshold,
                    weaknesses: state.weaknesses,
                    last_analysis: state.last_analysis,
                    symbols: state.symbols,
                };
            }
        }
        // Create with dummy in-memory database
        Self {
            memory: Arc::new(TradeMemory::new(":memory:").unwrap()),
            min_trades_for_analysis: DEFAULT_MIN_TRADES,
            weakness_threshold: DEFAULT_WEAKNESS_THRESHOLD,
            weaknesses: Vec::new(),
            last_analysis: None,
            symbols: Vec::new(),
        }
    }

    /// Save to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let state = WeaknessState {
            weaknesses: self.weaknesses.clone(),
            min_trades_for_analysis: self.min_trades_for_analysis,
            weakness_threshold: self.weakness_threshold,
            last_analysis: self.last_analysis,
            symbols: self.symbols.clone(),
        };
        let contents = serde_json::to_string_pretty(&state)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Set memory reference (required after load_or_new)
    pub fn set_memory(&mut self, memory: Arc<TradeMemory>) {
        self.memory = memory;
    }

    /// Format weaknesses for logging
    pub fn format_summary(&self) -> String {
        if self.weaknesses.is_empty() {
            return "No significant weaknesses identified".to_string();
        }

        let top: Vec<String> = self.weaknesses
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, w)| format!("{}. {}", i + 1, w.weakness_type.description()))
            .collect();

        format!("Top issues: {}", top.join(" | "))
    }

    /// Log current weaknesses
    pub fn log_weaknesses(&self) {
        if self.weaknesses.is_empty() {
            info!("[WEAKNESS] No significant weaknesses identified");
            return;
        }

        info!("[WEAKNESS] {}", self.format_summary());

        for w in &self.weaknesses {
            if w.is_critical() {
                info!(
                    "[WEAKNESS] CRITICAL: {} (severity: {:.2}) - Action: {}",
                    w.weakness_type.description(),
                    w.severity,
                    w.suggested_action
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weakness_type_description() {
        let wt = WeaknessType::RegimeWeakness {
            regime: Regime::Volatile,
            win_rate: 0.35,
            trade_count: 20,
        };
        let desc = wt.description();
        assert!(desc.contains("Volatile"));
        assert!(desc.contains("35.0%"));
        assert!(desc.contains("20"));
    }

    #[test]
    fn test_weakness_type_getters() {
        let wt = WeaknessType::SymbolWeakness {
            symbol: "AAPL".to_string(),
            win_rate: 0.40,
            trade_count: 50,
        };
        assert!((wt.win_rate() - 0.40).abs() < 0.001);
        assert_eq!(wt.trade_count(), 50);
    }

    #[test]
    fn test_severity_calculation() {
        let threshold = 0.45;

        // Low win rate, high trade count = positive severity
        // gap=0.15, confidence=ln(51)/ln(100)=0.854, severity=0.15*0.854*2=0.256
        let sev1 = Weakness::calculate_severity(0.30, 50, threshold);
        assert!(sev1 > 0.2);

        // Moderate win rate, low trade count = lower severity
        let sev2 = Weakness::calculate_severity(0.40, 10, threshold);
        assert!(sev2 < sev1);
        assert!(sev2 > 0.0);

        // Win rate at threshold = zero severity
        let sev3 = Weakness::calculate_severity(0.45, 100, threshold);
        assert!((sev3 - 0.0).abs() < 0.001);

        // Win rate above threshold = zero severity
        let sev4 = Weakness::calculate_severity(0.55, 100, threshold);
        assert!((sev4 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_weakness_is_critical() {
        let critical = Weakness {
            weakness_type: WeaknessType::RegimeWeakness {
                regime: Regime::Volatile,
                win_rate: 0.25,
                trade_count: 100,
            },
            severity: 0.8,
            identified_at: Utc::now(),
            trades_analyzed: 100,
            suggested_action: "Test".to_string(),
        };
        assert!(critical.is_critical());

        let minor = Weakness {
            weakness_type: WeaknessType::RegimeWeakness {
                regime: Regime::Ranging,
                win_rate: 0.42,
                trade_count: 20,
            },
            severity: 0.3,
            identified_at: Utc::now(),
            trades_analyzed: 20,
            suggested_action: "Test".to_string(),
        };
        assert!(!minor.is_critical());
    }

    #[test]
    fn test_suggest_action_regime() {
        let wt = WeaknessType::RegimeWeakness {
            regime: Regime::Volatile,
            win_rate: 0.30,
            trade_count: 50,
        };
        let action = Weakness::suggest_action(&wt);
        assert!(action.contains("Skip") || action.contains("Reduce"));
    }

    #[test]
    fn test_suggest_action_sr_score() {
        let wt = WeaknessType::SRScoreWeakness {
            score_range: (-4, -3),
            win_rate: 0.35,
            trade_count: 30,
        };
        let action = Weakness::suggest_action(&wt);
        assert!(action.contains("Require stronger S/R"));
    }

    #[test]
    fn test_suggest_action_volume() {
        let wt = WeaknessType::VolumeThresholdWeakness {
            threshold_range: (50.0, 65.0),
            win_rate: 0.38,
            trade_count: 25,
        };
        let action = Weakness::suggest_action(&wt);
        assert!(action.contains("Raise volume threshold"));
    }

    #[test]
    fn test_weakness_new() {
        let wt = WeaknessType::SymbolWeakness {
            symbol: "TEST".to_string(),
            win_rate: 0.35,
            trade_count: 30,
        };
        let weakness = Weakness::new(wt, 0.45);

        assert!(weakness.severity > 0.0);
        assert!(!weakness.suggested_action.is_empty());
        assert_eq!(weakness.trades_analyzed, 30);
    }

    #[test]
    fn test_improvement_priorities_format() {
        // This is more of a smoke test for the format
        let wt = WeaknessType::RegimeWeakness {
            regime: Regime::TrendingDown,
            win_rate: 0.38,
            trade_count: 40,
        };
        let desc = wt.description();
        assert!(desc.contains("TrendingDown"));
    }

    #[test]
    fn test_cluster_weakness() {
        let wt = WeaknessType::ClusterWeakness {
            cluster: AssetCluster::Crypto,
            win_rate: 0.32,
            trade_count: 25,
        };
        let desc = wt.description();
        assert!(desc.contains("Crypto"));

        let action = Weakness::suggest_action(&wt);
        assert!(action.contains("Avoid") || action.contains("Reduce"));
    }

    #[test]
    fn test_time_of_day_weakness() {
        let wt = WeaknessType::TimeOfDayWeakness {
            hour_utc: 14,
            win_rate: 0.33,
            trade_count: 15,
        };
        let desc = wt.description();
        assert!(desc.contains("14"));
        assert!(desc.contains("UTC"));

        let action = Weakness::suggest_action(&wt);
        assert!(action.contains("Avoid trading"));
    }
}
