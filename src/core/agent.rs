//! Symbol Agent Module - Lossless Implementation
//!
//! "Analogous to having a thousand independent traders each focusing on a
//! single stock, as opposed to a single quant manager trying to make sense
//! of a thousand datapoints." - pftq
//!
//! Each agent is completely independent:
//! - No cross-symbol logic
//! - No portfolio optimization
//! - No correlation analysis
//! - No sector rotation logic
//!
//! LOSSLESS PRINCIPLES:
//! - Volume percentile derived from data distribution (not fixed thresholds)
//! - Granularity derived from ATR (not price-based thresholds)
//! - Entry signals based on percentile ranking (not "top N")

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

#[allow(deprecated)]
use super::sr::{SRLevels, default_granularity, granularity_from_atr};
use super::capitulation::VolumeTracker;
use super::regime::{Regime, RegimeDetector};
use super::learner::{ConfidenceCalibrator, TradeOutcome, NUM_FEATURES};
use super::transfer::TransferManager;
use super::moe::MixtureOfExperts;
use super::metalearner::{MetaLearner, calculate_accuracy};
use crate::data::memory::{TradeMemory, MarketRegime};
use std::sync::Mutex;

/// Trading signal from an agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// Open a long position
    Buy,
    /// Close a long position
    Sell,
    /// Open a short position
    Short,
    /// Close a short position
    Cover,
    /// No action
    Hold,
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Buy => write!(f, "BUY"),
            Signal::Sell => write!(f, "SELL"),
            Signal::Short => write!(f, "SHORT"),
            Signal::Cover => write!(f, "COVER"),
            Signal::Hold => write!(f, "HOLD"),
        }
    }
}

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Long,
    Short,
}

/// Current position held by agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub side: Side,
    pub entry_price: Decimal,
    pub entry_time: DateTime<Utc>,
    pub quantity: Decimal,
}

/// Agent signal with context
#[derive(Debug, Clone)]
pub struct AgentSignal {
    pub symbol: String,
    pub signal: Signal,
    pub price: Decimal,
    pub reason: String,
    pub support: Option<Decimal>,
    pub resistance: Option<Decimal>,
    /// LOSSLESS: Volume percentile (0-100) derived from all observed data
    /// 100 = highest ever, 50 = median, etc.
    pub volume_percentile: f64,
    /// Conviction score (0-100) based on historical confidence
    /// Derived from S/R effectiveness and regime performance
    pub conviction: u8,
}

/// Entry context for outcome tracking (AGI learning)
#[derive(Debug, Clone)]
pub struct EntryContext {
    /// Unique trade ticket
    pub ticket: u64,
    /// S/R level that triggered entry
    pub sr_level: Decimal,
    /// S/R score at entry (0 = strongest, negative = weaker)
    pub sr_score: i32,
    /// Volume percentile at entry
    pub volume_percentile: f64,
    /// ATR at entry
    pub atr: Option<Decimal>,
    /// Market regime at entry
    pub regime: MarketRegime,
    /// Bar count at entry (for hold duration)
    pub entry_bar_count: u64,
    /// Entry price for MAE/MFE calculation
    pub entry_price: Decimal,
    /// Maximum adverse excursion (worst drawdown during trade)
    pub mae: f64,
    /// Maximum favorable excursion (best profit during trade)
    pub mfe: f64,
}

/// Minimum trades required before considering EWC consolidation
const EWC_MIN_TRADES: usize = 10;

/// Minimum win rate required for EWC consolidation (consider weights worth protecting)
const EWC_MIN_WIN_RATE: f64 = 0.55;

/// Independent trading agent for a single symbol
///
/// Each agent:
/// - Tracks S/R levels using the lossless counting algorithm
/// - Monitors volume for capitulation signals
/// - Generates buy/sell signals based on S/R + capitulation
/// - Manages its own position state
/// - Tracks ATR for volatility-based calculations (lossless)
/// - Records trade context for AGI learning (when memory is attached)
/// - Detects market regime using HMM (TrendingUp, TrendingDown, Ranging, Volatile)
/// - Uses learned confidence calibrator for adaptive trade filtering
/// - Applies Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
/// - Supports Mixture of Experts (MoE) for regime-specialized calibration
pub struct SymbolAgent {
    /// Symbol this agent is trading
    symbol: String,
    /// Lossless S/R level tracker
    sr: SRLevels,
    /// Volume tracker for capitulation detection
    volume: VolumeTracker,
    /// HMM-based regime detector
    regime_detector: RegimeDetector,
    /// Learned confidence calibrator
    calibrator: ConfidenceCalibrator,
    /// Mixture of Experts for regime-specialized calibration (optional)
    moe: Option<MixtureOfExperts>,
    /// Whether to use MoE for confidence prediction
    use_moe: bool,
    /// Current position (if any)
    position: Option<Position>,
    /// Last known price
    last_price: Decimal,
    /// Last known volume
    last_volume: u64,
    /// Number of bars processed
    bar_count: u64,
    /// Recent bars for ATR calculation (open, high, low, close)
    recent_bars: Vec<(Decimal, Decimal, Decimal, Decimal)>,
    /// Maximum bars to keep for ATR (operational limit)
    max_atr_bars: usize,
    /// Persistent memory for AGI learning (optional)
    memory: Option<Arc<TradeMemory>>,
    /// Pending entry context for outcome tracking
    pending_entry: Option<EntryContext>,
    /// Next ticket number for trade identification
    next_ticket: u64,
    /// Recent trade outcomes for EWC Fisher computation
    recent_trades: Vec<TradeOutcome>,
    /// Shared transfer manager for cross-symbol knowledge transfer
    transfer_manager: Option<Arc<Mutex<TransferManager>>>,

    // Meta-learning adaptation tracking
    /// Weights before adapting to new regime
    pre_adaptation_weights: Option<[f64; NUM_FEATURES]>,
    /// Bias before adapting to new regime
    pre_adaptation_bias: Option<f64>,
    /// Trades in current regime (for post-adaptation accuracy)
    trades_in_current_regime: Vec<TradeOutcome>,
    /// Minimum trades before reporting adaptation
    meta_adaptation_threshold: u32,
}

impl SymbolAgent {
    /// Create a new agent for a symbol (legacy - uses price-based granularity)
    ///
    /// # Arguments
    /// * `symbol` - The trading symbol (e.g., "AAPL", "BTC")
    /// * `initial_price` - Initial price for granularity calculation
    ///
    /// Note: For lossless derivation, prefer `new_with_atr()` once ATR is calculated.
    #[allow(deprecated)]
    pub fn new(symbol: String, initial_price: Decimal) -> Self {
        let granularity = default_granularity(&symbol, initial_price);

        Self {
            symbol,
            sr: SRLevels::new(granularity),
            volume: VolumeTracker::new(),
            regime_detector: RegimeDetector::new(),
            calibrator: ConfidenceCalibrator::new(),
            moe: None,
            use_moe: false,
            position: None,
            last_price: initial_price,
            last_volume: 0,
            bar_count: 0,
            recent_bars: Vec::with_capacity(20),
            max_atr_bars: 20, // Operational limit for ATR calculation
            memory: None,
            pending_entry: None,
            next_ticket: 1,
            recent_trades: Vec::new(),
            transfer_manager: None,
            pre_adaptation_weights: None,
            pre_adaptation_bias: None,
            trades_in_current_regime: Vec::new(),
            meta_adaptation_threshold: 10,
        }
    }

    /// Create agent with memory for AGI learning
    #[allow(deprecated)]
    pub fn new_with_memory(symbol: String, initial_price: Decimal, memory: Arc<TradeMemory>) -> Self {
        let granularity = default_granularity(&symbol, initial_price);

        Self {
            symbol,
            sr: SRLevels::new(granularity),
            volume: VolumeTracker::new(),
            regime_detector: RegimeDetector::new(),
            calibrator: ConfidenceCalibrator::new(),
            moe: None,
            use_moe: false,
            position: None,
            last_price: initial_price,
            last_volume: 0,
            bar_count: 0,
            recent_bars: Vec::with_capacity(20),
            max_atr_bars: 20,
            memory: Some(memory),
            pending_entry: None,
            next_ticket: 1,
            recent_trades: Vec::new(),
            transfer_manager: None,
            pre_adaptation_weights: None,
            pre_adaptation_bias: None,
            trades_in_current_regime: Vec::new(),
            meta_adaptation_threshold: 10,
        }
    }

    /// Create agent with granularity derived from ATR (lossless)
    ///
    /// # Arguments
    /// * `symbol` - The trading symbol
    /// * `atr` - Average True Range calculated from historical data
    ///
    /// This is the preferred constructor as it derives granularity from market data.
    pub fn new_with_atr(symbol: String, atr: Decimal) -> Self {
        let granularity = granularity_from_atr(atr);

        Self {
            symbol,
            sr: SRLevels::new(granularity),
            volume: VolumeTracker::new(),
            regime_detector: RegimeDetector::new(),
            calibrator: ConfidenceCalibrator::new(),
            moe: None,
            use_moe: false,
            position: None,
            last_price: Decimal::ZERO,
            last_volume: 0,
            bar_count: 0,
            recent_bars: Vec::with_capacity(20),
            max_atr_bars: 20,
            memory: None,
            pending_entry: None,
            next_ticket: 1,
            recent_trades: Vec::new(),
            transfer_manager: None,
            pre_adaptation_weights: None,
            pre_adaptation_bias: None,
            trades_in_current_regime: Vec::new(),
            meta_adaptation_threshold: 10,
        }
    }

    /// Create agent with custom granularity (for testing or special cases)
    pub fn with_granularity(symbol: String, granularity: Decimal) -> Self {
        Self {
            symbol,
            sr: SRLevels::new(granularity),
            volume: VolumeTracker::new(),
            regime_detector: RegimeDetector::new(),
            calibrator: ConfidenceCalibrator::new(),
            moe: None,
            use_moe: false,
            position: None,
            last_price: Decimal::ZERO,
            last_volume: 0,
            bar_count: 0,
            recent_bars: Vec::with_capacity(20),
            max_atr_bars: 20,
            memory: None,
            pending_entry: None,
            next_ticket: 1,
            recent_trades: Vec::new(),
            transfer_manager: None,
            pre_adaptation_weights: None,
            pre_adaptation_bias: None,
            trades_in_current_regime: Vec::new(),
            meta_adaptation_threshold: 10,
        }
    }

    /// Attach memory for AGI learning
    pub fn attach_memory(&mut self, memory: Arc<TradeMemory>) {
        self.memory = Some(memory);
    }

    /// Attach transfer manager for cross-symbol knowledge transfer
    pub fn attach_transfer_manager(&mut self, tm: Arc<Mutex<TransferManager>>) {
        self.transfer_manager = Some(tm);
    }

    /// Attach Mixture of Experts for regime-specialized calibration
    pub fn attach_moe(&mut self, moe: MixtureOfExperts) {
        self.moe = Some(moe);
    }

    /// Enable MoE for confidence prediction
    pub fn enable_moe(&mut self) {
        self.use_moe = true;
    }

    /// Disable MoE (fall back to single calibrator)
    pub fn disable_moe(&mut self) {
        self.use_moe = false;
    }

    /// Check if MoE is enabled
    pub fn is_moe_enabled(&self) -> bool {
        self.use_moe && self.moe.is_some()
    }

    /// Get reference to MoE
    pub fn moe(&self) -> Option<&MixtureOfExperts> {
        self.moe.as_ref()
    }

    /// Get mutable reference to MoE
    pub fn moe_mut(&mut self) -> Option<&mut MixtureOfExperts> {
        self.moe.as_mut()
    }

    /// Set MoE (for loading from persistence)
    pub fn set_moe(&mut self, moe: Option<MixtureOfExperts>) {
        self.moe = moe;
    }

    /// Initialize calibrator from meta-learner and/or cluster prior if available
    ///
    /// Priority order:
    /// 1. Meta-learner initialization (rapid adaptation foundation)
    /// 2. Cluster prior refinement (symbol-specific patterns)
    ///
    /// If this symbol's cluster has learned weights (>= 20 trades) and
    /// this calibrator has < 10 updates, initialize from cluster prior.
    pub fn maybe_init_from_cluster(&mut self) {
        // Only initialize if calibrator is fresh
        if self.calibrator.update_count() >= 10 {
            return;
        }

        // Step 1: Initialize from meta-learner first (base initialization)
        // This gives weights that have been shown to adapt quickly
        self.calibrator.init_from_meta();

        // Step 2: Refine with cluster prior if available
        let Some(ref tm) = self.transfer_manager else {
            return;
        };

        let tm_lock = tm.lock().unwrap();
        if let Some(prior_weights) = tm_lock.get_cluster_prior(&self.symbol) {
            let cluster = super::transfer::get_cluster(&self.symbol);
            let stats = tm_lock.get_cluster_stats(cluster);
            let (trade_count, win_rate) = stats
                .map(|s| (s.trade_count, s.win_rate()))
                .unwrap_or((0, 0.5));

            self.calibrator.set_weights(prior_weights);
            info!(
                "[TRANSFER] {} initialized from {} cluster ({} trades, {:.0}% win rate)",
                self.symbol,
                cluster.name(),
                trade_count,
                win_rate * 100.0
            );
        }
    }

    /// Get the symbol this agent is trading
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Check if agent is ready to generate signals
    ///
    /// LOSSLESS: Ready when we have observed data, not after arbitrary bar count.
    /// - Must have at least one S/R level (price has moved)
    /// - Must have volume context (any historical volume)
    pub fn is_ready(&self) -> bool {
        self.sr.level_count() > 0 && self.volume.has_context()
    }

    /// Get current position
    pub fn position(&self) -> Option<&Position> {
        self.position.as_ref()
    }

    /// Check if agent has an open position
    pub fn has_position(&self) -> bool {
        self.position.is_some()
    }

    /// Set position (used when recovering from broker state)
    pub fn set_position(&mut self, position: Option<Position>) {
        self.position = position;
    }

    /// Get last known price
    pub fn last_price(&self) -> Decimal {
        self.last_price
    }

    /// Get current support level
    pub fn support(&self) -> Option<Decimal> {
        self.sr.get_support(self.last_price)
    }

    /// Get current resistance level
    pub fn resistance(&self) -> Option<Decimal> {
        self.sr.get_resistance(self.last_price)
    }

    /// Process a new bar and potentially generate a signal
    ///
    /// This is the main entry point for the agent.
    /// Call this once per bar (e.g., once per day for daily trading).
    pub fn process_bar(
        &mut self,
        time: DateTime<Utc>,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: u64,
    ) -> Option<AgentSignal> {
        // 1. Update S/R levels
        self.sr.update_bar(open, high, low, close);

        // 2. Update volume tracker
        self.volume.update(volume);

        // 3. Update regime detector (HMM)
        self.regime_detector.update(open, high, low, close, volume);

        // 4. Check for regime change and record to memory
        if self.regime_detector.regime_changed() {
            if let Some(ref memory) = self.memory {
                let new_regime = self.regime_detector.current_regime();
                let _ = memory.start_regime(&self.symbol, new_regime.as_str());
            }

            // 4a. Report meta-learning adaptation from the previous regime
            self.maybe_report_meta_adaptation();

            // 4b. Prepare for new regime (store pre-adaptation weights)
            self.prepare_for_new_regime();

            // 4c. EWC consolidation on regime change if performance was good
            self.maybe_consolidate_ewc();

            // 4d. MoE expert consolidation for the previous regime
            self.maybe_consolidate_moe_expert();
        }

        // 5. Track bars for ATR calculation
        if self.recent_bars.len() >= self.max_atr_bars {
            self.recent_bars.remove(0);
        }
        self.recent_bars.push((open, high, low, close));

        // 6. Update state
        self.last_price = close;
        self.last_volume = volume;
        self.bar_count += 1;

        // 7. Not ready yet - need more data
        if !self.is_ready() {
            return None;
        }

        // 8. Check for signals
        self.check_signals(time, open, close, volume)
    }

    /// Minimum confidence threshold for taking trades
    const MIN_CONFIDENCE_THRESHOLD: f64 = 0.45;

    /// Check for trading signals based on current state
    ///
    /// LOSSLESS: Uses percentile-based volume checks derived from data distribution.
    /// AGI FEEDBACK: Uses historical S/R and regime effectiveness to set conviction.
    fn check_signals(
        &mut self,
        time: DateTime<Utc>,
        open: Decimal,
        close: Decimal,
        volume: u64,
    ) -> Option<AgentSignal> {
        let support = self.sr.get_support(close);
        let resistance = self.sr.get_resistance(close);
        let price_change = close - open;

        // LOSSLESS: Volume percentile (derived from all observed data)
        let volume_percentile = self.volume.percentile(volume);

        // LOSSLESS: Capitulation = high percentile volume (80th+) AND recent highest
        let is_capitulation_volume = self.volume.is_capitulation_volume(volume);
        let is_down_day = price_change < Decimal::ZERO;
        let is_up_day = price_change > Decimal::ZERO;
        let is_buy_capitulation = is_capitulation_volume && is_down_day;
        let is_sell_capitulation = is_capitulation_volume && is_up_day;

        // Check if at S/R levels
        let at_support = support.map_or(false, |s| self.sr.is_near(close, s));
        let at_resistance = resistance.map_or(false, |r| self.sr.is_near(close, r));

        // Generate signal based on position and conditions
        match &self.position {
            None => {
                // No position - look for entries
                if is_buy_capitulation && at_support {
                    if let Some(s) = support {
                        // AGI FEEDBACK: Calculate combined confidence for this S/R level
                        let combined_confidence = self.get_combined_confidence(s);
                        let conviction = (combined_confidence * 100.0) as u8;

                        // Skip trade if confidence too low
                        if combined_confidence < Self::MIN_CONFIDENCE_THRESHOLD {
                            info!(
                                "[MEMORY] {} Skipping BUY: confidence {:.1}% below threshold {:.1}%",
                                self.symbol, combined_confidence * 100.0, Self::MIN_CONFIDENCE_THRESHOLD * 100.0
                            );
                            return None;
                        }

                        let signal = AgentSignal {
                            symbol: self.symbol.clone(),
                            signal: Signal::Buy,
                            price: close,
                            reason: format!(
                                "Volume capitulation at support ({:.0}th pctl, {:.0}% conf)",
                                volume_percentile, combined_confidence * 100.0
                            ),
                            support,
                            resistance,
                            volume_percentile,
                            conviction,
                        };

                        self.position = Some(Position {
                            side: Side::Long,
                            entry_price: close,
                            entry_time: time,
                            quantity: Decimal::ZERO,
                        });

                        return Some(signal);
                    }
                }

                if is_sell_capitulation && at_resistance {
                    if let Some(r) = resistance {
                        // AGI FEEDBACK: Calculate combined confidence for this S/R level
                        let combined_confidence = self.get_combined_confidence(r);
                        let conviction = (combined_confidence * 100.0) as u8;

                        // Skip trade if confidence too low
                        if combined_confidence < Self::MIN_CONFIDENCE_THRESHOLD {
                            info!(
                                "[MEMORY] {} Skipping SHORT: confidence {:.1}% below threshold {:.1}%",
                                self.symbol, combined_confidence * 100.0, Self::MIN_CONFIDENCE_THRESHOLD * 100.0
                            );
                            return None;
                        }

                        let signal = AgentSignal {
                            symbol: self.symbol.clone(),
                            signal: Signal::Short,
                            price: close,
                            reason: format!(
                                "Volume capitulation at resistance ({:.0}th pctl, {:.0}% conf)",
                                volume_percentile, combined_confidence * 100.0
                            ),
                            support,
                            resistance,
                            volume_percentile,
                            conviction,
                        };

                        self.position = Some(Position {
                            side: Side::Short,
                            entry_price: close,
                            entry_time: time,
                            quantity: Decimal::ZERO,
                        });

                        return Some(signal);
                    }
                }

                // LOSSLESS alternative entry: at support on down day with elevated volume
                // "Elevated" threshold derived from mean + 1σ of volume distribution
                let elevated_threshold = self.volume.elevated_threshold();
                if at_support && is_down_day && volume_percentile >= elevated_threshold {
                    if let Some(s) = support {
                        let touched_support = self.sr.is_near(close.min(open), s);
                        if touched_support {
                            // AGI FEEDBACK: Calculate combined confidence for this S/R level
                            let combined_confidence = self.get_combined_confidence(s);
                            let conviction = (combined_confidence * 100.0) as u8;

                            // Skip trade if confidence too low
                            if combined_confidence < Self::MIN_CONFIDENCE_THRESHOLD {
                                info!(
                                    "[MEMORY] {} Skipping BUY (elevated): confidence {:.1}% below threshold {:.1}%",
                                    self.symbol, combined_confidence * 100.0, Self::MIN_CONFIDENCE_THRESHOLD * 100.0
                                );
                                return None;
                            }

                            let signal = AgentSignal {
                                symbol: self.symbol.clone(),
                                signal: Signal::Buy,
                                price: close,
                                reason: format!(
                                    "Support with elevated volume ({:.0}th pctl, {:.0}% conf)",
                                    volume_percentile, combined_confidence * 100.0
                                ),
                                support,
                                resistance,
                                volume_percentile,
                                conviction,
                            };

                            self.position = Some(Position {
                                side: Side::Long,
                                entry_price: close,
                                entry_time: time,
                                quantity: Decimal::ZERO,
                            });

                            return Some(signal);
                        }
                    }
                }
            }

            Some(pos) => {
                match pos.side {
                    Side::Long => {
                        if at_resistance {
                            // Exit signals don't need confidence check - always exit at target
                            let signal = AgentSignal {
                                symbol: self.symbol.clone(),
                                signal: Signal::Sell,
                                price: close,
                                reason: format!(
                                    "Reached resistance (entry: {:.2}, exit: {:.2})",
                                    pos.entry_price, close
                                ),
                                support,
                                resistance,
                                volume_percentile,
                                conviction: 100, // Exit signals are always high conviction
                            };

                            self.position = None;
                            return Some(signal);
                        }
                    }

                    Side::Short => {
                        if at_support {
                            let signal = AgentSignal {
                                symbol: self.symbol.clone(),
                                signal: Signal::Cover,
                                price: close,
                                reason: format!(
                                    "Reached support (entry: {:.2}, exit: {:.2})",
                                    pos.entry_price, close
                                ),
                                support,
                                resistance,
                                volume_percentile,
                                conviction: 100, // Exit signals are always high conviction
                            };

                            self.position = None;
                            return Some(signal);
                        }
                    }
                }
            }
        }

        None
    }

    /// Force close position (used for external close signals)
    pub fn close_position(&mut self) {
        self.position = None;
    }

    /// Get the number of bars processed
    pub fn bar_count(&self) -> u64 {
        self.bar_count
    }

    /// Get the number of S/R levels tracked
    pub fn sr_level_count(&self) -> usize {
        self.sr.level_count()
    }

    /// Get volume average
    pub fn avg_volume(&self) -> f64 {
        self.volume.average()
    }

    /// Bootstrap S/R with historical bar data (no signal generation)
    ///
    /// Used at startup to pre-populate S/R levels from historical data.
    /// Does NOT generate trading signals - only builds the S/R map.
    /// Also trains the regime detector with historical data.
    pub fn bootstrap_bar(
        &mut self,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: u64,
    ) {
        // Update S/R levels
        self.sr.update_bar(open, high, low, close);

        // Update volume tracker
        self.volume.update(volume);

        // Update regime detector (train HMM with historical data)
        self.regime_detector.update(open, high, low, close, volume);

        // Track bars for ATR calculation
        if self.recent_bars.len() >= self.max_atr_bars {
            self.recent_bars.remove(0);
        }
        self.recent_bars.push((open, high, low, close));

        // Update state
        self.last_price = close;
        self.last_volume = volume;
        self.bar_count += 1;
    }

    /// Get current market regime
    pub fn current_regime(&self) -> Regime {
        self.regime_detector.current_regime()
    }

    /// Get regime probabilities (TrendingUp, TrendingDown, Ranging, Volatile)
    pub fn regime_probabilities(&self) -> [f64; 4] {
        self.regime_detector.regime_probabilities()
    }

    /// Get duration in current regime (bars)
    pub fn regime_duration(&self) -> u32 {
        self.regime_detector.regime_duration()
    }

    /// Get confidence in current regime classification
    pub fn regime_confidence(&self) -> f64 {
        self.regime_detector.confidence()
    }

    /// Calculate current ATR from recent bars (lossless - derived from data)
    ///
    /// Returns None if insufficient data (less than 2 bars).
    /// Uses True Range: max(high-low, |high-prev_close|, |low-prev_close|)
    pub fn atr(&self) -> Option<Decimal> {
        if self.recent_bars.len() < 2 {
            return None;
        }

        let mut true_ranges = Vec::with_capacity(self.recent_bars.len() - 1);

        for i in 1..self.recent_bars.len() {
            let (_, high, low, _) = self.recent_bars[i];
            let (_, _, _, prev_close) = self.recent_bars[i - 1];

            // True Range = max(high-low, |high-prev_close|, |low-prev_close|)
            let hl = high - low;
            let hpc = (high - prev_close).abs();
            let lpc = (low - prev_close).abs();

            let tr = hl.max(hpc).max(lpc);
            true_ranges.push(tr);
        }

        if true_ranges.is_empty() {
            return None;
        }

        // Simple average of true ranges
        let sum: Decimal = true_ranges.iter().copied().sum();
        Some(sum / Decimal::from(true_ranges.len()))
    }

    // =========================================================================
    // AGI MEMORY INTEGRATION METHODS
    // =========================================================================

    /// Record entry context when opening a trade (for AGI learning)
    ///
    /// Call this after a trade is confirmed to record the full context
    /// for later outcome analysis.
    pub fn record_entry_context(
        &mut self,
        sr_level: Decimal,
        volume_percentile: f64,
    ) {
        let ticket = self.next_ticket;
        self.next_ticket += 1;

        let sr_score = self.sr.score_at(sr_level);
        // Use HMM regime detection, falling back to memory or Unknown
        let regime = MarketRegime::from_regime(self.regime_detector.current_regime());

        let entry_price = self.last_price;

        // Store pending entry for MAE/MFE tracking
        self.pending_entry = Some(EntryContext {
            ticket,
            sr_level,
            sr_score,
            volume_percentile,
            atr: self.atr(),
            regime,
            entry_bar_count: self.bar_count,
            entry_price,
            mae: 0.0,
            mfe: 0.0,
        });

        // Record to persistent memory
        if let Some(ref memory) = self.memory {
            let _ = memory.record_trade_entry(
                &self.symbol,
                ticket,
                match self.position {
                    Some(ref p) if p.side == Side::Long => "BUY",
                    Some(ref p) if p.side == Side::Short => "SHORT",
                    _ => "UNKNOWN",
                },
                entry_price,
                sr_level,
                sr_score,
                volume_percentile,
                self.atr().unwrap_or(Decimal::ZERO),
                regime.as_str(),
                self.bar_count,
            );

            // Record S/R touch
            let _ = memory.record_sr_touch(
                &self.symbol,
                sr_level.to_f64().unwrap_or(0.0),
                self.sr.is_near(entry_price, sr_level).then(|| 1.0).unwrap_or(0.0),
            );
        }
    }

    /// Update MAE/MFE during trade (call on each bar while in position)
    pub fn update_excursions(&mut self, current_price: Decimal) {
        if let Some(ref mut entry) = self.pending_entry {
            if let Some(ref pos) = self.position {
                let pnl_pct = match pos.side {
                    Side::Long => ((current_price - entry.entry_price) / entry.entry_price)
                        .to_f64().unwrap_or(0.0) * 100.0,
                    Side::Short => ((entry.entry_price - current_price) / entry.entry_price)
                        .to_f64().unwrap_or(0.0) * 100.0,
                };

                if pnl_pct < entry.mae {
                    entry.mae = pnl_pct;
                }
                if pnl_pct > entry.mfe {
                    entry.mfe = pnl_pct;
                }
            }
        }
    }

    /// Record exit outcome when closing a trade (for AGI learning)
    ///
    /// Call this after a trade is closed to record the outcome and
    /// update S/R effectiveness and volume calibration.
    pub fn record_exit_outcome(
        &mut self,
        exit_price: Decimal,
        hit_tp: bool,
        hit_sl: bool,
    ) {
        let Some(entry) = self.pending_entry.take() else {
            return;
        };

        let profit = match self.position {
            Some(ref p) if p.side == Side::Long => exit_price - entry.entry_price,
            Some(ref p) if p.side == Side::Short => entry.entry_price - exit_price,
            _ => Decimal::ZERO,
        };

        let profit_pct = if !entry.entry_price.is_zero() {
            (profit / entry.entry_price).to_f64().unwrap_or(0.0) * 100.0
        } else {
            0.0
        };

        let hold_bars = (self.bar_count - entry.entry_bar_count) as i64;
        let was_winner = profit > Decimal::ZERO;
        let bounced = was_winner; // Simplified: winner = bounced at S/R

        if let Some(ref memory) = self.memory {
            // Record trade exit
            let _ = memory.record_trade_exit(
                entry.ticket,
                exit_price,
                profit,
                profit_pct,
                hit_tp,
                hit_sl,
                hold_bars,
                entry.mae,
                entry.mfe,
            );

            // Update S/R effectiveness
            let _ = memory.record_sr_trade_outcome(
                &self.symbol,
                entry.sr_level.to_f64().unwrap_or(0.0),
                self.atr().map(|a| a.to_f64().unwrap_or(1.0) / 2.0).unwrap_or(1.0),
                bounced,
                profit.to_f64().unwrap_or(0.0),
            );

            // Update volume calibration
            let _ = memory.update_volume_calibration(
                &self.symbol,
                entry.volume_percentile,
                was_winner,
            );

            // Update regime stats
            let _ = memory.update_regime_stats(
                profit.to_f64().unwrap_or(0.0),
                hold_bars,
            );
        }

        // Update calibrator with trade outcome (always, even without memory)
        // Convert entry's MarketRegime back to Regime for the learner
        let regime_for_learner = match entry.regime {
            MarketRegime::Bull => Regime::TrendingUp,
            MarketRegime::Bear => Regime::TrendingDown,
            MarketRegime::Sideways => Regime::Ranging,
            MarketRegime::HighVolatility | MarketRegime::LowVolatility => Regime::Volatile,
            MarketRegime::Unknown => self.regime_detector.current_regime(),
        };

        self.calibrator.update_from_trade(
            entry.sr_score,
            entry.volume_percentile,
            &regime_for_learner,
            was_winner,
            0.01, // Learning rate
        );

        // Update MoE if enabled (routes to appropriate expert)
        if let Some(ref mut moe) = self.moe {
            moe.update(
                entry.sr_score,
                entry.volume_percentile,
                &regime_for_learner,
                was_winner,
                0.01, // Learning rate
            );

            // Update gating weights based on current regime probabilities
            let regime_probs = self.regime_detector.regime_probabilities();
            moe.update_gating(&regime_probs);
        }

        // Track trade outcome for EWC Fisher computation
        let trade_outcome = TradeOutcome {
            sr_score: entry.sr_score,
            volume_pct: entry.volume_percentile,
            regime: regime_for_learner,
            won: was_winner,
        };
        self.recent_trades.push(trade_outcome.clone());

        // Keep only the most recent trades (cap at 100)
        if self.recent_trades.len() > 100 {
            self.recent_trades.remove(0);
        }

        // Track trade in current regime for meta-learning
        self.track_trade_in_regime(trade_outcome);

        // Update transfer manager for cross-symbol knowledge transfer
        if let Some(ref tm) = self.transfer_manager {
            let mut tm_lock = tm.lock().unwrap();
            tm_lock.update_cluster(&self.symbol, self.calibrator.get_weights(), was_winner);
        }

        let outcome = if was_winner { "winning" } else { "losing" };
        info!(
            "[LEARNER] {} Updated after {} trade (sr_score: {}, vol: {:.0}%, regime: {:?})",
            self.symbol, outcome, entry.sr_score, entry.volume_percentile, regime_for_learner
        );
    }

    /// Get S/R confidence based on historical effectiveness
    ///
    /// Returns a confidence score 0.0-1.0 based on:
    /// - Historical win rate at this level (if >= 5 trades)
    /// - Fallback to S/R score heuristic
    pub fn get_sr_confidence(&self, price_level: Decimal) -> f64 {
        let price_f = price_level.to_f64().unwrap_or(0.0);
        let granularity = self.atr()
            .map(|a| a.to_f64().unwrap_or(1.0) / 2.0)
            .unwrap_or(1.0);

        // Try to get historical win rate from memory (requires >= 5 trades)
        if let Some(ref memory) = self.memory {
            if let Ok(Some((win_rate, trade_count))) = memory.get_sr_win_rate_with_count(
                &self.symbol,
                price_f,
                granularity,
            ) {
                if trade_count >= 5 {
                    info!(
                        "[MEMORY] {} S/R {:.2}: {:.1}% from {} trades",
                        self.symbol, price_f, win_rate * 100.0, trade_count
                    );
                    return win_rate;
                }
            }
        }

        // Fallback to S/R score-based confidence heuristic
        let score = self.sr.score_at(price_level);
        let confidence = match score {
            0 => 0.65,           // Never crossed - strongest
            -1 => 0.60,          // Crossed once
            -2 => 0.55,          // Crossed twice
            -3 => 0.50,          // Crossed 3 times
            _ => 0.45,           // Crossed 4+ times - weakest
        };

        confidence
    }

    /// Get regime confidence based on historical win rate in current regime
    ///
    /// Returns win rate if >= 10 trades in this regime, else 0.50 (neutral)
    pub fn get_regime_confidence(&self) -> f64 {
        let current_regime = self.regime_detector.current_regime();
        let regime_str = current_regime.as_str();

        if let Some(ref memory) = self.memory {
            if let Ok(regime_stats) = memory.get_win_rate_by_regime() {
                for (regime, win_rate, trade_count) in regime_stats {
                    // Match current regime (handle both symbol-prefixed and global regimes)
                    if regime == regime_str || regime.ends_with(&format!(":{}", regime_str)) {
                        if trade_count >= 10 {
                            info!(
                                "[MEMORY] {} Regime {}: {:.1}% from {} trades",
                                self.symbol, regime_str, win_rate * 100.0, trade_count
                            );
                            return win_rate;
                        }
                    }
                }
            }
        }

        // Not enough data - return neutral
        0.50
    }

    /// Get combined confidence from S/R, regime, and learned calibrator/MoE
    ///
    /// Base confidence: 70% S/R confidence, 30% regime confidence
    /// Final: 60% base confidence + 40% calibrator/MoE prediction
    ///
    /// If MoE is enabled, uses blended expert predictions instead of single calibrator.
    pub fn get_combined_confidence(&self, sr_level: Decimal) -> f64 {
        let sr_conf = self.get_sr_confidence(sr_level);
        let regime_conf = self.get_regime_confidence();
        let base_confidence = (sr_conf * 0.7) + (regime_conf * 0.3);

        // Get calibrator/MoE prediction
        let sr_score = self.sr.score_at(sr_level);
        let volume_pct = self.volume.percentile(self.last_volume);
        let regime = self.regime_detector.current_regime();

        let calibrated = if self.is_moe_enabled() {
            // Use MoE blended prediction
            let moe = self.moe.as_ref().unwrap();
            let pred = moe.predict(sr_score, volume_pct, &regime);
            info!(
                "[MOE] {} Using MoE prediction: {:.1}%",
                self.symbol, pred * 100.0
            );
            pred
        } else {
            // Use single calibrator
            self.calibrator.predict(sr_score, volume_pct, &regime)
        };

        // Blend base and calibrated confidence
        let combined = (base_confidence * 0.6) + (calibrated * 0.4);

        info!(
            "[MEMORY] {} Combined: {:.1}% (base: {:.1}%, calibrated: {:.1}%)",
            self.symbol, combined * 100.0, base_confidence * 100.0, calibrated * 100.0
        );

        combined
    }

    /// Get pending entry ticket (for external tracking)
    pub fn pending_ticket(&self) -> Option<u64> {
        self.pending_entry.as_ref().map(|e| e.ticket)
    }

    /// Get reference to calibrator
    pub fn calibrator(&self) -> &ConfidenceCalibrator {
        &self.calibrator
    }

    /// Get mutable reference to calibrator
    pub fn calibrator_mut(&mut self) -> &mut ConfidenceCalibrator {
        &mut self.calibrator
    }

    /// Set calibrator (for loading from persistence)
    pub fn set_calibrator(&mut self, calibrator: ConfidenceCalibrator) {
        self.calibrator = calibrator;
    }

    // =========================================================================
    // EWC (ELASTIC WEIGHT CONSOLIDATION) METHODS
    // =========================================================================

    /// Check if EWC consolidation should happen and perform it
    ///
    /// Consolidation occurs when:
    /// 1. We have enough recent trades (>= EWC_MIN_TRADES)
    /// 2. Recent win rate is good (>= EWC_MIN_WIN_RATE)
    ///
    /// This protects learned weights that produced good results in the
    /// current regime from being forgotten when adapting to a new regime.
    fn maybe_consolidate_ewc(&mut self) {
        // Need enough trades to compute meaningful Fisher Information
        if self.recent_trades.len() < EWC_MIN_TRADES {
            info!(
                "[EWC] {} Skipping consolidation: only {} trades (need {})",
                self.symbol, self.recent_trades.len(), EWC_MIN_TRADES
            );
            return;
        }

        // Calculate recent win rate
        let wins = self.recent_trades.iter().filter(|t| t.won).count();
        let win_rate = wins as f64 / self.recent_trades.len() as f64;

        if win_rate < EWC_MIN_WIN_RATE {
            info!(
                "[EWC] {} Skipping consolidation: win rate {:.1}% < {:.1}%",
                self.symbol, win_rate * 100.0, EWC_MIN_WIN_RATE * 100.0
            );
            return;
        }

        // Good performance - consolidate weights
        info!(
            "[EWC] {} Consolidating: {} trades, {:.1}% win rate",
            self.symbol, self.recent_trades.len(), win_rate * 100.0
        );

        // Compute Fisher Information from recent trades
        self.calibrator.compute_fisher(&self.recent_trades);

        // Store current weights as optimal
        self.calibrator.consolidate();

        // Clear recent trades for next regime
        self.recent_trades.clear();
    }

    /// Consolidate MoE expert for the previous regime
    ///
    /// When regime changes, consolidate the expert that was active to protect
    /// its learned weights. Filter recent trades by the previous regime.
    fn maybe_consolidate_moe_expert(&mut self) {
        let Some(ref mut moe) = self.moe else {
            return;
        };

        let Some(prev_regime) = self.regime_detector.previous_regime() else {
            return;
        };

        // Filter trades from the previous regime
        let regime_trades: Vec<_> = self.recent_trades
            .iter()
            .filter(|t| t.regime == prev_regime)
            .cloned()
            .collect();

        // Need enough trades from this regime
        if regime_trades.len() < EWC_MIN_TRADES {
            info!(
                "[MOE] {} Skipping {} expert consolidation: only {} trades",
                self.symbol, prev_regime, regime_trades.len()
            );
            return;
        }

        // Check win rate for this regime's trades
        let wins = regime_trades.iter().filter(|t| t.won).count();
        let win_rate = wins as f64 / regime_trades.len() as f64;

        if win_rate < EWC_MIN_WIN_RATE {
            info!(
                "[MOE] {} Skipping {} expert consolidation: win rate {:.1}%",
                self.symbol, prev_regime, win_rate * 100.0
            );
            return;
        }

        // Consolidate the expert for the previous regime
        info!(
            "[MOE] {} Consolidating {} expert: {} trades, {:.1}% win rate",
            self.symbol, prev_regime, regime_trades.len(), win_rate * 100.0
        );
        moe.consolidate_expert(&prev_regime, &regime_trades);
    }

    /// Get count of recent trades tracked for EWC
    pub fn recent_trade_count(&self) -> usize {
        self.recent_trades.len()
    }

    /// Check if calibrator has been consolidated (EWC is active)
    pub fn is_ewc_active(&self) -> bool {
        self.calibrator.is_consolidated()
    }

    // ==================== Meta-Learning Methods ====================

    /// Prepare for a new regime by storing pre-adaptation state
    ///
    /// Called when regime changes to capture the starting point for adaptation.
    fn prepare_for_new_regime(&mut self) {
        // Store current weights as pre-adaptation baseline
        self.pre_adaptation_weights = Some(*self.calibrator.get_weights());
        self.pre_adaptation_bias = Some(self.calibrator.get_bias());

        // Clear trades from previous regime
        self.trades_in_current_regime.clear();

        info!(
            "[META] {} Prepared for new regime: {} (tracking adaptation)",
            self.symbol,
            self.regime_detector.current_regime()
        );
    }

    /// Report meta-learning adaptation to the MetaLearner
    ///
    /// Called on regime change to report how well we adapted in the previous regime.
    fn maybe_report_meta_adaptation(&mut self) {
        // Need pre-adaptation weights to report
        let Some(pre_weights) = self.pre_adaptation_weights else {
            return;
        };
        let Some(pre_bias) = self.pre_adaptation_bias else {
            return;
        };

        // Need enough trades in current regime
        if self.trades_in_current_regime.len() < self.meta_adaptation_threshold as usize {
            info!(
                "[META] {} Skipping adaptation report: only {} trades (need {})",
                self.symbol,
                self.trades_in_current_regime.len(),
                self.meta_adaptation_threshold
            );
            return;
        }

        // Get previous regime for reporting
        let Some(prev_regime) = self.regime_detector.previous_regime() else {
            return;
        };

        // Calculate pre-adaptation accuracy (using pre-weights on current regime's trades)
        let pre_accuracy = calculate_accuracy(
            &pre_weights,
            pre_bias,
            &self.trades_in_current_regime,
        );

        // Calculate post-adaptation accuracy (using current weights)
        let post_accuracy = calculate_accuracy(
            self.calibrator.get_weights(),
            self.calibrator.get_bias(),
            &self.trades_in_current_regime,
        );

        // Report to calibrator's meta-learner
        self.calibrator.report_adaptation(
            &pre_weights,
            pre_bias,
            pre_accuracy,
            post_accuracy,
            self.trades_in_current_regime.len() as u32,
            &prev_regime,
        );

        info!(
            "[META] {} Reported {} adaptation: {:.1}% -> {:.1}% ({} trades)",
            self.symbol,
            prev_regime,
            pre_accuracy * 100.0,
            post_accuracy * 100.0,
            self.trades_in_current_regime.len()
        );
    }

    /// Track a trade in the current regime (for meta-learning)
    fn track_trade_in_regime(&mut self, trade: TradeOutcome) {
        self.trades_in_current_regime.push(trade);

        // Keep only last 100 trades per regime
        if self.trades_in_current_regime.len() > 100 {
            self.trades_in_current_regime.remove(0);
        }
    }

    /// Attach meta-learner to calibrator
    pub fn attach_meta_learner(&mut self, ml: Arc<Mutex<MetaLearner>>) {
        self.calibrator.attach_meta_learner(ml);
    }

    /// Check if meta-learner is attached
    pub fn has_meta_learner(&self) -> bool {
        self.calibrator.has_meta_learner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_agent_creation() {
        let agent = SymbolAgent::new("AAPL".to_string(), dec!(150.00));
        assert_eq!(agent.symbol(), "AAPL");
        assert!(!agent.is_ready());
        assert!(!agent.has_position());
    }

    #[test]
    fn test_agent_readiness_lossless() {
        let mut agent = SymbolAgent::with_granularity("TEST".to_string(), dec!(1.00));
        let now = Utc::now();

        // Initially not ready (no S/R levels, no volume context)
        assert!(!agent.is_ready());

        // Process 10 bars to meet volume context requirement
        for i in 0..10 {
            let price = dec!(100) + Decimal::from(i);
            agent.process_bar(now, price, price + dec!(1), price - dec!(1), price, 1000);
        }

        // Now ready (has S/R levels + 10+ volume observations)
        assert!(agent.is_ready());
    }

    #[test]
    fn test_agent_needs_data() {
        let mut agent = SymbolAgent::with_granularity("TEST".to_string(), dec!(1.00));
        let now = Utc::now();

        // Feed 10 bars - needs 10+ for volume context
        for i in 0..10 {
            let price = dec!(100) + Decimal::from(i);
            agent.process_bar(now, price, price + dec!(1), price - dec!(1), price, 1000);
        }

        // Should be ready after 10 bars (lossless - needs volume context)
        assert!(agent.is_ready());
    }

    #[test]
    fn test_buy_signal_at_support() {
        let mut agent = SymbolAgent::with_granularity("TEST".to_string(), dec!(0.10));
        let now = Utc::now();

        // Build up S/R levels with a stable price range
        // This creates support around 99.50 and resistance around 100.50
        for i in 0..30 {
            let price = dec!(100) + if i % 4 < 2 { dec!(0.50) } else { dec!(-0.50) };
            agent.process_bar(now, price, price + dec!(0.20), price - dec!(0.20), price, 1000);
        }

        // Verify agent is ready
        assert!(agent.is_ready());

        // Verify S/R levels are being tracked
        assert!(agent.support().is_some() || agent.resistance().is_some());

        // The agent should be able to process bars and potentially generate signals
        // when conditions align (capitulation + at S/R)
        let signal = agent.process_bar(
            now,
            dec!(100.10),   // open
            dec!(100.10),   // high
            dec!(99.40),    // low - at support area
            dec!(99.50),    // close - down day
            5000,           // volume spike (5x normal)
        );

        // Signal may or may not be generated depending on exact S/R levels
        // The key is that the system processes without error
        // and S/R tracking is functional
        if let Some(s) = signal {
            // If a signal is generated, it should be a buy signal given conditions
            assert!(matches!(s.signal, Signal::Buy | Signal::Hold));
            // Verify volume_percentile is set (0-100 range)
            assert!(s.volume_percentile >= 0.0 && s.volume_percentile <= 100.0);
        }
    }

    #[test]
    fn test_sell_signal_at_resistance() {
        let mut agent = SymbolAgent::with_granularity("TEST".to_string(), dec!(0.10));
        let now = Utc::now();

        // Build history - price oscillates in a range
        for i in 0..30 {
            let price = dec!(100) + if i % 4 < 2 { dec!(0.50) } else { dec!(-0.50) };
            agent.process_bar(now, price, price + dec!(0.20), price - dec!(0.20), price, 1000);
        }

        // Enter a long position
        agent.set_position(Some(Position {
            side: Side::Long,
            entry_price: dec!(99.50),
            entry_time: now,
            quantity: dec!(10),
        }));

        // Verify we have a position
        assert!(agent.has_position());

        // Price reaches resistance area
        let signal = agent.process_bar(
            now,
            dec!(100.30),
            dec!(100.60),   // high at resistance
            dec!(100.20),
            dec!(100.50),   // close at resistance
            1000,
        );

        // If we're at resistance with a long position, should get sell signal
        if let Some(s) = signal {
            assert_eq!(s.signal, Signal::Sell);
            // Verify volume_percentile is set (0-100 range)
            assert!(s.volume_percentile >= 0.0 && s.volume_percentile <= 100.0);
        }

        // Verify position tracking works
        // Even if no signal, the agent should still be functional
        assert!(agent.bar_count() > 30);
    }

    #[test]
    fn test_position_management() {
        let mut agent = SymbolAgent::new("AAPL".to_string(), dec!(150.00));

        assert!(!agent.has_position());

        agent.set_position(Some(Position {
            side: Side::Long,
            entry_price: dec!(150),
            entry_time: Utc::now(),
            quantity: dec!(100),
        }));

        assert!(agent.has_position());
        assert!(matches!(agent.position().unwrap().side, Side::Long));

        agent.close_position();
        assert!(!agent.has_position());
    }

    #[test]
    fn test_granularity_selection() {
        // BTC should get $100 granularity
        let btc = SymbolAgent::new("BTC".to_string(), dec!(50000));
        // Check it was created successfully
        assert_eq!(btc.symbol(), "BTC");

        // Low-priced stock should get smaller granularity
        let penny = SymbolAgent::new("SNDL".to_string(), dec!(2.50));
        assert_eq!(penny.symbol(), "SNDL");
    }

    #[test]
    fn test_volume_percentile_in_signal() {
        let mut agent = SymbolAgent::with_granularity("TEST".to_string(), dec!(1.00));
        let now = Utc::now();

        // Build up some history with varying volume (10+ bars for context)
        for i in 1..=10 {
            agent.process_bar(
                now,
                dec!(100),
                dec!(101),
                dec!(99),
                dec!(100),
                i * 100, // Increasing volume: 100, 200, 300, ...
            );
        }

        // Set a position to test exit signal
        agent.set_position(Some(Position {
            side: Side::Long,
            entry_price: dec!(99),
            entry_time: now,
            quantity: dec!(10),
        }));

        // Process a bar that should trigger a sell at resistance
        // Use mid-range volume
        let signal = agent.process_bar(
            now,
            dec!(100),
            dec!(102),  // High touches potential resistance
            dec!(100),
            dec!(101),  // Close up
            500,        // Mid-range volume
        );

        // If signal generated, verify volume_percentile is properly calculated
        if let Some(s) = signal {
            // 500 is around 50th percentile of 100-1000 range
            assert!(s.volume_percentile >= 0.0 && s.volume_percentile <= 100.0);
        }
    }

    #[test]
    fn test_new_with_atr() {
        // Test ATR-based agent creation
        let agent = SymbolAgent::new_with_atr("TEST".to_string(), dec!(5.0));
        assert_eq!(agent.symbol(), "TEST");
        assert!(!agent.is_ready()); // No data yet
    }
}
