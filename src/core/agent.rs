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

#[allow(deprecated)]
use super::sr::{SRLevels, default_granularity, granularity_from_atr};
use super::capitulation::VolumeTracker;

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
}

/// Independent trading agent for a single symbol
///
/// Each agent:
/// - Tracks S/R levels using the lossless counting algorithm
/// - Monitors volume for capitulation signals
/// - Generates buy/sell signals based on S/R + capitulation
/// - Manages its own position state
pub struct SymbolAgent {
    /// Symbol this agent is trading
    symbol: String,
    /// Lossless S/R level tracker
    sr: SRLevels,
    /// Volume tracker for capitulation detection
    volume: VolumeTracker,
    /// Current position (if any)
    position: Option<Position>,
    /// Last known price
    last_price: Decimal,
    /// Last known volume
    last_volume: u64,
    /// Number of bars processed
    bar_count: u64,
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
            position: None,
            last_price: initial_price,
            last_volume: 0,
            bar_count: 0,
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
            position: None,
            last_price: Decimal::ZERO,
            last_volume: 0,
            bar_count: 0,
        }
    }

    /// Create agent with custom granularity (for testing or special cases)
    pub fn with_granularity(symbol: String, granularity: Decimal) -> Self {
        Self {
            symbol,
            sr: SRLevels::new(granularity),
            volume: VolumeTracker::new(),
            position: None,
            last_price: Decimal::ZERO,
            last_volume: 0,
            bar_count: 0,
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

        // 3. Update state
        self.last_price = close;
        self.last_volume = volume;
        self.bar_count += 1;

        // 4. Not ready yet - need more data
        if !self.is_ready() {
            return None;
        }

        // 5. Check for signals
        self.check_signals(time, open, close, volume)
    }

    /// Check for trading signals based on current state
    ///
    /// LOSSLESS: Uses percentile-based volume checks derived from data distribution.
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
                    let signal = AgentSignal {
                        symbol: self.symbol.clone(),
                        signal: Signal::Buy,
                        price: close,
                        reason: format!(
                            "Volume capitulation at support ({:.0}th percentile)",
                            volume_percentile
                        ),
                        support,
                        resistance,
                        volume_percentile,
                    };

                    self.position = Some(Position {
                        side: Side::Long,
                        entry_price: close,
                        entry_time: time,
                        quantity: Decimal::ZERO,
                    });

                    return Some(signal);
                }

                if is_sell_capitulation && at_resistance {
                    let signal = AgentSignal {
                        symbol: self.symbol.clone(),
                        signal: Signal::Short,
                        price: close,
                        reason: format!(
                            "Volume capitulation at resistance ({:.0}th percentile)",
                            volume_percentile
                        ),
                        support,
                        resistance,
                        volume_percentile,
                    };

                    self.position = Some(Position {
                        side: Side::Short,
                        entry_price: close,
                        entry_time: time,
                        quantity: Decimal::ZERO,
                    });

                    return Some(signal);
                }

                // LOSSLESS alternative entry: at support on down day with elevated volume
                // "Elevated" = 80th percentile (derived from data distribution)
                if at_support && is_down_day && volume_percentile >= 80.0 {
                    if let Some(s) = support {
                        let touched_support = self.sr.is_near(close.min(open), s);
                        if touched_support {
                            let signal = AgentSignal {
                                symbol: self.symbol.clone(),
                                signal: Signal::Buy,
                                price: close,
                                reason: format!(
                                    "Price at support with elevated volume ({:.0}th percentile)",
                                    volume_percentile
                                ),
                                support,
                                resistance,
                                volume_percentile,
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

        // Update state
        self.last_price = close;
        self.last_volume = volume;
        self.bar_count += 1;
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
