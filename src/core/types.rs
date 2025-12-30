//! Core type definitions for the trading system
//!
//! These types are used throughout the system and represent
//! the fundamental concepts of trading.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Represents a single price candle (OHLCV)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    /// Timestamp of the candle open
    pub time: DateTime<Utc>,
    /// Opening price
    pub open: Decimal,
    /// Highest price during the period
    pub high: Decimal,
    /// Lowest price during the period
    pub low: Decimal,
    /// Closing price
    pub close: Decimal,
    /// Volume traded during the period
    pub volume: Decimal,
}

impl Candle {
    /// Create a new candle
    pub fn new(
        time: DateTime<Utc>,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
    ) -> Self {
        Self { time, open, high, low, close, volume }
    }
    
    /// Check if this is a bullish (green) candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
    
    /// Check if this is a bearish (red) candle
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
    
    /// Get the body size (absolute difference between open and close)
    pub fn body_size(&self) -> Decimal {
        (self.close - self.open).abs()
    }
    
    /// Get the full range (high - low)
    pub fn range(&self) -> Decimal {
        self.high - self.low
    }
}

/// Market tick (bid/ask)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub time: DateTime<Utc>,
    pub bid: Decimal,
    pub ask: Decimal,
}

impl Tick {
    /// Get the spread in decimal
    pub fn spread(&self) -> Decimal {
        self.ask - self.bid
    }
    
    /// Get the mid price
    pub fn mid(&self) -> Decimal {
        (self.bid + self.ask) / Decimal::from(2)
    }
}

/// Detected market trend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Up,
    Down,
    Neutral,
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trend::Up => write!(f, "UP"),
            Trend::Down => write!(f, "DOWN"),
            Trend::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// Momentum state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Momentum {
    StrongUp,
    Up,
    Neutral,
    Down,
    StrongDown,
}

impl std::fmt::Display for Momentum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Momentum::StrongUp => write!(f, "STRONG_UP"),
            Momentum::Up => write!(f, "UP"),
            Momentum::Neutral => write!(f, "NEUTRAL"),
            Momentum::Down => write!(f, "DOWN"),
            Momentum::StrongDown => write!(f, "STRONG_DOWN"),
        }
    }
}

/// Volume state (lossless - no fixed thresholds)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolumeState {
    /// Top 10% of recent volume - something significant happening
    Spike,
    /// Normal volume
    Normal,
    /// Bottom 10% of recent volume - market is dead
    Dead,
}

/// Trading signal from an agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Buy => write!(f, "BUY"),
            Signal::Sell => write!(f, "SELL"),
            Signal::Hold => write!(f, "HOLD"),
        }
    }
}

/// Complete market observation (what the agent "sees")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub trend: Trend,
    pub momentum: Momentum,
    pub volume_state: VolumeState,
    pub near_support: bool,
    pub near_resistance: bool,
    pub nearest_support: Option<Decimal>,
    pub nearest_resistance: Option<Decimal>,
    pub bounce_up: bool,
    pub bounce_down: bool,
}

/// Trading decision with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub signal: Signal,
    /// Conviction level 0-100
    pub conviction: u8,
    /// Human-readable reasons for the decision
    pub reasoning: Vec<String>,
    /// Suggested stop loss price
    pub stop_loss: Option<Decimal>,
    /// Suggested take profit price
    pub take_profit: Option<Decimal>,
}

impl Decision {
    /// Create a HOLD decision with no conviction
    pub fn hold() -> Self {
        Self {
            signal: Signal::Hold,
            conviction: 0,
            reasoning: vec!["No clear setup".to_string()],
            stop_loss: None,
            take_profit: None,
        }
    }
    
    /// Create a BUY decision
    pub fn buy(conviction: u8, reasons: Vec<String>, sl: Option<Decimal>, tp: Option<Decimal>) -> Self {
        Self {
            signal: Signal::Buy,
            conviction,
            reasoning: reasons,
            stop_loss: sl,
            take_profit: tp,
        }
    }
    
    /// Create a SELL decision
    pub fn sell(conviction: u8, reasons: Vec<String>, sl: Option<Decimal>, tp: Option<Decimal>) -> Self {
        Self {
            signal: Signal::Sell,
            conviction,
            reasoning: reasons,
            stop_loss: sl,
            take_profit: tp,
        }
    }
}

/// Position in the market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub ticket: u64,
    pub symbol: String,
    pub side: PositionSide,
    pub volume: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub profit: Decimal,
    pub opened_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}

impl Position {
    /// Calculate profit percentage
    pub fn profit_pct(&self) -> Decimal {
        if self.entry_price.is_zero() {
            return Decimal::ZERO;
        }
        
        match self.side {
            PositionSide::Long => {
                (self.current_price - self.entry_price) / self.entry_price * Decimal::from(100)
            }
            PositionSide::Short => {
                (self.entry_price - self.current_price) / self.entry_price * Decimal::from(100)
            }
        }
    }
}

/// Account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub balance: Decimal,
    pub equity: Decimal,
    pub margin_used: Decimal,
    pub margin_free: Decimal,
    pub profit: Decimal,
}

impl AccountInfo {
    /// Calculate profit/loss percentage
    pub fn pnl_pct(&self) -> Decimal {
        if self.balance.is_zero() {
            return Decimal::ZERO;
        }
        self.profit / self.balance * Decimal::from(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_candle_bullish() {
        let candle = Candle::new(
            Utc::now(),
            dec!(100),
            dec!(110),
            dec!(95),
            dec!(105),
            dec!(1000),
        );
        
        assert!(candle.is_bullish());
        assert!(!candle.is_bearish());
    }
    
    #[test]
    fn test_candle_bearish() {
        let candle = Candle::new(
            Utc::now(),
            dec!(105),
            dec!(110),
            dec!(95),
            dec!(100),
            dec!(1000),
        );
        
        assert!(candle.is_bearish());
        assert!(!candle.is_bullish());
    }
    
    #[test]
    fn test_tick_spread() {
        let tick = Tick {
            time: Utc::now(),
            bid: dec!(100.00),
            ask: dec!(100.50),
        };
        
        assert_eq!(tick.spread(), dec!(0.50));
        assert_eq!(tick.mid(), dec!(100.25));
    }
    
    #[test]
    fn test_decision_hold() {
        let decision = Decision::hold();
        assert_eq!(decision.signal, Signal::Hold);
        assert_eq!(decision.conviction, 0);
    }
}
