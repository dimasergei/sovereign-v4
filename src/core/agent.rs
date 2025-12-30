//! Agent Module
//!
//! Independent trading agent that focuses on a single symbol.
//!
//! "It is analogous to having a thousand independent traders each focusing 
//! on a single stock, as opposed to a single quant manager trying to make 
//! sense of a thousand datapoints." - pftq

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use crate::core::types::*;
use crate::core::lossless::MarketObserver;
use crate::core::strategy::{Strategy, TradeSignal, SignalDirection};

/// Agent signal with symbol context
#[derive(Debug, Clone)]
pub struct AgentSignal {
    pub symbol: String,
    pub signal: TradeSignal,
    pub price: Decimal,
    pub spread: Decimal,
}

/// Single-symbol trading agent
pub struct SymbolAgent {
    symbol: String,
    observer: MarketObserver,
    strategy: Strategy,
    tick_size: Decimal,
    is_forex: bool,
    candle_count: u64,
    last_price: Decimal,
    last_spread: Decimal,
    in_position: bool,
    position_side: Option<SignalDirection>,
}

impl SymbolAgent {
    /// Create a new agent for a symbol
    pub fn new(symbol: String, tick_size: Decimal, is_forex: bool) -> Self {
        Self {
            symbol,
            observer: MarketObserver::new(tick_size, is_forex),
            strategy: Strategy::default(),
            tick_size,
            is_forex,
            candle_count: 0,
            last_price: Decimal::ZERO,
            last_spread: Decimal::ZERO,
            in_position: false,
            position_side: None,
        }
    }

    /// Get symbol name
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Update with new candle
    pub fn update(&mut self, candle: &Candle) {
        self.observer.update(candle);
        self.last_price = candle.close;
        self.candle_count += 1;
    }

    /// Update tick data (bid/ask)
    pub fn update_tick(&mut self, bid: Decimal, ask: Decimal) {
        self.last_price = bid;
        self.last_spread = ask - bid;
    }

    /// Set position state
    pub fn set_position(&mut self, in_position: bool, side: Option<SignalDirection>) {
        self.in_position = in_position;
        self.position_side = side;
    }

    /// Check if agent has enough data
    pub fn is_ready(&self) -> bool {
        self.candle_count >= 5 // Need at least 5 candles for trend detection
    }

    /// Generate trading signal
    pub fn analyze(&self) -> Option<AgentSignal> {
        if !self.is_ready() {
            return None;
        }

        if self.in_position {
            return None; // Already in position
        }

        let obs = self.observer.observe(self.last_price);
        let signal = self.strategy.analyze(&obs, self.last_price);

        // Only return if we have a signal (not hold)
        if signal.direction == SignalDirection::Hold {
            return None;
        }

        Some(AgentSignal {
            symbol: self.symbol.clone(),
            signal,
            price: self.last_price,
            spread: self.last_spread,
        })
    }

    /// Get current observation for logging
    pub fn get_observation(&self) -> Observation {
        self.observer.observe(self.last_price)
    }

    /// Get candle count
    pub fn candle_count(&self) -> u64 {
        self.candle_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_agent_creation() {
        let agent = SymbolAgent::new("XAUUSD".to_string(), dec!(0.01), true);
        assert_eq!(agent.symbol(), "XAUUSD");
        assert!(!agent.is_ready());
    }

    #[test]
    fn test_agent_needs_data() {
        let mut agent = SymbolAgent::new("XAUUSD".to_string(), dec!(0.01), true);
        
        // Feed 5 candles
        for i in 0..5 {
            let candle = Candle::new(
                Utc::now(),
                dec!(100) + Decimal::from(i),
                dec!(101) + Decimal::from(i),
                dec!(99) + Decimal::from(i),
                dec!(100.5) + Decimal::from(i),
                dec!(1000),
            );
            agent.update(&candle);
        }
        
        assert!(agent.is_ready());
    }
}
