//! Agent Module
//!
//! Independent trading agent that focuses on a single symbol.
//!
//! "It is analogous to having a thousand independent traders each focusing 
//! on a single stock, as opposed to a single quant manager trying to make 
//! sense of a thousand datapoints." - pftq

use crate::core::types::*;
use crate::core::lossless::MarketObserver;

/// Agent trait - all agents must implement this
pub trait Agent: Send + Sync {
    /// Get the symbol this agent trades
    fn symbol(&self) -> &str;
    
    /// Update the agent with new market data
    fn update(&mut self, candle: &Candle);
    
    /// Make a trading decision
    fn decide(&self, current_price: rust_decimal::Decimal) -> Decision;
    
    /// Check if an open position should be closed
    fn should_close(&self, position: &Position) -> (bool, String);
}

/// Single-symbol trading agent
pub struct SymbolAgent {
    symbol: String,
    observer: MarketObserver,
}

impl SymbolAgent {
    /// Create a new agent for a symbol
    pub fn new(symbol: String, tick_size: rust_decimal::Decimal, is_forex: bool) -> Self {
        Self {
            symbol,
            observer: MarketObserver::new(tick_size, is_forex),
        }
    }
}

impl Agent for SymbolAgent {
    fn symbol(&self) -> &str {
        &self.symbol
    }
    
    fn update(&mut self, candle: &Candle) {
        self.observer.update(candle);
    }
    
    fn decide(&self, current_price: rust_decimal::Decimal) -> Decision {
        let observation = self.observer.observe(current_price);
        
        // TODO: Implement human-like thinking
        // This is where the "edge" will be developed
        
        Decision::hold()
    }
    
    fn should_close(&self, _position: &Position) -> (bool, String) {
        // TODO: Implement close logic
        (false, String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_agent_creation() {
        let agent = SymbolAgent::new("XAUUSD".to_string(), dec!(0.01), true);
        assert_eq!(agent.symbol(), "XAUUSD");
    }
}
