//! Portfolio Management Module
//!
//! From Tech Trader observations:
//! - Max exposure: 200% long OR 200% short
//! - Typically holds 10-20 positions simultaneously
//! - Both long AND short at the same time
//! - Reports positions as "X% long by Y% short"
//!
//! Position sizing: ~7% per position (based on ~15 positions avg)

use rust_decimal::Decimal;
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::agent::{Signal, Side, AgentSignal};

/// Maximum exposure per side (200%)
pub const MAX_EXPOSURE_PER_SIDE: f64 = 2.0;

/// Maximum number of positions
pub const MAX_POSITIONS: usize = 20;

/// Position size as percentage of account (7%)
pub const POSITION_SIZE_PCT: f64 = 0.07;

/// A position in the portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPosition {
    pub symbol: String,
    pub side: Side,
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub entry_time: DateTime<Utc>,
    pub market_value: Decimal,
}

impl PortfolioPosition {
    /// Calculate unrealized P&L
    pub fn unrealized_pnl(&self) -> Decimal {
        let price_diff = self.current_price - self.entry_price;
        match self.side {
            Side::Long => price_diff * self.quantity,
            Side::Short => -price_diff * self.quantity,
        }
    }

    /// Calculate unrealized P&L percentage
    pub fn unrealized_pnl_pct(&self) -> Decimal {
        if self.entry_price.is_zero() {
            return Decimal::ZERO;
        }
        let price_diff = self.current_price - self.entry_price;
        match self.side {
            Side::Long => price_diff / self.entry_price * dec!(100),
            Side::Short => -price_diff / self.entry_price * dec!(100),
        }
    }
}

/// Portfolio manager
///
/// Tracks positions, calculates exposure, and manages position sizing.
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Current positions by symbol
    positions: HashMap<String, PortfolioPosition>,
    /// Account equity
    equity: Decimal,
    /// Initial balance
    initial_balance: Decimal,
}

impl Portfolio {
    /// Create a new portfolio
    pub fn new(initial_balance: Decimal) -> Self {
        Self {
            positions: HashMap::new(),
            equity: initial_balance,
            initial_balance,
        }
    }

    /// Get current equity
    pub fn equity(&self) -> Decimal {
        self.equity
    }

    /// Update equity
    pub fn set_equity(&mut self, equity: Decimal) {
        self.equity = equity;
    }

    /// Get number of open positions
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }

    /// Get a position by symbol
    pub fn get_position(&self, symbol: &str) -> Option<&PortfolioPosition> {
        self.positions.get(symbol)
    }

    /// Check if we have a position in a symbol
    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    /// Get all positions
    pub fn positions(&self) -> &HashMap<String, PortfolioPosition> {
        &self.positions
    }

    /// Add or update a position
    pub fn add_position(&mut self, position: PortfolioPosition) {
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Remove a position
    pub fn remove_position(&mut self, symbol: &str) -> Option<PortfolioPosition> {
        self.positions.remove(symbol)
    }

    /// Update a position's current price
    pub fn update_price(&mut self, symbol: &str, price: Decimal) {
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.current_price = price;
            pos.market_value = pos.quantity * price;
        }
    }

    /// Calculate total long exposure as percentage of equity
    pub fn long_exposure_pct(&self) -> f64 {
        if self.equity.is_zero() {
            return 0.0;
        }

        let long_value: Decimal = self.positions
            .values()
            .filter(|p| matches!(p.side, Side::Long))
            .map(|p| p.market_value)
            .sum();

        (long_value / self.equity).to_f64().unwrap_or(0.0)
    }

    /// Calculate total short exposure as percentage of equity
    pub fn short_exposure_pct(&self) -> f64 {
        if self.equity.is_zero() {
            return 0.0;
        }

        let short_value: Decimal = self.positions
            .values()
            .filter(|p| matches!(p.side, Side::Short))
            .map(|p| p.market_value)
            .sum();

        (short_value / self.equity).to_f64().unwrap_or(0.0)
    }

    /// Get exposure summary string (e.g., "150% long by 50% short")
    pub fn exposure_summary(&self) -> String {
        let long = self.long_exposure_pct() * 100.0;
        let short = self.short_exposure_pct() * 100.0;
        format!("{:.0}% long by {:.0}% short", long, short)
    }

    /// Calculate total unrealized P&L
    pub fn unrealized_pnl(&self) -> Decimal {
        self.positions.values().map(|p| p.unrealized_pnl()).sum()
    }

    /// Calculate total unrealized P&L percentage
    pub fn unrealized_pnl_pct(&self) -> Decimal {
        if self.initial_balance.is_zero() {
            return Decimal::ZERO;
        }
        self.unrealized_pnl() / self.initial_balance * dec!(100)
    }

    /// Check if we can open a new position on the given side
    pub fn can_open_position(&self, side: Side) -> bool {
        // Check position limit
        if self.positions.len() >= MAX_POSITIONS {
            return false;
        }

        // Check exposure limit
        let current_exposure = match side {
            Side::Long => self.long_exposure_pct(),
            Side::Short => self.short_exposure_pct(),
        };

        current_exposure < MAX_EXPOSURE_PER_SIDE
    }

    /// Calculate position size for a new trade
    ///
    /// LOSSLESS: Scales with volume rank (derived from data, not threshold)
    /// Rank 1 = highest volume = maximum size
    /// Lower ranks = proportionally smaller
    pub fn calculate_position_size(&self, price: Decimal, volume_rank: usize) -> Decimal {
        if price.is_zero() {
            return Decimal::ZERO;
        }

        // Base position value = 7% of equity (portfolio management, not strategy)
        let base_pct = POSITION_SIZE_PCT;

        // LOSSLESS: Scale based on rank (derived from data)
        // Rank 1 = 2.0x, Rank 2 = 1.75x, Rank 3 = 1.5x, Rank 4 = 1.25x, Rank 5+ = 1.0x
        // Formula: scale = 2.0 - (rank - 1) * 0.25, clamped to [1.0, 2.0]
        let scale = (2.0 - (volume_rank as f64 - 1.0) * 0.25).clamp(1.0, 2.0);

        let position_value = self.equity * Decimal::from_f64(base_pct * scale).unwrap_or(dec!(0.07));

        // Shares = position value / price
        (position_value / price).round_dp(0)
    }

    /// Check if a signal should be executed based on portfolio constraints
    pub fn should_execute(&self, signal: &AgentSignal) -> bool {
        match signal.signal {
            Signal::Buy => {
                // Opening long - check if allowed
                !self.has_position(&signal.symbol) && self.can_open_position(Side::Long)
            }
            Signal::Short => {
                // Opening short - check if allowed
                !self.has_position(&signal.symbol) && self.can_open_position(Side::Short)
            }
            Signal::Sell | Signal::Cover => {
                // Closing position - always allowed if we have the position
                self.has_position(&signal.symbol)
            }
            Signal::Hold => false,
        }
    }

    /// Get count of long positions
    pub fn long_count(&self) -> usize {
        self.positions.values().filter(|p| matches!(p.side, Side::Long)).count()
    }

    /// Get count of short positions
    pub fn short_count(&self) -> usize {
        self.positions.values().filter(|p| matches!(p.side, Side::Short)).count()
    }

    /// Clear all positions (for testing/reset)
    pub fn clear(&mut self) {
        self.positions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new(dec!(100000));
        assert_eq!(portfolio.equity(), dec!(100000));
        assert_eq!(portfolio.position_count(), 0);
    }

    #[test]
    fn test_position_sizing_lossless() {
        let portfolio = Portfolio::new(dec!(100000));

        // Rank 1 (highest) = 2.0x scale
        // 7% * 2.0 = 14% of $100k = $14000, at $100/share = 140 shares
        let size_rank1 = portfolio.calculate_position_size(dec!(100), 1);
        assert_eq!(size_rank1, dec!(140));

        // Rank 2 = 1.75x scale
        // 7% * 1.75 = 12.25% of $100k = $12250, at $100/share = 122.5 shares (rounds to 122)
        let size_rank2 = portfolio.calculate_position_size(dec!(100), 2);
        assert_eq!(size_rank2, dec!(122));

        // Rank 3 = 1.5x scale
        // 7% * 1.5 = 10.5% of $100k = $10500, at $100/share = 105 shares
        let size_rank3 = portfolio.calculate_position_size(dec!(100), 3);
        assert_eq!(size_rank3, dec!(105));

        // Rank 4 = 1.25x scale
        // 7% * 1.25 = 8.75% of $100k = $8750, at $100/share = 88 shares (rounded)
        let size_rank4 = portfolio.calculate_position_size(dec!(100), 4);
        assert_eq!(size_rank4, dec!(88));

        // Rank 5+ = 1.0x scale (minimum)
        // 7% * 1.0 = 7% of $100k = $7000, at $100/share = 70 shares
        let size_rank5 = portfolio.calculate_position_size(dec!(100), 5);
        assert_eq!(size_rank5, dec!(70));

        // Higher ranks still get minimum scale
        let size_rank10 = portfolio.calculate_position_size(dec!(100), 10);
        assert_eq!(size_rank10, dec!(70));
    }

    #[test]
    fn test_exposure_calculation() {
        let mut portfolio = Portfolio::new(dec!(100000));

        // Add a long position worth $50k (50% exposure)
        portfolio.add_position(PortfolioPosition {
            symbol: "AAPL".to_string(),
            side: Side::Long,
            quantity: dec!(500),
            entry_price: dec!(100),
            current_price: dec!(100),
            entry_time: Utc::now(),
            market_value: dec!(50000),
        });

        assert!((portfolio.long_exposure_pct() - 0.5).abs() < 0.01);
        assert!(portfolio.short_exposure_pct() < 0.01);
    }

    #[test]
    fn test_can_open_position() {
        let mut portfolio = Portfolio::new(dec!(100000));

        // Should be able to open initially
        assert!(portfolio.can_open_position(Side::Long));
        assert!(portfolio.can_open_position(Side::Short));

        // Add positions up to 190% long exposure
        for i in 0..19 {
            portfolio.add_position(PortfolioPosition {
                symbol: format!("SYM{}", i),
                side: Side::Long,
                quantity: dec!(100),
                entry_price: dec!(100),
                current_price: dec!(100),
                entry_time: Utc::now(),
                market_value: dec!(10000), // 10% each
            });
        }

        // 190% exposure - can still open one more
        assert!(portfolio.can_open_position(Side::Long));

        // Add one more to hit 200%
        portfolio.add_position(PortfolioPosition {
            symbol: "SYM19".to_string(),
            side: Side::Long,
            quantity: dec!(100),
            entry_price: dec!(100),
            current_price: dec!(100),
            entry_time: Utc::now(),
            market_value: dec!(10000),
        });

        // Now at 200% - can't open more longs (hit position limit of 20)
        assert!(!portfolio.can_open_position(Side::Long));

        // But shorts should still be possible (if we had room)
        // Actually at 20 positions, can't open anything
        assert!(!portfolio.can_open_position(Side::Short));
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut portfolio = Portfolio::new(dec!(100000));

        // Add a long position that gained 10%
        portfolio.add_position(PortfolioPosition {
            symbol: "AAPL".to_string(),
            side: Side::Long,
            quantity: dec!(100),
            entry_price: dec!(100),
            current_price: dec!(110),
            entry_time: Utc::now(),
            market_value: dec!(11000),
        });

        // Should have $1000 unrealized gain
        assert_eq!(portfolio.unrealized_pnl(), dec!(1000));
    }

    #[test]
    fn test_exposure_summary() {
        let mut portfolio = Portfolio::new(dec!(100000));

        portfolio.add_position(PortfolioPosition {
            symbol: "LONG1".to_string(),
            side: Side::Long,
            quantity: dec!(100),
            entry_price: dec!(100),
            current_price: dec!(100),
            entry_time: Utc::now(),
            market_value: dec!(50000),
        });

        portfolio.add_position(PortfolioPosition {
            symbol: "SHORT1".to_string(),
            side: Side::Short,
            quantity: dec!(100),
            entry_price: dec!(100),
            current_price: dec!(100),
            entry_time: Utc::now(),
            market_value: dec!(25000),
        });

        let summary = portfolio.exposure_summary();
        assert!(summary.contains("50%"));
        assert!(summary.contains("25%"));
    }

    #[test]
    fn test_position_operations() {
        let mut portfolio = Portfolio::new(dec!(100000));
        let symbol = "AAPL";

        assert!(!portfolio.has_position(symbol));

        portfolio.add_position(PortfolioPosition {
            symbol: symbol.to_string(),
            side: Side::Long,
            quantity: dec!(100),
            entry_price: dec!(150),
            current_price: dec!(150),
            entry_time: Utc::now(),
            market_value: dec!(15000),
        });

        assert!(portfolio.has_position(symbol));
        assert_eq!(portfolio.position_count(), 1);

        // Update price
        portfolio.update_price(symbol, dec!(160));
        let pos = portfolio.get_position(symbol).unwrap();
        assert_eq!(pos.current_price, dec!(160));

        // Remove position
        let removed = portfolio.remove_position(symbol);
        assert!(removed.is_some());
        assert!(!portfolio.has_position(symbol));
    }
}
