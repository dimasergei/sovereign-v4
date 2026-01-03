//! Portfolio Management Module - Lossless Position Sizing
//!
//! From Tech Trader observations:
//! - Max exposure: 200% long OR 200% short
//! - Typically holds 10-20 positions simultaneously
//! - Both long AND short at the same time
//! - Reports positions as "X% long by Y% short"
//!
//! LOSSLESS PRINCIPLE: Position size is DERIVED from:
//! - Risk amount (1% of equity) - portfolio management, not strategy
//! - Stop distance from S/R levels - derived from market data
//! - NO volume scaling for size - volume affects ENTRY decision, not SIZE

use rust_decimal::Decimal;
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::agent::{Signal, Side, AgentSignal};
use crate::universe::Sector;

/// Maximum exposure per side (200%)
pub const MAX_EXPOSURE_PER_SIDE: f64 = 2.0;

/// Maximum number of positions
pub const MAX_POSITIONS: usize = 20;

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
    /// Sector classification for diversification reporting
    pub sector: Sector,
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

    /// Calculate position size using lossless principles
    ///
    /// LOSSLESS: Position size is DERIVED from:
    /// - Risk amount (1% of equity) - portfolio management, not strategy
    /// - Stop distance from S/R levels - derived from market data
    /// - ATR fallback when no S/R available - derived from volatility
    /// - NO volume scaling - volume affects ENTRY decision, not SIZE
    ///
    /// This matches pftq's philosophy: "No tweaking, no updates"
    ///
    /// # Arguments
    /// * `price` - Current price
    /// * `support` - Derived support level (preferred for stop distance)
    /// * `atr` - Average True Range fallback (used when no support)
    pub fn calculate_position_size(
        &self,
        price: Decimal,
        support: Option<Decimal>,
        atr: Option<Decimal>,
    ) -> Decimal {
        if price.is_zero() {
            return Decimal::ZERO;
        }

        // Portfolio risk management (not strategy parameter)
        // 1% risk per trade is standard risk management, not a tunable strategy param
        let risk_pct = dec!(0.01);
        let risk_amount = self.equity * risk_pct;

        // Stop distance DERIVED from S/R (lossless) or ATR fallback
        let stop_distance = match support {
            Some(s) if s < price && !s.is_zero() => price - s,
            _ => {
                // No support found - use ATR as derived fallback
                // ATR represents natural volatility, derived from price data
                match atr {
                    Some(a) if !a.is_zero() => a,
                    _ => {
                        // No ATR either - cannot size safely
                        // This should never happen after proper bootstrap
                        return Decimal::ZERO;
                    }
                }
            }
        };

        if stop_distance.is_zero() {
            return Decimal::ZERO;
        }

        // Position size = risk / stop distance
        // No volume scaling - volume affects entry decision, not size
        let shares = (risk_amount / stop_distance).round_dp(0);

        // Cap at reasonable position size (max ~10% of equity)
        let max_position_value = self.equity * dec!(0.10);
        let max_shares = (max_position_value / price).round_dp(0);

        shares.min(max_shares)
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

    /// Get exposure by sector (for logging/reporting only)
    pub fn sector_exposure(&self) -> HashMap<Sector, (f64, f64)> {
        let mut exposure: HashMap<Sector, (Decimal, Decimal)> = HashMap::new();

        for pos in self.positions.values() {
            let (long, short) = exposure.entry(pos.sector).or_insert((Decimal::ZERO, Decimal::ZERO));
            match pos.side {
                Side::Long => *long += pos.market_value,
                Side::Short => *short += pos.market_value,
            }
        }

        exposure.into_iter()
            .map(|(sector, (long, short))| {
                let long_pct = if self.equity.is_zero() {
                    0.0
                } else {
                    (long / self.equity).to_f64().unwrap_or(0.0)
                };
                let short_pct = if self.equity.is_zero() {
                    0.0
                } else {
                    (short / self.equity).to_f64().unwrap_or(0.0)
                };
                (sector, (long_pct, short_pct))
            })
            .collect()
    }

    /// Format sector exposure like Tech Trader: "14% short Finance, 7% long Healthcare"
    pub fn sector_summary(&self) -> String {
        let exposure = self.sector_exposure();
        let mut parts: Vec<String> = exposure.into_iter()
            .filter(|(_, (l, s))| *l > 0.01 || *s > 0.01)
            .flat_map(|(sector, (long, short))| {
                let mut v = vec![];
                if long > 0.01 {
                    v.push(format!("{:.0}% long {:?}", long * 100.0, sector));
                }
                if short > 0.01 {
                    v.push(format!("{:.0}% short {:?}", short * 100.0, sector));
                }
                v
            })
            .collect();
        parts.sort();
        if parts.is_empty() {
            "No sector exposure".to_string()
        } else {
            parts.join(", ")
        }
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

        // LOSSLESS: Position size derived from support distance
        // Risk = 1% of $100k = $1000
        // Price = $100, Support = $95 -> stop distance = $5
        // Size = $1000 / $5 = 200 shares
        // But capped at 10% of equity = $10000 / $100 = 100 shares
        let size_with_support = portfolio.calculate_position_size(dec!(100), Some(dec!(95)), None);
        assert_eq!(size_with_support, dec!(100)); // Capped at 10%

        // Tighter stop = larger position (but still capped)
        // Price = $100, Support = $98 -> stop distance = $2
        // Size = $1000 / $2 = 500 shares (capped to 100)
        let size_tight_stop = portfolio.calculate_position_size(dec!(100), Some(dec!(98)), None);
        assert_eq!(size_tight_stop, dec!(100)); // Capped

        // Wider stop = smaller position
        // Price = $100, Support = $80 -> stop distance = $20
        // Size = $1000 / $20 = 50 shares (under cap, so actual 50)
        let size_wide_stop = portfolio.calculate_position_size(dec!(100), Some(dec!(80)), None);
        assert_eq!(size_wide_stop, dec!(50));

        // No support but has ATR = use ATR as fallback
        // Price = $100, no support, ATR = $3 -> stop distance = $3
        // Size = $1000 / $3 = 333 shares (capped to 100)
        let size_atr_fallback = portfolio.calculate_position_size(dec!(100), None, Some(dec!(3)));
        assert_eq!(size_atr_fallback, dec!(100)); // Capped

        // No support and no ATR = cannot size safely (returns 0)
        let size_no_data = portfolio.calculate_position_size(dec!(100), None, None);
        assert_eq!(size_no_data, dec!(0));
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
            sector: Sector::Technology,
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
                sector: Sector::Unknown,
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
            sector: Sector::Unknown,
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
            sector: Sector::Technology,
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
            sector: Sector::Technology,
        });

        portfolio.add_position(PortfolioPosition {
            symbol: "SHORT1".to_string(),
            side: Side::Short,
            quantity: dec!(100),
            entry_price: dec!(100),
            current_price: dec!(100),
            entry_time: Utc::now(),
            market_value: dec!(25000),
            sector: Sector::Finance,
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
            sector: Sector::Technology,
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

    #[test]
    fn test_sector_exposure() {
        let mut portfolio = Portfolio::new(dec!(100000));

        // Add tech long position (30% of equity)
        portfolio.add_position(PortfolioPosition {
            symbol: "AAPL".to_string(),
            side: Side::Long,
            quantity: dec!(300),
            entry_price: dec!(100),
            current_price: dec!(100),
            entry_time: Utc::now(),
            market_value: dec!(30000),
            sector: Sector::Technology,
        });

        // Add finance short position (20% of equity)
        portfolio.add_position(PortfolioPosition {
            symbol: "JPM".to_string(),
            side: Side::Short,
            quantity: dec!(200),
            entry_price: dec!(100),
            current_price: dec!(100),
            entry_time: Utc::now(),
            market_value: dec!(20000),
            sector: Sector::Finance,
        });

        let sector_exp = portfolio.sector_exposure();

        // Check tech exposure
        let (tech_long, tech_short) = sector_exp.get(&Sector::Technology).unwrap_or(&(0.0, 0.0));
        assert!((*tech_long - 0.30).abs() < 0.01);
        assert!(*tech_short < 0.01);

        // Check finance exposure
        let (fin_long, fin_short) = sector_exp.get(&Sector::Finance).unwrap_or(&(0.0, 0.0));
        assert!(*fin_long < 0.01);
        assert!((*fin_short - 0.20).abs() < 0.01);

        // Check sector summary
        let summary = portfolio.sector_summary();
        assert!(summary.contains("Technology") || summary.contains("long"));
        assert!(summary.contains("Finance") || summary.contains("short"));
    }
}
