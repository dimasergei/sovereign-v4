//! Lossless Support/Resistance Algorithm
//!
//! "The idea is to just measure how often a price range has been traveled,
//! so that the areas never traveled can be deduced to be where price moves
//! least freely (aka support and resistance)." - pftq
//!
//! # Algorithm
//!
//! - Each price range gets a score starting at 0
//! - Every time price crosses a range, that range's score decreases by 1
//! - Untraveled ranges stay at 0 (strongest S/R)
//! - Support = highest-scoring range below current price
//! - Resistance = highest-scoring range above current price
//!
//! # Properties
//!
//! - **No thresholds** - just counting
//! - **No parameters** - granularity DERIVED from ATR (market data)
//! - **No statistics** - pure 1-to-1 mapping
//! - **Computationally light** - single subtraction per crossing

use std::collections::HashMap;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;

/// A discretized price range
///
/// Represents a price level like $50.00-$50.10 when granularity is $0.10
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PriceRange {
    /// The range index (price / granularity, rounded)
    index: i64,
    /// The granularity used for this range
    granularity_cents: i64,
}

impl PriceRange {
    /// Create a new price range from a price and granularity
    pub fn from_price(price: Decimal, granularity: Decimal) -> Self {
        // Convert to integer index for HashMap efficiency
        let index = (price / granularity).round().to_i64().unwrap_or(0);
        let granularity_cents = (granularity * Decimal::from(100))
            .round()
            .to_i64()
            .unwrap_or(1);

        Self { index, granularity_cents }
    }

    /// Get the midpoint price of this range
    pub fn mid(&self) -> Decimal {
        let granularity = Decimal::from(self.granularity_cents) / Decimal::from(100);
        Decimal::from(self.index) * granularity
    }

    /// Get the low boundary of this range
    pub fn low(&self) -> Decimal {
        let granularity = Decimal::from(self.granularity_cents) / Decimal::from(100);
        (Decimal::from(self.index) - Decimal::from(1) / Decimal::from(2)) * granularity
    }

    /// Get the high boundary of this range
    pub fn high(&self) -> Decimal {
        let granularity = Decimal::from(self.granularity_cents) / Decimal::from(100);
        (Decimal::from(self.index) + Decimal::from(1) / Decimal::from(2)) * granularity
    }
}

/// Lossless Support/Resistance tracker
///
/// Counts how often each price range has been crossed.
/// Score of 0 = never crossed = strongest level.
/// More negative = crossed more often = weaker level.
#[derive(Debug, Clone)]
pub struct SRLevels {
    /// Price range -> crossing count (0 = strongest, negative = weaker)
    scores: HashMap<i64, i32>,
    /// Granularity for discretizing prices
    granularity: Decimal,
    /// Last processed price (to detect crossings)
    last_price: Option<Decimal>,
}

impl SRLevels {
    /// Create a new S/R tracker
    ///
    /// # Arguments
    /// * `granularity` - The price increment for each range
    ///   - Stocks: Use $0.10 for most, $1.00 for high-priced
    ///   - ETFs: Use $0.05 or $0.10
    ///   - Crypto: BTC use $10-$100, XRP use $0.001
    pub fn new(granularity: Decimal) -> Self {
        Self {
            scores: HashMap::new(),
            granularity,
            last_price: None,
        }
    }

    /// Convert price to range index
    fn price_to_index(&self, price: Decimal) -> i64 {
        (price / self.granularity).round().to_i64().unwrap_or(0)
    }

    /// Convert range index back to price
    fn index_to_price(&self, index: i64) -> Decimal {
        Decimal::from(index) * self.granularity
    }

    /// Update with a new price
    ///
    /// Decrements the score for each range crossed between last price and current.
    pub fn update(&mut self, price: Decimal) {
        if let Some(last) = self.last_price {
            let last_idx = self.price_to_index(last);
            let curr_idx = self.price_to_index(price);

            if curr_idx != last_idx {
                // Decrement score for each range crossed
                let (start, end) = if curr_idx > last_idx {
                    (last_idx, curr_idx)
                } else {
                    (curr_idx, last_idx)
                };

                for idx in start..=end {
                    *self.scores.entry(idx).or_insert(0) -= 1;
                }
            }
        }

        self.last_price = Some(price);
    }

    /// Update with a candle (processes the full price journey)
    ///
    /// For a bar, we process: open -> high -> low -> close
    /// This captures the intrabar price movement.
    pub fn update_bar(&mut self, open: Decimal, high: Decimal, low: Decimal, close: Decimal) {
        // Process the candle's price journey
        // Assume: open -> high -> low -> close for bullish
        //         open -> low -> high -> close for bearish
        let is_bullish = close > open;

        if is_bullish {
            self.update(open);
            self.update(high);
            self.update(low);
            self.update(close);
        } else {
            self.update(open);
            self.update(low);
            self.update(high);
            self.update(close);
        }
    }

    /// Get the strongest support level below the given price
    ///
    /// Returns the price level with the highest score (closest to 0)
    /// that is below the current price.
    pub fn get_support(&self, current_price: Decimal) -> Option<Decimal> {
        let current_idx = self.price_to_index(current_price);

        self.scores
            .iter()
            .filter(|(idx, _)| **idx < current_idx)
            .max_by_key(|(_, score)| *score) // Highest score = closest to 0 = strongest
            .map(|(idx, _)| self.index_to_price(*idx))
    }

    /// Get the strongest resistance level above the given price
    pub fn get_resistance(&self, current_price: Decimal) -> Option<Decimal> {
        let current_idx = self.price_to_index(current_price);

        self.scores
            .iter()
            .filter(|(idx, _)| **idx > current_idx)
            .max_by_key(|(_, score)| *score)
            .map(|(idx, _)| self.index_to_price(*idx))
    }

    /// Get multiple support levels below the given price, sorted by strength
    pub fn get_support_levels(&self, current_price: Decimal, count: usize) -> Vec<(Decimal, i32)> {
        let current_idx = self.price_to_index(current_price);

        let mut levels: Vec<_> = self.scores
            .iter()
            .filter(|(idx, _)| **idx < current_idx)
            .map(|(idx, score)| (self.index_to_price(*idx), *score))
            .collect();

        // Sort by score descending (closest to 0 = strongest)
        levels.sort_by(|a, b| b.1.cmp(&a.1));
        levels.truncate(count);
        levels
    }

    /// Get multiple resistance levels above the given price, sorted by strength
    pub fn get_resistance_levels(&self, current_price: Decimal, count: usize) -> Vec<(Decimal, i32)> {
        let current_idx = self.price_to_index(current_price);

        let mut levels: Vec<_> = self.scores
            .iter()
            .filter(|(idx, _)| **idx > current_idx)
            .map(|(idx, score)| (self.index_to_price(*idx), *score))
            .collect();

        levels.sort_by(|a, b| b.1.cmp(&a.1));
        levels.truncate(count);
        levels
    }

    /// Check if price is "near" a level
    ///
    /// Near means within half a granularity unit.
    /// This is purely geometric, not a tunable threshold.
    pub fn is_near(&self, price: Decimal, level: Decimal) -> bool {
        let diff = (price - level).abs();
        diff <= self.granularity / Decimal::from(2)
    }

    /// Check if price is near support
    pub fn is_at_support(&self, current_price: Decimal) -> bool {
        if let Some(support) = self.get_support(current_price) {
            self.is_near(current_price, support)
        } else {
            false
        }
    }

    /// Check if price is near resistance
    pub fn is_at_resistance(&self, current_price: Decimal) -> bool {
        if let Some(resistance) = self.get_resistance(current_price) {
            self.is_near(current_price, resistance)
        } else {
            false
        }
    }

    /// Get the score at a specific price level
    pub fn score_at(&self, price: Decimal) -> i32 {
        let idx = self.price_to_index(price);
        *self.scores.get(&idx).unwrap_or(&0)
    }

    /// Get total number of levels tracked
    pub fn level_count(&self) -> usize {
        self.scores.len()
    }

    /// Clear all levels (start fresh)
    pub fn clear(&mut self) {
        self.scores.clear();
        self.last_price = None;
    }
}

/// Calculate ATR (Average True Range) from historical bars
///
/// LOSSLESS: ATR is derived purely from market data - no parameters.
/// The ATR represents the market's natural volatility/movement grid.
///
/// # Arguments
/// * `bars` - Historical bars as (open, high, low, close) tuples
///
/// # Returns
/// The average true range, or ZERO if insufficient data
pub fn calculate_atr(bars: &[(Decimal, Decimal, Decimal, Decimal)]) -> Decimal {
    if bars.len() < 2 {
        return Decimal::ZERO;
    }

    let mut tr_sum = Decimal::ZERO;
    for i in 1..bars.len() {
        let (_, high, low, _) = bars[i];
        let (_, _, _, prev_close) = bars[i - 1];

        // True Range = max(H-L, |H-PC|, |L-PC|)
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        tr_sum += tr;
    }

    tr_sum / Decimal::from(bars.len() as i64 - 1)
}

/// Derive granularity from ATR (lossless - derived from market data)
///
/// ATR represents the market's natural price grid - how much it typically moves.
/// Granularity = ATR / 10 gives us meaningful S/R buckets.
///
/// # Arguments
/// * `atr` - Average True Range calculated from historical data
///
/// # Returns
/// A "nice" granularity value rounded for readability
pub fn granularity_from_atr(atr: Decimal) -> Decimal {
    if atr.is_zero() {
        return dec!(0.10); // Bootstrap fallback only
    }

    // ATR / 10 gives us ~10 S/R levels per average day's range
    let raw = atr / dec!(10);

    // Round to nearest "nice" granularity for readability (not strategy-affecting)
    let nice_values = [
        dec!(0.01), dec!(0.02), dec!(0.05), dec!(0.10), dec!(0.25), dec!(0.50),
        dec!(1.00), dec!(2.50), dec!(5.00), dec!(10.00), dec!(25.00),
        dec!(50.00), dec!(100.00),
    ];

    nice_values
        .iter()
        .min_by_key(|&&v| {
            let diff = (v - raw).abs();
            // Multiply by large factor to convert to comparable integer
            (diff * dec!(10000)).to_i64().unwrap_or(i64::MAX)
        })
        .copied()
        .unwrap_or(dec!(0.10))
}

#[deprecated(since = "4.0.0", note = "Use granularity_from_atr for lossless derivation")]
/// Determine appropriate granularity for a symbol (legacy)
///
/// DEPRECATED: This uses price-based thresholds which violate lossless principles.
/// Use `granularity_from_atr()` instead for proper lossless derivation.
pub fn default_granularity(symbol: &str, price: Decimal) -> Decimal {
    // Crypto
    if symbol.ends_with("BTC") || symbol == "BTC" || symbol == "BTCUSD" {
        return Decimal::from(100); // $100 ranges for BTC
    }
    if symbol.ends_with("ETH") || symbol == "ETH" || symbol == "ETHUSD" {
        return Decimal::from(10); // $10 ranges for ETH
    }
    if symbol.contains("XRP") || symbol.contains("DOGE") || symbol.contains("ADA") {
        return Decimal::new(1, 3); // $0.001 for small-cap crypto
    }

    // Stock/ETF granularity based on price
    if price > Decimal::from(500) {
        Decimal::from(1) // $1.00 for high-priced stocks
    } else if price > Decimal::from(100) {
        Decimal::new(50, 2) // $0.50
    } else if price > Decimal::from(50) {
        Decimal::new(25, 2) // $0.25
    } else if price > Decimal::from(10) {
        Decimal::new(10, 2) // $0.10
    } else {
        Decimal::new(5, 2) // $0.05 for low-priced stocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_basic_sr_counting() {
        let mut sr = SRLevels::new(dec!(1.0));

        // Price moves $9 -> $10: range 9-10 gets score = -1
        sr.update(dec!(9));
        sr.update(dec!(10));

        assert_eq!(sr.score_at(dec!(9)), -1);
        assert_eq!(sr.score_at(dec!(10)), -1);

        // Price moves $10 -> $9: range 9-10 gets score = -2
        sr.update(dec!(9));

        assert_eq!(sr.score_at(dec!(9)), -2);
        assert_eq!(sr.score_at(dec!(10)), -2);
    }

    #[test]
    fn test_untraveled_is_strongest() {
        let mut sr = SRLevels::new(dec!(1.0));

        // Price bounces between 10 and 12 several times
        for _ in 0..5 {
            sr.update(dec!(10));
            sr.update(dec!(12));
        }

        // Levels 10-12 are well traveled (very negative scores)
        // Level 20 is never touched (score = 0 = strongest)
        assert!(sr.score_at(dec!(11)) < 0);
        assert_eq!(sr.score_at(dec!(20)), 0); // Never touched

        // Now touch 15 just once
        sr.update(dec!(15));

        // 15 was touched, but less than 10-12
        let resistance = sr.get_resistance(dec!(13));
        assert!(resistance.is_some());

        // The resistance should be above 13 and have been touched fewer times than 10-12
        let r = resistance.unwrap();
        assert!(r > dec!(13));
    }

    #[test]
    fn test_support_below_price() {
        let mut sr = SRLevels::new(dec!(0.10));

        // Build some history
        sr.update(dec!(100.00));
        sr.update(dec!(100.50));
        sr.update(dec!(100.00));
        sr.update(dec!(100.50));

        // Support should be below current price
        let support = sr.get_support(dec!(100.25));
        assert!(support.is_some());
        assert!(support.unwrap() < dec!(100.25));
    }

    #[test]
    fn test_resistance_above_price() {
        let mut sr = SRLevels::new(dec!(0.10));

        sr.update(dec!(100.00));
        sr.update(dec!(100.50));
        sr.update(dec!(100.00));

        let resistance = sr.get_resistance(dec!(100.25));
        assert!(resistance.is_some());
        assert!(resistance.unwrap() > dec!(100.25));
    }

    #[test]
    fn test_is_near_level() {
        let sr = SRLevels::new(dec!(0.10));

        // Within half granularity should be "near"
        assert!(sr.is_near(dec!(100.00), dec!(100.04)));
        assert!(sr.is_near(dec!(100.00), dec!(99.96)));

        // Outside half granularity should not be "near"
        assert!(!sr.is_near(dec!(100.00), dec!(100.10)));
        assert!(!sr.is_near(dec!(100.00), dec!(99.90)));
    }

    #[test]
    fn test_bar_update() {
        let mut sr = SRLevels::new(dec!(1.0));

        // Bullish bar: open=10, high=15, low=9, close=14
        sr.update_bar(dec!(10), dec!(15), dec!(9), dec!(14));

        // All ranges from 9 to 15 should be touched
        assert!(sr.score_at(dec!(9)) < 0);
        assert!(sr.score_at(dec!(12)) < 0);
        assert!(sr.score_at(dec!(15)) < 0);
    }

    #[test]
    fn test_pftq_example() {
        // From the spec:
        // Price moves $9 → $10: range $9-$10 gets score = -1
        // Price moves $10 → $9: range $9-$10 gets score = -2
        // Price moves $9 → $8.15: range $8.15-$9 gets score = -1
        // Price moves $8.15 → $10:
        //   - Range $8.15-$9: score = -2
        //   - Range $9-$10: score = -3

        let mut sr = SRLevels::new(dec!(1.0)); // Using $1 granularity

        sr.update(dec!(9));
        sr.update(dec!(10));
        assert_eq!(sr.score_at(dec!(9)), -1);
        assert_eq!(sr.score_at(dec!(10)), -1);

        sr.update(dec!(9));
        assert_eq!(sr.score_at(dec!(9)), -2);
        assert_eq!(sr.score_at(dec!(10)), -2);

        sr.update(dec!(8));
        assert_eq!(sr.score_at(dec!(8)), -1);
        assert_eq!(sr.score_at(dec!(9)), -3);

        sr.update(dec!(10));
        assert_eq!(sr.score_at(dec!(8)), -2);
        assert_eq!(sr.score_at(dec!(9)), -4);
        assert_eq!(sr.score_at(dec!(10)), -3);

        // Untraveled area at 11 stays at 0 = strongest resistance
        assert_eq!(sr.score_at(dec!(11)), 0);
    }

    #[test]
    fn test_calculate_atr() {
        // Test with simple bars: consistent $5 range
        let bars = vec![
            (dec!(100), dec!(102), dec!(98), dec!(101)),  // TR = 4 (H-L)
            (dec!(101), dec!(105), dec!(100), dec!(103)), // TR = 5 (H-L)
            (dec!(103), dec!(106), dec!(101), dec!(105)), // TR = 5 (H-L)
        ];

        let atr = calculate_atr(&bars);
        // ATR should be approximately (4+5+5)/3 = 4.67 but only uses n-1 bars
        // Actually: uses bars[1..] with prev_close from bars[i-1]
        // Bar 1: H=105, L=100, PC=101 -> TR = max(5, 4, 1) = 5
        // Bar 2: H=106, L=101, PC=103 -> TR = max(5, 3, 2) = 5
        // ATR = (5+5)/2 = 5
        assert_eq!(atr, dec!(5));
    }

    #[test]
    fn test_calculate_atr_insufficient_data() {
        // Single bar should return 0
        let single_bar = vec![(dec!(100), dec!(101), dec!(99), dec!(100))];
        assert_eq!(calculate_atr(&single_bar), Decimal::ZERO);

        // Empty should return 0
        let empty: Vec<(Decimal, Decimal, Decimal, Decimal)> = vec![];
        assert_eq!(calculate_atr(&empty), Decimal::ZERO);
    }

    #[test]
    fn test_granularity_from_atr() {
        // ATR of $10 -> raw = $1.0 -> should snap to $1.00
        assert_eq!(granularity_from_atr(dec!(10)), dec!(1.00));

        // ATR of $5 -> raw = $0.50 -> should snap to $0.50
        assert_eq!(granularity_from_atr(dec!(5)), dec!(0.50));

        // ATR of $2.5 -> raw = $0.25 -> should snap to $0.25
        assert_eq!(granularity_from_atr(dec!(2.5)), dec!(0.25));

        // ATR of $1 -> raw = $0.10 -> should snap to $0.10
        assert_eq!(granularity_from_atr(dec!(1)), dec!(0.10));

        // ATR of $100 -> raw = $10 -> should snap to $10.00
        assert_eq!(granularity_from_atr(dec!(100)), dec!(10.00));

        // Zero ATR should return default
        assert_eq!(granularity_from_atr(Decimal::ZERO), dec!(0.10));
    }

    #[test]
    fn test_granularity_from_atr_edge_cases() {
        // Very small ATR (penny stock)
        assert_eq!(granularity_from_atr(dec!(0.05)), dec!(0.01));

        // Very large ATR (BTC-like)
        assert_eq!(granularity_from_atr(dec!(1000)), dec!(100.00));
    }
}
