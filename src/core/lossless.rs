//! Lossless Algorithms Module
//!
//! Implementation of pftq's "lossless" algorithm philosophy.
//!
//! "The reason I call it a 'lossless' algorithm is because it doesn't estimate
//! anything or use any seeded values/thresholds. It is analogous to lossless
//! audio/image file formats. There is no sampling or use of statistics, just
//! a 1-to-1 map of where price moves most and least freely." - pftq
//!
//! # Key Principles
//!
//! 1. **No Parameters**: Algorithms discover everything from data
//! 2. **No Thresholds**: No magic numbers like "RSI > 70"
//! 3. **No Lookback Windows**: No "20-period average"
//! 4. **Direct Mapping**: Raw observation, not estimation

use std::collections::HashMap;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};

use super::types::{Candle, Trend, Momentum, VolumeState, Observation};

/// Lossless Support/Resistance Algorithm
///
/// On each price move, the price range traveled loses a point in score.
/// Untraveled ranges remain at 0 (strongest support/resistance).
///
/// Score of 0 = Never crossed = Strongest level
/// Score of -100 = Crossed 100 times = Weak level
#[derive(Debug, Clone)]
pub struct LosslessLevels {
    /// Price level -> score (0 is strongest, negative is weaker)
    levels: HashMap<i64, i64>,
    /// Tick size for rounding prices to levels
    tick_size: Decimal,
    /// Last processed price
    last_price: Option<Decimal>,
    /// Total number of updates
    update_count: u64,
    /// Decay factor for forex/crypto (recent crossings matter more)
    /// Set to 0 for stocks (no decay)
    decay_factor: f64,
}

impl LosslessLevels {
    /// Create a new LosslessLevels tracker
    ///
    /// # Arguments
    /// * `tick_size` - Minimum price increment (e.g., 0.01 for Gold)
    /// * `decay_factor` - Decay for old levels (0.0 = no decay, 0.001 = slow decay)
    pub fn new(tick_size: Decimal, decay_factor: f64) -> Self {
        Self {
            levels: HashMap::new(),
            tick_size,
            last_price: None,
            update_count: 0,
            decay_factor,
        }
    }
    
    /// Convert a price to a level key (integer for HashMap efficiency)
    fn price_to_level(&self, price: Decimal) -> i64 {
        let scaled = price / self.tick_size;
        scaled.round().to_string().parse().unwrap_or(0)
    }
    
    /// Convert a level key back to price
    fn level_to_price(&self, level: i64) -> Decimal {
        Decimal::from(level) * self.tick_size
    }
    
    /// Update levels with a new price
    ///
    /// Every price range crossed loses a point.
    pub fn update(&mut self, price: Decimal) {
        let current_level = self.price_to_level(price);
        
        if let Some(last_price) = self.last_price {
            let last_level = self.price_to_level(last_price);
            
            if current_level != last_level {
                // Mark all levels between last and current as crossed
                let (start, end) = if current_level > last_level {
                    (last_level, current_level)
                } else {
                    (current_level, last_level)
                };
                
                for level in start..=end {
                    // Decrease score (more negative = more traveled = weaker)
                    let score = self.levels.entry(level).or_insert(0);
                    *score -= 1;
                }
            }
        }
        
        self.last_price = Some(price);
        self.update_count += 1;
        
        // Apply decay to old levels (for forex/crypto where recent matters more)
        if self.decay_factor > 0.0 && self.update_count % 100 == 0 {
            self.apply_decay();
        }
    }
    
    /// Update with a full candle (processes high, low, close)
    pub fn update_candle(&mut self, candle: &Candle) {
        // Process the candle's price journey
        self.update(candle.open);
        self.update(candle.high);
        self.update(candle.low);
        self.update(candle.close);
    }
    
    /// Apply decay to old levels (forex/crypto forget faster)
    fn apply_decay(&mut self) {
        if self.decay_factor == 0.0 {
            return;
        }
        
        for score in self.levels.values_mut() {
            // Move score toward 0 (weaker levels recover slightly)
            let decay = (*score as f64 * self.decay_factor) as i64;
            if decay.abs() > 0 {
                *score -= decay.signum();
            }
        }
    }
    
    /// Get the N strongest support levels below the given price
    ///
    /// Strongest = score closest to 0 (least traveled)
    pub fn get_support_levels(&self, current_price: Decimal, count: usize) -> Vec<(Decimal, i64)> {
        let current_level = self.price_to_level(current_price);
        
        let mut supports: Vec<_> = self.levels
            .iter()
            .filter(|(level, _)| **level < current_level)
            .map(|(level, score)| (*level, *score))
            .collect();
        
        // Sort by score descending (closest to 0 first = strongest)
        supports.sort_by(|a, b| b.1.cmp(&a.1));
        
        supports
            .into_iter()
            .take(count)
            .map(|(level, score)| (self.level_to_price(level), score))
            .collect()
    }
    
    /// Get the N strongest resistance levels above the given price
    pub fn get_resistance_levels(&self, current_price: Decimal, count: usize) -> Vec<(Decimal, i64)> {
        let current_level = self.price_to_level(current_price);
        
        let mut resistances: Vec<_> = self.levels
            .iter()
            .filter(|(level, _)| **level > current_level)
            .map(|(level, score)| (*level, *score))
            .collect();
        
        // Sort by score descending (closest to 0 first = strongest)
        resistances.sort_by(|a, b| b.1.cmp(&a.1));
        
        resistances
            .into_iter()
            .take(count)
            .map(|(level, score)| (self.level_to_price(level), score))
            .collect()
    }
    
    /// Get the nearest strong support level
    pub fn nearest_support(&self, current_price: Decimal) -> Option<Decimal> {
        let supports = self.get_support_levels(current_price, 10);
        
        // Find nearest one within 2% of current price
        let threshold = current_price * dec!(0.02);
        
        for (price, _score) in &supports {
            if current_price - *price < threshold {
                return Some(*price);
            }
        }
        
        supports.first().map(|(price, _)| *price)
    }
    
    /// Get the nearest strong resistance level
    pub fn nearest_resistance(&self, current_price: Decimal) -> Option<Decimal> {
        let resistances = self.get_resistance_levels(current_price, 10);
        
        let threshold = current_price * dec!(0.02);
        
        for (price, _score) in &resistances {
            if *price - current_price < threshold {
                return Some(*price);
            }
        }
        
        resistances.first().map(|(price, _)| *price)
    }
    
    /// Check if price is near a strong support level
    pub fn is_near_support(&self, current_price: Decimal, threshold_pct: Decimal) -> bool {
        if let Some(support) = self.nearest_support(current_price) {
            let distance_pct = (current_price - support) / current_price;
            return distance_pct < threshold_pct;
        }
        false
    }
    
    /// Check if price is near a strong resistance level
    pub fn is_near_resistance(&self, current_price: Decimal, threshold_pct: Decimal) -> bool {
        if let Some(resistance) = self.nearest_resistance(current_price) {
            let distance_pct = (resistance - current_price) / current_price;
            return distance_pct < threshold_pct;
        }
        false
    }
}

/// Lossless Trend Detection
///
/// No EMAs. No MAs. Just counting higher highs and lower lows.
///
/// A human doesn't calculate a 21-period EMA in their head.
/// They just see "price is going up" or "price is going down".
#[derive(Debug, Clone)]
pub struct TrendObserver {
    highs: Vec<Decimal>,
    lows: Vec<Decimal>,
    max_history: usize,
}

impl TrendObserver {
    pub fn new(max_history: usize) -> Self {
        Self {
            highs: Vec::with_capacity(max_history),
            lows: Vec::with_capacity(max_history),
            max_history,
        }
    }
    
    /// Update with a new candle
    pub fn update(&mut self, candle: &Candle) {
        self.highs.push(candle.high);
        self.lows.push(candle.low);
        
        // Trim to max history
        if self.highs.len() > self.max_history {
            self.highs.remove(0);
            self.lows.remove(0);
        }
    }
    
    /// Detect trend using pure counts (lossless)
    ///
    /// No threshold except 1.5x ratio for clear trend
    pub fn detect(&self) -> Trend {
        if self.highs.len() < 4 {
            return Trend::Neutral;
        }
        
        let recent_highs = &self.highs[self.highs.len().saturating_sub(10)..];
        let recent_lows = &self.lows[self.lows.len().saturating_sub(10)..];
        
        // Count higher highs
        let higher_highs: usize = recent_highs
            .windows(2)
            .filter(|w| w[1] > w[0])
            .count();
        
        // Count lower lows
        let lower_lows: usize = recent_lows
            .windows(2)
            .filter(|w| w[1] < w[0])
            .count();
        
        // Count higher lows (bullish)
        let higher_lows: usize = recent_lows
            .windows(2)
            .filter(|w| w[1] > w[0])
            .count();
        
        // Count lower highs (bearish)
        let lower_highs: usize = recent_highs
            .windows(2)
            .filter(|w| w[1] < w[0])
            .count();
        
        // Uptrend: Higher highs AND higher lows dominate
        // Threshold: 50% more occurrences (the only "parameter")
        let hh_hl = higher_highs + higher_lows;
        let ll_lh = lower_lows + lower_highs;
        
        if hh_hl > 0 && ll_lh > 0 {
            let ratio = hh_hl as f64 / ll_lh as f64;
            
            if ratio > 1.5 {
                return Trend::Up;
            } else if ratio < 0.67 {
                return Trend::Down;
            }
        } else if hh_hl > 0 && ll_lh == 0 {
            return Trend::Up;
        } else if ll_lh > 0 && hh_hl == 0 {
            return Trend::Down;
        }
        
        Trend::Neutral
    }
    
    /// Get the recent swing high
    pub fn recent_high(&self) -> Option<Decimal> {
        self.highs.iter().copied().max_by(|a, b| a.cmp(b))
    }
    
    /// Get the recent swing low
    pub fn recent_low(&self) -> Option<Decimal> {
        self.lows.iter().copied().min_by(|a, b| a.cmp(b))
    }
}

/// Lossless Momentum Detection
///
/// No RSI. No MACD. Just comparing move sizes.
///
/// Question: Are the bulls/bears getting stronger or weaker?
#[derive(Debug, Clone)]
pub struct MomentumObserver {
    /// (direction: 1=up, -1=down, 0=neutral, magnitude)
    moves: Vec<(i8, Decimal)>,
    max_history: usize,
}

impl MomentumObserver {
    pub fn new(max_history: usize) -> Self {
        Self {
            moves: Vec::with_capacity(max_history),
            max_history,
        }
    }
    
    /// Update with a new candle
    pub fn update(&mut self, candle: &Candle) {
        let move_size = candle.close - candle.open;
        let magnitude = move_size.abs();
        let direction = if move_size > Decimal::ZERO {
            1
        } else if move_size < Decimal::ZERO {
            -1
        } else {
            0
        };
        
        self.moves.push((direction, magnitude));
        
        if self.moves.len() > self.max_history {
            self.moves.remove(0);
        }
    }
    
    /// Detect momentum state (lossless)
    pub fn detect(&self) -> Momentum {
        if self.moves.len() < 5 {
            return Momentum::Neutral;
        }
        
        let recent = &self.moves[self.moves.len().saturating_sub(5)..];
        
        let up_count = recent.iter().filter(|(d, _)| *d > 0).count();
        let down_count = recent.iter().filter(|(d, _)| *d < 0).count();
        
        let up_magnitude: Decimal = recent
            .iter()
            .filter(|(d, _)| *d > 0)
            .map(|(_, m)| *m)
            .sum();
        
        let down_magnitude: Decimal = recent
            .iter()
            .filter(|(d, _)| *d < 0)
            .map(|(_, m)| *m)
            .sum();
        
        // Strong momentum = 4+ moves in one direction AND 2x magnitude
        if up_count >= 4 && up_magnitude > down_magnitude * dec!(2) {
            Momentum::StrongUp
        } else if up_count >= 3 {
            Momentum::Up
        } else if down_count >= 4 && down_magnitude > up_magnitude * dec!(2) {
            Momentum::StrongDown
        } else if down_count >= 3 {
            Momentum::Down
        } else {
            Momentum::Neutral
        }
    }
    
    /// Check if momentum aligns with the given trend
    pub fn aligns_with(&self, trend: Trend) -> bool {
        let momentum = self.detect();
        
        match trend {
            Trend::Up => matches!(momentum, Momentum::Up | Momentum::StrongUp),
            Trend::Down => matches!(momentum, Momentum::Down | Momentum::StrongDown),
            Trend::Neutral => true,
        }
    }
    
    /// Check if momentum is slowing (moves getting smaller)
    pub fn is_slowing(&self) -> bool {
        if self.moves.len() < 6 {
            return false;
        }
        
        let recent: Vec<Decimal> = self.moves
            .iter()
            .rev()
            .take(3)
            .map(|(_, m)| *m)
            .collect();
        
        let previous: Vec<Decimal> = self.moves
            .iter()
            .rev()
            .skip(3)
            .take(3)
            .map(|(_, m)| *m)
            .collect();
        
        let avg_recent: Decimal = recent.iter().sum::<Decimal>() / Decimal::from(recent.len());
        let avg_previous: Decimal = previous.iter().sum::<Decimal>() / Decimal::from(previous.len());
        
        // 30% reduction = slowing
        avg_recent < avg_previous * dec!(0.7)
    }
}

/// Lossless Volume Analysis
///
/// No "volume > 20-day average" threshold.
/// Just: Where does today's volume rank among recent days?
#[derive(Debug, Clone)]
pub struct VolumeObserver {
    volumes: Vec<Decimal>,
    max_history: usize,
}

impl VolumeObserver {
    pub fn new(max_history: usize) -> Self {
        Self {
            volumes: Vec::with_capacity(max_history),
            max_history,
        }
    }
    
    /// Update with a new candle
    pub fn update(&mut self, candle: &Candle) {
        self.volumes.push(candle.volume);
        
        if self.volumes.len() > self.max_history {
            self.volumes.remove(0);
        }
    }
    
    /// Detect volume state using percentile (lossless)
    pub fn detect(&self) -> VolumeState {
        if self.volumes.len() < 5 {
            return VolumeState::Normal;
        }
        
        let current = self.volumes.last().copied().unwrap_or(Decimal::ZERO);
        
        // Calculate percentile: how many recent volumes are lower?
        let lower_count = self.volumes.iter().filter(|v| **v < current).count();
        let percentile = lower_count as f64 / self.volumes.len() as f64;
        
        if percentile > 0.9 {
            VolumeState::Spike
        } else if percentile < 0.1 {
            VolumeState::Dead
        } else {
            VolumeState::Normal
        }
    }
}

/// Bounce Detector
///
/// Detects price bounces from support/resistance levels.
/// Pattern-based, no parameters.
#[derive(Debug, Clone)]
pub struct BounceDetector {
    candles: Vec<Candle>,
    max_candles: usize,
}

impl BounceDetector {
    pub fn new(max_candles: usize) -> Self {
        Self {
            candles: Vec::with_capacity(max_candles),
            max_candles,
        }
    }
    
    /// Update with a new candle
    pub fn update(&mut self, candle: Candle) {
        self.candles.push(candle);
        
        if self.candles.len() > self.max_candles {
            self.candles.remove(0);
        }
    }
    
    /// Detect bounce up from support
    ///
    /// Pattern: Down move → Low point → Strong up move
    pub fn bounce_up(&self) -> bool {
        if self.candles.len() < 3 {
            return false;
        }
        
        let c1 = &self.candles[self.candles.len() - 3];
        let c2 = &self.candles[self.candles.len() - 2]; // The low point
        let c3 = &self.candles[self.candles.len() - 1];
        
        // c2 should have lowest low
        if c2.low > c1.low || c2.low > c3.low {
            return false;
        }
        
        // c3 should close higher than c2
        if c3.close <= c2.close {
            return false;
        }
        
        // c3 should be bullish
        if !c3.is_bullish() {
            return false;
        }
        
        // Bounce should be meaningful (at least 50% of candle range)
        let bounce_size = c3.close - c2.low;
        let candle_range = c3.range();
        
        if candle_range > Decimal::ZERO {
            let bounce_ratio = bounce_size / candle_range;
            return bounce_ratio > dec!(0.5);
        }
        
        false
    }
    
    /// Detect bounce down from resistance
    ///
    /// Pattern: Up move → High point → Strong down move
    pub fn bounce_down(&self) -> bool {
        if self.candles.len() < 3 {
            return false;
        }
        
        let c1 = &self.candles[self.candles.len() - 3];
        let c2 = &self.candles[self.candles.len() - 2]; // The high point
        let c3 = &self.candles[self.candles.len() - 1];
        
        // c2 should have highest high
        if c2.high < c1.high || c2.high < c3.high {
            return false;
        }
        
        // c3 should close lower than c2
        if c3.close >= c2.close {
            return false;
        }
        
        // c3 should be bearish
        if !c3.is_bearish() {
            return false;
        }
        
        true
    }
}

/// Complete Market Observer
///
/// Combines all lossless observers into a unified view.
#[derive(Debug)]
pub struct MarketObserver {
    pub levels: LosslessLevels,
    pub trend: TrendObserver,
    pub momentum: MomentumObserver,
    pub volume: VolumeObserver,
    pub bounce: BounceDetector,
}

impl MarketObserver {
    /// Create a new market observer
    ///
    /// # Arguments
    /// * `tick_size` - Minimum price increment
    /// * `is_forex` - If true, use decay for levels (forex forgets faster)
    pub fn new(tick_size: Decimal, is_forex: bool) -> Self {
        let decay_factor = if is_forex { 0.001 } else { 0.0 };
        
        Self {
            levels: LosslessLevels::new(tick_size, decay_factor),
            trend: TrendObserver::new(50),
            momentum: MomentumObserver::new(30),
            volume: VolumeObserver::new(20),
            bounce: BounceDetector::new(10),
        }
    }
    
    /// Update all observers with a new candle
    pub fn update(&mut self, candle: &Candle) {
        self.levels.update_candle(candle);
        self.trend.update(candle);
        self.momentum.update(candle);
        self.volume.update(candle);
        self.bounce.update(candle.clone());
    }
    
    /// Get complete market observation
    pub fn observe(&self, current_price: Decimal) -> Observation {
        let threshold = dec!(0.001); // 0.1% for near support/resistance
        
        Observation {
            trend: self.trend.detect(),
            momentum: self.momentum.detect(),
            volume_state: self.volume.detect(),
            near_support: self.levels.is_near_support(current_price, threshold),
            near_resistance: self.levels.is_near_resistance(current_price, threshold),
            nearest_support: self.levels.nearest_support(current_price),
            nearest_resistance: self.levels.nearest_resistance(current_price),
            bounce_up: self.bounce.bounce_up(),
            bounce_down: self.bounce.bounce_down(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_lossless_levels() {
        let mut levels = LosslessLevels::new(dec!(0.01), 0.0);
        
        // Simulate price movement
        levels.update(dec!(100.00));
        levels.update(dec!(100.50)); // Cross 50 levels
        levels.update(dec!(100.00)); // Cross back
        
        // Levels between 100.00 and 100.50 should have score -2
        let supports = levels.get_support_levels(dec!(100.25), 5);
        assert!(!supports.is_empty());
    }
    
    #[test]
    fn test_trend_detection() {
        let mut trend = TrendObserver::new(50);
        
        // Simulate uptrend: Higher highs and higher lows
        for i in 0..10 {
            let base = Decimal::from(100 + i * 2);
            let candle = Candle::new(
                Utc::now(),
                base,
                base + dec!(5),
                base - dec!(1),
                base + dec!(3),
                dec!(1000),
            );
            trend.update(&candle);
        }
        
        assert_eq!(trend.detect(), Trend::Up);
    }
    
    #[test]
    fn test_momentum_detection() {
        let mut momentum = MomentumObserver::new(30);
        
        // 5 bullish candles
        for _ in 0..5 {
            let candle = Candle::new(
                Utc::now(),
                dec!(100),
                dec!(105),
                dec!(99),
                dec!(104),
                dec!(1000),
            );
            momentum.update(&candle);
        }
        
        let result = momentum.detect();
        assert!(matches!(result, Momentum::Up | Momentum::StrongUp));
    }
}
