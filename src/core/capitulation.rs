//! Volume Capitulation Detection
//!
//! "BOT" labels on Tech Trader charts indicate "proprietary bottom signals
//! based on price and volume capitulation"
//!
//! # Concept
//!
//! - High volume = lots of participants
//! - Down day = sellers exhausted
//! - Combined = potential bottom
//!
//! This is NOT statistical. It's observational:
//! "The volume today is much higher than usual AND price went down"
//! A human can see this on a chart instantly.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Volume tracker for capitulation detection
///
/// LOSSLESS: No thresholds. Uses ranking instead of ratios.
/// A volume "spike" is simply the highest volume in the lookback window.
#[derive(Debug, Clone)]
pub struct VolumeTracker {
    /// Recent volume values (rolling window)
    volumes: Vec<u64>,
    /// Window size - derived from standard chart view (not a strategy parameter)
    window_size: usize,
}

impl VolumeTracker {
    /// Create a new volume tracker
    pub fn new() -> Self {
        Self {
            volumes: Vec::with_capacity(20),
            window_size: 20, // Standard chart lookback - infrastructure, not strategy
        }
    }

    /// Update with new volume
    pub fn update(&mut self, volume: u64) {
        if self.volumes.len() >= self.window_size {
            self.volumes.remove(0);
        }
        self.volumes.push(volume);
    }

    /// LOSSLESS: Check if this volume is the highest in the window
    /// No threshold - pure observation
    pub fn is_highest(&self, volume: u64) -> bool {
        if self.volumes.is_empty() {
            return false;
        }
        self.volumes.iter().all(|&v| volume > v)
    }

    /// LOSSLESS: Get the rank of this volume (1 = highest)
    /// Pure counting - no thresholds
    pub fn rank(&self, volume: u64) -> usize {
        self.volumes.iter().filter(|&&v| v > volume).count() + 1
    }

    /// LOSSLESS: Check if this volume is in the top N
    /// N is derived from window size (top 10% = top 2 of 20)
    pub fn is_top_n(&self, volume: u64, n: usize) -> bool {
        self.rank(volume) <= n
    }

    /// Check if we have any context (no fixed minimum)
    pub fn has_context(&self) -> bool {
        !self.volumes.is_empty()
    }

    /// Get the 20-bar simple moving average of volume (for logging only)
    pub fn average(&self) -> f64 {
        if self.volumes.is_empty() {
            return 0.0;
        }
        self.volumes.iter().sum::<u64>() as f64 / self.volumes.len() as f64
    }

    /// Get the volume ratio (for logging/display only, NOT for signals)
    pub fn ratio(&self, volume: u64) -> f64 {
        let avg = self.average();
        if avg <= 0.0 {
            return 0.0;
        }
        volume as f64 / avg
    }

    /// Get window fill percentage (for logging)
    pub fn fill_pct(&self) -> f64 {
        self.volumes.len() as f64 / self.window_size as f64 * 100.0
    }
}

impl Default for VolumeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Capitulation signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapitulationSignal {
    /// Volume spike on a down day (potential bottom)
    BuyCapitulation,
    /// Volume spike on an up day (potential top)
    SellCapitulation,
    /// No capitulation signal
    None,
}

/// LOSSLESS: Check for volume capitulation using ranking
/// No thresholds - spike = highest in window
pub fn check_capitulation_lossless(
    volume: u64,
    tracker: &VolumeTracker,
    price_change: Decimal,
) -> CapitulationSignal {
    // LOSSLESS: Spike = highest volume in window
    if !tracker.is_highest(volume) {
        return CapitulationSignal::None;
    }

    // Check price direction
    if price_change < Decimal::ZERO {
        CapitulationSignal::BuyCapitulation
    } else if price_change > Decimal::ZERO {
        CapitulationSignal::SellCapitulation
    } else {
        CapitulationSignal::None
    }
}

/// LOSSLESS helper: buy capitulation = highest volume + down day
pub fn is_buy_capitulation_lossless(
    volume: u64,
    tracker: &VolumeTracker,
    open: Decimal,
    close: Decimal,
) -> bool {
    tracker.is_highest(volume) && close < open
}

/// LOSSLESS helper: sell capitulation = highest volume + up day
pub fn is_sell_capitulation_lossless(
    volume: u64,
    tracker: &VolumeTracker,
    open: Decimal,
    close: Decimal,
) -> bool {
    tracker.is_highest(volume) && close > open
}

#[deprecated(note = "Use check_capitulation_lossless instead - no 2x threshold")]
/// Check for volume capitulation
///
/// # Arguments
/// * `current_volume` - Today's volume
/// * `average_volume` - Average daily volume (e.g., 20-day SMA)
/// * `price_change` - Today's price change (close - open)
///
/// # Returns
/// Capitulation signal if conditions are met
pub fn check_capitulation(
    current_volume: u64,
    average_volume: f64,
    price_change: Decimal,
) -> CapitulationSignal {
    // Need meaningful average
    if average_volume <= 0.0 {
        return CapitulationSignal::None;
    }

    // Check for volume spike (> 2x average)
    let volume_spike = current_volume as f64 > average_volume * 2.0;

    if !volume_spike {
        return CapitulationSignal::None;
    }

    // Check price direction
    if price_change < Decimal::ZERO {
        CapitulationSignal::BuyCapitulation // High volume selloff = potential bottom
    } else if price_change > Decimal::ZERO {
        CapitulationSignal::SellCapitulation // High volume rally = potential top
    } else {
        CapitulationSignal::None
    }
}

#[deprecated(note = "Use is_buy_capitulation_lossless instead")]
/// Simple helper to detect capitulation from a bar
pub fn is_buy_capitulation(
    volume: u64,
    avg_volume: f64,
    open: Decimal,
    close: Decimal,
) -> bool {
    let change = close - open;
    #[allow(deprecated)]
    matches!(
        check_capitulation(volume, avg_volume, change),
        CapitulationSignal::BuyCapitulation
    )
}

#[deprecated(note = "Use is_sell_capitulation_lossless instead")]
/// Simple helper to detect sell capitulation from a bar
pub fn is_sell_capitulation(
    volume: u64,
    avg_volume: f64,
    open: Decimal,
    close: Decimal,
) -> bool {
    let change = close - open;
    #[allow(deprecated)]
    matches!(
        check_capitulation(volume, avg_volume, change),
        CapitulationSignal::SellCapitulation
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_volume_tracker_ranking() {
        let mut tracker = VolumeTracker::new();

        // Add volumes: 1000, 2000, 3000, 4000, 5000
        for i in 1..=5 {
            tracker.update(i * 1000);
        }

        // 6000 is highest -> rank 1
        assert_eq!(tracker.rank(6000), 1);
        assert!(tracker.is_highest(6000));

        // 5000 ties highest -> rank 1 (not strictly greater)
        assert_eq!(tracker.rank(5000), 1);

        // 4500 is second highest -> rank 2
        assert_eq!(tracker.rank(4500), 2);

        // 1000 is lowest -> rank 5
        assert_eq!(tracker.rank(1000), 5);
    }

    #[test]
    fn test_lossless_capitulation() {
        let mut tracker = VolumeTracker::new();

        // Build history
        for _ in 0..10 {
            tracker.update(1000);
        }

        // Highest volume + down day = buy capitulation
        assert!(is_buy_capitulation_lossless(2000, &tracker, dec!(100), dec!(95)));

        // NOT highest = no capitulation
        assert!(!is_buy_capitulation_lossless(500, &tracker, dec!(100), dec!(95)));

        // Highest + up day = sell capitulation, not buy
        assert!(!is_buy_capitulation_lossless(2000, &tracker, dec!(100), dec!(105)));
        assert!(is_sell_capitulation_lossless(2000, &tracker, dec!(100), dec!(105)));
    }

    #[test]
    fn test_has_context() {
        let mut tracker = VolumeTracker::new();

        // Empty = no context
        assert!(!tracker.has_context());

        // Any data = has context
        tracker.update(1000);
        assert!(tracker.has_context());
    }

    #[test]
    fn test_volume_tracker_average() {
        let mut tracker = VolumeTracker::new();

        // Add 5 bars of volume
        for i in 1..=5 {
            tracker.update(i * 1000);
        }

        // Average should be (1000+2000+3000+4000+5000)/5 = 3000
        assert!((tracker.average() - 3000.0).abs() < 0.01);
    }

    #[test]
    fn test_tracker_rolling_window() {
        let mut tracker = VolumeTracker::new();

        // Fill with 20 values of 1000
        for _ in 0..20 {
            tracker.update(1000);
        }

        assert!((tracker.average() - 1000.0).abs() < 0.01);

        // Add 10 values of 2000 (replaces half the window)
        for _ in 0..10 {
            tracker.update(2000);
        }

        // Now average should be 1500 (10x1000 + 10x2000) / 20
        assert!((tracker.average() - 1500.0).abs() < 0.01);
    }

    #[test]
    fn test_is_top_n() {
        let mut tracker = VolumeTracker::new();

        // Add volumes: 1000, 2000, 3000, 4000, 5000
        for i in 1..=5 {
            tracker.update(i * 1000);
        }

        // 6000 should be in top 1, top 2, top 3
        assert!(tracker.is_top_n(6000, 1));
        assert!(tracker.is_top_n(6000, 2));
        assert!(tracker.is_top_n(6000, 3));

        // 4500 is rank 2, should be in top 2 and 3 but not top 1
        assert!(!tracker.is_top_n(4500, 1));
        assert!(tracker.is_top_n(4500, 2));
        assert!(tracker.is_top_n(4500, 3));
    }
}
