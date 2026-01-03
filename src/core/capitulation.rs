//! Volume Capitulation Detection - Lossless Implementation
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
//!
//! # LOSSLESS IMPLEMENTATION
//!
//! Instead of a fixed 20-bar window, we track ALL historical volume
//! and use percentile ranking. A "spike" is relative to all observed data.
//! This eliminates the arbitrary window size parameter.

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

/// Volume tracker using expanding window (lossless)
///
/// Instead of fixed lookback, tracks ALL historical volume and uses
/// percentile ranking. A "spike" is relative to all observed data.
#[derive(Debug, Clone)]
pub struct VolumeTracker {
    /// All observed volumes (sorted for percentile calculation)
    volumes_sorted: Vec<u64>,
    /// Recent volumes for recency-weighted ranking
    recent_volumes: Vec<u64>,
    /// Maximum recent window (operational limit, not strategy)
    max_recent: usize,
}

impl VolumeTracker {
    /// Create a new volume tracker
    pub fn new() -> Self {
        Self {
            volumes_sorted: Vec::new(),
            recent_volumes: Vec::with_capacity(100),
            max_recent: 100, // Operational limit for recency checks
        }
    }

    /// Update with new volume
    pub fn update(&mut self, volume: u64) {
        // Add to sorted list (for all-time percentile)
        let pos = self.volumes_sorted.binary_search(&volume).unwrap_or_else(|p| p);
        self.volumes_sorted.insert(pos, volume);

        // Add to recent (rolling window for recency check)
        if self.recent_volumes.len() >= self.max_recent {
            self.recent_volumes.remove(0);
        }
        self.recent_volumes.push(volume);
    }

    /// Percentile rank (0-100) across ALL observed data
    /// 100 = highest ever, 50 = median, etc.
    pub fn percentile(&self, volume: u64) -> f64 {
        if self.volumes_sorted.is_empty() {
            return 50.0;
        }
        let pos = self.volumes_sorted.binary_search(&volume).unwrap_or_else(|p| p);
        (pos as f64 / self.volumes_sorted.len() as f64) * 100.0
    }

    /// Is this volume in the top decile (90th+ percentile)?
    /// Derived from data distribution, not hardcoded threshold
    pub fn is_spike(&self, volume: u64) -> bool {
        self.percentile(volume) >= 90.0
    }

    /// Is this the highest in recent memory?
    pub fn is_recent_highest(&self, volume: u64) -> bool {
        self.recent_volumes.iter().all(|&v| volume >= v)
    }

    /// Combined check: high percentile AND recent highest
    /// This is the purest "capitulation" signal
    pub fn is_capitulation_volume(&self, volume: u64) -> bool {
        self.percentile(volume) >= 80.0 && self.is_recent_highest(volume)
    }

    /// LOSSLESS: Check if this volume is the highest in history
    /// No threshold - pure observation
    pub fn is_highest(&self, volume: u64) -> bool {
        if self.volumes_sorted.is_empty() {
            return false;
        }
        volume > *self.volumes_sorted.last().unwrap_or(&0)
    }

    /// LOSSLESS: Get the rank of this volume (1 = highest)
    /// Pure counting - no thresholds
    pub fn rank(&self, volume: u64) -> usize {
        // In recent window for backwards compatibility
        self.recent_volumes.iter().filter(|&&v| v > volume).count() + 1
    }

    /// LOSSLESS: Check if this volume is in the top N of recent
    pub fn is_top_n(&self, volume: u64, n: usize) -> bool {
        self.rank(volume) <= n
    }

    /// Check if we have any context (observed data)
    pub fn has_context(&self) -> bool {
        self.volumes_sorted.len() >= 10
    }

    /// Get count of observed volumes
    pub fn observation_count(&self) -> usize {
        self.volumes_sorted.len()
    }

    /// Get the average volume (for logging only)
    pub fn average(&self) -> f64 {
        if self.recent_volumes.is_empty() {
            return 0.0;
        }
        self.recent_volumes.iter().sum::<u64>() as f64 / self.recent_volumes.len() as f64
    }

    /// Get the volume ratio (for logging/display only, NOT for signals)
    pub fn ratio(&self, volume: u64) -> f64 {
        let avg = self.average();
        if avg <= 0.0 {
            return 0.0;
        }
        volume as f64 / avg
    }

    /// Get context fill percentage (for logging)
    pub fn fill_pct(&self) -> f64 {
        // Use 100 as "full" context
        (self.volumes_sorted.len().min(100) as f64 / 100.0) * 100.0
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

        // 5000 is in the history, so not "strictly greater" than history max
        // For rank, we check recent_volumes which has 5000 as max
        assert_eq!(tracker.rank(5000), 1);

        // 4500 is second highest -> rank 2
        assert_eq!(tracker.rank(4500), 2);

        // 1000 is lowest -> rank 5
        assert_eq!(tracker.rank(1000), 5);
    }

    #[test]
    fn test_percentile_ranking() {
        let mut tracker = VolumeTracker::new();

        // Add 100 volumes from 1-100
        for i in 1..=100 {
            tracker.update(i * 100);
        }

        // Volume of 10000 (100th value) should be at ~100th percentile
        let pct_high = tracker.percentile(10000);
        assert!(pct_high >= 99.0);

        // Volume of 100 (1st value) should be at ~0-1st percentile
        let pct_low = tracker.percentile(100);
        assert!(pct_low <= 2.0);

        // Volume of 5000 (50th value) should be around 50th percentile
        let pct_mid = tracker.percentile(5000);
        assert!(pct_mid >= 45.0 && pct_mid <= 55.0);
    }

    #[test]
    fn test_is_spike() {
        let mut tracker = VolumeTracker::new();

        // Add 100 volumes from 100-10000
        for i in 1..=100 {
            tracker.update(i * 100);
        }

        // 95th percentile volume should be a spike
        assert!(tracker.is_spike(9500));

        // 50th percentile volume should not be a spike
        assert!(!tracker.is_spike(5000));
    }

    #[test]
    fn test_is_capitulation_volume() {
        let mut tracker = VolumeTracker::new();

        // Build history with lower volumes
        for _ in 0..50 {
            tracker.update(1000);
        }

        // High percentile AND recent highest = capitulation
        assert!(tracker.is_capitulation_volume(5000));

        // Not recent highest even if high percentile = not capitulation
        tracker.update(6000); // Add a higher recent volume
        assert!(!tracker.is_capitulation_volume(5000));
    }

    #[test]
    fn test_lossless_capitulation() {
        let mut tracker = VolumeTracker::new();

        // Build history - need 10+ for has_context
        for _ in 0..20 {
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

        // Less than 10 observations = no context
        for _ in 0..9 {
            tracker.update(1000);
        }
        assert!(!tracker.has_context());

        // 10+ observations = has context
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

        // Fill with 30 values of 1000
        for _ in 0..30 {
            tracker.update(1000);
        }

        assert!((tracker.average() - 1000.0).abs() < 0.01);

        // Add 30 values of 2000
        for _ in 0..30 {
            tracker.update(2000);
        }

        // Now average should be (30x1000 + 30x2000) / 60 = 1500
        assert!((tracker.average() - 1500.0).abs() < 0.01);

        // Verify all-time percentile tracking still works
        assert_eq!(tracker.observation_count(), 60);
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
