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
/// Maintains a simple moving average of volume to detect spikes.
/// Uses a 20-bar lookback (standard viewing window on a chart).
#[derive(Debug, Clone)]
pub struct VolumeTracker {
    /// Recent volume values
    volumes: Vec<u64>,
    /// Running sum for efficient SMA calculation
    sum: u64,
}

impl VolumeTracker {
    /// Create a new volume tracker
    pub fn new() -> Self {
        Self {
            volumes: Vec::with_capacity(20),
            sum: 0,
        }
    }

    /// Update with new volume
    pub fn update(&mut self, volume: u64) {
        // Remove oldest if at capacity
        if self.volumes.len() >= 20 {
            if let Some(oldest) = self.volumes.first() {
                self.sum = self.sum.saturating_sub(*oldest);
            }
            self.volumes.remove(0);
        }

        self.volumes.push(volume);
        self.sum += volume;
    }

    /// Get the 20-bar simple moving average of volume
    pub fn average(&self) -> f64 {
        if self.volumes.is_empty() {
            return 0.0;
        }
        self.sum as f64 / self.volumes.len() as f64
    }

    /// Check if the given volume is a spike (> 2x average)
    pub fn is_spike(&self, volume: u64) -> bool {
        let avg = self.average();
        if avg <= 0.0 {
            return false;
        }
        volume as f64 > avg * 2.0
    }

    /// Get the volume ratio (current / average)
    pub fn ratio(&self, volume: u64) -> f64 {
        let avg = self.average();
        if avg <= 0.0 {
            return 0.0;
        }
        volume as f64 / avg
    }

    /// Check if we have enough data for reliable signals
    pub fn is_ready(&self) -> bool {
        self.volumes.len() >= 5 // Minimum 5 bars for context
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

/// Simple helper to detect capitulation from a bar
pub fn is_buy_capitulation(
    volume: u64,
    avg_volume: f64,
    open: Decimal,
    close: Decimal,
) -> bool {
    let change = close - open;
    matches!(
        check_capitulation(volume, avg_volume, change),
        CapitulationSignal::BuyCapitulation
    )
}

/// Simple helper to detect sell capitulation from a bar
pub fn is_sell_capitulation(
    volume: u64,
    avg_volume: f64,
    open: Decimal,
    close: Decimal,
) -> bool {
    let change = close - open;
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
    fn test_volume_spike_detection() {
        let mut tracker = VolumeTracker::new();

        // Normal volume around 1000
        for _ in 0..10 {
            tracker.update(1000);
        }

        // 2100 is just over 2x average - should be a spike
        assert!(tracker.is_spike(2100));

        // 1900 is under 2x - not a spike
        assert!(!tracker.is_spike(1900));
    }

    #[test]
    fn test_buy_capitulation() {
        // High volume on a down day = buy capitulation
        let signal = check_capitulation(3000, 1000.0, dec!(-5.0));
        assert_eq!(signal, CapitulationSignal::BuyCapitulation);
    }

    #[test]
    fn test_sell_capitulation() {
        // High volume on an up day = sell capitulation
        let signal = check_capitulation(3000, 1000.0, dec!(5.0));
        assert_eq!(signal, CapitulationSignal::SellCapitulation);
    }

    #[test]
    fn test_no_capitulation_low_volume() {
        // Volume not high enough - no signal
        let signal = check_capitulation(1500, 1000.0, dec!(-5.0));
        assert_eq!(signal, CapitulationSignal::None);
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
    fn test_helper_functions() {
        assert!(is_buy_capitulation(3000, 1000.0, dec!(100), dec!(95)));
        assert!(!is_buy_capitulation(3000, 1000.0, dec!(100), dec!(105)));

        assert!(is_sell_capitulation(3000, 1000.0, dec!(100), dec!(105)));
        assert!(!is_sell_capitulation(3000, 1000.0, dec!(100), dec!(95)));
    }
}
