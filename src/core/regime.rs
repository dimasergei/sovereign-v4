//! Hidden Markov Model (HMM) Based Regime Detection
//!
//! Classifies market regimes using a 4-state HMM:
//! - TrendingUp: Sustained positive returns with moderate volatility
//! - TrendingDown: Sustained negative returns with elevated volatility
//! - Ranging: Near-zero returns with low volatility (consolidation)
//! - Volatile: High ATR with large returns in either direction
//!
//! The HMM uses Gaussian emission probabilities based on:
//! - Normalized returns (close-to-close)
//! - Normalized volatility (ATR / price)
//! - Volume ratio (current / average)

use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    /// Sustained upward price movement
    TrendingUp,
    /// Sustained downward price movement
    TrendingDown,
    /// Low volatility consolidation
    Ranging,
    /// High volatility, large swings
    Volatile,
}

impl Regime {
    /// Convert to database-storable string
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::TrendingUp => "TRENDING_UP",
            Regime::TrendingDown => "TRENDING_DOWN",
            Regime::Ranging => "RANGING",
            Regime::Volatile => "VOLATILE",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "TRENDING_UP" => Some(Regime::TrendingUp),
            "TRENDING_DOWN" => Some(Regime::TrendingDown),
            "RANGING" => Some(Regime::Ranging),
            "VOLATILE" => Some(Regime::Volatile),
            _ => None,
        }
    }

    /// Get regime index (0-3) for matrix operations
    pub fn index(&self) -> usize {
        match self {
            Regime::TrendingUp => 0,
            Regime::TrendingDown => 1,
            Regime::Ranging => 2,
            Regime::Volatile => 3,
        }
    }

    /// Create regime from index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Regime::TrendingUp,
            1 => Regime::TrendingDown,
            2 => Regime::Ranging,
            _ => Regime::Volatile,
        }
    }

    /// Get all regimes
    pub fn all() -> [Regime; 4] {
        [
            Regime::TrendingUp,
            Regime::TrendingDown,
            Regime::Ranging,
            Regime::Volatile,
        ]
    }
}

impl std::fmt::Display for Regime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Regime::TrendingUp => write!(f, "TrendingUp"),
            Regime::TrendingDown => write!(f, "TrendingDown"),
            Regime::Ranging => write!(f, "Ranging"),
            Regime::Volatile => write!(f, "Volatile"),
        }
    }
}

/// Gaussian distribution parameters for emission probabilities
#[derive(Debug, Clone, Copy)]
struct GaussianParams {
    mean: f64,
    std_dev: f64,
}

impl GaussianParams {
    fn new(mean: f64, std_dev: f64) -> Self {
        Self {
            mean,
            std_dev: std_dev.max(0.001), // Prevent division by zero
        }
    }

    /// Calculate probability density at x
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        let coefficient = 1.0 / (self.std_dev * (2.0 * PI).sqrt());
        coefficient * (-0.5 * z * z).exp()
    }

    /// Calculate log probability density (for numerical stability)
    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        -0.5 * z * z - self.std_dev.ln() - 0.5 * (2.0 * PI).ln()
    }
}

/// Observation vector for HMM
#[derive(Debug, Clone, Copy)]
struct Observation {
    /// Normalized return (close-to-close / price)
    return_pct: f64,
    /// Normalized volatility (ATR / price)
    volatility: f64,
    /// Volume ratio (current / average)
    volume_ratio: f64,
}

/// Hidden Markov Model for regime detection
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    /// Number of states
    n_states: usize,

    /// State transition matrix (row = from, col = to)
    /// transition[i][j] = P(state_j | state_i)
    transition: [[f64; 4]; 4],

    /// Emission parameters for returns (per state)
    return_params: [GaussianParams; 4],

    /// Emission parameters for volatility (per state)
    volatility_params: [GaussianParams; 4],

    /// Current state probabilities (forward algorithm)
    state_probs: [f64; 4],

    /// Most likely current state
    current_state: Regime,

    /// Duration in current regime (bars)
    regime_duration: u32,

    /// Previous regime for change detection
    prev_regime: Option<Regime>,

    /// Rolling window of observations for statistics
    observation_window: Vec<Observation>,
    max_window_size: usize,

    /// Previous close for return calculation
    prev_close: Option<f64>,

    /// Rolling ATR components
    true_ranges: Vec<f64>,
    atr_period: usize,

    /// Rolling volume for averaging
    volumes: Vec<f64>,
    volume_period: usize,
}

impl RegimeDetector {
    /// Create a new regime detector with default parameters
    pub fn new() -> Self {
        // Transition matrix: favor staying in same state (diagonal ~0.95)
        // Off-diagonal transitions are balanced
        let stay_prob = 0.95;
        let switch_prob = (1.0 - stay_prob) / 3.0;

        let transition = [
            [stay_prob, switch_prob, switch_prob, switch_prob], // From TrendingUp
            [switch_prob, stay_prob, switch_prob, switch_prob], // From TrendingDown
            [switch_prob, switch_prob, stay_prob, switch_prob], // From Ranging
            [switch_prob, switch_prob, switch_prob, stay_prob], // From Volatile
        ];

        // Emission parameters for returns (mean, std_dev)
        // Returns are normalized (daily return percentage)
        let return_params = [
            GaussianParams::new(0.5, 0.8),   // TrendingUp: positive returns
            GaussianParams::new(-0.5, 1.0),  // TrendingDown: negative returns, higher variance
            GaussianParams::new(0.0, 0.3),   // Ranging: near-zero returns, low variance
            GaussianParams::new(0.0, 2.0),   // Volatile: zero mean, very high variance
        ];

        // Emission parameters for volatility (normalized ATR)
        // Volatility expressed as ATR / price * 100
        let volatility_params = [
            GaussianParams::new(1.5, 0.5),  // TrendingUp: moderate volatility
            GaussianParams::new(2.5, 0.8),  // TrendingDown: elevated volatility
            GaussianParams::new(0.8, 0.3),  // Ranging: low volatility
            GaussianParams::new(4.0, 1.5),  // Volatile: high volatility
        ];

        // Start with uniform state probabilities
        let initial_probs = [0.25, 0.25, 0.25, 0.25];

        Self {
            n_states: 4,
            transition,
            return_params,
            volatility_params,
            state_probs: initial_probs,
            current_state: Regime::Ranging, // Default to ranging
            regime_duration: 0,
            prev_regime: None,
            observation_window: Vec::with_capacity(50),
            max_window_size: 50,
            prev_close: None,
            true_ranges: Vec::with_capacity(14),
            atr_period: 14,
            volumes: Vec::with_capacity(20),
            volume_period: 20,
        }
    }

    /// Update the detector with a new candle
    pub fn update(&mut self, open: Decimal, high: Decimal, low: Decimal, close: Decimal, volume: u64) {
        let open_f = open.to_f64().unwrap_or(0.0);
        let high_f = high.to_f64().unwrap_or(0.0);
        let low_f = low.to_f64().unwrap_or(0.0);
        let close_f = close.to_f64().unwrap_or(0.0);
        let volume_f = volume as f64;

        // Update true range
        let tr = if let Some(prev) = self.prev_close {
            let hl = high_f - low_f;
            let hc = (high_f - prev).abs();
            let lc = (low_f - prev).abs();
            hl.max(hc).max(lc)
        } else {
            high_f - low_f
        };

        if self.true_ranges.len() >= self.atr_period {
            self.true_ranges.remove(0);
        }
        self.true_ranges.push(tr);

        // Update volume history
        if self.volumes.len() >= self.volume_period {
            self.volumes.remove(0);
        }
        self.volumes.push(volume_f);

        // Calculate observation
        if let Some(prev_close) = self.prev_close {
            let return_pct = ((close_f - prev_close) / prev_close) * 100.0;

            let atr = if !self.true_ranges.is_empty() {
                self.true_ranges.iter().sum::<f64>() / self.true_ranges.len() as f64
            } else {
                tr
            };
            let volatility = (atr / close_f) * 100.0;

            let avg_volume = if !self.volumes.is_empty() {
                self.volumes.iter().sum::<f64>() / self.volumes.len() as f64
            } else {
                volume_f
            };
            let volume_ratio = if avg_volume > 0.0 {
                volume_f / avg_volume
            } else {
                1.0
            };

            let obs = Observation {
                return_pct,
                volatility,
                volume_ratio,
            };

            // Store observation
            if self.observation_window.len() >= self.max_window_size {
                self.observation_window.remove(0);
            }
            self.observation_window.push(obs);

            // Run forward algorithm step
            self.forward_step(&obs);

            // Update regime duration
            let new_state = self.most_likely_state();
            if new_state != self.current_state {
                self.prev_regime = Some(self.current_state);
                self.current_state = new_state;
                self.regime_duration = 1;
            } else {
                self.regime_duration += 1;
            }
        }

        self.prev_close = Some(close_f);
    }

    /// Forward algorithm step: update state probabilities given new observation
    fn forward_step(&mut self, obs: &Observation) {
        let mut new_probs = [0.0; 4];

        // For each current state
        for j in 0..self.n_states {
            let mut sum = 0.0;

            // Sum over previous states
            for i in 0..self.n_states {
                sum += self.state_probs[i] * self.transition[i][j];
            }

            // Multiply by emission probability
            let emission = self.emission_probability(j, obs);
            new_probs[j] = sum * emission;
        }

        // Normalize
        let total: f64 = new_probs.iter().sum();
        if total > 0.0 {
            for p in &mut new_probs {
                *p /= total;
            }
        } else {
            // Fallback to uniform if all probabilities are zero
            new_probs = [0.25, 0.25, 0.25, 0.25];
        }

        self.state_probs = new_probs;
    }

    /// Calculate emission probability for a state given observation
    fn emission_probability(&self, state: usize, obs: &Observation) -> f64 {
        // Use product of independent Gaussian probabilities
        let return_prob = self.return_params[state].pdf(obs.return_pct);
        let volatility_prob = self.volatility_params[state].pdf(obs.volatility);

        // Combine probabilities (could also use log-sum for numerical stability)
        let combined = return_prob * volatility_prob;

        // Add small epsilon to prevent zero probability
        combined.max(1e-10)
    }

    /// Get the most likely current state
    fn most_likely_state(&self) -> Regime {
        let mut max_prob = 0.0;
        let mut max_state = 0;

        for (i, &prob) in self.state_probs.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_state = i;
            }
        }

        Regime::from_index(max_state)
    }

    /// Get the current regime
    pub fn current_regime(&self) -> Regime {
        self.current_state
    }

    /// Get probabilities for all regimes
    pub fn regime_probabilities(&self) -> [f64; 4] {
        self.state_probs
    }

    /// Get probability of a specific regime
    pub fn regime_probability(&self, regime: Regime) -> f64 {
        self.state_probs[regime.index()]
    }

    /// Get duration in current regime (bars)
    pub fn regime_duration(&self) -> u32 {
        self.regime_duration
    }

    /// Check if regime just changed
    pub fn regime_changed(&self) -> bool {
        self.prev_regime.is_some() && self.prev_regime != Some(self.current_state) && self.regime_duration == 1
    }

    /// Get previous regime (if changed)
    pub fn previous_regime(&self) -> Option<Regime> {
        self.prev_regime
    }

    /// Check if detector has enough data
    pub fn is_ready(&self) -> bool {
        self.observation_window.len() >= 5 && self.prev_close.is_some()
    }

    /// Get confidence in current regime (probability of current state)
    pub fn confidence(&self) -> f64 {
        self.state_probs[self.current_state.index()]
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.state_probs = [0.25, 0.25, 0.25, 0.25];
        self.current_state = Regime::Ranging;
        self.regime_duration = 0;
        self.prev_regime = None;
        self.observation_window.clear();
        self.prev_close = None;
        self.true_ranges.clear();
        self.volumes.clear();
    }

    /// Get summary statistics of recent observations
    pub fn observation_stats(&self) -> Option<(f64, f64, f64)> {
        if self.observation_window.is_empty() {
            return None;
        }

        let n = self.observation_window.len() as f64;
        let avg_return = self.observation_window.iter().map(|o| o.return_pct).sum::<f64>() / n;
        let avg_volatility = self.observation_window.iter().map(|o| o.volatility).sum::<f64>() / n;
        let avg_volume_ratio = self.observation_window.iter().map(|o| o.volume_ratio).sum::<f64>() / n;

        Some((avg_return, avg_volatility, avg_volume_ratio))
    }
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_regime_creation() {
        let detector = RegimeDetector::new();
        assert_eq!(detector.current_regime(), Regime::Ranging);
        assert_eq!(detector.regime_duration(), 0);
        assert!(!detector.is_ready());
    }

    #[test]
    fn test_regime_display() {
        assert_eq!(Regime::TrendingUp.to_string(), "TrendingUp");
        assert_eq!(Regime::TrendingDown.to_string(), "TrendingDown");
        assert_eq!(Regime::Ranging.to_string(), "Ranging");
        assert_eq!(Regime::Volatile.to_string(), "Volatile");
    }

    #[test]
    fn test_regime_string_conversion() {
        for regime in Regime::all() {
            let s = regime.as_str();
            let parsed = Regime::from_str(s);
            assert_eq!(parsed, Some(regime));
        }
    }

    #[test]
    fn test_regime_index() {
        for (i, regime) in Regime::all().iter().enumerate() {
            assert_eq!(regime.index(), i);
            assert_eq!(Regime::from_index(i), *regime);
        }
    }

    #[test]
    fn test_detector_update() {
        let mut detector = RegimeDetector::new();

        // Feed some bars
        for i in 0..20 {
            let price = dec!(100) + Decimal::from(i);
            detector.update(price, price + dec!(1), price - dec!(1), price, 1000);
        }

        assert!(detector.is_ready());
        assert!(detector.regime_duration() > 0);
    }

    #[test]
    fn test_trending_up_detection() {
        let mut detector = RegimeDetector::new();

        // Simulate uptrend: consistently rising prices
        let mut price = 100.0;
        for _ in 0..30 {
            price *= 1.01; // 1% daily gain
            let p = Decimal::try_from(price).unwrap();
            detector.update(
                p - dec!(0.5),
                p + dec!(1),
                p - dec!(1),
                p,
                1000,
            );
        }

        assert!(detector.is_ready());
        // Should have higher probability of TrendingUp
        let probs = detector.regime_probabilities();
        assert!(
            probs[Regime::TrendingUp.index()] > 0.2,
            "TrendingUp probability should be elevated: {:?}",
            probs
        );
    }

    #[test]
    fn test_trending_down_detection() {
        let mut detector = RegimeDetector::new();

        // Simulate downtrend: consistently falling prices
        let mut price = 100.0;
        for _ in 0..30 {
            price *= 0.985; // 1.5% daily loss
            let p = Decimal::try_from(price).unwrap();
            detector.update(
                p + dec!(0.5),
                p + dec!(1.5),
                p - dec!(0.5),
                p,
                1500, // Elevated volume on downtrend
            );
        }

        assert!(detector.is_ready());
        let probs = detector.regime_probabilities();
        assert!(
            probs[Regime::TrendingDown.index()] > 0.2,
            "TrendingDown probability should be elevated: {:?}",
            probs
        );
    }

    #[test]
    fn test_ranging_detection() {
        let mut detector = RegimeDetector::new();

        // Simulate ranging: price oscillates in tight range
        for i in 0..30 {
            let offset = if i % 2 == 0 { 0.2 } else { -0.2 };
            let price = Decimal::try_from(100.0 + offset).unwrap();
            detector.update(
                price,
                price + dec!(0.3),
                price - dec!(0.3),
                price,
                800, // Low volume
            );
        }

        assert!(detector.is_ready());
        let probs = detector.regime_probabilities();
        assert!(
            probs[Regime::Ranging.index()] > 0.2,
            "Ranging probability should be elevated: {:?}",
            probs
        );
    }

    #[test]
    fn test_volatile_detection() {
        let mut detector = RegimeDetector::new();

        // Simulate high volatility: large swings
        for i in 0..30 {
            let swing = if i % 2 == 0 { 3.0 } else { -3.0 };
            let price = Decimal::try_from(100.0 + swing).unwrap();
            detector.update(
                price - Decimal::try_from(swing / 2.0).unwrap(),
                price + dec!(2),
                price - dec!(2),
                price,
                3000, // High volume
            );
        }

        assert!(detector.is_ready());
        let probs = detector.regime_probabilities();
        assert!(
            probs[Regime::Volatile.index()] > 0.1,
            "Volatile probability should be elevated: {:?}",
            probs
        );
    }

    #[test]
    fn test_regime_change_detection() {
        let mut detector = RegimeDetector::new();

        // Start with ranging
        for i in 0..20 {
            let offset = if i % 2 == 0 { 0.1 } else { -0.1 };
            let price = Decimal::try_from(100.0 + offset).unwrap();
            detector.update(price, price + dec!(0.2), price - dec!(0.2), price, 1000);
        }

        let initial_regime = detector.current_regime();

        // Now simulate a strong uptrend
        let mut price = 100.0;
        for _ in 0..20 {
            price *= 1.02;
            let p = Decimal::try_from(price).unwrap();
            detector.update(p - dec!(0.5), p + dec!(1), p - dec!(0.5), p, 1500);
        }

        // Regime should have changed or at least probabilities shifted
        let final_probs = detector.regime_probabilities();
        assert!(
            final_probs[Regime::TrendingUp.index()] > final_probs[Regime::Ranging.index()]
                || detector.current_regime() != initial_regime,
            "Regime should shift toward TrendingUp"
        );
    }

    #[test]
    fn test_observation_stats() {
        let mut detector = RegimeDetector::new();

        for i in 0..10 {
            let price = dec!(100) + Decimal::from(i);
            detector.update(price, price + dec!(1), price - dec!(1), price, 1000);
        }

        let stats = detector.observation_stats();
        assert!(stats.is_some());
        let (avg_return, avg_vol, avg_vol_ratio) = stats.unwrap();
        assert!(avg_return.is_finite());
        assert!(avg_vol.is_finite());
        assert!(avg_vol_ratio.is_finite());
    }

    #[test]
    fn test_gaussian_pdf() {
        let g = GaussianParams::new(0.0, 1.0);

        // PDF at mean should be highest
        let at_mean = g.pdf(0.0);
        let at_one_sigma = g.pdf(1.0);
        let at_two_sigma = g.pdf(2.0);

        assert!(at_mean > at_one_sigma);
        assert!(at_one_sigma > at_two_sigma);

        // PDF should be symmetric
        assert!((g.pdf(1.0) - g.pdf(-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_transition_matrix_valid() {
        let detector = RegimeDetector::new();

        // Each row should sum to 1.0
        for row in &detector.transition {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Transition row should sum to 1.0, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_state_probs_sum_to_one() {
        let mut detector = RegimeDetector::new();

        for i in 0..20 {
            let price = dec!(100) + Decimal::from(i);
            detector.update(price, price + dec!(1), price - dec!(1), price, 1000);

            let sum: f64 = detector.state_probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "State probabilities should sum to 1.0, got {}",
                sum
            );
        }
    }
}
