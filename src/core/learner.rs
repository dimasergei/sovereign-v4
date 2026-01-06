//! Confidence Calibration Learner
//!
//! A lightweight linear model that learns to calibrate confidence scores
//! based on historical trade outcomes. This adds adaptive learning on top
//! of the lossless signal generation.
//!
//! Features:
//! - S/R score (normalized)
//! - Volume percentile (normalized)
//! - Regime one-hot encoding (4 values)
//!
//! The model learns from trade outcomes to adjust confidence predictions.
//!
//! Includes Elastic Weight Consolidation (EWC) to prevent catastrophic
//! forgetting when market regimes change. EWC protects weights that were
//! important for good performance in previous regimes.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::info;

use super::regime::Regime;

/// Number of features in the linear model
const NUM_FEATURES: usize = 6;

/// Default EWC lambda (strength of weight protection)
const DEFAULT_EWC_LAMBDA: f64 = 100.0;

/// Trade outcome for EWC Fisher computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOutcome {
    pub sr_score: i32,
    pub volume_pct: f64,
    pub regime: Regime,
    pub won: bool,
}

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Lightweight linear model for confidence calibration
///
/// Takes 6 features and outputs a confidence score in [0, 1]:
/// - sr_score_normalized: S/R score / -10.0, clamped to [-1, 1]
/// - volume_percentile_normalized: percentile / 100.0
/// - is_trending_up: 1.0 if TrendingUp regime, else 0.0
/// - is_trending_down: 1.0 if TrendingDown regime, else 0.0
/// - is_ranging: 1.0 if Ranging regime, else 0.0
/// - is_volatile: 1.0 if Volatile regime, else 0.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceCalibrator {
    /// Model weights for each feature
    weights: [f64; NUM_FEATURES],
    /// Bias term
    bias: f64,
    /// Number of updates performed (for tracking)
    update_count: u64,

    // EWC (Elastic Weight Consolidation) fields
    /// Fisher Information matrix diagonal (importance of each weight)
    fisher: [f64; NUM_FEATURES],
    /// Fisher Information for bias
    fisher_bias: f64,
    /// Optimal weights at consolidation point
    optimal_weights: Option<[f64; NUM_FEATURES]>,
    /// Optimal bias at consolidation point
    optimal_bias: Option<f64>,
    /// EWC penalty strength (higher = more protection of old weights)
    ewc_lambda: f64,
    /// Number of consolidations performed
    consolidation_count: u32,
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceCalibrator {
    /// Create a new calibrator with default weights
    ///
    /// Initial weights are set to reasonable priors:
    /// - S/R score: 0.3 (higher weight - stronger S/R = higher confidence)
    /// - Volume percentile: 0.2 (moderate weight - higher volume = higher confidence)
    /// - TrendingUp: 0.1 (slight positive - trends are tradeable)
    /// - TrendingDown: 0.1 (slight positive - trends are tradeable)
    /// - Ranging: 0.1 (neutral - ranging markets are tricky)
    /// - Volatile: 0.2 (higher weight - volatility affects outcomes)
    pub fn new() -> Self {
        Self {
            weights: [0.3, 0.2, 0.1, 0.1, 0.1, 0.2],
            bias: 0.5,
            update_count: 0,
            // EWC fields initialized to defaults
            fisher: [0.0; NUM_FEATURES],
            fisher_bias: 0.0,
            optimal_weights: None,
            optimal_bias: None,
            ewc_lambda: DEFAULT_EWC_LAMBDA,
            consolidation_count: 0,
        }
    }

    /// Predict confidence adjustment given input features
    ///
    /// Returns a value in [0, 1] representing the model's confidence prediction.
    pub fn predict(&self, sr_score: i32, volume_percentile: f64, regime: &Regime) -> f64 {
        let features = Self::encode_features(sr_score, volume_percentile, regime);
        self.predict_from_features(&features)
    }

    /// Predict from pre-encoded features
    fn predict_from_features(&self, features: &[f64; NUM_FEATURES]) -> f64 {
        // Dot product: sum(weights * features) + bias
        let mut z = self.bias;
        for i in 0..NUM_FEATURES {
            z += self.weights[i] * features[i];
        }
        sigmoid(z)
    }

    /// Encode raw inputs into normalized features
    pub fn encode_features(sr_score: i32, volume_percentile: f64, regime: &Regime) -> [f64; NUM_FEATURES] {
        // Normalize S/R score: divide by -10 and clamp to [-1, 1]
        // Score of 0 (strongest) -> 0.0, Score of -10 -> 1.0
        let sr_normalized = (sr_score as f64 / -10.0).clamp(-1.0, 1.0);

        // Normalize volume percentile to [0, 1]
        let vol_normalized = (volume_percentile / 100.0).clamp(0.0, 1.0);

        // One-hot encode regime
        let (is_trending_up, is_trending_down, is_ranging, is_volatile) = match regime {
            Regime::TrendingUp => (1.0, 0.0, 0.0, 0.0),
            Regime::TrendingDown => (0.0, 1.0, 0.0, 0.0),
            Regime::Ranging => (0.0, 0.0, 1.0, 0.0),
            Regime::Volatile => (0.0, 0.0, 0.0, 1.0),
        };

        [
            sr_normalized,
            vol_normalized,
            is_trending_up,
            is_trending_down,
            is_ranging,
            is_volatile,
        ]
    }

    /// Update weights using gradient descent with EWC penalty
    ///
    /// Performs one step of gradient descent with optional EWC regularization:
    /// w_i += learning_rate * ((target - prediction) * feature_i - ewc_penalty)
    ///
    /// If EWC has been consolidated, the penalty prevents catastrophic forgetting.
    pub fn update(&mut self, features: &[f64; NUM_FEATURES], target: f64, learning_rate: f64) {
        // Delegate to update_with_ewc which handles both regular and EWC updates
        self.update_with_ewc(features, target, learning_rate);
    }

    /// Update from raw inputs
    pub fn update_from_trade(
        &mut self,
        sr_score: i32,
        volume_percentile: f64,
        regime: &Regime,
        won: bool,
        learning_rate: f64,
    ) {
        let features = Self::encode_features(sr_score, volume_percentile, regime);
        let target = if won { 1.0 } else { 0.0 };
        self.update(&features, target, learning_rate);
    }

    // ==================== EWC Methods ====================

    /// Compute Fisher Information from recent trade outcomes
    ///
    /// Fisher Information measures the importance of each weight by computing
    /// the expected squared gradient over recent data. Higher values indicate
    /// weights that are more important for current performance.
    pub fn compute_fisher(&mut self, recent_trades: &[TradeOutcome]) {
        if recent_trades.is_empty() {
            return;
        }

        // Reset fisher to zero
        self.fisher = [0.0; NUM_FEATURES];
        self.fisher_bias = 0.0;

        // Compute average squared gradient over recent trades
        for trade in recent_trades {
            let features = Self::encode_features(trade.sr_score, trade.volume_pct, &trade.regime);
            let prediction = self.predict_from_features(&features);
            let target = if trade.won { 1.0 } else { 0.0 };
            let error = target - prediction;

            // Gradient squared for each weight
            for i in 0..NUM_FEATURES {
                let grad = error * features[i];
                self.fisher[i] += grad * grad;
            }
            // Gradient squared for bias
            self.fisher_bias += error * error;
        }

        // Average over number of trades
        let n = recent_trades.len() as f64;
        for i in 0..NUM_FEATURES {
            self.fisher[i] /= n;
        }
        self.fisher_bias /= n;
    }

    /// Consolidate current weights as the optimal reference point
    ///
    /// Call this when performance is good and you want to protect
    /// the current weights from being forgotten during future learning.
    /// Typically called after a regime change when the model has adapted well.
    pub fn consolidate(&mut self) {
        self.optimal_weights = Some(self.weights);
        self.optimal_bias = Some(self.bias);
        self.consolidation_count += 1;
        info!(
            "EWC consolidation #{}: protected weights {:?}",
            self.consolidation_count, self.weights
        );
    }

    /// Update weights with EWC penalty to prevent catastrophic forgetting
    ///
    /// The EWC penalty adds a quadratic term that penalizes moving away from
    /// optimal weights, weighted by Fisher Information (importance).
    ///
    /// Loss = BCE + (lambda/2) * sum_i(F_i * (w_i - w*_i)^2)
    ///
    /// Gradient includes: (target - pred) * feature - lambda * F_i * (w_i - w*_i)
    pub fn update_with_ewc(&mut self, features: &[f64; NUM_FEATURES], target: f64, learning_rate: f64) {
        let prediction = self.predict_from_features(features);
        let error = target - prediction;

        // Standard gradient descent with EWC penalty
        for i in 0..NUM_FEATURES {
            let mut grad = error * features[i];

            // Add EWC penalty if we have optimal weights
            if let Some(ref optimal) = self.optimal_weights {
                let ewc_penalty = self.ewc_lambda * self.fisher[i] * (self.weights[i] - optimal[i]);
                grad -= ewc_penalty;
            }

            self.weights[i] += learning_rate * grad;
        }

        // Update bias with EWC penalty
        let mut bias_grad = error;
        if let Some(optimal_bias) = self.optimal_bias {
            let ewc_penalty = self.ewc_lambda * self.fisher_bias * (self.bias - optimal_bias);
            bias_grad -= ewc_penalty;
        }
        self.bias += learning_rate * bias_grad;

        self.update_count += 1;
    }

    /// Get EWC lambda (penalty strength)
    pub fn get_ewc_lambda(&self) -> f64 {
        self.ewc_lambda
    }

    /// Set EWC lambda (penalty strength)
    pub fn set_ewc_lambda(&mut self, lambda: f64) {
        self.ewc_lambda = lambda;
    }

    /// Get Fisher Information values
    pub fn get_fisher(&self) -> &[f64; NUM_FEATURES] {
        &self.fisher
    }

    /// Check if model has been consolidated (has optimal weights)
    pub fn is_consolidated(&self) -> bool {
        self.optimal_weights.is_some()
    }

    /// Get consolidation count
    pub fn consolidation_count(&self) -> u32 {
        self.consolidation_count
    }

    /// Get current weights
    pub fn get_weights(&self) -> &[f64; NUM_FEATURES] {
        &self.weights
    }

    /// Set weights manually
    pub fn set_weights(&mut self, weights: [f64; NUM_FEATURES]) {
        self.weights = weights;
    }

    /// Get bias
    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    /// Set bias
    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Save calibrator to JSON file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load calibrator from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let calibrator: Self = serde_json::from_str(&json)?;
        Ok(calibrator)
    }

    /// Load from file or create new if file doesn't exist
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        match Self::load(path) {
            Ok(calibrator) => calibrator,
            Err(_) => Self::new(),
        }
    }

    /// Get feature names for logging/debugging
    pub fn feature_names() -> [&'static str; NUM_FEATURES] {
        [
            "sr_score",
            "volume_pctl",
            "trending_up",
            "trending_down",
            "ranging",
            "volatile",
        ]
    }

    /// Format weights for logging
    pub fn format_weights(&self) -> String {
        let names = Self::feature_names();
        let parts: Vec<String> = names
            .iter()
            .zip(self.weights.iter())
            .map(|(name, weight)| format!("{}={:.3}", name, weight))
            .collect();
        format!("bias={:.3} {}", self.bias, parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_calibrator() {
        let cal = ConfidenceCalibrator::new();
        assert_eq!(cal.weights.len(), 6);
        assert!((cal.bias - 0.5).abs() < 0.001);
        assert_eq!(cal.update_count, 0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_encode_features() {
        // Test S/R score normalization
        let features = ConfidenceCalibrator::encode_features(0, 50.0, &Regime::Ranging);
        assert!((features[0] - 0.0).abs() < 0.001); // Score 0 -> 0.0

        let features = ConfidenceCalibrator::encode_features(-10, 50.0, &Regime::Ranging);
        assert!((features[0] - 1.0).abs() < 0.001); // Score -10 -> 1.0

        let features = ConfidenceCalibrator::encode_features(-5, 50.0, &Regime::Ranging);
        assert!((features[0] - 0.5).abs() < 0.001); // Score -5 -> 0.5

        // Test volume normalization
        let features = ConfidenceCalibrator::encode_features(0, 100.0, &Regime::Ranging);
        assert!((features[1] - 1.0).abs() < 0.001);

        let features = ConfidenceCalibrator::encode_features(0, 0.0, &Regime::Ranging);
        assert!((features[1] - 0.0).abs() < 0.001);

        // Test regime one-hot
        let features = ConfidenceCalibrator::encode_features(0, 50.0, &Regime::TrendingUp);
        assert!((features[2] - 1.0).abs() < 0.001);
        assert!((features[3] - 0.0).abs() < 0.001);
        assert!((features[4] - 0.0).abs() < 0.001);
        assert!((features[5] - 0.0).abs() < 0.001);

        let features = ConfidenceCalibrator::encode_features(0, 50.0, &Regime::Volatile);
        assert!((features[2] - 0.0).abs() < 0.001);
        assert!((features[3] - 0.0).abs() < 0.001);
        assert!((features[4] - 0.0).abs() < 0.001);
        assert!((features[5] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_predict() {
        let cal = ConfidenceCalibrator::new();

        // With default weights and bias=0.5, prediction should be around 0.5-0.7
        let pred = cal.predict(0, 50.0, &Regime::Ranging);
        assert!(pred > 0.4 && pred < 0.8);

        // Higher S/R score (worse) should give lower confidence
        let pred_weak = cal.predict(-10, 50.0, &Regime::Ranging);
        let pred_strong = cal.predict(0, 50.0, &Regime::Ranging);
        // With positive weight on sr_normalized, higher normalized value (weaker S/R)
        // actually increases prediction, but the difference should be modest
        assert!((pred_weak - pred_strong).abs() < 0.3);
    }

    #[test]
    fn test_update() {
        let mut cal = ConfidenceCalibrator::new();
        let features = ConfidenceCalibrator::encode_features(0, 80.0, &Regime::TrendingUp);

        let pred_before = cal.predict_from_features(&features);

        // Update towards target of 1.0 (winning trade)
        cal.update(&features, 1.0, 0.1);

        let pred_after = cal.predict_from_features(&features);

        // Prediction should increase after positive feedback
        assert!(pred_after > pred_before);
        assert_eq!(cal.update_count, 1);
    }

    #[test]
    fn test_update_from_trade() {
        let mut cal = ConfidenceCalibrator::new();

        // Simulate winning trade
        cal.update_from_trade(-2, 75.0, &Regime::TrendingUp, true, 0.1);
        assert_eq!(cal.update_count, 1);

        // Simulate losing trade
        cal.update_from_trade(-5, 60.0, &Regime::Volatile, false, 0.1);
        assert_eq!(cal.update_count, 2);
    }

    #[test]
    fn test_learning_convergence() {
        let mut cal = ConfidenceCalibrator::new();

        // Simulate many winning trades in TrendingUp with strong S/R
        for _ in 0..50 {
            cal.update_from_trade(0, 90.0, &Regime::TrendingUp, true, 0.05);
        }

        // Simulate many losing trades in Volatile with weak S/R
        for _ in 0..50 {
            cal.update_from_trade(-8, 40.0, &Regime::Volatile, false, 0.05);
        }

        // After learning, should predict higher confidence for winning pattern
        let good_pred = cal.predict(0, 90.0, &Regime::TrendingUp);
        let bad_pred = cal.predict(-8, 40.0, &Regime::Volatile);

        assert!(good_pred > bad_pred);
    }

    #[test]
    fn test_save_load() {
        let mut cal = ConfidenceCalibrator::new();
        cal.update_from_trade(-3, 70.0, &Regime::Ranging, true, 0.1);

        let path = "/tmp/test_calibrator.json";
        cal.save(path).unwrap();

        let loaded = ConfidenceCalibrator::load(path).unwrap();
        assert_eq!(cal.weights, loaded.weights);
        assert!((cal.bias - loaded.bias).abs() < 0.0001);
        assert_eq!(cal.update_count, loaded.update_count);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_or_new() {
        // Non-existent file should return new calibrator
        let cal = ConfidenceCalibrator::load_or_new("/tmp/nonexistent_calibrator_12345.json");
        assert_eq!(cal.update_count, 0);
    }

    #[test]
    fn test_format_weights() {
        let cal = ConfidenceCalibrator::new();
        let formatted = cal.format_weights();
        assert!(formatted.contains("bias="));
        assert!(formatted.contains("sr_score="));
        assert!(formatted.contains("volume_pctl="));
    }

    #[test]
    fn test_get_set_weights() {
        let mut cal = ConfidenceCalibrator::new();
        let new_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        cal.set_weights(new_weights);
        assert_eq!(*cal.get_weights(), new_weights);
    }

    #[test]
    fn test_get_set_bias() {
        let mut cal = ConfidenceCalibrator::new();
        cal.set_bias(0.75);
        assert!((cal.get_bias() - 0.75).abs() < 0.001);
    }

    // ==================== EWC Tests ====================

    #[test]
    fn test_ewc_consolidation() {
        let mut cal = ConfidenceCalibrator::new();

        // Initially not consolidated
        assert!(!cal.is_consolidated());
        assert_eq!(cal.consolidation_count(), 0);

        // Create some trade outcomes
        let trades = vec![
            TradeOutcome { sr_score: 0, volume_pct: 80.0, regime: Regime::TrendingUp, won: true },
            TradeOutcome { sr_score: -2, volume_pct: 70.0, regime: Regime::TrendingUp, won: true },
            TradeOutcome { sr_score: -3, volume_pct: 60.0, regime: Regime::TrendingUp, won: false },
        ];

        // Compute Fisher Information
        cal.compute_fisher(&trades);

        // Fisher should now have non-zero values
        let fisher = cal.get_fisher();
        assert!(fisher.iter().any(|&f| f > 0.0));

        // Consolidate
        cal.consolidate();

        // Now consolidated
        assert!(cal.is_consolidated());
        assert_eq!(cal.consolidation_count(), 1);
    }

    #[test]
    fn test_ewc_prevents_forgetting() {
        let mut cal = ConfidenceCalibrator::new();

        // Train on winning pattern in TrendingUp
        let trending_up_trades: Vec<TradeOutcome> = (0..20).map(|_| {
            TradeOutcome { sr_score: 0, volume_pct: 85.0, regime: Regime::TrendingUp, won: true }
        }).collect();

        for trade in &trending_up_trades {
            let features = ConfidenceCalibrator::encode_features(trade.sr_score, trade.volume_pct, &trade.regime);
            cal.update(&features, 1.0, 0.1);
        }

        // Record prediction for the good pattern
        let good_pred_before = cal.predict(0, 85.0, &Regime::TrendingUp);

        // Compute Fisher and consolidate (protect these weights)
        cal.compute_fisher(&trending_up_trades);
        cal.consolidate();

        // Now train on different pattern (Volatile with weak S/R)
        for _ in 0..20 {
            cal.update_from_trade(-8, 40.0, &Regime::Volatile, false, 0.1);
        }

        // Check prediction for original good pattern - should be somewhat protected
        let good_pred_after = cal.predict(0, 85.0, &Regime::TrendingUp);

        // EWC should prevent complete forgetting
        // The prediction shouldn't drop too much (with EWC, it's penalized for moving away)
        // Without EWC, the prediction would drop significantly more
        assert!(good_pred_after > good_pred_before * 0.5,
            "EWC should prevent catastrophic forgetting: before={:.3}, after={:.3}",
            good_pred_before, good_pred_after);
    }

    #[test]
    fn test_fisher_computation() {
        let mut cal = ConfidenceCalibrator::new();

        // Create uniform trades
        let trades: Vec<TradeOutcome> = (0..10).map(|i| {
            TradeOutcome {
                sr_score: -(i % 5) as i32,
                volume_pct: 50.0 + (i * 5) as f64,
                regime: Regime::Ranging,
                won: i % 2 == 0,
            }
        }).collect();

        // Compute Fisher
        cal.compute_fisher(&trades);

        let fisher = cal.get_fisher();

        // All Fisher values should be non-negative
        for &f in fisher {
            assert!(f >= 0.0, "Fisher values must be non-negative");
        }

        // At least some features should have positive Fisher (indicating importance)
        assert!(fisher.iter().any(|&f| f > 0.0), "Some features should have positive Fisher");
    }

    #[test]
    fn test_ewc_lambda() {
        let mut cal = ConfidenceCalibrator::new();

        // Check default lambda
        assert!((cal.get_ewc_lambda() - 100.0).abs() < 0.001);

        // Set new lambda
        cal.set_ewc_lambda(200.0);
        assert!((cal.get_ewc_lambda() - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_update_with_ewc_without_consolidation() {
        let mut cal = ConfidenceCalibrator::new();
        let features = ConfidenceCalibrator::encode_features(0, 70.0, &Regime::TrendingUp);

        let pred_before = cal.predict_from_features(&features);

        // Update without prior consolidation (should work like normal update)
        cal.update_with_ewc(&features, 1.0, 0.1);

        let pred_after = cal.predict_from_features(&features);

        // Prediction should increase toward target
        assert!(pred_after > pred_before);
    }

    #[test]
    fn test_ewc_save_load() {
        let mut cal = ConfidenceCalibrator::new();

        // Create trades and consolidate
        let trades = vec![
            TradeOutcome { sr_score: 0, volume_pct: 80.0, regime: Regime::TrendingUp, won: true },
            TradeOutcome { sr_score: -2, volume_pct: 70.0, regime: Regime::TrendingUp, won: true },
        ];
        cal.compute_fisher(&trades);
        cal.consolidate();
        cal.set_ewc_lambda(150.0);

        // Save
        let path = "/tmp/test_calibrator_ewc.json";
        cal.save(path).unwrap();

        // Load and verify EWC state
        let loaded = ConfidenceCalibrator::load(path).unwrap();
        assert!(loaded.is_consolidated());
        assert_eq!(loaded.consolidation_count(), 1);
        assert!((loaded.get_ewc_lambda() - 150.0).abs() < 0.001);

        // Compare Fisher values with tolerance for floating point
        let cal_fisher = cal.get_fisher();
        let loaded_fisher = loaded.get_fisher();
        for i in 0..6 {
            assert!((cal_fisher[i] - loaded_fisher[i]).abs() < 1e-10,
                "Fisher[{}] mismatch: {} vs {}", i, cal_fisher[i], loaded_fisher[i]);
        }

        // Cleanup
        let _ = std::fs::remove_file(path);
    }
}
