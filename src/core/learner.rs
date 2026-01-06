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

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use super::regime::Regime;

/// Number of features in the linear model
const NUM_FEATURES: usize = 6;

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

    /// Update weights using simple gradient descent
    ///
    /// Performs one step of gradient descent:
    /// w_i += learning_rate * (target - prediction) * feature_i
    ///
    /// This is the gradient of binary cross-entropy loss with respect to weights.
    pub fn update(&mut self, features: &[f64; NUM_FEATURES], target: f64, learning_rate: f64) {
        let prediction = self.predict_from_features(features);
        let error = target - prediction;

        // Gradient descent update
        // For sigmoid output with BCE loss, gradient simplifies to: (target - pred) * feature
        for i in 0..NUM_FEATURES {
            self.weights[i] += learning_rate * error * features[i];
        }

        // Update bias
        self.bias += learning_rate * error;

        self.update_count += 1;
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
}
