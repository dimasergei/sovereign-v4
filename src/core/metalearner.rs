//! Reptile Meta-Learning for Rapid Adaptation
//!
//! Implements the Reptile algorithm (Nichol et al., 2018) for learning good
//! weight initializations that can quickly adapt to new market conditions.
//!
//! Benefits:
//! - New symbols/regimes start with weights that adapt quickly
//! - Learns across all trading experience which initializations work best
//! - Complements EWC (EWC prevents forgetting, Reptile speeds learning)
//!
//! Algorithm:
//! 1. Start task adaptation from meta_weights
//! 2. Run K gradient steps on task-specific data
//! 3. Update meta_weights toward adapted weights if successful
//! 4. Repeat across many tasks (regimes, symbols)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::info;

use super::learner::{ConfidenceCalibrator, TradeOutcome, NUM_FEATURES};
use super::regime::Regime;

/// Default outer learning rate for meta-updates
const DEFAULT_OUTER_LR: f64 = 0.1;

/// Default inner learning rate for task adaptation
const DEFAULT_INNER_LR: f64 = 0.02;

/// Default number of inner gradient steps per task
const DEFAULT_INNER_STEPS: usize = 5;

/// Maximum adaptation history to keep
const MAX_ADAPTATION_HISTORY: usize = 100;

/// Result of adapting to a new task (regime/symbol)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    /// The regime being adapted to
    pub regime: Regime,
    /// Weights before adaptation
    pub initial_weights: [f64; NUM_FEATURES],
    /// Weights after adaptation
    pub adapted_weights: [f64; NUM_FEATURES],
    /// Number of trades used for adaptation
    pub trades_used: u32,
    /// Accuracy before adaptation (on new regime data)
    pub pre_adaptation_accuracy: f64,
    /// Accuracy after adaptation
    pub post_adaptation_accuracy: f64,
}

impl AdaptationResult {
    /// Check if adaptation was successful (improved accuracy)
    pub fn was_successful(&self) -> bool {
        self.post_adaptation_accuracy > self.pre_adaptation_accuracy
    }

    /// Get the improvement in accuracy
    pub fn improvement(&self) -> f64 {
        self.post_adaptation_accuracy - self.pre_adaptation_accuracy
    }
}

/// Reptile Meta-Learner for learning good weight initializations
///
/// Learns a set of meta-weights that serve as a good starting point
/// for rapid adaptation to new tasks (regimes, symbols).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearner {
    /// Meta-learned weight initialization
    meta_weights: [f64; NUM_FEATURES],
    /// Meta-learned bias initialization
    meta_bias: f64,
    /// History of adaptation results for analysis
    adaptation_history: Vec<AdaptationResult>,
    /// Outer learning rate (meta-update step size)
    outer_lr: f64,
    /// Inner learning rate (task adaptation step size)
    inner_lr: f64,
    /// Number of gradient steps for task adaptation
    inner_steps: usize,
    /// Total meta-updates performed
    meta_update_count: u32,
    /// Successful adaptations count
    successful_adaptations: u32,
}

impl Default for MetaLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaLearner {
    /// Create a new MetaLearner with default parameters
    pub fn new() -> Self {
        Self {
            meta_weights: [0.0; NUM_FEATURES],
            meta_bias: 0.0,
            adaptation_history: Vec::new(),
            outer_lr: DEFAULT_OUTER_LR,
            inner_lr: DEFAULT_INNER_LR,
            inner_steps: DEFAULT_INNER_STEPS,
            meta_update_count: 0,
            successful_adaptations: 0,
        }
    }

    /// Create MetaLearner with custom learning rates
    pub fn with_learning_rates(outer_lr: f64, inner_lr: f64, inner_steps: usize) -> Self {
        Self {
            meta_weights: [0.0; NUM_FEATURES],
            meta_bias: 0.0,
            adaptation_history: Vec::new(),
            outer_lr,
            inner_lr,
            inner_steps,
            meta_update_count: 0,
            successful_adaptations: 0,
        }
    }

    /// Get the meta-learned initialization (weights and bias)
    ///
    /// Use this to initialize a new calibrator for fast adaptation.
    pub fn get_initialization(&self) -> ([f64; NUM_FEATURES], f64) {
        (self.meta_weights, self.meta_bias)
    }

    /// Get reference to meta weights
    pub fn meta_weights(&self) -> &[f64; NUM_FEATURES] {
        &self.meta_weights
    }

    /// Get meta bias
    pub fn meta_bias(&self) -> f64 {
        self.meta_bias
    }

    /// Adapt to a new task using inner gradient steps
    ///
    /// Starts from meta_weights and runs `inner_steps` gradient updates
    /// on the provided trades. Returns adapted weights without modifying
    /// the meta-weights (that happens in meta_update).
    pub fn adapt(&self, trades: &[TradeOutcome]) -> ([f64; NUM_FEATURES], f64) {
        if trades.is_empty() {
            return (self.meta_weights, self.meta_bias);
        }

        // Start from meta-weights
        let mut weights = self.meta_weights;
        let mut bias = self.meta_bias;

        // Run inner gradient steps
        for _step in 0..self.inner_steps {
            // Compute gradients over all trades
            let mut grad_weights = [0.0; NUM_FEATURES];
            let mut grad_bias = 0.0;

            for trade in trades {
                let features = ConfidenceCalibrator::encode_features(
                    trade.sr_score,
                    trade.volume_pct,
                    &trade.regime,
                );

                // Forward pass
                let mut z = bias;
                for i in 0..NUM_FEATURES {
                    z += weights[i] * features[i];
                }
                let pred = 1.0 / (1.0 + (-z).exp()); // sigmoid

                // Target
                let target = if trade.won { 1.0 } else { 0.0 };

                // Gradient of binary cross-entropy loss
                let error = pred - target;

                // Accumulate gradients
                for i in 0..NUM_FEATURES {
                    grad_weights[i] += error * features[i];
                }
                grad_bias += error;
            }

            // Average gradients
            let n = trades.len() as f64;
            for i in 0..NUM_FEATURES {
                grad_weights[i] /= n;
            }
            grad_bias /= n;

            // Update weights (gradient descent)
            for i in 0..NUM_FEATURES {
                weights[i] -= self.inner_lr * grad_weights[i];
            }
            bias -= self.inner_lr * grad_bias;
        }

        (weights, bias)
    }

    /// Perform Reptile meta-update
    ///
    /// Updates meta_weights toward the successfully adapted weights.
    /// Only updates if the adaptation was successful (improved accuracy).
    ///
    /// Reptile update rule: meta_weights += outer_lr * (adapted_weights - meta_weights)
    pub fn meta_update(
        &mut self,
        pre_weights: &[f64; NUM_FEATURES],
        pre_bias: f64,
        post_weights: &[f64; NUM_FEATURES],
        post_bias: f64,
        success: bool,
    ) {
        if !success {
            info!("[META] Skipping meta-update: adaptation was not successful");
            return;
        }

        // Calculate update magnitude for logging
        let mut delta_magnitude = 0.0;
        for i in 0..NUM_FEATURES {
            let delta = post_weights[i] - self.meta_weights[i];
            delta_magnitude += delta * delta;
        }
        delta_magnitude = delta_magnitude.sqrt();

        // Reptile update: move meta_weights toward adapted weights
        for i in 0..NUM_FEATURES {
            self.meta_weights[i] += self.outer_lr * (post_weights[i] - self.meta_weights[i]);
        }
        self.meta_bias += self.outer_lr * (post_bias - self.meta_bias);

        self.meta_update_count += 1;
        self.successful_adaptations += 1;

        info!(
            "[META] Updated meta-weights (delta: {:.4}, total updates: {})",
            delta_magnitude, self.meta_update_count
        );
    }

    /// Record an adaptation result for analysis
    pub fn record_adaptation(&mut self, result: AdaptationResult) {
        // Track successful adaptations
        if result.was_successful() {
            self.successful_adaptations += 1;
        }

        self.adaptation_history.push(result);

        // Keep only last MAX_ADAPTATION_HISTORY entries
        if self.adaptation_history.len() > MAX_ADAPTATION_HISTORY {
            self.adaptation_history.remove(0);
        }
    }

    /// Get average number of trades needed to reach 55% accuracy
    ///
    /// Measures adaptation speed across all recorded adaptations.
    pub fn get_adaptation_speed(&self) -> f64 {
        let successful: Vec<_> = self
            .adaptation_history
            .iter()
            .filter(|r| r.post_adaptation_accuracy >= 0.55)
            .collect();

        if successful.is_empty() {
            return f64::INFINITY;
        }

        let total_trades: u32 = successful.iter().map(|r| r.trades_used).sum();
        total_trades as f64 / successful.len() as f64
    }

    /// Get average improvement across adaptations
    pub fn get_average_improvement(&self) -> f64 {
        if self.adaptation_history.is_empty() {
            return 0.0;
        }

        let total: f64 = self.adaptation_history.iter().map(|r| r.improvement()).sum();
        total / self.adaptation_history.len() as f64
    }

    /// Get success rate (fraction of successful adaptations)
    pub fn get_success_rate(&self) -> f64 {
        if self.adaptation_history.is_empty() {
            return 0.0;
        }

        let successes = self.adaptation_history.iter().filter(|r| r.was_successful()).count();
        successes as f64 / self.adaptation_history.len() as f64
    }

    /// Get total meta-update count
    pub fn meta_update_count(&self) -> u32 {
        self.meta_update_count
    }

    /// Get adaptation history length
    pub fn adaptation_history_len(&self) -> usize {
        self.adaptation_history.len()
    }

    /// Get outer learning rate
    pub fn outer_lr(&self) -> f64 {
        self.outer_lr
    }

    /// Set outer learning rate
    pub fn set_outer_lr(&mut self, lr: f64) {
        self.outer_lr = lr.max(0.001).min(1.0);
    }

    /// Get inner learning rate
    pub fn inner_lr(&self) -> f64 {
        self.inner_lr
    }

    /// Set inner learning rate
    pub fn set_inner_lr(&mut self, lr: f64) {
        self.inner_lr = lr.max(0.001).min(1.0);
    }

    /// Get inner steps
    pub fn inner_steps(&self) -> usize {
        self.inner_steps
    }

    /// Set inner steps
    pub fn set_inner_steps(&mut self, steps: usize) {
        self.inner_steps = steps.max(1).min(20);
    }

    /// Save MetaLearner to JSON file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        info!(
            "[META] Saved to {} (updates: {}, history: {})",
            path,
            self.meta_update_count,
            self.adaptation_history.len()
        );
        Ok(())
    }

    /// Load MetaLearner from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let ml: Self = serde_json::from_str(&json)?;
        Ok(ml)
    }

    /// Load from file or create new if file doesn't exist
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        match Self::load(path) {
            Ok(ml) => ml,
            Err(_) => Self::new(),
        }
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        format!(
            "updates={}, history={}, speed={:.1}, success_rate={:.1}%",
            self.meta_update_count,
            self.adaptation_history.len(),
            self.get_adaptation_speed(),
            self.get_success_rate() * 100.0
        )
    }
}

/// Calculate accuracy of a calibrator on given trades
pub fn calculate_accuracy(
    weights: &[f64; NUM_FEATURES],
    bias: f64,
    trades: &[TradeOutcome],
) -> f64 {
    if trades.is_empty() {
        return 0.5;
    }

    let mut correct = 0;
    for trade in trades {
        let features = ConfidenceCalibrator::encode_features(
            trade.sr_score,
            trade.volume_pct,
            &trade.regime,
        );

        // Forward pass
        let mut z = bias;
        for i in 0..NUM_FEATURES {
            z += weights[i] * features[i];
        }
        let pred = 1.0 / (1.0 + (-z).exp()); // sigmoid

        // Prediction matches outcome
        let predicted_win = pred >= 0.5;
        if predicted_win == trade.won {
            correct += 1;
        }
    }

    correct as f64 / trades.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(sr_score: i32, volume_pct: f64, regime: Regime, won: bool) -> TradeOutcome {
        TradeOutcome {
            sr_score,
            volume_pct,
            regime,
            won,
        }
    }

    #[test]
    fn test_metalearner_creation() {
        let ml = MetaLearner::new();

        assert_eq!(ml.meta_update_count(), 0);
        assert_eq!(ml.adaptation_history_len(), 0);
        assert!((ml.outer_lr() - DEFAULT_OUTER_LR).abs() < 0.001);
        assert!((ml.inner_lr() - DEFAULT_INNER_LR).abs() < 0.001);
        assert_eq!(ml.inner_steps(), DEFAULT_INNER_STEPS);
    }

    #[test]
    fn test_get_initialization() {
        let ml = MetaLearner::new();
        let (weights, bias) = ml.get_initialization();

        // Default initialization is zeros
        for w in weights.iter() {
            assert!((w - 0.0).abs() < 0.001);
        }
        assert!((bias - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_adapt_empty_trades() {
        let ml = MetaLearner::new();
        let (weights, bias) = ml.adapt(&[]);

        // With no trades, should return meta_weights unchanged
        let (meta_w, meta_b) = ml.get_initialization();
        for i in 0..NUM_FEATURES {
            assert!((weights[i] - meta_w[i]).abs() < 0.001);
        }
        assert!((bias - meta_b).abs() < 0.001);
    }

    #[test]
    fn test_adapt_modifies_weights() {
        let ml = MetaLearner::new();

        // Create some trades
        let trades = vec![
            make_trade(0, 85.0, Regime::TrendingUp, true),
            make_trade(0, 80.0, Regime::TrendingUp, true),
            make_trade(-3, 50.0, Regime::TrendingUp, false),
        ];

        let (adapted_weights, adapted_bias) = ml.adapt(&trades);

        // Adapted weights should differ from meta_weights (all zeros)
        let mut any_changed = false;
        for w in adapted_weights.iter() {
            if w.abs() > 0.0001 {
                any_changed = true;
                break;
            }
        }
        assert!(any_changed || adapted_bias.abs() > 0.0001);
    }

    #[test]
    fn test_meta_update_success() {
        let mut ml = MetaLearner::new();

        let pre_weights = [0.0; NUM_FEATURES];
        let pre_bias = 0.0;
        let post_weights = [0.1, 0.2, -0.1, 0.05, -0.05, 0.15];
        let post_bias = 0.1;

        ml.meta_update(&pre_weights, pre_bias, &post_weights, post_bias, true);

        // Meta weights should have moved toward post_weights
        let (meta_w, meta_b) = ml.get_initialization();
        for i in 0..NUM_FEATURES {
            // outer_lr = 0.1, so meta_w[i] = 0 + 0.1 * post_weights[i]
            let expected = 0.1 * post_weights[i];
            assert!((meta_w[i] - expected).abs() < 0.001);
        }
        let expected_bias = 0.1 * post_bias;
        assert!((meta_b - expected_bias).abs() < 0.001);

        assert_eq!(ml.meta_update_count(), 1);
    }

    #[test]
    fn test_meta_update_skipped_on_failure() {
        let mut ml = MetaLearner::new();

        let pre_weights = [0.0; NUM_FEATURES];
        let pre_bias = 0.0;
        let post_weights = [0.1, 0.2, -0.1, 0.05, -0.05, 0.15];
        let post_bias = 0.1;

        ml.meta_update(&pre_weights, pre_bias, &post_weights, post_bias, false);

        // Meta weights should NOT have changed
        let (meta_w, meta_b) = ml.get_initialization();
        for i in 0..NUM_FEATURES {
            assert!((meta_w[i] - 0.0).abs() < 0.001);
        }
        assert!((meta_b - 0.0).abs() < 0.001);

        assert_eq!(ml.meta_update_count(), 0);
    }

    #[test]
    fn test_record_adaptation() {
        let mut ml = MetaLearner::new();

        let result = AdaptationResult {
            regime: Regime::TrendingUp,
            initial_weights: [0.0; NUM_FEATURES],
            adapted_weights: [0.1; NUM_FEATURES],
            trades_used: 15,
            pre_adaptation_accuracy: 0.45,
            post_adaptation_accuracy: 0.60,
        };

        ml.record_adaptation(result);

        assert_eq!(ml.adaptation_history_len(), 1);
    }

    #[test]
    fn test_adaptation_history_limit() {
        let mut ml = MetaLearner::new();

        // Add more than MAX_ADAPTATION_HISTORY entries
        for i in 0..MAX_ADAPTATION_HISTORY + 10 {
            let result = AdaptationResult {
                regime: Regime::TrendingUp,
                initial_weights: [0.0; NUM_FEATURES],
                adapted_weights: [i as f64 * 0.01; NUM_FEATURES],
                trades_used: 10,
                pre_adaptation_accuracy: 0.50,
                post_adaptation_accuracy: 0.55,
            };
            ml.record_adaptation(result);
        }

        // Should be capped at MAX_ADAPTATION_HISTORY
        assert_eq!(ml.adaptation_history_len(), MAX_ADAPTATION_HISTORY);
    }

    #[test]
    fn test_adaptation_result_was_successful() {
        let successful = AdaptationResult {
            regime: Regime::TrendingUp,
            initial_weights: [0.0; NUM_FEATURES],
            adapted_weights: [0.1; NUM_FEATURES],
            trades_used: 10,
            pre_adaptation_accuracy: 0.45,
            post_adaptation_accuracy: 0.60,
        };
        assert!(successful.was_successful());
        assert!((successful.improvement() - 0.15).abs() < 0.001);

        let failed = AdaptationResult {
            regime: Regime::TrendingDown,
            initial_weights: [0.0; NUM_FEATURES],
            adapted_weights: [0.1; NUM_FEATURES],
            trades_used: 10,
            pre_adaptation_accuracy: 0.55,
            post_adaptation_accuracy: 0.50,
        };
        assert!(!failed.was_successful());
        assert!((failed.improvement() - (-0.05)).abs() < 0.001);
    }

    #[test]
    fn test_get_success_rate() {
        let mut ml = MetaLearner::new();

        // Add 3 successful, 2 failed
        for i in 0..5 {
            let result = AdaptationResult {
                regime: Regime::TrendingUp,
                initial_weights: [0.0; NUM_FEATURES],
                adapted_weights: [0.1; NUM_FEATURES],
                trades_used: 10,
                pre_adaptation_accuracy: 0.50,
                post_adaptation_accuracy: if i < 3 { 0.60 } else { 0.45 },
            };
            ml.record_adaptation(result);
        }

        let success_rate = ml.get_success_rate();
        assert!((success_rate - 0.6).abs() < 0.001); // 3/5 = 0.6
    }

    #[test]
    fn test_get_adaptation_speed() {
        let mut ml = MetaLearner::new();

        // Add adaptations with varying trades_used
        for trades in [10, 15, 20] {
            let result = AdaptationResult {
                regime: Regime::TrendingUp,
                initial_weights: [0.0; NUM_FEATURES],
                adapted_weights: [0.1; NUM_FEATURES],
                trades_used: trades,
                pre_adaptation_accuracy: 0.50,
                post_adaptation_accuracy: 0.60, // All successful (>55%)
            };
            ml.record_adaptation(result);
        }

        let speed = ml.get_adaptation_speed();
        // Average of 10, 15, 20 = 15
        assert!((speed - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_accuracy() {
        let weights = [0.0; NUM_FEATURES];
        let bias = 0.0;

        // With zero weights, prediction is always 0.5 (sigmoid(0) = 0.5)
        // So it predicts win for all, accuracy depends on actual outcomes
        let trades = vec![
            make_trade(0, 80.0, Regime::TrendingUp, true),  // Correct
            make_trade(0, 80.0, Regime::TrendingUp, true),  // Correct
            make_trade(0, 80.0, Regime::TrendingUp, false), // Wrong
            make_trade(0, 80.0, Regime::TrendingUp, false), // Wrong
        ];

        let acc = calculate_accuracy(&weights, bias, &trades);
        assert!((acc - 0.5).abs() < 0.001); // 2/4 = 0.5
    }

    #[test]
    fn test_save_load() {
        let mut ml = MetaLearner::new();

        // Modify some state
        let trades = vec![
            make_trade(0, 85.0, Regime::TrendingUp, true),
            make_trade(-2, 60.0, Regime::TrendingUp, false),
        ];
        let (adapted_w, adapted_b) = ml.adapt(&trades);
        ml.meta_update(&[0.0; NUM_FEATURES], 0.0, &adapted_w, adapted_b, true);

        let result = AdaptationResult {
            regime: Regime::TrendingUp,
            initial_weights: [0.0; NUM_FEATURES],
            adapted_weights: adapted_w,
            trades_used: 2,
            pre_adaptation_accuracy: 0.50,
            post_adaptation_accuracy: 0.60,
        };
        ml.record_adaptation(result);

        // Save
        let path = "/tmp/test_metalearner.json";
        ml.save(path).unwrap();

        // Load
        let loaded = MetaLearner::load(path).unwrap();

        assert_eq!(loaded.meta_update_count(), ml.meta_update_count());
        assert_eq!(loaded.adaptation_history_len(), ml.adaptation_history_len());

        let (loaded_w, loaded_b) = loaded.get_initialization();
        let (orig_w, orig_b) = ml.get_initialization();
        for i in 0..NUM_FEATURES {
            assert!((loaded_w[i] - orig_w[i]).abs() < 1e-10);
        }
        assert!((loaded_b - orig_b).abs() < 1e-10);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_or_new() {
        // Non-existent file should return new MetaLearner
        let ml = MetaLearner::load_or_new("/tmp/nonexistent_metalearner_12345.json");
        assert_eq!(ml.meta_update_count(), 0);
    }

    #[test]
    fn test_with_learning_rates() {
        let ml = MetaLearner::with_learning_rates(0.05, 0.01, 10);

        assert!((ml.outer_lr() - 0.05).abs() < 0.001);
        assert!((ml.inner_lr() - 0.01).abs() < 0.001);
        assert_eq!(ml.inner_steps(), 10);
    }

    #[test]
    fn test_format_summary() {
        let ml = MetaLearner::new();
        let summary = ml.format_summary();

        assert!(summary.contains("updates="));
        assert!(summary.contains("history="));
        assert!(summary.contains("speed="));
        assert!(summary.contains("success_rate="));
    }

    #[test]
    fn test_set_learning_rates() {
        let mut ml = MetaLearner::new();

        ml.set_outer_lr(0.2);
        assert!((ml.outer_lr() - 0.2).abs() < 0.001);

        ml.set_inner_lr(0.05);
        assert!((ml.inner_lr() - 0.05).abs() < 0.001);

        ml.set_inner_steps(10);
        assert_eq!(ml.inner_steps(), 10);

        // Test bounds
        ml.set_outer_lr(0.0001); // Should clamp to 0.001
        assert!((ml.outer_lr() - 0.001).abs() < 0.001);

        ml.set_inner_steps(100); // Should clamp to 20
        assert_eq!(ml.inner_steps(), 20);
    }

    #[test]
    fn test_average_improvement() {
        let mut ml = MetaLearner::new();

        // Add adaptations with different improvements
        for (pre, post) in [(0.45, 0.55), (0.50, 0.60), (0.55, 0.50)] {
            let result = AdaptationResult {
                regime: Regime::TrendingUp,
                initial_weights: [0.0; NUM_FEATURES],
                adapted_weights: [0.1; NUM_FEATURES],
                trades_used: 10,
                pre_adaptation_accuracy: pre,
                post_adaptation_accuracy: post,
            };
            ml.record_adaptation(result);
        }

        // Improvements: 0.10, 0.10, -0.05 = 0.15 / 3 = 0.05
        let avg = ml.get_average_improvement();
        assert!((avg - 0.05).abs() < 0.001);
    }
}
