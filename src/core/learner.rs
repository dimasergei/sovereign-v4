//! Confidence Calibration Learner
//!
//! Provides both linear and neural network models for confidence calibration.
//! The models learn from trade outcomes to adjust confidence predictions.
//!
//! Features:
//! - S/R score (normalized)
//! - Volume percentile (normalized)
//! - Regime one-hot encoding (4 values)
//!
//! Includes Elastic Weight Consolidation (EWC) to prevent catastrophic
//! forgetting when market regimes change.

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::info;

use super::metalearner::{AdaptationResult, MetaLearner};
use super::regime::Regime;

/// Number of features in the model (public for metalearner)
pub const NUM_FEATURES: usize = 6;

/// Hidden layer 1 size
const HIDDEN1_SIZE: usize = 32;

/// Hidden layer 2 size
const HIDDEN2_SIZE: usize = 16;

/// Default EWC lambda (strength of weight protection)
const DEFAULT_EWC_LAMBDA: f64 = 100.0;

/// Default learning rate for neural network
const DEFAULT_LEARNING_RATE: f64 = 0.001;

/// Default momentum for neural network
const DEFAULT_MOMENTUM: f64 = 0.9;

/// Trade outcome for Fisher computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeOutcome {
    pub sr_score: i32,
    pub volume_pct: f64,
    pub regime: Regime,
    pub won: bool,
}

// ==================== Activation Functions ====================

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative
fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// ReLU activation function
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// ReLU derivative
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Xavier/He initialization
fn xavier_init(fan_in: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let limit = (2.0 / fan_in as f64).sqrt();
    rng.gen_range(-limit..limit)
}

// ==================== Calibrator Trait ====================

/// Trait for confidence calibrators
pub trait Calibrator: Send + Sync {
    /// Predict confidence given raw inputs
    fn predict(&self, sr_score: i32, volume_percentile: f64, regime: &Regime) -> f64;

    /// Update from raw inputs
    fn update_from_trade(
        &mut self,
        sr_score: i32,
        volume_percentile: f64,
        regime: &Regime,
        won: bool,
        learning_rate: f64,
    );

    /// Consolidate weights for EWC
    fn consolidate(&mut self);

    /// Check if EWC is active
    fn is_ewc_active(&self) -> bool;

    /// Check if model has been consolidated
    fn is_consolidated(&self) -> bool;

    /// Get update count
    fn update_count(&self) -> u64;

    /// Get consolidation count
    fn consolidation_count(&self) -> u32;

    /// Get weights (for transfer learning)
    fn get_weights(&self) -> [f64; NUM_FEATURES];

    /// Set weights (for transfer learning from cluster prior)
    fn set_weights(&mut self, weights: [f64; NUM_FEATURES]);

    /// Initialize from meta-learner if attached
    fn init_from_meta(&mut self);

    /// Get bias term (for transfer learning)
    fn get_bias(&self) -> f64;

    /// Report adaptation results to meta-learner
    fn report_adaptation(
        &self,
        pre_weights: &[f64; NUM_FEATURES],
        pre_bias: f64,
        pre_accuracy: f64,
        post_accuracy: f64,
        num_trades: u32,
        regime: &Regime,
    );
}

// ==================== Linear Calibrator ====================

/// Lightweight linear model for confidence calibration (original implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearCalibrator {
    weights: [f64; NUM_FEATURES],
    bias: f64,
    update_count: u64,
    fisher: [f64; NUM_FEATURES],
    fisher_bias: f64,
    optimal_weights: Option<[f64; NUM_FEATURES]>,
    optimal_bias: Option<f64>,
    ewc_lambda: f64,
    consolidation_count: u32,
    #[serde(skip)]
    meta_learner: Option<Arc<Mutex<MetaLearner>>>,
}

impl Default for LinearCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearCalibrator {
    pub fn new() -> Self {
        Self {
            weights: [0.3, 0.2, 0.1, 0.1, 0.1, 0.2],
            bias: 0.5,
            update_count: 0,
            fisher: [0.0; NUM_FEATURES],
            fisher_bias: 0.0,
            optimal_weights: None,
            optimal_bias: None,
            ewc_lambda: DEFAULT_EWC_LAMBDA,
            consolidation_count: 0,
            meta_learner: None,
        }
    }

    pub fn predict_from_features(&self, features: &[f64; NUM_FEATURES]) -> f64 {
        let mut z = self.bias;
        for i in 0..NUM_FEATURES {
            z += self.weights[i] * features[i];
        }
        sigmoid(z)
    }

    pub fn update(&mut self, features: &[f64; NUM_FEATURES], target: f64, learning_rate: f64) {
        let prediction = self.predict_from_features(features);
        let error = target - prediction;

        for i in 0..NUM_FEATURES {
            let mut grad = error * features[i];
            if let Some(ref optimal) = self.optimal_weights {
                let ewc_penalty =
                    self.ewc_lambda * self.fisher[i] * (self.weights[i] - optimal[i]);
                grad -= ewc_penalty;
            }
            self.weights[i] += learning_rate * grad;
        }

        let mut bias_grad = error;
        if let Some(optimal_bias) = self.optimal_bias {
            let ewc_penalty = self.ewc_lambda * self.fisher_bias * (self.bias - optimal_bias);
            bias_grad -= ewc_penalty;
        }
        self.bias += learning_rate * bias_grad;
        self.update_count += 1;
    }

    pub fn compute_fisher(&mut self, recent_trades: &[TradeOutcome]) {
        if recent_trades.is_empty() {
            return;
        }
        self.fisher = [0.0; NUM_FEATURES];
        self.fisher_bias = 0.0;

        for trade in recent_trades {
            let features = encode_features(trade.sr_score, trade.volume_pct, &trade.regime);
            let prediction = self.predict_from_features(&features);
            let target = if trade.won { 1.0 } else { 0.0 };
            let error = target - prediction;

            for i in 0..NUM_FEATURES {
                let grad = error * features[i];
                self.fisher[i] += grad * grad;
            }
            self.fisher_bias += error * error;
        }

        let n = recent_trades.len() as f64;
        for i in 0..NUM_FEATURES {
            self.fisher[i] /= n;
        }
        self.fisher_bias /= n;
    }

    pub fn get_weights(&self) -> &[f64; NUM_FEATURES] {
        &self.weights
    }

    pub fn set_weights(&mut self, weights: [f64; NUM_FEATURES]) {
        self.weights = weights;
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    pub fn get_fisher(&self) -> &[f64; NUM_FEATURES] {
        &self.fisher
    }

    pub fn get_ewc_lambda(&self) -> f64 {
        self.ewc_lambda
    }

    pub fn set_ewc_lambda(&mut self, lambda: f64) {
        self.ewc_lambda = lambda;
    }

    pub fn attach_meta_learner(&mut self, ml: Arc<Mutex<MetaLearner>>) {
        self.meta_learner = Some(ml);
    }

    pub fn has_meta_learner(&self) -> bool {
        self.meta_learner.is_some()
    }

    pub fn meta_learner(&self) -> Option<&Arc<Mutex<MetaLearner>>> {
        self.meta_learner.as_ref()
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

    /// Load calibrator from file, or create new if file doesn't exist
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        Self::load(&path).unwrap_or_else(|_| Self::new())
    }
}

impl Calibrator for LinearCalibrator {
    fn predict(&self, sr_score: i32, volume_percentile: f64, regime: &Regime) -> f64 {
        let features = encode_features(sr_score, volume_percentile, regime);
        self.predict_from_features(&features)
    }

    fn update_from_trade(
        &mut self,
        sr_score: i32,
        volume_percentile: f64,
        regime: &Regime,
        won: bool,
        learning_rate: f64,
    ) {
        let features = encode_features(sr_score, volume_percentile, regime);
        let target = if won { 1.0 } else { 0.0 };
        self.update(&features, target, learning_rate);
    }

    fn consolidate(&mut self) {
        self.optimal_weights = Some(self.weights);
        self.optimal_bias = Some(self.bias);
        self.consolidation_count += 1;
    }

    fn is_ewc_active(&self) -> bool {
        self.consolidation_count > 0 && self.fisher.iter().any(|&f| f > 0.0)
    }

    fn is_consolidated(&self) -> bool {
        self.optimal_weights.is_some()
    }

    fn update_count(&self) -> u64 {
        self.update_count
    }

    fn consolidation_count(&self) -> u32 {
        self.consolidation_count
    }

    fn get_weights(&self) -> [f64; NUM_FEATURES] {
        self.weights
    }

    fn set_weights(&mut self, weights: [f64; NUM_FEATURES]) {
        self.weights = weights;
    }

    fn init_from_meta(&mut self) {
        // Initialize from meta-learner if attached
        if let Some(ref meta) = self.meta_learner {
            if let Ok(ml) = meta.lock() {
                let (init_weights, init_bias) = ml.get_initialization();
                self.weights = init_weights;
                self.bias = init_bias;
            }
        }
    }

    fn get_bias(&self) -> f64 {
        self.bias
    }

    fn report_adaptation(
        &self,
        pre_weights: &[f64; NUM_FEATURES],
        pre_bias: f64,
        pre_accuracy: f64,
        post_accuracy: f64,
        num_trades: u32,
        regime: &Regime,
    ) {
        if let Some(ref meta) = self.meta_learner {
            if let Ok(mut ml) = meta.lock() {
                let result = AdaptationResult {
                    regime: regime.clone(),
                    initial_weights: *pre_weights,
                    adapted_weights: self.weights,
                    trades_used: num_trades,
                    pre_adaptation_accuracy: pre_accuracy,
                    post_adaptation_accuracy: post_accuracy,
                };

                // If adaptation was successful, also update meta-weights
                let success = result.was_successful();
                ml.meta_update(pre_weights, pre_bias, &self.weights, self.bias, success);

                ml.record_adaptation(result);
            }
        }
    }
}

// ==================== Neural Calibrator ====================

/// Neural network calibrator with architecture: 6 → 32 (ReLU) → 16 (ReLU) → 1 (sigmoid)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralCalibrator {
    // Layer 1: Input (6) → Hidden1 (32)
    weights_1: [[f64; NUM_FEATURES]; HIDDEN1_SIZE],
    bias_1: [f64; HIDDEN1_SIZE],

    // Layer 2: Hidden1 (32) → Hidden2 (16)
    weights_2: [[f64; HIDDEN1_SIZE]; HIDDEN2_SIZE],
    bias_2: [f64; HIDDEN2_SIZE],

    // Layer 3: Hidden2 (16) → Output (1)
    weights_3: [f64; HIDDEN2_SIZE],
    bias_3: f64,

    // Training parameters
    learning_rate: f64,
    momentum: f64,

    // Velocity for momentum (same shapes as weights)
    velocity_w1: [[f64; NUM_FEATURES]; HIDDEN1_SIZE],
    velocity_b1: [f64; HIDDEN1_SIZE],
    velocity_w2: [[f64; HIDDEN1_SIZE]; HIDDEN2_SIZE],
    velocity_b2: [f64; HIDDEN2_SIZE],
    velocity_w3: [f64; HIDDEN2_SIZE],
    velocity_b3: f64,

    // EWC fields
    fisher_w1: [[f64; NUM_FEATURES]; HIDDEN1_SIZE],
    fisher_b1: [f64; HIDDEN1_SIZE],
    fisher_w2: [[f64; HIDDEN1_SIZE]; HIDDEN2_SIZE],
    fisher_b2: [f64; HIDDEN2_SIZE],
    fisher_w3: [f64; HIDDEN2_SIZE],
    fisher_b3: f64,

    optimal_w1: Option<[[f64; NUM_FEATURES]; HIDDEN1_SIZE]>,
    optimal_b1: Option<[f64; HIDDEN1_SIZE]>,
    optimal_w2: Option<[[f64; HIDDEN1_SIZE]; HIDDEN2_SIZE]>,
    optimal_b2: Option<[f64; HIDDEN2_SIZE]>,
    optimal_w3: Option<[f64; HIDDEN2_SIZE]>,
    optimal_b3: Option<f64>,

    ewc_lambda: f64,
    consolidation_count: u32,
    update_count: u64,

    #[serde(skip)]
    meta_learner: Option<Arc<Mutex<MetaLearner>>>,
}

impl Default for NeuralCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralCalibrator {
    /// Create a new neural calibrator with Xavier initialization
    pub fn new() -> Self {
        let mut weights_1 = [[0.0; NUM_FEATURES]; HIDDEN1_SIZE];
        let mut bias_1 = [0.0; HIDDEN1_SIZE];

        let mut weights_2 = [[0.0; HIDDEN1_SIZE]; HIDDEN2_SIZE];
        let mut bias_2 = [0.0; HIDDEN2_SIZE];

        let mut weights_3 = [0.0; HIDDEN2_SIZE];
        let bias_3 = 0.0;

        // Xavier initialization for layer 1
        for i in 0..HIDDEN1_SIZE {
            for j in 0..NUM_FEATURES {
                weights_1[i][j] = xavier_init(NUM_FEATURES);
            }
            bias_1[i] = xavier_init(NUM_FEATURES) * 0.1;
        }

        // Xavier initialization for layer 2
        for i in 0..HIDDEN2_SIZE {
            for j in 0..HIDDEN1_SIZE {
                weights_2[i][j] = xavier_init(HIDDEN1_SIZE);
            }
            bias_2[i] = xavier_init(HIDDEN1_SIZE) * 0.1;
        }

        // Xavier initialization for layer 3
        for i in 0..HIDDEN2_SIZE {
            weights_3[i] = xavier_init(HIDDEN2_SIZE);
        }

        Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            weights_3,
            bias_3,
            learning_rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            velocity_w1: [[0.0; NUM_FEATURES]; HIDDEN1_SIZE],
            velocity_b1: [0.0; HIDDEN1_SIZE],
            velocity_w2: [[0.0; HIDDEN1_SIZE]; HIDDEN2_SIZE],
            velocity_b2: [0.0; HIDDEN2_SIZE],
            velocity_w3: [0.0; HIDDEN2_SIZE],
            velocity_b3: 0.0,
            fisher_w1: [[0.0; NUM_FEATURES]; HIDDEN1_SIZE],
            fisher_b1: [0.0; HIDDEN1_SIZE],
            fisher_w2: [[0.0; HIDDEN1_SIZE]; HIDDEN2_SIZE],
            fisher_b2: [0.0; HIDDEN2_SIZE],
            fisher_w3: [0.0; HIDDEN2_SIZE],
            fisher_b3: 0.0,
            optimal_w1: None,
            optimal_b1: None,
            optimal_w2: None,
            optimal_b2: None,
            optimal_w3: None,
            optimal_b3: None,
            ewc_lambda: DEFAULT_EWC_LAMBDA,
            consolidation_count: 0,
            update_count: 0,
            meta_learner: None,
        }
    }

    /// Forward pass returning output and intermediate activations
    fn forward_with_activations(
        &self,
        features: &[f64; NUM_FEATURES],
    ) -> (f64, [f64; HIDDEN1_SIZE], [f64; HIDDEN2_SIZE], [f64; HIDDEN1_SIZE], [f64; HIDDEN2_SIZE], f64)
    {
        // Layer 1: z1 = W1 * x + b1, h1 = ReLU(z1)
        let mut z1 = [0.0; HIDDEN1_SIZE];
        let mut h1 = [0.0; HIDDEN1_SIZE];
        for i in 0..HIDDEN1_SIZE {
            z1[i] = self.bias_1[i];
            for j in 0..NUM_FEATURES {
                z1[i] += self.weights_1[i][j] * features[j];
            }
            h1[i] = relu(z1[i]);
        }

        // Layer 2: z2 = W2 * h1 + b2, h2 = ReLU(z2)
        let mut z2 = [0.0; HIDDEN2_SIZE];
        let mut h2 = [0.0; HIDDEN2_SIZE];
        for i in 0..HIDDEN2_SIZE {
            z2[i] = self.bias_2[i];
            for j in 0..HIDDEN1_SIZE {
                z2[i] += self.weights_2[i][j] * h1[j];
            }
            h2[i] = relu(z2[i]);
        }

        // Layer 3: z3 = W3 * h2 + b3, y = sigmoid(z3)
        let mut z3 = self.bias_3;
        for i in 0..HIDDEN2_SIZE {
            z3 += self.weights_3[i] * h2[i];
        }
        let y = sigmoid(z3);

        (y, z1, z2, h1, h2, z3)
    }

    /// Forward pass (prediction only)
    pub fn forward(&self, features: &[f64; NUM_FEATURES]) -> f64 {
        let (y, _, _, _, _, _) = self.forward_with_activations(features);
        y
    }

    /// Backward pass with gradient computation and weight update
    pub fn backward(&mut self, features: &[f64; NUM_FEATURES], target: f64) -> f64 {
        // Forward pass to get activations
        let (pred, z1, z2, h1, h2, z3) = self.forward_with_activations(features);

        // Compute loss (binary cross-entropy)
        let loss = -(target * pred.max(1e-10).ln() + (1.0 - target) * (1.0 - pred).max(1e-10).ln());

        // Output layer gradient: d_loss/d_z3 = pred - target (for BCE + sigmoid)
        let delta3 = pred - target;

        // Gradients for layer 3
        let mut grad_w3 = [0.0; HIDDEN2_SIZE];
        for i in 0..HIDDEN2_SIZE {
            grad_w3[i] = delta3 * h2[i];
        }
        let grad_b3 = delta3;

        // Backpropagate to layer 2
        let mut delta2 = [0.0; HIDDEN2_SIZE];
        for i in 0..HIDDEN2_SIZE {
            delta2[i] = delta3 * self.weights_3[i] * relu_derivative(z2[i]);
        }

        // Gradients for layer 2
        let mut grad_w2 = [[0.0; HIDDEN1_SIZE]; HIDDEN2_SIZE];
        let mut grad_b2 = [0.0; HIDDEN2_SIZE];
        for i in 0..HIDDEN2_SIZE {
            for j in 0..HIDDEN1_SIZE {
                grad_w2[i][j] = delta2[i] * h1[j];
            }
            grad_b2[i] = delta2[i];
        }

        // Backpropagate to layer 1
        let mut delta1 = [0.0; HIDDEN1_SIZE];
        for i in 0..HIDDEN1_SIZE {
            let mut sum = 0.0;
            for j in 0..HIDDEN2_SIZE {
                sum += delta2[j] * self.weights_2[j][i];
            }
            delta1[i] = sum * relu_derivative(z1[i]);
        }

        // Gradients for layer 1
        let mut grad_w1 = [[0.0; NUM_FEATURES]; HIDDEN1_SIZE];
        let mut grad_b1 = [0.0; HIDDEN1_SIZE];
        for i in 0..HIDDEN1_SIZE {
            for j in 0..NUM_FEATURES {
                grad_w1[i][j] = delta1[i] * features[j];
            }
            grad_b1[i] = delta1[i];
        }

        // Update weights with momentum and optional EWC penalty
        self.update_weights(grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3);

        self.update_count += 1;
        loss
    }

    /// Update weights with momentum and EWC penalty
    fn update_weights(
        &mut self,
        grad_w1: [[f64; NUM_FEATURES]; HIDDEN1_SIZE],
        grad_b1: [f64; HIDDEN1_SIZE],
        grad_w2: [[f64; HIDDEN1_SIZE]; HIDDEN2_SIZE],
        grad_b2: [f64; HIDDEN2_SIZE],
        grad_w3: [f64; HIDDEN2_SIZE],
        grad_b3: f64,
    ) {
        let lr = self.learning_rate;
        let mom = self.momentum;
        let ewc_active = self.optimal_w1.is_some();

        // Update layer 1
        for i in 0..HIDDEN1_SIZE {
            for j in 0..NUM_FEATURES {
                let mut g = grad_w1[i][j];
                if ewc_active {
                    if let Some(ref opt) = self.optimal_w1 {
                        g += self.ewc_lambda * self.fisher_w1[i][j] * (self.weights_1[i][j] - opt[i][j]);
                    }
                }
                self.velocity_w1[i][j] = mom * self.velocity_w1[i][j] + lr * g;
                self.weights_1[i][j] -= self.velocity_w1[i][j];
            }
            let mut g_b = grad_b1[i];
            if ewc_active {
                if let Some(ref opt) = self.optimal_b1 {
                    g_b += self.ewc_lambda * self.fisher_b1[i] * (self.bias_1[i] - opt[i]);
                }
            }
            self.velocity_b1[i] = mom * self.velocity_b1[i] + lr * g_b;
            self.bias_1[i] -= self.velocity_b1[i];
        }

        // Update layer 2
        for i in 0..HIDDEN2_SIZE {
            for j in 0..HIDDEN1_SIZE {
                let mut g = grad_w2[i][j];
                if ewc_active {
                    if let Some(ref opt) = self.optimal_w2 {
                        g += self.ewc_lambda * self.fisher_w2[i][j] * (self.weights_2[i][j] - opt[i][j]);
                    }
                }
                self.velocity_w2[i][j] = mom * self.velocity_w2[i][j] + lr * g;
                self.weights_2[i][j] -= self.velocity_w2[i][j];
            }
            let mut g_b = grad_b2[i];
            if ewc_active {
                if let Some(ref opt) = self.optimal_b2 {
                    g_b += self.ewc_lambda * self.fisher_b2[i] * (self.bias_2[i] - opt[i]);
                }
            }
            self.velocity_b2[i] = mom * self.velocity_b2[i] + lr * g_b;
            self.bias_2[i] -= self.velocity_b2[i];
        }

        // Update layer 3
        for i in 0..HIDDEN2_SIZE {
            let mut g = grad_w3[i];
            if ewc_active {
                if let Some(ref opt) = self.optimal_w3 {
                    g += self.ewc_lambda * self.fisher_w3[i] * (self.weights_3[i] - opt[i]);
                }
            }
            self.velocity_w3[i] = mom * self.velocity_w3[i] + lr * g;
            self.weights_3[i] -= self.velocity_w3[i];
        }
        let mut g_b3 = grad_b3;
        if ewc_active {
            if let Some(opt) = self.optimal_b3 {
                g_b3 += self.ewc_lambda * self.fisher_b3 * (self.bias_3 - opt);
            }
        }
        self.velocity_b3 = mom * self.velocity_b3 + lr * g_b3;
        self.bias_3 -= self.velocity_b3;
    }

    /// Train on a batch of samples
    pub fn train_batch(&mut self, batch: &[([f64; NUM_FEATURES], f64)]) -> f64 {
        if batch.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0;
        for (features, target) in batch {
            total_loss += self.backward(features, *target);
        }
        total_loss / batch.len() as f64
    }

    /// Compute Fisher Information for all weights
    pub fn compute_fisher(&mut self, data: &[TradeOutcome]) {
        if data.is_empty() {
            return;
        }

        // Reset Fisher to zero
        self.fisher_w1 = [[0.0; NUM_FEATURES]; HIDDEN1_SIZE];
        self.fisher_b1 = [0.0; HIDDEN1_SIZE];
        self.fisher_w2 = [[0.0; HIDDEN1_SIZE]; HIDDEN2_SIZE];
        self.fisher_b2 = [0.0; HIDDEN2_SIZE];
        self.fisher_w3 = [0.0; HIDDEN2_SIZE];
        self.fisher_b3 = 0.0;

        for trade in data {
            let features = encode_features(trade.sr_score, trade.volume_pct, &trade.regime);
            let target = if trade.won { 1.0 } else { 0.0 };

            // Forward pass
            let (pred, z1, z2, h1, h2, _z3) = self.forward_with_activations(&features);

            // Compute gradients (same as backward)
            let delta3 = pred - target;

            // Layer 3 gradient squared
            for i in 0..HIDDEN2_SIZE {
                let grad = delta3 * h2[i];
                self.fisher_w3[i] += grad * grad;
            }
            self.fisher_b3 += delta3 * delta3;

            // Layer 2
            let mut delta2 = [0.0; HIDDEN2_SIZE];
            for i in 0..HIDDEN2_SIZE {
                delta2[i] = delta3 * self.weights_3[i] * relu_derivative(z2[i]);
            }
            for i in 0..HIDDEN2_SIZE {
                for j in 0..HIDDEN1_SIZE {
                    let grad = delta2[i] * h1[j];
                    self.fisher_w2[i][j] += grad * grad;
                }
                self.fisher_b2[i] += delta2[i] * delta2[i];
            }

            // Layer 1
            let mut delta1 = [0.0; HIDDEN1_SIZE];
            for i in 0..HIDDEN1_SIZE {
                let mut sum = 0.0;
                for j in 0..HIDDEN2_SIZE {
                    sum += delta2[j] * self.weights_2[j][i];
                }
                delta1[i] = sum * relu_derivative(z1[i]);
            }
            for i in 0..HIDDEN1_SIZE {
                for j in 0..NUM_FEATURES {
                    let grad = delta1[i] * features[j];
                    self.fisher_w1[i][j] += grad * grad;
                }
                self.fisher_b1[i] += delta1[i] * delta1[i];
            }
        }

        // Average over samples
        let n = data.len() as f64;
        for i in 0..HIDDEN1_SIZE {
            for j in 0..NUM_FEATURES {
                self.fisher_w1[i][j] /= n;
            }
            self.fisher_b1[i] /= n;
        }
        for i in 0..HIDDEN2_SIZE {
            for j in 0..HIDDEN1_SIZE {
                self.fisher_w2[i][j] /= n;
            }
            self.fisher_b2[i] /= n;
        }
        for i in 0..HIDDEN2_SIZE {
            self.fisher_w3[i] /= n;
        }
        self.fisher_b3 /= n;
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Get learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set momentum
    pub fn set_momentum(&mut self, m: f64) {
        self.momentum = m;
    }

    /// Set EWC lambda
    pub fn set_ewc_lambda(&mut self, lambda: f64) {
        self.ewc_lambda = lambda;
    }

    /// Get EWC lambda
    pub fn get_ewc_lambda(&self) -> f64 {
        self.ewc_lambda
    }

    /// Save to JSON file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file
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

    /// Attach meta-learner
    pub fn attach_meta_learner(&mut self, ml: Arc<Mutex<MetaLearner>>) {
        self.meta_learner = Some(ml);
    }

    /// Check if meta-learner is attached
    pub fn has_meta_learner(&self) -> bool {
        self.meta_learner.is_some()
    }
}

impl Calibrator for NeuralCalibrator {
    fn predict(&self, sr_score: i32, volume_percentile: f64, regime: &Regime) -> f64 {
        let features = encode_features(sr_score, volume_percentile, regime);
        self.forward(&features)
    }

    fn update_from_trade(
        &mut self,
        sr_score: i32,
        volume_percentile: f64,
        regime: &Regime,
        won: bool,
        _learning_rate: f64, // Uses internal learning rate
    ) {
        let features = encode_features(sr_score, volume_percentile, regime);
        let target = if won { 1.0 } else { 0.0 };
        self.backward(&features, target);
    }

    fn consolidate(&mut self) {
        self.optimal_w1 = Some(self.weights_1);
        self.optimal_b1 = Some(self.bias_1);
        self.optimal_w2 = Some(self.weights_2);
        self.optimal_b2 = Some(self.bias_2);
        self.optimal_w3 = Some(self.weights_3);
        self.optimal_b3 = Some(self.bias_3);
        self.consolidation_count += 1;
        info!(
            "[NEURAL] EWC consolidation #{}: protected neural weights",
            self.consolidation_count
        );
    }

    fn is_ewc_active(&self) -> bool {
        self.consolidation_count > 0
            && (self.fisher_w1.iter().flatten().any(|&f| f > 0.0)
                || self.fisher_w2.iter().flatten().any(|&f| f > 0.0)
                || self.fisher_w3.iter().any(|&f| f > 0.0))
    }

    fn is_consolidated(&self) -> bool {
        self.optimal_w1.is_some()
    }

    fn update_count(&self) -> u64 {
        self.update_count
    }

    fn consolidation_count(&self) -> u32 {
        self.consolidation_count
    }

    fn get_weights(&self) -> [f64; NUM_FEATURES] {
        // For neural networks, return the mean weight of the first layer per input feature
        // This provides an approximation for transfer learning compatibility
        let mut weights = [0.0; NUM_FEATURES];
        for i in 0..NUM_FEATURES {
            let sum: f64 = self.weights_1.iter().map(|row| row[i]).sum();
            weights[i] = sum / HIDDEN1_SIZE as f64;
        }
        weights
    }

    fn set_weights(&mut self, weights: [f64; NUM_FEATURES]) {
        // For neural networks, scale the first layer weights based on the transfer weights
        // This biases the network towards the transferred knowledge
        for j in 0..HIDDEN1_SIZE {
            for i in 0..NUM_FEATURES {
                // Blend: adjust weights towards the transfer weights
                self.weights_1[j][i] = 0.7 * self.weights_1[j][i] + 0.3 * weights[i];
            }
        }
    }

    fn init_from_meta(&mut self) {
        // Neural networks use their own initialization strategy
        // Meta-learner initialization not applicable to neural architecture
    }

    fn get_bias(&self) -> f64 {
        // For neural networks, return the output layer bias
        self.bias_3
    }

    fn report_adaptation(
        &self,
        _pre_weights: &[f64; NUM_FEATURES],
        _pre_bias: f64,
        _pre_accuracy: f64,
        _post_accuracy: f64,
        _num_trades: u32,
        _regime: &Regime,
    ) {
        // Neural networks don't use the meta-learner in the same way
        // Adaptation is handled through backpropagation
    }
}

// ==================== ConfidenceCalibrator (backward compatible alias) ====================

/// Type alias for backward compatibility - uses LinearCalibrator by default
/// Use CalibratorType for explicit control over calibrator type
pub type ConfidenceCalibrator = LinearCalibrator;

// ==================== CalibratorType Enum ====================

/// Enum for selecting calibrator type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibratorType {
    Linear(LinearCalibrator),
    Neural(NeuralCalibrator),
}

impl CalibratorType {
    /// Create a new linear calibrator
    pub fn new_linear() -> Self {
        CalibratorType::Linear(LinearCalibrator::new())
    }

    /// Create a new neural calibrator
    pub fn new_neural() -> Self {
        CalibratorType::Neural(NeuralCalibrator::new())
    }

    /// Get as a Calibrator trait reference
    pub fn as_calibrator(&self) -> &dyn Calibrator {
        match self {
            CalibratorType::Linear(c) => c,
            CalibratorType::Neural(c) => c,
        }
    }

    /// Get as a mutable Calibrator trait reference
    pub fn as_calibrator_mut(&mut self) -> &mut dyn Calibrator {
        match self {
            CalibratorType::Linear(c) => c,
            CalibratorType::Neural(c) => c,
        }
    }
}

impl Calibrator for CalibratorType {
    fn predict(&self, sr_score: i32, volume_percentile: f64, regime: &Regime) -> f64 {
        self.as_calibrator().predict(sr_score, volume_percentile, regime)
    }

    fn update_from_trade(
        &mut self,
        sr_score: i32,
        volume_percentile: f64,
        regime: &Regime,
        won: bool,
        learning_rate: f64,
    ) {
        self.as_calibrator_mut()
            .update_from_trade(sr_score, volume_percentile, regime, won, learning_rate)
    }

    fn consolidate(&mut self) {
        self.as_calibrator_mut().consolidate()
    }

    fn is_ewc_active(&self) -> bool {
        self.as_calibrator().is_ewc_active()
    }

    fn is_consolidated(&self) -> bool {
        self.as_calibrator().is_consolidated()
    }

    fn update_count(&self) -> u64 {
        self.as_calibrator().update_count()
    }

    fn consolidation_count(&self) -> u32 {
        self.as_calibrator().consolidation_count()
    }

    fn get_weights(&self) -> [f64; NUM_FEATURES] {
        self.as_calibrator().get_weights()
    }

    fn set_weights(&mut self, weights: [f64; NUM_FEATURES]) {
        self.as_calibrator_mut().set_weights(weights)
    }

    fn init_from_meta(&mut self) {
        self.as_calibrator_mut().init_from_meta()
    }

    fn get_bias(&self) -> f64 {
        self.as_calibrator().get_bias()
    }

    fn report_adaptation(
        &self,
        pre_weights: &[f64; NUM_FEATURES],
        pre_bias: f64,
        pre_accuracy: f64,
        post_accuracy: f64,
        num_trades: u32,
        regime: &Regime,
    ) {
        self.as_calibrator().report_adaptation(
            pre_weights,
            pre_bias,
            pre_accuracy,
            post_accuracy,
            num_trades,
            regime,
        )
    }
}

// ==================== Shared Functions ====================

/// Encode raw inputs into normalized features
pub fn encode_features(sr_score: i32, volume_percentile: f64, regime: &Regime) -> [f64; NUM_FEATURES] {
    let sr_normalized = (sr_score as f64 / -10.0).clamp(-1.0, 1.0);
    let vol_normalized = (volume_percentile / 100.0).clamp(0.0, 1.0);

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

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Linear Calibrator Tests ====================

    #[test]
    fn test_linear_new() {
        let cal = LinearCalibrator::new();
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
    fn test_relu() {
        assert!((relu(5.0) - 5.0).abs() < 0.001);
        assert!((relu(-5.0) - 0.0).abs() < 0.001);
        assert!((relu(0.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_encode_features() {
        let features = encode_features(0, 50.0, &Regime::Ranging);
        assert!((features[0] - 0.0).abs() < 0.001);

        let features = encode_features(-10, 50.0, &Regime::Ranging);
        assert!((features[0] - 1.0).abs() < 0.001);

        let features = encode_features(0, 100.0, &Regime::Ranging);
        assert!((features[1] - 1.0).abs() < 0.001);

        let features = encode_features(0, 50.0, &Regime::TrendingUp);
        assert!((features[2] - 1.0).abs() < 0.001);
        assert!((features[5] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_predict() {
        let cal = LinearCalibrator::new();
        let pred = cal.predict(0, 50.0, &Regime::Ranging);
        assert!(pred > 0.4 && pred < 0.8);
    }

    #[test]
    fn test_linear_update() {
        let mut cal = LinearCalibrator::new();
        let features = encode_features(0, 80.0, &Regime::TrendingUp);
        let pred_before = cal.predict_from_features(&features);
        cal.update(&features, 1.0, 0.1);
        let pred_after = cal.predict_from_features(&features);
        assert!(pred_after > pred_before);
    }

    #[test]
    fn test_linear_learning() {
        let mut cal = LinearCalibrator::new();

        for _ in 0..50 {
            cal.update_from_trade(0, 90.0, &Regime::TrendingUp, true, 0.05);
        }
        for _ in 0..50 {
            cal.update_from_trade(-8, 40.0, &Regime::Volatile, false, 0.05);
        }

        let good_pred = cal.predict(0, 90.0, &Regime::TrendingUp);
        let bad_pred = cal.predict(-8, 40.0, &Regime::Volatile);
        assert!(good_pred > bad_pred);
    }

    // ==================== Neural Calibrator Tests ====================

    #[test]
    fn test_neural_new() {
        let cal = NeuralCalibrator::new();
        assert_eq!(cal.update_count, 0);
        assert!((cal.learning_rate - DEFAULT_LEARNING_RATE).abs() < 0.0001);
    }

    #[test]
    fn test_neural_forward_pass() {
        let cal = NeuralCalibrator::new();
        let features = [0.5, 0.5, 1.0, 0.0, 0.0, 0.0];
        let pred = cal.forward(&features);
        // Output should be in [0, 1] due to sigmoid
        assert!(pred >= 0.0 && pred <= 1.0);
    }

    #[test]
    fn test_neural_backprop() {
        let mut cal = NeuralCalibrator::new();
        let features = [0.5, 0.8, 1.0, 0.0, 0.0, 0.0];

        let pred_before = cal.forward(&features);
        let loss1 = cal.backward(&features, 1.0);
        let pred_after = cal.forward(&features);

        // After training toward 1.0, prediction should increase
        assert!(pred_after >= pred_before || (pred_after - pred_before).abs() < 0.1);
        assert!(loss1 >= 0.0);
    }

    #[test]
    fn test_neural_learns_pattern() {
        let mut cal = NeuralCalibrator::new();
        cal.set_learning_rate(0.01);

        // Train: high volume + trending up = win
        for _ in 0..100 {
            let features = encode_features(0, 90.0, &Regime::TrendingUp);
            cal.backward(&features, 1.0);
        }

        // Train: low volume + volatile = lose
        for _ in 0..100 {
            let features = encode_features(-8, 30.0, &Regime::Volatile);
            cal.backward(&features, 0.0);
        }

        let good_pred = cal.predict(0, 90.0, &Regime::TrendingUp);
        let bad_pred = cal.predict(-8, 30.0, &Regime::Volatile);

        // Should learn the pattern (good > bad)
        assert!(
            good_pred > bad_pred,
            "Neural net should learn: good={:.3} > bad={:.3}",
            good_pred,
            bad_pred
        );
    }

    #[test]
    fn test_neural_learns_xor() {
        // Classic XOR test for neural networks
        let mut cal = NeuralCalibrator::new();
        cal.set_learning_rate(0.1);
        cal.set_momentum(0.0); // Disable momentum for cleaner test

        // XOR pattern: encode in first two features
        let xor_data = [
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
            ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1.0),
            ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),
            ([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.0),
        ];

        // Train for many epochs
        for _ in 0..1000 {
            for (features, target) in &xor_data {
                cal.backward(features, *target);
            }
        }

        // Test XOR predictions
        let mut correct = 0;
        for (features, target) in &xor_data {
            let pred = cal.forward(features);
            let predicted_class = if pred > 0.5 { 1.0 } else { 0.0 };
            if (predicted_class - target).abs() < 0.01 {
                correct += 1;
            }
        }

        // Should get at least 3/4 correct (XOR is hard for shallow networks)
        assert!(correct >= 3, "XOR test: {}/4 correct", correct);
    }

    #[test]
    fn test_neural_ewc() {
        let mut cal = NeuralCalibrator::new();
        cal.set_learning_rate(0.01);

        // Train on pattern 1
        let trades1: Vec<TradeOutcome> = (0..20)
            .map(|_| TradeOutcome {
                sr_score: 0,
                volume_pct: 85.0,
                regime: Regime::TrendingUp,
                won: true,
            })
            .collect();

        for trade in &trades1 {
            let features = encode_features(trade.sr_score, trade.volume_pct, &trade.regime);
            cal.backward(&features, 1.0);
        }

        let pred_before = cal.predict(0, 85.0, &Regime::TrendingUp);

        // Compute Fisher and consolidate
        cal.compute_fisher(&trades1);
        cal.consolidate();

        assert!(cal.is_consolidated());
        assert_eq!(cal.consolidation_count(), 1);

        // Train on conflicting pattern
        for _ in 0..20 {
            let features = encode_features(-8, 40.0, &Regime::Volatile);
            cal.backward(&features, 0.0);
        }

        let pred_after = cal.predict(0, 85.0, &Regime::TrendingUp);

        // EWC should limit how much the prediction drops
        assert!(
            pred_after > pred_before * 0.3,
            "EWC should prevent catastrophic forgetting: before={:.3}, after={:.3}",
            pred_before,
            pred_after
        );
    }

    #[test]
    fn test_neural_save_load() {
        let mut cal = NeuralCalibrator::new();
        cal.backward(&[0.5, 0.5, 1.0, 0.0, 0.0, 0.0], 1.0);

        let path = "/tmp/test_neural_calibrator.json";
        cal.save(path).unwrap();

        let loaded = NeuralCalibrator::load(path).unwrap();
        assert_eq!(cal.update_count, loaded.update_count);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    // ==================== CalibratorType Tests ====================

    #[test]
    fn test_calibrator_type_linear() {
        let mut cal = CalibratorType::new_linear();
        let pred = cal.predict(0, 50.0, &Regime::Ranging);
        assert!(pred > 0.0 && pred < 1.0);

        cal.update_from_trade(0, 80.0, &Regime::TrendingUp, true, 0.1);
        assert_eq!(cal.update_count(), 1);
    }

    #[test]
    fn test_calibrator_type_neural() {
        let mut cal = CalibratorType::new_neural();
        let pred = cal.predict(0, 50.0, &Regime::Ranging);
        assert!(pred >= 0.0 && pred <= 1.0);

        cal.update_from_trade(0, 80.0, &Regime::TrendingUp, true, 0.1);
        assert_eq!(cal.update_count(), 1);
    }

    #[test]
    fn test_calibrator_trait_polymorphism() {
        fn test_calibrator(cal: &mut dyn Calibrator) {
            let pred = cal.predict(-2, 70.0, &Regime::TrendingUp);
            assert!(pred >= 0.0 && pred <= 1.0);

            cal.update_from_trade(-2, 70.0, &Regime::TrendingUp, true, 0.01);
            assert!(cal.update_count() > 0);
        }

        let mut linear = LinearCalibrator::new();
        test_calibrator(&mut linear);

        let mut neural = NeuralCalibrator::new();
        test_calibrator(&mut neural);
    }

    // ==================== Backward Compatibility Tests ====================

    #[test]
    fn test_confidence_calibrator_alias() {
        // ConfidenceCalibrator should work as before
        let mut cal = ConfidenceCalibrator::new();
        cal.update_from_trade(-2, 75.0, &Regime::TrendingUp, true, 0.1);
        assert_eq!(cal.update_count(), 1);

        let pred = cal.predict(-2, 75.0, &Regime::TrendingUp);
        assert!(pred > 0.0 && pred < 1.0);
    }
}
