//! LSTM Sequence Modeling for Regime Transition Prediction
//!
//! Implements Long Short-Term Memory (LSTM) cells for learning temporal
//! patterns in market data to predict regime transitions.
//!
//! Architecture:
//! - LSTMCell: Core LSTM unit with input/forget/cell/output gates
//! - SequenceEncoder: Encodes rolling window of market features
//! - RegimePredictor: Predicts regime transition probabilities
//!
//! Benefits:
//! - Learn temporal patterns that precede regime changes
//! - Enable proactive position management before transitions
//! - Improve MoE gating with predicted regime probabilities

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use tracing::info;

use super::regime::Regime;

/// Default sequence length (number of bars to consider)
const DEFAULT_SEQUENCE_LENGTH: usize = 20;

/// Default hidden size for LSTM
const DEFAULT_HIDDEN_SIZE: usize = 32;

/// Default feature size (return, vol, regime, sr_score, vol_pct)
const FEATURE_SIZE: usize = 5;

/// Number of regime classes
const NUM_REGIMES: usize = 4;

// ==================== Activation Functions ====================

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Tanh activation function
fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

/// Softmax for probability distribution
fn softmax(x: &[f64]) -> Vec<f64> {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Vec<f64> = x.iter().map(|&xi| (xi - max_x).exp()).collect();
    let sum: f64 = exp_x.iter().sum();
    exp_x.iter().map(|&e| e / sum).collect()
}

// ==================== Matrix Operations ====================

/// Matrix-vector multiplication: W * x
fn mat_vec_mul(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Element-wise vector addition
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise vector multiplication (Hadamard product)
fn vec_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Xavier initialization for weight matrix
fn xavier_init(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-scale..scale)).collect())
        .collect()
}

// ==================== LSTM Cell ====================

/// LSTM Cell implementing the standard LSTM equations
///
/// Gates:
/// - Input gate (i): controls what new information to store
/// - Forget gate (f): controls what to forget from cell state
/// - Cell gate (c): candidate values to add to cell state
/// - Output gate (o): controls what to output from cell state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMCell {
    /// Input dimension
    pub input_size: usize,
    /// Hidden state dimension
    pub hidden_size: usize,

    // Input gate weights
    w_i: Vec<Vec<f64>>, // [hidden_size x input_size]
    u_i: Vec<Vec<f64>>, // [hidden_size x hidden_size]
    b_i: Vec<f64>,      // [hidden_size]

    // Forget gate weights
    w_f: Vec<Vec<f64>>,
    u_f: Vec<Vec<f64>>,
    b_f: Vec<f64>,

    // Cell gate weights
    w_c: Vec<Vec<f64>>,
    u_c: Vec<Vec<f64>>,
    b_c: Vec<f64>,

    // Output gate weights
    w_o: Vec<Vec<f64>>,
    u_o: Vec<Vec<f64>>,
    b_o: Vec<f64>,

    // State
    h: Vec<f64>, // Hidden state [hidden_size]
    c: Vec<f64>, // Cell state [hidden_size]
}

impl LSTMCell {
    /// Create a new LSTM cell with Xavier initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // Xavier initialize all weights
        let w_i = xavier_init(hidden_size, input_size);
        let u_i = xavier_init(hidden_size, hidden_size);
        let w_f = xavier_init(hidden_size, input_size);
        let u_f = xavier_init(hidden_size, hidden_size);
        let w_c = xavier_init(hidden_size, input_size);
        let u_c = xavier_init(hidden_size, hidden_size);
        let w_o = xavier_init(hidden_size, input_size);
        let u_o = xavier_init(hidden_size, hidden_size);

        // Zero initialize biases, except forget gate bias = 1.0
        // (helps with gradient flow in early training)
        let b_i = vec![0.0; hidden_size];
        let b_f = vec![1.0; hidden_size]; // Forget gate bias = 1.0
        let b_c = vec![0.0; hidden_size];
        let b_o = vec![0.0; hidden_size];

        // Zero initialize states
        let h = vec![0.0; hidden_size];
        let c = vec![0.0; hidden_size];

        Self {
            input_size,
            hidden_size,
            w_i,
            u_i,
            b_i,
            w_f,
            u_f,
            b_f,
            w_c,
            u_c,
            b_c,
            w_o,
            u_o,
            b_o,
            h,
            c,
        }
    }

    /// Reset hidden and cell states to zero
    pub fn reset_state(&mut self) {
        self.h = vec![0.0; self.hidden_size];
        self.c = vec![0.0; self.hidden_size];
    }

    /// Forward pass through the LSTM cell
    ///
    /// LSTM equations:
    /// i_t = σ(W_i * x_t + U_i * h_{t-1} + b_i)
    /// f_t = σ(W_f * x_t + U_f * h_{t-1} + b_f)
    /// c̃_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)
    /// c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    /// o_t = σ(W_o * x_t + U_o * h_{t-1} + b_o)
    /// h_t = o_t ⊙ tanh(c_t)
    pub fn forward(&mut self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.input_size, "Input size mismatch");

        // Input gate: i_t = σ(W_i * x + U_i * h + b_i)
        let wx_i = mat_vec_mul(&self.w_i, x);
        let uh_i = mat_vec_mul(&self.u_i, &self.h);
        let i_t: Vec<f64> = vec_add(&vec_add(&wx_i, &uh_i), &self.b_i)
            .iter()
            .map(|&v| sigmoid(v))
            .collect();

        // Forget gate: f_t = σ(W_f * x + U_f * h + b_f)
        let wx_f = mat_vec_mul(&self.w_f, x);
        let uh_f = mat_vec_mul(&self.u_f, &self.h);
        let f_t: Vec<f64> = vec_add(&vec_add(&wx_f, &uh_f), &self.b_f)
            .iter()
            .map(|&v| sigmoid(v))
            .collect();

        // Candidate cell state: c̃_t = tanh(W_c * x + U_c * h + b_c)
        let wx_c = mat_vec_mul(&self.w_c, x);
        let uh_c = mat_vec_mul(&self.u_c, &self.h);
        let c_tilde: Vec<f64> = vec_add(&vec_add(&wx_c, &uh_c), &self.b_c)
            .iter()
            .map(|&v| tanh_activation(v))
            .collect();

        // New cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        let fc = vec_mul(&f_t, &self.c);
        let ic = vec_mul(&i_t, &c_tilde);
        self.c = vec_add(&fc, &ic);

        // Output gate: o_t = σ(W_o * x + U_o * h + b_o)
        let wx_o = mat_vec_mul(&self.w_o, x);
        let uh_o = mat_vec_mul(&self.u_o, &self.h);
        let o_t: Vec<f64> = vec_add(&vec_add(&wx_o, &uh_o), &self.b_o)
            .iter()
            .map(|&v| sigmoid(v))
            .collect();

        // New hidden state: h_t = o_t ⊙ tanh(c_t)
        let tanh_c: Vec<f64> = self.c.iter().map(|&v| tanh_activation(v)).collect();
        self.h = vec_mul(&o_t, &tanh_c);

        self.h.clone()
    }

    /// Get current state (hidden, cell)
    pub fn get_state(&self) -> (&[f64], &[f64]) {
        (&self.h, &self.c)
    }

    /// Set state (for restoring from checkpoint)
    pub fn set_state(&mut self, h: Vec<f64>, c: Vec<f64>) {
        assert_eq!(h.len(), self.hidden_size);
        assert_eq!(c.len(), self.hidden_size);
        self.h = h;
        self.c = c;
    }
}

// ==================== Market Features ====================

/// Market features for sequence encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFeatures {
    /// 1-bar return (normalized)
    pub return_1bar: f64,
    /// Volatility estimate (normalized)
    pub volatility: f64,
    /// Current regime
    pub regime: Regime,
    /// S/R score at current price
    pub sr_score: i32,
    /// Volume percentile
    pub volume_percentile: f64,
}

impl MarketFeatures {
    /// Create new market features
    pub fn new(
        return_1bar: f64,
        volatility: f64,
        regime: Regime,
        sr_score: i32,
        volume_percentile: f64,
    ) -> Self {
        Self {
            return_1bar,
            volatility,
            regime,
            sr_score,
            volume_percentile,
        }
    }

    /// Convert to feature vector for LSTM input
    pub fn to_vector(&self) -> Vec<f64> {
        let regime_encoded = match self.regime {
            Regime::TrendingUp => 0.0,
            Regime::TrendingDown => 0.25,
            Regime::Ranging => 0.5,
            Regime::Volatile => 0.75,
        };

        vec![
            self.return_1bar.clamp(-1.0, 1.0),
            self.volatility.clamp(0.0, 1.0),
            regime_encoded,
            (self.sr_score as f64 / -10.0).clamp(-1.0, 1.0),
            (self.volume_percentile / 100.0).clamp(0.0, 1.0),
        ]
    }
}

// ==================== Sequence Encoder ====================

/// Encodes a sequence of market features using LSTM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceEncoder {
    /// LSTM cell for sequence processing
    lstm: LSTMCell,
    /// Number of time steps to consider
    sequence_length: usize,
    /// Feature dimension
    feature_size: usize,
    /// Rolling buffer of feature vectors
    buffer: VecDeque<Vec<f64>>,
}

impl SequenceEncoder {
    /// Create new sequence encoder
    pub fn new(hidden_size: usize) -> Self {
        Self {
            lstm: LSTMCell::new(FEATURE_SIZE, hidden_size),
            sequence_length: DEFAULT_SEQUENCE_LENGTH,
            feature_size: FEATURE_SIZE,
            buffer: VecDeque::with_capacity(DEFAULT_SEQUENCE_LENGTH),
        }
    }

    /// Create with custom sequence length
    pub fn with_sequence_length(hidden_size: usize, sequence_length: usize) -> Self {
        Self {
            lstm: LSTMCell::new(FEATURE_SIZE, hidden_size),
            sequence_length,
            feature_size: FEATURE_SIZE,
            buffer: VecDeque::with_capacity(sequence_length),
        }
    }

    /// Push new features into the buffer
    pub fn push(&mut self, features: MarketFeatures) {
        let feature_vec = features.to_vector();

        // Ring buffer: drop oldest if full
        if self.buffer.len() >= self.sequence_length {
            self.buffer.pop_front();
        }
        self.buffer.push_back(feature_vec);
    }

    /// Encode the current sequence into a fixed-size vector
    pub fn encode(&mut self) -> Vec<f64> {
        // Reset LSTM state for fresh encoding
        self.lstm.reset_state();

        // Forward pass through all buffered features
        let mut encoding = vec![0.0; self.lstm.hidden_size];
        for features in self.buffer.iter() {
            encoding = self.lstm.forward(features);
        }

        encoding
    }

    /// Check if buffer has enough data for encoding
    pub fn is_ready(&self) -> bool {
        self.buffer.len() >= self.sequence_length
    }

    /// Get current buffer length
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.lstm.reset_state();
    }
}

// ==================== Regime Predictor ====================

/// Predicts regime transitions using LSTM sequence encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePredictor {
    /// Sequence encoder
    encoder: SequenceEncoder,
    /// Prediction head weights [hidden_size x NUM_REGIMES]
    prediction_head: Vec<Vec<f64>>,
    /// Prediction bias [NUM_REGIMES]
    prediction_bias: [f64; NUM_REGIMES],
    /// Learning rate for updates
    learning_rate: f64,
    /// Number of predictions made
    prediction_count: u64,
    /// Number of correct predictions
    correct_predictions: u64,
    /// Recent prediction history for accuracy tracking
    #[serde(skip)]
    recent_predictions: VecDeque<(usize, usize)>, // (predicted, actual)
}

impl RegimePredictor {
    /// Create new regime predictor
    pub fn new(hidden_size: usize) -> Self {
        // Initialize prediction head with Xavier
        let prediction_head = xavier_init(NUM_REGIMES, hidden_size);
        let prediction_bias = [0.0; NUM_REGIMES];

        Self {
            encoder: SequenceEncoder::new(hidden_size),
            prediction_head,
            prediction_bias,
            learning_rate: 0.01,
            prediction_count: 0,
            correct_predictions: 0,
            recent_predictions: VecDeque::with_capacity(100),
        }
    }

    /// Predict next regime probabilities
    pub fn predict_next_regime(&mut self, features: MarketFeatures) -> [f64; NUM_REGIMES] {
        // Push features to encoder
        self.encoder.push(features);

        // If not ready, return uniform distribution
        if !self.encoder.is_ready() {
            return [0.25, 0.25, 0.25, 0.25];
        }

        // Get encoding
        let encoding = self.encoder.encode();

        // Linear layer: logits = W * encoding + bias
        let mut logits = [0.0; NUM_REGIMES];
        for (i, row) in self.prediction_head.iter().enumerate() {
            logits[i] = row.iter().zip(encoding.iter()).map(|(a, b)| a * b).sum::<f64>()
                + self.prediction_bias[i];
        }

        // Softmax to get probabilities
        let probs = softmax(&logits);
        let mut result = [0.0; NUM_REGIMES];
        for (i, p) in probs.iter().enumerate() {
            result[i] = *p;
        }

        result
    }

    /// Get regime index from Regime enum
    fn regime_to_index(regime: &Regime) -> usize {
        match regime {
            Regime::TrendingUp => 0,
            Regime::TrendingDown => 1,
            Regime::Ranging => 2,
            Regime::Volatile => 3,
        }
    }

    /// Get Regime enum from index
    fn index_to_regime(index: usize) -> Regime {
        match index {
            0 => Regime::TrendingUp,
            1 => Regime::TrendingDown,
            2 => Regime::Ranging,
            _ => Regime::Volatile,
        }
    }

    /// Update predictor with actual regime outcome
    pub fn update(&mut self, actual_regime: &Regime) {
        if !self.encoder.is_ready() {
            return;
        }

        let actual_idx = Self::regime_to_index(actual_regime);

        // Get current prediction
        let encoding = self.encoder.encode();
        let mut logits = [0.0; NUM_REGIMES];
        for (i, row) in self.prediction_head.iter().enumerate() {
            logits[i] = row.iter().zip(encoding.iter()).map(|(a, b)| a * b).sum::<f64>()
                + self.prediction_bias[i];
        }
        let probs = softmax(&logits);

        // Track accuracy
        let predicted_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.prediction_count += 1;
        if predicted_idx == actual_idx {
            self.correct_predictions += 1;
        }

        // Update recent predictions buffer
        if self.recent_predictions.len() >= 100 {
            self.recent_predictions.pop_front();
        }
        self.recent_predictions.push_back((predicted_idx, actual_idx));

        // Backprop through prediction head (simplified: cross-entropy gradient)
        // Gradient of softmax cross-entropy: pred - one_hot(actual)
        let mut grad = probs.clone();
        grad[actual_idx] -= 1.0;

        // Update weights: W -= lr * grad * encoding^T
        for i in 0..NUM_REGIMES {
            for j in 0..encoding.len() {
                self.prediction_head[i][j] -= self.learning_rate * grad[i] * encoding[j];
            }
            self.prediction_bias[i] -= self.learning_rate * grad[i];
        }
    }

    /// Get transition probability from one regime to another
    pub fn get_transition_probability(&self, from: &Regime, to: &Regime) -> f64 {
        // Count transitions in recent history
        let from_idx = Self::regime_to_index(from);
        let to_idx = Self::regime_to_index(to);

        let mut transitions = 0;
        let mut from_count = 0;

        for i in 1..self.recent_predictions.len() {
            let (_, prev_actual) = self.recent_predictions[i - 1];
            let (_, curr_actual) = self.recent_predictions[i];

            if prev_actual == from_idx {
                from_count += 1;
                if curr_actual == to_idx {
                    transitions += 1;
                }
            }
        }

        if from_count == 0 {
            // Default: small probability for transition, higher for staying
            if from_idx == to_idx {
                0.7
            } else {
                0.1
            }
        } else {
            transitions as f64 / from_count as f64
        }
    }

    /// Predict if a regime change is likely
    pub fn predict_regime_change(&mut self, current_regime: &Regime) -> Option<(Regime, f64)> {
        let probs = self.predict_next_regime(MarketFeatures::new(
            0.0,
            0.0,
            current_regime.clone(),
            0,
            50.0,
        ));

        let current_idx = Self::regime_to_index(current_regime);

        // Find highest probability regime that's different from current
        let mut max_prob = 0.0;
        let mut max_idx = current_idx;

        for (i, &p) in probs.iter().enumerate() {
            if i != current_idx && p > max_prob {
                max_prob = p;
                max_idx = i;
            }
        }

        // Return if probability > 35% (significant change likelihood)
        if max_prob > 0.35 && max_idx != current_idx {
            Some((Self::index_to_regime(max_idx), max_prob))
        } else {
            None
        }
    }

    /// Get prediction accuracy
    pub fn accuracy(&self) -> f64 {
        if self.prediction_count == 0 {
            0.5
        } else {
            self.correct_predictions as f64 / self.prediction_count as f64
        }
    }

    /// Get recent accuracy (last 100 predictions)
    pub fn recent_accuracy(&self) -> f64 {
        if self.recent_predictions.is_empty() {
            return 0.5;
        }

        let correct = self
            .recent_predictions
            .iter()
            .filter(|(pred, actual)| pred == actual)
            .count();

        correct as f64 / self.recent_predictions.len() as f64
    }

    /// Get prediction count
    pub fn prediction_count(&self) -> u64 {
        self.prediction_count
    }

    /// Check if encoder is ready
    pub fn is_ready(&self) -> bool {
        self.encoder.is_ready()
    }

    /// Push features without predicting
    pub fn push_features(&mut self, features: MarketFeatures) {
        self.encoder.push(features);
    }

    /// Get regime probabilities as array (for MoE integration)
    pub fn get_regime_probabilities(&mut self, features: MarketFeatures) -> [f64; NUM_REGIMES] {
        self.predict_next_regime(features)
    }

    /// Save predictor to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        info!(
            "[SEQUENCE] Saved to {} ({} predictions, {:.1}% accuracy)",
            path,
            self.prediction_count,
            self.accuracy() * 100.0
        );
        Ok(())
    }

    /// Load predictor from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let mut predictor: Self = serde_json::from_str(&json)?;
        // Reinitialize non-serialized fields
        predictor.recent_predictions = VecDeque::with_capacity(100);
        Ok(predictor)
    }

    /// Load or create new predictor
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        Self::load(&path).unwrap_or_else(|_| Self::new(DEFAULT_HIDDEN_SIZE))
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        format!(
            "predictions={}, accuracy={:.1}%, recent={:.1}%",
            self.prediction_count,
            self.accuracy() * 100.0,
            self.recent_accuracy() * 100.0
        )
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_forward() {
        let mut lstm = LSTMCell::new(5, 10);

        // Test forward pass
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let output = lstm.forward(&input);

        assert_eq!(output.len(), 10);
        // Output should be bounded by tanh
        for &val in output.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_lstm_sequence() {
        let mut lstm = LSTMCell::new(5, 10);

        // Process multiple time steps
        let inputs = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![0.2, 0.3, 0.4, 0.5, 0.6],
            vec![0.3, 0.4, 0.5, 0.6, 0.7],
        ];

        let mut outputs = Vec::new();
        for input in inputs {
            outputs.push(lstm.forward(&input));
        }

        // Each output should be different (state evolves)
        assert_ne!(outputs[0], outputs[1]);
        assert_ne!(outputs[1], outputs[2]);

        // After reset, same input should give same output
        lstm.reset_state();
        let output1 = lstm.forward(&vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        lstm.reset_state();
        let output2 = lstm.forward(&vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        assert_eq!(output1, output2);
    }

    #[test]
    fn test_lstm_state() {
        let mut lstm = LSTMCell::new(5, 10);

        // Initial state should be zeros
        let (h, c) = lstm.get_state();
        assert!(h.iter().all(|&x| x == 0.0));
        assert!(c.iter().all(|&x| x == 0.0));

        // After forward, state should change
        lstm.forward(&vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let (h, c) = lstm.get_state();
        assert!(!h.iter().all(|&x| x == 0.0));
        assert!(!c.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sequence_encoder() {
        let mut encoder = SequenceEncoder::with_sequence_length(16, 5);

        // Not ready initially
        assert!(!encoder.is_ready());

        // Push features
        for i in 0..5 {
            let features = MarketFeatures::new(
                0.01 * i as f64,
                0.5,
                Regime::TrendingUp,
                -2,
                70.0,
            );
            encoder.push(features);
        }

        // Should be ready after sequence_length features
        assert!(encoder.is_ready());
        assert_eq!(encoder.buffer_len(), 5);

        // Encoding should produce fixed-size output
        let encoding = encoder.encode();
        assert_eq!(encoding.len(), 16);
    }

    #[test]
    fn test_regime_prediction() {
        let mut predictor = RegimePredictor::new(16);

        // Push enough features to be ready
        for i in 0..25 {
            let features = MarketFeatures::new(
                0.01 * (i % 5) as f64,
                0.3,
                if i < 15 {
                    Regime::TrendingUp
                } else {
                    Regime::Volatile
                },
                -(i % 3) as i32,
                60.0 + (i % 20) as f64,
            );
            predictor.push_features(features);
        }

        assert!(predictor.is_ready());

        // Predict
        let features = MarketFeatures::new(0.02, 0.4, Regime::TrendingUp, -1, 75.0);
        let probs = predictor.predict_next_regime(features);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // All probabilities should be positive
        for p in probs.iter() {
            assert!(*p >= 0.0);
        }
    }

    #[test]
    fn test_regime_prediction_update() {
        let mut predictor = RegimePredictor::new(16);

        // Fill buffer
        for _ in 0..25 {
            let features = MarketFeatures::new(0.01, 0.3, Regime::TrendingUp, -2, 70.0);
            predictor.push_features(features);
        }

        // Update with actual regime
        predictor.update(&Regime::TrendingUp);
        assert_eq!(predictor.prediction_count(), 1);

        // Accuracy tracking
        predictor.update(&Regime::TrendingUp);
        predictor.update(&Regime::TrendingDown);
        assert_eq!(predictor.prediction_count(), 3);
    }

    #[test]
    fn test_market_features() {
        let features = MarketFeatures::new(0.05, 0.3, Regime::Volatile, -5, 80.0);
        let vec = features.to_vector();

        assert_eq!(vec.len(), FEATURE_SIZE);
        assert_eq!(vec[0], 0.05); // return (clamped)
        assert_eq!(vec[1], 0.3); // volatility
        assert_eq!(vec[2], 0.75); // regime encoded (Volatile)
        assert_eq!(vec[3], 0.5); // sr_score normalized
        assert_eq!(vec[4], 0.8); // volume_percentile normalized
    }

    #[test]
    fn test_transition_probability() {
        let mut predictor = RegimePredictor::new(16);

        // Fill buffer and make predictions
        for _ in 0..25 {
            let features = MarketFeatures::new(0.01, 0.3, Regime::TrendingUp, -2, 70.0);
            predictor.push_features(features);
        }

        // Record some transitions
        predictor.update(&Regime::TrendingUp);
        predictor.update(&Regime::TrendingUp);
        predictor.update(&Regime::TrendingDown);
        predictor.update(&Regime::TrendingDown);
        predictor.update(&Regime::TrendingUp);

        // Get transition probability
        let prob = predictor.get_transition_probability(&Regime::TrendingUp, &Regime::TrendingDown);
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);

        // Larger logit should have larger probability
        assert!(probs[3] > probs[2]);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_regime_change_prediction() {
        let mut predictor = RegimePredictor::new(16);

        // Fill with trending up data
        for _ in 0..25 {
            let features = MarketFeatures::new(0.02, 0.2, Regime::TrendingUp, -1, 75.0);
            predictor.push_features(features);
        }

        // Predict regime change (may or may not predict change depending on random init)
        let change = predictor.predict_regime_change(&Regime::TrendingUp);
        // Just verify it returns a valid result
        if let Some((regime, prob)) = change {
            assert!(prob > 0.0 && prob <= 1.0);
            assert_ne!(regime, Regime::TrendingUp);
        }
    }
}
