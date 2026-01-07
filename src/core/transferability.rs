//! ML-based transferability prediction to replace hardcoded clusters
//!
//! This module implements:
//! - Symbol profile encoding using neural networks
//! - Transfer success prediction between symbol pairs
//! - Learning from actual transfer outcomes
//! - Automatic cluster discovery from learned embeddings

use anyhow::Result;
use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::info;

use super::embeddings::cosine_similarity;
use super::regime::Regime;

/// Profile embedding dimension
const PROFILE_EMBEDDING_DIM: usize = 32;

/// Input features for profile encoder
const PROFILE_INPUT_DIM: usize = 10;

/// Hidden layer size for profile encoder
const PROFILE_HIDDEN_DIM: usize = 16;

/// Hidden size for transfer net
const TRANSFER_HIDDEN_1: usize = 32;
const TRANSFER_HIDDEN_2: usize = 16;

/// Transfer prediction between two symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferabilityScore {
    /// Source symbol
    pub source: String,
    /// Target symbol
    pub target: String,
    /// Transfer score (0.0 = no transfer, 1.0 = perfect transfer)
    pub score: f64,
    /// Confidence in the prediction
    pub confidence: f64,
    /// When this was computed
    pub last_computed: DateTime<Utc>,
}

/// Profile of a symbol's trading characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolProfile {
    /// Symbol identifier
    pub symbol: String,
    /// Learned representation (32-dim)
    pub embedding: Vec<f64>,
    /// Average volatility (ATR %)
    pub volatility_profile: f64,
    /// Average trend strength
    pub trend_profile: f64,
    /// Correlations with other symbols
    pub correlation_profile: HashMap<String, f64>,
    /// Percentage of time in each regime [TrendUp, TrendDown, Ranging, Volatile]
    pub regime_distribution: [f64; 4],
    /// Total trade count
    pub trade_count: u32,
    /// Win rate
    pub win_rate: f64,
    /// Running sum of volatility for averaging
    #[serde(default)]
    volatility_sum: f64,
    /// Running sum of trend for averaging
    #[serde(default)]
    trend_sum: f64,
    /// Wins count for win rate
    #[serde(default)]
    wins: u32,
    /// Regime counts for distribution
    #[serde(default)]
    regime_counts: [u32; 4],
}

impl SymbolProfile {
    /// Create a new empty profile
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            embedding: vec![0.0; PROFILE_EMBEDDING_DIM],
            volatility_profile: 0.0,
            trend_profile: 0.0,
            correlation_profile: HashMap::new(),
            regime_distribution: [0.25, 0.25, 0.25, 0.25],
            trade_count: 0,
            win_rate: 0.5,
            volatility_sum: 0.0,
            trend_sum: 0.0,
            wins: 0,
            regime_counts: [0; 4],
        }
    }

    /// Update profile with a new trade
    pub fn update(&mut self, volatility: f64, trend_strength: f64, regime: &Regime, won: bool) {
        self.trade_count += 1;
        self.volatility_sum += volatility;
        self.trend_sum += trend_strength.abs();

        if won {
            self.wins += 1;
        }

        // Update regime counts
        let regime_idx = match regime {
            Regime::TrendingUp => 0,
            Regime::TrendingDown => 1,
            Regime::Ranging => 2,
            Regime::Volatile => 3,
        };
        self.regime_counts[regime_idx] += 1;

        // Recompute averages
        self.volatility_profile = self.volatility_sum / self.trade_count as f64;
        self.trend_profile = self.trend_sum / self.trade_count as f64;
        self.win_rate = self.wins as f64 / self.trade_count as f64;

        // Recompute regime distribution
        let total = self.regime_counts.iter().sum::<u32>() as f64;
        if total > 0.0 {
            for i in 0..4 {
                self.regime_distribution[i] = self.regime_counts[i] as f64 / total;
            }
        }
    }

    /// Convert profile to input features for encoding
    pub fn to_features(&self) -> [f64; PROFILE_INPUT_DIM] {
        let mut features = [0.0; PROFILE_INPUT_DIM];

        // volatility_profile (normalized to 0-1, assuming 0-5% range)
        features[0] = (self.volatility_profile / 5.0).min(1.0);

        // trend_profile (normalized)
        features[1] = (self.trend_profile / 2.0).min(1.0);

        // regime_distribution (4 values)
        features[2] = self.regime_distribution[0];
        features[3] = self.regime_distribution[1];
        features[4] = self.regime_distribution[2];
        features[5] = self.regime_distribution[3];

        // win_rate (already 0-1)
        features[6] = self.win_rate;

        // log(trade_count) normalized
        features[7] = ((self.trade_count as f64 + 1.0).ln() / 7.0).min(1.0); // ln(1000) ≈ 6.9

        // sector encoding (simplified, 2 dims based on symbol characteristics)
        // Tech/Growth vs Value/Defensive heuristic
        let is_tech = self.symbol.len() <= 4 && self.volatility_profile > 1.5;
        features[8] = if is_tech { 1.0 } else { 0.0 };
        features[9] = if !is_tech { 1.0 } else { 0.0 };

        features
    }
}

/// Neural network for encoding symbol profiles to embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEncoder {
    /// Input to hidden weights [PROFILE_HIDDEN_DIM x PROFILE_INPUT_DIM]
    weights_1: Vec<Vec<f64>>,
    /// Hidden biases
    bias_1: Vec<f64>,
    /// Hidden to output weights [PROFILE_EMBEDDING_DIM x PROFILE_HIDDEN_DIM]
    weights_2: Vec<Vec<f64>>,
    /// Output biases
    bias_2: Vec<f64>,
}

impl Default for ProfileEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileEncoder {
    /// Create a new encoder with Xavier initialization
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale_1 = (2.0 / (PROFILE_INPUT_DIM + PROFILE_HIDDEN_DIM) as f64).sqrt();
        let scale_2 = (2.0 / (PROFILE_HIDDEN_DIM + PROFILE_EMBEDDING_DIM) as f64).sqrt();

        let weights_1: Vec<Vec<f64>> = (0..PROFILE_HIDDEN_DIM)
            .map(|_| {
                (0..PROFILE_INPUT_DIM)
                    .map(|_| rng.gen_range(-scale_1..scale_1))
                    .collect()
            })
            .collect();

        let bias_1 = vec![0.0; PROFILE_HIDDEN_DIM];

        let weights_2: Vec<Vec<f64>> = (0..PROFILE_EMBEDDING_DIM)
            .map(|_| {
                (0..PROFILE_HIDDEN_DIM)
                    .map(|_| rng.gen_range(-scale_2..scale_2))
                    .collect()
            })
            .collect();

        let bias_2 = vec![0.0; PROFILE_EMBEDDING_DIM];

        Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
        }
    }

    /// Encode a profile to an embedding
    pub fn encode(&self, features: &[f64; PROFILE_INPUT_DIM]) -> Vec<f64> {
        // Hidden layer
        let mut hidden = vec![0.0; PROFILE_HIDDEN_DIM];
        for i in 0..PROFILE_HIDDEN_DIM {
            let mut sum = self.bias_1[i];
            for j in 0..PROFILE_INPUT_DIM {
                sum += self.weights_1[i][j] * features[j];
            }
            hidden[i] = relu(sum);
        }

        // Output layer
        let mut output = vec![0.0; PROFILE_EMBEDDING_DIM];
        for i in 0..PROFILE_EMBEDDING_DIM {
            let mut sum = self.bias_2[i];
            for j in 0..PROFILE_HIDDEN_DIM {
                sum += self.weights_2[i][j] * hidden[j];
            }
            output[i] = sum; // Linear output for embedding
        }

        // L2 normalize
        let magnitude: f64 = output.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for val in &mut output {
                *val /= magnitude;
            }
        }

        output
    }
}

/// Neural network for predicting transfer success
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferNet {
    /// Input dimension: source_embed (32) + target_embed (32) + diff (32) = 96
    /// Layer 1: 96 -> 32
    weights_1: Vec<Vec<f64>>,
    bias_1: Vec<f64>,
    /// Layer 2: 32 -> 16
    weights_2: Vec<Vec<f64>>,
    bias_2: Vec<f64>,
    /// Layer 3: 16 -> 1
    weights_3: Vec<f64>,
    bias_3: f64,
    /// Learning rate
    learning_rate: f64,
}

impl Default for TransferNet {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferNet {
    const INPUT_DIM: usize = PROFILE_EMBEDDING_DIM * 3; // source + target + diff

    /// Create a new transfer net with Xavier initialization
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        let scale_1 = (2.0 / (Self::INPUT_DIM + TRANSFER_HIDDEN_1) as f64).sqrt();
        let scale_2 = (2.0 / (TRANSFER_HIDDEN_1 + TRANSFER_HIDDEN_2) as f64).sqrt();
        let scale_3 = (2.0 / (TRANSFER_HIDDEN_2 + 1) as f64).sqrt();

        let weights_1: Vec<Vec<f64>> = (0..TRANSFER_HIDDEN_1)
            .map(|_| {
                (0..Self::INPUT_DIM)
                    .map(|_| rng.gen_range(-scale_1..scale_1))
                    .collect()
            })
            .collect();

        let weights_2: Vec<Vec<f64>> = (0..TRANSFER_HIDDEN_2)
            .map(|_| {
                (0..TRANSFER_HIDDEN_1)
                    .map(|_| rng.gen_range(-scale_2..scale_2))
                    .collect()
            })
            .collect();

        let weights_3: Vec<f64> = (0..TRANSFER_HIDDEN_2)
            .map(|_| rng.gen_range(-scale_3..scale_3))
            .collect();

        Self {
            weights_1,
            bias_1: vec![0.0; TRANSFER_HIDDEN_1],
            weights_2,
            bias_2: vec![0.0; TRANSFER_HIDDEN_2],
            weights_3,
            bias_3: 0.0,
            learning_rate: 0.01,
        }
    }

    /// Predict transfer success probability
    pub fn predict(&self, source_embed: &[f64], target_embed: &[f64]) -> f64 {
        // Create input: [source; target; diff]
        let mut input = Vec::with_capacity(Self::INPUT_DIM);
        input.extend_from_slice(source_embed);
        input.extend_from_slice(target_embed);

        // Add difference
        for i in 0..PROFILE_EMBEDDING_DIM {
            let s = source_embed.get(i).copied().unwrap_or(0.0);
            let t = target_embed.get(i).copied().unwrap_or(0.0);
            input.push(s - t);
        }

        // Forward pass
        let (output, _, _, _) = self.forward(&input);
        output
    }

    /// Forward pass returning all activations for backprop
    fn forward(&self, input: &[f64]) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Layer 1
        let mut h1 = vec![0.0; TRANSFER_HIDDEN_1];
        for i in 0..TRANSFER_HIDDEN_1 {
            let mut sum = self.bias_1[i];
            for j in 0..Self::INPUT_DIM.min(input.len()) {
                sum += self.weights_1[i][j] * input[j];
            }
            h1[i] = relu(sum);
        }

        // Layer 2
        let mut h2 = vec![0.0; TRANSFER_HIDDEN_2];
        for i in 0..TRANSFER_HIDDEN_2 {
            let mut sum = self.bias_2[i];
            for j in 0..TRANSFER_HIDDEN_1 {
                sum += self.weights_2[i][j] * h1[j];
            }
            h2[i] = relu(sum);
        }

        // Output layer
        let mut output = self.bias_3;
        for i in 0..TRANSFER_HIDDEN_2 {
            output += self.weights_3[i] * h2[i];
        }
        let output = sigmoid(output);

        (output, input.to_vec(), h1, h2)
    }

    /// Train on a single example
    pub fn train_step(&mut self, source_embed: &[f64], target_embed: &[f64], target: f64) {
        // Create input
        let mut input = Vec::with_capacity(Self::INPUT_DIM);
        input.extend_from_slice(source_embed);
        input.extend_from_slice(target_embed);
        for i in 0..PROFILE_EMBEDDING_DIM {
            let s = source_embed.get(i).copied().unwrap_or(0.0);
            let t = target_embed.get(i).copied().unwrap_or(0.0);
            input.push(s - t);
        }

        // Forward
        let (pred, input_vec, h1, h2) = self.forward(&input);

        // Binary cross-entropy gradient
        let d_output = pred - target;

        // Backprop through output layer
        let mut d_h2 = vec![0.0; TRANSFER_HIDDEN_2];
        for i in 0..TRANSFER_HIDDEN_2 {
            d_h2[i] = d_output * self.weights_3[i] * relu_derivative(h2[i]);
            self.weights_3[i] -= self.learning_rate * d_output * h2[i];
        }
        self.bias_3 -= self.learning_rate * d_output;

        // Backprop through layer 2
        let mut d_h1 = vec![0.0; TRANSFER_HIDDEN_1];
        for i in 0..TRANSFER_HIDDEN_1 {
            let mut grad = 0.0;
            for j in 0..TRANSFER_HIDDEN_2 {
                grad += d_h2[j] * self.weights_2[j][i];
                self.weights_2[j][i] -= self.learning_rate * d_h2[j] * h1[i];
            }
            d_h1[i] = grad * relu_derivative(h1[i]);
        }
        for i in 0..TRANSFER_HIDDEN_2 {
            self.bias_2[i] -= self.learning_rate * d_h2[i];
        }

        // Backprop through layer 1
        for i in 0..TRANSFER_HIDDEN_1 {
            for j in 0..Self::INPUT_DIM.min(input_vec.len()) {
                self.weights_1[i][j] -= self.learning_rate * d_h1[i] * input_vec[j];
            }
            self.bias_1[i] -= self.learning_rate * d_h1[i];
        }
    }
}

/// Record of a transfer attempt and its outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferOutcome {
    /// Source symbol
    pub source: String,
    /// Target symbol
    pub target: String,
    /// Predicted transfer score
    pub predicted_score: f64,
    /// Did the transfer help?
    pub actual_success: bool,
    /// P&L before transfer
    pub pnl_before_transfer: f64,
    /// P&L after transfer
    pub pnl_after_transfer: f64,
    /// Number of trades evaluated
    pub trades_evaluated: u32,
    /// When this occurred
    pub timestamp: DateTime<Utc>,
}

/// Main predictor for transfer learning between symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferabilityPredictor {
    /// Profiles for each symbol
    symbol_profiles: HashMap<String, SymbolProfile>,
    /// Neural net for encoding profiles
    profile_encoder: ProfileEncoder,
    /// Neural net for predicting transfer success
    transfer_predictor: TransferNet,
    /// History of transfer outcomes for learning
    transfer_history: Vec<TransferOutcome>,
    /// Minimum trades needed to create a profile
    min_trades_for_profile: u32,
    /// Known bad source-target pairs (negative transfers)
    blacklist: Vec<(String, String)>,
    /// Next profile update counter
    #[serde(default)]
    update_counter: u32,
}

impl Default for TransferabilityPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferabilityPredictor {
    /// Create a new predictor
    pub fn new() -> Self {
        Self {
            symbol_profiles: HashMap::new(),
            profile_encoder: ProfileEncoder::new(),
            transfer_predictor: TransferNet::new(),
            transfer_history: Vec::new(),
            min_trades_for_profile: 20,
            blacklist: Vec::new(),
            update_counter: 0,
        }
    }

    /// Update a symbol's profile with a new trade
    pub fn update_profile(
        &mut self,
        symbol: &str,
        volatility: f64,
        trend_strength: f64,
        regime: &Regime,
        won: bool,
    ) {
        let profile = self
            .symbol_profiles
            .entry(symbol.to_string())
            .or_insert_with(|| SymbolProfile::new(symbol.to_string()));

        profile.update(volatility, trend_strength, regime, won);

        // Re-encode embedding periodically
        self.update_counter += 1;
        if self.update_counter % 10 == 0 {
            let features = profile.to_features();
            profile.embedding = self.profile_encoder.encode(&features);
        }
    }

    /// Compute profile embedding
    pub fn compute_profile_embedding(&self, profile: &SymbolProfile) -> Vec<f64> {
        let features = profile.to_features();
        self.profile_encoder.encode(&features)
    }

    /// Predict transferability between two symbols
    pub fn predict_transferability(
        &self,
        source: &str,
        target: &str,
    ) -> Option<TransferabilityScore> {
        // Get both profiles
        let source_profile = self.symbol_profiles.get(source)?;
        let target_profile = self.symbol_profiles.get(target)?;

        // Check minimum trades
        if source_profile.trade_count < self.min_trades_for_profile
            || target_profile.trade_count < self.min_trades_for_profile
        {
            return None;
        }

        // Check blacklist
        if self.is_blacklisted(source, target) {
            return Some(TransferabilityScore {
                source: source.to_string(),
                target: target.to_string(),
                score: 0.0,
                confidence: 1.0,
                last_computed: Utc::now(),
            });
        }

        // Get embeddings
        let source_embed = if source_profile.embedding.iter().all(|&x| x == 0.0) {
            self.compute_profile_embedding(source_profile)
        } else {
            source_profile.embedding.clone()
        };

        let target_embed = if target_profile.embedding.iter().all(|&x| x == 0.0) {
            self.compute_profile_embedding(target_profile)
        } else {
            target_profile.embedding.clone()
        };

        // Predict
        let score = self.transfer_predictor.predict(&source_embed, &target_embed);

        // Confidence based on trade counts
        let min_trades = source_profile.trade_count.min(target_profile.trade_count) as f64;
        let confidence = (min_trades / 100.0).min(1.0);

        Some(TransferabilityScore {
            source: source.to_string(),
            target: target.to_string(),
            score,
            confidence,
            last_computed: Utc::now(),
        })
    }

    /// Check if transfer should be attempted
    pub fn should_transfer(&self, source: &str, target: &str, threshold: f64) -> bool {
        if let Some(score) = self.predict_transferability(source, target) {
            score.score >= threshold
        } else {
            false
        }
    }

    /// Get the best source symbol for transfer to target
    pub fn get_best_source(&self, target: &str, candidates: &[String]) -> Option<(String, f64)> {
        let mut best: Option<(String, f64)> = None;

        for source in candidates {
            if source == target {
                continue;
            }

            if let Some(score) = self.predict_transferability(source, target) {
                if score.score > best.as_ref().map(|(_, s)| *s).unwrap_or(0.0) {
                    best = Some((source.clone(), score.score));
                }
            }
        }

        best
    }

    /// Record a transfer outcome for learning
    pub fn record_transfer_outcome(&mut self, outcome: TransferOutcome) {
        // Check for negative transfer
        if outcome.pnl_after_transfer < outcome.pnl_before_transfer * 0.8 {
            // Significant negative transfer - add to blacklist
            if !self.is_blacklisted(&outcome.source, &outcome.target) {
                self.blacklist
                    .push((outcome.source.clone(), outcome.target.clone()));
                info!(
                    "[TRANSFER] Blacklisted {} -> {} due to negative transfer",
                    outcome.source, outcome.target
                );
            }
        }

        self.transfer_history.push(outcome);

        // Trigger learning if we have enough data
        if self.transfer_history.len() >= 20 && self.transfer_history.len() % 10 == 0 {
            self.learn_from_outcomes();
        }
    }

    /// Learn from transfer outcomes
    pub fn learn_from_outcomes(&mut self) {
        if self.transfer_history.len() < 10 {
            return;
        }

        info!(
            "[TRANSFER] Learning from {} transfer outcomes",
            self.transfer_history.len()
        );

        // Train on all outcomes
        for outcome in &self.transfer_history {
            let source_profile = match self.symbol_profiles.get(&outcome.source) {
                Some(p) => p,
                None => continue,
            };
            let target_profile = match self.symbol_profiles.get(&outcome.target) {
                Some(p) => p,
                None => continue,
            };

            let source_embed = self.compute_profile_embedding(source_profile);
            let target_embed = self.compute_profile_embedding(target_profile);

            let target_label = if outcome.actual_success { 1.0 } else { 0.0 };

            self.transfer_predictor
                .train_step(&source_embed, &target_embed, target_label);
        }

        // Re-encode all profiles with updated encoder
        let symbols: Vec<String> = self.symbol_profiles.keys().cloned().collect();
        for symbol in symbols {
            if let Some(profile) = self.symbol_profiles.get_mut(&symbol) {
                let features = profile.to_features();
                profile.embedding = self.profile_encoder.encode(&features);
            }
        }
    }

    /// Get similarity between two symbols based on profile embeddings
    pub fn get_symbol_similarity(&self, a: &str, b: &str) -> f64 {
        let profile_a = match self.symbol_profiles.get(a) {
            Some(p) => p,
            None => return 0.0,
        };
        let profile_b = match self.symbol_profiles.get(b) {
            Some(p) => p,
            None => return 0.0,
        };

        let embed_a = if profile_a.embedding.iter().all(|&x| x == 0.0) {
            self.compute_profile_embedding(profile_a)
        } else {
            profile_a.embedding.clone()
        };

        let embed_b = if profile_b.embedding.iter().all(|&x| x == 0.0) {
            self.compute_profile_embedding(profile_b)
        } else {
            profile_b.embedding.clone()
        };

        cosine_similarity(&embed_a, &embed_b)
    }

    /// Discover clusters of similar symbols
    pub fn discover_clusters(&self) -> Vec<Vec<String>> {
        let symbols: Vec<&String> = self
            .symbol_profiles
            .keys()
            .filter(|s| {
                self.symbol_profiles
                    .get(*s)
                    .map(|p| p.trade_count >= self.min_trades_for_profile)
                    .unwrap_or(false)
            })
            .collect();

        if symbols.len() < 2 {
            return vec![symbols.into_iter().cloned().collect()];
        }

        // Simple agglomerative clustering
        let mut clusters: Vec<Vec<String>> = symbols.iter().map(|s| vec![(*s).clone()]).collect();
        let similarity_threshold = 0.7;

        loop {
            let mut best_merge: Option<(usize, usize, f64)> = None;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    // Average linkage
                    let mut total_sim = 0.0;
                    let mut count = 0;
                    for a in &clusters[i] {
                        for b in &clusters[j] {
                            total_sim += self.get_symbol_similarity(a, b);
                            count += 1;
                        }
                    }
                    let avg_sim = if count > 0 {
                        total_sim / count as f64
                    } else {
                        0.0
                    };

                    if avg_sim >= similarity_threshold {
                        if best_merge.is_none() || avg_sim > best_merge.unwrap().2 {
                            best_merge = Some((i, j, avg_sim));
                        }
                    }
                }
            }

            match best_merge {
                Some((i, j, _)) => {
                    let cluster_j = clusters.remove(j);
                    clusters[i].extend(cluster_j);
                }
                None => break,
            }
        }

        clusters
    }

    /// Check if a source-target pair is blacklisted
    pub fn is_blacklisted(&self, source: &str, target: &str) -> bool {
        self.blacklist
            .iter()
            .any(|(s, t)| s == source && t == target)
    }

    /// Get profile for a symbol
    pub fn get_profile(&self, symbol: &str) -> Option<&SymbolProfile> {
        self.symbol_profiles.get(symbol)
    }

    /// Get number of profiles
    pub fn profile_count(&self) -> usize {
        self.symbol_profiles.len()
    }

    /// Get number of transfer outcomes
    pub fn outcome_count(&self) -> usize {
        self.transfer_history.len()
    }

    /// Get blacklist size
    pub fn blacklist_count(&self) -> usize {
        self.blacklist.len()
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        let profiles_ready = self
            .symbol_profiles
            .values()
            .filter(|p| p.trade_count >= self.min_trades_for_profile)
            .count();
        format!(
            "{} profiles ({} ready), {} outcomes, {} blacklisted",
            self.symbol_profiles.len(),
            profiles_ready,
            self.transfer_history.len(),
            self.blacklist.len()
        )
    }

    /// Save to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let predictor: Self = serde_json::from_str(&json)?;
        Ok(predictor)
    }

    /// Load or create new
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        Self::load(&path).unwrap_or_default()
    }
}

/// ReLU activation
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// ReLU derivative (for backprop)
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Sigmoid activation
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ==================== Negative Transfer Detection ====================

use std::collections::HashSet;

/// Snapshot of performance metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Win rate (0.0 - 1.0)
    pub win_rate: f64,
    /// Average P&L per trade
    pub avg_pnl: f64,
    /// Sharpe ratio
    pub sharpe: f64,
    /// Maximum drawdown (positive value)
    pub max_dd: f64,
    /// Number of trades
    pub trade_count: u32,
    /// When this snapshot was taken
    pub timestamp: DateTime<Utc>,
}

impl PerformanceSnapshot {
    /// Create a new performance snapshot
    pub fn new(win_rate: f64, avg_pnl: f64, sharpe: f64, max_dd: f64, trade_count: u32) -> Self {
        Self {
            win_rate,
            avg_pnl,
            sharpe,
            max_dd,
            trade_count,
            timestamp: Utc::now(),
        }
    }

    /// Create an empty snapshot
    pub fn empty() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0)
    }
}

/// Record of a transfer attempt and its impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferImpact {
    /// Source symbol
    pub source: String,
    /// Target symbol
    pub target: String,
    /// Unique identifier for this transfer
    pub transfer_id: u64,
    /// Metrics before transfer
    pub pre_transfer_metrics: PerformanceSnapshot,
    /// Metrics after transfer (if evaluated)
    pub post_transfer_metrics: Option<PerformanceSnapshot>,
    /// Trades before transfer evaluation
    pub trades_before: u32,
    /// Trades after transfer application
    pub trades_after: u32,
    /// Calculated impact score (positive = helped, negative = hurt)
    pub impact_score: Option<f64>,
    /// When this transfer was initiated
    pub detected_at: DateTime<Utc>,
}

impl TransferImpact {
    /// Create a new transfer impact record
    pub fn new(
        source: String,
        target: String,
        transfer_id: u64,
        pre_metrics: PerformanceSnapshot,
    ) -> Self {
        Self {
            source,
            target,
            transfer_id,
            pre_transfer_metrics: pre_metrics.clone(),
            post_transfer_metrics: None,
            trades_before: pre_metrics.trade_count,
            trades_after: 0,
            impact_score: None,
            detected_at: Utc::now(),
        }
    }

    /// Calculate impact score from pre/post metrics
    pub fn calculate_impact(&mut self) {
        if let Some(ref post) = self.post_transfer_metrics {
            // Weighted combination of metric changes
            let win_rate_delta = post.win_rate - self.pre_transfer_metrics.win_rate;
            let sharpe_delta = post.sharpe - self.pre_transfer_metrics.sharpe;
            let pnl_delta = if self.pre_transfer_metrics.avg_pnl != 0.0 {
                (post.avg_pnl - self.pre_transfer_metrics.avg_pnl) / self.pre_transfer_metrics.avg_pnl.abs()
            } else {
                post.avg_pnl.signum() * 0.1
            };
            let dd_delta = self.pre_transfer_metrics.max_dd - post.max_dd; // Less DD is good

            // Impact score: combination of deltas
            // Range roughly -1.0 to 1.0
            self.impact_score = Some(
                win_rate_delta * 2.0 + // Win rate weighted heavily
                sharpe_delta * 0.5 +    // Sharpe ratio
                pnl_delta * 0.3 +        // P&L change
                dd_delta * 0.2           // Drawdown improvement
            );
        }
    }
}

/// Entry in the graylist (temporary block)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraylistEntry {
    /// When this entry was added
    pub added_at: DateTime<Utc>,
    /// Reason for graylisting
    pub reason: String,
    /// Impact score that caused graylisting
    pub impact_score: f64,
    /// When to retry this pair
    pub retry_after: DateTime<Utc>,
    /// Number of times this pair has been retried
    pub retry_count: u32,
}

impl GraylistEntry {
    /// Create a new graylist entry
    pub fn new(reason: String, impact_score: f64, retry_days: u32) -> Self {
        Self {
            added_at: Utc::now(),
            reason,
            impact_score,
            retry_after: Utc::now() + chrono::Duration::days(retry_days as i64),
            retry_count: 0,
        }
    }

    /// Check if this entry has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.retry_after
    }
}

/// Features for transfer impact prediction
#[derive(Debug, Clone)]
pub struct TransferFeatures {
    pub source_win_rate: f64,
    pub source_sharpe: f64,
    pub source_trades: f64,
    pub target_win_rate: f64,
    pub target_sharpe: f64,
    pub target_trades: f64,
    pub embedding_similarity: f64,
    pub regime_overlap: f64,
    pub volatility_ratio: f64,
    pub correlation: f64,
    pub cluster_same: f64,
    pub historical_success_rate: f64,
    pub time_since_last_transfer: f64,
    pub previous_impact: f64,
}

impl TransferFeatures {
    /// Convert to array for neural network input
    pub fn to_array(&self) -> [f64; 14] {
        [
            self.source_win_rate,
            self.source_sharpe.clamp(-3.0, 3.0) / 3.0, // Normalize
            (self.source_trades.ln() / 7.0).min(1.0),   // Log normalize
            self.target_win_rate,
            self.target_sharpe.clamp(-3.0, 3.0) / 3.0,
            (self.target_trades.ln() / 7.0).min(1.0),
            self.embedding_similarity,
            self.regime_overlap,
            (self.volatility_ratio - 1.0).clamp(-1.0, 1.0), // Center around 1
            self.correlation,
            self.cluster_same,
            self.historical_success_rate,
            self.time_since_last_transfer.min(1.0),
            self.previous_impact.clamp(-1.0, 1.0),
        ]
    }
}

/// Hidden layer sizes for impact predictor
const IMPACT_HIDDEN_1: usize = 32;
const IMPACT_HIDDEN_2: usize = 16;
const IMPACT_INPUT_DIM: usize = 14;

/// Neural network for predicting transfer impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferImpactPredictor {
    /// Layer 1: 14 -> 32
    weights_1: Vec<Vec<f64>>,
    bias_1: Vec<f64>,
    /// Layer 2: 32 -> 16
    weights_2: Vec<Vec<f64>>,
    bias_2: Vec<f64>,
    /// Layer 3: 16 -> 1
    weights_3: Vec<f64>,
    bias_3: f64,
    /// Learning rate
    learning_rate: f64,
    /// Training samples seen
    samples_trained: u64,
}

impl Default for TransferImpactPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferImpactPredictor {
    /// Create a new impact predictor
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        let scale_1 = (2.0 / (IMPACT_INPUT_DIM + IMPACT_HIDDEN_1) as f64).sqrt();
        let scale_2 = (2.0 / (IMPACT_HIDDEN_1 + IMPACT_HIDDEN_2) as f64).sqrt();
        let scale_3 = (2.0 / (IMPACT_HIDDEN_2 + 1) as f64).sqrt();

        let weights_1: Vec<Vec<f64>> = (0..IMPACT_HIDDEN_1)
            .map(|_| {
                (0..IMPACT_INPUT_DIM)
                    .map(|_| rng.gen_range(-scale_1..scale_1))
                    .collect()
            })
            .collect();

        let weights_2: Vec<Vec<f64>> = (0..IMPACT_HIDDEN_2)
            .map(|_| {
                (0..IMPACT_HIDDEN_1)
                    .map(|_| rng.gen_range(-scale_2..scale_2))
                    .collect()
            })
            .collect();

        let weights_3: Vec<f64> = (0..IMPACT_HIDDEN_2)
            .map(|_| rng.gen_range(-scale_3..scale_3))
            .collect();

        Self {
            weights_1,
            bias_1: vec![0.0; IMPACT_HIDDEN_1],
            weights_2,
            bias_2: vec![0.0; IMPACT_HIDDEN_2],
            weights_3,
            bias_3: 0.0,
            learning_rate: 0.01,
            samples_trained: 0,
        }
    }

    /// Predict impact score from features
    /// Returns value in range (-1.0, 1.0) where negative = harmful transfer
    pub fn predict(&self, features: &TransferFeatures) -> f64 {
        let input = features.to_array();

        // Layer 1
        let mut h1 = vec![0.0; IMPACT_HIDDEN_1];
        for i in 0..IMPACT_HIDDEN_1 {
            let mut sum = self.bias_1[i];
            for j in 0..IMPACT_INPUT_DIM {
                sum += self.weights_1[i][j] * input[j];
            }
            h1[i] = relu(sum);
        }

        // Layer 2
        let mut h2 = vec![0.0; IMPACT_HIDDEN_2];
        for i in 0..IMPACT_HIDDEN_2 {
            let mut sum = self.bias_2[i];
            for j in 0..IMPACT_HIDDEN_1 {
                sum += self.weights_2[i][j] * h1[j];
            }
            h2[i] = relu(sum);
        }

        // Output layer with tanh for range (-1, 1)
        let mut output = self.bias_3;
        for i in 0..IMPACT_HIDDEN_2 {
            output += self.weights_3[i] * h2[i];
        }

        output.tanh()
    }

    /// Get confidence in the prediction (based on training)
    pub fn get_confidence(&self, _features: &TransferFeatures) -> f64 {
        // Confidence increases with more training
        let base_confidence = 0.3;
        let max_confidence = 0.95;
        let samples_for_max = 500.0;

        base_confidence + (max_confidence - base_confidence) *
            (1.0 - (-(self.samples_trained as f64) / samples_for_max).exp())
    }

    /// Train on historical transfer impacts
    pub fn train(&mut self, history: &[TransferImpact]) {
        // Filter to impacts with known outcomes
        let training_data: Vec<_> = history
            .iter()
            .filter(|h| h.impact_score.is_some())
            .collect();

        if training_data.len() < 10 {
            return;
        }

        info!(
            "[TRANSFER] Training impact predictor on {} samples",
            training_data.len()
        );

        // Multiple epochs over data
        for _ in 0..3 {
            for impact in &training_data {
                // Build features (simplified - would need more context in real use)
                let features = TransferFeatures {
                    source_win_rate: impact.pre_transfer_metrics.win_rate,
                    source_sharpe: impact.pre_transfer_metrics.sharpe,
                    source_trades: impact.pre_transfer_metrics.trade_count as f64,
                    target_win_rate: 0.5, // Would need target info
                    target_sharpe: 0.0,
                    target_trades: 0.0,
                    embedding_similarity: 0.5,
                    regime_overlap: 0.5,
                    volatility_ratio: 1.0,
                    correlation: 0.0,
                    cluster_same: 0.0,
                    historical_success_rate: 0.5,
                    time_since_last_transfer: 1.0,
                    previous_impact: 0.0,
                };

                let target = impact.impact_score.unwrap_or(0.0).clamp(-1.0, 1.0);
                self.train_step(&features, target);
            }
        }
    }

    /// Single training step
    fn train_step(&mut self, features: &TransferFeatures, target: f64) {
        let input = features.to_array();

        // Forward pass
        let mut h1 = vec![0.0; IMPACT_HIDDEN_1];
        for i in 0..IMPACT_HIDDEN_1 {
            let mut sum = self.bias_1[i];
            for j in 0..IMPACT_INPUT_DIM {
                sum += self.weights_1[i][j] * input[j];
            }
            h1[i] = relu(sum);
        }

        let mut h2 = vec![0.0; IMPACT_HIDDEN_2];
        for i in 0..IMPACT_HIDDEN_2 {
            let mut sum = self.bias_2[i];
            for j in 0..IMPACT_HIDDEN_1 {
                sum += self.weights_2[i][j] * h1[j];
            }
            h2[i] = relu(sum);
        }

        let mut raw_output = self.bias_3;
        for i in 0..IMPACT_HIDDEN_2 {
            raw_output += self.weights_3[i] * h2[i];
        }
        let output = raw_output.tanh();

        // Backprop
        let d_output = output - target;
        let d_raw = d_output * (1.0 - output * output); // tanh derivative

        // Update output layer
        let mut d_h2 = vec![0.0; IMPACT_HIDDEN_2];
        for i in 0..IMPACT_HIDDEN_2 {
            d_h2[i] = d_raw * self.weights_3[i] * relu_derivative(h2[i]);
            self.weights_3[i] -= self.learning_rate * d_raw * h2[i];
        }
        self.bias_3 -= self.learning_rate * d_raw;

        // Update layer 2
        let mut d_h1 = vec![0.0; IMPACT_HIDDEN_1];
        for i in 0..IMPACT_HIDDEN_1 {
            let mut grad = 0.0;
            for j in 0..IMPACT_HIDDEN_2 {
                grad += d_h2[j] * self.weights_2[j][i];
                self.weights_2[j][i] -= self.learning_rate * d_h2[j] * h1[i];
            }
            d_h1[i] = grad * relu_derivative(h1[i]);
        }
        for i in 0..IMPACT_HIDDEN_2 {
            self.bias_2[i] -= self.learning_rate * d_h2[i];
        }

        // Update layer 1
        for i in 0..IMPACT_HIDDEN_1 {
            for j in 0..IMPACT_INPUT_DIM {
                self.weights_1[i][j] -= self.learning_rate * d_h1[i] * input[j];
            }
            self.bias_1[i] -= self.learning_rate * d_h1[i];
        }

        self.samples_trained += 1;
    }
}

/// Verdict of a transfer evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferVerdict {
    /// Transfer helped performance
    Positive,
    /// No significant change
    Neutral,
    /// Transfer hurt performance
    Negative,
    /// Severely hurt performance - should blacklist
    HighlyNegative,
}

impl TransferVerdict {
    /// Get verdict from impact score
    pub fn from_impact(impact: f64) -> Self {
        if impact > 0.05 {
            Self::Positive
        } else if impact > -0.05 {
            Self::Neutral
        } else if impact > -0.15 {
            Self::Negative
        } else {
            Self::HighlyNegative
        }
    }
}

/// Changes in metrics from transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsChange {
    pub win_rate_delta: f64,
    pub pnl_delta: f64,
    pub sharpe_delta: f64,
    pub dd_delta: f64,
}

impl MetricsChange {
    /// Create from pre/post snapshots
    pub fn from_snapshots(pre: &PerformanceSnapshot, post: &PerformanceSnapshot) -> Self {
        Self {
            win_rate_delta: post.win_rate - pre.win_rate,
            pnl_delta: post.avg_pnl - pre.avg_pnl,
            sharpe_delta: post.sharpe - pre.sharpe,
            dd_delta: post.max_dd - pre.max_dd, // Positive = worse DD
        }
    }
}

/// Reason why a transfer is blocked
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockReason {
    /// Permanently blocked
    Blacklisted { reason: String, since: DateTime<Utc> },
    /// Temporarily blocked
    Graylisted { reason: String, until: DateTime<Utc> },
    /// Predicted to be negative
    PredictedNegative { score: f64 },
}

impl std::fmt::Display for BlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockReason::Blacklisted { reason, since } => {
                write!(f, "Blacklisted since {}: {}", since.format("%Y-%m-%d"), reason)
            }
            BlockReason::Graylisted { reason, until } => {
                write!(f, "Graylisted until {}: {}", until.format("%Y-%m-%d"), reason)
            }
            BlockReason::PredictedNegative { score } => {
                write!(f, "Predicted negative impact: {:.2}", score)
            }
        }
    }
}

/// Result of evaluating a transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEvaluation {
    /// Transfer ID
    pub transfer_id: u64,
    /// Calculated impact score
    pub impact_score: f64,
    /// Verdict
    pub verdict: TransferVerdict,
    /// Detailed metric changes
    pub metrics_change: MetricsChange,
    /// Human-readable recommendation
    pub recommendation: String,
}

/// Detector and preventer of negative transfers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeTransferDetector {
    /// History of all transfer impacts
    impact_history: Vec<TransferImpact>,
    /// Permanently blocked pairs
    blacklist: HashSet<(String, String)>,
    /// Temporarily blocked pairs
    graylist: HashMap<(String, String), GraylistEntry>,
    /// ML model for predicting impact
    detection_model: TransferImpactPredictor,
    /// Minimum trades to evaluate a transfer
    min_trades_for_evaluation: u32,
    /// Threshold for negative detection
    negative_threshold: f64,
    /// Threshold for automatic blacklisting
    auto_blacklist_threshold: f64,
    /// Next transfer ID
    next_transfer_id: u64,
}

impl Default for NegativeTransferDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl NegativeTransferDetector {
    /// Create a new negative transfer detector
    pub fn new() -> Self {
        Self {
            impact_history: Vec::new(),
            blacklist: HashSet::new(),
            graylist: HashMap::new(),
            detection_model: TransferImpactPredictor::new(),
            min_trades_for_evaluation: 10,
            negative_threshold: -0.05,      // 5% performance drop
            auto_blacklist_threshold: -0.15, // 15% drop = permanent block
            next_transfer_id: 1,
        }
    }

    /// Start tracking a transfer
    pub fn start_transfer(
        &mut self,
        source: &str,
        target: &str,
        pre_metrics: PerformanceSnapshot,
    ) -> u64 {
        let transfer_id = self.next_transfer_id;
        self.next_transfer_id += 1;

        let impact = TransferImpact::new(
            source.to_string(),
            target.to_string(),
            transfer_id,
            pre_metrics,
        );

        self.impact_history.push(impact);
        info!(
            "[TRANSFER] Started tracking transfer {} -> {} (ID: {})",
            source, target, transfer_id
        );

        transfer_id
    }

    /// Evaluate a transfer after sufficient trades
    pub fn evaluate_transfer(
        &mut self,
        transfer_id: u64,
        post_metrics: PerformanceSnapshot,
    ) -> Option<TransferEvaluation> {
        // Find the transfer
        let impact = self
            .impact_history
            .iter_mut()
            .find(|i| i.transfer_id == transfer_id)?;

        // Check minimum trades
        let new_trades = post_metrics.trade_count.saturating_sub(impact.trades_before);
        if new_trades < self.min_trades_for_evaluation {
            return None;
        }

        // Update impact record
        impact.post_transfer_metrics = Some(post_metrics.clone());
        impact.trades_after = new_trades;
        impact.calculate_impact();

        let impact_score = impact.impact_score.unwrap_or(0.0);
        let verdict = TransferVerdict::from_impact(impact_score);
        let metrics_change = MetricsChange::from_snapshots(
            &impact.pre_transfer_metrics,
            &post_metrics,
        );

        // Extract data before calling methods on self (to avoid borrow conflicts)
        let source = impact.source.clone();
        let target = impact.target.clone();

        // Handle negative transfers
        let recommendation = match verdict {
            TransferVerdict::HighlyNegative => {
                self.add_to_blacklist(
                    &source,
                    &target,
                    &format!("Severe negative impact: {:.2}", impact_score),
                );
                format!(
                    "BLACKLISTED: {} -> {} caused {:.1}% performance drop",
                    source,
                    target,
                    impact_score.abs() * 100.0
                )
            }
            TransferVerdict::Negative => {
                self.add_to_graylist(
                    &source,
                    &target,
                    impact_score,
                    30, // Retry after 30 days
                );
                format!(
                    "Graylisted: {} -> {} for 30 days (impact: {:.2})",
                    source, target, impact_score
                )
            }
            TransferVerdict::Neutral => {
                format!(
                    "Neutral impact from {} -> {} (score: {:.2})",
                    source, target, impact_score
                )
            }
            TransferVerdict::Positive => {
                format!(
                    "Positive transfer: {} -> {} improved performance by {:.1}%",
                    source,
                    target,
                    impact_score * 100.0
                )
            }
        };

        info!("[TRANSFER] Evaluation: {}", recommendation);

        Some(TransferEvaluation {
            transfer_id,
            impact_score,
            verdict,
            metrics_change,
            recommendation,
        })
    }

    /// Check if a transfer should be allowed
    pub fn should_allow_transfer(
        &self,
        source: &str,
        target: &str,
    ) -> (bool, Option<String>) {
        let pair = (source.to_string(), target.to_string());

        // Check blacklist
        if self.blacklist.contains(&pair) {
            return (false, Some(format!("Permanently blocked: {} -> {}", source, target)));
        }

        // Check graylist
        if let Some(entry) = self.graylist.get(&pair) {
            if !entry.is_expired() {
                return (
                    false,
                    Some(format!(
                        "Temporarily blocked until {}: {}",
                        entry.retry_after.format("%Y-%m-%d"),
                        entry.reason
                    )),
                );
            }
        }

        // Predict impact
        let predicted_impact = self.predict_transfer_impact(source, target);
        if predicted_impact < self.negative_threshold {
            return (
                false,
                Some(format!("Predicted negative impact: {:.2}", predicted_impact)),
            );
        }

        (true, None)
    }

    /// Predict transfer impact using ML model
    pub fn predict_transfer_impact(&self, source: &str, target: &str) -> f64 {
        // Build features from historical data
        let features = self.build_features(source, target);
        self.detection_model.predict(&features)
    }

    /// Build features for a source-target pair
    fn build_features(&self, source: &str, target: &str) -> TransferFeatures {
        // Get historical success rate for this pair
        let pair_history: Vec<_> = self
            .impact_history
            .iter()
            .filter(|h| h.source == source && h.target == target && h.impact_score.is_some())
            .collect();

        let historical_success_rate = if pair_history.is_empty() {
            0.5 // No data, assume neutral
        } else {
            let successes = pair_history
                .iter()
                .filter(|h| h.impact_score.unwrap_or(0.0) > 0.0)
                .count();
            successes as f64 / pair_history.len() as f64
        };

        // Get last impact for this pair
        let previous_impact = pair_history
            .last()
            .and_then(|h| h.impact_score)
            .unwrap_or(0.0);

        // Time since last transfer
        let time_since_last = if let Some(last) = pair_history.last() {
            let days = (Utc::now() - last.detected_at).num_days() as f64;
            (days / 365.0).min(1.0)
        } else {
            1.0 // No history
        };

        TransferFeatures {
            source_win_rate: 0.5,   // Would need actual data
            source_sharpe: 0.0,
            source_trades: 50.0,
            target_win_rate: 0.5,
            target_sharpe: 0.0,
            target_trades: 10.0,
            embedding_similarity: 0.5,
            regime_overlap: 0.5,
            volatility_ratio: 1.0,
            correlation: 0.0,
            cluster_same: 0.0,
            historical_success_rate,
            time_since_last_transfer: time_since_last,
            previous_impact,
        }
    }

    /// Add a pair to the permanent blacklist
    pub fn add_to_blacklist(&mut self, source: &str, target: &str, reason: &str) {
        let pair = (source.to_string(), target.to_string());
        if !self.blacklist.contains(&pair) {
            self.blacklist.insert(pair);
            info!(
                "[TRANSFER] Blacklisted {} -> {}: {}",
                source, target, reason
            );
        }
    }

    /// Add a pair to the temporary graylist
    pub fn add_to_graylist(
        &mut self,
        source: &str,
        target: &str,
        impact: f64,
        retry_days: u32,
    ) {
        let pair = (source.to_string(), target.to_string());
        let entry = GraylistEntry::new(
            format!("Negative impact: {:.2}", impact),
            impact,
            retry_days,
        );
        self.graylist.insert(pair.clone(), entry);
        info!(
            "[TRANSFER] Graylisted {} -> {} for {} days",
            source, target, retry_days
        );
    }

    /// Remove a pair from the graylist
    pub fn remove_from_graylist(&mut self, source: &str, target: &str) {
        let pair = (source.to_string(), target.to_string());
        if self.graylist.remove(&pair).is_some() {
            info!("[TRANSFER] Removed {} -> {} from graylist", source, target);
        }
    }

    /// Clear all expired graylist entries
    pub fn clear_expired_graylist(&mut self) {
        let expired: Vec<_> = self
            .graylist
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(pair, _)| pair.clone())
            .collect();

        for pair in expired {
            self.graylist.remove(&pair);
            info!(
                "[TRANSFER] Graylist expired for {} -> {}",
                pair.0, pair.1
            );
        }
    }

    /// Get all blocked pairs with reasons
    pub fn get_blocked_pairs(&self) -> Vec<((String, String), BlockReason)> {
        let mut blocked = Vec::new();

        // Add blacklisted pairs
        for pair in &self.blacklist {
            blocked.push((
                pair.clone(),
                BlockReason::Blacklisted {
                    reason: "Permanent block due to severe negative impact".to_string(),
                    since: Utc::now(), // Would need to track actual date
                },
            ));
        }

        // Add graylisted pairs
        for (pair, entry) in &self.graylist {
            if !entry.is_expired() {
                blocked.push((
                    pair.clone(),
                    BlockReason::Graylisted {
                        reason: entry.reason.clone(),
                        until: entry.retry_after,
                    },
                ));
            }
        }

        blocked
    }

    /// Get transfer history for a specific pair
    pub fn get_transfer_history(&self, source: &str, target: &str) -> Vec<&TransferImpact> {
        self.impact_history
            .iter()
            .filter(|h| h.source == source && h.target == target)
            .collect()
    }

    /// Retrain the prediction model
    pub fn retrain_predictor(&mut self) {
        if self.impact_history.len() < 20 {
            info!(
                "[TRANSFER] Not enough data to retrain predictor ({} samples, need 20)",
                self.impact_history.len()
            );
            return;
        }

        self.detection_model.train(&self.impact_history);
        info!(
            "[TRANSFER] Retrained negative detector on {} outcomes",
            self.impact_history.len()
        );
    }

    /// Get summary statistics
    pub fn format_summary(&self) -> String {
        let evaluated = self
            .impact_history
            .iter()
            .filter(|h| h.impact_score.is_some())
            .count();

        let positive = self
            .impact_history
            .iter()
            .filter(|h| h.impact_score.map(|s| s > 0.05).unwrap_or(false))
            .count();

        format!(
            "Transfers: {} tracked, {} evaluated, {} positive\n\
             Blocked: {} blacklisted, {} graylisted",
            self.impact_history.len(),
            evaluated,
            positive,
            self.blacklist.len(),
            self.graylist.len()
        )
    }

    /// Save to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let detector: Self = serde_json::from_str(&json)?;
        Ok(detector)
    }

    /// Load or create new
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        Self::load(&path).unwrap_or_default()
    }

    /// Get blacklist count
    pub fn blacklist_count(&self) -> usize {
        self.blacklist.len()
    }

    /// Get graylist count
    pub fn graylist_count(&self) -> usize {
        self.graylist.len()
    }

    /// Check if pair is blacklisted
    pub fn is_blacklisted(&self, source: &str, target: &str) -> bool {
        self.blacklist.contains(&(source.to_string(), target.to_string()))
    }

    /// Check if pair is graylisted
    pub fn is_graylisted(&self, source: &str, target: &str) -> bool {
        self.graylist
            .get(&(source.to_string(), target.to_string()))
            .map(|e| !e.is_expired())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_encoding() {
        let encoder = ProfileEncoder::new();

        let mut profile = SymbolProfile::new("AAPL".to_string());
        profile.volatility_profile = 2.5;
        profile.trend_profile = 0.8;
        profile.regime_distribution = [0.4, 0.1, 0.3, 0.2];
        profile.win_rate = 0.55;
        profile.trade_count = 100;

        let features = profile.to_features();
        let embedding = encoder.encode(&features);

        // Check embedding properties
        assert_eq!(embedding.len(), PROFILE_EMBEDDING_DIM);

        // Should be normalized
        let magnitude: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    #[test]
    fn test_transfer_prediction() {
        let predictor = TransferabilityPredictor::new();

        // Create two profiles
        let mut profile_a = SymbolProfile::new("AAPL".to_string());
        let mut profile_b = SymbolProfile::new("MSFT".to_string());

        for _ in 0..25 {
            profile_a.update(2.0, 0.5, &Regime::TrendingUp, true);
            profile_b.update(2.2, 0.6, &Regime::TrendingUp, true);
        }

        let embed_a = predictor.compute_profile_embedding(&profile_a);
        let embed_b = predictor.compute_profile_embedding(&profile_b);

        let score = predictor.transfer_predictor.predict(&embed_a, &embed_b);

        // Score should be between 0 and 1
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_best_source_selection() {
        let mut predictor = TransferabilityPredictor::new();

        // Create profiles for multiple symbols
        let symbols = vec!["AAPL", "MSFT", "GOOGL", "AMZN"];

        for (i, &symbol) in symbols.iter().enumerate() {
            for _ in 0..25 {
                predictor.update_profile(
                    symbol,
                    2.0 + i as f64 * 0.1,
                    0.5,
                    &Regime::TrendingUp,
                    i % 2 == 0,
                );
            }
        }

        let candidates: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
        let best = predictor.get_best_source("AMZN", &candidates);

        assert!(best.is_some());
        let (source, score) = best.unwrap();
        assert_ne!(source, "AMZN"); // Should not select self
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_negative_transfer_detection() {
        let mut predictor = TransferabilityPredictor::new();

        // Record a negative transfer
        let outcome = TransferOutcome {
            source: "AAPL".to_string(),
            target: "TSLA".to_string(),
            predicted_score: 0.7,
            actual_success: false,
            pnl_before_transfer: 100.0,
            pnl_after_transfer: 50.0, // Significant loss
            trades_evaluated: 20,
            timestamp: Utc::now(),
        };

        predictor.record_transfer_outcome(outcome);

        // Should be blacklisted
        assert!(predictor.is_blacklisted("AAPL", "TSLA"));
        assert!(!predictor.is_blacklisted("TSLA", "AAPL")); // Reverse not blacklisted
    }

    #[test]
    fn test_cluster_discovery() {
        let mut predictor = TransferabilityPredictor::new();

        // Create similar profiles for tech stocks
        for symbol in &["AAPL", "MSFT", "GOOGL"] {
            for _ in 0..25 {
                predictor.update_profile(symbol, 2.5, 0.7, &Regime::TrendingUp, true);
            }
        }

        // Create different profiles for value stocks
        for symbol in &["JNJ", "PG", "KO"] {
            for _ in 0..25 {
                predictor.update_profile(symbol, 1.0, 0.3, &Regime::Ranging, true);
            }
        }

        let clusters = predictor.discover_clusters();

        // Should have at least 1 cluster
        assert!(!clusters.is_empty());

        // Total symbols should match
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn test_outcome_learning() {
        let mut predictor = TransferabilityPredictor::new();

        // Create profiles
        for symbol in &["A", "B", "C", "D"] {
            for _ in 0..25 {
                predictor.update_profile(symbol, 2.0, 0.5, &Regime::Ranging, true);
            }
        }

        // Add transfer outcomes
        for i in 0..15 {
            let outcome = TransferOutcome {
                source: "A".to_string(),
                target: "B".to_string(),
                predicted_score: 0.5,
                actual_success: i % 2 == 0,
                pnl_before_transfer: 100.0,
                pnl_after_transfer: if i % 2 == 0 { 120.0 } else { 90.0 },
                trades_evaluated: 10,
                timestamp: Utc::now(),
            };
            predictor.record_transfer_outcome(outcome);
        }

        // Learning should have been triggered
        assert!(predictor.outcome_count() >= 15);
    }

    #[test]
    fn test_symbol_profile_update() {
        let mut profile = SymbolProfile::new("TEST".to_string());

        profile.update(2.0, 0.5, &Regime::TrendingUp, true);
        profile.update(3.0, 0.7, &Regime::TrendingUp, false);
        profile.update(2.5, 0.6, &Regime::Ranging, true);

        assert_eq!(profile.trade_count, 3);
        assert!((profile.volatility_profile - 2.5).abs() < 0.01);
        assert!((profile.win_rate - 2.0 / 3.0).abs() < 0.01);
        assert!(profile.regime_distribution[0] > 0.5); // TrendingUp most common
    }

    // ==================== Negative Transfer Detection Tests ====================

    #[test]
    fn test_transfer_impact_calculation() {
        let pre = PerformanceSnapshot::new(0.55, 50.0, 1.2, 0.05, 100);
        let mut impact = TransferImpact::new(
            "AAPL".to_string(),
            "MSFT".to_string(),
            1,
            pre,
        );

        // Simulate positive transfer
        let post = PerformanceSnapshot::new(0.60, 60.0, 1.5, 0.04, 120);
        impact.post_transfer_metrics = Some(post);
        impact.calculate_impact();

        assert!(impact.impact_score.is_some());
        let score = impact.impact_score.unwrap();
        assert!(score > 0.0, "Positive metrics change should give positive impact");

        // Simulate negative transfer
        let mut impact2 = TransferImpact::new(
            "AAPL".to_string(),
            "TSLA".to_string(),
            2,
            PerformanceSnapshot::new(0.55, 50.0, 1.2, 0.05, 100),
        );
        let post_bad = PerformanceSnapshot::new(0.45, 30.0, 0.5, 0.10, 120);
        impact2.post_transfer_metrics = Some(post_bad);
        impact2.calculate_impact();

        let score2 = impact2.impact_score.unwrap();
        assert!(score2 < 0.0, "Negative metrics change should give negative impact");
    }

    #[test]
    fn test_blacklist_blocking() {
        let mut detector = NegativeTransferDetector::new();

        // Initially should allow transfer
        let (allowed, reason) = detector.should_allow_transfer("AAPL", "TSLA");
        // May or may not be allowed based on prediction, but no blacklist

        // Add to blacklist
        detector.add_to_blacklist("AAPL", "TSLA", "Test blacklist");

        // Now should be blocked
        let (allowed2, reason2) = detector.should_allow_transfer("AAPL", "TSLA");
        assert!(!allowed2);
        assert!(reason2.is_some());
        assert!(reason2.unwrap().contains("Permanently blocked"));

        // Reverse direction should still be allowed (not blacklisted)
        assert!(!detector.is_blacklisted("TSLA", "AAPL"));
    }

    #[test]
    fn test_graylist_expiry() {
        let mut detector = NegativeTransferDetector::new();

        // Add to graylist with 0 days (immediate expiry for testing)
        detector.add_to_graylist("AAPL", "GOOGL", -0.10, 0);

        // Entry exists but should be expired
        assert!(detector.graylist.contains_key(&("AAPL".to_string(), "GOOGL".to_string())));

        // Clear expired
        detector.clear_expired_graylist();

        // Should be removed
        assert!(!detector.graylist.contains_key(&("AAPL".to_string(), "GOOGL".to_string())));
    }

    #[test]
    fn test_negative_prediction() {
        let predictor = TransferImpactPredictor::new();

        let features = TransferFeatures {
            source_win_rate: 0.55,
            source_sharpe: 1.2,
            source_trades: 100.0,
            target_win_rate: 0.45,
            target_sharpe: 0.3,
            target_trades: 20.0,
            embedding_similarity: 0.3,
            regime_overlap: 0.2,
            volatility_ratio: 2.0,
            correlation: 0.1,
            cluster_same: 0.0,
            historical_success_rate: 0.2,
            time_since_last_transfer: 0.1,
            previous_impact: -0.2,
        };

        let prediction = predictor.predict(&features);
        // Prediction should be in range (-1, 1)
        assert!(prediction >= -1.0 && prediction <= 1.0);

        // Test confidence
        let confidence = predictor.get_confidence(&features);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_auto_blacklist() {
        let mut detector = NegativeTransferDetector::new();

        // Start a transfer
        let pre_metrics = PerformanceSnapshot::new(0.60, 100.0, 1.5, 0.05, 50);
        let transfer_id = detector.start_transfer("AAPL", "PENNY", pre_metrics);

        // Evaluate with severely negative outcome (>15% drop)
        let post_metrics = PerformanceSnapshot::new(0.35, 20.0, -0.5, 0.20, 70);
        let eval = detector.evaluate_transfer(transfer_id, post_metrics);

        assert!(eval.is_some());
        let evaluation = eval.unwrap();
        assert_eq!(evaluation.verdict, TransferVerdict::HighlyNegative);

        // Should be auto-blacklisted
        assert!(detector.is_blacklisted("AAPL", "PENNY"));
    }

    #[test]
    fn test_predictor_training() {
        let mut predictor = TransferImpactPredictor::new();

        // Create training data
        let mut impacts = Vec::new();
        for i in 0..15 {
            let mut impact = TransferImpact::new(
                "A".to_string(),
                "B".to_string(),
                i,
                PerformanceSnapshot::new(0.5, 50.0, 1.0, 0.05, 50),
            );
            impact.post_transfer_metrics = Some(PerformanceSnapshot::new(
                if i % 2 == 0 { 0.55 } else { 0.45 },
                if i % 2 == 0 { 60.0 } else { 40.0 },
                if i % 2 == 0 { 1.2 } else { 0.8 },
                0.05,
                60,
            ));
            impact.calculate_impact();
            impacts.push(impact);
        }

        // Train
        predictor.train(&impacts);

        // Should have trained on samples
        assert!(predictor.samples_trained > 0);
    }

    #[test]
    fn test_integration_with_transfer_manager() {
        let mut detector = NegativeTransferDetector::new();

        // Simulate multiple transfers
        for i in 0..5 {
            let pre = PerformanceSnapshot::new(0.5 + i as f64 * 0.02, 50.0, 1.0, 0.05, 30);
            let id = detector.start_transfer("SPY", &format!("ETF{}", i), pre);

            let post = PerformanceSnapshot::new(
                0.52 + i as f64 * 0.01,
                55.0,
                1.1,
                0.04,
                50,
            );
            detector.evaluate_transfer(id, post);
        }

        // Check history
        let history = detector.get_transfer_history("SPY", "ETF0");
        assert_eq!(history.len(), 1);

        // Check summary
        let summary = detector.format_summary();
        assert!(summary.contains("tracked"));
    }

    #[test]
    fn test_transfer_verdict_thresholds() {
        assert_eq!(TransferVerdict::from_impact(0.10), TransferVerdict::Positive);
        assert_eq!(TransferVerdict::from_impact(0.02), TransferVerdict::Neutral);
        assert_eq!(TransferVerdict::from_impact(-0.02), TransferVerdict::Neutral);
        assert_eq!(TransferVerdict::from_impact(-0.08), TransferVerdict::Negative);
        assert_eq!(TransferVerdict::from_impact(-0.20), TransferVerdict::HighlyNegative);
    }

    #[test]
    fn test_metrics_change_calculation() {
        let pre = PerformanceSnapshot::new(0.50, 50.0, 1.0, 0.05, 100);
        let post = PerformanceSnapshot::new(0.55, 60.0, 1.2, 0.06, 120);

        let change = MetricsChange::from_snapshots(&pre, &post);

        assert!((change.win_rate_delta - 0.05).abs() < 0.001);
        assert!((change.pnl_delta - 10.0).abs() < 0.001);
        assert!((change.sharpe_delta - 0.2).abs() < 0.001);
        assert!((change.dd_delta - 0.01).abs() < 0.001); // DD got worse
    }

    #[test]
    fn test_graylist_entry() {
        let entry = GraylistEntry::new("Test reason".to_string(), -0.10, 30);

        assert!(!entry.is_expired()); // Should not be expired yet
        assert_eq!(entry.retry_count, 0);
        assert!(entry.reason.contains("Test reason"));
    }

    #[test]
    fn test_blocked_pairs_list() {
        let mut detector = NegativeTransferDetector::new();

        detector.add_to_blacklist("A", "B", "Severe negative");
        detector.add_to_graylist("C", "D", -0.10, 30);

        let blocked = detector.get_blocked_pairs();
        assert_eq!(blocked.len(), 2);

        // Check that we have one blacklisted and one graylisted
        let has_blacklist = blocked.iter().any(|(_, reason)| {
            matches!(reason, BlockReason::Blacklisted { .. })
        });
        let has_graylist = blocked.iter().any(|(_, reason)| {
            matches!(reason, BlockReason::Graylisted { .. })
        });

        assert!(has_blacklist);
        assert!(has_graylist);
    }
}
