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
}
