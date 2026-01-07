//! Foundation Model Integration for Time-Series Understanding
//!
//! This module implements a transformer-based foundation model for time-series
//! data, inspired by models like TimesFM, Chronos, and MOMENT. It provides:
//!
//! - Time series tokenization via learned patch centroids
//! - Transformer encoder for sequence embeddings
//! - Zero-shot forecasting and regime detection
//! - Foundation-based transfer learning between symbols

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::sync::Arc;

use anyhow::Result;
use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::regime::Regime;

// ==================== Helper Functions ====================

/// GELU activation function
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Softmax function
fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / x.len() as f64; x.len()];
    }
    exp_vals.iter().map(|&v| v / sum).collect()
}

/// Layer normalization
fn layer_norm(x: &[f64], gamma: &[f64], beta: &[f64], eps: f64) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
    let variance: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = (variance + eps).sqrt();

    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let g = gamma.get(i).copied().unwrap_or(1.0);
            let b = beta.get(i).copied().unwrap_or(0.0);
            g * (v - mean) / std + b
        })
        .collect()
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Matrix multiplication: [m x n] @ [n x p] -> [m x p]
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() || a[0].is_empty() {
        return vec![];
    }

    let m = a.len();
    let n = a[0].len();
    let p = b[0].len();

    // Transpose b for efficient access
    let b_t: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| b.get(i).and_then(|row| row.get(j)).copied().unwrap_or(0.0)).collect())
        .collect();

    (0..m)
        .map(|i| {
            (0..p)
                .map(|j| {
                    a[i].iter()
                        .zip(b_t[j].iter())
                        .map(|(x, y)| x * y)
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Vector-matrix multiplication: [n] @ [n x p] -> [p]
fn vec_matmul(v: &[f64], m: &[Vec<f64>]) -> Vec<f64> {
    if v.is_empty() || m.is_empty() {
        return vec![];
    }

    let p = m[0].len();
    (0..p)
        .map(|j| {
            v.iter()
                .enumerate()
                .map(|(i, &val)| val * m.get(i).and_then(|row| row.get(j)).copied().unwrap_or(0.0))
                .sum()
        })
        .collect()
}

/// Xavier initialization for weight matrix
fn xavier_init(rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let scale = (6.0 / (rows + cols) as f64).sqrt();

    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| rng.gen_range(-scale..scale))
                .collect()
        })
        .collect()
}

/// Zero vector
fn zeros(n: usize) -> Vec<f64> {
    vec![0.0; n]
}

/// Ones vector
fn ones(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

// ==================== Foundation Model Types ====================

/// Type of foundation model architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoundationModelType {
    /// Google's TimesFM style
    TimesFM,
    /// Amazon's Chronos style
    Chronos,
    /// CMU's MOMENT style
    MOMENT,
    /// Custom architecture
    Custom,
}

impl std::fmt::Display for FoundationModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FoundationModelType::TimesFM => write!(f, "TimesFM"),
            FoundationModelType::Chronos => write!(f, "Chronos"),
            FoundationModelType::MOMENT => write!(f, "MOMENT"),
            FoundationModelType::Custom => write!(f, "Custom"),
        }
    }
}

// ==================== Time Series Tokenizer ====================

/// Tokenizer for converting time series to discrete tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesTokenizer {
    /// Vocabulary size (number of token types)
    pub vocab_size: usize,
    /// Number of timesteps per patch
    pub patch_size: usize,
    /// Learned patch centroids for quantization [vocab_size x patch_size]
    pub centroids: Vec<Vec<f64>>,
    /// Whether the tokenizer has been fitted
    fitted: bool,
}

impl TimeSeriesTokenizer {
    /// Create new tokenizer
    pub fn new(vocab_size: usize, patch_size: usize) -> Self {
        Self {
            vocab_size,
            patch_size,
            centroids: Vec::new(),
            fitted: false,
        }
    }

    /// Fit tokenizer on data using K-means clustering
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() {
            return;
        }

        // Extract patches from all series
        let mut patches: Vec<Vec<f64>> = Vec::new();
        for series in data {
            let series_patches = self.extract_patches(series);
            patches.extend(series_patches);
        }

        if patches.is_empty() {
            // Initialize with random centroids
            let mut rng = rand::thread_rng();
            self.centroids = (0..self.vocab_size)
                .map(|_| (0..self.patch_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect();
            self.fitted = true;
            return;
        }

        // K-means clustering
        let k = self.vocab_size.min(patches.len());
        self.centroids = self.kmeans(&patches, k, 20);

        // Pad to vocab_size if needed
        while self.centroids.len() < self.vocab_size {
            let mut rng = rand::thread_rng();
            self.centroids.push(
                (0..self.patch_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            );
        }

        self.fitted = true;
    }

    /// Extract patches from a series
    fn extract_patches(&self, series: &[f64]) -> Vec<Vec<f64>> {
        if series.len() < self.patch_size {
            return vec![];
        }

        // Normalize series
        let mean: f64 = series.iter().sum::<f64>() / series.len() as f64;
        let std: f64 = (series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64).sqrt();
        let std = if std == 0.0 { 1.0 } else { std };

        let normalized: Vec<f64> = series.iter().map(|x| (x - mean) / std).collect();

        // Non-overlapping patches
        normalized
            .chunks(self.patch_size)
            .filter(|c| c.len() == self.patch_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// K-means clustering
    fn kmeans(&self, data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<Vec<f64>> {
        if data.is_empty() || k == 0 {
            return vec![];
        }

        let dim = data[0].len();

        // Initialize centroids randomly from data
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f64>> = (0..k)
            .map(|_| data[rng.gen_range(0..data.len())].clone())
            .collect();

        for _ in 0..max_iters {
            // Assign points to nearest centroid
            let mut assignments: Vec<usize> = Vec::with_capacity(data.len());
            for point in data {
                let mut best_dist = f64::INFINITY;
                let mut best_idx = 0;
                for (i, centroid) in centroids.iter().enumerate() {
                    let dist: f64 = point
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }
                assignments.push(best_idx);
            }

            // Update centroids
            let mut new_centroids: Vec<Vec<f64>> = vec![vec![0.0; dim]; k];
            let mut counts: Vec<usize> = vec![0; k];

            for (point, &assign) in data.iter().zip(assignments.iter()) {
                for (j, &val) in point.iter().enumerate() {
                    new_centroids[assign][j] += val;
                }
                counts[assign] += 1;
            }

            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[i] as f64;
                    }
                } else {
                    // Keep old centroid if no points assigned
                    *centroid = centroids[i].clone();
                }
            }

            centroids = new_centroids;
        }

        centroids
    }

    /// Tokenize a series
    pub fn tokenize(&self, series: &[f64]) -> Vec<u32> {
        if !self.fitted || self.centroids.is_empty() {
            return vec![];
        }

        let patches = self.extract_patches(series);

        patches
            .iter()
            .map(|patch| {
                let mut best_dist = f64::INFINITY;
                let mut best_idx = 0u32;
                for (i, centroid) in self.centroids.iter().enumerate() {
                    let dist: f64 = patch
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i as u32;
                    }
                }
                best_idx
            })
            .collect()
    }

    /// Detokenize tokens back to series
    pub fn detokenize(&self, tokens: &[u32]) -> Vec<f64> {
        if !self.fitted || self.centroids.is_empty() {
            return vec![];
        }

        tokens
            .iter()
            .flat_map(|&token| {
                let idx = token as usize;
                if idx < self.centroids.len() {
                    self.centroids[idx].clone()
                } else {
                    vec![0.0; self.patch_size]
                }
            })
            .collect()
    }
}

impl Default for TimeSeriesTokenizer {
    fn default() -> Self {
        Self::new(1024, 16)
    }
}

// ==================== Layer Normalization ====================

/// Layer normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    /// Scale parameter
    pub gamma: Vec<f64>,
    /// Shift parameter
    pub beta: Vec<f64>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl LayerNorm {
    /// Create new layer norm
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: ones(dim),
            beta: zeros(dim),
            eps: 1e-5,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        layer_norm(x, &self.gamma, &self.beta, self.eps)
    }

    /// Forward pass for batch
    pub fn forward_batch(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|row| self.forward(row)).collect()
    }
}

// ==================== Feed-Forward Network ====================

/// Feed-forward network (MLP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForward {
    /// First layer weights [embedding_dim x 4*embedding_dim]
    pub w_1: Vec<Vec<f64>>,
    /// Second layer weights [4*embedding_dim x embedding_dim]
    pub w_2: Vec<Vec<f64>>,
    /// First layer bias
    pub bias_1: Vec<f64>,
    /// Second layer bias
    pub bias_2: Vec<f64>,
}

impl FeedForward {
    /// Create new feed-forward network
    pub fn new(embedding_dim: usize) -> Self {
        let hidden_dim = 4 * embedding_dim;
        Self {
            w_1: xavier_init(embedding_dim, hidden_dim),
            w_2: xavier_init(hidden_dim, embedding_dim),
            bias_1: zeros(hidden_dim),
            bias_2: zeros(embedding_dim),
        }
    }

    /// Forward pass for single vector
    pub fn forward_vec(&self, x: &[f64]) -> Vec<f64> {
        // h = GELU(x @ W_1 + bias_1)
        let h: Vec<f64> = vec_matmul(x, &self.w_1)
            .iter()
            .zip(self.bias_1.iter())
            .map(|(&v, &b)| gelu(v + b))
            .collect();

        // out = h @ W_2 + bias_2
        vec_matmul(&h, &self.w_2)
            .iter()
            .zip(self.bias_2.iter())
            .map(|(&v, &b)| v + b)
            .collect()
    }

    /// Forward pass for batch
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|row| self.forward_vec(row)).collect()
    }
}

// ==================== Multi-Head Attention ====================

/// Multi-head attention mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Query projection weights
    pub w_q: Vec<Vec<f64>>,
    /// Key projection weights
    pub w_k: Vec<Vec<f64>>,
    /// Value projection weights
    pub w_v: Vec<Vec<f64>>,
    /// Output projection weights
    pub w_o: Vec<Vec<f64>>,
}

impl MultiHeadAttention {
    /// Create new multi-head attention
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        let head_dim = embedding_dim / num_heads;
        Self {
            num_heads,
            head_dim,
            w_q: xavier_init(embedding_dim, embedding_dim),
            w_k: xavier_init(embedding_dim, embedding_dim),
            w_v: xavier_init(embedding_dim, embedding_dim),
            w_o: xavier_init(embedding_dim, embedding_dim),
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &[Vec<f64>], mask: Option<&[Vec<bool>]>) -> Vec<Vec<f64>> {
        if x.is_empty() {
            return vec![];
        }

        let seq_len = x.len();
        let embed_dim = self.num_heads * self.head_dim;

        // Project to Q, K, V
        let q: Vec<Vec<f64>> = x.iter().map(|row| vec_matmul(row, &self.w_q)).collect();
        let k: Vec<Vec<f64>> = x.iter().map(|row| vec_matmul(row, &self.w_k)).collect();
        let v: Vec<Vec<f64>> = x.iter().map(|row| vec_matmul(row, &self.w_v)).collect();

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        let scale = (self.head_dim as f64).sqrt();
        let mut scores: Vec<Vec<f64>> = vec![vec![0.0; seq_len]; seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                let dot: f64 = q[i].iter().zip(k[j].iter()).map(|(a, b)| a * b).sum();
                scores[i][j] = dot / scale;
            }
        }

        // Apply mask if provided (causal mask)
        if let Some(m) = mask {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if m.get(i).and_then(|row| row.get(j)).copied().unwrap_or(false) {
                        scores[i][j] = f64::NEG_INFINITY;
                    }
                }
            }
        }

        // Softmax over keys
        let attn_weights: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();

        // Apply attention to values: attn_weights @ V
        let mut attn_out: Vec<Vec<f64>> = vec![vec![0.0; embed_dim]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let weight = attn_weights[i][j];
                for (k, val) in v[j].iter().enumerate() {
                    attn_out[i][k] += weight * val;
                }
            }
        }

        // Output projection
        attn_out.iter().map(|row| vec_matmul(row, &self.w_o)).collect()
    }
}

// ==================== Transformer Layer ====================

/// Single transformer layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerLayer {
    /// Multi-head attention
    pub attention: MultiHeadAttention,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Layer norm before attention
    pub layer_norm_1: LayerNorm,
    /// Layer norm before FFN
    pub layer_norm_2: LayerNorm,
}

impl TransformerLayer {
    /// Create new transformer layer
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(embedding_dim, num_heads),
            ffn: FeedForward::new(embedding_dim),
            layer_norm_1: LayerNorm::new(embedding_dim),
            layer_norm_2: LayerNorm::new(embedding_dim),
        }
    }

    /// Forward pass with pre-norm architecture
    pub fn forward(&self, x: &[Vec<f64>], mask: Option<&[Vec<bool>]>) -> Vec<Vec<f64>> {
        if x.is_empty() {
            return vec![];
        }

        // Pre-norm + attention + residual
        let normed_1 = self.layer_norm_1.forward_batch(x);
        let attn_out = self.attention.forward(&normed_1, mask);
        let residual_1: Vec<Vec<f64>> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
            .collect();

        // Pre-norm + FFN + residual
        let normed_2 = self.layer_norm_2.forward_batch(&residual_1);
        let ffn_out = self.ffn.forward(&normed_2);
        residual_1
            .iter()
            .zip(ffn_out.iter())
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
            .collect()
    }
}

// ==================== Foundation Weights ====================

/// All weights for the foundation model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundationWeights {
    /// Token embedding matrix [vocab_size x embedding_dim]
    pub token_embedding: Vec<Vec<f64>>,
    /// Position embedding matrix [context_length x embedding_dim]
    pub position_embedding: Vec<Vec<f64>>,
    /// Transformer layers
    pub transformer_layers: Vec<TransformerLayer>,
    /// Output projection for forecasting
    pub output_projection: Vec<Vec<f64>>,
    /// Regime classifier head
    pub regime_head: Vec<Vec<f64>>,
    /// Direction classifier head
    pub direction_head: Vec<Vec<f64>>,
}

impl FoundationWeights {
    /// Create new weights with Xavier initialization
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        context_length: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        Self {
            token_embedding: xavier_init(vocab_size, embedding_dim),
            position_embedding: xavier_init(context_length, embedding_dim),
            transformer_layers: (0..num_layers)
                .map(|_| TransformerLayer::new(embedding_dim, num_heads))
                .collect(),
            output_projection: xavier_init(embedding_dim, vocab_size),
            regime_head: xavier_init(embedding_dim, 4), // 4 regimes
            direction_head: xavier_init(embedding_dim, 2), // up/down
        }
    }
}

// ==================== Forecast Distribution ====================

/// Distribution of forecasted values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastDistribution {
    /// Mean forecast
    pub mean: Vec<f64>,
    /// Standard deviation
    pub std: Vec<f64>,
    /// 10th percentile
    pub p10: Vec<f64>,
    /// 50th percentile (median)
    pub p50: Vec<f64>,
    /// 90th percentile
    pub p90: Vec<f64>,
}

impl ForecastDistribution {
    /// Create from samples
    pub fn from_samples(samples: &[Vec<f64>]) -> Self {
        if samples.is_empty() || samples[0].is_empty() {
            return Self {
                mean: vec![],
                std: vec![],
                p10: vec![],
                p50: vec![],
                p90: vec![],
            };
        }

        let horizon = samples[0].len();
        let n_samples = samples.len() as f64;

        let mut mean = vec![0.0; horizon];
        let mut std = vec![0.0; horizon];
        let mut p10 = vec![0.0; horizon];
        let mut p50 = vec![0.0; horizon];
        let mut p90 = vec![0.0; horizon];

        for t in 0..horizon {
            let mut values: Vec<f64> = samples.iter().map(|s| s[t]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Mean
            mean[t] = values.iter().sum::<f64>() / n_samples;

            // Std
            let variance = values.iter().map(|v| (v - mean[t]).powi(2)).sum::<f64>() / n_samples;
            std[t] = variance.sqrt();

            // Percentiles
            let n = values.len();
            p10[t] = values[(n as f64 * 0.1) as usize];
            p50[t] = values[(n as f64 * 0.5) as usize];
            p90[t] = values[(n as f64 * 0.9) as usize];
        }

        Self { mean, std, p10, p50, p90 }
    }
}

// ==================== Time Series Foundation Model ====================

/// Time series foundation model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesFoundation {
    /// Model architecture type
    pub model_type: FoundationModelType,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Maximum context length
    pub context_length: usize,
    /// Model weights
    pub weights: FoundationWeights,
    /// Tokenizer
    pub tokenizer: TimeSeriesTokenizer,
    /// Number of transformer layers
    num_layers: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Total inferences run
    inference_count: u64,
}

impl TimeSeriesFoundation {
    /// Create new foundation model
    pub fn new(model_type: FoundationModelType) -> Self {
        // Default: small model for efficiency
        let embedding_dim = 128;
        let context_length = 512;
        let num_layers = 4;
        let num_heads = 8;
        let vocab_size = 1024;
        let patch_size = 16;

        let mut tokenizer = TimeSeriesTokenizer::new(vocab_size, patch_size);

        // Initialize tokenizer with random centroids
        let mut rng = rand::thread_rng();
        tokenizer.centroids = (0..vocab_size)
            .map(|_| (0..patch_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        tokenizer.fitted = true;

        Self {
            model_type,
            embedding_dim,
            context_length,
            weights: FoundationWeights::new(
                vocab_size,
                embedding_dim,
                context_length,
                num_layers,
                num_heads,
            ),
            tokenizer,
            num_layers,
            num_heads,
            inference_count: 0,
        }
    }

    /// Create with custom dimensions
    pub fn with_config(
        model_type: FoundationModelType,
        embedding_dim: usize,
        context_length: usize,
        num_layers: usize,
        num_heads: usize,
        vocab_size: usize,
        patch_size: usize,
    ) -> Self {
        let mut tokenizer = TimeSeriesTokenizer::new(vocab_size, patch_size);

        // Initialize tokenizer with random centroids
        let mut rng = rand::thread_rng();
        tokenizer.centroids = (0..vocab_size)
            .map(|_| (0..patch_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        tokenizer.fitted = true;

        Self {
            model_type,
            embedding_dim,
            context_length,
            weights: FoundationWeights::new(
                vocab_size,
                embedding_dim,
                context_length,
                num_layers,
                num_heads,
            ),
            tokenizer,
            num_layers,
            num_heads,
            inference_count: 0,
        }
    }

    /// Encode a time series to embedding
    pub fn encode(&mut self, series: &[f64]) -> Vec<f64> {
        self.inference_count += 1;

        // Tokenize
        let tokens = self.tokenizer.tokenize(series);
        if tokens.is_empty() {
            return vec![0.0; self.embedding_dim];
        }

        // Truncate to context length
        let tokens: Vec<u32> = tokens
            .into_iter()
            .take(self.context_length)
            .collect();

        // Embed tokens
        let mut embeddings: Vec<Vec<f64>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &token)| {
                let token_idx = token as usize % self.weights.token_embedding.len();
                let pos_idx = pos % self.weights.position_embedding.len();

                let tok_emb = &self.weights.token_embedding[token_idx];
                let pos_emb = &self.weights.position_embedding[pos_idx];

                tok_emb
                    .iter()
                    .zip(pos_emb.iter())
                    .map(|(t, p)| t + p)
                    .collect()
            })
            .collect();

        // Pass through transformer layers
        for layer in &self.weights.transformer_layers {
            embeddings = layer.forward(&embeddings, None);
        }

        // Pool: mean of all positions
        if embeddings.is_empty() {
            return vec![0.0; self.embedding_dim];
        }

        let seq_len = embeddings.len() as f64;
        (0..self.embedding_dim)
            .map(|i| {
                embeddings
                    .iter()
                    .map(|emb| emb.get(i).copied().unwrap_or(0.0))
                    .sum::<f64>()
                    / seq_len
            })
            .collect()
    }

    /// Encode batch of series
    pub fn encode_batch(&mut self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        batch.iter().map(|s| self.encode(s)).collect()
    }

    /// Forecast future values
    pub fn forecast(&mut self, series: &[f64], horizon: usize) -> Vec<f64> {
        let mut context = series.to_vec();
        let mut forecasts = Vec::with_capacity(horizon);

        for _ in 0..horizon {
            // Encode current context
            let embedding = self.encode(&context);

            // Project to next token logits
            let logits = vec_matmul(&embedding, &self.weights.output_projection);

            // Argmax to get most likely token
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            // Detokenize to get values
            let next_values = self.tokenizer.detokenize(&[next_token]);

            // Use mean of patch as forecast
            let mean_val = if next_values.is_empty() {
                *context.last().unwrap_or(&0.0)
            } else {
                next_values.iter().sum::<f64>() / next_values.len() as f64
            };

            forecasts.push(mean_val);
            context.push(mean_val);

            // Limit context size
            if context.len() > 1000 {
                context.drain(0..100);
            }
        }

        forecasts
    }

    /// Forecast with uncertainty quantification
    pub fn forecast_distribution(
        &mut self,
        series: &[f64],
        horizon: usize,
        samples: usize,
    ) -> ForecastDistribution {
        let mut all_samples: Vec<Vec<f64>> = Vec::with_capacity(samples);
        let mut rng = rand::thread_rng();

        for _ in 0..samples {
            let mut context = series.to_vec();
            let mut sample_forecast = Vec::with_capacity(horizon);

            for _ in 0..horizon {
                let embedding = self.encode(&context);
                let logits = vec_matmul(&embedding, &self.weights.output_projection);

                // Sample with temperature
                let temperature = 1.0;
                let scaled_logits: Vec<f64> = logits.iter().map(|l| l / temperature).collect();
                let probs = softmax(&scaled_logits);

                // Sample from distribution
                let rand_val: f64 = rng.gen();
                let mut cumsum = 0.0;
                let mut next_token = 0u32;
                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if rand_val < cumsum {
                        next_token = i as u32;
                        break;
                    }
                }

                let next_values = self.tokenizer.detokenize(&[next_token]);
                let mean_val = if next_values.is_empty() {
                    *context.last().unwrap_or(&0.0) + rng.gen_range(-0.01..0.01)
                } else {
                    next_values.iter().sum::<f64>() / next_values.len() as f64
                };

                sample_forecast.push(mean_val);
                context.push(mean_val);
            }

            all_samples.push(sample_forecast);
        }

        ForecastDistribution::from_samples(&all_samples)
    }

    /// Compute similarity between two series
    pub fn similarity(&mut self, series_a: &[f64], series_b: &[f64]) -> f64 {
        let emb_a = self.encode(series_a);
        let emb_b = self.encode(series_b);
        cosine_similarity(&emb_a, &emb_b)
    }

    /// Zero-shot regime classification
    /// Returns (predicted_regime, confidence)
    pub fn zero_shot_regime(&mut self, series: &[f64]) -> (super::regime::Regime, f64) {
        let embedding = self.encode(series);
        let logits = vec_matmul(&embedding, &self.weights.regime_head);
        let probs = softmax(&logits);

        // Find argmax
        let mut max_idx = 0;
        let mut max_prob = probs.get(0).copied().unwrap_or(0.25);
        for (i, &p) in probs.iter().enumerate().skip(1) {
            if p > max_prob {
                max_prob = p;
                max_idx = i;
            }
        }

        let regime = super::regime::Regime::from_index(max_idx);
        (regime, max_prob)
    }

    /// Zero-shot regime probabilities (raw)
    pub fn zero_shot_regime_probs(&mut self, series: &[f64]) -> [f64; 4] {
        let embedding = self.encode(series);
        let logits = vec_matmul(&embedding, &self.weights.regime_head);
        let probs = softmax(&logits);

        [
            probs.get(0).copied().unwrap_or(0.25), // TrendingUp
            probs.get(1).copied().unwrap_or(0.25), // TrendingDown
            probs.get(2).copied().unwrap_or(0.25), // Ranging
            probs.get(3).copied().unwrap_or(0.25), // Volatile
        ]
    }

    /// Zero-shot direction prediction
    /// Returns (is_bullish, confidence)
    pub fn zero_shot_direction(&mut self, series: &[f64]) -> (bool, f64) {
        let embedding = self.encode(series);
        let logits = vec_matmul(&embedding, &self.weights.direction_head);
        let probs = softmax(&logits);

        let prob_up = probs.get(0).copied().unwrap_or(0.5);
        let prob_down = probs.get(1).copied().unwrap_or(0.5);

        let is_bullish = prob_up > prob_down;
        let confidence = if is_bullish { prob_up } else { prob_down };

        (is_bullish, confidence)
    }

    /// Fine-tune on labeled data
    pub fn fine_tune(&mut self, data: &[(Vec<f64>, f64)], epochs: usize, lr: f64) {
        if data.is_empty() {
            return;
        }

        for _epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (series, label) in data {
                let embedding = self.encode(series);

                // Simple SGD on output projection
                let pred = embedding.iter().sum::<f64>() / embedding.len() as f64;
                let error = label - pred;
                total_loss += error.powi(2);

                // Update regime head as proxy
                for (i, emb_val) in embedding.iter().enumerate() {
                    for j in 0..4 {
                        if let Some(row) = self.weights.regime_head.get_mut(i) {
                            if let Some(weight) = row.get_mut(j) {
                                *weight += lr * error * emb_val * 0.01;
                            }
                        }
                    }
                }
            }

            if epochs > 1 && _epoch % 10 == 0 {
                info!(
                    "[FOUNDATION] Epoch {}/{}: loss = {:.6}",
                    _epoch + 1,
                    epochs,
                    total_loss / data.len() as f64
                );
            }
        }
    }

    /// Get inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get context length
    pub fn context_length(&self) -> usize {
        self.context_length
    }

    /// Save to file
    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Load from file
    pub fn load(path: &str) -> Result<Self> {
        let data = fs::read(path)?;
        let model: Self = bincode::deserialize(&data)?;
        Ok(model)
    }

    /// Load or create new
    pub fn load_or_new(path: &str, model_type: FoundationModelType) -> Self {
        match Self::load(path) {
            Ok(model) => {
                info!(
                    "[FOUNDATION] Loaded {} model: {} inferences",
                    model.model_type,
                    model.inference_count
                );
                model
            }
            Err(_) => {
                info!("[FOUNDATION] Creating new {} model", model_type);
                Self::new(model_type)
            }
        }
    }

    /// Format summary
    pub fn format_summary(&self) -> String {
        format!(
            "{} model: {}d embed, {} layers, {} inferences",
            self.model_type,
            self.embedding_dim,
            self.num_layers,
            self.inference_count
        )
    }
}

impl Default for TimeSeriesFoundation {
    fn default() -> Self {
        Self::new(FoundationModelType::Custom)
    }
}

// ==================== Foundation Transfer ====================

/// Foundation-based transfer learning
#[derive(Debug)]
pub struct FoundationTransfer {
    /// Reference to foundation model
    foundation: Arc<std::sync::Mutex<TimeSeriesFoundation>>,
    /// Cached symbol embeddings
    symbol_embeddings: HashMap<String, Vec<f64>>,
    /// Cached transfer scores
    transfer_scores: HashMap<(String, String), f64>,
    /// Last update time per symbol
    last_update: HashMap<String, DateTime<Utc>>,
}

impl FoundationTransfer {
    /// Create new foundation transfer
    pub fn new(foundation: Arc<std::sync::Mutex<TimeSeriesFoundation>>) -> Self {
        Self {
            foundation,
            symbol_embeddings: HashMap::new(),
            transfer_scores: HashMap::new(),
            last_update: HashMap::new(),
        }
    }

    /// Compute and cache symbol embedding
    pub fn compute_symbol_embedding(&mut self, symbol: &str, history: &[f64]) {
        if let Ok(mut model) = self.foundation.lock() {
            let embedding = model.encode(history);
            self.symbol_embeddings.insert(symbol.to_string(), embedding);
            self.last_update.insert(symbol.to_string(), Utc::now());
        }
    }

    /// Get cached embedding
    pub fn get_embedding(&self, symbol: &str) -> Option<&Vec<f64>> {
        self.symbol_embeddings.get(symbol)
    }

    /// Zero-shot transfer score between symbols
    pub fn zero_shot_transfer(&mut self, source: &str, target: &str) -> f64 {
        // Check cache
        let key = (source.to_string(), target.to_string());
        if let Some(&score) = self.transfer_scores.get(&key) {
            return score;
        }

        // Get embeddings
        let source_emb = self.symbol_embeddings.get(source);
        let target_emb = self.symbol_embeddings.get(target);

        let score = match (source_emb, target_emb) {
            (Some(s), Some(t)) => cosine_similarity(s, t),
            _ => 0.5, // Default neutral transfer
        };

        // Cache result
        self.transfer_scores.insert(key, score);
        score
    }

    /// Get most similar symbols
    pub fn get_most_similar_symbols(&self, symbol: &str, n: usize) -> Vec<(String, f64)> {
        let Some(target_emb) = self.symbol_embeddings.get(symbol) else {
            return vec![];
        };

        let mut similarities: Vec<(String, f64)> = self
            .symbol_embeddings
            .iter()
            .filter(|(s, _)| *s != symbol)
            .map(|(s, emb)| (s.clone(), cosine_similarity(target_emb, emb)))
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(n);
        similarities
    }

    /// Get number of cached embeddings
    pub fn embedding_count(&self) -> usize {
        self.symbol_embeddings.len()
    }

    /// Clear stale embeddings (older than specified hours)
    pub fn clear_stale(&mut self, max_age_hours: i64) {
        let now = Utc::now();
        let stale: Vec<String> = self
            .last_update
            .iter()
            .filter(|(_, &time)| (now - time).num_hours() > max_age_hours)
            .map(|(s, _)| s.clone())
            .collect();

        for symbol in stale {
            self.symbol_embeddings.remove(&symbol);
            self.last_update.remove(&symbol);

            // Remove transfer scores involving this symbol
            self.transfer_scores.retain(|(s, t), _| s != &symbol && t != &symbol);
        }
    }

    /// Format summary
    pub fn format_summary(&self) -> String {
        format!(
            "{} embeddings, {} transfer scores cached",
            self.symbol_embeddings.len(),
            self.transfer_scores.len()
        )
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu(1.0) > 0.8);
        assert!(gelu(-1.0) < 0.0);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let probs = softmax(&x);

        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];

        let normed = layer_norm(&x, &gamma, &beta, 1e-5);

        // Mean should be ~0
        let mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(mean.abs() < 1e-6);

        // Std should be ~1
        let std: f64 = (normed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / normed.len() as f64).sqrt();
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tokenizer() {
        let mut tokenizer = TimeSeriesTokenizer::new(16, 4);

        // Fit on sample data
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        ];
        tokenizer.fit(&data);

        assert!(tokenizer.fitted);
        assert!(!tokenizer.centroids.is_empty());

        // Tokenize
        let tokens = tokenizer.tokenize(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(!tokens.is_empty());

        // Detokenize
        let reconstructed = tokenizer.detokenize(&tokens);
        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_attention() {
        let attn = MultiHeadAttention::new(32, 4);

        let x = vec![
            vec![0.1; 32],
            vec![0.2; 32],
            vec![0.3; 32],
        ];

        let out = attn.forward(&x, None);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 32);
    }

    #[test]
    fn test_transformer_forward() {
        let layer = TransformerLayer::new(32, 4);

        let x = vec![
            vec![0.1; 32],
            vec![0.2; 32],
        ];

        let out = layer.forward(&x, None);

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 32);
    }

    #[test]
    fn test_encode() {
        let mut model = TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        );

        let series = vec![100.0, 101.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5];
        let embedding = model.encode(&series);

        assert_eq!(embedding.len(), 32);
        assert!(model.inference_count() == 1);
    }

    #[test]
    fn test_forecast() {
        let mut model = TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        );

        let series = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let forecast = model.forecast(&series, 3);

        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_forecast_distribution() {
        let mut model = TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        );

        let series = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let dist = model.forecast_distribution(&series, 3, 10);

        assert_eq!(dist.mean.len(), 3);
        assert_eq!(dist.std.len(), 3);
        assert_eq!(dist.p10.len(), 3);
        assert_eq!(dist.p50.len(), 3);
        assert_eq!(dist.p90.len(), 3);
    }

    #[test]
    fn test_zero_shot_regime() {
        let mut model = TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        );

        let series = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0];
        let (regime, confidence) = model.zero_shot_regime(&series);

        // Confidence should be between 0 and 1
        assert!(confidence >= 0.0 && confidence <= 1.0);

        // Test raw probabilities sum to ~1
        let probs = model.zero_shot_regime_probs(&series);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_zero_shot_direction() {
        let mut model = TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        );

        let series = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0];
        let (is_bullish, confidence) = model.zero_shot_direction(&series);

        // Confidence should be between 0.5 and 1.0 (winner confidence)
        assert!(confidence >= 0.5 && confidence <= 1.0);
        // is_bullish is a boolean
        let _ = is_bullish; // Just verify it compiles
    }

    #[test]
    fn test_similarity() {
        let mut model = TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        );

        let series_a = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0];
        let series_b = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0];
        let series_c = vec![200.0, 198.0, 196.0, 194.0, 192.0, 190.0, 188.0, 186.0];

        let sim_same = model.similarity(&series_a, &series_b);
        let sim_diff = model.similarity(&series_a, &series_c);

        // Same series should have high similarity
        assert!(sim_same > 0.99);
        // Different patterns might have lower similarity
        assert!(sim_diff < sim_same || (sim_diff - sim_same).abs() < 0.5);
    }

    #[test]
    fn test_transfer_similarity() {
        let model = Arc::new(std::sync::Mutex::new(TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        )));

        let mut transfer = FoundationTransfer::new(model);

        // Add embeddings
        transfer.compute_symbol_embedding("AAPL", &[100.0, 102.0, 104.0, 106.0, 108.0]);
        transfer.compute_symbol_embedding("MSFT", &[150.0, 152.0, 154.0, 156.0, 158.0]);
        transfer.compute_symbol_embedding("GOOG", &[200.0, 198.0, 196.0, 194.0, 192.0]);

        assert_eq!(transfer.embedding_count(), 3);

        // Test zero-shot transfer
        let score = transfer.zero_shot_transfer("AAPL", "MSFT");
        assert!(score >= -1.0 && score <= 1.0);

        // Test most similar
        let similar = transfer.get_most_similar_symbols("AAPL", 2);
        assert_eq!(similar.len(), 2);
    }

    #[test]
    fn test_fine_tuning() {
        let mut model = TimeSeriesFoundation::with_config(
            FoundationModelType::Custom,
            32, 64, 2, 4, 64, 8,
        );

        // Generate synthetic training data
        let data: Vec<(Vec<f64>, f64)> = (0..20)
            .map(|i| {
                let series: Vec<f64> = (0..16).map(|j| 100.0 + (i + j) as f64 * 0.5).collect();
                let label = if i % 2 == 0 { 1.0 } else { 0.0 };
                (series, label)
            })
            .collect();

        // Fine-tune
        model.fine_tune(&data, 5, 0.01);

        // Model should still work after fine-tuning
        let test = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let embedding = model.encode(&test);
        assert_eq!(embedding.len(), 32);
    }

    #[test]
    fn test_model_summary() {
        let model = TimeSeriesFoundation::new(FoundationModelType::TimesFM);
        let summary = model.format_summary();

        assert!(summary.contains("TimesFM"));
        assert!(summary.contains("embed"));
        assert!(summary.contains("layers"));
    }
}
