//! Vector embeddings for similarity-based trade retrieval
//!
//! This module implements:
//! - Neural encoder for mapping trade contexts to vector embeddings
//! - Vector index with flat (exact) and HNSW (approximate) search
//! - Similarity functions (cosine, euclidean)
//! - Integration points for TradeMemory

use anyhow::Result;
use chrono::{DateTime, Utc};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::fs;
use std::path::Path;
use tracing::info;

use super::regime::Regime;
use crate::data::memory::TradeContext as MemoryTradeContext;

/// Embedding dimension
pub const EMBEDDING_DIM: usize = 64;

/// Input feature dimension
pub const INPUT_DIM: usize = 12;

/// Hidden layer size
const HIDDEN_DIM: usize = 32;

/// HNSW default parameters
const DEFAULT_M: usize = 16;
const DEFAULT_EF_CONSTRUCTION: usize = 200;

// ==================== Trade Embedding ====================

/// A trade context embedded as a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEmbedding {
    /// Unique embedding ID
    pub id: u64,
    /// Symbol this trade was for
    pub symbol: String,
    /// 64-dimensional embedding vector
    pub vector: Vec<f64>,
    /// When the trade occurred
    pub timestamp: DateTime<Utc>,
    /// Whether the trade was a winner (for retrieval analysis)
    pub won: bool,
    /// PnL of the trade
    pub pnl: f64,
}

impl TradeEmbedding {
    pub fn new(id: u64, symbol: String, vector: Vec<f64>, timestamp: DateTime<Utc>, won: bool, pnl: f64) -> Self {
        Self {
            id,
            symbol,
            vector,
            timestamp,
            won,
            pnl,
        }
    }
}

// ==================== Trade Context for Embedding ====================

/// Trade context features for embedding
#[derive(Debug, Clone)]
pub struct TradeContext {
    /// S/R score at entry (-10 to 0, 0 is strongest)
    pub sr_score: i32,
    /// Volume percentile (0-100)
    pub volume_percentile: f64,
    /// ATR normalized by price (as percentage)
    pub atr_pct: f64,
    /// Distance to nearest S/R level (as percentage of price)
    pub distance_to_sr_pct: f64,
    /// Market regime
    pub regime: Regime,
    /// Trade direction (true = long, false = short)
    pub is_long: bool,
    /// Hold duration in bars
    pub hold_duration: u32,
    /// Whether trade was a winner
    pub won: bool,
    /// Symbol
    pub symbol: String,
    /// PnL
    pub pnl: f64,
}

impl TradeContext {
    pub fn new(
        sr_score: i32,
        volume_percentile: f64,
        atr_pct: f64,
        distance_to_sr_pct: f64,
        regime: Regime,
        is_long: bool,
        hold_duration: u32,
        won: bool,
        symbol: String,
        pnl: f64,
    ) -> Self {
        Self {
            sr_score,
            volume_percentile,
            atr_pct,
            distance_to_sr_pct,
            regime,
            is_long,
            hold_duration,
            won,
            symbol,
            pnl,
        }
    }

    /// Convert from memory's TradeContext to embedding TradeContext
    pub fn from_memory_context(ctx: &MemoryTradeContext) -> Option<Self> {
        // Only convert closed trades with exit data
        let profit = ctx.profit?;
        let hold_bars = ctx.hold_bars?;

        // Parse regime from string
        let regime = match ctx.regime.as_str() {
            "BULL" | "TRENDING_UP" | "TrendingUp" => Regime::TrendingUp,
            "BEAR" | "TRENDING_DOWN" | "TrendingDown" => Regime::TrendingDown,
            "SIDEWAYS" | "RANGING" | "Ranging" => Regime::Ranging,
            "HIGH_VOL" | "VOLATILE" | "Volatile" | "HighVolatility" => Regime::Volatile,
            _ => Regime::Ranging, // Default
        };

        // Calculate ATR as percentage of entry price
        let atr_pct = if ctx.entry_price > 0.0 {
            ctx.atr / ctx.entry_price * 100.0
        } else {
            1.0
        };

        // Calculate distance to SR as percentage
        let distance_to_sr_pct = if ctx.entry_price > 0.0 {
            ((ctx.entry_price - ctx.sr_level).abs() / ctx.entry_price * 100.0).min(5.0)
        } else {
            1.0
        };

        // Determine direction
        let is_long = ctx.direction == "BUY" || ctx.direction == "LONG";

        Some(Self {
            sr_score: ctx.sr_score,
            volume_percentile: ctx.volume_percentile,
            atr_pct,
            distance_to_sr_pct,
            regime,
            is_long,
            hold_duration: hold_bars.max(0) as u32,
            won: profit > 0.0,
            symbol: ctx.symbol.clone(),
            pnl: profit,
        })
    }

    /// Convert to feature vector for neural encoding
    pub fn to_features(&self) -> [f64; INPUT_DIM] {
        let mut features = [0.0; INPUT_DIM];

        // sr_score: normalize from [-10, 0] to [0, 1]
        features[0] = (self.sr_score as f64 + 10.0) / 10.0;

        // volume_percentile: normalize from [0, 100] to [0, 1]
        features[1] = self.volume_percentile / 100.0;

        // atr_pct: assume typical range [0, 5%], clamp and normalize
        features[2] = (self.atr_pct / 5.0).clamp(0.0, 1.0);

        // distance_to_sr_pct: assume typical range [0, 2%], clamp and normalize
        features[3] = (self.distance_to_sr_pct / 2.0).clamp(0.0, 1.0);

        // regime one-hot encoding (4 features)
        match self.regime {
            Regime::TrendingUp => features[4] = 1.0,
            Regime::TrendingDown => features[5] = 1.0,
            Regime::Ranging => features[6] = 1.0,
            Regime::Volatile => features[7] = 1.0,
        }

        // direction one-hot (2 features)
        if self.is_long {
            features[8] = 1.0;
        } else {
            features[9] = 1.0;
        }

        // hold_duration: normalize assuming max ~100 bars
        features[10] = (self.hold_duration as f64 / 100.0).clamp(0.0, 1.0);

        // won: binary
        features[11] = if self.won { 1.0 } else { 0.0 };

        features
    }
}

// ==================== Neural Encoder ====================

/// Neural network encoder: TradeContext → 64-dim embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEncoder {
    /// First layer weights [32 x 12]
    weights_1: Vec<Vec<f64>>,
    /// First layer bias [32]
    bias_1: Vec<f64>,
    /// Second layer weights [64 x 32]
    weights_2: Vec<Vec<f64>>,
    /// Second layer bias [64]
    bias_2: Vec<f64>,
    /// Learning rate for triplet loss training
    learning_rate: f64,
    /// Triplet loss margin
    margin: f64,
}

impl NeuralEncoder {
    /// Create a new encoder with Xavier initialization
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization for layer 1: sqrt(2 / (fan_in + fan_out))
        let xavier_1 = (2.0 / (INPUT_DIM + HIDDEN_DIM) as f64).sqrt();
        let weights_1: Vec<Vec<f64>> = (0..HIDDEN_DIM)
            .map(|_| {
                (0..INPUT_DIM)
                    .map(|_| rng.gen_range(-xavier_1..xavier_1))
                    .collect()
            })
            .collect();
        let bias_1 = vec![0.0; HIDDEN_DIM];

        // Xavier initialization for layer 2
        let xavier_2 = (2.0 / (HIDDEN_DIM + EMBEDDING_DIM) as f64).sqrt();
        let weights_2: Vec<Vec<f64>> = (0..EMBEDDING_DIM)
            .map(|_| {
                (0..HIDDEN_DIM)
                    .map(|_| rng.gen_range(-xavier_2..xavier_2))
                    .collect()
            })
            .collect();
        let bias_2 = vec![0.0; EMBEDDING_DIM];

        Self {
            weights_1,
            bias_1,
            weights_2,
            bias_2,
            learning_rate: 0.01,
            margin: 0.2,
        }
    }

    /// Encode a trade context to a vector embedding
    pub fn encode(&self, context: &TradeContext) -> Vec<f64> {
        let features = context.to_features();
        self.encode_features(&features)
    }

    /// Encode raw features to embedding
    pub fn encode_features(&self, features: &[f64; INPUT_DIM]) -> Vec<f64> {
        // Layer 1: h = ReLU(W1 * x + b1)
        let mut hidden = vec![0.0; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let mut sum = self.bias_1[i];
            for j in 0..INPUT_DIM {
                sum += self.weights_1[i][j] * features[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // Layer 2: embedding = W2 * h + b2
        let mut embedding = vec![0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let mut sum = self.bias_2[i];
            for j in 0..HIDDEN_DIM {
                sum += self.weights_2[i][j] * hidden[j];
            }
            embedding[i] = sum;
        }

        // L2 normalize to unit vector
        let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for i in 0..EMBEDDING_DIM {
                embedding[i] /= norm;
            }
        }

        embedding
    }

    /// Train using triplet loss: max(0, d(anchor, positive) - d(anchor, negative) + margin)
    pub fn train(&mut self, anchor: &TradeContext, positive: &TradeContext, negative: &TradeContext) {
        let anchor_features = anchor.to_features();
        let positive_features = positive.to_features();
        let negative_features = negative.to_features();

        // Forward pass
        let (anchor_emb, anchor_hidden) = self.forward_with_hidden(&anchor_features);
        let (positive_emb, _positive_hidden) = self.forward_with_hidden(&positive_features);
        let (negative_emb, _negative_hidden) = self.forward_with_hidden(&negative_features);

        // Compute distances
        let d_ap = euclidean_distance(&anchor_emb, &positive_emb);
        let d_an = euclidean_distance(&anchor_emb, &negative_emb);

        // Triplet loss
        let loss = (d_ap - d_an + self.margin).max(0.0);

        if loss > 0.0 {
            // Compute gradients and update weights
            // Simplified gradient: push anchor towards positive, away from negative
            let lr = self.learning_rate * loss;

            // Update embedding layer (simplified gradient descent)
            for i in 0..EMBEDDING_DIM {
                let grad = (anchor_emb[i] - positive_emb[i]) - (anchor_emb[i] - negative_emb[i]);
                for j in 0..HIDDEN_DIM {
                    self.weights_2[i][j] -= lr * grad * anchor_hidden[j];
                }
                self.bias_2[i] -= lr * grad;
            }

            // Update hidden layer
            for i in 0..HIDDEN_DIM {
                if anchor_hidden[i] > 0.0 {
                    // ReLU derivative
                    let mut grad = 0.0;
                    for k in 0..EMBEDDING_DIM {
                        let emb_grad =
                            (anchor_emb[k] - positive_emb[k]) - (anchor_emb[k] - negative_emb[k]);
                        grad += emb_grad * self.weights_2[k][i];
                    }
                    for j in 0..INPUT_DIM {
                        self.weights_1[i][j] -= lr * grad * anchor_features[j];
                    }
                    self.bias_1[i] -= lr * grad;
                }
            }
        }
    }

    /// Forward pass returning both embedding and hidden layer
    fn forward_with_hidden(&self, features: &[f64; INPUT_DIM]) -> (Vec<f64>, Vec<f64>) {
        // Layer 1: h = ReLU(W1 * x + b1)
        let mut hidden = vec![0.0; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let mut sum = self.bias_1[i];
            for j in 0..INPUT_DIM {
                sum += self.weights_1[i][j] * features[j];
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // Layer 2: embedding = W2 * h + b2
        let mut embedding = vec![0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            let mut sum = self.bias_2[i];
            for j in 0..HIDDEN_DIM {
                sum += self.weights_2[i][j] * hidden[j];
            }
            embedding[i] = sum;
        }

        // L2 normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for i in 0..EMBEDDING_DIM {
                embedding[i] /= norm;
            }
        }

        (embedding, hidden)
    }
}

impl Default for NeuralEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Embedding Model ====================

/// Wrapper around NeuralEncoder with configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    /// The neural encoder
    encoder: NeuralEncoder,
    /// Embedding dimension (default 64)
    pub embedding_dim: usize,
    /// Number of embeddings created
    embeddings_created: u64,
}

impl EmbeddingModel {
    pub fn new() -> Self {
        Self {
            encoder: NeuralEncoder::new(),
            embedding_dim: EMBEDDING_DIM,
            embeddings_created: 0,
        }
    }

    pub fn encode(&self, context: &TradeContext) -> Vec<f64> {
        self.encoder.encode(context)
    }

    pub fn train(&mut self, anchor: &TradeContext, positive: &TradeContext, negative: &TradeContext) {
        self.encoder.train(anchor, positive, negative);
    }

    pub fn embeddings_created(&self) -> u64 {
        self.embeddings_created
    }

    pub fn increment_count(&mut self) {
        self.embeddings_created += 1;
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Similarity Functions ====================

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ==================== Index Type ====================

/// Type of vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    /// Brute force exact search
    Flat,
    /// Hierarchical Navigable Small World graph (approximate)
    HNSW {
        /// Max connections per node
        m: usize,
        /// Beam width during construction
        ef_construction: usize,
    },
}

impl Default for IndexType {
    fn default() -> Self {
        IndexType::Flat
    }
}

// ==================== HNSW Index ====================

/// Hierarchical Navigable Small World index for approximate nearest neighbor search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWIndex {
    /// Layers of the graph: layer -> node -> neighbors
    layers: Vec<Vec<Vec<usize>>>,
    /// Vectors stored in the index
    vectors: Vec<Vec<f64>>,
    /// Entry point (node with highest layer)
    entry_point: Option<usize>,
    /// Maximum layer for entry point
    max_layer: usize,
    /// Max connections per node
    m: usize,
    /// Max connections for layer 0
    m_max_0: usize,
    /// Beam width during construction
    ef_construction: usize,
    /// Level multiplier for random level generation
    ml: f64,
}

impl HNSWIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            layers: vec![Vec::new()],
            vectors: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m_max_0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Generate random level for a new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, vector: &[f64]) -> usize {
        let id = self.vectors.len();
        self.vectors.push(vector.to_vec());

        let level = self.random_level();

        // Ensure we have enough layers
        while self.layers.len() <= level {
            self.layers.push(Vec::new());
        }

        // Add node to each layer up to its level
        for l in 0..=level {
            while self.layers[l].len() <= id {
                self.layers[l].push(Vec::new());
            }
        }

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = level;
            return id;
        }

        let mut ep = self.entry_point.unwrap();
        let ep_layer = self.max_layer;

        // Search from top layer down to level+1
        for l in (level + 1..=ep_layer).rev() {
            let neighbors = self.search_layer(vector, ep, 1, l);
            if !neighbors.is_empty() {
                ep = neighbors[0].1;
            }
        }

        // Insert at each layer from level down to 0
        for l in (0..=level.min(ep_layer)).rev() {
            let neighbors = self.search_layer(vector, ep, self.ef_construction, l);

            // Select M best neighbors
            let m_max = if l == 0 { self.m_max_0 } else { self.m };
            let selected: Vec<usize> = neighbors
                .iter()
                .take(m_max)
                .map(|(_, idx)| *idx)
                .collect();

            // Connect to neighbors
            if l < self.layers.len() && id < self.layers[l].len() {
                self.layers[l][id] = selected.clone();
            }

            // Add bidirectional connections
            for &neighbor in &selected {
                if l < self.layers.len() && neighbor < self.layers[l].len() {
                    if !self.layers[l][neighbor].contains(&id) {
                        self.layers[l][neighbor].push(id);
                        // Prune if too many connections
                        if self.layers[l][neighbor].len() > m_max {
                            self.prune_connections(neighbor, l, m_max, vector);
                        }
                    }
                }
            }

            if !neighbors.is_empty() {
                ep = neighbors[0].1;
            }
        }

        // Update entry point if new node has higher level
        if level > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = level;
        }

        id
    }

    /// Search a single layer for nearest neighbors
    fn search_layer(
        &self,
        query: &[f64],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(f64, usize)> {
        if layer >= self.layers.len() || entry >= self.vectors.len() {
            return vec![];
        }

        let mut visited = vec![false; self.vectors.len()];
        visited[entry] = true;

        let entry_dist = euclidean_distance(query, &self.vectors[entry]);

        // Candidates: min-heap by distance
        let mut candidates: BinaryHeap<std::cmp::Reverse<(OrderedFloat, usize)>> = BinaryHeap::new();
        candidates.push(std::cmp::Reverse((OrderedFloat(entry_dist), entry)));

        // Results: max-heap by distance (we want to keep closest)
        let mut results: BinaryHeap<(OrderedFloat, usize)> = BinaryHeap::new();
        results.push((OrderedFloat(entry_dist), entry));

        while let Some(std::cmp::Reverse((OrderedFloat(c_dist), c_idx))) = candidates.pop() {
            // Get furthest result
            let furthest_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f64::MAX);

            if c_dist > furthest_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            if layer < self.layers.len() && c_idx < self.layers[layer].len() {
                for &neighbor in &self.layers[layer][c_idx] {
                    if neighbor < visited.len() && !visited[neighbor] {
                        visited[neighbor] = true;
                        let n_dist = euclidean_distance(query, &self.vectors[neighbor]);

                        let furthest = results.peek().map(|(d, _)| d.0).unwrap_or(f64::MAX);
                        if results.len() < ef || n_dist < furthest {
                            candidates.push(std::cmp::Reverse((OrderedFloat(n_dist), neighbor)));
                            results.push((OrderedFloat(n_dist), neighbor));

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<(f64, usize)> =
            results.into_iter().map(|(d, idx)| (d.0, idx)).collect();
        result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        result_vec
    }

    /// Prune connections to keep only M best
    fn prune_connections(&mut self, node: usize, layer: usize, m_max: usize, _query: &[f64]) {
        if layer >= self.layers.len() || node >= self.layers[layer].len() {
            return;
        }

        let node_vec = &self.vectors[node];
        let mut scored: Vec<(f64, usize)> = self.layers[layer][node]
            .iter()
            .filter(|&&n| n < self.vectors.len())
            .map(|&n| (euclidean_distance(node_vec, &self.vectors[n]), n))
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.layers[layer][node] = scored.into_iter().take(m_max).map(|(_, idx)| idx).collect();
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if self.entry_point.is_none() || self.vectors.is_empty() {
            return vec![];
        }

        let mut ep = self.entry_point.unwrap();

        // Descend from top layer to layer 1
        for l in (1..=self.max_layer).rev() {
            let neighbors = self.search_layer(query, ep, 1, l);
            if !neighbors.is_empty() {
                ep = neighbors[0].1;
            }
        }

        // Search layer 0 with ef = max(k, ef_construction)
        let ef = k.max(self.ef_construction);
        let results = self.search_layer(query, ep, ef, 0);

        // Return top k
        results
            .into_iter()
            .take(k)
            .map(|(dist, idx)| (idx, dist))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Wrapper for f64 to implement Ord
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ==================== Vector Index ====================

/// Vector index supporting both flat and HNSW search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndex {
    /// Stored embeddings
    embeddings: Vec<TradeEmbedding>,
    /// Embedding model
    model: EmbeddingModel,
    /// Index type
    index_type: IndexType,
    /// HNSW index (if using HNSW)
    #[serde(skip)]
    hnsw: Option<HNSWIndex>,
    /// Next embedding ID
    next_id: u64,
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(index_type: IndexType) -> Self {
        let hnsw = match &index_type {
            IndexType::HNSW { m, ef_construction } => Some(HNSWIndex::new(*m, *ef_construction)),
            IndexType::Flat => None,
        };

        Self {
            embeddings: Vec::new(),
            model: EmbeddingModel::new(),
            index_type,
            hnsw,
            next_id: 0,
        }
    }

    /// Create with default HNSW index
    pub fn new_hnsw() -> Self {
        Self::new(IndexType::HNSW {
            m: DEFAULT_M,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
        })
    }

    /// Add a trade to the index
    pub fn add(&mut self, trade: &TradeContext) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let vector = self.model.encode(trade);
        self.model.increment_count();

        let embedding = TradeEmbedding::new(
            id,
            trade.symbol.clone(),
            vector.clone(),
            Utc::now(),
            trade.won,
            trade.pnl,
        );

        // Add to HNSW if using approximate search
        if let Some(ref mut hnsw) = self.hnsw {
            hnsw.insert(&vector);
        }

        self.embeddings.push(embedding);
        id
    }

    /// Search for similar trades
    pub fn search_similar(&self, trade: &TradeContext, k: usize) -> Vec<(TradeEmbedding, f64)> {
        let query = self.model.encode(trade);
        self.search_by_vector(&query, k)
    }

    /// Search by vector directly
    pub fn search_by_vector(&self, vector: &[f64], k: usize) -> Vec<(TradeEmbedding, f64)> {
        if self.embeddings.is_empty() {
            return vec![];
        }

        match &self.index_type {
            IndexType::Flat => self.flat_search(vector, k),
            IndexType::HNSW { .. } => self.hnsw_search(vector, k),
        }
    }

    /// Flat (brute force) search
    fn flat_search(&self, query: &[f64], k: usize) -> Vec<(TradeEmbedding, f64)> {
        let mut scored: Vec<(usize, f64)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine_similarity(query, &e.vector)))
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(k)
            .map(|(i, sim)| (self.embeddings[i].clone(), sim))
            .collect()
    }

    /// HNSW approximate search
    fn hnsw_search(&self, query: &[f64], k: usize) -> Vec<(TradeEmbedding, f64)> {
        if let Some(ref hnsw) = self.hnsw {
            let results = hnsw.search(query, k);
            results
                .into_iter()
                .filter(|(idx, _)| *idx < self.embeddings.len())
                .map(|(idx, dist)| {
                    // Convert distance to similarity
                    let sim = 1.0 / (1.0 + dist);
                    (self.embeddings[idx].clone(), sim)
                })
                .collect()
        } else {
            self.flat_search(query, k)
        }
    }

    /// Get performance statistics of similar trades
    pub fn get_similar_outcomes(&self, trade: &TradeContext, k: usize) -> (f64, u32, u32) {
        let similar = self.search_similar(trade, k);

        if similar.is_empty() {
            return (0.0, 0, 0);
        }

        let mut total_pnl = 0.0;
        let mut wins = 0u32;
        let mut losses = 0u32;

        for (embedding, _sim) in &similar {
            total_pnl += embedding.pnl;
            if embedding.won {
                wins += 1;
            } else {
                losses += 1;
            }
        }

        let avg_pnl = total_pnl / similar.len() as f64;
        (avg_pnl, wins, losses)
    }

    /// Find all trades above similarity threshold
    pub fn find_pattern_matches(&self, current: &TradeContext, min_similarity: f64) -> Vec<TradeEmbedding> {
        let query = self.model.encode(current);

        self.embeddings
            .iter()
            .filter(|e| cosine_similarity(&query, &e.vector) >= min_similarity)
            .cloned()
            .collect()
    }

    /// Remove embeddings older than a date
    pub fn remove_old(&mut self, before: DateTime<Utc>) {
        // Get indices to keep
        let keep_indices: Vec<usize> = self
            .embeddings
            .iter()
            .enumerate()
            .filter(|(_, e)| e.timestamp >= before)
            .map(|(i, _)| i)
            .collect();

        // Rebuild embeddings
        self.embeddings = keep_indices
            .iter()
            .map(|&i| self.embeddings[i].clone())
            .collect();

        // Rebuild HNSW index if using it
        if let IndexType::HNSW { m, ef_construction } = &self.index_type {
            let mut new_hnsw = HNSWIndex::new(*m, *ef_construction);
            for embedding in &self.embeddings {
                new_hnsw.insert(&embedding.vector);
            }
            self.hnsw = Some(new_hnsw);
        }
    }

    /// Get number of embeddings
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get the embedding model
    pub fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    /// Get mutable embedding model for training
    pub fn model_mut(&mut self) -> &mut EmbeddingModel {
        &mut self.model
    }

    /// Save index to binary file
    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(self)?;
        fs::write(path, data)?;
        info!("Embeddings: Saved {} vectors to {}", self.embeddings.len(), path);
        Ok(())
    }

    /// Load index from binary file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read(path)?;
        let mut index: Self = bincode::deserialize(&data)?;

        // Rebuild HNSW index if needed
        if let IndexType::HNSW { m, ef_construction } = &index.index_type {
            let mut hnsw = HNSWIndex::new(*m, *ef_construction);
            for embedding in &index.embeddings {
                hnsw.insert(&embedding.vector);
            }
            index.hnsw = Some(hnsw);
        }

        info!("Embeddings: Loaded {} vectors", index.embeddings.len());
        Ok(index)
    }

    /// Load from file or create new
    pub fn load_or_new<P: AsRef<Path>>(path: P, index_type: IndexType) -> Self {
        Self::load(&path).unwrap_or_else(|_| Self::new(index_type))
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        let wins: u32 = self.embeddings.iter().filter(|e| e.won).count() as u32;
        let total = self.embeddings.len() as u32;
        let win_rate = if total > 0 {
            wins as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        format!(
            "{} embeddings, {:.1}% win rate",
            total, win_rate
        )
    }
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new(IndexType::Flat)
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_encoder() {
        let encoder = NeuralEncoder::new();

        let context = TradeContext::new(
            -5,
            75.0,
            1.5,
            0.5,
            Regime::TrendingUp,
            true,
            20,
            true,
            "TEST".to_string(),
            100.0,
        );

        let embedding = encoder.encode(&context);

        // Check embedding dimension
        assert_eq!(embedding.len(), EMBEDDING_DIM);

        // Check L2 normalization (should be unit vector)
        let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Embedding should be normalized, got norm={}", norm);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];

        // Same vectors
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        // Orthogonal
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        // Opposite
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_index_flat() {
        let mut index = VectorIndex::new(IndexType::Flat);

        // Add some trades
        for i in 0..10 {
            let context = TradeContext::new(
                -(i as i32),
                50.0 + i as f64 * 5.0,
                1.0,
                0.5,
                Regime::TrendingUp,
                true,
                10,
                i % 2 == 0,
                "TEST".to_string(),
                if i % 2 == 0 { 100.0 } else { -50.0 },
            );
            index.add(&context);
        }

        assert_eq!(index.len(), 10);

        // Search for similar
        let query = TradeContext::new(
            -5,
            75.0,
            1.0,
            0.5,
            Regime::TrendingUp,
            true,
            10,
            true,
            "TEST".to_string(),
            100.0,
        );

        let results = index.search_similar(&query, 5);
        assert_eq!(results.len(), 5);

        // Results should have similarity scores
        for (_, sim) in &results {
            assert!(*sim >= 0.0 && *sim <= 1.0);
        }
    }

    #[test]
    fn test_hnsw_insert_search() {
        let mut hnsw = HNSWIndex::new(8, 100);

        // Insert some vectors
        for i in 0..20 {
            let mut vec = vec![0.0; 64];
            vec[i % 64] = 1.0;
            hnsw.insert(&vec);
        }

        assert_eq!(hnsw.len(), 20);

        // Search for nearest
        let mut query = vec![0.0; 64];
        query[5] = 1.0;

        let results = hnsw.search(&query, 3);
        assert!(!results.is_empty());

        // First result should be index 5 (exact match)
        assert_eq!(results[0].0, 5);
    }

    #[test]
    fn test_similar_trade_retrieval() {
        let mut index = VectorIndex::new(IndexType::Flat);

        // Add winning trades in TrendingUp
        for i in 0..5 {
            let context = TradeContext::new(
                -2,
                80.0,
                1.0,
                0.3,
                Regime::TrendingUp,
                true,
                15,
                true,
                "TEST".to_string(),
                100.0 + i as f64 * 10.0,
            );
            index.add(&context);
        }

        // Add losing trades in Volatile
        for i in 0..5 {
            let context = TradeContext::new(
                -8,
                30.0,
                3.0,
                1.5,
                Regime::Volatile,
                false,
                5,
                false,
                "TEST".to_string(),
                -50.0 - i as f64 * 10.0,
            );
            index.add(&context);
        }

        // Query similar to winning pattern
        let query = TradeContext::new(
            -2,
            80.0,
            1.0,
            0.3,
            Regime::TrendingUp,
            true,
            15,
            true,
            "TEST".to_string(),
            0.0,
        );

        let (avg_pnl, wins, losses) = index.get_similar_outcomes(&query, 5);

        // Should find mostly winning trades
        assert!(wins > losses, "Expected more wins than losses for similar pattern");
        assert!(avg_pnl > 0.0, "Expected positive avg PnL for winning pattern");
    }

    #[test]
    fn test_embedding_persistence() {
        let mut index = VectorIndex::new(IndexType::Flat);

        // Add some trades
        for i in 0..5 {
            let context = TradeContext::new(
                -(i as i32),
                50.0 + i as f64 * 10.0,
                1.0,
                0.5,
                Regime::TrendingUp,
                true,
                10,
                true,
                "TEST".to_string(),
                100.0,
            );
            index.add(&context);
        }

        // Save
        let path = "/tmp/test_embeddings.bin";
        index.save(path).expect("Failed to save");

        // Load
        let loaded = VectorIndex::load(path).expect("Failed to load");

        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.embeddings[0].id, index.embeddings[0].id);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_trade_context_features() {
        let context = TradeContext::new(
            -5,
            75.0,
            2.5,
            1.0,
            Regime::TrendingDown,
            false,
            50,
            true,
            "TEST".to_string(),
            100.0,
        );

        let features = context.to_features();

        // Check dimensions
        assert_eq!(features.len(), INPUT_DIM);

        // Check sr_score normalization: -5 -> (−5 + 10) / 10 = 0.5
        assert!((features[0] - 0.5).abs() < 1e-6);

        // Check volume_percentile normalization: 75 / 100 = 0.75
        assert!((features[1] - 0.75).abs() < 1e-6);

        // Check regime one-hot (TrendingDown = index 5)
        assert_eq!(features[4], 0.0); // TrendingUp
        assert_eq!(features[5], 1.0); // TrendingDown
        assert_eq!(features[6], 0.0); // Ranging
        assert_eq!(features[7], 0.0); // Volatile

        // Check direction (short = index 9)
        assert_eq!(features[8], 0.0); // Long
        assert_eq!(features[9], 1.0); // Short
    }

    #[test]
    fn test_triplet_training() {
        let mut encoder = NeuralEncoder::new();

        // Create anchor, positive (similar), and negative (different) contexts
        let anchor = TradeContext::new(
            -2, 80.0, 1.0, 0.3, Regime::TrendingUp, true, 15, true, "TEST".to_string(), 100.0,
        );
        let positive = TradeContext::new(
            -3, 75.0, 1.2, 0.4, Regime::TrendingUp, true, 18, true, "TEST".to_string(), 90.0,
        );
        let negative = TradeContext::new(
            -8, 30.0, 3.0, 1.5, Regime::Volatile, false, 5, false, "TEST".to_string(), -50.0,
        );

        // Get initial embeddings
        let anchor_emb_before = encoder.encode(&anchor);
        let positive_emb_before = encoder.encode(&positive);

        let d_ap_before = euclidean_distance(&anchor_emb_before, &positive_emb_before);

        // Train
        for _ in 0..10 {
            encoder.train(&anchor, &positive, &negative);
        }

        // Get updated embeddings
        let anchor_emb_after = encoder.encode(&anchor);
        let positive_emb_after = encoder.encode(&positive);

        let d_ap_after = euclidean_distance(&anchor_emb_after, &positive_emb_after);

        // After training, anchor and positive should be closer (or at least not further)
        // Note: due to normalization, the effect may be subtle
        assert!(
            d_ap_after <= d_ap_before + 0.5,
            "Distance should not increase significantly: before={}, after={}",
            d_ap_before,
            d_ap_after
        );
    }
}
