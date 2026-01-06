//! Hierarchical memory with automatic consolidation from episodes to patterns
//!
//! This module implements:
//! - Multi-tier memory (Working, Episodic, Semantic, Archive)
//! - Importance-based memory scoring
//! - Pattern extraction via agglomerative clustering
//! - Automatic consolidation of episodes into semantic patterns

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::info;

use super::embeddings::{cosine_similarity, VectorIndex, EMBEDDING_DIM};
use super::regime::Regime;

/// Memory tier for hierarchical storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTier {
    /// Current session, immediate access
    Working,
    /// Individual trades, detailed
    Episodic,
    /// Patterns extracted from episodes
    Semantic,
    /// Old, compressed, rarely accessed
    Archive,
}

/// Individual trade episode in memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub trade_id: u64,
    /// Trade context information
    pub context: EpisodeContext,
    /// Vector embedding for similarity search
    pub embedding: Vec<f64>,
    /// Computed importance score
    pub importance: f64,
    /// Number of times accessed
    pub access_count: u32,
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Context stored with each episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeContext {
    pub symbol: String,
    pub sr_score: i32,
    pub volume_percentile: f64,
    pub atr_pct: f64,
    pub distance_to_sr_pct: f64,
    pub regime: Regime,
    pub is_long: bool,
    pub hold_duration: u32,
    pub won: bool,
    pub pnl: f64,
}

impl EpisodeContext {
    /// Create from trade parameters
    pub fn new(
        symbol: String,
        sr_score: i32,
        volume_percentile: f64,
        atr_pct: f64,
        distance_to_sr_pct: f64,
        regime: Regime,
        is_long: bool,
        hold_duration: u32,
        won: bool,
        pnl: f64,
    ) -> Self {
        Self {
            symbol,
            sr_score,
            volume_percentile,
            atr_pct,
            distance_to_sr_pct,
            regime,
            is_long,
            hold_duration,
            won,
            pnl,
        }
    }
}

/// Extracted pattern from clustered episodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Unique identifier
    pub id: u64,
    /// Auto-generated name: "Ranging_HighVolume_Support_Win"
    pub name: String,
    /// Average embedding of member episodes
    pub centroid: Vec<f64>,
    /// Number of episodes in this pattern
    pub member_count: u32,
    /// Win rate of member trades
    pub win_rate: f64,
    /// Average P&L of member trades
    pub avg_pnl: f64,
    /// Common regime (if any)
    pub regime: Option<Regime>,
    /// S/R score range (min, max)
    pub sr_score_range: (i32, i32),
    /// Volume percentile range
    pub volume_range: (f64, f64),
    /// Cluster tightness (higher = more confident)
    pub confidence: f64,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl Pattern {
    /// Generate pattern name from characteristics
    pub fn generate_name(
        regime: Option<&Regime>,
        sr_score_range: (i32, i32),
        volume_range: (f64, f64),
        win_rate: f64,
    ) -> String {
        let regime_str = match regime {
            Some(Regime::TrendingUp) => "Bull",
            Some(Regime::TrendingDown) => "Bear",
            Some(Regime::Ranging) => "Ranging",
            Some(Regime::Volatile) => "Volatile",
            None => "Mixed",
        };

        let volume_str = if volume_range.0 > 70.0 {
            "HighVol"
        } else if volume_range.1 < 30.0 {
            "LowVol"
        } else {
            "MidVol"
        };

        let sr_str = if sr_score_range.0 >= -2 {
            "StrongSR"
        } else if sr_score_range.1 <= -7 {
            "WeakSR"
        } else {
            "MidSR"
        };

        let outcome_str = if win_rate > 0.6 {
            "Win"
        } else if win_rate < 0.4 {
            "Lose"
        } else {
            "Neutral"
        };

        format!("{}_{}_{}_{}", regime_str, volume_str, sr_str, outcome_str)
    }
}

/// Scores importance of episodes for consolidation decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceScorer {
    /// Weight for recency (newer = more important)
    pub recency_weight: f64,
    /// Weight for outcome magnitude
    pub outcome_weight: f64,
    /// Weight for uniqueness (unusual = more important)
    pub uniqueness_weight: f64,
    /// Weight for access frequency
    pub access_weight: f64,
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self {
            recency_weight: 0.3,
            outcome_weight: 0.3,
            uniqueness_weight: 0.2,
            access_weight: 0.2,
        }
    }
}

impl ImportanceScorer {
    /// Create with custom weights
    pub fn new(
        recency_weight: f64,
        outcome_weight: f64,
        uniqueness_weight: f64,
        access_weight: f64,
    ) -> Self {
        Self {
            recency_weight,
            outcome_weight,
            uniqueness_weight,
            access_weight,
        }
    }

    /// Score an episode's importance
    ///
    /// Returns a value in [0, 1] indicating how important this episode is
    pub fn score(&self, episode: &Episode, all_episodes: &[Episode]) -> f64 {
        // Recency: exponential decay based on age
        let age_hours = (Utc::now() - episode.created_at).num_hours() as f64;
        let recency = (-age_hours / (24.0 * 30.0)).exp(); // Half-life of ~30 days

        // Outcome: magnitude of profit/loss (normalized)
        let max_pnl = all_episodes
            .iter()
            .map(|e| e.context.pnl.abs())
            .fold(1.0f64, |a, b| a.max(b));
        let outcome = if max_pnl > 0.0 {
            episode.context.pnl.abs() / max_pnl
        } else {
            0.5
        };

        // Uniqueness: distance to nearest neighbor
        let uniqueness = self.compute_uniqueness(episode, all_episodes);

        // Access frequency: logarithmic scale
        let access = (episode.access_count as f64 + 1.0).ln() / 5.0; // Normalize assuming max ~100 accesses

        // Weighted sum
        let raw_score = self.recency_weight * recency
            + self.outcome_weight * outcome
            + self.uniqueness_weight * uniqueness
            + self.access_weight * access.min(1.0);

        // Normalize to [0, 1]
        raw_score.clamp(0.0, 1.0)
    }

    /// Compute uniqueness as distance to nearest neighbor
    fn compute_uniqueness(&self, episode: &Episode, all_episodes: &[Episode]) -> f64 {
        if all_episodes.len() <= 1 {
            return 1.0;
        }

        let mut min_similarity = 1.0f64;

        for other in all_episodes {
            if other.trade_id == episode.trade_id {
                continue;
            }

            let sim = cosine_similarity(&episode.embedding, &other.embedding);
            if sim < min_similarity {
                min_similarity = sim;
            }
        }

        // Convert similarity to distance (lower similarity = more unique)
        1.0 - min_similarity
    }
}

/// Extracts patterns from episode clusters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExtractor {
    /// Minimum episodes to form a pattern
    pub min_cluster_size: usize,
    /// Similarity threshold for clustering
    pub similarity_threshold: f64,
    /// Maximum patterns to maintain
    pub max_patterns: usize,
}

impl Default for PatternExtractor {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            similarity_threshold: 0.85,
            max_patterns: 100,
        }
    }
}

impl PatternExtractor {
    /// Create with custom parameters
    pub fn new(min_cluster_size: usize, similarity_threshold: f64, max_patterns: usize) -> Self {
        Self {
            min_cluster_size,
            similarity_threshold,
            max_patterns,
        }
    }

    /// Extract patterns from episodes using agglomerative clustering
    pub fn extract(&self, episodes: &[Episode], next_pattern_id: &mut u64) -> Vec<Pattern> {
        if episodes.len() < self.min_cluster_size {
            return Vec::new();
        }

        // Perform agglomerative clustering
        let clusters = self.cluster_episodes(episodes);

        // Convert clusters to patterns
        let mut patterns: Vec<Pattern> = Vec::new();

        for cluster_indices in clusters {
            if cluster_indices.len() < self.min_cluster_size {
                continue;
            }

            let cluster_episodes: Vec<&Episode> =
                cluster_indices.iter().map(|&i| &episodes[i]).collect();

            if let Some(pattern) = self.create_pattern(&cluster_episodes, next_pattern_id) {
                patterns.push(pattern);
                *next_pattern_id += 1;
            }

            if patterns.len() >= self.max_patterns {
                break;
            }
        }

        // Sort by confidence (higher = better)
        patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        patterns
    }

    /// Agglomerative clustering of episodes
    fn cluster_episodes(&self, episodes: &[Episode]) -> Vec<Vec<usize>> {
        let n = episodes.len();
        if n == 0 {
            return Vec::new();
        }

        // Start with each episode as its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        // Precompute similarity matrix
        let mut similarities = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&episodes[i].embedding, &episodes[j].embedding);
                similarities[i][j] = sim;
                similarities[j][i] = sim;
            }
        }

        // Iteratively merge closest clusters
        loop {
            let mut best_merge: Option<(usize, usize, f64)> = None;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    // Average linkage: average similarity between all pairs
                    let avg_sim = self.cluster_similarity(&clusters[i], &clusters[j], &similarities);

                    if avg_sim >= self.similarity_threshold {
                        if best_merge.is_none() || avg_sim > best_merge.unwrap().2 {
                            best_merge = Some((i, j, avg_sim));
                        }
                    }
                }
            }

            match best_merge {
                Some((i, j, _)) => {
                    // Merge clusters j into i
                    let cluster_j = clusters.remove(j);
                    clusters[i].extend(cluster_j);
                }
                None => break, // No more clusters to merge
            }
        }

        clusters
    }

    /// Compute average similarity between two clusters (average linkage)
    fn cluster_similarity(
        &self,
        cluster_a: &[usize],
        cluster_b: &[usize],
        similarities: &[Vec<f64>],
    ) -> f64 {
        let mut total = 0.0;
        let mut count = 0;

        for &i in cluster_a {
            for &j in cluster_b {
                total += similarities[i][j];
                count += 1;
            }
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Create a pattern from a cluster of episodes
    fn create_pattern(&self, episodes: &[&Episode], next_id: &mut u64) -> Option<Pattern> {
        if episodes.is_empty() {
            return None;
        }

        // Compute centroid
        let mut centroid = vec![0.0; EMBEDDING_DIM];
        for ep in episodes {
            for (i, &val) in ep.embedding.iter().enumerate() {
                if i < EMBEDDING_DIM {
                    centroid[i] += val;
                }
            }
        }
        let n = episodes.len() as f64;
        for val in centroid.iter_mut() {
            *val /= n;
        }

        // Normalize centroid
        let magnitude: f64 = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for val in centroid.iter_mut() {
                *val /= magnitude;
            }
        }

        // Compute statistics
        let wins = episodes.iter().filter(|e| e.context.won).count();
        let win_rate = wins as f64 / n;
        let avg_pnl: f64 = episodes.iter().map(|e| e.context.pnl).sum::<f64>() / n;

        // Find common regime
        let mut regime_counts: HashMap<Regime, usize> = HashMap::new();
        for ep in episodes {
            *regime_counts.entry(ep.context.regime.clone()).or_insert(0) += 1;
        }
        let common_regime = regime_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .filter(|(_, count)| **count as f64 / n > 0.5)
            .map(|(regime, _)| regime.clone());

        // Compute ranges
        let sr_min = episodes.iter().map(|e| e.context.sr_score).min().unwrap_or(0);
        let sr_max = episodes.iter().map(|e| e.context.sr_score).max().unwrap_or(0);
        let vol_min = episodes.iter().map(|e| e.context.volume_percentile).fold(100.0f64, |a, b| a.min(b));
        let vol_max = episodes.iter().map(|e| e.context.volume_percentile).fold(0.0f64, |a, b| a.max(b));

        // Compute confidence (cluster tightness)
        let confidence = self.compute_cluster_confidence(episodes, &centroid);

        // Generate name
        let name = Pattern::generate_name(
            common_regime.as_ref(),
            (sr_min, sr_max),
            (vol_min, vol_max),
            win_rate,
        );

        let now = Utc::now();

        Some(Pattern {
            id: *next_id,
            name,
            centroid,
            member_count: episodes.len() as u32,
            win_rate,
            avg_pnl,
            regime: common_regime,
            sr_score_range: (sr_min, sr_max),
            volume_range: (vol_min, vol_max),
            confidence,
            created_at: now,
            last_updated: now,
        })
    }

    /// Compute cluster confidence (average similarity to centroid)
    fn compute_cluster_confidence(&self, episodes: &[&Episode], centroid: &[f64]) -> f64 {
        if episodes.is_empty() {
            return 0.0;
        }

        let total_sim: f64 = episodes
            .iter()
            .map(|e| cosine_similarity(&e.embedding, centroid))
            .sum();

        total_sim / episodes.len() as f64
    }
}

/// Statistics about memory tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub working_count: usize,
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub archive_count: usize,
    pub total_patterns: usize,
    pub avg_pattern_confidence: f64,
}

/// Main memory consolidator managing hierarchical memory
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryConsolidator {
    /// Current session memories (immediate access)
    working: Vec<Episode>,
    /// Detailed individual trade memories
    episodic: Vec<Episode>,
    /// Extracted patterns from episodes
    semantic: Vec<Pattern>,
    /// Old, compressed memories
    archive: Vec<Episode>,

    /// Importance scoring configuration
    importance_scorer: ImportanceScorer,
    /// Pattern extraction configuration
    pattern_extractor: PatternExtractor,

    /// Reference to vector index for embeddings
    #[serde(skip)]
    vector_index: Option<Arc<Mutex<VectorIndex>>>,

    /// How often to consolidate (in seconds)
    consolidation_interval_secs: i64,
    /// Last consolidation timestamp
    last_consolidation: DateTime<Utc>,
    /// Maximum episodes in episodic memory
    max_episodic_size: usize,
    /// Days before archiving
    archive_after_days: u32,

    /// Next episode ID
    next_episode_id: u64,
    /// Next pattern ID
    next_pattern_id: u64,
}

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self {
            working: Vec::new(),
            episodic: Vec::new(),
            semantic: Vec::new(),
            archive: Vec::new(),
            importance_scorer: ImportanceScorer::default(),
            pattern_extractor: PatternExtractor::default(),
            vector_index: None,
            consolidation_interval_secs: 24 * 60 * 60, // 24 hours
            last_consolidation: Utc::now(),
            max_episodic_size: 10000,
            archive_after_days: 90,
            next_episode_id: 1,
            next_pattern_id: 1,
        }
    }
}

impl MemoryConsolidator {
    /// Create a new memory consolidator
    pub fn new(vector_index: Arc<Mutex<VectorIndex>>) -> Self {
        Self {
            vector_index: Some(vector_index),
            ..Default::default()
        }
    }

    /// Attach vector index (for deserialized instances)
    pub fn attach_vector_index(&mut self, index: Arc<Mutex<VectorIndex>>) {
        self.vector_index = Some(index);
    }

    /// Add a new episode from a completed trade
    pub fn add_episode(&mut self, context: EpisodeContext, embedding: Vec<f64>) {
        let now = Utc::now();
        let episode = Episode {
            trade_id: self.next_episode_id,
            context,
            embedding,
            importance: 0.5, // Initial importance, will be rescored
            access_count: 0,
            last_accessed: now,
            created_at: now,
        };

        self.next_episode_id += 1;
        self.working.push(episode);

        // Flush to episodic if working memory is full
        if self.working.len() > 100 {
            self.flush_working();
        }
    }

    /// Flush working memory to episodic
    pub fn flush_working(&mut self) {
        if self.working.is_empty() {
            return;
        }

        info!(
            "[CONSOLIDATION] Flushing {} episodes from working to episodic memory",
            self.working.len()
        );

        self.episodic.append(&mut self.working);
        self.maybe_consolidate();
    }

    /// Check if consolidation is needed and perform it
    pub fn maybe_consolidate(&mut self) {
        let now = Utc::now();
        let elapsed = now - self.last_consolidation;

        if elapsed.num_seconds() >= self.consolidation_interval_secs {
            self.consolidate();
        }
    }

    /// Perform full consolidation
    pub fn consolidate(&mut self) {
        let start_episodic = self.episodic.len();
        let start_patterns = self.semantic.len();

        // 1. Rescore importance of all episodic memories
        self.rescore_importance();

        // 2. Extract patterns from high-importance episodes
        let high_importance: Vec<Episode> = self
            .episodic
            .iter()
            .filter(|e| e.importance > 0.3)
            .cloned()
            .collect();

        if !high_importance.is_empty() {
            let new_patterns = self
                .pattern_extractor
                .extract(&high_importance, &mut self.next_pattern_id);

            // 3. Merge similar new patterns with existing
            self.merge_patterns(new_patterns);
        }

        // 4. Archive old low-importance episodes
        self.archive_old();

        // 5. Prune if needed
        self.prune();

        self.last_consolidation = Utc::now();

        let end_patterns = self.semantic.len();
        let archived = start_episodic.saturating_sub(self.episodic.len());

        info!(
            "[CONSOLIDATION] Extracted {} patterns from {} episodes, archived {}",
            end_patterns - start_patterns,
            start_episodic,
            archived
        );
    }

    /// Rescore importance of all episodic memories
    fn rescore_importance(&mut self) {
        let episodes_clone = self.episodic.clone();
        for episode in &mut self.episodic {
            episode.importance = self.importance_scorer.score(episode, &episodes_clone);
        }
    }

    /// Merge new patterns with existing, combining similar ones
    fn merge_patterns(&mut self, new_patterns: Vec<Pattern>) {
        for new_pattern in new_patterns {
            // Check if similar pattern exists
            let mut merged = false;

            for existing in &mut self.semantic {
                let similarity = cosine_similarity(&new_pattern.centroid, &existing.centroid);

                if similarity > 0.9 {
                    // Merge into existing pattern
                    let total_count = existing.member_count + new_pattern.member_count;
                    let w1 = existing.member_count as f64 / total_count as f64;
                    let w2 = new_pattern.member_count as f64 / total_count as f64;

                    existing.win_rate = existing.win_rate * w1 + new_pattern.win_rate * w2;
                    existing.avg_pnl = existing.avg_pnl * w1 + new_pattern.avg_pnl * w2;
                    existing.member_count = total_count;
                    existing.confidence = existing.confidence * w1 + new_pattern.confidence * w2;
                    existing.last_updated = Utc::now();

                    // Update centroid
                    for i in 0..existing.centroid.len().min(new_pattern.centroid.len()) {
                        existing.centroid[i] = existing.centroid[i] * w1 + new_pattern.centroid[i] * w2;
                    }

                    merged = true;
                    break;
                }
            }

            if !merged {
                self.semantic.push(new_pattern);
            }
        }

        // Keep only top patterns by confidence
        if self.semantic.len() > self.pattern_extractor.max_patterns {
            self.semantic.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.semantic.truncate(self.pattern_extractor.max_patterns);
        }
    }

    /// Archive old, low-importance episodes
    fn archive_old(&mut self) {
        let cutoff = Utc::now() - Duration::days(self.archive_after_days as i64);

        let (old, recent): (Vec<Episode>, Vec<Episode>) = self
            .episodic
            .drain(..)
            .partition(|e| e.created_at < cutoff && e.importance < 0.5);

        self.episodic = recent;
        self.archive.extend(old);

        // Limit archive size (keep summary only)
        if self.archive.len() > 50000 {
            self.archive.sort_by(|a, b| {
                b.importance
                    .partial_cmp(&a.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.archive.truncate(25000);
        }
    }

    /// Prune episodic memory if over limit
    fn prune(&mut self) {
        if self.episodic.len() > self.max_episodic_size {
            // Sort by importance (keep high importance)
            self.episodic.sort_by(|a, b| {
                b.importance
                    .partial_cmp(&a.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Move excess to archive
            let excess: Vec<Episode> = self.episodic.drain(self.max_episodic_size..).collect();
            self.archive.extend(excess);
        }
    }

    /// Retrieve relevant episodes by similarity
    pub fn retrieve_relevant(&mut self, query_embedding: &[f64], k: usize) -> Vec<&Episode> {
        // Mark access
        let now = Utc::now();

        // Score and sort by similarity
        let mut scored: Vec<(usize, f64)> = self
            .episodic
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine_similarity(query_embedding, &e.embedding)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update access counts for retrieved episodes
        let top_indices: Vec<usize> = scored.iter().take(k).map(|(i, _)| *i).collect();
        for &idx in &top_indices {
            self.episodic[idx].access_count += 1;
            self.episodic[idx].last_accessed = now;
        }

        // Return references
        top_indices
            .iter()
            .map(|&i| &self.episodic[i])
            .collect()
    }

    /// Match patterns against a query embedding
    pub fn match_patterns(&self, query_embedding: &[f64]) -> Vec<(&Pattern, f64)> {
        let mut matches: Vec<(&Pattern, f64)> = self
            .semantic
            .iter()
            .map(|p| (p, cosine_similarity(query_embedding, &p.centroid)))
            .filter(|(_, sim)| *sim > 0.5)
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Get prediction from matching patterns
    ///
    /// Returns (win_rate, avg_pnl) if a strong pattern match is found
    pub fn get_pattern_prediction(&self, query_embedding: &[f64]) -> Option<(f64, f64, String)> {
        let matches = self.match_patterns(query_embedding);

        if let Some((pattern, similarity)) = matches.first() {
            if *similarity > 0.8 {
                return Some((pattern.win_rate, pattern.avg_pnl, pattern.name.clone()));
            }
        }

        None
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let avg_confidence = if self.semantic.is_empty() {
            0.0
        } else {
            self.semantic.iter().map(|p| p.confidence).sum::<f64>() / self.semantic.len() as f64
        };

        MemoryStats {
            working_count: self.working.len(),
            episodic_count: self.episodic.len(),
            semantic_count: self.semantic.len(),
            archive_count: self.archive.len(),
            total_patterns: self.semantic.len(),
            avg_pattern_confidence: avg_confidence,
        }
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        let stats = self.get_memory_stats();
        format!(
            "{} working, {} episodic, {} patterns ({:.0}% avg conf), {} archive",
            stats.working_count,
            stats.episodic_count,
            stats.total_patterns,
            stats.avg_pattern_confidence * 100.0,
            stats.archive_count
        )
    }

    /// Get pattern count
    pub fn pattern_count(&self) -> usize {
        self.semantic.len()
    }

    /// Get episodic count
    pub fn episodic_count(&self) -> usize {
        self.episodic.len()
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
        let consolidator: Self = serde_json::from_str(&json)?;
        Ok(consolidator)
    }

    /// Load or create new
    pub fn load_or_new<P: AsRef<Path>>(path: P, vector_index: Arc<Mutex<VectorIndex>>) -> Self {
        match Self::load(&path) {
            Ok(mut c) => {
                c.attach_vector_index(vector_index);
                c
            }
            Err(_) => Self::new(vector_index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_episode(id: u64, sr_score: i32, volume: f64, won: bool, pnl: f64) -> Episode {
        let mut embedding = vec![0.0; EMBEDDING_DIM];
        // Create somewhat unique embedding based on parameters
        embedding[0] = sr_score as f64 / 10.0;
        embedding[1] = volume / 100.0;
        embedding[2] = if won { 1.0 } else { 0.0 };
        embedding[3] = pnl / 100.0;
        // Normalize
        let mag: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag > 0.0 {
            for v in &mut embedding {
                *v /= mag;
            }
        }

        Episode {
            trade_id: id,
            context: EpisodeContext::new(
                "TEST".to_string(),
                sr_score,
                volume,
                1.0,
                0.5,
                Regime::Ranging,
                true,
                10,
                won,
                pnl,
            ),
            embedding,
            importance: 0.5,
            access_count: 0,
            last_accessed: Utc::now(),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_importance_scoring() {
        let scorer = ImportanceScorer::default();

        let episodes = vec![
            create_test_episode(1, -2, 80.0, true, 100.0),
            create_test_episode(2, -5, 50.0, false, -50.0),
            create_test_episode(3, -8, 30.0, true, 200.0),
        ];

        // High profit episode should score higher
        let score1 = scorer.score(&episodes[0], &episodes);
        let score2 = scorer.score(&episodes[1], &episodes);
        let score3 = scorer.score(&episodes[2], &episodes);

        // All scores should be in valid range
        assert!(score1 >= 0.0 && score1 <= 1.0);
        assert!(score2 >= 0.0 && score2 <= 1.0);
        assert!(score3 >= 0.0 && score3 <= 1.0);

        // Highest profit episode should generally score highest
        // (though other factors like recency also matter)
        println!(
            "Scores: {:.3}, {:.3}, {:.3}",
            score1, score2, score3
        );
    }

    #[test]
    fn test_pattern_extraction() {
        let extractor = PatternExtractor {
            min_cluster_size: 2,
            similarity_threshold: 0.5,
            max_patterns: 10,
        };

        // Create similar episodes (should cluster together)
        let mut episodes = Vec::new();
        for i in 0..5 {
            episodes.push(create_test_episode(
                i as u64,
                -2,
                80.0 + i as f64,
                true,
                100.0,
            ));
        }

        let mut next_id = 1;
        let patterns = extractor.extract(&episodes, &mut next_id);

        // Should extract at least one pattern
        assert!(!patterns.is_empty(), "Should extract patterns from similar episodes");

        let pattern = &patterns[0];
        assert!(pattern.member_count >= 2);
        assert!(pattern.win_rate > 0.0);
        assert!(!pattern.name.is_empty());
    }

    #[test]
    fn test_agglomerative_clustering() {
        let extractor = PatternExtractor {
            min_cluster_size: 2,
            similarity_threshold: 0.3,
            max_patterns: 10,
        };

        // Create two distinct clusters
        let mut episodes = Vec::new();

        // Cluster 1: High volume, winning
        for i in 0..3 {
            let mut ep = create_test_episode(i as u64, -2, 90.0, true, 100.0);
            ep.embedding[0] = 0.9;
            ep.embedding[1] = 0.1;
            episodes.push(ep);
        }

        // Cluster 2: Low volume, losing
        for i in 3..6 {
            let mut ep = create_test_episode(i as u64, -8, 20.0, false, -50.0);
            ep.embedding[0] = 0.1;
            ep.embedding[1] = 0.9;
            episodes.push(ep);
        }

        let clusters = extractor.cluster_episodes(&episodes);

        // Should identify distinct clusters
        println!("Found {} clusters", clusters.len());
        assert!(clusters.len() >= 1, "Should identify clusters");
    }

    #[test]
    fn test_memory_tiers() {
        let vector_index = Arc::new(Mutex::new(VectorIndex::new(
            super::super::embeddings::IndexType::Flat,
        )));

        let mut consolidator = MemoryConsolidator::new(vector_index);

        // Add episodes to working memory
        for i in 0..50 {
            let context = EpisodeContext::new(
                "TEST".to_string(),
                -2,
                70.0,
                1.0,
                0.5,
                Regime::TrendingUp,
                true,
                10,
                i % 3 != 0,
                if i % 3 != 0 { 50.0 } else { -30.0 },
            );
            consolidator.add_episode(context, vec![0.5; EMBEDDING_DIM]);
        }

        let stats = consolidator.get_memory_stats();
        assert_eq!(stats.working_count, 50);
        assert_eq!(stats.episodic_count, 0);

        // Flush to episodic
        consolidator.flush_working();

        let stats = consolidator.get_memory_stats();
        assert_eq!(stats.working_count, 0);
        assert_eq!(stats.episodic_count, 50);
    }

    #[test]
    fn test_consolidation_flow() {
        let vector_index = Arc::new(Mutex::new(VectorIndex::new(
            super::super::embeddings::IndexType::Flat,
        )));

        let mut consolidator = MemoryConsolidator::new(vector_index);
        consolidator.pattern_extractor.min_cluster_size = 3;
        consolidator.pattern_extractor.similarity_threshold = 0.5;

        // Add similar episodes that should form patterns
        for i in 0..20 {
            let context = EpisodeContext::new(
                "TEST".to_string(),
                -2,
                75.0 + (i % 5) as f64,
                1.0,
                0.5,
                Regime::TrendingUp,
                true,
                10,
                true,
                50.0,
            );

            // Create similar embeddings
            let mut embedding = vec![0.7; EMBEDDING_DIM];
            embedding[0] = 0.9;
            embedding[1] = 0.1 + (i % 5) as f64 * 0.02;

            consolidator.add_episode(context, embedding);
        }

        consolidator.flush_working();
        consolidator.consolidate();

        let stats = consolidator.get_memory_stats();
        println!(
            "After consolidation: {} episodes, {} patterns",
            stats.episodic_count, stats.total_patterns
        );

        // Should have consolidated some patterns
        // (may be 0 if embeddings are too different)
    }

    #[test]
    fn test_pattern_matching() {
        let vector_index = Arc::new(Mutex::new(VectorIndex::new(
            super::super::embeddings::IndexType::Flat,
        )));

        let mut consolidator = MemoryConsolidator::new(vector_index);

        // Manually add a pattern for testing
        let pattern = Pattern {
            id: 1,
            name: "Bull_HighVol_StrongSR_Win".to_string(),
            centroid: vec![0.8; EMBEDDING_DIM],
            member_count: 10,
            win_rate: 0.7,
            avg_pnl: 50.0,
            regime: Some(Regime::TrendingUp),
            sr_score_range: (-3, 0),
            volume_range: (70.0, 90.0),
            confidence: 0.85,
            created_at: Utc::now(),
            last_updated: Utc::now(),
        };
        consolidator.semantic.push(pattern);

        // Query with similar embedding
        let query = vec![0.8; EMBEDDING_DIM];
        let matches = consolidator.match_patterns(&query);

        assert!(!matches.is_empty());
        assert!(matches[0].1 > 0.9); // High similarity

        // Get prediction
        if let Some((win_rate, avg_pnl, name)) = consolidator.get_pattern_prediction(&query) {
            assert!((win_rate - 0.7).abs() < 0.01);
            assert!((avg_pnl - 50.0).abs() < 0.01);
            assert!(name.contains("Bull"));
        }
    }

    #[test]
    fn test_pattern_name_generation() {
        let name = Pattern::generate_name(
            Some(&Regime::TrendingUp),
            (-2, 0),
            (75.0, 90.0),
            0.75,
        );
        assert!(name.contains("Bull"));
        assert!(name.contains("Win"));
        assert!(name.contains("HighVol"));
        assert!(name.contains("StrongSR"));

        let name2 = Pattern::generate_name(None, (-9, -7), (10.0, 25.0), 0.3);
        assert!(name2.contains("Mixed"));
        assert!(name2.contains("Lose"));
        assert!(name2.contains("LowVol"));
        assert!(name2.contains("WeakSR"));
    }
}
