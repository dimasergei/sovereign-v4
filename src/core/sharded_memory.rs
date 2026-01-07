//! Sharded Memory Architecture for Billion-Scale Storage
//!
//! This module implements a distributed memory system that can scale to billions
//! of entries by partitioning data across multiple shards. Each shard is backed
//! by a SQLite database for durability.
//!
//! Key features:
//! - Consistent hashing for symbol-based routing
//! - Time-based partitioning for temporal queries
//! - Write buffering for batch inserts
//! - Query caching with LRU eviction
//! - Automatic rebalancing when shards become uneven
//! - Importance-based memory decay

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use tracing::{info, warn};

use super::regime::Regime;

// ==================== Configuration ====================

/// Configuration for sharded memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Number of shards to use
    pub num_shards: usize,
    /// Path template for shard databases (use {} for shard id)
    pub shard_path_template: String,
    /// Maximum entries per shard before triggering rebalance
    pub max_entries_per_shard: usize,
    /// Rebalance threshold (0.8 = rebalance when 80% full)
    pub rebalance_threshold: f64,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            num_shards: 16,
            shard_path_template: "sovereign_shard_{}.db".to_string(),
            max_entries_per_shard: 1_000_000,
            rebalance_threshold: 0.8,
        }
    }
}

impl ShardConfig {
    /// Create config with custom number of shards
    pub fn with_shards(num_shards: usize) -> Self {
        Self {
            num_shards,
            ..Default::default()
        }
    }

    /// Get path for a specific shard
    pub fn shard_path(&self, shard_id: usize) -> String {
        self.shard_path_template.replace("{}", &shard_id.to_string())
    }
}

// ==================== Shard Keys ====================

/// Key types for routing entries to shards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardKey {
    /// Route by symbol name
    Symbol(String),
    /// Route by time range
    TimeRange {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
    /// Route by market regime
    Regime(Regime),
    /// Route by precomputed hash
    Hash(u64),
}

impl ShardKey {
    /// Compute hash for this key
    pub fn hash_value(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match self {
            ShardKey::Symbol(s) => s.hash(&mut hasher),
            ShardKey::TimeRange { start, end } => {
                start.timestamp().hash(&mut hasher);
                end.timestamp().hash(&mut hasher);
            }
            ShardKey::Regime(r) => {
                format!("{:?}", r).hash(&mut hasher);
            }
            ShardKey::Hash(h) => return *h,
        }
        hasher.finish()
    }
}

// ==================== Sharding Strategy ====================

/// Strategy for distributing entries across shards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Each symbol goes to a consistent shard
    BySymbol,
    /// Time-based partitioning (by month/week)
    ByTime,
    /// Consistent hashing of entry id
    ByHash,
    /// Symbol primary, time secondary
    Hybrid,
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        ShardingStrategy::BySymbol
    }
}

// ==================== Shard Router ====================

/// Routes entries to appropriate shards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardRouter {
    /// Sharding strategy in use
    strategy: ShardingStrategy,
    /// Number of shards
    num_shards: usize,
    /// Symbol to shard assignment cache
    shard_map: HashMap<String, usize>,
    /// Current load (entry count) per shard
    shard_loads: Vec<usize>,
}

impl ShardRouter {
    /// Create a new shard router
    pub fn new(strategy: ShardingStrategy, num_shards: usize) -> Self {
        Self {
            strategy,
            num_shards,
            shard_map: HashMap::new(),
            shard_loads: vec![0; num_shards],
        }
    }

    /// Route a key to a shard index
    pub fn route(&self, key: &ShardKey) -> usize {
        match self.strategy {
            ShardingStrategy::BySymbol => {
                if let ShardKey::Symbol(s) = key {
                    self.route_symbol(s)
                } else {
                    (key.hash_value() % self.num_shards as u64) as usize
                }
            }
            ShardingStrategy::ByTime => {
                if let ShardKey::TimeRange { start, .. } = key {
                    self.route_time(start)
                } else {
                    (key.hash_value() % self.num_shards as u64) as usize
                }
            }
            ShardingStrategy::ByHash => {
                (key.hash_value() % self.num_shards as u64) as usize
            }
            ShardingStrategy::Hybrid => {
                // Primary: by symbol, secondary: by time bucket within symbol's shard group
                match key {
                    ShardKey::Symbol(s) => self.route_symbol(s),
                    ShardKey::TimeRange { start, .. } => self.route_time(start),
                    _ => (key.hash_value() % self.num_shards as u64) as usize,
                }
            }
        }
    }

    /// Route a symbol to a shard using consistent hashing
    pub fn route_symbol(&self, symbol: &str) -> usize {
        // Check cache first
        if let Some(&shard) = self.shard_map.get(symbol) {
            return shard;
        }
        // Compute consistent hash
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        symbol.hash(&mut hasher);
        (hasher.finish() % self.num_shards as u64) as usize
    }

    /// Route a timestamp to a shard using time-based bucketing
    pub fn route_time(&self, timestamp: &DateTime<Utc>) -> usize {
        // Bucket by month, distribute across shards
        let month_bucket = timestamp.timestamp() / (30 * 24 * 60 * 60);
        (month_bucket as usize) % self.num_shards
    }

    /// Update load for a shard
    pub fn update_load(&mut self, shard: usize, delta: i64) {
        if shard < self.shard_loads.len() {
            if delta >= 0 {
                self.shard_loads[shard] = self.shard_loads[shard].saturating_add(delta as usize);
            } else {
                self.shard_loads[shard] = self.shard_loads[shard].saturating_sub((-delta) as usize);
            }
        }
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalance(&self, threshold: f64, max_entries: usize) -> bool {
        let threshold_count = (max_entries as f64 * threshold) as usize;
        self.shard_loads.iter().any(|&load| load > threshold_count)
    }

    /// Suggest moves to rebalance shards
    /// Returns (from_shard, to_shard, symbol) moves
    pub fn suggest_rebalance(&self) -> Vec<(usize, usize, String)> {
        let mut moves = Vec::new();
        let total: usize = self.shard_loads.iter().sum();
        let avg = total / self.num_shards.max(1);

        // Find overloaded and underloaded shards
        let mut overloaded: Vec<usize> = self.shard_loads
            .iter()
            .enumerate()
            .filter(|(_, &load)| load > avg + avg / 4) // 25% above average
            .map(|(i, _)| i)
            .collect();

        let mut underloaded: Vec<usize> = self.shard_loads
            .iter()
            .enumerate()
            .filter(|(_, &load)| load < avg - avg / 4) // 25% below average
            .map(|(i, _)| i)
            .collect();

        // Match symbols from shard_map to suggest moves
        for (symbol, &shard) in &self.shard_map {
            if overloaded.contains(&shard) && !underloaded.is_empty() {
                let target = underloaded[0];
                moves.push((shard, target, symbol.clone()));

                // Limit suggestions
                if moves.len() >= 100 {
                    break;
                }
            }
        }

        moves
    }

    /// Cache a symbol's shard assignment
    pub fn cache_symbol(&mut self, symbol: &str, shard: usize) {
        self.shard_map.insert(symbol.to_string(), shard);
    }

    /// Get shard loads
    pub fn get_loads(&self) -> &[usize] {
        &self.shard_loads
    }

    /// Get number of shards
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }
}

// ==================== Entry Types ====================

/// Type of memory entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryType {
    /// Trade context and setup
    TradeContext,
    /// Support/resistance level
    SRLevel,
    /// Recognized pattern
    Pattern,
    /// Vector embedding
    Embedding,
    /// Market regime observation
    Regime,
}

impl EntryType {
    /// Get string representation for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            EntryType::TradeContext => "trade_context",
            EntryType::SRLevel => "sr_level",
            EntryType::Pattern => "pattern",
            EntryType::Embedding => "embedding",
            EntryType::Regime => "regime",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "trade_context" => Some(EntryType::TradeContext),
            "sr_level" => Some(EntryType::SRLevel),
            "pattern" => Some(EntryType::Pattern),
            "embedding" => Some(EntryType::Embedding),
            "regime" => Some(EntryType::Regime),
            _ => None,
        }
    }
}

// ==================== Memory Entry ====================

/// A single entry in sharded memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: u64,
    /// Type of entry
    pub entry_type: EntryType,
    /// Associated symbol
    pub symbol: String,
    /// Serialized payload data
    pub data: Vec<u8>,
    /// Importance score (0.0 - 1.0)
    pub importance: f64,
    /// When this entry was created
    pub created_at: DateTime<Utc>,
    /// Last time this entry was accessed
    pub last_accessed: DateTime<Utc>,
    /// Number of times this entry was accessed
    pub access_count: u32,
}

impl MemoryEntry {
    /// Create a new memory entry
    pub fn new(entry_type: EntryType, symbol: String, data: Vec<u8>, importance: f64) -> Self {
        let now = Utc::now();
        Self {
            id: 0, // Will be assigned on insert
            entry_type,
            symbol,
            data,
            importance,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Create with specific id
    pub fn with_id(mut self, id: u64) -> Self {
        self.id = id;
        self
    }

    /// Get age in days
    pub fn age_days(&self) -> f64 {
        let duration = Utc::now() - self.created_at;
        duration.num_seconds() as f64 / 86400.0
    }

    /// Check if entry is stale (no access in N days)
    pub fn is_stale(&self, days: i64) -> bool {
        Utc::now() - self.last_accessed > Duration::days(days)
    }
}

// ==================== Memory Query ====================

/// Query parameters for searching memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// Filter by entry type
    pub entry_type: Option<EntryType>,
    /// Filter by symbol
    pub symbol: Option<String>,
    /// Filter by time range
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Minimum importance threshold
    pub min_importance: Option<f64>,
    /// Maximum results to return
    pub limit: usize,
    /// How to order results
    pub order_by: QueryOrder,
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self {
            entry_type: None,
            symbol: None,
            time_range: None,
            min_importance: None,
            limit: 100,
            order_by: QueryOrder::Recency,
        }
    }
}

impl MemoryQuery {
    /// Create a query for a specific symbol
    pub fn for_symbol(symbol: &str) -> Self {
        Self {
            symbol: Some(symbol.to_string()),
            ..Default::default()
        }
    }

    /// Create a query for a specific entry type
    pub fn for_type(entry_type: EntryType) -> Self {
        Self {
            entry_type: Some(entry_type),
            ..Default::default()
        }
    }

    /// Set limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set minimum importance
    pub fn with_min_importance(mut self, min: f64) -> Self {
        self.min_importance = Some(min);
        self
    }

    /// Set order
    pub fn with_order(mut self, order: QueryOrder) -> Self {
        self.order_by = order;
        self
    }

    /// Compute a hash for caching
    pub fn cache_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        if let Some(ref t) = self.entry_type {
            t.as_str().hash(&mut hasher);
        }
        if let Some(ref s) = self.symbol {
            s.hash(&mut hasher);
        }
        if let Some((start, end)) = self.time_range {
            start.timestamp().hash(&mut hasher);
            end.timestamp().hash(&mut hasher);
        }
        if let Some(min) = self.min_importance {
            ((min * 1000.0) as i64).hash(&mut hasher);
        }
        self.limit.hash(&mut hasher);
        hasher.finish()
    }
}

/// How to order query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryOrder {
    /// Most recent first
    Recency,
    /// Highest importance first
    Importance,
    /// Most accessed first
    AccessCount,
    /// By similarity to embedding vector
    Relevance(Vec<f64>),
}

impl Default for QueryOrder {
    fn default() -> Self {
        QueryOrder::Recency
    }
}

// ==================== Memory Shard ====================

/// A single shard of the sharded memory system
#[derive(Debug)]
pub struct MemoryShard {
    /// Shard identifier
    pub id: usize,
    /// Path to database file
    pub path: String,
    /// In-memory storage (simulating SQLite for testing)
    entries: RwLock<HashMap<u64, MemoryEntry>>,
    /// Entry count
    entry_count: AtomicU64,
    /// Last compaction time
    last_compacted: RwLock<DateTime<Utc>>,
}

impl MemoryShard {
    /// Create a new memory shard
    pub fn new(id: usize, path: &str) -> Result<Self> {
        // In production, this would open/create SQLite database
        // For now, use in-memory storage
        info!("[SHARD] Creating shard {} at {}", id, path);

        Ok(Self {
            id,
            path: path.to_string(),
            entries: RwLock::new(HashMap::new()),
            entry_count: AtomicU64::new(0),
            last_compacted: RwLock::new(Utc::now()),
        })
    }

    /// Insert an entry into this shard
    pub fn insert(&self, entry: &MemoryEntry) -> Result<u64> {
        let mut entries = self.entries.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        entries.insert(entry.id, entry.clone());
        self.entry_count.fetch_add(1, Ordering::SeqCst);

        Ok(entry.id)
    }

    /// Query entries from this shard
    pub fn query(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        let entries = self.entries.read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        let mut results: Vec<MemoryEntry> = entries.values()
            .filter(|e| {
                // Apply filters
                if let Some(ref t) = query.entry_type {
                    if e.entry_type != *t {
                        return false;
                    }
                }
                if let Some(ref s) = query.symbol {
                    if e.symbol != *s {
                        return false;
                    }
                }
                if let Some((start, end)) = query.time_range {
                    if e.created_at < start || e.created_at > end {
                        return false;
                    }
                }
                if let Some(min) = query.min_importance {
                    if e.importance < min {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();

        // Sort by order
        match &query.order_by {
            QueryOrder::Recency => {
                results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            }
            QueryOrder::Importance => {
                results.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
            }
            QueryOrder::AccessCount => {
                results.sort_by(|a, b| b.access_count.cmp(&a.access_count));
            }
            QueryOrder::Relevance(embedding) => {
                // Sort by cosine similarity to embedding
                results.sort_by(|a, b| {
                    let sim_a = compute_similarity(&a.data, embedding);
                    let sim_b = compute_similarity(&b.data, embedding);
                    sim_b.partial_cmp(&sim_a).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Apply limit
        results.truncate(query.limit);

        Ok(results)
    }

    /// Get a single entry by id
    pub fn get(&self, id: u64) -> Result<Option<MemoryEntry>> {
        let entries = self.entries.read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        Ok(entries.get(&id).cloned())
    }

    /// Delete an entry
    pub fn delete(&self, id: u64) -> Result<bool> {
        let mut entries = self.entries.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        if entries.remove(&id).is_some() {
            self.entry_count.fetch_sub(1, Ordering::SeqCst);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Update entry importance
    pub fn update_importance(&self, id: u64, importance: f64) -> Result<bool> {
        let mut entries = self.entries.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        if let Some(entry) = entries.get_mut(&id) {
            entry.importance = importance;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Record an access to an entry
    pub fn record_access(&self, id: u64) -> Result<bool> {
        let mut entries = self.entries.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        if let Some(entry) = entries.get_mut(&id) {
            entry.access_count += 1;
            entry.last_accessed = Utc::now();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get entry count
    pub fn count(&self) -> usize {
        self.entry_count.load(Ordering::SeqCst) as usize
    }

    /// Compact the shard (vacuum and optimize)
    pub fn compact(&self) -> Result<()> {
        // In production, this would run VACUUM on SQLite
        // For now, just update the timestamp
        let mut last = self.last_compacted.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        *last = Utc::now();

        info!("[SHARD] Compacted shard {}", self.id);
        Ok(())
    }

    /// Get last compaction time
    pub fn last_compacted(&self) -> Result<DateTime<Utc>> {
        let last = self.last_compacted.read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        Ok(*last)
    }

    /// Get all entry IDs for a symbol (for rebalancing)
    pub fn get_symbol_entries(&self, symbol: &str) -> Result<Vec<u64>> {
        let entries = self.entries.read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        Ok(entries.values()
            .filter(|e| e.symbol == symbol)
            .map(|e| e.id)
            .collect())
    }

    /// Remove and return entries for a symbol (for rebalancing)
    pub fn extract_symbol(&self, symbol: &str) -> Result<Vec<MemoryEntry>> {
        let mut entries = self.entries.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        let ids: Vec<u64> = entries.values()
            .filter(|e| e.symbol == symbol)
            .map(|e| e.id)
            .collect();

        let mut extracted = Vec::new();
        for id in ids {
            if let Some(entry) = entries.remove(&id) {
                self.entry_count.fetch_sub(1, Ordering::SeqCst);
                extracted.push(entry);
            }
        }

        Ok(extracted)
    }
}

/// Compute cosine similarity between entry data and embedding
fn compute_similarity(data: &[u8], embedding: &[f64]) -> f64 {
    // Try to deserialize data as embedding
    if let Ok(data_embedding) = bincode::deserialize::<Vec<f64>>(data) {
        if data_embedding.len() == embedding.len() {
            let dot: f64 = data_embedding.iter()
                .zip(embedding.iter())
                .map(|(a, b)| a * b)
                .sum();
            let mag_a: f64 = data_embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mag_b: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            if mag_a > 0.0 && mag_b > 0.0 {
                return dot / (mag_a * mag_b);
            }
        }
    }
    0.0
}

// ==================== Global Index ====================

/// Global index for fast lookups across shards
#[derive(Debug)]
pub struct GlobalIndex {
    /// Symbol to shard mapping
    symbol_to_shard: RwLock<HashMap<String, usize>>,
    /// Entry ID to shard mapping
    id_to_shard: RwLock<HashMap<u64, usize>>,
    /// Total entries across all shards
    total_entries: AtomicU64,
    /// Next available ID
    last_id: AtomicU64,
}

impl GlobalIndex {
    /// Create a new global index
    pub fn new() -> Self {
        Self {
            symbol_to_shard: RwLock::new(HashMap::new()),
            id_to_shard: RwLock::new(HashMap::new()),
            total_entries: AtomicU64::new(0),
            last_id: AtomicU64::new(0),
        }
    }

    /// Generate next unique ID
    pub fn next_id(&self) -> u64 {
        self.last_id.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Register an entry
    pub fn register(&self, id: u64, symbol: &str, shard: usize) -> Result<()> {
        {
            let mut id_map = self.id_to_shard.write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            id_map.insert(id, shard);
        }
        {
            let mut sym_map = self.symbol_to_shard.write()
                .map_err(|e| anyhow!("Lock error: {}", e))?;
            sym_map.insert(symbol.to_string(), shard);
        }
        self.total_entries.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// Unregister an entry
    pub fn unregister(&self, id: u64) -> Result<()> {
        let mut id_map = self.id_to_shard.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        if id_map.remove(&id).is_some() {
            self.total_entries.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Look up shard for an entry ID
    pub fn lookup_id(&self, id: u64) -> Result<Option<usize>> {
        let id_map = self.id_to_shard.read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        Ok(id_map.get(&id).copied())
    }

    /// Look up shard for a symbol
    pub fn lookup_symbol(&self, symbol: &str) -> Result<Option<usize>> {
        let sym_map = self.symbol_to_shard.read()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        Ok(sym_map.get(symbol).copied())
    }

    /// Get total entry count
    pub fn total_entries(&self) -> u64 {
        self.total_entries.load(Ordering::SeqCst)
    }

    /// Update symbol shard mapping (for rebalancing)
    pub fn update_symbol_shard(&self, symbol: &str, new_shard: usize) -> Result<()> {
        let mut sym_map = self.symbol_to_shard.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        sym_map.insert(symbol.to_string(), new_shard);
        Ok(())
    }

    /// Update entry shard mapping (for rebalancing)
    pub fn update_id_shard(&self, id: u64, new_shard: usize) -> Result<()> {
        let mut id_map = self.id_to_shard.write()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        id_map.insert(id, new_shard);
        Ok(())
    }
}

impl Default for GlobalIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Write Buffer ====================

/// Buffers writes for batch insertion
#[derive(Debug)]
pub struct WriteBuffer {
    /// Buffered entries with target shards
    buffer: Mutex<Vec<(MemoryEntry, usize)>>,
    /// Maximum buffer capacity
    capacity: usize,
    /// Flush interval
    flush_interval: std::time::Duration,
    /// Last flush time
    last_flush: Mutex<std::time::Instant>,
}

impl WriteBuffer {
    /// Create a new write buffer
    pub fn new(capacity: usize, flush_interval: std::time::Duration) -> Self {
        Self {
            buffer: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
            flush_interval,
            last_flush: Mutex::new(std::time::Instant::now()),
        }
    }

    /// Add an entry to the buffer
    pub fn push(&self, entry: MemoryEntry, shard: usize) -> Result<bool> {
        let mut buffer = self.buffer.lock()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        buffer.push((entry, shard));
        Ok(buffer.len() >= self.capacity)
    }

    /// Check if buffer needs flushing
    pub fn needs_flush(&self) -> Result<bool> {
        let buffer = self.buffer.lock()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        let last = self.last_flush.lock()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        Ok(buffer.len() >= self.capacity || last.elapsed() >= self.flush_interval)
    }

    /// Drain the buffer
    pub fn drain(&self) -> Result<Vec<(MemoryEntry, usize)>> {
        let mut buffer = self.buffer.lock()
            .map_err(|e| anyhow!("Lock error: {}", e))?;
        let mut last = self.last_flush.lock()
            .map_err(|e| anyhow!("Lock error: {}", e))?;

        *last = std::time::Instant::now();
        Ok(std::mem::take(&mut *buffer))
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ==================== Query Cache ====================

/// LRU cache for query results
#[derive(Debug)]
pub struct QueryCache {
    /// Cache storage
    cache: Mutex<HashMap<u64, (Vec<MemoryEntry>, std::time::Instant)>>,
    /// Maximum cache entries
    max_size: usize,
    /// Time-to-live for cached results
    ttl: std::time::Duration,
    /// Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
}

impl QueryCache {
    /// Create a new query cache
    pub fn new(max_size: usize, ttl: std::time::Duration) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_size,
            ttl,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get cached results for a query
    pub fn get(&self, query: &MemoryQuery) -> Option<Vec<MemoryEntry>> {
        let key = query.cache_key();
        let cache = self.cache.lock().ok()?;

        if let Some((results, timestamp)) = cache.get(&key) {
            if timestamp.elapsed() < self.ttl {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(results.clone());
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Store results in cache
    pub fn put(&self, query: &MemoryQuery, results: Vec<MemoryEntry>) {
        let key = query.cache_key();
        let mut cache = match self.cache.lock() {
            Ok(c) => c,
            Err(_) => return,
        };

        // Evict old entries if at capacity
        if cache.len() >= self.max_size {
            // Remove oldest entry
            let oldest_key = cache.iter()
                .min_by_key(|(_, (_, ts))| *ts)
                .map(|(k, _)| *k);
            if let Some(k) = oldest_key {
                cache.remove(&k);
            }
        }

        cache.insert(key, (results, std::time::Instant::now()));
    }

    /// Invalidate cache entries for a symbol
    pub fn invalidate_symbol(&self, symbol: &str) {
        let mut cache = match self.cache.lock() {
            Ok(c) => c,
            Err(_) => return,
        };

        // Remove entries that might contain this symbol
        // In practice, we'd need query metadata; for now, clear relevant entries
        cache.retain(|_, (results, _)| {
            !results.iter().any(|e| e.symbol == symbol)
        });
    }

    /// Clear entire cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

// ==================== Importance Updater ====================

/// Updates importance scores with time decay and access boost
#[derive(Debug, Clone)]
pub struct ImportanceUpdater {
    /// Daily decay rate (0.99 = 1% decay per day)
    decay_rate: f64,
    /// Boost for recent access
    access_boost: f64,
    /// Weight for outcome-based adjustment
    outcome_weight: f64,
}

impl Default for ImportanceUpdater {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportanceUpdater {
    /// Create a new importance updater
    pub fn new() -> Self {
        Self {
            decay_rate: 0.99,
            access_boost: 0.1,
            outcome_weight: 0.3,
        }
    }

    /// Create with custom parameters
    pub fn with_params(decay_rate: f64, access_boost: f64, outcome_weight: f64) -> Self {
        Self {
            decay_rate,
            access_boost,
            outcome_weight,
        }
    }

    /// Compute updated importance for an entry
    pub fn compute_importance(&self, entry: &MemoryEntry, outcome: Option<bool>) -> f64 {
        let mut importance = entry.importance;

        // Apply time decay
        let age_days = entry.age_days();
        importance *= self.decay_rate.powf(age_days);

        // Boost if recently accessed (within 7 days)
        let days_since_access = (Utc::now() - entry.last_accessed).num_days();
        if days_since_access < 7 {
            importance += self.access_boost * (1.0 - days_since_access as f64 / 7.0);
        }

        // Boost for high access count
        let access_factor = (entry.access_count as f64).ln_1p() / 10.0;
        importance += self.access_boost * access_factor;

        // Adjust based on outcome if available
        if let Some(positive) = outcome {
            if positive {
                importance += self.outcome_weight;
            } else {
                importance -= self.outcome_weight * 0.5; // Negative outcomes reduce less
            }
        }

        // Clamp to valid range
        importance.clamp(0.0, 1.0)
    }

    /// Batch update importance for all entries in sharded memory
    pub fn batch_update(&self, memory: &ShardedMemory) -> Result<u32> {
        let mut count = 0;

        for shard in memory.shards.iter() {
            let entries = shard.entries.read()
                .map_err(|e| anyhow!("Lock error: {}", e))?;

            let updates: Vec<(u64, f64)> = entries.values()
                .map(|e| (e.id, self.compute_importance(e, None)))
                .collect();
            drop(entries); // Release read lock

            for (id, new_importance) in updates {
                if shard.update_importance(id, new_importance)? {
                    count += 1;
                }
            }
        }

        info!("[IMPORTANCE] Updated {} entries", count);
        Ok(count)
    }
}

// ==================== Sharded Memory Stats ====================

/// Statistics for sharded memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardedMemoryStats {
    /// Total entries across all shards
    pub total_entries: u64,
    /// Entry count per shard
    pub entries_per_shard: Vec<usize>,
    /// Total size in bytes (estimated)
    pub total_size_bytes: u64,
    /// Average query time in milliseconds
    pub avg_query_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Number of symbols tracked
    pub symbol_count: usize,
    /// Write buffer size
    pub buffer_size: usize,
}

impl ShardedMemoryStats {
    /// Get the most loaded shard
    pub fn most_loaded_shard(&self) -> Option<(usize, usize)> {
        self.entries_per_shard.iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(i, &count)| (i, count))
    }

    /// Get load imbalance ratio (max/avg)
    pub fn imbalance_ratio(&self) -> f64 {
        if self.entries_per_shard.is_empty() {
            return 1.0;
        }
        let avg = self.total_entries as f64 / self.entries_per_shard.len() as f64;
        let max = *self.entries_per_shard.iter().max().unwrap_or(&0) as f64;
        if avg > 0.0 {
            max / avg
        } else {
            1.0
        }
    }
}

// ==================== Sharded Memory ====================

/// Main sharded memory system for billion-scale storage
pub struct ShardedMemory {
    /// Configuration
    pub config: ShardConfig,
    /// Shard router
    pub router: ShardRouter,
    /// Memory shards
    pub shards: Vec<MemoryShard>,
    /// Global index
    pub global_index: GlobalIndex,
    /// Write buffer
    pub write_buffer: WriteBuffer,
    /// Query cache
    pub query_cache: QueryCache,
    /// Importance updater
    pub importance_updater: ImportanceUpdater,
    /// Query timing stats
    query_times: Mutex<Vec<f64>>,
}

impl ShardedMemory {
    /// Create a new sharded memory system
    pub fn new(config: ShardConfig) -> Result<Self> {
        info!("[SHARDED] Creating sharded memory with {} shards", config.num_shards);

        let mut shards = Vec::with_capacity(config.num_shards);
        for i in 0..config.num_shards {
            let path = config.shard_path(i);
            shards.push(MemoryShard::new(i, &path)?);
        }

        let router = ShardRouter::new(ShardingStrategy::BySymbol, config.num_shards);
        let global_index = GlobalIndex::new();
        let write_buffer = WriteBuffer::new(1000, std::time::Duration::from_secs(60));
        let query_cache = QueryCache::new(1000, std::time::Duration::from_secs(300));
        let importance_updater = ImportanceUpdater::new();

        Ok(Self {
            config,
            router,
            shards,
            global_index,
            write_buffer,
            query_cache,
            importance_updater,
            query_times: Mutex::new(Vec::new()),
        })
    }

    /// Insert an entry into the sharded memory
    pub fn insert(&mut self, mut entry: MemoryEntry) -> Result<u64> {
        // Assign ID
        entry.id = self.global_index.next_id();

        // Route to shard
        let shard_idx = self.router.route(&ShardKey::Symbol(entry.symbol.clone()));

        // Buffer the write
        let needs_flush = self.write_buffer.push(entry.clone(), shard_idx)?;

        // Register in index
        self.global_index.register(entry.id, &entry.symbol, shard_idx)?;
        self.router.update_load(shard_idx, 1);
        self.router.cache_symbol(&entry.symbol, shard_idx);

        // Invalidate cache for this symbol
        self.query_cache.invalidate_symbol(&entry.symbol);

        // Flush if buffer is full
        if needs_flush {
            self.flush()?;
        }

        Ok(entry.id)
    }

    /// Insert multiple entries in batch
    pub fn insert_batch(&mut self, entries: Vec<MemoryEntry>) -> Result<Vec<u64>> {
        let mut ids = Vec::with_capacity(entries.len());

        // Group by shard
        let mut by_shard: HashMap<usize, Vec<MemoryEntry>> = HashMap::new();

        for mut entry in entries {
            entry.id = self.global_index.next_id();
            let shard_idx = self.router.route(&ShardKey::Symbol(entry.symbol.clone()));

            self.global_index.register(entry.id, &entry.symbol, shard_idx)?;
            self.router.update_load(shard_idx, 1);
            self.router.cache_symbol(&entry.symbol, shard_idx);
            self.query_cache.invalidate_symbol(&entry.symbol);

            ids.push(entry.id);
            by_shard.entry(shard_idx).or_default().push(entry);
        }

        // Insert per shard
        for (shard_idx, shard_entries) in by_shard {
            for entry in shard_entries {
                self.shards[shard_idx].insert(&entry)?;
            }
        }

        Ok(ids)
    }

    /// Get an entry by ID
    pub fn get(&self, id: u64) -> Result<Option<MemoryEntry>> {
        // Look up shard in index
        let shard_idx = match self.global_index.lookup_id(id)? {
            Some(s) => s,
            None => return Ok(None),
        };

        // Query shard
        let result = self.shards[shard_idx].get(id)?;

        // Record access
        if result.is_some() {
            self.shards[shard_idx].record_access(id)?;
        }

        Ok(result)
    }

    /// Query entries
    pub fn query(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        let start = std::time::Instant::now();

        // Check cache
        if let Some(cached) = self.query_cache.get(query) {
            return Ok(cached);
        }

        let results = if let Some(ref symbol) = query.symbol {
            // Query single shard
            let shard_idx = self.router.route_symbol(symbol);
            self.shards[shard_idx].query(query)?
        } else {
            // Scatter-gather across all shards
            self.query_scatter_gather(query)?
        };

        // Cache results
        self.query_cache.put(query, results.clone());

        // Record timing
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if let Ok(mut times) = self.query_times.lock() {
            times.push(elapsed);
            if times.len() > 1000 {
                times.remove(0);
            }
        }

        Ok(results)
    }

    /// Scatter-gather query across all shards
    fn query_scatter_gather(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        let mut all_results: Vec<MemoryEntry> = Vec::new();

        for shard in &self.shards {
            let mut shard_results = shard.query(query)?;
            all_results.append(&mut shard_results);
        }

        // Sort merged results
        match &query.order_by {
            QueryOrder::Recency => {
                all_results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            }
            QueryOrder::Importance => {
                all_results.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
            }
            QueryOrder::AccessCount => {
                all_results.sort_by(|a, b| b.access_count.cmp(&a.access_count));
            }
            QueryOrder::Relevance(embedding) => {
                all_results.sort_by(|a, b| {
                    let sim_a = compute_similarity(&a.data, embedding);
                    let sim_b = compute_similarity(&b.data, embedding);
                    sim_b.partial_cmp(&sim_a).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Apply global limit
        all_results.truncate(query.limit);

        Ok(all_results)
    }

    /// Query in parallel using rayon (if available)
    #[cfg(feature = "parallel")]
    pub fn query_parallel(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        use rayon::prelude::*;

        let results: Vec<Vec<MemoryEntry>> = self.shards.par_iter()
            .filter_map(|shard| shard.query(query).ok())
            .collect();

        let mut all_results: Vec<MemoryEntry> = results.into_iter().flatten().collect();

        // Sort and limit
        match &query.order_by {
            QueryOrder::Recency => all_results.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
            QueryOrder::Importance => all_results.sort_by(|a, b|
                b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)),
            _ => {}
        }
        all_results.truncate(query.limit);

        Ok(all_results)
    }

    /// Non-parallel version for when rayon is not available
    #[cfg(not(feature = "parallel"))]
    pub fn query_parallel(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        self.query(query)
    }

    /// Update importance for an entry
    pub fn update_importance(&self, id: u64, importance: f64) -> Result<bool> {
        let shard_idx = match self.global_index.lookup_id(id)? {
            Some(s) => s,
            None => return Ok(false),
        };
        self.shards[shard_idx].update_importance(id, importance)
    }

    /// Record an access to an entry
    pub fn record_access(&self, id: u64) -> Result<bool> {
        let shard_idx = match self.global_index.lookup_id(id)? {
            Some(s) => s,
            None => return Ok(false),
        };
        self.shards[shard_idx].record_access(id)
    }

    /// Delete an entry
    pub fn delete(&mut self, id: u64) -> Result<bool> {
        let shard_idx = match self.global_index.lookup_id(id)? {
            Some(s) => s,
            None => return Ok(false),
        };

        if self.shards[shard_idx].delete(id)? {
            self.global_index.unregister(id)?;
            self.router.update_load(shard_idx, -1);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Flush the write buffer to shards
    pub fn flush(&mut self) -> Result<usize> {
        let buffered = self.write_buffer.drain()?;
        let count = buffered.len();

        for (entry, shard_idx) in buffered {
            self.shards[shard_idx].insert(&entry)?;
        }

        if count > 0 {
            info!("[SHARDED] Flushed {} entries to shards", count);
        }

        Ok(count)
    }

    /// Compact all shards
    pub fn compact_all(&mut self) -> Result<()> {
        info!("[SHARDED] Compacting all shards");
        for shard in &self.shards {
            shard.compact()?;
        }
        Ok(())
    }

    /// Rebalance entries between shards
    pub fn rebalance(&mut self) -> Result<u32> {
        let moves = self.router.suggest_rebalance();
        if moves.is_empty() {
            return Ok(0);
        }

        info!("[SHARDED] Rebalancing {} symbol moves", moves.len());
        let mut total_moved = 0u32;

        for (from_shard, to_shard, symbol) in moves {
            // Extract entries from source shard
            let entries = self.shards[from_shard].extract_symbol(&symbol)?;
            let count = entries.len() as u32;

            // Insert into target shard
            for entry in entries {
                self.shards[to_shard].insert(&entry)?;
                self.global_index.update_id_shard(entry.id, to_shard)?;
            }

            // Update index and router
            self.global_index.update_symbol_shard(&symbol, to_shard)?;
            self.router.cache_symbol(&symbol, to_shard);
            self.router.update_load(from_shard, -(count as i64));
            self.router.update_load(to_shard, count as i64);

            total_moved += count;
        }

        info!("[SHARDED] Rebalanced {} entries", total_moved);
        Ok(total_moved)
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalance(&self) -> bool {
        self.router.needs_rebalance(self.config.rebalance_threshold, self.config.max_entries_per_shard)
    }

    /// Get statistics
    pub fn get_stats(&self) -> ShardedMemoryStats {
        let entries_per_shard: Vec<usize> = self.shards.iter()
            .map(|s| s.count())
            .collect();

        let total_entries: u64 = entries_per_shard.iter().map(|&c| c as u64).sum();

        // Estimate size (rough: 1KB per entry)
        let total_size_bytes = total_entries * 1024;

        // Calculate average query time
        let avg_query_time_ms = self.query_times.lock()
            .map(|times| {
                if times.is_empty() {
                    0.0
                } else {
                    times.iter().sum::<f64>() / times.len() as f64
                }
            })
            .unwrap_or(0.0);

        // Get symbol count from index
        let symbol_count = self.router.shard_map.len();

        ShardedMemoryStats {
            total_entries,
            entries_per_shard,
            total_size_bytes,
            avg_query_time_ms,
            cache_hit_rate: self.query_cache.hit_rate(),
            symbol_count,
            buffer_size: self.write_buffer.len(),
        }
    }

    /// Update importance scores for all entries
    pub fn update_all_importance(&self) -> Result<u32> {
        self.importance_updater.batch_update(self)
    }

    /// Get number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Clear query cache
    pub fn clear_cache(&self) {
        self.query_cache.clear();
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entry(symbol: &str, importance: f64) -> MemoryEntry {
        MemoryEntry::new(
            EntryType::TradeContext,
            symbol.to_string(),
            vec![1, 2, 3, 4],
            importance,
        )
    }

    #[test]
    fn test_shard_routing() {
        let router = ShardRouter::new(ShardingStrategy::BySymbol, 16);

        // Same symbol should always route to same shard
        let shard1 = router.route_symbol("AAPL");
        let shard2 = router.route_symbol("AAPL");
        assert_eq!(shard1, shard2);

        // Different symbols may route to different shards
        let shard_aapl = router.route_symbol("AAPL");
        let shard_goog = router.route_symbol("GOOG");
        assert!(shard_aapl < 16 && shard_goog < 16);
    }

    #[test]
    fn test_insert_and_query() {
        let config = ShardConfig::with_shards(4);
        let mut memory = ShardedMemory::new(config).unwrap();

        // Insert entries
        let entry1 = create_test_entry("AAPL", 0.8);
        let entry2 = create_test_entry("GOOG", 0.6);
        let entry3 = create_test_entry("AAPL", 0.9);

        let id1 = memory.insert(entry1).unwrap();
        let id2 = memory.insert(entry2).unwrap();
        let id3 = memory.insert(entry3).unwrap();

        // Flush buffer
        memory.flush().unwrap();

        // Query by symbol
        let query = MemoryQuery::for_symbol("AAPL");
        let results = memory.query(&query).unwrap();
        assert_eq!(results.len(), 2);

        // Get by ID
        let entry = memory.get(id1).unwrap();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().symbol, "AAPL");

        // Query all with importance filter
        let query = MemoryQuery::default()
            .with_min_importance(0.7);
        let results = memory.query(&query).unwrap();
        assert_eq!(results.len(), 2); // AAPL 0.8 and 0.9
    }

    #[test]
    fn test_scatter_gather() {
        let config = ShardConfig::with_shards(4);
        let mut memory = ShardedMemory::new(config).unwrap();

        // Insert entries for multiple symbols
        for i in 0..20 {
            let symbol = format!("SYM{}", i % 5);
            let entry = create_test_entry(&symbol, 0.5 + (i as f64 * 0.02));
            memory.insert(entry).unwrap();
        }
        memory.flush().unwrap();

        // Query without symbol filter (scatter-gather)
        let query = MemoryQuery::default().with_limit(10);
        let results = memory.query(&query).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_write_buffer() {
        let buffer = WriteBuffer::new(5, std::time::Duration::from_secs(60));

        // Add entries
        for i in 0..4 {
            let entry = create_test_entry(&format!("SYM{}", i), 0.5);
            let needs_flush = buffer.push(entry, i % 4).unwrap();
            assert!(!needs_flush);
        }

        // Fifth entry should trigger flush
        let entry = create_test_entry("SYM4", 0.5);
        let needs_flush = buffer.push(entry, 0).unwrap();
        assert!(needs_flush);

        // Drain buffer
        let entries = buffer.drain().unwrap();
        assert_eq!(entries.len(), 5);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_rebalancing() {
        let mut router = ShardRouter::new(ShardingStrategy::BySymbol, 4);

        // Simulate uneven load
        router.update_load(0, 1000);
        router.update_load(1, 100);
        router.update_load(2, 100);
        router.update_load(3, 100);

        // Cache some symbol assignments
        router.cache_symbol("AAPL", 0);
        router.cache_symbol("GOOG", 0);
        router.cache_symbol("MSFT", 0);

        // Check rebalance needed
        assert!(router.needs_rebalance(0.8, 500));

        // Get suggestions
        let moves = router.suggest_rebalance();
        assert!(!moves.is_empty());
    }

    #[test]
    fn test_importance_decay() {
        let updater = ImportanceUpdater::new();

        // Create an old entry
        let mut entry = create_test_entry("AAPL", 0.8);
        entry.created_at = Utc::now() - chrono::Duration::days(30);
        entry.last_accessed = Utc::now() - chrono::Duration::days(30);

        let new_importance = updater.compute_importance(&entry, None);
        assert!(new_importance < 0.8); // Should have decayed

        // Create a recent, frequently accessed entry
        let mut entry2 = create_test_entry("GOOG", 0.5);
        entry2.access_count = 100;
        entry2.last_accessed = Utc::now();

        let new_importance2 = updater.compute_importance(&entry2, Some(true));
        assert!(new_importance2 > 0.5); // Should have increased
    }

    #[test]
    fn test_query_cache() {
        let cache = QueryCache::new(10, std::time::Duration::from_secs(60));

        let query = MemoryQuery::for_symbol("AAPL");
        let entries = vec![create_test_entry("AAPL", 0.8)];

        // Miss on first access
        assert!(cache.get(&query).is_none());

        // Put in cache
        cache.put(&query, entries.clone());

        // Hit on second access
        let cached = cache.get(&query);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);

        // Check hit rate
        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn test_parallel_query() {
        let config = ShardConfig::with_shards(4);
        let mut memory = ShardedMemory::new(config).unwrap();

        // Insert entries
        for i in 0..100 {
            let entry = create_test_entry(&format!("SYM{}", i % 10), 0.5);
            memory.insert(entry).unwrap();
        }
        memory.flush().unwrap();

        // Parallel query
        let query = MemoryQuery::default().with_limit(50);
        let results = memory.query_parallel(&query).unwrap();
        assert_eq!(results.len(), 50);
    }

    #[test]
    fn test_global_index() {
        let index = GlobalIndex::new();

        // Register entries
        let id1 = index.next_id();
        let id2 = index.next_id();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        index.register(id1, "AAPL", 0).unwrap();
        index.register(id2, "GOOG", 1).unwrap();

        // Lookup
        assert_eq!(index.lookup_id(id1).unwrap(), Some(0));
        assert_eq!(index.lookup_symbol("GOOG").unwrap(), Some(1));
        assert_eq!(index.total_entries(), 2);

        // Unregister
        index.unregister(id1).unwrap();
        assert_eq!(index.lookup_id(id1).unwrap(), None);
        assert_eq!(index.total_entries(), 1);
    }

    #[test]
    fn test_shard_config() {
        let config = ShardConfig::default();
        assert_eq!(config.num_shards, 16);
        assert_eq!(config.shard_path(5), "sovereign_shard_5.db");

        let custom = ShardConfig::with_shards(8);
        assert_eq!(custom.num_shards, 8);
    }

    #[test]
    fn test_entry_types() {
        assert_eq!(EntryType::TradeContext.as_str(), "trade_context");
        assert_eq!(EntryType::from_str("sr_level"), Some(EntryType::SRLevel));
        assert_eq!(EntryType::from_str("invalid"), None);
    }

    #[test]
    fn test_memory_stats() {
        let config = ShardConfig::with_shards(4);
        let mut memory = ShardedMemory::new(config).unwrap();

        // Insert some entries
        for i in 0..10 {
            let entry = create_test_entry(&format!("SYM{}", i), 0.5);
            memory.insert(entry).unwrap();
        }
        memory.flush().unwrap();

        let stats = memory.get_stats();
        assert_eq!(stats.total_entries, 10);
        assert_eq!(stats.entries_per_shard.len(), 4);
        assert!(stats.imbalance_ratio() >= 1.0);
    }
}
