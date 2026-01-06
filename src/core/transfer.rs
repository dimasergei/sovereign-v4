//! Cross-Symbol Knowledge Transfer
//!
//! Enables sharing of learned patterns between related assets. Assets are
//! grouped into clusters (e.g., metals, indices, energy) and calibrator
//! weights learned on one asset can be transferred to bootstrap new assets
//! in the same cluster.
//!
//! This accelerates learning for new symbols by leveraging patterns that
//! have already been discovered on related instruments.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::info;

/// Number of features in the calibrator model
const NUM_FEATURES: usize = 6;

/// Minimum trades in cluster before allowing transfer
const MIN_CLUSTER_TRADES_FOR_PRIOR: u32 = 20;

/// Minimum trades for source to be eligible for transfer
const MIN_SOURCE_TRADES_FOR_TRANSFER: u32 = 30;

/// Minimum win rate for transfer eligibility
const MIN_WIN_RATE_FOR_TRANSFER: f64 = 0.50;

/// Asset clusters for grouping related instruments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetCluster {
    /// Precious metals: gold, silver
    Metals,
    /// Stock indices: S&P 500, NASDAQ, etc.
    Indices,
    /// Energy commodities: oil, natural gas
    Energy,
    /// Foreign exchange pairs
    Forex,
    /// Cryptocurrencies
    Crypto,
    /// Individual stocks (no cluster transfer)
    Singleton,
}

impl AssetCluster {
    /// Get human-readable name for the cluster
    pub fn name(&self) -> &'static str {
        match self {
            AssetCluster::Metals => "Metals",
            AssetCluster::Indices => "Indices",
            AssetCluster::Energy => "Energy",
            AssetCluster::Forex => "Forex",
            AssetCluster::Crypto => "Crypto",
            AssetCluster::Singleton => "Singleton",
        }
    }
}

/// Determine the asset cluster for a given symbol
pub fn get_cluster(symbol: &str) -> AssetCluster {
    let symbol_upper = symbol.to_uppercase();

    // Metals - check first (specific symbols)
    if matches!(
        symbol_upper.as_str(),
        "XAUUSD" | "XAGUSD" | "GLD" | "SLV" | "GOLD" | "SILVER" | "GC" | "SI"
    ) {
        return AssetCluster::Metals;
    }

    // Indices - check before forex (specific symbols)
    if matches!(
        symbol_upper.as_str(),
        "SPY" | "QQQ" | "DIA" | "IWM" | "ES" | "NQ" | "YM" | "RTY" |
        "UK100" | "US30" | "US500" | "USTEC" | "DE40" | "JP225" |
        "SPX" | "NDX" | "VIX" | "UVXY" | "SVXY"
    ) {
        return AssetCluster::Indices;
    }

    // Energy - check before forex (specific symbols)
    if matches!(
        symbol_upper.as_str(),
        "USO" | "CL" | "NG" | "USOIL" | "WTICOUSD" | "BRENT" | "XLE" |
        "UCO" | "SCO" | "UNG" | "BOIL" | "KOLD"
    ) {
        return AssetCluster::Energy;
    }

    // Crypto - check BEFORE forex since many crypto symbols end with USD
    // Explicit crypto symbols
    if matches!(
        symbol_upper.as_str(),
        "BTCUSD" | "ETHUSD" | "BTC" | "ETH" | "BTCUSDT" | "ETHUSDT" |
        "SOLUSD" | "ADAUSD" | "DOGUSD" | "XRPUSD" | "DOGEUSD" |
        "AVAXUSD" | "LINKUSD" | "MATICUSD" | "LTCUSD" | "SOL" | "ADA" |
        "XRP" | "DOGE" | "AVAX" | "LINK" | "MATIC" | "LTC" | "DOT" | "UNI"
    ) || symbol_upper.contains("BTC") || symbol_upper.contains("ETH") {
        return AssetCluster::Crypto;
    }

    // Forex (major and minor pairs) - only traditional forex pairs
    // Check for standard 6-character forex pairs with fiat currencies
    if symbol_upper.len() == 6 {
        let first3 = &symbol_upper[0..3];
        let last3 = &symbol_upper[3..6];

        // List of fiat currency codes
        let fiat_codes = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK"];

        // Both parts must be fiat currencies for it to be forex
        if fiat_codes.contains(&first3) && fiat_codes.contains(&last3) {
            return AssetCluster::Forex;
        }
    }

    // Default to singleton (no cluster transfer)
    AssetCluster::Singleton
}

/// Statistics for a single cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    /// Aggregated weights from all symbols in this cluster
    pub weights: [f64; NUM_FEATURES],
    /// Total trades recorded across cluster
    pub trade_count: u32,
    /// Total wins across cluster
    pub win_count: u32,
}

impl Default for ClusterStats {
    fn default() -> Self {
        Self {
            weights: [0.3, 0.2, 0.1, 0.1, 0.1, 0.2], // Default calibrator weights
            trade_count: 0,
            win_count: 0,
        }
    }
}

impl ClusterStats {
    /// Calculate win rate
    pub fn win_rate(&self) -> f64 {
        if self.trade_count == 0 {
            0.5 // Neutral if no data
        } else {
            self.win_count as f64 / self.trade_count as f64
        }
    }
}

/// Manager for cross-symbol knowledge transfer
///
/// Tracks learned patterns per asset cluster and enables bootstrapping
/// new symbols with knowledge from related instruments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferManager {
    /// Statistics per cluster
    cluster_stats: HashMap<AssetCluster, ClusterStats>,
    /// Per-symbol trade counts (for tracking individual symbol maturity)
    symbol_trades: HashMap<String, u32>,
}

impl Default for TransferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferManager {
    /// Create a new transfer manager
    pub fn new() -> Self {
        Self {
            cluster_stats: HashMap::new(),
            symbol_trades: HashMap::new(),
        }
    }

    /// Update cluster knowledge after a trade outcome
    ///
    /// # Arguments
    /// * `symbol` - The symbol that was traded
    /// * `weights` - Current calibrator weights for this symbol
    /// * `won` - Whether the trade was a winner
    pub fn update_cluster(&mut self, symbol: &str, weights: &[f64; NUM_FEATURES], won: bool) {
        let cluster = get_cluster(symbol);

        // Don't update singleton cluster (no transfer benefit)
        if cluster == AssetCluster::Singleton {
            // Still track symbol trades
            *self.symbol_trades.entry(symbol.to_string()).or_insert(0) += 1;
            return;
        }

        // Get or create cluster stats
        let stats = self.cluster_stats.entry(cluster).or_default();

        // Update trade/win counts
        stats.trade_count += 1;
        if won {
            stats.win_count += 1;
        }

        // Running average of weights: 0.9 * old + 0.1 * new
        const ALPHA: f64 = 0.1;
        for i in 0..NUM_FEATURES {
            stats.weights[i] = (1.0 - ALPHA) * stats.weights[i] + ALPHA * weights[i];
        }

        // Track individual symbol trades
        *self.symbol_trades.entry(symbol.to_string()).or_insert(0) += 1;

        info!(
            "[TRANSFER] {} updated {} cluster: {} trades, {:.1}% win rate",
            symbol,
            cluster.name(),
            stats.trade_count,
            stats.win_rate() * 100.0
        );
    }

    /// Get cluster prior weights for initializing a new symbol
    ///
    /// Returns the cluster's aggregated weights if the cluster has
    /// sufficient data (>= MIN_CLUSTER_TRADES_FOR_PRIOR trades).
    pub fn get_cluster_prior(&self, symbol: &str) -> Option<[f64; NUM_FEATURES]> {
        let cluster = get_cluster(symbol);

        // No transfer for singleton symbols
        if cluster == AssetCluster::Singleton {
            return None;
        }

        self.cluster_stats.get(&cluster).and_then(|stats| {
            if stats.trade_count >= MIN_CLUSTER_TRADES_FOR_PRIOR {
                Some(stats.weights)
            } else {
                None
            }
        })
    }

    /// Get transfer confidence (cluster win rate)
    ///
    /// Returns the cluster's win rate if available, or 0.5 (neutral) if
    /// insufficient data.
    pub fn get_transfer_confidence(&self, symbol: &str) -> f64 {
        let cluster = get_cluster(symbol);

        if cluster == AssetCluster::Singleton {
            return 0.5;
        }

        self.cluster_stats
            .get(&cluster)
            .map(|stats| {
                if stats.trade_count >= MIN_CLUSTER_TRADES_FOR_PRIOR {
                    stats.win_rate()
                } else {
                    0.5
                }
            })
            .unwrap_or(0.5)
    }

    /// Check if knowledge should be transferred from source to target
    ///
    /// Transfer is allowed if:
    /// 1. Both symbols are in the same cluster
    /// 2. Source has >= MIN_SOURCE_TRADES_FOR_TRANSFER trades
    /// 3. Cluster win rate > MIN_WIN_RATE_FOR_TRANSFER
    pub fn should_transfer(&self, source: &str, target: &str) -> bool {
        let source_cluster = get_cluster(source);
        let target_cluster = get_cluster(target);

        // Must be same cluster
        if source_cluster != target_cluster {
            return false;
        }

        // No transfer for singletons
        if source_cluster == AssetCluster::Singleton {
            return false;
        }

        // Check source has enough trades
        let source_trades = self.symbol_trades.get(source).copied().unwrap_or(0);
        if source_trades < MIN_SOURCE_TRADES_FOR_TRANSFER {
            return false;
        }

        // Check cluster win rate
        if let Some(stats) = self.cluster_stats.get(&source_cluster) {
            stats.win_rate() > MIN_WIN_RATE_FOR_TRANSFER
        } else {
            false
        }
    }

    /// Get trade count for a specific symbol
    pub fn symbol_trade_count(&self, symbol: &str) -> u32 {
        self.symbol_trades.get(symbol).copied().unwrap_or(0)
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self, cluster: AssetCluster) -> Option<&ClusterStats> {
        self.cluster_stats.get(&cluster)
    }

    /// Get all cluster statistics
    pub fn all_cluster_stats(&self) -> &HashMap<AssetCluster, ClusterStats> {
        &self.cluster_stats
    }

    /// Save transfer manager to JSON file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        info!("[TRANSFER] Saved state to {}", path);
        Ok(())
    }

    /// Load transfer manager from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let manager: Self = serde_json::from_str(&json)?;
        Ok(manager)
    }

    /// Load from file or create new if file doesn't exist
    pub fn load_or_new<P: AsRef<Path>>(path: P) -> Self {
        match Self::load(&path) {
            Ok(manager) => {
                info!(
                    "[TRANSFER] Loaded state: {} clusters, {} symbols tracked",
                    manager.cluster_stats.len(),
                    manager.symbol_trades.len()
                );
                manager
            }
            Err(_) => {
                info!("[TRANSFER] Starting fresh (no saved state found)");
                Self::new()
            }
        }
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        let mut parts = vec![];
        for (cluster, stats) in &self.cluster_stats {
            if stats.trade_count > 0 {
                parts.push(format!(
                    "{}:{} ({:.0}%)",
                    cluster.name(),
                    stats.trade_count,
                    stats.win_rate() * 100.0
                ));
            }
        }
        if parts.is_empty() {
            "No cluster data yet".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// Get trade counts per cluster for monitoring
    pub fn cluster_trade_counts(&self) -> HashMap<String, u32> {
        self.cluster_stats
            .iter()
            .map(|(cluster, stats)| (cluster.name().to_string(), stats.trade_count))
            .collect()
    }

    /// Get total number of transfers applied (cluster initializations)
    pub fn transfers_applied(&self) -> u32 {
        self.cluster_stats.values()
            .filter(|s| s.trade_count > 0)
            .count() as u32
    }

    /// Get overall success rate across all clusters
    pub fn success_rate(&self) -> f64 {
        let total_trades: u32 = self.cluster_stats.values().map(|s| s.trade_count).sum();
        let total_wins: u32 = self.cluster_stats.values().map(|s| s.win_count).sum();
        if total_trades > 0 {
            total_wins as f64 / total_trades as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cluster_metals() {
        assert_eq!(get_cluster("XAUUSD"), AssetCluster::Metals);
        assert_eq!(get_cluster("XAGUSD"), AssetCluster::Metals);
        assert_eq!(get_cluster("GLD"), AssetCluster::Metals);
        assert_eq!(get_cluster("SLV"), AssetCluster::Metals);
    }

    #[test]
    fn test_get_cluster_indices() {
        assert_eq!(get_cluster("SPY"), AssetCluster::Indices);
        assert_eq!(get_cluster("QQQ"), AssetCluster::Indices);
        assert_eq!(get_cluster("ES"), AssetCluster::Indices);
        assert_eq!(get_cluster("UK100"), AssetCluster::Indices);
    }

    #[test]
    fn test_get_cluster_energy() {
        assert_eq!(get_cluster("USO"), AssetCluster::Energy);
        assert_eq!(get_cluster("CL"), AssetCluster::Energy);
        assert_eq!(get_cluster("NG"), AssetCluster::Energy);
    }

    #[test]
    fn test_get_cluster_forex() {
        assert_eq!(get_cluster("EURUSD"), AssetCluster::Forex);
        assert_eq!(get_cluster("GBPUSD"), AssetCluster::Forex);
        assert_eq!(get_cluster("USDJPY"), AssetCluster::Forex);
        assert_eq!(get_cluster("AUDUSD"), AssetCluster::Forex);
    }

    #[test]
    fn test_get_cluster_crypto() {
        assert_eq!(get_cluster("BTCUSD"), AssetCluster::Crypto);
        assert_eq!(get_cluster("ETHUSD"), AssetCluster::Crypto);
        assert_eq!(get_cluster("SOLUSD"), AssetCluster::Crypto);
    }

    #[test]
    fn test_get_cluster_singleton() {
        assert_eq!(get_cluster("AAPL"), AssetCluster::Singleton);
        assert_eq!(get_cluster("MSFT"), AssetCluster::Singleton);
        assert_eq!(get_cluster("GOOGL"), AssetCluster::Singleton);
    }

    #[test]
    fn test_transfer_manager_new() {
        let tm = TransferManager::new();
        assert!(tm.cluster_stats.is_empty());
        assert!(tm.symbol_trades.is_empty());
    }

    #[test]
    fn test_update_cluster() {
        let mut tm = TransferManager::new();
        let weights = [0.4, 0.3, 0.2, 0.1, 0.1, 0.2];

        // Update with a metals trade
        tm.update_cluster("XAUUSD", &weights, true);

        let stats = tm.get_cluster_stats(AssetCluster::Metals).unwrap();
        assert_eq!(stats.trade_count, 1);
        assert_eq!(stats.win_count, 1);
        assert!((stats.win_rate() - 1.0).abs() < 0.001);

        // Weights should have moved toward new values
        assert!(stats.weights[0] > 0.3); // Started at 0.3, moved toward 0.4
    }

    #[test]
    fn test_singleton_not_updated() {
        let mut tm = TransferManager::new();
        let weights = [0.4, 0.3, 0.2, 0.1, 0.1, 0.2];

        // Update with singleton (should track trades but not cluster)
        tm.update_cluster("AAPL", &weights, true);

        assert!(tm.get_cluster_stats(AssetCluster::Singleton).is_none());
        assert_eq!(tm.symbol_trade_count("AAPL"), 1);
    }

    #[test]
    fn test_get_cluster_prior() {
        let mut tm = TransferManager::new();
        let weights = [0.4, 0.3, 0.2, 0.1, 0.1, 0.2];

        // Not enough trades yet
        assert!(tm.get_cluster_prior("XAUUSD").is_none());

        // Add enough trades
        for _ in 0..25 {
            tm.update_cluster("XAUUSD", &weights, true);
        }

        // Now should have prior
        let prior = tm.get_cluster_prior("XAGUSD"); // Different symbol, same cluster
        assert!(prior.is_some());
    }

    #[test]
    fn test_get_transfer_confidence() {
        let mut tm = TransferManager::new();
        let weights = [0.4, 0.3, 0.2, 0.1, 0.1, 0.2];

        // No data - should return 0.5
        assert!((tm.get_transfer_confidence("XAUUSD") - 0.5).abs() < 0.001);

        // Add trades with 60% win rate (15 wins, 10 losses)
        for _ in 0..15 {
            tm.update_cluster("XAUUSD", &weights, true);
        }
        for _ in 0..10 {
            tm.update_cluster("XAUUSD", &weights, false);
        }

        // Should reflect cluster win rate
        let conf = tm.get_transfer_confidence("XAGUSD");
        assert!((conf - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_should_transfer() {
        let mut tm = TransferManager::new();
        let weights = [0.4, 0.3, 0.2, 0.1, 0.1, 0.2];

        // Not enough trades
        assert!(!tm.should_transfer("XAUUSD", "XAGUSD"));

        // Add enough winning trades
        for _ in 0..35 {
            tm.update_cluster("XAUUSD", &weights, true);
        }

        // Now should allow transfer within cluster
        assert!(tm.should_transfer("XAUUSD", "XAGUSD"));

        // Should not transfer across clusters
        assert!(!tm.should_transfer("XAUUSD", "SPY"));

        // Should not transfer to/from singletons
        assert!(!tm.should_transfer("AAPL", "MSFT"));
    }

    #[test]
    fn test_save_load() {
        let mut tm = TransferManager::new();
        let weights = [0.4, 0.3, 0.2, 0.1, 0.1, 0.2];

        for _ in 0..10 {
            tm.update_cluster("XAUUSD", &weights, true);
        }

        let path = "/tmp/test_transfer_manager.json";
        tm.save(path).unwrap();

        let loaded = TransferManager::load(path).unwrap();
        assert_eq!(
            loaded.get_cluster_stats(AssetCluster::Metals).unwrap().trade_count,
            10
        );

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_cluster_stats_win_rate() {
        let mut stats = ClusterStats::default();

        // No trades - neutral win rate
        assert!((stats.win_rate() - 0.5).abs() < 0.001);

        stats.trade_count = 10;
        stats.win_count = 7;

        assert!((stats.win_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_format_summary() {
        let mut tm = TransferManager::new();
        let weights = [0.4, 0.3, 0.2, 0.1, 0.1, 0.2];

        // Empty summary
        assert_eq!(tm.format_summary(), "No cluster data yet");

        // Add some data
        for _ in 0..5 {
            tm.update_cluster("XAUUSD", &weights, true);
            tm.update_cluster("SPY", &weights, false);
        }

        let summary = tm.format_summary();
        assert!(summary.contains("Metals"));
        assert!(summary.contains("Indices"));
    }
}
