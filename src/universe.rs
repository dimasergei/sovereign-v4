//! Trading Universe Module
//!
//! Manages the list of symbols to trade.
//! This is NOT a parameter - it's the market definition.
//!
//! Tech Trader trades essentially all liquid US stocks + ETFs + crypto.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Default symbols for the trading universe
///
/// 130 liquid symbols for IBKR mode - full universe for Tech Trader.
pub const DEFAULT_SYMBOLS: &[&str] = &[
    // Major Tech (20)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU", "LRCX",
    // Tech Extended (15)
    "NOW", "PANW", "SNOW", "CRWD", "ZS", "DDOG", "NET", "PLTR", "UBER", "ABNB",
    "COIN", "SQ", "SHOP", "PYPL", "ROKU",
    // Finance (15)
    "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "BLK", "AXP", "V",
    "MA", "COF", "USB", "PNC", "TFC",
    // Healthcare (15)
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT", "AMGN", "GILD",
    "BMY", "CVS", "CI", "HUM", "ISRG",
    // Industrial (15)
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "RTX", "DE", "MMM", "GD",
    "FDX", "NSC", "UNP", "WM", "EMR",
    // Consumer (15)
    "DIS", "NKE", "SBUX", "MCD", "HD", "LOW", "TGT", "COST", "WMT", "PG",
    "KO", "PEP", "CL", "EL", "LULU",
    // Energy (15)
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "MPC", "PBR",
    "HAL", "DVN", "FANG", "HES", "BKR",
    // ETFs (20)
    "SPY", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLE", "XLK", "GLD", "SLV",
    "USO", "UNG", "TLT", "HYG", "EEM", "EWZ", "FXI", "VXX", "ARKK", "XLV",
];

/// Trading universe containing all tradable symbols
#[derive(Debug, Clone)]
pub struct Universe {
    symbols: HashSet<String>,
    include_crypto: bool,
}

impl Universe {
    /// Create a new universe with default symbols
    pub fn new() -> Self {
        let symbols: HashSet<String> = DEFAULT_SYMBOLS
            .iter()
            .map(|s| s.to_string())
            .collect();

        Self {
            symbols,
            include_crypto: true,
        }
    }

    /// Create universe from a list of symbols
    pub fn from_symbols(symbols: Vec<String>) -> Self {
        Self {
            symbols: symbols.into_iter().collect(),
            include_crypto: true,
        }
    }

    /// Load universe from a file (one symbol per line)
    pub fn from_file(path: &Path) -> Result<Self, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let symbols: HashSet<String> = content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|s| s.to_uppercase())
            .collect();

        Ok(Self {
            symbols,
            include_crypto: true,
        })
    }

    /// Add a symbol to the universe
    pub fn add(&mut self, symbol: &str) {
        self.symbols.insert(symbol.to_uppercase());
    }

    /// Remove a symbol from the universe
    pub fn remove(&mut self, symbol: &str) {
        self.symbols.remove(&symbol.to_uppercase());
    }

    /// Check if a symbol is in the universe
    pub fn contains(&self, symbol: &str) -> bool {
        self.symbols.contains(&symbol.to_uppercase())
    }

    /// Get all symbols as a vector (sorted for consistency)
    pub fn symbols(&self) -> Vec<String> {
        let mut syms: Vec<_> = self.symbols.iter().cloned().collect();
        syms.sort();
        syms
    }

    /// Get stock symbols only (no crypto)
    pub fn stocks(&self) -> Vec<String> {
        self.symbols()
            .into_iter()
            .filter(|s| !Self::is_crypto(s))
            .collect()
    }

    /// Get crypto symbols only
    pub fn crypto(&self) -> Vec<String> {
        if !self.include_crypto {
            return vec![];
        }
        self.symbols()
            .into_iter()
            .filter(|s| Self::is_crypto(s))
            .collect()
    }

    /// Check if a symbol is crypto
    pub fn is_crypto(symbol: &str) -> bool {
        symbol.ends_with("USD")
            || symbol.ends_with("BTC")
            || symbol.ends_with("ETH")
            || matches!(symbol, "BTC" | "ETH" | "XRP" | "DOGE" | "ADA" | "SOL")
    }

    /// Get the number of symbols in the universe
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Check if the universe is empty
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Enable/disable crypto trading
    pub fn set_include_crypto(&mut self, include: bool) {
        self.include_crypto = include;
    }

    /// Filter symbols by minimum price and volume
    ///
    /// This is a liquidity filter - we only trade liquid symbols.
    pub fn filter_liquid(&self, min_price: f64, min_volume: u64) -> Vec<String> {
        // In practice, this would check against market data
        // For now, return all symbols (filtering happens at runtime)
        self.symbols()
    }
}

impl Default for Universe {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a universe for backtesting a single symbol
pub fn single_symbol(symbol: &str) -> Universe {
    Universe::from_symbols(vec![symbol.to_string()])
}

/// Create a universe for ETFs only
pub fn etfs_only() -> Universe {
    Universe::from_symbols(vec![
        "SPY".to_string(),
        "QQQ".to_string(),
        "IWM".to_string(),
        "DIA".to_string(),
        "GLD".to_string(),
        "SLV".to_string(),
        "USO".to_string(),
        "TLT".to_string(),
    ])
}

/// Create a universe for commodity ETFs (good for testing lossless)
pub fn commodity_etfs() -> Universe {
    Universe::from_symbols(vec![
        "USO".to_string(), // Oil
        "GLD".to_string(), // Gold
        "SLV".to_string(), // Silver
        "UNG".to_string(), // Natural Gas
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_universe() {
        let universe = Universe::new();
        assert!(!universe.is_empty());
        assert!(universe.contains("AAPL"));
        assert!(universe.contains("SPY"));
    }

    #[test]
    fn test_from_symbols() {
        let universe = Universe::from_symbols(vec![
            "AAPL".to_string(),
            "MSFT".to_string(),
        ]);
        assert_eq!(universe.len(), 2);
        assert!(universe.contains("AAPL"));
        assert!(universe.contains("MSFT"));
        assert!(!universe.contains("GOOGL"));
    }

    #[test]
    fn test_crypto_detection() {
        assert!(Universe::is_crypto("BTCUSD"));
        assert!(Universe::is_crypto("ETHUSD"));
        assert!(Universe::is_crypto("BTC"));
        assert!(!Universe::is_crypto("AAPL"));
        assert!(!Universe::is_crypto("SPY"));
    }

    #[test]
    fn test_add_remove() {
        let mut universe = Universe::from_symbols(vec!["AAPL".to_string()]);

        assert_eq!(universe.len(), 1);

        universe.add("MSFT");
        assert_eq!(universe.len(), 2);
        assert!(universe.contains("MSFT"));

        universe.remove("AAPL");
        assert_eq!(universe.len(), 1);
        assert!(!universe.contains("AAPL"));
    }

    #[test]
    fn test_stock_crypto_split() {
        let universe = Universe::from_symbols(vec![
            "AAPL".to_string(),
            "BTCUSD".to_string(),
            "SPY".to_string(),
            "ETHUSD".to_string(),
        ]);

        let stocks = universe.stocks();
        assert_eq!(stocks.len(), 2);
        assert!(stocks.contains(&"AAPL".to_string()));
        assert!(stocks.contains(&"SPY".to_string()));

        let crypto = universe.crypto();
        assert_eq!(crypto.len(), 2);
        assert!(crypto.contains(&"BTCUSD".to_string()));
        assert!(crypto.contains(&"ETHUSD".to_string()));
    }

    #[test]
    fn test_single_symbol() {
        let universe = single_symbol("USO");
        assert_eq!(universe.len(), 1);
        assert!(universe.contains("USO"));
    }

    #[test]
    fn test_commodity_etfs() {
        let universe = commodity_etfs();
        assert!(universe.contains("USO"));
        assert!(universe.contains("GLD"));
        assert!(!universe.contains("AAPL"));
    }
}
