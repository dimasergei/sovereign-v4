//! Multi-Symbol Data Feed
//!
//! Manages data streams for multiple symbols across multiple sources.

use anyhow::Result;
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::mpsc;
use crate::core::types::Candle;

#[derive(Debug, Clone)]
pub enum FeedMessage {
    Tick {
        symbol: String,
        bid: Decimal,
        ask: Decimal,
        timestamp: i64,
    },
    Candle {
        symbol: String,
        candle: Candle,
    },
    Connected {
        source: String,
    },
    Disconnected {
        source: String,
        error: String,
    },
}

#[derive(Debug, Clone)]
pub struct SymbolConfig {
    pub symbol: String,
    pub source: DataSource,
    pub tick_size: Decimal,
    pub is_forex: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataSource {
    Mt5,
    Alpaca,
    Polygon,
    Ibkr,
}

pub struct MultiFeed {
    symbols: HashMap<String, SymbolConfig>,
    tx: mpsc::Sender<FeedMessage>,
}

impl MultiFeed {
    pub fn new(tx: mpsc::Sender<FeedMessage>) -> Self {
        Self {
            symbols: HashMap::new(),
            tx,
        }
    }

    pub fn add_symbol(&mut self, config: SymbolConfig) {
        self.symbols.insert(config.symbol.clone(), config);
    }

    pub fn remove_symbol(&mut self, symbol: &str) {
        self.symbols.remove(symbol);
    }

    pub fn symbols(&self) -> Vec<&SymbolConfig> {
        self.symbols.values().collect()
    }

    pub fn symbols_by_source(&self, source: DataSource) -> Vec<&SymbolConfig> {
        self.symbols.values()
            .filter(|s| s.source == source)
            .collect()
    }

    /// Route a tick to the appropriate channel
    pub async fn emit_tick(&self, symbol: &str, bid: Decimal, ask: Decimal, timestamp: i64) {
        let _ = self.tx.send(FeedMessage::Tick {
            symbol: symbol.to_string(),
            bid,
            ask,
            timestamp,
        }).await;
    }

    /// Route a candle to the appropriate channel
    pub async fn emit_candle(&self, symbol: &str, candle: Candle) {
        let _ = self.tx.send(FeedMessage::Candle {
            symbol: symbol.to_string(),
            candle,
        }).await;
    }
}

/// Configuration for multi-feed from config file
#[derive(Debug, Clone)]
pub struct MultiFeedConfig {
    pub mt5_host: Option<String>,
    pub mt5_port: Option<u16>,
    pub alpaca_key: Option<String>,
    pub alpaca_secret: Option<String>,
    pub polygon_key: Option<String>,
}

impl MultiFeedConfig {
    pub fn has_mt5(&self) -> bool {
        self.mt5_host.is_some() && self.mt5_port.is_some()
    }

    pub fn has_alpaca(&self) -> bool {
        self.alpaca_key.is_some() && self.alpaca_secret.is_some()
    }

    pub fn has_polygon(&self) -> bool {
        self.polygon_key.is_some()
    }
}
