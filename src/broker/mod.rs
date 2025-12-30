//! Broker Module
//!
//! Abstractions for broker connections.
//! Currently supports: MT5 (via bridge)
//! Future: IBKR, Alpaca

pub mod mt5;

use async_trait::async_trait;
use anyhow::Result;

use crate::core::types::*;

/// Broker trait - all broker implementations must implement this
#[async_trait]
pub trait Broker: Send + Sync {
    /// Connect to the broker
    async fn connect(&mut self) -> Result<()>;
    
    /// Disconnect from the broker
    async fn disconnect(&mut self) -> Result<()>;
    
    /// Check if connected
    fn is_connected(&self) -> bool;
    
    /// Get account information
    async fn get_account(&self) -> Result<AccountInfo>;
    
    /// Get current tick for a symbol
    async fn get_tick(&self, symbol: &str) -> Result<Tick>;
    
    /// Get candles for a symbol
    async fn get_candles(&self, symbol: &str, timeframe: u32, count: usize) -> Result<Vec<Candle>>;
    
    /// Get open positions
    async fn get_positions(&self) -> Result<Vec<Position>>;
    
    /// Place an order
    async fn place_order(&self, order: OrderRequest) -> Result<OrderResult>;
    
    /// Close a position
    async fn close_position(&self, ticket: u64) -> Result<()>;
}

/// Order request
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: PositionSide,
    pub volume: rust_decimal::Decimal,
    pub stop_loss: Option<rust_decimal::Decimal>,
    pub take_profit: Option<rust_decimal::Decimal>,
    pub comment: String,
}

/// Order result
#[derive(Debug, Clone)]
pub struct OrderResult {
    pub success: bool,
    pub ticket: Option<u64>,
    pub price: Option<rust_decimal::Decimal>,
    pub error: Option<String>,
}
