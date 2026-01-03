//! Broker Module
//!
//! Broker integrations: IBKR, Alpaca, and Prime Broker (FIX).

pub mod alpaca;
pub mod ibkr;
pub mod prime_broker;

use rust_decimal::Decimal;
use async_trait::async_trait;
use thiserror::Error;

/// Broker error types
#[derive(Error, Debug)]
pub enum BrokerError {
    #[error("Not connected to broker")]
    NotConnected,

    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Order rejected: {0}")]
    OrderRejected(String),

    #[error("Insufficient funds")]
    InsufficientFunds,

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Timeout")]
    Timeout,

    #[error("Rate limited")]
    RateLimited,

    #[error("Authentication failed")]
    AuthenticationFailed,

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Unified broker interface
#[async_trait]
pub trait Broker: Send + Sync {
    /// Get account balance and equity
    async fn get_account(&self) -> anyhow::Result<AccountState>;
    
    /// Get open positions
    async fn get_positions(&self) -> anyhow::Result<Vec<BrokerPosition>>;
    
    /// Submit buy order
    async fn buy(&self, symbol: &str, qty: Decimal, sl: Decimal, tp: Decimal) 
        -> anyhow::Result<OrderResult>;
    
    /// Submit sell order
    async fn sell(&self, symbol: &str, qty: Decimal, sl: Decimal, tp: Decimal) 
        -> anyhow::Result<OrderResult>;
    
    /// Close position
    async fn close(&self, symbol: &str) -> anyhow::Result<OrderResult>;
}

#[derive(Debug, Clone)]
pub struct AccountState {
    pub balance: Decimal,
    pub equity: Decimal,
    pub margin_used: Decimal,
}

#[derive(Debug, Clone)]
pub struct BrokerPosition {
    pub symbol: String,
    pub side: PositionSide,
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
}

#[derive(Debug, Clone)]
pub enum PositionSide {
    Long,
    Short,
}

pub struct OrderRequest {
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub order_type: OrderType,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
}

#[derive(Debug, Clone)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market,
    Limit(Decimal),
    Stop(Decimal),
}

pub struct OrderResult {
    pub success: bool,
    pub order_id: String,
    pub filled_price: Option<Decimal>,
    pub filled_qty: Option<Decimal>,
    pub error: Option<String>,
}
