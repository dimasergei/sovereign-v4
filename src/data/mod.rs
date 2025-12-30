//! Data Module
//!
//! Data persistence and retrieval.
//! - PostgreSQL for historical data and trades
//! - Redis for real-time state and pub/sub
//! - MT5 Bridge for live market data

pub mod postgres;
pub mod redis;
pub mod mt5_bridge;
