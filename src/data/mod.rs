//! Data Module
//!
//! Data persistence and retrieval.
//! - PostgreSQL for historical data and trades
//! - Redis for real-time state and pub/sub

pub mod postgres;
pub mod redis;
