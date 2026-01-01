//! Sovereign v4.0 Library
//!
//! A "lossless" autonomous trading system based on pftq's Tech Trader philosophy.
//!
//! # Philosophy
//!
//! - No parameters, no thresholds, no statistics
//! - Pure counting-based S/R detection
//! - Volume capitulation for entry signals
//! - One independent agent per symbol
//!
//! # Modules
//!
//! - `core`: Core trading logic (S/R, capitulation, agents)
//! - `universe`: Trading universe (symbol lists)
//! - `portfolio`: Portfolio and position management
//! - `broker`: Broker integrations (Alpaca)
//! - `data`: Market data streams
//! - `comms`: Communications (Telegram)
//! - `config`: Configuration loading
//! - `backtest`: Backtesting harness

pub mod core;
pub mod universe;
pub mod portfolio;
pub mod broker;
pub mod data;
pub mod comms;
pub mod config;
pub mod backtest;
pub mod status;
