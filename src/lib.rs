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

pub mod core;
pub mod universe;
pub mod portfolio;
pub mod broker;
pub mod data;
pub mod comms;
pub mod config;
