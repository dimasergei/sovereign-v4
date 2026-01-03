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
//! # Execution Infrastructure
//!
//! Institutional-grade execution capabilities including:
//! - Tiered execution (Retail → Semi-Institutional → Institutional)
//! - Execution algorithms (VWAP, TWAP, POV, Iceberg, Adaptive)
//! - Smart order routing with venue scoring
//! - Dark pool integration
//! - Transaction Cost Analysis (TCA)
//! - FIX protocol support

pub mod core;
pub mod universe;
pub mod portfolio;
pub mod broker;
pub mod data;
pub mod comms;
pub mod config;
pub mod execution;
