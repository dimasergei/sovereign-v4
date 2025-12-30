//! MT5 Broker - Scaffold for direct MT5 integration
//!
//! Note: Current implementation uses mt5_bridge.rs for TCP communication.
//! This module is a scaffold for future direct integration.

use rust_decimal::Decimal;

pub struct Mt5Broker {
    pub host: String,
    pub port: u16,
    pub connected: bool,
}

impl Mt5Broker {
    pub fn new(host: String, port: u16) -> Self {
        Self {
            host,
            port,
            connected: false,
        }
    }
}
