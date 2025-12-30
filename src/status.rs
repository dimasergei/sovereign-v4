//! Status file for sharing state between processes

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const STATUS_FILE: &str = "sovereign_status.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemStatus {
    pub running: bool,
    pub start_time: i64,
    pub trades_today: u32,
    pub total_pnl: f64,
    pub active_positions: u32,
    pub last_signal: String,
    pub last_price: f64,
    pub balance: f64,
    pub equity: f64,
}

impl SystemStatus {
    pub fn save(&self) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(STATUS_FILE, json)?;
        Ok(())
    }

    pub fn load() -> Self {
        if Path::new(STATUS_FILE).exists() {
            if let Ok(contents) = fs::read_to_string(STATUS_FILE) {
                if let Ok(status) = serde_json::from_str(&contents) {
                    return status;
                }
            }
        }
        Self::default()
    }
}
