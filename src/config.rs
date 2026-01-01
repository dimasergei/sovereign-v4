//! Configuration loader
//!
//! NOTE: This config contains NO strategy parameters.
//! The strategy has NO parameters - that's the lossless philosophy.
//!
//! Only infrastructure settings are configured here.

use anyhow::Result;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub system: SystemConfig,
    #[serde(default)]
    pub broker: BrokerConfig,
    #[serde(default)]
    pub alpaca: Option<AlpacaConfig>,
    #[serde(default)]
    pub telegram: TelegramConfig,
    #[serde(default)]
    pub portfolio: PortfolioConfig,
    pub universe: UniverseConfig,
}

#[derive(Debug, Deserialize)]
pub struct UniverseConfig {
    pub symbols: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct SystemConfig {
    pub name: String,
    pub log_level: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct BrokerConfig {
    #[serde(default = "default_broker_type")]
    pub broker_type: String,
    #[serde(default)]
    pub paper: bool,
}

fn default_broker_type() -> String {
    "alpaca".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct AlpacaConfig {
    pub api_key: String,
    pub secret_key: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct TelegramConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub chat_id: String,
}

#[derive(Debug, Deserialize)]
pub struct PortfolioConfig {
    #[serde(default = "default_initial_balance")]
    pub initial_balance: f64,
}

fn default_initial_balance() -> f64 {
    100000.0
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            initial_balance: default_initial_balance(),
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn is_alpaca(&self) -> bool {
        self.alpaca.is_some()
    }

    pub fn alpaca_config(&self) -> Option<&AlpacaConfig> {
        self.alpaca.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_portfolio_config() {
        let cfg = PortfolioConfig::default();
        assert_eq!(cfg.initial_balance, 100000.0);
    }
}
