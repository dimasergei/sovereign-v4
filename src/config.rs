//! Configuration loader
use anyhow::Result;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::fs;
use std::str::FromStr;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub system: SystemConfig,
    #[serde(default)]
    pub broker: BrokerConfig,
    #[serde(default)]
    pub alpaca: Option<AlpacaConfig>,
    #[serde(default)]
    pub bridge: Option<BridgeConfig>,
    pub telegram: TelegramConfig,
    pub risk: RiskConfig,
    pub strategy: StrategyConfig,
    pub symbols: Vec<SymbolConfig>,
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

#[derive(Debug, Deserialize)]
pub struct BridgeConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize)]
pub struct TelegramConfig {
    pub enabled: bool,
    pub bot_token: String,
    pub chat_id: String,
}

#[derive(Debug, Deserialize)]
pub struct RiskConfig {
    pub risk_per_trade: f64,
    pub max_daily_loss: f64,
    pub max_floating_loss: f64,
    pub max_positions: usize,
    pub max_trades_per_day: usize,
    pub trade_cooldown_secs: u64,
    pub loss_cooldown_secs: u64,
}

#[derive(Debug, Deserialize)]
pub struct StrategyConfig {
    pub min_conviction: u8,
    pub risk_reward_ratio: f64,
}

#[derive(Debug, Deserialize)]
pub struct SymbolConfig {
    pub name: String,
    pub tick_size: f64,
    pub is_forex: bool,
    pub point_value: f64,
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

impl RiskConfig {
    pub fn to_guardian_config(&self) -> crate::core::guardian::RiskConfig {
        crate::core::guardian::RiskConfig {
            risk_per_trade: Decimal::from_str(&self.risk_per_trade.to_string()).unwrap(),
            max_daily_loss: Decimal::from_str(&self.max_daily_loss.to_string()).unwrap(),
            max_floating_loss: Decimal::from_str(&self.max_floating_loss.to_string()).unwrap(),
            max_positions: self.max_positions,
            max_trades_per_day: self.max_trades_per_day,
            trade_cooldown_secs: self.trade_cooldown_secs,
            loss_cooldown_secs: self.loss_cooldown_secs,
        }
    }
}

impl SymbolConfig {
    pub fn tick_size_decimal(&self) -> Decimal {
        Decimal::from_str(&self.tick_size.to_string()).unwrap()
    }
    
    pub fn point_value_decimal(&self) -> Decimal {
        Decimal::from_str(&self.point_value.to_string()).unwrap()
    }
}
