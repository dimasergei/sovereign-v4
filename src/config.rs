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
    pub system: Option<SystemConfig>,
    #[serde(default)]
    pub broker: BrokerConfig,
    #[serde(default)]
    pub alpaca: Option<AlpacaConfig>,
    #[serde(default)]
    pub ibkr: Option<IbkrConfig>,
    #[serde(default)]
    pub telegram: TelegramConfig,
    #[serde(default)]
    pub portfolio: PortfolioConfig,
    pub universe: UniverseConfig,
    #[serde(default)]
    pub capital: Option<CapitalConfig>,
    #[serde(default)]
    pub risk: Option<RiskConfig>,
    #[serde(default)]
    pub agi: Option<AgiConfig>,
    #[serde(default)]
    pub schedule: Option<ScheduleConfig>,
}

#[derive(Debug, Deserialize)]
pub struct UniverseConfig {
    pub symbols: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SystemConfig {
    pub name: String,
    pub log_level: String,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            name: "Sovereign v4".to_string(),
            log_level: "info".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct BrokerConfig {
    #[serde(default = "default_broker_type")]
    #[serde(rename = "type")]
    pub broker_type: String,
    #[serde(default)]
    pub paper: bool,
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub client_id: Option<i32>,
    #[serde(default)]
    pub account_currency: Option<String>,
}

fn default_broker_type() -> String {
    "alpaca".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct AlpacaConfig {
    pub api_key: String,
    pub secret_key: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IbkrConfig {
    /// Connection mode: "gateway" (REST API) or "tws" (Socket API)
    #[serde(default = "default_connection_mode")]
    pub connection_mode: String,
    /// Gateway URL for REST API mode (e.g., "https://localhost:5000")
    #[serde(default)]
    pub gateway_url: String,
    /// Account ID (e.g., "DU1234567")
    #[serde(default)]
    pub account_id: String,
}

fn default_connection_mode() -> String {
    "gateway".to_string()
}

#[derive(Debug, Deserialize, Default)]
pub struct TelegramConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub chat_id: String,
    #[serde(default)]
    pub alert_on_trade: bool,
    #[serde(default)]
    pub alert_on_dd: bool,
    #[serde(default)]
    pub daily_summary: bool,
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

#[derive(Debug, Deserialize, Clone)]
pub struct CapitalConfig {
    #[serde(default = "default_capital")]
    pub initial: f64,
    #[serde(default = "default_currency")]
    pub currency: String,
}

fn default_capital() -> f64 {
    1000000.0
}

fn default_currency() -> String {
    "CAD".to_string()
}

impl Default for CapitalConfig {
    fn default() -> Self {
        Self {
            initial: default_capital(),
            currency: default_currency(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct RiskConfig {
    #[serde(default = "default_max_position")]
    pub max_position_pct: f64,
    #[serde(default = "default_max_daily_dd")]
    pub max_daily_dd_pct: f64,
    #[serde(default = "default_max_total_dd")]
    pub max_total_dd_pct: f64,
    #[serde(default = "default_max_exposure")]
    pub max_exposure_pct: f64,
    #[serde(default = "default_max_positions")]
    pub max_positions: usize,
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
}

fn default_max_position() -> f64 { 0.5 }
fn default_max_daily_dd() -> f64 { 1.5 }
fn default_max_total_dd() -> f64 { 3.0 }
fn default_max_exposure() -> f64 { 10.0 }
fn default_max_positions() -> usize { 5 }
fn default_min_confidence() -> f64 { 0.50 }

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_pct: default_max_position(),
            max_daily_dd_pct: default_max_daily_dd(),
            max_total_dd_pct: default_max_total_dd(),
            max_exposure_pct: default_max_exposure(),
            max_positions: default_max_positions(),
            min_confidence: default_min_confidence(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgiConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub memory_enabled: bool,
    #[serde(default = "default_true")]
    pub learning_enabled: bool,
    #[serde(default = "default_true")]
    pub transfer_enabled: bool,
    #[serde(default = "default_true")]
    pub self_mod_enabled: bool,
    #[serde(default = "default_true")]
    pub causal_enabled: bool,
    #[serde(default = "default_true")]
    pub world_model_enabled: bool,
    #[serde(default = "default_true")]
    pub streaming_enabled: bool,
    #[serde(default = "default_auto_threshold")]
    pub auto_approve_threshold: f64,
    #[serde(default = "default_true")]
    pub require_backtest: bool,
    #[serde(default = "default_min_evidence")]
    pub min_evidence_trades: u32,
    #[serde(default)]
    pub constitution: Option<ConstitutionConfig>,
}

fn default_true() -> bool { true }
fn default_auto_threshold() -> f64 { 0.03 }
fn default_min_evidence() -> u32 { 30 }

impl Default for AgiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_enabled: true,
            learning_enabled: true,
            transfer_enabled: true,
            self_mod_enabled: true,
            causal_enabled: true,
            world_model_enabled: true,
            streaming_enabled: true,
            auto_approve_threshold: default_auto_threshold(),
            require_backtest: true,
            min_evidence_trades: default_min_evidence(),
            constitution: None,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ConstitutionConfig {
    #[serde(default = "default_max_position")]
    pub max_position_pct: f64,
    #[serde(default = "default_max_daily_dd")]
    pub max_daily_dd_pct: f64,
    #[serde(default = "default_max_total_dd")]
    pub max_total_dd_pct: f64,
    #[serde(default)]
    pub forbidden: Vec<String>,
}

impl Default for ConstitutionConfig {
    fn default() -> Self {
        Self {
            max_position_pct: default_max_position(),
            max_daily_dd_pct: default_max_daily_dd(),
            max_total_dd_pct: default_max_total_dd(),
            forbidden: vec!["max_daily_dd_pct".to_string(), "max_total_dd_pct".to_string(), "constitution".to_string()],
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ScheduleConfig {
    #[serde(default = "default_trading_start")]
    pub trading_start: String,
    #[serde(default = "default_trading_end")]
    pub trading_end: String,
    #[serde(default = "default_timezone")]
    pub timezone: String,
}

fn default_trading_start() -> String { "09:30".to_string() }
fn default_trading_end() -> String { "16:00".to_string() }
fn default_timezone() -> String { "America/Toronto".to_string() }

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            trading_start: default_trading_start(),
            trading_end: default_trading_end(),
            timezone: default_timezone(),
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
        self.broker.broker_type == "alpaca"
    }

    pub fn is_ibkr(&self) -> bool {
        self.broker.broker_type == "ibkr"
    }

    pub fn alpaca_config(&self) -> Option<&AlpacaConfig> {
        self.alpaca.as_ref()
    }

    pub fn ibkr_config(&self) -> Option<&IbkrConfig> {
        self.ibkr.as_ref()
    }

    /// Get the initial capital from either capital or portfolio config
    pub fn get_initial_capital(&self) -> f64 {
        if let Some(capital) = &self.capital {
            capital.initial
        } else {
            self.portfolio.initial_balance
        }
    }

    /// Get the currency
    pub fn get_currency(&self) -> String {
        if let Some(capital) = &self.capital {
            capital.currency.clone()
        } else {
            "USD".to_string()
        }
    }

    /// Get the system name
    pub fn get_system_name(&self) -> String {
        if let Some(system) = &self.system {
            system.name.clone()
        } else {
            "Sovereign v4".to_string()
        }
    }

    /// Get IBKR connection info from broker section
    pub fn get_ibkr_host(&self) -> String {
        self.broker.host.clone().unwrap_or_else(|| "127.0.0.1".to_string())
    }

    pub fn get_ibkr_port(&self) -> u16 {
        self.broker.port.unwrap_or(7497)
    }

    pub fn get_ibkr_client_id(&self) -> i32 {
        self.broker.client_id.unwrap_or(1)
    }

    /// Check if IBKR is in TWS socket mode
    pub fn is_ibkr_tws_mode(&self) -> bool {
        if let Some(ibkr) = &self.ibkr {
            ibkr.connection_mode == "tws"
        } else {
            false
        }
    }

    /// Check if IBKR is in Gateway REST mode
    pub fn is_ibkr_gateway_mode(&self) -> bool {
        if let Some(ibkr) = &self.ibkr {
            ibkr.connection_mode == "gateway" || ibkr.connection_mode.is_empty()
        } else {
            // Default to gateway mode if no ibkr config
            true
        }
    }

    /// Get IBKR account ID
    pub fn get_ibkr_account_id(&self) -> String {
        if let Some(ibkr) = &self.ibkr {
            ibkr.account_id.clone()
        } else {
            String::new()
        }
    }

    /// Get IBKR gateway URL
    pub fn get_ibkr_gateway_url(&self) -> String {
        if let Some(ibkr) = &self.ibkr {
            if ibkr.gateway_url.is_empty() {
                "https://localhost:5000".to_string()
            } else {
                ibkr.gateway_url.clone()
            }
        } else {
            "https://localhost:5000".to_string()
        }
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

    #[test]
    fn test_default_agi_config() {
        let cfg = AgiConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.learning_enabled);
        assert_eq!(cfg.auto_approve_threshold, 0.03);
    }
}
