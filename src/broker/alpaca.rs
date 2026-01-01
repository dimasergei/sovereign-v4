//! Alpaca Broker Integration
//!
//! REST API for stocks and crypto trading.
//! Free tier available at alpaca.markets

use anyhow::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub struct AlpacaBroker {
    api_key: String,
    api_secret: String,
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct OrderRequest {
    symbol: String,
    qty: String,
    side: String,
    #[serde(rename = "type")]
    order_type: String,
    time_in_force: String,
    stop_loss: Option<StopLoss>,
    take_profit: Option<TakeProfit>,
}

#[derive(Debug, Serialize)]
struct StopLoss {
    stop_price: String,
}

#[derive(Debug, Serialize)]
struct TakeProfit {
    limit_price: String,
}

#[derive(Debug, Deserialize)]
pub struct OrderResponse {
    pub id: String,
    pub status: String,
    pub filled_avg_price: Option<String>,
    pub filled_qty: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub qty: String,
    pub avg_entry_price: String,
    pub unrealized_pl: String,
    pub side: String,
}

#[derive(Debug, Deserialize)]
pub struct Account {
    pub equity: String,
    pub cash: String,
    pub buying_power: String,
}

impl AlpacaBroker {
    pub fn new(api_key: String, api_secret: String, paper: bool) -> Self {
        let base_url = if paper {
            "https://paper-api.alpaca.markets".to_string()
        } else {
            "https://api.alpaca.markets".to_string()
        };
        
        Self {
            api_key,
            api_secret,
            base_url,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_account(&self) -> Result<Account> {
        let resp = self.client
            .get(format!("{}/v2/account", self.base_url))
            .header("APCA-API-KEY-ID", &self.api_key)
            .header("APCA-API-SECRET-KEY", &self.api_secret)
            .send()
            .await?
            .json::<Account>()
            .await?;
        Ok(resp)
    }

    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        let resp = self.client
            .get(format!("{}/v2/positions", self.base_url))
            .header("APCA-API-KEY-ID", &self.api_key)
            .header("APCA-API-SECRET-KEY", &self.api_secret)
            .send()
            .await?
            .json::<Vec<Position>>()
            .await?;
        Ok(resp)
    }

    pub async fn buy(
        &self,
        symbol: &str,
        qty: Decimal,
        stop_loss: Option<Decimal>,
        take_profit: Option<Decimal>,
    ) -> Result<OrderResponse> {
        self.submit_order(symbol, qty, "buy", stop_loss, take_profit).await
    }

    pub async fn sell(
        &self,
        symbol: &str,
        qty: Decimal,
        stop_loss: Option<Decimal>,
        take_profit: Option<Decimal>,
    ) -> Result<OrderResponse> {
        self.submit_order(symbol, qty, "sell", stop_loss, take_profit).await
    }

    async fn submit_order(
        &self,
        symbol: &str,
        qty: Decimal,
        side: &str,
        stop_loss: Option<Decimal>,
        take_profit: Option<Decimal>,
    ) -> Result<OrderResponse> {
        let order = OrderRequest {
            symbol: symbol.to_string(),
            qty: qty.to_string(),
            side: side.to_string(),
            order_type: "market".to_string(),
            time_in_force: "gtc".to_string(),
            stop_loss: stop_loss.map(|p| StopLoss { stop_price: p.to_string() }),
            take_profit: take_profit.map(|p| TakeProfit { limit_price: p.to_string() }),
        };

        let resp = self.client
            .post(format!("{}/v2/orders", self.base_url))
            .header("APCA-API-KEY-ID", &self.api_key)
            .header("APCA-API-SECRET-KEY", &self.api_secret)
            .json(&order)
            .send()
            .await?
            .json::<OrderResponse>()
            .await?;
        Ok(resp)
    }

    pub async fn close_position(&self, symbol: &str) -> Result<OrderResponse> {
        let resp = self.client
            .delete(format!("{}/v2/positions/{}", self.base_url, symbol))
            .header("APCA-API-KEY-ID", &self.api_key)
            .header("APCA-API-SECRET-KEY", &self.api_secret)
            .send()
            .await?
            .json::<OrderResponse>()
            .await?;
        Ok(resp)
    }
}

/// Historical bar data for bootstrapping S/R levels
#[derive(Debug, Deserialize)]
pub struct HistoricalBar {
    #[serde(rename = "o")]
    pub open: f64,
    #[serde(rename = "h")]
    pub high: f64,
    #[serde(rename = "l")]
    pub low: f64,
    #[serde(rename = "c")]
    pub close: f64,
    #[serde(rename = "v")]
    pub volume: u64,
}

#[derive(Debug, Deserialize)]
struct BarsResponse {
    bars: Vec<HistoricalBar>,
    next_page_token: Option<String>,
}

impl AlpacaBroker {
    /// Fetch all available daily bars for a symbol
    /// Alpaca returns up to 10000 bars - we take whatever they give
    pub async fn get_all_daily_bars(&self, symbol: &str) -> Result<Vec<HistoricalBar>> {
        let url = format!(
            "https://data.alpaca.markets/v2/stocks/{}/bars?timeframe=1Day&limit=10000",
            symbol
        );

        let resp = self.client
            .get(&url)
            .header("APCA-API-KEY-ID", &self.api_key)
            .header("APCA-API-SECRET-KEY", &self.api_secret)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Failed to fetch bars for {}: {}", symbol, resp.status()));
        }

        let bars_resp: BarsResponse = resp.json().await?;
        Ok(bars_resp.bars)
    }
}

// Data stream for real-time prices
pub struct AlpacaDataStream {
    api_key: String,
    api_secret: String,
}

impl AlpacaDataStream {
    pub fn new(api_key: String, api_secret: String) -> Self {
        Self { api_key, api_secret }
    }

    pub fn stream_url(&self) -> String {
        "wss://stream.data.alpaca.markets/v2/iex".to_string()
    }

    pub fn auth_message(&self) -> String {
        format!(r#"{{"action":"auth","key":"{}","secret":"{}"}}"#, 
            self.api_key, self.api_secret)
    }

    pub fn subscribe_message(&self, symbols: &[&str]) -> String {
        let syms: Vec<String> = symbols.iter().map(|s| format!("\"{}\"", s)).collect();
        format!(r#"{{"action":"subscribe","bars":[{}]}}"#, syms.join(","))
    }
}
