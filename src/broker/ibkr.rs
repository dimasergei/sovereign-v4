//! Interactive Brokers Client Portal API Integration
//!
//! Uses the IBKR Client Portal Gateway REST API.
//! Gateway must be running locally on https://localhost:5000

use anyhow::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, debug};

/// IBKR Broker client
pub struct IbkrBroker {
    gateway_url: String,
    account_id: String,
    client: reqwest::Client,
    /// Cache of symbol -> conid mappings
    conid_cache: Arc<Mutex<HashMap<String, i64>>>,
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AccountInfo {
    #[serde(rename = "accountId")]
    pub account_id: String,
    #[serde(rename = "accountTitle")]
    pub account_title: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AccountSummary {
    #[serde(rename = "availableFunds")]
    pub available_funds: Option<AccountValue>,
    #[serde(rename = "netLiquidation")]
    pub net_liquidation: Option<AccountValue>,
    #[serde(rename = "buyingPower")]
    pub buying_power: Option<AccountValue>,
}

#[derive(Debug, Deserialize)]
pub struct AccountValue {
    pub amount: f64,
    pub currency: String,
}

#[derive(Debug, Deserialize)]
pub struct IbkrPosition {
    #[serde(rename = "contractDesc")]
    pub contract_desc: Option<String>,
    pub conid: i64,
    pub position: f64,
    #[serde(rename = "avgCost")]
    pub avg_cost: f64,
    #[serde(rename = "mktValue")]
    pub mkt_value: Option<f64>,
    #[serde(rename = "unrealizedPnl")]
    pub unrealized_pnl: Option<f64>,
    #[serde(rename = "assetClass")]
    pub asset_class: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContractSearch {
    #[serde(default)]
    pub conid: i64,
    #[serde(default)]
    pub symbol: String,
    #[serde(rename = "companyName")]
    pub company_name: Option<String>,
    #[serde(rename = "secType")]
    pub sec_type: Option<String>,
}

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
    #[serde(rename = "t")]
    pub time: i64,
}

#[derive(Debug, Deserialize)]
pub struct HistoricalData {
    pub data: Vec<HistoricalBar>,
    #[serde(rename = "mdAvailability")]
    pub md_availability: Option<String>,
}

#[derive(Debug, Serialize)]
struct OrderRequest {
    #[serde(rename = "conid")]
    conid: i64,
    #[serde(rename = "orderType")]
    order_type: String,
    side: String,
    quantity: f64,
    tif: String,
}

#[derive(Debug, Serialize)]
struct OrdersRequest {
    orders: Vec<OrderRequest>,
}

#[derive(Debug, Deserialize)]
pub struct OrderResponse {
    pub order_id: Option<String>,
    pub order_status: Option<String>,
    pub message: Option<Vec<String>>,
    pub id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OrderConfirmation {
    pub id: Option<String>,
    pub message: Option<Vec<String>>,
}

// ============================================================================
// Implementation
// ============================================================================

impl IbkrBroker {
    /// Create a new IBKR broker client
    pub fn new(gateway_url: String, account_id: String) -> Result<Self> {
        // Build client that accepts self-signed certificates
        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            gateway_url,
            account_id,
            client,
            conid_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Keep session alive - call every 60 seconds
    pub async fn tickle(&self) -> Result<()> {
        let url = format!("{}/v1/api/tickle", self.gateway_url);
        let resp = self.client.post(&url).send().await?;

        if !resp.status().is_success() {
            warn!("Tickle failed: {}", resp.status());
        } else {
            debug!("Session tickle OK");
        }
        Ok(())
    }

    /// Get account information
    pub async fn get_account(&self) -> Result<AccountSummary> {
        let url = format!(
            "{}/v1/api/portfolio/{}/summary",
            self.gateway_url, self.account_id
        );

        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get account: {}", resp.status()));
        }

        let summary: AccountSummary = resp.json().await?;
        Ok(summary)
    }

    /// Get all positions
    pub async fn get_positions(&self) -> Result<Vec<IbkrPosition>> {
        let url = format!(
            "{}/v1/api/portfolio/{}/positions/0",
            self.gateway_url, self.account_id
        );

        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get positions: {}", resp.status()));
        }

        let positions: Vec<IbkrPosition> = resp.json().await?;
        Ok(positions)
    }

    /// Look up contract ID (conid) for a symbol
    pub async fn get_conid(&self, symbol: &str) -> Result<i64> {
        // Check cache first
        {
            let cache = self.conid_cache.lock().await;
            if let Some(&conid) = cache.get(symbol) {
                return Ok(conid);
            }
        }

        // Search for contract
        let url = format!(
            "{}/v1/api/iserver/secdef/search?symbol={}",
            self.gateway_url, symbol
        );

        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Contract search failed for {}: {}", symbol, resp.status()));
        }

        let contracts: Vec<ContractSearch> = resp.json().await?;

        // Find the STK (stock) contract
        let conid = contracts
            .iter()
            .find(|c| c.sec_type.as_deref() == Some("STK"))
            .or_else(|| contracts.first())
            .map(|c| c.conid)
            .ok_or_else(|| anyhow::anyhow!("No contract found for {}", symbol))?;

        // Cache it
        {
            let mut cache = self.conid_cache.lock().await;
            cache.insert(symbol.to_string(), conid);
        }

        info!("{}: conid = {}", symbol, conid);
        Ok(conid)
    }

    /// Submit a market order
    pub async fn submit_order(&self, symbol: &str, side: &str, qty: Decimal) -> Result<OrderResponse> {
        let conid = self.get_conid(symbol).await?;

        let order = OrderRequest {
            conid,
            order_type: "MKT".to_string(),
            side: side.to_uppercase(),
            quantity: qty.to_string().parse().unwrap_or(0.0),
            tif: "DAY".to_string(),
        };

        let request = OrdersRequest {
            orders: vec![order],
        };

        let url = format!(
            "{}/v1/api/iserver/account/{}/orders",
            self.gateway_url, self.account_id
        );

        let resp = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Order failed: {}", error_text));
        }

        // IBKR may return a confirmation request
        let response_text = resp.text().await?;
        debug!("Order response: {}", response_text);

        // Try to parse as order response
        let order_resp: Vec<OrderResponse> = serde_json::from_str(&response_text)
            .unwrap_or_else(|_| vec![OrderResponse {
                order_id: None,
                order_status: Some("submitted".to_string()),
                message: None,
                id: None,
            }]);

        // Check if confirmation needed
        if let Some(resp) = order_resp.first() {
            if resp.id.is_some() {
                // Need to confirm the order
                self.confirm_order(&resp.id.clone().unwrap()).await?;
            }
        }

        Ok(order_resp.into_iter().next().unwrap_or(OrderResponse {
            order_id: None,
            order_status: None,
            message: None,
            id: None,
        }))
    }

    /// Confirm an order (for warning messages)
    async fn confirm_order(&self, reply_id: &str) -> Result<()> {
        let url = format!(
            "{}/v1/api/iserver/reply/{}",
            self.gateway_url, reply_id
        );

        let body = serde_json::json!({ "confirmed": true });

        let resp = self.client
            .post(&url)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            warn!("Order confirmation failed: {}", resp.status());
        }

        Ok(())
    }

    /// Buy shares
    pub async fn buy(&self, symbol: &str, qty: Decimal) -> Result<OrderResponse> {
        self.submit_order(symbol, "BUY", qty).await
    }

    /// Sell shares
    pub async fn sell(&self, symbol: &str, qty: Decimal) -> Result<OrderResponse> {
        self.submit_order(symbol, "SELL", qty).await
    }

    /// Close a position (sell all shares)
    pub async fn close_position(&self, symbol: &str) -> Result<OrderResponse> {
        // Get current position
        let positions = self.get_positions().await?;
        let conid = self.get_conid(symbol).await?;

        let position = positions.iter().find(|p| p.conid == conid);

        match position {
            Some(pos) => {
                let qty = Decimal::from_f64_retain(pos.position.abs())
                    .unwrap_or(Decimal::ZERO);

                if pos.position > 0.0 {
                    self.sell(symbol, qty).await
                } else {
                    self.buy(symbol, qty).await
                }
            }
            None => Err(anyhow::anyhow!("No position found for {}", symbol)),
        }
    }

    /// Get historical bars for a symbol
    pub async fn get_historical_bars(
        &self,
        symbol: &str,
        period: &str,  // e.g., "10y" for 10 years
        bar_size: &str, // e.g., "1d" for daily bars
    ) -> Result<Vec<HistoricalBar>> {
        let conid = self.get_conid(symbol).await?;

        let url = format!(
            "{}/v1/api/iserver/marketdata/history?conid={}&period={}&bar={}&outsideRth=false",
            self.gateway_url, conid, period, bar_size
        );

        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get historical bars for {}: {}", symbol, resp.status()));
        }

        let data: HistoricalData = resp.json().await?;
        Ok(data.data)
    }

    /// Get all available daily bars (up to 10 years)
    pub async fn get_all_daily_bars(&self, symbol: &str) -> Result<Vec<HistoricalBar>> {
        // Request 10 years of daily data
        self.get_historical_bars(symbol, "10y", "1d").await
    }
}

// ============================================================================
// Position helper for external use
// ============================================================================

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub qty: String,
    pub avg_entry_price: String,
    pub unrealized_pl: String,
    pub side: String,
}

impl From<&IbkrPosition> for Position {
    fn from(p: &IbkrPosition) -> Self {
        let side = if p.position >= 0.0 { "long" } else { "short" };
        Self {
            symbol: p.contract_desc.clone().unwrap_or_default(),
            qty: p.position.abs().to_string(),
            avg_entry_price: p.avg_cost.to_string(),
            unrealized_pl: p.unrealized_pnl.unwrap_or(0.0).to_string(),
            side: side.to_string(),
        }
    }
}
