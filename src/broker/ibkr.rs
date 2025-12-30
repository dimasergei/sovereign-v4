//! Interactive Brokers Integration
//!
//! Uses IBKR Client Portal API (Gateway required)
//! https://interactivebrokers.github.io/cpwebapi/

use anyhow::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

pub struct IbkrBroker {
    gateway_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct IbkrAccount {
    pub account_id: String,
    pub net_liquidation: f64,
    pub equity_with_loan: f64,
    pub available_funds: f64,
}

#[derive(Debug, Deserialize)]
pub struct IbkrPosition {
    pub contract_id: i64,
    pub symbol: String,
    pub position: f64,
    pub avg_cost: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug, Serialize)]
struct IbkrOrderRequest {
    acct_id: String,
    con_id: i64,
    order_type: String,
    side: String,
    quantity: f64,
    tif: String,
}

#[derive(Debug, Deserialize)]
pub struct IbkrOrderResponse {
    pub order_id: String,
    pub order_status: String,
}

impl IbkrBroker {
    /// Create new IBKR connection
    /// Gateway must be running at gateway_url (default: https://localhost:5000)
    pub fn new(gateway_url: String) -> Self {
        let client = reqwest::Client::builder()
            .danger_accept_invalid_certs(true) // Gateway uses self-signed cert
            .build()
            .unwrap();
        
        Self { gateway_url, client }
    }

    pub async fn get_accounts(&self) -> Result<Vec<IbkrAccount>> {
        let resp = self.client
            .get(format!("{}/v1/api/portfolio/accounts", self.gateway_url))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    pub async fn get_positions(&self, account_id: &str) -> Result<Vec<IbkrPosition>> {
        let resp = self.client
            .get(format!("{}/v1/api/portfolio/{}/positions/0", 
                self.gateway_url, account_id))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    pub async fn place_order(
        &self,
        account_id: &str,
        contract_id: i64,
        side: &str,
        quantity: f64,
    ) -> Result<IbkrOrderResponse> {
        let order = IbkrOrderRequest {
            acct_id: account_id.to_string(),
            con_id: contract_id,
            order_type: "MKT".to_string(),
            side: side.to_uppercase(),
            quantity,
            tif: "GTC".to_string(),
        };

        let resp = self.client
            .post(format!("{}/v1/api/iserver/account/{}/orders", 
                self.gateway_url, account_id))
            .json(&order)
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }

    pub async fn search_contract(&self, symbol: &str) -> Result<Vec<ContractInfo>> {
        let resp = self.client
            .get(format!("{}/v1/api/iserver/secdef/search?symbol={}", 
                self.gateway_url, symbol))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp)
    }
}

#[derive(Debug, Deserialize)]
pub struct ContractInfo {
    pub con_id: i64,
    pub symbol: String,
    pub sec_type: String,
    pub exchange: String,
}
