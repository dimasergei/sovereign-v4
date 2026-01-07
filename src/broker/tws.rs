//! Interactive Brokers TWS Socket API Integration
//!
//! Uses the native TWS Socket API via the ibapi crate.
//! Connect to TWS on port 7497 (paper) or 7496 (live).

use anyhow::Result;
use rust_decimal::Decimal;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use tracing::{info, warn, debug};

use ibapi::Client;
use ibapi::contracts::Contract as IbContract;
use ibapi::market_data::historical::{BarSize, WhatToShow, Duration as IbDuration};
use ibapi::market_data::TradingHours;
use ibapi::accounts::{AccountSummaryResult, PositionUpdate};
use ibapi::accounts::types::AccountGroup;

/// TWS Broker client using native socket API
pub struct TwsBroker {
    host: String,
    port: u16,
    client_id: i32,
    account_id: String,
    connected: Arc<AtomicBool>,
    client: Arc<Mutex<Option<Client>>>,
}

// ============================================================================
// Our Local Types (for API compatibility with rest of codebase)
// ============================================================================

/// Contract specification for trading (local type)
#[derive(Debug, Clone)]
pub struct Contract {
    pub symbol: String,
    pub sec_type: String,
    pub exchange: String,
    pub currency: String,
    pub primary_exchange: Option<String>,
    pub local_symbol: Option<String>,
    pub con_id: Option<i64>,
}

impl Contract {
    /// Create a stock contract
    pub fn stock(symbol: &str, exchange: &str, currency: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            sec_type: "STK".to_string(),
            exchange: exchange.to_string(),
            currency: currency.to_string(),
            primary_exchange: None,
            local_symbol: None,
            con_id: None,
        }
    }

    /// Create a forex contract
    pub fn forex(pair: &str) -> Self {
        let (base, quote) = if pair.contains('.') {
            let parts: Vec<&str> = pair.split('.').collect();
            (parts[0], parts[1])
        } else if pair.len() == 6 {
            (&pair[..3], &pair[3..])
        } else {
            (pair, "USD")
        };

        Self {
            symbol: base.to_string(),
            sec_type: "CASH".to_string(),
            exchange: "IDEALPRO".to_string(),
            currency: quote.to_string(),
            primary_exchange: None,
            local_symbol: None,
            con_id: None,
        }
    }

    /// Convert to ibapi Contract
    fn to_ibapi(&self) -> IbContract {
        if self.sec_type == "CASH" {
            IbContract::forex(&self.symbol, &self.currency).build()
        } else {
            let mut builder = IbContract::stock(&self.symbol);
            if self.exchange != "SMART" {
                builder = builder.on_exchange(&self.exchange);
            }
            if self.currency != "USD" {
                builder = builder.in_currency(&self.currency);
            }
            builder.build()
        }
    }
}

/// Order specification (local type for compatibility)
#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: i32,
    pub action: String,
    pub total_quantity: f64,
    pub order_type: String,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub tif: String,
    pub transmit: bool,
}

impl Order {
    pub fn market(action: &str, quantity: f64) -> Self {
        Self {
            order_id: 0,
            action: action.to_uppercase(),
            total_quantity: quantity,
            order_type: "MKT".to_string(),
            limit_price: None,
            stop_price: None,
            tif: "DAY".to_string(),
            transmit: true,
        }
    }

    pub fn limit(action: &str, quantity: f64, price: f64) -> Self {
        Self {
            order_id: 0,
            action: action.to_uppercase(),
            total_quantity: quantity,
            order_type: "LMT".to_string(),
            limit_price: Some(price),
            stop_price: None,
            tif: "DAY".to_string(),
            transmit: true,
        }
    }

    pub fn stop(action: &str, quantity: f64, stop_price: f64) -> Self {
        Self {
            order_id: 0,
            action: action.to_uppercase(),
            total_quantity: quantity,
            order_type: "STP".to_string(),
            limit_price: None,
            stop_price: Some(stop_price),
            tif: "DAY".to_string(),
            transmit: true,
        }
    }
}

/// Account summary data
#[derive(Debug, Clone, Default)]
pub struct AccountSummary {
    pub account_id: String,
    pub net_liquidation: f64,
    pub total_cash: f64,
    pub buying_power: f64,
    pub available_funds: f64,
    pub excess_liquidity: f64,
    pub maintenance_margin: f64,
    pub currency: String,
}

/// Position data
#[derive(Debug, Clone)]
pub struct TwsPosition {
    pub account: String,
    pub contract: Contract,
    pub position: f64,
    pub avg_cost: f64,
    pub unrealized_pnl: Option<f64>,
    pub market_value: Option<f64>,
}

/// Historical bar data
#[derive(Debug, Clone)]
pub struct HistoricalBar {
    pub time: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
    pub wap: f64,
    pub count: i32,
}

/// Order status
#[derive(Debug, Clone)]
pub struct OrderStatus {
    pub order_id: i32,
    pub status: String,
    pub filled: f64,
    pub remaining: f64,
    pub avg_fill_price: f64,
    pub perm_id: i64,
    pub parent_id: i32,
    pub last_fill_price: f64,
    pub client_id: i32,
    pub why_held: String,
}

// ============================================================================
// TwsBroker Implementation
// ============================================================================

impl TwsBroker {
    /// Create a new TWS broker client
    pub fn new(host: String, port: u16, client_id: i32, account_id: String) -> Self {
        Self {
            host,
            port,
            client_id,
            account_id,
            connected: Arc::new(AtomicBool::new(false)),
            client: Arc::new(Mutex::new(None)),
        }
    }

    /// Connect to TWS
    pub async fn connect(&self) -> Result<()> {
        let address = format!("{}:{}", self.host, self.port);
        info!("Connecting to TWS at {} (client_id={})", address, self.client_id);

        match Client::connect(&address, self.client_id).await {
            Ok(client) => {
                info!("Connected to TWS successfully (server version: {})", client.server_version());

                {
                    let mut guard = self.client.lock().await;
                    *guard = Some(client);
                }

                self.connected.store(true, Ordering::SeqCst);
                Ok(())
            }
            Err(e) => {
                warn!("Failed to connect to TWS: {}", e);
                Err(anyhow::anyhow!("TWS connection failed: {}", e))
            }
        }
    }

    /// Disconnect from TWS
    pub async fn disconnect(&self) -> Result<()> {
        info!("Disconnecting from TWS");

        {
            let mut guard = self.client.lock().await;
            *guard = None;
        }

        self.connected.store(false, Ordering::SeqCst);
        info!("Disconnected from TWS");
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Get account summary
    pub async fn get_account_summary(&self) -> Result<AccountSummary> {
        let guard = self.client.lock().await;
        let client = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Not connected to TWS"))?;

        debug!("Requesting account summary for {}", self.account_id);

        let tags = &[
            "NetLiquidation",
            "TotalCashValue",
            "BuyingPower",
            "AvailableFunds",
            "ExcessLiquidity",
            "MaintMarginReq",
        ];

        let mut subscription = client
            .account_summary(&AccountGroup("All".to_string()), tags)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request account summary: {}", e))?;

        let mut summary = AccountSummary {
            account_id: self.account_id.clone(),
            currency: "USD".to_string(),
            ..Default::default()
        };

        // Collect values from subscription using async iteration
        while let Some(result) = subscription.next().await {
            match result {
                Ok(AccountSummaryResult::Summary(s)) => {
                    if s.account == self.account_id || self.account_id.is_empty() {
                        summary.account_id = s.account.clone();
                        match s.tag.as_str() {
                            "NetLiquidation" => summary.net_liquidation = s.value.parse().unwrap_or(0.0),
                            "TotalCashValue" => summary.total_cash = s.value.parse().unwrap_or(0.0),
                            "BuyingPower" => summary.buying_power = s.value.parse().unwrap_or(0.0),
                            "AvailableFunds" => summary.available_funds = s.value.parse().unwrap_or(0.0),
                            "ExcessLiquidity" => summary.excess_liquidity = s.value.parse().unwrap_or(0.0),
                            "MaintMarginReq" => summary.maintenance_margin = s.value.parse().unwrap_or(0.0),
                            _ => {}
                        }
                        summary.currency = s.currency.clone();
                    }
                }
                Ok(AccountSummaryResult::End) => break,
                Err(e) => {
                    warn!("Error receiving account summary: {}", e);
                    break;
                }
            }
        }

        info!("Account summary: NAV=${:.2}, Cash=${:.2}, BP=${:.2}",
            summary.net_liquidation, summary.total_cash, summary.buying_power);

        Ok(summary)
    }

    /// Get all positions
    pub async fn get_positions(&self) -> Result<Vec<TwsPosition>> {
        let guard = self.client.lock().await;
        let client = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Not connected to TWS"))?;

        debug!("Requesting positions");

        let mut subscription = client
            .positions()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to request positions: {}", e))?;

        let mut positions = Vec::new();

        while let Some(result) = subscription.next().await {
            match result {
                Ok(PositionUpdate::Position(pos)) => {
                    let contract = Contract {
                        symbol: pos.contract.symbol.to_string(),
                        sec_type: pos.contract.security_type.to_string(),
                        exchange: pos.contract.exchange.to_string(),
                        currency: pos.contract.currency.to_string(),
                        primary_exchange: Some(pos.contract.primary_exchange.to_string()),
                        local_symbol: Some(pos.contract.local_symbol.clone()),
                        con_id: Some(pos.contract.contract_id as i64),
                    };

                    positions.push(TwsPosition {
                        account: pos.account.clone(),
                        contract,
                        position: pos.position,
                        avg_cost: pos.average_cost,
                        unrealized_pnl: None,
                        market_value: None,
                    });
                }
                Ok(PositionUpdate::PositionEnd) => break,
                Err(e) => {
                    warn!("Error receiving positions: {}", e);
                    break;
                }
            }
        }

        info!("Retrieved {} positions", positions.len());
        Ok(positions)
    }

    /// Get historical bars
    pub async fn get_historical_bars(
        &self,
        contract: &Contract,
        _end_date_time: &str,
        duration: &str,
        bar_size: &str,
        _what_to_show: &str,
        use_rth: bool,
    ) -> Result<Vec<HistoricalBar>> {
        let guard = self.client.lock().await;
        let client = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Not connected to TWS"))?;

        debug!("Requesting historical bars for {} ({} {})", contract.symbol, duration, bar_size);

        let ib_contract = contract.to_ibapi();

        // Parse duration string like "1 Y", "1 D", "1 M"
        let duration_val = parse_duration(duration);

        // Parse bar size string like "1 day", "1 hour", "5 mins"
        let bar_size_val = parse_bar_size(bar_size);

        let trading_hours = if use_rth {
            TradingHours::Regular
        } else {
            TradingHours::Extended
        };

        let historical_data = client
            .historical_data(
                &ib_contract,
                None, // end_date (None = now)
                duration_val,
                bar_size_val,
                Some(WhatToShow::Trades),
                trading_hours,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch historical data: {}", e))?;

        let bars: Vec<HistoricalBar> = historical_data
            .bars
            .iter()
            .map(|b| HistoricalBar {
                time: b.date.to_string(),
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
                volume: b.volume as i64,
                wap: b.wap,
                count: b.count,
            })
            .collect();

        info!("Retrieved {} historical bars for {}", bars.len(), contract.symbol);
        Ok(bars)
    }

    /// Place a market order
    pub async fn place_market_order(
        &self,
        contract: &Contract,
        action: &str,
        quantity: f64,
    ) -> Result<i32> {
        let guard = self.client.lock().await;
        let client = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Not connected to TWS"))?;

        info!("Placing market order: {} {} {} @ MKT", action, quantity, contract.symbol);

        let ib_contract = contract.to_ibapi();
        let qty = quantity.round() as i32;

        let order_id = if action.to_uppercase() == "BUY" {
            client
                .order(&ib_contract)
                .buy(qty)
                .market()
                .submit()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to place buy order: {}", e))?
        } else {
            client
                .order(&ib_contract)
                .sell(qty)
                .market()
                .submit()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to place sell order: {}", e))?
        };

        info!("Market order submitted with ID: {}", order_id);
        Ok(order_id.into())
    }

    /// Place a limit order
    pub async fn place_limit_order(
        &self,
        contract: &Contract,
        action: &str,
        quantity: f64,
        limit_price: f64,
    ) -> Result<i32> {
        let guard = self.client.lock().await;
        let client = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Not connected to TWS"))?;

        info!("Placing limit order: {} {} {} @ {}", action, quantity, contract.symbol, limit_price);

        let ib_contract = contract.to_ibapi();
        let qty = quantity.round() as i32;

        let order_id = if action.to_uppercase() == "BUY" {
            client
                .order(&ib_contract)
                .buy(qty)
                .limit(limit_price)
                .submit()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to place buy limit order: {}", e))?
        } else {
            client
                .order(&ib_contract)
                .sell(qty)
                .limit(limit_price)
                .submit()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to place sell limit order: {}", e))?
        };

        info!("Limit order submitted with ID: {}", order_id);
        Ok(order_id.into())
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: i32) -> Result<()> {
        let guard = self.client.lock().await;
        let client = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Not connected to TWS"))?;

        info!("Cancelling order {}", order_id);

        let _subscription = client
            .cancel_order(order_id, "")
            .await
            .map_err(|e| anyhow::anyhow!("Failed to cancel order: {}", e))?;

        info!("Order {} cancelled", order_id);
        Ok(())
    }

    /// Create a stock contract
    pub fn make_stock_contract(&self, symbol: &str, exchange: &str, currency: &str) -> Contract {
        Contract::stock(symbol, exchange, currency)
    }

    /// Create a forex contract
    pub fn make_forex_contract(&self, pair: &str) -> Contract {
        Contract::forex(pair)
    }

    /// Get the account ID
    pub fn account_id(&self) -> &str {
        &self.account_id
    }

    /// Get next valid order ID
    pub async fn next_order_id(&self) -> Result<i32> {
        let guard = self.client.lock().await;
        let client = guard.as_ref().ok_or_else(|| anyhow::anyhow!("Not connected to TWS"))?;
        Ok(client.next_order_id())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse duration string like "1 Y", "1 D", "1 M" to ibapi Duration
fn parse_duration(s: &str) -> IbDuration {
    let parts: Vec<&str> = s.trim().split_whitespace().collect();
    if parts.len() != 2 {
        return IbDuration::DAY;
    }

    let num: i32 = parts[0].parse().unwrap_or(1);
    match parts[1].to_uppercase().as_str() {
        "Y" | "YEAR" | "YEARS" => IbDuration::years(num),
        "M" | "MONTH" | "MONTHS" => IbDuration::months(num),
        "W" | "WEEK" | "WEEKS" => IbDuration::weeks(num),
        "D" | "DAY" | "DAYS" => IbDuration::days(num),
        "H" | "HOUR" | "HOURS" => IbDuration::seconds(num * 3600),
        _ => IbDuration::days(num),
    }
}

/// Parse bar size string like "1 day", "1 hour", "5 mins"
fn parse_bar_size(s: &str) -> BarSize {
    let s_lower = s.to_lowercase();
    if s_lower.contains("day") {
        BarSize::Day
    } else if s_lower.contains("hour") {
        BarSize::Hour
    } else if s_lower.contains("min") {
        if s_lower.contains("30") {
            BarSize::Min30
        } else if s_lower.contains("15") {
            BarSize::Min15
        } else if s_lower.contains("5") {
            BarSize::Min5
        } else {
            BarSize::Min
        }
    } else if s_lower.contains("sec") {
        BarSize::Sec30
    } else {
        BarSize::Day
    }
}

// ============================================================================
// Helper to convert to unified broker position
// ============================================================================

impl From<&TwsPosition> for super::BrokerPosition {
    fn from(p: &TwsPosition) -> Self {
        let side = if p.position >= 0.0 {
            super::PositionSide::Long
        } else {
            super::PositionSide::Short
        };

        Self {
            symbol: p.contract.symbol.clone(),
            side,
            quantity: Decimal::from_f64_retain(p.position.abs()).unwrap_or(Decimal::ZERO),
            entry_price: Decimal::from_f64_retain(p.avg_cost).unwrap_or(Decimal::ZERO),
            current_price: Decimal::ZERO,
            unrealized_pnl: Decimal::from_f64_retain(p.unrealized_pnl.unwrap_or(0.0))
                .unwrap_or(Decimal::ZERO),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stock_contract() {
        let contract = Contract::stock("AAPL", "SMART", "USD");
        assert_eq!(contract.symbol, "AAPL");
        assert_eq!(contract.sec_type, "STK");
        assert_eq!(contract.exchange, "SMART");
        assert_eq!(contract.currency, "USD");
    }

    #[test]
    fn test_forex_contract() {
        let contract = Contract::forex("EURUSD");
        assert_eq!(contract.symbol, "EUR");
        assert_eq!(contract.sec_type, "CASH");
        assert_eq!(contract.exchange, "IDEALPRO");
        assert_eq!(contract.currency, "USD");
    }

    #[test]
    fn test_forex_contract_with_dot() {
        let contract = Contract::forex("EUR.CAD");
        assert_eq!(contract.symbol, "EUR");
        assert_eq!(contract.currency, "CAD");
    }

    #[test]
    fn test_market_order() {
        let order = Order::market("BUY", 100.0);
        assert_eq!(order.action, "BUY");
        assert_eq!(order.total_quantity, 100.0);
        assert_eq!(order.order_type, "MKT");
        assert!(order.limit_price.is_none());
    }

    #[test]
    fn test_limit_order() {
        let order = Order::limit("SELL", 50.0, 150.50);
        assert_eq!(order.action, "SELL");
        assert_eq!(order.total_quantity, 50.0);
        assert_eq!(order.order_type, "LMT");
        assert_eq!(order.limit_price, Some(150.50));
    }

    #[test]
    fn test_stop_order() {
        let order = Order::stop("SELL", 100.0, 145.00);
        assert_eq!(order.action, "SELL");
        assert_eq!(order.order_type, "STP");
        assert_eq!(order.stop_price, Some(145.00));
    }

    #[test]
    fn test_parse_duration() {
        // Test that parse_duration returns the expected duration types
        let d = parse_duration("1 D");
        assert_eq!(d.to_string(), "1 D");

        let y = parse_duration("1 Y");
        assert_eq!(y.to_string(), "1 Y");

        let w = parse_duration("2 W");
        assert_eq!(w.to_string(), "2 W");

        let m = parse_duration("3 M");
        assert_eq!(m.to_string(), "3 M");
    }

    #[test]
    fn test_parse_bar_size() {
        assert!(matches!(parse_bar_size("1 day"), BarSize::Day));
        assert!(matches!(parse_bar_size("1 hour"), BarSize::Hour));
        assert!(matches!(parse_bar_size("5 mins"), BarSize::Min5));
        assert!(matches!(parse_bar_size("15 mins"), BarSize::Min15));
    }

    #[test]
    fn test_tws_broker_creation() {
        let broker = TwsBroker::new(
            "192.168.64.1".to_string(),
            7497,
            1,
            "DU1234567".to_string(),
        );
        assert!(!broker.is_connected());
        assert_eq!(broker.account_id(), "DU1234567");
    }
}
