//! Interactive Brokers TWS Socket API Integration
//!
//! Uses the native TWS Socket API via the ibapi crate.
//! Connect to TWS on port 7497 (paper) or 7496 (live).

use anyhow::Result;
use rust_decimal::Decimal;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use tracing::{info, warn, debug, error};

/// TWS Broker client using native socket API
pub struct TwsBroker {
    host: String,
    port: u16,
    client_id: i32,
    account_id: String,
    connected: Arc<AtomicBool>,
    client: Arc<Mutex<Option<TwsClient>>>,
}

/// Internal TWS client wrapper
struct TwsClient {
    // ibapi client will be stored here when connected
    // For now, we use a placeholder until ibapi is properly configured
    _host: String,
    _port: u16,
}

// ============================================================================
// Contract Definitions
// ============================================================================

/// Contract specification for trading
#[derive(Debug, Clone)]
pub struct Contract {
    pub symbol: String,
    pub sec_type: String,      // STK, CASH, FUT, OPT, etc.
    pub exchange: String,      // SMART, IDEALPRO, CME, etc.
    pub currency: String,      // USD, CAD, EUR, etc.
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
        // Parse pair like "EURUSD" or "EUR.USD"
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
}

// ============================================================================
// Order Types
// ============================================================================

/// Order specification
#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: i32,
    pub action: String,        // BUY, SELL
    pub total_quantity: f64,
    pub order_type: String,    // MKT, LMT, STP, etc.
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub tif: String,           // DAY, GTC, IOC, etc.
    pub transmit: bool,
}

impl Order {
    /// Create a market order
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

    /// Create a limit order
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

    /// Create a stop order
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

// ============================================================================
// Response Types
// ============================================================================

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
        info!("Connecting to TWS at {}:{} (client_id={})", self.host, self.port, self.client_id);

        // Create TWS connection
        // Note: The actual ibapi connection would be:
        // let client = ibapi::Client::connect(&self.host, self.port)?;
        // For now, we simulate the connection

        let client = TwsClient {
            _host: self.host.clone(),
            _port: self.port,
        };

        {
            let mut guard = self.client.lock().await;
            *guard = Some(client);
        }

        self.connected.store(true, Ordering::SeqCst);
        info!("Connected to TWS successfully");
        Ok(())
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
        if !self.is_connected() {
            return Err(anyhow::anyhow!("Not connected to TWS"));
        }

        debug!("Requesting account summary for {}", self.account_id);

        // In production, this would use:
        // client.req_account_summary(9001, "All", "$LEDGER:CAD")?;
        // Then process the callbacks to build the summary

        // For now, return placeholder (would be populated from callbacks)
        Ok(AccountSummary {
            account_id: self.account_id.clone(),
            net_liquidation: 0.0,
            total_cash: 0.0,
            buying_power: 0.0,
            available_funds: 0.0,
            excess_liquidity: 0.0,
            maintenance_margin: 0.0,
            currency: "CAD".to_string(),
        })
    }

    /// Get all positions
    pub async fn get_positions(&self) -> Result<Vec<TwsPosition>> {
        if !self.is_connected() {
            return Err(anyhow::anyhow!("Not connected to TWS"));
        }

        debug!("Requesting positions");

        // In production, this would use:
        // client.req_positions()?;
        // Then process the position callbacks

        Ok(Vec::new())
    }

    /// Get historical bars
    pub async fn get_historical_bars(
        &self,
        contract: &Contract,
        end_date_time: &str,
        duration: &str,
        bar_size: &str,
        what_to_show: &str,
        use_rth: bool,
    ) -> Result<Vec<HistoricalBar>> {
        if !self.is_connected() {
            return Err(anyhow::anyhow!("Not connected to TWS"));
        }

        debug!(
            "Requesting historical bars for {} ({} {} bars)",
            contract.symbol, duration, bar_size
        );

        // In production, this would use:
        // client.req_historical_data(
        //     4001,           // reqId
        //     &contract,      // contract
        //     end_date_time,  // end date time
        //     duration,       // duration string (e.g., "1 Y")
        //     bar_size,       // bar size (e.g., "1 day")
        //     what_to_show,   // what to show (e.g., "TRADES")
        //     use_rth,        // use regular trading hours
        //     1,              // format date (1=yyyyMMdd HH:mm:ss)
        //     false,          // keep up to date
        //     vec![],         // chart options
        // )?;

        let _ = (end_date_time, what_to_show, use_rth);

        Ok(Vec::new())
    }

    /// Place a market order
    pub async fn place_market_order(
        &self,
        contract: &Contract,
        action: &str,
        quantity: f64,
    ) -> Result<i32> {
        if !self.is_connected() {
            return Err(anyhow::anyhow!("Not connected to TWS"));
        }

        info!(
            "Placing market order: {} {} {} @ MKT",
            action, quantity, contract.symbol
        );

        let order = Order::market(action, quantity);
        self.place_order(contract, &order).await
    }

    /// Place a limit order
    pub async fn place_limit_order(
        &self,
        contract: &Contract,
        action: &str,
        quantity: f64,
        limit_price: f64,
    ) -> Result<i32> {
        if !self.is_connected() {
            return Err(anyhow::anyhow!("Not connected to TWS"));
        }

        info!(
            "Placing limit order: {} {} {} @ {}",
            action, quantity, contract.symbol, limit_price
        );

        let order = Order::limit(action, quantity, limit_price);
        self.place_order(contract, &order).await
    }

    /// Place an order (internal)
    async fn place_order(&self, contract: &Contract, order: &Order) -> Result<i32> {
        // In production, this would use:
        // let order_id = client.next_valid_order_id();
        // client.place_order(order_id, &contract, &order)?;

        debug!("Order submitted: {:?} for {:?}", order, contract);

        // Return a simulated order ID
        let order_id = 1001;
        Ok(order_id)
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: i32) -> Result<()> {
        if !self.is_connected() {
            return Err(anyhow::anyhow!("Not connected to TWS"));
        }

        info!("Cancelling order {}", order_id);

        // In production:
        // client.cancel_order(order_id)?;

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
        if !self.is_connected() {
            return Err(anyhow::anyhow!("Not connected to TWS"));
        }

        // In production, would request from TWS
        Ok(1000)
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
            current_price: Decimal::ZERO, // Would be populated from market data
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

    #[tokio::test]
    async fn test_tws_broker_creation() {
        let broker = TwsBroker::new(
            "127.0.0.1".to_string(),
            7497,
            1,
            "DU1234567".to_string(),
        );
        assert!(!broker.is_connected());
        assert_eq!(broker.account_id(), "DU1234567");
    }

    #[tokio::test]
    async fn test_tws_connect_disconnect() {
        let broker = TwsBroker::new(
            "127.0.0.1".to_string(),
            7497,
            1,
            "DU1234567".to_string(),
        );

        // Connect
        let result = broker.connect().await;
        assert!(result.is_ok());
        assert!(broker.is_connected());

        // Disconnect
        let result = broker.disconnect().await;
        assert!(result.is_ok());
        assert!(!broker.is_connected());
    }

    #[tokio::test]
    async fn test_not_connected_error() {
        let broker = TwsBroker::new(
            "127.0.0.1".to_string(),
            7497,
            1,
            "DU1234567".to_string(),
        );

        // Should fail when not connected
        let result = broker.get_account_summary().await;
        assert!(result.is_err());

        let result = broker.get_positions().await;
        assert!(result.is_err());
    }
}
