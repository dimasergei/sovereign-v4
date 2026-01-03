//! Prime Broker Module - FIX Protocol Interface
//!
//! Provides institutional connectivity via FIX protocol including:
//! - FIX session management
//! - Order submission, modification, and cancellation
//! - Execution report handling
//! - Position and balance queries
//!
//! This module defines the interface for prime broker connectivity.
//! Actual FIX implementation would require a FIX engine library.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::BrokerError;

/// FIX message field codes (subset)
pub mod fix_fields {
    pub const MSG_TYPE: u32 = 35;
    pub const SENDER_COMP_ID: u32 = 49;
    pub const TARGET_COMP_ID: u32 = 56;
    pub const CL_ORD_ID: u32 = 11;
    pub const ORDER_ID: u32 = 37;
    pub const EXEC_ID: u32 = 17;
    pub const EXEC_TYPE: u32 = 150;
    pub const ORD_STATUS: u32 = 39;
    pub const SYMBOL: u32 = 55;
    pub const SIDE: u32 = 54;
    pub const ORDER_QTY: u32 = 38;
    pub const ORD_TYPE: u32 = 40;
    pub const PRICE: u32 = 44;
    pub const STOP_PX: u32 = 99;
    pub const TIME_IN_FORCE: u32 = 59;
    pub const AVG_PX: u32 = 6;
    pub const CUM_QTY: u32 = 14;
    pub const LEAVES_QTY: u32 = 151;
    pub const LAST_QTY: u32 = 32;
    pub const LAST_PX: u32 = 31;
    pub const TEXT: u32 = 58;
}

/// FIX message types
pub mod fix_msg_types {
    pub const HEARTBEAT: &str = "0";
    pub const LOGON: &str = "A";
    pub const LOGOUT: &str = "5";
    pub const NEW_ORDER_SINGLE: &str = "D";
    pub const ORDER_CANCEL_REQUEST: &str = "F";
    pub const ORDER_CANCEL_REPLACE: &str = "G";
    pub const EXECUTION_REPORT: &str = "8";
    pub const ORDER_STATUS_REQUEST: &str = "H";
    pub const REJECT: &str = "3";
    pub const BUSINESS_REJECT: &str = "j";
}

/// Order side for FIX
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
    BuyMinus,  // Short sale
    SellShort,
}

impl OrderSide {
    /// Convert to FIX side value
    pub fn to_fix(&self) -> char {
        match self {
            OrderSide::Buy => '1',
            OrderSide::Sell => '2',
            OrderSide::BuyMinus => '3',
            OrderSide::SellShort => '5',
        }
    }

    /// Parse from FIX side value
    pub fn from_fix(c: char) -> Option<Self> {
        match c {
            '1' => Some(OrderSide::Buy),
            '2' => Some(OrderSide::Sell),
            '3' => Some(OrderSide::BuyMinus),
            '5' => Some(OrderSide::SellShort),
            _ => None,
        }
    }
}

/// Time in force for FIX
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    Gtc,
    Ioc,
    Fok,
    Gtd,
    AtTheOpening,
    AtTheClose,
}

impl TimeInForce {
    /// Convert to FIX TIF value
    pub fn to_fix(&self) -> char {
        match self {
            TimeInForce::Day => '0',
            TimeInForce::Gtc => '1',
            TimeInForce::Ioc => '3',
            TimeInForce::Fok => '4',
            TimeInForce::Gtd => '6',
            TimeInForce::AtTheOpening => '2',
            TimeInForce::AtTheClose => '7',
        }
    }

    /// Parse from FIX TIF value
    pub fn from_fix(c: char) -> Option<Self> {
        match c {
            '0' => Some(TimeInForce::Day),
            '1' => Some(TimeInForce::Gtc),
            '3' => Some(TimeInForce::Ioc),
            '4' => Some(TimeInForce::Fok),
            '6' => Some(TimeInForce::Gtd),
            '2' => Some(TimeInForce::AtTheOpening),
            '7' => Some(TimeInForce::AtTheClose),
            _ => None,
        }
    }
}

/// FIX order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixOrderStatus {
    New,
    PartiallyFilled,
    Filled,
    DoneForDay,
    Canceled,
    Replaced,
    PendingCancel,
    Stopped,
    Rejected,
    Suspended,
    PendingNew,
    Calculated,
    Expired,
    AcceptedForBidding,
    PendingReplace,
}

impl FixOrderStatus {
    /// Parse from FIX order status value
    pub fn from_fix(c: char) -> Option<Self> {
        match c {
            '0' => Some(FixOrderStatus::New),
            '1' => Some(FixOrderStatus::PartiallyFilled),
            '2' => Some(FixOrderStatus::Filled),
            '3' => Some(FixOrderStatus::DoneForDay),
            '4' => Some(FixOrderStatus::Canceled),
            '5' => Some(FixOrderStatus::Replaced),
            '6' => Some(FixOrderStatus::PendingCancel),
            '7' => Some(FixOrderStatus::Stopped),
            '8' => Some(FixOrderStatus::Rejected),
            '9' => Some(FixOrderStatus::Suspended),
            'A' => Some(FixOrderStatus::PendingNew),
            'B' => Some(FixOrderStatus::Calculated),
            'C' => Some(FixOrderStatus::Expired),
            'D' => Some(FixOrderStatus::AcceptedForBidding),
            'E' => Some(FixOrderStatus::PendingReplace),
            _ => None,
        }
    }

    /// Convert to FIX status value
    pub fn to_fix(&self) -> char {
        match self {
            FixOrderStatus::New => '0',
            FixOrderStatus::PartiallyFilled => '1',
            FixOrderStatus::Filled => '2',
            FixOrderStatus::DoneForDay => '3',
            FixOrderStatus::Canceled => '4',
            FixOrderStatus::Replaced => '5',
            FixOrderStatus::PendingCancel => '6',
            FixOrderStatus::Stopped => '7',
            FixOrderStatus::Rejected => '8',
            FixOrderStatus::Suspended => '9',
            FixOrderStatus::PendingNew => 'A',
            FixOrderStatus::Calculated => 'B',
            FixOrderStatus::Expired => 'C',
            FixOrderStatus::AcceptedForBidding => 'D',
            FixOrderStatus::PendingReplace => 'E',
        }
    }

    /// Check if status is terminal
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            FixOrderStatus::Filled
                | FixOrderStatus::Canceled
                | FixOrderStatus::Rejected
                | FixOrderStatus::Expired
                | FixOrderStatus::DoneForDay
        )
    }
}

/// FIX execution type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecType {
    New,
    DoneForDay,
    Canceled,
    Replaced,
    PendingCancel,
    Stopped,
    Rejected,
    Suspended,
    PendingNew,
    Calculated,
    Expired,
    Restated,
    PendingReplace,
    Trade,
    TradeCorrect,
    TradeCancel,
    OrderStatus,
    TradeInClearingHold,
    TradeReleasedToClearing,
    TriggeredOrActivated,
}

impl ExecType {
    /// Parse from FIX exec type value
    pub fn from_fix(c: char) -> Option<Self> {
        match c {
            '0' => Some(ExecType::New),
            '3' => Some(ExecType::DoneForDay),
            '4' => Some(ExecType::Canceled),
            '5' => Some(ExecType::Replaced),
            '6' => Some(ExecType::PendingCancel),
            '7' => Some(ExecType::Stopped),
            '8' => Some(ExecType::Rejected),
            '9' => Some(ExecType::Suspended),
            'A' => Some(ExecType::PendingNew),
            'B' => Some(ExecType::Calculated),
            'C' => Some(ExecType::Expired),
            'D' => Some(ExecType::Restated),
            'E' => Some(ExecType::PendingReplace),
            'F' => Some(ExecType::Trade),
            'G' => Some(ExecType::TradeCorrect),
            'H' => Some(ExecType::TradeCancel),
            'I' => Some(ExecType::OrderStatus),
            'J' => Some(ExecType::TradeInClearingHold),
            'K' => Some(ExecType::TradeReleasedToClearing),
            'L' => Some(ExecType::TriggeredOrActivated),
            _ => None,
        }
    }
}

/// FIX session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSessionConfig {
    /// Sender comp ID
    pub sender_comp_id: String,
    /// Target comp ID
    pub target_comp_id: String,
    /// FIX host
    pub host: String,
    /// FIX port
    pub port: u16,
    /// Heartbeat interval (seconds)
    pub heartbeat_interval: u32,
    /// Use SSL/TLS
    pub use_ssl: bool,
    /// Reset sequence on logon
    pub reset_on_logon: bool,
    /// FIX version (e.g., "FIX.4.2", "FIX.4.4")
    pub fix_version: String,
}

impl Default for FixSessionConfig {
    fn default() -> Self {
        Self {
            sender_comp_id: "CLIENT".to_string(),
            target_comp_id: "BROKER".to_string(),
            host: "localhost".to_string(),
            port: 9878,
            heartbeat_interval: 30,
            use_ssl: true,
            reset_on_logon: true,
            fix_version: "FIX.4.4".to_string(),
        }
    }
}

/// Order for prime broker submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeOrder {
    /// Client order ID
    pub cl_ord_id: String,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: OrderSide,
    /// Quantity
    pub quantity: Decimal,
    /// Order type (1=Market, 2=Limit, etc.)
    pub ord_type: char,
    /// Limit price (if applicable)
    pub price: Option<Decimal>,
    /// Stop price (if applicable)
    pub stop_px: Option<Decimal>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Account
    pub account: Option<String>,
    /// Algo parameters
    pub algo_params: HashMap<String, String>,
    /// Target venues (comma-separated)
    pub ex_destination: Option<String>,
}

impl PrimeOrder {
    /// Create a new market order
    pub fn market(cl_ord_id: &str, symbol: &str, side: OrderSide, quantity: Decimal) -> Self {
        Self {
            cl_ord_id: cl_ord_id.to_string(),
            symbol: symbol.to_string(),
            side,
            quantity,
            ord_type: '1',  // Market
            price: None,
            stop_px: None,
            time_in_force: TimeInForce::Day,
            account: None,
            algo_params: HashMap::new(),
            ex_destination: None,
        }
    }

    /// Create a new limit order
    pub fn limit(
        cl_ord_id: &str,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> Self {
        Self {
            cl_ord_id: cl_ord_id.to_string(),
            symbol: symbol.to_string(),
            side,
            quantity,
            ord_type: '2',  // Limit
            price: Some(price),
            stop_px: None,
            time_in_force: TimeInForce::Day,
            account: None,
            algo_params: HashMap::new(),
            ex_destination: None,
        }
    }

    /// Set time in force
    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Set account
    pub fn with_account(mut self, account: &str) -> Self {
        self.account = Some(account.to_string());
        self
    }

    /// Set algo parameter
    pub fn with_algo_param(mut self, key: &str, value: &str) -> Self {
        self.algo_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Set execution destination
    pub fn with_destination(mut self, dest: &str) -> Self {
        self.ex_destination = Some(dest.to_string());
        self
    }
}

/// Order acknowledgement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAck {
    /// Client order ID
    pub cl_ord_id: String,
    /// Broker order ID
    pub order_id: String,
    /// Status
    pub status: FixOrderStatus,
    /// Text/description
    pub text: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Cancel acknowledgement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelAck {
    /// Client order ID
    pub cl_ord_id: String,
    /// Original order ID
    pub orig_cl_ord_id: String,
    /// Broker order ID
    pub order_id: String,
    /// Success
    pub success: bool,
    /// Reason (if failed)
    pub reason: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Modify acknowledgement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifyAck {
    /// Client order ID (new)
    pub cl_ord_id: String,
    /// Original order ID
    pub orig_cl_ord_id: String,
    /// Broker order ID
    pub order_id: String,
    /// Success
    pub success: bool,
    /// New quantity (if modified)
    pub new_qty: Option<Decimal>,
    /// New price (if modified)
    pub new_price: Option<Decimal>,
    /// Reason (if failed)
    pub reason: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Order modification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderModification {
    /// New quantity (if changing)
    pub new_qty: Option<Decimal>,
    /// New price (if changing)
    pub new_price: Option<Decimal>,
    /// New stop price (if changing)
    pub new_stop_px: Option<Decimal>,
}

/// Order status from prime broker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderStatus {
    /// Broker order ID
    pub order_id: String,
    /// Client order ID
    pub cl_ord_id: String,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: OrderSide,
    /// Status
    pub status: FixOrderStatus,
    /// Order quantity
    pub order_qty: Decimal,
    /// Cumulative filled quantity
    pub cum_qty: Decimal,
    /// Leaves quantity
    pub leaves_qty: Decimal,
    /// Average price
    pub avg_px: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Execution report from FIX
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReportFix {
    /// Execution ID
    pub exec_id: String,
    /// Broker order ID
    pub order_id: String,
    /// Client order ID
    pub cl_ord_id: String,
    /// Execution type
    pub exec_type: ExecType,
    /// Order status
    pub ord_status: FixOrderStatus,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: OrderSide,
    /// Last quantity (this fill)
    pub last_qty: Decimal,
    /// Last price (this fill)
    pub last_px: Decimal,
    /// Cumulative quantity
    pub cum_qty: Decimal,
    /// Leaves quantity
    pub leaves_qty: Decimal,
    /// Average price
    pub avg_px: Decimal,
    /// Execution venue
    pub last_mkt: Option<String>,
    /// Commission
    pub commission: Decimal,
    /// Text
    pub text: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Account balance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountBalance {
    /// Account ID
    pub account: String,
    /// Cash balance
    pub cash: Decimal,
    /// Buying power
    pub buying_power: Decimal,
    /// Margin used
    pub margin_used: Decimal,
    /// Portfolio value
    pub portfolio_value: Decimal,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Position from prime broker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Quantity (positive = long, negative = short)
    pub quantity: Decimal,
    /// Average cost
    pub avg_cost: Decimal,
    /// Current price
    pub current_price: Decimal,
    /// Market value
    pub market_value: Decimal,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
}

/// Prime broker trait
#[async_trait]
pub trait PrimeBroker: Send + Sync {
    /// Connect to the prime broker
    async fn connect(&mut self) -> Result<(), BrokerError>;

    /// Disconnect from the prime broker
    async fn disconnect(&mut self) -> Result<(), BrokerError>;

    /// Check if connected
    fn is_connected(&self) -> bool;

    /// Submit a new order
    async fn submit_order(&self, order: PrimeOrder) -> Result<OrderAck, BrokerError>;

    /// Cancel an order
    async fn cancel_order(&self, order_id: &str) -> Result<CancelAck, BrokerError>;

    /// Modify an order
    async fn modify_order(
        &self,
        order_id: &str,
        modification: OrderModification,
    ) -> Result<ModifyAck, BrokerError>;

    /// Get order status
    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatus, BrokerError>;

    /// Get account balance
    async fn get_balance(&self) -> Result<AccountBalance, BrokerError>;

    /// Get positions
    async fn get_positions(&self) -> Result<Vec<Position>, BrokerError>;

    /// Get available venues
    fn available_venues(&self) -> Vec<String>;

    /// Subscribe to execution reports
    async fn subscribe_executions(
        &self,
        callback: Box<dyn Fn(ExecutionReportFix) + Send + Sync>,
    ) -> Result<(), BrokerError>;
}

/// Mock prime broker for testing
pub struct MockPrimeBroker {
    config: FixSessionConfig,
    connected: bool,
    orders: HashMap<String, OrderStatus>,
    positions: Vec<Position>,
    balance: AccountBalance,
    next_order_id: u64,
}

impl MockPrimeBroker {
    /// Create a new mock broker
    pub fn new(config: FixSessionConfig) -> Self {
        Self {
            config,
            connected: false,
            orders: HashMap::new(),
            positions: Vec::new(),
            balance: AccountBalance {
                account: "MOCK".to_string(),
                cash: dec!(1_000_000),
                buying_power: dec!(2_000_000),
                margin_used: dec!(0),
                portfolio_value: dec!(1_000_000),
                unrealized_pnl: dec!(0),
                timestamp: Utc::now(),
            },
            next_order_id: 1000,
        }
    }

    /// Add a mock position
    pub fn add_position(&mut self, position: Position) {
        self.positions.push(position);
    }

    /// Set balance
    pub fn set_balance(&mut self, balance: AccountBalance) {
        self.balance = balance;
    }

    fn next_order_id(&mut self) -> String {
        self.next_order_id += 1;
        format!("ORD{}", self.next_order_id)
    }
}

#[async_trait]
impl PrimeBroker for MockPrimeBroker {
    async fn connect(&mut self) -> Result<(), BrokerError> {
        self.connected = true;
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<(), BrokerError> {
        self.connected = false;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    async fn submit_order(&self, order: PrimeOrder) -> Result<OrderAck, BrokerError> {
        if !self.connected {
            return Err(BrokerError::NotConnected);
        }

        Ok(OrderAck {
            cl_ord_id: order.cl_ord_id,
            order_id: format!("ORD{}", self.next_order_id + 1),
            status: FixOrderStatus::New,
            text: None,
            timestamp: Utc::now(),
        })
    }

    async fn cancel_order(&self, order_id: &str) -> Result<CancelAck, BrokerError> {
        if !self.connected {
            return Err(BrokerError::NotConnected);
        }

        Ok(CancelAck {
            cl_ord_id: "".to_string(),
            orig_cl_ord_id: order_id.to_string(),
            order_id: order_id.to_string(),
            success: true,
            reason: None,
            timestamp: Utc::now(),
        })
    }

    async fn modify_order(
        &self,
        order_id: &str,
        modification: OrderModification,
    ) -> Result<ModifyAck, BrokerError> {
        if !self.connected {
            return Err(BrokerError::NotConnected);
        }

        Ok(ModifyAck {
            cl_ord_id: format!("{}_M", order_id),
            orig_cl_ord_id: order_id.to_string(),
            order_id: order_id.to_string(),
            success: true,
            new_qty: modification.new_qty,
            new_price: modification.new_price,
            reason: None,
            timestamp: Utc::now(),
        })
    }

    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatus, BrokerError> {
        if !self.connected {
            return Err(BrokerError::NotConnected);
        }

        self.orders
            .get(order_id)
            .cloned()
            .ok_or_else(|| BrokerError::OrderNotFound(order_id.to_string()))
    }

    async fn get_balance(&self) -> Result<AccountBalance, BrokerError> {
        if !self.connected {
            return Err(BrokerError::NotConnected);
        }

        Ok(self.balance.clone())
    }

    async fn get_positions(&self) -> Result<Vec<Position>, BrokerError> {
        if !self.connected {
            return Err(BrokerError::NotConnected);
        }

        Ok(self.positions.clone())
    }

    fn available_venues(&self) -> Vec<String> {
        vec![
            "NYSE".to_string(),
            "NASDAQ".to_string(),
            "ARCA".to_string(),
            "BATS".to_string(),
            "IEX".to_string(),
        ]
    }

    async fn subscribe_executions(
        &self,
        _callback: Box<dyn Fn(ExecutionReportFix) + Send + Sync>,
    ) -> Result<(), BrokerError> {
        if !self.connected {
            return Err(BrokerError::NotConnected);
        }

        // Mock implementation - would set up callback in real implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_side_conversion() {
        assert_eq!(OrderSide::Buy.to_fix(), '1');
        assert_eq!(OrderSide::Sell.to_fix(), '2');
        assert_eq!(OrderSide::from_fix('1'), Some(OrderSide::Buy));
        assert_eq!(OrderSide::from_fix('2'), Some(OrderSide::Sell));
        assert_eq!(OrderSide::from_fix('X'), None);
    }

    #[test]
    fn test_tif_conversion() {
        assert_eq!(TimeInForce::Day.to_fix(), '0');
        assert_eq!(TimeInForce::Gtc.to_fix(), '1');
        assert_eq!(TimeInForce::Ioc.to_fix(), '3');
        assert_eq!(TimeInForce::from_fix('0'), Some(TimeInForce::Day));
    }

    #[test]
    fn test_fix_order_status() {
        assert_eq!(FixOrderStatus::from_fix('0'), Some(FixOrderStatus::New));
        assert_eq!(FixOrderStatus::from_fix('2'), Some(FixOrderStatus::Filled));
        assert!(FixOrderStatus::Filled.is_terminal());
        assert!(FixOrderStatus::Canceled.is_terminal());
        assert!(!FixOrderStatus::New.is_terminal());
    }

    #[test]
    fn test_prime_order_creation() {
        let order = PrimeOrder::market("CLT001", "AAPL", OrderSide::Buy, dec!(100));

        assert_eq!(order.cl_ord_id, "CLT001");
        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.ord_type, '1');  // Market
        assert_eq!(order.price, None);
    }

    #[test]
    fn test_limit_order_creation() {
        let order = PrimeOrder::limit("CLT002", "AAPL", OrderSide::Sell, dec!(100), dec!(150))
            .with_tif(TimeInForce::Gtc)
            .with_account("ACCT001")
            .with_destination("NYSE");

        assert_eq!(order.ord_type, '2');  // Limit
        assert_eq!(order.price, Some(dec!(150)));
        assert_eq!(order.time_in_force, TimeInForce::Gtc);
        assert_eq!(order.account, Some("ACCT001".to_string()));
        assert_eq!(order.ex_destination, Some("NYSE".to_string()));
    }

    #[test]
    fn test_fix_session_config() {
        let config = FixSessionConfig::default();

        assert_eq!(config.fix_version, "FIX.4.4");
        assert!(config.use_ssl);
        assert_eq!(config.heartbeat_interval, 30);
    }

    #[tokio::test]
    async fn test_mock_broker_connect() {
        let mut broker = MockPrimeBroker::new(FixSessionConfig::default());

        assert!(!broker.is_connected());

        broker.connect().await.unwrap();
        assert!(broker.is_connected());

        broker.disconnect().await.unwrap();
        assert!(!broker.is_connected());
    }

    #[tokio::test]
    async fn test_mock_broker_submit_order() {
        let mut broker = MockPrimeBroker::new(FixSessionConfig::default());
        broker.connect().await.unwrap();

        let order = PrimeOrder::market("CLT001", "AAPL", OrderSide::Buy, dec!(100));
        let ack = broker.submit_order(order).await.unwrap();

        assert_eq!(ack.cl_ord_id, "CLT001");
        assert_eq!(ack.status, FixOrderStatus::New);
    }

    #[tokio::test]
    async fn test_mock_broker_not_connected() {
        let broker = MockPrimeBroker::new(FixSessionConfig::default());

        let order = PrimeOrder::market("CLT001", "AAPL", OrderSide::Buy, dec!(100));
        let result = broker.submit_order(order).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_broker_balance() {
        let mut broker = MockPrimeBroker::new(FixSessionConfig::default());
        broker.connect().await.unwrap();

        let balance = broker.get_balance().await.unwrap();

        assert_eq!(balance.cash, dec!(1_000_000));
        assert_eq!(balance.buying_power, dec!(2_000_000));
    }

    #[tokio::test]
    async fn test_mock_broker_positions() {
        let mut broker = MockPrimeBroker::new(FixSessionConfig::default());
        broker.add_position(Position {
            symbol: "AAPL".to_string(),
            quantity: dec!(100),
            avg_cost: dec!(150),
            current_price: dec!(155),
            market_value: dec!(15500),
            unrealized_pnl: dec!(500),
        });

        broker.connect().await.unwrap();

        let positions = broker.get_positions().await.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].symbol, "AAPL");
    }

    #[test]
    fn test_exec_type_conversion() {
        assert_eq!(ExecType::from_fix('F'), Some(ExecType::Trade));
        assert_eq!(ExecType::from_fix('0'), Some(ExecType::New));
        assert_eq!(ExecType::from_fix('4'), Some(ExecType::Canceled));
    }

    #[test]
    fn test_available_venues() {
        let broker = MockPrimeBroker::new(FixSessionConfig::default());
        let venues = broker.available_venues();

        assert!(venues.contains(&"NYSE".to_string()));
        assert!(venues.contains(&"NASDAQ".to_string()));
    }
}
