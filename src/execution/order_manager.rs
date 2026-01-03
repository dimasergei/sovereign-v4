//! Order Manager Module
//!
//! Manages order lifecycle including:
//! - Order state tracking
//! - Parent/child order relationships (for algo slicing)
//! - Fill recording
//! - Retry logic with exponential backoff
//! - Execution reporting

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::venue::Venue;

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    MarketOnClose,
    LimitOnClose,
    Pegged,
    MidpointPeg,
    PrimaryPeg,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Market => write!(f, "MKT"),
            OrderType::Limit => write!(f, "LMT"),
            OrderType::Stop => write!(f, "STP"),
            OrderType::StopLimit => write!(f, "STP_LMT"),
            OrderType::MarketOnClose => write!(f, "MOC"),
            OrderType::LimitOnClose => write!(f, "LOC"),
            OrderType::Pegged => write!(f, "PEG"),
            OrderType::MidpointPeg => write!(f, "MID_PEG"),
            OrderType::PrimaryPeg => write!(f, "PRI_PEG"),
        }
    }
}

/// Time in force
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Day order (expires at end of day)
    Day,
    /// Good till cancelled
    Gtc,
    /// Immediate or cancel
    Ioc,
    /// Fill or kill
    Fok,
    /// Good till date
    Gtd(DateTime<Utc>),
    /// At the open
    Opg,
    /// At the close
    Cls,
}

impl std::fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeInForce::Day => write!(f, "DAY"),
            TimeInForce::Gtc => write!(f, "GTC"),
            TimeInForce::Ioc => write!(f, "IOC"),
            TimeInForce::Fok => write!(f, "FOK"),
            TimeInForce::Gtd(_) => write!(f, "GTD"),
            TimeInForce::Opg => write!(f, "OPG"),
            TimeInForce::Cls => write!(f, "CLS"),
        }
    }
}

/// Order state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderState {
    /// Order created but not yet submitted
    Pending,
    /// Order submitted to venue
    Submitted,
    /// Order acknowledged by venue
    Acknowledged,
    /// Order partially filled
    PartialFill,
    /// Order completely filled
    Filled,
    /// Order cancelled
    Cancelled,
    /// Order rejected by venue
    Rejected,
    /// Order expired
    Expired,
    /// Order failed (system error)
    Failed,
}

impl OrderState {
    /// Check if order is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderState::Filled
                | OrderState::Cancelled
                | OrderState::Rejected
                | OrderState::Expired
                | OrderState::Failed
        )
    }

    /// Check if order is working (can receive fills)
    pub fn is_working(&self) -> bool {
        matches!(
            self,
            OrderState::Submitted | OrderState::Acknowledged | OrderState::PartialFill
        )
    }
}

impl std::fmt::Display for OrderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderState::Pending => write!(f, "PENDING"),
            OrderState::Submitted => write!(f, "SUBMITTED"),
            OrderState::Acknowledged => write!(f, "ACK"),
            OrderState::PartialFill => write!(f, "PARTIAL"),
            OrderState::Filled => write!(f, "FILLED"),
            OrderState::Cancelled => write!(f, "CANCELLED"),
            OrderState::Rejected => write!(f, "REJECTED"),
            OrderState::Expired => write!(f, "EXPIRED"),
            OrderState::Failed => write!(f, "FAILED"),
        }
    }
}

/// Liquidity indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LiquidityIndicator {
    /// Added liquidity (maker)
    Added,
    /// Removed liquidity (taker)
    Removed,
    /// Unknown
    Unknown,
}

/// A single fill event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// Unique fill ID
    pub fill_id: String,
    /// Quantity filled
    pub quantity: Decimal,
    /// Fill price
    pub price: Decimal,
    /// Fill timestamp
    pub timestamp: DateTime<Utc>,
    /// Venue where fill occurred
    pub venue: String,
    /// Commission for this fill
    pub commission: Decimal,
    /// Liquidity indicator
    pub liquidity: LiquidityIndicator,
    /// Exchange-assigned execution ID
    pub exec_id: Option<String>,
}

impl Fill {
    /// Create a new fill
    pub fn new(quantity: Decimal, price: Decimal, venue: &str) -> Self {
        Self {
            fill_id: Uuid::new_v4().to_string(),
            quantity,
            price,
            timestamp: Utc::now(),
            venue: venue.to_string(),
            commission: dec!(0),
            liquidity: LiquidityIndicator::Unknown,
            exec_id: None,
        }
    }

    /// Set commission
    pub fn with_commission(mut self, commission: Decimal) -> Self {
        self.commission = commission;
        self
    }

    /// Set liquidity indicator
    pub fn with_liquidity(mut self, liquidity: LiquidityIndicator) -> Self {
        self.liquidity = liquidity;
        self
    }

    /// Set execution ID
    pub fn with_exec_id(mut self, exec_id: &str) -> Self {
        self.exec_id = Some(exec_id.to_string());
        self
    }

    /// Get fill value (quantity * price)
    pub fn value(&self) -> Decimal {
        self.quantity * self.price
    }

    /// Get net value (after commission)
    pub fn net_value(&self) -> Decimal {
        self.value() - self.commission
    }
}

/// A managed order with full lifecycle tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedOrder {
    /// Unique order ID (internal)
    pub order_id: String,
    /// Broker-assigned order ID
    pub broker_order_id: Option<String>,
    /// Parent order ID (for child orders in algo execution)
    pub parent_id: Option<String>,
    /// Symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Total quantity
    pub quantity: Decimal,
    /// Filled quantity
    pub filled_qty: Decimal,
    /// Remaining quantity
    pub remaining_qty: Decimal,
    /// Limit price (if applicable)
    pub limit_price: Option<Decimal>,
    /// Stop price (if applicable)
    pub stop_price: Option<Decimal>,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Current state
    pub state: OrderState,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Submission time
    pub submitted_at: Option<DateTime<Utc>>,
    /// List of fills
    pub fills: Vec<Fill>,
    /// Target venue
    pub target_venue: Option<String>,
    /// Algorithm tag
    pub algo_tag: Option<String>,
    /// Retry count
    pub retry_count: u32,
    /// Maximum retries
    pub max_retries: u32,
    /// Last error message
    pub last_error: Option<String>,
    /// Average fill price
    pub avg_fill_price: Decimal,
    /// Total commission
    pub total_commission: Decimal,
    /// Custom tags
    pub tags: HashMap<String, String>,
}

impl ManagedOrder {
    /// Create a new managed order
    pub fn new(symbol: &str, side: OrderSide, quantity: Decimal) -> Self {
        let order_id = Uuid::new_v4().to_string();
        Self {
            order_id,
            broker_order_id: None,
            parent_id: None,
            symbol: symbol.to_string(),
            side,
            order_type: OrderType::Market,
            quantity,
            filled_qty: dec!(0),
            remaining_qty: quantity,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
            state: OrderState::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            submitted_at: None,
            fills: Vec::new(),
            target_venue: None,
            algo_tag: None,
            retry_count: 0,
            max_retries: 3,
            last_error: None,
            avg_fill_price: dec!(0),
            total_commission: dec!(0),
            tags: HashMap::new(),
        }
    }

    /// Create a market order
    pub fn market(symbol: &str, side: OrderSide, quantity: Decimal) -> Self {
        Self::new(symbol, side, quantity)
    }

    /// Create a limit order
    pub fn limit(symbol: &str, side: OrderSide, quantity: Decimal, limit_price: Decimal) -> Self {
        let mut order = Self::new(symbol, side, quantity);
        order.order_type = OrderType::Limit;
        order.limit_price = Some(limit_price);
        order
    }

    /// Create a child order from a parent
    pub fn child_of(parent: &ManagedOrder, quantity: Decimal) -> Self {
        let mut order = Self::new(&parent.symbol, parent.side, quantity);
        order.parent_id = Some(parent.order_id.clone());
        order.order_type = parent.order_type;
        order.limit_price = parent.limit_price;
        order.time_in_force = parent.time_in_force;
        order.algo_tag = parent.algo_tag.clone();
        order
    }

    /// Set order type
    pub fn with_order_type(mut self, order_type: OrderType) -> Self {
        self.order_type = order_type;
        self
    }

    /// Set limit price
    pub fn with_limit(mut self, price: Decimal) -> Self {
        self.limit_price = Some(price);
        if matches!(self.order_type, OrderType::Market) {
            self.order_type = OrderType::Limit;
        }
        self
    }

    /// Set stop price
    pub fn with_stop(mut self, price: Decimal) -> Self {
        self.stop_price = Some(price);
        self.order_type = OrderType::Stop;
        self
    }

    /// Set time in force
    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Set target venue
    pub fn with_venue(mut self, venue: &str) -> Self {
        self.target_venue = Some(venue.to_string());
        self
    }

    /// Set algorithm tag
    pub fn with_algo(mut self, algo: &str) -> Self {
        self.algo_tag = Some(algo.to_string());
        self
    }

    /// Add a custom tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Mark as submitted
    pub fn mark_submitted(&mut self, broker_order_id: Option<&str>) {
        self.state = OrderState::Submitted;
        self.submitted_at = Some(Utc::now());
        self.updated_at = Utc::now();
        if let Some(id) = broker_order_id {
            self.broker_order_id = Some(id.to_string());
        }
    }

    /// Mark as acknowledged
    pub fn mark_acknowledged(&mut self, broker_order_id: Option<&str>) {
        self.state = OrderState::Acknowledged;
        self.updated_at = Utc::now();
        if let Some(id) = broker_order_id {
            self.broker_order_id = Some(id.to_string());
        }
    }

    /// Record a fill
    pub fn record_fill(&mut self, fill: Fill) {
        let fill_value = fill.value();
        let fill_qty = fill.quantity;

        // Update average price
        let old_value = self.avg_fill_price * self.filled_qty;
        self.filled_qty += fill_qty;
        self.remaining_qty = self.quantity - self.filled_qty;

        if self.filled_qty > dec!(0) {
            self.avg_fill_price = (old_value + fill_value) / self.filled_qty;
        }

        self.total_commission += fill.commission;
        self.fills.push(fill);

        // Update state
        if self.remaining_qty <= dec!(0) {
            self.state = OrderState::Filled;
        } else {
            self.state = OrderState::PartialFill;
        }
        self.updated_at = Utc::now();
    }

    /// Mark as cancelled
    pub fn mark_cancelled(&mut self) {
        self.state = OrderState::Cancelled;
        self.updated_at = Utc::now();
    }

    /// Mark as rejected with reason
    pub fn mark_rejected(&mut self, reason: &str) {
        self.state = OrderState::Rejected;
        self.last_error = Some(reason.to_string());
        self.updated_at = Utc::now();
    }

    /// Mark as failed with error
    pub fn mark_failed(&mut self, error: &str) {
        self.state = OrderState::Failed;
        self.last_error = Some(error.to_string());
        self.updated_at = Utc::now();
    }

    /// Mark as expired
    pub fn mark_expired(&mut self) {
        self.state = OrderState::Expired;
        self.updated_at = Utc::now();
    }

    /// Increment retry count and check if can retry
    pub fn can_retry(&mut self) -> bool {
        if self.retry_count < self.max_retries {
            self.retry_count += 1;
            true
        } else {
            false
        }
    }

    /// Get delay for next retry (exponential backoff)
    pub fn retry_delay(&self) -> Duration {
        Duration::milliseconds((2_i64.pow(self.retry_count) * 1000).min(16000))
    }

    /// Check if order is complete
    pub fn is_complete(&self) -> bool {
        self.state.is_terminal()
    }

    /// Check if order is working
    pub fn is_working(&self) -> bool {
        self.state.is_working()
    }

    /// Get fill percentage
    pub fn fill_pct(&self) -> f64 {
        if self.quantity.is_zero() {
            return 0.0;
        }
        (self.filled_qty / self.quantity).to_f64().unwrap_or(0.0) * 100.0
    }

    /// Get total notional value
    pub fn notional_value(&self) -> Decimal {
        self.avg_fill_price * self.filled_qty
    }

    /// Check if this is a child order
    pub fn is_child(&self) -> bool {
        self.parent_id.is_some()
    }
}

/// Execution report for a completed order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    /// Order ID
    pub order_id: String,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: OrderSide,
    /// Total quantity ordered
    pub ordered_qty: Decimal,
    /// Total quantity filled
    pub filled_qty: Decimal,
    /// Average fill price
    pub avg_price: Decimal,
    /// Total commission
    pub total_commission: Decimal,
    /// Final state
    pub final_state: OrderState,
    /// Number of fills
    pub fill_count: usize,
    /// Venues used
    pub venues: Vec<String>,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
    /// Duration
    pub duration_ms: i64,
    /// Algorithm used
    pub algorithm: Option<String>,
}

impl ExecutionReport {
    /// Create from a managed order
    pub fn from_order(order: &ManagedOrder) -> Self {
        let venues: Vec<String> = order
            .fills
            .iter()
            .map(|f| f.venue.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let end_time = Utc::now();
        let duration_ms = (end_time - order.created_at).num_milliseconds();

        Self {
            order_id: order.order_id.clone(),
            symbol: order.symbol.clone(),
            side: order.side,
            ordered_qty: order.quantity,
            filled_qty: order.filled_qty,
            avg_price: order.avg_fill_price,
            total_commission: order.total_commission,
            final_state: order.state,
            fill_count: order.fills.len(),
            venues,
            start_time: order.created_at,
            end_time,
            duration_ms,
            algorithm: order.algo_tag.clone(),
        }
    }

    /// Check if execution was successful
    pub fn is_success(&self) -> bool {
        matches!(self.final_state, OrderState::Filled)
    }

    /// Get fill rate
    pub fn fill_rate(&self) -> f64 {
        if self.ordered_qty.is_zero() {
            return 0.0;
        }
        (self.filled_qty / self.ordered_qty).to_f64().unwrap_or(0.0) * 100.0
    }

    /// Get total cost (fills + commission)
    pub fn total_cost(&self) -> Decimal {
        self.avg_price * self.filled_qty + self.total_commission
    }
}

/// Order manager for tracking multiple orders
#[derive(Debug, Default)]
pub struct OrderManager {
    /// All orders by ID
    orders: HashMap<String, ManagedOrder>,
    /// Parent to children mapping
    parent_children: HashMap<String, Vec<String>>,
    /// Symbol to active orders mapping
    symbol_orders: HashMap<String, Vec<String>>,
    /// Completed execution reports
    reports: Vec<ExecutionReport>,
}

impl OrderManager {
    /// Create a new order manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new order
    pub fn add_order(&mut self, order: ManagedOrder) -> String {
        let order_id = order.order_id.clone();
        let symbol = order.symbol.clone();

        // Track parent-child relationship
        if let Some(ref parent_id) = order.parent_id {
            self.parent_children
                .entry(parent_id.clone())
                .or_default()
                .push(order_id.clone());
        }

        // Track symbol mapping
        self.symbol_orders
            .entry(symbol)
            .or_default()
            .push(order_id.clone());

        self.orders.insert(order_id.clone(), order);
        order_id
    }

    /// Get an order by ID
    pub fn get_order(&self, order_id: &str) -> Option<&ManagedOrder> {
        self.orders.get(order_id)
    }

    /// Get a mutable order by ID
    pub fn get_order_mut(&mut self, order_id: &str) -> Option<&mut ManagedOrder> {
        self.orders.get_mut(order_id)
    }

    /// Get all orders for a symbol
    pub fn orders_for_symbol(&self, symbol: &str) -> Vec<&ManagedOrder> {
        self.symbol_orders
            .get(symbol)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.orders.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all working orders
    pub fn working_orders(&self) -> Vec<&ManagedOrder> {
        self.orders.values().filter(|o| o.is_working()).collect()
    }

    /// Get all pending orders
    pub fn pending_orders(&self) -> Vec<&ManagedOrder> {
        self.orders
            .values()
            .filter(|o| matches!(o.state, OrderState::Pending))
            .collect()
    }

    /// Get children of a parent order
    pub fn get_children(&self, parent_id: &str) -> Vec<&ManagedOrder> {
        self.parent_children
            .get(parent_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.orders.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Record a fill for an order
    pub fn record_fill(&mut self, order_id: &str, fill: Fill) -> bool {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.record_fill(fill);
            true
        } else {
            false
        }
    }

    /// Cancel an order
    pub fn cancel_order(&mut self, order_id: &str) -> bool {
        if let Some(order) = self.orders.get_mut(order_id) {
            order.mark_cancelled();
            true
        } else {
            false
        }
    }

    /// Complete an order and generate execution report
    pub fn complete_order(&mut self, order_id: &str) -> Option<ExecutionReport> {
        if let Some(order) = self.orders.get(order_id) {
            let report = ExecutionReport::from_order(order);
            self.reports.push(report.clone());
            Some(report)
        } else {
            None
        }
    }

    /// Get all execution reports
    pub fn get_reports(&self) -> &[ExecutionReport] {
        &self.reports
    }

    /// Get reports for a symbol
    pub fn get_reports_for_symbol(&self, symbol: &str) -> Vec<&ExecutionReport> {
        self.reports.iter().filter(|r| r.symbol == symbol).collect()
    }

    /// Get aggregate fill rate for a symbol
    pub fn symbol_fill_rate(&self, symbol: &str) -> f64 {
        let reports = self.get_reports_for_symbol(symbol);
        if reports.is_empty() {
            return 0.0;
        }

        let total_ordered: Decimal = reports.iter().map(|r| r.ordered_qty).sum();
        let total_filled: Decimal = reports.iter().map(|r| r.filled_qty).sum();

        if total_ordered.is_zero() {
            return 0.0;
        }

        (total_filled / total_ordered).to_f64().unwrap_or(0.0) * 100.0
    }

    /// Clean up old completed orders
    pub fn cleanup(&mut self, max_age: Duration) {
        let cutoff = Utc::now() - max_age;

        let to_remove: Vec<String> = self
            .orders
            .iter()
            .filter(|(_, o)| o.is_complete() && o.updated_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.orders.remove(&id);

            // Clean up symbol mapping
            for orders in self.symbol_orders.values_mut() {
                orders.retain(|o| o != &id);
            }

            // Clean up parent-child mapping
            for children in self.parent_children.values_mut() {
                children.retain(|c| c != &id);
            }
        }
    }

    /// Get order count
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Get working order count
    pub fn working_count(&self) -> usize {
        self.orders.values().filter(|o| o.is_working()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_managed_order_creation() {
        let order = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(100));

        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.quantity, dec!(100));
        assert_eq!(order.state, OrderState::Pending);
        assert_eq!(order.order_type, OrderType::Market);
    }

    #[test]
    fn test_limit_order() {
        let order = ManagedOrder::limit("AAPL", OrderSide::Buy, dec!(100), dec!(150.00));

        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.limit_price, Some(dec!(150.00)));
    }

    #[test]
    fn test_order_lifecycle() {
        let mut order = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(100));

        // Submit
        order.mark_submitted(Some("BROKER123"));
        assert_eq!(order.state, OrderState::Submitted);
        assert_eq!(order.broker_order_id, Some("BROKER123".to_string()));

        // Acknowledge
        order.mark_acknowledged(None);
        assert_eq!(order.state, OrderState::Acknowledged);

        // Partial fill
        let fill = Fill::new(dec!(50), dec!(150.00), "NASDAQ");
        order.record_fill(fill);
        assert_eq!(order.state, OrderState::PartialFill);
        assert_eq!(order.filled_qty, dec!(50));
        assert_eq!(order.remaining_qty, dec!(50));

        // Complete fill
        let fill = Fill::new(dec!(50), dec!(150.10), "NYSE");
        order.record_fill(fill);
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.filled_qty, dec!(100));
        assert!(order.is_complete());

        // Average price
        assert!((order.avg_fill_price - dec!(150.05)).abs() < dec!(0.01));
    }

    #[test]
    fn test_child_orders() {
        let parent = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(1000))
            .with_algo("VWAP");

        let child = ManagedOrder::child_of(&parent, dec!(100));

        assert_eq!(child.parent_id, Some(parent.order_id.clone()));
        assert_eq!(child.symbol, "AAPL");
        assert_eq!(child.side, OrderSide::Buy);
        assert_eq!(child.algo_tag, Some("VWAP".to_string()));
        assert!(child.is_child());
    }

    #[test]
    fn test_retry_logic() {
        let mut order = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(100));
        order.max_retries = 3;

        assert!(order.can_retry());
        assert_eq!(order.retry_count, 1);

        assert!(order.can_retry());
        assert_eq!(order.retry_count, 2);

        assert!(order.can_retry());
        assert_eq!(order.retry_count, 3);

        assert!(!order.can_retry());  // Max reached
        assert_eq!(order.retry_count, 3);
    }

    #[test]
    fn test_retry_delay() {
        let mut order = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(100));

        order.retry_count = 0;
        assert_eq!(order.retry_delay().num_milliseconds(), 1000);

        order.retry_count = 1;
        assert_eq!(order.retry_delay().num_milliseconds(), 2000);

        order.retry_count = 2;
        assert_eq!(order.retry_delay().num_milliseconds(), 4000);

        order.retry_count = 3;
        assert_eq!(order.retry_delay().num_milliseconds(), 8000);

        order.retry_count = 4;
        assert_eq!(order.retry_delay().num_milliseconds(), 16000);

        order.retry_count = 5;  // Should cap at 16000
        assert_eq!(order.retry_delay().num_milliseconds(), 16000);
    }

    #[test]
    fn test_execution_report() {
        let mut order = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(100))
            .with_algo("TWAP");

        order.mark_submitted(Some("B123"));
        order.record_fill(Fill::new(dec!(50), dec!(150), "NYSE").with_commission(dec!(0.50)));
        order.record_fill(Fill::new(dec!(50), dec!(151), "NASDAQ").with_commission(dec!(0.50)));

        let report = ExecutionReport::from_order(&order);

        assert_eq!(report.filled_qty, dec!(100));
        assert_eq!(report.fill_count, 2);
        assert_eq!(report.total_commission, dec!(1.00));
        assert!(report.venues.contains(&"NYSE".to_string()));
        assert!(report.venues.contains(&"NASDAQ".to_string()));
        assert_eq!(report.algorithm, Some("TWAP".to_string()));
    }

    #[test]
    fn test_order_manager() {
        let mut manager = OrderManager::new();

        // Add parent order
        let parent = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(1000));
        let parent_id = parent.order_id.clone();
        manager.add_order(parent);

        // Add child orders
        let parent = manager.get_order(&parent_id).unwrap();
        let child1 = ManagedOrder::child_of(parent, dec!(500));
        let child2 = ManagedOrder::child_of(parent, dec!(500));
        let child1_id = child1.order_id.clone();
        let child2_id = child2.order_id.clone();
        manager.add_order(child1);
        manager.add_order(child2);

        // Check parent-child relationship
        let children = manager.get_children(&parent_id);
        assert_eq!(children.len(), 2);

        // Record fills
        manager.record_fill(&child1_id, Fill::new(dec!(500), dec!(150), "NYSE"));
        manager.record_fill(&child2_id, Fill::new(dec!(500), dec!(150), "NASDAQ"));

        // Check order states
        assert!(manager.get_order(&child1_id).unwrap().is_complete());
        assert!(manager.get_order(&child2_id).unwrap().is_complete());

        // Generate reports
        let report = manager.complete_order(&child1_id).unwrap();
        assert!(report.is_success());
    }

    #[test]
    fn test_fill_with_details() {
        let fill = Fill::new(dec!(100), dec!(150.50), "NASDAQ")
            .with_commission(dec!(0.30))
            .with_liquidity(LiquidityIndicator::Removed)
            .with_exec_id("EX12345");

        assert_eq!(fill.quantity, dec!(100));
        assert_eq!(fill.price, dec!(150.50));
        assert_eq!(fill.venue, "NASDAQ");
        assert_eq!(fill.commission, dec!(0.30));
        assert_eq!(fill.liquidity, LiquidityIndicator::Removed);
        assert_eq!(fill.exec_id, Some("EX12345".to_string()));
        assert_eq!(fill.value(), dec!(15050));
        assert_eq!(fill.net_value(), dec!(15049.70));
    }

    #[test]
    fn test_order_states() {
        assert!(!OrderState::Pending.is_terminal());
        assert!(!OrderState::Submitted.is_terminal());
        assert!(!OrderState::PartialFill.is_terminal());
        assert!(OrderState::Filled.is_terminal());
        assert!(OrderState::Cancelled.is_terminal());
        assert!(OrderState::Rejected.is_terminal());

        assert!(!OrderState::Pending.is_working());
        assert!(OrderState::Submitted.is_working());
        assert!(OrderState::Acknowledged.is_working());
        assert!(OrderState::PartialFill.is_working());
        assert!(!OrderState::Filled.is_working());
    }
}
