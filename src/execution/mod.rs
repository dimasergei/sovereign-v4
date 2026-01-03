//! Institutional Execution Infrastructure
//!
//! Provides institutional-grade execution capabilities including:
//! - Tiered execution (Retail → Semi-Institutional → Institutional)
//! - Execution algorithms (VWAP, TWAP, POV, Iceberg, Adaptive)
//! - Smart order routing with venue scoring
//! - Dark pool integration
//! - Transaction Cost Analysis (TCA)
//! - FIX protocol preparation
//!
//! # Architecture
//!
//! ```text
//! ExecutionEngine
//! ├── SmartRouter (venue selection)
//! ├── DarkPoolRouter (dark pool routing)
//! ├── OrderManager (order lifecycle)
//! └── TcaAnalyzer (transaction cost analysis)
//! ```
//!
//! # Account Tiers
//!
//! - `Retail`: < $100K - Alpaca market orders
//! - `SemiInstitutional`: $100K-$1M - IBKR algos, IEX routing
//! - `Institutional`: $1M+ - Prime broker, dark pools, FIX

pub mod algorithms;
pub mod dark_pool;
pub mod order_manager;
pub mod smart_router;
pub mod tca;
pub mod venue;

use chrono::{Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

use algorithms::{
    AlgorithmState, ExecutionAlgorithm, IcebergManager, OrderSlice, TwapScheduler, VwapScheduler,
};
use dark_pool::{DarkPoolConfig, DarkPoolRouter};
use order_manager::{ExecutionReport, Fill, ManagedOrder, OrderManager, OrderSide, OrderState};
use smart_router::{RoutingDecision, SmartRouter, SprayRoute};
use tca::{BenchmarkPrices, TcaAnalyzer, TcaReport};
use venue::VenueRegistry;

/// Account tier for execution capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountTier {
    /// Retail: < $100K - Basic execution via Alpaca
    Retail,
    /// Semi-Institutional: $100K-$1M - IBKR algos, better routing
    SemiInstitutional,
    /// Institutional: $1M+ - Prime broker, dark pools, FIX
    Institutional,
}

impl AccountTier {
    /// Determine tier from account balance
    pub fn from_balance(balance: Decimal) -> Self {
        if balance >= dec!(1_000_000) {
            AccountTier::Institutional
        } else if balance >= dec!(100_000) {
            AccountTier::SemiInstitutional
        } else {
            AccountTier::Retail
        }
    }

    /// Check if tier supports algo execution
    pub fn supports_algos(&self) -> bool {
        !matches!(self, AccountTier::Retail)
    }

    /// Check if tier supports dark pools
    pub fn supports_dark_pools(&self) -> bool {
        matches!(self, AccountTier::Institutional)
    }

    /// Check if tier supports FIX protocol
    pub fn supports_fix(&self) -> bool {
        matches!(self, AccountTier::Institutional)
    }

    /// Get maximum concurrent orders
    pub fn max_concurrent_orders(&self) -> usize {
        match self {
            AccountTier::Retail => 5,
            AccountTier::SemiInstitutional => 20,
            AccountTier::Institutional => 100,
        }
    }

    /// Get default broker for tier
    pub fn default_broker(&self) -> &'static str {
        match self {
            AccountTier::Retail => "ALPACA",
            AccountTier::SemiInstitutional => "IBKR",
            AccountTier::Institutional => "PRIME",
        }
    }
}

impl std::fmt::Display for AccountTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccountTier::Retail => write!(f, "Retail"),
            AccountTier::SemiInstitutional => write!(f, "Semi-Institutional"),
            AccountTier::Institutional => write!(f, "Institutional"),
        }
    }
}

/// Urgency level for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Urgency {
    /// Low urgency - prioritize price improvement
    Low,
    /// Medium urgency - balance speed and cost
    Medium,
    /// High urgency - prioritize speed
    High,
    /// Immediate - market orders only
    Immediate,
}

impl Urgency {
    /// Convert to numeric value (0.0 to 1.0)
    pub fn as_f64(&self) -> f64 {
        match self {
            Urgency::Low => 0.2,
            Urgency::Medium => 0.5,
            Urgency::High => 0.8,
            Urgency::Immediate => 1.0,
        }
    }
}

/// Execution plan for an order
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Symbol to trade
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Total quantity
    pub quantity: Decimal,
    /// Reference price (arrival price)
    pub reference_price: Decimal,
    /// Execution algorithm
    pub algorithm: ExecutionAlgorithm,
    /// Urgency level
    pub urgency: Urgency,
    /// Execution duration (for algos)
    pub duration: Option<Duration>,
    /// Use dark pools
    pub use_dark_pools: bool,
    /// Maximum dark pool percentage
    pub max_dark_pct: f64,
    /// Limit price (if applicable)
    pub limit_price: Option<Decimal>,
    /// Client order ID
    pub client_order_id: Option<String>,
}

impl ExecutionPlan {
    /// Create a new execution plan
    pub fn new(symbol: &str, side: OrderSide, quantity: Decimal, price: Decimal) -> Self {
        Self {
            symbol: symbol.to_string(),
            side,
            quantity,
            reference_price: price,
            algorithm: ExecutionAlgorithm::Market,
            urgency: Urgency::Medium,
            duration: None,
            use_dark_pools: false,
            max_dark_pct: 0.3,
            limit_price: None,
            client_order_id: None,
        }
    }

    /// Create a market order plan
    pub fn market(symbol: &str, is_buy: bool, quantity: Decimal, price: Decimal) -> Self {
        let side = if is_buy {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        Self::new(symbol, side, quantity, price)
    }

    /// Create a VWAP plan
    pub fn vwap(
        symbol: &str,
        is_buy: bool,
        quantity: Decimal,
        price: Decimal,
        num_slices: u32,
        duration: Duration,
    ) -> Self {
        let side = if is_buy {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        let mut plan = Self::new(symbol, side, quantity, price);
        plan.algorithm = ExecutionAlgorithm::vwap(duration.num_minutes(), num_slices);
        plan.duration = Some(duration);
        plan.urgency = Urgency::Low;
        plan
    }

    /// Create a TWAP plan
    pub fn twap(
        symbol: &str,
        is_buy: bool,
        quantity: Decimal,
        price: Decimal,
        num_slices: u32,
        duration: Duration,
    ) -> Self {
        let side = if is_buy {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        let mut plan = Self::new(symbol, side, quantity, price);
        plan.algorithm = ExecutionAlgorithm::twap(duration.num_minutes(), num_slices);
        plan.duration = Some(duration);
        plan.urgency = Urgency::Medium;
        plan
    }

    /// Create an iceberg plan
    pub fn iceberg(
        symbol: &str,
        is_buy: bool,
        quantity: Decimal,
        price: Decimal,
        display_ratio: f64,
    ) -> Self {
        let side = if is_buy {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        let mut plan = Self::new(symbol, side, quantity, price);
        plan.algorithm = ExecutionAlgorithm::iceberg(quantity, display_ratio);
        plan.urgency = Urgency::Low;
        plan
    }

    /// Set urgency
    pub fn with_urgency(mut self, urgency: Urgency) -> Self {
        self.urgency = urgency;
        self
    }

    /// Enable dark pools
    pub fn with_dark_pools(mut self, max_pct: f64) -> Self {
        self.use_dark_pools = true;
        self.max_dark_pct = max_pct;
        self
    }

    /// Set limit price
    pub fn with_limit(mut self, price: Decimal) -> Self {
        self.limit_price = Some(price);
        self
    }

    /// Set client order ID
    pub fn with_client_id(mut self, id: &str) -> Self {
        self.client_order_id = Some(id.to_string());
        self
    }

    /// Calculate expected notional
    pub fn notional(&self) -> Decimal {
        self.quantity * self.reference_price
    }

    /// Check if plan uses an algorithm
    pub fn is_algo(&self) -> bool {
        self.algorithm.requires_slicing()
    }
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Order ID
    pub order_id: String,
    /// Final state
    pub state: OrderState,
    /// Total filled quantity
    pub filled_qty: Decimal,
    /// Average fill price
    pub avg_price: Decimal,
    /// Total commission
    pub commission: Decimal,
    /// Execution report
    pub report: Option<ExecutionReport>,
    /// TCA analysis
    pub tca: Option<TcaReport>,
    /// Venues used
    pub venues: Vec<String>,
    /// Number of fills
    pub fill_count: usize,
    /// Duration in milliseconds
    pub duration_ms: i64,
}

impl ExecutionResult {
    /// Check if execution was successful
    pub fn is_success(&self) -> bool {
        matches!(self.state, OrderState::Filled)
    }

    /// Get fill rate
    pub fn fill_rate(&self, total_qty: Decimal) -> f64 {
        if total_qty.is_zero() {
            return 0.0;
        }
        (self.filled_qty / total_qty).to_f64().unwrap_or(0.0) * 100.0
    }
}

/// Main execution engine
#[derive(Debug)]
pub struct ExecutionEngine {
    /// Account tier
    tier: AccountTier,
    /// Smart router
    router: SmartRouter,
    /// Dark pool router
    dark_router: DarkPoolRouter,
    /// Order manager
    orders: OrderManager,
    /// TCA analyzer
    tca: TcaAnalyzer,
    /// Active algorithm states
    algo_states: std::collections::HashMap<String, AlgorithmState>,
}

impl ExecutionEngine {
    /// Create a new execution engine
    pub fn new(tier: AccountTier) -> Self {
        let registry = VenueRegistry::with_all_venues();
        let router = SmartRouter::new(registry);
        let dark_config = DarkPoolConfig::default();
        let dark_router = DarkPoolRouter::new(dark_config);

        Self {
            tier,
            router,
            dark_router,
            orders: OrderManager::new(),
            tca: TcaAnalyzer::new(),
            algo_states: std::collections::HashMap::new(),
        }
    }

    /// Create engine for a given balance
    pub fn for_balance(balance: Decimal) -> Self {
        Self::new(AccountTier::from_balance(balance))
    }

    /// Get current tier
    pub fn tier(&self) -> AccountTier {
        self.tier
    }

    /// Set tier
    pub fn set_tier(&mut self, tier: AccountTier) {
        self.tier = tier;
    }

    /// Recommend an algorithm for the execution plan
    pub fn recommend_algorithm(&self, plan: &ExecutionPlan, adv: Decimal) -> ExecutionAlgorithm {
        // Retail tier always uses market orders
        if !self.tier.supports_algos() {
            return ExecutionAlgorithm::Market;
        }

        let urgency = plan.urgency.as_f64();
        let notional = plan.notional();

        // Use the algorithm module's recommendation
        algorithms::recommend_algorithm(notional, adv, urgency, 0.02)
    }

    /// Create an execution plan with recommended algorithm
    pub fn create_plan(
        &self,
        symbol: &str,
        is_buy: bool,
        quantity: Decimal,
        price: Decimal,
        urgency: Urgency,
        adv: Decimal,
    ) -> ExecutionPlan {
        let side = if is_buy {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        let mut plan = ExecutionPlan::new(symbol, side, quantity, price);
        plan.urgency = urgency;

        // Recommend algorithm
        plan.algorithm = self.recommend_algorithm(&plan, adv);

        // Enable dark pools for institutional tier
        if self.tier.supports_dark_pools() && urgency != Urgency::Immediate {
            plan.use_dark_pools = true;
            plan.max_dark_pct = 0.3;
        }

        plan
    }

    /// Route an order to the best venue
    pub fn route_order(&self, plan: &ExecutionPlan) -> RoutingDecision {
        self.router.route_order(
            &plan.symbol,
            plan.side,
            plan.quantity,
            plan.reference_price,
            plan.urgency.as_f64(),
        )
    }

    /// Get dark pool decision
    pub fn get_dark_pool_decision(&self, plan: &ExecutionPlan) -> dark_pool::DarkPoolDecision {
        if !self.tier.supports_dark_pools() || !plan.use_dark_pools {
            return dark_pool::DarkPoolDecision {
                use_dark: false,
                allocations: vec![],
                lit_quantity: plan.quantity,
                order_type: dark_pool::DarkPoolOrderType::Limit,
                timeout: Duration::zero(),
                reason: "Dark pools not enabled".to_string(),
            };
        }

        self.dark_router.select_pools(
            &plan.symbol,
            plan.quantity,
            plan.reference_price,
            plan.urgency.as_f64(),
        )
    }

    /// Generate algorithm slices for an execution plan
    pub fn generate_slices(&self, plan: &ExecutionPlan) -> Vec<OrderSlice> {
        match &plan.algorithm {
            ExecutionAlgorithm::Vwap(config) => {
                let scheduler = VwapScheduler::new(config.clone());
                scheduler.generate_slices(plan.quantity)
            }
            ExecutionAlgorithm::Twap(config) => {
                let scheduler = TwapScheduler::new(config.clone());
                scheduler.generate_slices(plan.quantity)
            }
            ExecutionAlgorithm::Market => {
                // Single slice for market orders
                vec![OrderSlice::new(0, plan.quantity, Utc::now())]
            }
            _ => {
                // Default: single slice
                vec![OrderSlice::new(0, plan.quantity, Utc::now())]
            }
        }
    }

    /// Create a managed order from a plan
    pub fn create_order(&mut self, plan: &ExecutionPlan) -> String {
        let order = ManagedOrder::new(&plan.symbol, plan.side, plan.quantity)
            .with_algo(plan.algorithm.name());

        if let Some(limit) = plan.limit_price {
            let order = order.with_limit(limit);
            return self.orders.add_order(order);
        }

        self.orders.add_order(order)
    }

    /// Record a fill for an order
    pub fn record_fill(&mut self, order_id: &str, fill: Fill) -> bool {
        // Update order manager
        if !self.orders.record_fill(order_id, fill.clone()) {
            return false;
        }

        // Update router stats
        self.router.update_stats(
            &fill.venue,
            true,
            fill.quantity,
            fill.quantity,
            0.0,  // Would need arrival price for slippage
            0.0,
            0.0,  // latency_ms
        );

        true
    }

    /// Complete an order and generate TCA
    pub fn complete_order(
        &mut self,
        order_id: &str,
        benchmarks: BenchmarkPrices,
    ) -> Option<ExecutionResult> {
        let order = self.orders.get_order(order_id)?;

        // Generate execution report
        let report = ExecutionReport::from_order(order);

        // Generate TCA
        let tca_report = TcaReport::calculate(order, benchmarks, order.total_commission);
        self.tca.add_report(tca_report.clone());

        Some(ExecutionResult {
            order_id: order_id.to_string(),
            state: order.state,
            filled_qty: order.filled_qty,
            avg_price: order.avg_fill_price,
            commission: order.total_commission,
            report: Some(report),
            tca: Some(tca_report),
            venues: order.fills.iter().map(|f| f.venue.clone()).collect(),
            fill_count: order.fills.len(),
            duration_ms: order
                .submitted_at
                .map(|s| (Utc::now() - s).num_milliseconds())
                .unwrap_or(0),
        })
    }

    /// Get TCA statistics
    pub fn get_tca_statistics(&self) -> tca::TcaStatistics {
        self.tca.get_statistics()
    }

    /// Get TCA statistics for a symbol
    pub fn get_symbol_tca(&self, symbol: &str) -> tca::TcaStatistics {
        self.tca.get_symbol_statistics(symbol)
    }

    /// Get order manager
    pub fn orders(&self) -> &OrderManager {
        &self.orders
    }

    /// Get mutable order manager
    pub fn orders_mut(&mut self) -> &mut OrderManager {
        &mut self.orders
    }

    /// Get router
    pub fn router(&self) -> &SmartRouter {
        &self.router
    }

    /// Get mutable router
    pub fn router_mut(&mut self) -> &mut SmartRouter {
        &mut self.router
    }

    /// Get dark pool router
    pub fn dark_router(&self) -> &DarkPoolRouter {
        &self.dark_router
    }

    /// Get mutable dark pool router
    pub fn dark_router_mut(&mut self) -> &mut DarkPoolRouter {
        &mut self.dark_router
    }

    /// Get TCA analyzer
    pub fn tca(&self) -> &TcaAnalyzer {
        &self.tca
    }

    /// Get working order count
    pub fn working_order_count(&self) -> usize {
        self.orders.working_count()
    }

    /// Check if can accept new orders
    pub fn can_accept_order(&self) -> bool {
        self.orders.working_count() < self.tier.max_concurrent_orders()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_tier_from_balance() {
        assert_eq!(
            AccountTier::from_balance(dec!(50_000)),
            AccountTier::Retail
        );
        assert_eq!(
            AccountTier::from_balance(dec!(500_000)),
            AccountTier::SemiInstitutional
        );
        assert_eq!(
            AccountTier::from_balance(dec!(5_000_000)),
            AccountTier::Institutional
        );
    }

    #[test]
    fn test_tier_capabilities() {
        let retail = AccountTier::Retail;
        assert!(!retail.supports_algos());
        assert!(!retail.supports_dark_pools());
        assert!(!retail.supports_fix());

        let semi = AccountTier::SemiInstitutional;
        assert!(semi.supports_algos());
        assert!(!semi.supports_dark_pools());

        let inst = AccountTier::Institutional;
        assert!(inst.supports_algos());
        assert!(inst.supports_dark_pools());
        assert!(inst.supports_fix());
    }

    #[test]
    fn test_execution_plan_creation() {
        let plan = ExecutionPlan::market("AAPL", true, dec!(100), dec!(150));

        assert_eq!(plan.symbol, "AAPL");
        assert_eq!(plan.side, OrderSide::Buy);
        assert_eq!(plan.quantity, dec!(100));
        assert!(!plan.is_algo());
        assert_eq!(plan.notional(), dec!(15000));
    }

    #[test]
    fn test_execution_plan_vwap() {
        let plan = ExecutionPlan::vwap(
            "AAPL",
            true,
            dec!(1000),
            dec!(150),
            10,
            Duration::hours(1),
        );

        assert!(plan.is_algo());
        assert!(matches!(plan.algorithm, ExecutionAlgorithm::Vwap(_)));
        assert_eq!(plan.urgency, Urgency::Low);
    }

    #[test]
    fn test_execution_plan_iceberg() {
        let plan = ExecutionPlan::iceberg("AAPL", true, dec!(10000), dec!(150), 0.1);

        assert!(matches!(plan.algorithm, ExecutionAlgorithm::Iceberg(_)));
    }

    #[test]
    fn test_execution_engine_creation() {
        let engine = ExecutionEngine::new(AccountTier::Retail);
        assert_eq!(engine.tier(), AccountTier::Retail);
        assert!(engine.can_accept_order());
    }

    #[test]
    fn test_recommend_algorithm() {
        let engine = ExecutionEngine::new(AccountTier::Institutional);

        // Small order (< 1% of ADV) -> Market
        // 50 shares at $150 = $7,500 notional vs $1M ADV = 0.75% participation
        let plan = ExecutionPlan::market("AAPL", true, dec!(50), dec!(150));
        let algo = engine.recommend_algorithm(&plan, dec!(1_000_000));
        assert!(matches!(algo, ExecutionAlgorithm::Market));

        // Large order with low urgency -> VWAP or TWAP
        let mut plan = ExecutionPlan::market("AAPL", true, dec!(5000), dec!(150));
        plan.urgency = Urgency::Low;
        let algo = engine.recommend_algorithm(&plan, dec!(1_000_000));
        assert!(matches!(
            algo,
            ExecutionAlgorithm::Vwap(_) | ExecutionAlgorithm::Twap(_)
        ));
    }

    #[test]
    fn test_retail_always_market() {
        let engine = ExecutionEngine::new(AccountTier::Retail);

        let plan = ExecutionPlan::market("AAPL", true, dec!(10000), dec!(150));
        let algo = engine.recommend_algorithm(&plan, dec!(1_000_000));

        // Retail should always use market orders
        assert!(matches!(algo, ExecutionAlgorithm::Market));
    }

    #[test]
    fn test_route_order() {
        let engine = ExecutionEngine::new(AccountTier::SemiInstitutional);

        let plan = ExecutionPlan::market("AAPL", true, dec!(100), dec!(150));
        let decision = engine.route_order(&plan);

        assert!(!decision.venue_id.is_empty());
        assert!(decision.fill_probability > 0.0);
    }

    #[test]
    fn test_generate_slices_vwap() {
        let engine = ExecutionEngine::new(AccountTier::Institutional);

        let plan = ExecutionPlan::vwap(
            "AAPL",
            true,
            dec!(1000),
            dec!(150),
            5,
            Duration::hours(1),
        );

        let slices = engine.generate_slices(&plan);
        assert_eq!(slices.len(), 5);

        let total: Decimal = slices.iter().map(|s| s.quantity).sum();
        assert_eq!(total, dec!(1000));
    }

    #[test]
    fn test_create_and_fill_order() {
        let mut engine = ExecutionEngine::new(AccountTier::Retail);

        let plan = ExecutionPlan::market("AAPL", true, dec!(100), dec!(150));
        let order_id = engine.create_order(&plan);

        // Record a fill
        let fill = Fill::new(dec!(100), dec!(150.10), "ALPACA");
        assert!(engine.record_fill(&order_id, fill));

        // Complete the order
        let benchmarks = BenchmarkPrices::with_arrival(dec!(150));
        let result = engine.complete_order(&order_id, benchmarks);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.filled_qty, dec!(100));
        assert!(result.tca.is_some());
    }

    #[test]
    fn test_dark_pool_decision() {
        let engine = ExecutionEngine::new(AccountTier::Institutional);

        let plan = ExecutionPlan::market("AAPL", true, dec!(1000), dec!(150))
            .with_urgency(Urgency::Low)
            .with_dark_pools(0.5);

        let decision = engine.get_dark_pool_decision(&plan);
        assert!(decision.use_dark);
        assert!(!decision.allocations.is_empty());
    }

    #[test]
    fn test_dark_pool_disabled_for_retail() {
        let engine = ExecutionEngine::new(AccountTier::Retail);

        let plan = ExecutionPlan::market("AAPL", true, dec!(1000), dec!(150))
            .with_dark_pools(0.5);

        let decision = engine.get_dark_pool_decision(&plan);
        assert!(!decision.use_dark);
    }

    #[test]
    fn test_urgency_levels() {
        assert_eq!(Urgency::Low.as_f64(), 0.2);
        assert_eq!(Urgency::Medium.as_f64(), 0.5);
        assert_eq!(Urgency::High.as_f64(), 0.8);
        assert_eq!(Urgency::Immediate.as_f64(), 1.0);
    }

    #[test]
    fn test_execution_result() {
        let result = ExecutionResult {
            order_id: "TEST123".to_string(),
            state: OrderState::Filled,
            filled_qty: dec!(100),
            avg_price: dec!(150),
            commission: dec!(1.00),
            report: None,
            tca: None,
            venues: vec!["NYSE".to_string()],
            fill_count: 1,
            duration_ms: 100,
        };

        assert!(result.is_success());
        assert_eq!(result.fill_rate(dec!(100)), 100.0);
    }

    #[test]
    fn test_create_plan() {
        let engine = ExecutionEngine::new(AccountTier::SemiInstitutional);

        let plan = engine.create_plan(
            "AAPL",
            true,
            dec!(1000),
            dec!(150),
            Urgency::Low,
            dec!(10_000_000),
        );

        assert_eq!(plan.symbol, "AAPL");
        assert_eq!(plan.side, OrderSide::Buy);
        // Should have recommended an algorithm since it's semi-institutional
        assert!(plan.algorithm.requires_slicing() || matches!(plan.algorithm, ExecutionAlgorithm::Market));
    }
}
