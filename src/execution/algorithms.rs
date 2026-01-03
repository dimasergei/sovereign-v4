//! Execution Algorithms Module
//!
//! Implements institutional execution algorithms including:
//! - VWAP (Volume Weighted Average Price)
//! - TWAP (Time Weighted Average Price)
//! - POV (Percentage of Volume)
//! - Iceberg (Hidden quantity management)
//! - Adaptive (Market-responsive algorithm)
//! - Implementation Shortfall (Arrival price benchmark)

use chrono::{DateTime, Duration, Timelike, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// VWAP algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapConfig {
    /// Start time for VWAP execution
    pub start_time: DateTime<Utc>,
    /// End time for VWAP execution
    pub end_time: DateTime<Utc>,
    /// Number of slices to divide the order into
    pub num_slices: u32,
    /// Maximum participation rate (0.0 to 1.0)
    pub max_participation: f64,
    /// Minimum slice size (shares)
    pub min_slice_size: u32,
    /// Use historical volume profile
    pub use_volume_profile: bool,
    /// Allowed slippage from VWAP (bps)
    pub max_slippage_bps: f64,
}

impl Default for VwapConfig {
    fn default() -> Self {
        Self {
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::hours(1),
            num_slices: 10,
            max_participation: 0.10,
            min_slice_size: 100,
            use_volume_profile: true,
            max_slippage_bps: 10.0,
        }
    }
}

impl VwapConfig {
    /// Create VWAP config for a given duration
    pub fn with_duration(duration: Duration, num_slices: u32) -> Self {
        Self {
            start_time: Utc::now(),
            end_time: Utc::now() + duration,
            num_slices,
            ..Default::default()
        }
    }
}

/// TWAP algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapConfig {
    /// Start time for TWAP execution
    pub start_time: DateTime<Utc>,
    /// End time for TWAP execution
    pub end_time: DateTime<Utc>,
    /// Number of slices to divide the order into
    pub num_slices: u32,
    /// Randomization factor (0.0 to 1.0) - adds variance to slice timing
    pub randomization: f64,
    /// Minimum slice size (shares)
    pub min_slice_size: u32,
    /// Maximum price deviation from slice price (bps)
    pub max_deviation_bps: f64,
}

impl Default for TwapConfig {
    fn default() -> Self {
        Self {
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::hours(1),
            num_slices: 10,
            randomization: 0.20,
            min_slice_size: 100,
            max_deviation_bps: 5.0,
        }
    }
}

impl TwapConfig {
    /// Create TWAP config for a given duration
    pub fn with_duration(duration: Duration, num_slices: u32) -> Self {
        Self {
            start_time: Utc::now(),
            end_time: Utc::now() + duration,
            num_slices,
            ..Default::default()
        }
    }
}

/// POV (Percentage of Volume) algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PovConfig {
    /// Target participation rate (0.0 to 1.0)
    pub target_rate: f64,
    /// Maximum participation rate
    pub max_rate: f64,
    /// Minimum slice size (shares)
    pub min_slice_size: u32,
    /// Volume observation window (seconds)
    pub observation_window_secs: u32,
    /// Maximum duration for order completion
    pub max_duration: Duration,
}

impl Default for PovConfig {
    fn default() -> Self {
        Self {
            target_rate: 0.05,
            max_rate: 0.10,
            min_slice_size: 100,
            observation_window_secs: 60,
            max_duration: Duration::hours(4),
        }
    }
}

/// Iceberg algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergConfig {
    /// Visible (display) quantity
    pub display_qty: Decimal,
    /// Total order quantity
    pub total_qty: Decimal,
    /// Variance in display quantity (0.0 to 1.0)
    pub display_variance: f64,
    /// Minimum time between refreshes (milliseconds)
    pub min_refresh_ms: u32,
    /// Maximum time between refreshes (milliseconds)
    pub max_refresh_ms: u32,
    /// Price limit
    pub limit_price: Option<Decimal>,
}

impl Default for IcebergConfig {
    fn default() -> Self {
        Self {
            display_qty: dec!(100),
            total_qty: dec!(1000),
            display_variance: 0.20,
            min_refresh_ms: 100,
            max_refresh_ms: 500,
            limit_price: None,
        }
    }
}

impl IcebergConfig {
    /// Create iceberg with display ratio
    pub fn with_display_ratio(total_qty: Decimal, display_ratio: f64) -> Self {
        let display_qty = total_qty * Decimal::try_from(display_ratio).unwrap_or(dec!(0.1));
        Self {
            display_qty: display_qty.round_dp(0),
            total_qty,
            ..Default::default()
        }
    }
}

/// Adaptive algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Urgency level (0.0 = passive, 1.0 = aggressive)
    pub urgency: f64,
    /// Target completion time
    pub target_end: DateTime<Utc>,
    /// Risk aversion (0.0 = risk neutral, 1.0 = highly risk averse)
    pub risk_aversion: f64,
    /// Enable dark pool routing
    pub use_dark_pools: bool,
    /// Maximum market impact (bps)
    pub max_impact_bps: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            urgency: 0.5,
            target_end: Utc::now() + Duration::hours(1),
            risk_aversion: 0.5,
            use_dark_pools: true,
            max_impact_bps: 20.0,
        }
    }
}

/// Implementation Shortfall algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsConfig {
    /// Arrival price (benchmark)
    pub arrival_price: Decimal,
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Expected volatility (annualized)
    pub volatility: f64,
    /// Target completion time
    pub target_end: DateTime<Utc>,
    /// Maximum participation rate
    pub max_participation: f64,
}

impl Default for IsConfig {
    fn default() -> Self {
        Self {
            arrival_price: dec!(0),
            risk_aversion: 0.5,
            volatility: 0.20,
            target_end: Utc::now() + Duration::hours(2),
            max_participation: 0.15,
        }
    }
}

/// Execution algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionAlgorithm {
    /// Simple market order
    Market,
    /// Volume Weighted Average Price
    Vwap(VwapConfig),
    /// Time Weighted Average Price
    Twap(TwapConfig),
    /// Percentage of Volume
    Pov(PovConfig),
    /// Iceberg / Reserve order
    Iceberg(IcebergConfig),
    /// Adaptive market-responsive algorithm
    Adaptive(AdaptiveConfig),
    /// Implementation Shortfall (arrival price benchmark)
    ImplementationShortfall(IsConfig),
}

impl ExecutionAlgorithm {
    /// Get algorithm name
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionAlgorithm::Market => "MARKET",
            ExecutionAlgorithm::Vwap(_) => "VWAP",
            ExecutionAlgorithm::Twap(_) => "TWAP",
            ExecutionAlgorithm::Pov(_) => "POV",
            ExecutionAlgorithm::Iceberg(_) => "ICEBERG",
            ExecutionAlgorithm::Adaptive(_) => "ADAPTIVE",
            ExecutionAlgorithm::ImplementationShortfall(_) => "IS",
        }
    }

    /// Check if algorithm requires slicing
    pub fn requires_slicing(&self) -> bool {
        !matches!(self, ExecutionAlgorithm::Market)
    }

    /// Get expected duration
    pub fn expected_duration(&self) -> Option<Duration> {
        match self {
            ExecutionAlgorithm::Market => None,
            ExecutionAlgorithm::Vwap(c) => Some(c.end_time - c.start_time),
            ExecutionAlgorithm::Twap(c) => Some(c.end_time - c.start_time),
            ExecutionAlgorithm::Pov(c) => Some(c.max_duration),
            ExecutionAlgorithm::Iceberg(_) => None,  // Depends on fills
            ExecutionAlgorithm::Adaptive(c) => Some(c.target_end - Utc::now()),
            ExecutionAlgorithm::ImplementationShortfall(c) => Some(c.target_end - Utc::now()),
        }
    }
}

/// A slice of an order to be executed
#[derive(Debug, Clone)]
pub struct OrderSlice {
    /// Slice index
    pub index: u32,
    /// Quantity for this slice
    pub quantity: Decimal,
    /// Target execution time
    pub target_time: DateTime<Utc>,
    /// Limit price (if any)
    pub limit_price: Option<Decimal>,
    /// Whether slice has been executed
    pub executed: bool,
    /// Actual execution time
    pub execution_time: Option<DateTime<Utc>>,
    /// Actual execution price
    pub execution_price: Option<Decimal>,
    /// Actual quantity filled
    pub filled_qty: Decimal,
}

impl OrderSlice {
    /// Create a new order slice
    pub fn new(index: u32, quantity: Decimal, target_time: DateTime<Utc>) -> Self {
        Self {
            index,
            quantity,
            target_time,
            limit_price: None,
            executed: false,
            execution_time: None,
            execution_price: None,
            filled_qty: dec!(0),
        }
    }

    /// Set limit price
    pub fn with_limit(mut self, price: Decimal) -> Self {
        self.limit_price = Some(price);
        self
    }

    /// Mark slice as executed
    pub fn mark_executed(&mut self, price: Decimal, qty: Decimal) {
        self.executed = true;
        self.execution_time = Some(Utc::now());
        self.execution_price = Some(price);
        self.filled_qty = qty;
    }

    /// Check if slice is due for execution
    pub fn is_due(&self) -> bool {
        !self.executed && Utc::now() >= self.target_time
    }
}

/// Tracks state of algorithm execution
#[derive(Debug, Clone)]
pub struct AlgorithmState {
    /// Total order quantity
    pub total_qty: Decimal,
    /// Quantity filled so far
    pub filled_qty: Decimal,
    /// Quantity remaining
    pub remaining_qty: Decimal,
    /// Order slices
    pub slices: Vec<OrderSlice>,
    /// Current slice index
    pub current_slice: usize,
    /// Weighted average execution price
    pub avg_price: Decimal,
    /// Algorithm start time
    pub start_time: DateTime<Utc>,
    /// Target end time
    pub target_end: Option<DateTime<Utc>>,
    /// Whether algorithm is complete
    pub complete: bool,
}

impl AlgorithmState {
    /// Create new algorithm state
    pub fn new(total_qty: Decimal) -> Self {
        Self {
            total_qty,
            filled_qty: dec!(0),
            remaining_qty: total_qty,
            slices: Vec::new(),
            current_slice: 0,
            avg_price: dec!(0),
            start_time: Utc::now(),
            target_end: None,
            complete: false,
        }
    }

    /// Update state with a fill
    pub fn record_fill(&mut self, price: Decimal, qty: Decimal) {
        let old_value = self.avg_price * self.filled_qty;
        let new_value = price * qty;
        self.filled_qty += qty;
        self.remaining_qty = self.total_qty - self.filled_qty;

        if self.filled_qty > dec!(0) {
            self.avg_price = (old_value + new_value) / self.filled_qty;
        }

        if self.remaining_qty <= dec!(0) {
            self.complete = true;
        }
    }

    /// Get completion percentage
    pub fn completion_pct(&self) -> f64 {
        if self.total_qty.is_zero() {
            return 100.0;
        }
        (self.filled_qty / self.total_qty).to_f64().unwrap_or(0.0) * 100.0
    }

    /// Get next due slice
    pub fn next_due_slice(&self) -> Option<&OrderSlice> {
        self.slices.iter().find(|s| s.is_due())
    }
}

/// VWAP Scheduler - generates execution schedule based on volume profile
#[derive(Debug)]
pub struct VwapScheduler {
    config: VwapConfig,
    /// Intraday volume profile (normalized weights for each half-hour)
    volume_profile: Vec<f64>,
}

impl VwapScheduler {
    /// Create a new VWAP scheduler with default U-shaped volume profile
    pub fn new(config: VwapConfig) -> Self {
        // Default U-shaped intraday volume profile (per half hour, 9:30-16:00)
        // Higher at open and close, lower midday
        let volume_profile = vec![
            0.08,  // 9:30-10:00
            0.07,  // 10:00-10:30
            0.06,  // 10:30-11:00
            0.055, // 11:00-11:30
            0.05,  // 11:30-12:00
            0.045, // 12:00-12:30
            0.045, // 12:30-13:00
            0.05,  // 13:00-13:30
            0.055, // 13:30-14:00
            0.06,  // 14:00-14:30
            0.07,  // 14:30-15:00
            0.08,  // 15:00-15:30
            0.12,  // 15:30-16:00 (MOC orders)
        ];

        Self {
            config,
            volume_profile,
        }
    }

    /// Set custom volume profile
    pub fn with_volume_profile(mut self, profile: Vec<f64>) -> Self {
        self.volume_profile = profile;
        self
    }

    /// Generate execution slices
    pub fn generate_slices(&self, total_qty: Decimal) -> Vec<OrderSlice> {
        let mut slices = Vec::new();
        let duration = self.config.end_time - self.config.start_time;
        let slice_duration = duration / self.config.num_slices as i32;

        // Calculate weights for our time window
        let weights = self.calculate_weights();
        let total_weight: f64 = weights.iter().sum();

        for i in 0..self.config.num_slices {
            let weight = if i < weights.len() as u32 {
                weights[i as usize] / total_weight
            } else {
                1.0 / self.config.num_slices as f64
            };

            let qty = (total_qty.to_f64().unwrap_or(0.0) * weight).round();
            let qty = Decimal::try_from(qty).unwrap_or(dec!(0));
            let qty = qty.max(Decimal::from(self.config.min_slice_size));

            let target_time = self.config.start_time + slice_duration * i as i32;

            slices.push(OrderSlice::new(i, qty, target_time));
        }

        // Adjust last slice to ensure total matches
        let allocated: Decimal = slices.iter().map(|s| s.quantity).sum();
        if let Some(last) = slices.last_mut() {
            let adjustment = total_qty - allocated;
            last.quantity = (last.quantity + adjustment).max(dec!(0));
        }

        slices
    }

    /// Calculate weights based on volume profile for our time window
    fn calculate_weights(&self) -> Vec<f64> {
        if !self.config.use_volume_profile {
            // Uniform distribution
            return vec![1.0 / self.config.num_slices as f64; self.config.num_slices as usize];
        }

        // Map our execution window to volume profile buckets
        let mut weights = Vec::new();
        let duration = self.config.end_time - self.config.start_time;
        let slice_duration = duration / self.config.num_slices as i32;

        for i in 0..self.config.num_slices {
            let slice_time = self.config.start_time + slice_duration * i as i32;
            let bucket = self.get_volume_bucket(slice_time);
            let weight = if bucket < self.volume_profile.len() {
                self.volume_profile[bucket]
            } else {
                0.077 // Average weight if out of bounds
            };
            weights.push(weight);
        }

        weights
    }

    /// Get volume profile bucket for a given time
    fn get_volume_bucket(&self, time: DateTime<Utc>) -> usize {
        // Convert to market hours (assume ET, 9:30-16:00)
        let hour = time.time().hour();
        let minute = time.time().minute();
        let minutes_from_open = if hour >= 14 {  // UTC offset for ET
            (hour - 14) * 60 + minute - 30  // 9:30 ET = 14:30 UTC
        } else {
            0
        };

        // Each bucket is 30 minutes
        (minutes_from_open / 30) as usize
    }
}

/// TWAP Scheduler - generates time-uniform execution schedule
#[derive(Debug)]
pub struct TwapScheduler {
    config: TwapConfig,
}

impl TwapScheduler {
    /// Create a new TWAP scheduler
    pub fn new(config: TwapConfig) -> Self {
        Self { config }
    }

    /// Generate execution slices with optional randomization
    pub fn generate_slices(&self, total_qty: Decimal) -> Vec<OrderSlice> {
        let mut slices = Vec::new();
        let duration = self.config.end_time - self.config.start_time;
        let base_slice_duration = duration / self.config.num_slices as i32;

        let qty_per_slice = total_qty / Decimal::from(self.config.num_slices);
        let qty_per_slice = qty_per_slice.max(Decimal::from(self.config.min_slice_size));

        for i in 0..self.config.num_slices {
            // Add randomization to timing
            let randomization_offset = if self.config.randomization > 0.0 {
                let max_offset_secs = (base_slice_duration.num_seconds() as f64
                    * self.config.randomization) as i64;
                // Deterministic "random" based on slice index for reproducibility
                let offset = ((i as i64 * 7919) % (max_offset_secs * 2 + 1)) - max_offset_secs;
                Duration::seconds(offset)
            } else {
                Duration::zero()
            };

            let target_time = self.config.start_time
                + base_slice_duration * i as i32
                + randomization_offset;

            slices.push(OrderSlice::new(i, qty_per_slice, target_time));
        }

        // Adjust last slice quantity
        let allocated: Decimal = slices.iter().map(|s| s.quantity).sum();
        if let Some(last) = slices.last_mut() {
            let adjustment = total_qty - allocated;
            last.quantity = (last.quantity + adjustment).max(dec!(0));
        }

        slices
    }
}

/// Iceberg Manager - manages hidden quantity for iceberg orders
#[derive(Debug)]
pub struct IcebergManager {
    config: IcebergConfig,
    /// Remaining hidden quantity
    hidden_qty: Decimal,
    /// Currently displayed quantity
    displayed_qty: Decimal,
    /// Total filled quantity
    filled_qty: Decimal,
    /// Last refresh time
    last_refresh: DateTime<Utc>,
}

impl IcebergManager {
    /// Create a new iceberg manager
    pub fn new(config: IcebergConfig) -> Self {
        let displayed_qty = config.display_qty;
        let hidden_qty = config.total_qty - displayed_qty;

        Self {
            config,
            hidden_qty,
            displayed_qty,
            filled_qty: dec!(0),
            last_refresh: Utc::now(),
        }
    }

    /// Get current display quantity
    pub fn display_qty(&self) -> Decimal {
        self.displayed_qty
    }

    /// Get remaining hidden quantity
    pub fn hidden_qty(&self) -> Decimal {
        self.hidden_qty
    }

    /// Get total remaining quantity
    pub fn remaining_qty(&self) -> Decimal {
        self.displayed_qty + self.hidden_qty
    }

    /// Record a fill and potentially refresh display
    pub fn record_fill(&mut self, qty: Decimal) -> bool {
        self.filled_qty += qty;
        self.displayed_qty -= qty;

        // Check if we need to refresh display
        if self.displayed_qty <= dec!(0) && self.hidden_qty > dec!(0) {
            return self.refresh_display();
        }

        false
    }

    /// Refresh the displayed quantity from hidden reserve
    pub fn refresh_display(&mut self) -> bool {
        if self.hidden_qty <= dec!(0) {
            return false;
        }

        // Calculate new display quantity with variance
        let base_display = self.config.display_qty;
        let variance_factor = 1.0 + (self.config.display_variance * self.variance_offset());
        let new_display = base_display * Decimal::try_from(variance_factor).unwrap_or(dec!(1));
        let new_display = new_display.min(self.hidden_qty);

        self.displayed_qty = new_display.round_dp(0);
        self.hidden_qty -= self.displayed_qty;
        self.last_refresh = Utc::now();

        true
    }

    /// Check if enough time has passed for a refresh
    pub fn can_refresh(&self) -> bool {
        let elapsed = (Utc::now() - self.last_refresh).num_milliseconds() as u32;
        elapsed >= self.config.min_refresh_ms
    }

    /// Get variance offset based on current state
    fn variance_offset(&self) -> f64 {
        // Simple deterministic variance based on filled quantity
        let seed = self.filled_qty.to_f64().unwrap_or(0.0) as u64;
        ((seed * 7919) % 200) as f64 / 100.0 - 1.0  // Returns -1.0 to 1.0
    }

    /// Check if order is complete
    pub fn is_complete(&self) -> bool {
        self.remaining_qty() <= dec!(0)
    }
}

/// Algorithm factory methods
impl ExecutionAlgorithm {
    /// Create a market order algorithm
    pub fn market() -> Self {
        ExecutionAlgorithm::Market
    }

    /// Create a VWAP algorithm with duration
    pub fn vwap(duration_minutes: i64, num_slices: u32) -> Self {
        ExecutionAlgorithm::Vwap(VwapConfig::with_duration(
            Duration::minutes(duration_minutes),
            num_slices,
        ))
    }

    /// Create a TWAP algorithm with duration
    pub fn twap(duration_minutes: i64, num_slices: u32) -> Self {
        ExecutionAlgorithm::Twap(TwapConfig::with_duration(
            Duration::minutes(duration_minutes),
            num_slices,
        ))
    }

    /// Create an iceberg algorithm with display ratio
    pub fn iceberg(total_qty: Decimal, display_ratio: f64) -> Self {
        ExecutionAlgorithm::Iceberg(IcebergConfig::with_display_ratio(total_qty, display_ratio))
    }

    /// Create a POV algorithm with target participation
    pub fn pov(target_rate: f64) -> Self {
        ExecutionAlgorithm::Pov(PovConfig {
            target_rate,
            ..Default::default()
        })
    }

    /// Create an adaptive algorithm with urgency level
    pub fn adaptive(urgency: f64, duration_minutes: i64) -> Self {
        ExecutionAlgorithm::Adaptive(AdaptiveConfig {
            urgency,
            target_end: Utc::now() + Duration::minutes(duration_minutes),
            ..Default::default()
        })
    }
}

/// Estimate market impact for a given order size
pub fn estimate_market_impact(
    order_size: Decimal,
    adv: Decimal,  // Average Daily Volume
    volatility: f64,  // Daily volatility
    urgency: f64,  // 0.0 to 1.0
) -> f64 {
    if adv.is_zero() {
        return 0.0;
    }

    // Simple market impact model: I = σ * √(Q/ADV) * urgency_factor
    let participation = (order_size / adv).to_f64().unwrap_or(0.0);
    let urgency_factor = 0.5 + urgency * 0.5;  // 0.5 to 1.0

    volatility * participation.sqrt() * urgency_factor * 10000.0  // Convert to bps
}

/// Recommend algorithm based on order characteristics
pub fn recommend_algorithm(
    order_value: Decimal,
    adv_value: Decimal,  // ADV in dollars
    urgency: f64,
    volatility: f64,
) -> ExecutionAlgorithm {
    if adv_value.is_zero() {
        return ExecutionAlgorithm::market();
    }

    let participation = (order_value / adv_value).to_f64().unwrap_or(0.0);

    // High volatility should take priority for adaptive execution
    if volatility > 0.03 && urgency < 0.9 && participation >= 0.01 {
        return ExecutionAlgorithm::adaptive(urgency, 60);
    }

    // Decision matrix
    match (participation, urgency) {
        // Small orders or high urgency -> market
        (p, u) if p < 0.01 || u > 0.9 => ExecutionAlgorithm::market(),

        // Large orders, low urgency -> VWAP
        (p, u) if p > 0.05 && u < 0.3 => ExecutionAlgorithm::vwap(120, 20),

        // Medium orders, medium urgency -> TWAP
        (p, u) if p > 0.02 && u < 0.6 => ExecutionAlgorithm::twap(60, 10),

        // Default -> TWAP
        _ => ExecutionAlgorithm::twap(30, 6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_scheduler() {
        let config = VwapConfig::with_duration(Duration::hours(1), 6);
        let scheduler = VwapScheduler::new(config);

        let slices = scheduler.generate_slices(dec!(6000));

        assert_eq!(slices.len(), 6);

        // Total should equal order size
        let total: Decimal = slices.iter().map(|s| s.quantity).sum();
        assert_eq!(total, dec!(6000));

        // Each slice should have target time
        for (i, slice) in slices.iter().enumerate() {
            assert_eq!(slice.index, i as u32);
            assert!(!slice.executed);
        }
    }

    #[test]
    fn test_twap_scheduler() {
        let config = TwapConfig::with_duration(Duration::minutes(30), 6);
        let scheduler = TwapScheduler::new(config);

        let slices = scheduler.generate_slices(dec!(1200));

        assert_eq!(slices.len(), 6);

        let total: Decimal = slices.iter().map(|s| s.quantity).sum();
        assert_eq!(total, dec!(1200));

        // Each slice should be roughly equal (200 shares)
        for slice in &slices {
            assert!(slice.quantity >= dec!(100)); // Minimum slice size
        }
    }

    #[test]
    fn test_iceberg_manager() {
        let config = IcebergConfig {
            display_qty: dec!(100),
            total_qty: dec!(1000),
            ..Default::default()
        };

        let mut manager = IcebergManager::new(config);

        assert_eq!(manager.display_qty(), dec!(100));
        assert_eq!(manager.hidden_qty(), dec!(900));
        assert_eq!(manager.remaining_qty(), dec!(1000));

        // Record a fill
        let refreshed = manager.record_fill(dec!(100));
        assert!(refreshed);  // Should trigger refresh

        // Hidden should decrease, display should be refreshed
        assert!(manager.display_qty() > dec!(0));
        assert!(manager.hidden_qty() < dec!(900));
    }

    #[test]
    fn test_algorithm_state() {
        let mut state = AlgorithmState::new(dec!(1000));

        assert_eq!(state.completion_pct(), 0.0);
        assert!(!state.complete);

        // Record some fills
        state.record_fill(dec!(100), dec!(250));
        assert_eq!(state.filled_qty, dec!(250));
        assert_eq!(state.remaining_qty, dec!(750));
        assert!((state.completion_pct() - 25.0).abs() < 0.1);
        assert_eq!(state.avg_price, dec!(100));

        // Another fill at different price
        state.record_fill(dec!(102), dec!(250));
        assert_eq!(state.filled_qty, dec!(500));
        assert_eq!(state.avg_price, dec!(101)); // Weighted average
    }

    #[test]
    fn test_market_impact_estimation() {
        // Small order
        let impact = estimate_market_impact(
            dec!(10000),   // $10k order
            dec!(1000000), // $1M ADV
            0.02,          // 2% daily vol
            0.5,           // Medium urgency
        );
        assert!(impact < 20.0);  // Should be low impact

        // Large order
        let impact = estimate_market_impact(
            dec!(100000),  // $100k order
            dec!(1000000), // $1M ADV
            0.02,
            0.9,  // High urgency
        );
        assert!(impact > 20.0);  // Should be higher impact
    }

    #[test]
    fn test_algorithm_recommendation() {
        // Very small order with high urgency -> Market
        let algo = recommend_algorithm(
            dec!(5000),    // $5k order
            dec!(1000000), // $1M ADV
            0.95,          // Very high urgency forces market
            0.02,
        );
        assert!(matches!(algo, ExecutionAlgorithm::Market));

        // Large order, low urgency -> VWAP or TWAP
        let algo = recommend_algorithm(
            dec!(100000),  // $100k order (10% of ADV)
            dec!(1000000), // $1M ADV
            0.2,           // Low urgency
            0.02,
        );
        assert!(matches!(algo, ExecutionAlgorithm::Vwap(_) | ExecutionAlgorithm::Twap(_)));

        // High volatility -> Adaptive
        let algo = recommend_algorithm(
            dec!(50000),
            dec!(1000000),
            0.5,
            0.05,  // 5% daily vol
        );
        assert!(matches!(algo, ExecutionAlgorithm::Adaptive(_)));
    }

    #[test]
    fn test_execution_algorithm_factory() {
        let market = ExecutionAlgorithm::market();
        assert_eq!(market.name(), "MARKET");
        assert!(!market.requires_slicing());

        let vwap = ExecutionAlgorithm::vwap(60, 10);
        assert_eq!(vwap.name(), "VWAP");
        assert!(vwap.requires_slicing());

        let iceberg = ExecutionAlgorithm::iceberg(dec!(10000), 0.1);
        assert_eq!(iceberg.name(), "ICEBERG");
        if let ExecutionAlgorithm::Iceberg(config) = iceberg {
            assert_eq!(config.total_qty, dec!(10000));
            assert_eq!(config.display_qty, dec!(1000));
        }
    }

    #[test]
    fn test_order_slice() {
        let mut slice = OrderSlice::new(0, dec!(100), Utc::now() - Duration::seconds(1));

        assert!(slice.is_due());
        assert!(!slice.executed);

        slice.mark_executed(dec!(150.50), dec!(100));

        assert!(slice.executed);
        assert!(slice.execution_time.is_some());
        assert_eq!(slice.execution_price, Some(dec!(150.50)));
        assert_eq!(slice.filled_qty, dec!(100));
    }
}
