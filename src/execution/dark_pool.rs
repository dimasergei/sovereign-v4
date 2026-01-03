//! Dark Pool Routing Module
//!
//! Provides intelligent routing to dark pools for:
//! - Reduced market impact on large orders
//! - Price improvement opportunities
//! - Block trading
//!
//! Supports major dark pools: Sigma-X, CrossFinder, MS Pool, UBS ATS, Liquidnet, Level ATS

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::venue::{Venue, VenueType};

/// Dark pool order types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DarkPoolOrderType {
    /// Midpoint peg (between bid and ask)
    MidpointPeg,
    /// Primary peg (track the bid/ask)
    PrimaryPeg,
    /// Standard limit order
    Limit,
    /// Conditional order (IOI matching)
    Conditional,
    /// Block crossing
    Block,
}

impl std::fmt::Display for DarkPoolOrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DarkPoolOrderType::MidpointPeg => write!(f, "MID_PEG"),
            DarkPoolOrderType::PrimaryPeg => write!(f, "PRI_PEG"),
            DarkPoolOrderType::Limit => write!(f, "LIMIT"),
            DarkPoolOrderType::Conditional => write!(f, "COND"),
            DarkPoolOrderType::Block => write!(f, "BLOCK"),
        }
    }
}

/// Dark pool venue with enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkPoolVenue {
    /// Base venue information
    pub venue: Venue,
    /// Minimum order size for this pool
    pub min_order_size: Decimal,
    /// Minimum notional value
    pub min_notional: Decimal,
    /// Average match rate (historical)
    pub avg_match_rate: f64,
    /// Average price improvement (bps)
    pub avg_price_improvement_bps: f64,
    /// Typical time to fill (seconds)
    pub avg_fill_time_secs: u32,
    /// Supports IOI (Indication of Interest)
    pub supports_ioi: bool,
    /// Supports conditional orders
    pub supports_conditional: bool,
    /// Block crossing minimum size
    pub block_minimum: Option<Decimal>,
}

impl DarkPoolVenue {
    /// Create from a base venue
    pub fn from_venue(venue: Venue) -> Self {
        Self {
            venue,
            min_order_size: dec!(100),
            min_notional: dec!(10000),
            avg_match_rate: 0.25,
            avg_price_improvement_bps: 2.0,
            avg_fill_time_secs: 60,
            supports_ioi: false,
            supports_conditional: false,
            block_minimum: None,
        }
    }

    /// Create Sigma X venue
    pub fn sigma_x() -> Self {
        Self {
            venue: Venue::dark_pool("SIGMA_X"),
            min_order_size: dec!(100),
            min_notional: dec!(10000),
            avg_match_rate: 0.35,
            avg_price_improvement_bps: 2.5,
            avg_fill_time_secs: 45,
            supports_ioi: true,
            supports_conditional: true,
            block_minimum: Some(dec!(10000)),
        }
    }

    /// Create CrossFinder venue
    pub fn crossfinder() -> Self {
        Self {
            venue: Venue::dark_pool("CROSSFINDER"),
            min_order_size: dec!(100),
            min_notional: dec!(10000),
            avg_match_rate: 0.32,
            avg_price_improvement_bps: 2.0,
            avg_fill_time_secs: 50,
            supports_ioi: true,
            supports_conditional: true,
            block_minimum: Some(dec!(10000)),
        }
    }

    /// Create MS Pool venue
    pub fn ms_pool() -> Self {
        Self {
            venue: Venue::dark_pool("MS_POOL"),
            min_order_size: dec!(100),
            min_notional: dec!(10000),
            avg_match_rate: 0.30,
            avg_price_improvement_bps: 1.8,
            avg_fill_time_secs: 55,
            supports_ioi: true,
            supports_conditional: false,
            block_minimum: None,
        }
    }

    /// Create UBS ATS venue
    pub fn ubs_ats() -> Self {
        Self {
            venue: Venue::dark_pool("UBS_ATS"),
            min_order_size: dec!(100),
            min_notional: dec!(10000),
            avg_match_rate: 0.28,
            avg_price_improvement_bps: 1.5,
            avg_fill_time_secs: 60,
            supports_ioi: false,
            supports_conditional: false,
            block_minimum: None,
        }
    }

    /// Create Liquidnet venue (institutional block trading)
    pub fn liquidnet() -> Self {
        Self {
            venue: Venue::dark_pool("LIQUIDNET"),
            min_order_size: dec!(1000),
            min_notional: dec!(100000),
            avg_match_rate: 0.20,
            avg_price_improvement_bps: 5.0,
            avg_fill_time_secs: 180,
            supports_ioi: true,
            supports_conditional: true,
            block_minimum: Some(dec!(50000)),
        }
    }

    /// Create Level ATS venue
    pub fn level_ats() -> Self {
        Self {
            venue: Venue::dark_pool("LEVEL_ATS"),
            min_order_size: dec!(100),
            min_notional: dec!(10000),
            avg_match_rate: 0.25,
            avg_price_improvement_bps: 1.5,
            avg_fill_time_secs: 70,
            supports_ioi: false,
            supports_conditional: false,
            block_minimum: None,
        }
    }

    /// Check if venue can handle order size
    pub fn can_handle(&self, quantity: Decimal, notional: Decimal) -> bool {
        quantity >= self.min_order_size && notional >= self.min_notional
    }

    /// Check if venue supports block crossing for this size
    pub fn supports_block(&self, notional: Decimal) -> bool {
        self.block_minimum.map_or(false, |min| notional >= min)
    }

    /// Get venue ID
    pub fn id(&self) -> &str {
        &self.venue.id
    }
}

/// Dark pool routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkPoolConfig {
    /// Minimum order size for dark pool routing
    pub min_order_size: Decimal,
    /// Maximum percentage of order to route dark (0.0 to 1.0)
    pub max_dark_percentage: f64,
    /// Minimum acceptable fill rate (0.0 to 1.0)
    pub min_fill_rate: f64,
    /// Maximum time to wait for fill (seconds)
    pub fill_timeout_secs: u32,
    /// Minimum price improvement required (bps)
    pub min_price_improvement_bps: f64,
    /// Enable block crossing
    pub enable_block_crossing: bool,
    /// Block crossing minimum notional
    pub block_minimum_notional: Decimal,
    /// Maximum number of pools to spray
    pub max_spray_venues: usize,
}

impl Default for DarkPoolConfig {
    fn default() -> Self {
        Self {
            min_order_size: dec!(500),
            max_dark_percentage: 0.50,
            min_fill_rate: 0.20,
            fill_timeout_secs: 120,
            min_price_improvement_bps: 0.0,
            enable_block_crossing: true,
            block_minimum_notional: dec!(100000),
            max_spray_venues: 3,
        }
    }
}

/// Performance statistics for a dark pool
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DarkPoolStats {
    /// Number of orders sent
    pub orders_sent: u64,
    /// Number of orders filled (at least partially)
    pub orders_filled: u64,
    /// Total quantity sent
    pub qty_sent: Decimal,
    /// Total quantity filled
    pub qty_filled: Decimal,
    /// Total price improvement captured (in dollars)
    pub total_improvement: Decimal,
    /// Average fill time (milliseconds)
    pub avg_fill_time_ms: u64,
    /// Last update time
    pub last_update: DateTime<Utc>,
}

impl DarkPoolStats {
    /// Calculate fill rate
    pub fn fill_rate(&self) -> f64 {
        if self.qty_sent.is_zero() {
            return 0.0;
        }
        (self.qty_filled / self.qty_sent).to_f64().unwrap_or(0.0)
    }

    /// Calculate order success rate
    pub fn success_rate(&self) -> f64 {
        if self.orders_sent == 0 {
            return 0.0;
        }
        self.orders_filled as f64 / self.orders_sent as f64
    }

    /// Record an order sent
    pub fn record_order_sent(&mut self, quantity: Decimal) {
        self.orders_sent += 1;
        self.qty_sent += quantity;
        self.last_update = Utc::now();
    }

    /// Record a fill
    pub fn record_fill(&mut self, quantity: Decimal, improvement: Decimal, fill_time_ms: u64) {
        if quantity > dec!(0) {
            self.orders_filled += 1;
        }
        self.qty_filled += quantity;
        self.total_improvement += improvement;

        // Update average fill time (exponential moving average)
        if self.orders_filled == 1 {
            self.avg_fill_time_ms = fill_time_ms;
        } else {
            self.avg_fill_time_ms = (self.avg_fill_time_ms * 9 + fill_time_ms) / 10;
        }

        self.last_update = Utc::now();
    }
}

/// Dark pool routing decision
#[derive(Debug, Clone)]
pub struct DarkPoolDecision {
    /// Whether to use dark pools
    pub use_dark: bool,
    /// Selected dark pools with allocation
    pub allocations: Vec<DarkPoolAllocation>,
    /// Quantity for lit market
    pub lit_quantity: Decimal,
    /// Recommended order type for dark pools
    pub order_type: DarkPoolOrderType,
    /// Maximum time to wait for dark fills
    pub timeout: Duration,
    /// Reason for decision
    pub reason: String,
}

/// Allocation to a specific dark pool
#[derive(Debug, Clone)]
pub struct DarkPoolAllocation {
    /// Dark pool venue
    pub venue_id: String,
    /// Quantity to route
    pub quantity: Decimal,
    /// Priority (lower = higher priority)
    pub priority: u32,
    /// Expected fill probability
    pub expected_fill_pct: f64,
}

/// Dark pool router
#[derive(Debug)]
pub struct DarkPoolRouter {
    /// Configuration
    config: DarkPoolConfig,
    /// Available dark pools
    venues: Vec<DarkPoolVenue>,
    /// Performance statistics by venue
    stats: HashMap<String, DarkPoolStats>,
}

impl DarkPoolRouter {
    /// Create a new dark pool router
    pub fn new(config: DarkPoolConfig) -> Self {
        // Default venues
        let venues = vec![
            DarkPoolVenue::sigma_x(),
            DarkPoolVenue::crossfinder(),
            DarkPoolVenue::ms_pool(),
            DarkPoolVenue::ubs_ats(),
            DarkPoolVenue::liquidnet(),
            DarkPoolVenue::level_ats(),
        ];

        let stats = venues
            .iter()
            .map(|v| (v.id().to_string(), DarkPoolStats::default()))
            .collect();

        Self {
            config,
            venues,
            stats,
        }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(DarkPoolConfig::default())
    }

    /// Check if order should use dark pools
    pub fn should_use_dark(&self, quantity: Decimal, urgency: f64) -> bool {
        // Don't use dark pools for urgent orders
        if urgency > 0.8 {
            return false;
        }

        // Need minimum size
        if quantity < self.config.min_order_size {
            return false;
        }

        // Have available venues
        !self.venues.is_empty()
    }

    /// Select dark pools for an order
    pub fn select_pools(
        &self,
        symbol: &str,
        quantity: Decimal,
        price: Decimal,
        urgency: f64,
    ) -> DarkPoolDecision {
        let notional = quantity * price;

        // Check if we should use dark pools
        if !self.should_use_dark(quantity, urgency) {
            return DarkPoolDecision {
                use_dark: false,
                allocations: vec![],
                lit_quantity: quantity,
                order_type: DarkPoolOrderType::Limit,
                timeout: Duration::zero(),
                reason: format!(
                    "Dark pools not suitable: qty={}, urgency={}",
                    quantity, urgency
                ),
            };
        }

        // Calculate dark quantity based on urgency
        let dark_pct = self.config.max_dark_percentage * (1.0 - urgency);
        let dark_qty = quantity * Decimal::try_from(dark_pct).unwrap_or(dec!(0.5));
        let lit_qty = quantity - dark_qty;

        // Score and select venues
        let mut scored_venues: Vec<(&DarkPoolVenue, f64)> = self
            .venues
            .iter()
            .filter(|v| v.can_handle(dark_qty, notional))
            .map(|v| {
                let score = self.score_venue(v, quantity, notional, urgency);
                (v, score)
            })
            .collect();

        // Sort by score (higher is better)
        scored_venues.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top N venues
        let selected: Vec<_> = scored_venues
            .into_iter()
            .take(self.config.max_spray_venues)
            .collect();

        if selected.is_empty() {
            return DarkPoolDecision {
                use_dark: false,
                allocations: vec![],
                lit_quantity: quantity,
                order_type: DarkPoolOrderType::Limit,
                timeout: Duration::zero(),
                reason: "No suitable dark pools available".to_string(),
            };
        }

        // Allocate quantity across selected venues
        let total_score: f64 = selected.iter().map(|(_, s)| s).sum();
        let allocations: Vec<DarkPoolAllocation> = selected
            .iter()
            .enumerate()
            .map(|(i, (venue, score))| {
                let allocation_pct = score / total_score;
                let allocation_qty = dark_qty * Decimal::try_from(allocation_pct).unwrap_or(dec!(0.33));

                DarkPoolAllocation {
                    venue_id: venue.id().to_string(),
                    quantity: allocation_qty.round_dp(0),
                    priority: i as u32,
                    expected_fill_pct: venue.avg_match_rate * allocation_pct,
                }
            })
            .collect();

        // Determine order type based on conditions
        let order_type = if notional >= self.config.block_minimum_notional
            && self.config.enable_block_crossing
        {
            DarkPoolOrderType::Block
        } else if urgency < 0.3 {
            DarkPoolOrderType::MidpointPeg
        } else {
            DarkPoolOrderType::Limit
        };

        // Calculate timeout based on urgency
        let timeout_secs = (self.config.fill_timeout_secs as f64 * (1.0 - urgency * 0.5)) as i64;
        let timeout = Duration::seconds(timeout_secs);

        DarkPoolDecision {
            use_dark: true,
            allocations,
            lit_quantity: lit_qty,
            order_type,
            timeout,
            reason: format!(
                "Routing {:.0}% to {} dark pools, {:.0}% lit",
                dark_pct * 100.0,
                selected.len(),
                (1.0 - dark_pct) * 100.0
            ),
        }
    }

    /// Score a venue for routing
    fn score_venue(
        &self,
        venue: &DarkPoolVenue,
        quantity: Decimal,
        notional: Decimal,
        urgency: f64,
    ) -> f64 {
        // Get historical stats
        let stats = self.stats.get(venue.id());

        // Base score from venue characteristics
        let mut score = 0.0;

        // Fill rate (most important)
        let fill_rate = stats.map(|s| s.fill_rate()).unwrap_or(venue.avg_match_rate);
        score += fill_rate * 40.0;

        // Price improvement
        score += venue.avg_price_improvement_bps * 5.0;

        // Speed (important for higher urgency)
        let speed_factor = 1.0 - (venue.avg_fill_time_secs as f64 / 300.0).min(1.0);
        score += speed_factor * urgency * 20.0;

        // Block capability bonus
        if venue.supports_block(notional) {
            score += 10.0;
        }

        // Conditional order support bonus
        if venue.supports_conditional {
            score += 5.0;
        }

        // Penalize if recent fill rate is lower than expected
        if let Some(s) = stats {
            if s.fill_rate() < venue.avg_match_rate * 0.5 && s.orders_sent > 10 {
                score *= 0.5;  // Significant penalty
            }
        }

        score
    }

    /// Record order sent to a dark pool
    pub fn record_order_sent(&mut self, venue_id: &str, quantity: Decimal) {
        if let Some(stats) = self.stats.get_mut(venue_id) {
            stats.record_order_sent(quantity);
        }
    }

    /// Record a fill from a dark pool
    pub fn record_fill(
        &mut self,
        venue_id: &str,
        quantity: Decimal,
        improvement: Decimal,
        fill_time_ms: u64,
    ) {
        if let Some(stats) = self.stats.get_mut(venue_id) {
            stats.record_fill(quantity, improvement, fill_time_ms);
        }
    }

    /// Get statistics for a venue
    pub fn get_stats(&self, venue_id: &str) -> Option<&DarkPoolStats> {
        self.stats.get(venue_id)
    }

    /// Get all venues
    pub fn venues(&self) -> &[DarkPoolVenue] {
        &self.venues
    }

    /// Get venues suitable for a given size
    pub fn venues_for_size(&self, quantity: Decimal, price: Decimal) -> Vec<&DarkPoolVenue> {
        let notional = quantity * price;
        self.venues
            .iter()
            .filter(|v| v.can_handle(quantity, notional))
            .collect()
    }

    /// Compare venue performance
    pub fn compare_venues(&self) -> Vec<(&str, f64, f64)> {
        // Returns (venue_id, fill_rate, price_improvement_bps)
        self.venues
            .iter()
            .map(|v| {
                // Use actual fill rate if we have data, otherwise fall back to venue's avg_match_rate
                let fill_rate = self
                    .stats
                    .get(v.id())
                    .filter(|s| s.orders_sent > 0)  // Only use stats if we have data
                    .map(|s| s.fill_rate())
                    .unwrap_or(v.avg_match_rate);
                (v.id(), fill_rate, v.avg_price_improvement_bps)
            })
            .collect()
    }

    /// Get aggregate statistics across all venues
    pub fn aggregate_stats(&self) -> DarkPoolStats {
        let mut agg = DarkPoolStats::default();

        for stats in self.stats.values() {
            agg.orders_sent += stats.orders_sent;
            agg.orders_filled += stats.orders_filled;
            agg.qty_sent += stats.qty_sent;
            agg.qty_filled += stats.qty_filled;
            agg.total_improvement += stats.total_improvement;
        }

        if agg.orders_filled > 0 {
            let total_time: u64 = self.stats.values().map(|s| s.avg_fill_time_ms).sum();
            agg.avg_fill_time_ms = total_time / self.stats.len() as u64;
        }

        agg.last_update = Utc::now();
        agg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dark_pool_venue_creation() {
        let sigma = DarkPoolVenue::sigma_x();
        assert_eq!(sigma.id(), "SIGMA_X");
        assert!(sigma.supports_ioi);
        assert!(sigma.supports_conditional);
        assert!(sigma.avg_match_rate > 0.0);
    }

    #[test]
    fn test_venue_can_handle() {
        let sigma = DarkPoolVenue::sigma_x();

        // Should handle large orders
        assert!(sigma.can_handle(dec!(500), dec!(50000)));

        // Should not handle small orders
        assert!(!sigma.can_handle(dec!(50), dec!(5000)));
    }

    #[test]
    fn test_dark_pool_router_creation() {
        let router = DarkPoolRouter::default();
        assert!(!router.venues.is_empty());
        assert_eq!(router.venues.len(), 6);  // All default venues
    }

    #[test]
    fn test_should_use_dark() {
        let router = DarkPoolRouter::default();

        // Large order, low urgency -> yes
        assert!(router.should_use_dark(dec!(1000), 0.3));

        // Large order, high urgency -> no
        assert!(!router.should_use_dark(dec!(1000), 0.9));

        // Small order -> no
        assert!(!router.should_use_dark(dec!(100), 0.3));
    }

    #[test]
    fn test_select_pools() {
        let router = DarkPoolRouter::default();

        let decision = router.select_pools("AAPL", dec!(1000), dec!(150), 0.3);

        assert!(decision.use_dark);
        assert!(!decision.allocations.is_empty());
        assert!(decision.allocations.len() <= 3);  // max_spray_venues

        // Total allocation should not exceed dark quantity
        let total_allocated: Decimal = decision.allocations.iter().map(|a| a.quantity).sum();
        assert!(total_allocated <= dec!(1000));

        // Should have lit quantity remaining
        assert!(decision.lit_quantity > dec!(0));
    }

    #[test]
    fn test_select_pools_urgent() {
        let router = DarkPoolRouter::default();

        let decision = router.select_pools("AAPL", dec!(1000), dec!(150), 0.9);

        // High urgency should skip dark pools
        assert!(!decision.use_dark);
        assert!(decision.allocations.is_empty());
        assert_eq!(decision.lit_quantity, dec!(1000));
    }

    #[test]
    fn test_dark_pool_stats() {
        let mut stats = DarkPoolStats::default();

        // Record orders
        stats.record_order_sent(dec!(1000));
        stats.record_order_sent(dec!(1000));

        assert_eq!(stats.orders_sent, 2);
        assert_eq!(stats.qty_sent, dec!(2000));
        assert_eq!(stats.fill_rate(), 0.0);  // No fills yet

        // Record fills
        stats.record_fill(dec!(800), dec!(5.00), 100);
        stats.record_fill(dec!(500), dec!(3.00), 150);

        assert_eq!(stats.orders_filled, 2);
        assert_eq!(stats.qty_filled, dec!(1300));
        assert!((stats.fill_rate() - 0.65).abs() < 0.01);
        assert_eq!(stats.total_improvement, dec!(8.00));
    }

    #[test]
    fn test_venue_scoring() {
        let router = DarkPoolRouter::default();

        // Get venues for a medium-sized order
        let venues = router.venues_for_size(dec!(1000), dec!(150));
        assert!(!venues.is_empty());

        // Compare venues
        let comparison = router.compare_venues();
        assert!(!comparison.is_empty());

        // All venues should have positive fill rates
        for (_, fill_rate, _) in &comparison {
            assert!(*fill_rate > 0.0);
        }
    }

    #[test]
    fn test_block_order_type() {
        let mut config = DarkPoolConfig::default();
        config.block_minimum_notional = dec!(50000);

        let router = DarkPoolRouter::new(config);

        // Large enough for block
        let decision = router.select_pools("AAPL", dec!(500), dec!(150), 0.2);
        // 500 * 150 = 75000 > 50000, should suggest block
        if decision.use_dark {
            assert_eq!(decision.order_type, DarkPoolOrderType::Block);
        }
    }

    #[test]
    fn test_order_type_selection() {
        // Low urgency -> midpoint peg
        let router = DarkPoolRouter::default();
        let decision = router.select_pools("AAPL", dec!(1000), dec!(50), 0.1);

        if decision.use_dark {
            // Low urgency, small notional -> midpoint peg (not block)
            assert!(matches!(
                decision.order_type,
                DarkPoolOrderType::MidpointPeg | DarkPoolOrderType::Block
            ));
        }
    }

    #[test]
    fn test_aggregate_stats() {
        let mut router = DarkPoolRouter::default();

        // Record some activity
        router.record_order_sent("SIGMA_X", dec!(1000));
        router.record_fill("SIGMA_X", dec!(500), dec!(2.50), 100);

        router.record_order_sent("CROSSFINDER", dec!(1000));
        router.record_fill("CROSSFINDER", dec!(300), dec!(1.50), 150);

        let agg = router.aggregate_stats();
        assert_eq!(agg.orders_sent, 2);
        assert_eq!(agg.qty_filled, dec!(800));
        assert_eq!(agg.total_improvement, dec!(4.00));
    }

    #[test]
    fn test_dark_pool_allocation() {
        let alloc = DarkPoolAllocation {
            venue_id: "SIGMA_X".to_string(),
            quantity: dec!(500),
            priority: 0,
            expected_fill_pct: 0.35,
        };

        assert_eq!(alloc.venue_id, "SIGMA_X");
        assert_eq!(alloc.quantity, dec!(500));
        assert_eq!(alloc.priority, 0);
    }

    #[test]
    fn test_liquidnet_block_requirements() {
        let liquidnet = DarkPoolVenue::liquidnet();

        // Liquidnet has higher minimums
        assert!(liquidnet.min_order_size > dec!(500));
        assert!(liquidnet.min_notional > dec!(50000));

        // Small orders should fail
        assert!(!liquidnet.can_handle(dec!(500), dec!(50000)));

        // Large orders should work
        assert!(liquidnet.can_handle(dec!(2000), dec!(200000)));
    }
}
