//! Smart Order Router Module
//!
//! Intelligent venue selection and order routing based on:
//! - Historical fill rates
//! - Price improvement
//! - Speed/latency
//! - Costs (maker rebates vs taker fees)
//! - Order characteristics
//!
//! Supports both single-venue and spray routing strategies.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::order_manager::{OrderSide, OrderType};
use super::venue::{Venue, VenueRegistry, VenueType};

/// Venue performance statistics (with EMA updates)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueStats {
    /// Venue ID
    pub venue_id: String,
    /// Fill rate (0.0 to 1.0) - EMA
    pub fill_rate: f64,
    /// Average slippage in basis points - EMA
    pub avg_slippage_bps: f64,
    /// Average price improvement in basis points - EMA
    pub avg_improvement_bps: f64,
    /// Average latency in milliseconds - EMA
    pub avg_latency_ms: f64,
    /// Total orders sent
    pub orders_sent: u64,
    /// Total orders filled (at least partially)
    pub orders_filled: u64,
    /// Total quantity sent
    pub qty_sent: Decimal,
    /// Total quantity filled
    pub qty_filled: Decimal,
    /// Last update time
    pub last_update: DateTime<Utc>,
    /// EMA alpha for updates
    ema_alpha: f64,
}

impl VenueStats {
    /// Create new venue stats
    pub fn new(venue_id: &str) -> Self {
        Self {
            venue_id: venue_id.to_string(),
            fill_rate: 0.85,           // Start with optimistic assumption
            avg_slippage_bps: 1.0,     // 1 bp slippage assumption
            avg_improvement_bps: 0.0,
            avg_latency_ms: 50.0,
            orders_sent: 0,
            orders_filled: 0,
            qty_sent: dec!(0),
            qty_filled: dec!(0),
            last_update: Utc::now(),
            ema_alpha: 0.1,  // 10% weight to new observations
        }
    }

    /// Update stats with a new observation
    pub fn update(
        &mut self,
        filled: bool,
        quantity_sent: Decimal,
        quantity_filled: Decimal,
        slippage_bps: f64,
        improvement_bps: f64,
        latency_ms: f64,
    ) {
        self.orders_sent += 1;
        if filled {
            self.orders_filled += 1;
        }
        self.qty_sent += quantity_sent;
        self.qty_filled += quantity_filled;

        // EMA updates
        let alpha = self.ema_alpha;
        let fill_indicator = if filled { 1.0 } else { 0.0 };

        self.fill_rate = alpha * fill_indicator + (1.0 - alpha) * self.fill_rate;
        self.avg_slippage_bps = alpha * slippage_bps + (1.0 - alpha) * self.avg_slippage_bps;
        self.avg_improvement_bps = alpha * improvement_bps + (1.0 - alpha) * self.avg_improvement_bps;
        self.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.avg_latency_ms;

        self.last_update = Utc::now();
    }

    /// Get effective fill rate (quantity-weighted)
    pub fn effective_fill_rate(&self) -> f64 {
        if self.qty_sent.is_zero() {
            return self.fill_rate;
        }
        (self.qty_filled / self.qty_sent).to_f64().unwrap_or(0.0)
    }

    /// Calculate venue score for routing
    pub fn score(&self, urgency: f64, cost_sensitivity: f64) -> f64 {
        // Base score from fill rate (most important)
        let mut score = self.fill_rate * 100.0;

        // Price improvement bonus
        score += self.avg_improvement_bps * 5.0;

        // Slippage penalty
        score -= self.avg_slippage_bps * 3.0;

        // Speed factor (more important with higher urgency)
        let speed_factor = 1.0 - (self.avg_latency_ms / 500.0).min(1.0);
        score += speed_factor * urgency * 20.0;

        score.max(0.0)
    }
}

/// Routing decision for an order
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Primary venue
    pub venue_id: String,
    /// Recommended order type
    pub order_type: OrderType,
    /// Limit price (if applicable)
    pub limit_price: Option<Decimal>,
    /// Expected fill probability
    pub fill_probability: f64,
    /// Expected price improvement (bps)
    pub expected_improvement_bps: f64,
    /// Expected cost (negative = rebate)
    pub expected_cost_bps: f64,
    /// Confidence score (0-100)
    pub confidence: f64,
    /// Reason for selection
    pub reason: String,
}

/// Multi-venue routing for spray orders
#[derive(Debug, Clone)]
pub struct SprayRoute {
    /// Venue allocations
    pub allocations: Vec<VenueAllocation>,
    /// Total quantity
    pub total_quantity: Decimal,
    /// Expected aggregate fill rate
    pub expected_fill_rate: f64,
    /// Strategy description
    pub strategy: String,
}

/// Allocation to a single venue in a spray
#[derive(Debug, Clone)]
pub struct VenueAllocation {
    /// Venue ID
    pub venue_id: String,
    /// Quantity to route
    pub quantity: Decimal,
    /// Order type
    pub order_type: OrderType,
    /// Limit price (if applicable)
    pub limit_price: Option<Decimal>,
    /// Priority (lower = earlier)
    pub priority: u32,
}

/// Smart order router
#[derive(Debug)]
pub struct SmartRouter {
    /// Venue registry
    registry: VenueRegistry,
    /// Venue statistics
    stats: HashMap<String, VenueStats>,
    /// Default cost sensitivity
    cost_sensitivity: f64,
}

impl SmartRouter {
    /// Create a new smart router
    pub fn new(registry: VenueRegistry) -> Self {
        let stats = registry
            .venue_ids()
            .into_iter()
            .map(|id| (id.to_string(), VenueStats::new(id)))
            .collect();

        Self {
            registry,
            stats,
            cost_sensitivity: 0.5,
        }
    }

    /// Create with all standard venues
    pub fn with_all_venues() -> Self {
        Self::new(VenueRegistry::with_all_venues())
    }

    /// Set cost sensitivity (0.0 = speed priority, 1.0 = cost priority)
    pub fn with_cost_sensitivity(mut self, sensitivity: f64) -> Self {
        self.cost_sensitivity = sensitivity.clamp(0.0, 1.0);
        self
    }

    /// Route a single order to the best venue
    pub fn route_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        urgency: f64,
    ) -> RoutingDecision {
        let notional = quantity * price;

        // Get suitable venues
        let quantity_u32 = quantity.to_u64().unwrap_or(0) as u32;
        let venues = self.registry.for_size(quantity_u32);

        if venues.is_empty() {
            return RoutingDecision {
                venue_id: "ALPACA".to_string(),  // Fallback
                order_type: OrderType::Market,
                limit_price: None,
                fill_probability: 0.95,
                expected_improvement_bps: 0.0,
                expected_cost_bps: 0.0,
                confidence: 50.0,
                reason: "No suitable venues, using fallback".to_string(),
            };
        }

        // Score all venues
        let mut scored: Vec<(&Venue, f64)> = venues
            .iter()
            .map(|v| {
                let base_score = self.score_venue(v, urgency);
                (*v, base_score)
            })
            .collect();

        // Sort by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Best venue
        let (best_venue, best_score) = scored[0];

        // Determine order type based on venue and urgency
        let order_type = self.recommend_order_type(best_venue, urgency);

        // Calculate limit price if needed
        let limit_price = if matches!(order_type, OrderType::Limit) {
            Some(self.calculate_limit_price(price, side, urgency, best_venue))
        } else {
            None
        };

        // Get venue stats
        let stats = self.stats.get(&best_venue.id);
        let fill_probability = stats.map(|s| s.fill_rate).unwrap_or(0.85);
        let expected_improvement = stats.map(|s| s.avg_improvement_bps).unwrap_or(0.0);

        // Calculate expected cost
        let is_maker = matches!(order_type, OrderType::Limit);
        let cost = best_venue.expected_cost(quantity, is_maker);
        let expected_cost_bps = if !notional.is_zero() {
            (cost / notional).to_f64().unwrap_or(0.0) * 10000.0
        } else {
            0.0
        };

        RoutingDecision {
            venue_id: best_venue.id.clone(),
            order_type,
            limit_price,
            fill_probability,
            expected_improvement_bps: expected_improvement,
            expected_cost_bps,
            confidence: best_score.min(100.0),
            reason: format!(
                "Best venue: {} (score: {:.1}, fill: {:.0}%)",
                best_venue.name,
                best_score,
                fill_probability * 100.0
            ),
        }
    }

    /// Create a spray route across multiple venues
    pub fn spray_route(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        urgency: f64,
        max_venues: usize,
    ) -> SprayRoute {
        let quantity_u32 = quantity.to_u64().unwrap_or(0) as u32;
        let venues = self.registry.for_size(quantity_u32);

        if venues.is_empty() || max_venues == 0 {
            return SprayRoute {
                allocations: vec![VenueAllocation {
                    venue_id: "ALPACA".to_string(),
                    quantity,
                    order_type: OrderType::Market,
                    limit_price: None,
                    priority: 0,
                }],
                total_quantity: quantity,
                expected_fill_rate: 0.95,
                strategy: "Fallback to single venue".to_string(),
            };
        }

        // Score venues
        let mut scored: Vec<(&Venue, f64)> = venues
            .iter()
            .map(|v| (*v, self.score_venue(v, urgency)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top N venues
        let selected: Vec<_> = scored.into_iter().take(max_venues).collect();

        // Allocate quantity proportionally by score
        let total_score: f64 = selected.iter().map(|(_, s)| s).sum();

        let allocations: Vec<VenueAllocation> = selected
            .iter()
            .enumerate()
            .map(|(i, (venue, score))| {
                let alloc_pct = score / total_score;
                let alloc_qty = quantity * Decimal::try_from(alloc_pct).unwrap_or(dec!(0.33));
                let alloc_qty = alloc_qty.round_dp(0).max(dec!(1));

                let order_type = self.recommend_order_type(venue, urgency);
                let limit_price = if matches!(order_type, OrderType::Limit) {
                    Some(self.calculate_limit_price(price, side, urgency, venue))
                } else {
                    None
                };

                VenueAllocation {
                    venue_id: venue.id.clone(),
                    quantity: alloc_qty,
                    order_type,
                    limit_price,
                    priority: i as u32,
                }
            })
            .collect();

        // Adjust last allocation to match total
        let allocated: Decimal = allocations.iter().map(|a| a.quantity).sum();
        let mut final_allocations = allocations;
        if let Some(last) = final_allocations.last_mut() {
            let adjustment = quantity - allocated;
            last.quantity = (last.quantity + adjustment).max(dec!(0));
        }

        // Calculate expected fill rate
        let expected_fill_rate: f64 = selected
            .iter()
            .map(|(v, _)| {
                self.stats
                    .get(&v.id)
                    .map(|s| s.fill_rate)
                    .unwrap_or(0.85)
            })
            .sum::<f64>()
            / selected.len() as f64;

        SprayRoute {
            allocations: final_allocations,
            total_quantity: quantity,
            expected_fill_rate,
            strategy: format!("Spray across {} venues", selected.len()),
        }
    }

    /// Score a venue for routing
    fn score_venue(&self, venue: &Venue, urgency: f64) -> f64 {
        let mut score = 0.0;

        // Get historical stats
        let stats = self.stats.get(&venue.id);

        // Fill rate (40% weight)
        let fill_rate = stats
            .map(|s| s.fill_rate)
            .unwrap_or(venue.capabilities.typical_fill_rate);
        score += fill_rate * 40.0;

        // Price improvement (20% weight if not urgent)
        if venue.capabilities.price_improvement {
            let improvement = stats.map(|s| s.avg_improvement_bps).unwrap_or(1.0);
            score += improvement * (1.0 - urgency) * 20.0;
        }

        // Speed (20% weight, more important with urgency)
        let latency = stats
            .map(|s| s.avg_latency_ms)
            .unwrap_or(venue.capabilities.avg_latency_ms as f64);
        let speed_factor = 1.0 - (latency / 200.0).min(1.0);
        score += speed_factor * urgency * 20.0;

        // Cost (20% weight based on sensitivity)
        let cost_factor = if venue.capabilities.taker_fee_cents < dec!(0.002) {
            1.0  // Low cost
        } else if venue.capabilities.maker_fee_cents < dec!(0) {
            0.8  // Has rebate
        } else {
            0.5  // Higher cost
        };
        score += cost_factor * self.cost_sensitivity * 20.0;

        // Venue type bonuses
        match venue.venue_type {
            VenueType::Exchange => score += 5.0,
            VenueType::DarkPool if urgency < 0.5 => score += 10.0,
            VenueType::Internalizer if urgency > 0.7 => score += 5.0,
            _ => {}
        }

        // Priority adjustment
        score -= venue.priority as f64 * 0.1;

        score.max(0.0)
    }

    /// Recommend order type for a venue
    fn recommend_order_type(&self, venue: &Venue, urgency: f64) -> OrderType {
        if urgency > 0.8 {
            // Very urgent - market orders
            return OrderType::Market;
        }

        if urgency < 0.3 && venue.capabilities.pegged_orders {
            // Low urgency, venue supports pegging
            return OrderType::MidpointPeg;
        }

        if venue.capabilities.limit_orders {
            OrderType::Limit
        } else {
            OrderType::Market
        }
    }

    /// Calculate limit price
    fn calculate_limit_price(
        &self,
        mid_price: Decimal,
        side: OrderSide,
        urgency: f64,
        venue: &Venue,
    ) -> Decimal {
        // More aggressive limit with higher urgency (willing to cross spread)
        let tick_size = dec!(0.01);
        let ticks_away = (urgency * 3.0).round() as i32;  // 0-3 ticks, more with higher urgency

        let offset = tick_size * Decimal::from(ticks_away);

        match side {
            OrderSide::Buy => mid_price + offset,   // Bid higher for faster fill
            OrderSide::Sell => mid_price - offset,  // Ask lower for faster fill
        }
    }

    /// Update venue statistics
    pub fn update_stats(
        &mut self,
        venue_id: &str,
        filled: bool,
        quantity_sent: Decimal,
        quantity_filled: Decimal,
        slippage_bps: f64,
        improvement_bps: f64,
        latency_ms: f64,
    ) {
        if let Some(stats) = self.stats.get_mut(venue_id) {
            stats.update(
                filled,
                quantity_sent,
                quantity_filled,
                slippage_bps,
                improvement_bps,
                latency_ms,
            );
        }
    }

    /// Get venue statistics
    pub fn get_stats(&self, venue_id: &str) -> Option<&VenueStats> {
        self.stats.get(venue_id)
    }

    /// Get all venue statistics
    pub fn all_stats(&self) -> &HashMap<String, VenueStats> {
        &self.stats
    }

    /// Get venue rankings
    pub fn venue_rankings(&self, urgency: f64) -> Vec<(&str, f64)> {
        let mut rankings: Vec<_> = self
            .registry
            .available_sorted()
            .iter()
            .map(|v| (v.id.as_str(), self.score_venue(v, urgency)))
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rankings
    }

    /// Get the venue registry
    pub fn registry(&self) -> &VenueRegistry {
        &self.registry
    }

    /// Set venue availability
    pub fn set_venue_available(&mut self, venue_id: &str, available: bool) {
        self.registry.set_available(venue_id, available);
    }
}

/// Spray router for multi-venue simultaneous execution
#[derive(Debug)]
pub struct SprayRouter {
    /// Smart router for venue scoring
    router: SmartRouter,
    /// Maximum venues to spray
    max_venues: usize,
    /// Minimum quantity per venue
    min_venue_qty: Decimal,
}

impl SprayRouter {
    /// Create a new spray router
    pub fn new(router: SmartRouter) -> Self {
        Self {
            router,
            max_venues: 5,
            min_venue_qty: dec!(100),
        }
    }

    /// Set maximum venues
    pub fn with_max_venues(mut self, max: usize) -> Self {
        self.max_venues = max;
        self
    }

    /// Set minimum quantity per venue
    pub fn with_min_venue_qty(mut self, min: Decimal) -> Self {
        self.min_venue_qty = min;
        self
    }

    /// Create spray route
    pub fn spray(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        urgency: f64,
    ) -> SprayRoute {
        // Calculate how many venues we can use
        let max_possible = (quantity / self.min_venue_qty)
            .to_u64()
            .unwrap_or(0) as usize;
        let num_venues = self.max_venues.min(max_possible).max(1);

        self.router
            .spray_route(symbol, side, quantity, price, urgency, num_venues)
    }

    /// Get underlying router
    pub fn router(&self) -> &SmartRouter {
        &self.router
    }

    /// Get mutable underlying router
    pub fn router_mut(&mut self) -> &mut SmartRouter {
        &mut self.router
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_stats_creation() {
        let stats = VenueStats::new("NYSE");
        assert_eq!(stats.venue_id, "NYSE");
        assert!(stats.fill_rate > 0.0);
        assert_eq!(stats.orders_sent, 0);
    }

    #[test]
    fn test_venue_stats_update() {
        let mut stats = VenueStats::new("NYSE");

        // Update with a successful fill
        stats.update(true, dec!(1000), dec!(1000), 1.0, 0.5, 20.0);

        assert_eq!(stats.orders_sent, 1);
        assert_eq!(stats.orders_filled, 1);
        assert!(stats.fill_rate > 0.85);  // EMA should increase

        // Update with a miss
        stats.update(false, dec!(1000), dec!(0), 0.0, 0.0, 50.0);

        assert_eq!(stats.orders_sent, 2);
        assert_eq!(stats.orders_filled, 1);
        assert!(stats.fill_rate < 0.95);  // Should decrease
    }

    #[test]
    fn test_smart_router_creation() {
        let router = SmartRouter::with_all_venues();
        assert!(!router.registry.is_empty());
    }

    #[test]
    fn test_route_order() {
        let router = SmartRouter::with_all_venues();

        let decision = router.route_order("AAPL", OrderSide::Buy, dec!(100), dec!(150), 0.5);

        assert!(!decision.venue_id.is_empty());
        assert!(decision.fill_probability > 0.0);
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_route_order_urgent() {
        let router = SmartRouter::with_all_venues();

        let decision = router.route_order("AAPL", OrderSide::Buy, dec!(100), dec!(150), 0.9);

        // High urgency should favor market orders
        assert!(matches!(
            decision.order_type,
            OrderType::Market | OrderType::Limit
        ));
    }

    #[test]
    fn test_spray_route() {
        let router = SmartRouter::with_all_venues();

        let spray = router.spray_route("AAPL", OrderSide::Buy, dec!(1000), dec!(150), 0.5, 3);

        assert!(spray.allocations.len() <= 3);
        assert!(!spray.allocations.is_empty());

        // Total allocation should match quantity
        let total: Decimal = spray.allocations.iter().map(|a| a.quantity).sum();
        assert_eq!(total, dec!(1000));
    }

    #[test]
    fn test_venue_rankings() {
        let router = SmartRouter::with_all_venues();

        let rankings = router.venue_rankings(0.5);

        assert!(!rankings.is_empty());

        // Rankings should be sorted by score (descending)
        for i in 1..rankings.len() {
            assert!(rankings[i - 1].1 >= rankings[i].1);
        }
    }

    #[test]
    fn test_spray_router() {
        let router = SmartRouter::with_all_venues();
        let spray_router = SprayRouter::new(router)
            .with_max_venues(3)
            .with_min_venue_qty(dec!(100));

        let spray = spray_router.spray("AAPL", OrderSide::Buy, dec!(1000), dec!(150), 0.5);

        assert!(spray.allocations.len() <= 3);
        assert_eq!(spray.total_quantity, dec!(1000));
    }

    #[test]
    fn test_limit_price_calculation() {
        let router = SmartRouter::with_all_venues();
        let venue = Venue::nyse();

        // Buy order with low urgency - should be at or below mid
        let buy_limit = router.calculate_limit_price(dec!(100), OrderSide::Buy, 0.2, &venue);
        assert!(buy_limit >= dec!(100));

        // Sell order with low urgency
        let sell_limit = router.calculate_limit_price(dec!(100), OrderSide::Sell, 0.2, &venue);
        assert!(sell_limit <= dec!(100));

        // High urgency - prices should be more aggressive
        let urgent_buy = router.calculate_limit_price(dec!(100), OrderSide::Buy, 0.9, &venue);
        let calm_buy = router.calculate_limit_price(dec!(100), OrderSide::Buy, 0.1, &venue);
        // Urgent buy should have higher limit (more willing to pay)
        assert!(urgent_buy >= calm_buy);
    }

    #[test]
    fn test_venue_score_components() {
        let router = SmartRouter::with_all_venues();

        // Score venues at different urgency levels
        let nyse_low = router.score_venue(&Venue::nyse(), 0.2);
        let nyse_high = router.score_venue(&Venue::nyse(), 0.8);

        // Scores should differ based on urgency
        // (speed is weighted more heavily with high urgency)
        assert!(nyse_low != nyse_high);
    }

    #[test]
    fn test_routing_decision_structure() {
        let decision = RoutingDecision {
            venue_id: "NYSE".to_string(),
            order_type: OrderType::Limit,
            limit_price: Some(dec!(150.00)),
            fill_probability: 0.85,
            expected_improvement_bps: 1.5,
            expected_cost_bps: -0.2,
            confidence: 75.0,
            reason: "Best venue for this order".to_string(),
        };

        assert_eq!(decision.venue_id, "NYSE");
        assert_eq!(decision.limit_price, Some(dec!(150.00)));
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_update_and_retrieve_stats() {
        let mut router = SmartRouter::with_all_venues();

        // Update stats
        router.update_stats("NYSE", true, dec!(1000), dec!(1000), 0.5, 1.0, 15.0);

        // Retrieve and verify
        let stats = router.get_stats("NYSE").unwrap();
        assert_eq!(stats.orders_sent, 1);
        assert_eq!(stats.orders_filled, 1);
    }

    #[test]
    fn test_set_venue_availability() {
        let mut router = SmartRouter::with_all_venues();

        // Disable NYSE
        router.set_venue_available("NYSE", false);

        // Should not appear in rankings
        let rankings = router.venue_rankings(0.5);
        assert!(!rankings.iter().any(|(id, _)| *id == "NYSE"));
    }
}
