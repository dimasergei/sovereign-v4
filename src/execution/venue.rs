//! Venue Definitions Module
//!
//! Defines all execution venues including exchanges, dark pools, retail brokers,
//! and internalizers with their capabilities and characteristics.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Type of execution venue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VenueType {
    /// Lit exchange (NYSE, NASDAQ, ARCA, BATS, IEX)
    Exchange,
    /// Dark pool (Sigma-X, CrossFinder, MS Pool, UBS ATS)
    DarkPool,
    /// Retail broker (Alpaca, IBKR)
    RetailBroker,
    /// Market maker / internalizer (Citadel, Virtu)
    Internalizer,
}

impl fmt::Display for VenueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VenueType::Exchange => write!(f, "Exchange"),
            VenueType::DarkPool => write!(f, "Dark Pool"),
            VenueType::RetailBroker => write!(f, "Retail Broker"),
            VenueType::Internalizer => write!(f, "Internalizer"),
        }
    }
}

/// Capabilities of an execution venue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueCapabilities {
    /// Supports market orders
    pub market_orders: bool,
    /// Supports limit orders
    pub limit_orders: bool,
    /// Supports pegged orders (midpoint peg, primary peg)
    pub pegged_orders: bool,
    /// Supports iceberg/reserve orders
    pub iceberg_orders: bool,
    /// Provides price improvement opportunities
    pub price_improvement: bool,
    /// Good for large block orders
    pub large_orders: bool,
    /// Typical fill rate (0.0 to 1.0)
    pub typical_fill_rate: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: u32,
    /// Maker rebate in cents per share (negative = rebate, positive = fee)
    pub maker_fee_cents: Decimal,
    /// Taker fee in cents per share
    pub taker_fee_cents: Decimal,
    /// Minimum order size (shares)
    pub min_order_size: u32,
    /// Maximum order size (shares), None = unlimited
    pub max_order_size: Option<u32>,
}

impl Default for VenueCapabilities {
    fn default() -> Self {
        Self {
            market_orders: true,
            limit_orders: true,
            pegged_orders: false,
            iceberg_orders: false,
            price_improvement: false,
            large_orders: false,
            typical_fill_rate: 0.90,
            avg_latency_ms: 50,
            maker_fee_cents: dec!(0.0030),  // 30 mils
            taker_fee_cents: dec!(0.0030),
            min_order_size: 1,
            max_order_size: None,
        }
    }
}

/// An execution venue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Venue {
    /// Unique venue identifier (e.g., "NYSE", "SIGMA_X")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Type of venue
    pub venue_type: VenueType,
    /// Market identifier code (ISO 10383)
    pub mic: Option<String>,
    /// Venue capabilities
    pub capabilities: VenueCapabilities,
    /// Whether venue is currently available
    pub available: bool,
    /// Priority for routing (lower = higher priority)
    pub priority: u32,
}

impl Venue {
    /// Create a new venue
    pub fn new(id: &str, name: &str, venue_type: VenueType) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            venue_type,
            mic: None,
            capabilities: VenueCapabilities::default(),
            available: true,
            priority: 100,
        }
    }

    /// Set Market Identifier Code
    pub fn with_mic(mut self, mic: &str) -> Self {
        self.mic = Some(mic.to_string());
        self
    }

    /// Set capabilities
    pub fn with_capabilities(mut self, capabilities: VenueCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    // ========== Retail Broker Factory Methods ==========

    /// Alpaca Markets broker
    pub fn alpaca() -> Self {
        Self::new("ALPACA", "Alpaca Markets", VenueType::RetailBroker)
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: false,
                iceberg_orders: false,
                price_improvement: false,
                large_orders: false,
                typical_fill_rate: 0.95,
                avg_latency_ms: 100,
                maker_fee_cents: dec!(0.0),  // Commission-free
                taker_fee_cents: dec!(0.0),
                min_order_size: 1,
                max_order_size: None,
            })
            .with_priority(90)
    }

    /// Interactive Brokers with Smart Routing
    pub fn ibkr_smart() -> Self {
        Self::new("IBKR_SMART", "IBKR Smart Router", VenueType::RetailBroker)
            .with_mic("SMART")
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.98,
                avg_latency_ms: 20,
                maker_fee_cents: dec!(-0.002),  // Rebate
                taker_fee_cents: dec!(0.005),
                min_order_size: 1,
                max_order_size: None,
            })
            .with_priority(50)
    }

    // ========== Exchange Factory Methods ==========

    /// New York Stock Exchange
    pub fn nyse() -> Self {
        Self::new("NYSE", "New York Stock Exchange", VenueType::Exchange)
            .with_mic("XNYS")
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.85,
                avg_latency_ms: 5,
                maker_fee_cents: dec!(-0.0020),  // Maker rebate
                taker_fee_cents: dec!(0.0030),
                min_order_size: 1,
                max_order_size: None,
            })
            .with_priority(10)
    }

    /// NASDAQ Stock Market
    pub fn nasdaq() -> Self {
        Self::new("NASDAQ", "NASDAQ Stock Market", VenueType::Exchange)
            .with_mic("XNAS")
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.88,
                avg_latency_ms: 3,
                maker_fee_cents: dec!(-0.0025),  // Maker rebate
                taker_fee_cents: dec!(0.0030),
                min_order_size: 1,
                max_order_size: None,
            })
            .with_priority(10)
    }

    /// NYSE Arca
    pub fn arca() -> Self {
        Self::new("ARCA", "NYSE Arca", VenueType::Exchange)
            .with_mic("ARCX")
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.82,
                avg_latency_ms: 4,
                maker_fee_cents: dec!(-0.0021),
                taker_fee_cents: dec!(0.0030),
                min_order_size: 1,
                max_order_size: None,
            })
            .with_priority(15)
    }

    /// BATS Global Markets (now Cboe)
    pub fn bats() -> Self {
        Self::new("BATS", "Cboe BZX Exchange", VenueType::Exchange)
            .with_mic("BATS")
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.80,
                avg_latency_ms: 3,
                maker_fee_cents: dec!(-0.0025),
                taker_fee_cents: dec!(0.0030),
                min_order_size: 1,
                max_order_size: None,
            })
            .with_priority(15)
    }

    /// IEX Exchange (Investors Exchange)
    pub fn iex() -> Self {
        Self::new("IEX", "Investors Exchange", VenueType::Exchange)
            .with_mic("IEXG")
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.75,
                avg_latency_ms: 10,  // Speed bump included
                maker_fee_cents: dec!(-0.0009),  // Lower rebate but better protection
                taker_fee_cents: dec!(0.0009),   // Lower take fee
                min_order_size: 1,
                max_order_size: None,
            })
            .with_priority(20)
    }

    // ========== Dark Pool Factory Methods ==========

    /// Create a dark pool venue by name
    pub fn dark_pool(name: &str) -> Self {
        match name.to_uppercase().as_str() {
            "SIGMA_X" | "SIGMAX" => Self::sigma_x(),
            "CROSSFINDER" | "CROSS_FINDER" => Self::crossfinder(),
            "MS_POOL" | "MSPOOL" => Self::ms_pool(),
            "UBS_ATS" | "UBSATS" => Self::ubs_ats(),
            "LIQUIDNET" => Self::liquidnet(),
            "LEVEL_ATS" | "LEVELATS" => Self::level_ats(),
            _ => {
                // Generic dark pool
                Self::new(name, name, VenueType::DarkPool)
                    .with_capabilities(VenueCapabilities {
                        market_orders: false,
                        limit_orders: true,
                        pegged_orders: true,
                        iceberg_orders: true,
                        price_improvement: true,
                        large_orders: true,
                        typical_fill_rate: 0.30,
                        avg_latency_ms: 50,
                        maker_fee_cents: dec!(0.0010),
                        taker_fee_cents: dec!(0.0010),
                        min_order_size: 100,
                        max_order_size: None,
                    })
                    .with_priority(30)
            }
        }
    }

    /// Goldman Sachs Sigma X
    fn sigma_x() -> Self {
        Self::new("SIGMA_X", "Goldman Sachs Sigma X", VenueType::DarkPool)
            .with_mic("SGMX")
            .with_capabilities(VenueCapabilities {
                market_orders: false,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.35,
                avg_latency_ms: 30,
                maker_fee_cents: dec!(0.0010),
                taker_fee_cents: dec!(0.0010),
                min_order_size: 100,
                max_order_size: None,
            })
            .with_priority(25)
    }

    /// Credit Suisse CrossFinder
    fn crossfinder() -> Self {
        Self::new("CROSSFINDER", "Credit Suisse CrossFinder", VenueType::DarkPool)
            .with_mic("CXFN")
            .with_capabilities(VenueCapabilities {
                market_orders: false,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.32,
                avg_latency_ms: 35,
                maker_fee_cents: dec!(0.0008),
                taker_fee_cents: dec!(0.0012),
                min_order_size: 100,
                max_order_size: None,
            })
            .with_priority(25)
    }

    /// Morgan Stanley Pool
    fn ms_pool() -> Self {
        Self::new("MS_POOL", "Morgan Stanley Pool", VenueType::DarkPool)
            .with_mic("MSPL")
            .with_capabilities(VenueCapabilities {
                market_orders: false,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.30,
                avg_latency_ms: 40,
                maker_fee_cents: dec!(0.0010),
                taker_fee_cents: dec!(0.0010),
                min_order_size: 100,
                max_order_size: None,
            })
            .with_priority(25)
    }

    /// UBS ATS
    fn ubs_ats() -> Self {
        Self::new("UBS_ATS", "UBS ATS", VenueType::DarkPool)
            .with_mic("UBSA")
            .with_capabilities(VenueCapabilities {
                market_orders: false,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.28,
                avg_latency_ms: 45,
                maker_fee_cents: dec!(0.0010),
                taker_fee_cents: dec!(0.0010),
                min_order_size: 100,
                max_order_size: None,
            })
            .with_priority(25)
    }

    /// Liquidnet (institutional block trading)
    fn liquidnet() -> Self {
        Self::new("LIQUIDNET", "Liquidnet", VenueType::DarkPool)
            .with_mic("LQDN")
            .with_capabilities(VenueCapabilities {
                market_orders: false,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: false,  // Block trading only
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.20,  // Lower fill rate but larger blocks
                avg_latency_ms: 100,
                maker_fee_cents: dec!(0.0005),
                taker_fee_cents: dec!(0.0005),
                min_order_size: 1000,  // Institutional minimum
                max_order_size: None,
            })
            .with_priority(20)
    }

    /// Level ATS
    fn level_ats() -> Self {
        Self::new("LEVEL_ATS", "Level ATS", VenueType::DarkPool)
            .with_mic("LVLA")
            .with_capabilities(VenueCapabilities {
                market_orders: false,
                limit_orders: true,
                pegged_orders: true,
                iceberg_orders: true,
                price_improvement: true,
                large_orders: true,
                typical_fill_rate: 0.25,
                avg_latency_ms: 50,
                maker_fee_cents: dec!(0.0010),
                taker_fee_cents: dec!(0.0010),
                min_order_size: 100,
                max_order_size: None,
            })
            .with_priority(30)
    }

    // ========== Internalizer Factory Methods ==========

    /// Create an internalizer venue by name
    pub fn internalization(name: &str) -> Self {
        match name.to_uppercase().as_str() {
            "CITADEL" => Self::citadel(),
            "VIRTU" => Self::virtu(),
            _ => {
                // Generic internalizer
                Self::new(name, name, VenueType::Internalizer)
                    .with_capabilities(VenueCapabilities {
                        market_orders: true,
                        limit_orders: true,
                        pegged_orders: false,
                        iceberg_orders: false,
                        price_improvement: true,
                        large_orders: false,
                        typical_fill_rate: 0.95,
                        avg_latency_ms: 5,
                        maker_fee_cents: dec!(0.0),
                        taker_fee_cents: dec!(0.0),
                        min_order_size: 1,
                        max_order_size: Some(10000),
                    })
                    .with_priority(40)
            }
        }
    }

    /// Citadel Securities
    fn citadel() -> Self {
        Self::new("CITADEL", "Citadel Securities", VenueType::Internalizer)
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: false,
                iceberg_orders: false,
                price_improvement: true,
                large_orders: false,
                typical_fill_rate: 0.98,
                avg_latency_ms: 2,
                maker_fee_cents: dec!(0.0),
                taker_fee_cents: dec!(0.0),
                min_order_size: 1,
                max_order_size: Some(10000),
            })
            .with_priority(40)
    }

    /// Virtu Financial
    fn virtu() -> Self {
        Self::new("VIRTU", "Virtu Financial", VenueType::Internalizer)
            .with_capabilities(VenueCapabilities {
                market_orders: true,
                limit_orders: true,
                pegged_orders: false,
                iceberg_orders: false,
                price_improvement: true,
                large_orders: false,
                typical_fill_rate: 0.97,
                avg_latency_ms: 3,
                maker_fee_cents: dec!(0.0),
                taker_fee_cents: dec!(0.0),
                min_order_size: 1,
                max_order_size: Some(10000),
            })
            .with_priority(40)
    }

    // ========== Utility Methods ==========

    /// Check if venue supports a given order type
    pub fn supports_market_orders(&self) -> bool {
        self.capabilities.market_orders
    }

    pub fn supports_limit_orders(&self) -> bool {
        self.capabilities.limit_orders
    }

    pub fn supports_pegged_orders(&self) -> bool {
        self.capabilities.pegged_orders
    }

    pub fn supports_iceberg_orders(&self) -> bool {
        self.capabilities.iceberg_orders
    }

    /// Check if venue is suitable for large orders
    pub fn suitable_for_large_orders(&self) -> bool {
        self.capabilities.large_orders
    }

    /// Check if venue offers price improvement
    pub fn offers_price_improvement(&self) -> bool {
        self.capabilities.price_improvement
    }

    /// Get expected cost for a given quantity (in cents)
    pub fn expected_cost(&self, quantity: Decimal, is_maker: bool) -> Decimal {
        let fee_per_share = if is_maker {
            self.capabilities.maker_fee_cents
        } else {
            self.capabilities.taker_fee_cents
        };
        quantity * fee_per_share
    }

    /// Check if venue can handle given order size
    pub fn can_handle_size(&self, quantity: u32) -> bool {
        if quantity < self.capabilities.min_order_size {
            return false;
        }
        if let Some(max) = self.capabilities.max_order_size {
            if quantity > max {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for Venue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.venue_type)
    }
}

/// Collection of venues for routing
#[derive(Debug, Clone, Default)]
pub struct VenueRegistry {
    venues: Vec<Venue>,
}

impl VenueRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self { venues: Vec::new() }
    }

    /// Create a registry with all standard venues
    pub fn with_all_venues() -> Self {
        let mut registry = Self::new();

        // Retail brokers
        registry.add(Venue::alpaca());
        registry.add(Venue::ibkr_smart());

        // Exchanges
        registry.add(Venue::nyse());
        registry.add(Venue::nasdaq());
        registry.add(Venue::arca());
        registry.add(Venue::bats());
        registry.add(Venue::iex());

        // Dark pools
        registry.add(Venue::dark_pool("SIGMA_X"));
        registry.add(Venue::dark_pool("CROSSFINDER"));
        registry.add(Venue::dark_pool("MS_POOL"));
        registry.add(Venue::dark_pool("UBS_ATS"));
        registry.add(Venue::dark_pool("LIQUIDNET"));
        registry.add(Venue::dark_pool("LEVEL_ATS"));

        // Internalizers
        registry.add(Venue::internalization("CITADEL"));
        registry.add(Venue::internalization("VIRTU"));

        registry
    }

    /// Add a venue to the registry
    pub fn add(&mut self, venue: Venue) {
        self.venues.push(venue);
    }

    /// Get a venue by ID
    pub fn get(&self, id: &str) -> Option<&Venue> {
        self.venues.iter().find(|v| v.id == id)
    }

    /// Get all venues of a given type
    pub fn by_type(&self, venue_type: VenueType) -> Vec<&Venue> {
        self.venues
            .iter()
            .filter(|v| v.venue_type == venue_type && v.available)
            .collect()
    }

    /// Get all available venues sorted by priority
    pub fn available_sorted(&self) -> Vec<&Venue> {
        let mut venues: Vec<&Venue> = self.venues.iter().filter(|v| v.available).collect();
        venues.sort_by_key(|v| v.priority);
        venues
    }

    /// Get venues suitable for a given order size
    pub fn for_size(&self, quantity: u32) -> Vec<&Venue> {
        self.venues
            .iter()
            .filter(|v| v.available && v.can_handle_size(quantity))
            .collect()
    }

    /// Get dark pools only
    pub fn dark_pools(&self) -> Vec<&Venue> {
        self.by_type(VenueType::DarkPool)
    }

    /// Get exchanges only
    pub fn exchanges(&self) -> Vec<&Venue> {
        self.by_type(VenueType::Exchange)
    }

    /// Set venue availability
    pub fn set_available(&mut self, id: &str, available: bool) {
        if let Some(venue) = self.venues.iter_mut().find(|v| v.id == id) {
            venue.available = available;
        }
    }

    /// Get all venue IDs
    pub fn venue_ids(&self) -> Vec<&str> {
        self.venues.iter().map(|v| v.id.as_str()).collect()
    }

    /// Get count of venues
    pub fn len(&self) -> usize {
        self.venues.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.venues.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_creation() {
        let alpaca = Venue::alpaca();
        assert_eq!(alpaca.id, "ALPACA");
        assert_eq!(alpaca.venue_type, VenueType::RetailBroker);
        assert!(alpaca.capabilities.market_orders);
        assert!(alpaca.available);
    }

    #[test]
    fn test_exchange_venues() {
        let nyse = Venue::nyse();
        assert_eq!(nyse.id, "NYSE");
        assert_eq!(nyse.venue_type, VenueType::Exchange);
        assert!(nyse.capabilities.pegged_orders);
        assert!(nyse.capabilities.iceberg_orders);

        let iex = Venue::iex();
        assert_eq!(iex.id, "IEX");
        assert!(iex.capabilities.price_improvement);
    }

    #[test]
    fn test_dark_pool_factory() {
        let sigma = Venue::dark_pool("SIGMA_X");
        assert_eq!(sigma.id, "SIGMA_X");
        assert_eq!(sigma.venue_type, VenueType::DarkPool);
        assert!(!sigma.capabilities.market_orders);
        assert!(sigma.capabilities.pegged_orders);

        let liquidnet = Venue::dark_pool("LIQUIDNET");
        assert_eq!(liquidnet.capabilities.min_order_size, 1000);
    }

    #[test]
    fn test_internalizer_factory() {
        let citadel = Venue::internalization("CITADEL");
        assert_eq!(citadel.id, "CITADEL");
        assert_eq!(citadel.venue_type, VenueType::Internalizer);
        assert!(citadel.capabilities.price_improvement);
        assert!(citadel.capabilities.max_order_size.is_some());
    }

    #[test]
    fn test_venue_registry() {
        let registry = VenueRegistry::with_all_venues();

        assert!(registry.len() > 10);
        assert!(registry.get("NYSE").is_some());
        assert!(registry.get("SIGMA_X").is_some());

        let dark_pools = registry.dark_pools();
        assert!(dark_pools.len() >= 6);

        let exchanges = registry.exchanges();
        assert!(exchanges.len() >= 5);
    }

    #[test]
    fn test_venue_capabilities() {
        let nyse = Venue::nyse();

        // Cost calculation
        let cost = nyse.expected_cost(dec!(1000), true);  // Maker
        assert!(cost < dec!(0));  // Should be negative (rebate)

        let cost = nyse.expected_cost(dec!(1000), false);  // Taker
        assert!(cost > dec!(0));  // Should be positive (fee)

        // Size handling
        assert!(nyse.can_handle_size(100));
        assert!(nyse.can_handle_size(1000000));
    }

    #[test]
    fn test_venue_for_size() {
        let registry = VenueRegistry::with_all_venues();

        // Small order - most venues should work
        let venues = registry.for_size(100);
        assert!(!venues.is_empty());

        // Very small order - dark pools may not work
        let venues = registry.for_size(10);
        let dark_pools_count = venues.iter().filter(|v| v.venue_type == VenueType::DarkPool).count();
        // Most dark pools have 100 share minimum
        assert!(dark_pools_count < 6);
    }

    #[test]
    fn test_registry_availability() {
        let mut registry = VenueRegistry::with_all_venues();

        let nyse = registry.get("NYSE").unwrap();
        assert!(nyse.available);

        registry.set_available("NYSE", false);
        let nyse = registry.get("NYSE").unwrap();
        assert!(!nyse.available);

        // Available sorted should not include NYSE now
        let available = registry.available_sorted();
        assert!(available.iter().all(|v| v.id != "NYSE"));
    }
}
