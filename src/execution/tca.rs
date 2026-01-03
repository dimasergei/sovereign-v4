//! Transaction Cost Analysis (TCA) Module
//!
//! Provides comprehensive analysis of execution quality including:
//! - Arrival slippage
//! - VWAP performance
//! - Implementation shortfall
//! - Market impact estimation
//! - Timing cost
//! - Commission analysis

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::order_manager::{ExecutionReport, ManagedOrder, OrderSide};

/// Benchmark type for TCA comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Benchmark {
    /// Price at order arrival (decision price)
    Arrival,
    /// VWAP over execution interval
    IntervalVwap,
    /// Full day VWAP
    DayVwap,
    /// TWAP over execution interval
    Twap,
    /// Previous day's close
    PreviousClose,
    /// Market close price
    MarketClose,
    /// Opening price
    Open,
}

impl std::fmt::Display for Benchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Benchmark::Arrival => write!(f, "Arrival"),
            Benchmark::IntervalVwap => write!(f, "Interval VWAP"),
            Benchmark::DayVwap => write!(f, "Day VWAP"),
            Benchmark::Twap => write!(f, "TWAP"),
            Benchmark::PreviousClose => write!(f, "Previous Close"),
            Benchmark::MarketClose => write!(f, "Market Close"),
            Benchmark::Open => write!(f, "Open"),
        }
    }
}

/// Execution quality grade
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExecutionQuality {
    /// Outstanding execution (< -5 bps slippage - beat benchmark)
    Excellent,
    /// Good execution (< 2 bps slippage)
    Good,
    /// Acceptable execution (< 10 bps slippage)
    Acceptable,
    /// Poor execution (< 25 bps slippage)
    Poor,
    /// Very poor execution (>= 25 bps slippage)
    VeryPoor,
}

impl ExecutionQuality {
    /// Grade based on slippage in basis points
    pub fn from_slippage_bps(slippage_bps: f64) -> Self {
        if slippage_bps < -5.0 {
            ExecutionQuality::Excellent
        } else if slippage_bps < 2.0 {
            ExecutionQuality::Good
        } else if slippage_bps < 10.0 {
            ExecutionQuality::Acceptable
        } else if slippage_bps < 25.0 {
            ExecutionQuality::Poor
        } else {
            ExecutionQuality::VeryPoor
        }
    }
}

impl std::fmt::Display for ExecutionQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionQuality::Excellent => write!(f, "EXCELLENT"),
            ExecutionQuality::Good => write!(f, "GOOD"),
            ExecutionQuality::Acceptable => write!(f, "ACCEPTABLE"),
            ExecutionQuality::Poor => write!(f, "POOR"),
            ExecutionQuality::VeryPoor => write!(f, "VERY POOR"),
        }
    }
}

/// Benchmark prices for TCA calculation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkPrices {
    /// Price at order arrival
    pub arrival_price: Option<Decimal>,
    /// VWAP over execution interval
    pub interval_vwap: Option<Decimal>,
    /// Full day VWAP
    pub day_vwap: Option<Decimal>,
    /// TWAP over execution interval
    pub twap: Option<Decimal>,
    /// Previous close
    pub previous_close: Option<Decimal>,
    /// Market close
    pub market_close: Option<Decimal>,
    /// Opening price
    pub open: Option<Decimal>,
    /// Mid-quote at arrival
    pub arrival_mid: Option<Decimal>,
    /// Bid at arrival
    pub arrival_bid: Option<Decimal>,
    /// Ask at arrival
    pub arrival_ask: Option<Decimal>,
}

impl BenchmarkPrices {
    /// Create with arrival price only
    pub fn with_arrival(price: Decimal) -> Self {
        Self {
            arrival_price: Some(price),
            ..Default::default()
        }
    }

    /// Set interval VWAP
    pub fn with_interval_vwap(mut self, vwap: Decimal) -> Self {
        self.interval_vwap = Some(vwap);
        self
    }

    /// Set day VWAP
    pub fn with_day_vwap(mut self, vwap: Decimal) -> Self {
        self.day_vwap = Some(vwap);
        self
    }

    /// Set previous close
    pub fn with_previous_close(mut self, price: Decimal) -> Self {
        self.previous_close = Some(price);
        self
    }

    /// Get benchmark price
    pub fn get(&self, benchmark: Benchmark) -> Option<Decimal> {
        match benchmark {
            Benchmark::Arrival => self.arrival_price,
            Benchmark::IntervalVwap => self.interval_vwap,
            Benchmark::DayVwap => self.day_vwap,
            Benchmark::Twap => self.twap,
            Benchmark::PreviousClose => self.previous_close,
            Benchmark::MarketClose => self.market_close,
            Benchmark::Open => self.open,
        }
    }
}

/// Transaction Cost Analysis report for a single execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcaReport {
    /// Order ID
    pub order_id: String,
    /// Symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Total quantity
    pub quantity: Decimal,
    /// Average execution price
    pub avg_price: Decimal,
    /// Benchmark prices used
    pub benchmarks: BenchmarkPrices,

    // ===== Slippage Metrics (in basis points) =====

    /// Slippage vs arrival price
    pub arrival_slippage_bps: f64,
    /// Slippage vs interval VWAP
    pub vwap_slippage_bps: Option<f64>,
    /// Implementation shortfall
    pub implementation_shortfall_bps: f64,
    /// Estimated market impact
    pub market_impact_bps: f64,
    /// Timing cost (delay cost)
    pub timing_cost_bps: f64,
    /// Commission cost in bps
    pub commission_cost_bps: f64,

    // ===== Cost Components =====

    /// Total cost in dollars
    pub total_cost_usd: Decimal,
    /// Slippage cost in dollars
    pub slippage_cost_usd: Decimal,
    /// Commission in dollars
    pub commission_usd: Decimal,

    // ===== Quality Assessment =====

    /// Overall execution quality grade
    pub grade: ExecutionQuality,
    /// Primary benchmark used
    pub primary_benchmark: Benchmark,
    /// Execution timestamp
    pub timestamp: DateTime<Utc>,
    /// Execution duration in milliseconds
    pub duration_ms: i64,
    /// Number of fills
    pub fill_count: usize,
    /// Venues used
    pub venues: Vec<String>,
}

impl TcaReport {
    /// Calculate TCA from order and benchmark prices
    pub fn calculate(
        order: &ManagedOrder,
        benchmarks: BenchmarkPrices,
        commission: Decimal,
    ) -> Self {
        let avg_price = order.avg_fill_price;
        let quantity = order.filled_qty;
        let notional = avg_price * quantity;

        // Calculate arrival slippage
        let arrival_slippage_bps = if let Some(arrival) = benchmarks.arrival_price {
            Self::calculate_slippage_bps(avg_price, arrival, order.side)
        } else {
            0.0
        };

        // Calculate VWAP slippage
        let vwap_slippage_bps = benchmarks.interval_vwap.map(|vwap| {
            Self::calculate_slippage_bps(avg_price, vwap, order.side)
        });

        // Calculate timing cost (difference between arrival and first trade)
        let timing_cost_bps = if let Some(arrival) = benchmarks.arrival_price {
            if let Some(first_fill) = order.fills.first() {
                Self::calculate_slippage_bps(first_fill.price, arrival, order.side)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Estimate market impact (total slippage - timing cost)
        let market_impact_bps = (arrival_slippage_bps - timing_cost_bps).max(0.0);

        // Commission cost in bps
        let commission_cost_bps = if !notional.is_zero() {
            (commission / notional).to_f64().unwrap_or(0.0) * 10000.0
        } else {
            0.0
        };

        // Implementation shortfall = arrival slippage + commission
        let implementation_shortfall_bps = arrival_slippage_bps + commission_cost_bps;

        // Calculate dollar costs
        let slippage_cost_usd = if let Some(arrival) = benchmarks.arrival_price {
            let ideal_notional = arrival * quantity;
            match order.side {
                OrderSide::Buy => notional - ideal_notional,
                OrderSide::Sell => ideal_notional - notional,
            }
        } else {
            dec!(0)
        };

        let total_cost_usd = slippage_cost_usd + commission;

        // Determine grade based on implementation shortfall
        let grade = ExecutionQuality::from_slippage_bps(implementation_shortfall_bps);

        // Collect venues
        let venues: Vec<String> = order
            .fills
            .iter()
            .map(|f| f.venue.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let duration_ms = order
            .submitted_at
            .map(|s| (Utc::now() - s).num_milliseconds())
            .unwrap_or(0);

        Self {
            order_id: order.order_id.clone(),
            symbol: order.symbol.clone(),
            side: order.side,
            quantity,
            avg_price,
            benchmarks,
            arrival_slippage_bps,
            vwap_slippage_bps,
            implementation_shortfall_bps,
            market_impact_bps,
            timing_cost_bps,
            commission_cost_bps,
            total_cost_usd,
            slippage_cost_usd,
            commission_usd: commission,
            grade,
            primary_benchmark: Benchmark::Arrival,
            timestamp: Utc::now(),
            duration_ms,
            fill_count: order.fills.len(),
            venues,
        }
    }

    /// Calculate slippage in basis points
    fn calculate_slippage_bps(exec_price: Decimal, benchmark: Decimal, side: OrderSide) -> f64 {
        if benchmark.is_zero() {
            return 0.0;
        }

        let diff = exec_price - benchmark;
        let slippage = match side {
            OrderSide::Buy => diff / benchmark,  // Higher is worse for buys
            OrderSide::Sell => -diff / benchmark, // Lower is worse for sells
        };

        slippage.to_f64().unwrap_or(0.0) * 10000.0  // Convert to bps
    }

    /// Check if execution beat the benchmark
    pub fn beat_benchmark(&self, benchmark: Benchmark) -> Option<bool> {
        match benchmark {
            Benchmark::Arrival => Some(self.arrival_slippage_bps < 0.0),
            Benchmark::IntervalVwap => self.vwap_slippage_bps.map(|s| s < 0.0),
            _ => None,
        }
    }

    /// Get total cost as percentage of notional
    pub fn total_cost_pct(&self) -> f64 {
        let notional = self.avg_price * self.quantity;
        if notional.is_zero() {
            return 0.0;
        }
        (self.total_cost_usd / notional).to_f64().unwrap_or(0.0) * 100.0
    }
}

/// Aggregate TCA statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TcaStatistics {
    /// Total number of executions
    pub execution_count: usize,
    /// Total quantity traded
    pub total_quantity: Decimal,
    /// Total notional traded
    pub total_notional: Decimal,
    /// Average arrival slippage (bps)
    pub avg_arrival_slippage_bps: f64,
    /// Average VWAP slippage (bps)
    pub avg_vwap_slippage_bps: f64,
    /// Average implementation shortfall (bps)
    pub avg_is_bps: f64,
    /// Total commission paid
    pub total_commission: Decimal,
    /// Total slippage cost
    pub total_slippage_cost: Decimal,
    /// Quality distribution
    pub quality_distribution: HashMap<String, usize>,
    /// Best execution (lowest slippage)
    pub best_slippage_bps: f64,
    /// Worst execution (highest slippage)
    pub worst_slippage_bps: f64,
    /// Standard deviation of slippage
    pub slippage_std_dev: f64,
}

impl TcaStatistics {
    /// Calculate statistics from a collection of TCA reports
    pub fn from_reports(reports: &[TcaReport]) -> Self {
        if reports.is_empty() {
            return Self::default();
        }

        let mut stats = Self::default();
        stats.execution_count = reports.len();

        let mut slippages = Vec::new();
        let mut vwap_slippages = Vec::new();
        let mut quality_counts: HashMap<String, usize> = HashMap::new();

        for report in reports {
            let notional = report.avg_price * report.quantity;

            stats.total_quantity += report.quantity;
            stats.total_notional += notional;
            stats.total_commission += report.commission_usd;
            stats.total_slippage_cost += report.slippage_cost_usd;

            slippages.push(report.arrival_slippage_bps);

            if let Some(vwap_slip) = report.vwap_slippage_bps {
                vwap_slippages.push(vwap_slip);
            }

            *quality_counts.entry(report.grade.to_string()).or_insert(0) += 1;
        }

        // Calculate averages
        stats.avg_arrival_slippage_bps = slippages.iter().sum::<f64>() / slippages.len() as f64;

        if !vwap_slippages.is_empty() {
            stats.avg_vwap_slippage_bps =
                vwap_slippages.iter().sum::<f64>() / vwap_slippages.len() as f64;
        }

        // Implementation shortfall (arrival + commission)
        let total_is: f64 = reports.iter().map(|r| r.implementation_shortfall_bps).sum();
        stats.avg_is_bps = total_is / reports.len() as f64;

        // Best/worst
        stats.best_slippage_bps = slippages
            .iter()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        stats.worst_slippage_bps = slippages
            .iter()
            .cloned()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Standard deviation
        let mean = stats.avg_arrival_slippage_bps;
        let variance: f64 = slippages.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / slippages.len() as f64;
        stats.slippage_std_dev = variance.sqrt();

        stats.quality_distribution = quality_counts;

        stats
    }

    /// Get grade distribution as percentages
    pub fn grade_percentages(&self) -> HashMap<String, f64> {
        self.quality_distribution
            .iter()
            .map(|(grade, count)| {
                let pct = (*count as f64 / self.execution_count as f64) * 100.0;
                (grade.clone(), pct)
            })
            .collect()
    }
}

/// TCA analyzer for tracking and analyzing executions
#[derive(Debug, Default)]
pub struct TcaAnalyzer {
    /// All TCA reports
    reports: Vec<TcaReport>,
    /// Reports by symbol
    by_symbol: HashMap<String, Vec<usize>>,
    /// Reports by venue
    by_venue: HashMap<String, Vec<usize>>,
}

impl TcaAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a TCA report
    pub fn add_report(&mut self, report: TcaReport) {
        let idx = self.reports.len();

        // Index by symbol
        self.by_symbol
            .entry(report.symbol.clone())
            .or_default()
            .push(idx);

        // Index by venues
        for venue in &report.venues {
            self.by_venue
                .entry(venue.clone())
                .or_default()
                .push(idx);
        }

        self.reports.push(report);
    }

    /// Analyze an order and add report
    pub fn analyze_order(
        &mut self,
        order: &ManagedOrder,
        benchmarks: BenchmarkPrices,
        commission: Decimal,
    ) -> TcaReport {
        let report = TcaReport::calculate(order, benchmarks, commission);
        self.add_report(report.clone());
        report
    }

    /// Get all reports
    pub fn get_reports(&self) -> &[TcaReport] {
        &self.reports
    }

    /// Get reports for a symbol
    pub fn get_reports_for_symbol(&self, symbol: &str) -> Vec<&TcaReport> {
        self.by_symbol
            .get(symbol)
            .map(|indices| indices.iter().map(|&i| &self.reports[i]).collect())
            .unwrap_or_default()
    }

    /// Get reports for a venue
    pub fn get_reports_for_venue(&self, venue: &str) -> Vec<&TcaReport> {
        self.by_venue
            .get(venue)
            .map(|indices| indices.iter().map(|&i| &self.reports[i]).collect())
            .unwrap_or_default()
    }

    /// Get aggregate statistics
    pub fn get_statistics(&self) -> TcaStatistics {
        TcaStatistics::from_reports(&self.reports)
    }

    /// Get statistics for a symbol
    pub fn get_symbol_statistics(&self, symbol: &str) -> TcaStatistics {
        let reports: Vec<TcaReport> = self
            .get_reports_for_symbol(symbol)
            .into_iter()
            .cloned()
            .collect();
        TcaStatistics::from_reports(&reports)
    }

    /// Get statistics for a venue
    pub fn get_venue_statistics(&self, venue: &str) -> TcaStatistics {
        let reports: Vec<TcaReport> = self
            .get_reports_for_venue(venue)
            .into_iter()
            .cloned()
            .collect();
        TcaStatistics::from_reports(&reports)
    }

    /// Get statistics for a date range
    pub fn get_period_statistics(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> TcaStatistics {
        let reports: Vec<TcaReport> = self
            .reports
            .iter()
            .filter(|r| r.timestamp >= start && r.timestamp <= end)
            .cloned()
            .collect();
        TcaStatistics::from_reports(&reports)
    }

    /// Get worst executions
    pub fn worst_executions(&self, limit: usize) -> Vec<&TcaReport> {
        let mut reports: Vec<&TcaReport> = self.reports.iter().collect();
        reports.sort_by(|a, b| {
            b.implementation_shortfall_bps
                .partial_cmp(&a.implementation_shortfall_bps)
                .unwrap()
        });
        reports.into_iter().take(limit).collect()
    }

    /// Get best executions
    pub fn best_executions(&self, limit: usize) -> Vec<&TcaReport> {
        let mut reports: Vec<&TcaReport> = self.reports.iter().collect();
        reports.sort_by(|a, b| {
            a.implementation_shortfall_bps
                .partial_cmp(&b.implementation_shortfall_bps)
                .unwrap()
        });
        reports.into_iter().take(limit).collect()
    }

    /// Compare venues
    pub fn compare_venues(&self) -> HashMap<String, TcaStatistics> {
        let mut result = HashMap::new();
        for venue in self.by_venue.keys() {
            result.insert(venue.clone(), self.get_venue_statistics(venue));
        }
        result
    }

    /// Get report count
    pub fn report_count(&self) -> usize {
        self.reports.len()
    }

    /// Clear old reports
    pub fn clear_before(&mut self, cutoff: DateTime<Utc>) {
        let to_remove: Vec<usize> = self
            .reports
            .iter()
            .enumerate()
            .filter(|(_, r)| r.timestamp < cutoff)
            .map(|(i, _)| i)
            .collect();

        // Remove in reverse order to maintain indices
        for idx in to_remove.into_iter().rev() {
            self.reports.remove(idx);
        }

        // Rebuild indices
        self.by_symbol.clear();
        self.by_venue.clear();

        for (idx, report) in self.reports.iter().enumerate() {
            self.by_symbol
                .entry(report.symbol.clone())
                .or_default()
                .push(idx);
            for venue in &report.venues {
                self.by_venue
                    .entry(venue.clone())
                    .or_default()
                    .push(idx);
            }
        }
    }
}

/// Generate a summary string for a TCA report
pub fn format_tca_summary(report: &TcaReport) -> String {
    format!(
        "{} {} {}: {} @ {:.2} | Slippage: {:.1} bps | IS: {:.1} bps | Grade: {}",
        report.side,
        report.quantity,
        report.symbol,
        report.fill_count,
        report.avg_price,
        report.arrival_slippage_bps,
        report.implementation_shortfall_bps,
        report.grade
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::order_manager::Fill;

    fn create_test_order() -> ManagedOrder {
        let mut order = ManagedOrder::market("AAPL", OrderSide::Buy, dec!(1000));
        order.mark_submitted(Some("B123"));

        // Add fills at progressively worse prices (simulating market impact)
        order.record_fill(Fill::new(dec!(300), dec!(150.00), "NYSE"));
        order.record_fill(Fill::new(dec!(300), dec!(150.10), "NASDAQ"));
        order.record_fill(Fill::new(dec!(400), dec!(150.20), "ARCA"));

        order
    }

    #[test]
    fn test_tca_report_calculation() {
        let order = create_test_order();
        let benchmarks = BenchmarkPrices::with_arrival(dec!(149.90))
            .with_interval_vwap(dec!(150.05));

        let report = TcaReport::calculate(&order, benchmarks, dec!(5.00));

        assert_eq!(report.symbol, "AAPL");
        assert_eq!(report.quantity, dec!(1000));
        assert_eq!(report.fill_count, 3);

        // Average price should be weighted: (300*150 + 300*150.10 + 400*150.20) / 1000
        // = (45000 + 45030 + 60080) / 1000 = 150.11
        assert!((report.avg_price - dec!(150.11)).abs() < dec!(0.01));

        // Arrival slippage: (150.11 - 149.90) / 149.90 * 10000 ≈ 14 bps (for buy)
        assert!(report.arrival_slippage_bps > 10.0);
        assert!(report.arrival_slippage_bps < 20.0);

        // Should include commission
        assert!(report.commission_cost_bps > 0.0);
        assert_eq!(report.commission_usd, dec!(5.00));
    }

    #[test]
    fn test_execution_quality_grading() {
        // Excellent (beat benchmark significantly)
        assert_eq!(
            ExecutionQuality::from_slippage_bps(-10.0),
            ExecutionQuality::Excellent
        );

        // Good
        assert_eq!(
            ExecutionQuality::from_slippage_bps(1.0),
            ExecutionQuality::Good
        );

        // Acceptable
        assert_eq!(
            ExecutionQuality::from_slippage_bps(5.0),
            ExecutionQuality::Acceptable
        );

        // Poor
        assert_eq!(
            ExecutionQuality::from_slippage_bps(15.0),
            ExecutionQuality::Poor
        );

        // Very Poor
        assert_eq!(
            ExecutionQuality::from_slippage_bps(30.0),
            ExecutionQuality::VeryPoor
        );
    }

    #[test]
    fn test_tca_statistics() {
        let order1 = create_test_order();
        let order2 = create_test_order();

        let benchmarks = BenchmarkPrices::with_arrival(dec!(149.90));

        let report1 = TcaReport::calculate(&order1, benchmarks.clone(), dec!(5.00));
        let report2 = TcaReport::calculate(&order2, benchmarks, dec!(5.00));

        let stats = TcaStatistics::from_reports(&[report1, report2]);

        assert_eq!(stats.execution_count, 2);
        assert_eq!(stats.total_quantity, dec!(2000));
        assert_eq!(stats.total_commission, dec!(10.00));
        assert!(stats.avg_arrival_slippage_bps > 0.0);
    }

    #[test]
    fn test_tca_analyzer() {
        let mut analyzer = TcaAnalyzer::new();

        // Add some reports
        let order1 = create_test_order();
        let benchmarks = BenchmarkPrices::with_arrival(dec!(149.90));
        analyzer.analyze_order(&order1, benchmarks.clone(), dec!(5.00));

        let order2 = create_test_order();
        analyzer.analyze_order(&order2, benchmarks, dec!(5.00));

        assert_eq!(analyzer.report_count(), 2);

        // Check symbol stats
        let symbol_stats = analyzer.get_symbol_statistics("AAPL");
        assert_eq!(symbol_stats.execution_count, 2);

        // Check venue stats (orders used NYSE, NASDAQ, ARCA)
        let nyse_stats = analyzer.get_venue_statistics("NYSE");
        assert!(nyse_stats.execution_count > 0);
    }

    #[test]
    fn test_slippage_calculation_sides() {
        // For buys: higher execution price = worse (positive slippage)
        let buy_slippage = TcaReport::calculate_slippage_bps(dec!(101), dec!(100), OrderSide::Buy);
        assert!(buy_slippage > 0.0);
        assert!((buy_slippage - 100.0).abs() < 1.0); // ~100 bps

        // For sells: lower execution price = worse (positive slippage)
        let sell_slippage = TcaReport::calculate_slippage_bps(dec!(99), dec!(100), OrderSide::Sell);
        assert!(sell_slippage > 0.0);
        assert!((sell_slippage - 100.0).abs() < 1.0); // ~100 bps

        // Beat benchmark for buy (lower price)
        let beat_buy = TcaReport::calculate_slippage_bps(dec!(99), dec!(100), OrderSide::Buy);
        assert!(beat_buy < 0.0); // Negative = beat benchmark

        // Beat benchmark for sell (higher price)
        let beat_sell = TcaReport::calculate_slippage_bps(dec!(101), dec!(100), OrderSide::Sell);
        assert!(beat_sell < 0.0); // Negative = beat benchmark
    }

    #[test]
    fn test_benchmark_prices() {
        let benchmarks = BenchmarkPrices::with_arrival(dec!(100))
            .with_interval_vwap(dec!(100.50))
            .with_day_vwap(dec!(101.00))
            .with_previous_close(dec!(99.00));

        assert_eq!(benchmarks.get(Benchmark::Arrival), Some(dec!(100)));
        assert_eq!(benchmarks.get(Benchmark::IntervalVwap), Some(dec!(100.50)));
        assert_eq!(benchmarks.get(Benchmark::DayVwap), Some(dec!(101.00)));
        assert_eq!(benchmarks.get(Benchmark::PreviousClose), Some(dec!(99.00)));
        assert_eq!(benchmarks.get(Benchmark::MarketClose), None);
    }

    #[test]
    fn test_worst_best_executions() {
        let mut analyzer = TcaAnalyzer::new();

        // Create orders with different slippage
        for arrival in [dec!(149.0), dec!(149.5), dec!(150.0)] {
            let order = create_test_order();
            let benchmarks = BenchmarkPrices::with_arrival(arrival);
            analyzer.analyze_order(&order, benchmarks, dec!(5.00));
        }

        let worst = analyzer.worst_executions(1);
        let best = analyzer.best_executions(1);

        assert_eq!(worst.len(), 1);
        assert_eq!(best.len(), 1);

        // Worst should have highest slippage (arrival price 149.0 with avg exec ~150.11)
        // Best should have lowest slippage (arrival price 150.0 with avg exec ~150.11)
        assert!(worst[0].arrival_slippage_bps > best[0].arrival_slippage_bps);
    }
}
