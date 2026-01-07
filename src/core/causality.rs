//! Causal Reasoning Foundation with Granger Causality
//!
//! Implements causal discovery and reasoning for trading:
//! - Granger causality tests to identify leading indicators
//! - Causal graph of symbol/factor relationships
//! - Regime prediction from causal factors
//! - Trade confidence adjustment based on causal alignment
//!
//! Key insight: Understanding WHY markets move (causality) is more
//! valuable than just THAT they move (correlation).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::info;

use super::regime::Regime;

/// Default window size for Granger tests (bars)
const DEFAULT_WINDOW_SIZE: usize = 252;

/// Default significance threshold (p-value)
const DEFAULT_SIGNIFICANCE: f64 = 0.05;

/// Minimum data points for Granger test
const MIN_DATA_POINTS: usize = 30;

/// Maximum lag to test
const MAX_LAG: u32 = 10;

/// Direction of causal relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalDirection {
    /// Source up -> Target up
    Positive,
    /// Source up -> Target down
    Negative,
}

impl std::fmt::Display for CausalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CausalDirection::Positive => write!(f, "+"),
            CausalDirection::Negative => write!(f, "-"),
        }
    }
}

/// Types of causal factors
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalFactor {
    /// A trading symbol
    Symbol(String),
    /// VIX volatility index
    VIX,
    /// US Dollar Index
    DXY,
    /// 10-Year Treasury Yields
    Yields,
    /// Crude Oil
    Oil,
    /// Gold
    Gold,
}

impl CausalFactor {
    /// Get string representation
    pub fn as_str(&self) -> String {
        match self {
            CausalFactor::Symbol(s) => s.clone(),
            CausalFactor::VIX => "VIX".to_string(),
            CausalFactor::DXY => "DXY".to_string(),
            CausalFactor::Yields => "10Y".to_string(),
            CausalFactor::Oil => "OIL".to_string(),
            CausalFactor::Gold => "GOLD".to_string(),
        }
    }

    /// Create from string (symbol or known factor)
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "VIX" | "^VIX" => CausalFactor::VIX,
            "DXY" | "DX-Y.NYB" => CausalFactor::DXY,
            "10Y" | "^TNX" | "YIELDS" => CausalFactor::Yields,
            "OIL" | "CL=F" | "USO" => CausalFactor::Oil,
            "GOLD" | "GC=F" | "GLD" | "XAUUSD" => CausalFactor::Gold,
            _ => CausalFactor::Symbol(s.to_string()),
        }
    }
}

impl std::fmt::Display for CausalFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A causal relationship between two factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRelationship {
    /// Source factor (the cause)
    pub source: String,
    /// Target factor (the effect)
    pub target: String,
    /// How many bars source leads target
    pub lag_bars: u32,
    /// Correlation strength (0.0-1.0)
    pub strength: f64,
    /// Direction of effect
    pub direction: CausalDirection,
    /// Statistical confidence (1 - p_value)
    pub confidence: f64,
    /// When this relationship was last validated
    pub last_validated: DateTime<Utc>,
}

impl CausalRelationship {
    /// Create a new causal relationship
    pub fn new(
        source: String,
        target: String,
        lag_bars: u32,
        strength: f64,
        direction: CausalDirection,
        p_value: f64,
    ) -> Self {
        Self {
            source,
            target,
            lag_bars,
            strength,
            direction,
            confidence: 1.0 - p_value,
            last_validated: Utc::now(),
        }
    }

    /// Check if relationship is still valid (not too old)
    pub fn is_valid(&self, max_age_days: i64) -> bool {
        let age = Utc::now() - self.last_validated;
        age.num_days() < max_age_days
    }

    /// Get description of this relationship
    pub fn description(&self) -> String {
        format!(
            "{} {} {} (lag {}, strength {:.2}, conf {:.1}%)",
            self.source,
            match self.direction {
                CausalDirection::Positive => "->",
                CausalDirection::Negative => "-|",
            },
            self.target,
            self.lag_bars,
            self.strength,
            self.confidence * 100.0
        )
    }
}

/// Result of a Granger causality test
#[derive(Debug, Clone)]
pub struct GrangerResult {
    /// Optimal lag found
    pub lag: u32,
    /// F-statistic
    pub f_statistic: f64,
    /// P-value (lower = more significant)
    pub p_value: f64,
    /// Is result statistically significant?
    pub significant: bool,
    /// Correlation at optimal lag
    pub correlation: f64,
}

/// Granger causality test implementation
pub struct GrangerCausalityTest {
    /// Significance threshold
    significance: f64,
}

impl Default for GrangerCausalityTest {
    fn default() -> Self {
        Self::new()
    }
}

impl GrangerCausalityTest {
    /// Create new test with default significance
    pub fn new() -> Self {
        Self {
            significance: DEFAULT_SIGNIFICANCE,
        }
    }

    /// Create with custom significance threshold
    pub fn with_significance(significance: f64) -> Self {
        Self { significance }
    }

    /// Run Granger causality test
    ///
    /// Tests if `source` Granger-causes `target`.
    /// Returns None if insufficient data.
    pub fn test(&self, source: &[f64], target: &[f64], max_lag: u32) -> Option<GrangerResult> {
        let n = source.len().min(target.len());

        if n < MIN_DATA_POINTS {
            return None;
        }

        let max_lag = max_lag.min((n / 3) as u32).max(1);
        let mut best_result: Option<GrangerResult> = None;

        for lag in 1..=max_lag {
            if let Some(result) = self.test_single_lag(source, target, lag) {
                if result.significant {
                    match &best_result {
                        None => best_result = Some(result),
                        Some(best) if result.p_value < best.p_value => {
                            best_result = Some(result);
                        }
                        _ => {}
                    }
                }
            }
        }

        // If no significant result, return best non-significant
        if best_result.is_none() {
            for lag in 1..=max_lag {
                if let Some(result) = self.test_single_lag(source, target, lag) {
                    match &best_result {
                        None => best_result = Some(result),
                        Some(best) if result.p_value < best.p_value => {
                            best_result = Some(result);
                        }
                        _ => {}
                    }
                }
            }
        }

        best_result
    }

    /// Test a single lag value
    fn test_single_lag(&self, source: &[f64], target: &[f64], lag: u32) -> Option<GrangerResult> {
        let n = source.len().min(target.len());
        let lag = lag as usize;

        if n <= lag + 1 {
            return None;
        }

        // Align series: we want to predict target[t] using target[t-1..t-lag] and source[t-lag]
        let effective_n = n - lag;

        // Build target vector (what we're predicting)
        let y: Vec<f64> = target[lag..].iter().copied().collect();

        // Build restricted model: AR(1) on target only
        // y_t = a0 + a1 * y_{t-1}
        let y_lagged: Vec<f64> = target[lag - 1..n - 1].iter().copied().collect();
        let ssr_restricted = self.fit_ar1_ssr(&y, &y_lagged);

        // Build unrestricted model: AR(1) on target + lagged source
        // y_t = a0 + a1 * y_{t-1} + b1 * x_{t-lag}
        let x_lagged: Vec<f64> = source[0..effective_n].iter().copied().collect();
        let ssr_unrestricted = self.fit_arx_ssr(&y, &y_lagged, &x_lagged);

        // F-test: F = ((SSR_r - SSR_u) / q) / (SSR_u / (n - k))
        // q = number of restrictions (1 in this case)
        // k = number of parameters in unrestricted model (3: intercept, y_lag, x_lag)
        let q = 1.0;
        let k = 3.0;
        let n_f = effective_n as f64;

        if ssr_unrestricted <= 0.0 || ssr_restricted < ssr_unrestricted {
            return None;
        }

        let f_stat = ((ssr_restricted - ssr_unrestricted) / q) / (ssr_unrestricted / (n_f - k));

        // Compute approximate p-value using F-distribution approximation
        // For F(1, n-k), use simplified approximation
        let df1 = q;
        let df2 = n_f - k;
        let p_value = self.f_distribution_pvalue(f_stat, df1, df2);

        // Compute correlation at this lag
        let correlation = self.pearson_correlation(&x_lagged, &y);

        Some(GrangerResult {
            lag: lag as u32,
            f_statistic: f_stat,
            p_value,
            significant: p_value < self.significance,
            correlation,
        })
    }

    /// Fit AR(1) model and return sum of squared residuals
    fn fit_ar1_ssr(&self, y: &[f64], y_lag: &[f64]) -> f64 {
        let n = y.len().min(y_lag.len());
        if n < 2 {
            return f64::MAX;
        }

        // Simple OLS for y = a + b * y_lag
        let (a, b) = self.simple_ols(y_lag, y);

        // Compute SSR
        let mut ssr = 0.0;
        for i in 0..n {
            let predicted = a + b * y_lag[i];
            let residual = y[i] - predicted;
            ssr += residual * residual;
        }

        ssr
    }

    /// Fit ARX model (AR + exogenous) and return sum of squared residuals
    fn fit_arx_ssr(&self, y: &[f64], y_lag: &[f64], x_lag: &[f64]) -> f64 {
        let n = y.len().min(y_lag.len()).min(x_lag.len());
        if n < 3 {
            return f64::MAX;
        }

        // Multiple regression: y = a + b1 * y_lag + b2 * x_lag
        // Use normal equations with 2x2 system (after centering)
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_lag_mean: f64 = y_lag.iter().sum::<f64>() / n as f64;
        let x_lag_mean: f64 = x_lag.iter().sum::<f64>() / n as f64;

        // Build covariance matrix elements
        let mut s_yy_lag = 0.0;
        let mut s_yx_lag = 0.0;
        let mut s_ylag_ylag = 0.0;
        let mut s_xlag_xlag = 0.0;
        let mut s_ylag_xlag = 0.0;

        for i in 0..n {
            let y_c = y[i] - y_mean;
            let yl_c = y_lag[i] - y_lag_mean;
            let xl_c = x_lag[i] - x_lag_mean;

            s_yy_lag += y_c * yl_c;
            s_yx_lag += y_c * xl_c;
            s_ylag_ylag += yl_c * yl_c;
            s_xlag_xlag += xl_c * xl_c;
            s_ylag_xlag += yl_c * xl_c;
        }

        // Solve 2x2 system: [s_ylag_ylag, s_ylag_xlag; s_ylag_xlag, s_xlag_xlag] * [b1; b2] = [s_yy_lag; s_yx_lag]
        let det = s_ylag_ylag * s_xlag_xlag - s_ylag_xlag * s_ylag_xlag;
        if det.abs() < 1e-10 {
            return self.fit_ar1_ssr(y, y_lag); // Degenerate, fall back to AR(1)
        }

        let b1 = (s_xlag_xlag * s_yy_lag - s_ylag_xlag * s_yx_lag) / det;
        let b2 = (s_ylag_ylag * s_yx_lag - s_ylag_xlag * s_yy_lag) / det;
        let a = y_mean - b1 * y_lag_mean - b2 * x_lag_mean;

        // Compute SSR
        let mut ssr = 0.0;
        for i in 0..n {
            let predicted = a + b1 * y_lag[i] + b2 * x_lag[i];
            let residual = y[i] - predicted;
            ssr += residual * residual;
        }

        ssr
    }

    /// Simple OLS regression: y = a + b * x
    fn simple_ols(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len().min(y.len()) as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        let x_mean: f64 = x.iter().sum::<f64>() / n;
        let y_mean: f64 = y.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den = 0.0;

        for i in 0..n as usize {
            let x_diff = x[i] - x_mean;
            num += x_diff * (y[i] - y_mean);
            den += x_diff * x_diff;
        }

        if den.abs() < 1e-10 {
            return (y_mean, 0.0);
        }

        let b = num / den;
        let a = y_mean - b * x_mean;

        (a, b)
    }

    /// Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return 0.0;
        }

        let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            cov += x_diff * y_diff;
            var_x += x_diff * x_diff;
            var_y += y_diff * y_diff;
        }

        if var_x < 1e-10 || var_y < 1e-10 {
            return 0.0;
        }

        cov / (var_x.sqrt() * var_y.sqrt())
    }

    /// Approximate p-value for F-distribution
    /// Uses a simplified approximation for F(df1, df2)
    fn f_distribution_pvalue(&self, f: f64, df1: f64, df2: f64) -> f64 {
        if f <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
            return 1.0;
        }

        // Use beta function relationship: F(df1, df2) relates to Beta distribution
        // P(F > f) = I_x(df2/2, df1/2) where x = df2 / (df2 + df1 * f)
        let x = df2 / (df2 + df1 * f);

        // Approximate incomplete beta using continued fraction or series
        // For simplicity, use a rough approximation based on normal approximation
        // This is less accurate but sufficient for our purposes
        let z = ((f.powf(1.0 / 3.0) * (1.0 - 2.0 / (9.0 * df2)))
            - (1.0 - 2.0 / (9.0 * df1)))
            / ((2.0 / (9.0 * df1) + f.powf(2.0 / 3.0) * 2.0 / (9.0 * df2)).sqrt());

        // Standard normal CDF approximation
        self.normal_cdf(-z)
    }

    /// Standard normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        // Approximation using error function
        0.5 * (1.0 + self.erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation (Abramowitz and Stegun)
    fn erf(&self, x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

/// Causal graph of relationships between factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// All discovered relationships
    relationships: Vec<CausalRelationship>,
    /// All known factors
    factors: HashSet<String>,
    /// Last time graph was updated
    last_updated: DateTime<Utc>,
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalGraph {
    /// Create empty causal graph
    pub fn new() -> Self {
        Self {
            relationships: Vec::new(),
            factors: HashSet::new(),
            last_updated: Utc::now(),
        }
    }

    /// Add a relationship to the graph
    pub fn add_relationship(&mut self, rel: CausalRelationship) {
        // Remove existing relationship between same source/target if exists
        self.relationships
            .retain(|r| !(r.source == rel.source && r.target == rel.target));

        // Add factors
        self.factors.insert(rel.source.clone());
        self.factors.insert(rel.target.clone());

        // Add new relationship
        self.relationships.push(rel);
        self.last_updated = Utc::now();
    }

    /// Get all relationships where factor is the cause
    pub fn get_effects(&self, factor: &str) -> Vec<&CausalRelationship> {
        self.relationships
            .iter()
            .filter(|r| r.source == factor)
            .collect()
    }

    /// Get all relationships where factor is the effect
    pub fn get_causes(&self, factor: &str) -> Vec<&CausalRelationship> {
        self.relationships
            .iter()
            .filter(|r| r.target == factor)
            .collect()
    }

    /// Get strongest cause for a factor
    pub fn get_leading_indicator(&self, factor: &str) -> Option<(&str, u32)> {
        self.get_causes(factor)
            .into_iter()
            .max_by(|a, b| {
                a.strength
                    .partial_cmp(&b.strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| (r.source.as_str(), r.lag_bars))
    }

    /// Get all relationships
    pub fn relationships(&self) -> &[CausalRelationship] {
        &self.relationships
    }

    /// Get relationship count
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    /// Get count of significant relationships (confidence > 0.7)
    pub fn significant_relationship_count(&self) -> usize {
        self.relationships.iter().filter(|r| r.confidence > 0.7).count()
    }

    /// Get factor count
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    /// Prune old relationships
    pub fn prune_old(&mut self, max_age_days: i64) {
        self.relationships.retain(|r| r.is_valid(max_age_days));
    }
}

/// Causal analyzer for discovering and using causal relationships
pub struct CausalAnalyzer {
    /// The causal graph
    graph: CausalGraph,
    /// Price history for each symbol (returns)
    price_history: HashMap<String, Vec<f64>>,
    /// Last raw price for computing returns
    last_prices: HashMap<String, f64>,
    /// Rolling window size
    window_size: usize,
    /// Significance threshold for tests
    significance_threshold: f64,
    /// Granger test implementation
    granger: GrangerCausalityTest,
}

impl CausalAnalyzer {
    /// Create new causal analyzer
    pub fn new() -> Self {
        Self {
            graph: CausalGraph::new(),
            price_history: HashMap::new(),
            last_prices: HashMap::new(),
            window_size: DEFAULT_WINDOW_SIZE,
            significance_threshold: DEFAULT_SIGNIFICANCE,
            granger: GrangerCausalityTest::new(),
        }
    }

    /// Create with custom parameters
    pub fn with_params(window_size: usize, significance: f64) -> Self {
        Self {
            graph: CausalGraph::new(),
            price_history: HashMap::new(),
            last_prices: HashMap::new(),
            window_size,
            significance_threshold: significance,
            granger: GrangerCausalityTest::with_significance(significance),
        }
    }

    /// Update price for a symbol
    pub fn update_prices(&mut self, symbol: &str, price: f64) {
        if price <= 0.0 {
            return;
        }

        // Compute return if we have last price
        if let Some(&last_price) = self.last_prices.get(symbol) {
            if last_price > 0.0 {
                let ret = (price / last_price).ln();

                // Append to history
                let history = self.price_history.entry(symbol.to_string()).or_default();
                history.push(ret);

                // Trim to window size
                if history.len() > self.window_size {
                    history.remove(0);
                }
            }
        }

        // Update last price
        self.last_prices.insert(symbol.to_string(), price);
    }

    /// Discover causal relationships between all tracked symbols
    pub fn discover_relationships(&mut self) -> Vec<CausalRelationship> {
        let mut new_relationships = Vec::new();
        let symbols: Vec<String> = self.price_history.keys().cloned().collect();

        // Test each pair
        for i in 0..symbols.len() {
            for j in 0..symbols.len() {
                if i == j {
                    continue;
                }

                let source = &symbols[i];
                let target = &symbols[j];

                if let Some(rel) = self.test_pair(source, target) {
                    info!(
                        "[CAUSAL] Found: {}",
                        rel.description()
                    );
                    self.graph.add_relationship(rel.clone());
                    new_relationships.push(rel);
                }
            }
        }

        new_relationships
    }

    /// Test if source Granger-causes target
    fn test_pair(&self, source: &str, target: &str) -> Option<CausalRelationship> {
        let source_data = self.price_history.get(source)?;
        let target_data = self.price_history.get(target)?;

        let result = self.granger.test(source_data, target_data, MAX_LAG)?;

        if !result.significant {
            return None;
        }

        let direction = if result.correlation >= 0.0 {
            CausalDirection::Positive
        } else {
            CausalDirection::Negative
        };

        Some(CausalRelationship::new(
            source.to_string(),
            target.to_string(),
            result.lag,
            result.correlation.abs(),
            direction,
            result.p_value,
        ))
    }

    /// Get what factors predict this symbol
    pub fn get_causes(&self, symbol: &str) -> Vec<&CausalRelationship> {
        self.graph.get_causes(symbol)
    }

    /// Get what this symbol predicts
    pub fn get_effects(&self, symbol: &str) -> Vec<&CausalRelationship> {
        self.graph.get_effects(symbol)
    }

    /// Get the best leading indicator for a symbol
    pub fn get_leading_indicator(&self, symbol: &str) -> Option<(String, u32)> {
        self.graph
            .get_leading_indicator(symbol)
            .map(|(s, lag)| (s.to_string(), lag))
    }

    /// Explain a price move based on causal factors
    pub fn explain_move(&self, symbol: &str, direction: &str) -> Vec<String> {
        let mut explanations = Vec::new();
        let causes = self.get_causes(symbol);

        for cause in causes {
            // Get recent return of the cause
            if let Some(history) = self.price_history.get(&cause.source) {
                if let Some(&recent_return) = history.last() {
                    let cause_direction = if recent_return > 0.0 { "up" } else { "down" };

                    let expected_effect = match cause.direction {
                        CausalDirection::Positive => cause_direction,
                        CausalDirection::Negative => {
                            if cause_direction == "up" {
                                "down"
                            } else {
                                "up"
                            }
                        }
                    };

                    if expected_effect == direction {
                        explanations.push(format!(
                            "{} {} may be caused by: {} {} (lag {}, strength {:.2})",
                            symbol,
                            direction,
                            cause.source,
                            cause_direction,
                            cause.lag_bars,
                            cause.strength
                        ));
                    }
                }
            }
        }

        explanations
    }

    /// Predict regime from causal factors
    pub fn predict_regime_from_causes(&self, symbol: &str) -> Option<Regime> {
        let causes = self.get_causes(symbol);

        // Check VIX influence
        for cause in &causes {
            if cause.source == "VIX" || cause.source.contains("VIX") {
                if let Some(history) = self.price_history.get(&cause.source) {
                    if let Some(&recent) = history.last() {
                        // VIX spiking (>2% move)
                        if recent > 0.02 {
                            return Some(Regime::Volatile);
                        }
                    }
                }
            }
        }

        // Check trend indicators
        if let Some((leader, _)) = self.get_leading_indicator(symbol) {
            if let Some(history) = self.price_history.get(&leader) {
                // Look at recent trend (last 5 bars)
                if history.len() >= 5 {
                    let recent_sum: f64 = history.iter().rev().take(5).sum();

                    // Find the direction relationship
                    if let Some(rel) = causes.iter().find(|r| r.source == leader) {
                        let effective_direction = match rel.direction {
                            CausalDirection::Positive => recent_sum,
                            CausalDirection::Negative => -recent_sum,
                        };

                        if effective_direction > 0.01 {
                            return Some(Regime::TrendingUp);
                        } else if effective_direction < -0.01 {
                            return Some(Regime::TrendingDown);
                        }
                    }
                }
            }
        }

        None
    }

    /// Get confidence adjustment based on causal alignment
    ///
    /// Returns:
    /// - Positive value (e.g., 0.1) if causes agree with trade
    /// - Negative value (e.g., -0.1) if causes disagree
    /// - 0.0 if no relevant causal info
    pub fn get_causal_confidence_adjustment(
        &self,
        symbol: &str,
        is_long: bool,
    ) -> f64 {
        let causes = self.get_causes(symbol);

        if causes.is_empty() {
            return 0.0;
        }

        let mut alignment_score = 0.0;
        let mut total_weight = 0.0;

        for cause in causes {
            if let Some(history) = self.price_history.get(&cause.source) {
                if let Some(&recent_return) = history.last() {
                    // Determine expected target direction based on cause movement
                    let expected_up = match cause.direction {
                        CausalDirection::Positive => recent_return > 0.0,
                        CausalDirection::Negative => recent_return < 0.0,
                    };

                    let agrees = expected_up == is_long;

                    // Weight by strength and confidence
                    let weight = cause.strength * cause.confidence;
                    alignment_score += if agrees { weight } else { -weight };
                    total_weight += weight;
                }
            }
        }

        if total_weight < 0.01 {
            return 0.0;
        }

        // Normalize to [-0.15, +0.15] range
        let normalized = alignment_score / total_weight;
        normalized * 0.15
    }

    /// Get causal context for logging
    pub fn get_causal_context(&self, symbol: &str) -> String {
        let causes = self.get_causes(symbol);

        if causes.is_empty() {
            return format!("{}: no causal factors identified", symbol);
        }

        let cause_strs: Vec<String> = causes
            .iter()
            .take(3)
            .map(|c| {
                format!(
                    "{}{} (lag {}, {:.2})",
                    c.source,
                    c.direction,
                    c.lag_bars,
                    c.strength
                )
            })
            .collect();

        format!("{} caused by: {}", symbol, cause_strs.join(", "))
    }

    /// Get the causal graph
    pub fn graph(&self) -> &CausalGraph {
        &self.graph
    }

    /// Get mutable graph
    pub fn graph_mut(&mut self) -> &mut CausalGraph {
        &mut self.graph
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get history length for a symbol
    pub fn history_len(&self, symbol: &str) -> usize {
        self.price_history.get(symbol).map(|h| h.len()).unwrap_or(0)
    }

    /// Get number of tracked symbols
    pub fn tracked_count(&self) -> usize {
        self.price_history.len()
    }

    /// Save to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let state = CausalState {
            graph: self.graph.clone(),
            window_size: self.window_size,
            significance_threshold: self.significance_threshold,
        };
        let contents = serde_json::to_string_pretty(&state)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Load from file or create new
    pub fn load_or_new(path: &str) -> Self {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(state) = serde_json::from_str::<CausalState>(&contents) {
                info!(
                    "[CAUSAL] Loaded {} relationships from {}",
                    state.graph.relationship_count(),
                    path
                );
                return Self {
                    graph: state.graph,
                    price_history: HashMap::new(),
                    last_prices: HashMap::new(),
                    window_size: state.window_size,
                    significance_threshold: state.significance_threshold,
                    granger: GrangerCausalityTest::with_significance(state.significance_threshold),
                };
            }
        }
        Self::new()
    }

    /// Format summary for logging
    pub fn format_summary(&self) -> String {
        format!(
            "{} relationships, {} factors tracked",
            self.graph.relationship_count(),
            self.tracked_count()
        )
    }

    // ==================== Monitor Methods ====================

    /// Get total relationship count (wrapper for graph method)
    pub fn relationship_count(&self) -> usize {
        self.graph.relationship_count()
    }

    /// Get count of significant relationships (wrapper for graph method)
    pub fn significant_relationship_count(&self) -> usize {
        self.graph.significant_relationship_count()
    }
}

impl Default for CausalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable state for CausalAnalyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CausalState {
    graph: CausalGraph,
    window_size: usize,
    significance_threshold: f64,
}

// ==================== Do-Calculus Types ====================

/// Type of intervention in a causal query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InterventionType {
    /// Observational conditioning P(Y|X=x)
    Observe { variable: String, value: f64 },
    /// Interventional do() operator P(Y|do(X=x))
    Intervene { variable: String, value: f64 },
    /// Counterfactual query: what if X had been different?
    Counterfactual {
        variable: String,
        observed: f64,
        intervened: f64,
    },
}

impl InterventionType {
    /// Get the variable name
    pub fn variable(&self) -> &str {
        match self {
            InterventionType::Observe { variable, .. } => variable,
            InterventionType::Intervene { variable, .. } => variable,
            InterventionType::Counterfactual { variable, .. } => variable,
        }
    }

    /// Get the value being set
    pub fn value(&self) -> f64 {
        match self {
            InterventionType::Observe { value, .. } => *value,
            InterventionType::Intervene { value, .. } => *value,
            InterventionType::Counterfactual { intervened, .. } => *intervened,
        }
    }
}

/// A causal query specifying what to compute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalQuery {
    /// Target variable to predict
    pub target: String,
    /// Interventions (do() operators or observations)
    pub interventions: Vec<InterventionType>,
    /// Conditioning variables (given clause)
    pub given: Vec<(String, f64)>,
}

impl CausalQuery {
    /// Create a simple do() query: P(target | do(treatment = value))
    pub fn do_query(target: &str, treatment: &str, value: f64) -> Self {
        Self {
            target: target.to_string(),
            interventions: vec![InterventionType::Intervene {
                variable: treatment.to_string(),
                value,
            }],
            given: Vec::new(),
        }
    }

    /// Create an observational query: P(target | X = value)
    pub fn observe_query(target: &str, observed: &str, value: f64) -> Self {
        Self {
            target: target.to_string(),
            interventions: vec![InterventionType::Observe {
                variable: observed.to_string(),
                value,
            }],
            given: Vec::new(),
        }
    }

    /// Create a counterfactual query
    pub fn counterfactual_query(
        target: &str,
        variable: &str,
        observed: f64,
        intervened: f64,
    ) -> Self {
        Self {
            target: target.to_string(),
            interventions: vec![InterventionType::Counterfactual {
                variable: variable.to_string(),
                observed,
                intervened,
            }],
            given: Vec::new(),
        }
    }

    /// Add a conditioning variable
    pub fn with_given(mut self, variable: &str, value: f64) -> Self {
        self.given.push((variable.to_string(), value));
        self
    }
}

/// Method used to estimate causal effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimationMethod {
    /// Backdoor adjustment (control for confounders)
    BackdoorAdjustment,
    /// Frontdoor adjustment (use mediators)
    FrontdoorAdjustment,
    /// Instrumental variable estimation
    InstrumentalVariable,
    /// Direct estimation (no confounders)
    DirectEstimation,
    /// Not identifiable from observational data
    NotIdentifiable,
}

impl std::fmt::Display for EstimationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EstimationMethod::BackdoorAdjustment => write!(f, "backdoor"),
            EstimationMethod::FrontdoorAdjustment => write!(f, "frontdoor"),
            EstimationMethod::InstrumentalVariable => write!(f, "IV"),
            EstimationMethod::DirectEstimation => write!(f, "direct"),
            EstimationMethod::NotIdentifiable => write!(f, "not-identifiable"),
        }
    }
}

/// Result of a causal query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// The query that was executed
    pub query: CausalQuery,
    /// Estimated causal effect
    pub estimated_effect: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
    /// Method used for estimation
    pub method_used: EstimationMethod,
    /// Variables adjusted for
    pub confounders_adjusted: Vec<String>,
}

impl QueryResult {
    /// Create a new query result
    pub fn new(
        query: CausalQuery,
        effect: f64,
        ci: (f64, f64),
        method: EstimationMethod,
        confounders: Vec<String>,
    ) -> Self {
        Self {
            query,
            estimated_effect: effect,
            confidence_interval: ci,
            method_used: method,
            confounders_adjusted: confounders,
        }
    }

    /// Create a result indicating non-identifiability
    pub fn not_identifiable(query: CausalQuery) -> Self {
        Self {
            query,
            estimated_effect: 0.0,
            confidence_interval: (f64::NEG_INFINITY, f64::INFINITY),
            method_used: EstimationMethod::NotIdentifiable,
            confounders_adjusted: Vec::new(),
        }
    }

    /// Is the effect statistically significant?
    pub fn is_significant(&self) -> bool {
        let (lo, hi) = self.confidence_interval;
        // Significant if CI doesn't contain 0
        !(lo <= 0.0 && hi >= 0.0)
    }
}

/// Structural equation for a variable in the causal model
/// Y = sum(coef_i * X_i) + noise
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralEquation {
    /// Target variable this equation defines
    pub target: String,
    /// Parent variables
    pub parents: Vec<String>,
    /// Linear coefficients for each parent
    pub coefficients: Vec<f64>,
    /// Standard deviation of noise term
    pub noise_std: f64,
}

impl StructuralEquation {
    /// Create a new structural equation
    pub fn new(target: &str, parents: Vec<String>, coefficients: Vec<f64>, noise_std: f64) -> Self {
        Self {
            target: target.to_string(),
            parents,
            coefficients,
            noise_std,
        }
    }

    /// Evaluate the equation given parent values
    pub fn evaluate(&self, parent_values: &HashMap<String, f64>) -> f64 {
        let mut result = 0.0;
        for (parent, coef) in self.parents.iter().zip(self.coefficients.iter()) {
            if let Some(&val) = parent_values.get(parent) {
                result += coef * val;
            }
        }
        result
    }

    /// Evaluate with noise
    pub fn evaluate_with_noise(&self, parent_values: &HashMap<String, f64>, noise: f64) -> f64 {
        self.evaluate(parent_values) + noise * self.noise_std
    }
}

/// Full structural causal model (SCM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalModel {
    /// The causal graph structure
    pub graph: CausalGraph,
    /// Structural equations for each variable
    pub structural_equations: HashMap<String, StructuralEquation>,
    /// Observed data for each variable
    pub observed_data: HashMap<String, Vec<f64>>,
    /// Known latent confounders (pairs with hidden common cause)
    pub latent_confounders: Vec<(String, String)>,
}

impl CausalModel {
    /// Create a new causal model from an existing graph
    pub fn new(graph: CausalGraph) -> Self {
        Self {
            graph,
            structural_equations: HashMap::new(),
            observed_data: HashMap::new(),
            latent_confounders: Vec::new(),
        }
    }

    /// Add a structural equation
    pub fn add_equation(&mut self, eq: StructuralEquation) {
        self.structural_equations.insert(eq.target.clone(), eq);
    }

    /// Add observed data for a variable
    pub fn add_observed_data(&mut self, variable: &str, data: Vec<f64>) {
        self.observed_data.insert(variable.to_string(), data);
    }

    /// Add a latent confounder between two variables
    pub fn add_latent_confounder(&mut self, var1: &str, var2: &str) {
        self.latent_confounders
            .push((var1.to_string(), var2.to_string()));
    }

    /// Check if two variables have a latent confounder
    pub fn has_latent_confounder(&self, var1: &str, var2: &str) -> bool {
        self.latent_confounders.iter().any(|(a, b)| {
            (a == var1 && b == var2) || (a == var2 && b == var1)
        })
    }

    /// Get all variables in the model
    pub fn variables(&self) -> Vec<String> {
        let mut vars: HashSet<String> = self.structural_equations.keys().cloned().collect();
        for eq in self.structural_equations.values() {
            for parent in &eq.parents {
                vars.insert(parent.clone());
            }
        }
        vars.into_iter().collect()
    }

    /// Get parents of a variable (from structural equations)
    pub fn get_parents(&self, variable: &str) -> Vec<String> {
        self.structural_equations
            .get(variable)
            .map(|eq| eq.parents.clone())
            .unwrap_or_default()
    }

    /// Get children of a variable
    pub fn get_children(&self, variable: &str) -> Vec<String> {
        self.structural_equations
            .iter()
            .filter_map(|(target, eq)| {
                if eq.parents.contains(&variable.to_string()) {
                    Some(target.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get mean of observed data for a variable
    pub fn get_mean(&self, variable: &str) -> Option<f64> {
        self.observed_data.get(variable).map(|data| {
            if data.is_empty() {
                0.0
            } else {
                data.iter().sum::<f64>() / data.len() as f64
            }
        })
    }

    /// Get variance of observed data
    pub fn get_variance(&self, variable: &str) -> Option<f64> {
        self.observed_data.get(variable).and_then(|data| {
            if data.len() < 2 {
                return None;
            }
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
            Some(variance)
        })
    }
}

impl Default for CausalModel {
    fn default() -> Self {
        Self::new(CausalGraph::new())
    }
}

/// Rules of do-calculus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoRule {
    /// Rule 1: Insertion/deletion of observations
    /// P(y|do(x),z,w) = P(y|do(x),w) if (Y ⊥ Z | X, W)_{G_{\overline{X}}}
    Rule1InsertDelete,
    /// Rule 2: Action/observation exchange
    /// P(y|do(x),do(z),w) = P(y|do(x),z,w) if (Y ⊥ Z | X, W)_{G_{\overline{X}\underline{Z}}}
    Rule2ActionExchange,
    /// Rule 3: Insertion/deletion of actions
    /// P(y|do(x),do(z),w) = P(y|do(x),w) if (Y ⊥ Z | X, W)_{G_{\overline{X}\overline{Z(W)}}}
    Rule3ActionDeletion,
}

impl std::fmt::Display for DoRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DoRule::Rule1InsertDelete => write!(f, "Rule 1 (insert/delete obs)"),
            DoRule::Rule2ActionExchange => write!(f, "Rule 2 (action/obs exchange)"),
            DoRule::Rule3ActionDeletion => write!(f, "Rule 3 (action deletion)"),
        }
    }
}

/// Do-calculus engine for computing causal effects
#[derive(Debug, Clone)]
pub struct DoCalculusEngine {
    /// The structural causal model
    model: CausalModel,
    /// Rules applied during last computation
    rules_applied: Vec<DoRule>,
}

impl DoCalculusEngine {
    /// Create a new do-calculus engine
    pub fn new(model: CausalModel) -> Self {
        Self {
            model,
            rules_applied: Vec::new(),
        }
    }

    /// Execute a causal query
    pub fn query(&mut self, q: CausalQuery) -> QueryResult {
        self.rules_applied.clear();

        // Get the treatment variable(s) from interventions
        let treatments: Vec<_> = q
            .interventions
            .iter()
            .filter_map(|i| match i {
                InterventionType::Intervene { variable, .. } => Some(variable.clone()),
                _ => None,
            })
            .collect();

        if treatments.is_empty() {
            // Pure observational query
            return self.observational_query(&q);
        }

        let treatment = &treatments[0];
        let target = &q.target;

        // Check identifiability
        if !self.is_identifiable(target, treatment) {
            return QueryResult::not_identifiable(q);
        }

        // Find adjustment set
        let adjustment_set = self.find_adjustment_set(treatment, target);

        match adjustment_set {
            Some(confounders) if !confounders.is_empty() => {
                // Use backdoor adjustment
                let value = q.interventions.iter().find_map(|i| match i {
                    InterventionType::Intervene { variable, value } if variable == treatment => {
                        Some(*value)
                    }
                    _ => None,
                }).unwrap_or(0.0);

                let effect = self.backdoor_adjustment(target, treatment, value, &confounders);
                let ci = self.compute_confidence_interval(target, treatment, &confounders);

                QueryResult::new(
                    q,
                    effect,
                    ci,
                    EstimationMethod::BackdoorAdjustment,
                    confounders,
                )
            }
            _ => {
                // Direct estimation
                let value = q.interventions.iter().find_map(|i| match i {
                    InterventionType::Intervene { variable, value } if variable == treatment => {
                        Some(*value)
                    }
                    _ => None,
                }).unwrap_or(0.0);

                let effect = self.do_intervention(target, treatment, value);
                let ci = (effect - 0.1 * effect.abs().max(0.1), effect + 0.1 * effect.abs().max(0.1));

                QueryResult::new(
                    q,
                    effect,
                    ci,
                    EstimationMethod::DirectEstimation,
                    Vec::new(),
                )
            }
        }
    }

    /// Handle pure observational query
    fn observational_query(&self, q: &CausalQuery) -> QueryResult {
        // For observations, compute conditional expectation
        let observed: Vec<_> = q
            .interventions
            .iter()
            .filter_map(|i| match i {
                InterventionType::Observe { variable, value } => Some((variable.clone(), *value)),
                _ => None,
            })
            .collect();

        let effect = self.conditional_expectation(&q.target, &observed);

        QueryResult::new(
            q.clone(),
            effect,
            (effect - 0.1 * effect.abs().max(0.1), effect + 0.1 * effect.abs().max(0.1)),
            EstimationMethod::DirectEstimation,
            Vec::new(),
        )
    }

    /// Compute P(target | do(variable = value))
    pub fn do_intervention(&self, target: &str, variable: &str, value: f64) -> f64 {
        // Check if there are confounders
        let backdoor_paths = self.find_backdoor_paths(variable, target);

        if backdoor_paths.is_empty() {
            // No confounders, direct computation using structural equation
            self.direct_intervention_effect(target, variable, value)
        } else {
            // Use backdoor adjustment
            let adjustment_set = self.find_adjustment_set(variable, target)
                .unwrap_or_default();
            self.backdoor_adjustment(target, variable, value, &adjustment_set)
        }
    }

    /// Direct intervention effect using structural equations
    fn direct_intervention_effect(&self, target: &str, variable: &str, value: f64) -> f64 {
        // Use structural equation if available
        if let Some(eq) = self.model.structural_equations.get(target) {
            if eq.parents.contains(&variable.to_string()) {
                // Find coefficient for this variable
                if let Some(idx) = eq.parents.iter().position(|p| p == variable) {
                    return eq.coefficients[idx] * value;
                }
            }
        }

        // Fall back to relationship in causal graph
        for rel in self.model.graph.relationships() {
            if rel.source == variable && rel.target == target {
                let sign = match rel.direction {
                    CausalDirection::Positive => 1.0,
                    CausalDirection::Negative => -1.0,
                };
                return sign * rel.strength * value;
            }
        }

        0.0
    }

    /// Backdoor adjustment formula
    /// P(Y|do(X)) = sum_Z P(Y|X,Z) * P(Z)
    pub fn backdoor_adjustment(
        &self,
        target: &str,
        treatment: &str,
        value: f64,
        confounders: &[String],
    ) -> f64 {
        if confounders.is_empty() {
            return self.direct_intervention_effect(target, treatment, value);
        }

        // Discretize confounders and compute weighted sum
        let mut total_effect = 0.0;
        let mut total_weight = 0.0;

        // Use observed data to compute adjustment
        let n_samples = self.model.observed_data
            .get(treatment)
            .map(|d| d.len())
            .unwrap_or(100);

        // For each "stratum" of confounders
        let num_strata = 10.min(n_samples / 10).max(1);

        for stratum in 0..num_strata {
            let stratum_weight = self.get_stratum_weight(confounders, stratum, num_strata);
            let stratum_effect = self.get_stratum_effect(target, treatment, value, confounders, stratum, num_strata);

            total_effect += stratum_weight * stratum_effect;
            total_weight += stratum_weight;
        }

        if total_weight > 0.0 {
            total_effect / total_weight
        } else {
            self.direct_intervention_effect(target, treatment, value)
        }
    }

    /// Get weight for a stratum (approximates P(Z))
    fn get_stratum_weight(&self, confounders: &[String], stratum: usize, num_strata: usize) -> f64 {
        // Uniform strata for now
        let _ = confounders;
        let _ = stratum;
        1.0 / num_strata as f64
    }

    /// Get effect in a stratum (approximates E[Y|X,Z] at stratum)
    fn get_stratum_effect(
        &self,
        target: &str,
        treatment: &str,
        value: f64,
        confounders: &[String],
        stratum: usize,
        num_strata: usize,
    ) -> f64 {
        // Compute effect in this stratum
        let base_effect = self.direct_intervention_effect(target, treatment, value);

        // Adjust for confounder influence
        let mut confounder_adjustment = 0.0;
        for confounder in confounders {
            if let Some(eq) = self.model.structural_equations.get(target) {
                if let Some(idx) = eq.parents.iter().position(|p| p == confounder) {
                    // Effect of confounder on target
                    let coef = eq.coefficients[idx];
                    // Approximate stratum value
                    let stratum_val = (stratum as f64 + 0.5) / num_strata as f64;
                    confounder_adjustment += coef * stratum_val;
                }
            }
        }

        base_effect + confounder_adjustment * 0.1 // Attenuate confounder effect
    }

    /// Find all backdoor paths from treatment to outcome
    pub fn find_backdoor_paths(&self, from: &str, to: &str) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut current_path = vec![from.to_string()];

        // Find paths that go through parents (backdoor = into treatment)
        let parents = self.model.get_parents(from);
        for parent in parents {
            visited.insert(from.to_string());
            self.find_paths_to(&parent, to, &mut visited, &mut current_path, &mut paths);
            visited.remove(from);
        }

        // Also check latent confounders
        for (a, b) in &self.model.latent_confounders {
            if a == from || b == from {
                let other = if a == from { b } else { a };
                visited.insert(from.to_string());
                self.find_paths_to(other, to, &mut visited, &mut current_path, &mut paths);
                visited.remove(from);
            }
        }

        paths
    }

    /// Helper to find paths between nodes
    fn find_paths_to(
        &self,
        current: &str,
        target: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        paths: &mut Vec<Vec<String>>,
    ) {
        if visited.contains(current) {
            return;
        }

        path.push(current.to_string());
        visited.insert(current.to_string());

        if current == target {
            paths.push(path.clone());
        } else {
            // Check children
            let children = self.model.get_children(current);
            for child in children {
                self.find_paths_to(&child, target, visited, path, paths);
            }

            // Check parents (for undirected search)
            let parents = self.model.get_parents(current);
            for parent in parents {
                self.find_paths_to(&parent, target, visited, path, paths);
            }
        }

        path.pop();
        visited.remove(current);
    }

    /// Find minimal adjustment set that blocks all backdoor paths
    pub fn find_adjustment_set(&self, treatment: &str, outcome: &str) -> Option<Vec<String>> {
        let backdoor_paths = self.find_backdoor_paths(treatment, outcome);

        if backdoor_paths.is_empty() {
            return Some(Vec::new()); // No adjustment needed
        }

        // Collect all variables on backdoor paths (except treatment and outcome)
        let mut candidates: HashSet<String> = HashSet::new();
        for path in &backdoor_paths {
            for node in path {
                if node != treatment && node != outcome {
                    candidates.insert(node.clone());
                }
            }
        }

        // Add parents of treatment (common approach)
        for parent in self.model.get_parents(treatment) {
            candidates.insert(parent);
        }

        // Check if this set blocks all backdoor paths
        let adjustment: Vec<String> = candidates.into_iter().collect();

        if self.blocks_all_backdoor_paths(treatment, outcome, &adjustment) {
            Some(adjustment)
        } else {
            // Check for latent confounders making it non-identifiable
            if self.model.has_latent_confounder(treatment, outcome) {
                None
            } else {
                Some(adjustment)
            }
        }
    }

    /// Check if adjustment set blocks all backdoor paths
    fn blocks_all_backdoor_paths(&self, treatment: &str, outcome: &str, adjustment: &[String]) -> bool {
        let paths = self.find_backdoor_paths(treatment, outcome);

        for path in paths {
            let mut blocked = false;
            for node in &path {
                if adjustment.contains(node) {
                    blocked = true;
                    break;
                }
            }
            if !blocked && path.len() > 2 {
                // Path not blocked
                return false;
            }
        }

        true
    }

    /// Check if causal effect is identifiable
    pub fn is_identifiable(&self, target: &str, treatment: &str) -> bool {
        // Effect is identifiable if:
        // 1. No latent confounders between treatment and outcome, OR
        // 2. We can find a valid adjustment set

        if self.model.has_latent_confounder(treatment, target) {
            // Check if we can still identify via other methods
            // (simplified: return false for latent confounders)
            return false;
        }

        // Check if adjustment set exists
        self.find_adjustment_set(treatment, target).is_some()
    }

    /// Compute Average Treatment Effect
    /// ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
    pub fn average_treatment_effect(&self, treatment: &str, outcome: &str) -> f64 {
        let effect_1 = self.do_intervention(outcome, treatment, 1.0);
        let effect_0 = self.do_intervention(outcome, treatment, 0.0);
        effect_1 - effect_0
    }

    /// Compute Conditional Average Treatment Effect for subgroup
    pub fn conditional_ate(
        &self,
        treatment: &str,
        outcome: &str,
        condition: &str,
        value: f64,
    ) -> f64 {
        // Simplified: scale ATE by condition value
        let base_ate = self.average_treatment_effect(treatment, outcome);

        // Adjust based on condition's relationship to outcome
        if let Some(rel) = self.model.graph.relationships().iter()
            .find(|r| r.source == condition && r.target == outcome) {
            let modifier = match rel.direction {
                CausalDirection::Positive => 1.0 + 0.1 * value,
                CausalDirection::Negative => 1.0 - 0.1 * value,
            };
            base_ate * modifier
        } else {
            base_ate
        }
    }

    /// Compute counterfactual: given factual observations, what would Y have been if X were different?
    /// Three steps: abduction, action, prediction
    pub fn counterfactual(
        &self,
        factual: &[(String, f64)],
        intervention: &str,
        new_value: f64,
        target: &str,
    ) -> f64 {
        // Step 1: Abduction - infer noise terms from factual observations
        let noise_terms = self.abduction(factual, target);

        // Step 2: Action - perform intervention
        let mut intervened_values: HashMap<String, f64> = factual.iter().cloned().collect();
        intervened_values.insert(intervention.to_string(), new_value);

        // Step 3: Prediction - compute target under intervention with inferred noise
        self.prediction(target, &intervened_values, &noise_terms)
    }

    /// Abduction step: infer noise terms from observations
    fn abduction(&self, factual: &[(String, f64)], target: &str) -> HashMap<String, f64> {
        let mut noise = HashMap::new();
        let factual_map: HashMap<_, _> = factual.iter().cloned().collect();

        if let Some(eq) = self.model.structural_equations.get(target) {
            // noise = observed - predicted
            let predicted = eq.evaluate(&factual_map);
            if let Some(&observed) = factual_map.get(target) {
                noise.insert(target.to_string(), observed - predicted);
            }
        }

        noise
    }

    /// Prediction step: compute target under intervention with noise
    fn prediction(
        &self,
        target: &str,
        values: &HashMap<String, f64>,
        noise: &HashMap<String, f64>,
    ) -> f64 {
        if let Some(eq) = self.model.structural_equations.get(target) {
            let base = eq.evaluate(values);
            let noise_term = noise.get(target).copied().unwrap_or(0.0);
            base + noise_term
        } else {
            // Fall back to observed value or 0
            values.get(target).copied().unwrap_or(0.0)
        }
    }

    /// Compute conditional expectation E[Y | conditions]
    fn conditional_expectation(&self, target: &str, conditions: &[(String, f64)]) -> f64 {
        let condition_map: HashMap<_, _> = conditions.iter().cloned().collect();

        if let Some(eq) = self.model.structural_equations.get(target) {
            eq.evaluate(&condition_map)
        } else {
            // Use mean of observed data
            self.model.get_mean(target).unwrap_or(0.0)
        }
    }

    /// Compute confidence interval for causal effect
    fn compute_confidence_interval(
        &self,
        target: &str,
        treatment: &str,
        _confounders: &[String],
    ) -> (f64, f64) {
        // Use variance of observed data to estimate CI
        let target_var = self.model.get_variance(target).unwrap_or(0.1);
        let treatment_var = self.model.get_variance(treatment).unwrap_or(0.1);

        let n = self.model.observed_data
            .get(target)
            .map(|d| d.len())
            .unwrap_or(30) as f64;

        let se = (target_var / n + treatment_var / n).sqrt();
        let z = 1.96; // 95% CI

        let effect = self.average_treatment_effect(treatment, target);
        (effect - z * se, effect + z * se)
    }

    /// Get the causal model
    pub fn model(&self) -> &CausalModel {
        &self.model
    }

    /// Get mutable causal model
    pub fn model_mut(&mut self) -> &mut CausalModel {
        &mut self.model
    }

    /// Get rules applied in last query
    pub fn rules_applied(&self) -> &[DoRule] {
        &self.rules_applied
    }
}

// ==================== Backdoor Criterion Functions ====================

/// Check if a set blocks all backdoor paths
pub fn blocks_backdoor(
    graph: &CausalGraph,
    treatment: &str,
    outcome: &str,
    adjustment: &[String],
) -> bool {
    // Build a simple model to check
    let model = CausalModel::new(graph.clone());
    let engine = DoCalculusEngine::new(model);
    engine.blocks_all_backdoor_paths(treatment, outcome, adjustment)
}

/// Find minimal adjustment set for backdoor criterion
pub fn find_minimal_adjustment(
    graph: &CausalGraph,
    treatment: &str,
    outcome: &str,
) -> Option<Vec<String>> {
    let model = CausalModel::new(graph.clone());
    let engine = DoCalculusEngine::new(model);
    engine.find_adjustment_set(treatment, outcome)
}

// ==================== Extended CausalAnalyzer ====================

impl CausalAnalyzer {
    /// Create a do-calculus engine from current state
    pub fn create_do_engine(&self) -> DoCalculusEngine {
        let mut model = CausalModel::new(self.graph.clone());

        // Add observed data from price history
        for (symbol, history) in &self.price_history {
            model.add_observed_data(symbol, history.clone());
        }

        // Infer structural equations from relationships
        for rel in self.graph.relationships() {
            let coef = match rel.direction {
                CausalDirection::Positive => rel.strength,
                CausalDirection::Negative => -rel.strength,
            };

            // Check if we already have an equation for this target
            if let Some(eq) = model.structural_equations.get_mut(&rel.target) {
                if !eq.parents.contains(&rel.source) {
                    eq.parents.push(rel.source.clone());
                    eq.coefficients.push(coef);
                }
            } else {
                let eq = StructuralEquation::new(
                    &rel.target,
                    vec![rel.source.clone()],
                    vec![coef],
                    0.1, // Default noise
                );
                model.add_equation(eq);
            }
        }

        DoCalculusEngine::new(model)
    }

    /// Compute intervention effect: "What happens to target if we intervene on treatment?"
    pub fn intervention_effect(&self, treatment: &str, target: &str) -> Option<f64> {
        let engine = self.create_do_engine();

        if !engine.is_identifiable(target, treatment) {
            return None;
        }

        Some(engine.average_treatment_effect(treatment, target))
    }

    /// Counterfactual query for trade analysis
    pub fn would_trade_succeed_if(
        &self,
        current_outcome: f64,
        intervention_var: &str,
        current_value: f64,
        new_value: f64,
        target: &str,
    ) -> f64 {
        let engine = self.create_do_engine();

        let factual = vec![
            (intervention_var.to_string(), current_value),
            (target.to_string(), current_outcome),
        ];

        engine.counterfactual(&factual, intervention_var, new_value, target)
    }

    /// Get intervention-adjusted confidence for a trade setup
    pub fn get_intervention_adjusted_confidence(
        &self,
        symbol: &str,
        is_long: bool,
        base_confidence: f64,
    ) -> f64 {
        let causes = self.get_causes(symbol);

        if causes.is_empty() {
            return base_confidence;
        }

        let engine = self.create_do_engine();
        let mut adjustment = 0.0;

        for cause in causes {
            // Get current value of cause
            if let Some(history) = self.price_history.get(&cause.source) {
                if let Some(&recent) = history.last() {
                    // Compute intervention effect
                    let effect = engine.do_intervention(symbol, &cause.source, recent);

                    // Adjust based on trade direction
                    let directional_effect = if is_long { effect } else { -effect };

                    adjustment += directional_effect * cause.confidence * 0.1;
                }
            }
        }

        // Clamp to valid range
        (base_confidence + adjustment).clamp(0.0, 1.0)
    }

    /// Enhanced explain_move using causal interventions
    pub fn explain_move_causal(&self, symbol: &str, direction: &str) -> Vec<String> {
        let mut explanations = self.explain_move(symbol, direction);

        // Add intervention-based explanations
        let engine = self.create_do_engine();

        for rel in self.graph.get_causes(symbol) {
            if let Some(history) = self.price_history.get(&rel.source) {
                if let Some(&recent) = history.last() {
                    let effect = engine.do_intervention(symbol, &rel.source, recent);

                    if effect.abs() > 0.001 {
                        let effect_dir = if effect > 0.0 { "increase" } else { "decrease" };
                        explanations.push(format!(
                            "[CAUSAL] do({} = {:.4}) would {} {} by {:.4}",
                            rel.source,
                            recent,
                            effect_dir,
                            symbol,
                            effect.abs()
                        ));
                    }
                }
            }
        }

        explanations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_direction_display() {
        assert_eq!(format!("{}", CausalDirection::Positive), "+");
        assert_eq!(format!("{}", CausalDirection::Negative), "-");
    }

    #[test]
    fn test_causal_factor_from_str() {
        assert_eq!(CausalFactor::from_str("VIX"), CausalFactor::VIX);
        assert_eq!(CausalFactor::from_str("DXY"), CausalFactor::DXY);
        assert_eq!(CausalFactor::from_str("10Y"), CausalFactor::Yields);
        assert_eq!(CausalFactor::from_str("OIL"), CausalFactor::Oil);
        assert_eq!(CausalFactor::from_str("GOLD"), CausalFactor::Gold);
        assert_eq!(
            CausalFactor::from_str("AAPL"),
            CausalFactor::Symbol("AAPL".to_string())
        );
    }

    #[test]
    fn test_causal_relationship_description() {
        let rel = CausalRelationship::new(
            "VIX".to_string(),
            "SPY".to_string(),
            2,
            0.65,
            CausalDirection::Negative,
            0.01,
        );

        let desc = rel.description();
        assert!(desc.contains("VIX"));
        assert!(desc.contains("SPY"));
        assert!(desc.contains("lag 2"));
    }

    #[test]
    fn test_granger_test_insufficient_data() {
        let granger = GrangerCausalityTest::new();
        let source = vec![1.0, 2.0, 3.0]; // Too few points
        let target = vec![1.0, 2.0, 3.0];

        let result = granger.test(&source, &target, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_granger_test_with_data() {
        let granger = GrangerCausalityTest::new();

        // Create synthetic data where source leads target
        let mut source = Vec::new();
        let mut target = Vec::new();

        for i in 0..100 {
            source.push((i as f64 * 0.1).sin());
        }

        // Target is lagged version of source
        for i in 0..100 {
            if i < 2 {
                target.push(0.0);
            } else {
                target.push(source[i - 2] + 0.1 * ((i as f64 * 0.3).cos()));
            }
        }

        let result = granger.test(&source, &target, 5);
        assert!(result.is_some());

        let res = result.unwrap();
        assert!(res.f_statistic > 0.0);
        // Lag should be around 2
        assert!(res.lag >= 1 && res.lag <= 5);
    }

    #[test]
    fn test_causal_graph_add_relationship() {
        let mut graph = CausalGraph::new();

        let rel = CausalRelationship::new(
            "A".to_string(),
            "B".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        );

        graph.add_relationship(rel);

        assert_eq!(graph.relationship_count(), 1);
        assert_eq!(graph.factor_count(), 2);
    }

    #[test]
    fn test_causal_graph_get_causes_effects() {
        let mut graph = CausalGraph::new();

        graph.add_relationship(CausalRelationship::new(
            "A".to_string(),
            "B".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        ));

        graph.add_relationship(CausalRelationship::new(
            "C".to_string(),
            "B".to_string(),
            2,
            0.3,
            CausalDirection::Negative,
            0.02,
        ));

        let causes_b = graph.get_causes("B");
        assert_eq!(causes_b.len(), 2);

        let effects_a = graph.get_effects("A");
        assert_eq!(effects_a.len(), 1);
    }

    #[test]
    fn test_causal_graph_leading_indicator() {
        let mut graph = CausalGraph::new();

        graph.add_relationship(CausalRelationship::new(
            "A".to_string(),
            "B".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        ));

        graph.add_relationship(CausalRelationship::new(
            "C".to_string(),
            "B".to_string(),
            2,
            0.7, // Stronger
            CausalDirection::Negative,
            0.02,
        ));

        let (leader, lag) = graph.get_leading_indicator("B").unwrap();
        assert_eq!(leader, "C");
        assert_eq!(lag, 2);
    }

    #[test]
    fn test_causal_analyzer_update_prices() {
        let mut analyzer = CausalAnalyzer::new();

        analyzer.update_prices("AAPL", 100.0);
        analyzer.update_prices("AAPL", 101.0);
        analyzer.update_prices("AAPL", 102.0);

        // Should have 2 returns (need 3 prices for 2 returns)
        assert_eq!(analyzer.history_len("AAPL"), 2);
    }

    #[test]
    fn test_causal_analyzer_confidence_adjustment() {
        let mut analyzer = CausalAnalyzer::new();

        // Add some price history
        for i in 0..50 {
            analyzer.update_prices("VIX", 20.0 + (i as f64 * 0.1));
            analyzer.update_prices("SPY", 400.0 - (i as f64 * 0.5));
        }

        // Manually add a relationship
        analyzer.graph.add_relationship(CausalRelationship::new(
            "VIX".to_string(),
            "SPY".to_string(),
            1,
            0.7,
            CausalDirection::Negative,
            0.01,
        ));

        // VIX is going up, which with negative correlation means SPY should go down
        // So a long position should get negative adjustment
        let adj_long = analyzer.get_causal_confidence_adjustment("SPY", true);
        let adj_short = analyzer.get_causal_confidence_adjustment("SPY", false);

        // Long should be penalized when VIX up (negative correlation)
        assert!(adj_long < adj_short);
    }

    #[test]
    fn test_causal_analyzer_context() {
        let mut analyzer = CausalAnalyzer::new();

        // No relationships
        let ctx = analyzer.get_causal_context("AAPL");
        assert!(ctx.contains("no causal factors"));

        // Add relationship
        analyzer.graph.add_relationship(CausalRelationship::new(
            "SPY".to_string(),
            "AAPL".to_string(),
            1,
            0.6,
            CausalDirection::Positive,
            0.01,
        ));

        let ctx = analyzer.get_causal_context("AAPL");
        assert!(ctx.contains("SPY"));
        assert!(ctx.contains("caused by"));
    }

    #[test]
    fn test_pearson_correlation() {
        let granger = GrangerCausalityTest::new();

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = granger.pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.01);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = granger.pearson_correlation(&x, &y_neg);
        assert!((corr_neg - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_relationship_validity() {
        let rel = CausalRelationship::new(
            "A".to_string(),
            "B".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        );

        // Should be valid for recent check
        assert!(rel.is_valid(30));
    }

    #[test]
    fn test_causal_graph_prune() {
        let mut graph = CausalGraph::new();

        let mut old_rel = CausalRelationship::new(
            "A".to_string(),
            "B".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        );
        // Make it old
        old_rel.last_validated = Utc::now() - chrono::Duration::days(100);

        graph.add_relationship(old_rel);

        let new_rel = CausalRelationship::new(
            "C".to_string(),
            "D".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        );
        graph.add_relationship(new_rel);

        assert_eq!(graph.relationship_count(), 2);

        graph.prune_old(30);

        assert_eq!(graph.relationship_count(), 1);
    }

    // ==================== Do-Calculus Tests ====================

    #[test]
    fn test_backdoor_criterion() {
        // Create a simple causal graph: Z -> X -> Y, Z -> Y (Z is a confounder)
        let mut graph = CausalGraph::new();

        // Z -> X
        graph.add_relationship(CausalRelationship::new(
            "Z".to_string(),
            "X".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        ));

        // X -> Y
        graph.add_relationship(CausalRelationship::new(
            "X".to_string(),
            "Y".to_string(),
            1,
            0.7,
            CausalDirection::Positive,
            0.01,
        ));

        // Z -> Y (backdoor path)
        graph.add_relationship(CausalRelationship::new(
            "Z".to_string(),
            "Y".to_string(),
            1,
            0.3,
            CausalDirection::Positive,
            0.01,
        ));

        // Z should block the backdoor path
        assert!(blocks_backdoor(&graph, "X", "Y", &["Z".to_string()]));
    }

    #[test]
    fn test_adjustment_set() {
        let mut graph = CausalGraph::new();

        // Simple chain: A -> B -> C
        graph.add_relationship(CausalRelationship::new(
            "A".to_string(),
            "B".to_string(),
            1,
            0.6,
            CausalDirection::Positive,
            0.01,
        ));

        graph.add_relationship(CausalRelationship::new(
            "B".to_string(),
            "C".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        ));

        // For A -> C effect via B, no adjustment needed
        let adjustment = find_minimal_adjustment(&graph, "A", "C");
        assert!(adjustment.is_some());
    }

    #[test]
    fn test_do_intervention() {
        let mut model = CausalModel::default();

        // Add structural equation: Y = 0.5 * X
        model.add_equation(StructuralEquation::new(
            "Y",
            vec!["X".to_string()],
            vec![0.5],
            0.1,
        ));

        // Add relationship to graph
        model.graph.add_relationship(CausalRelationship::new(
            "X".to_string(),
            "Y".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        ));

        let engine = DoCalculusEngine::new(model);

        // do(X = 2) should give Y = 0.5 * 2 = 1.0
        let effect = engine.do_intervention("Y", "X", 2.0);
        assert!((effect - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_counterfactual() {
        let mut model = CausalModel::default();

        // Y = 0.6 * X + noise
        model.add_equation(StructuralEquation::new(
            "Y",
            vec!["X".to_string()],
            vec![0.6],
            0.1,
        ));

        let engine = DoCalculusEngine::new(model);

        // Factual: X = 1.0, Y = 0.8 (noise = 0.2)
        // Counterfactual: What if X = 2.0?
        // Should be: 0.6 * 2.0 + 0.2 = 1.4
        let factual = vec![
            ("X".to_string(), 1.0),
            ("Y".to_string(), 0.8),
        ];

        let counterfactual_y = engine.counterfactual(&factual, "X", 2.0, "Y");
        assert!((counterfactual_y - 1.4).abs() < 0.01);
    }

    #[test]
    fn test_ate_computation() {
        let mut model = CausalModel::default();

        // Y = 0.5 * X (linear effect)
        model.add_equation(StructuralEquation::new(
            "Y",
            vec!["X".to_string()],
            vec![0.5],
            0.0,
        ));

        model.graph.add_relationship(CausalRelationship::new(
            "X".to_string(),
            "Y".to_string(),
            1,
            0.5,
            CausalDirection::Positive,
            0.01,
        ));

        let engine = DoCalculusEngine::new(model);

        // ATE = E[Y|do(X=1)] - E[Y|do(X=0)] = 0.5 - 0 = 0.5
        let ate = engine.average_treatment_effect("X", "Y");
        assert!((ate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_identifiability() {
        let mut model = CausalModel::default();

        // Simple identifiable case: X -> Y
        model.add_equation(StructuralEquation::new(
            "Y",
            vec!["X".to_string()],
            vec![0.5],
            0.1,
        ));

        let engine = DoCalculusEngine::new(model.clone());
        assert!(engine.is_identifiable("Y", "X"));

        // Add latent confounder - makes it non-identifiable
        model.add_latent_confounder("X", "Y");
        let engine2 = DoCalculusEngine::new(model);
        assert!(!engine2.is_identifiable("Y", "X"));
    }

    #[test]
    fn test_causal_query() {
        let mut model = CausalModel::default();

        model.add_equation(StructuralEquation::new(
            "Y",
            vec!["X".to_string()],
            vec![0.7],
            0.1,
        ));

        model.graph.add_relationship(CausalRelationship::new(
            "X".to_string(),
            "Y".to_string(),
            1,
            0.7,
            CausalDirection::Positive,
            0.01,
        ));

        let mut engine = DoCalculusEngine::new(model);

        // Query P(Y | do(X = 1.0))
        let query = CausalQuery::do_query("Y", "X", 1.0);
        let result = engine.query(query);

        assert!((result.estimated_effect - 0.7).abs() < 0.1);
        assert!(result.method_used != EstimationMethod::NotIdentifiable);
    }

    #[test]
    fn test_structural_equation_evaluation() {
        let eq = StructuralEquation::new(
            "Y",
            vec!["X1".to_string(), "X2".to_string()],
            vec![0.5, 0.3],
            0.1,
        );

        let mut values = HashMap::new();
        values.insert("X1".to_string(), 2.0);
        values.insert("X2".to_string(), 3.0);

        // Y = 0.5 * 2 + 0.3 * 3 = 1.0 + 0.9 = 1.9
        let result = eq.evaluate(&values);
        assert!((result - 1.9).abs() < 0.01);
    }

    #[test]
    fn test_causal_model_parents_children() {
        let mut model = CausalModel::default();

        // X -> Y -> Z
        model.add_equation(StructuralEquation::new(
            "Y",
            vec!["X".to_string()],
            vec![0.5],
            0.1,
        ));

        model.add_equation(StructuralEquation::new(
            "Z",
            vec!["Y".to_string()],
            vec![0.6],
            0.1,
        ));

        assert_eq!(model.get_parents("Y"), vec!["X".to_string()]);
        assert_eq!(model.get_children("Y"), vec!["Z".to_string()]);
    }

    #[test]
    fn test_intervention_adjusted_confidence() {
        let mut analyzer = CausalAnalyzer::new();

        // Add price history
        for i in 0..50 {
            analyzer.update_prices("VIX", 20.0 + (i as f64 * 0.05));
            analyzer.update_prices("SPY", 400.0 - (i as f64 * 0.1));
        }

        // VIX causes SPY (negative relationship)
        analyzer.graph.add_relationship(CausalRelationship::new(
            "VIX".to_string(),
            "SPY".to_string(),
            1,
            0.6,
            CausalDirection::Negative,
            0.01,
        ));

        let base_conf = 0.7;
        let adjusted = analyzer.get_intervention_adjusted_confidence("SPY", true, base_conf);

        // Should be adjusted (could be higher or lower depending on VIX state)
        assert!(adjusted >= 0.0 && adjusted <= 1.0);
    }

    #[test]
    fn test_do_rule_display() {
        assert_eq!(format!("{}", DoRule::Rule1InsertDelete), "Rule 1 (insert/delete obs)");
        assert_eq!(format!("{}", DoRule::Rule2ActionExchange), "Rule 2 (action/obs exchange)");
        assert_eq!(format!("{}", DoRule::Rule3ActionDeletion), "Rule 3 (action deletion)");
    }

    #[test]
    fn test_estimation_method_display() {
        assert_eq!(format!("{}", EstimationMethod::BackdoorAdjustment), "backdoor");
        assert_eq!(format!("{}", EstimationMethod::DirectEstimation), "direct");
        assert_eq!(format!("{}", EstimationMethod::NotIdentifiable), "not-identifiable");
    }

    #[test]
    fn test_query_result_significance() {
        let query = CausalQuery::do_query("Y", "X", 1.0);

        // Significant: CI doesn't contain 0
        let significant = QueryResult::new(
            query.clone(),
            0.5,
            (0.2, 0.8),
            EstimationMethod::DirectEstimation,
            Vec::new(),
        );
        assert!(significant.is_significant());

        // Not significant: CI contains 0
        let not_significant = QueryResult::new(
            query,
            0.1,
            (-0.2, 0.4),
            EstimationMethod::DirectEstimation,
            Vec::new(),
        );
        assert!(!not_significant.is_significant());
    }
}
