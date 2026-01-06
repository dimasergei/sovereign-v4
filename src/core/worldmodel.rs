//! World Model for Market Dynamics Simulation
//!
//! Implements a mental model of market dynamics for forward planning:
//! - Tracks market state (prices, volatilities, regimes, positions)
//! - Learns transition dynamics from historical data
//! - Simulates future scenarios for decision evaluation
//! - Uses Monte Carlo methods for price distribution forecasting
//!
//! Key insight: Trading decisions should consider not just current state,
//! but how actions affect future states and outcomes.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use super::regime::Regime;
use super::causality::CausalRelationship;

/// Default number of simulation steps
const DEFAULT_SIMULATION_STEPS: usize = 20;

/// Default number of Monte Carlo samples
const DEFAULT_MONTE_CARLO_SAMPLES: usize = 1000;

/// Minimum history for learning transition dynamics
const MIN_HISTORY_FOR_LEARNING: usize = 50;

/// Direction of a simulated position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionDirection {
    Long,
    Short,
    Flat,
}

impl std::fmt::Display for PositionDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionDirection::Long => write!(f, "LONG"),
            PositionDirection::Short => write!(f, "SHORT"),
            PositionDirection::Flat => write!(f, "FLAT"),
        }
    }
}

/// Action that can be taken in the world model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Open a long position
    Buy,
    /// Close a long position
    Sell,
    /// Open a short position
    Short,
    /// Close a short position
    Cover,
    /// No action
    Hold,
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Buy => write!(f, "BUY"),
            Action::Sell => write!(f, "SELL"),
            Action::Short => write!(f, "SHORT"),
            Action::Cover => write!(f, "COVER"),
            Action::Hold => write!(f, "HOLD"),
        }
    }
}

impl Action {
    /// Get all possible actions
    pub fn all() -> [Action; 5] {
        [Action::Buy, Action::Sell, Action::Short, Action::Cover, Action::Hold]
    }

    /// Check if action is valid given current position
    pub fn is_valid(&self, current_position: PositionDirection) -> bool {
        match (self, current_position) {
            (Action::Buy, PositionDirection::Flat) => true,
            (Action::Buy, PositionDirection::Short) => false, // Must cover first
            (Action::Sell, PositionDirection::Long) => true,
            (Action::Sell, PositionDirection::Flat) => false,
            (Action::Sell, PositionDirection::Short) => false,
            (Action::Short, PositionDirection::Flat) => true,
            (Action::Short, PositionDirection::Long) => false, // Must sell first
            (Action::Cover, PositionDirection::Short) => true,
            (Action::Cover, PositionDirection::Long) => false,
            (Action::Cover, PositionDirection::Flat) => false,
            (Action::Hold, _) => true,
            _ => false,
        }
    }

    /// Get valid actions for current position
    pub fn valid_actions(position: PositionDirection) -> Vec<Action> {
        Action::all()
            .into_iter()
            .filter(|a| a.is_valid(position))
            .collect()
    }
}

/// A position in the simulated world
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimPosition {
    /// Symbol being traded
    pub symbol: String,
    /// Position direction
    pub direction: PositionDirection,
    /// Entry price
    pub entry_price: f64,
    /// Position size (shares/contracts)
    pub size: f64,
    /// Unrealized PnL at current price
    pub unrealized_pnl: f64,
}

impl SimPosition {
    /// Create a new flat position
    pub fn flat(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            direction: PositionDirection::Flat,
            entry_price: 0.0,
            size: 0.0,
            unrealized_pnl: 0.0,
        }
    }

    /// Create a new long position
    pub fn long(symbol: &str, entry_price: f64, size: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            direction: PositionDirection::Long,
            entry_price,
            size,
            unrealized_pnl: 0.0,
        }
    }

    /// Create a new short position
    pub fn short(symbol: &str, entry_price: f64, size: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            direction: PositionDirection::Short,
            entry_price,
            size,
            unrealized_pnl: 0.0,
        }
    }

    /// Update unrealized PnL given current price
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = match self.direction {
            PositionDirection::Long => (current_price - self.entry_price) * self.size,
            PositionDirection::Short => (self.entry_price - current_price) * self.size,
            PositionDirection::Flat => 0.0,
        };
    }

    /// Close the position and return realized PnL
    pub fn close(&mut self, exit_price: f64) -> f64 {
        let pnl = match self.direction {
            PositionDirection::Long => (exit_price - self.entry_price) * self.size,
            PositionDirection::Short => (self.entry_price - exit_price) * self.size,
            PositionDirection::Flat => 0.0,
        };
        self.direction = PositionDirection::Flat;
        self.entry_price = 0.0;
        self.size = 0.0;
        self.unrealized_pnl = 0.0;
        pnl
    }
}

/// Complete state of the simulated market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Current prices by symbol
    pub prices: HashMap<String, f64>,
    /// Recent returns by symbol (last N bars)
    pub returns: HashMap<String, Vec<f64>>,
    /// Current volatilities by symbol (realized vol)
    pub volatilities: HashMap<String, f64>,
    /// Current regimes by symbol
    pub regimes: HashMap<String, Regime>,
    /// Current positions by symbol
    pub positions: HashMap<String, SimPosition>,
    /// Total portfolio equity
    pub equity: f64,
    /// Timestamp of this state
    pub timestamp: DateTime<Utc>,
    /// Step number in simulation
    pub step: usize,
}

impl MarketState {
    /// Create a new empty market state
    pub fn new(equity: f64) -> Self {
        Self {
            prices: HashMap::new(),
            returns: HashMap::new(),
            volatilities: HashMap::new(),
            regimes: HashMap::new(),
            positions: HashMap::new(),
            equity,
            timestamp: Utc::now(),
            step: 0,
        }
    }

    /// Create initial state from current market data
    pub fn from_current(
        prices: HashMap<String, Decimal>,
        regimes: HashMap<String, Regime>,
        equity: f64,
    ) -> Self {
        let prices_f64: HashMap<String, f64> = prices
            .into_iter()
            .map(|(k, v)| (k, v.to_f64().unwrap_or(0.0)))
            .collect();

        let positions: HashMap<String, SimPosition> = prices_f64
            .keys()
            .map(|s| (s.clone(), SimPosition::flat(s)))
            .collect();

        Self {
            prices: prices_f64,
            returns: HashMap::new(),
            volatilities: HashMap::new(),
            regimes,
            positions,
            equity,
            timestamp: Utc::now(),
            step: 0,
        }
    }

    /// Get total portfolio value (equity + unrealized PnL)
    pub fn total_value(&self) -> f64 {
        self.equity + self.positions.values().map(|p| p.unrealized_pnl).sum::<f64>()
    }

    /// Get price for a symbol
    pub fn get_price(&self, symbol: &str) -> Option<f64> {
        self.prices.get(symbol).copied()
    }

    /// Get position for a symbol
    pub fn get_position(&self, symbol: &str) -> Option<&SimPosition> {
        self.positions.get(symbol)
    }

    /// Get position direction for a symbol
    pub fn get_position_direction(&self, symbol: &str) -> PositionDirection {
        self.positions
            .get(symbol)
            .map(|p| p.direction)
            .unwrap_or(PositionDirection::Flat)
    }

    /// Get regime for a symbol
    pub fn get_regime(&self, symbol: &str) -> Option<Regime> {
        self.regimes.get(symbol).copied()
    }

    /// Clone state with incremented step
    pub fn next_step(&self) -> Self {
        let mut state = self.clone();
        state.step += 1;
        state.timestamp = Utc::now();
        state
    }
}

/// Model of market transition dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionModel {
    /// Mean returns by symbol (drift)
    pub mean_returns: HashMap<String, f64>,
    /// Volatilities by symbol (diffusion)
    pub volatilities: HashMap<String, f64>,
    /// Correlation matrix (symbol pairs)
    pub correlations: HashMap<(String, String), f64>,
    /// Causal effects from CausalAnalyzer
    pub causal_effects: Vec<CausalRelationship>,
    /// Regime transition probabilities (from regime -> to regime)
    pub regime_transitions: [[f64; 4]; 4],
    /// Number of samples used to estimate
    pub sample_count: usize,
}

impl Default for TransitionModel {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionModel {
    /// Create a new transition model with default parameters
    pub fn new() -> Self {
        // Default regime transition matrix (equal probabilities)
        let regime_transitions = [
            [0.7, 0.1, 0.1, 0.1],  // From TrendingUp
            [0.1, 0.7, 0.1, 0.1],  // From TrendingDown
            [0.15, 0.15, 0.6, 0.1], // From Ranging
            [0.1, 0.1, 0.1, 0.7],  // From Volatile
        ];

        Self {
            mean_returns: HashMap::new(),
            volatilities: HashMap::new(),
            correlations: HashMap::new(),
            causal_effects: Vec::new(),
            regime_transitions,
            sample_count: 0,
        }
    }

    /// Learn transition dynamics from historical returns
    pub fn learn_from_returns(&mut self, symbol: &str, returns: &[f64]) {
        if returns.len() < MIN_HISTORY_FOR_LEARNING {
            return;
        }

        // Calculate mean return
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        self.mean_returns.insert(symbol.to_string(), mean);

        // Calculate volatility (standard deviation)
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;
        let volatility = variance.sqrt();
        self.volatilities.insert(symbol.to_string(), volatility);

        self.sample_count = returns.len();
    }

    /// Learn correlation between two symbols
    pub fn learn_correlation(&mut self, symbol1: &str, symbol2: &str, returns1: &[f64], returns2: &[f64]) {
        let n = returns1.len().min(returns2.len());
        if n < MIN_HISTORY_FOR_LEARNING {
            return;
        }

        let mean1: f64 = returns1[..n].iter().sum::<f64>() / n as f64;
        let mean2: f64 = returns2[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..n {
            let d1 = returns1[i] - mean1;
            let d2 = returns2[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let correlation = if var1 > 0.0 && var2 > 0.0 {
            cov / (var1.sqrt() * var2.sqrt())
        } else {
            0.0
        };

        self.correlations.insert(
            (symbol1.to_string(), symbol2.to_string()),
            correlation,
        );
        self.correlations.insert(
            (symbol2.to_string(), symbol1.to_string()),
            correlation,
        );
    }

    /// Learn regime transition probabilities
    pub fn learn_regime_transitions(&mut self, regime_history: &[Regime]) {
        if regime_history.len() < 2 {
            return;
        }

        // Count transitions
        let mut counts = [[0u32; 4]; 4];
        let mut row_totals = [0u32; 4];

        for i in 0..regime_history.len() - 1 {
            let from = regime_history[i].index();
            let to = regime_history[i + 1].index();
            counts[from][to] += 1;
            row_totals[from] += 1;
        }

        // Convert to probabilities
        for from in 0..4 {
            if row_totals[from] > 0 {
                for to in 0..4 {
                    self.regime_transitions[from][to] =
                        counts[from][to] as f64 / row_totals[from] as f64;
                }
            }
        }
    }

    /// Set causal effects from CausalAnalyzer
    pub fn set_causal_effects(&mut self, effects: Vec<CausalRelationship>) {
        self.causal_effects = effects;
    }

    /// Get mean return for a symbol
    pub fn get_mean_return(&self, symbol: &str) -> f64 {
        self.mean_returns.get(symbol).copied().unwrap_or(0.0)
    }

    /// Get volatility for a symbol
    pub fn get_volatility(&self, symbol: &str) -> f64 {
        self.volatilities.get(symbol).copied().unwrap_or(0.02) // Default 2% daily vol
    }

    /// Get correlation between two symbols
    pub fn get_correlation(&self, symbol1: &str, symbol2: &str) -> f64 {
        self.correlations
            .get(&(symbol1.to_string(), symbol2.to_string()))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get next regime probability
    pub fn get_next_regime_prob(&self, current_regime: Regime) -> [f64; 4] {
        self.regime_transitions[current_regime.index()]
    }

    /// Sample next regime
    pub fn sample_next_regime(&self, current_regime: Regime, random: f64) -> Regime {
        let probs = self.get_next_regime_prob(current_regime);
        let mut cumsum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if random < cumsum {
                return Regime::from_index(i);
            }
        }
        Regime::Volatile // Fallback
    }

    /// Get causal adjustment factor for a symbol
    pub fn get_causal_adjustment(&self, symbol: &str, factor_returns: &HashMap<String, f64>) -> f64 {
        let mut adjustment = 0.0;

        for rel in &self.causal_effects {
            if rel.target == symbol {
                if let Some(&factor_return) = factor_returns.get(&rel.source) {
                    // Apply causal effect with direction and strength
                    let effect = match rel.direction {
                        super::causality::CausalDirection::Positive => {
                            factor_return * rel.strength
                        }
                        super::causality::CausalDirection::Negative => {
                            -factor_return * rel.strength
                        }
                    };
                    adjustment += effect * rel.confidence;
                }
            }
        }

        adjustment
    }
}

/// Result of a simulation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Initial market state
    pub initial_state: MarketState,
    /// Final market state
    pub final_state: MarketState,
    /// Actions taken during simulation
    pub actions_taken: Vec<(usize, String, Action)>, // (step, symbol, action)
    /// Total realized + unrealized PnL
    pub total_pnl: f64,
    /// Maximum drawdown during simulation
    pub max_drawdown: f64,
    /// Number of steps simulated
    pub steps_simulated: usize,
    /// Final equity
    pub final_equity: f64,
    /// Sharpe ratio of simulation path
    pub sharpe_ratio: f64,
}

impl SimulationResult {
    /// Create a new simulation result
    pub fn new(initial_state: MarketState, final_state: MarketState) -> Self {
        let total_pnl = final_state.total_value() - initial_state.total_value();
        Self {
            initial_state,
            final_state: final_state.clone(),
            actions_taken: Vec::new(),
            total_pnl,
            max_drawdown: 0.0,
            steps_simulated: final_state.step,
            final_equity: final_state.equity,
            sharpe_ratio: 0.0,
        }
    }

    /// Check if simulation was profitable
    pub fn is_profitable(&self) -> bool {
        self.total_pnl > 0.0
    }

    /// Get return on equity
    pub fn return_on_equity(&self) -> f64 {
        if self.initial_state.equity > 0.0 {
            self.total_pnl / self.initial_state.equity
        } else {
            0.0
        }
    }
}

/// Price forecast distribution from Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct PriceForecast {
    /// Symbol being forecasted
    pub symbol: String,
    /// Number of steps ahead
    pub steps_ahead: usize,
    /// Mean forecast price
    pub mean: f64,
    /// Median forecast price
    pub median: f64,
    /// 5th percentile (downside)
    pub percentile_5: f64,
    /// 25th percentile
    pub percentile_25: f64,
    /// 75th percentile
    pub percentile_75: f64,
    /// 95th percentile (upside)
    pub percentile_95: f64,
    /// Standard deviation of forecast
    pub std_dev: f64,
    /// Number of samples used
    pub samples: usize,
}

impl PriceForecast {
    /// Get expected return
    pub fn expected_return(&self, current_price: f64) -> f64 {
        if current_price > 0.0 {
            (self.mean - current_price) / current_price
        } else {
            0.0
        }
    }

    /// Get probability of price above threshold
    pub fn prob_above(&self, threshold: f64) -> f64 {
        // Assuming approximately normal distribution
        if self.std_dev > 0.0 {
            let z = (threshold - self.mean) / self.std_dev;
            1.0 - normal_cdf(z)
        } else {
            if self.mean > threshold { 1.0 } else { 0.0 }
        }
    }

    /// Get probability of price below threshold
    pub fn prob_below(&self, threshold: f64) -> f64 {
        1.0 - self.prob_above(threshold)
    }

    /// Get expected profit for long position
    pub fn expected_long_profit(&self, entry_price: f64, size: f64) -> f64 {
        (self.mean - entry_price) * size
    }

    /// Get expected profit for short position
    pub fn expected_short_profit(&self, entry_price: f64, size: f64) -> f64 {
        (entry_price - self.mean) * size
    }
}

/// World Model for forward planning and simulation
#[derive(Clone, Serialize, Deserialize)]
pub struct WorldModel {
    /// Transition dynamics model
    pub transition_model: TransitionModel,
    /// Current market state
    pub current_state: MarketState,
    /// Historical states for learning
    #[serde(skip)]
    history: Vec<MarketState>,
    /// Number of simulation steps
    pub simulation_steps: usize,
    /// Random seed for reproducibility
    seed: u64,
}

impl WorldModel {
    /// Create a new world model
    pub fn new(initial_equity: f64) -> Self {
        Self {
            transition_model: TransitionModel::new(),
            current_state: MarketState::new(initial_equity),
            history: Vec::new(),
            simulation_steps: DEFAULT_SIMULATION_STEPS,
            seed: 42,
        }
    }

    /// Create with custom simulation steps
    pub fn with_steps(initial_equity: f64, steps: usize) -> Self {
        let mut model = Self::new(initial_equity);
        model.simulation_steps = steps;
        model
    }

    /// Update current state with new market data
    pub fn update_state(
        &mut self,
        prices: HashMap<String, Decimal>,
        regimes: HashMap<String, Regime>,
    ) {
        // Store previous state in history
        self.history.push(self.current_state.clone());

        // Calculate returns from price changes
        let mut returns: HashMap<String, Vec<f64>> = HashMap::new();
        for (symbol, new_price) in &prices {
            let new_price_f64 = new_price.to_f64().unwrap_or(0.0);
            if let Some(old_price) = self.current_state.get_price(symbol) {
                if old_price > 0.0 {
                    let ret = (new_price_f64 - old_price) / old_price;
                    let mut symbol_returns = self.current_state.returns
                        .get(symbol)
                        .cloned()
                        .unwrap_or_default();
                    symbol_returns.push(ret);
                    // Keep last 100 returns
                    if symbol_returns.len() > 100 {
                        symbol_returns.remove(0);
                    }
                    returns.insert(symbol.clone(), symbol_returns);
                }
            }
        }

        // Update prices
        for (symbol, price) in &prices {
            self.current_state.prices.insert(
                symbol.clone(),
                price.to_f64().unwrap_or(0.0),
            );
        }

        // Update returns
        self.current_state.returns = returns;

        // Update regimes
        self.current_state.regimes = regimes;

        // Update position PnLs
        for (symbol, position) in &mut self.current_state.positions {
            if let Some(&price) = self.current_state.prices.get(symbol) {
                position.update_pnl(price);
            }
        }

        // Learn transition dynamics from returns
        for (symbol, rets) in &self.current_state.returns {
            self.transition_model.learn_from_returns(symbol, rets);
        }

        // Update timestamp
        self.current_state.timestamp = Utc::now();

        // Limit history size
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    /// Simulate one step forward
    fn simulate_step(
        &self,
        state: &MarketState,
        action: Option<(&str, Action)>,
        random_values: &HashMap<String, f64>,
    ) -> MarketState {
        let mut new_state = state.next_step();

        // Simulate price changes using GBM
        for (symbol, &current_price) in &state.prices {
            let mean_return = self.transition_model.get_mean_return(symbol);
            let volatility = self.transition_model.get_volatility(symbol);
            let random = random_values.get(symbol).copied().unwrap_or(0.0);

            // Geometric Brownian Motion: S(t+1) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            // For daily, dt = 1
            let drift = mean_return - 0.5 * volatility.powi(2);
            let diffusion = volatility * random;
            let new_price = current_price * (drift + diffusion).exp();

            new_state.prices.insert(symbol.clone(), new_price);
        }

        // Apply action if provided
        if let Some((symbol, action)) = action {
            if let Some(position) = new_state.positions.get_mut(symbol) {
                if let Some(&price) = new_state.prices.get(symbol) {
                    let position_size = new_state.equity * 0.1 / price; // 10% of equity

                    match action {
                        Action::Buy => {
                            if position.direction == PositionDirection::Flat {
                                *position = SimPosition::long(symbol, price, position_size);
                            }
                        }
                        Action::Sell => {
                            if position.direction == PositionDirection::Long {
                                let pnl = position.close(price);
                                new_state.equity += pnl;
                            }
                        }
                        Action::Short => {
                            if position.direction == PositionDirection::Flat {
                                *position = SimPosition::short(symbol, price, position_size);
                            }
                        }
                        Action::Cover => {
                            if position.direction == PositionDirection::Short {
                                let pnl = position.close(price);
                                new_state.equity += pnl;
                            }
                        }
                        Action::Hold => {}
                    }
                }
            }
        }

        // Update position PnLs
        for (symbol, position) in &mut new_state.positions {
            if let Some(&price) = new_state.prices.get(symbol) {
                position.update_pnl(price);
            }
        }

        // Simulate regime transitions
        for (symbol, &regime) in &state.regimes {
            let random = random_values.get(symbol).copied().unwrap_or(0.5);
            let new_regime = self.transition_model.sample_next_regime(regime, random.abs());
            new_state.regimes.insert(symbol.clone(), new_regime);
        }

        new_state
    }

    /// Run a full simulation with given action policy
    pub fn simulate<F>(&self, policy: F) -> SimulationResult
    where
        F: Fn(&MarketState) -> Vec<(String, Action)>,
    {
        let mut state = self.current_state.clone();
        let initial_state = state.clone();
        let mut actions_taken = Vec::new();
        let mut equity_history = vec![state.total_value()];
        let mut peak_equity = state.total_value();
        let mut max_drawdown = 0.0;

        // Simple pseudo-random for simulation
        let mut rng_state = self.seed;

        for step in 0..self.simulation_steps {
            // Get actions from policy
            let actions = policy(&state);

            // Generate random values for each symbol
            let mut random_values: HashMap<String, f64> = HashMap::new();
            for symbol in state.prices.keys() {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let random = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                random_values.insert(symbol.clone(), random);
            }

            // Apply first action (simplified - could handle multiple)
            let action = actions.first().cloned();
            if let Some((ref sym, act)) = action {
                actions_taken.push((step, sym.clone(), act));
                state = self.simulate_step(&state, Some((sym, act)), &random_values);
            } else {
                state = self.simulate_step(&state, None, &random_values);
            }

            // Track drawdown
            let current_value = state.total_value();
            equity_history.push(current_value);
            if current_value > peak_equity {
                peak_equity = current_value;
            }
            let drawdown = (peak_equity - current_value) / peak_equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calculate Sharpe ratio
        let returns: Vec<f64> = equity_history
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let sharpe = if returns.len() > 1 {
            let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            if variance > 0.0 {
                mean_ret / variance.sqrt() * (252.0_f64).sqrt() // Annualized
            } else {
                0.0
            }
        } else {
            0.0
        };

        let mut result = SimulationResult::new(initial_state, state);
        result.actions_taken = actions_taken;
        result.max_drawdown = max_drawdown;
        result.sharpe_ratio = sharpe;

        result
    }

    /// Evaluate an action using Monte Carlo simulation
    pub fn evaluate_action(
        &self,
        symbol: &str,
        action: Action,
        samples: usize,
    ) -> f64 {
        let samples = samples.max(10);
        let mut total_pnl = 0.0;

        for i in 0..samples {
            let seed = self.seed.wrapping_add(i as u64);
            let mut model_copy = self.clone();
            model_copy.seed = seed;

            // Simple policy: take the action, then hold
            let action_symbol = symbol.to_string();
            let action_to_take = action;
            let policy = move |state: &MarketState| {
                if state.step == 0 {
                    vec![(action_symbol.clone(), action_to_take)]
                } else {
                    vec![]
                }
            };

            let result = model_copy.simulate(policy);
            total_pnl += result.total_pnl;
        }

        total_pnl / samples as f64
    }

    /// Get best action for a symbol using evaluation
    pub fn get_best_action(&self, symbol: &str) -> (Action, f64) {
        let position = self.current_state.get_position_direction(symbol);
        let valid_actions = Action::valid_actions(position);

        let mut best_action = Action::Hold;
        let mut best_value = f64::NEG_INFINITY;

        for action in valid_actions {
            let value = self.evaluate_action(symbol, action, 100);
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        (best_action, best_value)
    }

    /// Monte Carlo price forecast
    pub fn forecast_price(
        &self,
        symbol: &str,
        steps_ahead: usize,
        samples: usize,
    ) -> Option<PriceForecast> {
        let current_price = self.current_state.get_price(symbol)?;
        let samples = samples.max(100);
        let mut final_prices: Vec<f64> = Vec::with_capacity(samples);

        let mean_return = self.transition_model.get_mean_return(symbol);
        let volatility = self.transition_model.get_volatility(symbol);

        for i in 0..samples {
            let mut price = current_price;
            let mut rng_state = self.seed.wrapping_add(i as u64);

            for _ in 0..steps_ahead {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;

                // Box-Muller transform for normal distribution
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

                let drift = mean_return - 0.5 * volatility.powi(2);
                let diffusion = volatility * z;
                price *= (drift + diffusion).exp();
            }

            final_prices.push(price);
        }

        // Sort for percentiles
        final_prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = final_prices.iter().sum::<f64>() / samples as f64;
        let median = final_prices[samples / 2];
        let variance: f64 = final_prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
            / (samples - 1) as f64;
        let std_dev = variance.sqrt();

        Some(PriceForecast {
            symbol: symbol.to_string(),
            steps_ahead,
            mean,
            median,
            percentile_5: final_prices[(samples as f64 * 0.05) as usize],
            percentile_25: final_prices[(samples as f64 * 0.25) as usize],
            percentile_75: final_prices[(samples as f64 * 0.75) as usize],
            percentile_95: final_prices[(samples as f64 * 0.95) as usize],
            std_dev,
            samples,
        })
    }

    /// Get expected value for taking a position
    pub fn get_expected_value(&self, symbol: &str, direction: PositionDirection, horizon: usize) -> f64 {
        let forecast = match self.forecast_price(symbol, horizon, DEFAULT_MONTE_CARLO_SAMPLES) {
            Some(f) => f,
            None => return 0.0,
        };

        let current_price = match self.current_state.get_price(symbol) {
            Some(p) => p,
            None => return 0.0,
        };

        match direction {
            PositionDirection::Long => forecast.mean - current_price,
            PositionDirection::Short => current_price - forecast.mean,
            PositionDirection::Flat => 0.0,
        }
    }

    /// Apply causal effects from CausalAnalyzer
    pub fn set_causal_effects(&mut self, effects: Vec<CausalRelationship>) {
        self.transition_model.set_causal_effects(effects);
        info!("WorldModel: Applied {} causal effects", self.transition_model.causal_effects.len());
    }

    /// Get confidence adjustment based on world model predictions
    pub fn get_prediction_confidence(&self, symbol: &str, direction: PositionDirection) -> f64 {
        let expected = self.get_expected_value(symbol, direction, 5);
        let current_price = self.current_state.get_price(symbol).unwrap_or(100.0);

        // Normalize expected value by current price
        let normalized = expected / current_price;

        // Convert to confidence: positive expectation = higher confidence
        let confidence = (normalized * 100.0).tanh();

        // Scale to 0.5-1.5 range
        0.5 + confidence * 0.5
    }

    /// Get symbols in the world model
    pub fn get_symbols(&self) -> Vec<String> {
        self.current_state.prices.keys().cloned().collect()
    }

    /// Get current equity
    pub fn get_equity(&self) -> f64 {
        self.current_state.equity
    }

    /// Update equity after trade
    pub fn update_equity(&mut self, new_equity: f64) {
        self.current_state.equity = new_equity;
    }

    /// Get forecast accuracy (placeholder - would track historical forecasts)
    pub fn forecast_accuracy(&self) -> f64 {
        // In a full implementation, this would track:
        // - Forecasts made
        // - Actual outcomes
        // - Percentage of correct direction predictions
        // For now, return a default based on simulation performance
        0.52 // Slightly better than random
    }
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_validity() {
        assert!(Action::Buy.is_valid(PositionDirection::Flat));
        assert!(!Action::Buy.is_valid(PositionDirection::Long));
        assert!(Action::Sell.is_valid(PositionDirection::Long));
        assert!(!Action::Sell.is_valid(PositionDirection::Flat));
        assert!(Action::Short.is_valid(PositionDirection::Flat));
        assert!(Action::Cover.is_valid(PositionDirection::Short));
        assert!(Action::Hold.is_valid(PositionDirection::Flat));
        assert!(Action::Hold.is_valid(PositionDirection::Long));
    }

    #[test]
    fn test_valid_actions() {
        let actions = Action::valid_actions(PositionDirection::Flat);
        assert!(actions.contains(&Action::Buy));
        assert!(actions.contains(&Action::Short));
        assert!(actions.contains(&Action::Hold));
        assert!(!actions.contains(&Action::Sell));
        assert!(!actions.contains(&Action::Cover));
    }

    #[test]
    fn test_sim_position_long() {
        let mut pos = SimPosition::long("AAPL", 100.0, 10.0);
        pos.update_pnl(110.0);
        assert!((pos.unrealized_pnl - 100.0).abs() < 0.01);

        let pnl = pos.close(120.0);
        assert!((pnl - 200.0).abs() < 0.01);
        assert_eq!(pos.direction, PositionDirection::Flat);
    }

    #[test]
    fn test_sim_position_short() {
        let mut pos = SimPosition::short("AAPL", 100.0, 10.0);
        pos.update_pnl(90.0);
        assert!((pos.unrealized_pnl - 100.0).abs() < 0.01);

        let pnl = pos.close(80.0);
        assert!((pnl - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_market_state() {
        let mut state = MarketState::new(100000.0);
        state.prices.insert("AAPL".to_string(), 150.0);
        state.positions.insert("AAPL".to_string(), SimPosition::flat("AAPL"));

        assert_eq!(state.get_price("AAPL"), Some(150.0));
        assert_eq!(state.get_position_direction("AAPL"), PositionDirection::Flat);
        assert!((state.total_value() - 100000.0).abs() < 0.01);
    }

    #[test]
    fn test_transition_model_learning() {
        let mut model = TransitionModel::new();
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01).sin() * 0.02).collect();

        model.learn_from_returns("AAPL", &returns);

        assert!(model.mean_returns.contains_key("AAPL"));
        assert!(model.volatilities.contains_key("AAPL"));
    }

    #[test]
    fn test_transition_model_correlation() {
        let mut model = TransitionModel::new();
        let returns1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
        let returns2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.015).collect();

        model.learn_correlation("AAPL", "MSFT", &returns1, &returns2);

        let corr = model.get_correlation("AAPL", "MSFT");
        assert!(corr > 0.9); // Should be highly correlated
    }

    #[test]
    fn test_regime_transitions() {
        let mut model = TransitionModel::new();
        let history = vec![
            Regime::TrendingUp, Regime::TrendingUp, Regime::TrendingUp,
            Regime::Ranging, Regime::Ranging,
            Regime::TrendingDown,
        ];

        model.learn_regime_transitions(&history);

        let probs = model.get_next_regime_prob(Regime::TrendingUp);
        assert!(probs[0] > 0.5); // High probability to stay in TrendingUp
    }

    #[test]
    fn test_world_model_creation() {
        let model = WorldModel::new(100000.0);
        assert!((model.current_state.equity - 100000.0).abs() < 0.01);
        assert_eq!(model.simulation_steps, DEFAULT_SIMULATION_STEPS);
    }

    #[test]
    fn test_world_model_update() {
        let mut model = WorldModel::new(100000.0);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::new(15000, 2)); // 150.00

        let mut regimes = HashMap::new();
        regimes.insert("AAPL".to_string(), Regime::TrendingUp);

        model.update_state(prices.clone(), regimes.clone());

        assert_eq!(model.current_state.get_price("AAPL"), Some(150.0));
        assert_eq!(model.current_state.get_regime("AAPL"), Some(Regime::TrendingUp));
    }

    #[test]
    fn test_world_model_simulation() {
        let mut model = WorldModel::with_steps(100000.0, 5);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::new(15000, 2));

        let mut regimes = HashMap::new();
        regimes.insert("AAPL".to_string(), Regime::TrendingUp);

        model.update_state(prices, regimes);

        // Simple hold policy
        let result = model.simulate(|_| vec![]);

        assert_eq!(result.steps_simulated, 5);
        assert!(result.max_drawdown >= 0.0);
    }

    #[test]
    fn test_price_forecast() {
        let mut model = WorldModel::new(100000.0);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::new(15000, 2));

        let mut regimes = HashMap::new();
        regimes.insert("AAPL".to_string(), Regime::TrendingUp);

        model.update_state(prices, regimes);

        // Add some return history
        for i in 0..60 {
            let price = Decimal::new(15000 + (i * 10) as i64, 2);
            let mut prices = HashMap::new();
            prices.insert("AAPL".to_string(), price);
            let mut regimes = HashMap::new();
            regimes.insert("AAPL".to_string(), Regime::TrendingUp);
            model.update_state(prices, regimes);
        }

        let forecast = model.forecast_price("AAPL", 5, 100);
        assert!(forecast.is_some());

        let f = forecast.unwrap();
        assert_eq!(f.symbol, "AAPL");
        assert_eq!(f.steps_ahead, 5);
        assert!(f.percentile_5 < f.median);
        assert!(f.median < f.percentile_95);
    }

    #[test]
    fn test_action_evaluation() {
        let mut model = WorldModel::with_steps(100000.0, 5);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::new(15000, 2));

        let mut regimes = HashMap::new();
        regimes.insert("AAPL".to_string(), Regime::TrendingUp);

        model.update_state(prices, regimes);

        let hold_value = model.evaluate_action("AAPL", Action::Hold, 10);
        // Value should be defined (though may be positive or negative)
        assert!(hold_value.is_finite());
    }

    #[test]
    fn test_simulation_result() {
        let state1 = MarketState::new(100000.0);
        let mut state2 = MarketState::new(110000.0);
        state2.step = 10;

        let result = SimulationResult::new(state1, state2);

        assert!(result.is_profitable());
        assert!((result.total_pnl - 10000.0).abs() < 0.01);
        assert!((result.return_on_equity() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        // Standard normal CDF at 0 should be 0.5
        let cdf_0 = normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.01);

        // CDF at large positive should be close to 1
        let cdf_3 = normal_cdf(3.0);
        assert!(cdf_3 > 0.99);

        // CDF at large negative should be close to 0
        let cdf_neg3 = normal_cdf(-3.0);
        assert!(cdf_neg3 < 0.01);
    }

    #[test]
    fn test_price_forecast_probabilities() {
        let forecast = PriceForecast {
            symbol: "AAPL".to_string(),
            steps_ahead: 5,
            mean: 150.0,
            median: 149.0,
            percentile_5: 140.0,
            percentile_25: 145.0,
            percentile_75: 155.0,
            percentile_95: 160.0,
            std_dev: 8.0,
            samples: 1000,
        };

        let expected_return = forecast.expected_return(145.0);
        assert!((expected_return - 0.0345).abs() < 0.01);

        let prob_above_mean = forecast.prob_above(150.0);
        assert!((prob_above_mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_expected_value() {
        let mut model = WorldModel::new(100000.0);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), Decimal::new(15000, 2));

        let mut regimes = HashMap::new();
        regimes.insert("AAPL".to_string(), Regime::TrendingUp);

        model.update_state(prices, regimes);

        let ev_flat = model.get_expected_value("AAPL", PositionDirection::Flat, 5);
        assert!((ev_flat - 0.0).abs() < 0.01);
    }
}
