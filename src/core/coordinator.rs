//! Coordinator Module
//!
//! Central brain that manages all agents and makes portfolio-level decisions.
//!
//! Collects signals from 1000+ agents, ranks by conviction, and executes
//! the best opportunities while respecting portfolio risk limits.

use std::collections::HashMap;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use crate::core::agent::{SymbolAgent, AgentSignal};
use crate::core::types::Candle;
use crate::core::strategy::SignalDirection;
use crate::core::guardian::{RiskGuardian, RiskConfig};

/// Portfolio-level signal after coordinator ranking
#[derive(Debug, Clone)]
pub struct RankedSignal {
    pub rank: usize,
    pub symbol: String,
    pub direction: SignalDirection,
    pub conviction: u8,
    pub price: Decimal,
    pub stop_loss: Decimal,
    pub take_profit: Decimal,
    pub spread: Decimal,
}

/// Coordinator manages all agents
pub struct Coordinator {
    agents: HashMap<String, SymbolAgent>,
    guardian: RiskGuardian,
    max_concurrent_positions: usize,
    active_positions: HashMap<String, u64>,
}

impl Coordinator {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            guardian: RiskGuardian::new(RiskConfig::default()),
            max_concurrent_positions: 3,
            active_positions: HashMap::new(),
        }
    }

    pub fn with_config(max_positions: usize, risk_config: RiskConfig) -> Self {
        Self {
            agents: HashMap::new(),
            guardian: RiskGuardian::new(risk_config),
            max_concurrent_positions: max_positions,
            active_positions: HashMap::new(),
        }
    }

    pub fn add_agent(&mut self, symbol: &str, tick_size: Decimal, is_forex: bool) {
        let agent = SymbolAgent::new(symbol.to_string(), tick_size, is_forex);
        self.agents.insert(symbol.to_string(), agent);
    }

    pub fn remove_agent(&mut self, symbol: &str) {
        self.agents.remove(symbol);
    }

    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    pub fn update_candle(&mut self, symbol: &str, candle: &Candle) {
        if let Some(agent) = self.agents.get_mut(symbol) {
            agent.update(candle);
        }
    }

    pub fn update_tick(&mut self, symbol: &str, bid: Decimal, ask: Decimal) {
        if let Some(agent) = self.agents.get_mut(symbol) {
            agent.update_tick(bid, ask);
        }
    }

    pub fn position_opened(&mut self, symbol: &str, ticket: u64, side: SignalDirection) {
        self.active_positions.insert(symbol.to_string(), ticket);
        if let Some(agent) = self.agents.get_mut(symbol) {
            agent.set_position(true, Some(side));
        }
        self.guardian.record_trade_opened();
    }

    pub fn position_closed(&mut self, symbol: &str, profit: Decimal) {
        self.active_positions.remove(symbol);
        if let Some(agent) = self.agents.get_mut(symbol) {
            agent.set_position(false, None);
        }
        self.guardian.record_trade_closed(profit);
    }

    pub fn active_position_count(&self) -> usize {
        self.active_positions.len()
    }

    pub fn can_open_position(&self) -> bool {
        self.active_positions.len() < self.max_concurrent_positions
    }

    pub fn collect_signals(&mut self, balance: Decimal, equity: Decimal) -> Vec<RankedSignal> {
        if !self.can_open_position() {
            return Vec::new();
        }

        let (can_trade, _reason) = self.guardian.can_trade(
            balance,
            equity,
            self.active_positions.len(),
        );
        if !can_trade {
            return Vec::new();
        }

        let mut signals: Vec<AgentSignal> = self.agents
            .values()
            .filter_map(|agent| agent.analyze())
            .collect();

        signals.sort_by(|a, b| b.signal.conviction.cmp(&a.signal.conviction));

        signals
            .into_iter()
            .enumerate()
            .map(|(rank, s)| RankedSignal {
                rank: rank + 1,
                symbol: s.symbol,
                direction: s.signal.direction,
                conviction: s.signal.conviction,
                price: s.price,
                stop_loss: s.signal.stop_loss,
                take_profit: s.signal.take_profit,
                spread: s.spread,
            })
            .collect()
    }

    pub fn best_signal(&mut self, balance: Decimal, equity: Decimal) -> Option<RankedSignal> {
        self.collect_signals(balance, equity).into_iter().next()
    }

    pub fn guardian_status(&self) -> String {
        self.guardian.status()
    }

    pub fn check_daily_reset(&mut self, balance: Decimal) {
        self.guardian.check_daily_reset(balance);
    }

    pub fn calculate_lots(
        &self,
        balance: Decimal,
        sl_distance: Decimal,
        point_value: Decimal,
    ) -> Decimal {
        self.guardian.calculate_position_size(
            balance,
            sl_distance,
            point_value,
            dec!(0.01),
            dec!(10.0),
            dec!(0.01),
            dec!(1.0),
        )
    }

    pub fn get_observation(&self, symbol: &str) -> Option<crate::core::types::Observation> {
        self.agents.get(symbol).map(|a| a.get_observation())
    }
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::new()
    }
}
