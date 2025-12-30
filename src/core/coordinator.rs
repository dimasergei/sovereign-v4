//! Coordinator Module
//!
//! Central brain that manages all agents and allocates capital.
//!
//! Receives signals from all agents, aggregates them, and executes trades.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::agent::Agent;
use crate::core::types::*;

/// Central coordinator managing all agents
pub struct Coordinator {
    /// All active agents
    agents: HashMap<String, Arc<RwLock<Box<dyn Agent>>>>,
    /// Current portfolio state
    positions: HashMap<String, Position>,
    /// Account info
    account: AccountInfo,
}

impl Coordinator {
    /// Create a new coordinator
    pub fn new(initial_balance: rust_decimal::Decimal) -> Self {
        Self {
            agents: HashMap::new(),
            positions: HashMap::new(),
            account: AccountInfo {
                balance: initial_balance,
                equity: initial_balance,
                margin_used: rust_decimal::Decimal::ZERO,
                margin_free: initial_balance,
                profit: rust_decimal::Decimal::ZERO,
            },
        }
    }
    
    /// Add an agent to the pool
    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        let symbol = agent.symbol().to_string();
        self.agents.insert(symbol, Arc::new(RwLock::new(agent)));
    }
    
    /// Get number of active agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
    
    /// Process signals from all agents and decide on trades
    pub async fn process_signals(&self) -> Vec<Decision> {
        let mut decisions = Vec::new();
        
        for (_symbol, agent) in &self.agents {
            let agent = agent.read().await;
            // TODO: Get current price, make decision
            let decision = agent.decide(rust_decimal::Decimal::ZERO);
            
            if decision.conviction >= 50 {
                decisions.push(decision);
            }
        }
        
        decisions
    }
    
    /// Run the coordinator loop
    pub async fn run(&mut self) {
        loop {
            // TODO: 
            // 1. Fetch market data for all symbols
            // 2. Update all agents
            // 3. Collect signals
            // 4. Allocate capital
            // 5. Execute trades
            // 6. Monitor positions
            
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_coordinator_creation() {
        let coordinator = Coordinator::new(dec!(10000));
        assert_eq!(coordinator.agent_count(), 0);
    }
}
