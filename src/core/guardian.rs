//! Risk Guardian Module
//!
//! Account protection layer. The gatekeeper.
//!
//! Strategy parameters are discovered. Risk limits are ENFORCED.
//! No trade passes without the guardian's approval.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, Datelike};

use crate::core::types::*;

/// Risk configuration (these ARE parameters - but for protection, not strategy)
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Max percentage of account to risk per trade (e.g., 0.005 = 0.5%)
    pub risk_per_trade: Decimal,
    /// Max daily loss percentage before stopping (e.g., 0.02 = 2%)
    pub max_daily_loss: Decimal,
    /// Max floating loss percentage before emergency close (e.g., 0.015 = 1.5%)
    pub max_floating_loss: Decimal,
    /// Max simultaneous positions
    pub max_positions: usize,
    /// Max trades per day
    pub max_trades_per_day: usize,
    /// Cooldown between trades in seconds
    pub trade_cooldown_secs: u64,
    /// Cooldown after a loss in seconds
    pub loss_cooldown_secs: u64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            risk_per_trade: dec!(0.005),      // 0.5%
            max_daily_loss: dec!(0.02),        // 2%
            max_floating_loss: dec!(0.015),    // 1.5%
            max_positions: 1,
            max_trades_per_day: 6,
            trade_cooldown_secs: 1800,         // 30 minutes
            loss_cooldown_secs: 3600,          // 1 hour
        }
    }
}

/// Risk Guardian - protects the account at all costs
pub struct RiskGuardian {
    config: RiskConfig,
    /// Start of day balance for daily P&L tracking
    daily_start_balance: Decimal,
    /// Current daily P&L
    daily_pnl: Decimal,
    /// Trades executed today
    trades_today: usize,
    /// Timestamp of last trade
    last_trade_time: Option<DateTime<Utc>>,
    /// Timestamp of last loss
    last_loss_time: Option<DateTime<Utc>>,
    /// Current day (for reset detection)
    current_day: u32,
    /// Is trading enabled?
    trading_enabled: bool,
    /// Reason for disabling (if disabled)
    disabled_reason: String,
}

impl RiskGuardian {
    /// Create a new risk guardian
    pub fn new(config: RiskConfig) -> Self {
        let now = Utc::now();
        Self {
            config,
            daily_start_balance: Decimal::ZERO,
            daily_pnl: Decimal::ZERO,
            trades_today: 0,
            last_trade_time: None,
            last_loss_time: None,
            current_day: now.day(),
            trading_enabled: true,
            disabled_reason: String::new(),
        }
    }
    
    /// Check and reset daily tracking if it's a new day
    pub fn check_daily_reset(&mut self, current_balance: Decimal) {
        let today = Utc::now().day();
        
        if today != self.current_day {
            self.current_day = today;
            self.daily_start_balance = current_balance;
            self.daily_pnl = Decimal::ZERO;
            self.trades_today = 0;
            self.trading_enabled = true;
            self.disabled_reason.clear();
        }
    }
    
    /// Check if we're allowed to trade
    ///
    /// Returns (can_trade, reason)
    pub fn can_trade(
        &mut self,
        balance: Decimal,
        equity: Decimal,
        open_positions: usize,
    ) -> (bool, String) {
        self.check_daily_reset(balance);
        
        // Update daily P&L
        if self.daily_start_balance > Decimal::ZERO {
            self.daily_pnl = (equity - self.daily_start_balance) / self.daily_start_balance;
        }
        
        // Check 1: Trading enabled?
        if !self.trading_enabled {
            return (false, format!("Trading disabled: {}", self.disabled_reason));
        }
        
        // Check 2: Daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss {
            self.trading_enabled = false;
            self.disabled_reason = format!(
                "Daily loss limit hit ({:.2}%)", 
                self.daily_pnl * dec!(100)
            );
            return (false, self.disabled_reason.clone());
        }
        
        // Check 3: Floating loss limit
        let floating_pnl = if balance > Decimal::ZERO {
            (equity - balance) / balance
        } else {
            Decimal::ZERO
        };
        
        if floating_pnl < -self.config.max_floating_loss {
            return (false, format!(
                "Floating loss too high ({:.2}%)",
                floating_pnl * dec!(100)
            ));
        }
        
        // Check 4: Max positions
        if open_positions >= self.config.max_positions {
            return (false, format!(
                "Max positions reached ({}/{})",
                open_positions, self.config.max_positions
            ));
        }
        
        // Check 5: Max trades per day
        if self.trades_today >= self.config.max_trades_per_day {
            return (false, format!(
                "Max daily trades reached ({}/{})",
                self.trades_today, self.config.max_trades_per_day
            ));
        }
        
        // Check 6: Trade cooldown
        if let Some(last_trade) = self.last_trade_time {
            let elapsed = (Utc::now() - last_trade).num_seconds() as u64;
            if elapsed < self.config.trade_cooldown_secs {
                let remaining = self.config.trade_cooldown_secs - elapsed;
                return (false, format!("Trade cooldown: {}s remaining", remaining));
            }
        }
        
        // Check 7: Loss cooldown
        if let Some(last_loss) = self.last_loss_time {
            let elapsed = (Utc::now() - last_loss).num_seconds() as u64;
            if elapsed < self.config.loss_cooldown_secs {
                let remaining = self.config.loss_cooldown_secs - elapsed;
                return (false, format!("Loss cooldown: {}s remaining", remaining));
            }
        }
        
        (true, "OK".to_string())
    }
    
    /// Calculate position size based on risk
    pub fn calculate_position_size(
        &self,
        balance: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        point_value: Decimal,
        min_lot: Decimal,
        max_lot: Decimal,
        lot_step: Decimal,
    ) -> Decimal {
        let risk_amount = balance * self.config.risk_per_trade;
        let sl_distance = (entry_price - stop_loss).abs();
        
        if sl_distance.is_zero() || point_value.is_zero() {
            return min_lot;
        }
        
        // Position size = Risk / (SL_Distance * Point_Value)
        let position_size = risk_amount / (sl_distance * point_value);
        
        // Round to lot step
        let lots = (position_size / lot_step).round() * lot_step;
        
        // Enforce limits
        lots.max(min_lot).min(max_lot)
    }
    
    /// Record that a trade was opened
    pub fn record_trade_opened(&mut self) {
        self.trades_today += 1;
        self.last_trade_time = Some(Utc::now());
    }
    
    /// Record that a trade was closed
    pub fn record_trade_closed(&mut self, pnl: Decimal) {
        if pnl < Decimal::ZERO {
            self.last_loss_time = Some(Utc::now());
        }
    }
    
    /// Check if emergency close is needed
    pub fn check_emergency_close(&self, balance: Decimal, equity: Decimal) -> (bool, String) {
        if balance.is_zero() {
            return (false, String::new());
        }
        
        let floating_pnl = (equity - balance) / balance;
        
        // Emergency at 150% of normal limit
        let emergency_threshold = -self.config.max_floating_loss * dec!(1.5);
        
        if floating_pnl < emergency_threshold {
            return (true, format!(
                "EMERGENCY: Floating loss {:.2}% exceeds limit",
                floating_pnl * dec!(100)
            ));
        }
        
        (false, String::new())
    }
    
    /// Get current status
    pub fn status(&self) -> String {
        format!(
            "Trading: {} | Daily P&L: {:.2}% | Trades: {}/{}",
            if self.trading_enabled { "Enabled" } else { "DISABLED" },
            self.daily_pnl * dec!(100),
            self.trades_today,
            self.config.max_trades_per_day,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_guardian_creation() {
        let guardian = RiskGuardian::new(RiskConfig::default());
        assert!(guardian.trading_enabled);
    }
    
    #[test]
    fn test_can_trade_basic() {
        let mut guardian = RiskGuardian::new(RiskConfig::default());
        guardian.daily_start_balance = dec!(10000);
        
        let (can_trade, reason) = guardian.can_trade(dec!(10000), dec!(10000), 0);
        assert!(can_trade, "Should be able to trade: {}", reason);
    }
    
    #[test]
    fn test_position_sizing() {
        let guardian = RiskGuardian::new(RiskConfig::default());
        
        let size = guardian.calculate_position_size(
            dec!(10000),   // balance
            dec!(2650),    // entry
            dec!(2640),    // stop loss (10 points away)
            dec!(1),       // point value
            dec!(0.01),    // min lot
            dec!(100),     // max lot
            dec!(0.01),    // lot step
        );
        
        // Risk = 10000 * 0.005 = 50
        // SL distance = 10
        // Size = 50 / 10 = 5 lots
        assert_eq!(size, dec!(5.00));
    }
}
