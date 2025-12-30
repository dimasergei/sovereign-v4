//! Trading Strategy - The Brain
//!
//! Converts market observations into trading decisions.
//! No parameters, no thresholds - pure pattern recognition.

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

use crate::core::types::{Trend, Momentum, VolumeState, Observation};

/// Trading signal with conviction level
#[derive(Debug, Clone)]
pub struct TradeSignal {
    pub direction: SignalDirection,
    pub conviction: u8,          // 0-100
    pub stop_loss: Decimal,
    pub take_profit: Decimal,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignalDirection {
    Buy,
    Sell,
    Hold,
}

/// Strategy configuration
pub struct Strategy {
    min_conviction: u8,
    risk_reward_ratio: Decimal,
}

impl Default for Strategy {
    fn default() -> Self {
        Self {
            min_conviction: 60,
            risk_reward_ratio: dec!(2.0), // 1:2 risk/reward
        }
    }
}

impl Strategy {
    pub fn new(min_conviction: u8, risk_reward_ratio: Decimal) -> Self {
        Self { min_conviction, risk_reward_ratio }
    }

    /// Analyze observation and generate signal
    pub fn analyze(&self, obs: &Observation, current_price: Decimal) -> TradeSignal {
        let mut conviction: i32 = 0;
        let mut direction = SignalDirection::Hold;
        let mut reasons = Vec::new();

        // ═══════════════════════════════════════════════════════════
        // TREND ANALYSIS (weight: 30 points)
        // ═══════════════════════════════════════════════════════════
        match obs.trend {
            Trend::Up => {
                conviction += 30;
                direction = SignalDirection::Buy;
                reasons.push("Trend: UP".to_string());
            }
            Trend::Down => {
                conviction += 30;
                direction = SignalDirection::Sell;
                reasons.push("Trend: DOWN".to_string());
            }
            Trend::Neutral => {
                // No conviction from trend
            }
        }

        // ═══════════════════════════════════════════════════════════
        // MOMENTUM CONFIRMATION (weight: 25 points)
        // ═══════════════════════════════════════════════════════════
        let momentum_aligns = match (&direction, &obs.momentum) {
            (SignalDirection::Buy, Momentum::Up) => true,
            (SignalDirection::Sell, Momentum::Down) => true,
            _ => false,
        };

        if momentum_aligns {
            conviction += 25;
            reasons.push("Momentum confirms".to_string());
        } else if obs.momentum != Momentum::Neutral {
            conviction -= 15;
            reasons.push("Momentum diverges".to_string());
        }

        // ═══════════════════════════════════════════════════════════
        // SUPPORT/RESISTANCE (weight: 25 points)
        // ═══════════════════════════════════════════════════════════
        match direction {
            SignalDirection::Buy => {
                if obs.near_support {
                    conviction += 25;
                    reasons.push("Near support - good entry".to_string());
                }
                if obs.near_resistance {
                    conviction -= 20;
                    reasons.push("Near resistance - risky".to_string());
                }
            }
            SignalDirection::Sell => {
                if obs.near_resistance {
                    conviction += 25;
                    reasons.push("Near resistance - good entry".to_string());
                }
                if obs.near_support {
                    conviction -= 20;
                    reasons.push("Near support - risky".to_string());
                }
            }
            SignalDirection::Hold => {}
        }

        // ═══════════════════════════════════════════════════════════
        // BOUNCE DETECTION (weight: 20 points)
        // ═══════════════════════════════════════════════════════════
        match direction {
            SignalDirection::Buy => {
                if obs.bounce_up {
                    conviction += 20;
                    reasons.push("Bounce up detected".to_string());
                }
                if obs.bounce_down {
                    conviction -= 25;
                    direction = SignalDirection::Hold;
                    reasons.push("Bounce down - wrong direction".to_string());
                }
            }
            SignalDirection::Sell => {
                if obs.bounce_down {
                    conviction += 20;
                    reasons.push("Bounce down detected".to_string());
                }
                if obs.bounce_up {
                    conviction -= 25;
                    direction = SignalDirection::Hold;
                    reasons.push("Bounce up - wrong direction".to_string());
                }
            }
            SignalDirection::Hold => {}
        }

	// ═══════════════════════════════════════════════════════════
        // VOLUME CONFIRMATION (weight: 10 points)
        // ═══════════════════════════════════════════════════════════
        match obs.volume_state {
            VolumeState::Spike => {
                conviction += 10;
                reasons.push("Volume spike - strong move".to_string());
            }
            VolumeState::Dead => {
                conviction -= 10;
                reasons.push("Dead volume - weak move".to_string());
            }
            VolumeState::Normal => {}
        }
        // ═══════════════════════════════════════════════════════════
        // FINAL DECISION
        // ═══════════════════════════════════════════════════════════
        let conviction = conviction.max(0).min(100) as u8;

        // Not enough conviction - hold
        if conviction < self.min_conviction {
            return TradeSignal {
                direction: SignalDirection::Hold,
                conviction,
                stop_loss: Decimal::ZERO,
                take_profit: Decimal::ZERO,
                reasons,
            };
        }

        // Calculate SL/TP based on ATR-like range (simplified)
        let sl_distance = current_price * dec!(0.003); // 0.3% SL
        let tp_distance = sl_distance * self.risk_reward_ratio;

        let (stop_loss, take_profit) = match direction {
            SignalDirection::Buy => (
                current_price - sl_distance,
                current_price + tp_distance,
            ),
            SignalDirection::Sell => (
                current_price + sl_distance,
                current_price - tp_distance,
            ),
            SignalDirection::Hold => (Decimal::ZERO, Decimal::ZERO),
        };

        TradeSignal {
            direction,
            conviction,
            stop_loss,
            take_profit,
            reasons,
        }
    }
}
