//! Sovereign v4.0 "Perpetual" - Institutional Autonomous Trading System

use anyhow::Result;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use tokio::sync::mpsc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::sync::Arc;
use tokio::sync::Mutex;

mod core;
mod broker;
mod data;
mod comms;

use crate::core::lossless::MarketObserver;
use crate::core::types::Candle;
use crate::core::strategy::{Strategy, SignalDirection};
use crate::core::guardian::{RiskGuardian, RiskConfig};
use crate::data::mt5_bridge::{self, BridgeMessage, BridgeWriter};
use crate::comms::telegram;

const VPS_HOST: &str = "213.136.76.40";
const VPS_PORT: u16 = 5555;

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("═══════════════════════════════════════════════════════════");
    info!("  SOVEREIGN v4.0 - Perpetual Autonomous Trading System");
    info!("═══════════════════════════════════════════════════════════");
    
    telegram::send_startup().await;
    
    let (tx, mut rx) = mpsc::channel::<BridgeMessage>(100);
    let writer: BridgeWriter = Arc::new(Mutex::new(None));
    let writer_clone = writer.clone();
    
    // Initialize components
    let mut observer = MarketObserver::new(dec!(0.01), true);
    let strategy = Strategy::default();
    let mut guardian = RiskGuardian::new(RiskConfig::default());
    
    let mut tick_count = 0u64;
    let mut candle_count = 0u64;
    let mut in_position = false;
    let mut current_ticket: u64 = 0;
    let mut last_direction = String::new();
    let mut total_pnl = Decimal::ZERO;
    let mut trade_count = 0u32;
    let mut current_balance = dec!(10000);
    let mut current_equity = dec!(10000);
    
    tokio::spawn(async move {
        loop {
            if let Err(e) = mt5_bridge::connect(VPS_HOST, VPS_PORT, tx.clone(), writer_clone.clone()).await {
                info!("Bridge error: {}. Reconnecting in 5s...", e);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    });
    
    info!("Strategy: min_conviction=60, risk_reward=1:2");
    info!("Risk Guardian: {}", guardian.status());
    info!("Waiting for market data...");
    
    let writer_for_account = writer.clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        let _ = mt5_bridge::request_account(&writer_for_account).await;
    });
    
    while let Some(msg) = rx.recv().await {
        match msg {
            BridgeMessage::Tick(tick) => {
                tick_count += 1;
                if tick_count % 100 == 0 {
                    info!("Tick #{}: bid={} ask={}", tick_count, tick.bid, tick.ask);
                }
            }
            BridgeMessage::Candle(candle) => {
                candle_count += 1;
                
                // Check daily reset
                guardian.check_daily_reset(current_balance);
                
                let c = Candle::new(
                    chrono::Utc::now(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume.into(),
                );
                
                observer.update(&c);
                let obs = observer.observe(candle.close);
                let signal = strategy.analyze(&obs, candle.close);
                
                info!("═══════════════════════════════════════════════════════════");
                info!("CANDLE #{}: O={} H={} L={} C={}", 
                    candle_count, candle.open, candle.high, candle.low, candle.close);
                info!("Trend: {} | Momentum: {} | Volume: {:?}", 
                    obs.trend, obs.momentum, obs.volume_state);
                info!("Signal: {:?} | Conviction: {}% | In Position: {}", 
                    signal.direction, signal.conviction, in_position);
                
                for reason in &signal.reasons {
                    info!("  → {}", reason);
                }
                
                // Check emergency close conditions
                let (emergency, emergency_reason) = guardian.check_emergency_close(current_balance, current_equity);
                if emergency && in_position {
                    warn!("🚨 EMERGENCY CLOSE: {}", emergency_reason);
                    let _ = telegram::send(&format!("🚨 EMERGENCY: {}", emergency_reason)).await;
                    if current_ticket > 0 {
                        let _ = mt5_bridge::send_close(&writer, current_ticket).await;
                    }
                }
                
                // Check if we should trade
                if !in_position && signal.direction != SignalDirection::Hold {
                    let (can_trade, reject_reason) = guardian.can_trade(
                        current_balance,
                        current_equity,
                        1, // current positions
                    );
                    
                    if can_trade {
                        info!("═══════════════════════════════════════════════════════════");
                        info!("🚨 TRADE SIGNAL: {:?}", signal.direction);
                        info!("   Entry: {}", candle.close);
                        info!("   SL: {}", signal.stop_loss);
                        info!("   TP: {}", signal.take_profit);
                        info!("   Conviction: {}%", signal.conviction);
                        info!("═══════════════════════════════════════════════════════════");
                        
                        let dir_str = format!("{:?}", signal.direction);
                        telegram::send_signal(
                            &dir_str,
                            &candle.close.to_string(),
                            &signal.stop_loss.to_string(),
                            &signal.take_profit.to_string(),
                            signal.conviction,
                        ).await;
                        
                        last_direction = dir_str.clone();
                        
                        // Calculate position size based on risk
                        let sl_distance = (candle.close - signal.stop_loss).abs();
                        let lots = guardian.calculate_position_size(
                            current_balance,
                            sl_distance,
                            dec!(100),   // point value for gold
                            dec!(0.01),  // min lot
                            dec!(10.0),  // max lot
                            dec!(0.01),  // lot step
                            dec!(1.0),   // contract size multiplier
                        );
                        
                        info!("   Lots: {} (risk-adjusted)", lots);
                        
                        match signal.direction {
                            SignalDirection::Buy => {
                                if let Err(e) = mt5_bridge::send_buy(&writer, lots, signal.stop_loss, signal.take_profit).await {
                                    warn!("Failed to send buy: {}", e);
                                }
                            }
                            SignalDirection::Sell => {
                                if let Err(e) = mt5_bridge::send_sell(&writer, lots, signal.stop_loss, signal.take_profit).await {
                                    warn!("Failed to send sell: {}", e);
                                }
                            }
                            SignalDirection::Hold => {}
                        }
                    } else {
                        info!("⛔ Trade blocked by Risk Guardian: {}", reject_reason);
                    }
                }
                
                info!("═══════════════════════════════════════════════════════════");
            }
            BridgeMessage::OrderResult { success, ticket, price, error } => {
                if success {
                    info!("✅ ORDER FILLED: ticket={} price={}", ticket, price);
                    telegram::send_fill(&last_direction, ticket, &price.to_string()).await;
                    in_position = true;
                    current_ticket = ticket;
                    trade_count += 1;
                    guardian.record_trade_opened();
                } else {
                    warn!("❌ ORDER FAILED: {}", error);
                    let _ = telegram::send(&format!("❌ Order failed: {}", error)).await;
                }
            }
            BridgeMessage::PositionOpen(pos) => {
                info!("📊 Position Open: ticket={} side={} profit={}", 
                    pos.ticket, if pos.side == 0 { "BUY" } else { "SELL" }, pos.profit);
                in_position = true;
                current_ticket = pos.ticket;
            }
            BridgeMessage::PositionUpdate { ticket, profit } => {
                if tick_count % 50 == 0 {
                    info!("📊 Position {}: P&L ${}", ticket, profit);
                }
            }
            BridgeMessage::PositionClosed => {
                info!("═══════════════════════════════════════════════════════════");
                info!("📊 POSITION CLOSED");
                info!("═══════════════════════════════════════════════════════════");
                let _ = telegram::send("📊 Position closed by SL/TP").await;
                in_position = false;
                current_ticket = 0;
            }
            BridgeMessage::CloseResult { success, ticket, profit, error } => {
                if success {
                    info!("✅ CLOSED: ticket={} profit=${}", ticket, profit);
                    total_pnl += profit;
                    guardian.record_trade_closed(profit);
                    let _ = telegram::send(&format!("✅ Closed ticket {} | P&L: ${} | Total: ${}", 
                        ticket, profit, total_pnl)).await;
                    in_position = false;
                    current_ticket = 0;
                } else {
                    warn!("❌ Close failed: {}", error);
                }
            }
            BridgeMessage::AccountInfo { balance, equity, profit } => {
                current_balance = balance;
                current_equity = equity;
                info!("═══════════════════════════════════════════════════════════");
                info!("ACCOUNT: Balance=${} Equity=${} Profit=${}", balance, equity, profit);
                info!("Session: {} trades | Total P&L: ${}", trade_count, total_pnl);
                info!("Guardian: {}", guardian.status());
                info!("═══════════════════════════════════════════════════════════");
            }
        }
    }
    
    Ok(())
}
