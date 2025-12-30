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
    
    let mut observer = MarketObserver::new(dec!(0.01), true);
    let strategy = Strategy::default();
    
    let mut tick_count = 0u64;
    let mut candle_count = 0u64;
    let mut in_position = false;
    let mut current_ticket: u64 = 0;
    let mut last_direction = String::new();
    let mut total_pnl = Decimal::ZERO;
    let mut trade_count = 0u32;
    
    tokio::spawn(async move {
        loop {
            if let Err(e) = mt5_bridge::connect(VPS_HOST, VPS_PORT, tx.clone(), writer_clone.clone()).await {
                info!("Bridge error: {}. Reconnecting in 5s...", e);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    });
    
    info!("Strategy: min_conviction=60, risk_reward=1:2");
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
                
                if !in_position && signal.direction != SignalDirection::Hold {
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
                    
                    let lots = dec!(0.01);
                    
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
                } else {
                    warn!("❌ ORDER FAILED: {}", error);
                    telegram::send(&format!("❌ Order failed: {}", error)).await;
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
                telegram::send("📊 Position closed by SL/TP").await;
                in_position = false;
                current_ticket = 0;
            }
            BridgeMessage::CloseResult { success, ticket, profit, error } => {
                if success {
                    info!("✅ CLOSED: ticket={} profit=${}", ticket, profit);
                    total_pnl += profit;
                    telegram::send(&format!("✅ Closed ticket {} | P&L: ${} | Total: ${}", 
                        ticket, profit, total_pnl)).await;
                    in_position = false;
                    current_ticket = 0;
                } else {
                    warn!("❌ Close failed: {}", error);
                }
            }
            BridgeMessage::AccountInfo { balance, equity, profit } => {
                info!("═══════════════════════════════════════════════════════════");
                info!("ACCOUNT: Balance=${} Equity=${} Profit=${}", balance, equity, profit);
                info!("Session: {} trades | Total P&L: ${}", trade_count, total_pnl);
                info!("═══════════════════════════════════════════════════════════");
            }
        }
    }
    
    Ok(())
}
