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
mod config;

use crate::core::types::Candle;
use crate::core::coordinator::Coordinator;
use crate::core::strategy::SignalDirection;
use crate::data::mt5_bridge::{self, BridgeMessage, BridgeWriter};
use crate::data::database::TradeDb;
use crate::comms::telegram;
use crate::config::Config;

const SEP: &str = "===========================================================";

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::load("config.toml").unwrap_or_else(|e| {
        eprintln!("Failed to load config.toml: {}. Using defaults.", e);
        std::process::exit(1);
    });
    
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("{}", SEP);
    info!("  {} - Perpetual Autonomous Trading System", cfg.system.name);
    info!("{}", SEP);
    
    let db = TradeDb::new("sovereign_trades.db")?;
    
    if let Ok((total, wins, _losses, pnl)) = db.get_total_stats() {
        if total > 0 {
            let win_rate = (wins as f64 / total as f64) * 100.0;
            info!("Historical: {} trades | {:.1}% win rate | ${:.2} P&L", total, win_rate, pnl);
        }
    }
    
    let risk_config = cfg.risk.to_guardian_config();
    let mut coordinator = Coordinator::with_config(cfg.risk.max_positions, risk_config);
    
    for sym in &cfg.symbols {
        coordinator.add_agent(&sym.name, sym.tick_size_decimal(), sym.is_forex);
        info!("Agent: {} (tick={}, forex={})", sym.name, sym.tick_size, sym.is_forex);
    }
    
    info!("Agents: {} symbols loaded", coordinator.agent_count());
    
    if cfg.telegram.enabled {
        telegram::send_startup().await;
    }
    
    let (tx, mut rx) = mpsc::channel::<BridgeMessage>(100);
    let writer: BridgeWriter = Arc::new(Mutex::new(None));
    let writer_clone = writer.clone();
    
    let mut tick_count = 0u64;
    let mut candle_count = 0u64;
    let mut last_direction = String::new();
    let mut last_lots = dec!(0.01);
    let mut last_sl = Decimal::ZERO;
    let mut last_tp = Decimal::ZERO;
    let mut last_conviction: u8 = 0;
    let mut total_pnl = Decimal::ZERO;
    let mut trade_count = 0u32;
    let mut current_balance = dec!(10000);
    let mut current_equity = dec!(10000);
    
    let current_symbol = cfg.symbols.first()
        .map(|s| s.name.clone())
        .unwrap_or_else(|| "XAUUSD".to_string());
    
    let point_value = cfg.symbols.first()
        .map(|s| s.point_value_decimal())
        .unwrap_or(dec!(100));
    
    let bridge_host = cfg.bridge.host.clone();
    let bridge_port = cfg.bridge.port;
    let max_positions = cfg.risk.max_positions;
    let min_conviction = cfg.strategy.min_conviction;
    let risk_reward = cfg.strategy.risk_reward_ratio;
    let telegram_enabled = cfg.telegram.enabled;
    
    tokio::spawn(async move {
        loop {
            if let Err(e) = mt5_bridge::connect(&bridge_host, bridge_port, tx.clone(), writer_clone.clone()).await {
                info!("Bridge error: {}. Reconnecting in 5s...", e);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    });
    
    info!("Strategy: min_conviction={}, risk_reward=1:{}", min_conviction, risk_reward);
    info!("Guardian: {}", coordinator.guardian_status());
    info!("Bridge: {}:{}", cfg.bridge.host, cfg.bridge.port);
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
                coordinator.update_tick(&current_symbol, tick.bid, tick.ask);
                
                if tick_count % 100 == 0 {
                    info!("Tick #{}: bid={} ask={}", tick_count, tick.bid, tick.ask);
                }
            }
            BridgeMessage::Candle(candle) => {
                candle_count += 1;
                coordinator.check_daily_reset(current_balance);
                
                let c = Candle::new(
                    chrono::Utc::now(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume.into(),
                );
                
                coordinator.update_candle(&current_symbol, &c);
                
                if let Some(obs) = coordinator.get_observation(&current_symbol) {
                    info!("{}", SEP);
                    info!("CANDLE #{}: {} | O={} H={} L={} C={}", 
                        candle_count, current_symbol, candle.open, candle.high, candle.low, candle.close);
                    info!("Trend: {} | Momentum: {} | Volume: {:?}", 
                        obs.trend, obs.momentum, obs.volume_state);
                    info!("Positions: {}/{}", coordinator.active_position_count(), max_positions);
                }
                
                let signals = coordinator.collect_signals(current_balance, current_equity);
                
                if signals.is_empty() {
                    info!("Signal: HOLD | No opportunities");
                } else {
                    for signal in &signals {
                        info!("Signal #{}: {} {:?} | Conviction: {}%", 
                            signal.rank, signal.symbol, signal.direction, signal.conviction);
                    }
                    
                    if let Some(best) = signals.first() {
                        info!("{}", SEP);
                        info!("EXECUTING: {} {:?}", best.symbol, best.direction);
                        info!("  Entry: {} | SL: {} | TP: {}", best.price, best.stop_loss, best.take_profit);
                        info!("  Conviction: {}%", best.conviction);
                        info!("{}", SEP);
                        
                        let dir_str = format!("{:?}", best.direction);
                        if telegram_enabled {
                            telegram::send_signal(
                                &dir_str,
                                &best.price.to_string(),
                                &best.stop_loss.to_string(),
                                &best.take_profit.to_string(),
                                best.conviction,
                            ).await;
                        }
                        
                        last_direction = dir_str;
                        last_sl = best.stop_loss;
                        last_tp = best.take_profit;
                        last_conviction = best.conviction;
                        
                        let sl_distance = (best.price - best.stop_loss).abs();
                        let lots = coordinator.calculate_lots(current_balance, sl_distance, point_value);
                        last_lots = lots;
                        
                        info!("  Lots: {} (risk-adjusted)", lots);
                        
                        match best.direction {
                            SignalDirection::Buy => {
                                if let Err(e) = mt5_bridge::send_buy(&writer, lots, best.stop_loss, best.take_profit).await {
                                    warn!("Failed to send buy: {}", e);
                                }
                            }
                            SignalDirection::Sell => {
                                if let Err(e) = mt5_bridge::send_sell(&writer, lots, best.stop_loss, best.take_profit).await {
                                    warn!("Failed to send sell: {}", e);
                                }
                            }
                            SignalDirection::Hold => {}
                        }
                    }
                }
                
                info!("{}", SEP);
            }
            BridgeMessage::OrderResult { success, ticket, price, error } => {
                if success {
                    info!("ORDER FILLED: ticket={} price={}", ticket, price);
                    if telegram_enabled {
                        telegram::send_fill(&last_direction, ticket, &price.to_string()).await;
                    }
                    
                    let side = if last_direction == "Buy" { SignalDirection::Buy } else { SignalDirection::Sell };
                    coordinator.position_opened(&current_symbol, ticket, side);
                    trade_count += 1;
                    
                    if let Err(e) = db.record_open(
                        ticket,
                        &last_direction,
                        last_lots,
                        price,
                        last_sl,
                        last_tp,
                        last_conviction,
                    ) {
                        warn!("DB error: {}", e);
                    }
                } else {
                    warn!("ORDER FAILED: {}", error);
                    if telegram_enabled {
                        let _ = telegram::send(&format!("Order failed: {}", error)).await;
                    }
                }
            }
            BridgeMessage::PositionOpen(pos) => {
                info!("Position Open: {} ticket={} profit={}", current_symbol, pos.ticket, pos.profit);
            }
            BridgeMessage::PositionUpdate { ticket, profit } => {
                if tick_count % 50 == 0 {
                    info!("Position {}: P&L ${}", ticket, profit);
                }
            }
            BridgeMessage::PositionClosed => {
                info!("{}", SEP);
                info!("POSITION CLOSED: {}", current_symbol);
                info!("{}", SEP);
                coordinator.position_closed(&current_symbol, Decimal::ZERO);
                if telegram_enabled {
                    let _ = telegram::send("Position closed by SL/TP").await;
                }
            }
            BridgeMessage::CloseResult { success, ticket, profit, error } => {
                if success {
                    info!("CLOSED: ticket={} profit=${}", ticket, profit);
                    total_pnl += profit;
                    coordinator.position_closed(&current_symbol, profit);
                    
                    if let Err(e) = db.record_close(ticket, Decimal::ZERO, profit) {
                        warn!("DB error: {}", e);
                    }
                    
                    if let Ok((total, wins, losses, pnl)) = db.get_today_stats() {
                        info!("Today: {} trades | W:{} L:{} | ${:.2}", total, wins, losses, pnl);
                    }
                    
                    if telegram_enabled {
                        let _ = telegram::send(&format!("Closed {} | P&L: ${} | Total: ${}", 
                            current_symbol, profit, total_pnl)).await;
                    }
                } else {
                    warn!("Close failed: {}", error);
                }
            }
            BridgeMessage::AccountInfo { balance, equity, profit } => {
                current_balance = balance;
                current_equity = equity;
                info!("{}", SEP);
                info!("ACCOUNT: Balance=${} Equity=${} Profit=${}", balance, equity, profit);
                info!("Session: {} trades | Total P&L: ${}", trade_count, total_pnl);
                info!("Active: {}/{} positions", coordinator.active_position_count(), max_positions);
                info!("{}", SEP);
            }
        }
    }
    
    Ok(())
}
