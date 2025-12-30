//! Sovereign v4.0 "Perpetual" - Institutional Autonomous Trading System

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use tokio::sync::mpsc;
use rust_decimal_macros::dec;
use std::sync::Arc;
use tokio::sync::Mutex;

mod core;
mod broker;
mod data;
mod comms;

use crate::core::lossless::MarketObserver;
use crate::core::types::Candle;
use crate::data::mt5_bridge::{self, BridgeMessage, BridgeWriter};

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
    
    let (tx, mut rx) = mpsc::channel::<BridgeMessage>(100);
    let writer: BridgeWriter = Arc::new(Mutex::new(None));
    let writer_clone = writer.clone();
    
    let mut observer = MarketObserver::new(dec!(0.01), true);
    let mut tick_count = 0u64;
    let mut candle_count = 0u64;
    
    // Spawn bridge connection
    tokio::spawn(async move {
        loop {
            if let Err(e) = mt5_bridge::connect(VPS_HOST, VPS_PORT, tx.clone(), writer_clone.clone()).await {
                info!("Bridge error: {}. Reconnecting in 5s...", e);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    });
    
    info!("Waiting for market data...");
    
    // Request account info after 3 seconds
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
                    info!("Tick #{}: bid={} ask={} spread={}", 
                        tick_count, tick.bid, tick.ask, tick.ask - tick.bid);
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
                    candle.volume,
                );
                
                observer.update(&c);
                let obs = observer.observe(candle.close);
                
                info!("═══════════════════════════════════════════════════════════");
                info!("CANDLE #{}: O={} H={} L={} C={}", 
                    candle_count, candle.open, candle.high, candle.low, candle.close);
                info!("Trend: {} | Momentum: {} | Volume: {:?}", 
                    obs.trend, obs.momentum, obs.volume_state);
                info!("Near Support: {} | Near Resistance: {}", 
                    obs.near_support, obs.near_resistance);
                info!("Bounce Up: {} | Bounce Down: {}", 
                    obs.bounce_up, obs.bounce_down);
                info!("═══════════════════════════════════════════════════════════");
            }
            BridgeMessage::OrderResult { success, ticket, price, error } => {
                if success {
                    info!("ORDER FILLED: ticket={} price={}", ticket, price);
                } else {
                    info!("ORDER FAILED: {}", error);
                }
            }
            BridgeMessage::AccountInfo { balance, equity, profit } => {
                info!("═══════════════════════════════════════════════════════════");
                info!("ACCOUNT: Balance=${} Equity=${} Profit=${}", balance, equity, profit);
                info!("═══════════════════════════════════════════════════════════");
            }
        }
    }
    
    Ok(())
}
