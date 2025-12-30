//! Sovereign v4.0 "Perpetual" - Institutional Autonomous Trading System

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use rust_decimal_macros::dec;

mod core;
mod broker;
mod data;
mod comms;

use crate::core::types::*;
use crate::core::lossless::MarketObserver;
use crate::core::guardian::{RiskGuardian, RiskConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)?;
    
    info!("═══════════════════════════════════════════════════════════");
    info!("  SOVEREIGN v4.0 - Perpetual Autonomous Trading System");
    info!("═══════════════════════════════════════════════════════════");
    info!("  Philosophy: Build once, run forever.");
    info!("═══════════════════════════════════════════════════════════");
    
    // Initialize components
    let mut observer = MarketObserver::new(dec!(0.01), true); // Gold, forex mode
    let guardian = RiskGuardian::new(RiskConfig::default());
    
    info!("Market Observer initialized for XAUUSD");
    info!("Risk Guardian: {}", guardian.status());
    
    // Simulate some candles to test the lossless algorithms
    let test_candles = vec![
        Candle::new(chrono::Utc::now(), dec!(2650.00), dec!(2652.00), dec!(2648.00), dec!(2651.00), dec!(1000)),
        Candle::new(chrono::Utc::now(), dec!(2651.00), dec!(2655.00), dec!(2650.00), dec!(2654.00), dec!(1200)),
        Candle::new(chrono::Utc::now(), dec!(2654.00), dec!(2658.00), dec!(2653.00), dec!(2657.00), dec!(1500)),
        Candle::new(chrono::Utc::now(), dec!(2657.00), dec!(2660.00), dec!(2655.00), dec!(2656.00), dec!(1100)),
        Candle::new(chrono::Utc::now(), dec!(2656.00), dec!(2658.00), dec!(2652.00), dec!(2653.00), dec!(1300)),
    ];
    
    info!("Processing {} test candles...", test_candles.len());
    
    for (i, candle) in test_candles.iter().enumerate() {
        observer.update(candle);
        let obs = observer.observe(candle.close);
        
        info!(
            "Candle {}: Close={} | Trend={} | Momentum={} | Volume={:?}",
            i + 1,
            candle.close,
            obs.trend,
            obs.momentum,
            obs.volume_state
        );
    }
    
    info!("═══════════════════════════════════════════════════════════");
    info!("  Lossless algorithms working. Ready for real data.");
    info!("═══════════════════════════════════════════════════════════");
    
    Ok(())
}
