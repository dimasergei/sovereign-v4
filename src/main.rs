//! Sovereign v4.0 "Perpetual" - Lossless Autonomous Trading System
//!
//! Based on pftq's Tech Trader philosophy:
//! - No parameters, no thresholds, no statistics
//! - Pure counting-based S/R detection
//! - Volume capitulation for entry signals
//! - One independent agent per symbol

use anyhow::Result;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use tokio::sync::mpsc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};
use std::collections::HashMap;
use chrono::{Utc, Weekday, Timelike, Datelike};

mod core;
mod universe;
mod portfolio;
mod broker;
mod data;
mod comms;
mod config;

use crate::core::{SymbolAgent, AgentSignal, Signal, Side, Position, HealthMonitor};
use crate::core::health::HealthStatus;
use crate::universe::Universe;
use crate::portfolio::{Portfolio, PortfolioPosition};
use crate::data::alpaca_stream::{self, AlpacaMessage};
use crate::data::database::TradeDb;
use crate::broker::alpaca::AlpacaBroker;
use crate::comms::telegram;
use crate::config::Config;

const SEP: &str = "===========================================================";

/// Check if US stock market is open
fn is_market_open() -> bool {
    let now = Utc::now();
    let weekday = now.weekday();

    if weekday == Weekday::Sat || weekday == Weekday::Sun {
        return false;
    }

    let hour = now.hour();
    let minute = now.minute();
    let time_minutes = hour * 60 + minute;

    // 14:30 UTC = 870 minutes, 21:00 UTC = 1260 minutes
    time_minutes >= 870 && time_minutes < 1260
}

/// Smart alerting - reduces spam
struct AlertManager {
    last_gap_alert: std::time::Instant,
    gap_alert_interval: std::time::Duration,
}

impl AlertManager {
    fn new() -> Self {
        Self {
            last_gap_alert: std::time::Instant::now(),
            gap_alert_interval: std::time::Duration::from_secs(300),
        }
    }

    fn should_alert_gap(&mut self) -> bool {
        if self.last_gap_alert.elapsed() >= self.gap_alert_interval && is_market_open() {
            self.last_gap_alert = std::time::Instant::now();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        // Reset alert timer when data flows
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let cfg = Config::load("config.toml").unwrap_or_else(|e| {
        eprintln!("Failed to load config.toml: {}. Exiting.", e);
        std::process::exit(1);
    });

    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("{}", SEP);
    info!("  {} - Lossless Autonomous Trading System", cfg.system.name);
    info!("  Philosophy: No parameters. No thresholds. Pure counting.");
    info!("{}", SEP);

    // Initialize database
    let db = TradeDb::new("sovereign_trades.db")?;
    if let Ok((total, wins, _losses, pnl)) = db.get_total_stats() {
        if total > 0 {
            let win_rate = (wins as f64 / total as f64) * 100.0;
            info!("Historical: {} trades | {:.1}% win rate | ${:.2} P&L", total, win_rate, pnl);
        }
    }

    // Get Alpaca config
    let alpaca_cfg = cfg.alpaca.clone().expect("Alpaca config required");

    // Initialize broker
    let broker = AlpacaBroker::new(
        alpaca_cfg.api_key.clone(),
        alpaca_cfg.secret_key.clone(),
        cfg.broker.paper,
    );

    // Get account info
    let mut initial_balance = dec!(100000);
    match broker.get_account().await {
        Ok(account) => {
            info!("Alpaca Account: ${} equity, ${} cash, ${} buying power",
                account.equity, account.cash, account.buying_power);
            if let Ok(eq) = account.equity.parse::<f64>() {
                initial_balance = Decimal::from_f64(eq).unwrap_or(dec!(100000));
            }
        }
        Err(e) => {
            warn!("Failed to get account info: {}", e);
        }
    }

    // Initialize universe
    let symbols: Vec<String> = cfg.symbols.iter().map(|s| s.name.clone()).collect();
    let universe = Universe::from_symbols(symbols.clone());
    info!("Universe: {} symbols loaded", universe.len());

    // Initialize agents (one per symbol)
    let mut agents: HashMap<String, SymbolAgent> = HashMap::new();
    for sym in &cfg.symbols {
        let price = sym.tick_size_decimal() * dec!(10000); // Estimate initial price
        let agent = SymbolAgent::new(sym.name.clone(), price);
        info!("Agent: {} (auto-granularity based on price)", sym.name);
        agents.insert(sym.name.clone(), agent);
    }
    info!("Agents: {} independent traders ready", agents.len());

    // Initialize portfolio
    let mut portfolio = Portfolio::new(initial_balance);

    // Recover existing positions
    match broker.get_positions().await {
        Ok(positions) => {
            if !positions.is_empty() {
                info!("{}", SEP);
                info!("RECOVERING {} EXISTING POSITION(S):", positions.len());
                for pos in &positions {
                    info!("  {} {} shares @ ${} | P&L: ${}",
                        pos.symbol, pos.qty, pos.avg_entry_price, pos.unrealized_pl);

                    // Add to portfolio
                    let side = if pos.side == "long" { Side::Long } else { Side::Short };
                    let qty: Decimal = pos.qty.parse().unwrap_or(Decimal::ZERO);
                    let entry_price: Decimal = pos.avg_entry_price.parse().unwrap_or(Decimal::ZERO);
                    let market_value: Decimal = qty * entry_price;

                    portfolio.add_position(PortfolioPosition {
                        symbol: pos.symbol.clone(),
                        side,
                        quantity: qty,
                        entry_price,
                        current_price: entry_price,
                        entry_time: Utc::now(),
                        market_value,
                    });

                    // Sync agent state
                    if let Some(agent) = agents.get_mut(&pos.symbol) {
                        agent.set_position(Some(Position {
                            side,
                            entry_price,
                            entry_time: Utc::now(),
                            quantity: qty,
                        }));
                    }
                }
                info!("{}", SEP);
            } else {
                info!("No existing positions to recover");
            }
        }
        Err(e) => {
            warn!("Failed to get positions: {}", e);
        }
    }

    // Send startup notification
    if cfg.telegram.enabled {
        telegram::send_startup().await;
    }

    // Setup message channel
    let (tx, mut rx) = mpsc::channel::<AlpacaMessage>(100);

    // Initialize health monitoring
    let mut health = HealthMonitor::new();
    let mut last_health_check = std::time::Instant::now();
    let mut alert_manager = AlertManager::new();

    // Counters
    let mut tick_count = 0u64;
    let mut bar_count = 0u64;

    // Config copies for closures
    let telegram_enabled = cfg.telegram.enabled;
    let api_key = alpaca_cfg.api_key.clone();
    let api_secret = alpaca_cfg.secret_key.clone();
    let symbols_clone = symbols.clone();

    // Spawn WebSocket connection
    tokio::spawn(async move {
        loop {
            info!("Connecting to Alpaca stream...");
            if let Err(e) = alpaca_stream::connect(&api_key, &api_secret, &symbols_clone, tx.clone()).await {
                warn!("Alpaca stream error: {}. Reconnecting in 5s...", e);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    });

    info!("Strategy: Lossless S/R + Volume Capitulation");
    info!("Portfolio: Max {:.0}% exposure per side, ~{:.0}% per position",
        portfolio::MAX_EXPOSURE_PER_SIDE * 100.0,
        portfolio::POSITION_SIZE_PCT * 100.0);
    info!("Broker: Alpaca (paper={})", cfg.broker.paper);
    info!("Health: Bar-based monitoring (90s timeout)");
    info!("Market Status: {}", if is_market_open() { "OPEN" } else { "CLOSED" });
    info!("Waiting for market data...");

    // Main event loop
    loop {
        // Health check every 10 seconds
        if last_health_check.elapsed().as_secs() >= 10 {
            last_health_check = std::time::Instant::now();
            health.set_market_open(is_market_open());

            match health.check() {
                HealthStatus::Healthy { gaps: _ } => {
                    alert_manager.reset();
                }
                HealthStatus::MarketClosed => {
                    // No warnings when market closed
                }
                HealthStatus::StaleData { seconds, gaps } => {
                    warn!("HEALTH: No bars for {}s (gap #{})", seconds, gaps);

                    if alert_manager.should_alert_gap() && telegram_enabled {
                        let _ = telegram::send(&format!(
                            "Data gap: {}s without bars (gap #{}) - Market: {}",
                            seconds, gaps, if is_market_open() { "OPEN" } else { "CLOSED" }
                        )).await;
                    }
                }
            }
        }

        // Process messages
        match tokio::time::timeout(
            tokio::time::Duration::from_millis(100),
            rx.recv()
        ).await {
            Ok(Some(msg)) => {
                match msg {
                    AlpacaMessage::Connected => {
                        info!("Connected to Alpaca WebSocket");
                    }
                    AlpacaMessage::Tick(tick) => {
                        tick_count += 1;
                        // Update portfolio prices
                        if let Some(price) = Decimal::from_f64(tick.bid.to_f64().unwrap_or(0.0)) {
                            portfolio.update_price(&tick.symbol, price);
                        }

                        if tick_count % 100 == 0 {
                            info!("Tick #{}: {} bid={} ask={}", tick_count, tick.symbol, tick.bid, tick.ask);
                        }
                    }
                    AlpacaMessage::Bar(bar) => {
                        bar_count += 1;
                        health.record_bar();
                        alert_manager.reset();

                        // Log bar
                        info!("{}", SEP);
                        info!("BAR #{}: {} | O={} H={} L={} C={} V={}",
                            bar_count, bar.symbol, bar.open, bar.high, bar.low, bar.close, bar.volume);

                        // Process through agent
                        if let Some(agent) = agents.get_mut(&bar.symbol) {
                            let signal = agent.process_bar(
                                Utc::now(),
                                bar.open,
                                bar.high,
                                bar.low,
                                bar.close,
                                bar.volume as u64,
                            );

                            // Log S/R levels
                            if let (Some(support), Some(resistance)) = (agent.support(), agent.resistance()) {
                                info!("S/R: Support={:.2} | Resistance={:.2}", support, resistance);
                            }
                            info!("Agent: {} bars processed, {} S/R levels tracked",
                                agent.bar_count(), agent.sr_level_count());
                            info!("Portfolio: {} | Positions: {}",
                                portfolio.exposure_summary(), portfolio.position_count());

                            // Check for signals
                            if !is_market_open() {
                                info!("Signal: HOLD | Market closed");
                                info!("{}", SEP);
                                continue;
                            }

                            if let Some(sig) = signal {
                                // Check portfolio constraints
                                if portfolio.should_execute(&sig) {
                                    info!("{}", SEP);
                                    info!("SIGNAL: {} {} @ {:.2}", sig.signal, sig.symbol, sig.price);
                                    info!("Reason: {}", sig.reason);
                                    if let Some(s) = sig.support {
                                        info!("Support: {:.2}", s);
                                    }
                                    if let Some(r) = sig.resistance {
                                        info!("Resistance: {:.2}", r);
                                    }
                                    info!("{}", SEP);

                                    // Execute trade
                                    let qty = portfolio.calculate_position_size(sig.price);

                                    match sig.signal {
                                        Signal::Buy => {
                                            // Use resistance as take profit
                                            let tp = sig.resistance;
                                            let sl = sig.support.map(|s| s - (sig.price - s) * dec!(0.5));

                                            match broker.buy(&sig.symbol, qty, sl, tp).await {
                                                Ok(order) => {
                                                    info!("BUY ORDER: {} qty={} status={}", order.id, qty, order.status);

                                                    // Add to portfolio
                                                    portfolio.add_position(PortfolioPosition {
                                                        symbol: sig.symbol.clone(),
                                                        side: Side::Long,
                                                        quantity: qty,
                                                        entry_price: sig.price,
                                                        current_price: sig.price,
                                                        entry_time: Utc::now(),
                                                        market_value: qty * sig.price,
                                                    });

                                                    if telegram_enabled {
                                                        telegram::send_signal(
                                                            "BUY",
                                                            &sig.price.to_string(),
                                                            &sl.map_or("N/A".to_string(), |s| s.to_string()),
                                                            &tp.map_or("N/A".to_string(), |t| t.to_string()),
                                                            100,
                                                        ).await;
                                                    }
                                                }
                                                Err(e) => {
                                                    warn!("Buy failed: {}", e);
                                                    // Rollback agent position
                                                    agent.close_position();
                                                }
                                            }
                                        }
                                        Signal::Sell => {
                                            // Close long position
                                            match broker.close_position(&sig.symbol).await {
                                                Ok(order) => {
                                                    info!("SELL ORDER: {} status={}", order.id, order.status);
                                                    portfolio.remove_position(&sig.symbol);

                                                    if telegram_enabled {
                                                        telegram::send(&format!(
                                                            "SELL {} @ {:.2} - {}",
                                                            sig.symbol, sig.price, sig.reason
                                                        )).await;
                                                    }
                                                }
                                                Err(e) => warn!("Sell failed: {}", e),
                                            }
                                        }
                                        Signal::Short => {
                                            // Use support as take profit
                                            let tp = sig.support;
                                            let sl = sig.resistance.map(|r| r + (r - sig.price) * dec!(0.5));

                                            match broker.sell(&sig.symbol, qty, sl, tp).await {
                                                Ok(order) => {
                                                    info!("SHORT ORDER: {} qty={} status={}", order.id, qty, order.status);

                                                    portfolio.add_position(PortfolioPosition {
                                                        symbol: sig.symbol.clone(),
                                                        side: Side::Short,
                                                        quantity: qty,
                                                        entry_price: sig.price,
                                                        current_price: sig.price,
                                                        entry_time: Utc::now(),
                                                        market_value: qty * sig.price,
                                                    });

                                                    if telegram_enabled {
                                                        telegram::send_signal(
                                                            "SHORT",
                                                            &sig.price.to_string(),
                                                            &sl.map_or("N/A".to_string(), |s| s.to_string()),
                                                            &tp.map_or("N/A".to_string(), |t| t.to_string()),
                                                            100,
                                                        ).await;
                                                    }
                                                }
                                                Err(e) => {
                                                    warn!("Short failed: {}", e);
                                                    agent.close_position();
                                                }
                                            }
                                        }
                                        Signal::Cover => {
                                            // Close short position
                                            match broker.close_position(&sig.symbol).await {
                                                Ok(order) => {
                                                    info!("COVER ORDER: {} status={}", order.id, order.status);
                                                    portfolio.remove_position(&sig.symbol);

                                                    if telegram_enabled {
                                                        telegram::send(&format!(
                                                            "COVER {} @ {:.2} - {}",
                                                            sig.symbol, sig.price, sig.reason
                                                        )).await;
                                                    }
                                                }
                                                Err(e) => warn!("Cover failed: {}", e),
                                            }
                                        }
                                        Signal::Hold => {}
                                    }
                                } else {
                                    info!("Signal: {} {} blocked by portfolio constraints", sig.signal, sig.symbol);
                                }
                            } else {
                                info!("Signal: HOLD | No opportunity");
                            }
                        }

                        info!("{}", SEP);
                    }
                    AlpacaMessage::Error(err) => {
                        warn!("Alpaca error: {}", err);
                    }
                }
            }
            Ok(None) => {
                warn!("Channel closed. Exiting.");
                break;
            }
            Err(_) => {
                // Timeout - continue loop for health checks
            }
        }
    }

    Ok(())
}
