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
use std::sync::Arc;
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
use crate::universe::{Universe, Sector};
use crate::portfolio::{Portfolio, PortfolioPosition};
use crate::data::alpaca_stream::{self, AlpacaMessage};
use crate::data::database::TradeDb;
use crate::data::memory::TradeMemory;
use crate::broker::alpaca::AlpacaBroker;
use crate::broker::ibkr::IbkrBroker;
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

/// Check if market is open for a given symbol
/// Crypto trades 24/7, stocks follow US market hours
fn is_market_open_for_symbol(symbol: &str) -> bool {
    if crate::universe::Universe::is_crypto(symbol) {
        return true;
    }
    is_market_open()
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

/// Broker type enum for runtime dispatch
enum BrokerType {
    Alpaca(AlpacaBroker),
    Ibkr(Arc<IbkrBroker>),
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

    // Initialize AGI memory for learning
    let memory = Arc::new(TradeMemory::new("sovereign_memory.db")?);
    if let Ok((total, wins, total_profit, avg_profit)) = memory.get_overall_stats() {
        if total > 0 {
            let win_rate = wins as f64 / total as f64 * 100.0;
            info!("Memory: {} trades learned | {:.1}% win rate | ${:.2} avg profit",
                total, win_rate, avg_profit);
        }
    }

    // Initialize broker based on config
    let broker_type = if cfg.is_ibkr() {
        let ibkr_cfg = cfg.ibkr_config().expect("IBKR config required");
        let broker = IbkrBroker::new(
            ibkr_cfg.gateway_url.clone(),
            ibkr_cfg.account_id.clone(),
        )?;
        info!("Broker: IBKR Client Portal (account: {})", ibkr_cfg.account_id);
        BrokerType::Ibkr(Arc::new(broker))
    } else {
        let alpaca_cfg = cfg.alpaca.clone().expect("Alpaca config required");
        let broker = AlpacaBroker::new(
            alpaca_cfg.api_key.clone(),
            alpaca_cfg.secret_key.clone(),
            cfg.broker.paper,
        );
        info!("Broker: Alpaca (paper={})", cfg.broker.paper);
        BrokerType::Alpaca(broker)
    };

    // Get account info and initial balance
    let mut initial_balance = dec!(100000);
    match &broker_type {
        BrokerType::Alpaca(broker) => {
            match broker.get_account().await {
                Ok(account) => {
                    info!("Account: ${} equity, ${} cash, ${} buying power",
                        account.equity, account.cash, account.buying_power);
                    if let Ok(eq) = account.equity.parse::<f64>() {
                        initial_balance = Decimal::from_f64(eq).unwrap_or(dec!(100000));
                    }
                }
                Err(e) => warn!("Failed to get account info: {}", e),
            }
        }
        BrokerType::Ibkr(broker) => {
            match broker.get_account().await {
                Ok(summary) => {
                    if let Some(nl) = &summary.net_liquidation {
                        info!("Account: ${:.2} net liquidation", nl.amount);
                        initial_balance = Decimal::from_f64(nl.amount).unwrap_or(dec!(100000));
                    }
                    if let Some(bp) = &summary.buying_power {
                        info!("Buying Power: ${:.2}", bp.amount);
                    }
                }
                Err(e) => warn!("Failed to get account info: {}", e),
            }
        }
    }

    // Initialize universe
    let symbols: Vec<String> = cfg.universe.symbols.clone();
    let universe = Universe::from_symbols(symbols.clone());
    info!("Universe: {} symbols loaded", universe.len());

    // Initialize agents (one per symbol) with AGI memory
    let mut agents: HashMap<String, SymbolAgent> = HashMap::new();
    for sym in &symbols {
        let agent = SymbolAgent::new_with_memory(sym.clone(), dec!(100), Arc::clone(&memory));
        agents.insert(sym.clone(), agent);
    }
    info!("Agents: {} created with memory, bootstrapping with historical data...", agents.len());

    // Bootstrap each agent with historical S/R data
    match &broker_type {
        BrokerType::Alpaca(broker) => {
            for sym in &symbols {
                match broker.get_all_daily_bars(sym).await {
                    Ok(bars) => {
                        if let Some(agent) = agents.get_mut(sym) {
                            for bar in &bars {
                                let open = Decimal::from_f64(bar.open).unwrap_or(dec!(0));
                                let high = Decimal::from_f64(bar.high).unwrap_or(dec!(0));
                                let low = Decimal::from_f64(bar.low).unwrap_or(dec!(0));
                                let close = Decimal::from_f64(bar.close).unwrap_or(dec!(0));
                                agent.bootstrap_bar(open, high, low, close, bar.volume);
                            }
                            info!("{}: Bootstrapped S/R from {} historical bars", sym, bars.len());
                        }
                    }
                    Err(e) => warn!("{}: Failed to fetch historical bars: {}", sym, e),
                }
            }
        }
        BrokerType::Ibkr(broker) => {
            for sym in &symbols {
                match broker.get_all_daily_bars(sym).await {
                    Ok(bars) => {
                        if let Some(agent) = agents.get_mut(sym) {
                            for bar in &bars {
                                let open = Decimal::from_f64(bar.open).unwrap_or(dec!(0));
                                let high = Decimal::from_f64(bar.high).unwrap_or(dec!(0));
                                let low = Decimal::from_f64(bar.low).unwrap_or(dec!(0));
                                let close = Decimal::from_f64(bar.close).unwrap_or(dec!(0));
                                agent.bootstrap_bar(open, high, low, close, bar.volume as u64);
                            }
                            info!("{}: Bootstrapped S/R from {} historical bars", sym, bars.len());
                        }
                    }
                    Err(e) => warn!("{}: Failed to fetch historical bars: {}", sym, e),
                }
            }
        }
    }
    info!("Agents: {} independent traders ready", agents.len());

    // Initialize portfolio
    let mut portfolio = Portfolio::new(initial_balance);

    // Recover existing positions
    match &broker_type {
        BrokerType::Alpaca(broker) => {
            recover_alpaca_positions(broker, &mut portfolio, &mut agents).await;
        }
        BrokerType::Ibkr(broker) => {
            recover_ibkr_positions(broker, &mut portfolio, &mut agents).await;
        }
    }

    // Send startup notification
    if cfg.telegram.enabled {
        telegram::send_startup().await;
    }

    // Initialize health monitoring
    let mut health = HealthMonitor::new();
    let mut last_health_check = std::time::Instant::now();
    let mut last_tickle = std::time::Instant::now();
    let mut alert_manager = AlertManager::new();

    // Counters
    let mut tick_count = 0u64;
    let mut bar_count = 0u64;

    let telegram_enabled = cfg.telegram.enabled;

    info!("Strategy: Lossless S/R + Volume Capitulation");
    info!("Portfolio: Max {:.0}% exposure per side, position size derived from S/R",
        portfolio::MAX_EXPOSURE_PER_SIDE * 100.0);
    info!("Position Sizing: 1% risk per trade, size = risk / stop distance");
    info!("Health: Bar-based monitoring (90s timeout)");
    info!("Market Status: {}", if is_market_open() { "OPEN" } else { "CLOSED" });

    // Run appropriate event loop based on broker
    match broker_type {
        BrokerType::Alpaca(broker) => {
            run_alpaca_loop(
                broker,
                cfg,
                &mut agents,
                &mut portfolio,
                &mut health,
                &mut alert_manager,
                telegram_enabled,
            ).await
        }
        BrokerType::Ibkr(broker) => {
            run_ibkr_loop(
                broker,
                cfg,
                &mut agents,
                &mut portfolio,
                &mut health,
                &mut alert_manager,
                &mut last_tickle,
                telegram_enabled,
            ).await
        }
    }
}

/// Recover positions from Alpaca
async fn recover_alpaca_positions(
    broker: &AlpacaBroker,
    portfolio: &mut Portfolio,
    agents: &mut HashMap<String, SymbolAgent>,
) {
    match broker.get_positions().await {
        Ok(positions) => {
            if !positions.is_empty() {
                info!("{}", SEP);
                info!("RECOVERING {} EXISTING POSITION(S):", positions.len());
                for pos in &positions {
                    info!("  {} {} shares @ ${} | P&L: ${}",
                        pos.symbol, pos.qty, pos.avg_entry_price, pos.unrealized_pl);

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
                        sector: Sector::from_symbol(&pos.symbol),
                    });

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
        Err(e) => warn!("Failed to get positions: {}", e),
    }
}

/// Recover positions from IBKR
async fn recover_ibkr_positions(
    broker: &IbkrBroker,
    portfolio: &mut Portfolio,
    agents: &mut HashMap<String, SymbolAgent>,
) {
    match broker.get_positions().await {
        Ok(positions) => {
            if !positions.is_empty() {
                info!("{}", SEP);
                info!("RECOVERING {} EXISTING POSITION(S):", positions.len());
                for pos in &positions {
                    let symbol = pos.contract_desc.clone().unwrap_or_default();
                    info!("  {} {:.0} shares @ ${:.2} | P&L: ${:.2}",
                        symbol, pos.position.abs(), pos.avg_cost,
                        pos.unrealized_pnl.unwrap_or(0.0));

                    let side = if pos.position >= 0.0 { Side::Long } else { Side::Short };
                    let qty = Decimal::from_f64(pos.position.abs()).unwrap_or(Decimal::ZERO);
                    let entry_price = Decimal::from_f64(pos.avg_cost).unwrap_or(Decimal::ZERO);
                    let market_value = qty * entry_price;

                    portfolio.add_position(PortfolioPosition {
                        symbol: symbol.clone(),
                        side,
                        quantity: qty,
                        entry_price,
                        current_price: entry_price,
                        entry_time: Utc::now(),
                        market_value,
                        sector: Sector::from_symbol(&symbol),
                    });

                    if let Some(agent) = agents.get_mut(&symbol) {
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
        Err(e) => warn!("Failed to get positions: {}", e),
    }
}

/// Run the Alpaca event loop with WebSocket streaming
async fn run_alpaca_loop(
    broker: AlpacaBroker,
    cfg: Config,
    agents: &mut HashMap<String, SymbolAgent>,
    portfolio: &mut Portfolio,
    health: &mut HealthMonitor,
    alert_manager: &mut AlertManager,
    telegram_enabled: bool,
) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<AlpacaMessage>(100);

    let alpaca_cfg = cfg.alpaca.clone().expect("Alpaca config required");
    let api_key = alpaca_cfg.api_key.clone();
    let api_secret = alpaca_cfg.secret_key.clone();
    let symbols_clone: Vec<String> = cfg.universe.symbols.clone();

    let mut last_health_check = std::time::Instant::now();
    let mut bar_count = 0u64;
    let mut tick_count = 0u64;
    let mut last_summary_date: Option<chrono::NaiveDate> = None;

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

    info!("Waiting for market data...");

    // Main event loop
    loop {
        // Health check every 10 seconds
        if last_health_check.elapsed().as_secs() >= 10 {
            last_health_check = std::time::Instant::now();
            health.set_market_open(is_market_open());

            match health.check() {
                HealthStatus::Healthy { gaps: _ } => alert_manager.reset(),
                HealthStatus::MarketClosed => {}
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
        match tokio::time::timeout(tokio::time::Duration::from_millis(100), rx.recv()).await {
            Ok(Some(msg)) => {
                match msg {
                    AlpacaMessage::Connected => {
                        info!("Connected to Alpaca WebSocket");
                    }
                    AlpacaMessage::Tick(tick) => {
                        tick_count += 1;
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

                        process_bar_signal(
                            &bar.symbol,
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close,
                            bar.volume as u64,
                            bar_count,
                            agents,
                            portfolio,
                            &broker,
                            telegram_enabled,
                        ).await;
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
            Err(_) => {}
        }

        // Daily summary at 21:05 UTC (5 min after market close)
        let now = Utc::now();
        let today = now.date_naive();
        if now.hour() == 21 && now.minute() >= 5 && now.minute() < 15 {
            if last_summary_date != Some(today) {
                last_summary_date = Some(today);
                if telegram_enabled {
                    let sector_info = portfolio.sector_summary();
                    telegram::send_daily_summary(
                        portfolio.position_count(),
                        portfolio.long_exposure_pct(),
                        portfolio.short_exposure_pct(),
                        portfolio.unrealized_pnl(),
                        bar_count,
                        &sector_info,
                    ).await;
                }
            }
        }
    }

    Ok(())
}

/// Run the IBKR event loop with polling
async fn run_ibkr_loop(
    broker: Arc<IbkrBroker>,
    cfg: Config,
    agents: &mut HashMap<String, SymbolAgent>,
    portfolio: &mut Portfolio,
    health: &mut HealthMonitor,
    alert_manager: &mut AlertManager,
    last_tickle: &mut std::time::Instant,
    telegram_enabled: bool,
) -> Result<()> {
    let symbols: Vec<String> = cfg.universe.symbols.clone();
    let mut last_health_check = std::time::Instant::now();
    let mut last_data_poll = std::time::Instant::now();
    let mut bar_count = 0u64;
    let mut last_summary_date: Option<chrono::NaiveDate> = None;

    info!("IBKR mode: Polling for market data every 60 seconds");
    info!("Waiting for market data...");

    loop {
        // Session keep-alive every 60 seconds
        if last_tickle.elapsed().as_secs() >= 60 {
            *last_tickle = std::time::Instant::now();
            if let Err(e) = broker.tickle().await {
                warn!("Session tickle failed: {}", e);
            }
        }

        // Health check every 10 seconds
        if last_health_check.elapsed().as_secs() >= 10 {
            last_health_check = std::time::Instant::now();
            health.set_market_open(is_market_open());

            match health.check() {
                HealthStatus::Healthy { gaps: _ } => alert_manager.reset(),
                HealthStatus::MarketClosed => {}
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

        // Poll for new data every 60 seconds during market hours
        if last_data_poll.elapsed().as_secs() >= 60 && is_market_open() {
            last_data_poll = std::time::Instant::now();

            // Get latest bar for each symbol
            for sym in &symbols {
                match broker.get_historical_bars(sym, "1d", "1d").await {
                    Ok(bars) => {
                        if let Some(bar) = bars.last() {
                            bar_count += 1;
                            health.record_bar();
                            alert_manager.reset();

                            process_bar_signal_ibkr(
                                sym,
                                bar.open,
                                bar.high,
                                bar.low,
                                bar.close,
                                bar.volume as u64,
                                bar_count,
                                agents,
                                portfolio,
                                &broker,
                                telegram_enabled,
                            ).await;
                        }
                    }
                    Err(e) => warn!("{}: Failed to get latest bar: {}", sym, e),
                }
            }
        }

        // Daily summary at 21:05 UTC (5 min after market close)
        let now = Utc::now();
        let today = now.date_naive();
        if now.hour() == 21 && now.minute() >= 5 && now.minute() < 15 {
            if last_summary_date != Some(today) {
                last_summary_date = Some(today);
                if telegram_enabled {
                    let sector_info = portfolio.sector_summary();
                    telegram::send_daily_summary(
                        portfolio.position_count(),
                        portfolio.long_exposure_pct(),
                        portfolio.short_exposure_pct(),
                        portfolio.unrealized_pnl(),
                        bar_count,
                        &sector_info,
                    ).await;
                }
            }
        }

        // Small sleep to prevent busy loop
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

/// Process a bar and generate/execute signals (Alpaca version)
async fn process_bar_signal(
    symbol: &str,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: u64,
    bar_count: u64,
    agents: &mut HashMap<String, SymbolAgent>,
    portfolio: &mut Portfolio,
    broker: &AlpacaBroker,
    telegram_enabled: bool,
) {
    info!("{}", SEP);
    info!("BAR #{}: {} | O={} H={} L={} C={} V={}",
        bar_count, symbol, open, high, low, close, volume);

    if let Some(agent) = agents.get_mut(symbol) {
        let signal = agent.process_bar(Utc::now(), open, high, low, close, volume);

        if let (Some(support), Some(resistance)) = (agent.support(), agent.resistance()) {
            info!("S/R: Support={:.2} | Resistance={:.2}", support, resistance);
        }
        info!("Agent: {} bars processed, {} S/R levels tracked",
            agent.bar_count(), agent.sr_level_count());
        info!("Portfolio: {} | Positions: {}",
            portfolio.exposure_summary(), portfolio.position_count());

        if !is_market_open_for_symbol(symbol) {
            info!("Signal: HOLD | Market closed for {}", symbol);
            info!("{}", SEP);
            return;
        }

        if let Some(sig) = signal {
            if portfolio.should_execute(&sig) {
                info!("{}", SEP);
                info!("SIGNAL: {} {} @ {:.2} ({:.0}th percentile volume)",
                    sig.signal, sig.symbol, sig.price, sig.volume_percentile);
                info!("Reason: {}", sig.reason);
                info!("{}", SEP);

                // LOSSLESS: Position size derived from support distance (or ATR fallback)
                let qty = portfolio.calculate_position_size(sig.price, sig.support, agent.atr());
                execute_alpaca_signal(&sig, qty, agent, portfolio, broker, telegram_enabled).await;
            } else {
                info!("Signal: {} {} blocked by portfolio constraints", sig.signal, sig.symbol);
            }
        } else {
            info!("Signal: HOLD | No opportunity");
        }
    }

    info!("{}", SEP);
}

/// Process a bar and generate/execute signals (IBKR version)
async fn process_bar_signal_ibkr(
    symbol: &str,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: u64,
    bar_count: u64,
    agents: &mut HashMap<String, SymbolAgent>,
    portfolio: &mut Portfolio,
    broker: &IbkrBroker,
    telegram_enabled: bool,
) {
    let open = Decimal::from_f64(open).unwrap_or(dec!(0));
    let high = Decimal::from_f64(high).unwrap_or(dec!(0));
    let low = Decimal::from_f64(low).unwrap_or(dec!(0));
    let close = Decimal::from_f64(close).unwrap_or(dec!(0));

    info!("{}", SEP);
    info!("BAR #{}: {} | O={} H={} L={} C={} V={}",
        bar_count, symbol, open, high, low, close, volume);

    if let Some(agent) = agents.get_mut(symbol) {
        let signal = agent.process_bar(Utc::now(), open, high, low, close, volume);

        if let (Some(support), Some(resistance)) = (agent.support(), agent.resistance()) {
            info!("S/R: Support={:.2} | Resistance={:.2}", support, resistance);
        }
        info!("Agent: {} bars processed, {} S/R levels tracked",
            agent.bar_count(), agent.sr_level_count());
        info!("Portfolio: {} | Positions: {}",
            portfolio.exposure_summary(), portfolio.position_count());

        if !is_market_open_for_symbol(symbol) {
            info!("Signal: HOLD | Market closed for {}", symbol);
            info!("{}", SEP);
            return;
        }

        if let Some(sig) = signal {
            if portfolio.should_execute(&sig) {
                info!("{}", SEP);
                info!("SIGNAL: {} {} @ {:.2} ({:.0}th percentile volume)",
                    sig.signal, sig.symbol, sig.price, sig.volume_percentile);
                info!("Reason: {}", sig.reason);
                info!("{}", SEP);

                // LOSSLESS: Position size derived from support distance (or ATR fallback)
                let qty = portfolio.calculate_position_size(sig.price, sig.support, agent.atr());
                execute_ibkr_signal(&sig, qty, agent, portfolio, broker, telegram_enabled).await;
            } else {
                info!("Signal: {} {} blocked by portfolio constraints", sig.signal, sig.symbol);
            }
        } else {
            info!("Signal: HOLD | No opportunity");
        }
    }

    info!("{}", SEP);
}

/// Execute a trading signal via Alpaca
async fn execute_alpaca_signal(
    sig: &AgentSignal,
    qty: Decimal,
    agent: &mut SymbolAgent,
    portfolio: &mut Portfolio,
    broker: &AlpacaBroker,
    telegram_enabled: bool,
) {
    match sig.signal {
        Signal::Buy => {
            let tp = sig.resistance;
            let sl = sig.support.map(|s| s - (sig.price - s) * dec!(0.5));

            match broker.buy(&sig.symbol, qty, sl, tp).await {
                Ok(order) => {
                    info!("BUY ORDER: {} qty={} status={}", order.id, qty, order.status);

                    portfolio.add_position(PortfolioPosition {
                        symbol: sig.symbol.clone(),
                        side: Side::Long,
                        quantity: qty,
                        entry_price: sig.price,
                        current_price: sig.price,
                        entry_time: Utc::now(),
                        market_value: qty * sig.price,
                        sector: Sector::from_symbol(&sig.symbol),
                    });

                    if telegram_enabled {
                        telegram::send_signal(
                            "BUY", &sig.price.to_string(),
                            &sl.map_or("N/A".to_string(), |s| s.to_string()),
                            &tp.map_or("N/A".to_string(), |t| t.to_string()),
                            100,
                        ).await;
                    }
                }
                Err(e) => {
                    warn!("Buy failed: {}", e);
                    agent.close_position();
                }
            }
        }
        Signal::Sell => {
            match broker.close_position(&sig.symbol).await {
                Ok(order) => {
                    info!("SELL ORDER: {} status={}", order.id, order.status);
                    portfolio.remove_position(&sig.symbol);
                    if telegram_enabled {
                        let _ = telegram::send(&format!("SELL {} @ {:.2} - {}", sig.symbol, sig.price, sig.reason)).await;
                    }
                }
                Err(e) => warn!("Sell failed: {}", e),
            }
        }
        Signal::Short => {
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
                        sector: Sector::from_symbol(&sig.symbol),
                    });

                    if telegram_enabled {
                        telegram::send_signal(
                            "SHORT", &sig.price.to_string(),
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
            match broker.close_position(&sig.symbol).await {
                Ok(order) => {
                    info!("COVER ORDER: {} status={}", order.id, order.status);
                    portfolio.remove_position(&sig.symbol);
                    if telegram_enabled {
                        let _ = telegram::send(&format!("COVER {} @ {:.2} - {}", sig.symbol, sig.price, sig.reason)).await;
                    }
                }
                Err(e) => warn!("Cover failed: {}", e),
            }
        }
        Signal::Hold => {}
    }
}

/// Execute a trading signal via IBKR
async fn execute_ibkr_signal(
    sig: &AgentSignal,
    qty: Decimal,
    agent: &mut SymbolAgent,
    portfolio: &mut Portfolio,
    broker: &IbkrBroker,
    telegram_enabled: bool,
) {
    match sig.signal {
        Signal::Buy => {
            match broker.buy(&sig.symbol, qty).await {
                Ok(order) => {
                    info!("BUY ORDER: {:?} qty={}", order.order_id, qty);

                    portfolio.add_position(PortfolioPosition {
                        symbol: sig.symbol.clone(),
                        side: Side::Long,
                        quantity: qty,
                        entry_price: sig.price,
                        current_price: sig.price,
                        entry_time: Utc::now(),
                        market_value: qty * sig.price,
                        sector: Sector::from_symbol(&sig.symbol),
                    });

                    if telegram_enabled {
                        telegram::send_signal(
                            "BUY", &sig.price.to_string(), "N/A", "N/A", 100,
                        ).await;
                    }
                }
                Err(e) => {
                    warn!("Buy failed: {}", e);
                    agent.close_position();
                }
            }
        }
        Signal::Sell => {
            match broker.close_position(&sig.symbol).await {
                Ok(order) => {
                    info!("SELL ORDER: {:?}", order.order_id);
                    portfolio.remove_position(&sig.symbol);
                    if telegram_enabled {
                        let _ = telegram::send(&format!("SELL {} @ {:.2} - {}", sig.symbol, sig.price, sig.reason)).await;
                    }
                }
                Err(e) => warn!("Sell failed: {}", e),
            }
        }
        Signal::Short => {
            match broker.sell(&sig.symbol, qty).await {
                Ok(order) => {
                    info!("SHORT ORDER: {:?} qty={}", order.order_id, qty);

                    portfolio.add_position(PortfolioPosition {
                        symbol: sig.symbol.clone(),
                        side: Side::Short,
                        quantity: qty,
                        entry_price: sig.price,
                        current_price: sig.price,
                        entry_time: Utc::now(),
                        market_value: qty * sig.price,
                        sector: Sector::from_symbol(&sig.symbol),
                    });

                    if telegram_enabled {
                        telegram::send_signal(
                            "SHORT", &sig.price.to_string(), "N/A", "N/A", 100,
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
            match broker.close_position(&sig.symbol).await {
                Ok(order) => {
                    info!("COVER ORDER: {:?}", order.order_id);
                    portfolio.remove_position(&sig.symbol);
                    if telegram_enabled {
                        let _ = telegram::send(&format!("COVER {} @ {:.2} - {}", sig.symbol, sig.price, sig.reason)).await;
                    }
                }
                Err(e) => warn!("Cover failed: {}", e),
            }
        }
        Signal::Hold => {}
    }
}
