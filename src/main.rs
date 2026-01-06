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

use crate::core::{SymbolAgent, AgentSignal, Signal, Side, Position, HealthMonitor, ConfidenceCalibrator, Calibrator, TransferManager, MixtureOfExperts, MetaLearner, WeaknessAnalyzer, CausalAnalyzer, WorldModel, CounterfactualAnalyzer, AGIMonitor, RegimePredictor, VectorIndex, IndexType, MemoryConsolidator, TransferabilityPredictor, SelfModificationEngine, Constitution};
use crate::core::health::HealthStatus;
use std::sync::Mutex;
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
const CALIBRATOR_PATH: &str = "sovereign_calibrator.json";
const TRANSFER_PATH: &str = "sovereign_transfer.json";
const MOE_PATH: &str = "sovereign_moe.json";
const META_PATH: &str = "sovereign_meta.json";
const WEAKNESS_PATH: &str = "sovereign_weakness.json";
const CAUSALITY_PATH: &str = "sovereign_causality.json";
const WORLDMODEL_PATH: &str = "sovereign_worldmodel.json";
const COUNTERFACTUAL_PATH: &str = "sovereign_counterfactual.json";
const MONITOR_PATH: &str = "sovereign_monitor.json";
const SEQUENCE_PATH: &str = "sovereign_sequence.json";
const EMBEDDINGS_PATH: &str = "sovereign_embeddings.bin";
const CONSOLIDATION_PATH: &str = "sovereign_consolidation.json";
const TRANSFERABILITY_PATH: &str = "sovereign_transferability.json";
const SELFMOD_PATH: &str = "sovereign_selfmod.json";

/// Save the best calibrator from all agents (one with most updates)
fn save_calibrator(agents: &HashMap<String, SymbolAgent>) {
    // Find the agent with the most calibrator updates
    let best = agents.values()
        .max_by_key(|a| a.calibrator().update_count());

    if let Some(agent) = best {
        let cal = agent.calibrator();
        if cal.update_count() > 0 {
            if let Err(e) = cal.save(CALIBRATOR_PATH) {
                warn!("Failed to save calibrator: {}", e);
            } else {
                info!("Calibrator: Saved {} updates to {}", cal.update_count(), CALIBRATOR_PATH);
            }
        }
    }
}

/// Save transfer manager state
fn save_transfer(transfer_manager: &Arc<Mutex<TransferManager>>) {
    let tm = transfer_manager.lock().unwrap();
    if let Err(e) = tm.save(TRANSFER_PATH) {
        warn!("Failed to save transfer state: {}", e);
    } else {
        info!("Transfer: Saved state - {}", tm.format_summary());
    }
}

/// Save the best MoE from all agents (one with most trades)
fn save_moe(agents: &HashMap<String, SymbolAgent>) {
    // Find the agent with the most MoE trades
    let best = agents.values()
        .filter_map(|a| a.moe().map(|m| (a, m)))
        .max_by_key(|(_, m)| m.total_trades());

    if let Some((agent, moe)) = best {
        if moe.total_trades() > 0 {
            if let Err(e) = moe.save(MOE_PATH) {
                warn!("Failed to save MoE: {}", e);
            } else {
                info!("MoE: Saved {} trades - {} (from {})",
                    moe.total_trades(), moe.format_stats(), agent.symbol());
            }
        }
    }
}

/// Save MetaLearner state
fn save_meta(meta_learner: &Arc<Mutex<MetaLearner>>) {
    let ml = meta_learner.lock().unwrap();
    if let Err(e) = ml.save(META_PATH) {
        warn!("Failed to save MetaLearner: {}", e);
    } else {
        info!("Meta: Saved - {}", ml.format_summary());
    }
}

/// Save WeaknessAnalyzer state
fn save_weakness(weakness_analyzer: &Arc<Mutex<WeaknessAnalyzer>>) {
    let wa = weakness_analyzer.lock().unwrap();
    if let Err(e) = wa.save(WEAKNESS_PATH) {
        warn!("Failed to save WeaknessAnalyzer: {}", e);
    } else {
        info!("Weakness: Saved - {} weaknesses identified", wa.weakness_count());
    }
}

/// Save CausalAnalyzer state
fn save_causality(causal_analyzer: &Arc<Mutex<CausalAnalyzer>>) {
    let ca = causal_analyzer.lock().unwrap();
    if let Err(e) = ca.save(CAUSALITY_PATH) {
        warn!("Failed to save CausalAnalyzer: {}", e);
    } else {
        info!("Causal: Saved - {}", ca.format_summary());
    }
}

/// Save WorldModel state
fn save_worldmodel(world_model: &Arc<Mutex<WorldModel>>) {
    let wm = world_model.lock().unwrap();
    let symbols = wm.get_symbols();
    // Serialize and save
    match serde_json::to_string_pretty(&*wm) {
        Ok(json) => {
            if let Err(e) = std::fs::write(WORLDMODEL_PATH, json) {
                warn!("Failed to save WorldModel: {}", e);
            } else {
                info!("WorldModel: Saved - {} symbols, equity {:.2}", symbols.len(), wm.get_equity());
            }
        }
        Err(e) => warn!("Failed to serialize WorldModel: {}", e),
    }
}

/// Load WorldModel from file
fn load_worldmodel(initial_equity: f64) -> WorldModel {
    match std::fs::read_to_string(WORLDMODEL_PATH) {
        Ok(json) => {
            match serde_json::from_str::<WorldModel>(&json) {
                Ok(wm) => {
                    info!("WorldModel: Loaded from {}", WORLDMODEL_PATH);
                    wm
                }
                Err(e) => {
                    warn!("Failed to parse WorldModel: {}, creating new", e);
                    WorldModel::new(initial_equity)
                }
            }
        }
        Err(_) => {
            info!("WorldModel: Creating new (no saved state)");
            WorldModel::new(initial_equity)
        }
    }
}

/// Save CounterfactualAnalyzer state
fn save_counterfactual(counterfactual: &Arc<Mutex<CounterfactualAnalyzer>>) {
    let cf = counterfactual.lock().unwrap();
    if let Err(e) = cf.save(COUNTERFACTUAL_PATH) {
        warn!("Failed to save CounterfactualAnalyzer: {}", e);
    } else {
        info!("Counterfactual: Saved - {}", cf.format_summary());
    }
}

/// Save AGIMonitor state
fn save_monitor(monitor: &Arc<Mutex<AGIMonitor>>) {
    let mon = monitor.lock().unwrap();
    if let Err(e) = mon.save(MONITOR_PATH) {
        warn!("Failed to save AGIMonitor: {}", e);
    } else {
        info!("Monitor: Saved - {}", mon.format_summary());
    }
}

/// Save RegimePredictor state
fn save_sequence(regime_predictor: &Arc<Mutex<RegimePredictor>>) {
    let rp = regime_predictor.lock().unwrap();
    if let Err(e) = rp.save(SEQUENCE_PATH) {
        warn!("Failed to save RegimePredictor: {}", e);
    } else {
        info!("Sequence: Saved - {} predictions, {:.1}% accuracy",
            rp.prediction_count(),
            rp.accuracy() * 100.0);
    }
}

/// Save VectorIndex state (binary for efficiency)
fn save_embeddings(vector_index: &Arc<Mutex<VectorIndex>>) {
    let idx = vector_index.lock().unwrap();
    if let Err(e) = idx.save(EMBEDDINGS_PATH) {
        warn!("Failed to save VectorIndex: {}", e);
    } else {
        info!("Embeddings: Saved - {}", idx.format_summary());
    }
}

/// Save MemoryConsolidator state
fn save_consolidation(memory_consolidator: &Arc<Mutex<MemoryConsolidator>>) {
    let mc = memory_consolidator.lock().unwrap();
    if let Err(e) = mc.save(CONSOLIDATION_PATH) {
        warn!("Failed to save MemoryConsolidator: {}", e);
    } else {
        info!("Consolidation: Saved - {}", mc.format_summary());
    }
}

/// Run consolidation on the memory consolidator (extract patterns)
fn run_consolidation(memory_consolidator: &Arc<Mutex<MemoryConsolidator>>) {
    let mut mc = memory_consolidator.lock().unwrap();
    let pre_patterns = mc.pattern_count();
    mc.consolidate();
    let post_patterns = mc.pattern_count();
    info!(
        "Consolidation: Extracted {} new patterns ({} -> {} total)",
        post_patterns.saturating_sub(pre_patterns),
        pre_patterns,
        post_patterns
    );
}

/// Save TransferabilityPredictor state
fn save_transferability(transferability_predictor: &Arc<Mutex<TransferabilityPredictor>>) {
    let tp = transferability_predictor.lock().unwrap();
    if let Err(e) = tp.save(TRANSFERABILITY_PATH) {
        warn!("Failed to save TransferabilityPredictor: {}", e);
    } else {
        info!("Transferability: Saved - {}", tp.format_summary());
    }
}

/// Save SelfModificationEngine state
fn save_selfmod(selfmod: &Arc<Mutex<SelfModificationEngine>>) {
    let engine = selfmod.lock().unwrap();
    if let Err(e) = engine.save(SELFMOD_PATH) {
        warn!("Failed to save SelfModificationEngine: {}", e);
    } else {
        info!("SelfMod: Saved - {} rules active, {} pending",
            engine.rule_engine().active_count(),
            engine.guard().pending_count()
        );
    }
}

/// Run daily self-modification analysis and proposal generation
fn run_analyze_and_propose(
    selfmod: &Arc<Mutex<SelfModificationEngine>>,
    weakness_analyzer: &Arc<Mutex<crate::core::WeaknessAnalyzer>>,
    counterfactual: &Arc<Mutex<crate::core::CounterfactualAnalyzer>>,
) {
    let mut engine = selfmod.lock().unwrap();
    let wa = weakness_analyzer.lock().unwrap();
    let cf = counterfactual.lock().unwrap();

    // Get weaknesses and insights for rule generation
    let weaknesses = wa.get_weaknesses();
    let insights = cf.get_insights();

    let mut proposal_count = 0;

    // Generate rules from weaknesses
    for weakness in weaknesses {
        if let Some(rule) = engine.generate_rule_from_weakness(weakness) {
            info!("[SELF-MOD] Proposed rule from weakness: {}", rule.name);
            proposal_count += 1;
            // Apply the rule addition
            let _ = engine.apply_rule_addition(rule);
        }
    }

    // Generate threshold changes from insights
    for insight in insights {
        if let Some(mod_type) = engine.generate_threshold_change(insight) {
            info!("[SELF-MOD] Proposed threshold change: {:?}", mod_type);
            proposal_count += 1;
        }
    }

    if proposal_count > 0 {
        info!("[SELF-MOD] Generated {} proposals from daily analysis", proposal_count);
    }
}

/// Run weekly learning from transfer outcomes
fn learn_from_transfers(transfer_manager: &Arc<Mutex<TransferManager>>) {
    let tm = transfer_manager.lock().unwrap();
    tm.learn_from_outcomes();
    info!("Transfer: Triggered learning from outcomes");
}

/// Discover and log ML-based clusters
fn discover_ml_clusters(transfer_manager: &Arc<Mutex<TransferManager>>) {
    let tm = transfer_manager.lock().unwrap();
    let clusters = tm.discover_clusters_ml();
    if clusters.is_empty() {
        info!("Transfer: No ML clusters discovered (insufficient data)");
    } else {
        info!("Transfer: Discovered {} ML clusters:", clusters.len());
        for (i, cluster) in clusters.iter().enumerate() {
            info!("  Cluster {}: {} symbols - {:?}",
                i + 1, cluster.len(), cluster.iter().take(5).cloned().collect::<Vec<_>>());
        }
    }
}

/// Process Telegram self-modification commands
async fn process_telegram_commands(selfmod: &Arc<Mutex<SelfModificationEngine>>) {
    let commands = telegram::poll_commands().await;

    for cmd in commands {
        match cmd.command.as_str() {
            "/pending" => {
                let engine = selfmod.lock().unwrap();
                let pending: Vec<(String, String, String)> = engine
                    .get_pending()
                    .iter()
                    .map(|p| (
                        p.id.to_string(),
                        format!("{:?}", p.modification),
                        p.reason.clone(),
                    ))
                    .collect();
                drop(engine);
                telegram::send_pending_mods(&pending).await;
            }
            "/rules" => {
                let engine = selfmod.lock().unwrap();
                let rules: Vec<(String, String, String, u32)> = engine
                    .get_active_rules()
                    .iter()
                    .map(|r| (
                        r.name.clone(),
                        format!("{:?}", r.condition),
                        format!("{:?}", r.action),
                        r.performance.times_triggered,
                    ))
                    .collect();
                drop(engine);
                telegram::send_rules(&rules).await;
            }
            "/constitution" => {
                let engine = selfmod.lock().unwrap();
                let c = engine.guard().constitution();
                telegram::send_constitution(
                    c.max_position_size,
                    c.max_daily_loss,
                    c.max_drawdown,
                    c.min_confidence_for_trade,
                    c.max_active_rules as usize,
                    c.forbidden_modifications.len(),
                ).await;
            }
            "/approve" => {
                if let Some(id_str) = cmd.args.first() {
                    if let Ok(id) = id_str.parse::<u64>() {
                        let mut engine = selfmod.lock().unwrap();
                        match engine.approve_pending(id, "telegram") {
                            Ok(()) => {
                                drop(engine);
                                telegram::send_approval(id_str, true, "Modification approved and applied.").await;
                            }
                            Err(e) => {
                                drop(engine);
                                telegram::send_approval(id_str, false, &format!("Error: {}", e)).await;
                            }
                        }
                    } else {
                        telegram::send_approval(id_str, false, "Invalid ID format").await;
                    }
                } else {
                    telegram::send_approval("", false, "Usage: /approve <id>").await;
                }
            }
            "/reject" => {
                if let Some(id_str) = cmd.args.first() {
                    if let Ok(id) = id_str.parse::<u64>() {
                        let mut engine = selfmod.lock().unwrap();
                        match engine.reject_pending(id, "rejected via telegram") {
                            Ok(()) => {
                                drop(engine);
                                telegram::send_rejection(id_str, true, "Modification rejected.").await;
                            }
                            Err(e) => {
                                drop(engine);
                                telegram::send_rejection(id_str, false, &format!("Error: {}", e)).await;
                            }
                        }
                    } else {
                        telegram::send_rejection(id_str, false, "Invalid ID format").await;
                    }
                } else {
                    telegram::send_rejection("", false, "Usage: /reject <id>").await;
                }
            }
            "/rollback" => {
                if let Some(id_str) = cmd.args.first() {
                    if let Ok(id) = id_str.parse::<u64>() {
                        let mut engine = selfmod.lock().unwrap();
                        match engine.rollback_modification(id) {
                            Ok(()) => {
                                drop(engine);
                                telegram::send_rollback(id_str, true, "Modification rolled back.").await;
                            }
                            Err(e) => {
                                drop(engine);
                                telegram::send_rollback(id_str, false, &format!("Error: {}", e)).await;
                            }
                        }
                    } else {
                        telegram::send_rollback(id_str, false, "Invalid ID format").await;
                    }
                } else {
                    telegram::send_rollback("", false, "Usage: /rollback <id>").await;
                }
            }
            "/selfmod" | "/selfmodhelp" => {
                telegram::send_selfmod_help().await;
            }
            _ => {}
        }
    }
}

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

    // Load MetaLearner for rapid adaptation (must be before calibrator init)
    let meta_learner = Arc::new(Mutex::new(MetaLearner::load_or_new(META_PATH)));
    {
        let ml = meta_learner.lock().unwrap();
        info!("Meta: Loaded - {}", ml.format_summary());
    }

    // Attach MetaLearner to all agents
    for agent in agents.values_mut() {
        agent.attach_meta_learner(Arc::clone(&meta_learner));
    }

    // Load learned calibrator weights if available
    let calibrator = ConfidenceCalibrator::load_or_new(CALIBRATOR_PATH);
    if calibrator.update_count() > 0 {
        info!("Calibrator: Loaded with {} updates, applying to all agents", calibrator.update_count());
        for agent in agents.values_mut() {
            agent.set_calibrator(calibrator.clone());
        }
    } else {
        info!("Calibrator: Starting fresh (no saved weights found)");
    }

    // Load transferability predictor for ML-based transfer decisions
    let transferability_predictor = Arc::new(Mutex::new(
        TransferabilityPredictor::load_or_new(TRANSFERABILITY_PATH)
    ));
    {
        let tp = transferability_predictor.lock().unwrap();
        info!("Transferability: Loaded - {}", tp.format_summary());
    }

    // Load transfer manager for cross-symbol knowledge transfer
    let transfer_manager = Arc::new(Mutex::new(TransferManager::load_or_new(TRANSFER_PATH)));
    {
        let mut tm = transfer_manager.lock().unwrap();
        // Attach ML predictor to transfer manager
        tm.attach_predictor(Arc::clone(&transferability_predictor));
        info!("Transfer: {} (ML predictions enabled)", tm.format_summary());
    }

    // Attach transfer manager to all agents and try cluster initialization
    for agent in agents.values_mut() {
        agent.attach_transfer_manager(Arc::clone(&transfer_manager));
        agent.maybe_init_from_cluster();
    }

    // Load MoE for regime-specialized calibration
    let moe = MixtureOfExperts::load_or_new(MOE_PATH);
    if moe.total_trades() > 0 {
        info!("MoE: Loaded with {} trades - {}", moe.total_trades(), moe.format_stats());
        for agent in agents.values_mut() {
            agent.attach_moe(moe.clone());
            agent.enable_moe();
        }
    } else {
        info!("MoE: Starting fresh (no saved state found), attaching empty MoE");
        for agent in agents.values_mut() {
            agent.attach_moe(MixtureOfExperts::new());
            agent.enable_moe();
        }
    }

    // Load WeaknessAnalyzer for self-directed improvement
    let weakness_analyzer = Arc::new(Mutex::new(WeaknessAnalyzer::load_or_new(WEAKNESS_PATH)));
    {
        let mut wa = weakness_analyzer.lock().unwrap();
        // Set memory reference and symbols for live analysis
        wa.set_memory(Arc::clone(&memory));
        wa.set_symbols(symbols.clone());
        info!("Weakness: Loaded - {} weaknesses identified", wa.weakness_count());
    }

    // Attach WeaknessAnalyzer to all agents
    for agent in agents.values_mut() {
        agent.attach_weakness_analyzer(Arc::clone(&weakness_analyzer));
    }

    // Load CausalAnalyzer for understanding market relationships
    let causal_analyzer = Arc::new(Mutex::new(CausalAnalyzer::load_or_new(CAUSALITY_PATH)));
    {
        let ca = causal_analyzer.lock().unwrap();
        info!("Causal: Loaded - {}", ca.format_summary());
    }

    // Attach CausalAnalyzer to all agents
    for agent in agents.values_mut() {
        agent.attach_causal_analyzer(Arc::clone(&causal_analyzer));
    }

    // Load WorldModel for forward planning
    let initial_equity_f64 = initial_balance.to_f64().unwrap_or(100000.0);
    let world_model = Arc::new(Mutex::new(load_worldmodel(initial_equity_f64)));
    {
        let wm = world_model.lock().unwrap();
        info!("WorldModel: Loaded - {} symbols, equity {:.2}", wm.get_symbols().len(), wm.get_equity());
    }

    // Attach WorldModel to all agents
    for agent in agents.values_mut() {
        agent.attach_world_model(Arc::clone(&world_model));
    }

    // Load CounterfactualAnalyzer for learning from alternative decisions
    let counterfactual = Arc::new(Mutex::new(
        CounterfactualAnalyzer::load_or_new(COUNTERFACTUAL_PATH, Arc::clone(&memory))
    ));
    {
        let cf = counterfactual.lock().unwrap();
        info!("Counterfactual: Loaded - {}", cf.format_summary());
    }

    // Attach CounterfactualAnalyzer to all agents
    for agent in agents.values_mut() {
        agent.attach_counterfactual_analyzer(Arc::clone(&counterfactual));
    }

    // Load AGI Monitor for comprehensive system monitoring
    let agi_monitor = Arc::new(Mutex::new(AGIMonitor::load_or_new(MONITOR_PATH)));
    {
        let mut mon = agi_monitor.lock().unwrap();
        // Attach all components
        mon.attach_memory(Arc::clone(&memory));
        mon.attach_calibrator(calibrator.clone());
        mon.attach_moe(moe.clone());
        mon.attach_meta_learner(Arc::clone(&meta_learner));
        mon.attach_transfer_manager(Arc::clone(&transfer_manager));
        mon.attach_weakness_analyzer(Arc::clone(&weakness_analyzer));
        mon.attach_causal_analyzer(Arc::clone(&causal_analyzer));
        mon.attach_world_model(Arc::clone(&world_model));
        mon.attach_counterfactual(Arc::clone(&counterfactual));
        info!("Monitor: Loaded - {}", mon.format_summary());
    }

    // Load RegimePredictor for LSTM-based regime transition prediction
    let regime_predictor = Arc::new(Mutex::new(RegimePredictor::load_or_new(SEQUENCE_PATH)));
    {
        let rp = regime_predictor.lock().unwrap();
        info!("Sequence: Loaded - {} predictions, {:.1}% accuracy",
            rp.prediction_count(),
            rp.accuracy() * 100.0);
    }

    // Attach RegimePredictor to all agents
    for agent in agents.values_mut() {
        agent.attach_regime_predictor(Arc::clone(&regime_predictor));
    }

    // Load VectorIndex for similarity-based trade retrieval
    let vector_index = Arc::new(Mutex::new(
        VectorIndex::load_or_new(EMBEDDINGS_PATH, IndexType::HNSW { m: 16, ef_construction: 200 })
    ));
    {
        let idx = vector_index.lock().unwrap();
        info!("Embeddings: Loaded - {}", idx.format_summary());
    }

    // Attach VectorIndex to all agents
    for agent in agents.values_mut() {
        agent.attach_vector_index(Arc::clone(&vector_index));
    }

    // Load MemoryConsolidator for hierarchical memory and pattern extraction
    let memory_consolidator = Arc::new(Mutex::new(
        MemoryConsolidator::load_or_new(CONSOLIDATION_PATH, Arc::clone(&vector_index))
    ));
    {
        let mc = memory_consolidator.lock().unwrap();
        info!("Consolidation: Loaded - {}", mc.format_summary());
    }

    // Attach MemoryConsolidator to all agents
    for agent in agents.values_mut() {
        agent.attach_memory_consolidator(Arc::clone(&memory_consolidator));
    }

    // Load SelfModificationEngine with conservative constitution
    let constitution = Constitution::default();
    let selfmod = Arc::new(Mutex::new(
        SelfModificationEngine::load_or_new(SELFMOD_PATH, constitution)
    ));
    {
        let engine = selfmod.lock().unwrap();
        info!("SelfMod: Loaded - {} rules active, {} pending, {} applied",
            engine.rule_engine().active_count(),
            engine.guard().pending_count(),
            engine.applied_count()
        );
    }

    // Attach SelfModificationEngine to all agents
    for agent in agents.values_mut() {
        agent.attach_self_mod(Arc::clone(&selfmod));
    }

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
                &transfer_manager,
                &meta_learner,
                &weakness_analyzer,
                &causal_analyzer,
                &world_model,
                &counterfactual,
                &agi_monitor,
                &regime_predictor,
                &vector_index,
                &memory_consolidator,
                &transferability_predictor,
                &selfmod,
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
                &transfer_manager,
                &meta_learner,
                &weakness_analyzer,
                &causal_analyzer,
                &world_model,
                &counterfactual,
                &agi_monitor,
                &regime_predictor,
                &vector_index,
                &memory_consolidator,
                &transferability_predictor,
                &selfmod,
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
    transfer_manager: &Arc<Mutex<TransferManager>>,
    meta_learner: &Arc<Mutex<MetaLearner>>,
    weakness_analyzer: &Arc<Mutex<WeaknessAnalyzer>>,
    causal_analyzer: &Arc<Mutex<CausalAnalyzer>>,
    world_model: &Arc<Mutex<WorldModel>>,
    counterfactual: &Arc<Mutex<CounterfactualAnalyzer>>,
    agi_monitor: &Arc<Mutex<AGIMonitor>>,
    regime_predictor: &Arc<Mutex<RegimePredictor>>,
    vector_index: &Arc<Mutex<VectorIndex>>,
    memory_consolidator: &Arc<Mutex<MemoryConsolidator>>,
    transferability_predictor: &Arc<Mutex<TransferabilityPredictor>>,
    selfmod: &Arc<Mutex<SelfModificationEngine>>,
) -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<AlpacaMessage>(100);

    let alpaca_cfg = cfg.alpaca.clone().expect("Alpaca config required");
    let api_key = alpaca_cfg.api_key.clone();
    let api_secret = alpaca_cfg.secret_key.clone();
    let symbols_clone: Vec<String> = cfg.universe.symbols.clone();

    let mut last_health_check = std::time::Instant::now();
    let mut last_hourly_snapshot = std::time::Instant::now();
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

            // Poll for Telegram commands
            if telegram_enabled {
                process_telegram_commands(selfmod).await;
            }

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

        // Hourly snapshot for AGI monitoring
        if last_hourly_snapshot.elapsed().as_secs() >= 3600 {
            last_hourly_snapshot = std::time::Instant::now();
            let mut mon = agi_monitor.lock().unwrap();
            mon.snapshot_hourly();
        }

        // Daily summary at 21:05 UTC (5 min after market close)
        let now = Utc::now();
        let today = now.date_naive();
        if now.hour() == 21 && now.minute() >= 5 && now.minute() < 15 {
            if last_summary_date != Some(today) {
                last_summary_date = Some(today);

                // Daily snapshot for AGI monitoring
                {
                    let mut mon = agi_monitor.lock().unwrap();
                    mon.snapshot_daily();
                }

                // Save learned calibrator weights, transfer state, MoE, MetaLearner, WeaknessAnalyzer, CausalAnalyzer, WorldModel, Counterfactual, and Monitor
                save_calibrator(agents);
                save_transfer(transfer_manager);
                save_moe(agents);
                save_meta(meta_learner);
                save_weakness(weakness_analyzer);
                save_causality(causal_analyzer);
                save_worldmodel(world_model);
                save_counterfactual(counterfactual);
                save_monitor(agi_monitor);
                save_sequence(regime_predictor);
                save_embeddings(vector_index);
                save_consolidation(memory_consolidator);
                save_transferability(&transferability_predictor);
                save_selfmod(selfmod);

                // Run pattern consolidation
                run_consolidation(memory_consolidator);

                // Run daily self-modification analysis
                run_analyze_and_propose(selfmod, weakness_analyzer, counterfactual);

                // Run weekly ML transfer learning and cluster discovery
                learn_from_transfers(transfer_manager);
                discover_ml_clusters(transfer_manager);

                // Run periodic weakness analysis on trade history
                {
                    let mut wa = weakness_analyzer.lock().unwrap();
                    // Get trade history from all agents to analyze weaknesses
                    // (WeaknessAnalyzer accumulates from record_trade calls)
                    let weaknesses = wa.analyze_all();
                    if !weaknesses.is_empty() {
                        info!("[WEAKNESS] Identified {} weakness patterns:", weaknesses.len());
                        for w in weaknesses.iter().take(5) {
                            info!("  - {:?}: {:.1}% severity - {}", w.weakness_type, w.severity * 100.0, w.suggested_action);
                        }
                    }
                }

                // Run counterfactual analysis on recent trades
                {
                    let mut cf = counterfactual.lock().unwrap();
                    let insights = cf.analyze_all_recent(50);
                    if !insights.is_empty() {
                        info!("[COUNTERFACTUAL] Identified {} patterns:", insights.len());
                        for insight in insights.iter().take(5) {
                            info!("  - {:?}: {} trades, ${:.2} avg improvement - {}",
                                insight.insight_type, insight.evidence_count, insight.avg_improvement, insight.description);
                        }
                    }
                    // Log recommendations
                    let recommendations = cf.get_recommendations();
                    if !recommendations.is_empty() {
                        info!("[COUNTERFACTUAL] Top recommendations:");
                        for rec in recommendations.iter().take(3) {
                            info!("  - {}", rec);
                        }
                    }
                }

                // Run weekly causal discovery on Sundays
                if now.weekday() == Weekday::Sun {
                    let mut ca = causal_analyzer.lock().unwrap();
                    let new_relationships = ca.discover_relationships();
                    if !new_relationships.is_empty() {
                        info!("[CAUSAL] Discovered {} new relationships:", new_relationships.len());
                        for rel in new_relationships.iter().take(5) {
                            info!("  - {}", rel.description());
                        }
                    }
                    // Prune old relationships (older than 90 days)
                    ca.graph_mut().prune_old(90);
                }

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
    transfer_manager: &Arc<Mutex<TransferManager>>,
    meta_learner: &Arc<Mutex<MetaLearner>>,
    weakness_analyzer: &Arc<Mutex<WeaknessAnalyzer>>,
    causal_analyzer: &Arc<Mutex<CausalAnalyzer>>,
    world_model: &Arc<Mutex<WorldModel>>,
    counterfactual: &Arc<Mutex<CounterfactualAnalyzer>>,
    agi_monitor: &Arc<Mutex<AGIMonitor>>,
    regime_predictor: &Arc<Mutex<RegimePredictor>>,
    vector_index: &Arc<Mutex<VectorIndex>>,
    memory_consolidator: &Arc<Mutex<MemoryConsolidator>>,
    transferability_predictor: &Arc<Mutex<TransferabilityPredictor>>,
    selfmod: &Arc<Mutex<SelfModificationEngine>>,
) -> Result<()> {
    let symbols: Vec<String> = cfg.universe.symbols.clone();
    let mut last_health_check = std::time::Instant::now();
    let mut last_hourly_snapshot = std::time::Instant::now();
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

            // Poll for Telegram commands
            if telegram_enabled {
                process_telegram_commands(selfmod).await;
            }

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

        // Hourly snapshot for AGI monitoring
        if last_hourly_snapshot.elapsed().as_secs() >= 3600 {
            last_hourly_snapshot = std::time::Instant::now();
            let mut mon = agi_monitor.lock().unwrap();
            mon.snapshot_hourly();
        }

        // Daily summary at 21:05 UTC (5 min after market close)
        let now = Utc::now();
        let today = now.date_naive();
        if now.hour() == 21 && now.minute() >= 5 && now.minute() < 15 {
            if last_summary_date != Some(today) {
                last_summary_date = Some(today);

                // Daily snapshot for AGI monitoring
                {
                    let mut mon = agi_monitor.lock().unwrap();
                    mon.snapshot_daily();
                }

                // Save learned calibrator weights, transfer state, MoE, MetaLearner, WeaknessAnalyzer, CausalAnalyzer, WorldModel, Counterfactual, and Monitor
                save_calibrator(agents);
                save_transfer(transfer_manager);
                save_moe(agents);
                save_meta(meta_learner);
                save_weakness(weakness_analyzer);
                save_causality(causal_analyzer);
                save_worldmodel(world_model);
                save_counterfactual(counterfactual);
                save_monitor(agi_monitor);
                save_sequence(regime_predictor);
                save_embeddings(vector_index);
                save_consolidation(memory_consolidator);
                save_transferability(&transferability_predictor);
                save_selfmod(selfmod);

                // Run pattern consolidation
                run_consolidation(memory_consolidator);

                // Run daily self-modification analysis
                run_analyze_and_propose(selfmod, weakness_analyzer, counterfactual);

                // Run weekly ML transfer learning and cluster discovery
                learn_from_transfers(transfer_manager);
                discover_ml_clusters(transfer_manager);

                // Run periodic weakness analysis on trade history
                {
                    let mut wa = weakness_analyzer.lock().unwrap();
                    let weaknesses = wa.analyze_all();
                    if !weaknesses.is_empty() {
                        info!("[WEAKNESS] Identified {} weakness patterns:", weaknesses.len());
                        for w in weaknesses.iter().take(5) {
                            info!("  - {:?}: {:.1}% severity - {}", w.weakness_type, w.severity * 100.0, w.suggested_action);
                        }
                    }
                }

                // Run counterfactual analysis on recent trades
                {
                    let mut cf = counterfactual.lock().unwrap();
                    let insights = cf.analyze_all_recent(50);
                    if !insights.is_empty() {
                        info!("[COUNTERFACTUAL] Identified {} patterns:", insights.len());
                        for insight in insights.iter().take(5) {
                            info!("  - {:?}: {} trades, ${:.2} avg improvement - {}",
                                insight.insight_type, insight.evidence_count, insight.avg_improvement, insight.description);
                        }
                    }
                    // Log recommendations
                    let recommendations = cf.get_recommendations();
                    if !recommendations.is_empty() {
                        info!("[COUNTERFACTUAL] Top recommendations:");
                        for rec in recommendations.iter().take(3) {
                            info!("  - {}", rec);
                        }
                    }
                }

                // Run weekly causal discovery on Sundays
                if now.weekday() == Weekday::Sun {
                    let mut ca = causal_analyzer.lock().unwrap();
                    let new_relationships = ca.discover_relationships();
                    if !new_relationships.is_empty() {
                        info!("[CAUSAL] Discovered {} new relationships:", new_relationships.len());
                        for rel in new_relationships.iter().take(5) {
                            info!("  - {}", rel.description());
                        }
                    }
                    // Prune old relationships (older than 90 days)
                    ca.graph_mut().prune_old(90);
                }

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
