//! Telegram notifications for Sovereign v4

use anyhow::Result;
use tracing::warn;

const BOT_TOKEN: &str = "8570067655:AAHezpMffYJIc6oHkhmAX-MwXIpI91TvzR8";
const CHAT_ID: &str = "7898079111";

pub async fn send(message: &str) -> Result<()> {
    let url = format!(
        "https://api.telegram.org/bot{}/sendMessage",
        BOT_TOKEN
    );
    
    println!("[TELEGRAM] Sending: {}", message);
    
    let client = reqwest::Client::new();
    let params = [
        ("chat_id", CHAT_ID),
        ("text", message),
        ("parse_mode", "HTML"),
    ];
    
    match client.post(&url).form(&params).send().await {
        Ok(resp) => {
            println!("[TELEGRAM] Response: {}", resp.status());
            if !resp.status().is_success() {
                warn!("Telegram send failed: {}", resp.status());
            }
        }
        Err(e) => {
            println!("[TELEGRAM] Error: {}", e);
            warn!("Telegram error: {}", e);
        }
    }
    
    Ok(())
}

pub async fn send_signal(direction: &str, price: &str, sl: &str, tp: &str, conviction: u8) {
    let emoji = match direction {
        "Buy" => "🟢",
        "Sell" => "🔴",
        _ => "⚪",
    };
    
    let msg = format!(
        "{} <b>SOVEREIGN v4</b>\n\n\
        Signal: <b>{}</b>\n\
        Price: {}\n\
        SL: {}\n\
        TP: {}\n\
        Conviction: {}%",
        emoji, direction, price, sl, tp, conviction
    );
    
    let _ = send(&msg).await;
}

pub async fn send_fill(direction: &str, ticket: u64, price: &str) {
    let msg = format!(
        "✅ <b>ORDER FILLED</b>\n\n\
        Direction: {}\n\
        Ticket: {}\n\
        Price: {}",
        direction, ticket, price
    );
    
    let _ = send(&msg).await;
}

pub async fn send_startup() {
    let msg = "🚀 <b>Sovereign v4</b> started\n\nListening for signals...";
    let _ = send(msg).await;
}

pub async fn send_daily_summary(
    positions: usize,
    long_exposure: f64,
    short_exposure: f64,
    unrealized_pnl: rust_decimal::Decimal,
    total_bars: u64,
    sector_info: &str,
) {
    let sector_line = if sector_info.is_empty() || sector_info == "No sector exposure" {
        String::new()
    } else {
        format!("\nSectors: {}", sector_info)
    };

    let msg = format!(
        "📊 <b>Daily Summary</b>\n\n\
        Positions: {}\n\
        Exposure: {:.0}% long / {:.0}% short\n\
        Unrealized P&L: ${:.2}\n\
        Bars Processed: {}{}",
        positions,
        long_exposure * 100.0,
        short_exposure * 100.0,
        unrealized_pnl,
        total_bars,
        sector_line
    );
    let _ = send(&msg).await;
}

// ==================== AGI Notification Functions ====================

/// Send AGI progress summary
pub async fn send_agi_summary(
    agi_progress_pct: f64,
    learning_velocity: f64,
    total_trades: u32,
    win_rate: f64,
) {
    let progress_bar = render_progress_bar(agi_progress_pct, 20);
    let velocity_emoji = if learning_velocity > 0.01 { "📈" } else if learning_velocity < -0.01 { "📉" } else { "➡️" };

    let msg = format!(
        "🤖 <b>AGI Progress</b>\n\n\
        Progress: {} {:.1}%\n\
        Learning: {} {:.3}\n\n\
        📊 Trades: {} | Win Rate: {:.1}%",
        progress_bar, agi_progress_pct * 100.0,
        velocity_emoji, learning_velocity,
        total_trades, win_rate * 100.0
    );
    let _ = send(&msg).await;
}

/// Send AGI metrics summary
pub async fn send_agi_metrics(
    calibrator_updates: u32,
    calibrator_accuracy: f64,
    meta_adaptations: u32,
    weaknesses_found: u32,
    causal_relationships: u32,
    insights_generated: u32,
) {
    let msg = format!(
        "📈 <b>AGI Metrics</b>\n\n\
        🎯 Calibrator: {} updates ({:.0}% accuracy)\n\
        🧠 Meta-Learning: {} adaptations\n\
        ⚠️ Weaknesses: {} identified\n\
        🔗 Causal: {} relationships\n\
        💡 Insights: {} generated",
        calibrator_updates, calibrator_accuracy * 100.0,
        meta_adaptations,
        weaknesses_found,
        causal_relationships,
        insights_generated
    );
    let _ = send(&msg).await;
}

/// Send system health status
pub async fn send_health_status(
    uptime_hours: u64,
    components_healthy: usize,
    total_components: usize,
    errors_last_hour: u32,
) {
    let health_emoji = if components_healthy == total_components { "✅" } else { "⚠️" };

    let msg = format!(
        "🏥 <b>System Health</b>\n\n\
        {} Components: {}/{}\n\
        ⏱️ Uptime: {}h\n\
        ❌ Errors (1h): {}",
        health_emoji, components_healthy, total_components,
        uptime_hours,
        errors_last_hour
    );
    let _ = send(&msg).await;
}

/// Send top weaknesses
pub async fn send_weaknesses(weaknesses: &[(String, f64)]) {
    if weaknesses.is_empty() {
        let _ = send("✅ <b>No Weaknesses</b>\n\nNo significant weaknesses identified.").await;
        return;
    }

    let mut msg = String::from("⚠️ <b>Top Weaknesses</b>\n\n");
    for (i, (desc, severity)) in weaknesses.iter().take(5).enumerate() {
        let severity_bar = render_progress_bar(*severity, 10);
        msg.push_str(&format!("{}. {} {:.0}%\n   {}\n\n", i + 1, severity_bar, severity * 100.0, desc));
    }
    let _ = send(&msg).await;
}

/// Send counterfactual insights
pub async fn send_insights(insights: &[(String, f64)]) {
    if insights.is_empty() {
        let _ = send("📊 <b>No Insights</b>\n\nNo counterfactual insights yet.").await;
        return;
    }

    let mut msg = String::from("💡 <b>Counterfactual Insights</b>\n\n");
    for (i, (desc, improvement)) in insights.iter().take(5).enumerate() {
        let emoji = if *improvement > 0.0 { "📈" } else { "📉" };
        msg.push_str(&format!("{}. {} ${:.2} potential\n   {}\n\n", i + 1, emoji, improvement, desc));
    }
    let _ = send(&msg).await;
}

/// Render an ASCII progress bar
fn render_progress_bar(progress: f64, width: usize) -> String {
    let filled = (progress * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

// ==================== Self-Modification Commands ====================

use std::sync::atomic::{AtomicI64, Ordering};

/// Last processed update ID for command polling
static LAST_UPDATE_ID: AtomicI64 = AtomicI64::new(0);

/// Command received from Telegram
#[derive(Debug, Clone)]
pub struct TelegramCommand {
    pub command: String,
    pub args: Vec<String>,
    pub chat_id: String,
}

/// Poll for new commands from Telegram
pub async fn poll_commands() -> Vec<TelegramCommand> {
    let url = format!(
        "https://api.telegram.org/bot{}/getUpdates?offset={}&timeout=1",
        BOT_TOKEN,
        LAST_UPDATE_ID.load(Ordering::Relaxed) + 1
    );

    let client = reqwest::Client::new();
    match client.get(&url).send().await {
        Ok(resp) => {
            if let Ok(body) = resp.text().await {
                parse_updates(&body)
            } else {
                Vec::new()
            }
        }
        Err(_) => Vec::new(),
    }
}

/// Parse updates from Telegram API response
fn parse_updates(json: &str) -> Vec<TelegramCommand> {
    let mut commands = Vec::new();

    // Simple JSON parsing without serde
    if !json.contains("\"ok\":true") {
        return commands;
    }

    // Parse update_id and text from each result
    for result_chunk in json.split("\"update_id\":").skip(1) {
        // Extract update_id
        if let Some(id_end) = result_chunk.find(',') {
            if let Ok(update_id) = result_chunk[..id_end].trim().parse::<i64>() {
                LAST_UPDATE_ID.store(update_id, Ordering::Relaxed);

                // Extract message text if present
                if let Some(text_start) = result_chunk.find("\"text\":\"") {
                    let text_portion = &result_chunk[text_start + 8..];
                    if let Some(text_end) = text_portion.find('"') {
                        let text = &text_portion[..text_end];

                        // Check if it's a command (starts with /)
                        if text.starts_with('/') {
                            let parts: Vec<&str> = text.split_whitespace().collect();
                            if !parts.is_empty() {
                                commands.push(TelegramCommand {
                                    command: parts[0].to_lowercase(),
                                    args: parts[1..].iter().map(|s| s.to_string()).collect(),
                                    chat_id: CHAT_ID.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    commands
}

/// Send pending modifications list
pub async fn send_pending_mods(pending: &[(String, String, String)]) {
    if pending.is_empty() {
        let _ = send("⏳ <b>No Pending Modifications</b>\n\nAll modifications have been processed.").await;
        return;
    }

    let mut msg = String::from("⏳ <b>Pending Modifications</b>\n\n");
    for (i, (id, mod_type, description)) in pending.iter().enumerate() {
        msg.push_str(&format!(
            "{}. <b>{}</b>\n   ID: <code>{}</code>\n   {}\n\n",
            i + 1, mod_type, id, description
        ));
    }
    msg.push_str("\nUse /approve &lt;id&gt; or /reject &lt;id&gt; to process.");
    let _ = send(&msg).await;
}

/// Send active rules list
pub async fn send_rules(rules: &[(String, String, String, u32)]) {
    if rules.is_empty() {
        let _ = send("📋 <b>No Active Rules</b>\n\nNo self-modification rules active.").await;
        return;
    }

    let mut msg = String::from("📋 <b>Active Rules</b>\n\n");
    for (i, (name, condition, action, triggered)) in rules.iter().enumerate() {
        msg.push_str(&format!(
            "{}. <b>{}</b>\n   When: {}\n   Then: {}\n   Triggered: {} times\n\n",
            i + 1, name, condition, action, triggered
        ));
    }
    let _ = send(&msg).await;
}

/// Send constitution summary
pub async fn send_constitution(
    max_position_pct: f64,
    max_daily_loss_pct: f64,
    max_drawdown_pct: f64,
    min_confidence: f64,
    max_auto_rules: usize,
    forbidden_count: usize,
) {
    let msg = format!(
        "📜 <b>Constitution</b>\n\n\
        🔒 Max Position: {:.0}%\n\
        🔒 Max Daily Loss: {:.0}%\n\
        🔒 Max Drawdown: {:.0}%\n\
        🔒 Min Confidence: {:.0}%\n\
        🤖 Max Auto-Rules: {}\n\
        🚫 Forbidden Types: {}",
        max_position_pct * 100.0,
        max_daily_loss_pct * 100.0,
        max_drawdown_pct * 100.0,
        min_confidence * 100.0,
        max_auto_rules,
        forbidden_count
    );
    let _ = send(&msg).await;
}

/// Send approval confirmation
pub async fn send_approval(id: &str, success: bool, message: &str) {
    let emoji = if success { "✅" } else { "❌" };
    let status = if success { "Approved" } else { "Failed" };
    let msg = format!(
        "{} <b>Modification {}</b>\n\nID: <code>{}</code>\n{}",
        emoji, status, id, message
    );
    let _ = send(&msg).await;
}

/// Send rejection confirmation
pub async fn send_rejection(id: &str, success: bool, message: &str) {
    let emoji = if success { "🗑️" } else { "❌" };
    let status = if success { "Rejected" } else { "Failed" };
    let msg = format!(
        "{} <b>Modification {}</b>\n\nID: <code>{}</code>\n{}",
        emoji, status, id, message
    );
    let _ = send(&msg).await;
}

/// Send rollback confirmation
pub async fn send_rollback(id: &str, success: bool, message: &str) {
    let emoji = if success { "⏪" } else { "❌" };
    let status = if success { "Rolled Back" } else { "Rollback Failed" };
    let msg = format!(
        "{} <b>Modification {}</b>\n\nID: <code>{}</code>\n{}",
        emoji, status, id, message
    );
    let _ = send(&msg).await;
}

/// Send command help
pub async fn send_selfmod_help() {
    let msg = "🤖 <b>Self-Modification Commands</b>\n\n\
        /pending - List pending modifications\n\
        /rules - List active trading rules\n\
        /constitution - Show safety constraints\n\
        /approve &lt;id&gt; - Approve a modification\n\
        /reject &lt;id&gt; - Reject a modification\n\
        /rollback &lt;id&gt; - Rollback an applied modification";
    let _ = send(msg).await;
}

// ==================== Codegen Commands ====================

/// Send codegen help
pub async fn send_codegen_help() {
    let msg = "🧬 <b>Code Generation Commands</b>\n\n\
        /codegen - Show code generation status\n\
        /gencode list - List all generated code\n\
        /gencode pending - List pending code proposals\n\
        /gencode active - List active deployed code\n\
        /gencode deploy &lt;id&gt; - Deploy pending code\n\
        /gencode rollback &lt;id&gt; - Rollback deployed code";
    let _ = send(msg).await;
}

/// Send codegen status
pub async fn send_codegen_status(active: usize, pending: usize, history: usize) {
    let msg = format!(
        "🧬 <b>Code Generation Status</b>\n\n\
        Active: {}\n\
        Pending: {}\n\
        Total Generated: {}",
        active, pending, history
    );
    let _ = send(&msg).await;
}

/// Send list of pending generated code
pub async fn send_pending_code(code_list: &[(u64, String, String)]) {
    if code_list.is_empty() {
        let _ = send("⏳ <b>No Pending Code</b>\n\nNo code proposals awaiting deployment.").await;
        return;
    }

    let mut msg = String::from("⏳ <b>Pending Code Proposals</b>\n\n");
    for (id, code_type, description) in code_list.iter().take(10) {
        msg.push_str(&format!(
            "<code>{}</code> [{}]\n{}\n\n",
            id, code_type, description
        ));
    }
    let _ = send(&msg).await;
}

/// Send list of active generated code
pub async fn send_active_code(code_list: &[(u64, String, String, u32)]) {
    if code_list.is_empty() {
        let _ = send("✅ <b>No Active Code</b>\n\nNo generated code is currently deployed.").await;
        return;
    }

    let mut msg = String::from("✅ <b>Active Generated Code</b>\n\n");
    for (id, code_type, description, executions) in code_list.iter().take(10) {
        msg.push_str(&format!(
            "<code>{}</code> [{}]\n{}\nExecutions: {}\n\n",
            id, code_type, description, executions
        ));
    }
    let _ = send(&msg).await;
}

/// Send code deployment confirmation
pub async fn send_code_deploy(id: &str, success: bool, message: &str) {
    let emoji = if success { "🚀" } else { "❌" };
    let status = if success { "Deployed" } else { "Deploy Failed" };
    let msg = format!(
        "{} <b>Code {}</b>\n\nID: <code>{}</code>\n{}",
        emoji, status, id, message
    );
    let _ = send(&msg).await;
}

/// Send code rollback confirmation
pub async fn send_code_rollback(id: &str, success: bool, message: &str) {
    let emoji = if success { "⏪" } else { "❌" };
    let status = if success { "Rolled Back" } else { "Rollback Failed" };
    let msg = format!(
        "{} <b>Code {}</b>\n\nID: <code>{}</code>\n{}",
        emoji, status, id, message
    );
    let _ = send(&msg).await;
}

// ==================== Causal Commands ====================

/// Send latent confounders list
pub async fn send_causal_latents(latents: &[(String, String)], summary: &str) {
    if latents.is_empty() {
        let msg = format!("🔍 <b>Latent Confounders</b>\n\n{}", summary);
        let _ = send(&msg).await;
        return;
    }

    let mut msg = String::from("🔍 <b>Latent Confounders (Hidden Variables)</b>\n\n");
    msg.push_str(&format!("{}\n\n", summary));

    for (i, (a, b)) in latents.iter().take(10).enumerate() {
        msg.push_str(&format!(
            "{}. <b>{}</b> ↔ <b>{}</b>\n   Both influenced by unknown hidden factor\n\n",
            i + 1, a, b
        ));
    }

    if latents.len() > 10 {
        msg.push_str(&format!("... and {} more", latents.len() - 10));
    }

    let _ = send(&msg).await;
}

/// Send PAG structure summary
pub async fn send_causal_pag(summary: &str) {
    let msg = format!(
        "🕸️ <b>Partial Ancestral Graph (PAG)</b>\n\n\
        <pre>{}</pre>\n\n\
        Legend:\n\
        → Definite direction\n\
        ↔ Latent confounder\n\
        ○ Uncertain endpoint",
        summary
    );
    let _ = send(&msg).await;
}

// ==================== Transfer Blocking Commands ====================

/// Send transfer command help
pub async fn send_transfer_help() {
    let msg = "🔄 <b>Transfer Commands</b>\n\n\
        /transfer blocked - List blacklisted & graylisted pairs\n\
        /transfer history - Show transfer impact history\n\
        /transfer unblock &lt;source&gt; &lt;target&gt; - Remove from blacklist";
    let _ = send(msg).await;
}

/// Send blocked transfer pairs (blacklist and graylist)
pub async fn send_blocked_transfers(
    blacklist: &[(String, String, String)],  // (source, target, reason)
    graylist: &[(String, String, f64, String)],  // (source, target, impact, expires)
) {
    let mut msg = String::from("🚫 <b>Blocked Transfer Pairs</b>\n\n");

    // Blacklist section
    if blacklist.is_empty() {
        msg.push_str("⛔ <b>Blacklist:</b> Empty\n\n");
    } else {
        msg.push_str(&format!("⛔ <b>Blacklist</b> ({} pairs):\n", blacklist.len()));
        for (i, (source, target, reason)) in blacklist.iter().take(10).enumerate() {
            msg.push_str(&format!(
                "{}. {} → {}\n   {}\n",
                i + 1, source, target, reason
            ));
        }
        if blacklist.len() > 10 {
            msg.push_str(&format!("... and {} more\n", blacklist.len() - 10));
        }
        msg.push('\n');
    }

    // Graylist section
    if graylist.is_empty() {
        msg.push_str("⏸️ <b>Graylist:</b> Empty");
    } else {
        msg.push_str(&format!("⏸️ <b>Graylist</b> ({} pairs):\n", graylist.len()));
        for (i, (source, target, impact, expires)) in graylist.iter().take(10).enumerate() {
            msg.push_str(&format!(
                "{}. {} → {} ({:.1}%)\n   Expires: {}\n",
                i + 1, source, target, impact * 100.0, expires
            ));
        }
        if graylist.len() > 10 {
            msg.push_str(&format!("... and {} more", graylist.len() - 10));
        }
    }

    let _ = send(&msg).await;
}

/// Send transfer impact history
pub async fn send_transfer_history(history: &[(String, String, f64, String, String)]) {
    // (source, target, impact, verdict, date)
    if history.is_empty() {
        let _ = send("📜 <b>Transfer History</b>\n\nNo transfer evaluations yet.").await;
        return;
    }

    let mut msg = String::from("📜 <b>Transfer History</b>\n\n");
    for (i, (source, target, impact, verdict, date)) in history.iter().take(15).enumerate() {
        let emoji = match verdict.as_str() {
            "Positive" => "✅",
            "Neutral" => "➖",
            "Negative" => "⚠️",
            "HighlyNegative" => "🚫",
            _ => "❓",
        };
        msg.push_str(&format!(
            "{}. {} {} → {}\n   Impact: {:.1}% | {}\n",
            i + 1, emoji, source, target,
            impact * 100.0, date
        ));
    }

    if history.len() > 15 {
        msg.push_str(&format!("\n... and {} more", history.len() - 15));
    }

    let _ = send(&msg).await;
}

/// Send transfer unblock confirmation
pub async fn send_transfer_unblock(source: &str, target: &str, success: bool, message: &str) {
    let emoji = if success { "✅" } else { "❌" };
    let status = if success { "Unblocked" } else { "Unblock Failed" };
    let msg = format!(
        "{} <b>Transfer {}</b>\n\n{} → {}\n{}",
        emoji, status, source, target, message
    );
    let _ = send(&msg).await;
}

/// Send negative transfer alert
pub async fn send_negative_transfer_alert(
    source: &str,
    target: &str,
    impact: f64,
    verdict: &str,
    action: &str,
) {
    let emoji = if verdict == "HighlyNegative" { "🚨" } else { "⚠️" };
    let msg = format!(
        "{} <b>Negative Transfer Detected</b>\n\n\
        {} → {}\n\
        Impact: <b>{:.1}%</b>\n\
        Verdict: {}\n\
        Action: {}",
        emoji, source, target, impact * 100.0, verdict, action
    );
    let _ = send(&msg).await;
}

// ==================== Sharded Memory Commands ====================

/// Send memory command help
pub async fn send_memory_help() {
    let msg = "💾 <b>Memory Commands</b>\n\n\
        /memory stats - Show sharded memory statistics\n\
        /memory shards - Show per-shard information\n\
        /memory compact - Trigger shard compaction\n\
        /memory rebalance - Trigger shard rebalancing";
    let _ = send(msg).await;
}

/// Send sharded memory statistics
pub async fn send_memory_stats(
    total_entries: u64,
    num_shards: usize,
    total_size_mb: f64,
    avg_query_ms: f64,
    cache_hit_rate: f64,
    symbol_count: usize,
    buffer_size: usize,
) {
    let msg = format!(
        "💾 <b>Sharded Memory Stats</b>\n\n\
        📊 Total Entries: {}\n\
        🗂️ Shards: {}\n\
        📦 Total Size: {:.1} MB\n\
        ⚡ Avg Query: {:.2} ms\n\
        🎯 Cache Hit Rate: {:.1}%\n\
        🏷️ Symbols: {}\n\
        ✏️ Write Buffer: {}",
        format_number(total_entries),
        num_shards,
        total_size_mb,
        avg_query_ms,
        cache_hit_rate * 100.0,
        symbol_count,
        buffer_size
    );
    let _ = send(&msg).await;
}

/// Format large numbers with K/M/B suffixes
fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Send per-shard information
pub async fn send_shard_info(shards: &[(usize, usize, f64)]) {
    // (shard_id, entry_count, load_pct)
    let mut msg = String::from("🗂️ <b>Shard Information</b>\n\n");

    for (id, count, load_pct) in shards {
        let bar = render_load_bar(*load_pct);
        msg.push_str(&format!(
            "Shard {}: {} {} ({:.1}%)\n",
            id, bar, format_number(*count as u64), load_pct * 100.0
        ));
    }

    let _ = send(&msg).await;
}

/// Render a load bar for shard visualization
fn render_load_bar(load_pct: f64) -> String {
    let width = 10;
    let filled = ((load_pct * width as f64).round() as usize).min(width);
    let empty = width - filled;

    let fill_char = if load_pct > 0.9 {
        "🔴"
    } else if load_pct > 0.7 {
        "🟡"
    } else {
        "🟢"
    };

    format!("[{}{}]", fill_char.repeat(filled), "░".repeat(empty))
}

/// Send compaction result
pub async fn send_memory_compact(success: bool, message: &str) {
    let emoji = if success { "✅" } else { "❌" };
    let status = if success { "Compaction Complete" } else { "Compaction Failed" };
    let msg = format!(
        "{} <b>{}</b>\n\n{}",
        emoji, status, message
    );
    let _ = send(&msg).await;
}

/// Send rebalance result
pub async fn send_memory_rebalance(success: bool, entries_moved: u32, message: &str) {
    let emoji = if success { "⚖️" } else { "❌" };
    let status = if success { "Rebalance Complete" } else { "Rebalance Failed" };
    let msg = format!(
        "{} <b>{}</b>\n\n\
        Entries Moved: {}\n\
        {}",
        emoji, status, entries_moved, message
    );
    let _ = send(&msg).await;
}

// ==================== Streaming Commands ====================

/// Send streaming command help
pub async fn send_streaming_help() {
    let msg = "📡 <b>Streaming Commands</b>\n\n\
        /streaming stats - Show streaming statistics\n\
        /streaming flush - Force flush all pending updates\n\
        /streaming pause - Pause learning (buffer still accepts)\n\
        /streaming resume - Resume learning";
    let _ = send(msg).await;
}

/// Send streaming statistics
pub async fn send_streaming_stats(
    updates_processed: u64,
    updates_pending: usize,
    learning_updates: u64,
    avg_latency_us: f64,
    buffer_utilization: f64,
    cache_size: usize,
    is_paused: bool,
) {
    let status_emoji = if is_paused { "⏸️" } else { "▶️" };
    let status_text = if is_paused { "PAUSED" } else { "RUNNING" };

    let msg = format!(
        "📡 <b>Streaming Stats</b> {} {}\n\n\
        📥 Updates Processed: {}\n\
        ⏳ Updates Pending: {}\n\
        🧠 Learning Updates: {}\n\
        ⚡ Avg Latency: {:.1} µs\n\
        📊 Buffer: {:.1}%\n\
        🗂️ Cache Size: {}",
        status_emoji, status_text,
        format_number(updates_processed),
        updates_pending,
        format_number(learning_updates),
        avg_latency_us,
        buffer_utilization * 100.0,
        cache_size
    );
    let _ = send(&msg).await;
}

/// Send streaming flush result
pub async fn send_streaming_flush(success: bool, message: &str) {
    let emoji = if success { "✅" } else { "❌" };
    let status = if success { "Flush Complete" } else { "Flush Failed" };
    let msg = format!(
        "{} <b>{}</b>\n\n{}",
        emoji, status, message
    );
    let _ = send(&msg).await;
}

/// Send streaming pause/resume confirmation
pub async fn send_streaming_state(paused: bool) {
    let (emoji, status, desc) = if paused {
        ("⏸️", "Learning Paused", "Buffer continues accepting updates")
    } else {
        ("▶️", "Learning Resumed", "Processing updates in background")
    };

    let msg = format!(
        "{} <b>{}</b>\n\n{}",
        emoji, status, desc
    );
    let _ = send(&msg).await;
}

/// Send online learning update notification
pub async fn send_learning_update(
    symbol: &str,
    update_type: &str,
    learning_count: u64,
    accuracy: f64,
) {
    let msg = format!(
        "🧠 <b>Online Learning</b>\n\n\
        Symbol: {}\n\
        Update: {}\n\
        Total Learning Updates: {}\n\
        Model Accuracy: {:.1}%",
        symbol, update_type, learning_count, accuracy * 100.0
    );
    let _ = send(&msg).await;
}
