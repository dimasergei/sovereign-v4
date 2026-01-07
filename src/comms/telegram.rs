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
