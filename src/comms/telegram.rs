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
