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
