//! MT5 Bridge - Bidirectional: Data in, Orders out

use anyhow::Result;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, WriteHalf, ReadHalf};
use tracing::{info, warn};
use rust_decimal::Decimal;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

#[derive(Debug, Clone)]
pub struct Tick {
    pub bid: Decimal,
    pub ask: Decimal,
}

#[derive(Debug, Clone)]
pub struct BridgeCandle {
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

#[derive(Debug, Clone)]
pub enum BridgeMessage {
    Tick(Tick),
    Candle(BridgeCandle),
    OrderResult { success: bool, ticket: u64, price: Decimal, error: String },
    AccountInfo { balance: Decimal, equity: Decimal, profit: Decimal },
}

/// Shared writer for sending commands
pub type BridgeWriter = Arc<Mutex<Option<WriteHalf<TcpStream>>>>;

/// Connect to MT5 bridge
pub async fn connect(
    host: &str, 
    port: u16, 
    tx: mpsc::Sender<BridgeMessage>,
    writer: BridgeWriter,
) -> Result<()> {
    let addr = format!("{}:{}", host, port);
    
    info!("Connecting to MT5 bridge at {}...", addr);
    
    let stream = TcpStream::connect(&addr).await?;
    let (read_half, write_half) = tokio::io::split(stream);
    
    // Store writer for sending commands
    {
        let mut w = writer.lock().await;
        *w = Some(write_half);
    }
    
    info!("Connected to MT5 bridge!");
    
    let reader = BufReader::new(read_half);
    let mut lines = reader.lines();
    
    while let Ok(Some(line)) = lines.next_line().await {
        if line.starts_with("TICK:") {
            if let Some(tick) = parse_tick(&line[5..]) {
                let _ = tx.send(BridgeMessage::Tick(tick)).await;
            }
        } else if line.starts_with("CANDLE:") {
            if let Some(candle) = parse_candle(&line[7..]) {
                info!("Candle: O={} H={} L={} C={}", 
                    candle.open, candle.high, candle.low, candle.close);
                let _ = tx.send(BridgeMessage::Candle(candle)).await;
            }
        } else if line.starts_with("ORDER_OK:") {
            let parts: Vec<&str> = line[9..].split(',').collect();
            if parts.len() >= 2 {
                let ticket = parts[0].parse().unwrap_or(0);
                let price = Decimal::from_str(parts[1]).unwrap_or(Decimal::ZERO);
                info!("Order filled: ticket={} price={}", ticket, price);
                let _ = tx.send(BridgeMessage::OrderResult {
                    success: true, ticket, price, error: String::new()
                }).await;
            }
        } else if line.starts_with("ORDER_ERROR:") {
            let error = line[12..].to_string();
            warn!("Order failed: {}", error);
            let _ = tx.send(BridgeMessage::OrderResult {
                success: false, ticket: 0, price: Decimal::ZERO, error
            }).await;
        } else if line.starts_with("ACCOUNT:") {
            let parts: Vec<&str> = line[8..].split(',').collect();
            if parts.len() >= 3 {
                let _ = tx.send(BridgeMessage::AccountInfo {
                    balance: Decimal::from_str(parts[0]).unwrap_or(Decimal::ZERO),
                    equity: Decimal::from_str(parts[1]).unwrap_or(Decimal::ZERO),
                    profit: Decimal::from_str(parts[2]).unwrap_or(Decimal::ZERO),
                }).await;
            }
        }
    }
    
    // Clear writer on disconnect
    {
        let mut w = writer.lock().await;
        *w = None;
    }
    
    warn!("Disconnected from MT5 bridge");
    Ok(())
}

/// Send a buy order
pub async fn send_buy(writer: &BridgeWriter, lots: Decimal, sl: Decimal, tp: Decimal) -> Result<()> {
    let cmd = format!("BUY:{},{},{}\n", lots, sl, tp);
    send_command(writer, &cmd).await
}

/// Send a sell order
pub async fn send_sell(writer: &BridgeWriter, lots: Decimal, sl: Decimal, tp: Decimal) -> Result<()> {
    let cmd = format!("SELL:{},{},{}\n", lots, sl, tp);
    send_command(writer, &cmd).await
}

/// Close a position
pub async fn send_close(writer: &BridgeWriter, ticket: u64) -> Result<()> {
    let cmd = format!("CLOSE:{}\n", ticket);
    send_command(writer, &cmd).await
}

/// Request account info
pub async fn request_account(writer: &BridgeWriter) -> Result<()> {
    send_command(writer, "ACCOUNT\n").await
}

async fn send_command(writer: &BridgeWriter, cmd: &str) -> Result<()> {
    let mut guard = writer.lock().await;
    if let Some(ref mut w) = *guard {
        w.write_all(cmd.as_bytes()).await?;
        info!("Sent: {}", cmd.trim());
    }
    Ok(())
}

fn parse_tick(data: &str) -> Option<Tick> {
    let parts: Vec<&str> = data.split(',').collect();
    if parts.len() >= 2 {
        Some(Tick {
            bid: Decimal::from_str(parts[0]).ok()?,
            ask: Decimal::from_str(parts[1]).ok()?,
        })
    } else {
        None
    }
}

fn parse_candle(data: &str) -> Option<BridgeCandle> {
    let parts: Vec<&str> = data.split(',').collect();
    if parts.len() >= 5 {
        Some(BridgeCandle {
            open: Decimal::from_str(parts[0]).ok()?,
            high: Decimal::from_str(parts[1]).ok()?,
            low: Decimal::from_str(parts[2]).ok()?,
            close: Decimal::from_str(parts[3]).ok()?,
            volume: Decimal::from_str(parts[4]).ok()?,
        })
    } else {
        None
    }
}
