//! MT5 Bridge - Connects to VPS and receives live data

use anyhow::Result;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{info, warn, error};
use rust_decimal::Decimal;
use std::str::FromStr;
use tokio::sync::mpsc;

/// Tick data from MT5
#[derive(Debug, Clone)]
pub struct Tick {
    pub bid: Decimal,
    pub ask: Decimal,
}

/// Candle data from MT5
#[derive(Debug, Clone)]
pub struct BridgeCandle {
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

/// Messages from the bridge
#[derive(Debug, Clone)]
pub enum BridgeMessage {
    Tick(Tick),
    Candle(BridgeCandle),
}

/// Connect to MT5 bridge and receive data
pub async fn connect(host: &str, port: u16, tx: mpsc::Sender<BridgeMessage>) -> Result<()> {
    let addr = format!("{}:{}", host, port);
    
    info!("Connecting to MT5 bridge at {}...", addr);
    
    let stream = TcpStream::connect(&addr).await?;
    info!("Connected to MT5 bridge!");
    
    let reader = BufReader::new(stream);
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
        }
    }
    
    warn!("Disconnected from MT5 bridge");
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
