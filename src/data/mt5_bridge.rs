//! MT5 Bridge - TCP connection to VPS

use anyhow::Result;
use rust_decimal::Decimal;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, WriteHalf};
use tracing::info;

pub type BridgeWriter = Arc<Mutex<Option<WriteHalf<TcpStream>>>>;

#[derive(Debug, Clone)]
pub struct TickData {
    pub bid: Decimal,
    pub ask: Decimal,
}

#[derive(Debug, Clone)]
pub struct CandleData {
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: i64,
}

#[derive(Debug, Clone)]
pub struct PositionData {
    pub ticket: u64,
    pub side: i32,      // 0=buy, 1=sell
    pub volume: Decimal,
    pub open_price: Decimal,
    pub sl: Decimal,
    pub tp: Decimal,
    pub profit: Decimal,
}

#[derive(Debug, Clone)]
pub enum BridgeMessage {
    Tick(TickData),
    Candle(CandleData),
    OrderResult { success: bool, ticket: u64, price: Decimal, error: String },
    AccountInfo { balance: Decimal, equity: Decimal, profit: Decimal },
    PositionOpen(PositionData),
    PositionUpdate { ticket: u64, profit: Decimal },
    PositionClosed,
    CloseResult { success: bool, ticket: u64, profit: Decimal, error: String },
}

pub async fn connect(
    host: &str,
    port: u16,
    tx: mpsc::Sender<BridgeMessage>,
    writer: BridgeWriter,
) -> Result<()> {
    info!("Connecting to MT5 bridge at {}:{}...", host, port);
    
    let stream = TcpStream::connect(format!("{}:{}", host, port)).await?;
    let (read_half, write_half) = tokio::io::split(stream);
    
    {
        let mut w = writer.lock().await;
        *w = Some(write_half);
    }
    
    info!("Connected to MT5 bridge!");
    
    let reader = BufReader::new(read_half);
    let mut lines = reader.lines();
    
    while let Some(line) = lines.next_line().await? {
        if let Some(msg) = parse_message(&line) {
            if let BridgeMessage::Candle(ref c) = msg {
                info!("Candle: O={} H={} L={} C={}", c.open, c.high, c.low, c.close);
            }
            tx.send(msg).await?;
        }
    }
    
    Ok(())
}

fn parse_message(line: &str) -> Option<BridgeMessage> {
    let parts: Vec<&str> = line.splitn(2, ':').collect();
    
    match parts[0] {
        "TICK" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::Tick(TickData {
                bid: Decimal::from_str(values.get(0)?).ok()?,
                ask: Decimal::from_str(values.get(1)?).ok()?,
            }))
        }
        "CANDLE" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::Candle(CandleData {
                open: Decimal::from_str(values.get(0)?).ok()?,
                high: Decimal::from_str(values.get(1)?).ok()?,
                low: Decimal::from_str(values.get(2)?).ok()?,
                close: Decimal::from_str(values.get(3)?).ok()?,
                volume: values.get(4)?.parse().ok()?,
            }))
        }
        "ORDER_OK" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::OrderResult {
                success: true,
                ticket: values.get(0)?.parse().ok()?,
                price: Decimal::from_str(values.get(1)?).ok()?,
                error: String::new(),
            })
        }
        "ORDER_ERROR" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::OrderResult {
                success: false,
                ticket: 0,
                price: Decimal::ZERO,
                error: values.get(1).unwrap_or(&"Unknown").to_string(),
            })
        }
        "ACCOUNT" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::AccountInfo {
                balance: Decimal::from_str(values.get(0)?).ok()?,
                equity: Decimal::from_str(values.get(1)?).ok()?,
                profit: Decimal::from_str(values.get(2)?).ok()?,
            })
        }
        "POSITION_OPEN" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::PositionOpen(PositionData {
                ticket: values.get(0)?.parse().ok()?,
                side: values.get(1)?.parse().ok()?,
                volume: Decimal::from_str(values.get(2)?).ok()?,
                open_price: Decimal::from_str(values.get(3)?).ok()?,
                sl: Decimal::from_str(values.get(4)?).ok()?,
                tp: Decimal::from_str(values.get(5)?).ok()?,
                profit: Decimal::from_str(values.get(6)?).ok()?,
            }))
        }
        "POSITION_UPDATE" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::PositionUpdate {
                ticket: values.get(0)?.parse().ok()?,
                profit: Decimal::from_str(values.get(1)?).ok()?,
            })
        }
        "POSITION_CLOSED" => {
            Some(BridgeMessage::PositionClosed)
        }
        "CLOSE_OK" => {
            let values: Vec<&str> = parts.get(1)?.split(',').collect();
            Some(BridgeMessage::CloseResult {
                success: true,
                ticket: values.get(0)?.parse().ok()?,
                profit: Decimal::from_str(values.get(1)?).ok()?,
                error: String::new(),
            })
        }
        "CLOSE_ERROR" => {
            Some(BridgeMessage::CloseResult {
                success: false,
                ticket: 0,
                profit: Decimal::ZERO,
                error: parts.get(1).unwrap_or(&"Unknown").to_string(),
            })
        }
        _ => None,
    }
}

async fn send_command(writer: &BridgeWriter, cmd: &str) -> Result<()> {
    let mut w = writer.lock().await;
    if let Some(ref mut stream) = *w {
        stream.write_all(format!("{}\n", cmd).as_bytes()).await?;
        info!("Sent: {}", cmd);
    }
    Ok(())
}

pub async fn send_buy(writer: &BridgeWriter, lots: Decimal, sl: Decimal, tp: Decimal) -> Result<()> {
    send_command(writer, &format!("BUY:{},{},{}", lots, sl, tp)).await
}

pub async fn send_sell(writer: &BridgeWriter, lots: Decimal, sl: Decimal, tp: Decimal) -> Result<()> {
    send_command(writer, &format!("SELL:{},{},{}", lots, sl, tp)).await
}

pub async fn send_close(writer: &BridgeWriter, ticket: u64) -> Result<()> {
    send_command(writer, &format!("CLOSE:{}", ticket)).await
}

pub async fn request_account(writer: &BridgeWriter) -> Result<()> {
    send_command(writer, "ACCOUNT").await
}

pub async fn request_positions(writer: &BridgeWriter) -> Result<()> {
    send_command(writer, "POSITIONS").await
}
