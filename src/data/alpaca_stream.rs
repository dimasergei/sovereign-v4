//! Alpaca WebSocket Data Stream
//! Connects to Alpaca's real-time market data

use anyhow::Result;
use rust_decimal::Decimal;
use std::str::FromStr;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct TickData {
    pub symbol: String,
    pub bid: Decimal,
    pub ask: Decimal,
}

#[derive(Debug, Clone)]
pub struct BarData {
    pub symbol: String,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: i64,
}

#[derive(Debug, Clone)]
pub enum AlpacaMessage {
    Tick(TickData),
    Bar(BarData),
    Connected,
    Error(String),
}

#[derive(Debug, Serialize)]
struct AuthMessage {
    action: String,
    key: String,
    secret: String,
}

#[derive(Debug, Serialize)]
struct SubscribeMessage {
    action: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    bars: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    quotes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StreamMessage {
    #[serde(rename = "T")]
    msg_type: String,
    #[serde(rename = "S")]
    symbol: Option<String>,
    // Bar fields
    o: Option<f64>,
    h: Option<f64>,
    l: Option<f64>,
    c: Option<f64>,
    v: Option<i64>,
    // Quote fields
    bp: Option<f64>,
    ap: Option<f64>,
    // Error
    msg: Option<String>,
}

pub async fn connect(
    api_key: &str,
    api_secret: &str,
    symbols: &[String],
    tx: mpsc::Sender<AlpacaMessage>,
) -> Result<()> {
    // Use IEX feed (free) - for SIP feed use wss://stream.data.alpaca.markets/v2/sip
    let url = "wss://stream.data.alpaca.markets/v2/iex";
    
    info!("Connecting to Alpaca stream: {}", url);
    
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();
    
    info!("Connected to Alpaca WebSocket");
    
    // Authenticate
    let auth = AuthMessage {
        action: "auth".to_string(),
        key: api_key.to_string(),
        secret: api_secret.to_string(),
    };
    write.send(Message::Text(serde_json::to_string(&auth)?)).await?;
    
    // Wait for auth response
    if let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                info!("Auth response: {}", text);
                if text.contains("authorized") {
                    let _ = tx.send(AlpacaMessage::Connected).await;
                } else if text.contains("error") {
                    let _ = tx.send(AlpacaMessage::Error(text.clone())).await;
                    return Err(anyhow::anyhow!("Auth failed: {}", text));
                }
            }
            _ => {}
        }
    }
    
    // Subscribe to bars and quotes
    let subscribe = SubscribeMessage {
        action: "subscribe".to_string(),
        bars: symbols.to_vec(),
        quotes: symbols.to_vec(),
    };
    write.send(Message::Text(serde_json::to_string(&subscribe)?)).await?;
    info!("Subscribed to: {:?}", symbols);
    
    // Process incoming messages
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Alpaca sends arrays of messages
                if let Ok(messages) = serde_json::from_str::<Vec<StreamMessage>>(&text) {
                    for m in messages {
                        match m.msg_type.as_str() {
                            "b" => {
                                // Bar
                                if let (Some(sym), Some(o), Some(h), Some(l), Some(c), Some(v)) = 
                                    (m.symbol, m.o, m.h, m.l, m.c, m.v) {
                                    let bar = BarData {
                                        symbol: sym,
                                        open: Decimal::from_str(&o.to_string()).unwrap_or_default(),
                                        high: Decimal::from_str(&h.to_string()).unwrap_or_default(),
                                        low: Decimal::from_str(&l.to_string()).unwrap_or_default(),
                                        close: Decimal::from_str(&c.to_string()).unwrap_or_default(),
                                        volume: v,
                                    };
                                    let _ = tx.send(AlpacaMessage::Bar(bar)).await;
                                }
                            }
                            "q" => {
                                // Quote
                                if let (Some(sym), Some(bp), Some(ap)) = (m.symbol, m.bp, m.ap) {
                                    let tick = TickData {
                                        symbol: sym,
                                        bid: Decimal::from_str(&bp.to_string()).unwrap_or_default(),
                                        ask: Decimal::from_str(&ap.to_string()).unwrap_or_default(),
                                    };
                                    let _ = tx.send(AlpacaMessage::Tick(tick)).await;
                                }
                            }
                            "error" => {
                                warn!("Alpaca error: {:?}", m.msg);
                                let _ = tx.send(AlpacaMessage::Error(m.msg.unwrap_or_default())).await;
                            }
                            "subscription" => {
                                info!("Subscription confirmed");
                            }
                            "success" => {
                                info!("Success: {:?}", m.msg);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                let _ = write.send(Message::Pong(data)).await;
            }
            Ok(Message::Close(_)) => {
                warn!("WebSocket closed");
                break;
            }
            Err(e) => {
                warn!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
    
    Ok(())
}
