//! Alpaca WebSocket Data Stream
//! Connects to Alpaca's real-time market data
//! Supports HTTP proxy via https_proxy environment variable

use anyhow::Result;
use rust_decimal::Decimal;
use std::str::FromStr;
use std::env;
use tokio::sync::mpsc;
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream, MaybeTlsStream};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug};

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

/// Get HTTP proxy from environment
fn get_proxy() -> Option<(String, u16, Option<String>)> {
    // Check https_proxy first, then HTTP_PROXY
    let proxy_url = env::var("https_proxy")
        .or_else(|_| env::var("HTTPS_PROXY"))
        .or_else(|_| env::var("http_proxy"))
        .or_else(|_| env::var("HTTP_PROXY"))
        .ok()?;

    // Parse proxy URL (format: http://[user:pass@]host:port)
    let proxy_url = proxy_url.trim_start_matches("http://").trim_start_matches("https://");

    // Check for credentials (user:pass@host:port format)
    let (auth, host_port) = if let Some(at_pos) = proxy_url.rfind('@') {
        let auth = Some(proxy_url[..at_pos].to_string());
        let host_port = &proxy_url[at_pos + 1..];
        (auth, host_port)
    } else {
        (None, proxy_url)
    };

    // Parse host:port
    if let Some(colon_pos) = host_port.rfind(':') {
        let host = host_port[..colon_pos].to_string();
        let port = host_port[colon_pos + 1..].trim_end_matches('/').parse().ok()?;
        Some((host, port, auth))
    } else {
        None
    }
}

/// Connect to target through HTTP CONNECT proxy
async fn connect_via_proxy(
    proxy_host: &str,
    proxy_port: u16,
    target_host: &str,
    target_port: u16,
    proxy_auth: Option<&str>,
) -> Result<TcpStream> {
    info!("Connecting via proxy {}:{}", proxy_host, proxy_port);

    let mut stream = TcpStream::connect((proxy_host, proxy_port)).await?;

    // Build CONNECT request with optional Proxy-Authorization
    let auth_header = if let Some(auth) = proxy_auth {
        use std::io::Write;
        let mut encoder = base64::write::EncoderStringWriter::new(&base64::engine::general_purpose::STANDARD);
        let _ = encoder.write_all(auth.as_bytes());
        let encoded = encoder.into_inner();
        format!("Proxy-Authorization: Basic {}\r\n", encoded)
    } else {
        String::new()
    };

    let connect_request = format!(
        "CONNECT {}:{} HTTP/1.1\r\nHost: {}:{}\r\n{}\r\n",
        target_host, target_port, target_host, target_port, auth_header
    );
    stream.write_all(connect_request.as_bytes()).await?;

    // Read response
    let mut buffer = [0u8; 1024];
    let n = stream.read(&mut buffer).await?;
    let response = String::from_utf8_lossy(&buffer[..n]);

    debug!("Proxy response: {}", response);

    // Check for 200 OK
    if !response.contains("200") {
        return Err(anyhow::anyhow!("Proxy CONNECT failed: {}", response));
    }

    info!("Proxy tunnel established to {}:{}", target_host, target_port);
    Ok(stream)
}

pub async fn connect(
    api_key: &str,
    api_secret: &str,
    symbols: &[String],
    tx: mpsc::Sender<AlpacaMessage>,
) -> Result<()> {
    // Use IEX feed (free) - for SIP feed use wss://stream.data.alpaca.markets/v2/sip
    let url = "wss://stream.data.alpaca.markets/v2/iex";
    let target_host = "stream.data.alpaca.markets";
    let target_port = 443u16;

    info!("Connecting to Alpaca stream: {}", url);

    // Check for proxy and connect appropriately
    if let Some((proxy_host, proxy_port, proxy_auth)) = get_proxy() {
        connect_via_proxy_ws(api_key, api_secret, symbols, tx, &proxy_host, proxy_port, proxy_auth.as_deref(), target_host, target_port, url).await
    } else {
        connect_direct_ws(api_key, api_secret, symbols, tx, url).await
    }
}

/// Connect directly without proxy
async fn connect_direct_ws(
    api_key: &str,
    api_secret: &str,
    symbols: &[String],
    tx: mpsc::Sender<AlpacaMessage>,
    url: &str,
) -> Result<()> {
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

/// Connect via HTTP proxy
async fn connect_via_proxy_ws(
    api_key: &str,
    api_secret: &str,
    symbols: &[String],
    tx: mpsc::Sender<AlpacaMessage>,
    proxy_host: &str,
    proxy_port: u16,
    proxy_auth: Option<&str>,
    target_host: &str,
    target_port: u16,
    url: &str,
) -> Result<()> {
    // Connect via HTTP proxy
    let tcp_stream = connect_via_proxy(proxy_host, proxy_port, target_host, target_port, proxy_auth).await?;

    // Upgrade to TLS
    let connector = tokio_native_tls::TlsConnector::from(
        native_tls::TlsConnector::new()?
    );
    let tls_stream = connector.connect(target_host, tcp_stream).await?;

    // Upgrade to WebSocket
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::client_async;

    let mut request = url.into_client_request()?;
    request.headers_mut().insert("Host", target_host.parse().unwrap());

    let (ws_stream, _) = client_async(request, tls_stream).await?;
    let (mut write, mut read) = ws_stream.split();

    info!("Connected to Alpaca WebSocket via proxy");

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
