//! Web Dashboard

use axum::{routing::get, Router, response::Html, Json};
use serde::Serialize;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use sovereign::status::SystemStatus;

#[derive(Serialize)]
struct ApiStatus {
    name: String,
    version: String,
    status: String,
    uptime_secs: u64,
    trades_today: u32,
    total_pnl: f64,
    active_positions: u32,
    last_signal: String,
    last_price: f64,
    balance: f64,
    equity: f64,
}

async fn index() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html>
<head>
    <title>Sovereign v4</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { font-family: monospace; background: #0a0a0a; color: #0f0; padding: 40px; max-width: 800px; margin: 0 auto; }
        h1 { border-bottom: 2px solid #0f0; padding-bottom: 10px; }
        .card { background: #111; border: 1px solid #0f0; padding: 20px; margin: 20px 0; border-radius: 4px; }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; }
        .label { color: #888; }
        .value { color: #0f0; font-weight: bold; }
        .pos { color: #0f0; }
        .neg { color: #f44; }
    </style>
</head>
<body>
    <h1>SOVEREIGN v4.0</h1>
    <p>Perpetual Autonomous Trading System</p>
    <div class="card">
        <h3>System</h3>
        <div class="stat"><span class="label">Status:</span><span class="value" id="status">Loading...</span></div>
        <div class="stat"><span class="label">Uptime:</span><span class="value" id="uptime">-</span></div>
        <div class="stat"><span class="label">Price:</span><span class="value" id="price">-</span></div>
    </div>
    <div class="card">
        <h3>Account</h3>
        <div class="stat"><span class="label">Balance:</span><span class="value" id="balance">-</span></div>
        <div class="stat"><span class="label">Equity:</span><span class="value" id="equity">-</span></div>
        <div class="stat"><span class="label">Positions:</span><span class="value" id="positions">-</span></div>
    </div>
    <div class="card">
        <h3>Trading</h3>
        <div class="stat"><span class="label">Trades Today:</span><span class="value" id="trades">-</span></div>
        <div class="stat"><span class="label">Total P&L:</span><span class="value" id="pnl">-</span></div>
        <div class="stat"><span class="label">Last Signal:</span><span class="value" id="signal">-</span></div>
    </div>
    <script>
        async function update() {
            try {
                const r = await fetch('/api/status');
                const d = await r.json();
                document.getElementById('status').textContent = d.status.toUpperCase();
                document.getElementById('uptime').textContent = Math.floor(d.uptime_secs/3600)+'h '+Math.floor((d.uptime_secs%3600)/60)+'m';
                document.getElementById('price').textContent = '$'+d.last_price.toFixed(2);
                document.getElementById('balance').textContent = '$'+d.balance.toFixed(2);
                document.getElementById('equity').textContent = '$'+d.equity.toFixed(2);
                document.getElementById('positions').textContent = d.active_positions;
                document.getElementById('trades').textContent = d.trades_today;
                const pnl = document.getElementById('pnl');
                pnl.textContent = (d.total_pnl>=0?'+':'')+d.total_pnl.toFixed(2);
                pnl.className = d.total_pnl>=0?'value pos':'value neg';
                document.getElementById('signal').textContent = d.last_signal;
            } catch(e) { console.error(e); }
        }
        update(); setInterval(update, 5000);
    </script>
</body>
</html>"#)
}

async fn api_status() -> Json<ApiStatus> {
    let s = SystemStatus::load();
    let now = chrono::Utc::now().timestamp();
    let uptime = if s.start_time > 0 { (now - s.start_time) as u64 } else { 0 };
    
    Json(ApiStatus {
        name: "Sovereign v4".to_string(),
        version: "4.0.0".to_string(),
        status: if s.running { "running" } else { "stopped" }.to_string(),
        uptime_secs: uptime,
        trades_today: s.trades_today,
        total_pnl: s.total_pnl,
        active_positions: s.active_positions,
        last_signal: s.last_signal,
        last_price: s.last_price,
        balance: s.balance,
        equity: s.equity,
    })
}

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("  SOVEREIGN v4 - Web Dashboard");
    println!("============================================================");
    
    let app = Router::new()
        .route("/", get(index))
        .route("/api/status", get(api_status));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("Dashboard: http://localhost:8080");
    
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
