//! Web Dashboard - Simple status monitor

use axum::{
    routing::get,
    Router,
    response::Html,
    Json,
};
use serde::Serialize;
use std::net::SocketAddr;
use tokio::net::TcpListener;

#[derive(Serialize)]
struct Status {
    name: String,
    version: String,
    status: String,
    uptime_secs: u64,
    trades_today: u32,
    total_pnl: f64,
    active_positions: u32,
    last_signal: String,
}

async fn index() -> Html<String> {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Sovereign v4 Dashboard</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #0a0a0a; 
            color: #00ff00; 
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }
        h1 { border-bottom: 2px solid #00ff00; padding-bottom: 10px; }
        .card {
            background: #111;
            border: 1px solid #00ff00;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; }
        .label { color: #888; }
        .value { color: #00ff00; font-weight: bold; }
        .positive { color: #00ff00; }
        .negative { color: #ff4444; }
        .neutral { color: #ffff00; }
    </style>
</head>
<body>
    <h1>SOVEREIGN v4.0</h1>
    <p>Perpetual Autonomous Trading System</p>
    
    <div class="card">
        <h3>System Status</h3>
        <div class="stat"><span class="label">Status:</span><span class="value positive">RUNNING</span></div>
        <div class="stat"><span class="label">Version:</span><span class="value">4.0.0</span></div>
        <div class="stat"><span class="label">Uptime:</span><span class="value" id="uptime">Loading...</span></div>
    </div>
    
    <div class="card">
        <h3>Trading Stats</h3>
        <div class="stat"><span class="label">Active Positions:</span><span class="value" id="positions">0</span></div>
        <div class="stat"><span class="label">Trades Today:</span><span class="value" id="trades">0</span></div>
        <div class="stat"><span class="label">Total P&L:</span><span class="value" id="pnl">$0.00</span></div>
    </div>
    
    <div class="card">
        <h3>Last Signal</h3>
        <div class="stat"><span class="label">Signal:</span><span class="value neutral" id="signal">Waiting...</span></div>
    </div>
    
    <script>
        async function updateStats() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                document.getElementById('uptime').textContent = formatUptime(data.uptime_secs);
                document.getElementById('positions').textContent = data.active_positions;
                document.getElementById('trades').textContent = data.trades_today;
                document.getElementById('pnl').textContent = '$' + data.total_pnl.toFixed(2);
                document.getElementById('signal').textContent = data.last_signal;
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }
        
        function formatUptime(secs) {
            const h = Math.floor(secs / 3600);
            const m = Math.floor((secs % 3600) / 60);
            const s = secs % 60;
            return h + 'h ' + m + 'm ' + s + 's';
        }
        
        updateStats();
        setInterval(updateStats, 5000);
    </script>
</body>
</html>
"#;
    Html(html.to_string())
}

async fn api_status() -> Json<Status> {
    // TODO: Read from shared state file or database
    Json(Status {
        name: "Sovereign v4".to_string(),
        version: "4.0.0".to_string(),
        status: "running".to_string(),
        uptime_secs: 0,
        trades_today: 0,
        total_pnl: 0.0,
        active_positions: 0,
        last_signal: "HOLD".to_string(),
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
    println!("Dashboard running at http://localhost:8080");
    
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
