# Sovereign v4.0 "Perpetual"

**Institutional-Grade Autonomous Trading System**

Built in Rust for 12+ year runtime with zero human intervention.

---

## Philosophy

Inspired by [Tech Trader](https://techtrader.ai/) by pftq:

> "The reason I call it a 'lossless' algorithm is because it doesn't estimate anything or use any seeded values/thresholds."

### Core Principles

1. **Lossless Algorithms**: No parameters, no thresholds, no magic numbers
2. **Human Thinking**: Pattern recognition, not statistics  
3. **Perpetual Operation**: Self-healing, never crashes
4. **Scale**: Designed for 1000+ concurrent agents

---

## Quick Start
```bash
# Build
cargo build --release

# Run trading system
./target/release/sovereign

# Run with auto-restart
./target/release/watchdog

# Run as Linux service
sudo systemctl start sovereign

# Run backtest
./target/release/backtest data/XAUUSD.csv

# Run web dashboard
./target/release/dashboard
```

---

## Binaries

| Binary | Purpose |
|--------|---------|
| sovereign | Main trading system |
| watchdog | Auto-restart guardian |
| backtest | Strategy testing on CSV |
| dashboard | Web UI (port 8080) |

---

## Project Structure
```
sovereign_v4_rust/
├── src/
│   ├── main.rs              # Entry point
│   ├── lib.rs               # Library exports
│   ├── config.rs            # TOML config loader
│   ├── status.rs            # Shared status file
│   ├── backtest.rs          # Backtesting engine
│   ├── core/
│   │   ├── types.rs         # Core data structures
│   │   ├── lossless.rs      # Lossless algorithms
│   │   ├── strategy.rs      # Signal generation
│   │   ├── agent.rs         # Single-symbol agent
│   │   ├── coordinator.rs   # Multi-agent manager
│   │   └── guardian.rs      # Risk management
│   ├── broker/
│   │   ├── mt5.rs           # MetaTrader 5
│   │   ├── alpaca.rs        # Alpaca Markets
│   │   └── ibkr.rs          # Interactive Brokers
│   ├── data/
│   │   ├── mt5_bridge.rs    # TCP bridge to MT5
│   │   ├── database.rs      # SQLite storage
│   │   └── multi_feed.rs    # Multi-symbol data
│   ├── comms/
│   │   └── telegram.rs      # Notifications
│   └── bin/
│       ├── watchdog.rs      # Process monitor
│       ├── backtest.rs      # CLI backtester
│       └── dashboard.rs     # Web UI
├── config.toml              # Runtime configuration
└── Cargo.toml               # Dependencies
```

---

## Infrastructure Status: 100%

| Component | Status |
|-----------|--------|
| Lossless Algorithms | ✅ |
| Trading Strategy | ✅ |
| MT5 Bridge | ✅ |
| Position Tracking | ✅ |
| Risk Guardian | ✅ |
| Telegram Alerts | ✅ |
| Watchdog | ✅ |
| Systemd Service | ✅ |
| SQLite Database | ✅ |
| Multi-Agent Coordinator | ✅ |
| Config File | ✅ |
| Backtesting | ✅ |
| Web Dashboard | ✅ |
| Alpaca Broker | ✅ |
| IBKR Broker | ✅ |
| Multi-Symbol Feed | ✅ |

---

## License

Proprietary - All rights reserved.
