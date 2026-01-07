# Sovereign V4 AGI Deployment Guide

## Prerequisites
- Rust 1.70+
- SQLite3
- IBKR or Alpaca account (paper trading enabled)

## Pre-Deployment Checklist

### 1. Build & Test
```bash
cargo build --release
cargo test
cargo run --bin integration_test
```

All 11 integration tests should pass:
- Memory (trade storage and retrieval)
- Learning (calibrator weight updates)
- EWC (elastic weight consolidation)
- MoE (mixture of experts routing)
- Meta-Learning (rapid adaptation)
- Transfer (cross-symbol knowledge)
- Weakness (pattern identification)
- Causality (relationship discovery)
- World Model (price forecasting)
- Counterfactual (what-if analysis)
- Monitor (AGI progress tracking)

### 2. Configuration
- [ ] config.toml has correct broker credentials
- [ ] Universe symbols defined
- [ ] Risk parameters set (max position, max exposure)

Example config.toml:
```toml
[system]
name = "Sovereign v4"

[broker]
type = "alpaca"  # or "ibkr"
paper = true     # Start with paper trading!

[alpaca]
api_key = "your_api_key"
secret_key = "your_secret_key"

[ibkr]
gateway_url = "https://localhost:5000"
account_id = "your_account"

[universe]
symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "XAUUSD"]

[telegram]
enabled = true
```

### 3. Persistence Directory
```bash
# All state files will be created in the working directory:
# - sovereign_trades.db (trade history)
# - sovereign_memory.db (AGI memory)
# - sovereign_calibrator.json (learned weights)
# - sovereign_transfer.json (cluster knowledge)
# - sovereign_moe.json (regime experts)
# - sovereign_meta.json (meta-learning state)
# - sovereign_weakness.json (identified weaknesses)
# - sovereign_causality.json (causal relationships)
# - sovereign_worldmodel.json (market model)
# - sovereign_counterfactual.json (what-if insights)
# - sovereign_monitor.json (AGI metrics history)
```

### 4. First Run (Paper Trading)
```bash
RUST_LOG=info cargo run --release
```

### 5. Verify Telegram
- [ ] Startup message received
- [ ] Trade signals appear
- [ ] Daily summary at 21:05 UTC

### 6. Monitor Initial Learning

Watch for these log messages:
```
[MEMORY] Recording trade context...
[LEARNER] Updated calibrator (X updates)
[TRANSFER] Cluster prior applied to...
[META] Reported adaptation...
[WEAKNESS] Identified X weakness patterns
[CAUSAL] Discovered X new relationships
[COUNTERFACTUAL] Identified X patterns
[MONITOR] Hourly snapshot #X
```

Learning milestones:
- After 10 trades: Calibrator should show updates
- After 50 trades: Weaknesses should be identified
- After 100 trades: Meta-learning should have adaptations
- After 7 days: Causal relationships should appear

## Go-Live Criteria

Before switching to live trading, verify:

- [ ] 100+ paper trades completed
- [ ] AGI progress > 50% (check logs for Monitor summary)
- [ ] No critical weaknesses (severity > 0.8)
- [ ] Calibrator accuracy > 55%
- [ ] All components healthy for 7 consecutive days
- [ ] Win rate > 45% on paper trades
- [ ] Positive total P&L on paper

## Go-Live Procedure

1. **Stop paper trading**
   ```bash
   # Gracefully stop (Ctrl+C) or
   pkill sovereign
   ```

2. **Backup all state files**
   ```bash
   mkdir -p backup/$(date +%Y%m%d)
   cp sovereign_*.json sovereign_*.db backup/$(date +%Y%m%d)/
   ```

3. **Switch config.toml to live credentials**
   ```toml
   [broker]
   paper = false  # LIVE MODE
   ```

4. **Start with reduced position size**
   - Edit portfolio.rs: `RISK_PER_TRADE = 0.005` (0.5% instead of 1%)
   - Or manually size down in config

5. **Start live trading**
   ```bash
   RUST_LOG=info cargo run --release 2>&1 | tee trading.log
   ```

6. **Monitor first 10 live trades closely**
   - Verify order execution matches signals
   - Check P&L tracking
   - Confirm Telegram notifications

7. **Scale up after 3 profitable days**
   - Increase position size gradually
   - Return to 1% risk per trade

## Rollback Procedure

If issues occur in live trading:

```bash
# 1. Stop immediately
pkill sovereign

# 2. Restore paper mode
# Edit config.toml: paper = true

# 3. Optionally restore previous state
cp backup/YYYYMMDD/sovereign_*.json .
cp backup/YYYYMMDD/sovereign_*.db .

# 4. Restart in paper mode
cargo run --release
```

## Daily Operations

### Morning Checklist (before market open)
- [ ] Check system is running (`ps aux | grep sovereign`)
- [ ] Review overnight logs for errors
- [ ] Verify component health in logs

### During Market Hours
- Monitor Telegram for signals
- Watch for unusual patterns

### End of Day (after 21:05 UTC)
- Daily summary should appear
- All state files auto-saved
- Check AGI metrics in logs

### Weekly Review
- Review weakness patterns
- Check meta-learning progress
- Verify causal relationships make sense
- Run integration test to verify system health

## Emergency Procedures

### Emergency Stop
```bash
# Immediate stop
pkill -9 sovereign

# Or if running in tmux
tmux kill-session -t sovereign
```

### Close All Positions (Manual)
If the system fails with open positions:

**Alpaca:**
```bash
curl -X DELETE \
  "https://paper-api.alpaca.markets/v2/positions" \
  -H "APCA-API-KEY-ID: YOUR_KEY" \
  -H "APCA-API-SECRET-KEY: YOUR_SECRET"
```

**IBKR:**
- Log into Client Portal
- Navigate to Portfolio
- Close all positions manually

### Recovery After Crash
```bash
# 1. Check what state was saved
ls -la sovereign_*.json

# 2. Restart - system will recover positions from broker
cargo run --release

# 3. Verify position recovery in logs
grep "RECOVERING" trading.log
```

## Performance Tuning

### Memory Usage
- Trade memory grows over time
- Prune old data periodically if needed
- SQLite files can be compacted

### CPU Usage
- Most CPU during bar processing
- Causal discovery runs weekly (Sunday)
- World model simulations are lightweight

### Network
- Alpaca: WebSocket streaming (low bandwidth)
- IBKR: Polling every 60s (minimal)
- Telegram: On signals only

## Troubleshooting

### "No bars received"
- Check market hours (14:30-21:00 UTC for US stocks)
- Verify API credentials
- Check symbol validity

### "Confidence below threshold"
- Normal during learning phase
- Will improve as calibrator trains
- Check memory for trade history

### "Weakness identified"
- Review weakness details in logs
- System will auto-adjust sizing
- May skip trades in weak conditions

### "Failed to save state"
- Check disk space
- Verify file permissions
- Ensure working directory is writable

## Monitoring Commands

```bash
# Watch logs in real-time
tail -f trading.log | grep -E "(SIGNAL|ORDER|ERROR)"

# Check recent trades
sqlite3 sovereign_trades.db "SELECT * FROM trades ORDER BY id DESC LIMIT 10;"

# Count trades by outcome
sqlite3 sovereign_trades.db "SELECT outcome, COUNT(*) FROM trades GROUP BY outcome;"

# Check AGI component health
grep "Monitor:" trading.log | tail -5
```

## Version History

- v4.0.0: Initial AGI release
  - Memory-based learning
  - Confidence calibration with EWC
  - Mixture of Experts
  - Meta-learning (Reptile)
  - Cross-symbol transfer
  - Weakness identification
  - Causal discovery
  - World model forecasting
  - Counterfactual analysis
  - Comprehensive monitoring
