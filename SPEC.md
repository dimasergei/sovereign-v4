# Sovereign v4 Trading System Specification

## 1. Philosophy & Design Goals

**Core Philosophy**: Fully autonomous, self-improving trading system with constitutional safety constraints. The system operates without human intervention while respecting immutable safety limits.

**Design Goals**:
- **Full Autonomy**: No human approval workflow - modifications auto-deploy if constitutional, auto-reject if not
- **Self-Modification**: System learns from weaknesses and automatically generates/deploys trading rules
- **Constitutional Safety**: Hard limits that cannot be overridden (max drawdown, position limits)
- **Adaptive Learning**: Regime detection, causal analysis, and counterfactual reasoning
- **Transparency**: All modifications logged to Telegram, full audit trail

**Differentiators from v3**:
- Removed human approval workflow for self-modifications
- Added causal inference (FCI algorithm, do-calculus)
- Sharded memory architecture for scalability
- Foundation model integration for regime detection

---

## 2. Architecture

### Language & Platform
- **Language**: Rust (async/tokio runtime)
- **Platform**: Linux (tested on kernel 4.4.0)
- **Database**: SQLite (rusqlite) for trade memory
- **Cache**: Redis for distributed state (optional)

### Broker Integration
- **Primary**: Interactive Brokers TWS via `ibapi` crate
- **Paper Trading**: Uses MIDPOINT data (TRADES unavailable on paper)
- **Backup**: Alpaca API integration available

### Core Components
```
src/
├── core/
│   ├── selfmod.rs       # Self-modification engine with constitutional guard
│   ├── regime.rs        # Market regime detection (Trending/Ranging/Volatile)
│   ├── causality.rs     # FCI causal discovery, PAG, do-calculus
│   ├── weakness.rs      # Weakness analyzer (identifies poor patterns)
│   ├── counterfactual.rs # "What-if" trade analysis
│   ├── learner.rs       # Neural network confidence calibration
│   ├── metalearner.rs   # Meta-learning adaptation
│   ├── foundation.rs    # Transformer-based foundation model
│   ├── streaming.rs     # Online learning pipeline
│   ├── sharded_memory.rs # Distributed trade storage
│   ├── codegen.rs       # Runtime code generation
│   ├── monitor.rs       # AGI progress tracking
│   └── transfer.rs      # Transfer learning between symbols
├── comms/
│   └── telegram.rs      # Telegram bot notifications & commands
├── execution/
│   ├── order_manager.rs # Order lifecycle management
│   ├── smart_router.rs  # Venue selection
│   ├── dark_pool.rs     # Dark pool routing
│   └── algos.rs         # TWAP, VWAP, Iceberg algorithms
├── broker/
│   ├── tws.rs           # IBKR TWS integration
│   └── alpaca.rs        # Alpaca fallback
├── data/
│   ├── memory.rs        # Trade memory SQLite
│   └── alpaca_stream.rs # Market data streaming
└── portfolio.rs         # Position management
```

### State Management
- **Constitution**: Immutable safety constraints (max 10% drawdown, 5% position, 3% daily loss)
- **Rule Engine**: Self-generated trading rules with conditions/actions
- **Modification History**: Full audit log of all auto-deployed changes
- **Trade Memory**: SQLite with sharded architecture for scale

---

## 3. Signal Generation

### Entry Logic
- **Regime-Aware**: Different strategies per regime (TrendingUp, TrendingDown, Ranging, Volatile)
- **Support/Resistance Score**: Integer score based on price levels
- **Volume Confirmation**: Volume percentile thresholds
- **Confidence Calibration**: Neural network calibrated confidence scores
- **Causal Factors**: Signals informed by discovered causal relationships

### Exit Logic
- **ATR-Based Stops**: Configurable ATR multipliers for stop loss
- **Take Profit**: ATR-based targets
- **Time-Based**: Maximum hold duration rules
- **Regime Change**: Exit on regime transition

### Self-Generated Rules
System automatically creates rules from identified weaknesses:
```rust
RuleCondition::RegimeIs(Regime::Volatile)
  .and(RuleCondition::VolumeBelow(30.0))
  => RuleAction::SkipTrade { reason: "Low volume in volatile regime" }
```

### Indicators Used
- ATR (Average True Range)
- Volume percentile
- Support/Resistance levels
- Regime state (HMM-based)
- Causal factor activation

---

## 4. Risk Management

### Position Sizing
- **Max Position**: 5% of capital per position (constitutional limit)
- **Scaling**: Based on confidence score and regime

### Stop Loss
- **Method**: ATR-multiplier based
- **Default**: 1.5x ATR
- **Adjustment**: Self-modification can tighten/widen within limits

### Constitutional Limits (Immutable)
```rust
max_position_size: 0.05      // 5% max per position
max_daily_loss: 0.03         // 3% max daily loss
max_drawdown: 0.10           // 10% max drawdown - HARD STOP
min_confidence_for_trade: 0.40
max_rule_changes_per_day: 3
max_active_rules: 50
```

### Kill Conditions
- Drawdown exceeds 10%: System halts all trading
- Daily loss exceeds 3%: No new positions for the day
- Consecutive losses trigger rule evaluation

### Forbidden Modifications
These can NEVER be changed by self-modification:
- `max_drawdown`
- `constitution`
- `forbidden_modifications`

---

## 5. Universe

### Asset Classes
- US Equities (primary)
- Exchange: SMART routing via IBKR

### Symbol Selection
- Configurable watchlist
- Liquidity filtering available
- Sector exposure tracking

### Transfer Learning
- Symbols clustered by behavior similarity
- Negative transfer detection prevents harmful knowledge sharing
- Blacklist/graylist for blocked symbol pairs

---

## 6. Execution

### Order Types
- Market orders (primary for fills)
- Limit orders (for algos)

### Execution Algorithms
- **TWAP**: Time-weighted average price
- **VWAP**: Volume-weighted average price
- **Iceberg**: Hidden size orders

### Smart Routing
- Multi-venue routing decisions
- Dark pool access
- Cost vs urgency optimization

### Filling Mode
- Paper: MIDPOINT historical data polling (60s intervals)
- Live: Real-time TWS streaming

---

## 7. Performance Metrics

| Metric | Value |
|--------|-------|
| Win Rate | ___ |
| Profit Factor | ___ |
| Sharpe Ratio | ___ |
| Max Drawdown | ___ |
| Avg Trade Duration | ___ |
| Total Trades | ___ |
| Total P&L | ___ |

---

## 8. Current Status

### Development Stage
- **Phase**: Paper trading validation
- **Broker**: IBKR TWS (paper account)
- **Data**: MIDPOINT polling every 60s

### Known Issues
- Paper accounts don't support TRADES data type (using MIDPOINT)
- Need real-time streaming for live trading

### Recent Changes
- Made self-modification fully autonomous (no human approval)
- Removed /pending, /approve, /reject commands
- Added /history for modification audit trail

### Next Steps
- [ ] Live trading readiness validation
- [ ] Real-time data streaming integration
- [ ] Performance metrics collection
- [ ] Prop firm rule compliance verification

---

## 9. Prop Firm Compatibility

### Target Accounts
- _To be specified_

### Rule Compliance
| Rule | System Constraint | Status |
|------|-------------------|--------|
| Max Daily Loss | 3% (constitutional) | Compliant |
| Max Drawdown | 10% (constitutional) | Compliant |
| Max Position Size | 5% (constitutional) | Compliant |
| Consistency | Self-modification rules | Adaptive |

### Monitoring Commands (Telegram)
```
/rules        - List active trading rules
/constitution - Show safety constraints
/history      - Show modification history
/capital      - Show capital status
/readiness    - Check live trading readiness
/rollback <id> - Rollback a deployed modification
```

---

## Appendix: Self-Modification Flow

```
┌─────────────────────────────────────┐
│ Weakness/Insight Detected           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ propose_modification()              │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│ Constitution│  │ Constitution│
│ PASSES      │  │ FAILS       │
└──────┬──────┘  └──────┬──────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│ AutoDeployed│  │ AutoRejected│
│ + Telegram  │  │ + Telegram  │
│ notification│  │ notification│
└─────────────┘  └─────────────┘
```

No human approval required. Constitutional constraints are the only gate.
