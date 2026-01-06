# Sovereign v4 - AGI Gap Analysis

**Document Purpose**: Identify gaps between current v4 capabilities and AGI-level autonomous trading
**Analysis Date**: 2026-01-06
**Codebase Version**: v4.0 (Lossless Implementation)

---

## 1. CURRENT ARCHITECTURE

### 1.1 Module Overview

```
sovereign-v4/src/
├── main.rs              # Entry point, event loops (Alpaca/IBKR)
├── lib.rs               # Library exports
├── config.rs            # TOML configuration loader
│
├── core/                # [TRADING LOGIC - 1,200 LOC]
│   ├── agent.rs         # SymbolAgent - independent trader per symbol
│   ├── sr.rs            # SRLevels - lossless counting algorithm
│   ├── capitulation.rs  # VolumeTracker - percentile-based detection
│   ├── types.rs         # Candle, Signal, Position, Decision
│   └── health.rs        # HealthMonitor - data gap detection
│
├── broker/              # [EXCHANGE INTEGRATION - 1,500 LOC]
│   ├── alpaca.rs        # Alpaca REST API (paper/live)
│   ├── ibkr.rs          # IBKR Client Portal Gateway
│   ├── prime_broker.rs  # FIX protocol (placeholder)
│   └── mod.rs           # Broker traits and errors
│
├── execution/           # [INSTITUTIONAL EXECUTION - 5,800 LOC]
│   ├── mod.rs           # ExecutionEngine, AccountTier
│   ├── algorithms.rs    # VWAP, TWAP, POV, Iceberg, Adaptive
│   ├── smart_router.rs  # Venue selection & scoring
│   ├── dark_pool.rs     # Dark pool routing
│   ├── order_manager.rs # Order lifecycle management
│   ├── tca.rs           # Transaction cost analysis
│   └── venue.rs         # Exchange definitions
│
├── portfolio.rs         # Position management, risk constraints
├── universe.rs          # Symbol universe, sector classification
│
├── data/                # [DATA LAYER - 400 LOC]
│   ├── database.rs      # SQLite trade history
│   └── alpaca_stream.rs # WebSocket market data
│
└── comms/               # [NOTIFICATIONS - 150 LOC]
    └── telegram.rs      # Telegram alerts
```

### 1.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────────┐
│  MARKET DATA │     │  HISTORICAL  │     │         BROKER STATE            │
│  (WebSocket) │     │    BARS      │     │  (Positions, Account)           │
└──────┬───────┘     └──────┬───────┘     └─────────────┬──────────────────┬─┘
       │                    │                           │                  │
       ▼                    ▼                           ▼                  │
┌──────────────────────────────────────────────────────────────────────┐  │
│                      SymbolAgent (per symbol)                        │  │
│  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │ ┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐  │ │  │
│  │ │   SRLevels    │  │  VolumeTracker   │  │  Position State   │  │ │  │
│  │ │ (counting     │  │  (percentile     │  │  (Long/Short/     │  │ │  │
│  │ │  algorithm)   │  │   ranking)       │  │   None)           │  │ │  │
│  │ └───────────────┘  └──────────────────┘  └───────────────────┘  │ │  │
│  │                                                                  │ │  │
│  │ process_bar() → check_signals() → AgentSignal                   │ │  │
│  └─────────────────────────────────────────────────────────────────┘ │  │
└──────────────────────────────────────────┬───────────────────────────┘  │
                                           │                              │
                                           ▼                              │
┌──────────────────────────────────────────────────────────────────────┐  │
│                        Portfolio                                     │  │
│  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  should_execute(signal) → position_size() → execute()          │  │  │
│  │  • Max 20 positions, 200% exposure/side                        │  │  │
│  │  • Position size = 1% risk / stop_distance                     │  │  │
│  └────────────────────────────────────────────────────────────────┘  │  │
└──────────────────────────────────────────┬───────────────────────────┘  │
                                           │                              │
                                           ▼                              │
┌──────────────────────────────────────────────────────────────────────┐  │
│                    ExecutionEngine (Institutional)                   │  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │ SmartRouter  │  │ DarkPoolRtr  │  │ OrderManager │                │  │
│  │ (venue score)│  │ (allocation) │  │ (lifecycle)  │                │  │
│  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│                                                                       │  │
│  Algorithm Selection: Market | VWAP | TWAP | Iceberg | Adaptive     │  │
└──────────────────────────────────────────┬───────────────────────────┘  │
                                           │                              │
                                           ▼                              │
┌──────────────────────────────────────────────────────────────────────┐  │
│                         Broker                                       │  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │  │
│  │    Alpaca        │  │      IBKR        │  │   Prime Broker     │  │  │
│  │  (retail/paper)  │  │  (institutional) │  │   (FIX - future)   │  │  │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘  │  │
└──────────────────────────────────────────┬───────────────────────────┘  │
                                           │                              │
                                           ▼                              │
┌──────────────────────────────────────────────────────────────────────┐  │
│                       TradeDb (SQLite)                               │◀─┘
│  • trades: ticket, direction, entry/exit, S/L, T/P, profit          │
│  • daily_stats: trades, wins, losses, PnL                           │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 Trading Logic (Lossless S/R)

```
SIGNAL GENERATION ALGORITHM
═══════════════════════════

1. S/R LEVEL TRACKING (SRLevels)
   ┌────────────────────────────────────────────────────────────────┐
   │ For each price range (granularity = ATR / 2):                 │
   │   • Count = 0 → Never crossed (STRONGEST S/R)                 │
   │   • Count = -1 → Crossed once                                 │
   │   • Count = -N → Crossed N times (weaker)                     │
   │                                                                │
   │ Support = Nearest level below price with best (lowest) count  │
   │ Resistance = Nearest level above price with best count        │
   └────────────────────────────────────────────────────────────────┘

2. VOLUME CAPITULATION (VolumeTracker)
   ┌────────────────────────────────────────────────────────────────┐
   │ • Track ALL historical volume observations (no fixed window)  │
   │ • Percentile = rank(current_vol) / count(all_vols) * 100      │
   │ • Capitulation = percentile >= 80 AND recent_highest          │
   │ • Elevated = percentile >= mean + 1σ                          │
   └────────────────────────────────────────────────────────────────┘

3. ENTRY CONDITIONS
   ┌────────────────────────────────────────────────────────────────┐
   │ BUY:   at_support AND (capitulation OR elevated_volume)       │
   │        AND is_down_day                                         │
   │                                                                │
   │ SHORT: at_resistance AND (capitulation OR elevated_volume)    │
   │        AND is_up_day                                           │
   └────────────────────────────────────────────────────────────────┘

4. EXIT CONDITIONS
   ┌────────────────────────────────────────────────────────────────┐
   │ SELL (close long):  at_resistance                             │
   │ COVER (close short): at_support                               │
   └────────────────────────────────────────────────────────────────┘
```

### 1.4 Parameters Inventory

| Category | Parameter | Source | Lossless? |
|----------|-----------|--------|-----------|
| **S/R Granularity** | ATR / 2 | Derived from price data | ✅ Yes |
| **Volume Threshold** | Percentile ranking | Derived from volume distribution | ✅ Yes |
| **Position Size** | 1% risk / stop_distance | Derived from S/R placement | ✅ Yes |
| **Max Positions** | 20 | Hardcoded operational limit | ⚠️ Operational |
| **Max Exposure** | 200% per side | Hardcoded operational limit | ⚠️ Operational |
| **ATR Period** | 14 bars (rolling window) | Hardcoded | ⚠️ Operational |
| **Health Timeout** | 90 seconds | Hardcoded operational limit | ⚠️ Operational |

**Summary**: Core trading logic is lossless. Only operational limits (max positions, exposure caps) are hardcoded.

---

## 2. MEMORY/STORAGE

### 2.1 Persistent Storage

| Storage | Purpose | Location | Format |
|---------|---------|----------|--------|
| **TradeDb** | Trade history | `sovereign_trades.db` | SQLite |
| **Config** | System settings | `config.toml` | TOML |

**TradeDb Schema**:
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    ticket INTEGER NOT NULL,
    direction TEXT NOT NULL,        -- 'BUY' | 'SELL' | 'SHORT' | 'COVER'
    lots REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    sl REAL NOT NULL,               -- Stop loss
    tp REAL NOT NULL,               -- Take profit
    profit REAL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    conviction INTEGER NOT NULL     -- 0-100
);

CREATE TABLE daily_stats (
    date TEXT PRIMARY KEY,
    trades INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    profit REAL NOT NULL
);
```

### 2.2 Runtime State (In-Memory Only)

| State | Lifetime | Lost on Restart? |
|-------|----------|------------------|
| S/R levels | Session | ✅ Yes (rebuilt from historical bars) |
| Volume distributions | Session | ✅ Yes (rebuilt from historical bars) |
| Active positions | Session | ❌ No (recovered from broker) |
| Venue statistics (EMA) | Session | ✅ Yes |
| TCA history | Session | ✅ Yes |
| Algorithm state | Session | ✅ Yes |

### 2.3 Learned State

**Current**: ❌ **NONE**

The system does NOT persist:
- S/R level effectiveness metrics
- Which S/R levels led to profitable trades
- Volume threshold calibration per symbol
- Venue performance history
- Execution quality patterns
- Strategy parameter adaptations

---

## 3. LEARNING CAPABILITY

### 3.1 Machine Learning Components

**Current**: ❌ **NONE**

```
CODEBASE SEARCH RESULTS:
  "learn"           → 0 matches in trading logic
  "train"           → 0 matches
  "model"           → 0 matches (except "market impact model" - static formula)
  "neural"          → 0 matches
  "ML"              → 0 matches
  "gradient"        → 0 matches
  "backprop"        → 0 matches
  "tensorflow"      → 0 matches
  "pytorch"         → 0 matches
```

### 3.2 Adaptation Mechanisms

| Capability | Current State | How It Works |
|------------|---------------|--------------|
| **S/R Adaptation** | ⚠️ Passive | Counts update as price crosses levels (no learning from outcomes) |
| **Volume Calibration** | ⚠️ Passive | Percentiles update as new data arrives (no learning from trades) |
| **Venue Selection** | ⚠️ Session-only | EMA updates fill rates (lost on restart) |
| **Algorithm Selection** | ⚠️ Rule-based | Static rules based on order size/ADV (no learning) |

### 3.3 Can It Update Without Code Changes?

| Aspect | Can Update? | How? |
|--------|-------------|------|
| Trading symbols | ✅ Yes | Edit `config.toml` universe |
| Broker selection | ✅ Yes | Edit `config.toml` broker type |
| Position limits | ❌ No | Hardcoded constants |
| Entry/exit logic | ❌ No | Requires code changes |
| New strategies | ❌ No | Requires code changes |
| Risk parameters | ❌ No | Hardcoded in portfolio.rs |

---

## 4. BROKER INTEGRATION

### 4.1 Alpaca (Retail)

| Feature | Status | Notes |
|---------|--------|-------|
| **Connection** | ✅ Working | REST API + WebSocket |
| **Paper Trading** | ✅ Working | Configurable |
| **Live Trading** | ✅ Working | Configurable |
| **Market Orders** | ✅ Supported | Primary order type |
| **Limit Orders** | ⚠️ Partial | In execution engine, not main.rs |
| **Stop Loss** | ✅ Supported | Bracket orders |
| **Take Profit** | ✅ Supported | Bracket orders |
| **Trailing Stops** | ❌ Not implemented | |
| **Historical Data** | ✅ Working | Daily bars for S/R bootstrap |
| **Streaming Data** | ✅ Working | 1-min bars via WebSocket |

### 4.2 IBKR (Institutional)

| Feature | Status | Notes |
|---------|--------|-------|
| **Connection** | ✅ Working | Client Portal Gateway (localhost:5000) |
| **Authentication** | ✅ Working | Cookie-based with tickle keep-alive |
| **Market Orders** | ✅ Supported | Via Client Portal API |
| **Limit Orders** | ❌ Not in main.rs | Execution engine only |
| **Stop Loss** | ❌ Not implemented | |
| **Take Profit** | ❌ Not implemented | |
| **Contract Search** | ✅ Working | POST with JSON body |
| **Historical Data** | ✅ Working | Daily bars |
| **Position Recovery** | ✅ Working | |

### 4.3 Prime Broker (Future)

| Feature | Status | Notes |
|---------|--------|-------|
| **FIX Protocol** | ⚠️ Placeholder | Structs defined, no connection |
| **Dark Pool Access** | ⚠️ Simulated | Router logic exists, no real venues |
| **Block Trading** | ⚠️ Simulated | Logic exists, no real execution |

---

## 5. AGI GAP ASSESSMENT

### 5.1 Capability Ratings

```
AGI CAPABILITY ASSESSMENT
═════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ CAPABILITY              │ CURRENT │ NOTES                          │
├─────────────────────────┼─────────┼────────────────────────────────┤
│ Continuous Learning     │   5%    │ No ML, no outcome feedback     │
│ Knowledge Transfer      │   0%    │ No cross-symbol learning       │
│ Self-Directed Improve   │   0%    │ Cannot modify own strategy     │
│ Persistent Memory       │  15%    │ Trade history only, no learned │
│                         │         │ weights or calibrations        │
│ Causal Reasoning        │  10%    │ S/R counting is correlational, │
│                         │         │ not causal                     │
│ Meta-Learning           │   0%    │ Cannot learn how to learn      │
│ Adaptive Parameters     │   5%    │ Session-only venue EMA         │
│ Strategy Discovery      │   0%    │ No new pattern recognition     │
│ Risk Self-Assessment    │   5%    │ Static limits, no dynamic adj  │
│ Market Regime Detection │   0%    │ No regime classification       │
└─────────────────────────┴─────────┴────────────────────────────────┘

                           OVERALL AGI READINESS: 4%
```

### 5.2 Gap Visualization

```
AGI CAPABILITY RADAR
════════════════════

                    Continuous Learning
                           │ 5%
                           │
                           ▼
           Strategy    ────┼────    Knowledge
           Discovery       │        Transfer
              0%           │           0%
                    ╲      │      ╱
                     ╲     │     ╱
                      ╲    │    ╱
          Adaptive ────────┼────────── Self-Directed
          Parameters       │           Improvement
              5%           │               0%
                      ╱    │    ╲
                     ╱     │     ╲
                    ╱      │      ╲
              Risk     ────┼────    Causal
          Assessment       │        Reasoning
              5%           │           10%
                           │
                           ▼
                    Persistent Memory
                          15%

                    ══════════════════
                    Current: 4% AGI
                    ══════════════════
```

### 5.3 Detailed Gap Analysis

#### Gap 1: Continuous Learning (Current: 5%)

**What's Missing**:
- No feedback loop from trade outcomes to strategy
- S/R levels don't learn which levels are actually predictive
- Volume thresholds don't calibrate based on success rates
- No reinforcement learning from P&L

**Required Components**:
```
┌────────────────────────────────────────────────────────────┐
│ FEEDBACK LOOP (Missing)                                    │
│                                                            │
│   Trade Outcome                                            │
│        │                                                   │
│        ▼                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ │
│   │  Evaluate   │────▶│  Update     │────▶│  Improve    │ │
│   │  Decision   │     │  Weights    │     │  Strategy   │ │
│   └─────────────┘     └─────────────┘     └─────────────┘ │
│        │                     │                   │        │
│        └─────────────────────┴───────────────────┘        │
│                        MISSING                             │
└────────────────────────────────────────────────────────────┘
```

#### Gap 2: Knowledge Transfer (Current: 0%)

**What's Missing**:
- Each SymbolAgent is completely independent (by design)
- No learned patterns shared between symbols
- No sector-wide learnings
- No market regime learnings applied globally

**Required Components**:
- Shared embedding space for pattern representation
- Cross-symbol attention mechanism
- Hierarchical learning (symbol → sector → market)

#### Gap 3: Persistent Memory (Current: 15%)

**What's Persisted**:
- Trade history (entries, exits, P&L)
- Daily statistics

**What's Missing**:
- S/R level effectiveness scores
- Volume threshold calibrations
- Venue performance history
- Strategy weights/parameters
- Market regime classifications
- Pattern success rates

#### Gap 4: Causal Reasoning (Current: 10%)

**Current State**:
- S/R counting is purely correlational ("price bounced here before")
- No understanding of WHY levels work
- No distinction between correlation and causation

**Required Components**:
- Causal graph construction
- Intervention analysis
- Counterfactual reasoning

---

## 6. PRIORITY RECOMMENDATIONS

### 6.1 High Priority (Foundation)

| Priority | Gap | Effort | Impact | Description |
|----------|-----|--------|--------|-------------|
| **P1** | Persistent Memory | Medium | High | Store S/R effectiveness, volume calibrations per symbol |
| **P2** | Outcome Feedback | Medium | High | Track which S/R levels led to profitable trades |
| **P3** | Adaptive Thresholds | Low | Medium | Let volume percentile thresholds evolve based on results |

### 6.2 Medium Priority (Enhancement)

| Priority | Gap | Effort | Impact | Description |
|----------|-----|--------|--------|-------------|
| **P4** | Venue Learning | Low | Medium | Persist venue EMA statistics across restarts |
| **P5** | Regime Detection | Medium | Medium | Classify bull/bear/sideways markets |
| **P6** | Risk Self-Tuning | Medium | Medium | Adjust position sizing based on recent performance |

### 6.3 Long-Term (AGI Path)

| Priority | Gap | Effort | Impact | Description |
|----------|-----|--------|--------|-------------|
| **P7** | Continuous Learning | High | Very High | Add RL agent for strategy optimization |
| **P8** | Knowledge Transfer | High | High | Shared pattern embeddings across symbols |
| **P9** | Strategy Discovery | Very High | Very High | Automated pattern mining and strategy generation |
| **P10** | Self-Improvement | Very High | Transformative | System that can modify its own code |

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1: Memory Foundation (2-3 weeks)

```
┌────────────────────────────────────────────────────────────────┐
│ MEMORY SCHEMA EXTENSION                                        │
│                                                                │
│  sr_effectiveness                                              │
│  ├── symbol TEXT                                               │
│  ├── price_level REAL                                          │
│  ├── touch_count INTEGER                                       │
│  ├── bounce_success_rate REAL                                  │
│  ├── avg_profit_when_hit REAL                                  │
│  └── last_updated TIMESTAMP                                    │
│                                                                │
│  volume_calibration                                            │
│  ├── symbol TEXT                                               │
│  ├── optimal_percentile REAL                                   │
│  ├── success_rate_at_threshold REAL                            │
│  └── last_calibrated TIMESTAMP                                 │
│                                                                │
│  venue_performance                                             │
│  ├── venue_id TEXT                                             │
│  ├── fill_rate_ema REAL                                        │
│  ├── slippage_ema REAL                                         │
│  ├── improvement_ema REAL                                      │
│  └── total_orders INTEGER                                      │
└────────────────────────────────────────────────────────────────┘
```

### Phase 2: Feedback Loops (3-4 weeks)

```rust
// Pseudo-code for outcome tracking
impl SymbolAgent {
    fn record_trade_outcome(&mut self, entry_level: Decimal, exit_price: Decimal, profit: Decimal) {
        // Update S/R effectiveness
        self.sr.record_bounce_outcome(entry_level, profit > 0);

        // Update volume threshold if this was a volume-triggered entry
        if self.last_entry_was_volume_triggered {
            self.volume.update_threshold_effectiveness(profit > 0);
        }

        // Persist to database
        self.db.store_sr_effectiveness(self.symbol, entry_level, profit);
    }
}
```

### Phase 3: Adaptive Learning (4-6 weeks)

- Implement gradient-free optimization for threshold tuning
- Add Bayesian updating for S/R confidence scores
- Implement regime detection (HMM or similar)

### Phase 4: AGI Foundation (3-6 months)

- Add reinforcement learning agent (PPO/A2C)
- Implement pattern embedding space
- Add cross-symbol attention mechanism
- Build strategy discovery pipeline

---

## 8. CONCLUSION

Sovereign v4 is a **well-architected rule-based trading system** with:

✅ Clean separation of concerns
✅ Lossless trading philosophy (no magic numbers)
✅ Solid execution infrastructure
✅ Multiple broker support

However, it has **significant gaps for AGI-level autonomy**:

❌ No continuous learning from outcomes
❌ No persistent learned state
❌ No cross-symbol knowledge transfer
❌ No self-improvement capability
❌ No causal reasoning

**Overall AGI Readiness: 4%**

The foundation is solid, but the system is fundamentally reactive rather than learning. The path to AGI requires implementing persistent memory, feedback loops, and eventually ML-based strategy discovery.

---

*Generated by AGI Gap Analysis Tool v1.0*
