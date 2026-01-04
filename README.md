# Sovereign v4.0

**Institutional-Grade Autonomous Trading System**

A Rust-based trading system implementing pftq's "Tech Trader" philosophy with a strict **lossless** approach: no hardcoded parameters, no magic numbers—just pure data-driven support/resistance detection and volume capitulation signals.

---

## Philosophy

### The Lossless Principle

Traditional trading systems are riddled with arbitrary parameters: 50-day moving averages, RSI thresholds of 30/70, 2x volume spikes. These "magic numbers" are the enemy of robust trading—they're curve-fitted to historical data and fail when markets evolve.

**Sovereign v4 eliminates all hardcoded thresholds:**

| Traditional (Lossy) | Sovereign (Lossless) |
|---------------------|----------------------|
| "Volume > 2x average" | Volume percentile from full distribution |
| "RSI below 30" | No oscillators—pure price action |
| "50-bar lookback" | All available historical data |
| "S/R at round numbers" | S/R granularity derived from ATR |
| "Fixed 2% position size" | Size from risk/stop-distance |

The result: a system that adapts to each symbol's unique characteristics without manual tuning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SOVEREIGN v4.0                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Universe  │───▶│SymbolAgents │───▶│  ExecutionEngine    │ │
│  │  (30 max)   │    │ (1 per sym) │    │                     │ │
│  └─────────────┘    └──────┬──────┘    │  ┌───────────────┐  │ │
│                            │           │  │ SmartRouter   │  │ │
│  ┌─────────────┐           │           │  ├───────────────┤  │ │
│  │  Portfolio  │◀──────────┤           │  │ DarkPoolRouter│  │ │
│  │ (20 pos max)│           │           │  ├───────────────┤  │ │
│  └─────────────┘           │           │  │ OrderManager  │  │ │
│                            │           │  ├───────────────┤  │ │
│  ┌─────────────┐           │           │  │ TcaAnalyzer   │  │ │
│  │   Brokers   │◀──────────┘           │  └───────────────┘  │ │
│  │ Alpaca/IBKR │                       └─────────────────────┘ │
│  └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **SymbolAgent** | `src/core/agent.rs` | Independent trader per symbol. Tracks S/R, volume, generates signals. |
| **SRLevels** | `src/core/sr.rs` | Lossless support/resistance via counting algorithm |
| **VolumeTracker** | `src/core/capitulation.rs` | Percentile-based volume spike detection |
| **ExecutionEngine** | `src/execution/mod.rs` | Tier-aware order execution orchestrator |
| **SmartRouter** | `src/execution/smart_router.rs` | Venue selection with fill rate/cost scoring |
| **Portfolio** | `src/portfolio.rs` | Position management with risk constraints |

---

## Signal Generation

### The Counting Algorithm (SRLevels)

Every price range maintains a **cross count**:
- `0` = Never crossed (strongest S/R)
- `-1` = Crossed once (weaker)
- `-N` = Crossed N times (weakest)

When price approaches a level with count `0`, it's a high-probability reversal zone.

```
Price Range    Count    Interpretation
─────────────────────────────────────────
$149.50-150.00   0     Strong resistance (untouched)
$148.00-148.50  -3     Weak (crossed 3 times)
$145.00-145.50   0     Strong support (untouched)
```

**Granularity**: Derived from 14-period ATR, not hardcoded. Volatile symbols get wider ranges.

### Volume Capitulation

Entry signals require volume confirmation:

1. Calculate volume percentile against **all historical data** (not fixed window)
2. Detect spikes using statistical distribution (mean + 2σ)
3. High-percentile volume at S/R = high-conviction entry

### Signal Types

| Signal | Trigger | Action |
|--------|---------|--------|
| **Buy** | Bounce off support + volume spike | Enter long |
| **Sell** | Bounce off resistance + volume spike | Enter short |
| **Cover** | Long position + resistance rejection | Exit long |
| **Short** | Short position + support bounce | Exit short |
| **Hold** | No actionable setup | Wait |

---

## Account Tiers

Execution capabilities scale with account size:

| Tier | Balance | Features |
|------|---------|----------|
| **Retail** | < $100K | Alpaca market orders only |
| **Semi-Institutional** | $100K - $1M | IBKR algos, IEX routing |
| **Institutional** | $1M+ | Dark pools, FIX protocol, prime broker |

```rust
let tier = AccountTier::from_balance(dec!(500_000)); // SemiInstitutional
assert!(tier.supports_algos());      // true
assert!(!tier.supports_dark_pools()); // false (need $1M+)
```

---

## Execution Algorithms

For larger orders, Sovereign slices execution to minimize market impact:

| Algorithm | Use Case | Description |
|-----------|----------|-------------|
| **VWAP** | Low urgency, large orders | Volume-weighted scheduling (U-shaped intraday profile) |
| **TWAP** | Medium urgency | Time-weighted with randomization |
| **POV** | Track volume | Percentage of volume participation |
| **Iceberg** | Hide size | Show small quantity, refill on fills |
| **Adaptive** | High volatility | Adjusts aggression based on conditions |
| **IS** | Minimize shortfall | Implementation shortfall optimization |

Algorithm selection is automatic based on order size, ADV participation, and urgency.

---

## Dark Pool Integration

Institutional tier routes to dark pools for price improvement:

| Venue | Type | Min Notional |
|-------|------|--------------|
| Sigma-X | Goldman Sachs | $10K |
| CrossFinder | Credit Suisse | $10K |
| MS Pool | Morgan Stanley | $10K |
| Liquidnet | Block trading | $100K |
| Level ATS | Alternative | $10K |

The `DarkPoolRouter` scores venues by:
- Historical fill rate
- Average price improvement (bps)
- Match speed
- Block capability

---

## Transaction Cost Analysis

Every execution is analyzed post-trade:

```
┌────────────────────────────────────────┐
│           TCA Report                   │
├────────────────────────────────────────┤
│ Arrival Slippage:    +2.5 bps          │
│ VWAP Performance:    -1.2 bps          │
│ Implementation SF:   +3.8 bps          │
│ Market Impact:       +1.5 bps          │
│ ─────────────────────────────────────  │
│ Quality Grade:       GOOD              │
└────────────────────────────────────────┘
```

Grades: Excellent (< -5 bps) → Good → Acceptable → Poor → Very Poor

---

## Portfolio Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max positions | 20 | Diversification without over-dilution |
| Max exposure per side | 200% | Can be 200% long AND 200% short |
| Risk per trade | 1% of equity | Standard risk management |
| Position sizing | Risk / Stop distance | S/R-derived stops |

---

## Configuration

Create `config.toml` from the example:

```toml
[system]
name = "sovereign"
log_level = "info"

[broker]
type = "alpaca"  # or "ibkr"
paper = true

[alpaca]
api_key = "your-api-key"
secret_key = "your-secret-key"

[ibkr]
gateway_url = "https://localhost:5000"
account_id = "your-account"

[telegram]
enabled = true
bot_token = "your-bot-token"
chat_id = "your-chat-id"

[portfolio]
initial_balance = 100000

[universe]
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
```

**Note**: Configuration contains ONLY infrastructure settings. No strategy parameters—that would violate lossless philosophy.

---

## Building & Running

### Prerequisites

- Rust 1.70+
- SQLite3
- (For IBKR) Client Portal Gateway running on `localhost:5000`

### Build

```bash
git clone https://github.com/dimasergei/sovereign-v4.git
cd sovereign-v4
cargo build --release
```

### Run

```bash
# Paper trading with Alpaca
./target/release/sovereign

# Or specify config
./target/release/sovereign --config /path/to/config.toml
```

### Market Hours

- **US Stocks**: Monday-Friday, 09:30-16:00 ET (14:30-21:00 UTC)
- **Crypto**: 24/7 (if enabled in universe)

---

## Project Structure

```
sovereign-v4/
├── src/
│   ├── main.rs              # Entry point, event loop
│   ├── lib.rs               # Library exports
│   ├── config.rs            # Configuration loading
│   │
│   ├── core/                # Trading logic
│   │   ├── agent.rs         # SymbolAgent (signal generation)
│   │   ├── sr.rs            # Support/Resistance counting
│   │   ├── capitulation.rs  # Volume spike detection
│   │   ├── types.rs         # Core data structures
│   │   └── health.rs        # Data gap monitoring
│   │
│   ├── broker/              # Exchange integrations
│   │   ├── alpaca.rs        # Alpaca REST API
│   │   ├── ibkr.rs          # IBKR Client Portal
│   │   └── prime_broker.rs  # FIX protocol (institutional)
│   │
│   ├── execution/           # Institutional execution
│   │   ├── mod.rs           # ExecutionEngine
│   │   ├── algorithms.rs    # VWAP, TWAP, POV, Iceberg
│   │   ├── smart_router.rs  # Venue selection
│   │   ├── dark_pool.rs     # Dark pool routing
│   │   ├── order_manager.rs # Order lifecycle
│   │   ├── tca.rs           # Transaction cost analysis
│   │   └── venue.rs         # Exchange definitions
│   │
│   ├── portfolio.rs         # Position management
│   ├── universe.rs          # Symbol universe & sectors
│   │
│   ├── data/                # Market data & storage
│   │   ├── database.rs      # Trade history (SQLite)
│   │   └── alpaca_stream.rs # WebSocket data feed
│   │
│   └── comms/               # Notifications
│       └── telegram.rs      # Telegram alerts
│
├── config.toml.example      # Configuration template
├── Cargo.toml               # Dependencies
└── README.md                # This file
```

---

## Data Flow

```
                    ┌──────────────────┐
                    │  Market Data     │
                    │  (WebSocket)     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  1-Minute Bar    │
                    │  OHLCV           │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │       SymbolAgent            │
              │  ┌────────────────────────┐  │
              │  │ SRLevels.update()      │  │
              │  │ VolumeTracker.update() │  │
              │  │ check_signals()        │  │
              │  └────────────────────────┘  │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    AgentSignal               │
              │  {signal, conviction,        │
              │   support, resistance,       │
              │   volume_percentile}         │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    ExecutionEngine           │
              │  ├─ recommend_algorithm()    │
              │  ├─ calculate_position_size()│
              │  ├─ route_order()            │
              │  └─ get_dark_pool_decision() │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    Broker                    │
              │  buy() / sell() / close()    │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    TradeDb                   │
              │  record_open/close()         │
              └──────────────────────────────┘
```

---

## Development Guidelines

### Contributing

When modifying Sovereign, maintain the lossless philosophy:

1. **No hardcoded thresholds** - Derive from data
2. **No magic numbers** - If you need a constant, it must have statistical justification
3. **No cross-symbol optimization** - Each SymbolAgent is independent
4. **Test coverage** - All new code needs tests
5. **Type safety** - Use `Decimal` for prices/quantities, never `f64`

### Running Tests

```bash
cargo test
```

Current: **201 tests** (138 library + 63 integration)

---

## Roadmap

- [ ] Backtesting framework with historical data replay
- [ ] Options strategy support (covered calls, spreads)
- [ ] Multi-region deployment (co-location)
- [ ] Machine learning signal enhancement (while maintaining lossless principles)
- [ ] Real-time performance dashboard

---

## License

Proprietary. All rights reserved.

---

## Acknowledgments

- **pftq** - Tech Trader philosophy and counting-based S/R methodology
- **Alpaca** - Commission-free trading API
- **Interactive Brokers** - Professional-grade execution
