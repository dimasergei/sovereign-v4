# Sovereign v4.0 "Perpetual" - Rust Development Roadmap

## Mission
Build an institutional-grade, fully autonomous trading system in Rust that can run for 12+ years without human intervention, capable of scaling to 1000+ agents.

---

## Phase 0: Environment Setup (Day 1)

### Install Rust
```bash
# Windows (PowerShell as Admin)
winget install Rustlang.Rust.MSVC

# Or download from: https://rustup.rs/
```

### Install VS Code Extensions
1. **rust-analyzer** - Core Rust support
2. **CodeLLDB** - Debugging
3. **Even Better TOML** - Cargo.toml support
4. **Error Lens** - Inline error display

### Verify Installation
```bash
rustc --version
cargo --version
```

### Create First Project
```bash
cargo new sovereign_v4
cd sovereign_v4
cargo run
```

---

## Phase 1: Rust Fundamentals (Weeks 1-4)

### Week 1: Core Syntax
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | Variables, Types | Rust Book Ch 3 | exercises/day01_variables.rs |
| 2 | Functions | Rust Book Ch 3 | exercises/day02_functions.rs |
| 3 | Control Flow | Rust Book Ch 3 | exercises/day03_control.rs |
| 4 | Ownership | Rust Book Ch 4 | exercises/day04_ownership.rs |
| 5 | References & Borrowing | Rust Book Ch 4 | exercises/day05_borrowing.rs |
| 6 | Slices | Rust Book Ch 4 | exercises/day06_slices.rs |
| 7 | Review + Mini Project | - | Build: Price tracker struct |

### Week 2: Data Structures
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | Structs | Rust Book Ch 5 | Build: Candle struct |
| 2 | Methods | Rust Book Ch 5 | Add methods to Candle |
| 3 | Enums | Rust Book Ch 6 | Build: TradeSignal enum |
| 4 | Pattern Matching | Rust Book Ch 6 | Match on signals |
| 5 | Option<T> | Rust Book Ch 6 | Handle missing data |
| 6 | Result<T, E> | Rust Book Ch 9 | Error handling |
| 7 | Review + Mini Project | - | Build: Trade struct with validation |

### Week 3: Collections & Modules
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | Vectors | Rust Book Ch 8 | Store candle history |
| 2 | HashMaps | Rust Book Ch 8 | Build: Lossless levels |
| 3 | Strings | Rust Book Ch 8 | Parse market data |
| 4 | Modules | Rust Book Ch 7 | Organize code |
| 5 | Packages & Crates | Rust Book Ch 7 | Use external crates |
| 6 | Error Handling Deep | Rust Book Ch 9 | Custom error types |
| 7 | Review + Mini Project | - | Build: LosslessLevels module |

### Week 4: Traits & Generics
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | Generics | Rust Book Ch 10 | Generic Agent<T> |
| 2 | Traits | Rust Book Ch 10 | Build: Analyzer trait |
| 3 | Trait Bounds | Rust Book Ch 10 | Constrained generics |
| 4 | Lifetimes Intro | Rust Book Ch 10 | Basic lifetime syntax |
| 5 | Lifetimes Practice | Rust Book Ch 10 | References in structs |
| 6 | Trait Objects | Rust Book Ch 17 | Dynamic dispatch |
| 7 | Review + Mini Project | - | Build: Pluggable analyzers |

---

## Phase 2: Async & Concurrency (Weeks 5-8)

### Week 5: Async Basics
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | Threads | Rust Book Ch 16 | Spawn agent threads |
| 2 | Message Passing | Rust Book Ch 16 | Channel communication |
| 3 | Shared State | Rust Book Ch 16 | Mutex, Arc |
| 4 | Async/Await Intro | Tokio Tutorial | First async function |
| 5 | Tokio Runtime | Tokio Tutorial | Set up runtime |
| 6 | Async Channels | Tokio Tutorial | mpsc, broadcast |
| 7 | Review + Mini Project | - | Build: Multi-agent spawner |

### Week 6: Network & I/O
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | File I/O | Rust Book Ch 12 | Read/write JSON |
| 2 | Serde JSON | serde docs | Serialize candles |
| 3 | HTTP Client | reqwest docs | Fetch market data |
| 4 | WebSockets | tokio-tungstenite | Real-time data |
| 5 | TCP Sockets | Tokio Tutorial | Build simple server |
| 6 | Error Handling Async | anyhow/thiserror | Async error patterns |
| 7 | Review + Mini Project | - | Build: Data fetcher |

### Week 7: Database Integration
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | PostgreSQL Setup | - | Install, create DB |
| 2 | sqlx Basics | sqlx docs | Connect, query |
| 3 | Migrations | sqlx docs | Schema management |
| 4 | CRUD Operations | sqlx docs | Insert/select trades |
| 5 | Connection Pooling | sqlx docs | Production patterns |
| 6 | Redis Basics | redis-rs docs | Pub/sub, caching |
| 7 | Review + Mini Project | - | Build: Trade storage |

### Week 8: Real-World Patterns
| Day | Topic | Resource | Practice |
|-----|-------|----------|----------|
| 1 | Actor Pattern | actix docs | Agent as actor |
| 2 | State Machines | - | Trade state machine |
| 3 | Retry Logic | - | Resilient connections |
| 4 | Graceful Shutdown | Tokio Tutorial | Clean termination |
| 5 | Configuration | config-rs docs | Config management |
| 6 | Logging | tracing docs | Structured logging |
| 7 | Review + Mini Project | - | Build: Resilient agent |

---

## Phase 3: Core Trading System (Weeks 9-16)

### Week 9-10: Lossless Algorithms
- Implement LosslessLevels (support/resistance)
- Implement TrendObserver (higher highs/lower lows)
- Implement MomentumObserver (volume analysis)
- Implement BounceDetector
- Full test coverage

### Week 11-12: Agent Architecture
- Agent trait definition
- Single-symbol agent implementation
- Agent pool manager
- Signal aggregation
- Conviction scoring

### Week 13-14: Coordinator & Risk
- Central coordinator
- Capital allocation
- Risk guardian (position sizing, limits)
- Portfolio management
- Emergency shutdown

### Week 15-16: Broker Integration
- Abstract broker trait
- MT5 bridge (via FFI or TCP)
- Order execution
- Position tracking
- Error recovery

---

## Phase 4: Infrastructure (Weeks 17-20)

### Week 17: Monitoring & Alerts
- Telegram bot integration
- Metrics collection (prometheus)
- Grafana dashboards
- Alert rules

### Week 18: Deployment
- Linux server setup
- systemd service files
- Automated deployment
- Backup scripts

### Week 19: Testing & Validation
- Unit tests
- Integration tests
- Paper trading mode
- Backtesting framework

### Week 20: Documentation & Polish
- API documentation
- Runbooks
- Recovery procedures
- Performance tuning

---

## Phase 5: Live Trading (Weeks 21-24)

### Week 21-22: Paper Trading
- Run full system on paper
- Monitor performance
- Fix bugs
- Tune parameters (only risk params, not strategy)

### Week 23-24: Micro Live
- $1K test account
- Monitor closely
- Validate execution
- Scale up gradually

---

## Key Resources

### Official Documentation
- The Rust Book: https://doc.rust-lang.org/book/
- Rust by Example: https://doc.rust-lang.org/rust-by-example/
- Tokio Tutorial: https://tokio.rs/tokio/tutorial

### Crates We'll Use
```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres"] }
redis = "0.24"
reqwest = { version = "0.11", features = ["json"] }
tokio-tungstenite = "0.21"
tracing = "0.1"
tracing-subscriber = "0.3"
config = "0.14"
anyhow = "1"
thiserror = "1"
chrono = { version = "0.4", features = ["serde"] }
rust_decimal = "1"
```

### Trading-Specific Resources
- QuantLib (C++ library, study architecture)
- LMAX Exchange (open source trading examples)
- Jane Street Tech Blog

---

## Daily Practice Routine

```
Morning (1-2 hours):
├── Read Rust Book chapter
├── Take notes
└── Understand concepts

Afternoon (2-3 hours):
├── Complete daily exercise
├── Build mini-project piece
└── Debug and experiment

Evening (30 min):
├── Review what you learned
├── Plan tomorrow
└── Update progress log
```

---

## Progress Tracking

Create a file: `PROGRESS.md`

```markdown
# Sovereign v4 Progress

## Week 1
- [ ] Day 1: Variables ___/___
- [ ] Day 2: Functions ___/___
- [ ] Day 3: Control Flow ___/___
- [ ] Day 4: Ownership ___/___
- [ ] Day 5: Borrowing ___/___
- [ ] Day 6: Slices ___/___
- [ ] Day 7: Mini Project ___/___

Notes:
- 
- 

## Week 2
...
```

---

## Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 4 | Rust Fundamentals | Can write basic Rust programs |
| 8 | Async Proficient | Can build concurrent systems |
| 12 | Core Algorithms | Lossless levels working |
| 16 | Full System | Complete trading engine |
| 20 | Deployed | Running on Linux server |
| 24 | Live Trading | Real money, autonomous |

---

## Remember

> "pftq built Tech Trader at age 21. You're building something similar. 
> The difference: You have a roadmap. Take it one day at a time."

The goal isn't to rush. The goal is to build something that runs for 12+ years.

Quality > Speed.
