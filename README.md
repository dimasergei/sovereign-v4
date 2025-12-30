# Sovereign v4.0 "Perpetual"

**Institutional-Grade Autonomous Trading System**

Built in Rust for 12+ year runtime with zero human intervention.

---

## Philosophy

Inspired by [Tech Trader](https://techtrader.ai/) by pftq:

> "The reason I call it a 'lossless' algorithm is because it doesn't estimate anything or use any seeded values/thresholds. It is analogous to lossless audio/image file formats."

### Core Principles

1. **Lossless Algorithms**: No parameters, no thresholds, no magic numbers
2. **Human Thinking**: Pattern recognition, not statistics  
3. **Perpetual Operation**: Self-healing, never crashes
4. **Scale**: Designed for 1000+ concurrent agents

---

## Quick Start

### Prerequisites

1. Install Rust: https://rustup.rs/
2. Install VS Code + rust-analyzer extension

### Run the First Exercise

```bash
cd sovereign_v4_rust
cargo run --bin day01
```

### Run All Tests

```bash
cargo test
```

### Build for Production

```bash
cargo build --release
```

---

## Project Structure

```
sovereign_v4_rust/
├── src/
│   ├── main.rs              # Entry point
│   ├── core/                # Trading logic
│   │   ├── mod.rs
│   │   ├── types.rs         # Core data structures
│   │   ├── lossless.rs      # Lossless algorithms
│   │   ├── agent.rs         # Trading agents
│   │   ├── coordinator.rs   # Central brain
│   │   └── guardian.rs      # Risk management
│   ├── broker/              # Broker connections
│   │   ├── mod.rs
│   │   └── mt5.rs           # MT5 bridge
│   ├── data/                # Persistence
│   │   ├── mod.rs
│   │   ├── postgres.rs      # PostgreSQL
│   │   └── redis.rs         # Redis
│   ├── comms/               # Communications
│   │   ├── mod.rs
│   │   ├── telegram.rs      # Notifications
│   │   └── monitor.rs       # System monitoring
│   └── bin/
│       └── watchdog.rs      # External guardian
├── exercises/               # Learning exercises
│   ├── day01_variables.rs
│   ├── day02_functions.rs
│   ├── day04_ownership.rs
│   └── week1_project.rs
├── config/                  # Configuration files
├── migrations/              # Database migrations
├── Cargo.toml               # Dependencies
└── ROADMAP.md              # Development roadmap
```

---

## Learning Path

### Week 1: Rust Fundamentals
- Day 1: Variables and Types → `cargo run --bin day01`
- Day 2: Functions → `cargo run --bin day02`
- Day 3: Control Flow
- Day 4: Ownership → `cargo run --bin day04`
- Day 5: References & Borrowing
- Day 6: Slices
- Day 7: Mini Project → `cargo run --bin week1_project`

### Week 2-4: Data Structures & Modules
See `ROADMAP.md` for full curriculum.

### Week 5-8: Async & Concurrency
See `ROADMAP.md` for full curriculum.

---

## Key Components

### Lossless Levels (`src/core/lossless.rs`)

Support/resistance detection with no parameters:

```rust
// Every price range crossed loses a point
for each_price_move {
    price_range_traveled.score -= 1;
}
// Score of 0 = never crossed = strongest level
```

### Risk Guardian (`src/core/guardian.rs`)

Account protection (the ONLY place with hard limits):

- Max risk per trade: 0.5%
- Max daily loss: 2%
- Max floating loss: 1.5%
- Max positions: 1

### Agent Architecture

Each agent:
- Focuses on ONE symbol
- Makes independent decisions
- Reports to coordinator
- Can scale to 1000+ agents

---

## Deployment

### Development

```bash
cargo run
```

### Production (Linux)

```bash
# Build
cargo build --release

# Create systemd service
sudo cp sovereign.service /etc/systemd/system/
sudo systemctl enable sovereign
sudo systemctl start sovereign
```

---

## Resources

- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Tokio Tutorial](https://tokio.rs/tokio/tutorial)
- [pftq's Lossless Algorithms](https://www.pftq.com/blabberbox/?page=Lossless_Algorithms)

---

## License

Proprietary - All rights reserved.
