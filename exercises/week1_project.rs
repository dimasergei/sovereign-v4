//! Week 1 Mini-Project: Price Tracker
//!
//! Combine everything from Week 1 into a simple price tracker.
//!
//! Run with: cargo run --bin week1_project
//!
//! This project uses:
//! - Variables and types (Day 1)
//! - Functions (Day 2)
//! - Control flow (Day 3)
//! - Ownership (Day 4)
//! - References & Borrowing (Day 5)
//! - Slices (Day 6)

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Week 1 Project: Simple Price Tracker");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Create a price tracker for Gold
    let mut tracker = PriceTracker::new("XAUUSD", 2650.00);
    
    // Simulate price updates
    let prices = [2651.50, 2649.00, 2652.75, 2648.25, 2655.00, 2653.50];
    
    println!("Simulating price updates...\n");
    
    for price in prices {
        tracker.update(price);
        tracker.print_status();
    }
    
    println!("\n--- Summary ---");
    println!("Symbol: {}", tracker.symbol);
    println!("Current: ${:.2}", tracker.current_price);
    println!("High: ${:.2}", tracker.high);
    println!("Low: ${:.2}", tracker.low);
    println!("Change: {:.2}%", tracker.change_percent());
    println!("Trend: {}", tracker.trend());
    
    // =========================================================================
    // EXERCISE: Extend the tracker
    // =========================================================================
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  EXERCISE: Add these features to PriceTracker");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("TODO:");
    println!("1. Add a method `is_near_high(&self, threshold: f64) -> bool`");
    println!("   Returns true if current price is within threshold of high");
    println!("");
    println!("2. Add a method `is_near_low(&self, threshold: f64) -> bool`");
    println!("   Returns true if current price is within threshold of low");
    println!("");
    println!("3. Add a method `volatility(&self) -> f64`");
    println!("   Returns (high - low) / open * 100 (percentage)");
    println!("");
    println!("4. Add `update_count: u32` field to track number of updates");
    
    println!("\n✅ Week 1 complete! You now know Rust fundamentals.");
    println!("Next week: Structs, Enums, and Pattern Matching!");
}

/// Simple price tracker for a trading symbol
struct PriceTracker {
    /// Trading symbol (e.g., "XAUUSD")
    symbol: String,
    /// Opening price
    open_price: f64,
    /// Current price
    current_price: f64,
    /// Highest price seen
    high: f64,
    /// Lowest price seen
    low: f64,
    /// Previous price (for trend detection)
    previous_price: f64,
}

impl PriceTracker {
    /// Create a new price tracker
    fn new(symbol: &str, initial_price: f64) -> Self {
        Self {
            symbol: String::from(symbol),
            open_price: initial_price,
            current_price: initial_price,
            high: initial_price,
            low: initial_price,
            previous_price: initial_price,
        }
    }
    
    /// Update with a new price
    fn update(&mut self, new_price: f64) {
        self.previous_price = self.current_price;
        self.current_price = new_price;
        
        // Update high/low
        if new_price > self.high {
            self.high = new_price;
        }
        if new_price < self.low {
            self.low = new_price;
        }
    }
    
    /// Calculate percentage change from open
    fn change_percent(&self) -> f64 {
        ((self.current_price - self.open_price) / self.open_price) * 100.0
    }
    
    /// Get current trend direction
    fn trend(&self) -> &str {
        if self.current_price > self.previous_price {
            "UP ↑"
        } else if self.current_price < self.previous_price {
            "DOWN ↓"
        } else {
            "FLAT →"
        }
    }
    
    /// Print current status
    fn print_status(&self) {
        let change = self.current_price - self.previous_price;
        let sign = if change >= 0.0 { "+" } else { "" };
        
        println!(
            "{}: ${:.2} ({}{:.2}) {}",
            self.symbol,
            self.current_price,
            sign,
            change,
            self.trend()
        );
    }
}
