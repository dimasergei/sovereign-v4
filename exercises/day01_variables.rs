//! Day 1: Variables and Types
//!
//! Learning Rust basics through trading examples.
//!
//! Run with: cargo run --bin day01
//!
//! Key concepts:
//! - let vs let mut
//! - Type inference
//! - Basic types: i32, f64, bool, char, String, &str

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Day 1: Variables and Types");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // =========================================================================
    // IMMUTABLE VARIABLES (default in Rust)
    // =========================================================================
    
    // In Rust, variables are immutable by default
    let price = 2650.50;  // f64 inferred
    println!("Gold price: ${}", price);
    
    // This would cause a compile error:
    // price = 2651.00;  // ERROR: cannot assign twice to immutable variable
    
    // =========================================================================
    // MUTABLE VARIABLES
    // =========================================================================
    
    // Use `mut` to make a variable mutable
    let mut balance: f64 = 10000.0;  // Explicit type annotation
    println!("Starting balance: ${}", balance);
    
    balance += 500.0;  // Now we can modify it
    println!("After profit: ${}", balance);
    
    // =========================================================================
    // TYPE ANNOTATIONS
    // =========================================================================
    
    // Rust can usually infer types, but you can be explicit
    let account_number: i32 = 123456;
    let lot_size: f64 = 0.01;
    let is_live: bool = false;
    let currency: char = '$';
    let broker: &str = "GFT";  // String slice (borrowed)
    let symbol: String = String::from("XAUUSD");  // Owned String
    
    println!("\nAccount Info:");
    println!("  Number: {}", account_number);
    println!("  Lot Size: {}", lot_size);
    println!("  Live Mode: {}", is_live);
    println!("  Currency: {}", currency);
    println!("  Broker: {}", broker);
    println!("  Symbol: {}", symbol);
    
    // =========================================================================
    // SHADOWING (re-declaring with same name)
    // =========================================================================
    
    // You can "shadow" a variable by declaring it again
    let spread = 50;  // i32
    println!("\nSpread (integer): {} points", spread);
    
    let spread = spread as f64 / 100.0;  // Now f64
    println!("Spread (decimal): ${}", spread);
    
    // =========================================================================
    // CONSTANTS
    // =========================================================================
    
    // Constants are ALWAYS immutable and must have type annotations
    const MAX_DAILY_LOSS_PCT: f64 = 0.02;  // 2%
    const MAX_POSITIONS: i32 = 1;
    
    println!("\nRisk Limits:");
    println!("  Max Daily Loss: {}%", MAX_DAILY_LOSS_PCT * 100.0);
    println!("  Max Positions: {}", MAX_POSITIONS);
    
    // =========================================================================
    // EXERCISE: Your turn!
    // =========================================================================
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  EXERCISE: Complete the code below");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // TODO 1: Declare an immutable variable for the entry price (use 2648.50)
    // let entry_price = ???;
    
    // TODO 2: Declare a mutable variable for position PnL starting at 0.0
    // let mut pnl = ???;
    
    // TODO 3: Calculate PnL if current price is 2652.50 and we have 0.1 lots
    // (Formula: (current - entry) * lots * 100)
    // let current_price = ???;
    // let lots = ???;
    // pnl = ???;
    
    // TODO 4: Print the results
    // println!("Entry: ${}", entry_price);
    // println!("Current: ${}", current_price);
    // println!("PnL: ${}", pnl);
    
    // Uncomment and complete the code above!
    
    println!("\n✅ Day 1 complete! Run 'cargo run --bin day02' for the next lesson.");
}
