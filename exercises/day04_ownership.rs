//! Day 4: Ownership
//!
//! THE most important concept in Rust. This is what makes Rust unique.
//!
//! Run with: cargo run --bin day04
//!
//! Key concepts:
//! - Every value has an owner
//! - Only one owner at a time
//! - Value is dropped when owner goes out of scope
//! - Move semantics

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Day 4: Ownership - The Heart of Rust");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // =========================================================================
    // STACK vs HEAP
    // =========================================================================
    
    // Stack: Fixed size, fast, copied automatically
    let price1 = 2650.50;  // f64 lives on stack
    let price2 = price1;    // Copied! Both are valid
    println!("Price 1: ${}", price1);
    println!("Price 2: ${}", price2);
    
    // Heap: Dynamic size, slower, MOVED (not copied)
    let symbol1 = String::from("XAUUSD");  // String lives on heap
    let symbol2 = symbol1;                  // MOVED! symbol1 is now invalid
    // println!("{}", symbol1);  // ERROR: value borrowed after move
    println!("Symbol 2: {}", symbol2);
    
    // =========================================================================
    // OWNERSHIP RULES
    // =========================================================================
    
    println!("\n--- Ownership Rules ---\n");
    
    // Rule 1: Each value has one owner
    let trade = create_trade();
    println!("Trade created: {}", trade);
    
    // Rule 2: When owner goes out of scope, value is dropped
    {
        let temp_data = String::from("temporary");
        println!("Inside scope: {}", temp_data);
    } // temp_data is dropped here
    // println!("{}", temp_data);  // ERROR: not found in this scope
    
    // Rule 3: Ownership can be transferred (moved)
    let trade2 = trade;  // trade is moved to trade2
    // println!("{}", trade);  // ERROR: value borrowed after move
    println!("Trade moved to: {}", trade2);
    
    // =========================================================================
    // CLONE (explicit deep copy)
    // =========================================================================
    
    println!("\n--- Cloning ---\n");
    
    let account1 = String::from("GFT-12345");
    let account2 = account1.clone();  // Deep copy, both are valid
    println!("Account 1: {}", account1);
    println!("Account 2: {}", account2);
    
    // =========================================================================
    // FUNCTIONS AND OWNERSHIP
    // =========================================================================
    
    println!("\n--- Functions and Ownership ---\n");
    
    let my_trade = String::from("BUY XAUUSD 0.1");
    
    // This function takes ownership
    print_trade(my_trade);
    // println!("{}", my_trade);  // ERROR: value moved into function
    
    // To keep using it, we could:
    // 1. Clone before passing
    // 2. Have function return ownership back
    // 3. Use references (tomorrow's lesson!)
    
    let another_trade = String::from("SELL EURUSD 0.2");
    let returned_trade = process_and_return(another_trade);
    println!("Got back: {}", returned_trade);
    
    // =========================================================================
    // WHY THIS MATTERS FOR TRADING SYSTEMS
    // =========================================================================
    
    println!("\n--- Why Ownership Matters ---\n");
    
    /*
    In a trading system, ownership prevents bugs like:
    
    1. Double-free: Trying to close the same position twice
       - Rust: Once position is "consumed", it can't be used again
    
    2. Use-after-free: Using data that's been deallocated
       - Rust: Compiler catches this at compile time
    
    3. Data races: Multiple threads modifying the same data
       - Rust: Ownership rules prevent this (more in async lessons)
    
    4. Memory leaks: Forgetting to free memory
       - Rust: Automatic drop when owner goes out of scope
    
    These bugs can cost REAL MONEY in trading systems!
    Rust catches them at COMPILE TIME, not runtime.
    */
    
    println!("Ownership prevents:");
    println!("  - Double-closing positions");
    println!("  - Using stale market data");
    println!("  - Data races in multi-threaded agents");
    println!("  - Memory leaks in long-running systems");
    
    // =========================================================================
    // EXERCISE
    // =========================================================================
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  EXERCISE: Fix the ownership errors");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // TODO: Uncomment and fix the code below
    
    /*
    let position = String::from("LONG XAUUSD 0.1 @ 2650");
    
    // This takes ownership
    log_position(position);
    
    // ERROR: How do we use position again?
    // Option 1: Clone before passing
    // Option 2: Use references (day 5)
    
    println!("Position is: {}", position);
    */
    
    println!("\n✅ Day 4 complete! This is the foundation of Rust's safety.");
    println!("Tomorrow: References & Borrowing (how to share without moving)");
}

fn create_trade() -> String {
    String::from("New Trade")
}

fn print_trade(trade: String) {
    println!("Printing trade: {}", trade);
    // trade is dropped here when function ends
}

fn process_and_return(trade: String) -> String {
    println!("Processing: {}", trade);
    trade  // Return ownership to caller
}

fn log_position(pos: String) {
    println!("Logged: {}", pos);
}
