//! Day 2: Functions
//!
//! Learning functions through trading examples.
//!
//! Run with: cargo run --bin day02
//!
//! Key concepts:
//! - Function syntax
//! - Parameters and return types
//! - Expressions vs statements
//! - Early returns

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Day 2: Functions");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // =========================================================================
    // BASIC FUNCTION
    // =========================================================================
    
    // Call a simple function
    print_welcome();
    
    // =========================================================================
    // FUNCTION WITH PARAMETERS
    // =========================================================================
    
    // Calculate position size
    let balance = 10000.0;
    let risk_pct = 0.005;  // 0.5%
    let sl_distance = 10.0;  // 10 points
    
    let position_size = calculate_position_size(balance, risk_pct, sl_distance);
    println!("Position size: {} lots", position_size);
    
    // =========================================================================
    // FUNCTION WITH RETURN VALUE (explicit return)
    // =========================================================================
    
    let pnl = calculate_pnl(2650.0, 2660.0, 0.1, true);
    println!("PnL: ${:.2}", pnl);
    
    // =========================================================================
    // FUNCTION WITH EXPRESSION RETURN (no semicolon)
    // =========================================================================
    
    let risk_amount = get_risk_amount(10000.0, 0.005);
    println!("Risk amount: ${:.2}", risk_amount);
    
    // =========================================================================
    // MULTIPLE PARAMETERS AND LOGIC
    // =========================================================================
    
    let can_trade = check_can_trade(10000.0, 10050.0, 0, 2);
    println!("Can trade: {}", can_trade);
    
    // =========================================================================
    // EXERCISE: Your turn!
    // =========================================================================
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  EXERCISE: Create the functions below");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // TODO: Uncomment and implement these functions at the bottom of this file:
    
    // let spread = calculate_spread(2650.00, 2650.50);
    // println!("Spread: {} points", spread);
    
    // let rr_ratio = calculate_risk_reward(2650.0, 2640.0, 2670.0);
    // println!("Risk/Reward: {:.2}", rr_ratio);
    
    // let is_valid = is_valid_trade(rr_ratio, spread);
    // println!("Valid trade: {}", is_valid);
    
    println!("\n✅ Day 2 complete! Run 'cargo run --bin day03' for the next lesson.");
}

// =========================================================================
// FUNCTION DEFINITIONS
// =========================================================================

/// Simple function with no parameters or return value
fn print_welcome() {
    println!("Welcome to Sovereign v4.0!");
    println!("Learning Rust through trading.\n");
}

/// Function with parameters and explicit return type
fn calculate_position_size(balance: f64, risk_pct: f64, sl_distance: f64) -> f64 {
    let risk_amount = balance * risk_pct;
    let position_size = risk_amount / sl_distance;
    
    // Round to 2 decimal places
    (position_size * 100.0).round() / 100.0
}

/// Function with explicit return statement
fn calculate_pnl(entry: f64, current: f64, lots: f64, is_long: bool) -> f64 {
    let price_diff = current - entry;
    
    if is_long {
        return price_diff * lots * 100.0;  // Explicit return
    } else {
        return -price_diff * lots * 100.0;
    }
}

/// Function with expression return (no semicolon = return value)
fn get_risk_amount(balance: f64, risk_pct: f64) -> f64 {
    balance * risk_pct  // No semicolon = this is the return value
}

/// Function with boolean return
fn check_can_trade(balance: f64, equity: f64, open_positions: i32, max_positions: i32) -> bool {
    // Check floating PnL
    let floating_pnl_pct = (equity - balance) / balance;
    
    if floating_pnl_pct < -0.02 {
        return false;  // Too much floating loss
    }
    
    if open_positions >= max_positions {
        return false;  // Too many positions
    }
    
    true  // All checks passed
}

// =========================================================================
// EXERCISE: Implement these functions
// =========================================================================

// TODO: Calculate spread in points (ask - bid) * 100
// fn calculate_spread(bid: f64, ask: f64) -> f64 {
//     ???
// }

// TODO: Calculate risk/reward ratio
// Risk = entry - stop_loss, Reward = take_profit - entry
// fn calculate_risk_reward(entry: f64, stop_loss: f64, take_profit: f64) -> f64 {
//     ???
// }

// TODO: Check if trade is valid (R:R >= 1.5 AND spread < 100)
// fn is_valid_trade(rr_ratio: f64, spread: f64) -> bool {
//     ???
// }
