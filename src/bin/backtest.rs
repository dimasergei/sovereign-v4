//! Backtest Runner
//!
//! Test the lossless trading strategy on historical data.
//!
//! Usage:
//!   cargo run --bin backtest data/USO_daily.csv
//!   cargo run --bin backtest data/GLD_daily.csv USO

use anyhow::Result;
use rust_decimal_macros::dec;
use std::env;

use sovereign::backtest::{Backtester, load_csv};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("================================================================");
        println!("              SOVEREIGN LOSSLESS BACKTESTER                    ");
        println!("================================================================");
        println!();
        println!("Usage: backtest <csv_file> [symbol]");
        println!();
        println!("Arguments:");
        println!("  csv_file  Path to historical data (required)");
        println!("  symbol    Trading symbol name (default: derived from filename)");
        println!();
        println!("CSV Format: timestamp,open,high,low,close,volume");
        println!("  - timestamp: Unix epoch seconds");
        println!("  - First row is header (skipped)");
        println!();
        println!("Examples:");
        println!("  backtest data/USO_daily.csv");
        println!("  backtest data/GLD_daily.csv GLD");
        println!();
        println!("Recommended test symbols (commodity ETFs):");
        println!("  - USO (Oil)   - Should show mean reversion");
        println!("  - GLD (Gold)  - Should show range trading");
        println!("  - SLV (Silver) - Should show volatility trades");
        println!("================================================================");
        return Ok(());
    }

    let csv_path = &args[1];

    // Derive symbol from filename or use provided
    let symbol = if args.len() > 2 {
        args[2].clone()
    } else {
        std::path::Path::new(csv_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("UNKNOWN")
            .split('_')
            .next()
            .unwrap_or("UNKNOWN")
            .to_string()
    };

    println!("================================================================");
    println!("              SOVEREIGN LOSSLESS BACKTESTER                    ");
    println!("================================================================");
    println!();
    println!("Loading: {}", csv_path);
    println!("Symbol:  {}", symbol);
    println!();

    let bars = load_csv(csv_path)?;
    println!("Loaded {} bars", bars.len());

    if bars.is_empty() {
        println!("No bars loaded. Check CSV format.");
        return Ok(());
    }

    if let (Some(first), Some(last)) = (bars.first(), bars.last()) {
        println!("Period:  {} to {}",
            first.time.format("%Y-%m-%d"),
            last.time.format("%Y-%m-%d"));
        println!("Price:   ${:.2} to ${:.2}",
            first.close, last.close);
    }

    println!();
    println!("Running backtest...");
    println!();

    let backtester = Backtester::new(&symbol, dec!(100000));
    let result = backtester.run(&bars);

    result.print_summary();

    if !result.trades.is_empty() {
        result.print_trades(10);
    } else {
        println!();
        println!("No trades generated.");
        println!();
        println!("This could mean:");
        println!("  - Not enough data (need 20+ bars for S/R)");
        println!("  - No volume capitulation events");
        println!("  - Price never reached S/R levels");
        println!();
    }

    Ok(())
}
