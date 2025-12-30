//! Backtest Runner

use anyhow::Result;
use rust_decimal_macros::dec;
use std::env;

use sovereign::backtest::Backtester;
use sovereign::core::strategy::SignalDirection;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: backtest <csv_file>");
        println!("CSV format: timestamp,open,high,low,close,volume");
        println!("");
        println!("Example: backtest data/XAUUSD_M5_2024.csv");
        return Ok(());
    }

    let csv_path = &args[1];
    
    println!("Loading candles from {}...", csv_path);
    
    let candles = Backtester::load_csv(csv_path)?;
    println!("Loaded {} candles", candles.len());
    
    if candles.is_empty() {
        println!("No candles loaded. Check CSV format.");
        return Ok(());
    }

    println!("First candle: {:?}", candles.first());
    println!("Last candle: {:?}", candles.last());
    println!("");
    println!("Running backtest...");
    
    let mut backtester = Backtester::new(dec!(10000));
    let result = backtester.run(candles);
    
    result.print_summary();
    
    if !result.trades.is_empty() {
        println!("");
        println!("Last 10 trades:");
        for trade in result.trades.iter().rev().take(10) {
            let dir = match trade.direction {
                SignalDirection::Buy => "BUY ",
                SignalDirection::Sell => "SELL",
                _ => "HOLD",
            };
            let profit = trade.profit.unwrap_or(dec!(0));
            let symbol = if profit > dec!(0) { "+" } else { "" };
            println!("  {} @ {} -> {} | P&L: {}${:.2} | Conv: {}%",
                dir,
                trade.entry_price,
                trade.exit_price.unwrap_or(dec!(0)),
                symbol,
                profit,
                trade.conviction
            );
        }
    }

    Ok(())
}
