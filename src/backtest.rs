//! Backtesting Engine
//!
//! Tests the lossless trading strategy on historical data.
//!
//! Usage:
//! ```text
//! cargo run --bin backtest data/USO_daily.csv
//! ```

use rust_decimal::Decimal;
use rust_decimal::prelude::{ToPrimitive, FromPrimitive};
use rust_decimal_macros::dec;
use std::fs::File;
use std::io::{BufRead, BufReader};
use chrono::{DateTime, Utc, TimeZone};

use crate::core::agent::{SymbolAgent, Signal, Side};
use crate::core::sr::default_granularity;
use crate::portfolio::{Portfolio, PortfolioPosition, POSITION_SIZE_PCT};

/// A completed trade in the backtest
#[derive(Debug, Clone)]
pub struct BacktestTrade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub side: Side,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub quantity: Decimal,
    pub profit: Decimal,
    pub profit_pct: Decimal,
    pub reason: String,
}

/// Results of a backtest run
#[derive(Debug)]
pub struct BacktestResult {
    pub symbol: String,
    pub total_bars: usize,
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub total_profit: Decimal,
    pub total_return_pct: Decimal,
    pub max_drawdown_pct: Decimal,
    pub win_rate: f64,
    pub profit_factor: Decimal,
    pub avg_win: Decimal,
    pub avg_loss: Decimal,
    pub trades: Vec<BacktestTrade>,
}

/// Historical bar data
#[derive(Debug, Clone)]
pub struct Bar {
    pub time: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: u64,
}

/// Backtester that uses the SymbolAgent architecture
pub struct Backtester {
    symbol: String,
    starting_balance: Decimal,
}

impl Backtester {
    /// Create a new backtester for a symbol
    pub fn new(symbol: &str, starting_balance: Decimal) -> Self {
        Self {
            symbol: symbol.to_string(),
            starting_balance,
        }
    }

    /// Run backtest on historical bars
    pub fn run(&self, bars: &[Bar]) -> BacktestResult {
        let mut trades: Vec<BacktestTrade> = Vec::new();
        let mut balance = self.starting_balance;
        let mut peak_balance = balance;
        let mut max_drawdown_pct = Decimal::ZERO;
        let mut gross_profit = Decimal::ZERO;
        let mut gross_loss = Decimal::ZERO;

        // Determine granularity from first bar
        let initial_price = bars.first().map(|b| b.close).unwrap_or(dec!(100));
        let granularity = default_granularity(&self.symbol, initial_price);

        // Create agent
        let mut agent = SymbolAgent::with_granularity(self.symbol.clone(), granularity);

        // Track open position
        let mut open_position: Option<(DateTime<Utc>, Side, Decimal, Decimal)> = None; // (time, side, price, qty)

        for bar in bars {
            // Process bar through agent
            let signal = agent.process_bar(
                bar.time,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            );

            // Calculate position size
            let position_value = balance * Decimal::from_f64(POSITION_SIZE_PCT).unwrap_or(dec!(0.07));
            let qty = if bar.close > Decimal::ZERO {
                (position_value / bar.close).round_dp(0)
            } else {
                Decimal::ZERO
            };

            // Check for signals
            if let Some(sig) = signal {
                match sig.signal {
                    Signal::Buy => {
                        if open_position.is_none() {
                            open_position = Some((bar.time, Side::Long, bar.close, qty));
                        }
                    }
                    Signal::Sell => {
                        if let Some((entry_time, Side::Long, entry_price, pos_qty)) = open_position {
                            let profit = (bar.close - entry_price) * pos_qty;
                            let profit_pct = (bar.close - entry_price) / entry_price * dec!(100);

                            balance += profit;
                            if profit > Decimal::ZERO {
                                gross_profit += profit;
                            } else {
                                gross_loss += profit.abs();
                            }

                            trades.push(BacktestTrade {
                                entry_time,
                                exit_time: bar.time,
                                side: Side::Long,
                                entry_price,
                                exit_price: bar.close,
                                quantity: pos_qty,
                                profit,
                                profit_pct,
                                reason: sig.reason.clone(),
                            });

                            open_position = None;
                        }
                    }
                    Signal::Short => {
                        if open_position.is_none() {
                            open_position = Some((bar.time, Side::Short, bar.close, qty));
                        }
                    }
                    Signal::Cover => {
                        if let Some((entry_time, Side::Short, entry_price, pos_qty)) = open_position {
                            let profit = (entry_price - bar.close) * pos_qty;
                            let profit_pct = (entry_price - bar.close) / entry_price * dec!(100);

                            balance += profit;
                            if profit > Decimal::ZERO {
                                gross_profit += profit;
                            } else {
                                gross_loss += profit.abs();
                            }

                            trades.push(BacktestTrade {
                                entry_time,
                                exit_time: bar.time,
                                side: Side::Short,
                                entry_price,
                                exit_price: bar.close,
                                quantity: pos_qty,
                                profit,
                                profit_pct,
                                reason: sig.reason.clone(),
                            });

                            open_position = None;
                        }
                    }
                    Signal::Hold => {}
                }
            }

            // Track drawdown
            if balance > peak_balance {
                peak_balance = balance;
            }
            if peak_balance > Decimal::ZERO {
                let drawdown = (peak_balance - balance) / peak_balance * dec!(100);
                if drawdown > max_drawdown_pct {
                    max_drawdown_pct = drawdown;
                }
            }
        }

        // Calculate statistics
        let wins = trades.iter().filter(|t| t.profit > Decimal::ZERO).count();
        let losses = trades.iter().filter(|t| t.profit < Decimal::ZERO).count();
        let total_profit = balance - self.starting_balance;
        let total_return_pct = if self.starting_balance > Decimal::ZERO {
            total_profit / self.starting_balance * dec!(100)
        } else {
            Decimal::ZERO
        };
        let win_rate = if !trades.is_empty() {
            wins as f64 / trades.len() as f64 * 100.0
        } else {
            0.0
        };
        let profit_factor = if gross_loss > Decimal::ZERO {
            gross_profit / gross_loss
        } else if gross_profit > Decimal::ZERO {
            dec!(999.99) // Infinite (no losses)
        } else {
            Decimal::ZERO
        };

        let avg_win = if wins > 0 {
            trades.iter()
                .filter(|t| t.profit > Decimal::ZERO)
                .map(|t| t.profit)
                .sum::<Decimal>() / Decimal::from(wins)
        } else {
            Decimal::ZERO
        };

        let avg_loss = if losses > 0 {
            trades.iter()
                .filter(|t| t.profit < Decimal::ZERO)
                .map(|t| t.profit.abs())
                .sum::<Decimal>() / Decimal::from(losses)
        } else {
            Decimal::ZERO
        };

        BacktestResult {
            symbol: self.symbol.clone(),
            total_bars: bars.len(),
            total_trades: trades.len(),
            wins,
            losses,
            total_profit,
            total_return_pct,
            max_drawdown_pct,
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            trades,
        }
    }
}

/// Load bars from a CSV file
///
/// Expected format: timestamp,open,high,low,close,volume
/// First line is header (skipped)
pub fn load_csv(path: &str) -> Result<Vec<Bar>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut bars = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }

        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        if parts.len() >= 6 {
            let timestamp: i64 = parts[0].parse().unwrap_or(0);
            let time = Utc.timestamp_opt(timestamp, 0).unwrap();

            let open = parts[1].parse::<Decimal>().unwrap_or(Decimal::ZERO);
            let high = parts[2].parse::<Decimal>().unwrap_or(Decimal::ZERO);
            let low = parts[3].parse::<Decimal>().unwrap_or(Decimal::ZERO);
            let close = parts[4].parse::<Decimal>().unwrap_or(Decimal::ZERO);
            let volume: u64 = parts[5].parse().unwrap_or(0);

            bars.push(Bar {
                time,
                open,
                high,
                low,
                close,
                volume,
            });
        }
    }

    Ok(bars)
}

impl BacktestResult {
    /// Print a summary of the results
    pub fn print_summary(&self) {
        println!("================================================================");
        println!("                    BACKTEST RESULTS                           ");
        println!("================================================================");
        println!(" Symbol:          {}", self.symbol);
        println!(" Bars Processed:  {}", self.total_bars);
        println!("----------------------------------------------------------------");
        println!(" Total Trades:    {:>6}", self.total_trades);
        println!(" Wins:            {:>6}", self.wins);
        println!(" Losses:          {:>6}", self.losses);
        println!(" Win Rate:        {:>6.1}%", self.win_rate);
        println!("----------------------------------------------------------------");
        println!(" Total Profit:   ${:>10.2}", self.total_profit);
        println!(" Total Return:    {:>6.2}%", self.total_return_pct);
        println!(" Max Drawdown:    {:>6.2}%", self.max_drawdown_pct);
        println!(" Profit Factor:   {:>6.2}", self.profit_factor);
        println!("----------------------------------------------------------------");
        println!(" Avg Win:        ${:>10.2}", self.avg_win);
        println!(" Avg Loss:       ${:>10.2}", self.avg_loss);
        println!("================================================================");
    }

    /// Print the last N trades
    pub fn print_trades(&self, count: usize) {
        println!("\nLast {} Trades:", count);
        println!("----------------------------------------------------------------");
        for trade in self.trades.iter().rev().take(count).rev() {
            let dir = match trade.side {
                Side::Long => "LONG ",
                Side::Short => "SHORT",
            };
            println!(
                " {} {} @ {:.2} -> {:.2} | P&L: ${:.2} ({:+.2}%)",
                dir,
                trade.entry_time.format("%Y-%m-%d"),
                trade.entry_price,
                trade.exit_price,
                trade.profit,
                trade.profit_pct
            );
        }
        println!("----------------------------------------------------------------");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bars() -> Vec<Bar> {
        let mut bars = Vec::new();
        let base_time = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

        // Create oscillating price pattern to trigger S/R
        for i in 0..100 {
            let day = base_time + chrono::Duration::days(i);
            let base_price = dec!(100);

            // Oscillate between 95 and 105
            let price = if i % 10 < 5 {
                base_price + Decimal::from(i % 5)
            } else {
                base_price + dec!(5) - Decimal::from(i % 5)
            };

            bars.push(Bar {
                time: day,
                open: price - dec!(0.5),
                high: price + dec!(1),
                low: price - dec!(1),
                close: price,
                volume: if i % 20 == 19 { 5000 } else { 1000 }, // Volume spike every 20 bars
            });
        }

        bars
    }

    #[test]
    fn test_backtester_creation() {
        let bt = Backtester::new("USO", dec!(100000));
        assert_eq!(bt.symbol, "USO");
        assert_eq!(bt.starting_balance, dec!(100000));
    }

    #[test]
    fn test_backtest_run() {
        let bars = create_test_bars();
        let bt = Backtester::new("TEST", dec!(100000));
        let result = bt.run(&bars);

        assert_eq!(result.symbol, "TEST");
        assert_eq!(result.total_bars, 100);
    }

    #[test]
    fn test_profit_calculation() {
        // Create a simple uptrend for testing
        let mut bars = Vec::new();
        let base_time = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

        // Build S/R levels first (20 bars)
        for i in 0..25 {
            bars.push(Bar {
                time: base_time + chrono::Duration::days(i),
                open: dec!(100),
                high: dec!(102),
                low: dec!(99),
                close: dec!(100),
                volume: 1000,
            });
        }

        // Add capitulation bar at support
        bars.push(Bar {
            time: base_time + chrono::Duration::days(25),
            open: dec!(101),
            high: dec!(101),
            low: dec!(99),
            close: dec!(100), // Down day, high volume
            volume: 5000,
        });

        // Add upward movement to resistance
        for i in 26..35 {
            bars.push(Bar {
                time: base_time + chrono::Duration::days(i),
                open: dec!(100) + Decimal::from(i - 26),
                high: dec!(102) + Decimal::from(i - 26),
                low: dec!(99) + Decimal::from(i - 26),
                close: dec!(101) + Decimal::from(i - 26),
                volume: 1000,
            });
        }

        let bt = Backtester::new("TEST", dec!(100000));
        let result = bt.run(&bars);

        // Should have processed all bars
        assert_eq!(result.total_bars, bars.len());
    }
}
