//! Backtesting Engine

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::core::types::Candle;
use crate::core::lossless::MarketObserver;
use crate::core::strategy::{Strategy, SignalDirection};
use chrono::{DateTime, Utc, TimeZone};

#[derive(Debug, Clone)]
pub struct BacktestTrade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: Option<DateTime<Utc>>,
    pub direction: SignalDirection,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub stop_loss: Decimal,
    pub take_profit: Decimal,
    pub profit: Option<Decimal>,
    pub conviction: u8,
}

#[derive(Debug)]
pub struct BacktestResult {
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub total_profit: Decimal,
    pub max_drawdown: Decimal,
    pub win_rate: f64,
    pub profit_factor: Decimal,
    pub trades: Vec<BacktestTrade>,
}

pub struct Backtester {
    observer: MarketObserver,
    strategy: Strategy,
    starting_balance: Decimal,
    risk_per_trade: Decimal,
}

impl Backtester {
    pub fn new(starting_balance: Decimal) -> Self {
        Self {
            observer: MarketObserver::new(dec!(0.01), true),
            strategy: Strategy::default(),
            starting_balance,
            risk_per_trade: dec!(0.005),
        }
    }

    pub fn run(&mut self, candles: Vec<Candle>) -> BacktestResult {
        let mut trades: Vec<BacktestTrade> = Vec::new();
        let mut open_trade: Option<BacktestTrade> = None;
        let mut balance = self.starting_balance;
        let mut peak_balance = balance;
        let mut max_drawdown = Decimal::ZERO;
        let mut gross_profit = Decimal::ZERO;
        let mut gross_loss = Decimal::ZERO;

        for candle in &candles {
            self.observer.update(candle);
            let obs = self.observer.observe(candle.close);

            if let Some(ref mut trade) = open_trade {
                let hit_sl = match trade.direction {
                    SignalDirection::Buy => candle.low <= trade.stop_loss,
                    SignalDirection::Sell => candle.high >= trade.stop_loss,
                    SignalDirection::Hold => false,
                };

                let hit_tp = match trade.direction {
                    SignalDirection::Buy => candle.high >= trade.take_profit,
                    SignalDirection::Sell => candle.low <= trade.take_profit,
                    SignalDirection::Hold => false,
                };

                if hit_sl {
                    let exit_price = trade.stop_loss;
                    let profit = match trade.direction {
                        SignalDirection::Buy => exit_price - trade.entry_price,
                        SignalDirection::Sell => trade.entry_price - exit_price,
                        SignalDirection::Hold => Decimal::ZERO,
                    };
                    
                    let risk_amount = self.starting_balance * self.risk_per_trade;
                    let sl_distance = (trade.entry_price - trade.stop_loss).abs();
                    let position_value = if sl_distance > Decimal::ZERO {
                        risk_amount / sl_distance * trade.entry_price
                    } else {
                        Decimal::ZERO
                    };
                    let actual_profit = profit / trade.entry_price * position_value;
                    
                    trade.exit_time = Some(candle.time);
                    trade.exit_price = Some(exit_price);
                    trade.profit = Some(actual_profit);
                    
                    balance += actual_profit;
                    gross_loss += actual_profit.abs();
                    
                    trades.push(trade.clone());
                    open_trade = None;
                } else if hit_tp {
                    let exit_price = trade.take_profit;
                    let profit = match trade.direction {
                        SignalDirection::Buy => exit_price - trade.entry_price,
                        SignalDirection::Sell => trade.entry_price - exit_price,
                        SignalDirection::Hold => Decimal::ZERO,
                    };
                    
                    let risk_amount = self.starting_balance * self.risk_per_trade;
                    let sl_distance = (trade.entry_price - trade.stop_loss).abs();
                    let position_value = if sl_distance > Decimal::ZERO {
                        risk_amount / sl_distance * trade.entry_price
                    } else {
                        Decimal::ZERO
                    };
                    let actual_profit = profit / trade.entry_price * position_value;
                    
                    trade.exit_time = Some(candle.time);
                    trade.exit_price = Some(exit_price);
                    trade.profit = Some(actual_profit);
                    
                    balance += actual_profit;
                    gross_profit += actual_profit;
                    
                    trades.push(trade.clone());
                    open_trade = None;
                }

                if balance > peak_balance {
                    peak_balance = balance;
                }
                let drawdown = (peak_balance - balance) / peak_balance;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            } else {
                let signal = self.strategy.analyze(&obs, candle.close);
                
                if signal.direction != SignalDirection::Hold {
                    open_trade = Some(BacktestTrade {
                        entry_time: candle.time,
                        exit_time: None,
                        direction: signal.direction,
                        entry_price: candle.close,
                        exit_price: None,
                        stop_loss: signal.stop_loss,
                        take_profit: signal.take_profit,
                        profit: None,
                        conviction: signal.conviction,
                    });
                }
            }
        }

        let wins = trades.iter().filter(|t| t.profit.unwrap_or(Decimal::ZERO) > Decimal::ZERO).count();
        let losses = trades.iter().filter(|t| t.profit.unwrap_or(Decimal::ZERO) < Decimal::ZERO).count();
        let total_profit = balance - self.starting_balance;
        let win_rate = if trades.is_empty() { 0.0 } else { wins as f64 / trades.len() as f64 * 100.0 };
        let profit_factor = if gross_loss > Decimal::ZERO {
            gross_profit / gross_loss
        } else {
            Decimal::ZERO
        };

        BacktestResult {
            total_trades: trades.len(),
            wins,
            losses,
            total_profit,
            max_drawdown,
            win_rate,
            profit_factor,
            trades,
        }
    }

    pub fn load_csv(path: &str) -> Result<Vec<Candle>, std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut candles = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if i == 0 { continue; }
            
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            
            if parts.len() >= 6 {
                let timestamp: i64 = parts[0].parse().unwrap_or(0);
                let time = Utc.timestamp_opt(timestamp, 0).unwrap();
                
                let open = parts[1].parse::<Decimal>().unwrap_or(Decimal::ZERO);
                let high = parts[2].parse::<Decimal>().unwrap_or(Decimal::ZERO);
                let low = parts[3].parse::<Decimal>().unwrap_or(Decimal::ZERO);
                let close = parts[4].parse::<Decimal>().unwrap_or(Decimal::ZERO);
                let volume = parts[5].parse::<Decimal>().unwrap_or(Decimal::ZERO);

                candles.push(Candle::new(time, open, high, low, close, volume));
            }
        }

        Ok(candles)
    }
}

impl BacktestResult {
    pub fn print_summary(&self) {
        println!("============================================================");
        println!("              BACKTEST RESULTS                              ");
        println!("============================================================");
        println!(" Total Trades:    {:>6}", self.total_trades);
        println!(" Wins:            {:>6}", self.wins);
        println!(" Losses:          {:>6}", self.losses);
        println!(" Win Rate:        {:>6.1}%", self.win_rate);
        println!(" Total Profit:   ${:>10.2}", self.total_profit);
        println!(" Max Drawdown:    {:>6.2}%", self.max_drawdown * dec!(100));
        println!(" Profit Factor:   {:>6.2}", self.profit_factor);
        println!("============================================================");
    }
}

