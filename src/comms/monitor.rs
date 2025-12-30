//! System Monitor
//!
//! Collects and reports system health metrics.

use std::time::{Duration, Instant};
use rust_decimal::Decimal;

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub uptime_secs: u64,
    pub ticks_processed: u64,
    pub candles_processed: u64,
    pub signals_generated: u64,
    pub trades_executed: u64,
    pub errors_count: u64,
    pub last_tick_time: Option<Instant>,
    pub last_candle_time: Option<Instant>,
    pub balance: Decimal,
    pub equity: Decimal,
    pub total_pnl: Decimal,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            uptime_secs: 0,
            ticks_processed: 0,
            candles_processed: 0,
            signals_generated: 0,
            trades_executed: 0,
            errors_count: 0,
            last_tick_time: None,
            last_candle_time: None,
            balance: Decimal::ZERO,
            equity: Decimal::ZERO,
            total_pnl: Decimal::ZERO,
        }
    }
}

pub struct Monitor {
    start_time: Instant,
    metrics: SystemMetrics,
    stale_threshold: Duration,
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: SystemMetrics::default(),
            stale_threshold: Duration::from_secs(60),
        }
    }

    pub fn record_tick(&mut self) {
        self.metrics.ticks_processed += 1;
        self.metrics.last_tick_time = Some(Instant::now());
    }

    pub fn record_candle(&mut self) {
        self.metrics.candles_processed += 1;
        self.metrics.last_candle_time = Some(Instant::now());
    }

    pub fn record_signal(&mut self) {
        self.metrics.signals_generated += 1;
    }

    pub fn record_trade(&mut self) {
        self.metrics.trades_executed += 1;
    }

    pub fn record_error(&mut self) {
        self.metrics.errors_count += 1;
    }

    pub fn update_account(&mut self, balance: Decimal, equity: Decimal, pnl: Decimal) {
        self.metrics.balance = balance;
        self.metrics.equity = equity;
        self.metrics.total_pnl = pnl;
    }

    pub fn get_metrics(&self) -> SystemMetrics {
        let mut m = self.metrics.clone();
        m.uptime_secs = self.start_time.elapsed().as_secs();
        m
    }

    pub fn is_data_stale(&self) -> bool {
        match self.metrics.last_tick_time {
            Some(t) => t.elapsed() > self.stale_threshold,
            None => true,
        }
    }

    pub fn health_check(&self) -> HealthStatus {
        let uptime = self.start_time.elapsed();
        
        // Check if we're receiving data
        let data_ok = !self.is_data_stale();
        
        // Check error rate (more than 10 errors per hour is concerning)
        let error_rate = if uptime.as_secs() > 0 {
            self.metrics.errors_count as f64 / (uptime.as_secs() as f64 / 3600.0)
        } else {
            0.0
        };
        let errors_ok = error_rate < 10.0;
        
        // Check if system is trading (at least 1 candle per 10 minutes after warmup)
        let trading_ok = if uptime.as_secs() > 600 {
            self.metrics.candles_processed > 0
        } else {
            true // Still warming up
        };

        if data_ok && errors_ok && trading_ok {
            HealthStatus::Healthy
        } else if !data_ok {
            HealthStatus::Degraded("No data received".to_string())
        } else if !errors_ok {
            HealthStatus::Degraded(format!("High error rate: {:.1}/hr", error_rate))
        } else {
            HealthStatus::Degraded("Not processing candles".to_string())
        }
    }

    pub fn summary(&self) -> String {
        let m = self.get_metrics();
        let health = self.health_check();
        
        format!(
            "Uptime: {}h {}m | Ticks: {} | Candles: {} | Trades: {} | P&L: ${} | Health: {:?}",
            m.uptime_secs / 3600,
            (m.uptime_secs % 3600) / 60,
            m.ticks_processed,
            m.candles_processed,
            m.trades_executed,
            m.total_pnl,
            health
        )
    }
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded(String),
    Critical(String),
}
