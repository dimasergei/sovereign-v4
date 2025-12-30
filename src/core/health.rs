//! Health Monitor - Detects and recovers from data gaps

use std::time::{Duration, Instant};

pub struct HealthMonitor {
    last_tick: Instant,
    last_candle: Instant,
    tick_timeout: Duration,
    candle_timeout: Duration,
    gap_count: u32,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            last_tick: Instant::now(),
            last_candle: Instant::now(),
            tick_timeout: Duration::from_secs(30),
            candle_timeout: Duration::from_secs(360),
            gap_count: 0,
        }
    }

    pub fn record_tick(&mut self) {
        let was_stale = self.is_tick_stale();
        self.last_tick = Instant::now();
        if was_stale {
            self.gap_count = 0; // Reset on recovery
        }
    }

    pub fn record_candle(&mut self) {
        self.last_candle = Instant::now();
    }

    pub fn is_tick_stale(&self) -> bool {
        self.last_tick.elapsed() > self.tick_timeout
    }

    pub fn is_candle_stale(&self) -> bool {
        self.last_candle.elapsed() > self.candle_timeout
    }

    pub fn check(&mut self) -> HealthStatus {
        let tick_age = self.last_tick.elapsed();
        let candle_age = self.last_candle.elapsed();

        if tick_age > self.tick_timeout {
            self.gap_count += 1;
            return HealthStatus::StaleData {
                seconds: tick_age.as_secs(),
                gaps: self.gap_count,
            };
        }

        if candle_age > self.candle_timeout {
            return HealthStatus::MissingCandles {
                seconds: candle_age.as_secs(),
            };
        }

        HealthStatus::Healthy
    }

    pub fn should_reconnect(&self) -> bool {
        self.gap_count >= 3
    }

    pub fn should_alert(&self) -> bool {
        self.gap_count >= 5
    }

    pub fn gap_count(&self) -> u32 {
        self.gap_count
    }

    pub fn tick_age_secs(&self) -> u64 {
        self.last_tick.elapsed().as_secs()
    }

    pub fn candle_age_secs(&self) -> u64 {
        self.last_candle.elapsed().as_secs()
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    StaleData { seconds: u64, gaps: u32 },
    MissingCandles { seconds: u64 },
}
