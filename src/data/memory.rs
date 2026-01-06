//! Trade Memory - Persistent AGI Learning Storage
//!
//! Stores learned patterns and outcomes for continuous improvement:
//! - S/R level effectiveness (which levels actually predict bounces)
//! - Volume calibration (optimal thresholds per symbol)
//! - Trade context (full entry context for outcome analysis)
//! - Market regime history (bull/bear/sideways classification)

use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::path::Path;
use std::sync::Mutex;

use crate::core::regime::Regime;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    HighVolatility,
    LowVolatility,
    Unknown,
}

impl MarketRegime {
    pub fn as_str(&self) -> &'static str {
        match self {
            MarketRegime::Bull => "BULL",
            MarketRegime::Bear => "BEAR",
            MarketRegime::Sideways => "SIDEWAYS",
            MarketRegime::HighVolatility => "HIGH_VOL",
            MarketRegime::LowVolatility => "LOW_VOL",
            MarketRegime::Unknown => "UNKNOWN",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "BULL" | "TRENDING_UP" => MarketRegime::Bull,
            "BEAR" | "TRENDING_DOWN" => MarketRegime::Bear,
            "SIDEWAYS" | "RANGING" => MarketRegime::Sideways,
            "HIGH_VOL" | "VOLATILE" => MarketRegime::HighVolatility,
            "LOW_VOL" => MarketRegime::LowVolatility,
            _ => MarketRegime::Unknown,
        }
    }

    /// Convert from HMM Regime to MarketRegime
    pub fn from_regime(regime: Regime) -> Self {
        match regime {
            Regime::TrendingUp => MarketRegime::Bull,
            Regime::TrendingDown => MarketRegime::Bear,
            Regime::Ranging => MarketRegime::Sideways,
            Regime::Volatile => MarketRegime::HighVolatility,
        }
    }
}

/// S/R effectiveness record
#[derive(Debug, Clone)]
pub struct SrEffectiveness {
    pub symbol: String,
    pub price_level: f64,
    pub granularity: f64,
    pub touch_count: i32,
    pub bounce_count: i32,
    pub break_count: i32,
    pub total_profit: f64,
    pub avg_profit_per_trade: f64,
    pub last_touched: DateTime<Utc>,
}

/// Trade context for learning
#[derive(Debug, Clone)]
pub struct TradeContext {
    pub id: i64,
    pub symbol: String,
    pub ticket: u64,
    pub direction: String,
    pub entry_price: f64,
    pub sr_level: f64,
    pub sr_score: i32,
    pub volume_percentile: f64,
    pub atr: f64,
    pub regime: String,
    pub entry_bar_count: u64,
    pub opened_at: DateTime<Utc>,
    // Exit fields (filled on close)
    pub exit_price: Option<f64>,
    pub profit: Option<f64>,
    pub profit_pct: Option<f64>,
    pub hit_tp: Option<bool>,
    pub hit_sl: Option<bool>,
    pub hold_bars: Option<i64>,
    pub mae: Option<f64>,  // Maximum Adverse Excursion
    pub mfe: Option<f64>,  // Maximum Favorable Excursion
    pub closed_at: Option<DateTime<Utc>>,
}

/// Volume calibration record
#[derive(Debug, Clone)]
pub struct VolumeCalibration {
    pub symbol: String,
    pub current_threshold: f64,
    pub optimal_threshold: f64,
    pub trades_at_threshold: i32,
    pub win_rate_at_threshold: f64,
    pub last_calibrated: DateTime<Utc>,
}

/// Regime statistics
#[derive(Debug, Clone)]
pub struct RegimeStats {
    pub regime: String,
    pub total_trades: i32,
    pub wins: i32,
    pub losses: i32,
    pub total_profit: f64,
    pub avg_hold_bars: f64,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
}

/// Persistent memory for AGI learning
pub struct TradeMemory {
    conn: Mutex<rusqlite::Connection>,
}

// Implement Send + Sync for Arc usage
unsafe impl Send for TradeMemory {}
unsafe impl Sync for TradeMemory {}

impl TradeMemory {
    /// Create or open the memory database
    pub fn new(path: &str) -> Result<Self> {
        let exists = Path::new(path).exists();
        let conn = rusqlite::Connection::open(path)?;

        if !exists {
            // S/R effectiveness tracking
            conn.execute(
                "CREATE TABLE sr_effectiveness (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    price_level REAL NOT NULL,
                    granularity REAL NOT NULL,
                    touch_count INTEGER DEFAULT 0,
                    bounce_count INTEGER DEFAULT 0,
                    break_count INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    last_touched TEXT NOT NULL,
                    UNIQUE(symbol, price_level, granularity)
                )",
                [],
            )?;

            // Trade context for outcome analysis
            conn.execute(
                "CREATE TABLE trade_context (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    ticket INTEGER NOT NULL UNIQUE,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    sr_level REAL NOT NULL,
                    sr_score INTEGER NOT NULL,
                    volume_percentile REAL NOT NULL,
                    atr REAL NOT NULL,
                    regime TEXT NOT NULL,
                    entry_bar_count INTEGER NOT NULL,
                    opened_at TEXT NOT NULL,
                    exit_price REAL,
                    profit REAL,
                    profit_pct REAL,
                    hit_tp INTEGER,
                    hit_sl INTEGER,
                    hold_bars INTEGER,
                    mae REAL,
                    mfe REAL,
                    closed_at TEXT
                )",
                [],
            )?;

            // Volume calibration per symbol
            conn.execute(
                "CREATE TABLE volume_calibration (
                    symbol TEXT PRIMARY KEY,
                    current_threshold REAL NOT NULL,
                    optimal_threshold REAL NOT NULL,
                    trades_at_threshold INTEGER DEFAULT 0,
                    wins_at_threshold INTEGER DEFAULT 0,
                    last_calibrated TEXT NOT NULL
                )",
                [],
            )?;

            // Market regime history
            conn.execute(
                "CREATE TABLE regime_history (
                    id INTEGER PRIMARY KEY,
                    regime TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    total_hold_bars INTEGER DEFAULT 0,
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                )",
                [],
            )?;

            // Create indexes for performance
            conn.execute(
                "CREATE INDEX idx_sr_symbol ON sr_effectiveness(symbol)",
                [],
            )?;
            conn.execute(
                "CREATE INDEX idx_trade_symbol ON trade_context(symbol)",
                [],
            )?;
            conn.execute(
                "CREATE INDEX idx_trade_regime ON trade_context(regime)",
                [],
            )?;

            println!("[MEMORY] Created new memory database: {}", path);
        } else {
            println!("[MEMORY] Opened existing memory database: {}", path);
        }

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // =========================================================================
    // S/R EFFECTIVENESS METHODS
    // =========================================================================

    /// Record a touch of an S/R level (price approached the level)
    pub fn record_sr_touch(
        &self,
        symbol: &str,
        price_level: f64,
        granularity: f64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO sr_effectiveness (symbol, price_level, granularity, touch_count, last_touched)
             VALUES (?1, ?2, ?3, 1, ?4)
             ON CONFLICT(symbol, price_level, granularity) DO UPDATE SET
                touch_count = touch_count + 1,
                last_touched = ?4",
            rusqlite::params![symbol, price_level, granularity, now],
        )?;

        Ok(())
    }

    /// Record trade outcome at an S/R level
    pub fn record_sr_trade_outcome(
        &self,
        symbol: &str,
        price_level: f64,
        granularity: f64,
        bounced: bool,
        profit: f64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        if bounced {
            conn.execute(
                "INSERT INTO sr_effectiveness (symbol, price_level, granularity, touch_count, bounce_count, total_profit, last_touched)
                 VALUES (?1, ?2, ?3, 1, 1, ?4, ?5)
                 ON CONFLICT(symbol, price_level, granularity) DO UPDATE SET
                    touch_count = touch_count + 1,
                    bounce_count = bounce_count + 1,
                    total_profit = total_profit + ?4,
                    last_touched = ?5",
                rusqlite::params![symbol, price_level, granularity, profit, now],
            )?;
        } else {
            conn.execute(
                "INSERT INTO sr_effectiveness (symbol, price_level, granularity, touch_count, break_count, total_profit, last_touched)
                 VALUES (?1, ?2, ?3, 1, 1, ?4, ?5)
                 ON CONFLICT(symbol, price_level, granularity) DO UPDATE SET
                    touch_count = touch_count + 1,
                    break_count = break_count + 1,
                    total_profit = total_profit + ?4,
                    last_touched = ?5",
                rusqlite::params![symbol, price_level, granularity, profit, now],
            )?;
        }

        Ok(())
    }

    /// Get win rate for a specific S/R level
    pub fn get_sr_win_rate(
        &self,
        symbol: &str,
        price_level: f64,
        granularity: f64,
    ) -> Result<Option<f64>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT bounce_count, touch_count FROM sr_effectiveness
             WHERE symbol = ?1 AND ABS(price_level - ?2) < ?3"
        )?;

        let result = stmt.query_row(
            rusqlite::params![symbol, price_level, granularity / 2.0],
            |row| {
                let bounces: i32 = row.get(0)?;
                let touches: i32 = row.get(1)?;
                Ok((bounces, touches))
            },
        );

        match result {
            Ok((bounces, touches)) if touches > 0 => {
                Ok(Some(bounces as f64 / touches as f64))
            }
            _ => Ok(None),
        }
    }

    /// Get win rate and trade count for a specific S/R level
    /// Returns (win_rate, trade_count) if found
    pub fn get_sr_win_rate_with_count(
        &self,
        symbol: &str,
        price_level: f64,
        granularity: f64,
    ) -> Result<Option<(f64, i32)>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT bounce_count, touch_count FROM sr_effectiveness
             WHERE symbol = ?1 AND ABS(price_level - ?2) < ?3"
        )?;

        let result = stmt.query_row(
            rusqlite::params![symbol, price_level, granularity / 2.0],
            |row| {
                let bounces: i32 = row.get(0)?;
                let touches: i32 = row.get(1)?;
                Ok((bounces, touches))
            },
        );

        match result {
            Ok((bounces, touches)) if touches > 0 => {
                let win_rate = bounces as f64 / touches as f64;
                Ok(Some((win_rate, touches)))
            }
            _ => Ok(None),
        }
    }

    /// Get S/R effectiveness for a symbol
    pub fn get_sr_effectiveness(&self, symbol: &str, min_touches: i32) -> Result<Vec<SrEffectiveness>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT symbol, price_level, granularity, touch_count, bounce_count, break_count,
                    total_profit, last_touched
             FROM sr_effectiveness
             WHERE symbol = ?1 AND touch_count >= ?2
             ORDER BY bounce_count DESC"
        )?;

        let rows = stmt.query_map(rusqlite::params![symbol, min_touches], |row| {
            let touches: i32 = row.get(3)?;
            let total_profit: f64 = row.get(6)?;
            let bounces: i32 = row.get(4)?;

            Ok(SrEffectiveness {
                symbol: row.get(0)?,
                price_level: row.get(1)?,
                granularity: row.get(2)?,
                touch_count: touches,
                bounce_count: bounces,
                break_count: row.get(5)?,
                total_profit,
                avg_profit_per_trade: if bounces > 0 { total_profit / bounces as f64 } else { 0.0 },
                last_touched: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // =========================================================================
    // TRADE CONTEXT METHODS
    // =========================================================================

    /// Record trade entry with full context
    pub fn record_trade_entry(
        &self,
        symbol: &str,
        ticket: u64,
        direction: &str,
        entry_price: Decimal,
        sr_level: Decimal,
        sr_score: i32,
        volume_percentile: f64,
        atr: Decimal,
        regime: &str,
        entry_bar_count: u64,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO trade_context
             (symbol, ticket, direction, entry_price, sr_level, sr_score, volume_percentile,
              atr, regime, entry_bar_count, opened_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            rusqlite::params![
                symbol,
                ticket as i64,
                direction,
                entry_price.to_f64().unwrap_or(0.0),
                sr_level.to_f64().unwrap_or(0.0),
                sr_score,
                volume_percentile,
                atr.to_f64().unwrap_or(0.0),
                regime,
                entry_bar_count as i64,
                now,
            ],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Record trade exit with outcome
    pub fn record_trade_exit(
        &self,
        ticket: u64,
        exit_price: Decimal,
        profit: Decimal,
        profit_pct: f64,
        hit_tp: bool,
        hit_sl: bool,
        hold_bars: i64,
        mae: f64,
        mfe: f64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "UPDATE trade_context SET
                exit_price = ?1,
                profit = ?2,
                profit_pct = ?3,
                hit_tp = ?4,
                hit_sl = ?5,
                hold_bars = ?6,
                mae = ?7,
                mfe = ?8,
                closed_at = ?9
             WHERE ticket = ?10",
            rusqlite::params![
                exit_price.to_f64().unwrap_or(0.0),
                profit.to_f64().unwrap_or(0.0),
                profit_pct,
                hit_tp as i32,
                hit_sl as i32,
                hold_bars,
                mae,
                mfe,
                now,
                ticket as i64,
            ],
        )?;

        Ok(())
    }

    /// Get trade context by ticket
    pub fn get_trade_context(&self, ticket: u64) -> Result<Option<TradeContext>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, symbol, ticket, direction, entry_price, sr_level, sr_score,
                    volume_percentile, atr, regime, entry_bar_count, opened_at,
                    exit_price, profit, profit_pct, hit_tp, hit_sl, hold_bars,
                    mae, mfe, closed_at
             FROM trade_context WHERE ticket = ?1"
        )?;

        let result = stmt.query_row([ticket as i64], |row| {
            Ok(TradeContext {
                id: row.get(0)?,
                symbol: row.get(1)?,
                ticket: row.get::<_, i64>(2)? as u64,
                direction: row.get(3)?,
                entry_price: row.get(4)?,
                sr_level: row.get(5)?,
                sr_score: row.get(6)?,
                volume_percentile: row.get(7)?,
                atr: row.get(8)?,
                regime: row.get(9)?,
                entry_bar_count: row.get::<_, i64>(10)? as u64,
                opened_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(11)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                exit_price: row.get(12)?,
                profit: row.get(13)?,
                profit_pct: row.get(14)?,
                hit_tp: row.get::<_, Option<i32>>(15)?.map(|v| v != 0),
                hit_sl: row.get::<_, Option<i32>>(16)?.map(|v| v != 0),
                hold_bars: row.get(17)?,
                mae: row.get(18)?,
                mfe: row.get(19)?,
                closed_at: row.get::<_, Option<String>>(20)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
        });

        match result {
            Ok(ctx) => Ok(Some(ctx)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get all trade contexts for a symbol
    pub fn get_trade_contexts(&self, symbol: &str, limit: i32) -> Result<Vec<TradeContext>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, symbol, ticket, direction, entry_price, sr_level, sr_score,
                    volume_percentile, atr, regime, entry_bar_count, opened_at,
                    exit_price, profit, profit_pct, hit_tp, hit_sl, hold_bars,
                    mae, mfe, closed_at
             FROM trade_context
             WHERE symbol = ?1 AND closed_at IS NOT NULL
             ORDER BY closed_at DESC
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(rusqlite::params![symbol, limit], |row| {
            Ok(TradeContext {
                id: row.get(0)?,
                symbol: row.get(1)?,
                ticket: row.get::<_, i64>(2)? as u64,
                direction: row.get(3)?,
                entry_price: row.get(4)?,
                sr_level: row.get(5)?,
                sr_score: row.get(6)?,
                volume_percentile: row.get(7)?,
                atr: row.get(8)?,
                regime: row.get(9)?,
                entry_bar_count: row.get::<_, i64>(10)? as u64,
                opened_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(11)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                exit_price: row.get(12)?,
                profit: row.get(13)?,
                profit_pct: row.get(14)?,
                hit_tp: row.get::<_, Option<i32>>(15)?.map(|v| v != 0),
                hit_sl: row.get::<_, Option<i32>>(16)?.map(|v| v != 0),
                hold_bars: row.get(17)?,
                mae: row.get(18)?,
                mfe: row.get(19)?,
                closed_at: row.get::<_, Option<String>>(20)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // =========================================================================
    // VOLUME CALIBRATION METHODS
    // =========================================================================

    /// Update volume calibration for a symbol
    pub fn update_volume_calibration(
        &self,
        symbol: &str,
        threshold_used: f64,
        was_winner: bool,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        // First, try to get existing calibration
        let existing: Option<(f64, i32, i32)> = conn
            .query_row(
                "SELECT optimal_threshold, trades_at_threshold, wins_at_threshold
                 FROM volume_calibration WHERE symbol = ?1",
                [symbol],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .ok();

        if let Some((current_optimal, trades, wins)) = existing {
            let new_trades = trades + 1;
            let new_wins = if was_winner { wins + 1 } else { wins };
            let new_win_rate = new_wins as f64 / new_trades as f64;

            // Adjust optimal threshold using exponential moving average
            // If win rate is improving, keep current threshold
            // If declining, adjust towards the threshold that won
            let alpha = 0.1;
            let new_optimal = if was_winner {
                alpha * threshold_used + (1.0 - alpha) * current_optimal
            } else {
                current_optimal // Don't adjust on losses
            };

            conn.execute(
                "UPDATE volume_calibration SET
                    current_threshold = ?1,
                    optimal_threshold = ?2,
                    trades_at_threshold = ?3,
                    wins_at_threshold = ?4,
                    last_calibrated = ?5
                 WHERE symbol = ?6",
                rusqlite::params![
                    threshold_used,
                    new_optimal,
                    new_trades,
                    new_wins,
                    now,
                    symbol
                ],
            )?;
        } else {
            // First trade for this symbol
            conn.execute(
                "INSERT INTO volume_calibration
                 (symbol, current_threshold, optimal_threshold, trades_at_threshold, wins_at_threshold, last_calibrated)
                 VALUES (?1, ?2, ?3, 1, ?4, ?5)",
                rusqlite::params![
                    symbol,
                    threshold_used,
                    threshold_used,
                    if was_winner { 1 } else { 0 },
                    now
                ],
            )?;
        }

        Ok(())
    }

    /// Get optimal volume threshold for a symbol
    pub fn get_optimal_volume_threshold(&self, symbol: &str) -> Result<Option<f64>> {
        let conn = self.conn.lock().unwrap();

        let result = conn.query_row(
            "SELECT optimal_threshold FROM volume_calibration
             WHERE symbol = ?1 AND trades_at_threshold >= 5",
            [symbol],
            |row| row.get(0),
        );

        match result {
            Ok(threshold) => Ok(Some(threshold)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get volume calibration for a symbol
    pub fn get_volume_calibration(&self, symbol: &str) -> Result<Option<VolumeCalibration>> {
        let conn = self.conn.lock().unwrap();

        let result = conn.query_row(
            "SELECT symbol, current_threshold, optimal_threshold, trades_at_threshold,
                    wins_at_threshold, last_calibrated
             FROM volume_calibration WHERE symbol = ?1",
            [symbol],
            |row| {
                let trades: i32 = row.get(3)?;
                let wins: i32 = row.get(4)?;
                Ok(VolumeCalibration {
                    symbol: row.get(0)?,
                    current_threshold: row.get(1)?,
                    optimal_threshold: row.get(2)?,
                    trades_at_threshold: trades,
                    win_rate_at_threshold: if trades > 0 { wins as f64 / trades as f64 } else { 0.0 },
                    last_calibrated: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            },
        );

        match result {
            Ok(cal) => Ok(Some(cal)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // =========================================================================
    // REGIME METHODS
    // =========================================================================

    /// Start a new market regime (global)
    pub fn start_regime_global(&self, regime: MarketRegime) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        // Close any open regime
        conn.execute(
            "UPDATE regime_history SET ended_at = ?1 WHERE ended_at IS NULL",
            [&now],
        )?;

        // Start new regime
        conn.execute(
            "INSERT INTO regime_history (regime, started_at) VALUES (?1, ?2)",
            rusqlite::params![regime.as_str(), now],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Start a new market regime for a specific symbol (HMM-detected)
    pub fn start_regime(&self, symbol: &str, regime_str: &str) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        // Close any open regime for this symbol
        // Note: For symbol-specific regimes, we use the symbol as a prefix in the regime field
        let regime_key = format!("{}:{}", symbol, regime_str);

        conn.execute(
            "UPDATE regime_history SET ended_at = ?1 WHERE regime LIKE ?2 AND ended_at IS NULL",
            rusqlite::params![now, format!("{}:%", symbol)],
        )?;

        // Start new regime
        conn.execute(
            "INSERT INTO regime_history (regime, started_at) VALUES (?1, ?2)",
            rusqlite::params![regime_key, now],
        )?;

        Ok(conn.last_insert_rowid())
    }

    /// Get current market regime
    pub fn get_current_regime(&self) -> Result<Option<MarketRegime>> {
        let conn = self.conn.lock().unwrap();

        let result = conn.query_row(
            "SELECT regime FROM regime_history WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        );

        match result {
            Ok(regime_str) => Ok(Some(MarketRegime::from_str(&regime_str))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Update regime statistics after a trade
    pub fn update_regime_stats(
        &self,
        profit: f64,
        hold_bars: i64,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        let is_win = profit > 0.0;

        conn.execute(
            "UPDATE regime_history SET
                total_trades = total_trades + 1,
                wins = wins + ?1,
                losses = losses + ?2,
                total_profit = total_profit + ?3,
                total_hold_bars = total_hold_bars + ?4
             WHERE ended_at IS NULL",
            rusqlite::params![
                if is_win { 1 } else { 0 },
                if is_win { 0 } else { 1 },
                profit,
                hold_bars,
            ],
        )?;

        Ok(())
    }

    /// Get regime statistics
    pub fn get_regime_stats(&self, regime: MarketRegime) -> Result<Option<RegimeStats>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT regime, total_trades, wins, losses, total_profit, total_hold_bars,
                    started_at, ended_at
             FROM regime_history
             WHERE regime = ?1
             ORDER BY id DESC LIMIT 1"
        )?;

        let result = stmt.query_row([regime.as_str()], |row| {
            let trades: i32 = row.get(1)?;
            let total_hold_bars: i64 = row.get(5)?;
            Ok(RegimeStats {
                regime: row.get(0)?,
                total_trades: trades,
                wins: row.get(2)?,
                losses: row.get(3)?,
                total_profit: row.get(4)?,
                avg_hold_bars: if trades > 0 { total_hold_bars as f64 / trades as f64 } else { 0.0 },
                started_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                ended_at: row.get::<_, Option<String>>(7)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
        });

        match result {
            Ok(stats) => Ok(Some(stats)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // =========================================================================
    // ANALYTICS METHODS
    // =========================================================================

    /// Get win rate by regime
    pub fn get_win_rate_by_regime(&self) -> Result<Vec<(String, f64, i32)>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT regime,
                    CAST(SUM(wins) AS REAL) / NULLIF(SUM(total_trades), 0) as win_rate,
                    SUM(total_trades) as trades
             FROM regime_history
             GROUP BY regime
             HAVING SUM(total_trades) > 0"
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                row.get::<_, i32>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get S/R score effectiveness (which scores lead to wins)
    pub fn get_sr_score_effectiveness(&self, symbol: &str) -> Result<Vec<(i32, f64, i32)>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT sr_score,
                    CAST(SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as win_rate,
                    COUNT(*) as trades
             FROM trade_context
             WHERE symbol = ?1 AND closed_at IS NOT NULL
             GROUP BY sr_score
             ORDER BY sr_score"
        )?;

        let rows = stmt.query_map([symbol], |row| {
            Ok((
                row.get::<_, i32>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, i32>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get overall statistics
    pub fn get_overall_stats(&self) -> Result<(i32, i32, f64, f64)> {
        let conn = self.conn.lock().unwrap();

        let result = conn.query_row(
            "SELECT COUNT(*),
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END),
                    COALESCE(SUM(profit), 0),
                    COALESCE(AVG(profit), 0)
             FROM trade_context
             WHERE closed_at IS NOT NULL",
            [],
            |row| {
                Ok((
                    row.get::<_, i32>(0)?,
                    row.get::<_, Option<i32>>(1)?.unwrap_or(0),
                    row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                    row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                ))
            },
        )?;

        Ok(result)
    }

    /// Get volume percentile effectiveness
    pub fn get_volume_percentile_effectiveness(&self, symbol: &str) -> Result<Vec<(i32, f64, i32)>> {
        let conn = self.conn.lock().unwrap();

        // Group by volume percentile buckets (0-10, 10-20, etc.)
        let mut stmt = conn.prepare(
            "SELECT CAST(volume_percentile / 10 AS INTEGER) * 10 as bucket,
                    CAST(SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as win_rate,
                    COUNT(*) as trades
             FROM trade_context
             WHERE symbol = ?1 AND closed_at IS NOT NULL
             GROUP BY bucket
             ORDER BY bucket"
        )?;

        let rows = stmt.query_map([symbol], |row| {
            Ok((
                row.get::<_, i32>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, i32>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_memory_creation() {
        let memory = TradeMemory::new(":memory:").unwrap();
        assert!(memory.get_overall_stats().is_ok());
    }

    #[test]
    fn test_sr_effectiveness() {
        let memory = TradeMemory::new(":memory:").unwrap();

        // Record some touches and outcomes
        memory.record_sr_touch("AAPL", 150.0, 1.0).unwrap();
        memory.record_sr_touch("AAPL", 150.0, 1.0).unwrap();
        memory.record_sr_trade_outcome("AAPL", 150.0, 1.0, true, 100.0).unwrap();
        memory.record_sr_trade_outcome("AAPL", 150.0, 1.0, false, -50.0).unwrap();

        // Check win rate
        let win_rate = memory.get_sr_win_rate("AAPL", 150.0, 1.0).unwrap();
        assert!(win_rate.is_some());
        // 1 bounce out of 4 touches = 25%
        assert!((win_rate.unwrap() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_trade_context() {
        let memory = TradeMemory::new(":memory:").unwrap();

        // Record entry
        let id = memory.record_trade_entry(
            "AAPL",
            12345,
            "BUY",
            dec!(150.0),
            dec!(148.0),
            -2,
            85.0,
            dec!(2.5),
            "BULL",
            100,
        ).unwrap();
        assert!(id > 0);

        // Record exit
        memory.record_trade_exit(
            12345,
            dec!(155.0),
            dec!(500.0),
            3.33,
            true,
            false,
            5,
            -1.5,
            4.0,
        ).unwrap();

        // Retrieve context
        let ctx = memory.get_trade_context(12345).unwrap();
        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert_eq!(ctx.symbol, "AAPL");
        assert_eq!(ctx.profit, Some(500.0));
        assert_eq!(ctx.hit_tp, Some(true));
    }

    #[test]
    fn test_volume_calibration() {
        let memory = TradeMemory::new(":memory:").unwrap();

        // Record several trades at different thresholds
        memory.update_volume_calibration("AAPL", 80.0, true).unwrap();
        memory.update_volume_calibration("AAPL", 75.0, true).unwrap();
        memory.update_volume_calibration("AAPL", 85.0, false).unwrap();
        memory.update_volume_calibration("AAPL", 80.0, true).unwrap();
        memory.update_volume_calibration("AAPL", 80.0, true).unwrap();

        // Get optimal threshold
        let threshold = memory.get_optimal_volume_threshold("AAPL").unwrap();
        assert!(threshold.is_some());
    }

    #[test]
    fn test_regime_tracking() {
        let memory = TradeMemory::new(":memory:").unwrap();

        // Start a bull regime (global)
        memory.start_regime_global(MarketRegime::Bull).unwrap();

        // Record some trades
        memory.update_regime_stats(100.0, 5).unwrap();
        memory.update_regime_stats(-50.0, 3).unwrap();
        memory.update_regime_stats(75.0, 4).unwrap();

        // Check current regime
        let current = memory.get_current_regime().unwrap();
        assert_eq!(current, Some(MarketRegime::Bull));

        // Get stats
        let stats = memory.get_regime_stats(MarketRegime::Bull).unwrap();
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.total_trades, 3);
        assert_eq!(stats.wins, 2);
        assert_eq!(stats.losses, 1);
    }

    #[test]
    fn test_symbol_regime_tracking() {
        let memory = TradeMemory::new(":memory:").unwrap();

        // Start a trending up regime for AAPL
        memory.start_regime("AAPL", "TRENDING_UP").unwrap();

        // Start a different regime for MSFT
        memory.start_regime("MSFT", "RANGING").unwrap();

        // Both should be recorded separately
        // (This is basic existence test - actual regime retrieval would need more work)
    }

    #[test]
    fn test_analytics() {
        let memory = TradeMemory::new(":memory:").unwrap();

        // Record some trades
        for i in 0..10 {
            memory.record_trade_entry(
                "AAPL",
                1000 + i,
                "BUY",
                dec!(150.0),
                dec!(148.0),
                -(i as i32 % 5), // Different S/R scores
                (50 + i * 5) as f64, // Different volume percentiles
                dec!(2.5),
                "BULL",
                100,
            ).unwrap();

            memory.record_trade_exit(
                1000 + i,
                dec!(155.0),
                if i % 3 == 0 { dec!(-50.0) } else { dec!(100.0) },
                3.0,
                i % 3 != 0,
                i % 3 == 0,
                5,
                -1.0,
                3.0,
            ).unwrap();
        }

        // Test S/R score effectiveness
        let sr_eff = memory.get_sr_score_effectiveness("AAPL").unwrap();
        assert!(!sr_eff.is_empty());

        // Test volume percentile effectiveness
        let vol_eff = memory.get_volume_percentile_effectiveness("AAPL").unwrap();
        assert!(!vol_eff.is_empty());

        // Test overall stats
        let (total, wins, profit, _avg) = memory.get_overall_stats().unwrap();
        assert_eq!(total, 10);
        assert!(wins > 0);
        assert!(profit != 0.0);
    }
}
