//! Trade Memory - Extended storage for AGI learning capabilities
//! 
//! Extends TradeDb with:
//! - S/R level effectiveness tracking
//! - Volume threshold calibration
//! - Trade context for outcome learning
//! - Regime classification history

use anyhow::Result;
use rusqlite::{Connection, params};
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};

/// Extended memory for AGI learning
pub struct TradeMemory {
    conn: Connection,
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// S/R level effectiveness record
#[derive(Debug, Clone)]
pub struct SREffectiveness {
    pub symbol: String,
    pub price_level: f64,
    pub touch_count: i32,
    pub trades_taken: i32,
    pub wins: i32,
    pub total_profit: f64,
    pub avg_hold_bars: f64,
    pub last_touch: DateTime<Utc>,
    pub last_trade: Option<DateTime<Utc>>,
}

/// Volume calibration record
#[derive(Debug, Clone)]
pub struct VolumeCalibration {
    pub symbol: String,
    pub current_threshold: f64,      // percentile threshold
    pub trades_at_threshold: i32,
    pub wins_at_threshold: i32,
    pub total_profit: f64,
    pub last_calibrated: DateTime<Utc>,
}

/// Trade context - full state at decision time
#[derive(Debug, Clone)]
pub struct TradeContext {
    pub id: i64,
    pub ticket: u64,
    pub symbol: String,
    pub direction: String,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub profit: Option<f64>,
    
    // Decision context
    pub sr_level: f64,               // S/R level that triggered entry
    pub sr_score: i32,               // Count at that level (0 = strongest)
    pub volume_percentile: f64,      // Volume percentile at entry
    pub atr: f64,                    // ATR at entry
    pub distance_to_sr_pct: f64,     // How close price was to S/R (%)
    
    // Outcome
    pub won: Option<bool>,
    pub hit_tp: Option<bool>,
    pub hit_sl: Option<bool>,
    pub hold_duration_bars: Option<i32>,
    pub max_adverse_excursion: Option<f64>,  // Worst drawdown during trade
    pub max_favorable_excursion: Option<f64>, // Best profit during trade
    
    // Regime at entry
    pub regime: String,              // trending_up/trending_down/ranging/volatile
    
    pub opened_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
}

/// Regime period record
#[derive(Debug, Clone)]
pub struct RegimePeriod {
    pub id: i64,
    pub symbol: String,
    pub regime: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub trades_during: i32,
    pub win_rate: Option<f64>,
    pub total_profit: Option<f64>,
}

// ============================================================================
// IMPLEMENTATION
// ============================================================================

impl TradeMemory {
    /// Create new memory store (extends existing db or creates new)
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        
        // Create memory tables if they don't exist
        Self::init_schema(&conn)?;
        
        println!("[MEMORY] Initialized trade memory: {}", path);
        Ok(Self { conn })
    }
    
    /// Initialize AGI memory schema
    fn init_schema(conn: &Connection) -> Result<()> {
        // S/R Effectiveness - tracks which levels actually work
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sr_effectiveness (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                price_level REAL NOT NULL,
                touch_count INTEGER DEFAULT 0,
                trades_taken INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0.0,
                avg_hold_bars REAL DEFAULT 0.0,
                last_touch TEXT NOT NULL,
                last_trade TEXT,
                UNIQUE(symbol, price_level)
            )",
            [],
        )?;
        
        // Volume Calibration - optimal thresholds per symbol
        conn.execute(
            "CREATE TABLE IF NOT EXISTS volume_calibration (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                current_threshold REAL NOT NULL,
                trades_at_threshold INTEGER DEFAULT 0,
                wins_at_threshold INTEGER DEFAULT 0,
                total_profit REAL DEFAULT 0.0,
                last_calibrated TEXT NOT NULL
            )",
            [],
        )?;
        
        // Trade Context - full state at decision time
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_context (
                id INTEGER PRIMARY KEY,
                ticket INTEGER NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                profit REAL,
                
                sr_level REAL NOT NULL,
                sr_score INTEGER NOT NULL,
                volume_percentile REAL NOT NULL,
                atr REAL NOT NULL,
                distance_to_sr_pct REAL NOT NULL,
                
                won INTEGER,
                hit_tp INTEGER,
                hit_sl INTEGER,
                hold_duration_bars INTEGER,
                max_adverse_excursion REAL,
                max_favorable_excursion REAL,
                
                regime TEXT NOT NULL,
                
                opened_at TEXT NOT NULL,
                closed_at TEXT
            )",
            [],
        )?;
        
        // Regime History - track market regime periods
        conn.execute(
            "CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                trades_during INTEGER DEFAULT 0,
                win_rate REAL,
                total_profit REAL
            )",
            [],
        )?;
        
        // Indexes for fast queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sr_symbol ON sr_effectiveness(symbol)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_symbol ON trade_context(symbol)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_regime ON trade_context(regime)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_history(symbol)",
            [],
        )?;
        
        Ok(())
    }
    
    // ========================================================================
    // S/R EFFECTIVENESS
    // ========================================================================
    
    /// Record an S/R level touch (price approached level)
    pub fn record_sr_touch(&self, symbol: &str, price_level: f64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        
        self.conn.execute(
            "INSERT INTO sr_effectiveness (symbol, price_level, touch_count, last_touch)
             VALUES (?1, ?2, 1, ?3)
             ON CONFLICT(symbol, price_level) DO UPDATE SET
                touch_count = touch_count + 1,
                last_touch = ?3",
            params![symbol, price_level, now],
        )?;
        
        Ok(())
    }
    
    /// Record trade outcome at an S/R level
    pub fn record_sr_trade_outcome(
        &self,
        symbol: &str,
        price_level: f64,
        won: bool,
        profit: f64,
        hold_bars: i32,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        
        // Update running averages
        self.conn.execute(
            "INSERT INTO sr_effectiveness (symbol, price_level, touch_count, trades_taken, wins, total_profit, avg_hold_bars, last_touch, last_trade)
             VALUES (?1, ?2, 0, 1, ?3, ?4, ?5, ?6, ?6)
             ON CONFLICT(symbol, price_level) DO UPDATE SET
                trades_taken = trades_taken + 1,
                wins = wins + ?3,
                total_profit = total_profit + ?4,
                avg_hold_bars = (avg_hold_bars * trades_taken + ?5) / (trades_taken + 1),
                last_trade = ?6",
            params![
                symbol,
                price_level,
                if won { 1 } else { 0 },
                profit,
                hold_bars as f64,
                now,
            ],
        )?;
        
        Ok(())
    }
    
    /// Get S/R effectiveness for a symbol
    pub fn get_sr_effectiveness(&self, symbol: &str) -> Result<Vec<SREffectiveness>> {
        let mut stmt = self.conn.prepare(
            "SELECT symbol, price_level, touch_count, trades_taken, wins, 
                    total_profit, avg_hold_bars, last_touch, last_trade
             FROM sr_effectiveness
             WHERE symbol = ?1
             ORDER BY trades_taken DESC"
        )?;
        
        let rows = stmt.query_map([symbol], |row| {
            Ok(SREffectiveness {
                symbol: row.get(0)?,
                price_level: row.get(1)?,
                touch_count: row.get(2)?,
                trades_taken: row.get(3)?,
                wins: row.get(4)?,
                total_profit: row.get(5)?,
                avg_hold_bars: row.get(6)?,
                last_touch: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                last_trade: row.get::<_, Option<String>>(8)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
        })?;
        
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
    
    /// Get win rate for a specific S/R level
    pub fn get_sr_win_rate(&self, symbol: &str, price_level: f64, tolerance: f64) -> Result<Option<f64>> {
        let mut stmt = self.conn.prepare(
            "SELECT CAST(wins AS REAL) / NULLIF(trades_taken, 0) as win_rate
             FROM sr_effectiveness
             WHERE symbol = ?1 AND ABS(price_level - ?2) < ?3
             LIMIT 1"
        )?;
        
        let result = stmt.query_row(params![symbol, price_level, tolerance], |row| {
            row.get::<_, Option<f64>>(0)
        }).ok().flatten();
        
        Ok(result)
    }
    
    // ========================================================================
    // VOLUME CALIBRATION
    // ========================================================================
    
    /// Update volume threshold calibration
    pub fn update_volume_calibration(
        &self,
        symbol: &str,
        threshold: f64,
        won: bool,
        profit: f64,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        
        self.conn.execute(
            "INSERT INTO volume_calibration (symbol, current_threshold, trades_at_threshold, wins_at_threshold, total_profit, last_calibrated)
             VALUES (?1, ?2, 1, ?3, ?4, ?5)
             ON CONFLICT(symbol) DO UPDATE SET
                current_threshold = ?2,
                trades_at_threshold = trades_at_threshold + 1,
                wins_at_threshold = wins_at_threshold + ?3,
                total_profit = total_profit + ?4,
                last_calibrated = ?5",
            params![
                symbol,
                threshold,
                if won { 1 } else { 0 },
                profit,
                now,
            ],
        )?;
        
        Ok(())
    }
    
    /// Get volume calibration for symbol
    pub fn get_volume_calibration(&self, symbol: &str) -> Result<Option<VolumeCalibration>> {
        let mut stmt = self.conn.prepare(
            "SELECT symbol, current_threshold, trades_at_threshold, wins_at_threshold, 
                    total_profit, last_calibrated
             FROM volume_calibration
             WHERE symbol = ?1"
        )?;
        
        let result = stmt.query_row([symbol], |row| {
            Ok(VolumeCalibration {
                symbol: row.get(0)?,
                current_threshold: row.get(1)?,
                trades_at_threshold: row.get(2)?,
                wins_at_threshold: row.get(3)?,
                total_profit: row.get(4)?,
                last_calibrated: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        }).ok();
        
        Ok(result)
    }
    
    // ========================================================================
    // TRADE CONTEXT
    // ========================================================================
    
    /// Record full trade context at entry
    pub fn record_trade_entry(
        &self,
        ticket: u64,
        symbol: &str,
        direction: &str,
        entry_price: f64,
        sr_level: f64,
        sr_score: i32,
        volume_percentile: f64,
        atr: f64,
        regime: &str,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        let distance_to_sr_pct = ((entry_price - sr_level).abs() / entry_price) * 100.0;
        
        self.conn.execute(
            "INSERT INTO trade_context (
                ticket, symbol, direction, entry_price, sr_level, sr_score,
                volume_percentile, atr, distance_to_sr_pct, regime, opened_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                ticket as i64,
                symbol,
                direction,
                entry_price,
                sr_level,
                sr_score,
                volume_percentile,
                atr,
                distance_to_sr_pct,
                regime,
                now,
            ],
        )?;
        
        Ok(self.conn.last_insert_rowid())
    }
    
    /// Record trade exit with outcome
    pub fn record_trade_exit(
        &self,
        ticket: u64,
        exit_price: f64,
        profit: f64,
        hit_tp: bool,
        hit_sl: bool,
        hold_duration_bars: i32,
        max_adverse_excursion: f64,
        max_favorable_excursion: f64,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let won = profit > 0.0;
        
        self.conn.execute(
            "UPDATE trade_context SET
                exit_price = ?1,
                profit = ?2,
                won = ?3,
                hit_tp = ?4,
                hit_sl = ?5,
                hold_duration_bars = ?6,
                max_adverse_excursion = ?7,
                max_favorable_excursion = ?8,
                closed_at = ?9
             WHERE ticket = ?10",
            params![
                exit_price,
                profit,
                if won { 1 } else { 0 },
                if hit_tp { 1 } else { 0 },
                if hit_sl { 1 } else { 0 },
                hold_duration_bars,
                max_adverse_excursion,
                max_favorable_excursion,
                now,
                ticket as i64,
            ],
        )?;
        
        Ok(())
    }
    
    /// Get trade contexts for learning (closed trades with outcomes)
    pub fn get_trade_contexts(
        &self,
        symbol: Option<&str>,
        regime: Option<&str>,
        limit: i32,
    ) -> Result<Vec<TradeContext>> {
        let sql = format!(
            "SELECT id, ticket, symbol, direction, entry_price, exit_price, profit,
                    sr_level, sr_score, volume_percentile, atr, distance_to_sr_pct,
                    won, hit_tp, hit_sl, hold_duration_bars, 
                    max_adverse_excursion, max_favorable_excursion,
                    regime, opened_at, closed_at
             FROM trade_context
             WHERE closed_at IS NOT NULL
             {}
             {}
             ORDER BY closed_at DESC
             LIMIT ?",
            symbol.map(|_| "AND symbol = ?").unwrap_or(""),
            regime.map(|_| "AND regime = ?").unwrap_or(""),
        );
        
        let mut stmt = self.conn.prepare(&sql)?;
        
        // Build params dynamically
        let mut param_idx = 0;
        let params: Vec<Box<dyn rusqlite::ToSql>> = {
            let mut p: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            if let Some(s) = symbol {
                p.push(Box::new(s.to_string()));
                param_idx += 1;
            }
            if let Some(r) = regime {
                p.push(Box::new(r.to_string()));
                param_idx += 1;
            }
            p.push(Box::new(limit));
            p
        };
        
        let params_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(TradeContext {
                id: row.get(0)?,
                ticket: row.get::<_, i64>(1)? as u64,
                symbol: row.get(2)?,
                direction: row.get(3)?,
                entry_price: row.get(4)?,
                exit_price: row.get(5)?,
                profit: row.get(6)?,
                sr_level: row.get(7)?,
                sr_score: row.get(8)?,
                volume_percentile: row.get(9)?,
                atr: row.get(10)?,
                distance_to_sr_pct: row.get(11)?,
                won: row.get::<_, Option<i32>>(12)?.map(|v| v == 1),
                hit_tp: row.get::<_, Option<i32>>(13)?.map(|v| v == 1),
                hit_sl: row.get::<_, Option<i32>>(14)?.map(|v| v == 1),
                hold_duration_bars: row.get(15)?,
                max_adverse_excursion: row.get(16)?,
                max_favorable_excursion: row.get(17)?,
                regime: row.get(18)?,
                opened_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(19)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                closed_at: row.get::<_, Option<String>>(20)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
        })?;
        
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
    
    // ========================================================================
    // REGIME TRACKING
    // ========================================================================
    
    /// Start a new regime period
    pub fn start_regime(&self, symbol: &str, regime: &str) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        
        // End any open regime for this symbol
        self.conn.execute(
            "UPDATE regime_history SET ended_at = ?1 
             WHERE symbol = ?2 AND ended_at IS NULL",
            params![now, symbol],
        )?;
        
        // Start new regime
        self.conn.execute(
            "INSERT INTO regime_history (symbol, regime, started_at)
             VALUES (?1, ?2, ?3)",
            params![symbol, regime, now],
        )?;
        
        Ok(self.conn.last_insert_rowid())
    }
    
    /// Get current regime for symbol
    pub fn get_current_regime(&self, symbol: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT regime FROM regime_history
             WHERE symbol = ?1 AND ended_at IS NULL
             ORDER BY started_at DESC LIMIT 1"
        )?;
        
        let result = stmt.query_row([symbol], |row| {
            row.get::<_, String>(0)
        }).ok();
        
        Ok(result)
    }
    
    /// Update regime statistics when trade closes
    pub fn update_regime_stats(&self, symbol: &str, won: bool, profit: f64) -> Result<()> {
        self.conn.execute(
            "UPDATE regime_history SET
                trades_during = trades_during + 1,
                win_rate = (COALESCE(win_rate, 0) * trades_during + ?1) / (trades_during + 1),
                total_profit = COALESCE(total_profit, 0) + ?2
             WHERE symbol = ?3 AND ended_at IS NULL",
            params![if won { 1.0 } else { 0.0 }, profit, symbol],
        )?;
        
        Ok(())
    }
    
    /// Get regime performance history
    pub fn get_regime_history(&self, symbol: &str) -> Result<Vec<RegimePeriod>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, symbol, regime, started_at, ended_at, trades_during, win_rate, total_profit
             FROM regime_history
             WHERE symbol = ?1
             ORDER BY started_at DESC"
        )?;
        
        let rows = stmt.query_map([symbol], |row| {
            Ok(RegimePeriod {
                id: row.get(0)?,
                symbol: row.get(1)?,
                regime: row.get(2)?,
                started_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                ended_at: row.get::<_, Option<String>>(4)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
                trades_during: row.get(5)?,
                win_rate: row.get(6)?,
                total_profit: row.get(7)?,
            })
        })?;
        
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
    
    // ========================================================================
    // ANALYTICS
    // ========================================================================
    
    /// Get overall win rate by regime
    pub fn get_win_rate_by_regime(&self) -> Result<Vec<(String, f64, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT regime, 
                    CAST(SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as win_rate,
                    COUNT(*) as trade_count
             FROM trade_context
             WHERE closed_at IS NOT NULL
             GROUP BY regime
             ORDER BY win_rate DESC"
        )?;
        
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, i32>(2)?,
            ))
        })?;
        
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
    
    /// Get S/R score effectiveness (do lower scores = better trades?)
    pub fn get_sr_score_effectiveness(&self) -> Result<Vec<(i32, f64, i32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT sr_score,
                    CAST(SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as win_rate,
                    COUNT(*) as trade_count
             FROM trade_context
             WHERE closed_at IS NOT NULL
             GROUP BY sr_score
             ORDER BY sr_score ASC"
        )?;
        
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i32>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, i32>(2)?,
            ))
        })?;
        
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
    
    /// Get optimal volume percentile threshold (which threshold has best results?)
    pub fn get_optimal_volume_threshold(&self, symbol: &str) -> Result<Option<f64>> {
        let mut stmt = self.conn.prepare(
            "SELECT 
                ROUND(volume_percentile / 5) * 5 as bucket,
                CAST(SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as win_rate,
                COUNT(*) as n
             FROM trade_context
             WHERE symbol = ?1 AND closed_at IS NOT NULL
             GROUP BY bucket
             HAVING n >= 5
             ORDER BY win_rate DESC
             LIMIT 1"
        )?;
        
        let result = stmt.query_row([symbol], |row| {
            row.get::<_, f64>(0)
        }).ok();
        
        Ok(result)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_creation() {
        let mem = TradeMemory::new(":memory:").unwrap();
        assert!(mem.get_sr_effectiveness("AAPL").unwrap().is_empty());
    }
    
    #[test]
    fn test_sr_effectiveness() {
        let mem = TradeMemory::new(":memory:").unwrap();
        
        // Record touches and trades
        mem.record_sr_touch("AAPL", 150.0).unwrap();
        mem.record_sr_touch("AAPL", 150.0).unwrap();
        mem.record_sr_trade_outcome("AAPL", 150.0, true, 100.0, 5).unwrap();
        mem.record_sr_trade_outcome("AAPL", 150.0, false, -50.0, 3).unwrap();
        
        let effectiveness = mem.get_sr_effectiveness("AAPL").unwrap();
        assert_eq!(effectiveness.len(), 1);
        assert_eq!(effectiveness[0].touch_count, 2);
        assert_eq!(effectiveness[0].trades_taken, 2);
        assert_eq!(effectiveness[0].wins, 1);
        assert_eq!(effectiveness[0].total_profit, 50.0);
    }
    
    #[test]
    fn test_trade_context() {
        let mem = TradeMemory::new(":memory:").unwrap();
        
        // Record entry
        mem.record_trade_entry(
            1001, "AAPL", "BUY", 150.0, 148.0, -2, 85.0, 2.5, "trending_up"
        ).unwrap();
        
        // Record exit
        mem.record_trade_exit(1001, 155.0, 500.0, true, false, 10, -50.0, 600.0).unwrap();
        
        let contexts = mem.get_trade_contexts(Some("AAPL"), None, 10).unwrap();
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].won, Some(true));
        assert_eq!(contexts[0].profit, Some(500.0));
    }
    
    #[test]
    fn test_regime_tracking() {
        let mem = TradeMemory::new(":memory:").unwrap();
        
        mem.start_regime("AAPL", "trending_up").unwrap();
        assert_eq!(mem.get_current_regime("AAPL").unwrap(), Some("trending_up".to_string()));
        
        mem.start_regime("AAPL", "ranging").unwrap();
        assert_eq!(mem.get_current_regime("AAPL").unwrap(), Some("ranging".to_string()));
        
        let history = mem.get_regime_history("AAPL").unwrap();
        assert_eq!(history.len(), 2);
    }
}
