//! Trade Database - SQLite storage for trade history

use anyhow::Result;
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use std::path::Path;

pub struct TradeDb {
    conn: rusqlite::Connection,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub id: i64,
    pub ticket: u64,
    pub direction: String,
    pub lots: Decimal,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub sl: Decimal,
    pub tp: Decimal,
    pub profit: Option<Decimal>,
    pub opened_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    pub conviction: u8,
}

impl TradeDb {
    pub fn new(path: &str) -> Result<Self> {
        let exists = Path::new(path).exists();
        let conn = rusqlite::Connection::open(path)?;
        
        if !exists {
            conn.execute(
                "CREATE TABLE trades (
                    id INTEGER PRIMARY KEY,
                    ticket INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    lots REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    sl REAL NOT NULL,
                    tp REAL NOT NULL,
                    profit REAL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    conviction INTEGER NOT NULL
                )",
                [],
            )?;
            
            conn.execute(
                "CREATE TABLE daily_stats (
                    date TEXT PRIMARY KEY,
                    trades INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    losses INTEGER NOT NULL,
                    profit REAL NOT NULL
                )",
                [],
            )?;
            
            println!("[DB] Created new database: {}", path);
        } else {
            println!("[DB] Opened existing database: {}", path);
        }
        
        Ok(Self { conn })
    }
    
    pub fn record_open(
        &self,
        ticket: u64,
        direction: &str,
        lots: Decimal,
        entry_price: Decimal,
        sl: Decimal,
        tp: Decimal,
        conviction: u8,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        
        self.conn.execute(
            "INSERT INTO trades (ticket, direction, lots, entry_price, sl, tp, opened_at, conviction)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                ticket as i64,
                direction,
                lots.to_string().parse::<f64>()?,
                entry_price.to_string().parse::<f64>()?,
                sl.to_string().parse::<f64>()?,
                tp.to_string().parse::<f64>()?,
                now,
                conviction as i32,
            ],
        )?;
        
        Ok(self.conn.last_insert_rowid())
    }
    
    pub fn record_close(
        &self,
        ticket: u64,
        exit_price: Decimal,
        profit: Decimal,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        
        self.conn.execute(
            "UPDATE trades SET exit_price = ?1, profit = ?2, closed_at = ?3 WHERE ticket = ?4",
            rusqlite::params![
                exit_price.to_string().parse::<f64>()?,
                profit.to_string().parse::<f64>()?,
                now,
                ticket as i64,
            ],
        )?;
        
        Ok(())
    }
    
    pub fn get_today_stats(&self) -> Result<(i32, i32, i32, f64)> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        
        let mut stmt = self.conn.prepare(
            "SELECT COUNT(*), 
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END),
                    COALESCE(SUM(profit), 0)
             FROM trades 
             WHERE date(opened_at) = ?1 AND closed_at IS NOT NULL"
        )?;
        
        let result = stmt.query_row([&today], |row| {
            Ok((
                row.get::<_, i32>(0)?,
                row.get::<_, Option<i32>>(1)?.unwrap_or(0),
                row.get::<_, Option<i32>>(2)?.unwrap_or(0),
                row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
            ))
        })?;
        
        Ok(result)
    }
    
    pub fn get_total_stats(&self) -> Result<(i32, i32, i32, f64)> {
        let mut stmt = self.conn.prepare(
            "SELECT COUNT(*), 
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END),
                    COALESCE(SUM(profit), 0)
             FROM trades 
             WHERE closed_at IS NOT NULL"
        )?;
        
        let result = stmt.query_row([], |row| {
            Ok((
                row.get::<_, i32>(0)?,
                row.get::<_, Option<i32>>(1)?.unwrap_or(0),
                row.get::<_, Option<i32>>(2)?.unwrap_or(0),
                row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
            ))
        })?;
        
        Ok(result)
    }
}
