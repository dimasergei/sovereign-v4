//! Sovereign v4.0 - Watchdog
//!
//! External guardian process that monitors the main system.
//! If the main process dies, the watchdog restarts it.

use std::process::{Command, Child, Stdio};
use std::time::{Duration, Instant};
use std::thread;

const MAX_RESTARTS_PER_HOUR: u32 = 5;
const RESTART_DELAY_SECS: u64 = 10;

fn send_telegram(message: &str) {
    let token = "8570067655:AAHezpMffYJIc6oHkhmAX-MwXIpI91TvzR8";
    let chat_id = "7898079111";
    let url = format!(
        "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}",
        token, chat_id, urlencoding(message)
    );
    
    // Fire and forget - don't block watchdog
    let _ = Command::new("curl")
        .args(["-s", &url])
        .stdout(Stdio::null())
        .spawn();
}

fn urlencoding(s: &str) -> String {
    s.replace(' ', "%20")
        .replace('\n', "%0A")
        .replace('!', "%21")
        .replace('#', "%23")
}

fn start_sovereign() -> Option<Child> {
    println!("[WATCHDOG] Starting Sovereign...");
    
    match Command::new("./target/release/sovereign")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => {
            println!("[WATCHDOG] Sovereign started (PID: {})", child.id());
            Some(child)
        }
        Err(e) => {
            println!("[WATCHDOG] Failed to start: {}", e);
            send_telegram(&format!("❌ WATCHDOG: Failed to start Sovereign: {}", e));
            None
        }
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  SOVEREIGN WATCHDOG - External Guardian");
    println!("═══════════════════════════════════════════════════════════");
    
    send_telegram("🐕 Watchdog started - monitoring Sovereign");
    
    let mut restart_count: u32 = 0;
    let mut hour_start = Instant::now();
    
    loop {
        // Reset restart counter every hour
        if hour_start.elapsed() > Duration::from_secs(3600) {
            restart_count = 0;
            hour_start = Instant::now();
        }
        
        // Check restart limit
        if restart_count >= MAX_RESTARTS_PER_HOUR {
            println!("[WATCHDOG] Too many restarts! Waiting for next hour...");
            send_telegram("🚨 WATCHDOG: Too many restarts! Pausing for 1 hour.");
            thread::sleep(Duration::from_secs(3600));
            restart_count = 0;
            hour_start = Instant::now();
            continue;
        }
        
        // Start the process
        let mut child = match start_sovereign() {
            Some(c) => c,
            None => {
                thread::sleep(Duration::from_secs(RESTART_DELAY_SECS));
                restart_count += 1;
                continue;
            }
        };
        
        // Wait for process to exit
        match child.wait() {
            Ok(status) => {
                if status.success() {
                    println!("[WATCHDOG] Sovereign exited normally");
                    send_telegram("ℹ️ Sovereign exited normally");
                } else {
                    println!("[WATCHDOG] Sovereign crashed! Exit code: {:?}", status.code());
                    send_telegram(&format!("💀 Sovereign CRASHED! Exit: {:?}. Restarting...", status.code()));
                    restart_count += 1;
                }
            }
            Err(e) => {
                println!("[WATCHDOG] Error waiting for process: {}", e);
                send_telegram(&format!("❌ Watchdog error: {}", e));
                restart_count += 1;
            }
        }
        
        // Delay before restart
        println!("[WATCHDOG] Restarting in {} seconds...", RESTART_DELAY_SECS);
        thread::sleep(Duration::from_secs(RESTART_DELAY_SECS));
    }
}
