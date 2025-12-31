#!/usr/bin/env python3
"""
Bridge Watchdog - Auto-restarts mt5_bridge_v3.py if it crashes
"""

import subprocess
import time
import sys
import os

BRIDGE_SCRIPT = r"C:\sovereign-v4\vps\mt5_bridge_v3.py"
RESTART_DELAY = 5  # seconds between restart attempts
MAX_RAPID_RESTARTS = 5  # max restarts within RAPID_WINDOW
RAPID_WINDOW = 60  # seconds

def main():
    print("=" * 50)
    print("Bridge Watchdog Starting")
    print(f"Monitoring: {BRIDGE_SCRIPT}")
    print("=" * 50)
    
    restart_times = []
    
    while True:
        # Check for rapid restart loop
        now = time.time()
        restart_times = [t for t in restart_times if now - t < RAPID_WINDOW]
        
        if len(restart_times) >= MAX_RAPID_RESTARTS:
            print(f"ERROR: {MAX_RAPID_RESTARTS} restarts in {RAPID_WINDOW}s - possible crash loop")
            print("Waiting 5 minutes before retry...")
            time.sleep(300)
            restart_times = []
        
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting bridge...")
        
        try:
            process = subprocess.Popen(
                [sys.executable, BRIDGE_SCRIPT],
                cwd=os.path.dirname(BRIDGE_SCRIPT)
            )
            
            restart_times.append(time.time())
            
            # Wait for process to exit
            exit_code = process.wait()
            
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Bridge exited with code {exit_code}")
            
        except FileNotFoundError:
            print(f"ERROR: Bridge script not found: {BRIDGE_SCRIPT}")
            time.sleep(30)
            continue
        except Exception as e:
            print(f"ERROR: Failed to start bridge: {e}")
        
        print(f"Restarting in {RESTART_DELAY} seconds...")
        time.sleep(RESTART_DELAY)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nWatchdog stopped")