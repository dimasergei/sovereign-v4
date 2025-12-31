#!/usr/bin/env python3
"""
Bridge Watchdog - Auto-restarts mt5_bridge_v3.py if it crashes
"""

import subprocess
import time
import sys
import os

BRIDGE_SCRIPT = r"C:\sovereign-v4\vps\mt5_bridge_v3.py"
RESTART_DELAY = 5
MAX_RAPID_RESTARTS = 5
RAPID_WINDOW = 60


def main():
    print("=" * 50)
    print("Bridge Watchdog Starting")
    print(f"Monitoring: {BRIDGE_SCRIPT}")
    print("=" * 50)
    
    restart_times = []
    
    while True:
        now = time.time()
        restart_times = [t for t in restart_times if now - t < RAPID_WINDOW]
        
        if len(restart_times) >= MAX_RAPID_RESTARTS:
            print(f"ERROR: {MAX_RAPID_RESTARTS} restarts in {RAPID_WINDOW}s")
            print("Waiting 5 minutes...")
            time.sleep(300)
            restart_times = []
        
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting bridge...")
        
        try:
            process = subprocess.Popen(
                [sys.executable, BRIDGE_SCRIPT],
                cwd=os.path.dirname(BRIDGE_SCRIPT)
            )
            
            restart_times.append(time.time())
            exit_code = process.wait()
            
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Bridge exited: {exit_code}")
            
        except FileNotFoundError:
            print(f"ERROR: Not found: {BRIDGE_SCRIPT}")
            time.sleep(30)
            continue
        except Exception as e:
            print(f"ERROR: {e}")
        
        print(f"Restarting in {RESTART_DELAY}s...")
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nWatchdog stopped")