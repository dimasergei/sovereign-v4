#!/bin/bash
echo "========================================="
echo "EMERGENCY STOP - Sovereign v4"
echo "========================================="
echo ""

# Kill all sovereign processes
pkill -f "sovereign.*--config" 2>/dev/null && echo "Killed sovereign processes." || echo "No sovereign processes found."

# Also try to kill cargo run processes
pkill -f "cargo run.*sovereign" 2>/dev/null && echo "Killed cargo run processes." || true

echo ""
echo "========================================="
echo "All trading processes killed."
echo "========================================="
echo ""
echo "IMPORTANT: Check IBKR TWS for any open positions!"
echo "Positions may still be open - close them manually if needed."
echo ""
