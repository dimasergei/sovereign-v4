#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "========================================="
echo "=== LIVE TRADING PRE-FLIGHT CHECK ==="
echo "========================================="
echo ""

# Run readiness check
echo "Running readiness check..."
cargo run --bin readiness_check
if [ $? -ne 0 ]; then
    echo ""
    echo "========================================="
    echo "READINESS CHECK FAILED. Aborting."
    echo "========================================="
    echo ""
    echo "Continue paper trading until all criteria are met."
    exit 1
fi

echo ""
echo "========================================="
echo "WARNING: LIVE TRADING MODE"
echo "========================================="
echo ""
echo "You are about to trade with REAL CAPITAL"
echo "Config: config/ibkr_live.toml"
echo "Port: 7496 (live)"
echo ""

read -p "Go LIVE with real capital? Type 'LIVE' to confirm: " confirm
if [ "$confirm" != "LIVE" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "========================================="
echo "Starting Sovereign v4 AGI LIVE Trading"
echo "========================================="
echo ""

# Create data directory if it doesn't exist
mkdir -p data

RUST_LOG=info cargo run --release -- --config config/ibkr_live.toml --live
