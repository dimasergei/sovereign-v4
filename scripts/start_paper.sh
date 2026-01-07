#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "========================================="
echo "Starting Sovereign v4 AGI Paper Trading"
echo "========================================="

# Verify build is up to date
echo "Checking build..."
cargo build --release 2>/dev/null || {
    echo "Build failed! Run 'cargo build --release' to see errors."
    exit 1
}

# Create data directory if it doesn't exist
mkdir -p data

echo "Starting with config: config/ibkr_paper.toml"
echo "Connecting to IBKR TWS on port 7497 (paper)..."
echo ""

RUST_LOG=info cargo run --release -- --config config/ibkr_paper.toml
