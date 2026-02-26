#!/bin/bash

# Trading AI Application Runner

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
COMMAND=""
CONFIG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        download)
            COMMAND="download"
            shift
            ;;
        train)
            COMMAND="train"
            shift
            ;;
        backtest)
            COMMAND="backtest"
            shift
            ;;
        test)
            COMMAND="test"
            shift
            ;;
        testnet)
            COMMAND="testnet"
            shift
            ;;
        live)
            COMMAND="live"
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to run commands
run_command() {
    echo -e "${GREEN}Running: $1${NC}"
    case $1 in
        download)
            python scripts/download_data.py "$CONFIG"
            ;;
        train)
            python scripts/train_all_models.py "$CONFIG"
            ;;
        backtest)
            python scripts/backtest.py "$CONFIG"
            ;;
        test)
            python scripts/test_all.py
            ;;
        testnet)
            python scripts/run_testnet.py "$CONFIG"
            ;;
        live)
            python scripts/run_live.py "$CONFIG"
            ;;
    esac
}

# Main
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Trading AI Application${NC}"
echo -e "${YELLOW}========================================${NC}"

if [ -z "$COMMAND" ]; then
    echo "Usage: ./run.sh <command> [--config config_file]"
    echo ""
    echo "Commands:"
    echo "  download  - Download market data"
    echo "  train     - Train all AI models"
    echo "  backtest - Run backtest"
    echo "  test      - Run tests"
    echo "  testnet  - Run on testnet"
    echo "  live     - Run live trading"
    echo ""
    echo "Options:"
    echo "  --config  - Configuration file (YAML)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh download"
    echo "  ./run.sh train"
    echo "  ./run.sh backtest --config config/testnet.yaml"
    echo "  ./run.sh testnet --config config/testnet.yaml"
    exit 1
fi

# Add config if provided
CONFIG_ARG=""
if [ -n "$CONFIG" ]; then
    CONFIG_ARG="--config $CONFIG"
fi

run_command "$COMMAND" "$CONFIG_ARG"

echo -e "${GREEN}Done!${NC}"
