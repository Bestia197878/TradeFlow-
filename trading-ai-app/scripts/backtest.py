"""Backtest Script"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import yaml

from app.strategies import RSIStrategy
from app.backtesting import BacktestEngine
from app.services.exchange import ExchangeService


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_backtest(
    data_path: str = None,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    strategy: str = "rsi",
    initial_capital: float = 100000,
    output_dir: str = "results"
) -> dict:
    """
    Run backtest on historical data.

    Args:
        data_path: Path to data file
        symbol: Trading pair
        timeframe: Timeframe
        strategy: Strategy name
        initial_capital: Initial capital
        output_dir: Output directory

    Returns:
        Backtest results
    """
    print("="*50)
    print("BACKTEST")
    print("="*50)

    # Load data
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        print("Generating simulated data...")
        exchange = ExchangeService(testnet=True, simulate=True)
        df = exchange.fetch_ohlcv(symbol, timeframe, limit=500)

    print(f"Loaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Create strategy
    if strategy.lower() == "rsi":
        trading_strategy = RSIStrategy(
            initial_capital=initial_capital,
            rsi_period=14,
            overbought_threshold=70,
            oversold_threshold=30,
            use_divergence=False,
            use_momentum=True
        )
        print("Using RSI Strategy")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,
        slippage=0.0005
    )

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run(df, trading_strategy, verbose=1)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"backtest_{symbol.replace('/', '_')}")
    engine.save_results(results, output_path)

    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--strategy", type=str, default="rsi", help="Strategy name")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, help="Config file (YAML)")

    args = parser.parse_args()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        symbol = config.get('symbol', args.symbol)
        timeframe = config.get('timeframe', args.timeframe)
        strategy = config.get('strategy', args.strategy)
        initial_capital = config.get('initial_capital', args.capital)
    else:
        symbol = args.symbol
        timeframe = args.timeframe
        strategy = args.strategy
        initial_capital = args.capital

    run_backtest(
        data_path=args.data,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        initial_capital=initial_capital,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
