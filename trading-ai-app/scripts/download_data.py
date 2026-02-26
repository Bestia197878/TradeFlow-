"""Download Market Data Script"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime, timedelta
import time

from app.services.exchange import ExchangeService


def download_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = None,
    end_date: str = None,
    output_dir: str = "data"
) -> pd.DataFrame:
    """
    Download historical market data.

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} {timeframe} data...")

    # Initialize exchange
    exchange = ExchangeService(testnet=True, simulate=False)

    # Parse dates
    if end_date:
        end = pd.to_datetime(end_date)
    else:
        end = datetime.now()

    if start_date:
        start = pd.to_datetime(start_date)
    else:
        start = end - timedelta(days=365)

    # Calculate required candles
    tf_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }
    minutes_per_candle = tf_minutes.get(timeframe, 60)
    total_minutes = (end - start).total_seconds() / 60
    limit = int(total_minutes / minutes_per_candle) + 1

    # Limit max candles
    limit = min(limit, 1000)

    print(f"Date range: {start} to {end}")
    print(f"Fetching {limit} candles...")

    # Fetch data
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    if data is not None and len(data) > 0:
        # Filter by date range
        data = data[data.index >= start]
        data = data[data.index <= end]

        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath)

        print(f"Downloaded {len(data)} candles")
        print(f"Saved to {filepath}")

        return data
    else:
        print("No data downloaded")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data", help="Output directory")

    args = parser.parse_args()

    download_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
