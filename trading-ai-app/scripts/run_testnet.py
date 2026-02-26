"""Run Testnet Script"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import time
import signal

from app.strategies import RSIStrategy
from app.services.exchange import ExchangeService
from app.services.risk import RiskManagement
from app.execution.live_trader import LiveTrader


# Global trader instance for signal handling
trader = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down...")
    if trader:
        trader.stop()
    sys.exit(0)


def run_testnet(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    interval: int = 60,
    initial_capital: float = 100000,
    paper_trading: bool = True
):
    """
    Run trading on testnet.

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        interval: Update interval in seconds
        initial_capital: Initial capital
        paper_trading: Use paper trading mode
    """
    global trader

    print("="*50)
    print("TESTNET TRADING")
    print("="*50)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Interval: {interval}s")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Paper Trading: {paper_trading}")
    print("="*50)

    # Initialize exchange
    print("\nInitializing exchange...")
    exchange = ExchangeService(
        exchange_id="binance",
        testnet=True,
        simulate=paper_trading
    )

    # Initialize strategy
    strategy = RSIStrategy(
        initial_capital=initial_capital,
        rsi_period=14,
        overbought_threshold=70,
        oversold_threshold=30,
        use_momentum=True
    )

    # Initialize risk management
    risk_manager = RiskManagement(
        max_position_size=0.5,
        max_daily_loss=0.05,
        max_drawdown=0.15,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )

    # Configuration
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'interval': interval,
        'initial_capital': initial_capital
    }

    # Initialize live trader
    trader = LiveTrader(
        exchange=exchange,
        strategy=strategy,
        config=config,
        risk_manager=risk_manager,
        paper_trading=paper_trading
    )

    # Set up callbacks
    def on_trade(trade):
        print(f"\n{'='*30}")
        print(f"TRADE EXECUTED")
        print(f"{'='*30}")
        print(f"Action: {trade['action'].upper()}")
        print(f"Price: ${trade['price']:,.2f}")
        print(f"Value: ${trade.get('value', 0):,.2f}")
        print(f"Status: {trade['status']}")
        print(f"Reason: {trade.get('reason', 'N/A')}")

        # Print status
        status = trader.get_status()
        print(f"\nPosition: {status['position']:.4f}")
        print(f"Capital: ${status['capital']:,.2f}")
        print(f"Total Trades: {status['total_trades']}")

    def on_error(error):
        print(f"Error: {error}")

    def on_signal(signal):
        print(f"Signal: {signal.action} (confidence: {signal.confidence:.2f}) - {signal.reason}")

    trader.on_trade = on_trade
    trader.on_error = on_error
    trader.on_signal = on_signal

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Start trading
    print("\nStarting testnet trading...")
    trader.start()

    print("\nTrading is running. Press Ctrl+C to stop.")

    # Keep running
    try:
        while True:
            status = trader.get_status()
            print(f"\rRunning | Position: {status['position']:.4f} | Capital: ${status['capital']:,.2f} | Trades: {status['total_trades']}", end="")
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if trader:
            trader.stop()

        # Print final performance
        print("\n\nFinal Performance:")
        perf = trader.get_performance()
        print(f"Total Return: {perf['total_return']*100:.2f}%")
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run testnet trading")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--interval", type=int, default=60, help="Update interval (seconds)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--live", action="store_true", help="Use real money (not paper trading)")
    parser.add_argument("--config", type=str, help="Config file (YAML)")

    args = parser.parse_args()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            symbol = config.get('symbol', args.symbol)
            timeframe = config.get('timeframe', args.timeframe)
            interval = config.get('interval', args.interval)
            initial_capital = config.get('initial_capital', args.capital)
            paper_trading = not args.live
    else:
        symbol = args.symbol
        timeframe = args.timeframe
        interval = args.interval
        initial_capital = args.capital
        paper_trading = not args.live

    run_testnet(
        symbol=symbol,
        timeframe=timeframe,
        interval=interval,
        initial_capital=initial_capital,
        paper_trading=paper_trading
    )


if __name__ == "__main__":
    main()
