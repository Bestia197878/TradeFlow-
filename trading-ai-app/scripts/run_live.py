"""Run Live Trading Script"""

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


def run_live(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    interval: int = 60,
    initial_capital: float = 100000,
    api_key: str = None,
    api_secret: str = None,
    # Strategy parameters from config
    rsi_period: int = 14,
    overbought_threshold: float = 65,
    oversold_threshold: float = 35,
    use_momentum: bool = True,
    # Risk parameters from config
    max_position_size: float = 0.3,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.10,
    stop_loss_pct: float = 0.015,
    take_profit_pct: float = 0.03,
    trailing_stop: bool = False,
    trailing_stop_pct: float = 0.01
):
    """
    Run live trading with real money.

    Args:
        symbol: Trading pair
        timeframe: Timeframe
        interval: Update interval in seconds
        initial_capital: Initial capital
        api_key: Exchange API key
        api_secret: Exchange API secret
    """
    global trader

    print("="*50)
    print("LIVE TRADING")
    print("="*50)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Interval: {interval}s")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("="*50)

    # Warning
    print("\n⚠️  WARNING: Live trading with real money!")
    print("⚠️  Make sure you understand the risks.")
    confirm = input("Type 'YES' to confirm: ")
    if confirm != 'YES':
        print("Cancelled.")
        return

    # Initialize exchange
    print("\nInitializing exchange...")
    exchange = ExchangeService(
        exchange_id="binance",
        api_key=api_key,
        api_secret=api_secret,
        testnet=False,
        simulate=False
    )

    # Check balance
    balance = exchange.fetch_balance()
    print(f"Account balance: {balance}")

    # Initialize strategy (use config parameters)
    strategy = RSIStrategy(
        initial_capital=initial_capital,
        rsi_period=rsi_period,
        overbought_threshold=overbought_threshold,
        oversold_threshold=oversold_threshold,
        use_momentum=use_momentum
    )

    # Initialize risk management (use config parameters)
    risk_manager = RiskManagement(
        max_position_size=max_position_size,
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop=trailing_stop,
        trailing_stop_pct=trailing_stop_pct
    )

    # Configuration
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'interval': interval,
        'initial_capital': initial_capital
    }

    # Initialize live trader (not paper trading)
    trader = LiveTrader(
        exchange=exchange,
        strategy=strategy,
        config=config,
        risk_manager=risk_manager,
        paper_trading=False  # REAL TRADING
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
        print(f"Order ID: {trade.get('order_id', 'N/A')}")

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
    print("\nStarting live trading...")
    trader.start()

    print("\nLive trading is running. Press Ctrl+C to stop.")
    print("⚠️  WARNING: Real money is at risk!")

    # Keep running
    try:
        while True:
            status = trader.get_status()
            print(f"\rLive | Position: {status['position']:.4f} | Capital: ${status['capital']:,.2f} | Trades: {status['total_trades']}", end="")
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
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run live trading")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--interval", type=int, default=60, help="Update interval (seconds)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--api-key", type=str, help="Exchange API key")
    parser.add_argument("--api-secret", type=str, help="Exchange API secret")
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
            api_key = args.api_key or config.get('api_key')
            api_secret = args.api_secret or config.get('api_secret')
            
            # Load strategy settings
            strategy_config = config.get('strategy', {})
            rsi_period = strategy_config.get('rsi_period', 14)
            overbought_threshold = strategy_config.get('overbought_threshold', 65)
            oversold_threshold = strategy_config.get('oversold_threshold', 35)
            use_momentum = strategy_config.get('use_momentum', True)
            
            # Load risk settings
            risk_config = config.get('risk', {})
            max_position_size = risk_config.get('max_position_size', 0.3)
            max_daily_loss = risk_config.get('max_daily_loss', 0.03)
            max_drawdown = risk_config.get('max_drawdown', 0.10)
            stop_loss_pct = risk_config.get('stop_loss_pct', 0.015)
            take_profit_pct = risk_config.get('take_profit_pct', 0.03)
            trailing_stop = risk_config.get('trailing_stop', False)
            trailing_stop_pct = risk_config.get('trailing_stop_pct', 0.01)
    else:
        symbol = args.symbol
        timeframe = args.timeframe
        interval = args.interval
        initial_capital = args.capital
        api_key = args.api_key
        api_secret = args.api_secret
        
        # Default values
        rsi_period = 14
        overbought_threshold = 65
        oversold_threshold = 35
        use_momentum = True
        max_position_size = 0.3
        max_daily_loss = 0.03
        max_drawdown = 0.10
        stop_loss_pct = 0.015
        take_profit_pct = 0.03
        trailing_stop = False
        trailing_stop_pct = 0.01

    # Check for API credentials
    if not api_key or not api_secret:
        print("Error: API key and secret are required for live trading")
        print("Provide them via --api-key and --api-secret or config file")
        return

    run_live(
        symbol=symbol,
        timeframe=timeframe,
        interval=interval,
        initial_capital=initial_capital,
        api_key=api_key,
        api_secret=api_secret,
        # Strategy parameters
        rsi_period=rsi_period,
        overbought_threshold=overbought_threshold,
        oversold_threshold=oversold_threshold,
        use_momentum=use_momentum,
        # Risk parameters
        max_position_size=max_position_size,
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop=trailing_stop,
        trailing_stop_pct=trailing_stop_pct
    )


if __name__ == "__main__":
    main()
