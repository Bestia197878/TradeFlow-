"""Live Trading Execution"""

import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os


class LiveTrader:
    """
    Live trading execution engine.
    """

    def __init__(
        self,
        exchange,
        strategy,
        config: Dict,
        risk_manager=None,
        paper_trading: bool = True
    ):
        """
        Initialize live trader.

        Args:
            exchange: Exchange service instance
            strategy: Trading strategy instance
            config: Configuration dictionary
            risk_manager: Risk management instance
            paper_trading: If True, simulate trades without real money
        """
        self.exchange = exchange
        self.strategy = strategy
        self.config = config
        self.risk_manager = risk_manager
        self.paper_trading = paper_trading

        # Trading state
        self.is_running = False
        self.position = 0.0
        self.capital = config.get('initial_capital', 100000)
        self.trades: List[Dict] = []
        self.order_history: List[Dict] = []

        # Performance tracking
        self.equity_curve: List[float] = []
        self.daily_pnl: float = 0.0

        # Callbacks
        self.on_trade: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_signal: Optional[Callable] = None

    def start(self) -> None:
        """Start live trading."""
        if self.is_running:
            print("Trader is already running")
            return

        self.is_running = True
        print("Starting live trader...")

        # Get current market data
        symbol = self.config.get('symbol', 'BTC/USDT')
        timeframe = self.config.get('timeframe', '1h')

        # Reset strategy
        self.strategy.reset()

        # Start trading loop in background thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()

    def stop(self) -> None:
        """Stop live trading."""
        if not self.is_running:
            return

        self.is_running = False
        print("Stopping live trader...")

        if hasattr(self, 'trading_thread'):
            self.trading_thread.join(timeout=10)

    def _trading_loop(self) -> None:
        """Main trading loop."""
        symbol = self.config.get('symbol', 'BTC/USDT')
        timeframe = self.config.get('timeframe', '1h')
        interval = self.config.get('interval', 60)  # seconds

        while self.is_running:
            try:
                # Fetch current market data
                data = self.exchange.fetch_ohlcv(symbol, timeframe)

                if data is not None and len(data) > 0:
                    # Get current price
                    current_price = data['close'].iloc[-1]

                    # Generate signal
                    signal = self.strategy.generate_signal(data)

                    # Log signal
                    if self.on_signal:
                        self.on_signal(signal)

                    # Check risk limits
                    if self.risk_manager:
                        signal = self.risk_manager.check_signal(signal, self.position, self.capital)

                    # Execute if signal is not hold
                    if signal.action != 'hold':
                        self._execute_trade(signal, current_price)

                    # Update equity
                    equity = self.capital + self.position * current_price
                    self.equity_curve.append(equity)

                # Wait for next iteration
                time.sleep(interval)

            except Exception as e:
                print(f"Error in trading loop: {e}")
                if self.on_error:
                    self.on_error(e)
                time.sleep(5)  # Wait before retrying

    def _execute_trade(self, signal, current_price: float) -> Dict:
        """Execute a trade."""
        symbol = self.config.get('symbol', 'BTC/USDT')

        # Calculate position size
        position_size = self.strategy.calculate_position_size(signal, current_price)

        if position_size <= 0:
            return {'executed': False, 'reason': 'position size too small'}

        order = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': signal.action,
            'price': current_price,
            'quantity': position_size / current_price,
            'value': position_size,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'status': 'pending'
        }

        if self.paper_trading:
            # Paper trading - simulate execution
            order['status'] = 'filled'
            order['fill_price'] = current_price
            order['commission'] = position_size * 0.001

            if signal.action == 'buy':
                if self.capital >= position_size * 1.001:
                    self.capital -= position_size * 1.001
                    self.position += position_size / current_price
                else:
                    order['status'] = 'rejected'
                    order['reason'] = 'insufficient capital'

            elif signal.action == 'sell':
                if self.position >= position_size / current_price:
                    self.capital += position_size * 0.999
                    self.position -= position_size / current_price
                else:
                    order['status'] = 'rejected'
                    order['reason'] = 'insufficient position'

        else:
            # Real trading - execute on exchange
            try:
                if signal.action == 'buy':
                    result = self.exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side='buy',
                        amount=position_size / current_price
                    )
                else:
                    result = self.exchange.create_order(
                        symbol=symbol,
                        order_type='market',
                        side='sell',
                        amount=position_size / current_price
                    )

                order['status'] = 'filled'
                order['order_id'] = result.get('id')

            except Exception as e:
                order['status'] = 'failed'
                order['error'] = str(e)

        # Record trade
        self.trades.append(order)
        self.order_history.append(order)

        # Update risk manager with new position
        if self.risk_manager and order['status'] == 'filled':
            self.risk_manager.update_position(
                entry_price=order['fill_price'],
                position_size=order['quantity']
            )

        # Trigger callback
        if self.on_trade:
            self.on_trade(order)

        return order

    def get_status(self) -> Dict:
        """Get current trading status."""
        return {
            'is_running': self.is_running,
            'position': self.position,
            'capital': self.capital,
            'total_trades': len(self.trades),
            'equity_curve_length': len(self.equity_curve),
            'last_equity': self.equity_curve[-1] if self.equity_curve else self.capital
        }

    def get_performance(self) -> Dict:
        """Get performance metrics."""
        if not self.equity_curve:
            return {
                'total_pnl': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0
            }

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        winning_trades = len([t for t in self.trades if t.get('action') == 'sell' and t.get('value', 0) > 0])
        total_exits = len([t for t in self.trades if t.get('action') == 'sell'])

        return {
            'initial_capital': self.capital,
            'current_equity': equity[-1],
            'total_pnl': equity[-1] - equity[0],
            'total_return': (equity[-1] - equity[0]) / equity[0],
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_exits if total_exits > 0 else 0.0,
            'avg_return': np.mean(returns) if len(returns) > 0 else 0.0,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 1 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(equity)
        }

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(equity) == 0:
            return 0.0

        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return np.min(drawdown)

    def save_state(self, path: str) -> None:
        """Save trading state to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'position': self.position,
            'capital': self.capital,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'timestamp': datetime.now().isoformat()
        }
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        """Load trading state from file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.position = state['position']
            self.capital = state['capital']
            self.trades = state['trades']
            self.equity_curve = state['equity_curve']
