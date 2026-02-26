"""Base Strategy Class"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class Signal:
    """Trading signal representation."""

    def __init__(
        self,
        action: str,  # 'buy', 'sell', 'hold'
        confidence: float = 0.0,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        reason: str = ""
    ):
        self.action = action
        self.confidence = confidence
        self.price = price
        self.quantity = quantity
        self.reason = reason
        self.timestamp = pd.Timestamp.now()

    def __repr__(self):
        return f"Signal({self.action}, confidence={self.confidence:.2f}, price={self.price})"


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(
        self,
        name: str = "BaseStrategy",
        initial_capital: float = 100000,
        max_position_size: float = 1.0,  # 100% of capital
        commission: float = 0.001  # 0.1%
    ):
        self.name = name
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission = commission
        self.position = 0.0
        self.capital = initial_capital
        self.trades: List[Dict] = []
        self.signals: List[Signal] = []

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        indicators: Optional[Dict] = None
    ) -> Signal:
        """
        Generate trading signal based on market data.

        Args:
            data: DataFrame with OHLCV data
            indicators: Pre-calculated technical indicators

        Returns:
            Signal object with action and confidence
        """
        pass

    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float
    ) -> float:
        """Calculate position size based on signal and risk management."""
        if signal.action == 'hold' or signal.confidence < 0.5:
            return 0.0

        # Calculate max position based on capital and confidence
        available_capital = self.capital * self.max_position_size
        position_size = available_capital * signal.confidence

        # Account for commission
        position_size = position_size / (1 + self.commission)

        return position_size

    def execute_signal(
        self,
        signal: Signal,
        current_price: float
    ) -> Dict:
        """Execute a trading signal."""
        if signal.action == 'hold':
            return {'executed': False, 'reason': 'hold signal'}

        position_size = self.calculate_position_size(signal, current_price)

        if position_size <= 0:
            return {'executed': False, 'reason': 'zero position size'}

        trade = {
            'timestamp': signal.timestamp,
            'action': signal.action,
            'price': current_price,
            'quantity': position_size / current_price,
            'value': position_size,
            'commission': position_size * self.commission,
            'reason': signal.reason,
            'confidence': signal.confidence
        }

        # Update capital and position
        if signal.action == 'buy':
            self.capital -= position_size + position_size * self.commission
            self.position += position_size / current_price
            trade['position_after'] = self.position
            trade['capital_after'] = self.capital

        elif signal.action == 'sell':
            if self.position >= position_size / current_price:
                self.capital += position_size - position_size * self.commission
                self.position -= position_size / current_price
                trade['position_after'] = self.position
                trade['capital_after'] = self.capital
            else:
                return {'executed': False, 'reason': 'insufficient position'}

        self.trades.append(trade)
        return {'executed': True, 'trade': trade}

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        return self.capital + self.position * current_price

    def get_performance(self) -> Dict:
        """Calculate strategy performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

        returns = []
        for i in range(1, len(self.trades)):
            if self.trades[i]['action'] == 'sell' and self.trades[i-1]['action'] == 'buy':
                trade_return = (self.trades[i]['price'] - self.trades[i-1]['price']) / self.trades[i-1]['price']
                returns.append(trade_return)

        if not returns:
            return {
                'total_trades': len(self.trades),
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

        returns = np.array(returns)

        return {
            'total_trades': len(self.trades),
            'winning_trades': sum(returns > 0),
            'losing_trades': sum(returns < 0),
            'win_rate': sum(returns > 0) / len(returns) if len(returns) > 0 else 0,
            'total_return': np.sum(returns),
            'avg_return': np.mean(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'volatility': np.std(returns) * np.sqrt(252)
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def reset(self) -> None:
        """Reset strategy state."""
        self.position = 0.0
        self.capital = self.initial_capital
        self.trades = []
        self.signals = []
