"""Backtesting Engine"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime
import os


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []

    def run(
        self,
        data: pd.DataFrame,
        strategy,
        verbose: int = 1
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data
            strategy: Trading strategy instance
            verbose: Verbosity level

        Returns:
            Dictionary with backtest results
        """
        if len(data) == 0:
            return {'error': 'No data provided'}

        # Reset strategy state
        strategy.reset()

        # Initialize tracking variables
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0

        # Track equity curve
        self.equity_curve = []
        self.trades = []

        # Pre-calculate indicators if strategy supports it
        indicators = None
        if hasattr(strategy, 'get_indicators'):
            indicators = strategy.get_indicators(data)

        # Iterate through data
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_price = data['close'].iloc[i]

            # Get indicators for current point
            current_indicators = None
            if indicators is not None:
                current_indicators = {
                    k: v.iloc[:i+1] if hasattr(v, 'iloc') else v
                    for k, v in indicators.items()
                }

            # Generate signal
            signal = strategy.generate_signal(current_data, current_indicators)

            # Execute signal
            if signal.action == 'buy' and position == 0:
                # Buy signal and no current position
                buy_price = current_price * (1 + self.slippage)
                position_size = strategy.calculate_position_size(signal, buy_price)

                if position_size > 0 and capital >= position_size * (1 + self.commission):
                    cost = position_size * (1 + self.commission)
                    capital -= cost
                    position = position_size / buy_price
                    entry_price = buy_price

                    self.trades.append({
                        'index': i,
                        'date': data.index[i] if hasattr(data.index, '__getitem__') else i,
                        'action': 'buy',
                        'price': buy_price,
                        'quantity': position,
                        'value': position_size,
                        'capital_after': capital
                    })

            elif signal.action == 'sell' and position > 0:
                # Sell signal and have position
                sell_price = current_price * (1 - self.slippage)
                proceeds = position * sell_price * (1 - self.commission)

                self.trades.append({
                    'index': i,
                    'date': data.index[i] if hasattr(data.index, '__getitem__') else i,
                    'action': 'sell',
                    'price': sell_price,
                    'quantity': position,
                    'value': position * sell_price,
                    'profit': proceeds - (position * entry_price) - (position * entry_price * self.commission),
                    'capital_after': capital + proceeds
                })

                capital += proceeds
                position = 0.0
                entry_price = 0.0

            # Update equity curve
            equity = capital + position * current_price
            self.equity_curve.append(equity)

        # Calculate final metrics
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital

        # Calculate daily returns
        if len(self.equity_curve) > 1:
            self.daily_returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        else:
            self.daily_returns = []

        # Get strategy performance
        performance = strategy.get_performance()

        results = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': (final_equity - self.initial_capital) / self.initial_capital,
            'total_trades': len([t for t in self.trades if t['action'] == 'buy']),
            'winning_trades': performance.get('winning_trades', 0),
            'losing_trades': performance.get('losing_trades', 0),
            'win_rate': performance.get('win_rate', 0),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'profit_factor': self._calculate_profit_factor(),
            'avg_trade': performance.get('avg_return', 0),
            'volatility': np.std(self.daily_returns) * np.sqrt(252) if len(self.daily_returns) > 0 else 0,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns
        }

        if verbose:
            self._print_results(results)

        return results

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(self.daily_returns) == 0:
            return 0.0

        excess_returns = np.array(self.daily_returns) - risk_free_rate / 252
        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(self.daily_returns) == 0:
            return 0.0

        excess_returns = np.array(self.daily_returns) - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.equity_curve) == 0:
            return 0.0

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        return np.min(drawdown)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        sells = [t.get('profit', 0) for t in self.trades if t['action'] == 'sell']

        if not sells:
            return 0.0

        gross_profit = sum(p for p in sells if p > 0)
        gross_loss = abs(sum(p for p in sells if p < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _print_results(self, results: Dict) -> None:
        """Print backtest results."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
        print(f"Final Equity:       ${results['final_equity']:,.2f}")
        print(f"Total Return:       {results['total_return']*100:.2f}%")
        print("-" * 50)
        print(f"Total Trades:       {results['total_trades']}")
        print(f"Winning Trades:     {results['winning_trades']}")
        print(f"Losing Trades:      {results['losing_trades']}")
        print(f"Win Rate:           {results['win_rate']*100:.2f}%")
        print("-" * 50)
        print(f"Sharpe Ratio:       {results['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio:      {results['sortino_ratio']:.4f}")
        print(f"Max Drawdown:       {results['max_drawdown']*100:.2f}%")
        print(f"Profit Factor:      {results['profit_factor']:.4f}")
        print(f"Volatility:         {results['volatility']*100:.2f}%")
        print("=" * 50 + "\n")

    def save_results(self, results: Dict, path: str) -> None:
        """Save backtest results to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save summary as text
        with open(path + '.txt', 'w') as f:
            f.write("BACKTEST RESULTS\n")
            f.write("=" * 50 + "\n")
            for key, value in results.items():
                if key not in ['trades', 'equity_curve', 'daily_returns']:
                    f.write(f"{key}: {value}\n")

        # Save trades as CSV
        if results.get('trades'):
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(path + '_trades.csv', index=False)

        # Save equity curve as CSV
        if results.get('equity_curve'):
            equity_df = pd.DataFrame({
                'equity': results['equity_curve'],
                'daily_return': [0.0] + list(results.get('daily_returns', []))
            })
            equity_df.to_csv(path + '_equity.csv', index=False)

        print(f"Results saved to {path}")
