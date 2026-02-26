"""Risk Management Service"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta


class RiskManagement:
    """
    Risk management service for controlling trading risk.
    """

    def __init__(
        self,
        max_position_size: float = 1.0,  # Max % of capital per position
        max_daily_loss: float = 0.05,    # Max daily loss % (5%)
        max_drawdown: float = 0.15,      # Max drawdown % (15%)
        max_leverage: float = 1.0,       # Max leverage (1x = no leverage)
        stop_loss_pct: float = 0.02,      # Stop loss % (2%)
        take_profit_pct: float = 0.05,    # Take profit % (5%)
        trailing_stop: bool = False,
        trailing_stop_pct: float = 0.015  # Trailing stop % (1.5%)
    ):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop = trailing_stop
        self.trailing_stop_pct = trailing_stop_pct

        # State tracking
        self.daily_pnl = 0.0
        self.daily_starting_capital = 0.0
        self.peak_capital = 0.0
        self.current_drawdown = 0.0
        self.last_reset_date = datetime.now().date()

        # Trailing stop state
        self.highest_price = 0.0
        self.stop_loss_triggered = False

    def check_signal(
        self,
        signal,
        position: float,
        capital: float
    ):
        """
        Check if signal passes risk management rules.

        Args:
            signal: TradingSignal object
            position: Current position size
            capital: Current capital

        Returns:
            Modified signal or None to reject
        """
        # Check if daily loss limit reached
        if self._is_daily_loss_limit_reached(capital):
            signal.action = 'hold'
            signal.reason += ' | Daily loss limit reached'
            return signal

        # Check drawdown
        if self._is_drawdown_limit_reached(capital):
            signal.action = 'hold'
            signal.reason += ' | Max drawdown limit reached'
            return signal

        # Check position size limit
        if signal.action == 'buy':
            position_value = signal.price * (signal.quantity or 1)
            if position_value > capital * self.max_position_size:
                # Reduce position size
                max_value = capital * self.max_position_size
                signal.quantity = max_value / signal.price
                signal.reason += ' | Position size reduced'

        # Apply stop loss to existing position
        if position > 0 and signal.price > 0:
            # Check if stop loss triggered
            if self._is_stop_loss_triggered(signal.price, position, capital):
                signal.action = 'sell'
                signal.reason = 'Stop loss triggered'
                self.stop_loss_triggered = True
                return signal

            # Check if take profit triggered
            if self._is_take_profit_triggered(signal.price, position, capital):
                signal.action = 'sell'
                signal.reason = 'Take profit triggered'
                return signal

        return signal

    def _is_daily_loss_limit_reached(self, capital: float) -> bool:
        """Check if daily loss limit is reached."""
        today = datetime.now().date()

        # Reset daily tracking if new day
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_starting_capital = capital
            self.last_reset_date = today

        if self.daily_starting_capital == 0:
            self.daily_starting_capital = capital

        daily_loss = (self.daily_starting_capital - capital) / self.daily_starting_capital

        return daily_loss >= self.max_daily_loss

    def _is_drawdown_limit_reached(self, capital: float) -> bool:
        """Check if max drawdown limit is reached."""
        # Update peak capital
        if capital > self.peak_capital:
            self.peak_capital = capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_capital - capital) / self.peak_capital

        return self.current_drawdown >= self.max_drawdown

    def _is_stop_loss_triggered(
        self,
        current_price: float,
        position: float,
        capital: float
    ) -> bool:
        """Check if stop loss is triggered."""
        # For long positions
        if position > 0:
            # Update highest price for trailing stop
            if self.trailing_stop:
                if current_price > self.highest_price:
                    self.highest_price = current_price
                    return False

                stop_price = self.highest_price * (1 - self.trailing_stop_pct)
                if current_price <= stop_price:
                    return True
            else:
                # Fixed stop loss
                if hasattr(self, 'entry_price') and self.entry_price > 0:
                    if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                        return True

        return False

    def _is_take_profit_triggered(
        self,
        current_price: float,
        position: float,
        capital: float
    ) -> bool:
        """Check if take profit is triggered."""
        if position > 0 and hasattr(self, 'entry_price') and self.entry_price > 0:
            if current_price >= self.entry_price * (1 + self.take_profit_pct):
                return True

        return False

    def update_position(self, entry_price: float, position_size: float) -> None:
        """Update position tracking."""
        self.entry_price = entry_price
        self.position_size = position_size
        self.highest_price = entry_price
        self.stop_loss_triggered = False

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        risk_per_trade: float = 0.02  # 2% risk per trade
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            capital: Available capital
            entry_price: Entry price
            risk_per_trade: Risk percentage per trade

        Returns:
            Position size (number of units)
        """
        risk_amount = capital * risk_per_trade
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)

        # Position size = risk amount / (entry price - stop loss)
        position_value = risk_amount / (1 - self.stop_loss_pct)

        # Ensure within max position size
        max_position_value = capital * self.max_position_size
        position_value = min(position_value, max_position_value)

        # Account for leverage
        position_value = min(position_value, capital * self.max_leverage)

        return position_value / entry_price

    def get_risk_metrics(self, capital: float) -> Dict:
        """Get current risk metrics."""
        return {
            'daily_pnl_pct': (self.daily_pnl / self.daily_starting_capital * 100) if self.daily_starting_capital > 0 else 0,
            'daily_loss_limit': self.max_daily_loss * 100,
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_drawdown_limit': self.max_drawdown * 100,
            'position_limit_pct': self.max_position_size * 100,
            'leverage_limit': self.max_leverage,
            'stop_loss_pct': self.stop_loss_pct * 100,
            'take_profit_pct': self.take_profit_pct * 100,
            'is_trading_allowed': not self._is_daily_loss_limit_reached(capital) and
                                  not self._is_drawdown_limit_reached(capital)
        }

    def reset_daily(self, capital: float) -> None:
        """Reset daily tracking."""
        self.daily_pnl = 0.0
        self.daily_starting_capital = capital
        self.last_reset_date = datetime.now().date()

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily PnL."""
        self.daily_pnl += pnl
