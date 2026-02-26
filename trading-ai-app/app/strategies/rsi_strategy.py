"""RSI Trading Strategy"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .base import BaseStrategy, Signal


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) based trading strategy.

    Buy when RSI is oversold (< oversold_threshold)
    Sell when RSI is overbought (> overbought_threshold)
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        rsi_period: int = 14,
        overbought_threshold: float = 70,
        oversold_threshold: float = 30,
        max_position_size: float = 1.0,
        commission: float = 0.001,
        use_divergence: bool = False,
        use_momentum: bool = True
    ):
        super().__init__(
            name="RSIStrategy",
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            commission=commission
        )

        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.use_divergence = use_divergence
        self.use_momentum = use_momentum
        self.last_rsi = None

    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator."""
        if period is None:
            period = self.rsi_period

        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use exponential moving average
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_momentum(
        self,
        prices: pd.Series,
        fast_period: int = 5,
        slow_period: int = 20
    ) -> pd.Series:
        """Calculate momentum indicator."""
        fast_ma = prices.ewm(span=fast_period).mean()
        slow_ma = prices.ewm(span=slow_period).mean()
        return fast_ma - slow_ma

    def detect_divergence(
        self,
        prices: pd.Series,
        rsi: pd.Series,
        lookback: int = 5
    ) -> Optional[str]:
        """Detect bullish or bearish divergence."""
        if len(prices) < lookback + 1:
            return None

        # Get last 'lookback' prices and RSI
        price_diff = prices.iloc[-1] - prices.iloc[-lookback]
        rsi_diff = rsi.iloc[-1] - rsi.iloc[-lookback]

        # Bullish divergence: price makes lower low, RSI makes higher low
        if price_diff < 0 and rsi_diff > 0:
            return 'bullish'

        # Bearish divergence: price makes higher high, RSI makes lower high
        if price_diff > 0 and rsi_diff < 0:
            return 'bearish'

        return None

    def generate_signal(
        self,
        data: pd.DataFrame,
        indicators: Optional[Dict] = None
    ) -> Signal:
        """
        Generate trading signal based on RSI strategy.

        Args:
            data: DataFrame with OHLCV data (must have 'close' column)
            indicators: Optional pre-calculated indicators

        Returns:
            Signal object
        """
        # Ensure we have enough data
        if len(data) < self.rsi_period + 1:
            return Signal('hold', confidence=0.0, reason='insufficient data')

        prices = data['close']

        # Calculate RSI
        if indicators and 'rsi' in indicators:
            rsi = indicators['rsi']
        else:
            rsi = self.calculate_rsi(prices)

        current_rsi = rsi.iloc[-1]
        current_price = prices.iloc[-1]

        # Calculate momentum if enabled
        momentum = None
        if self.use_momentum:
            momentum = self.calculate_momentum(prices).iloc[-1]

        # Detect divergence if enabled
        divergence = None
        if self.use_divergence:
            divergence = self.detect_divergence(prices, rsi)

        # Generate signal based on RSI levels
        signal = self._generate_rsi_signal(current_rsi, momentum, divergence)

        # Add price and reason
        signal.price = current_price
        signal.reason = f"RSI: {current_rsi:.2f}"

        if momentum is not None:
            signal.reason += f", Momentum: {momentum:.2f}"

        if divergence:
            signal.reason += f", Divergence: {divergence}"

        self.last_rsi = current_rsi
        self.signals.append(signal)

        return signal

    def _generate_rsi_signal(
        self,
        rsi: float,
        momentum: Optional[float] = None,
        divergence: Optional[str] = None
    ) -> Signal:
        """Generate signal based on RSI, momentum, and divergence."""

        # Strong buy signal: Oversold + bullish divergence
        if rsi < self.oversold_threshold:
            confidence = (self.oversold_threshold - rsi) / self.oversold_threshold

            if divergence == 'bullish':
                confidence = min(1.0, confidence + 0.2)
                return Signal(
                    'buy',
                    confidence=confidence,
                    reason='Oversold with bullish divergence'
                )

            # Check momentum for confirmation
            if momentum is not None and momentum > 0:
                confidence = min(1.0, confidence + 0.1)
                return Signal(
                    'buy',
                    confidence=confidence,
                    reason='Oversold with positive momentum'
                )

            return Signal(
                'buy',
                confidence=confidence,
                reason='Oversold'
            )

        # Strong sell signal: Overbought + bearish divergence
        elif rsi > self.overbought_threshold:
            confidence = (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)

            if divergence == 'bearish':
                confidence = min(1.0, confidence + 0.2)
                return Signal(
                    'sell',
                    confidence=confidence,
                    reason='Overbought with bearish divergence'
                )

            # Check momentum for confirmation
            if momentum is not None and momentum < 0:
                confidence = min(1.0, confidence + 0.1)
                return Signal(
                    'sell',
                    confidence=confidence,
                    reason='Overbought with negative momentum'
                )

            return Signal(
                'sell',
                confidence=confidence,
                reason='Overbought'
            )

        # RSI in neutral zone - check for momentum reversal
        elif rsi < 40 and momentum is not None and momentum > 0:
            # Approaching oversold with positive momentum
            return Signal(
                'buy',
                confidence=0.4,
                reason='RSI approaching oversold with positive momentum'
            )

        elif rsi > 60 and momentum is not None and momentum < 0:
            # Approaching overbought with negative momentum
            return Signal(
                'sell',
                confidence=0.4,
                reason='RSI approaching overbought with negative momentum'
            )

        # Hold signal
        return Signal(
            'hold',
            confidence=0.5,
            reason='RSI in neutral zone'
        )

    def get_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all indicators used by this strategy."""
        prices = data['close']

        return {
            'rsi': self.calculate_rsi(prices),
            'momentum': self.calculate_momentum(prices) if self.use_momentum else None
        }
