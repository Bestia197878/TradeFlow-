"""AI Ensemble Trading Strategy

Combines RSI strategy with AI models (Random Forest, LSTM, RL Agent)
for enhanced signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import os
import pickle

from .base import BaseStrategy, Signal


class AIEnsembleStrategy(BaseStrategy):
    """
    AI Ensemble Strategy that combines:
    - RSI-based signals
    - Random Forest predictions
    - LSTM price predictions
    - RL Agent signals
    
    The final signal is a weighted combination of all models.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        # RSI parameters
        rsi_period: int = 14,
        overbought_threshold: float = 65,
        oversold_threshold: float = 35,
        # Model weights (must sum to 1.0)
        rsi_weight: float = 0.3,
        rf_weight: float = 0.3,
        lstm_weight: float = 0.2,
        rl_weight: float = 0.2,
        # Risk parameters
        max_position_size: float = 1.0,
        commission: float = 0.001,
        use_momentum: bool = True,
        # Model paths
        model_dir: str = "models"
    ):
        super().__init__(
            name="AIEnsemble",
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            commission=commission
        )

        # RSI parameters
        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.use_momentum = use_momentum

        # Model weights
        self.rsi_weight = rsi_weight
        self.rf_weight = rf_weight
        self.lstm_weight = lstm_weight
        self.rl_weight = rl_weight
        
        # Validate weights
        total = rsi_weight + rf_weight + lstm_weight + rl_weight
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.rsi_weight /= total
            self.rf_weight /= total
            self.lstm_weight /= total
            self.rl_weight /= total

        # AI Models
        self.rf_model = None
        self.lstm_model = None
        self.rl_agent = None
        self.sentiment_model = None
        
        # Model directory
        self.model_dir = model_dir
        
        # Track if models are loaded
        self.models_loaded = False
        
        # Try to load models
        self._load_models()
        
        # Last predictions for display
        self.last_rsi_signal = None
        self.last_rf_prediction = None
        self.last_lstm_prediction = None
        self.last_rl_action = None
        self.last_sentiment = None

    def _load_models(self):
        """Load trained AI models if available."""
        model_dir = self.model_dir
        
        # Try to load Random Forest
        rf_path = os.path.join(model_dir, "rf_model.pkl")
        if os.path.exists(rf_path):
            try:
                from app.ai_models.rf_model import RandomForestModel
                self.rf_model = RandomForestModel()
                self.rf_model.load(rf_path)
                print(f"Loaded Random Forest model from {rf_path}")
            except Exception as e:
                print(f"Failed to load RF model: {e}")
                self.rf_model = None
        
        # Try to load LSTM
        lstm_path = os.path.join(model_dir, "lstm_model.pkl")
        if os.path.exists(lstm_path):
            try:
                from app.ai_models.lstm_model import LSTMModel
                self.lstm_model = LSTMModel()
                self.lstm_model.load(lstm_path)
                print(f"Loaded LSTM model from {lstm_path}")
            except Exception as e:
                print(f"Failed to load LSTM model: {e}")
                self.lstm_model = None
        
        # Try to load RL Agent
        rl_path = os.path.join(model_dir, "rl_agent.pkl")
        if os.path.exists(rl_path):
            try:
                from app.ai_models.rl_agent import RLAgent
                self.rl_agent = RLAgent()
                self.rl_agent.load(rl_path)
                print(f"Loaded RL Agent from {rl_path}")
            except Exception as e:
                print(f"Failed to load RL agent: {e}")
                self.rl_agent = None
        
        # Try to load Sentiment model
        sentiment_path = os.path.join(model_dir, "sentiment_model.pkl")
        if os.path.exists(sentiment_path):
            try:
                from app.ai_models.llm_sentiment import LLMSentiment
                self.sentiment_model = LLMSentiment()
                self.sentiment_model.load(sentiment_path)
                print(f"Loaded Sentiment model from {sentiment_path}")
            except Exception as e:
                print(f"Failed to load Sentiment model: {e}")
                self.sentiment_model = None
        
        self.models_loaded = (
            self.rf_model is not None or 
            self.lstm_model is not None or 
            self.rl_agent is not None
        )

    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator."""
        if period is None:
            period = self.rsi_period

        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

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

    def _get_rsi_signal(self, rsi: float, momentum: Optional[float] = None) -> tuple:
        """Get RSI-based signal. Returns (action, confidence)."""
        if rsi < self.oversold_threshold:
            distance = self.oversold_threshold - rsi
            confidence = min(0.95, 0.5 + distance / 20)
            if momentum is not None and momentum > 0:
                confidence = min(0.95, confidence + 0.1)
            return ('buy', confidence)
        elif rsi > self.overbought_threshold:
            distance = rsi - self.overbought_threshold
            confidence = min(0.95, 0.5 + distance / 20)
            if momentum is not None and momentum < 0:
                confidence = min(0.95, confidence + 0.1)
            return ('sell', confidence)
        elif rsi < 40 and momentum is not None and momentum > 0:
            return ('buy', 0.6)
        elif rsi > 60 and momentum is not None and momentum < 0:
            return ('sell', 0.6)
        else:
            return ('hold', 0.5)

    def _get_rf_signal(self, data: pd.DataFrame) -> tuple:
        """Get Random Forest signal. Returns (action, confidence)."""
        if self.rf_model is None or not self.rf_model.is_trained:
            return ('hold', 0.5)
        
        try:
            prices = data['close'].values
            prediction, confidence = self.rf_model.predict(prices)
            
            if prediction == 1:  # Price going up
                return ('buy', confidence)
            else:  # Price going down
                return ('sell', confidence)
        except Exception as e:
            print(f"RF prediction error: {e}")
            return ('hold', 0.5)

    def _get_lstm_signal(self, data: pd.DataFrame) -> tuple:
        """Get LSTM signal based on price prediction. Returns (action, confidence)."""
        if self.lstm_model is None or not self.lstm_model.is_trained:
            return ('hold', 0.5)
        
        try:
            prices = data['close'].values
            predicted_price = self.lstm_model.predict(prices)[0][0]
            current_price = prices[-1]
            
            change_pct = (predicted_price - current_price) / current_price
            
            # Determine action based on predicted change
            if change_pct > 0.005:  # > 0.5% expected increase
                confidence = min(0.9, 0.5 + abs(change_pct) * 10)
                return ('buy', confidence)
            elif change_pct < -0.005:  # > 0.5% expected decrease
                confidence = min(0.9, 0.5 + abs(change_pct) * 10)
                return ('sell', confidence)
            else:
                return ('hold', 0.5)
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return ('hold', 0.5)

    def _get_rl_signal(self, data: pd.DataFrame) -> tuple:
        """Get RL Agent signal. Returns (action, confidence)."""
        if self.rl_agent is None:
            return ('hold', 0.5)
        
        try:
            prices = data['close'].values
            
            # Create state from recent prices
            if len(prices) >= 20:
                state = self._create_rl_state(prices)
                action = self.rl_agent.act(state, epsilon=0.0)  # No exploration
                
                # Map action to signal
                action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
                action_name = action_map.get(action, 'hold')
                
                return (action_name, 0.7)  # Fixed confidence for RL
            else:
                return ('hold', 0.5)
        except Exception as e:
            print(f"RL prediction error: {e}")
            return ('hold', 0.5)

    def _create_rl_state(self, prices: np.ndarray) -> np.ndarray:
        """Create state vector for RL agent."""
        # Simple state: returns, volatility, position features
        returns = np.diff(prices) / prices[:-1]
        
        # Features
        recent_return = returns[-1] if len(returns) > 0 else 0
        volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        
        # RSI
        rsi = self.calculate_rsi(pd.Series(prices)).iloc[-1]
        
        # MACD
        ema_12 = pd.Series(prices).ewm(span=12).mean()
        ema_26 = pd.Series(prices).ewm(span=26).mean()
        macd = (ema_12 - ema_26).iloc[-1]
        
        state = np.array([
            recent_return,
            volatility,
            momentum,
            rsi / 100.0,  # Normalize RSI
            macd / prices[-1]  # Normalize MACD
        ])
        
        return state

    def _combine_signals(
        self,
        rsi_signal: tuple,
        rf_signal: tuple,
        lstm_signal: tuple,
        rl_signal: tuple,
        sentiment: Optional[float] = None
    ) -> Signal:
        """Combine all signals using weighted voting."""
        
        # Collect all votes
        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        
        # Add RSI votes
        action, confidence = rsi_signal
        votes[action] += confidence * self.rsi_weight
        
        # Add RF votes
        action, confidence = rf_signal
        votes[action] += confidence * self.rf_weight
        
        # Add LSTM votes
        action, confidence = lstm_signal
        votes[action] += confidence * self.lstm_weight
        
        # Add RL votes
        action, confidence = rl_signal
        votes[action] += confidence * self.rl_weight
        
        # Add sentiment bias if available
        if sentiment is not None:
            # Positive sentiment adds buy weight
            # Negative sentiment adds sell weight
            sentiment_bias = abs(sentiment) * 0.1  # Small bias
            if sentiment > 0:
                votes['buy'] += sentiment_bias
            else:
                votes['sell'] += sentiment_bias
        
        # Determine final action
        max_vote = max(votes.values())
        final_action = max(votes, key=votes.get)
        
        # Calculate confidence
        total_votes = sum(votes.values())
        final_confidence = max_vote / total_votes if total_votes > 0 else 0.5
        
        # Generate reason
        reasons = []
        if rsi_signal[0] != 'hold':
            reasons.append(f"RSI({rsi_signal[0]}:{rsi_signal[1]:.2f})")
        if rf_signal[0] != 'hold':
            reasons.append(f"RF({rf_signal[0]}:{rf_signal[1]:.2f})")
        if lstm_signal[0] != 'hold':
            reasons.append(f"LSTM({lstm_signal[0]}:{lstm_signal[1]:.2f})")
        if rl_signal[0] != 'hold':
            reasons.append(f"RL({rl_signal[0]}:{rl_signal[1]:.2f})")
        if sentiment is not None:
            reasons.append(f"Sentiment({sentiment:+.2f})")
        
        reason = " | ".join(reasons) if reasons else "No clear signal"
        
        return Signal(
            action=final_action,
            confidence=final_confidence,
            reason=reason
        )

    def generate_signal(
        self,
        data: pd.DataFrame,
        indicators: Optional[Dict] = None
    ) -> Signal:
        """Generate trading signal using AI ensemble."""
        
        # Ensure we have enough data
        if len(data) < self.rsi_period + 1:
            return Signal('hold', confidence=0.0, reason='insufficient data')
        
        prices = data['close']
        current_price = prices.iloc[-1]
        
        # Calculate RSI
        if indicators and 'rsi' in indicators:
            rsi = indicators['rsi']
        else:
            rsi = self.calculate_rsi(prices)
        
        current_rsi = rsi.iloc[-1]
        
        # Calculate momentum
        momentum = None
        if self.use_momentum:
            momentum = self.calculate_momentum(prices).iloc[-1]
        
        # Get individual signals
        rsi_signal = self._get_rsi_signal(current_rsi, momentum)
        rf_signal = self._get_rf_signal(data)
        lstm_signal = self._get_lstm_signal(data)
        rl_signal = self._get_rl_signal(data)
        
        # Store for display
        self.last_rsi_signal = rsi_signal
        self.last_rf_prediction = rf_signal
        self.last_lstm_prediction = lstm_signal
        self.last_rl_action = rl_signal
        
        # Get sentiment if available
        sentiment = None
        if self.sentiment_model is not None:
            try:
                result = self.sentiment_model.analyze("BTC crypto market news")
                sentiment = result.get('sentiment', 0)
                self.last_sentiment = sentiment
            except:
                pass
        
        # Combine signals
        signal = self._combine_signals(
            rsi_signal,
            rf_signal,
            lstm_signal,
            rl_signal,
            sentiment
        )
        
        # Add price and additional info
        signal.price = current_price
        signal.reason += f" | RSI: {current_rsi:.2f}"
        
        self.signals.append(signal)
        return signal

    def get_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all indicators used by this strategy."""
        prices = data['close']

        return {
            'rsi': self.calculate_rsi(prices),
            'momentum': self.calculate_momentum(prices) if self.use_momentum else None
        }

    def get_model_status(self) -> Dict:
        """Get status of loaded AI models."""
        return {
            'rf_loaded': self.rf_model is not None,
            'lstm_loaded': self.lstm_model is not None,
            'rl_loaded': self.rl_agent is not None,
            'sentiment_loaded': self.sentiment_model is not None,
            'models_loaded': self.models_loaded,
            'last_predictions': {
                'rsi': self.last_rsi_signal,
                'rf': self.last_rf_prediction,
                'lstm': self.last_lstm_prediction,
                'rl': self.last_rl_action,
                'sentiment': self.last_sentiment
            }
        }
