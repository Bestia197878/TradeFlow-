"""Reinforcement Learning Agent for Trading"""

import numpy as np
from typing import Optional, Tuple
import pickle
import os
from collections import deque
import random


class RLAgent:
    """Deep Q-Learning agent for automated trading."""

    def __init__(
        self,
        state_size: int = 10,
        action_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32
    ):
        self.state_size = state_size
        self.action_size = action_size  # 0: hold, 1: buy, 2: sell
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.model = None
        self.target_model = None
        self.is_trained = False

    def _build_model(self) -> 'keras.Model':
        """Build the Q-network."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                Dense(64, input_dim=self.state_size, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
            return model
        except ImportError:
            print("TensorFlow not installed. Using mock model.")
            return None

    def _build_state(
        self,
        prices: np.ndarray,
        position: float,
        capital: float
    ) -> np.ndarray:
        """Build state from market data."""
        # Normalize prices
        if len(prices) > 0:
            price_norm = prices[-1] / (prices.mean() + 1e-10)
        else:
            price_norm = 1.0

        # Returns
        returns = 0.0
        if len(prices) > 1:
            returns = (prices[-1] - prices[-2]) / (prices[-2] + 1e-10)

        # Volatility
        volatility = 0.0
        if len(prices) > 5:
            ret = np.diff(prices) / (prices[:-1] + 1e-10)
            volatility = np.std(ret[-5:])

        # Position ratio
        position_ratio = position / (capital + 1e-10)

        # Simple moving average ratio
        sma_ratio = 1.0
        if len(prices) >= 10:
            sma = np.mean(prices[-10:])
            sma_ratio = prices[-1] / (sma + 1e-10)

        # RSI
        rsi = self._calculate_rsi(prices)

        # MACD
        macd = self._calculate_macd(prices)

        state = np.array([
            price_norm,
            returns,
            volatility,
            position_ratio,
            sma_ratio,
            rsi / 100.0,  # Normalize RSI
            macd,
            capital / 100000.0,  # Normalize capital
            position,
            returns - volatility  # Risk-adjusted return
        ])

        return state

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD."""
        if len(prices) < 26:
            return 0.0

        ema_12 = self._ema(prices[-26:], 12)
        ema_26 = self._ema(prices[-26:], 26)
        macd = ema_12 - ema_26

        # Signal line
        signal = self._ema(prices[-9:], 9) if len(prices) >= 9 else macd

        return macd - signal

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return np.mean(data)

        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.model is not None:
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            return np.argmax(q_values[0])
        else:
            # Mock action
            return random.randrange(self.action_size)

    def replay(self) -> float:
        """Train on batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        if self.model is not None:
            current_q = self.model.predict(states, verbose=0)
            next_q = self.target_model.predict(next_states, verbose=0)

            for i in range(len(batch)):
                if dones[i]:
                    current_q[i][actions[i]] = rewards[i]
                else:
                    current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

            loss = self.model.train_on_batch(states, current_q)
            return float(loss) if isinstance(loss, (int, float)) else float(loss[0])
        else:
            return 0.0

    def train(
        self,
        prices: np.ndarray,
        initial_capital: float = 100000,
        episodes: int = 100,
        max_steps: int = 1000,
        verbose: int = 1
    ) -> dict:
        """Train the RL agent."""
        self.model = self._build_model()
        self.target_model = self._build_model()

        if self.model is None:
            self.is_trained = True
            return {"episodes": episodes, "total_reward": 0}

        rewards_history = []

        for episode in range(episodes):
            capital = initial_capital
            position = 0.0
            total_reward = 0
            prices_idx = random.randint(20, len(prices) - max_steps - 1)
            episode_prices = prices[prices_idx:prices_idx + max_steps]

            state = self._build_state(episode_prices[:self.state_size], position, capital)

            for t in range(len(episode_prices) - self.state_size):
                current_prices = episode_prices[:self.state_size + t + 1]
                state = self._build_state(current_prices, position, capital)

                action = self.act(state)

                # Execute action
                if action == 1 and capital > episode_prices[self.state_size + t]:  # Buy
                    position += 1
                    capital -= episode_prices[self.state_size + t]
                elif action == 2 and position > 0:  # Sell
                    position -= 1
                    capital += episode_prices[self.state_size + t]

                # Calculate reward
                next_prices = current_prices
                if len(episode_prices) > self.state_size + t + 1:
                    next_prices = episode_prices[:self.state_size + t + 2]

                next_state = self._build_state(next_prices, position, capital)

                # Reward = change in portfolio value
                portfolio_before = capital + position * episode_prices[self.state_size + t]
                if len(episode_prices) > self.state_size + t + 1:
                    portfolio_after = capital + position * episode_prices[self.state_size + t + 1]
                else:
                    portfolio_after = portfolio_before

                reward = portfolio_after - portfolio_before
                total_reward += reward

                done = t == len(episode_prices) - self.state_size - 1

                self.remember(state, action, reward, next_state, done)
                state = next_state

                # Train
                self.replay()

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Update target model periodically
            if episode % 10 == 0:
                self.target_model.set_weights(self.model.get_weights())

            rewards_history.append(total_reward)

            if verbose and episode % 10 == 0:
                print(f"Episode {episode}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {self.epsilon:.4f}")

        self.is_trained = True
        return {
            "episodes": episodes,
            "avg_reward": np.mean(rewards_history),
            "max_reward": max(rewards_history)
        }

    def get_action(self, prices: np.ndarray, position: float, capital: float) -> int:
        """Get action for current state."""
        state = self._build_state(prices, position, capital)
        return self.act(state)

    def save(self, path: str) -> None:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.model is not None:
            self.model.save(path + '_model.keras')
        with open(path + '_params.pkl', 'wb') as f:
            pickle.dump({
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'is_trained': self.is_trained
            }, f)

    def load(self, path: str) -> None:
        """Load the model from disk."""
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(path + '_model.keras')
        except Exception:
            print("Could not load Keras model")

        with open(path + '_params.pkl', 'rb') as f:
            data = pickle.load(f)
            self.state_size = data['state_size']
            self.action_size = data['action_size']
            self.learning_rate = data['learning_rate']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon']
            self.epsilon_min = data['epsilon_min']
            self.epsilon_decay = data['epsilon_decay']
            self.is_trained = data['is_trained']

        self.target_model = self._build_model()
