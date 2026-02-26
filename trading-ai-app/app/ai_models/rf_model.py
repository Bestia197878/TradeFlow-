"""Random Forest Model for Price Prediction"""

import numpy as np
from typing import Optional, Tuple
import pickle
import os


class RandomForestModel:
    """Random Forest model for price direction prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        features: int = 20
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.features = features
        self.model = None
        self.scaler = None
        self.is_trained = False

    def _create_features(self, data: np.ndarray) -> np.ndarray:
        """Create technical indicators as features."""
        features = []

        # Price returns
        returns = np.diff(data) / data[:-1]
        features.append(returns)

        # Moving averages
        for window in [5, 10, 20]:
            if len(data) >= window:
                ma = np.convolve(data, np.ones(window)/window, mode='valid')
                ma = np.pad(ma, (len(data) - len(ma), 0), mode='edge')
                features.append(ma / data - 1)

        # Volatility
        for window in [5, 10, 20]:
            if len(data) >= window:
                volatility = np.array([np.std(data[max(0,i-window):i+1]) for i in range(len(data))])
                features.append(volatility)

        # RSI
        rsi = self._calculate_rsi(data)
        features.append(rsi)

        # MACD
        macd = self._calculate_macd(data)
        features.append(macd)

        # Combine features
        feature_array = np.column_stack(features)

        # Limit to last 'features' columns
        return feature_array[-len(data):, :self.features]

    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(period)/period, mode='full')[:len(data)-1]
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='full')[:len(data)-1]

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return np.pad(rsi, (1, 0), mode='edge')

    def _calculate_macd(self, data: np.ndarray) -> np.ndarray:
        """Calculate MACD indicator."""
        ema_12 = self._ema(data, 12)
        ema_26 = self._ema(data, 26)
        macd = ema_12 - ema_26
        signal = self._ema(macd, 9)
        return macd - signal

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        ema = np.zeros_like(data)
        ema[0] = data[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema

    def _create_labels(self, data: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Create binary labels: 1 if price goes up, 0 otherwise."""
        returns = np.diff(data) / data[:-1]
        labels = (returns > threshold).astype(int)
        return np.pad(labels, (1, 0), mode='edge')

    def train(
        self,
        data: np.ndarray,
        test_size: float = 0.2,
        verbose: int = 1
    ) -> dict:
        """Train the Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        # Create features
        X = self._create_features(data)
        y = self._create_labels(data)

        # Remove NaN values
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        self.is_trained = True

        if verbose:
            print(f"Random Forest - Train Accuracy: {train_score:.4f}")
            print(f"Random Forest - Test Accuracy: {test_score:.4f}")

        return {"train_accuracy": train_score, "test_accuracy": test_score}

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions. Returns (predictions, confidence)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if self.scaler is None:
            raise ValueError("Scaler not initialized")

        # Create features
        X = self._create_features(data)
        X = X[-1:, :]  # Use last row
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)

        return prediction, confidence

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X = self._create_features(data)
        X = X[-1:, :]
        X_scaled = self.scaler.transform(X)

        return self.model.predict_proba(X_scaled)[0]

    def save(self, path: str) -> None:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'features': self.features,
                'is_trained': self.is_trained
            }, f)

    def load(self, path: str) -> None:
        """Load the model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.n_estimators = data['n_estimators']
            self.max_depth = data['max_depth']
            self.features = data['features']
            self.is_trained = data['is_trained']
