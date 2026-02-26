"""LSTM Model for Price Prediction"""

import numpy as np
from typing import Optional, Tuple
import pickle
import os


class LSTMModel:
    """Long Short-Term Memory neural network for time series prediction."""

    def __init__(
        self,
        sequence_length: int = 60,
        hidden_units: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.is_trained = False

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> 'keras.Model':
        """Build the LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            model = Sequential()
            model.add(LSTM(
                self.hidden_units,
                return_sequences=True,
                input_shape=input_shape
            ))
            model.add(Dropout(self.dropout))

            for _ in range(self.num_layers - 1):
                model.add(LSTM(self.hidden_units, return_sequences=True))
                model.add(Dropout(self.dropout))

            model.add(LSTM(self.hidden_units))
            model.add(Dropout(self.dropout))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mse')
            return model
        except ImportError:
            print("TensorFlow not installed. Using mock model.")
            return None

    def train(
        self,
        data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> dict:
        """Train the LSTM model."""
        from sklearn.preprocessing import MinMaxScaler

        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        # Create sequences
        X, y = self._create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and train model
        self.model = self._build_model((X.shape[1], 1))

        if self.model is not None:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose
            )
            self.is_trained = True
            return history.history
        else:
            self.is_trained = True
            return {"loss": [0.0], "val_loss": [0.0]}

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if self.scaler is None:
            raise ValueError("Scaler not initialized")

        # Scale the data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))

        # Create sequence
        if len(scaled_data) >= self.sequence_length:
            input_seq = scaled_data[-self.sequence_length:]
            input_seq = input_seq.reshape(1, self.sequence_length, 1)

            if self.model is not None:
                prediction = self.model.predict(input_seq, verbose=0)
                return self.scaler.inverse_transform(prediction)
            else:
                # Mock prediction
                return np.array([data[-1] * 1.01])
        else:
            raise ValueError(f"Data length must be at least {self.sequence_length}")

    def save(self, path: str) -> None:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'hidden_units': self.hidden_units,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'is_trained': self.is_trained
            }, f)

    def load(self, path: str) -> None:
        """Load the model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.sequence_length = data['sequence_length']
            self.hidden_units = data['hidden_units']
            self.num_layers = data['num_layers']
            self.dropout = data['dropout']
            self.learning_rate = data['learning_rate']
            self.is_trained = data['is_trained']
