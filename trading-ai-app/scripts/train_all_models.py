"""Train All Models Script"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd

from app.ai_models import LSTMModel, RandomForestModel, RLAgent
from app.services.exchange import ExchangeService


def generate_training_data(symbol: str = "BTC/USDT", samples: int = 1000) -> np.ndarray:
    """Generate or fetch training data."""
    print(f"Generating training data for {symbol}...")

    # Try to load from file first
    data_file = f"data/{symbol.replace('/', '_')}_1h.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        return df['close'].values

    # Generate simulated data
    print("Using simulated data for training...")
    exchange = ExchangeService(testnet=True, simulate=True)
    data = exchange.fetch_ohlcv(symbol, "1h", limit=samples)
    return data['close'].values


def train_lstm_model(data: np.ndarray, output_dir: str = "models") -> LSTMModel:
    """Train LSTM model."""
    print("\n" + "="*50)
    print("Training LSTM Model...")
    print("="*50)

    model = LSTMModel(
        sequence_length=60,
        hidden_units=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001
    )

    # Train model
    history = model.train(
        data,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "lstm_model.pkl"))
    print(f"LSTM model saved to {output_dir}/lstm_model.pkl")

    return model


def train_rf_model(data: np.ndarray, output_dir: str = "models") -> RandomForestModel:
    """Train Random Forest model."""
    print("\n" + "="*50)
    print("Training Random Forest Model...")
    print("="*50)

    model = RandomForestModel(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        features=20
    )

    # Train model
    results = model.train(data, test_size=0.2, verbose=1)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "rf_model.pkl"))
    print(f"Random Forest model saved to {output_dir}/rf_model.pkl")

    return model


def train_rl_agent(data: np.ndarray, output_dir: str = "models") -> RLAgent:
    """Train RL agent."""
    print("\n" + "="*50)
    print("Training RL Agent...")
    print("="*50)

    agent = RLAgent(
        state_size=10,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32
    )

    # Train agent
    results = agent.train(
        prices=data,
        initial_capital=100000,
        episodes=50,
        max_steps=500,
        verbose=1
    )

    # Save agent
    os.makedirs(output_dir, exist_ok=True)
    agent.save(os.path.join(output_dir, "rl_agent"))
    print(f"RL agent saved to {output_dir}/rl_agent")

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train all AI models")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--skip-rf", action="store_true", help="Skip Random Forest training")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training")

    args = parser.parse_args()

    print("="*50)
    print("TRAINING ALL MODELS")
    print("="*50)

    # Generate training data
    data = generate_training_data(args.symbol, args.samples)

    # Train models
    if not args.skip_lstm:
        train_lstm_model(data, args.output)

    if not args.skip_rf:
        train_rf_model(data, args.output)

    if not args.skip_rl:
        train_rl_agent(data, args.output)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
