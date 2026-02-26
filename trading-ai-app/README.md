# Trading AI Application

A comprehensive trading AI application with multiple AI models, strategies, backtesting, and live execution capabilities.

## Features

- **AI Models**: LSTM, Random Forest, Reinforcement Learning, LLM Sentiment Analysis
- **Strategies**: RSI Strategy and extensible base strategy
- **Backtesting**: Full backtesting engine with performance metrics
- **Live Trading**: Testnet and live execution support
- **Risk Management**: Built-in risk management services

## Project Structure

```
trading-ai-app/
├── app/
│   ├── ai_models/         # AI trading models
│   ├── strategies/        # Trading strategies
│   ├── backtesting/      # Backtesting engine
│   ├── execution/        # Live trading execution
│   └── services/         # Exchange and risk services
├── scripts/              # Training and execution scripts
├── config/               # Configuration files
├── data/                 # Market data storage
└── models/              # Trained model storage
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Download market data:
```bash
python scripts/download_data.py
```

2. Train all models:
```bash
python scripts/train_all_models.py
```

3. Run backtest:
```bash
python scripts/backtest.py
```

4. Run on testnet:
```bash
python scripts/run_testnet.py
```

5. Run live:
```bash
python scripts/run_live.py
```

## Configuration

Edit `config/testnet.yaml` for testnet settings or `config/live.yaml` for live trading.

## License

MIT
