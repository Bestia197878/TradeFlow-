"""Test All Script - Run tests for all components"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.ai_models import LSTMModel, RandomForestModel, RLAgent, LLMSentiment
from app.strategies import BaseStrategy, RSIStrategy
from app.backtesting import BacktestEngine
from app.services.exchange import ExchangeService
from app.services.risk import RiskManagement


class TestExchangeService(unittest.TestCase):
    """Test Exchange Service."""

    def setUp(self):
        self.exchange = ExchangeService(testnet=True, simulate=True)

    def test_fetch_ohlcv(self):
        """Test fetching OHLCV data."""
        data = self.exchange.fetch_ohlcv("BTC/USDT", "1h", limit=100)
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 100)
        self.assertIn('close', data.columns)

    def test_fetch_ticker(self):
        """Test fetching ticker."""
        ticker = self.exchange.fetch_ticker("BTC/USDT")
        self.assertIsNotNone(ticker)
        self.assertIn('last', ticker)

    def test_create_order(self):
        """Test creating order."""
        order = self.exchange.create_order("BTC/USDT", "market", "buy", 0.001)
        self.assertIn('id', order)


class TestAIModels(unittest.TestCase):
    """Test AI Models."""

    def setUp(self):
        # Generate sample data
        np.random.seed(42)
        self.prices = np.cumsum(np.random.randn(200) * 100 + 50000)

    def test_lstm_model(self):
        """Test LSTM model."""
        model = LSTMModel(sequence_length=20, hidden_units=32)
        history = model.train(self.prices[:100], epochs=2, verbose=0)
        self.assertTrue(model.is_trained)

    def test_rf_model(self):
        """Test Random Forest model."""
        model = RandomForestModel(n_estimators=10)
        results = model.train(self.prices, verbose=0)
        self.assertTrue(model.is_trained)
        self.assertIn('test_accuracy', results)

    def test_rl_agent(self):
        """Test RL agent."""
        agent = RLAgent(state_size=10, action_size=3, epsilon=1.0)
        results = agent.train(self.prices, episodes=2, max_steps=50, verbose=0)
        self.assertTrue(agent.is_trained)

    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        sentiment = LLMSentiment()
        result = sentiment.analyze("Bitcoin is going to the moon! This is bullish news.")
        self.assertIn('sentiment', result)
        self.assertIn('positive_prob', result)


class TestStrategies(unittest.TestCase):
    """Test Trading Strategies."""

    def setUp(self):
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        self.prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        self.data = pd.DataFrame({
            'open': self.prices * 0.99,
            'high': self.prices * 1.01,
            'low': self.prices * 0.98,
            'close': self.prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_rsi_strategy(self):
        """Test RSI strategy."""
        strategy = RSIStrategy(initial_capital=100000)
        signal = strategy.generate_signal(self.data)
        self.assertIn(signal.action, ['buy', 'sell', 'hold'])
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)

    def test_strategy_execution(self):
        """Test strategy execution."""
        strategy = RSIStrategy(initial_capital=100000)
        signal = strategy.generate_signal(self.data)
        result = strategy.execute_signal(signal, self.data['close'].iloc[-1])
        self.assertIn('executed', result)


class TestBacktest(unittest.TestCase):
    """Test Backtest Engine."""

    def setUp(self):
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='h')
        self.prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        self.data = pd.DataFrame({
            'open': self.prices * 0.99,
            'high': self.prices * 1.01,
            'low': self.prices * 0.98,
            'close': self.prices,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)

    def test_backtest_run(self):
        """Test running backtest."""
        strategy = RSIStrategy(initial_capital=100000)
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run(self.data, strategy, verbose=0)

        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)


class TestRiskManagement(unittest.TestCase):
    """Test Risk Management."""

    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        risk = RiskManagement(max_position_size=0.5, max_daily_loss=0.03)
        metrics = risk.get_risk_metrics(100000)
        self.assertIn('is_trading_allowed', metrics)
        self.assertTrue(metrics['is_trading_allowed'])

    def test_position_size_calculation(self):
        """Test position size calculation."""
        risk = RiskManagement()
        size = risk.calculate_position_size(100000, 50000, risk_per_trade=0.02)
        self.assertGreater(size, 0)


def run_tests():
    """Run all tests."""
    print("="*50)
    print("RUNNING ALL TESTS")
    print("="*50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestExchangeService))
    suite.addTests(loader.loadTestsFromTestCase(TestAIModels))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktest))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManagement))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
