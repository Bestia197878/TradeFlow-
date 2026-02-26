"""Exchange Service"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import random


class ExchangeService:
    """
    Exchange service for fetching market data and executing trades.
    Supports both real exchange APIs and simulated data.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        simulate: bool = True
    ):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.simulate = simulate
        self.exchange = None

        # Initialize exchange if possible
        if not simulate:
            self._init_exchange()

        # Simulated data state
        self.simulated_price = 50000.0  # Starting price for BTC
        self.price_history: List[float] = [self.simulated_price]

    def _init_exchange(self) -> None:
        """Initialize exchange connection."""
        try:
            import ccxt
            exchange_class = getattr(ccxt, self.exchange_id)

            if self.testnet:
                # Use testnet
                self.exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
                # Load markets
                self.exchange.load_markets()
            else:
                self.exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True
                })
                self.exchange.load_markets()

            print(f"Connected to {self.exchange_id} {'testnet' if self.testnet else 'mainnet'}")

        except ImportError:
            print("ccxt not installed. Using simulated mode.")
            self.simulate = True
        except Exception as e:
            print(f"Error connecting to exchange: {e}")
            print("Falling back to simulated mode.")
            self.simulate = True

    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair
            timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        if self.simulate:
            return self._generate_simulated_data(symbol, timeframe, limit)

        try:
            if self.exchange:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                return df
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")

        return self._generate_simulated_data(symbol, timeframe, limit)

    def _generate_simulated_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """Generate simulated OHLCV data with realistic market movements."""
        # Determine number of periods based on timeframe
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        minutes = tf_minutes.get(timeframe, 60)

        # Generate timestamps
        end_time = datetime.now()
        timestamps = [
            end_time - timedelta(minutes=minutes * (limit - i - 1))
            for i in range(limit)
        ]

        # Generate price data with trends and volatility clusters
        data = []
        price = self.simulated_price
        
        # Create trend phases
        trend_direction = 1  # 1 = up, -1 = down
        trend_duration = 0
        
        for i in range(limit):
            # Change trend occasionally (every 20-50 bars)
            if trend_duration <= 0:
                trend_direction = random.choice([-1, 1])
                trend_duration = random.randint(20, 50)
            trend_duration -= 1
            
            # Add trend component + random component
            trend_change = trend_direction * random.uniform(0.001, 0.003)
            random_change = random.gauss(0, 0.015)  # 1.5% std dev
            
            # Occasionally add spike (5% chance)
            if random.random() < 0.05:
                random_change += random.choice([-1, 1]) * random.uniform(0.02, 0.05)
            
            change = (trend_change + random_change) * price
            price = max(price + change, price * 0.5)  # Prevent negative prices

            # Generate OHLC
            volatility = abs(random.gauss(0, 0.01))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = random.uniform(low, high)
            close_price = random.uniform(low, high)
            volume = random.uniform(1000, 10000)

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        # Update simulated price
        self.simulated_price = price
        self.price_history.append(price)

        return df

    def fetch_ticker(self, symbol: str = "BTC/USDT") -> Optional[Dict]:
        """Fetch current ticker data."""
        if self.simulate:
            return {
                'symbol': symbol,
                'bid': self.simulated_price * 0.999,
                'ask': self.simulated_price * 1.001,
                'last': self.simulated_price,
                'volume': random.uniform(1000, 10000),
                'timestamp': datetime.now()
            }

        try:
            if self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                return {
                    'symbol': symbol,
                    'bid': ticker.get('bid', 0),
                    'ask': ticker.get('ask', 0),
                    'last': ticker.get('last', 0),
                    'volume': ticker.get('quoteVolume', 0),
                    'timestamp': ticker.get('timestamp', 0)
                }
        except Exception as e:
            print(f"Error fetching ticker: {e}")

        return None

    def create_order(
        self,
        symbol: str,
        order_type: str = "market",
        side: str = "buy",
        amount: float = 0.001,
        price: Optional[float] = None
    ) -> Dict:
        """Create a trade order."""
        if self.simulate:
            return self._simulate_order(symbol, order_type, side, amount, price)

        try:
            if self.exchange:
                order_params = {
                    'symbol': symbol,
                    'type': order_type,
                    'side': side,
                    'amount': amount
                }
                if price:
                    order_params['price'] = price

                result = self.exchange.create_order(**order_params)
                return result
        except Exception as e:
            print(f"Error creating order: {e}")
            raise

        return {}

    def _simulate_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float]
    ) -> Dict:
        """Simulate order execution."""
        if price is None:
            price = self.simulated_price

        return {
            'id': f"sim_{int(time.time())}_{random.randint(1000, 9999)}",
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'filled',
            'filled': amount,
            'remaining': 0,
            'cost': amount * price,
            'timestamp': int(time.time() * 1000)
        }

    def fetch_balance(self) -> Dict:
        """Fetch account balance."""
        if self.simulate:
            return {
                'USDT': {'free': 100000, 'used': 0, 'total': 100000},
                'BTC': {'free': 0, 'used': 0, 'total': 0}
            }

        try:
            if self.exchange:
                balance = self.exchange.fetch_balance()
                return balance.get('info', balance)
        except Exception as e:
            print(f"Error fetching balance: {e}")

        return {'USDT': {'free': 0, 'used': 0, 'total': 0}}

    def fetch_positions(self) -> List[Dict]:
        """Fetch open positions."""
        if self.simulate:
            return []

        try:
            if self.exchange:
                positions = self.exchange.fetch_positions()
                return positions
        except Exception as e:
            print(f"Error fetching positions: {e}")

        return []

    def get_market_price(self, symbol: str = "BTC/USDT") -> float:
        """Get current market price."""
        ticker = self.fetch_ticker(symbol)
        if ticker:
            return ticker.get('last', self.simulated_price)
        return self.simulated_price
