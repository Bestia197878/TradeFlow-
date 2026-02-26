"""Trading API Server - FastAPI"""

import sys
import os
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random

from app.strategies import RSIStrategy
from app.strategies.ai_ensemble import AIEnsembleStrategy
from app.services.exchange import ExchangeService
from app.services.risk import RiskManagement
from app.execution.live_trader import LiveTrader

# Create FastAPI app
app = FastAPI(title="Trading API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trader: Optional[LiveTrader] = None
trading_state = {
    "is_running": False,
    "capital": 40.0,
    "position": 0.0,
    "total_trades": 0,
    "win_rate": 0.0,
    "pnl": 0.0,
    "last_trade": None,
    "current_price": 0.0,
    "symbol": "BTC/USDT",
    "strategy": "rsi"
}

# Market data cache
market_data = {
    "prices": {
        "BTC/USDT": 0.0,
        "ETH/USDT": 0.0,
        "SOL/USDT": 0.0,
        "BNB/USDT": 0.0
    },
    "order_book": {
        "bids": [],
        "asks": []
    }
}


# Models
class TradeRequest(BaseModel):
    symbol: str
    side: str  # buy or sell
    amount: float
    price: Optional[float] = None


class ConfigRequest(BaseModel):
    symbol: str = "BTC/USDT"
    initial_capital: float = 40.0
    max_position_size: float = 0.7
    max_daily_loss: float = 0.1
    max_drawdown: float = 0.2
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    strategy: str = "rsi"
    rsi_period: int = 14
    overbought_threshold: float = 55
    oversold_threshold: float = 45


# Initialize exchange service for market data
exchange = ExchangeService(
    exchange_id="binance",
    testnet=True,
    simulate=True
)


def update_market_data():
    """Update cached market data."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    
    for symbol in symbols:
        ticker = exchange.fetch_ticker(symbol)
        if ticker:
            market_data["prices"][symbol] = ticker.get("last", 0)
    
    # Generate order book from current price
    btc_price = market_data["prices"]["BTC/USDT"]
    if btc_price > 0:
        bids = []
        asks = []
        for i in range(5):
            bid_price = btc_price * (1 - (i + 1) * 0.0005)
            ask_price = btc_price * (1 + (i + 1) * 0.0005)
            bids.append({
                "price": round(bid_price, 2),
                "amount": round(random.uniform(0.001, 0.01), 4)
            })
            asks.append({
                "price": round(ask_price, 2),
                "amount": round(random.uniform(0.001, 0.01), 4)
            })
        market_data["order_book"]["bids"] = bids
        market_data["order_book"]["asks"] = asks
    
    # Update current price in trading state
    trading_state["current_price"] = btc_price


def trading_loop(interval: int = 30):
    """Background trading loop."""
    global trader, trading_state
    
    while trading_state["is_running"]:
        try:
            if trader:
                status = trader.get_status()
                trading_state["capital"] = status.get("capital", 40.0)
                trading_state["position"] = status.get("position", 0.0)
                trading_state["total_trades"] = status.get("total_trades", 0)
                
                perf = trader.get_performance()
                trading_state["win_rate"] = perf.get("win_rate", 0.0) * 100
                trading_state["pnl"] = perf.get("total_pnl", 0.0)
            
            update_market_data()
            asyncio.sleep(interval)
        except Exception as e:
            print(f"Trading loop error: {e}")
            break


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Trading API Server", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get current trading status."""
    update_market_data()
    
    return {
        "is_running": trading_state["is_running"],
        "capital": trading_state["capital"],
        "position": trading_state["position"],
        "total_trades": trading_state["total_trades"],
        "win_rate": trading_state["win_rate"],
        "pnl": trading_state["pnl"],
        "current_price": trading_state["current_price"],
        "symbol": trading_state["symbol"],
        "strategy": trading_state["strategy"],
        "prices": market_data["prices"]
    }


@app.get("/api/market/prices")
async def get_market_prices():
    """Get current market prices."""
    update_market_data()
    return market_data["prices"]


@app.get("/api/market/orderbook")
async def get_orderbook(symbol: str = "BTC/USDT"):
    """Get order book for a symbol."""
    update_market_data()
    return market_data["order_book"]


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio information."""
    return {
        "total_balance": trading_state["capital"],
        "available": trading_state["capital"] - (trading_state["position"] * trading_state["current_price"]),
        "in_positions": trading_state["position"] * trading_state["current_price"],
        "position": trading_state["position"],
        "asset": "USDT",
        "price": trading_state["current_price"]
    }


@app.get("/api/trades")
async def get_trades(limit: int = 10):
    """Get recent trades."""
    if trader:
        trades = trader.get_trade_history(limit)
        return trades
    return []


@app.post("/api/trading/start")
async def start_trading(config: ConfigRequest, background_tasks: BackgroundTasks):
    """Start trading."""
    global trader, trading_state
    
    if trading_state["is_running"]:
        raise HTTPException(status_code=400, detail="Trading is already running")
    
    # Update trading state
    trading_state["capital"] = config.initial_capital
    trading_state["strategy"] = config.strategy
    trading_state["symbol"] = config.symbol
    
    # Initialize exchange
    exchange_service = ExchangeService(
        exchange_id="binance",
        testnet=True,
        simulate=True
    )
    
    # Initialize strategy
    if config.strategy == "ai_ensemble":
        strategy = AIEnsembleStrategy(
            initial_capital=config.initial_capital,
            rsi_period=config.rsi_period,
            overbought_threshold=config.overbought_threshold,
            oversold_threshold=config.oversold_threshold,
            max_position_size=config.max_position_size
        )
    else:
        strategy = RSIStrategy(
            initial_capital=config.initial_capital,
            rsi_period=config.rsi_period,
            overbought_threshold=config.overbought_threshold,
            oversold_threshold=config.oversold_threshold
        )
    
    # Initialize risk management
    risk_manager = RiskManagement(
        max_position_size=config.max_position_size,
        max_daily_loss=config.max_daily_loss,
        max_drawdown=config.max_drawdown,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        trailing_stop=True,
        trailing_stop_pct=0.01
    )
    
    # Configuration
    cfg = {
        'symbol': config.symbol,
        'timeframe': '15m',
        'interval': 30,
        'initial_capital': config.initial_capital
    }
    
    # Initialize trader
    trader = LiveTrader(
        exchange=exchange_service,
        strategy=strategy,
        config=cfg,
        risk_manager=risk_manager,
        paper_trading=True
    )
    
    # Set up callbacks
    def on_trade(trade):
        trading_state["last_trade"] = trade
        trading_state["total_trades"] += 1
    
    def on_signal(signal):
        print(f"Signal: {signal.action} - {signal.reason}")
    
    trader.on_trade = on_trade
    trader.on_signal = on_signal
    
    # Start trading
    trader.start()
    trading_state["is_running"] = True
    
    # Initial update
    update_market_data()
    
    return {
        "status": "started",
        "message": f"Trading started with {config.strategy} strategy",
        "config": {
            "capital": config.initial_capital,
            "strategy": config.strategy,
            "symbol": config.symbol
        }
    }


@app.post("/api/trading/stop")
async def stop_trading():
    """Stop trading."""
    global trader, trading_state
    
    if not trading_state["is_running"]:
        raise HTTPException(status_code=400, detail="Trading is not running")
    
    if trader:
        trader.stop()
    
    trading_state["is_running"] = False
    
    return {
        "status": "stopped",
        "message": "Trading stopped"
    }


@app.post("/api/trading/execute")
async def execute_trade(trade: TradeRequest):
    """Execute a manual trade."""
    global trading_state
    
    if not trading_state["is_running"]:
        raise HTTPException(status_code=400, detail="Start trading first")
    
    if not trader:
        raise HTTPException(status_code=400, detail="No active trader")
    
    try:
        # Execute the trade
        result = trader.execute_manual_trade(
            symbol=trade.symbol,
            side=trade.side,
            amount=trade.amount,
            price=trade.price
        )
        
        # Update state
        trading_state["total_trades"] += 1
        
        return {
            "status": "success",
            "trade": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    """Get current trading configuration."""
    return {
        "initial_capital": 40.0,
        "max_position_size": 0.7,
        "max_daily_loss": 0.1,
        "max_drawdown": 0.2,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
        "strategy": trading_state["strategy"],
        "rsi_period": 14,
        "overbought_threshold": 55,
        "oversold_threshold": 45
    }


@app.get("/api/performance")
async def get_performance():
    """Get performance metrics."""
    if trader:
        perf = trader.get_performance()
        return perf
    
    return {
        "total_return": 0.0,
        "total_trades": 0,
        "win_rate": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_pnl": 0.0
    }


@app.get("/api/ai/signals")
async def get_ai_signals():
    """Get AI trading signals."""
    # Generate signals based on current market data
    signals = []
    
    # Check if we have a trader with strategy
    if trader and hasattr(trader, 'strategy'):
        try:
            # Get latest analysis
            signal = trader.strategy.get_signal(exchange, trading_state["symbol"])
            if signal:
                signals.append({
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "reason": signal.reason
                })
        except:
            pass
    
    # If no signals from trader, generate demo signals
    if not signals:
        rsi_value = random.uniform(30, 70)
        action = "buy" if rsi_value < 45 else "sell" if rsi_value > 55 else "hold"
        signals.append({
            "action": action,
            "confidence": round(random.uniform(0.5, 0.9), 2),
            "reason": f"RSI: {rsi_value:.1f}",
            "rsi": rsi_value
        })
    
    return {
        "signals": signals,
        "timestamp": datetime.now().isoformat()
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
