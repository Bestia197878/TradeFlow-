// Trading API client
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface MarketPrice {
  symbol: string;
  price: number;
  change24h?: number;
}

export interface OrderBook {
  bids: Array<{ price: number; amount: number }>;
  asks: Array<{ price: number; amount: number }>;
}

export interface TradeStatus {
  is_running: boolean;
  capital: number;
  position: number;
  total_trades: number;
  win_rate: number;
  pnl: number;
  current_price: number;
  symbol: string;
  strategy: string;
  prices: Record<string, number>;
}

export interface Portfolio {
  total_balance: number;
  available: number;
  in_positions: number;
  position: number;
  asset: string;
  price: number;
}

export interface Trade {
  id: string;
  symbol: string;
  side: string;
  amount: number;
  price: number;
  value: number;
  pnl?: number;
  time: string;
  status: string;
}

export interface AISignal {
  action: string;
  confidence: number;
  reason: string;
  rsi?: number;
}

export interface Performance {
  total_return: number;
  total_trades: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_pnl: number;
}

export interface TradingConfig {
  initial_capital: number;
  max_position_size: number;
  max_daily_loss: number;
  max_drawdown: number;
  stop_loss_pct: number;
  take_profit_pct: number;
  strategy: string;
  rsi_period: number;
  overbought_threshold: number;
  oversold_threshold: number;
}

// API Functions
export async function getStatus(): Promise<TradeStatus> {
  const res = await fetch(`${API_BASE}/api/status`, { 
    cache: 'no-store' 
  });
  return res.json();
}

export async function getMarketPrices(): Promise<Record<string, number>> {
  const res = await fetch(`${API_BASE}/api/market/prices`, { 
    cache: 'no-store' 
  });
  return res.json();
}

export async function getOrderBook(symbol: string = "BTC/USDT"): Promise<OrderBook> {
  const res = await fetch(`${API_BASE}/api/market/orderbook?symbol=${symbol}`, { 
    cache: 'no-store' 
  });
  return res.json();
}

export async function getPortfolio(): Promise<Portfolio> {
  const res = await fetch(`${API_BASE}/api/portfolio`, { 
    cache: 'no-store' 
  });
  return res.json();
}

export async function getTrades(limit: number = 10): Promise<Trade[]> {
  const res = await fetch(`${API_BASE}/api/trades?limit=${limit}`, { 
    cache: 'no-store' 
  });
  return res.json();
}

export async function startTrading(config: Partial<TradingConfig>): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/trading/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  return res.json();
}

export async function stopTrading(): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/trading/stop`, {
    method: 'POST'
  });
  return res.json();
}

export async function executeTrade(
  symbol: string, 
  side: string, 
  amount: number, 
  price?: number
): Promise<{ status: string; trade: any }> {
  const res = await fetch(`${API_BASE}/api/trading/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, side, amount, price })
  });
  return res.json();
}

export async function getConfig(): Promise<TradingConfig> {
  const res = await fetch(`${API_BASE}/api/config`, { 
    cache: 'no-store' 
  });
  return res.json();
}

export async function getPerformance(): Promise<Performance> {
  const res = await fetch(`${API_BASE}/api/performance`, { 
    cache: 'no-store' 
  });
  return res.json();
}

export async function getAISignals(): Promise<{ signals: AISignal[]; timestamp: string }> {
  const res = await fetch(`${API_BASE}/api/ai/signals`, { 
    cache: 'no-store' 
  });
  return res.json();
}
