"use client";

import { useState, useEffect, useCallback } from "react";
import { 
  getStatus, 
  getMarketPrices, 
  getOrderBook,
  getPortfolio, 
  getTrades, 
  getAISignals,
  getPerformance,
  startTrading,
  stopTrading,
  executeTrade,
  getConfig,
  TradeStatus,
  Portfolio,
  Trade,
  AISignal,
  Performance,
  TradingConfig
} from "./lib/trading-api";

// Sidebar navigation items
const navItems = [
  { id: "dashboard", label: "Dashboard", icon: "◫" },
  { id: "trading", label: "Trading", icon: "⬡" },
  { id: "portfolio", label: "Portfolio", icon: "◈" },
  { id: "analytics", label: "Analytics", icon: "◇" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

// Trading pairs
const tradingPairs = [
  { id: "BTC/USDT", name: "Bitcoin", symbol: "BTC" },
  { id: "ETH/USDT", name: "Ethereum", symbol: "ETH" },
  { id: "SOL/USDT", name: "Solana", symbol: "SOL" },
  { id: "BNB/USDT", name: "BNB", symbol: "BNB" },
];

export default function Dashboard() {
  const [activeNav, setActiveNav] = useState("dashboard");
  const [chartPeriod, setChartPeriod] = useState("1W");
  
  // Trading state
  const [isTrading, setIsTrading] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedPair, setSelectedPair] = useState("BTC/USDT");
  
  // Data states
  const [status, setStatus] = useState<TradeStatus | null>(null);
  const [prices, setPrices] = useState<Record<string, number>>({});
  const [orderBook, setOrderBook] = useState<{bids: any[], asks: any[]}>({ bids: [], asks: [] });
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [aiSignals, setAiSignals] = useState<AISignal[]>([]);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [config, setConfig] = useState<TradingConfig | null>(null);
  
  // Form states
  const [tradeAmount, setTradeAmount] = useState("");
  const [tradePrice, setTradePrice] = useState("");
  
  // Chart data
  const [chartData, setChartData] = useState<number[]>([]);

  // Fetch data from API
  const fetchData = useCallback(async () => {
    try {
      const [statusData, pricesData, orderBookData, portfolioData, tradesData, signalsData, perfData, configData] = await Promise.all([
        getStatus(),
        getMarketPrices(),
        getOrderBook(selectedPair),
        getPortfolio(),
        getTrades(10),
        getAISignals(),
        getPerformance(),
        getConfig()
      ]);
      
      setStatus(statusData);
      setPrices(pricesData);
      setOrderBook(orderBookData);
      setPortfolio(portfolioData);
      setTrades(tradesData);
      setAiSignals(signalsData.signals);
      setPerformance(perfData);
      setConfig(configData);
      
      // Set trading state
      setIsTrading(statusData.is_running);
      
      // Generate chart data based on period
      generateChartData(chartPeriod, pricesData["BTC/USDT"] || 40000);
      
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setIsLoading(false);
    }
  }, [selectedPair, chartPeriod]);

  // Generate mock chart data
  const generateChartData = (period: string, basePrice: number) => {
    const data: number[] = [];
    let price = basePrice * 0.9;
    const points = period === "1H" ? 12 : period === "1D" ? 12 : period === "1W" ? 12 : period === "1M" ? 12 : 12;
    
    for (let i = 0; i < points; i++) {
      const change = (Math.random() - 0.45) * 0.1 * price;
      price = Math.max(price + change, price * 0.5);
      data.push(Math.min(100, Math.max(10, (price / basePrice) * 50)));
    }
    setChartData(data);
  };

  // Initial fetch and polling
  useEffect(() => {
    fetchData();
    
    // Poll every 5 seconds when trading
    const interval = setInterval(() => {
      if (isTrading) {
        fetchData();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [fetchData, isTrading]);

  // Handle start/stop trading
  const handleToggleTrading = async () => {
    setIsLoading(true);
    try {
      if (isTrading) {
        await stopTrading();
        setIsTrading(false);
      } else {
        await startTrading({
          initial_capital: 40,
          max_position_size: 0.7,
          max_daily_loss: 0.1,
          max_drawdown: 0.2,
          stop_loss_pct: 0.015,
          take_profit_pct: 0.03,
          strategy: "rsi"
        });
        setIsTrading(true);
      }
      await fetchData();
    } catch (error) {
      console.error("Error toggling trading:", error);
    }
    setIsLoading(false);
  };

  // Handle manual trade
  const handleTrade = async (side: "buy" | "sell") => {
    if (!tradeAmount || !status) return;
    
    const amount = parseFloat(tradeAmount);
    const price = prices[selectedPair] || parseFloat(tradePrice) || 0;
    
    if (amount <= 0 || price <= 0) return;
    
    try {
      await executeTrade(selectedPair, side, amount, price);
      setTradeAmount("");
      await fetchData();
    } catch (error) {
      console.error("Error executing trade:", error);
    }
  };

  // Get current price for selected pair
  const currentPrice = prices[selectedPair] || 0;

  // Render content based on active navigation
  const renderContent = () => {
    switch (activeNav) {
      case "dashboard":
        return (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-4 gap-4 mb-10">
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl hover:border-neutral-700 transition-colors">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Total Balance</p>
                <div className="flex items-end justify-between">
                  <span className="text-2xl font-medium">${(portfolio?.total_balance || status?.capital || 40).toFixed(2)}</span>
                  <span className={`text-sm ${(performance?.total_return || 0) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                    {(performance?.total_return || 0) >= 0 ? "+" : ""}{(performance?.total_return || 0 * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl hover:border-neutral-700 transition-colors">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Today's P&L</p>
                <div className="flex items-end justify-between">
                  <span className="text-2xl font-medium">${(performance?.total_pnl || 0).toFixed(2)}</span>
                  <span className={`text-sm ${(performance?.total_pnl || 0) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                    {(performance?.total_pnl || 0) >= 0 ? "+" : ""}{(performance?.total_pnl || 0).toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl hover:border-neutral-700 transition-colors">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Open Positions</p>
                <div className="flex items-end justify-between">
                  <span className="text-2xl font-medium">{(portfolio?.position || 0).toFixed(4)}</span>
                  <span className="text-sm text-neutral-400">
                    {(portfolio?.position || 0) > 0 ? `${((portfolio?.position || 0) * (prices["BTC/USDT"] || 0)).toFixed(2)} USDT` : "0 USDT"}
                  </span>
                </div>
              </div>
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl hover:border-neutral-700 transition-colors">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Win Rate</p>
                <div className="flex items-end justify-between">
                  <span className="text-2xl font-medium">{(performance?.win_rate || 0).toFixed(1)}%</span>
                  <span className="text-sm text-emerald-500">
                    {performance?.total_trades || 0} trades
                  </span>
                </div>
              </div>
            </div>

            {/* Content Grid */}
            <div className="grid grid-cols-3 gap-6">
              {/* Recent Trades */}
              <div className="col-span-2 bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium">Recent Trades</h3>
                  <button 
                    onClick={() => setActiveNav("analytics")}
                    className="text-sm text-neutral-400 hover:text-white transition-colors"
                  >
                    View All →
                  </button>
                </div>
                
                <div className="space-y-3">
                  {trades.length > 0 ? (
                    trades.map((trade) => (
                      <div
                        key={trade.id}
                        className="flex items-center justify-between p-4 bg-neutral-900 rounded-lg hover:bg-neutral-800 transition-colors"
                      >
                        <div className="flex items-center gap-4">
                          <span
                            className={`text-xs font-medium px-2 py-1 rounded ${
                              trade.side === "buy"
                                ? "bg-emerald-500/20 text-emerald-500"
                                : "bg-red-500/20 text-red-500"
                            }`}
                          >
                            {trade.side.toUpperCase()}
                          </span>
                          <div>
                            <p className="text-sm font-medium">{trade.symbol}</p>
                            <p className="text-xs text-neutral-500">{trade.amount} @ ${trade.price.toLocaleString()}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={`text-sm font-medium ${(trade.pnl || 0) >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                            {(trade.pnl || 0) >= 0 ? "+" : ""}${(trade.pnl || 0).toFixed(2)}
                          </p>
                          <p className="text-xs text-neutral-500">{trade.time}</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8 text-neutral-400">
                      <p>No trades yet. Start trading to see your history.</p>
                    </div>
                  )}
                </div>
              </div>

              {/* AI Insights */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">AI Insights</h3>
                
                <div className="space-y-4">
                  {/* Trading Status */}
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`w-2 h-2 rounded-full ${isTrading ? "bg-emerald-500 animate-pulse" : "bg-neutral-500"}`}></span>
                      <span className="text-sm font-medium">
                        {isTrading ? "Trading Active" : "Trading Stopped"}
                      </span>
                    </div>
                    <p className="text-sm text-neutral-400">
                      {isTrading 
                        ? `Bot running with ${status?.strategy || 'RSI'} strategy. Current position: ${(status?.position || 0).toFixed(4)} BTC`
                        : "Start trading to begin automated trading"
                      }
                    </p>
                  </div>
                  
                  {/* AI Signals */}
                  {aiSignals.length > 0 && (
                    <div className="p-4 bg-neutral-900 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`text-sm font-medium ${
                          aiSignals[0].action === "buy" ? "text-emerald-500" : 
                          aiSignals[0].action === "sell" ? "text-red-500" : "text-neutral-400"
                        }`}>
                          {aiSignals[0].action.toUpperCase()} Signal
                        </span>
                        <span className="text-xs text-neutral-500">
                          {(aiSignals[0].confidence * 100).toFixed(0)}% confidence
                        </span>
                      </div>
                      <p className="text-sm text-neutral-400">{aiSignals[0].reason}</p>
                    </div>
                  )}
                  
                  {/* Risk Settings */}
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-blue-500">◉</span>
                      <span className="text-sm font-medium">Risk Settings</span>
                    </div>
                    <p className="text-sm text-neutral-400">
                      Max position: {(config?.max_position_size || 0.7) * 100}%, 
                      Stop loss: {((config?.stop_loss_pct || 0.015) * 100).toFixed(1)}%, 
                      Take profit: {((config?.take_profit_pct || 0.03) * 100).toFixed(1)}%
                    </p>
                  </div>
                  
                  {/* Capital Protection */}
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-amber-500">◉</span>
                      <span className="text-sm font-medium">Capital Protection</span>
                    </div>
                    <p className="text-sm text-neutral-400">
                      Daily loss limit: {((config?.max_daily_loss || 0.1) * 100).toFixed(0)}%, 
                      Max drawdown: {((config?.max_drawdown || 0.2) * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Chart Area */}
            <div className="mt-6 bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-medium">Portfolio Performance</h3>
                <div className="flex gap-2">
                  {["1H", "1D", "1W", "1M", "1Y"].map((period) => (
                    <button
                      key={period}
                      onClick={() => {
                        setChartPeriod(period);
                        generateChartData(period, prices["BTC/USDT"] || 40000);
                      }}
                      className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                        chartPeriod === period
                          ? "bg-neutral-700 text-white"
                          : "text-neutral-500 hover:text-white hover:bg-neutral-800"
                      }`}
                    >
                      {period}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Simple chart visualization */}
              <div className="h-48 flex items-end gap-2">
                {chartData.length > 0 ? (
                  chartData.map((height, index) => (
                    <div
                      key={index}
                      className="flex-1 bg-gradient-to-t from-emerald-500/30 to-emerald-500/80 rounded-t hover:from-emerald-400/40 hover:to-emerald-400/90 transition-all cursor-pointer"
                      style={{ height: `${height}%` }}
                    ></div>
                  ))
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-neutral-500">
                    Loading chart data...
                  </div>
                )}
              </div>
            </div>
          </>
        );

      case "trading":
        return (
          <div className="space-y-6">
            {/* Trading Controls */}
            <div className="flex items-center justify-between">
              <div className="flex gap-2">
                {tradingPairs.map((pair) => (
                  <button
                    key={pair.id}
                    onClick={() => setSelectedPair(pair.id)}
                    className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                      selectedPair === pair.id
                        ? "bg-neutral-700 text-white"
                        : "bg-neutral-900 text-neutral-400 hover:text-white"
                    }`}
                  >
                    {pair.id}
                  </button>
                ))}
              </div>
              
              <button
                onClick={handleToggleTrading}
                disabled={isLoading}
                className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                  isTrading 
                    ? "bg-red-600 hover:bg-red-500" 
                    : "bg-emerald-600 hover:bg-emerald-500"
                } disabled:opacity-50`}
              >
                {isLoading ? "Loading..." : isTrading ? "Stop Trading" : "Start Trading"}
              </button>
            </div>
            
            <div className="grid grid-cols-2 gap-6">
              {/* Trading Panel */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium">New Trade</h3>
                  <span className="text-emerald-500 font-medium">
                    ${currentPrice.toLocaleString()}
                  </span>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-neutral-400">Trading Pair</label>
                    <select 
                      value={selectedPair}
                      onChange={(e) => setSelectedPair(e.target.value)}
                      className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white"
                    >
                      {tradingPairs.map((pair) => (
                        <option key={pair.id} value={pair.id}>
                          {pair.id} - {pair.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-neutral-400">Amount</label>
                      <input 
                        type="number" 
                        value={tradeAmount}
                        onChange={(e) => setTradeAmount(e.target.value)}
                        placeholder="0.00" 
                        className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500" 
                      />
                    </div>
                    <div>
                      <label className="text-sm text-neutral-400">Price</label>
                      <input 
                        type="number" 
                        value={tradePrice || currentPrice}
                        onChange={(e) => setTradePrice(e.target.value)}
                        placeholder={currentPrice.toString()} 
                        className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500" 
                      />
                    </div>
                  </div>
                  
                  {/* Quick amount buttons */}
                  <div className="flex gap-2">
                    {[10, 25, 50, 100].map((pct) => (
                      <button
                        key={pct}
                        onClick={() => setTradeAmount(((portfolio?.total_balance || 40) * pct / 100 / currentPrice).toFixed(6))}
                        className="flex-1 py-2 text-xs bg-neutral-800 hover:bg-neutral-700 rounded-lg transition-colors"
                      >
                        {pct}%
                      </button>
                    ))}
                  </div>
                  
                  <div className="flex gap-3 mt-6">
                    <button 
                      onClick={() => handleTrade("buy")}
                      className="flex-1 py-3 bg-emerald-600 hover:bg-emerald-500 rounded-lg font-medium transition-colors"
                    >
                      BUY
                    </button>
                    <button 
                      onClick={() => handleTrade("sell")}
                      className="flex-1 py-3 bg-red-600 hover:bg-red-500 rounded-lg font-medium transition-colors"
                    >
                      SELL
                    </button>
                  </div>
                </div>
              </div>

              {/* Active Positions */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">Active Positions</h3>
                {(portfolio?.position || 0) > 0 ? (
                  <div className="space-y-4">
                    <div className="p-4 bg-neutral-900 rounded-lg">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">{selectedPair}</p>
                          <p className="text-sm text-neutral-400">
                            {(portfolio?.position || 0).toFixed(6)} BTC
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="font-medium">
                            ${((portfolio?.position || 0) * currentPrice).toFixed(2)}
                          </p>
                          <p className="text-sm text-neutral-400">
                            @ ${currentPrice.toLocaleString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-neutral-400">
                    <p>No active positions. Start trading to open positions.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Order Book */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Order Book - {selectedPair}</h3>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-neutral-500 mb-2">Price (USDT)</p>
                  <div className="space-y-1">
                    {(orderBook.asks || []).map((ask, i) => (
                      <p key={i} className="text-red-500">{ask.price.toLocaleString()}</p>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-neutral-500 mb-2">Amount</p>
                  <div className="space-y-1">
                    {(orderBook.asks || []).map((ask, i) => (
                      <p key={i}>{ask.amount.toFixed(4)}</p>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-neutral-500 mb-2">Total</p>
                  <div className="space-y-1">
                    {(orderBook.asks || []).map((ask, i) => (
                      <p key={i}>{(ask.price * ask.amount).toFixed(2)}</p>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Current Price */}
              <div className="mt-4 pt-4 border-t border-neutral-800">
                <div className="flex items-center justify-between">
                  <span className="text-neutral-400">Current Price</span>
                  <span className="text-xl font-medium text-emerald-500">
                    ${currentPrice.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );

      case "portfolio":
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-4">
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Total Balance</p>
                <p className="text-3xl font-medium">${(portfolio?.total_balance || 40).toFixed(2)}</p>
                <p className="text-emerald-500 text-sm mt-2">
                  +{(performance?.total_return || 0 * 100).toFixed(2)}%
                </p>
              </div>
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Available</p>
                <p className="text-3xl font-medium">${(portfolio?.available || 40).toFixed(2)}</p>
                <p className="text-neutral-400 text-sm mt-2">
                  {((portfolio?.available || 40) / (portfolio?.total_balance || 40) * 100).toFixed(0)}%
                </p>
              </div>
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">In Positions</p>
                <p className="text-3xl font-medium">${(portfolio?.in_positions || 0).toFixed(2)}</p>
                <p className="text-neutral-400 text-sm mt-2">
                  {((portfolio?.in_positions || 0) / (portfolio?.total_balance || 40) * 100).toFixed(0)}%
                </p>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Assets</h3>
              <div className="space-y-3">
                {/* USDT */}
                <div className="flex items-center justify-between p-4 bg-neutral-900 rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center text-black font-bold">₮</div>
                    <div>
                      <p className="font-medium">Tether</p>
                      <p className="text-sm text-neutral-400">USDT</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">{(portfolio?.available || 40).toFixed(2)} USDT</p>
                    <p className="text-sm text-neutral-400">${(portfolio?.available || 40).toFixed(2)}</p>
                  </div>
                </div>
                
                {/* BTC Position */}
                {(portfolio?.position || 0) > 0 && (
                  <div className="flex items-center justify-between p-4 bg-neutral-900 rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 bg-orange-500 rounded-full flex items-center justify-center text-black font-bold">₿</div>
                      <div>
                        <p className="font-medium">Bitcoin</p>
                        <p className="text-sm text-neutral-400">BTC</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">{(portfolio?.position || 0).toFixed(6)} BTC</p>
                      <p className="text-sm text-neutral-400">
                        ${((portfolio?.position || 0) * currentPrice).toFixed(2)}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Market Prices */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Market Prices</h3>
              <div className="grid grid-cols-4 gap-4">
                {Object.entries(prices).map(([symbol, price]) => (
                  <div key={symbol} className="p-4 bg-neutral-900 rounded-lg">
                    <p className="text-sm text-neutral-400">{symbol}</p>
                    <p className="text-lg font-medium">${(price || 0).toLocaleString()}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case "analytics":
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">Performance Metrics</h3>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Total P&L</span>
                    <span className={(performance?.total_pnl || 0) >= 0 ? "text-emerald-500" : "text-red-500"}>
                      ${(performance?.total_pnl || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Win Rate</span>
                    <span>{(performance?.win_rate || 0).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Profit Factor</span>
                    <span>{(performance?.total_trades || 0) > 0 ? (performance?.win_rate || 0) / (1 - (performance?.win_rate || 0.5)).toFixed(2) : "0.00"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Sharpe Ratio</span>
                    <span>{(performance?.sharpe_ratio || 0).toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Max Drawdown</span>
                    <span className="text-red-500">
                      -{((performance?.max_drawdown || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">AI Model Performance</h3>
                <div className="space-y-4">
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">LSTM Model</span>
                      <span className="text-neutral-400">
                        {aiSignals.some(s => s.reason?.includes("LSTM")) ? "Active" : "Standby"}
                      </span>
                    </div>
                    <div className="h-2 bg-neutral-800 rounded-full">
                      <div 
                        className="h-2 bg-blue-500 rounded-full" 
                        style={{ width: aiSignals.length > 0 ? '75%' : '0%' }}
                      ></div>
                    </div>
                  </div>
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">Random Forest</span>
                      <span className="text-neutral-400">Active</span>
                    </div>
                    <div className="h-2 bg-neutral-800 rounded-full">
                      <div className="h-2 bg-green-500 rounded-full" style={{ width: '80%' }}></div>
                    </div>
                  </div>
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">RL Agent</span>
                      <span className="text-neutral-400">Learning</span>
                    </div>
                    <div className="h-2 bg-neutral-800 rounded-full">
                      <div className="h-2 bg-purple-500 rounded-full" style={{ width: '60%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Trade History</h3>
              {trades.length > 0 ? (
                <div className="space-y-2">
                  {trades.map((trade) => (
                    <div key={trade.id} className="flex items-center justify-between p-3 bg-neutral-900 rounded-lg">
                      <div className="flex items-center gap-4">
                        <span className={`px-2 py-1 text-xs rounded ${
                          trade.side === "buy" ? "bg-emerald-500/20 text-emerald-500" : "bg-red-500/20 text-red-500"
                        }`}>
                          {trade.side.toUpperCase()}
                        </span>
                        <span>{trade.symbol}</span>
                        <span className="text-neutral-400">{trade.amount}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span>${trade.price.toLocaleString()}</span>
                        <span className={(trade.pnl || 0) >= 0 ? "text-emerald-500" : "text-red-500"}>
                          ${(trade.pnl || 0).toFixed(2)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-neutral-400">
                  <p>No trades yet. Start trading to see your history.</p>
                </div>
              )}
            </div>
          </div>
        );

      case "settings":
        return (
          <div className="space-y-6">
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Trading Settings</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Auto Trading</p>
                    <p className="text-sm text-neutral-400">Allow AI to execute trades automatically</p>
                  </div>
                  <button 
                    onClick={handleToggleTrading}
                    className={`w-12 h-6 rounded-full relative transition-colors ${
                      isTrading ? "bg-emerald-600" : "bg-neutral-700"
                    }`}
                  >
                    <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all ${
                      isTrading ? "right-1" : "left-1"
                    }`}></span>
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Risk Management</p>
                    <p className="text-sm text-neutral-400">Enable stop loss and take profit</p>
                  </div>
                  <button className="w-12 h-6 bg-emerald-600 rounded-full relative">
                    <span className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full"></span>
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Notifications</p>
                    <p className="text-sm text-neutral-400">Receive alerts for trades and signals</p>
                  </div>
                  <button className="w-12 h-6 bg-neutral-700 rounded-full relative">
                    <span className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full"></span>
                  </button>
                </div>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">API Configuration</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-neutral-400">Binance API Key</label>
                  <input type="password" placeholder="Enter API key" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500" />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Binance Secret Key</label>
                  <input type="password" placeholder="Enter secret key" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500" />
                </div>
                <p className="text-xs text-neutral-500">
                  API keys are only required for live trading. Testnet mode works without API keys.
                </p>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Risk Limits</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-neutral-400">Max Position Size (%)</label>
                  <input 
                    type="number" 
                    defaultValue={(config?.max_position_size || 0.7) * 100} 
                    className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" 
                  />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Stop Loss (%)</label>
                  <input 
                    type="number" 
                    defaultValue={(config?.stop_loss_pct || 0.015) * 100} 
                    className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" 
                  />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Take Profit (%)</label>
                  <input 
                    type="number" 
                    defaultValue={(config?.take_profit_pct || 0.03) * 100} 
                    className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" 
                  />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Daily Loss Limit (%)</label>
                  <input 
                    type="number" 
                    defaultValue={(config?.max_daily_loss || 0.1) * 100} 
                    className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" 
                  />
                </div>
              </div>
              <button className="mt-4 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors">
                Save Settings
              </button>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (isLoading && !status) {
    return (
      <div className="flex min-h-screen bg-black text-white items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-neutral-400">Connecting to trading server...</p>
          <p className="text-sm text-neutral-500 mt-2">Make sure the Python server is running on port 8000</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-black text-white">
      {/* Sidebar */}
      <aside className="w-64 border-r border-neutral-800 p-6">
        <div className="mb-10">
          <h1 className="text-xl font-semibold tracking-tight">TradeFlow</h1>
          <p className="text-xs text-neutral-500 mt-1">AI Trading Platform</p>
        </div>
        
        <nav className="space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveNav(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 text-sm rounded-lg transition-all duration-200 ${
                activeNav === item.id
                  ? "bg-neutral-800 text-white"
                  : "text-neutral-400 hover:text-white hover:bg-neutral-900"
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="mt-auto pt-10">
          <div className="p-4 bg-neutral-900 rounded-xl">
            <p className="text-xs text-neutral-500 mb-2">AI Status</p>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isTrading ? "bg-emerald-500 animate-pulse" : "bg-neutral-500"}`}></span>
              <span className={`text-sm ${isTrading ? "text-emerald-500" : "text-neutral-400"}`}>
                {isTrading ? "Active" : "Standby"}
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-auto">
        <div className="max-w-7xl mx-auto">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}
