"use client";

import { useState } from "react";

// Sidebar navigation items
const navItems = [
  { id: "dashboard", label: "Dashboard", icon: "◫" },
  { id: "trading", label: "Trading", icon: "⬡" },
  { id: "portfolio", label: "Portfolio", icon: "◈" },
  { id: "analytics", label: "Analytics", icon: "◇" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

// Mock data for dashboard stats
const stats = [
  { label: "Total Balance", value: "$40.00", change: "+0.0%", positive: true },
  { label: "Today's P&L", value: "$0.00", change: "+0.0%", positive: true },
  { label: "Open Positions", value: "0", change: "0", positive: true },
  { label: "Win Rate", value: "0.0%", change: "+0.0%", positive: true },
];

// Recent trades data
const recentTrades = [
  { id: 1, pair: "BTC/USDT", side: "BUY", amount: "0.001 BTC", price: "$42,350", pnl: "$0.00", time: "Now" },
];

export default function Dashboard() {
  const [activeNav, setActiveNav] = useState("dashboard");
  const [chartPeriod, setChartPeriod] = useState("1W");
  const [isTrading, setIsTrading] = useState(false);

  // Dynamic chart data based on period
  const chartData: Record<string, number[]> = {
    "1H": [5, 6, 7, 8, 9, 8, 10, 11, 12, 11, 13, 14],
    "1D": [4, 5, 6, 7, 8, 9, 8, 10, 11, 12, 13, 14],
    "1W": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "1M": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "1Y": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  };

  const currentChartData = chartData[chartPeriod] || chartData["1W"];

  // Render content based on active navigation
  const renderContent = () => {
    switch (activeNav) {
      case "dashboard":
        return (
          <>
            {/* Stats Grid */}
            <div className="grid grid-cols-4 gap-4 mb-10">
              {stats.map((stat, index) => (
                <div
                  key={index}
                  className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl hover:border-neutral-700 transition-colors"
                >
                  <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">{stat.label}</p>
                  <div className="flex items-end justify-between">
                    <span className="text-2xl font-medium">{stat.value}</span>
                    <span className={`text-sm ${stat.positive ? "text-emerald-500" : "text-red-500"}`}>
                      {stat.change}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Content Grid */}
            <div className="grid grid-cols-3 gap-6">
              {/* Recent Trades */}
              <div className="col-span-2 bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium">Recent Trades</h3>
                  <button className="text-sm text-neutral-400 hover:text-white transition-colors">
                    View All →
                  </button>
                </div>
                
                <div className="space-y-3">
                  {recentTrades.map((trade) => (
                    <div
                      key={trade.id}
                      className="flex items-center justify-between p-4 bg-neutral-900 rounded-lg hover:bg-neutral-800 transition-colors"
                    >
                      <div className="flex items-center gap-4">
                        <span
                          className={`text-xs font-medium px-2 py-1 rounded ${
                            trade.side === "BUY"
                              ? "bg-emerald-500/20 text-emerald-500"
                              : "bg-red-500/20 text-red-500"
                          }`}
                        >
                          {trade.side}
                        </span>
                        <div>
                          <p className="text-sm font-medium">{trade.pair}</p>
                          <p className="text-xs text-neutral-500">{trade.amount} @ {trade.price}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`text-sm font-medium ${trade.pnl.startsWith("+") ? "text-emerald-500" : "text-red-500"}`}>
                          {trade.pnl}
                        </p>
                        <p className="text-xs text-neutral-500">{trade.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* AI Insights */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">AI Insights</h3>
                
                <div className="space-y-4">
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-blue-500">◉</span>
                      <span className="text-sm font-medium">Ready to Trade</span>
                    </div>
                    <p className="text-sm text-neutral-400">Bot configured for $40 USDT. Starting with conservative parameters.</p>
                  </div>
                  
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-emerald-500">◉</span>
                      <span className="text-sm font-medium">Risk Settings</span>
                    </div>
                    <p className="text-sm text-neutral-400">Max position: 25% ($10), Stop loss: 1%, Take profit: 2%</p>
                  </div>
                  
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-amber-500">◉</span>
                      <span className="text-sm font-medium">Capital Protection</span>
                    </div>
                    <p className="text-sm text-neutral-400">Daily loss limit: 2% ($0.80). Max drawdown: 5% ($2)</p>
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
                      onClick={() => setChartPeriod(period)}
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
                {currentChartData.map((height, index) => (
                  <div
                    key={index}
                    className="flex-1 bg-gradient-to-t from-emerald-500/30 to-emerald-500/80 rounded-t hover:from-emerald-400/40 hover:to-emerald-400/90 transition-all cursor-pointer"
                    style={{ height: `${height}%` }}
                  ></div>
                ))}
              </div>
            </div>
          </>
        );

      case "trading":
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              {/* Trading Panel */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">New Trade</h3>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-neutral-400">Trading Pair</label>
                    <select className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white">
                      <option>BTC/USDT</option>
                      <option>ETH/USDT</option>
                      <option>SOL/USDT</option>
                      <option>BNB/USDT</option>
                    </select>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-neutral-400">Amount</label>
                      <input type="number" placeholder="0.00" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500" />
                    </div>
                    <div>
                      <label className="text-sm text-neutral-400">Price</label>
                      <input type="number" placeholder="0.00" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white placeholder-neutral-500" />
                    </div>
                  </div>
                  <div className="flex gap-3 mt-6">
                    <button className="flex-1 py-3 bg-emerald-600 hover:bg-emerald-500 rounded-lg font-medium transition-colors">
                      BUY
                    </button>
                    <button className="flex-1 py-3 bg-red-600 hover:bg-red-500 rounded-lg font-medium transition-colors">
                      SELL
                    </button>
                  </div>
                </div>
              </div>

              {/* Active Positions */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">Active Positions</h3>
                <div className="text-center py-8 text-neutral-400">
                  <p>No active positions. Start trading to open positions.</p>
                </div>
              </div>
            </div>

            {/* Order Book */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Order Book - BTC/USDT</h3>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-neutral-500 mb-2">Price (USDT)</p>
                  <div className="space-y-1">
                    <p className="text-red-500">42,520.00</p>
                    <p className="text-red-500">42,510.00</p>
                    <p className="text-red-500">42,500.00</p>
                    <p className="text-red-500">42,490.00</p>
                    <p className="text-red-500">42,480.00</p>
                  </div>
                </div>
                <div>
                  <p className="text-neutral-500 mb-2">Amount (BTC)</p>
                  <div className="space-y-1">
                    <p>0.001</p>
                    <p>0.002</p>
                    <p>0.005</p>
                    <p>0.003</p>
                    <p>0.001</p>
                  </div>
                </div>
                <div>
                  <p className="text-neutral-500 mb-2">Total</p>
                  <div className="space-y-1">
                    <p>42.52</p>
                    <p>85.02</p>
                    <p>212.50</p>
                    <p>127.47</p>
                    <p>42.48</p>
                  </div>
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
                <p className="text-3xl font-medium">$40.00</p>
                <p className="text-emerald-500 text-sm mt-2">+0.0%</p>
              </div>
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">Available</p>
                <p className="text-3xl font-medium">$40.00</p>
                <p className="text-neutral-400 text-sm mt-2">100%</p>
              </div>
              <div className="p-5 bg-neutral-900/50 border border-neutral-800 rounded-xl">
                <p className="text-neutral-500 text-xs uppercase tracking-wider mb-2">In Positions</p>
                <p className="text-3xl font-medium">$0.00</p>
                <p className="text-neutral-400 text-sm mt-2">0%</p>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Assets</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-4 bg-neutral-900 rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center text-black font-bold">$</div>
                    <div>
                      <p className="font-medium">Tether</p>
                      <p className="text-sm text-neutral-400">USDT</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">40.00 USDT</p>
                    <p className="text-sm text-neutral-400">$40.00</p>
                  </div>
                </div>
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
                    <span className="text-emerald-500">$0.00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Win Rate</span>
                    <span>0.0%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Profit Factor</span>
                    <span>0.00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Sharpe Ratio</span>
                    <span>0.00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-400">Max Drawdown</span>
                    <span className="text-red-500">0.0%</span>
                  </div>
                </div>
              </div>

              <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                <h3 className="text-lg font-medium mb-6">AI Model Performance</h3>
                <div className="space-y-4">
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">LSTM Model</span>
                      <span className="text-neutral-400">Pending</span>
                    </div>
                    <div className="h-2 bg-neutral-800 rounded-full">
                      <div className="h-2 bg-neutral-600 rounded-full" style={{ width: '0%' }}></div>
                    </div>
                  </div>
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">Random Forest</span>
                      <span className="text-neutral-400">Pending</span>
                    </div>
                    <div className="h-2 bg-neutral-800 rounded-full">
                      <div className="h-2 bg-neutral-600 rounded-full" style={{ width: '0%' }}></div>
                    </div>
                  </div>
                  <div className="p-4 bg-neutral-900 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="font-medium">RL Agent</span>
                      <span className="text-neutral-400">Pending</span>
                    </div>
                    <div className="h-2 bg-neutral-800 rounded-full">
                      <div className="h-2 bg-neutral-600 rounded-full" style={{ width: '0%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Trade History</h3>
              <div className="text-center py-8 text-neutral-400">
                <p>No trades yet. Start trading to see your history.</p>
              </div>
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
                  <button className="w-12 h-6 bg-emerald-600 rounded-full relative">
                    <span className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full"></span>
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
                  <input type="password" value="••••••••••••••••" readOnly className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Binance Secret Key</label>
                  <input type="password" value="••••••••••••••••" readOnly className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" />
                </div>
                <button className="px-4 py-2 bg-neutral-700 hover:bg-neutral-600 rounded-lg transition-colors">
                  Update Keys
                </button>
              </div>
            </div>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
              <h3 className="text-lg font-medium mb-6">Risk Limits</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-neutral-400">Max Position Size (%)</label>
                  <input type="number" defaultValue="25" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Stop Loss (%)</label>
                  <input type="number" defaultValue="1" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Take Profit (%)</label>
                  <input type="number" defaultValue="2" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" />
                </div>
                <div>
                  <label className="text-sm text-neutral-400">Daily Loss Limit (%)</label>
                  <input type="number" defaultValue="2" className="w-full mt-2 p-3 bg-neutral-900 border border-neutral-700 rounded-lg text-white" />
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
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
              <span className="text-sm text-emerald-500">Active</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8">
        {/* Header */}
        <header className="flex items-center justify-between mb-10">
          <div>
            <h2 className="text-2xl font-medium">
              {navItems.find((item) => item.id === activeNav)?.label || "Dashboard"}
            </h2>
            <p className="text-neutral-500 text-sm mt-1">Portfolio: $40.00 USDT</p>
          </div>
          <div className="flex items-center gap-4">
            {/* Trade Control Buttons */}
            {!isTrading ? (
              <button
                onClick={() => setIsTrading(true)}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
              >
                <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
                Trade ON
              </button>
            ) : (
              <button
                onClick={() => setIsTrading(false)}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-sm font-medium rounded-lg transition-colors"
              >
                <span className="w-2 h-2 bg-white rounded-full"></span>
                Trade STOP
              </button>
            )}
            <button className="p-2 text-neutral-400 hover:text-white transition-colors">
              <span className="text-xl">🔔</span>
            </button>
            <div className="w-10 h-10 bg-neutral-800 rounded-full flex items-center justify-center">
              <span className="text-sm">JD</span>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        {renderContent()}
      </main>
    </div>
  );
}
