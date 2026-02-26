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
  { label: "Total Balance", value: "$124,532.89", change: "+12.5%", positive: true },
  { label: "Today's P&L", value: "+$2,341.56", change: "+3.2%", positive: true },
  { label: "Open Positions", value: "3", change: "0", positive: true },
  { label: "Win Rate", value: "67.8%", change: "+2.1%", positive: true },
];

// Recent trades data
const recentTrades = [
  { id: 1, pair: "BTC/USDT", side: "BUY", amount: "0.25 BTC", price: "$42,350", pnl: "+$234.50", time: "2 min ago" },
  { id: 2, pair: "ETH/USDT", side: "SELL", amount: "2.5 ETH", price: "$2,280", pnl: "+$89.20", time: "15 min ago" },
  { id: 3, pair: "SOL/USDT", side: "BUY", amount: "50 SOL", price: "$98.50", pnl: "-$45.00", time: "1 hr ago" },
  { id: 4, pair: "BTC/USDT", side: "SELL", amount: "0.1 BTC", price: "$42,500", pnl: "+$156.80", time: "3 hrs ago" },
];

export default function Dashboard() {
  const [activeNav, setActiveNav] = useState("dashboard");

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
            <h2 className="text-2xl font-medium">Dashboard</h2>
            <p className="text-neutral-500 text-sm mt-1">Welcome back, Trader</p>
          </div>
          <div className="flex items-center gap-4">
            <button className="p-2 text-neutral-400 hover:text-white transition-colors">
              <span className="text-xl">🔔</span>
            </button>
            <div className="w-10 h-10 bg-neutral-800 rounded-full flex items-center justify-center">
              <span className="text-sm">JD</span>
            </div>
          </div>
        </header>

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
                  <span className="text-emerald-500">◉</span>
                  <span className="text-sm font-medium">Market Trend</span>
                </div>
                <p className="text-sm text-neutral-400">Bullish momentum detected on BTC. Consider increasing position size.</p>
              </div>
              
              <div className="p-4 bg-neutral-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-amber-500">◉</span>
                  <span className="text-sm font-medium">Risk Alert</span>
                </div>
                <p className="text-sm text-neutral-400">Volatility increasing. Consider tightening stop loss.</p>
              </div>
              
              <div className="p-4 bg-neutral-900 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-blue-500">◉</span>
                  <span className="text-sm font-medium">Signal</span>
                </div>
                <p className="text-sm text-neutral-400">RSI oversold on ETH. Potential buy opportunity in next 24h.</p>
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
                  className={`px-3 py-1 text-xs rounded-lg transition-colors ${
                    period === "1W"
                      ? "bg-neutral-700 text-white"
                      : "text-neutral-500 hover:text-white"
                  }`}
                >
                  {period}
                </button>
              ))}
            </div>
          </div>
          
          {/* Simple chart visualization */}
          <div className="h-48 flex items-end gap-2">
            {[35, 45, 38, 52, 48, 65, 58, 72, 68, 85, 78, 92].map((height, index) => (
              <div
                key={index}
                className="flex-1 bg-gradient-to-t from-emerald-500/30 to-emerald-500/80 rounded-t"
                style={{ height: `${height}%` }}
              ></div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
