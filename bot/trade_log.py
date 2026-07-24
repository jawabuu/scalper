"""
Append-only in-memory ledger of closed trades.
Thread-safe: engine appends, API reads.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ClosedTrade:
    symbol: str
    entry_price: float
    exit_price: float
    qty: float
    pnl_pct: float
    pnl_usdt: float
    reason: str  # take_profit | trailing_stop | timeout | kill_switch | profit_lock | hard_stop
    opened_at: datetime
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Strategy attribution + diagnostics (for evaluating gainer vs dipper, and
    # whether entry conditions like a big upper wick predict weak trades).
    strategy: str = "breakout"          # breakout | pullback
    regime: str | None = None           # gainer | dipper | None (breakout)
    peak_pnl_pct: float = 0.0           # highest P&L the trade reached
    entry_stamps: dict = field(default_factory=dict)  # rsi, candle_pos, upper_wick, etc.


class TradeLog:
    def __init__(self, max_entries: int = 500):
        self._trades: list[ClosedTrade] = []
        self._lock = threading.Lock()
        self._max = max_entries

    def append(self, trade: ClosedTrade):
        with self._lock:
            self._trades.append(trade)
            if len(self._trades) > self._max:
                self._trades = self._trades[-self._max:]

    def all(self) -> list[ClosedTrade]:
        with self._lock:
            return list(self._trades)

    def summary(self) -> dict:
        with self._lock:
            trades = list(self._trades)

        if not trades:
            return {
                "total_trades": 0,
                "winning": 0,
                "losing": 0,
                "win_rate": 0.0,
                "total_pnl_pct": 0.0,
                "total_pnl_usdt": 0.0,
                "avg_pnl_pct": 0.0,
                "best_trade_pct": 0.0,
                "worst_trade_pct": 0.0,
            }

        winning = [t for t in trades if t.pnl_pct > 0]
        losing  = [t for t in trades if t.pnl_pct <= 0]
        pnls    = [t.pnl_pct for t in trades]

        def _breakdown(subset):
            if not subset:
                return {"count": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0,
                        "total_pnl_pct": 0.0, "avg_peak_pct": 0.0}
            w = [t for t in subset if t.pnl_pct > 0]
            return {
                "count":         len(subset),
                "win_rate":      round(len(w) / len(subset) * 100, 1),
                "avg_pnl_pct":   round(sum(t.pnl_pct for t in subset) / len(subset), 3),
                "total_pnl_pct": round(sum(t.pnl_pct for t in subset), 3),
                # Average peak reached — distinguishes "gave back a real gain" from
                # "never got traction" (e.g. a trade peaking at +0.04% never moved).
                "avg_peak_pct":  round(sum(t.peak_pnl_pct for t in subset) / len(subset), 3),
            }

        return {
            "total_trades":   len(trades),
            "winning":        len(winning),
            "losing":         len(losing),
            "win_rate":       round(len(winning) / len(trades) * 100, 1),
            "total_pnl_pct":  round(sum(pnls), 3),
            "total_pnl_usdt": round(sum(t.pnl_usdt for t in trades), 2),
            "avg_pnl_pct":    round(sum(pnls) / len(pnls), 3),
            "best_trade_pct": round(max(pnls), 3),
            "worst_trade_pct": round(min(pnls), 3),
            # Per-regime breakdown — the key to the gainer-vs-dipper question.
            "by_regime": {
                "gainer": _breakdown([t for t in trades if t.regime == "gainer"]),
                "dipper": _breakdown([t for t in trades if t.regime == "dipper"]),
            },
        }
