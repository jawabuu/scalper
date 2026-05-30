"""
REST API for the scalping bot dashboard.
Runs in a background thread alongside the engine.
All reads are non-blocking; kill switch write is thread-safe via Python GIL + bool assignment.
"""

import time
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Engine is injected at startup — see main.py
_engine = None


def create_app(engine) -> FastAPI:
    global _engine
    _engine = engine

    app = FastAPI(title="Scalping Bot API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    def status():
        """Bot health, mode, and current cycle info."""
        # Fetch live balance directly — don't serve the stale engine cache
        # which only refreshes each cycle. This keeps the UI consistent
        # immediately after a trade executes.
        try:
            raw = _engine.exchange.fetch_balance()
            balance_usdt = float(raw.get("USDT", {}).get("free") or _engine.last_balance)
            _engine.last_balance = balance_usdt  # keep cache in sync
        except Exception:
            balance_usdt = _engine.last_balance

        return {
            "running":       True,
            "testnet":       _engine.cfg.testnet,
            "kill_switch":   _engine.kill_switch,
            "timeframe":     _engine.cfg.timeframe,
            "last_cycle_ts": _engine.last_cycle_ts,
            "last_cycle_ago_s": round(time.time() - _engine.last_cycle_ts, 1)
                                 if _engine.last_cycle_ts else None,
            "balance_usdt":  balance_usdt,
            "server_time":   datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/positions")
    def positions():
        """All currently open positions with live unrealised P&L."""
        # Use store's thread-safe snapshot — avoids RuntimeError if engine
        # adds/removes a position while we're iterating.
        snapshot = _engine.positions.snapshot()
        result = []
        for sym, pos in snapshot.items():
            # Best-effort live price from last OHLCV — no extra API call
            try:
                ticker = _engine.exchange.fetch_ticker(sym)
                current_price = ticker["last"] or pos.entry_price
            except Exception:
                current_price = pos.entry_price

            pnl_pct  = (current_price - pos.entry_price) / pos.entry_price * 100
            pnl_usdt = (current_price - pos.entry_price) * pos.qty

            result.append({
                "symbol":        sym,
                "entry_price":   pos.entry_price,
                "current_price": current_price,
                "qty":           pos.qty,
                "trailing_stop": pos.trailing_stop,
                "candles_held":  pos.candles_held,
                "opened_at":     pos.opened_at.isoformat(),
                "pnl_pct":       round(pnl_pct, 4),
                "pnl_usdt":      round(pnl_usdt, 4),
            })
        return result

    @app.get("/api/trades")
    def trades(limit: int = 100):
        """Closed trade history, most recent first."""
        all_trades = _engine.trade_log.all()
        recent = list(reversed(all_trades))[:limit]
        return [
            {
                "symbol":      t.symbol,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "qty":         t.qty,
                "pnl_pct":     t.pnl_pct,
                "pnl_usdt":    t.pnl_usdt,
                "reason":      t.reason,
                "opened_at":   t.opened_at.isoformat(),
                "closed_at":   t.closed_at.isoformat(),
            }
            for t in recent
        ]

    @app.get("/api/summary")
    def summary():
        """Aggregate P&L stats across all closed trades."""
        return _engine.trade_log.summary()

    @app.post("/api/kill")
    def kill():
        """Engage kill switch — stops new entries and closes all open positions."""
        _engine.kill_switch = True
        log_msg = "Kill switch ENGAGED via API"
        import logging
        logging.getLogger("api").warning(log_msg)
        return {"ok": True, "message": "Kill switch engaged. Open positions will be closed on next cycle."}

    @app.post("/api/resume")
    def resume():
        """Disengage kill switch — bot resumes normal operation."""
        _engine.kill_switch = False
        import logging
        logging.getLogger("api").info("Kill switch DISENGAGED via API")
        return {"ok": True, "message": "Bot resumed."}

    @app.post("/api/close-all")
    def close_all():
        """Immediately close all open positions (does not pause the bot)."""
        count = len(_engine.positions)
        if count == 0:
            return {"ok": True, "message": "No open positions to close.", "closed": 0}
        _engine.close_all_positions(reason="manual")
        return {"ok": True, "message": f"Closed {count} position(s).", "closed": count}

    @app.post("/api/close/{symbol:path}")
    def close_one(symbol: str):
        """Close a single position by symbol (e.g. BTC/USDT)."""
        import logging
        if symbol not in _engine.positions:
            raise HTTPException(status_code=404, detail=f"{symbol} not in open positions")
        pos = _engine.positions[symbol]
        try:
            ticker = _engine.exchange.fetch_ticker(symbol)
            price = float(ticker["last"] or pos.entry_price)
        except Exception:
            price = pos.entry_price
        _engine._close_position(symbol, price, reason="manual")
        logging.getLogger("api").info(f"Manually closed {symbol} @ {price}")
        return {"ok": True, "message": f"Closed {symbol}.", "symbol": symbol}

    return app


def run_api(engine, host: str = "0.0.0.0", port: int = 8000):
    """Start uvicorn in a daemon thread. Returns immediately."""
    import uvicorn
    app = create_app(engine)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="api-server")
    thread.start()

    import logging
    logging.getLogger("api").info(f"Dashboard API listening on {host}:{port}")
    return thread
