"""
REST API for the scalping bot dashboard.
Runs in a background thread alongside the engine.
All reads are non-blocking; kill switch write is thread-safe via Python GIL + bool assignment.

All /api/* routes require GitHub OAuth authentication.
Auth routes (/auth/*) are public.
"""

import time
import logging
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import (
    get_current_user,
    login_route,
    callback_route,
    me_route,
    logout_route,
)

log = logging.getLogger("api")

# Engine is injected at startup — see main.py
_engine = None


def _require_auth(request: Request) -> dict:
    """FastAPI dependency — returns user dict or raises 401."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def create_app(engine) -> FastAPI:
    global _engine
    _engine = engine

    app = FastAPI(title="Scalping Bot API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # ── Auth routes (public) ─────────────────────────────────────────────────

    @app.get("/auth/login")
    def login(request: Request):
        return login_route(request)

    @app.get("/auth/callback")
    async def callback(request: Request):
        return await callback_route(request)

    @app.get("/auth/me")
    def me(request: Request):
        return me_route(request)

    @app.post("/auth/logout")
    def logout(request: Request):
        return logout_route(request)

    # ── Bot API routes (require auth) ────────────────────────────────────────

    @app.get("/api/status")
    def status(user: dict = Depends(_require_auth)):
        """Bot health, mode, and current cycle info."""
        try:
            raw = _engine.exchange.fetch_balance()
            balance_usdt = float(raw.get("USDT", {}).get("free") or _engine.last_balance)
            _engine.last_balance = balance_usdt
        except Exception:
            balance_usdt = _engine.last_balance

        return {
            "running":          True,
            "testnet":          _engine.cfg.testnet,
            "kill_switch":      _engine.kill_switch,
            "timeframe":        _engine.cfg.timeframe,
            "last_cycle_ts":    _engine.last_cycle_ts,
            "last_cycle_ago_s": round(time.time() - _engine.last_cycle_ts, 1)
                                if _engine.last_cycle_ts else None,
            "balance_usdt":     balance_usdt,
            "server_time":      datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/positions")
    def positions(user: dict = Depends(_require_auth)):
        """All currently open positions with live unrealised P&L."""
        snapshot = _engine.positions.snapshot()
        result = []
        for sym, pos in snapshot.items():
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
    def trades(limit: int = 100, user: dict = Depends(_require_auth)):
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
    def summary(user: dict = Depends(_require_auth)):
        """Aggregate P&L stats across all closed trades."""
        return _engine.trade_log.summary()

    @app.post("/api/kill")
    def kill(user: dict = Depends(_require_auth)):
        """Engage kill switch — stops new entries and closes all open positions."""
        _engine.kill_switch = True
        log.warning(f"Kill switch ENGAGED by {user['username']}")
        return {"ok": True, "message": "Kill switch engaged. Open positions will be closed on next cycle."}

    @app.post("/api/resume")
    def resume(user: dict = Depends(_require_auth)):
        """Disengage kill switch — bot resumes normal operation."""
        _engine.kill_switch = False
        log.info(f"Kill switch DISENGAGED by {user['username']}")
        return {"ok": True, "message": "Bot resumed."}

    @app.post("/api/close-all")
    def close_all(user: dict = Depends(_require_auth)):
        """Immediately close all open positions (does not pause the bot)."""
        count = len(_engine.positions)
        if count == 0:
            return {"ok": True, "message": "No open positions to close.", "closed": 0}
        log.warning(f"Close-all triggered by {user['username']}")
        _engine.close_all_positions(reason="manual")
        return {"ok": True, "message": f"Closed {count} position(s).", "closed": count}

    @app.post("/api/close/{symbol:path}")
    def close_one(symbol: str, user: dict = Depends(_require_auth)):
        """Close a single position by symbol (e.g. BTC/USDT)."""
        if symbol not in _engine.positions:
            raise HTTPException(status_code=404, detail=f"{symbol} not in open positions")
        pos = _engine.positions[symbol]
        try:
            ticker = _engine.exchange.fetch_ticker(symbol)
            price = float(ticker["last"] or pos.entry_price)
        except Exception:
            price = pos.entry_price
        _engine._close_position(symbol, price, reason="manual")
        log.info(f"Manually closed {symbol} @ {price} by {user['username']}")
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

    log.info(f"Dashboard API listening on {host}:{port}")
    return thread
