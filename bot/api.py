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

from bot import __version__
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
        # Read the engine-maintained balance (refreshed every monitor pass, ~7s)
        # rather than fetching independently. Single source of truth: the available
        # USDT shown here is the same figure the engine uses, so the dashboard's
        # portfolio total (USDT + position values) reconciles consistently.
        balance_usdt = _engine.last_balance

        return {
            "running":          True,
            "version":          __version__,
            "testnet":          _engine.cfg.testnet,
            "kill_switch":      _engine.kill_switch,
            "trailing_activation_enabled": _engine.trailing_activation_enabled,
            "trailing_activation_pct":     _engine.trailing_activation_pct,
            "btc_filter_enabled":          _engine.btc_filter_enabled,
            "btc_trend_lookback":          _engine.btc_trend_lookback,
            "btc_trend_threshold_pct":     _engine.btc_trend_threshold_pct,
            "btc_regime":                  (_engine._btc_regime_cache or {}),
            "entry_timing_enabled":        _engine.entry_timing_enabled,
            "entry_timing_ema_len":        _engine.entry_timing_ema_len,
            "entry_timing_band_pct":       _engine.entry_timing_band_pct,
            "momentum_enabled":            _engine.momentum_enabled,
            "momentum_lookback":           _engine.momentum_lookback,
            "momentum_min_slope_pct":      _engine.momentum_min_slope_pct,
            "profit_lock_enabled":         _engine.profit_lock_enabled,
            "profit_lock_arm_pct":         _engine.profit_lock_arm_pct,
            "profit_lock_giveback_pct":    _engine.profit_lock_giveback_pct,
            "hard_stop_enabled":           _engine.hard_stop_enabled,
            "hard_stop_pct":               _engine.hard_stop_pct,
            "reentry_guard_enabled":       _engine.reentry_guard_enabled,
            "timeframe":        _engine.cfg.timeframe,
            "last_cycle_ts":    _engine.last_cycle_ts,
            "last_cycle_ago_s": round(time.time() - _engine.last_cycle_ts, 1)
                                if _engine.last_cycle_ts else None,
            "balance_usdt":     balance_usdt,
            "server_time":      datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/positions")
    def positions(user: dict = Depends(_require_auth)):
        """
        All currently open positions with live unrealised P&L.

        Reads the live price and P&L the ENGINE maintains (updated every monitor
        pass), rather than fetching its own ticker. This guarantees a single source
        of truth: the number shown here is exactly the number the engine acts on
        for the trailing stop and profit lock.
        """
        snapshot = _engine.positions.snapshot()
        result = []
        for sym, pos in snapshot.items():
            # Use the engine-maintained live values. Fall back to entry price only
            # if the monitor hasn't populated them yet (brand-new position).
            if pos.current_price and pos.current_price > 0:
                current_price = pos.current_price
                pnl_pct  = pos.pnl_pct
                pnl_usdt = pos.pnl_usdt
            else:
                current_price = pos.entry_price
                pnl_pct  = 0.0
                pnl_usdt = 0.0

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
                "peak_pnl_pct":  round(pos.peak_pnl_pct, 4),
                "backstop_type": pos.backstop_type,
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

    @app.post("/api/profit-lock")
    def set_profit_lock(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update the continuous profit lock (in-memory, not persisted).
        Body: {"enabled": bool, "arm_pct": float, "giveback_pct": float}
        Once P&L crosses arm_pct, a profit floor ratchets up with the peak and
        locks a rising fraction of the gain. Applies to all open and future
        positions on the next cycle.
        """
        enabled   = payload.get("enabled")
        arm_pct   = payload.get("arm_pct")
        giveback  = payload.get("giveback_pct")

        if enabled is not None:
            _engine.profit_lock_enabled = bool(enabled)

        if arm_pct is not None:
            try:
                ap = float(arm_pct)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="arm_pct must be a number")
            if ap <= 0 or ap > 50:
                raise HTTPException(status_code=400, detail="arm_pct must be between 0 and 50")
            _engine.profit_lock_arm_pct = ap

        if giveback is not None:
            try:
                gb = float(giveback)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="giveback_pct must be a number")
            if gb < 0 or gb > 5:
                raise HTTPException(status_code=400, detail="giveback_pct must be between 0 and 5")
            _engine.profit_lock_giveback_pct = gb

        log.info(
            f"Profit lock updated by {user['username']}: "
            f"enabled={_engine.profit_lock_enabled} "
            f"arm_pct={_engine.profit_lock_arm_pct} "
            f"giveback_pct={_engine.profit_lock_giveback_pct}"
        )
        return {
            "ok": True,
            "profit_lock_enabled": _engine.profit_lock_enabled,
            "profit_lock_arm_pct": _engine.profit_lock_arm_pct,
            "profit_lock_giveback_pct": _engine.profit_lock_giveback_pct,
        }

    @app.post("/api/hard-stop")
    def set_hard_stop(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update the hard stop-loss (in-memory, not persisted).
        Body: {"enabled": bool, "pct": float}
        Cuts a losing position at -pct% P&L, checked before the trailing stop.
        """
        enabled = payload.get("enabled")
        pct     = payload.get("pct")

        if enabled is not None:
            _engine.hard_stop_enabled = bool(enabled)

        if pct is not None:
            try:
                p = float(pct)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="pct must be a number")
            if p <= 0 or p > 50:
                raise HTTPException(status_code=400, detail="pct must be between 0 and 50")
            _engine.hard_stop_pct = p

        log.info(
            f"Hard stop updated by {user['username']}: "
            f"enabled={_engine.hard_stop_enabled} pct={_engine.hard_stop_pct}"
        )
        return {
            "ok": True,
            "hard_stop_enabled": _engine.hard_stop_enabled,
            "hard_stop_pct": _engine.hard_stop_pct,
        }

    @app.post("/api/reentry-guard")
    def set_reentry_guard(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update the smart re-entry guard (in-memory, not persisted).
        Body: {"enabled": bool}
        When on, refuses to re-enter a coin at a price higher than its last
        loss exit — stops the bot chasing a just-lost coin back up.
        """
        enabled = payload.get("enabled")
        if enabled is not None:
            _engine.reentry_guard_enabled = bool(enabled)
        log.info(
            f"Re-entry guard updated by {user['username']}: "
            f"enabled={_engine.reentry_guard_enabled}"
        )
        return {"ok": True, "reentry_guard_enabled": _engine.reentry_guard_enabled}

    @app.post("/api/momentum")
    def set_momentum(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update the momentum confirmation gate (in-memory, not persisted).
        Body: {"enabled": bool, "lookback": int, "min_slope_pct": float}
        Confirms the coin is rising at entry via raw-price slope. Applies to NEW
        entries from the next cycle; never affects open positions.
        """
        enabled    = payload.get("enabled")
        lookback   = payload.get("lookback")
        min_slope  = payload.get("min_slope_pct")

        if enabled is not None:
            _engine.momentum_enabled = bool(enabled)

        if lookback is not None:
            try:
                lb = int(lookback)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="lookback must be an integer")
            if lb < 1 or lb > 50:
                raise HTTPException(status_code=400, detail="lookback must be between 1 and 50")
            _engine.momentum_lookback = lb

        if min_slope is not None:
            try:
                ms = float(min_slope)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="min_slope_pct must be a number")
            if ms < 0 or ms > 10:
                raise HTTPException(status_code=400, detail="min_slope_pct must be between 0 and 10")
            _engine.momentum_min_slope_pct = ms

        log.info(
            f"Momentum gate updated by {user['username']}: "
            f"enabled={_engine.momentum_enabled} "
            f"lookback={_engine.momentum_lookback} "
            f"min_slope={_engine.momentum_min_slope_pct}"
        )
        return {
            "ok": True,
            "momentum_enabled": _engine.momentum_enabled,
            "momentum_lookback": _engine.momentum_lookback,
            "momentum_min_slope_pct": _engine.momentum_min_slope_pct,
        }

    @app.post("/api/entry-timing")
    def set_entry_timing(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update the per-coin entry-timing gate (in-memory, not persisted).
        Body: {"enabled": bool, "ema_len": int, "band_pct": float}
        Skips entries where price is extended above the fast EMA. Applies to
        NEW entries from the next cycle; never affects open positions.
        """
        enabled  = payload.get("enabled")
        ema_len  = payload.get("ema_len")
        band_pct = payload.get("band_pct")

        if enabled is not None:
            _engine.entry_timing_enabled = bool(enabled)

        if ema_len is not None:
            try:
                el = int(ema_len)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="ema_len must be an integer")
            if el < 2 or el > 100:
                raise HTTPException(status_code=400, detail="ema_len must be between 2 and 100")
            _engine.entry_timing_ema_len = el

        if band_pct is not None:
            try:
                bp = float(band_pct)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="band_pct must be a number")
            if bp < 0 or bp > 10:
                raise HTTPException(status_code=400, detail="band_pct must be between 0 and 10")
            _engine.entry_timing_band_pct = bp

        log.info(
            f"Entry-timing gate updated by {user['username']}: "
            f"enabled={_engine.entry_timing_enabled} "
            f"ema_len={_engine.entry_timing_ema_len} "
            f"band_pct={_engine.entry_timing_band_pct}"
        )
        return {
            "ok": True,
            "entry_timing_enabled": _engine.entry_timing_enabled,
            "entry_timing_ema_len": _engine.entry_timing_ema_len,
            "entry_timing_band_pct": _engine.entry_timing_band_pct,
        }

    @app.post("/api/btc-filter")
    def set_btc_filter(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update the BTC market-regime filter (in-memory, not persisted).
        Body: {"enabled": bool, "lookback": int, "threshold_pct": float}
        Only gates NEW entries when BTC short-term trend is falling; never
        affects open positions. Applies from the next cycle.
        """
        enabled   = payload.get("enabled")
        lookback  = payload.get("lookback")
        threshold = payload.get("threshold_pct")

        if enabled is not None:
            _engine.btc_filter_enabled = bool(enabled)

        if lookback is not None:
            try:
                lb = int(lookback)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="lookback must be an integer")
            if lb < 1 or lb > 50:
                raise HTTPException(status_code=400, detail="lookback must be between 1 and 50")
            _engine.btc_trend_lookback = lb

        if threshold is not None:
            try:
                th = float(threshold)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="threshold_pct must be a number")
            if th < 0 or th > 10:
                raise HTTPException(status_code=400, detail="threshold_pct must be between 0 and 10")
            _engine.btc_trend_threshold_pct = th

        log.info(
            f"BTC filter updated by {user['username']}: "
            f"enabled={_engine.btc_filter_enabled} "
            f"lookback={_engine.btc_trend_lookback} "
            f"threshold={_engine.btc_trend_threshold_pct}"
        )
        return {
            "ok": True,
            "btc_filter_enabled": _engine.btc_filter_enabled,
            "btc_trend_lookback": _engine.btc_trend_lookback,
            "btc_trend_threshold_pct": _engine.btc_trend_threshold_pct,
        }

    @app.post("/api/trailing-activation")
    def set_trailing_activation(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update the trailing-stop activation threshold (in-memory, not persisted).
        Body: {"enabled": bool, "pct": float}
        Applies to NEW positions entered after this change — not retroactively.
        """
        enabled = payload.get("enabled")
        pct     = payload.get("pct")

        if enabled is not None:
            _engine.trailing_activation_enabled = bool(enabled)

        if pct is not None:
            try:
                pct_val = float(pct)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="pct must be a number")
            if pct_val < 0 or pct_val > 50:
                raise HTTPException(status_code=400, detail="pct must be between 0 and 50")
            _engine.trailing_activation_pct = pct_val

        log.info(
            f"Trailing activation updated by {user['username']}: "
            f"enabled={_engine.trailing_activation_enabled} "
            f"pct={_engine.trailing_activation_pct}"
        )
        return {
            "ok": True,
            "trailing_activation_enabled": _engine.trailing_activation_enabled,
            "trailing_activation_pct": _engine.trailing_activation_pct,
        }

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
        # Take the position lock so this can't race the fast monitor or main cycle.
        with _engine._pos_lock:
            if symbol not in _engine.positions:
                raise HTTPException(status_code=404, detail=f"{symbol} already closed")
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
