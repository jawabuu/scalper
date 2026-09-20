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
_scanner = None
_guardian = None
_auto = None


def _instance_identity() -> dict:
    """Who is actually answering this request."""
    import os
    import socket
    demo = getattr(_guardian, "demo", None)
    return {
        "host": socket.gethostname(),
        "service": os.environ.get("INSTANCE_NAME", ""),
        "futures_env": ("unknown" if demo is None else ("demo" if demo else "live")),
        "state_owner": getattr(_guardian, "state_owner", ""),
    }


_auto_error: str = ""
_peer_eval = None


def set_peer_eval(pe):
    """Attach the cross-evaluation service."""
    global _peer_eval
    _peer_eval = pe


def set_auto_trader(auto):
    """Attach the auto-trader so the dashboard can toggle and inspect it."""
    global _auto
    _auto = auto


def set_auto_trader_error(msg: str):
    """
    Record why the auto-trader is absent, so the dashboard can SAY so.

    Without this the failure is one ERROR line in a startup log: the scanner
    scans, the guardian guards, the dashboard looks healthy, and nothing opens.
    """
    global _auto_error
    _auto_error = str(msg or "")
_entry = None


def set_entry_service(svc):
    """Attach the EntryService that backs the UI entry button."""
    global _entry
    _entry = svc


def set_guardian(guardian):
    """Attach a FuturesGuardian so the dashboard can display its state."""
    global _guardian
    _guardian = guardian


def set_scanner(runner):
    """Attach a ScanRunner so the dashboard can display its candidates."""
    global _scanner
    _scanner = runner


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

    @app.post("/api/futures/entry/preview")
    def entry_preview(payload: dict, user: dict = Depends(_require_auth)):
        """Step 1: compute a plan and return a single-use confirm token. Sends nothing."""
        if _entry is None:
            raise HTTPException(status_code=400, detail="Futures entry not enabled")
        try:
            res = _entry.preview(
                symbol=str(payload.get("symbol") or ""),
                side=str(payload.get("side") or ""),
                margin_pct=payload.get("margin_pct"),
                callback_pct=payload.get("callback_pct"),
            )
        except Exception as e:
            log.exception("entry preview failed")
            return {"ok": False, "errors": [f"{type(e).__name__}: {e}"]}
        log.info(f"Entry preview by {user['username']}: {payload} -> ok={res.get('ok')}")
        return res

    @app.post("/api/futures/entry/execute")
    def entry_execute(payload: dict, user: dict = Depends(_require_auth)):
        """Step 2: place the entry, but only with a valid unexpired token."""
        if _entry is None:
            raise HTTPException(status_code=400, detail="Futures entry not enabled")
        token = str(payload.get("token") or "")
        try:
            res = _entry.execute(token)
        except Exception as e:
            log.exception("entry execute failed")
            return {"ok": False, "errors": [f"{type(e).__name__}: {e}"]}
        log.warning(f"Entry execute by {user['username']}: ok={res.get('ok')} "
                    f"dry_run={res.get('dry_run')}")
        return res

    @app.get("/api/guardian")
    def guardian(user: dict = Depends(_require_auth)):
        """Futures guardian state: tracked positions, stop levels, recent actions."""
        if _guardian is None:
            return {"enabled": False, "states": {}, "recent_actions": [],
                    "message": "Guardian not enabled on this instance"}
        try:
            snap = _guardian.snapshot()
            snap["entry_enabled"] = _entry is not None
            return snap
        except Exception as e:
            log.exception("guardian snapshot failed")
            return {"enabled": True, "states": {}, "recent_actions": [],
                    "config": {}, "error": f"{type(e).__name__}: {e}"}

    @app.post("/api/futures/close")
    def futures_close(payload: dict, user: dict = Depends(_require_auth)):
        """Close an open futures position at market (reduce-only)."""
        if _guardian is None:
            raise HTTPException(status_code=400, detail="Guardian not enabled")
        symbol = (payload or {}).get("symbol")
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol is required")
        try:
            res = _guardian.close_position(symbol)
        except Exception as e:
            log.exception("futures close failed")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        log.warning(f"Futures close {symbol} by {user['username']}: ok={res.get('ok')} "
                    f"dry_run={res.get('dry_run')}")
        return res

    @app.get("/api/analysis")
    def analysis(user: dict = Depends(_require_auth)):
        """
        Which entry conditions actually paid, from recorded trades.

        Reports sample size alongside every result — a striking difference
        across a handful of trades is noise, and this is exactly where a
        strategy gets overfitted.
        """
        if _guardian is None:
            return {"enabled": False, "message": "Guardian not enabled"}
        try:
            from bot.analysis import (analyse, reconcile, account_return,
                                      day_report)
            trades = _guardian.closed_trades()
            report = analyse(trades)
            try:
                day_start = None
                day_base = None
                if _auto is not None:
                    day_base = getattr(_auto.state, "day_start_balance", None)
                # Start of the current LOCAL day, matching the daily-loss
                # window. The boundary is DAY_TZ_OFFSET_H hours east of UTC so
                # the day an operator sees is their own.
                import time as _t
                from bot.auto_trader import day_start_ts as _day_start
                day_start = _day_start(_t.time())
                report["account_return"] = account_return(
                    trades,
                    baseline=getattr(_guardian, "wallet_start", None),
                    wallet_now=_account_wallet(_guardian),
                    day_baseline=day_base,
                    day_start_ts=day_start)
                # Its own card: account_return has one subtitle line and gives
                # it to the wallet-gap warning when the two disagree, which on
                # demo is nearly permanent — so the daily figure disappeared
                # exactly where it was wanted.
                from bot.auto_trader import DAY_TZ_OFFSET_H
                # The dashboard is a static nginx file, so a server-side
                # default cannot be templated into it — it travels here.
                from bot.config import BotConfig as _BC
                report["ui"] = {
                    "hide_amounts_default": bool(_BC().hide_card_amounts)}
                report["day"] = day_report(
                    trades,
                    day_baseline=day_base,
                    day_start_ts=(getattr(_auto.state, "day_started_at", 0.0)
                                  if _auto is not None else 0.0) or day_start,
                    wallet_now=_account_wallet(_guardian),
                    tz_offset_h=DAY_TZ_OFFSET_H,
                    baseline_source=(getattr(_auto.state, "day_baseline_source", "")
                                     if _auto is not None else ""))
            except Exception as e:
                report["account_return"] = {"error": str(e)}
                report["day"] = {"error": str(e)}
            try:
                report["reconciliation"] = reconcile(
                    trades,
                    wallet_now=_account_wallet(_guardian),
                    wallet_start=getattr(_guardian, "wallet_start", None))
            except Exception as e:
                report["reconciliation"] = {"error": str(e)}
            report["enabled"] = True
            return report
        except Exception as e:
            log.exception("analysis failed")
            return {"enabled": True, "error": f"{type(e).__name__}: {e}"}

    @app.get("/api/futures/orders/diagnose")
    def futures_orders_diagnose(user: dict = Depends(_require_auth)):
        """What each candidate order-listing call returns, with URLs."""
        if _guardian is None:
            return {"enabled": False}
        try:
            return _guardian.diagnose_order_listing()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    @app.get("/api/futures/orders")
    def futures_orders(user: dict = Depends(_require_auth)):
        """Resting orders reconciled against positions — orphans made visible."""
        if _guardian is None:
            return {"enabled": False}
        try:
            rep = _guardian.reconcile_orders()
            rep["enabled"] = True
            return rep
        except Exception as e:
            log.exception("order reconciliation failed")
            return {"enabled": True, "error": f"{type(e).__name__}: {e}"}

    @app.get("/api/futures/trades")
    def futures_trades(user: dict = Depends(_require_auth)):
        """Closed futures positions observed by the guardian."""
        if _guardian is None:
            return {"enabled": False, "trades": []}
        try:
            return {"enabled": True, "trades": _guardian.closed_trades()}
        except Exception as e:
            log.exception("futures trades failed")
            return {"enabled": True, "trades": [], "error": f"{type(e).__name__}: {e}"}

    @app.get("/api/futures/trades.csv")
    def futures_trades_csv(user: dict = Depends(_require_auth)):
        """
        The full closed-trade record as CSV, including every recorded field.

        The dashboard exports the same thing client-side; this exists so a run
        can be pulled with curl without opening a browser.
        """
        from fastapi.responses import PlainTextResponse
        if _guardian is None:
            return PlainTextResponse("", media_type="text/csv")
        trades = _guardian.closed_trades()
        if not trades:
            return PlainTextResponse("", media_type="text/csv")

        keys, ctx_keys = [], []
        for t in trades:
            for k in t:
                if k != "entry_context" and k not in keys:
                    keys.append(k)
            for k in (t.get("entry_context") or {}):
                if k not in ctx_keys:
                    ctx_keys.append(k)

        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(keys + [f"ctx_{k}" for k in ctx_keys])
        for t in trades:
            ctx = t.get("entry_context") or {}
            w.writerow([t.get(k) for k in keys] + [ctx.get(k) for k in ctx_keys])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")

    @app.post("/api/peer/evaluate")
    def peer_evaluate(payload: dict):
        """
        INTERNAL, UNAUTHENTICATED, EVALUATION ONLY.

        Reachable only on the Docker network between the two containers; it is
        not exposed by nginx. It reads the scan snapshot this instance already
        holds and returns a verdict — it places no orders, changes no config
        and makes no exchange calls, so the worst a caller can do is fill the
        report file. Rate-limited to 60/min for that reason.
        """
        pe = _peer_eval
        if pe is None:
            return {"verdict": "disabled", "detail": "peer eval not configured"}
        return pe.evaluate(payload, _scanner, _auto)

    @app.post("/api/baselines/reset")
    def reset_baselines(payload: dict, user: dict = Depends(_require_auth)):
        """
        Re-base TODAY and/or ACCOUNT RETURN without clearing trade history.

        Both baselines are sticky by design — today's is cached per day so it
        stops drifting, and wallet_start is written once and persisted. That
        stickiness is right until a figure is WRONG: a baseline reconstructed
        at an unlucky moment, or a deposit that makes the old start
        meaningless. Until now the only way out was wiping history, which
        throws away the trades to fix a single number.

        Nothing here touches trades, orders or positions.
        """
        done = {}
        if payload.get("day", True):
            try:
                from bot.analysis import _DAY_BASELINE, day_report
                # DAY_TZ_OFFSET_H is imported per-function in this module, so
                # it must be imported HERE too — it is not module-level.
                from bot.auto_trader import DAY_TZ_OFFSET_H
                import time as _t
                st = getattr(_auto, "state", None) if _auto else None
                # WRITE a baseline, do not clear a cache.
                #
                # The old version cleared _DAY_BASELINE and zeroed
                # day_start_balance. Clearing a cache of a deterministic
                # function recomputes the identical number, so the card never
                # changed — while zeroing day_start_balance silently moved the
                # DAILY HALT's threshold to the current balance, erasing the
                # day's drawdown with no visible sign.
                #
                # Now it computes today's 00:00 balance once, from the wallet
                # and the P&L since midnight, and STORES it. It stops drifting
                # because nothing recomputes it, and the halt and the card
                # then share one figure.
                _DAY_BASELINE.clear()
                wallet = getattr(_guardian, "_wallet_balance_cached", 0.0) or 0.0
                started = getattr(st, "day_started_at", 0.0) if st else 0.0
                base = None
                if st is not None and wallet > 0 and started:
                    rep = day_report(
                        (_guardian.closed_trades() if _guardian else []),
                        day_baseline=None, day_start_ts=started,
                        wallet_now=wallet, tz_offset_h=DAY_TZ_OFFSET_H)
                    net = float(rep.get("net_pnl") or 0.0)
                    base = wallet - net
                if st is not None and base and base > 0:
                    prev = st.day_start_balance
                    st.day_start_balance = float(base)
                    st.day_baseline_at = _t.time()
                    st.day_baseline_source = "operator"
                    done["day"] = (
                        f"{prev:.2f} -> {base:.2f} (stored; the daily halt "
                        f"now measures from this figure too)")
                elif st is None:
                    done["day"] = "no auto-trader"
                else:
                    done["day"] = ("wallet or day start unavailable — "
                                   "nothing changed")
            except Exception as e:
                done["day"] = f"failed: {e}"
        if payload.get("account"):
            try:
                g = _guardian
                if g is None:
                    done["account"] = "no guardian"
                else:
                    # An EXPLICIT value wins. "Re-base to now" is only right
                    # when the run genuinely starts now; after a deposit, or
                    # when reconstructing a run whose true start is known, the
                    # operator has the correct figure and the bot does not.
                    given = payload.get("account_value")
                    bal = None
                    if given is not None:
                        try:
                            bal = float(given)
                        except (TypeError, ValueError):
                            bal = None
                        if bal is None or bal <= 0:
                            done["account"] = (
                                f"rejected {given!r} — must be a positive "
                                f"number")
                            bal = None
                        else:
                            src = "set explicitly"
                    if bal is None and given is None:
                        # The guardian caches the balance each cycle; there is
                        # no wallet_balance() method.
                        bal = getattr(g, "_wallet_balance_cached", 0.0) or None
                        src = "current wallet"
                    if bal:
                        prev = g.wallet_start
                        g.wallet_start = float(bal)
                        g.save_state()
                        done["account"] = (
                            f"{prev if prev else 'unset'} -> {bal:.2f} ({src})")
                    elif "account" not in done:
                        done["account"] = "wallet balance unavailable"
            except Exception as e:
                done["account"] = f"failed: {e}"
        log.warning(f"Baselines reset by {user['username']}: {done}")
        return {"ok": True, "result": done}

    @app.post("/api/peer/compare")
    def peer_compare(payload: dict):
        """Pair the peer's candidate readings with ours. Same guarantees as
        /evaluate: no orders, no config change, no exchange calls."""
        pe = _peer_eval
        if pe is None:
            return {"paired": 0, "detail": "disabled"}
        return pe.compare(payload, _scanner)

    @app.get("/api/peer/calibration")
    def peer_calibration(limit: int = 5000,
                         user: dict = Depends(_require_auth)):
        pe = _peer_eval
        if pe is None:
            return {"enabled": False}
        return {"enabled": True, "label": pe.label,
                "calibration": pe.calibration(limit=limit)}

    @app.get("/api/peer/report")
    def peer_report(limit: int = 500, user: dict = Depends(_require_auth)):
        pe = _peer_eval
        if pe is None:
            # "not enabled" covered both "never constructed" and "mode=off",
            # which are diagnosed completely differently.
            return {"enabled": False, "rows": [],
                    "detail": ("peer eval was not constructed at startup — "
                               "check the log for 'peer eval failed to start'")}
        st = pe.status()
        if not (pe.sends or pe.receives):
            return {"enabled": False, "status": st, "rows": [],
                    "detail": (f"PEER_EVAL_MODE={st['mode']!r}"
                               + ("; PEER_EVAL_URL is empty so it cannot send"
                                  if not st.get("peer") else ""))}
        return {"enabled": True, "status": st, "rows": pe.report(limit=limit)}

    @app.get("/api/auto-trade")
    def auto_trade_status(user: dict = Depends(_require_auth)):
        if _auto is None:
            return {"available": False, "enabled": False,
                    "start_error": _auto_error,
                    "message": (f"AUTO-TRADE FAILED TO START: {_auto_error}"
                                if _auto_error
                                else "Auto-trade not configured on this instance")}
        snap = _auto.snapshot()
        snap["available"] = True
        return snap

    @app.post("/api/auto-trade")
    def auto_trade_set(payload: dict, user: dict = Depends(_require_auth)):
        """Toggle unattended trading, or clear a daily-loss halt."""
        if _auto is None:
            raise HTTPException(status_code=400, detail="Auto-trade not configured")
        rules = payload.get("rules")
        if rules:
            try:
                applied, errors = _auto.update_rules(rules)
            except HTTPException:
                raise
            except Exception as e:
                # A rule the validator cannot handle used to escape as a 500
                # with a stack trace in the log and nothing useful on the
                # page. A bad value is a 400 carrying its reason.
                log.error(f"rule update failed for {list(rules)}: {e}",
                          exc_info=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"could not apply {', '.join(map(str, rules))}: {e}")
            if errors:
                raise HTTPException(status_code=400, detail="; ".join(errors))
            log.warning(f"Auto-trade rules changed by {user['username']}: {applied}")
            snap = _auto.snapshot()
            snap["available"] = True
            snap["applied"] = applied
            return snap

        if payload.get("reset_halt"):
            log.warning(f"Auto-trade halt cleared by {user['username']}")
            snap = _auto.reset_halt()
        else:
            on = bool(payload.get("enabled"))
            log.warning(f"Auto-trade {'ENABLED' if on else 'DISABLED'} by {user['username']}")
            snap = _auto.set_enabled(on)
        snap["available"] = True
        return snap

    def _account_wallet(g):
        """
        What the RETURN cards measure: USDT plus the fee reserve at cost.

        The USDT balance alone cannot see a fee paid in BNB. LSK 2026-09-20,
        the only trade on a clean account, proved it: the USDT wallet moved
        -0.2273 — exactly the closing PnL — while the 0.0395 fee left the BNB
        balance. That 0.0395 was the whole reported gap.

        Degrades to the USDT figure on any failure, never to a smaller number,
        so a price-lookup blip cannot read as a loss.
        """
        if g is None:
            return None
        try:
            av = g.account_value()
            v = float(av.get("value") or 0.0)
            if v > 0:
                return v
        except Exception:
            pass
        return getattr(g, "_wallet_balance_cached", None)

    @app.get("/api/shadow")
    def api_shadow(limit: int = 500, symbol: str = "", verdict: str = "",
                   since: float = 0.0, user: dict = Depends(_require_auth)):
        """
        The shadow decision log — jev's advisory ENTER/SKIP verdict on every
        AUTO-ENTRY candidate, entered or refused, recorded before the outcome
        was known. Per JEV-BRIEF.md §5b: modelled on /api/scan, same four
        rules — auth required, never 500, degrade to enabled: false, and
        read-only (a GET here never calls the model, only reads the log
        bot/shadow_decision.py already wrote).
        """
        shadow = getattr(_auto, "shadow", None) if _auto is not None else None
        if shadow is None:
            return {"enabled": False, "decisions": [], "summary": {},
                    "message": "Shadow decision log not enabled on this "
                              "instance (SHADOW_ENABLED)"}
        try:
            from .shadow_decision import read_decisions, summarize
            decisions = read_decisions(
                shadow.path, limit=limit, symbol=symbol or None,
                verdict=verdict or None, since=since or None)
            closed = (_guardian.closed_trades()
                     if _guardian is not None else [])
            summary = summarize(decisions, closed)
        except Exception as e:
            log.exception("shadow decision read failed")
            return {"enabled": True, "decisions": [], "summary": {},
                    "error": f"{type(e).__name__}: {e}"}
        return {
            "enabled": True,
            "decisions": decisions,
            "summary": summary,
            "config": {"model": shadow.model, "path": str(shadow.path)},
        }

    @app.get("/api/scan")
    def scan(user: dict = Depends(_require_auth)):
        """
        Current candidate list from the read-only market screen.

        These are INDICATORS OF POTENTIAL for operator review — the scanner
        holds no keys and never trades.
        """
        if _scanner is None:
            return {"enabled": False, "candidates": [],
                    "message": "Scanner not enabled on this instance"}
        # Never let a snapshot fault become an opaque 500 — return the error so
        # it is visible in the dashboard and diagnosable without server logs.
        try:
            snap = _scanner.snapshot()
        except Exception as e:
            log.exception("scanner snapshot failed")
            return {"enabled": True, "candidates": [], "config": {},
                    "error": f"{type(e).__name__}: {e}"}
        snap["enabled"] = True
        return snap

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
            # Which backend answered. When two instances are fronted by
            # separate hostnames, a swapped nginx upstream is otherwise
            # invisible — the UI looks right while showing another bot's data.
            "instance":         _instance_identity(),
            "testnet":          _engine.cfg.testnet,
            "strategy":         _engine.cfg.strategy,
            "kill_switch":      _engine.kill_switch,
            # Whether cross-instance evaluation is actually doing anything, so
            # the UI can hide its Peer control rather than offering a button
            # that only ever explains it is off. PEER_EVAL_MODE=off -> False.
            "peer_eval_enabled": bool(
                _peer_eval is not None
                and (getattr(_peer_eval, "sends", False)
                     or getattr(_peer_eval, "receives", False))),
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
            # Pullback strategy settings (read-only; set via env, shown in the UI so
            # the operator can see what's actually governing entries on this instance).
            "pullback": {
                "gainer_enabled":     _engine.pb_gainer_enabled,
                "dipper_enabled":     _engine.pb_dipper_enabled,
                "rsi_min":            _engine.pb_rsi_min,
                "rsi_max":            _engine.pb_rsi_max,
                "rsi_rising_lookback":_engine.pb_rsi_rising_lookback,
                "ema_fast":           _engine.cfg.pb_ema_fast,
                "ema_slow":           _engine.cfg.pb_ema_slow,
                "ema_buffer_pct":     _engine.cfg.pb_ema_buffer_pct,
                "vol_floor_pct":      _engine.pb_vol_floor_pct,
                "candle_pos_max":     _engine.pb_candle_pos_max,
                "upper_wick_max":     _engine.pb_upper_wick_max,
                "low_proximity_pct":  _engine.pb_low_proximity_pct,
                "max_wick_stop_pct":  _engine.pb_max_wick_stop_pct,
                "min_volume_usdt":    _engine.pb_min_volume_usdt,
                "timeout_candles":    _engine.pb_timeout_candles,
                "session_windows":    _engine.pb_session_windows or "always active",
                "session_tz_offset":  _engine.cfg.pb_session_tz_offset,
            } if _engine.cfg.strategy == "pullback" else None,
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
                "strategy":    getattr(t, "strategy", "breakout"),
                "regime":      getattr(t, "regime", None),
                "peak_pnl_pct": getattr(t, "peak_pnl_pct", 0.0),
                "entry_stamps": getattr(t, "entry_stamps", {}) or {},
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

    @app.post("/api/pullback")
    def set_pullback(payload: dict, user: dict = Depends(_require_auth)):
        """
        Update live-editable pullback entry-tuning knobs (in-memory, not persisted;
        take effect on the next scan). Structural params (EMA/MA periods, sizing)
        are env-only and NOT settable here. Body accepts any subset of the keys.
        Each is validated; invalid values are rejected without changing anything.
        """
        if _engine.cfg.strategy != "pullback":
            raise HTTPException(status_code=400, detail="not running the pullback strategy")

        # (attr, type, min, max) — bounds guard against fat-finger mistakes.
        spec = {
            "rsi_min":             (float, 0, 100),
            "rsi_max":             (float, 0, 100),
            "rsi_rising_lookback": (int, 1, 50),
            "vol_floor_pct":       (float, 0, 100),
            "candle_pos_max":      (float, 0.01, 1.0),
            "upper_wick_max":      (float, 0.0, 1.0),
            "low_proximity_pct":   (float, 0.0, 100),
            "max_wick_stop_pct":   (float, 0.01, 50),
            "min_volume_usdt":     (float, 0, 1e12),
            "timeout_candles":     (int, 1, 1000),
        }
        bool_keys = {"gainer_enabled", "dipper_enabled"}

        applied = {}
        for key, val in payload.items():
            if key in bool_keys:
                setattr(_engine, f"pb_{key}", bool(val))
                applied[key] = bool(val)
            elif key == "session_windows":
                # Empty string = always active; otherwise validated by the parser.
                _engine.pb_session_windows = str(val)
                applied[key] = str(val)
            elif key in spec:
                typ, lo, hi = spec[key]
                try:
                    v = typ(val)
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400, detail=f"{key} must be {typ.__name__}")
                if not (lo <= v <= hi):
                    raise HTTPException(status_code=400, detail=f"{key} must be in [{lo}, {hi}]")
                setattr(_engine, f"pb_{key}", v)
                applied[key] = v
            # unknown keys ignored

        # Coherence check: rsi_min < rsi_max after applying.
        if _engine.pb_rsi_min >= _engine.pb_rsi_max:
            raise HTTPException(status_code=400, detail="rsi_min must be < rsi_max")

        log.info(f"Pullback knobs updated by {user['username']}: {applied}")
        return {"ok": True, "applied": applied}

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
