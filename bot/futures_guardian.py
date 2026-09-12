"""
Futures guardian — exchange layer.

Connects the pure decision logic in futures_guard.py to a live Binance USD-M
futures account. It protects positions the OPERATOR opened; it never opens one.

SAFETY PROPERTIES (each enforced in code, not convention):

  1. Every order it places is reduceOnly=True. A reduce-only order can only
     shrink or close an existing position — it can never open one, and never
     increase exposure.
  2. It only ever cancels orders that pass is_protective_stop(): reduce-only,
     closing side, stop-typed. A trailing-stop ENTRY order (not reduce-only) is
     invisible to it and can never be cancelled.
  3. Stop replacement is PLACE-THEN-CANCEL: the new stop is resting before the
     old one is removed. A brief moment with two stops is harmless (both are
     reduce-only; the first to trigger closes the position and the other is
     cleaned up). A moment with ZERO stops on a leveraged position is not.
  4. DRY_RUN mode logs every intended action without sending it.
  5. Credential cross-wire check: refuses to start if testnet and live keys are
     identical, or if the resolved key does not match the declared environment.

It sets no leverage and no margin mode — the operator configures those on
Binance. The guardian only reads what is there.
"""
from __future__ import annotations

import logging
import threading
import time

import ccxt

from .futures_state import identity as futures_state_identity
from .futures_guard import (
    FuturesPosition, GuardState, GuardConfig,
    roi_pct, price_for_roi, stop_side,
    adopt_state, is_protective_stop, evaluate,
    callback_roi_at, trail_locks_in, is_armed, atr_stop_roi,
    trail_callback_price_pct,
)

log = logging.getLogger("futures_guardian")


def _safe_err(err) -> str:
    """Exchange error as code + message, never the signed request URL."""
    from bot.futures_entry import _safe_err as _f
    try:
        return _f(err)
    except Exception:
        return f"{type(err).__name__}"


def resolve_usdt_balance(bal: dict) -> tuple[float, str]:
    """
    Total USDT wallet balance from a ccxt futures balance payload.

    Single source of truth: the guardian displays this figure and the entry
    service sizes positions from it, so they must never resolve it differently.
    Several shapes are tried because the payload differs between live and demo.
    Returns (value, source) so the origin can be logged when a figure surprises.
    """
    candidates: list[tuple[str, object]] = []

    # ORDER MATTERS. Sizing must use REALISED equity.
    #
    # ccxt maps USDT.total to marginBalance = walletBalance + unrealised PnL,
    # so preferring it made the risk budget swing with the mark price of every
    # open position: winning trades inflated the budget and sized the NEXT
    # position larger, precisely when exposure was already highest. One
    # observed jump of +74% in 34 seconds scaled every subsequent entry 1.74x.
    #
    # totalWalletBalance excludes unrealised PnL and is the figure "0.5% of
    # the wallet" is supposed to mean.
    info = (bal or {}).get("info") or {}
    if isinstance(info, dict):
        candidates.append(("info.totalWalletBalance", info.get("totalWalletBalance")))
        assets = info.get("assets")
        if isinstance(assets, list):
            for a in assets:
                if (a or {}).get("asset") == "USDT":
                    candidates.append(("assets[USDT].walletBalance", a.get("walletBalance")))

    usdt = (bal or {}).get("USDT") or {}
    if isinstance(usdt, dict):
        # Fallbacks only: these include unrealised PnL.
        candidates.append(("USDT.total (margin balance)", usdt.get("total")))
        candidates.append(("USDT.free", usdt.get("free")))

    total_map = (bal or {}).get("total") or {}
    if isinstance(total_map, dict):
        candidates.append(("total.USDT", total_map.get("USDT")))

    if isinstance(info, dict):
        candidates.append(("info.availableBalance", info.get("availableBalance")))

    for source, raw in candidates:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val, source
    return 0.0, "unresolved"


def resolve_price(exchange, symbol: str) -> tuple[float, str]:
    """
    Current price for a symbol, from whichever field the endpoint provides.

    Single source of truth for pricing, shared by the guardian and the entry
    service. Payload shape varies between live and demo — some responses carry
    no `last` at all — so several fields are tried before falling back to the
    most recent candle close. Returns (price, source); price is 0.0 if nothing
    could be read, with the reason in the source string.
    """
    try:
        t = exchange.fetch_ticker(symbol) or {}
    except Exception as e:
        return 0.0, f"ticker error: {type(e).__name__}: {e}"

    for field in ("last", "close", "markPrice", "mark", "previousClose"):
        try:
            v = float(t.get(field))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v, f"ticker.{field}"

    # Some payloads only carry the book.
    try:
        bid, ask = float(t.get("bid")), float(t.get("ask"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2, "ticker.bid/ask mid"
    except (TypeError, ValueError):
        pass

    info = t.get("info") or {}
    for field in ("markPrice", "lastPrice", "indexPrice"):
        try:
            v = float(info.get(field))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v, f"ticker.info.{field}"

    # Last resort: the most recent candle close.
    try:
        raw = exchange.fetch_ohlcv(symbol, "1m", limit=2)
        if raw:
            v = float(raw[-1][4])
            if v > 0:
                return v, "ohlcv close"
    except Exception as e:
        return 0.0, f"no price field; ohlcv fallback failed: {e}"

    return 0.0, "no usable price field in ticker"


def _r2(v):
    return None if v is None else round(float(v), 2)


def _r3(v):
    return None if v is None else round(float(v), 3)


class FuturesGuardian:
    def __init__(self, guard_cfg: GuardConfig, *, api_key: str, api_secret: str,
                 demo: bool = True, dry_run: bool = True, atr_timeframe: str = "3m",
                 poll_interval: float = 5.0, socks_proxy: str | None = None):
        self.cfg = guard_cfg.validate()
        if self.cfg.fail_fast_s:
            log.info(
                f"Fail-fast ARMED: cut after {self.cfg.fail_fast_s:.0f}s when "
                f"peak <= {self.cfg.fail_fast_max_peak_roi:+.1f}% ROI and "
                f"current <= {-abs(self.cfg.fail_fast_loss_roi):.1f}% ROI")
        else:
            log.warning(
                "Fail-fast DISABLED (GUARD_FAIL_FAST_S is 0 or unset) — losing "
                "trades will run to their stop. Set it to enable early cuts.")
        self.demo = demo
        # ATR must be measured on the timeframe actually traded.
        self.atr_timeframe = atr_timeframe or "3m"
        # Identifies which environment owns the state file, so a demo and a
        # live container sharing a volume cannot load each other's state.
        self.state_owner = futures_state_identity(demo, api_key)
        self.dry_run = dry_run
        self.poll_interval = poll_interval

        params: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                # ccxt REFUSES fetchOpenOrders() without a symbol until this is
                # acknowledged, raising what looks like an ExchangeError but is
                # only a rate-limit warning. Without it the account-wide order
                # listing never runs, so the orphan sweep had no input at all.
                # The higher weight is acceptable: it runs once every 120s.
                "fetchOpenOrders": {"warnWithoutSymbol": False},
            },
        }
        if socks_proxy:
            params["proxies"] = {"http": socks_proxy, "https": socks_proxy}
        self.exchange = ccxt.binanceusdm(params)
        if demo:
            # ccxt removed sandbox/testnet support for binanceusdm; Binance now
            # offers "demo trading", which routes to demo-fapi.binance.com with
            # separate demo credentials. enable_demo_trading() swaps the API
            # URLs. Note ccxt refuses demo mode if sandbox mode is also on, so
            # the two must never both be set.
            try:
                self.exchange.enable_demo_trading(True)
                log.info("Futures DEMO trading enabled (demo-fapi.binance.com)")
            except AttributeError:
                # Older ccxt without demo support — fall back to the legacy call.
                log.warning("ccxt has no enable_demo_trading(); falling back to "
                            "set_sandbox_mode(). Upgrade ccxt if this fails.")
                self.exchange.set_sandbox_mode(True)

        self._init_runtime_state()

        mode = "DRY RUN (no orders sent)" if dry_run else "LIVE (places real orders)"
        log.warning(
            f"Futures guardian starting — {'DEMO' if demo else 'LIVE ACCOUNT'} — {mode} | "
            f"stop {-self.cfg.initial_stop_roi:+.0f}% ROI, arm +{self.cfg.arm_roi:.0f}% ROI, "
            f"native trail {self.cfg.trail_callback_pct:.2f}% price"
        )

    def _init_runtime_state(self):
        """
        All mutable runtime state in one place.

        __init__ calls this, and so can any construction path that bypasses it
        (test doubles). Keeping the attribute list in a single method stops the
        two from drifting apart and producing AttributeErrors that get silently
        swallowed by the per-position exception handler.
        """
        # symbol -> GuardState
        self._states: dict[str, GuardState] = {}
        self._lock = threading.RLock()
        self._last_cycle_ts: float = 0.0
        self._last_error: str | None = None
        self._wallet_balance_cached: float = 0.0
        self._actions: list[dict] = []   # recent actions, for the dashboard
        self._pos_meta: dict[str, dict] = {}   # symbol -> sizing snapshot
        self._closed_trades: list[dict] = []   # futures trade history
        # symbol -> (high, low, fetched_at). Some ticker payloads omit high/low,
        # so the 24h range is derived from candles as a fallback and cached —
        # the guardian polls every few seconds and must not refetch 24h of
        # klines that often.
        self._range_cache: dict[str, tuple[float, float, float]] = {}
        # symbol -> where the 24h range came from, or why it is missing. Shown
        # in the snapshot so an "n/a" is diagnosable without server logs.
        self._range_source: dict[str, str] = {}
        self._risk_overshoots: dict[str, dict] = {}
        self._capped_stop_reported: dict[str, float] = {}
        self._listing_warned: bool = False
        self.max_closed_trades: int = getattr(
            self, "max_closed_trades", self.DEFAULT_MAX_CLOSED_TRADES)
        self._empty_listings: int = 0
        self._balance_outliers: int = 0
        self._untracked_reported: dict = {}
        self._dropped_reported: dict = {}
        # Consecutive API failures across ALL endpoints. Each call used to
        # handle its own failure locally, so the bot had no concept of "I
        # cannot currently see the exchange" — and cancelled a filled order
        # two minutes after failing to read the order book at all.
        self._api_failures: int = 0
        self._blind_since: float | None = None
        self._pending_cancels: dict[str, list] = {}
        self._cancel_attempts: dict[str, int] = {}
        self._last_trade_fees: float | None = None
        # Ids confirmed cancelled or confirmed absent from both books. Nothing
        # here is ever retried — a successful cancel followed by a -2011 retry
        # was re-queueing orders that no longer existed.
        self._cancelled_ids: set = set()
        # Wallet at the first observation, so the reconciliation has a baseline
        # to measure against. Persisted with the rest of the state.
        self.wallet_start: float | None = getattr(self, "wallet_start", None)
        self._last_wide_probe: float = 0.0
        self._missing_counts: dict[str, int] = {}
        self._stop_source_reported: set = set()
        self._atr_cache: dict[str, tuple[float, float]] = {}   # sym -> (atr%, ts)
        # Extra order ids when a stop had to be split across the per-order cap.
        self._split_stop_ids: dict[str, list[str]] = {}
        # EVERY protective stop id placed per symbol, not just the current one.
        # Ratcheting replaces a stop each time it moves; a failed cancel used to
        # orphan that stop permanently, because only the latest id was tracked.
        # Three stops were left resting on one position for exactly that reason.
        self._all_stop_ids: dict[str, list[str]] = {}
        # Where restart-critical state is persisted. Empty disables it.
        self.state_path: str = getattr(self, "state_path", "")
        self.state_owner: str = getattr(self, "state_owner", "")
        self._entry_service = getattr(self, "_entry_service", None)
        # How long an unfilled entry order may rest before it is cancelled.
        self.entry_order_ttl_s: float = getattr(self, "entry_order_ttl_s", 900.0)
        # Poll interval used while an entry order is resting and could fill at
        # any moment. Shrinks the unprotected window after a fill.
        self.pending_poll_interval: float = getattr(
            self, "pending_poll_interval", 1.0)
        self.reap_untracked: bool = getattr(self, "reap_untracked", False)
        self.sweep_orphan_stops: bool = getattr(self, "sweep_orphan_stops", True)
        self._last_untracked_sweep: float = 0.0
        self.untracked_sweep_interval_s: float = 120.0

    # ── reading ─────────────────────────────────────────────────────────────

    def note_api_failure(self, where: str, err=None):
        """Record a failed exchange read and enter read-only if they pile up."""
        self._api_failures = getattr(self, "_api_failures", 0) + 1
        if self._api_failures == self.BLIND_AFTER_FAILURES:
            self._blind_since = time.time()
            log.error(
                f"BLIND: {self._api_failures} consecutive exchange failures "
                f"(latest at {where}). Suspending entries, cancellations and "
                f"reaping until reads succeed. Positions are still monitored.")

    def note_api_success(self):
        if getattr(self, "_api_failures", 0):
            if getattr(self, "_blind_since", None):
                log.warning(
                    f"Exchange reads recovered after "
                    f"{time.time() - self._blind_since:.0f}s blind — resuming.")
            self._api_failures = 0
            self._blind_since = None

    @property
    def is_blind(self) -> bool:
        """True when the bot cannot currently see the exchange."""
        return getattr(self, "_api_failures", 0) >= self.BLIND_AFTER_FAILURES

    def _log_dropped_position(self, raw, why: str):
        """Report a position row discarded during normalisation."""
        try:
            info = (raw or {}).get("info") or {}
            sym = (raw or {}).get("symbol") or info.get("symbol") or "?"
            key = f"{sym}:{why}"
            if self._dropped_reported.get(key):
                return
            self._dropped_reported[key] = True
            recovered = "RECOVERED" in why
            log.warning(
                f"POSITION ROW {'REPAIRED' if recovered else 'DISCARDED'} "
                f"{sym}: {why}. "
                f"positionAmt={info.get('positionAmt')!r} "
                f"entryPrice={info.get('entryPrice')!r} "
                f"positionSide={info.get('positionSide')!r} "
                f"ccxt side={(raw or {}).get('side')!r} "
                f"contracts={(raw or {}).get('contracts')!r}. "
                + ("" if recovered
                   else "If a position IS open on this symbol it is UNGUARDED."))
        except Exception as e:
            # Do NOT swallow: a silent reporter is how the original defect
            # stayed invisible.
            log.warning(f"could not report a discarded position row: {e}")

    def fetch_positions(self) -> list[FuturesPosition]:
        """Open positions with non-zero size, normalised to FuturesPosition."""
        try:
            raw = self.exchange.fetch_positions()
            self.note_api_success()
        except Exception as e:
            self.note_api_failure("fetch_positions", e)
            raise
        out = []
        for p in raw:
            try:
                contracts = float(p.get("contracts") or 0)
                if contracts == 0:
                    # Never drop a position silently: an unguarded position is
                    # the worst failure the system has, and this path used to
                    # produce no output at all.
                    self._log_dropped_position(p, "zero contracts")
                    continue
                side = (p.get("side") or "").lower()
                if side not in ("long", "short"):
                    # RECOVER rather than discard. On one-way mode Binance
                    # reports positionSide BOTH and the direction lives in the
                    # SIGN of positionAmt; if ccxt fails to derive it, throwing
                    # the row away leaves a real position unguarded.
                    amt = ((p.get("info") or {}).get("positionAmt")
                           or p.get("contracts"))
                    try:
                        amt = float(amt)
                    except (TypeError, ValueError):
                        amt = 0.0
                    if amt > 0:
                        side = "long"
                    elif amt < 0:
                        side = "short"
                    if side in ("long", "short"):
                        self._log_dropped_position(
                            p, f"side missing from ccxt — RECOVERED as {side}")
                    else:
                        self._log_dropped_position(p, "unrecognised side")
                        continue
                entry = float(p.get("entryPrice") or 0)
                if entry <= 0:
                    # Never drop a position silently: an unguarded position is
                    # the worst failure the system has, and this path used to
                    # produce no output at all.
                    self._log_dropped_position(p, "no entry price")
                    continue
                info = p.get("info") or {}
                # The `leverage` field has been observed to come back as 1 on
                # isolated positions, which silently scales every ROI and stop
                # distance by the leverage factor. Try several sources, but the
                # authoritative value is derived from notional/margin below.
                lev_raw = (p.get("leverage") or info.get("leverage")
                           or info.get("marginRatio") and None)
                try:
                    lev = int(float(lev_raw)) if lev_raw else 1
                except (TypeError, ValueError):
                    lev = 1

                notional = entry * abs(contracts)
                margin_raw = (p.get("initialMargin") or info.get("initialMargin")
                              or p.get("collateral") or info.get("isolatedMargin"))
                try:
                    margin = float(margin_raw) if margin_raw else 0.0
                except (TypeError, ValueError):
                    margin = 0.0
                if margin <= 0:
                    # Last resort: reconstruct from the reported leverage.
                    margin = notional / max(lev, 1)
                    log.warning(
                        f"{p.get('symbol')}: no usable margin field; reconstructed "
                        f"{margin:.4f} from leverage={lev}. ROI figures depend on "
                        f"this — verify against the Binance UI."
                    )

                # Binance reports updateTime in ms on positionRisk. On a
                # freshly opened position that is the fill time.
                upd = None
                try:
                    raw = (p.get("info") or {}).get("updateTime") or p.get("timestamp")
                    if raw:
                        upd = float(raw) / 1000.0
                except (TypeError, ValueError):
                    upd = None

                pos = FuturesPosition(
                    symbol=p["symbol"], side=side, entry_price=entry,
                    qty=abs(contracts), leverage=max(lev, 1), margin=margin,
                    updated_at=upd,
                )
                # Sanity-check the derived leverage against the reported one.
                eff = pos.effective_leverage
                if lev > 1 and abs(eff - lev) / lev > 0.2:
                    log.warning(
                        f"{pos.symbol}: derived leverage {eff:.1f}x differs from "
                        f"reported {lev}x — using derived (notional/margin)."
                    )
                out.append(pos)
            except Exception as e:
                log.debug(f"skipping unparseable position {p.get('symbol')}: {e}")
        return out

    def fetch_open_orders(self, symbol: str) -> list[dict]:
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            log.warning(f"fetch_open_orders failed for {symbol}: {e}")
            return []

    def mark_price(self, pos: FuturesPosition) -> float | None:
        return self._price_from_ticker(pos.symbol, self._ticker(pos.symbol))

    def _price_from_ticker(self, symbol: str, t: dict | None) -> float | None:
        if not t:
            price, source = resolve_price(self.exchange, symbol)
            if price <= 0:
                log.warning(f"{symbol}: could not read a price ({source})")
                return None
            return price
        for field in ("last", "close", "markPrice"):
            try:
                v = float(t.get(field))
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        price, source = resolve_price(self.exchange, symbol)
        if price <= 0:
            log.warning(f"{symbol}: could not read a price ({source})")
            return None
        return price

    RANGE_CACHE_TTL_S = 300.0
    # Consecutive cycles a position must be absent before it counts as closed.
    MISSING_CONFIRMATIONS = 3
    # Cancels are retried well past the position's life: an order that reports
    # -2011 may still be resting, so give it many chances before abandoning it.
    MAX_CANCEL_ATTEMPTS = 60
    # How many closed trades to keep. Analysis needs the whole run, not a
    # window; at ~100 trades a day this holds well over a month.
    DEFAULT_MAX_CLOSED_TRADES = 5000
    # Guard against a single bad balance reading scaling position size.
    BALANCE_JUMP_TOLERANCE = 0.25
    BALANCE_JUMP_CONFIRMATIONS = 3
    # Consecutive failures before the bot treats itself as unable to see the
    # exchange and stops taking destructive action.
    BLIND_AFTER_FAILURES = 3
    # Seconds of slack before the order-placed stamp when querying the income
    # ledger, so the entry fill's commission is inside the window.
    INCOME_LOOKBACK_PAD_S = 120.0
    # Give up on the expensive account-wide listing after this many empties,
    # then re-probe on a slow clock in case the environment starts reporting.
    EMPTY_LISTING_LIMIT = 5
    WIDE_PROBE_INTERVAL_S = 1800.0
    # Candle timeframe for ATR. Must match the timeframe the strategy trades:
    # a 15m ATR is ~2.2x a 3m ATR, so stops sized off 15m were more than twice
    # as wide as the 3m chart justified (FORM: 24.4% ROI where 10.9% was right).
    atr_timeframe: str = "3m"

    def _range_24h(self, symbol: str, ticker: dict | None) -> tuple[float | None, float | None]:
        """
        24h high/low for a symbol.

        Prefers the ticker (one field lookup, no extra call). Falls back to
        deriving them from 24h of candles when the ticker omits them — which the
        demo endpoint does — and caches the result so a 5-second poll loop does
        not refetch klines every cycle.
        """
        hi = (ticker or {}).get("high")
        lo = (ticker or {}).get("low")
        try:
            if hi and lo and float(hi) > float(lo):
                self._range_source[symbol] = "ticker"
                return float(hi), float(lo)
        except (TypeError, ValueError):
            pass

        cached = self._range_cache.get(symbol)
        if cached and (time.time() - cached[2]) < self.RANGE_CACHE_TTL_S:
            self._range_source[symbol] = "candles (cached)"
            return cached[0], cached[1]

        try:
            # 96 x 15m = 24h, one request, coarse enough for a daily range.
            raw = self.exchange.fetch_ohlcv(symbol, "15m", limit=96)
            if raw:
                highs = [r[2] for r in raw if r[2] is not None]
                lows = [r[3] for r in raw if r[3] is not None]
                if highs and lows:
                    h, l = float(max(highs)), float(min(lows))
                    self._range_cache[symbol] = (h, l, time.time())
                    self._range_source[symbol] = "candles"
                    return h, l
            self._range_source[symbol] = "empty candle response"
            log.warning(f"{symbol}: 24h range fallback returned no candles")
        except Exception as e:
            # WARNING, not debug: this was swallowed silently and the range just
            # showed "n/a" with no way to tell why.
            self._range_source[symbol] = f"{type(e).__name__}: {e}"
            log.warning(f"{symbol}: 24h range fallback failed: {type(e).__name__}: {e}")
        return None, None

    def atr_pct(self, symbol: str) -> float | None:
        """
        ATR as a % of price, from the same cached candle window used for the
        24h range. Cached because the guardian polls every few seconds and this
        needs a klines call.
        """
        now = time.time()
        cached = self._atr_cache.get(symbol)
        if cached and (now - cached[1]) < self.RANGE_CACHE_TTL_S:
            return cached[0]
        try:
            raw = self.exchange.fetch_ohlcv(symbol, self.atr_timeframe, limit=120)
            if not raw or len(raw) < 15:
                return None
            trs = []
            prev_close = None
            for _ts, _o, h, l, c, _v in raw:
                if None in (h, l, c):
                    continue
                tr = h - l
                if prev_close is not None:
                    tr = max(tr, abs(h - prev_close), abs(l - prev_close))
                trs.append(tr)
                prev_close = c
            if not trs or not prev_close:
                return None
            atr = sum(trs[-14:]) / min(len(trs), 14)
            pct = atr / prev_close * 100
            self._atr_cache[symbol] = (pct, now)
            return pct
        except Exception as e:
            log.warning(f"{symbol}: ATR lookup failed: {type(e).__name__}: {e}")
            return None

    def _check_risk_invariant(self, pos: FuturesPosition, stop_roi: float):
        """
        Assert the loss at the placed stop matches the configured risk budget.

        Sizing and stop placement are computed in different components at
        different times. When they disagree the position is silently oversized
        — one trade risked 2.8x its budget before anyone noticed. This is the
        backstop: it cannot prevent the mismatch, but it makes it loud the
        moment a stop is placed rather than when the loss lands.
        """
        budget_pct = getattr(self, "risk_pct", 0.0)
        wallet = self._wallet_balance_cached
        if not budget_pct or wallet <= 0 or not pos.margin:
            return
        budget = wallet * budget_pct / 100.0
        loss = pos.margin * abs(stop_roi) / 100.0
        if budget > 0 and loss > budget * 1.25:
            msg = (f"RISK OVERSHOOT: stop at {-abs(stop_roi):.1f}% ROI on "
                   f"{pos.margin:.2f} margin risks {loss:.2f} USDT, but the "
                   f"budget is {budget:.2f} ({budget_pct}% of {wallet:.2f}) "
                   f"— {loss/budget:.1f}x. Sizing and stop disagree.")
            log.error(f"{pos.symbol}: {msg}")
            self._record(pos.symbol, "RISK_OVERSHOOT",
                         f"{loss:.2f} at risk vs {budget:.2f} budget "
                         f"({loss/budget:.1f}x)")
            with self._lock:
                self._risk_overshoots[pos.symbol] = {
                    "loss_at_stop": round(loss, 2), "budget": round(budget, 2),
                    "ratio": round(loss / budget, 2), "stop_roi": round(stop_roi, 2),
                }

    # Ages at which the position's ROI is sampled, in seconds. 0 captures the
    # ROI at FIRST OBSERVATION: for a bot-placed entry the pending poll runs
    # every second, so this is the position's ROI before the market has had
    # time to move it — i.e. the fill-versus-trigger gap expressed in ROI.
    # peak_roi cannot carry this (it is clamped at 0 and only ratchets up) and
    # neither can trough_roi (it cannot separate "opened at -13%" from "opened
    # flat and fell to -13%").
    ROI_CHECKPOINTS_S = (0, 60, 180, 300)

    @staticmethod
    def _drift_pct(meta: dict, side: str) -> float | None:
        """
        Adverse price movement between sizing and the actual entry, as a % of
        the sized price. Positive is always against the position, whichever
        side it is, so the two directions can be pooled.
        """
        try:
            ctx = meta.get("entry_context") or {}
            sized = ctx.get("sized_price")
            entry = meta.get("entry_price")
            if not sized or not entry:
                return None
            sized, entry = float(sized), float(entry)
            if sized <= 0:
                return None
            move = (entry - sized) / sized * 100.0
            # Sign so that POSITIVE is favourable to the position. A short
            # filled ABOVE where it was sized sold higher, which is better; a
            # long filled BELOW bought cheaper. An earlier version had this
            # backwards and reported a good fill as "against the position".
            return round(move if str(side).lower() == "short" else -move, 3)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _should_fail_fast(self, pos, state, current_roi: float) -> bool:
        """
        Cut a trade that never went green and is now losing.

        The loss floor and the never-green test are always required. Time
        alone would cut trades that are merely slow; the floor targets "going
        wrong" rather than "not going right yet".

        One further gate, off by default:

        fail_fast_require_worsening  cut only what is worse than where it
                                   started, so a recovering position is left
                                   alone regardless of depth.
        """
        cfg = self.cfg
        if not getattr(cfg, "fail_fast_s", 0):
            return False
        if state.peak_roi > getattr(cfg, "fail_fast_max_peak_roi", 0.0):
            return False                      # it has been green — leave it
        floor = getattr(cfg, "fail_fast_loss_roi", 5.0)
        if current_roi > -abs(floor):
            return False                      # losing, but not badly

        # ROI at FIRST OBSERVATION. peak_roi cannot stand in for this: it is
        # clamped at 0, so a position first seen at -10.6% records peak 0.00
        # and looks identical to one that opened flat. The age-0 checkpoint
        # keeps the real figure.
        entry_roi = state.roi_checkpoints.get("0")

        # NOTE: an earlier version cut immediately when the age-0 checkpoint
        # was far underwater. That was wrong. roi_pct measures against
        # pos.entry_price, which IS the fill price, so ROI at the moment of
        # fill is identically zero. A non-zero age-0 reading therefore measures
        # how LATE the guardian looked, not how the trade opened — KOMA read
        # 0.00 because it was seen on the fill tick, GRIFFAIN -13.31 because it
        # was seen ~3s later. Cutting on it would cut on poll latency.

        # A position that is recovering is going the other way, whatever its
        # absolute ROI. RIVER reached -17.54% and closed +59.99%; cutting on
        # depth alone would have taken it. Only cut what is getting WORSE than
        # where it started.
        if getattr(cfg, "fail_fast_require_worsening", False):
            if entry_roi is not None and current_roi > entry_roi:
                return False

        opened = (self._pos_meta.get(pos.symbol, {}) or {}).get("opened_seen_at")
        if not opened:
            return False
        return (time.time() - float(opened)) >= cfg.fail_fast_s

    def _note_progress(self, pos, state, current_roi: float):
        """Record when the trade first went green, and its ROI at fixed ages."""
        now = time.time()
        if state.first_positive_at is None and current_roi > 0:
            state.first_positive_at = now
        opened = (self._pos_meta.get(pos.symbol, {}) or {}).get("opened_seen_at")
        if not opened:
            return
        age = now - float(opened)
        for mark in self.ROI_CHECKPOINTS_S:
            key = str(mark)
            if age >= mark and key not in state.roi_checkpoints:
                state.roi_checkpoints[key] = round(current_roi, 2)

    def _capture_entry_context(self, pos: FuturesPosition, price: float,
                               range_pos: float | None) -> dict:
        """
        Snapshot the conditions a position was entered under.

        Recorded once, on first sight, and never updated — the point is what was
        true AT ENTRY, so later analysis can ask which conditions actually paid.
        External entries are captured approximately (first observation rather
        than the true fill moment); auto-trade entries supply exact values via
        note_entry_context().
        """
        ctx = {
            "side": pos.side,
            "atr_pct": self.atr_pct(pos.symbol),
            "range_pos_24h": range_pos,
            "leverage": pos.effective_leverage,
            "margin": pos.margin,
            "captured": "observed",
        }
        meta = self._pos_meta.get(pos.symbol, {})
        hi, lo = meta.get("high_24h"), meta.get("low_24h")
        try:
            if hi and lo:
                ctx["dist_to_extreme_pct"] = (
                    abs((price - float(lo)) / float(lo) * 100) if pos.side == "long"
                    else abs((price - float(hi)) / float(hi) * 100))
        except (TypeError, ValueError):
            pass
        return ctx

    def note_entry_context(self, symbol: str, ctx: dict):
        """
        Record exact entry conditions from whoever opened the position.

        The auto-trader knows the RSI, distance and re-entry status it acted on;
        the guardian can only observe after the fact. Exact beats observed.
        """
        with self._lock:
            meta = self._pos_meta.setdefault(symbol, {})
            existing = meta.get("entry_context") or {}
            merged = {**existing, **ctx, "captured": "exact"}
            meta["entry_context"] = merged
        log.info(f"{symbol}: entry context recorded {ctx}")

    def effective_stop_roi(self, pos: FuturesPosition) -> float:
        """
        The initial stop distance to use for this position.

        With atr_stop_mult set, the stop scales with the coin's own volatility
        rather than being a constant — a fixed stop sits inside the noise on a
        volatile coin and needlessly far away on a calm one.
        """
        if not self.cfg.atr_stop_mult:
            return self.cfg.initial_stop_roi

        # Prefer the stop the position was actually SIZED for.
        #
        # A trailing-stop entry rests before it fills, so the guardian first
        # sees the position minutes after the entry sized it. Recomputing ATR
        # then can give a very different answer — a position sized for a 12.9%
        # stop received a 24.4% one, nearly 2x the intended risk. Sharing the
        # ATR cache only helps inside its TTL; the sized stop must be carried
        # with the position instead.
        with self._lock:
            sized = (self._pos_meta.get(pos.symbol, {})
                     .get("entry_context") or {}).get("sized_stop_roi")
        if sized:
            if pos.symbol not in self._stop_source_reported:
                self._stop_source_reported.add(pos.symbol)
                log.info(f"{pos.symbol}: stop {float(sized):.1f}% ROI from the "
                         f"SIZED handoff (entry and guardian agree)")
            return float(sized)

        if pos.symbol not in self._stop_source_reported:
            self._stop_source_reported.add(pos.symbol)
            log.warning(
                f"{pos.symbol}: no sized stop recorded — deriving from a fresh "
                f"ATR read. This position was opened before the handoff existed, "
                f"or its state was lost. The budget cap is the safety net."
            )

        a = self.atr_pct(pos.symbol)
        roi = atr_stop_roi(a, pos.effective_leverage, self.cfg)
        if roi is None:
            return self.cfg.initial_stop_roi
        return roi

    def _cap_stop_to_budget(self, pos: FuturesPosition, stop_roi: float) -> float:
        """
        Tighten a stop that would risk more than the configured budget.

        The budget and the position's ACTUAL margin uniquely determine the
        widest acceptable stop: loss = margin x stop_roi, so the cap is simply
        budget / margin. When sizing and stop placement disagree, this recovers
        the stop the position was really sized for rather than trusting an ATR
        reading taken at a different moment.

        Tightening is always safe in risk terms, but a tighter stop sits closer
        to the noise, so it is floored at atr_stop_min_roi. If even that floor
        exceeds the budget the position is simply too large for it, which is
        reported rather than silently accepted.
        """
        budget_pct = getattr(self, "risk_pct", 0.0)
        wallet = self._wallet_balance_cached
        if not budget_pct or wallet <= 0 or not pos.margin:
            return stop_roi

        budget = wallet * budget_pct / 100.0
        max_stop_roi = budget / pos.margin * 100.0
        if stop_roi <= max_stop_roi * 1.05:
            return stop_roi          # already within budget

        floor = self.cfg.atr_stop_min_roi or 0.0
        capped = max(max_stop_roi, floor)
        if capped >= stop_roi:
            return stop_roi

        # Margin drifts with unrealised PnL, so this recomputes every cycle and
        # would otherwise log an almost-identical line every few seconds. Report
        # it once, and again only if the level moves meaningfully.
        prev = self._capped_stop_reported.get(pos.symbol)
        if prev is None or abs(prev - capped) >= 0.5:
            log.warning(
                f"{pos.symbol}: stop {stop_roi:.1f}% ROI on {pos.margin:.2f} margin "
                f"would risk {pos.margin * stop_roi / 100:.2f} vs a {budget:.2f} "
                f"budget — tightening to {capped:.1f}% ROI "
                f"({pos.margin * capped / 100:.2f} at risk)."
            )
            self._record(pos.symbol, "stop_capped",
                         f"{stop_roi:.1f}% -> {capped:.1f}% ROI to hold risk at "
                         f"{budget:.2f} USDT")
            self._capped_stop_reported[pos.symbol] = capped
        if capped > max_stop_roi * 1.05:
            log.error(
                f"{pos.symbol}: even the {floor:.1f}% minimum stop risks "
                f"{pos.margin * capped / 100:.2f} vs a {budget:.2f} budget — "
                f"the position is too large for the risk setting."
            )
        return capped
        if abs(roi - self.cfg.initial_stop_roi) > 0.5:
            log.info(f"{pos.symbol}: ATR {a:.2f}% -> initial stop {roi:.0f}% ROI "
                     f"(fixed default would be {self.cfg.initial_stop_roi:.0f}%)")
        return roi

    def _ticker(self, symbol: str) -> dict | None:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            log.warning(f"ticker failed for {symbol}: {e}")
            return None

    # ── writing (the only two methods that mutate the account) ──────────────

    def market_max_qty(self, symbol: str) -> float | None:
        """
        The per-order quantity cap that applies to a MARKET-executing order.

        Binance publishes two caps: LOT_SIZE (limit orders) and
        MARKET_LOT_SIZE (market orders), and the market one is usually far
        smaller. Stops, trailing stops and entries here all execute as market,
        so the MARKET cap governs — but only LOT_SIZE was being read, so orders
        sized against it were rejected with -4005 "Quantity greater than max
        quantity".

        Returns the smaller of the two that are published.
        """
        try:
            m = self.exchange.market(symbol) or {}
            limits = m.get("limits") or {}
            caps = []
            for key in ("market", "amount"):
                mx = ((limits.get(key) or {}).get("max"))
                if mx:
                    caps.append(float(mx))
            # Fall back to the raw filter if ccxt did not surface it.
            if not caps:
                for f in ((m.get("info") or {}).get("filters") or []):
                    if f.get("filterType") in ("MARKET_LOT_SIZE", "LOT_SIZE"):
                        try:
                            caps.append(float(f.get("maxQty")))
                        except (TypeError, ValueError):
                            pass
            return min(caps) if caps else None
        except Exception:
            return None

    def _place_stop(self, pos: FuturesPosition, stop_price: float) -> str | None:
        """
        Place a reduce-only STOP_MARKET that closes `pos` at `stop_price`.

        reduceOnly=True is non-negotiable: it makes it impossible for this order
        to open or enlarge a position, whatever else goes wrong.
        """
        price_str = self.exchange.price_to_precision(pos.symbol, stop_price)
        side = stop_side(pos)

        # A position can be larger than the per-order cap. Placing one oversized
        # stop is rejected outright, so split it into several reduce-only stops
        # at the same trigger — together they still close the whole position.
        max_qty = self.market_max_qty(pos.symbol)
        if max_qty and pos.qty > max_qty:
            return self._place_split_stops(pos, stop_price, max_qty, side)

        qty_str = self.exchange.amount_to_precision(pos.symbol, pos.qty)

        if self.dry_run:
            log.info(f"[DRY RUN] would place {side} STOP_MARKET reduceOnly "
                     f"{qty_str} {pos.symbol} trigger={price_str}")
            return f"dry-{int(time.time()*1000)}"

        order = self.exchange.create_order(
            symbol=pos.symbol, type="STOP_MARKET", side=side,
            amount=float(qty_str), price=None,
            params={"stopPrice": float(price_str), "reduceOnly": True},
        )
        oid = str(order.get("id") or order.get("orderId") or "")
        log.info(f"Placed stop for {pos.symbol}: {side} STOP_MARKET "
                 f"trigger={price_str} id={oid}")
        return oid

    def _place_split_stops(self, pos: FuturesPosition, stop_price: float,
                           max_qty: float, side: str) -> str | None:
        """Place several reduce-only stops when one would exceed the order cap."""
        remaining = pos.qty
        ids: list[str] = []
        price_str = self.exchange.price_to_precision(pos.symbol, stop_price)
        n = 0
        while remaining > 0 and n < 20:
            chunk = min(remaining, max_qty)
            qty_str = self.exchange.amount_to_precision(pos.symbol, chunk)
            if float(qty_str) <= 0:
                break
            if self.dry_run:
                log.info(f"[DRY RUN] would place {side} STOP_MARKET reduceOnly "
                         f"{qty_str} {pos.symbol} trigger={price_str} (split)")
                ids.append(f"dry-split-{n}")
            else:
                order = self.exchange.create_order(
                    symbol=pos.symbol, type="STOP_MARKET", side=side,
                    amount=float(qty_str), price=None,
                    params={"stopPrice": float(price_str), "reduceOnly": True},
                )
                ids.append(str(order.get("id") or order.get("orderId") or ""))
            remaining -= float(qty_str)
            n += 1
        if not ids:
            return None
        log.warning(
            f"{pos.symbol}: position {pos.qty:g} exceeds the per-order cap "
            f"{max_qty:g} — placed {len(ids)} split stops at {price_str}")
        self._split_stop_ids[pos.symbol] = ids
        return ids[0]

    def _place_native_trail(self, pos: FuturesPosition) -> str | None:
        """
        Place Binance's own TRAILING_STOP_MARKET for the armed phase.

        Once this is resting the exchange tracks the peak continuously, tick by
        tick, so the trail no longer depends on how often the guardian polls —
        the sampling gap that let a spike go unnoticed disappears.

        callbackRate is a PRICE percentage, so its ROI cost scales with the
        position's leverage; that is why the lock-in check happens here rather
        than at startup.
        """
        lev = pos.effective_leverage
        cb = trail_callback_price_pct(lev, self.cfg)
        locked = trail_locks_in(lev, self.cfg)
        if locked <= 0:
            log.warning(
                f"{pos.symbol}: arming a {cb}% trail at {lev:.0f}x gives back "
                f"{callback_roi_at(lev, self.cfg):.0f}% ROI, but the arm level is "
                f"+{self.cfg.arm_roi:.0f}% — the trail would engage at "
                f"{locked:+.0f}% ROI (at or below entry). Keeping the fixed stop "
                f"instead. Raise GUARD_ARM_ROI or lower GUARD_TRAIL_CALLBACK_PCT."
            )
            return None

        qty_str = self.exchange.amount_to_precision(pos.symbol, pos.qty)
        side = stop_side(pos)
        if self.dry_run:
            log.info(f"[DRY RUN] would place {side} TRAILING_STOP_MARKET reduceOnly "
                     f"{qty_str} {pos.symbol} callbackRate={cb}%")
            return f"dry-trail-{int(time.time()*1000)}"

        order = self.exchange.create_order(
            symbol=pos.symbol, type="TRAILING_STOP_MARKET", side=side,
            amount=float(qty_str), price=None,
            params={"callbackRate": cb, "reduceOnly": True},
        )
        oid = str(order.get("id") or order.get("orderId") or "")
        log.info(
            f"{pos.symbol}: ARMED native trailing stop (callback {cb}% price = "
            f"{callback_roi_at(lev, self.cfg):.0f}% ROI at {lev:.0f}x), locks in "
            f"~{locked:+.0f}% ROI. id={oid}"
        )
        return oid

    def _cancel_superseded_stops(self, pos: FuturesPosition, keep: str | None):
        """Cancel protective stops for this symbol other than the current one."""
        ids = list(self._all_stop_ids.get(pos.symbol) or [])
        for oid in ids:
            if keep and oid == keep:
                continue
            if self._cancel_stop(pos, oid):
                self._all_stop_ids[pos.symbol] = [
                    i for i in self._all_stop_ids.get(pos.symbol, []) if i != oid]

    @staticmethod
    def _order_already_gone(err) -> bool:
        """
        -2011 "Unknown order sent" means the order does NOT exist — already
        filled, already cancelled, or from a previous run. It is a SUCCESSFUL
        outcome for a cancel, not a failure, and must not be reported as
        "still resting".
        """
        t = str(err).lower()
        return "-2011" in t or "unknown order" in t

    def _queue_pending_cancel(self, symbol: str, order_id: str):
        """
        Remember a cancel that did not take, so it survives the position
        closing and a container restart.

        Neither order listing nor the cancel result is trustworthy on this
        venue, so the only durable record of "this order should not exist" is
        the bot's own queue.
        """
        with self._lock:
            q = self._pending_cancels.setdefault(symbol, [])
            if order_id not in q:
                q.append(order_id)

    def _clear_pending_cancel(self, symbol: str, order_id: str):
        with self._lock:
            q = self._pending_cancels.get(symbol) or []
            self._pending_cancels[symbol] = [i for i in q if i != order_id]
            if not self._pending_cancels[symbol]:
                self._pending_cancels.pop(symbol, None)
        self._cancel_attempts.pop(order_id, None)

    def drain_pending_cancels(self) -> int:
        """
        Retry every cancel that has not yet succeeded, including for symbols
        with no open position. Runs on the sweep clock.
        """
        with self._lock:
            work = {k: list(v) for k, v in self._pending_cancels.items()}
        done = 0
        for symbol, ids in work.items():
            for oid in ids:
                if oid in self._cancelled_ids:
                    self._clear_pending_cancel(symbol, oid)
                    continue
                n = self._cancel_attempts.get(oid, 0)
                if n >= self.MAX_CANCEL_ATTEMPTS:
                    continue
                try:
                    self.exchange.cancel_order(oid, symbol)
                    log.warning(f"{symbol}: pending cancel of {oid} SUCCEEDED "
                                f"on attempt {n + 1}")
                    self._record(symbol, "pending_cancel_ok", f"id={oid}")
                    self._clear_pending_cancel(symbol, oid)
                    done += 1
                except Exception as e:
                    ok, gone = self._cancel_algo_order(oid, symbol)
                    if ok or gone:
                        self._cancelled_ids.add(oid)
                        self._clear_pending_cancel(symbol, oid)
                        done += 1
                        continue
                    self._cancel_attempts[oid] = n + 1
                    if self._cancel_attempts[oid] == self.MAX_CANCEL_ATTEMPTS:
                        log.error(
                            f"{symbol}: giving up on cancelling {oid} after "
                            f"{self.MAX_CANCEL_ATTEMPTS} attempts ({e}). If it "
                            f"is still on the exchange, cancel it by hand.")
                        self._record(symbol, "cancel_abandoned", f"id={oid}")
        return done

    def _cancel_stop(self, pos: FuturesPosition, order_id: str) -> bool:
        """
        Cancel a stop the guardian is managing.

        Callers must only pass IDs of orders that satisfied is_protective_stop().
        """
        if self.dry_run:
            log.info(f"[DRY RUN] would cancel order {order_id} on {pos.symbol}")
            return True
        try:
            self.exchange.cancel_order(order_id, pos.symbol)
            log.info(f"Cancelled superseded stop {order_id} on {pos.symbol}")
            self._clear_pending_cancel(pos.symbol, order_id)
            return True
        except Exception as e:
            # -2011 "Unknown order sent" is NOT proof the order is gone here.
            # An order that was still resting on the exchange returned -2011 to
            # a cancel, so treating it as success dropped it from tracking and
            # orphaned it permanently. Count the attempts instead and keep it
            # queued; a retry costs one request and may succeed.
            # These stops are ALGO orders, so a regular cancel misses them.
            ok, gone = self._cancel_algo_order(order_id, pos.symbol)
            if ok or gone:
                self._cancelled_ids.add(order_id)
                self._clear_pending_cancel(pos.symbol, order_id)
                return True
            n = self._cancel_attempts.get(order_id, 0) + 1
            self._cancel_attempts[order_id] = n
            gone = self._order_already_gone(e)
            if n == 1 or n % 20 == 0:
                log.warning(
                    f"cancel {order_id} on {pos.symbol} failed (attempt {n}): {e}"
                    + ("  [-2011 is not reliable on this venue; still queued]"
                       if gone else ""))
            self._queue_pending_cancel(pos.symbol, order_id)
            return False

    # ── per-position management ─────────────────────────────────────────────

    def _record(self, symbol: str, action: str, detail: str):
        entry = {"ts": time.time(), "symbol": symbol,
                 "action": action, "detail": detail}
        with self._lock:
            self._actions.append(entry)
            self._actions = self._actions[-50:]

    def manage_position(self, pos: FuturesPosition):
        price = self.mark_price(pos)
        if price is None:
            return
        t = self._ticker(pos.symbol) or {}
        hi, lo = self._range_24h(pos.symbol, t)
        current_roi = roi_pct(pos, price)
        range_pos = None
        dist_low = dist_high = None
        try:
            if hi and lo and float(hi) > float(lo):
                hi_f, lo_f = float(hi), float(lo)
                range_pos = (price - lo_f) / (hi_f - lo_f)
                # How far price has travelled from each extreme, as a %.
                dist_low = (price - lo_f) / lo_f * 100
                dist_high = (price - hi_f) / hi_f * 100
        except (TypeError, ValueError):
            pass

        with self._lock:
            self._pos_meta[pos.symbol] = {
                "margin": pos.margin, "notional": pos.notional,
                "leverage": pos.effective_leverage, "side": pos.side,
                "entry_price": pos.entry_price, "current_price": price,
                "current_roi": current_roi,
                "high_24h": float(hi) if hi else None,
                "low_24h": float(lo) if lo else None,
                "range_pos_24h": range_pos,
                "pct_above_24h_low": dist_low,
                "pct_below_24h_high": dist_high,
                "opened_seen_at": self._pos_meta.get(pos.symbol, {}).get(
                    "opened_seen_at", time.time()),
                # Exchange-side fill time, captured once. opened_seen_at minus
                # this is the guardian's observation lag — the thing the age-0
                # ROI checkpoint was accidentally measuring.
                "fill_time": self._pos_meta.get(pos.symbol, {}).get(
                    "fill_time", pos.updated_at),
                "fill_price": self._pos_meta.get(pos.symbol, {}).get(
                    "fill_price", pos.entry_price),
                # Entry conditions, captured once and preserved, so closed
                # trades can later be analysed by what they were entered on.
                "entry_context": self._pos_meta.get(pos.symbol, {}).get(
                    "entry_context") or self._capture_entry_context(pos, price, range_pos),
            }

        with self._lock:
            state = self._states.get(pos.symbol)

        if state is None:
            # First sight: adopt any protective stop already resting so we
            # manage it rather than stacking a second one on top.
            orders = self.fetch_open_orders(pos.symbol)
            state = adopt_state(pos, orders, roi_pct(pos, price), self.cfg)
            self._record(pos.symbol, "adopted",
                         f"{pos.side} @ {pos.entry_price} | "
                         f"existing stop: {state.stop_roi is not None}")

        prev_order_id = state.stop_order_id
        prev_stop_roi = state.stop_roi
        was_armed = state.armed
        # Instrumentation for a fail-fast rule: when the position first turned
        # positive, and where it stood at fixed ages. Recorded only — nothing
        # acts on these yet, but without them any cutoff would be chosen blind.
        self._note_progress(pos, state, current_roi)

        if self._should_fail_fast(pos, state, current_roi):
            log.warning(
                f"FAIL-FAST {pos.symbol}: no positive peak after "
                f"{self.cfg.fail_fast_s:.0f}s and ROI {current_roi:+.1f}% — "
                f"closing at market rather than waiting for the stop.")
            self._record(pos.symbol, "fail_fast",
                         f"peak {state.peak_roi:+.1f}%, ROI {current_roi:+.1f}%")
            try:
                self.close_position(pos.symbol)
            except Exception as e:
                log.warning(f"{pos.symbol}: fail-fast close failed: {e}")
            return

        _stop_roi_used = self._cap_stop_to_budget(pos, self.effective_stop_roi(pos))
        stop_roi_used = _stop_roi_used
        self._check_risk_invariant(pos, _stop_roi_used)
        state, stop_price, reason = evaluate(
            pos, price, state, self.cfg,
            initial_stop_override=_stop_roi_used)

        # ── Armed phase: Binance owns the trail ──────────────────────────────
        # Once a native trailing stop is resting the exchange tracks the peak
        # continuously, so the guardian must NOT keep repositioning stops — it
        # only watches. This is what removes the polling gap.
        if state.native_trail_id:
            # Binance owns the trail, so no repositioning — but a fixed stop
            # whose cancel FAILED at arming would otherwise sit untouched until
            # the position closed, still able to fire at a level the trade has
            # long left behind. Retry the sweep each cycle; it is a no-op once
            # nothing is superseded.
            self._cancel_superseded_stops(pos, keep=state.native_trail_id)
            with self._lock:
                self._states[pos.symbol] = state
            return

        # ── Transition: arm the native trail, replacing the fixed stop ───────
        # Arm whenever the position IS armed and no trail is resting yet — not
        # only on the transition. A position opened by a trailing-stop ENTRY can
        # be deep in profit the first time the guardian sees it, in which case
        # it is born armed and the transition never fires. That left it falling
        # through to the fixed stop, which the exchange rejects as "would
        # trigger immediately", looping UNPROTECTED.
        if self.cfg.use_native_trail and state.armed and not state.native_trail_id:
            trail_id = None
            try:
                trail_id = self._place_native_trail(pos)
            except Exception as e:
                log.error(f"FAILED to arm native trail for {pos.symbol}: {e} — "
                          f"keeping the fixed stop")
                self._record(pos.symbol, "trail_failed", str(e))

            if trail_id:
                # Place-then-cancel, same as everywhere else: the trail is
                # resting before the fixed stop is removed.
                if prev_order_id:
                    if self._cancel_stop(pos, prev_order_id):
                        # Drop it from the sweep list so it is not re-cancelled
                        # on every subsequent cycle.
                        self._all_stop_ids[pos.symbol] = [
                            i for i in self._all_stop_ids.get(pos.symbol, [])
                            if i != prev_order_id]
                    else:
                        # Reached only for a genuine failure: an order that is
                        # already gone now reports success (-2011 handling in
                        # _cancel_stop), so this really does mean still resting.
                        log.error(
                            f"{pos.symbol}: armed the trail but could NOT cancel "
                            f"the fixed stop {prev_order_id} — it is still "
                            f"resting and will be retried each cycle.")
                        self._record(pos.symbol, "stop_cancel_failed",
                                     f"id={prev_order_id} left resting at arming")
                else:
                    log.warning(
                        f"{pos.symbol}: armed with NO fixed stop id on record "
                        f"(state lost, or the order listing could not see it). "
                        f"Any existing stop is untracked.")
                state.native_trail_id = trail_id
                state.stop_order_id = None
                # Track the trail too, so close-time cleanup cancels it.
                self._all_stop_ids.setdefault(pos.symbol, [])
                if trail_id not in self._all_stop_ids[pos.symbol]:
                    self._all_stop_ids[pos.symbol].append(trail_id)
                # Report the callbackRate actually sent, not the raw config
                # value — printing trail_callback_pct made a correctly-placed
                # 0.25% trail look like a 1.0% one.
                self._record(pos.symbol, "trail_armed",
                             f"native trailing stop, callback "
                             f"{trail_callback_price_pct(pos.effective_leverage, self.cfg):.2f}% price "
                             f"({callback_roi_at(pos.effective_leverage, self.cfg):.0f}% ROI)")
                with self._lock:
                    self._states[pos.symbol] = state
                return
            # Falling through means the trail could not be armed (e.g. the
            # lock-in check failed) — carry on managing the fixed stop.

        if stop_price is not None:
            stop_price = float(self.exchange.price_to_precision(pos.symbol, stop_price))
            # PLACE-THEN-CANCEL: never leave the position unprotected.
            new_id = None
            try:
                new_id = self._place_stop(pos, stop_price)
                state.unprotected_reason = None
            except Exception as e:
                msg = str(e)
                if "-2021" in msg or "immediately trigger" in msg.lower():
                    # The position is already worse than its stop level, so the
                    # stop cannot be placed at all. This is NOT a transient
                    # error: the position is unprotected until the operator acts.
                    # Deliberately not auto-closing — that is the operator's call.
                    cur = roi_pct(pos, price)
                    # The fixed stop cannot be placed because the position has
                    # already moved past it — which means it is IN PROFIT and
                    # its gains are exactly what needs protecting. Fall back to
                    # a trailing stop rather than leaving it naked.
                    # Which side of the stop are we on? Past it in the WINNING
                    # direction means gains to protect — trail. Past it in the
                    # LOSING direction means the stop level has been breached
                    # and the position should already be closed.
                    breached = cur <= -abs(stop_roi_used)
                    if breached:
                        log.error(
                            f"{pos.symbol}: already at {cur:+.1f}% ROI, past its "
                            f"{-abs(stop_roi_used):+.1f}% stop — the stop level has "
                            f"been breached."
                        )
                        if self.cfg.close_if_past_stop:
                            res = self.close_position(pos.symbol)
                            self._record(
                                pos.symbol, "closed_past_stop",
                                f"discovered at {cur:+.1f}% ROI, past the "
                                f"{-abs(stop_roi_used):+.1f}% stop — closed "
                                f"({'ok' if res.get('ok') else res.get('error')})")
                            state.unprotected_reason = None
                            with self._lock:
                                self._states[pos.symbol] = state
                            return
                        state.unprotected_reason = (
                            f"at {cur:+.1f}% ROI, already past its "
                            f"{-abs(stop_roi_used):+.1f}% stop and NOT closed "
                            f"(close_if_past_stop is off)")
                        self._record(pos.symbol, "UNPROTECTED",
                                     state.unprotected_reason)
                        with self._lock:
                            self._states[pos.symbol] = state
                        return

                    trail_id = None
                    if self.cfg.use_native_trail and not state.native_trail_id:
                        try:
                            trail_id = self._place_native_trail(pos)
                        except Exception as te:
                            log.error(f"{pos.symbol}: trailing fallback failed: {te}")
                    if trail_id:
                        state.native_trail_id = trail_id
                        state.stop_order_id = None
                        state.armed = True
                        state.unprotected_reason = None
                        log.warning(
                            f"{pos.symbol}: fixed stop rejected at {cur:+.1f}% ROI "
                            f"— protected with a trailing stop instead."
                        )
                        self._record(pos.symbol, "trail_fallback",
                                     f"fixed stop rejected at {cur:+.1f}% ROI; "
                                     f"trailing stop placed")
                        with self._lock:
                            self._states[pos.symbol] = state
                        return

                    # Report the stop ACTUALLY used. Printing the config value
                    # made a message read "at -5.7% ROI, past the -10% stop",
                    # which is self-contradictory and hid that the stop had been
                    # tightened by the budget cap.
                    state.unprotected_reason = (
                        f"already at {cur:+.1f}% ROI, past the "
                        f"{-abs(stop_roi_used):+.1f}% stop (config default "
                        f"{-self.cfg.initial_stop_roi:+.0f}%) — exchange rejected "
                        f"the stop, and the trailing fallback also failed"
                    )
                    log.error(
                        f"{pos.symbol}: UNPROTECTED — {state.unprotected_reason}. "
                        f"Close it or accept the risk; the guardian will not "
                        f"close a position on its own."
                    )
                    self._record(pos.symbol, "UNPROTECTED", state.unprotected_reason)
                else:
                    log.error(f"FAILED to place stop for {pos.symbol}: {e} — "
                              f"leaving the existing stop in place")
                    self._record(pos.symbol, "place_failed", msg)
                # evaluate() optimistically recorded the new stop level before
                # the order was sent. Roll it back so state reflects what is
                # ACTUALLY resting — otherwise should_replace_stop sees the
                # level as already achieved and never retries, leaving the
                # position unprotected indefinitely.
                state.stop_roi = prev_stop_roi
                with self._lock:
                    self._states[pos.symbol] = state
                return   # keep old state/order; do not cancel anything

            if prev_order_id and new_id:
                self._cancel_stop(pos, prev_order_id)

            if new_id:
                self._all_stop_ids.setdefault(pos.symbol, [])
                if new_id not in self._all_stop_ids[pos.symbol]:
                    self._all_stop_ids[pos.symbol].append(new_id)
            state.stop_order_id = new_id
            # Sweep any earlier stops whose cancel did not take. Harmless
            # individually, but they accumulate and can fire against a LATER
            # position on the same symbol.
            self._cancel_superseded_stops(pos, keep=new_id)
            log.info(f"{pos.symbol}: {reason} | ROI now {roi_pct(pos, price):+.1f}% "
                     f"| stop @ {stop_price}")
            self._record(pos.symbol, "stop_set",
                         f"{reason} (stop {state.stop_roi:+.1f}% ROI @ {stop_price})")

        with self._lock:
            self._states[pos.symbol] = state

    # ── cycle ───────────────────────────────────────────────────────────────

    def load_state(self, path: str):
        """Restore state from a previous run. Best-effort; never blocks startup."""
        from . import futures_state
        self.state_path = path
        data = futures_state.load(path, owner=self.state_owner)
        if not data:
            return
        restored = futures_state.restore_states(data)
        with self._lock:
            self._states.update(restored)
            for sym, m in (data.get("pos_meta") or {}).items():
                meta = self._pos_meta.setdefault(sym, {})
                if m.get("entry_context"):
                    meta["entry_context"] = m["entry_context"]
                # Restore anything the close record needs. manage_position
                # overwrites these within a poll, so they only matter for a
                # position that closes before the first pass after a restart —
                # which previously produced a row with no side, entry or ROI.
                for k in ("opened_seen_at", "side", "entry_price", "margin",
                          "leverage", "current_roi", "current_price",
                          "fill_time", "fill_price"):
                    if m.get(k) is not None:
                        meta[k] = m[k]
            if not self._closed_trades:
                self._closed_trades = list(data.get("closed_trades") or [])
        self._restored_safety = data.get("safety") or {}
        self._restored_placed_orders = data.get("placed_orders") or {}
        if data.get("wallet_start") and self.wallet_start is None:
            self.wallet_start = float(data["wallet_start"])
        with self._lock:
            for sym, ids in (data.get("stop_ids") or {}).items():
                cur = self._all_stop_ids.setdefault(sym, [])
                for oid in ids:
                    if oid not in cur:
                        cur.append(oid)
        with self._lock:
            for sym, ids in (data.get("pending_cancels") or {}).items():
                q = self._pending_cancels.setdefault(sym, [])
                for oid in ids:
                    if oid not in q:
                        q.append(oid)
        if data.get("pending_cancels"):
            n = sum(len(v) for v in data["pending_cancels"].values())
            log.warning(f"restored {n} order(s) still awaiting cancellation "
                        f"from a previous run — they will be retried")
        if data.get("stop_ids"):
            log.info(f"restored {sum(len(v) for v in data['stop_ids'].values())} "
                     f"tracked stop id(s) across "
                     f"{len(data['stop_ids'])} symbol(s)")
        for sym, st in restored.items():
            log.info(f"{sym}: restored peak {st.peak_roi:+.1f}% ROI, "
                     f"armed={st.armed}, trail={bool(st.native_trail_id)}")

    def verify_state_path(self) -> bool:
        """
        Confirm the state file can actually be written.

        Writes a throwaway probe file beside the real one. The previous version
        proved writability by SAVING EMPTY STATE to the real path — and ran
        after load_state, so every startup read the file and then immediately
        overwrote it with nothing. Restored trades, position records and the
        daily-loss baseline were destroyed on disk each boot.
        """
        if not self.state_path:
            log.warning("Futures state persistence DISABLED (no path set) — "
                        "sized stops and the daily-loss baseline will not "
                        "survive a restart.")
            return False
        import os
        probe = f"{self.state_path}.probe"
        try:
            os.makedirs(os.path.dirname(probe) or ".", exist_ok=True)
            with open(probe, "w") as fh:
                fh.write("ok")
            os.unlink(probe)
            log.info(f"Futures state persistence OK -> {self.state_path}")
            return True
        except Exception as e:
            log.error(
                f"Futures state NOT writable at {self.state_path}: {e} — sized "
                f"stops and the daily-loss baseline will be lost on restart. "
                f"Check the volume mount is writable.")
            return False

    def save_state(self):
        if not self.state_path:
            return
        from . import futures_state
        with self._lock:
            states = dict(self._states)
            meta = dict(self._pos_meta)
            trades = list(self._closed_trades)
        entry = getattr(self, "_entry_service", None)
        placed = entry.export_placed_orders() if entry is not None else {}
        with self._lock:
            stop_ids = {k: list(v) for k, v in self._all_stop_ids.items()}
            pending = {k: list(v) for k, v in self._pending_cancels.items()}
        futures_state.save(self.state_path, states=states, pos_meta=meta,
                           closed_trades=trades, owner=self.state_owner,
                           placed_orders=placed, stop_ids=stop_ids,
                           pending_cancels=pending,
                           wallet_start=self.wallet_start,
                           safety=getattr(self, "_safety_snapshot", lambda: {})())

    def run_cycle(self):
        try:
            positions = self.fetch_positions()
        except Exception as e:
            self._last_error = f"fetch_positions failed: {e}"
            log.warning(self._last_error)
            return

        try:
            b = self.exchange.fetch_balance()
            val, source = resolve_usdt_balance(b)
            if val <= 0:
                log.warning("Could not resolve a positive USDT futures balance")
            elif abs(val - self._wallet_balance_cached) > 0.01:
                log.info(f"Futures wallet balance {val:.2f} USDT (from {source})")
            # A single reading must not be able to scale position size. A jump
            # beyond this fraction between polls is treated as suspect: the
            # last good value is kept and the reading is logged rather than
            # acted on. Real deposits and withdrawals settle after
            # BALANCE_JUMP_CONFIRMATIONS consecutive readings agree.
            prev = self._wallet_balance_cached
            if prev and val > 0:
                ratio = val / prev
                if ratio > (1 + self.BALANCE_JUMP_TOLERANCE) or ratio < (1 - self.BALANCE_JUMP_TOLERANCE):
                    self._balance_outliers += 1
                    if self._balance_outliers < self.BALANCE_JUMP_CONFIRMATIONS:
                        log.warning(
                            f"Balance reading {val:.2f} is {(ratio - 1) * 100:+.1f}% "
                            f"from {prev:.2f} ({source}) — IGNORED for sizing "
                            f"({self._balance_outliers}/{self.BALANCE_JUMP_CONFIRMATIONS}). "
                            f"A bad reading would scale every new position.")
                        val = prev
                    else:
                        log.warning(
                            f"Balance {val:.2f} confirmed after "
                            f"{self._balance_outliers} readings — accepting.")
                        self._balance_outliers = 0
                else:
                    self._balance_outliers = 0
            self._wallet_balance_cached = val
            if self.wallet_start is None and val > 0:
                self.wallet_start = val
                log.info(f"Reconciliation baseline: wallet {val:.2f} USDT")
        except Exception as e:
            log.debug(f"balance fetch failed: {e}")

        live_symbols = {p.symbol for p in positions}

        # BACKSTOP: is there a position on the exchange we are not guarding?
        #
        # Every other consistency check is internal — they compare the bot's
        # view against itself. This is the only one that asks the exchange.
        # An unguarded position is the worst failure the system has (one ran
        # to +70% ROI untracked, unprotected, and absent from the history),
        # and it can arrive by several routes: a filled order mistaken for a
        # cancelled one, a row dropped in normalisation, or a restart that
        # lost state. This catches all of them regardless of cause.
        with self._lock:
            tracked = set(self._states)
        untracked = live_symbols - tracked
        for sym in sorted(untracked):
            if self._untracked_reported.get(sym):
                continue
            self._untracked_reported[sym] = True
            log.error(
                f"UNTRACKED POSITION {sym} is open on the exchange but the "
                f"guardian has no state for it — it is unprotected. Adopting "
                f"it now and placing a stop.")
            self._record(sym, "untracked_adopted", "found open but not guarded")
        for sym in list(self._untracked_reported):
            if sym not in live_symbols:
                self._untracked_reported.pop(sym, None)
        with self._lock:
            tracked = len(self._states)
        if tracked and not positions:
            # Every tracked position vanishing at once is far more likely to be
            # an API hiccup than a simultaneous close.
            log.warning(
                f"fetch_positions returned NO positions while tracking {tracked} "
                f"— treating as a transient reply, not a mass close.")

        # Forget state for positions that have closed (stopped out or closed by
        # the operator) so a future position on the same symbol starts fresh.
        with self._lock:
            # A symbol missing from ONE fetch_positions response is not proof it
            # closed. A transient or partial reply used to trigger the full
            # close path: a phantom trade recorded, state wiped, and the
            # protective stop cancelled as orphaned — leaving a position that
            # was still open now unprotected. One position was recorded closed
            # twice while its loss grew from 141 to 240 USDT. Require the
            # absence to repeat before believing it.
            gone = []
            for sym in list(self._states):
                if sym in live_symbols:
                    self._missing_counts.pop(sym, None)
                    continue
                n = self._missing_counts.get(sym, 0) + 1
                self._missing_counts[sym] = n
                if n < self.MISSING_CONFIRMATIONS:
                    log.warning(
                        f"{sym}: absent from the position list "
                        f"({n}/{self.MISSING_CONFIRMATIONS}) — holding state and "
                        f"its stop until confirmed. A transient reply must not "
                        f"look like a close."
                    )
                    continue
                gone.append((sym, self._states[sym], self._pos_meta.get(sym, {})))
                self._missing_counts.pop(sym, None)
            for sym, st, meta in gone:
                self._record_closed_trade(sym, st, meta)
                log.info(f"{sym}: position gone — clearing guard state")
                self._record(sym, "closed", "position no longer open")
                del self._states[sym]
                self._pos_meta.pop(sym, None)
                self._capped_stop_reported.pop(sym, None)
                self._stop_source_reported.discard(sym)
                self._missing_counts.pop(sym, None)

        # Cancel any protective stop left resting after the position closed.
        # An orphaned reduce-only stop is not harmless: if a NEW position is
        # later opened on the same symbol, that stale order can trigger against
        # it at a level chosen for the old trade. Done outside the state lock
        # because it makes network calls.
        for sym, st, _meta in gone:
            ids = set(self._all_stop_ids.pop(sym, []) or [])
            if st.stop_order_id:
                ids.add(st.stop_order_id)
            if st.native_trail_id:
                ids.add(st.native_trail_id)
            for oid in ids:
                if not self._cancel_orphan_stop(sym, oid):
                    # Keep it queued: the position is gone but the order may
                    # not be, and nothing else will ever discover it.
                    self._queue_pending_cancel(sym, oid)

        for pos in positions:
            try:
                self.manage_position(pos)
            except Exception as e:
                log.warning(f"manage_position failed for {pos.symbol}: {e}")

        # Reap entry orders this bot placed that should no longer rest: stale
        # unfilled ones (which block their symbol and would open a position
        # sized for conditions that have passed), and any left over on a symbol
        # that now has a position (which would add to it).
        entry = getattr(self, "_entry_service", None)
        if entry is not None:
            try:
                entry.reap_stale_entry_orders(self.entry_order_ttl_s, live_symbols)
            except Exception as e:
                log.warning(f"entry-order reap failed: {e}")

            # Sweep for orders the bot has no record of. Anything placed before
            # tracking existed rests indefinitely and can fill hours later.
            sweep_due = self._untracked_due()
            if self._pending_cancels:
                try:
                    self.drain_pending_cancels()
                except Exception as e:
                    log.warning(f"pending-cancel drain failed: {e}")
            if self.sweep_orphan_stops and sweep_due:
                try:
                    self.reap_orphan_stops()
                except Exception as e:
                    log.warning(f"orphan-stop sweep failed: {e}")

            if self.reap_untracked and sweep_due:
                try:
                    entry.reap_untracked_entry_orders(
                        self.entry_order_ttl_s, self._reap_scan_symbols())
                except Exception as e:
                    log.warning(f"untracked reap failed: {e}")

        self._last_cycle_ts = time.time()
        self._last_error = None
        self.save_state()

    def _record_closed_trade(self, symbol: str, state: GuardState, meta: dict):
        """
        Log a position that has disappeared from the account.

        The exit price is the LAST OBSERVED mark price, not the actual fill —
        the guardian polls, so a stop that triggered between cycles filled at a
        price it never saw. Binance's realised PnL is fetched where available
        and preferred; otherwise the figures here are an approximation and are
        labelled as such.
        """
        entry = meta.get("entry_price")
        last = meta.get("current_price")
        side = meta.get("side")
        notional = meta.get("notional") or 0.0

        # Compute the result from our own record first. This is direction-aware
        # and cannot come out sign-inverted: a short profits when price falls.
        computed = None
        if entry and last and side and notional:
            move = ((last - entry) / entry) if side == "long" else ((entry - last) / entry)
            computed = move * notional

        # Binance's realizedPnl is the exchange's own figure, so prefer
        # it — but only when it AGREES IN SIGN with the direction-aware figure.
        # A short that closed in profit was being reported as a loss because the
        # exchange value was taken on trust.
        realised = None
        pnl_source = "none"
        ledger_pnl = None
        # Reset per close. This is instance state, and leaving it set meant a
        # trade whose fill lookup failed inherited the PREVIOUS trade's
        # commission — a wrong number that looks entirely plausible.
        self._last_trade_fees = None
        try:
            # Scope the fills to THIS position's lifetime. fetch_my_trades
            # returns the last N fills for the symbol regardless of which
            # position they belonged to, so summing them blindly gave every
            # trade on a symbol the SAME realised figure — two UAI trades with
            # +24% and -5% ROI both reported an identical -1.4133.
            opened_at = meta.get("opened_seen_at")
            since_ms = int(opened_at * 1000) if opened_at else None
            trades = self.exchange.fetch_my_trades(symbol, since=since_ms, limit=50)

            scoped = []
            for t in trades:
                ts = t.get("timestamp")
                if since_ms and ts and ts < since_ms:
                    continue        # belongs to an earlier position
                scoped.append(t)

            pnl = sum(float((t.get("info") or {}).get("realizedPnl") or 0)
                      for t in scoped)
            # Binance's realizedPnl EXCLUDES commission — that is a separate
            # field on the same fill. Summing it alone gives a GROSS figure,
            # which is why a positive total P&L can sit alongside a falling
            # wallet balance. Capture the commission so net is reportable.
            fees = 0.0
            for t in scoped:
                info = t.get("info") or {}
                try:
                    fees += abs(float(info.get("commission") or
                                      (t.get("fee") or {}).get("cost") or 0))
                except (TypeError, ValueError):
                    pass
            self._last_trade_fees = round(fees, 6)
            if scoped and not pnl:
                log.debug(f"{symbol}: {len(scoped)} fill(s) in scope, all zero realisedPnl")
            if pnl:
                if computed is None or (pnl >= 0) == (computed >= 0):
                    realised = pnl
                    pnl_source = "fills"
                else:
                    log.warning(
                        f"{symbol}: exchange realisedPnl {pnl:+.4f} disagrees in sign "
                        f"with the {side} result computed from entry/exit "
                        f"({computed:+.4f}) — using the computed value."
                    )
                    realised = computed
                    pnl_source = "computed"
        except Exception as e:
            log.debug(f"realised PnL lookup failed for {symbol}: {e}")

        # The income ledger is queried INDEPENDENTLY of the fills. It used to
        # sit inside the same try, so a fill-lookup failure skipped it — losing
        # the authoritative source exactly when the fallback was needed.
        try:
            # The window must start when the ORDER WAS PLACED, not when the
            # guardian first saw the position. The entry commission is charged
            # at the fill, which precedes the first poll — starting at
            # opened_seen_at filtered it out and captured only the EXIT side,
            # halving every fee figure and leaving the wallet unreconciled.
            ctx = meta.get("entry_context") or {}
            placed_at = ctx.get("sized_at")
            opened_at = meta.get("opened_seen_at")
            floor_ts = placed_at or opened_at
            if floor_ts:
                # A small margin in case the fill preceded the recorded stamp.
                income_since = int((float(floor_ts) - self.INCOME_LOOKBACK_PAD_S)
                                   * 1000)
            else:
                income_since = None
            led_pnl, led_comm, led_found = self._income_for_position(
                symbol, income_since)
            if led_found:
                if led_comm:
                    self._last_trade_fees = round(led_comm, 6)
                ledger_pnl = led_pnl
                log.info(f"{symbol}: ledger realised {led_pnl:+.4f}, "
                         f"fees {led_comm:.4f}")
            else:
                log.warning(
                    f"{symbol}: income ledger returned nothing — P&L and fees "
                    f"fall back to fills or a price estimate, which can be wrong.")
        except Exception as e:
            log.warning(f"{symbol}: income ledger lookup failed: {e}")

        if realised is None and computed is not None:
            realised = computed
            pnl_source = "computed"

        # The income ledger is the exchange's own accounting and wins over both
        # the fill sum and the price estimate — INCLUDING when it reports zero,
        # which is exactly the case a price-based estimate gets wrong. One
        # trade opened and closed at the same price (true result 0) and was
        # recorded as +4.32 because the estimated exit was never checked.
        if ledger_pnl is not None:
            if realised is not None and abs(realised - ledger_pnl) > 0.01:
                log.warning(
                    f"{symbol}: {pnl_source} P&L {realised:+.4f} disagrees with "
                    f"the income ledger {ledger_pnl:+.4f} — using the ledger.")
            realised = ledger_pnl
            pnl_source = "ledger"

        # When the exchange reports realised PnL it reflects the ACTUAL fill.
        # The guardian polls, so a stop that triggered between cycles filled at
        # a price it never observed — final_roi and exit_price computed from the
        # last observed price then understate the move. Derive both from the
        # realised figure instead, which is why a stopped-out trade could show
        # -3% ROI while the money said -10%.
        margin = meta.get("margin") or 0.0
        leverage = meta.get("leverage") or 0.0
        final_roi = _r2(meta.get("current_roi"))
        exit_price = last
        exit_from_exchange = False

        if realised is not None and realised != computed and margin > 0:
            final_roi = round(realised / margin * 100.0, 2)
            if entry and leverage > 0:
                move = (final_roi / 100.0) / leverage
                exit_price = (entry * (1 + move) if side == "long"
                              else entry * (1 - move))
            exit_from_exchange = True
            observed = _r2(meta.get("current_roi"))
            if observed is not None and abs(final_roi - observed) > 1.0:
                log.info(
                    f"{symbol}: exit reconstructed from realised PnL — "
                    f"{final_roi:+.2f}% ROI (last observed was {observed:+.2f}%, "
                    f"the stop filled between polls)"
                )

        rec = {
            "symbol": symbol,
            "side": meta.get("side"),
            "entry_price": entry,
            "exit_price": exit_price,
            "margin_usdt": meta.get("margin"),
            "leverage": meta.get("leverage"),
            "peak_roi": round(state.peak_roi, 2),
            "trough_roi": round(state.trough_roi, 2),
            # Seconds from first sighting to the first positive ROI. None means
            # it never went green — the case a fail-fast rule targets.
            "secs_to_first_positive": (
                round(state.first_positive_at - float(meta["opened_seen_at"]), 1)
                if state.first_positive_at and meta.get("opened_seen_at") else None),
            # ROI at first sight. NOT a property of the entry: ROI against the
            # fill price is zero at the fill, so any non-zero value here is the
            # market moving during the guardian's observation lag.
            "roi_at_first_sight": state.roi_checkpoints.get("0"),
            "roi_at_0s": state.roi_checkpoints.get("0"),   # kept for old rows
            # Seconds between the exchange filling the entry and the guardian
            # first seeing the position. The honest measure of that lag.
            "observation_lag_s": (
                round(meta["opened_seen_at"] - float(meta["fill_time"]), 2)
                if meta.get("opened_seen_at") and meta.get("fill_time")
                else None),
            "fill_price": meta.get("fill_price"),
            "roi_at_60s": state.roi_checkpoints.get("60"),
            "roi_at_180s": state.roi_checkpoints.get("180"),
            "roi_at_300s": state.roi_checkpoints.get("300"),
            "final_roi": final_roi,
            # ROI DERIVED from the same figure as the money, so the two can
            # never disagree. Previously ROI came from entry/exit prices while
            # realised came from the ledger, and one trade reported +4.18% ROI
            # against a true result of zero P&L.
            "roi_from_realised": (
                round(realised / margin * 100, 2)
                if realised is not None and margin else None),
            "net_roi": (
                round((realised - self._last_trade_fees) / margin * 100, 2)
                if realised is not None and margin
                and self._last_trade_fees is not None else None),
            "pnl_source": pnl_source,
            # Whether the money figure came from the EXCHANGE at all. When
            # neither the income ledger nor the fills are available, the
            # fallback reconstructs the exit from a price it has to guess —
            # which produced a -75% ROI on a position whose stop was capped at
            # -30%, and a flat 0 on another. Such numbers are not merely
            # imprecise, they are invented, so they are marked and excluded
            # from every total rather than silently averaged in.
            "pnl_verified": pnl_source in ("ledger", "fills"),
            "observed_roi": _r2(meta.get("current_roi")),
            "exit_from_exchange": exit_from_exchange,
            "stop_roi": None if state.stop_roi is None else round(state.stop_roi, 2),
            "armed": state.armed,
            # The capital actually committed. Needed to compute return on
            # capital: averaging ROI percentages weights a large trade the same
            # as a small one, so one big loss can hide behind two small wins.
            "margin": round(margin, 4) if margin else None,
            "realised_pnl_usdt": None if realised is None else round(realised, 4),
            # Commission from the income ledger. GROSS realised P&L excludes
            # it, which is why a positive P&L can sit beside a falling wallet.
            "fees_usdt": (None if self._last_trade_fees is None
                          else round(self._last_trade_fees, 4)),
            "net_pnl_usdt": (
                round(realised - self._last_trade_fees, 4)
                if realised is not None and self._last_trade_fees is not None
                else None),
            "exit_is_estimate": not exit_from_exchange,
            "entry_context": meta.get("entry_context") or {},
            # How far price DRIFTED between sizing and the fill, signed so
            # positive is always against the position. Elapsed time is only a
            # proxy for this: a calm coin can rest for many minutes and be the
            # same trade, a vertical one is a different trade in seconds.
            "drift_since_sizing_pct": self._drift_pct(meta, meta.get("side")),
            # Seconds between the entry being SIZED and the position first
            # being seen. A TRAILING_STOP_MARKET entry rests until price
            # retraces, so the conditions that sized the callback can be many
            # minutes stale by the time it fills — STORJ rested 305s while
            # price ran. Computed from fields already recorded.
            "signal_age_s": (
                round(meta["opened_seen_at"] - (meta.get("entry_context") or {})["sized_at"], 1)
                if meta.get("opened_seen_at")
                and (meta.get("entry_context") or {}).get("sized_at")
                else None),
            "exit_reason": ("trail" if state.native_trail_id
                            else ("stop" if state.stop_roi is not None else "unknown")),
            "opened_at": meta.get("opened_seen_at"),
            "closed_at": time.time(),
        }
        with self._lock:
            self._closed_trades.append(rec)
            # A 100-trade cap silently truncated the record: a 24-hour run kept
            # reporting exactly 100 trades while far more had closed, so every
            # split was computed on a moving window rather than the whole
            # sample. Data collection is the point, so the ceiling is now high
            # enough not to bind in practice and configurable if it ever does.
            if len(self._closed_trades) > self.max_closed_trades:
                self._closed_trades = self._closed_trades[-self.max_closed_trades:]
        log.info(
            f"CLOSED {symbol} {rec['side']} peak={rec['peak_roi']:+.1f}% ROI "
            f"final={rec['final_roi']}% realised="
            f"{'n/a' if realised is None else f'{realised:+.4f}'}"
        )

    def close_position(self, symbol: str) -> dict:
        """
        Close an open position at market with a REDUCE-ONLY order.

        Reduce-only means this can only ever shrink or flatten the position; it
        cannot open or reverse one, whatever size is passed. The resting
        protective stop is cancelled afterwards, not before, so the position is
        never left unprotected while the close is in flight.
        """
        positions = {p.symbol: p for p in self.fetch_positions()}
        pos = positions.get(symbol)
        if pos is None:
            return {"ok": False, "error": f"no open position on {symbol}"}

        side = stop_side(pos)          # the side that closes this position

        # Split across the per-order quantity cap, same as the protective stop.
        # A single oversized MARKET order is rejected with -4005, which is why
        # closing from the dashboard failed and had to be done on Binance.
        max_qty = self.market_max_qty(symbol)
        chunks: list[float] = []
        remaining = pos.qty
        if max_qty and remaining > max_qty:
            while remaining > 0 and len(chunks) < 20:
                c = float(self.exchange.amount_to_precision(symbol,
                                                            min(remaining, max_qty)))
                if c <= 0:
                    break
                chunks.append(c)
                remaining -= c
        else:
            chunks = [float(self.exchange.amount_to_precision(symbol, pos.qty))]

        qty = sum(chunks)
        if self.dry_run:
            log.info(f"[DRY RUN] would close {symbol}: {side} MARKET reduceOnly "
                     f"{qty} in {len(chunks)} order(s)")
            self._record(symbol, "close_dry_run", f"{side} MARKET {qty}")
            return {"ok": True, "dry_run": True, "symbol": symbol,
                    "side": side, "qty": qty, "message": "Dry run — no order sent."}

        ids: list[str] = []
        for c in chunks:
            try:
                order = self.exchange.create_order(
                    symbol=symbol, type="MARKET", side=side, amount=c,
                    price=None, params={"reduceOnly": True},
                )
                ids.append(str(order.get("id") or order.get("orderId") or ""))
            except Exception as e:
                log.error(f"close failed for {symbol}: {e}")
                self._record(symbol, "close_failed", str(e))
                # Report partial success honestly rather than claiming a clean
                # close — some of the position may already be flat.
                return {"ok": False, "error": str(e),
                        "partially_closed_qty": sum(chunks[:len(ids)]),
                        "order_ids": ids}

        oid = ids[0] if ids else ""
        if len(ids) > 1:
            log.warning(f"{symbol}: closed in {len(ids)} orders (per-order cap)")
        log.warning(f"CLOSED {symbol} by operator: {side} MARKET reduceOnly qty={qty} id={oid}")
        self._record(symbol, "closed_by_operator", f"{side} MARKET {qty} id={oid}")

        # Remove the now-redundant protective stop.
        with self._lock:
            st = self._states.get(symbol)
        if st and st.stop_order_id:
            self._cancel_stop(pos, st.stop_order_id)

        return {"ok": True, "dry_run": False, "symbol": symbol,
                "side": side, "qty": qty, "order_id": oid}

    def _endpoint_hint(self) -> str:
        """Best-effort description of the API host actually in use."""
        try:
            urls = getattr(self.exchange, "urls", {}) or {}
            api = urls.get("api")
            if isinstance(api, dict):
                for key in ("fapiPrivate", "fapiPublic", "future", "private"):
                    if api.get(key):
                        return str(api[key])
                return str(next(iter(api.values()), ""))
            return str(api or "")
        except Exception:
            return "unknown"

    def diagnose_order_listing(self) -> dict:
        """
        Report what every candidate order-listing call actually returns.

        Balance and position calls succeed while order listings come back
        empty, so the question is which specific endpoint disagrees with the
        exchange UI. Guessing has been wrong repeatedly; this reports the URL
        used, the count, and the raw error for each candidate so the answer is
        observed rather than inferred.
        """
        urls = getattr(self.exchange, "urls", {}) or {}
        api = urls.get("api") if isinstance(urls.get("api"), dict) else {}
        out = {
            "demo_flag": self.demo,
            "demo_enabled_in_ccxt": bool(
                (getattr(self.exchange, "options", {}) or {}).get("enableDemoTrading")),
            "fapiPrivate_url": api.get("fapiPrivate", "?"),
            "private_url": api.get("private", "?"),
            "default_type": (getattr(self.exchange, "options", {}) or {}).get("defaultType"),
            "attempts": [],
        }
        try:
            out["positions"] = len(self.exchange.fetch_positions() or [])
        except Exception as e:
            out["positions"] = f"error: {e}"

        candidates = [
            ("fetch_open_orders()", lambda: self.exchange.fetch_open_orders()),
            ("fetch_open_orders(type=future)",
             lambda: self.exchange.fetch_open_orders(None, None, None,
                                                     {"type": "future"})),
            ("fapiPrivateGetOpenOrders",
             lambda: self.exchange.fapiPrivateGetOpenOrders()),
            ("fapiPrivateGetOpenOrders(recvWindow)",
             lambda: self.exchange.fapiPrivateGetOpenOrders({"recvWindow": 10000})),
        ]
        for name, call in candidates:
            row = {"call": name}
            try:
                got = call() or []
                row["count"] = len(got)
                row["sample"] = [
                    {k: v for k, v in (o.items() if isinstance(o, dict) else [])
                     if k in ("orderId", "id", "symbol", "origType", "type",
                              "reduceOnly", "status")}
                    for o in list(got)[:3]
                ]
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            out["attempts"].append(row)
        return out

    def _algo_orders(self) -> list:
        """Open ALGO orders — the book conditional and trailing stops live in."""
        for name in ("fapiPrivateGetOpenAlgoOrders", "fapiPrivateGetAllAlgoOrders"):
            fn = getattr(self.exchange, name, None)
            if fn is None:
                continue
            try:
                res = fn() or []
            except Exception as e:
                log.debug(f"{name} failed: {e}")
                continue
            # Binance wraps the list; accept either shape.
            if isinstance(res, dict):
                res = (res.get("orders") or res.get("data")
                       or res.get("algoOrders") or [])
            if res:
                return list(res)
        return []

    def _income_for_position(self, symbol: str, since_ms: int | None) -> tuple:
        """
        Realised P&L and commission from Binance's INCOME ledger.

        This is the authoritative record of what a position actually paid or
        earned. It matters because the fallback — reconstructing the exit from
        the stop level — can invent a profit: one trade opened and closed at
        the SAME price (true result: 0 P&L, -2.07 USDT in fees) and was
        recorded as +4.32 USDT because the estimated exit was never checked
        against the ledger.

        Returns (realised_pnl, commission, found).
        """
        fn = getattr(self.exchange, "fapiPrivateGetIncome", None)
        if fn is None:
            return 0.0, 0.0, False
        try:
            params = {"symbol": self.exchange.market_id(symbol), "limit": 200}
            if since_ms:
                params["startTime"] = int(since_ms)
            rows = fn(params) or []
        except Exception as e:
            log.debug(f"{symbol}: income lookup failed: {e}")
            return 0.0, 0.0, False

        pnl = comm = 0.0
        seen = False
        for r in rows:
            if not isinstance(r, dict):
                continue
            kind = str(r.get("incomeType") or "").upper()
            try:
                val = float(r.get("income") or 0)
            except (TypeError, ValueError):
                continue
            if kind == "REALIZED_PNL":
                pnl += val
                seen = True
            elif kind == "COMMISSION":
                comm += abs(val)
                seen = True
        return round(pnl, 8), round(comm, 8), seen

    def _cancel_any(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order that may live in EITHER book.

        Conditional and trailing stops are algo orders, so a regular cancel
        returns -2011 for them — which means "not in the book I searched",
        not "does not exist". Always try both before concluding anything.
        """
        if self.dry_run:
            log.info(f"[DRY RUN] would cancel {order_id} on {symbol}")
            return True
        if order_id in self._cancelled_ids:
            return True                       # already done; never retry
        try:
            self.exchange.cancel_order(order_id, symbol)
            self._cancelled_ids.add(order_id)
            return True
        except Exception as e:
            ok, gone = self._cancel_algo_order(order_id, symbol)
            if ok or gone:
                # Cancelled, or absent from BOTH books — either way it is not
                # resting and must not be queued for another attempt.
                self._cancelled_ids.add(order_id)
                return True
            log.debug(f"{symbol}: cancel {order_id} failed in both books: {e}")
            return False

    def _cancel_algo_order(self, order_id: str, symbol: str) -> tuple:
        """
        Cancel via the ALGO book. Returns (cancelled, definitely_gone).

        -2011 from the REGULAR book only means "not in the book I searched" —
        these are algo orders. But -2011 from the ALGO book too means the order
        exists in neither, so it really is gone and must stop being retried.
        """
        fn = getattr(self.exchange, "fapiPrivateDeleteAlgoOrder", None)
        if fn is None:
            return False, False
        try:
            fn({"algoId": order_id})
            log.info(f"Cancelled ALGO order {order_id} on {symbol}")
            return True, True
        except Exception as e:
            gone = self._order_already_gone(e)
            log.debug(f"algo cancel {order_id} failed: {e}")
            return False, gone

    def _all_open_orders_raw(self) -> list:
        """
        Every open order on the account, from the exchange directly.

        The unified fetch_open_orders() call with no symbol has been observed
        returning nothing on this account while per-symbol queries return
        orders, so the raw endpoint is tried as well. Asking the exchange for
        the full list is the only approach that does not depend on the bot
        remembering which symbols to check — and that memory is cleared by a
        history reset or a failed persist.
        """
        # Belt and braces: ccxt raises a rate-limit WARNING as an ExchangeError
        # unless this is acknowledged, and the request never leaves the
        # process. Setting it in the constructor is not enough if anything
        # replaces options later, and the failure mode is silent — an empty
        # order list that looks like a clean account.
        try:
            opts = self.exchange.options
            if not isinstance(opts.get("fetchOpenOrders"), dict):
                opts["fetchOpenOrders"] = {}
            opts["fetchOpenOrders"]["warnWithoutSymbol"] = False
        except Exception:
            pass

        # The account-wide listing carries 40x the normal request weight. On
        # Binance demo futures it returns nothing no matter what, so paying
        # that weight every sweep buys exactly nothing. After a run of empty
        # replies, stop asking and retry only occasionally — cleanup relies on
        # the bot's persisted record of what it placed, not on discovery.
        now = time.time()
        skip_wide = (self._empty_listings >= self.EMPTY_LISTING_LIMIT
                     and now - self._last_wide_probe < self.WIDE_PROBE_INTERVAL_S)
        if skip_wide:
            log.debug(f"skipping account-wide order listing "
                      f"({self._empty_listings} empty replies; retry in "
                      f"{self.WIDE_PROBE_INTERVAL_S/60:.0f}m)")
            return []
        self._last_wide_probe = now

        rows = []
        for label, call in (
                ("unified", lambda: self.exchange.fetch_open_orders()),
                ("raw fapi", lambda: self.exchange.fapiPrivateGetOpenOrders()),
                # Conditional and trailing stops are ALGO orders on Binance
                # futures — placed via POST /fapi/v1/algoOrder and held in a
                # SEPARATE book. They never appear in /fapi/v1/openOrders,
                # which is why every listing returned zero while orders were
                # plainly resting on the exchange.
                ("algo", lambda: self._algo_orders()),
        ):
            try:
                got = call() or []
                log.info(f"open-order listing via {label}: {len(got)} order(s)")
                for o in got:
                    if isinstance(o, dict):
                        rows.append(o)
            except Exception as e:
                # Log the MESSAGE, not just the type. "ExchangeError" alone
                # hides exactly what Binance said, which is the only useful
                # part when two order listings disagree with the UI.
                log.warning(f"open-order listing via {label} FAILED: "
                            f"{type(e).__name__}: {e}")

        if rows:
            self._empty_listings = 0
        else:
            self._empty_listings += 1
            if self._empty_listings == self.EMPTY_LISTING_LIMIT:
                log.warning(
                    f"account-wide order listing has returned nothing "
                    f"{self._empty_listings} times; it costs 40x request "
                    f"weight, so it will now be probed only every "
                    f"{self.WIDE_PROBE_INTERVAL_S/60:.0f} minutes. Cleanup "
                    f"continues from the bot's own persisted record of the "
                    f"stops it placed.")
        return rows

    def _normalise_order(self, o: dict) -> dict:
        """Accept either a ccxt order or a raw Binance one."""
        info = o.get("info") if isinstance(o.get("info"), dict) else o
        sym = o.get("symbol") or ""
        if sym and "/" not in sym:
            # raw Binance gives BRUSDT; map it back to the unified symbol
            try:
                sym = self.exchange.markets_by_id[sym][0]["symbol"]
            except Exception:
                base = sym[:-4] if sym.endswith("USDT") else sym
                sym = f"{base}/USDT:USDT"
        # Careful: the raw endpoint returns the STRING "false", and
        # bool("false") is True. Coerce explicitly — getting this wrong would
        # classify an entry order as a protective stop and cancel it, breaking
        # the guarantee that the bot never touches an entry it did not place.
        ro = o.get("reduceOnly")
        if ro is None:
            ro = info.get("reduceOnly")
        if isinstance(ro, str):
            ro = ro.strip().lower() == "true"
        ro = bool(ro)
        return {
            # Algo orders identify themselves with algoId, not orderId.
            "id": str(o.get("id") or info.get("orderId")
                      or info.get("algoId") or o.get("algoId") or ""),
            "symbol": sym,
            "type": (o.get("type") or info.get("origType")
                     or info.get("type") or info.get("algoType")
                     or info.get("strategyType") or "").upper(),
            "reduce_only": ro,
            "ts": o.get("timestamp") or float(info.get("time") or 0),
            "qty": o.get("amount") or info.get("origQty"),
        }

    def _orphan_scan_symbols(self, live: set) -> list:
        """
        Symbols worth checking for orphaned stops: anywhere the bot has traded
        or is looking. Symbols with a CLOSED trade are the likeliest source —
        that is exactly when a stop is left behind.
        """
        syms = set()
        with self._lock:
            syms.update(self._states.keys())
            syms.update(self._pos_meta.keys())
            syms.update(self._all_stop_ids.keys())
            for t in self._closed_trades[-60:]:
                if t.get("symbol"):
                    syms.add(t["symbol"])
        entry = getattr(self, "_entry_service", None)
        if entry is not None:
            try:
                syms.update(entry.export_placed_orders().keys())
            except Exception:
                pass
        scanner = getattr(self, "_scanner", None)
        if scanner is not None:
            try:
                for row in (scanner.snapshot().get("candidates") or []):
                    if row.get("symbol"):
                        syms.add(row["symbol"])
            except Exception:
                pass
        return sorted(syms - live)

    def reconcile_orders(self) -> dict:
        """
        Compare every resting order against actual positions.

        Exists so a mismatch is visible in the dashboard rather than requiring
        the exchange UI to be read by hand. Classifies each order as
        protecting a real position, an orphaned stop, or a stale entry.
        """
        out = {"positions": [], "protecting": [], "orphan_stops": [],
               "stale_entries": [], "account_wide_count": 0,
               "per_symbol_count": 0, "error": None,
               # Which account/endpoint this reflects. When an order listing
               # disagrees with the exchange UI, the first question is whether
               # both are looking at the same place.
               "endpoint": self._endpoint_hint(),
               "demo": self.demo}
        try:
            live = {p.symbol: p for p in self.fetch_positions()}
        except Exception as e:
            out["error"] = f"positions: {e}"
            return out
        out["positions"] = [
            {"symbol": sym, "side": p.side, "qty": p.qty,
             "margin": round(p.margin, 2)} for sym, p in live.items()]

        orders, seen = [], set()
        for raw in self._all_open_orders_raw():
            o = self._normalise_order(raw)
            if o["id"] and o["id"] not in seen:
                seen.add(o["id"]); orders.append(o); out["account_wide_count"] += 1
        for sym in sorted(live) + self._orphan_scan_symbols(set(live)):
            try:
                for raw in (self.exchange.fetch_open_orders(sym) or []):
                    o = self._normalise_order(raw)
                    o["symbol"] = o["symbol"] or sym
                    if o["id"] and o["id"] not in seen:
                        seen.add(o["id"]); orders.append(o)
                        out["per_symbol_count"] += 1
            except Exception:
                pass

        for o in orders:
            ro = o["reduce_only"]
            sym = o["symbol"]
            row = dict(o)
            if not ro:
                row["has_position"] = sym in live
                out["stale_entries"].append(row)
            elif sym in live:
                out["protecting"].append(row)
            else:
                out["orphan_stops"].append(row)

        # A position needs exactly ONE protective stop. Extras are superseded
        # ratchets whose cancel never took; they can still fire at a stale
        # level. Keep the most recent, list the rest as surplus.
        out["duplicate_stops"] = []
        by_sym: dict = {}
        for r in out["protecting"]:
            by_sym.setdefault(r["symbol"], []).append(r)
        keep = []
        for sym, rows in by_sym.items():
            rows.sort(key=lambda r: r.get("ts") or 0, reverse=True)
            keep.append(rows[0])
            out["duplicate_stops"].extend(rows[1:])
        out["protecting"] = keep
        return out

    def reap_orphan_stops(self) -> list:
        """
        Cancel reduce-only stops resting on symbols with NO open position.

        A reduce-only stop with nothing to reduce serves no purpose and cannot
        protect anything — but it CAN fire against a future position on the
        same symbol, closing it at a level chosen for a trade that already
        ended. Eight such orders accumulated across five symbols, three of them
        on one symbol.

        Unlike the entry reaper this does not need to know who placed the
        order: a protective stop with no position is unambiguously stale
        whoever created it. Non-reduce-only orders are never touched, so the
        operator's own entries are safe.
        """
        try:
            live = {p.symbol for p in self.fetch_positions()}
        except Exception as e:
            log.warning(f"orphan-stop sweep could not list positions: {_safe_err(e)}")
            return []

        # Collect from BOTH paths and de-duplicate by order id. The
        # account-wide call is cheaper, but conditional (stop / trailing)
        # orders have been observed missing from it, so every symbol the bot
        # has touched is also queried directly. Six orphans survived a sweep
        # that relied on the account-wide call alone.
        orders, seen_ids = [], set()
        account_wide = 0
        for raw in self._all_open_orders_raw():
            o = self._normalise_order(raw)
            if o["id"] and o["id"] not in seen_ids:
                seen_ids.add(o["id"]); orders.append(o); account_wide += 1

        per_symbol = 0
        # Per-symbol is now a top-up, not the primary source: live symbols may
        # carry SURPLUS stops, and remembered symbols are still worth checking.
        for sym in sorted(live) + self._orphan_scan_symbols(live):
            try:
                for raw in (self.exchange.fetch_open_orders(sym) or []):
                    o = self._normalise_order(raw)
                    o["symbol"] = o["symbol"] or sym
                    if o["id"] and o["id"] not in seen_ids:
                        seen_ids.add(o["id"]); orders.append(o); per_symbol += 1
            except Exception as e:
                log.debug(f"{sym}: per-symbol order list failed: {e}")

        log.info(f"orphan-stop sweep: {account_wide} order(s) account-wide, "
                 f"{per_symbol} more per-symbol, {len(orders)} total")
        # An empty result is now the NORMAL, healthy state: everything the bot
        # placed has been cleaned up. It used to mean the listing was broken,
        # because conditional stops live in the algo book and only the regular
        # book was queried. Warn only if orders are still queued for
        # cancellation while the listing claims there is nothing there — that
        # combination is genuinely contradictory.
        with self._lock:
            queued = sum(len(v) for v in self._pending_cancels.values())
        if not orders and queued and not self._listing_warned:
            self._listing_warned = True
            log.warning(
                f"Order listing reports nothing, yet {queued} order(s) are "
                f"still queued for cancellation. Either they are gone and the "
                f"queue is stale, or a book is not being read. Both listings "
                f"and the algo book were tried.")

        cancelled = []
        surplus: dict = {}
        for o in orders or []:
            if not o["reduce_only"]:
                continue                     # an entry order; not ours to judge
            sym = o["symbol"]
            if not sym:
                continue
            if sym in live:
                # A live position keeps its most recent stop; earlier ones are
                # superseded ratchets that can still fire at a stale level.
                surplus.setdefault(sym, []).append(o)
                continue
            otype = o["type"]
            if "STOP" not in otype and "TRAILING" not in otype:
                continue                     # not a protective stop
            oid = o["id"]
            if self._cancel_any(oid, sym):
                log.warning(f"{sym}: cancelled ORPHANED protective stop {oid} "
                            f"— no open position to protect")
                self._record(sym, "orphan_stop_swept", f"id={oid} (no position)")
                cancelled.append((sym, oid))
                self._clear_pending_cancel(sym, oid)
            else:
                self._queue_pending_cancel(sym, oid)
        # Now trim any live position down to a single protective stop.
        for sym, rows in surplus.items():
            if len(rows) < 2:
                continue
            rows.sort(key=lambda r: r.get("ts") or 0, reverse=True)
            for o in rows[1:]:
                otype = o["type"]
                if "STOP" not in otype and "TRAILING" not in otype:
                    continue
                oid = o["id"]
                if self._cancel_any(oid, sym):
                    log.warning(f"{sym}: cancelled SURPLUS protective stop {oid} "
                                f"— position already has a newer one")
                    self._record(sym, "surplus_stop_swept", f"id={oid}")
                    cancelled.append((sym, oid))
                else:
                    self._queue_pending_cancel(sym, oid)

        if cancelled:
            log.warning(f"orphan-stop sweep cancelled {len(cancelled)} order(s)")
        return cancelled

    def _cancel_orphan_stop(self, symbol: str, order_id: str) -> bool:
        """
        Cancel a stop left behind by a closed position. Returns True only on a
        confirmed cancel.

        The order has often already triggered (that is why the position
        closed), so an error here is common. But -2011 has been observed on
        orders that were STILL RESTING, so a failure is queued for retry rather
        than assumed benign — nothing else will ever discover it.
        """
        if self.dry_run:
            log.info(f"[DRY RUN] would cancel orphaned stop {order_id} on {symbol}")
            return True
        try:
            self.exchange.cancel_order(order_id, symbol)
            log.info(f"Cancelled orphaned stop {order_id} on {symbol} "
                     f"(position already closed)")
            self._record(symbol, "orphan_stop_cancelled", f"id={order_id}")
            return True
        except Exception as e:
            # Often the stop is what closed the position — but -2011 has been
            # seen on orders still resting, so report failure and let the
            # caller queue it.
            ok, gone = self._cancel_algo_order(order_id, symbol)
            if ok or gone:
                self._cancelled_ids.add(order_id)
                if ok:
                    self._record(symbol, "orphan_stop_cancelled",
                                 f"id={order_id} (algo)")
                return True
            log.debug(f"orphaned stop {order_id} on {symbol} not cancellable: {e}")
            return False

    def closed_trades(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._closed_trades))

    def _untracked_due(self) -> bool:
        """Sweeping costs one call per symbol, so it runs on its own slower clock."""
        now = time.time()
        if now - self._last_untracked_sweep < self.untracked_sweep_interval_s:
            return False
        self._last_untracked_sweep = now
        return True

    def _reap_scan_symbols(self) -> list:
        """
        Symbols worth sweeping: whatever the bot has touched, plus anything the
        scanner is currently looking at.
        """
        syms = set()
        entry = getattr(self, "_entry_service", None)
        if entry is not None:
            syms.update(entry.export_placed_orders().keys())
        scanner = getattr(self, "_scanner", None)
        if scanner is not None:
            try:
                for row in (scanner.snapshot().get("candidates") or []):
                    if row.get("symbol"):
                        syms.add(row["symbol"])
            except Exception:
                pass
        with self._lock:
            syms.update(self._states.keys())
            syms.update(self._pos_meta.keys())
        return sorted(syms)

    def _has_pending_entries(self) -> bool:
        entry = getattr(self, "_entry_service", None)
        if entry is None:
            return False
        try:
            return any(entry.bot_placed_orders(sym)
                       for sym in list(entry.export_placed_orders().keys()))
        except Exception:
            return False

    def run_forever(self):
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                log.warning(f"guardian cycle error: {e}")

            # Between an entry FILLING and the guardian first seeing it, the
            # position has no stop at all. On a sharp move that window is long
            # enough to blow past the stop level entirely — at 20x, a 0.5% move
            # in 5s is already -10% ROI. Poll faster while an entry order is
            # resting, so a new position is protected sooner.
            interval = self.poll_interval
            if self._has_pending_entries():
                interval = min(interval, self.pending_poll_interval)
            time.sleep(interval)

    def start_background(self):
        t = threading.Thread(target=self.run_forever, daemon=True,
                             name="futures-guardian")
        t.start()
        return t

    # ── API surface ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            states = {
                sym: {
                    "peak_roi": round(s.peak_roi, 2),
                    "trough_roi": round(s.trough_roi, 2),
                    "armed": s.armed,
                    "stop_roi": None if s.stop_roi is None else round(s.stop_roi, 2),
                    "stop_order_id": s.stop_order_id,
                    "native_trail_id": s.native_trail_id,
                    "unprotected_reason": s.unprotected_reason,
                    # Sizing, so a surprising position size is visible rather
                    # than something to reconstruct from the exchange UI.
                    "margin_usdt": round(self._pos_meta.get(sym, {}).get("margin", 0.0), 2),
                    "notional_usdt": round(self._pos_meta.get(sym, {}).get("notional", 0.0), 2),
                    "leverage": round(self._pos_meta.get(sym, {}).get("leverage", 0.0), 1),
                    "side": self._pos_meta.get(sym, {}).get("side"),
                    "current_roi": _r2(self._pos_meta.get(sym, {}).get("current_roi")),
                    "entry_price": self._pos_meta.get(sym, {}).get("entry_price"),
                    "current_price": self._pos_meta.get(sym, {}).get("current_price"),
                    "range_pos_24h": _r3(self._pos_meta.get(sym, {}).get("range_pos_24h")),
                    "pct_above_24h_low": _r2(self._pos_meta.get(sym, {}).get("pct_above_24h_low")),
                    "pct_below_24h_high": _r2(self._pos_meta.get(sym, {}).get("pct_below_24h_high")),
                    "high_24h": self._pos_meta.get(sym, {}).get("high_24h"),
                    "low_24h": self._pos_meta.get(sym, {}).get("low_24h"),
                    "range_source": self._range_source.get(sym, "unknown"),
                    "risk_overshoot": self._risk_overshoots.get(sym),
                }
                for sym, s in self._states.items()
            }
            actions = list(reversed(self._actions[-20:]))
        return {
            "enabled": True,
            "dry_run": self.dry_run,
            "demo": self.demo,
            "states": states,
            "recent_actions": actions,
            "last_cycle_ago_s": (time.time() - self._last_cycle_ts) if self._last_cycle_ts else None,
            "error": self._last_error,
            "wallet_balance": self._wallet_balance_cached,
            "config": {
                "initial_stop_roi": self.cfg.initial_stop_roi,
                "arm_roi": self.cfg.arm_roi,
                "callback_roi": self.cfg.callback_roi,
                "trail_callback_pct": self.cfg.trail_callback_pct,
                "use_native_trail": self.cfg.use_native_trail,
                # Fail-fast was invisible here, so a value of 0 — which
                # disables it outright on the first line of the predicate —
                # looked identical to a working configuration. 119 trades ran
                # with it silently off.
                "fail_fast_s": self.cfg.fail_fast_s,
                "fail_fast_max_peak_roi": self.cfg.fail_fast_max_peak_roi,
                "fail_fast_loss_roi": self.cfg.fail_fast_loss_roi,
                "fail_fast_enabled": bool(self.cfg.fail_fast_s),
            },
        }
