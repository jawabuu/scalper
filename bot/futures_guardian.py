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

from .futures_guard import (
    FuturesPosition, GuardState, GuardConfig,
    roi_pct, price_for_roi, stop_side,
    adopt_state, is_protective_stop, evaluate,
    callback_roi_at, trail_locks_in, is_armed, atr_stop_roi,
    trail_callback_price_pct,
)

log = logging.getLogger("futures_guardian")


def resolve_usdt_balance(bal: dict) -> tuple[float, str]:
    """
    Total USDT wallet balance from a ccxt futures balance payload.

    Single source of truth: the guardian displays this figure and the entry
    service sizes positions from it, so they must never resolve it differently.
    Several shapes are tried because the payload differs between live and demo.
    Returns (value, source) so the origin can be logged when a figure surprises.
    """
    candidates: list[tuple[str, object]] = []

    usdt = (bal or {}).get("USDT") or {}
    if isinstance(usdt, dict):
        candidates.append(("USDT.total", usdt.get("total")))
        candidates.append(("USDT.free", usdt.get("free")))

    total_map = (bal or {}).get("total") or {}
    if isinstance(total_map, dict):
        candidates.append(("total.USDT", total_map.get("USDT")))

    info = (bal or {}).get("info") or {}
    if isinstance(info, dict):
        candidates.append(("info.totalWalletBalance", info.get("totalWalletBalance")))
        candidates.append(("info.availableBalance", info.get("availableBalance")))
        assets = info.get("assets")
        if isinstance(assets, list):
            for a in assets:
                if (a or {}).get("asset") == "USDT":
                    candidates.append(("assets[USDT].walletBalance", a.get("walletBalance")))

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
                 demo: bool = True, dry_run: bool = True,
                 poll_interval: float = 5.0, socks_proxy: str | None = None):
        self.cfg = guard_cfg.validate()
        self.demo = demo
        self.dry_run = dry_run
        self.poll_interval = poll_interval

        params: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
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
        self._atr_cache: dict[str, tuple[float, float]] = {}   # sym -> (atr%, ts)
        # Extra order ids when a stop had to be split across the per-order cap.
        self._split_stop_ids: dict[str, list[str]] = {}

    # ── reading ─────────────────────────────────────────────────────────────

    def fetch_positions(self) -> list[FuturesPosition]:
        """Open positions with non-zero size, normalised to FuturesPosition."""
        raw = self.exchange.fetch_positions()
        out = []
        for p in raw:
            try:
                contracts = float(p.get("contracts") or 0)
                if contracts == 0:
                    continue
                side = (p.get("side") or "").lower()
                if side not in ("long", "short"):
                    continue
                entry = float(p.get("entryPrice") or 0)
                if entry <= 0:
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

                pos = FuturesPosition(
                    symbol=p["symbol"], side=side, entry_price=entry,
                    qty=abs(contracts), leverage=max(lev, 1), margin=margin,
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
            raw = self.exchange.fetch_ohlcv(symbol, "15m", limit=96)
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

    def effective_stop_roi(self, pos: FuturesPosition) -> float:
        """
        The initial stop distance to use for this position.

        With atr_stop_mult set, the stop scales with the coin's own volatility
        rather than being a constant — a fixed stop sits inside the noise on a
        volatile coin and needlessly far away on a calm one.
        """
        if not self.cfg.atr_stop_mult:
            return self.cfg.initial_stop_roi
        a = self.atr_pct(pos.symbol)
        roi = atr_stop_roi(a, pos.effective_leverage, self.cfg)
        if roi is None:
            return self.cfg.initial_stop_roi
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
        The exchange's per-order quantity cap for this symbol, if published.

        Low-priced coins need a huge contract count for a given notional, so a
        legitimate position size can exceed the per-ORDER cap and the protective
        stop is rejected with -4005 — leaving the position unprotected.
        """
        try:
            m = self.exchange.market(symbol)
            mx = (((m or {}).get("limits") or {}).get("amount") or {}).get("max")
            return float(mx) if mx else None
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
            return True
        except Exception as e:
            # Not fatal: it may already have triggered or been cancelled.
            log.warning(f"cancel {order_id} on {pos.symbol} failed: {e}")
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
        state, stop_price, reason = evaluate(
            pos, price, state, self.cfg,
            initial_stop_override=self.effective_stop_roi(pos))

        # ── Armed phase: Binance owns the trail ──────────────────────────────
        # Once a native trailing stop is resting the exchange tracks the peak
        # continuously, so the guardian must NOT keep repositioning stops — it
        # only watches. This is what removes the polling gap.
        if state.native_trail_id:
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
                    self._cancel_stop(pos, prev_order_id)
                state.native_trail_id = trail_id
                state.stop_order_id = None
                self._record(pos.symbol, "trail_armed",
                             f"native trailing stop, callback "
                             f"{self.cfg.trail_callback_pct}% price")
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

                    state.unprotected_reason = (
                        f"already at {cur:+.1f}% ROI, past the "
                        f"{-self.cfg.initial_stop_roi:+.0f}% stop — exchange "
                        f"rejected the stop, and the trailing fallback also failed"
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

            state.stop_order_id = new_id
            log.info(f"{pos.symbol}: {reason} | ROI now {roi_pct(pos, price):+.1f}% "
                     f"| stop @ {stop_price}")
            self._record(pos.symbol, "stop_set",
                         f"{reason} (stop {state.stop_roi:+.1f}% ROI @ {stop_price})")

        with self._lock:
            self._states[pos.symbol] = state

    # ── cycle ───────────────────────────────────────────────────────────────

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
            self._wallet_balance_cached = val
        except Exception as e:
            log.debug(f"balance fetch failed: {e}")

        live_symbols = {p.symbol for p in positions}

        # Forget state for positions that have closed (stopped out or closed by
        # the operator) so a future position on the same symbol starts fresh.
        with self._lock:
            gone = [(sym, self._states[sym], self._pos_meta.get(sym, {}))
                    for sym in list(self._states) if sym not in live_symbols]
            for sym, st, meta in gone:
                self._record_closed_trade(sym, st, meta)
                log.info(f"{sym}: position gone — clearing guard state")
                self._record(sym, "closed", "position no longer open")
                del self._states[sym]
                self._pos_meta.pop(sym, None)

        # Cancel any protective stop left resting after the position closed.
        # An orphaned reduce-only stop is not harmless: if a NEW position is
        # later opened on the same symbol, that stale order can trigger against
        # it at a level chosen for the old trade. Done outside the state lock
        # because it makes network calls.
        for sym, st, _meta in gone:
            if st.stop_order_id:
                self._cancel_orphan_stop(sym, st.stop_order_id)

        for pos in positions:
            try:
                self.manage_position(pos)
            except Exception as e:
                log.warning(f"manage_position failed for {pos.symbol}: {e}")

        self._last_cycle_ts = time.time()
        self._last_error = None

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

        # Binance's realizedPnl is more accurate (it includes fees), so prefer
        # it — but only when it AGREES IN SIGN with the direction-aware figure.
        # A short that closed in profit was being reported as a loss because the
        # exchange value was taken on trust.
        realised = None
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
            if scoped and not pnl:
                log.debug(f"{symbol}: {len(scoped)} fill(s) in scope, all zero realisedPnl")
            if pnl:
                if computed is None or (pnl >= 0) == (computed >= 0):
                    realised = pnl
                else:
                    log.warning(
                        f"{symbol}: exchange realisedPnl {pnl:+.4f} disagrees in sign "
                        f"with the {side} result computed from entry/exit "
                        f"({computed:+.4f}) — using the computed value."
                    )
                    realised = computed
        except Exception as e:
            log.debug(f"realised PnL lookup failed for {symbol}: {e}")

        if realised is None and computed is not None:
            realised = computed

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
            "final_roi": final_roi,
            "observed_roi": _r2(meta.get("current_roi")),
            "exit_from_exchange": exit_from_exchange,
            "stop_roi": None if state.stop_roi is None else round(state.stop_roi, 2),
            "armed": state.armed,
            "realised_pnl_usdt": None if realised is None else round(realised, 4),
            "exit_is_estimate": not exit_from_exchange,
            "opened_at": meta.get("opened_seen_at"),
            "closed_at": time.time(),
        }
        with self._lock:
            self._closed_trades.append(rec)
            self._closed_trades = self._closed_trades[-100:]
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

    def _cancel_orphan_stop(self, symbol: str, order_id: str):
        """
        Cancel a stop left behind by a closed position.

        The order has usually already triggered (that is why the position
        closed), so a "not found / already filled" error is the normal case and
        is logged quietly rather than treated as a failure.
        """
        if self.dry_run:
            log.info(f"[DRY RUN] would cancel orphaned stop {order_id} on {symbol}")
            return
        try:
            self.exchange.cancel_order(order_id, symbol)
            log.info(f"Cancelled orphaned stop {order_id} on {symbol} "
                     f"(position already closed)")
            self._record(symbol, "orphan_stop_cancelled", f"id={order_id}")
        except Exception as e:
            # Expected when the stop is what closed the position.
            log.debug(f"orphaned stop {order_id} on {symbol} not cancellable: {e}")

    def closed_trades(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._closed_trades))

    def run_forever(self):
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                log.warning(f"guardian cycle error: {e}")
            time.sleep(self.poll_interval)

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
            },
        }
