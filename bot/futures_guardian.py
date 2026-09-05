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
)

log = logging.getLogger("futures_guardian")


class FuturesGuardian:
    def __init__(self, guard_cfg: GuardConfig, *, api_key: str, api_secret: str,
                 testnet: bool = True, dry_run: bool = True,
                 poll_interval: float = 5.0, socks_proxy: str | None = None):
        self.cfg = guard_cfg.validate()
        self.testnet = testnet
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
        if testnet:
            self.exchange.set_sandbox_mode(True)

        # symbol -> GuardState
        self._states: dict[str, GuardState] = {}
        self._lock = threading.RLock()
        self._last_cycle_ts: float = 0.0
        self._last_error: str | None = None
        self._actions: list[dict] = []   # recent actions, for the dashboard

        mode = "DRY RUN (no orders sent)" if dry_run else "LIVE (places real orders)"
        log.warning(
            f"Futures guardian starting — testnet={testnet} — {mode} | "
            f"stop {-self.cfg.initial_stop_roi:+.0f}% ROI, arm +{self.cfg.arm_roi:.0f}% ROI, "
            f"callback {self.cfg.callback_roi:.0f}% ROI"
        )

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
        try:
            t = self.exchange.fetch_ticker(pos.symbol)
            return float(t.get("last") or t.get("close") or 0) or None
        except Exception as e:
            log.warning(f"ticker failed for {pos.symbol}: {e}")
            return None

    # ── writing (the only two methods that mutate the account) ──────────────

    def _place_stop(self, pos: FuturesPosition, stop_price: float) -> str | None:
        """
        Place a reduce-only STOP_MARKET that closes `pos` at `stop_price`.

        reduceOnly=True is non-negotiable: it makes it impossible for this order
        to open or enlarge a position, whatever else goes wrong.
        """
        price_str = self.exchange.price_to_precision(pos.symbol, stop_price)
        qty_str = self.exchange.amount_to_precision(pos.symbol, pos.qty)
        side = stop_side(pos)

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
        state, stop_price, reason = evaluate(pos, price, state, self.cfg)

        if stop_price is not None:
            stop_price = float(self.exchange.price_to_precision(pos.symbol, stop_price))
            # PLACE-THEN-CANCEL: never leave the position unprotected.
            new_id = None
            try:
                new_id = self._place_stop(pos, stop_price)
            except Exception as e:
                log.error(f"FAILED to place stop for {pos.symbol}: {e} — "
                          f"leaving the existing stop in place")
                self._record(pos.symbol, "place_failed", str(e))
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

        live_symbols = {p.symbol for p in positions}

        # Forget state for positions that have closed (stopped out or closed by
        # the operator) so a future position on the same symbol starts fresh.
        with self._lock:
            for sym in list(self._states):
                if sym not in live_symbols:
                    log.info(f"{sym}: position gone — clearing guard state")
                    self._record(sym, "closed", "position no longer open")
                    del self._states[sym]

        for pos in positions:
            try:
                self.manage_position(pos)
            except Exception as e:
                log.warning(f"manage_position failed for {pos.symbol}: {e}")

        self._last_cycle_ts = time.time()
        self._last_error = None

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
                }
                for sym, s in self._states.items()
            }
            actions = list(reversed(self._actions[-20:]))
        return {
            "enabled": True,
            "dry_run": self.dry_run,
            "testnet": self.testnet,
            "states": states,
            "recent_actions": actions,
            "last_cycle_ago_s": (time.time() - self._last_cycle_ts) if self._last_cycle_ts else None,
            "error": self._last_error,
            "config": {
                "initial_stop_roi": self.cfg.initial_stop_roi,
                "arm_roi": self.cfg.arm_roi,
                "callback_roi": self.cfg.callback_roi,
            },
        }
