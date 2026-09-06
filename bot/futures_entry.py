"""
Futures entry — operator-initiated position opening.

This is the ONLY component in the system that can open a position, so it is
deliberately the most constrained:

  * TWO STEP. `preview()` computes and returns a plan plus a short-lived
    confirm token; nothing is sent. `execute()` requires that exact token.
    A stray or replayed POST to execute cannot open a position.
  * SERVER-SIDE GUARDRAILS. Every limit is re-checked at execute time, not
    just when the preview was built. A UI that is buggy, stale, or bypassed
    still cannot exceed the limits.
  * DRY RUN. Honours the guardian's dry-run flag; logs the intended order and
    sends nothing.
  * ENTRY ORDER TYPE is TRAILING_STOP_MARKET, so the market must move in the
    intended direction by the callback rate before the fill happens. A short
    only fills once price is actually falling; a long once it is rising.

Once filled, the position is picked up by the FuturesGuardian on its next
cycle, which places the protective stop.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field

log = logging.getLogger("futures_entry")

# A preview must be confirmed within this window. Prevents acting on a plan
# priced against a market that has since moved.
CONFIRM_TTL_S = 60.0


@dataclass
class EntryLimits:
    """Hard limits, re-checked at execute time."""
    max_positions: int = 1              # operator runs one position at a time
    max_margin_pct: float = 25.0        # ceiling on margin as % of wallet
    default_margin_pct: float = 10.0
    default_callback_pct: float = 0.1   # Binance minimum
    min_callback_pct: float = 0.1
    max_callback_pct: float = 10.0
    # Operator-declared leverage, used ONLY when the exchange reports none
    # (seen on demo for symbols with no open position). Zero means "not
    # declared" and the entry is refused rather than sized on a guess.
    assumed_leverage: float = 0.0
    # ── Volatility-scaled sizing ────────────────────────────────────────
    # 0 disables (margin stays a flat % of wallet). When set, the stop sits
    # atr_stop_mult x ATR away and the position is sized so the loss at that
    # stop equals risk_pct of the wallet — so a volatile coin gets a wider stop
    # and a SMALLER position, keeping the money at risk constant.
    atr_stop_mult: float = 0.0
    risk_pct: float = 1.0
    atr_stop_min_roi: float = 4.0
    atr_stop_max_roi: float = 30.0


@dataclass
class EntryPlan:
    symbol: str
    side: str            # "long" | "short"
    order_side: str      # "buy" | "sell"
    margin_usdt: float
    notional_usdt: float
    qty: float
    leverage: float
    ref_price: float
    callback_pct: float
    wallet_balance: float
    margin_pct: float
    projected_stop_price: float | None = None
    projected_stop_roi: float | None = None
    projected_stop_loss_usdt: float | None = None
    atr_pct: float | None = None
    leverage_source: str = ""      # "info.leverage" | "assumed" | ...
    token: str = ""
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "order_side": self.order_side,
            "margin_usdt": round(self.margin_usdt, 2),
            "notional_usdt": round(self.notional_usdt, 2),
            "qty": self.qty,
            "leverage": round(self.leverage, 1),
            "leverage_source": self.leverage_source,
            "leverage_assumed": self.leverage_source == "assumed",
            "ref_price": self.ref_price,
            "callback_pct": self.callback_pct,
            "wallet_balance": round(self.wallet_balance, 2),
            "margin_pct": round(self.margin_pct, 2),
            "projected_stop_price": self.projected_stop_price,
            "projected_stop_roi": self.projected_stop_roi,
            "projected_stop_loss_usdt": self.projected_stop_loss_usdt,
            "atr_pct": self.atr_pct,
            "token": self.token,
            "expires_in_s": max(0, round(CONFIRM_TTL_S - (time.time() - self.created_at))),
        }


# ── pure helpers ─────────────────────────────────────────────────────────────

def compute_size(wallet_balance: float, margin_pct: float, leverage: float,
                 price: float) -> tuple[float, float, float]:
    """(margin, notional, qty) for a given wallet %, leverage and price."""
    if wallet_balance <= 0 or price <= 0 or leverage <= 0:
        return 0.0, 0.0, 0.0
    margin = wallet_balance * (margin_pct / 100.0)
    notional = margin * leverage
    return margin, notional, notional / price


def order_side_for(side: str) -> str:
    """Opening a long buys; opening a short sells."""
    if side not in ("long", "short"):
        raise ValueError(f"side must be long|short, got {side!r}")
    return "buy" if side == "long" else "sell"


def validate_request(*, side: str, margin_pct: float, callback_pct: float,
                     wallet_balance: float, open_positions: int,
                     symbol_has_position: bool, limits: EntryLimits) -> list[str]:
    """
    Every blocking reason for this request. Empty list == allowed.

    Called from BOTH preview and execute, so a plan that was valid when
    previewed is re-checked against the account as it is at execute time.
    """
    errs: list[str] = []
    if side not in ("long", "short"):
        errs.append(f"invalid side {side!r}")
    if wallet_balance <= 0:
        errs.append("wallet balance is zero or unavailable")
    if margin_pct <= 0:
        errs.append("margin % must be positive")
    if margin_pct > limits.max_margin_pct:
        errs.append(f"margin {margin_pct}% exceeds the {limits.max_margin_pct}% cap")
    if callback_pct < limits.min_callback_pct:
        errs.append(f"callback {callback_pct}% below the {limits.min_callback_pct}% minimum")
    if callback_pct > limits.max_callback_pct:
        errs.append(f"callback {callback_pct}% above the {limits.max_callback_pct}% maximum")
    if symbol_has_position:
        errs.append("a position is already open on this symbol")
    if open_positions >= limits.max_positions:
        errs.append(f"already at the {limits.max_positions}-position limit")
    return errs


class EntryService:
    """
    Builds and executes operator-initiated futures entries.

    Depends on a FuturesGuardian for exchange access, account state and the
    dry-run flag — deliberately, so entries cannot be enabled independently of
    the protection that will manage them.
    """

    def __init__(self, guardian, limits: EntryLimits | None = None):
        self.guardian = guardian
        self.limits = limits or EntryLimits()
        self._pending: dict[str, EntryPlan] = {}

    # -- account reads
    def wallet_balance(self) -> float:
        """
        Total USDT wallet balance backing futures positions.

        Delegates to the shared resolver so the figure sizing an entry is
        always the same one the dashboard displays — two different readings
        here would silently mis-size positions.
        """
        from .futures_guardian import resolve_usdt_balance
        bal = self.guardian.exchange.fetch_balance()
        val, source = resolve_usdt_balance(bal)
        if val > 0:
            log.info(f"Wallet balance {val:.2f} USDT (from {source})")
        else:
            log.warning("Could not resolve a positive USDT wallet balance")
        return val

    def _atr_pct(self, symbol: str) -> float | None:
        """
        ATR as a % of price, from the GUARDIAN's cached value.

        Previously this was a second, uncached implementation. When it failed
        transiently while the guardian's succeeded, the entry silently fell back
        to flat sizing while the guardian still placed a volatility-scaled stop
        — a full-size position behind a wide stop, which is how a trade lost 3x
        the configured risk. One source, one cache, one answer.
        """
        try:
            if hasattr(self.guardian, "atr_pct"):
                return self.guardian.atr_pct(symbol)
        except Exception as e:
            log.warning(f"{symbol}: ATR lookup failed: {type(e).__name__}: {e}")
        return None

    def _resolve_symbol(self, symbol: str) -> tuple[bool, str, str]:
        """
        Check the symbol is tradable on the account we will actually order on.

        Candidates surfaced by the scanner come from the live market feed; the
        trading account may be demo, whose market list can differ. A couple of
        common notations are tried before giving up so a scanner symbol like
        "X/USDT:USDT" still matches a market listed as "X/USDT".
        """
        ex = self.guardian.exchange
        try:
            if not getattr(ex, "markets", None):
                ex.load_markets()
        except Exception as e:
            return False, f"could not load markets from the trading account: {e}", symbol

        markets = getattr(ex, "markets", None) or {}
        if not markets:
            # Could not determine the market list. The exchange is the authority,
            # not this pre-check, so proceed and let the ticker/order surface a
            # real error rather than blocking on an unknown.
            log.warning("market list unavailable; skipping the tradability check")
            return True, "", symbol

        if symbol in markets:
            return True, "", symbol

        base = symbol.split("/")[0]
        for alt in (f"{base}/USDT:USDT", f"{base}/USDT", symbol.replace(":USDT", "")):
            if alt in markets:
                log.info(f"resolved {symbol} -> {alt} on the trading account")
                return True, "", alt

        return False, (
            f"{symbol} is not tradable on this account "
            f"({'demo' if getattr(self.guardian, 'demo', False) else 'live'}). "
            f"The scanner screens the live market, whose symbol list can differ."
        ), symbol

    def symbol_leverage_detail(self, symbol: str) -> tuple[float, str]:
        """
        Leverage for this symbol, plus the source it came from.

        Hazards handled:

        1. ccxt's parsed `leverage` field has been observed reporting 1 on
           isolated positions, so the RAW `info.leverage` is preferred.
        2. Demo does not report leverage for a symbol with no open position.
           A configured ENTRY_ASSUMED_LEVERAGE is used then, but the source is
           returned as "assumed" so the preview can label it — sizing is only
           correct if that value matches what is set on Binance.
        3. If nothing resolves and nothing is declared, this returns 0.0 so the
           entry is REFUSED. A silent 1.0 previously mis-sized an order 10x.
        """
        candidates: list[tuple[str, object]] = []
        ex = self.guardian.exchange
        try:
            for p in ex.fetch_positions([symbol]):
                info = p.get("info") or {}
                candidates.append(("info.leverage", info.get("leverage")))
                candidates.append(("leverage", p.get("leverage")))
        except Exception as e:
            log.warning(f"leverage lookup failed for {symbol}: {e}")

        # ccxt's fetch_positions_risk DROPS any position with entryPrice <= 0,
        # so a symbol with no open position is filtered out and its leverage
        # goes with it — on live and demo alike. Query the raw endpoint that
        # ccxt is filtering, which reports leverage for every symbol.
        if not any(v for _, v in candidates):
            market_id = None
            try:
                market_id = ex.market_id(symbol)
            except Exception:
                market_id = symbol.split("/")[0] + "USDT"
            for meth in ("fapiPrivateV3GetPositionRisk",
                         "fapiPrivateV2GetPositionRisk",
                         "fapiPrivateGetPositionRisk"):
                fn = getattr(ex, meth, None)
                if fn is None:
                    continue
                try:
                    rows = fn({"symbol": market_id})
                    for row in (rows if isinstance(rows, list) else [rows]):
                        if (row or {}).get("leverage"):
                            candidates.append((meth, row.get("leverage")))
                    if any(v for _, v in candidates):
                        break
                except Exception as e:
                    log.debug(f"{meth} failed for {symbol}: {e}")

        if not any(v for _, v in candidates):
            try:
                if hasattr(ex, "fetch_leverage"):
                    lv = ex.fetch_leverage(symbol)
                    val = (lv.get("leverage") or (lv.get("info") or {}).get("leverage")) \
                        if isinstance(lv, dict) else lv
                    candidates.append(("fetch_leverage", val))
            except Exception as e:
                log.debug(f"fetch_leverage unavailable for {symbol}: {e}")

        for source, raw in candidates:
            try:
                lev = float(raw)
            except (TypeError, ValueError):
                continue
            if lev > 0:
                log.info(f"{symbol}: leverage {lev:g}x (from {source})")
                return lev, source

        declared = getattr(self.limits, "assumed_leverage", 0) or 0
        if declared > 0:
            log.warning(
                f"{symbol}: exchange reported no leverage — ASSUMING {declared:g}x "
                f"(ENTRY_ASSUMED_LEVERAGE). Sizing is only correct if this matches "
                f"the leverage set on Binance for this symbol."
            )
            return float(declared), "assumed"

        log.error(f"{symbol}: could not resolve leverage and none is declared")
        return 0.0, ""

    def symbol_leverage(self, symbol: str) -> float:
        return self.symbol_leverage_detail(symbol)[0]

    def _account_state(self, symbol: str) -> tuple[int, bool]:
        positions = self.guardian.fetch_positions()
        return len(positions), any(p.symbol == symbol for p in positions)

    # -- step 1
    def preview(self, symbol: str, side: str, margin_pct: float | None = None,
                callback_pct: float | None = None) -> dict:
        margin_pct = self.limits.default_margin_pct if margin_pct is None else float(margin_pct)
        callback_pct = self.limits.default_callback_pct if callback_pct is None else float(callback_pct)

        balance = self.wallet_balance()
        open_count, has_pos = self._account_state(symbol)
        errors = validate_request(
            side=side, margin_pct=margin_pct, callback_pct=callback_pct,
            wallet_balance=balance, open_positions=open_count,
            symbol_has_position=has_pos, limits=self.limits,
        )
        if errors:
            return {"ok": False, "errors": errors}

        # The scanner screens the LIVE market while entries execute on the
        # trading account (demo or live). Those universes are not identical, so
        # confirm the symbol exists here before anything else — otherwise the
        # failure surfaces as a vague "no price" further down.
        ok, why, resolved = self._resolve_symbol(symbol)
        if not ok:
            return {"ok": False, "errors": [why]}
        symbol = resolved

        try:
            ticker = self.guardian.exchange.fetch_ticker(symbol)
        except Exception as e:
            log.warning(f"ticker fetch failed for {symbol}: {e}")
            return {"ok": False, "errors": [
                f"could not read a price for {symbol}: {type(e).__name__}: {e}"]}
        price = float(ticker.get("last") or ticker.get("close") or 0)
        if price <= 0:
            return {"ok": False, "errors": [
                f"{symbol} returned no usable price (ticker had no last/close)"]}

        leverage, lev_source = self.symbol_leverage_detail(symbol)
        if leverage <= 0:
            return {"ok": False, "errors": [
                "could not read this symbol's leverage from Binance — refusing "
                "to size an entry. Set leverage for the symbol first."]}

        # Volatility-scaled sizing, when enabled: derive the stop from ATR and
        # size the position so the loss at that stop is a fixed % of the wallet.
        atr_pct = None
        stop_roi_override = None
        if getattr(self.limits, "atr_stop_mult", 0):
            atr_pct = self._atr_pct(symbol)
            if atr_pct:
                raw_roi = self.limits.atr_stop_mult * atr_pct * leverage
                stop_roi_override = max(self.limits.atr_stop_min_roi,
                                        min(self.limits.atr_stop_max_roi, raw_roi))
            else:
                # Refuse rather than fall back to flat sizing. The guardian will
                # still place an ATR-scaled stop, so a flat-sized position would
                # sit behind a stop it was not sized for — the exact mismatch
                # that cost 3x the intended risk on a single trade.
                return {"ok": False, "errors": [
                    f"volatility sizing is enabled but ATR is unavailable for "
                    f"{symbol} — refusing rather than sizing this at a flat "
                    f"{margin_pct}% behind a volatility-scaled stop"]}

        if stop_roi_override:
            stop_move_pct = stop_roi_override / leverage      # price % to the stop
            risk_usdt = balance * (self.limits.risk_pct / 100.0)
            notional = risk_usdt / (stop_move_pct / 100.0)
            margin = notional / leverage
            # Never exceed the wallet ceiling regardless of what ATR suggests.
            cap = balance * (self.limits.max_margin_pct / 100.0)
            if margin > cap:
                margin = cap
                notional = margin * leverage
            qty_raw = notional / price
        else:
            margin, notional, qty_raw = compute_size(balance, margin_pct, leverage, price)
        # Never size beyond the exchange's per-ORDER quantity cap. A position
        # above it cannot have a single protective stop placed (-4005), which is
        # how a position ended up unprotected while the guardian retried.
        try:
            m = self.guardian.exchange.market(symbol)
            max_qty = (((m or {}).get("limits") or {}).get("amount") or {}).get("max")
            if max_qty and qty_raw > float(max_qty):
                log.warning(
                    f"{symbol}: size {qty_raw:g} exceeds the per-order cap "
                    f"{float(max_qty):g} — trimming so the stop can be placed")
                qty_raw = float(max_qty)
        except Exception:
            pass

        qty = float(self.guardian.exchange.amount_to_precision(symbol, qty_raw))
        if qty <= 0:
            return {"ok": False, "errors": ["computed quantity rounds to zero — margin too small"]}

        # Lot-size rounding can shrink a position substantially on high-priced
        # coins with coarse steps. Silently shipping a fraction of the requested
        # size is how a 10.7 USDT margin became 0.7, so surface it loudly and
        # report the size that will ACTUALLY be opened, not the requested one.
        size_warnings: list[str] = []
        if qty_raw > 0:
            shrink = (qty_raw - qty) / qty_raw
            if shrink >= 0.10:
                size_warnings.append(
                    f"lot rounding reduced size {shrink*100:.0f}% "
                    f"({qty_raw:.4f} -> {qty:g}) — actual margin will be "
                    f"{qty * price / leverage:.2f} USDT, not {margin:.2f}"
                )
        # Report the real, post-rounding economics.
        notional = qty * price
        margin = notional / leverage

        # Show where the guardian's initial stop will land, so the operator sees
        # the downside before confirming rather than after the fill.
        from .futures_guard import FuturesPosition, price_for_roi
        provisional = FuturesPosition(symbol, side, price, qty, int(leverage), margin)
        # Use the SAME stop the guardian will place. With volatility scaling on,
        # that is the ATR-derived level, not the fixed default — reporting the
        # fixed one would understate or overstate the real downside.
        stop_roi = -(stop_roi_override or self.guardian.cfg.initial_stop_roi)
        stop_price = price_for_roi(provisional, stop_roi)
        risk_usdt_at_stop = margin * abs(stop_roi) / 100.0

        plan = EntryPlan(
            symbol=symbol, side=side, order_side=order_side_for(side),
            margin_usdt=margin, notional_usdt=notional, qty=qty,
            leverage=leverage, leverage_source=lev_source,
            ref_price=price, callback_pct=callback_pct,
            wallet_balance=balance, margin_pct=margin_pct,
            projected_stop_price=float(self.guardian.exchange.price_to_precision(symbol, stop_price)),
            projected_stop_roi=stop_roi,
            projected_stop_loss_usdt=round(risk_usdt_at_stop, 2),
            atr_pct=(None if atr_pct is None else round(atr_pct, 3)),
            token=secrets.token_urlsafe(12),
        )
        self._pending[plan.token] = plan
        self._prune()
        if lev_source == "assumed":
            size_warnings.insert(0, (
                f"leverage {leverage:g}x is ASSUMED (the exchange reported none) "
                f"— confirm this matches the leverage set on Binance for {symbol}"))
        if size_warnings:
            for w in size_warnings:
                log.warning(f"{symbol}: {w}")
        return {"ok": True, "plan": plan.as_dict(),
                "warnings": size_warnings,
                "dry_run": self.guardian.dry_run}

    def _prune(self):
        now = time.time()
        for tok in [t for t, p in self._pending.items()
                    if now - p.created_at > CONFIRM_TTL_S]:
            self._pending.pop(tok, None)

    # -- step 2
    def execute(self, token: str) -> dict:
        self._prune()
        plan = self._pending.pop(token, None)
        if plan is None:
            return {"ok": False, "errors": ["confirmation expired or already used — preview again"]}

        # Re-validate against the account as it is NOW, not as it was at preview.
        balance = self.wallet_balance()
        open_count, has_pos = self._account_state(plan.symbol)
        errors = validate_request(
            side=plan.side, margin_pct=plan.margin_pct, callback_pct=plan.callback_pct,
            wallet_balance=balance, open_positions=open_count,
            symbol_has_position=has_pos, limits=self.limits,
        )
        if errors:
            return {"ok": False, "errors": errors}

        if self.guardian.dry_run:
            log.warning(
                f"[DRY RUN] would open {plan.side.upper()} {plan.symbol}: "
                f"{plan.order_side} TRAILING_STOP_MARKET qty={plan.qty} "
                f"callback={plan.callback_pct}% margin={plan.margin_usdt:.2f} "
                f"notional={plan.notional_usdt:.2f}"
            )
            return {"ok": True, "dry_run": True, "plan": plan.as_dict(),
                    "message": "Dry run — no order sent."}

        try:
            order = self.guardian.exchange.create_order(
                symbol=plan.symbol, type="TRAILING_STOP_MARKET",
                side=plan.order_side, amount=plan.qty, price=None,
                params={"callbackRate": plan.callback_pct, "reduceOnly": False},
            )
        except Exception as e:
            log.exception("entry order failed")
            return {"ok": False, "errors": [f"{type(e).__name__}: {e}"]}

        oid = str(order.get("id") or order.get("orderId") or "")
        log.warning(
            f"ENTRY PLACED {plan.side.upper()} {plan.symbol} id={oid} "
            f"qty={plan.qty} callback={plan.callback_pct}% "
            f"margin={plan.margin_usdt:.2f} notional={plan.notional_usdt:.2f}"
        )
        return {"ok": True, "dry_run": False, "order_id": oid,
                "plan": plan.as_dict(),
                "message": ("Trailing-stop entry placed. It fills only once price "
                            "moves in your direction by the callback rate. The "
                            "guardian will protect the position once filled.")}
