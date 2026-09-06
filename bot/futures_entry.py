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
            "ref_price": self.ref_price,
            "callback_pct": self.callback_pct,
            "wallet_balance": round(self.wallet_balance, 2),
            "margin_pct": round(self.margin_pct, 2),
            "projected_stop_price": self.projected_stop_price,
            "projected_stop_roi": self.projected_stop_roi,
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

        Position size is a percentage of this, so a wrong reading silently
        mis-sizes every entry. Several shapes are tried because the futures
        balance payload differs between environments (live vs demo), and the
        resolved value is logged so a surprising size can be traced.
        """
        bal = self.guardian.exchange.fetch_balance()
        candidates = []

        usdt = bal.get("USDT") or {}
        if isinstance(usdt, dict):
            candidates.append(("USDT.total", usdt.get("total")))
            candidates.append(("USDT.free", usdt.get("free")))

        total_map = bal.get("total") or {}
        if isinstance(total_map, dict):
            candidates.append(("total.USDT", total_map.get("USDT")))

        info = bal.get("info") or {}
        if isinstance(info, dict):
            candidates.append(("info.totalWalletBalance", info.get("totalWalletBalance")))
            candidates.append(("info.availableBalance", info.get("availableBalance")))
            assets = info.get("assets")
            if isinstance(assets, list):
                for a in assets:
                    if (a or {}).get("asset") == "USDT":
                        candidates.append(("assets[USDT].walletBalance",
                                           a.get("walletBalance")))

        for source, raw in candidates:
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val > 0:
                log.info(f"Wallet balance {val:.2f} USDT (from {source})")
                return val

        log.warning(f"Could not resolve a positive USDT wallet balance; "
                    f"tried {[c[0] for c in candidates]}")
        return 0.0

    def symbol_leverage(self, symbol: str) -> float:
        """
        The leverage the operator configured for this symbol on Binance. The
        guardian never sets leverage; it is read so sizing matches reality.

        Two hazards handled here:

        1. ccxt's parsed `leverage` field has been observed reporting 1 on
           isolated positions, so the RAW `info.leverage` from Binance is
           preferred — that is the authoritative value.
        2. On failure this returns 0.0, NOT 1.0. A silent 1.0 sized an entry at
           a tenth of the intended notional (which then floored to the minimum
           lot), producing a position a fraction of the requested size. Zero
           makes the plan invalid so the entry is REFUSED instead of mis-sized.
        """
        raw_candidates: list[tuple[str, object]] = []
        try:
            for p in self.guardian.exchange.fetch_positions([symbol]):
                info = p.get("info") or {}
                # Raw Binance value first — the parsed field is less reliable.
                raw_candidates.append(("info.leverage", info.get("leverage")))
                raw_candidates.append(("leverage", p.get("leverage")))
        except Exception as e:
            log.warning(f"leverage lookup failed for {symbol}: {e}")

        for source, raw in raw_candidates:
            try:
                lev = float(raw)
            except (TypeError, ValueError):
                continue
            if lev > 0:
                log.info(f"{symbol}: leverage {lev:g}x (from {source})")
                return lev

        log.error(
            f"{symbol}: could not resolve leverage — refusing to size an entry. "
            f"Set leverage for this symbol on Binance first."
        )
        return 0.0

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

        ticker = self.guardian.exchange.fetch_ticker(symbol)
        price = float(ticker.get("last") or ticker.get("close") or 0)
        if price <= 0:
            return {"ok": False, "errors": ["could not read a price for this symbol"]}

        leverage = self.symbol_leverage(symbol)
        if leverage <= 0:
            return {"ok": False, "errors": [
                "could not read this symbol's leverage from Binance — refusing "
                "to size an entry. Set leverage for the symbol first."]}

        margin, notional, qty_raw = compute_size(balance, margin_pct, leverage, price)
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
        stop_roi = -self.guardian.cfg.initial_stop_roi
        stop_price = price_for_roi(provisional, stop_roi)

        plan = EntryPlan(
            symbol=symbol, side=side, order_side=order_side_for(side),
            margin_usdt=margin, notional_usdt=notional, qty=qty,
            leverage=leverage, ref_price=price, callback_pct=callback_pct,
            wallet_balance=balance, margin_pct=margin_pct,
            projected_stop_price=float(self.guardian.exchange.price_to_precision(symbol, stop_price)),
            projected_stop_roi=stop_roi,
            token=secrets.token_urlsafe(12),
        )
        self._pending[plan.token] = plan
        self._prune()
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
