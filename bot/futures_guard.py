"""
Futures position guardian — Phase 1.

Watches the Binance USD-M futures account for positions the OPERATOR opened
manually, and protects them:

  1. Immediately places a protective stop at -INITIAL_STOP_ROI% (closing the
     unprotected window between opening a position and setting a stop by hand).
  2. Tracks peak ROI. Once peak ROI reaches ARM_ROI%, arms a trailing stop that
     follows the peak by CALLBACK_ROI%, ratcheting up and never down.

It does NOT decide entries. It never opens a position. Every order it places is
reduce-only, so it can only ever shrink or close an existing position.

ROI convention matches the Binance UI: ROI% = PnL / margin * 100.
At L leverage, an ROI move of R% corresponds to a price move of R/L %.

Long and short are both supported — the direction-dependent maths is isolated in
roi_pct() and price_for_roi() so the rest of the logic is side-agnostic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("futures_guard")


@dataclass
class FuturesPosition:
    """A position as reported by the exchange."""
    symbol: str
    side: str            # "long" | "short"
    entry_price: float
    qty: float           # absolute size in contracts/base units
    leverage: int
    margin: float        # isolated margin backing the position

    def __post_init__(self):
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be long|short, got {self.side!r}")

    @property
    def notional(self) -> float:
        """Position size in quote terms: entry price x contracts."""
        return self.entry_price * self.qty

    @property
    def effective_leverage(self) -> float:
        """
        Leverage DERIVED from notional / margin rather than trusting the
        exchange's `leverage` field, which has been observed to come back as 1
        on isolated-margin positions. Getting this wrong scales every ROI figure
        and every stop distance by the leverage factor, so it is derived from
        two values that are reliably reported (size and margin) and only falls
        back to the reported field if margin is unusable.
        """
        if self.margin > 0 and self.notional > 0:
            return self.notional / self.margin
        return float(self.leverage or 1)


@dataclass
class GuardState:
    """Per-position tracking the guardian keeps in memory."""
    peak_roi: float = 0.0          # highest ROI% seen (monotonic)
    armed: bool = False            # has the trailing stop armed?
    stop_order_id: str | None = None
    stop_roi: float | None = None  # ROI level the resting stop sits at
    # Order id of the native Binance trailing stop, once armed. While this is
    # set the exchange manages the trail tick-by-tick and the guardian stops
    # repositioning anything.
    native_trail_id: str | None = None


@dataclass
class GuardConfig:
    """
    All thresholds in ROI% (Binance UI convention: PnL / margin * 100).

    Defaults encode the agreed Option A:
      - initial stop at -7% ROI
      - trailing arms at +15% ROI
      - callback 10% ROI (narrower than the 15% arm, so arming LOCKS IN +5% ROI)
    """
    initial_stop_roi: float = 7.0    # positive number; applied as -7% ROI
    arm_roi: float = 15.0
    callback_roi: float = 10.0
    # Native Binance trailing stop used for the ARMED phase, expressed as a
    # PRICE percentage (Binance's callbackRate). At L leverage this equals
    # callback_pct * L in ROI terms, so its ROI cost depends on the position's
    # leverage and cannot be validated at startup — it is checked at arm time.
    trail_callback_pct: float = 1.0
    use_native_trail: bool = True
    # Only move a resting stop if the new level differs by at least this much
    # ROI, to avoid spamming cancel/replace on every tick.
    min_stop_move_roi: float = 1.0

    def validate(self):
        """
        Enforce the invariant that makes the trail actually protect a gain.

        If callback >= arm, then at the moment the trail arms the stop sits at
        (arm - callback) <= 0 — i.e. at or BELOW entry. The operator would be up
        arm% ROI and still exit at a loss. This is exactly the flaw found in the
        manual setup (activation +15% ROI with a 2% price callback = 20% ROI),
        so the guardian refuses to run a config that reproduces it.
        """
        assert self.initial_stop_roi > 0, "initial_stop_roi must be positive"
        assert self.arm_roi > 0, "arm_roi must be positive"
        assert self.callback_roi > 0, "callback_roi must be positive"
        assert self.callback_roi < self.arm_roi, (
            f"callback_roi ({self.callback_roi}) must be LESS than arm_roi "
            f"({self.arm_roi}) — otherwise arming the trail puts the stop at or "
            f"below entry, turning a winning trade into a loss."
        )
        return self


# ── Direction-aware maths (the only place long/short differ) ─────────────────

def roi_pct(pos: FuturesPosition, price: float) -> float:
    """
    ROI% as the Binance UI reports it: PnL / margin * 100.

    Equivalent to the price move times leverage, sign-corrected for side.
    """
    if pos.entry_price <= 0:
        return 0.0
    if pos.side == "long":
        price_move = (price - pos.entry_price) / pos.entry_price
    else:  # short profits when price falls
        price_move = (pos.entry_price - price) / pos.entry_price
    # ROI% = PnL / margin * 100, which equals price_move * leverage * 100.
    return price_move * pos.effective_leverage * 100


def price_for_roi(pos: FuturesPosition, target_roi: float) -> float:
    """
    The price at which this position reaches `target_roi` (may be negative ROI).

    Inverse of roi_pct(). For a long, higher ROI = higher price; for a short,
    higher ROI = lower price.
    """
    price_move = (target_roi / 100.0) / pos.effective_leverage
    if pos.side == "long":
        return pos.entry_price * (1 + price_move)
    return pos.entry_price * (1 - price_move)


def stop_side(pos: FuturesPosition) -> str:
    """The order side that closes this position."""
    return "sell" if pos.side == "long" else "buy"


# ── Order classification & adoption ──────────────────────────────────────────
#
# SAFETY RULE: the operator sometimes OPENS a position with a trailing-stop
# order (so the market must confirm direction before the fill). That resting
# order is an ENTRY, not protection. The guardian must never cancel it.
#
# The distinguishing property is reduceOnly: a protective stop can only ever
# shrink/close a position and is flagged reduce-only; an entry order is not.
# Therefore the guardian ONLY ever cancels or replaces reduce-only orders.

def is_protective_stop(order: dict, pos: FuturesPosition) -> bool:
    """
    True only for an order that is unambiguously protection for `pos`:
      - reduce-only (cannot open or increase a position), AND
      - on the side that closes this position, AND
      - a stop-type order.

    Anything failing these checks — notably a non-reduce-only trailing-stop
    ENTRY order — is treated as none of the guardian's business.
    """
    if not order.get("reduceOnly", order.get("reduce_only", False)):
        return False
    side = (order.get("side") or "").lower()
    if side != stop_side(pos):
        return False
    otype = (order.get("type") or "").upper().replace("-", "_")
    return "STOP" in otype


def adoptable_stop(orders: list[dict], pos: FuturesPosition) -> dict | None:
    """
    Find an existing protective stop for this position, so the guardian adopts
    and manages it rather than stacking a second stop on top.

    Returns the order dict, or None if the position is currently unprotected.
    Non-reduce-only orders (entries) are never returned.
    """
    for o in orders:
        if is_protective_stop(o, pos):
            return o
    return None


def adopt_state(pos: FuturesPosition, orders: list[dict], current_roi: float,
                cfg: GuardConfig) -> GuardState:
    """
    Build initial guard state for a position discovered at startup (or one the
    operator opened by hand), adopting any protective stop already resting.

    The adopted stop's price is converted back to an ROI level so the trailing
    logic continues from where the operator left off. Peak ROI is seeded from
    the current ROI so an already-profitable position doesn't immediately look
    like it has fallen from a peak of zero.
    """
    state = GuardState(peak_roi=max(current_roi, 0.0))
    existing = adoptable_stop(orders, pos)
    if existing is not None:
        stop_price = existing.get("stopPrice") or existing.get("triggerPrice")
        if stop_price:
            state.stop_roi = roi_pct(pos, float(stop_price))
            state.stop_order_id = str(existing.get("id") or existing.get("orderId") or "")
            log.info(
                f"{pos.symbol}: adopted existing protective stop at "
                f"{float(stop_price):.6f} ({state.stop_roi:+.1f}% ROI)"
            )
    state.armed = is_armed(state, cfg)
    return state


# ── Stop level decision (pure) ───────────────────────────────────────────────

# ROI comparisons run through float round-trips (price_for_roi -> roi_pct), which
# can land a hair below an exact threshold (e.g. 14.999999999999858 for 15.0).
# A tiny tolerance keeps an exact-boundary arm from being silently missed.
_ROI_EPS = 1e-6


def is_armed(state: GuardState, cfg: GuardConfig) -> bool:
    """Has the peak reached the arming threshold (float-tolerant)?"""
    return state.peak_roi >= (cfg.arm_roi - _ROI_EPS)


def callback_roi_at(leverage: float, cfg: GuardConfig) -> float:
    """The native trail's give-back expressed in ROI% for a given leverage."""
    return cfg.trail_callback_pct * max(leverage, 0.0)


def trail_locks_in(leverage: float, cfg: GuardConfig) -> float:
    """
    ROI the stop lands at the moment the trail arms.

    Positive means arming locks in profit. Zero or negative means the trail
    engages at or below entry — the flaw the callback<arm invariant exists to
    prevent, but here it depends on leverage so it is evaluated per position.
    """
    return cfg.arm_roi - callback_roi_at(leverage, cfg)


def desired_stop_roi(state: GuardState, cfg: GuardConfig) -> float:
    """
    The ROI level the protective stop should sit at, given the peak seen so far.

    Before the trail arms: the fixed initial stop at -initial_stop_roi.
    After it arms: peak - callback, which ratchets up with the peak and never
    falls back (peak_roi is monotonic). Because callback < arm (enforced in
    validate), the first armed level is strictly positive — the trade is locked
    into profit the moment the trail engages.
    """
    if is_armed(state, cfg):
        return state.peak_roi - cfg.callback_roi
    return -cfg.initial_stop_roi


def update_peak(state: GuardState, current_roi: float) -> GuardState:
    """Track the high-water ROI mark. Monotonic — never decreases."""
    if current_roi > state.peak_roi:
        state.peak_roi = current_roi
    return state


def should_replace_stop(state: GuardState, new_stop_roi: float,
                        cfg: GuardConfig) -> bool:
    """
    Whether the resting stop needs to be cancelled and re-placed.

    True when there is no stop yet, or when the desired level has moved in the
    FAVOURABLE direction by at least min_stop_move_roi. A stop is never moved
    to a worse level — protection only ever tightens.
    """
    if state.stop_roi is None:
        return True
    if new_stop_roi <= state.stop_roi:
        return False  # never loosen
    return (new_stop_roi - state.stop_roi) >= cfg.min_stop_move_roi


def evaluate(pos: FuturesPosition, price: float, state: GuardState,
             cfg: GuardConfig) -> tuple[GuardState, float | None, str]:
    """
    Full per-tick decision for one position (pure — no exchange calls).

    Returns (updated_state, stop_price_to_place_or_None, human_reason).
    """
    current_roi = roi_pct(pos, price)
    state = update_peak(state, current_roi)

    new_stop_roi = desired_stop_roi(state, cfg)
    newly_armed = (not state.armed) and is_armed(state, cfg)
    if newly_armed:
        state.armed = True

    if not should_replace_stop(state, new_stop_roi, cfg):
        return state, None, "stop unchanged"

    stop_price = price_for_roi(pos, new_stop_roi)
    state.stop_roi = new_stop_roi

    if newly_armed:
        reason = (f"trail ARMED at peak {state.peak_roi:+.1f}% ROI -> stop "
                  f"{new_stop_roi:+.1f}% ROI (locked in profit)")
    elif state.armed:
        reason = (f"trail ratcheted: peak {state.peak_roi:+.1f}% ROI -> stop "
                  f"{new_stop_roi:+.1f}% ROI")
    else:
        reason = f"initial protective stop at {new_stop_roi:+.1f}% ROI"

    return state, stop_price, reason
