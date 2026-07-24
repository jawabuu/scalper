"""
Pullback strategy — mean-reversion / good-price scalp.

Two regimes:
  - GAINER: coin clearly trending up; scalp on momentum. 24h low not required.
  - DIPPER: coin falling but high-volume; enter near 24h-low support as it stabilizes.

Both share a universal "good entry price" gate: never enter high in the candle,
never at the tip of a rejection (upper-wick) spike.

This module is PURE LOGIC — it takes a prepared DataFrame + config and returns a
decision. No network, no exchange, no side effects — so it is fully unit-testable.
The engine calls `evaluate_entry()` and acts on the result.

Exits are handled separately (strictly: TP at MA10 or fixed %, wick SL bounded,
5-candle timeout).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import pandas as pd
import pandas_ta as ta


@dataclass
class PullbackDecision:
    enter: bool
    regime: str | None = None          # "gainer" | "dipper" | None
    reason: str = ""                   # human-readable why enter / why skip
    stop_price: float | None = None    # computed wick-based stop (bounded)
    entry_ref_price: float | None = None
    stamps: dict | None = None         # logged diagnostics (RSI, vol ratio, etc.)


def prepare_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add the indicators the pullback strategy needs to a raw OHLCV frame."""
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=cfg.pb_ema_fast)
    df["ema_slow"] = ta.ema(df["close"], length=cfg.pb_ema_slow)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["price_ma"] = ta.sma(df["close"], length=cfg.pb_price_ma_len)
    df["vol_ma"] = ta.sma(df["volume"], length=cfg.pb_vol_ma_len)
    return df


def candle_position(row) -> float:
    """Where the close sits within the candle range: 0.0=low, 1.0=high."""
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 0.5
    return (row["close"] - row["low"]) / rng


def upper_wick_fraction(row) -> float:
    """Upper wick length as a fraction of the total candle range."""
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 0.0
    body_top = max(row["open"], row["close"])
    return (row["high"] - body_top) / rng


def lower_wick_fraction(row) -> float:
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 0.0
    body_bot = min(row["open"], row["close"])
    return (body_bot - row["low"]) / rng


def passes_good_price(row, cfg) -> tuple[bool, str]:
    """
    Universal good-entry-price gate. Reject entries high in the candle or under a
    long upper (rejection) wick.
    """
    pos = candle_position(row)
    if pos > cfg.pb_candle_pos_max:
        return False, f"price high in candle (pos={pos:.2f} > {cfg.pb_candle_pos_max})"
    uw = upper_wick_fraction(row)
    # A long upper wick with price near the top = rejection spike → skip.
    if uw >= cfg.pb_upper_wick_max and pos > 0.5:
        return False, f"rejection upper-wick (wick={uw:.2f}, pos={pos:.2f})"
    return True, "good price"


def rsi_rising(df: pd.DataFrame, cfg) -> bool:
    """
    RSI must be ticking up on the confirmation candle AND be above its value
    `pb_rsi_rising_lookback` candles back — blocks a genuine multi-candle decline
    (e.g. "declining 15 min, now at 50").
    """
    lb = cfg.pb_rsi_rising_lookback
    if len(df) < lb + 1:
        return False
    rsi_now = df["rsi"].iloc[-1]
    rsi_prev = df["rsi"].iloc[-2]
    rsi_back = df["rsi"].iloc[-1 - lb]
    if pd.isna(rsi_now) or pd.isna(rsi_prev) or pd.isna(rsi_back):
        return False
    return (rsi_now > rsi_prev) and (rsi_now > rsi_back)


def ema_uptrend(row, cfg) -> bool:
    """EMA fast clearly above slow (with a small buffer so near-crossovers fail)."""
    if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]):
        return False
    return row["ema_fast"] >= row["ema_slow"] * (1 + cfg.pb_ema_buffer_pct / 100)


def _bounded_wick_stop(df: pd.DataFrame, entry_price: float, cfg):
    """
    Stop just below the recent candle's low (the dip wick). Returns (stop, too_deep).
    If the implied stop distance exceeds pb_max_wick_stop_pct, flag too_deep so the
    caller SKIPS the trade (a flush that deep is a breakdown, not a clean setup).
    """
    recent_low = min(df["low"].iloc[-1], df["low"].iloc[-2]) if len(df) >= 2 else df["low"].iloc[-1]
    stop = recent_low * 0.999  # a hair below the wick
    dist_pct = (entry_price - stop) / entry_price * 100
    too_deep = dist_pct > cfg.pb_max_wick_stop_pct
    return stop, too_deep, dist_pct


def evaluate_entry(df: pd.DataFrame, cfg, *, vol_24h_usdt: float | None = None,
                   low_24h: float | None = None) -> PullbackDecision:
    """
    Decide whether to enter, using the confirmation candle (last row of df).
    df must already have indicators (call prepare_indicators first).
    """
    if len(df) < max(cfg.pb_ema_slow, cfg.pb_rsi_rising_lookback + 2, cfg.pb_price_ma_len):
        return PullbackDecision(False, reason="insufficient history")

    row = df.iloc[-1]
    entry_price = float(row["close"])
    rsi_now = float(row["rsi"]) if not pd.isna(row["rsi"]) else None

    stamps = {
        "rsi": round(rsi_now, 2) if rsi_now is not None else None,
        "candle_pos": round(candle_position(row), 3),
        "upper_wick": round(upper_wick_fraction(row), 3),
        "vol_ratio": None,
    }
    if not pd.isna(row["vol_ma"]) and row["vol_ma"] > 0:
        stamps["vol_ratio"] = round(row["volume"] / row["vol_ma"], 2)

    # ── Universal good-price gate ──
    ok, why = passes_good_price(row, cfg)
    if not ok:
        return PullbackDecision(False, reason=why, stamps=stamps)

    # ── Volume expansion (strict, both regimes) ──
    if pd.isna(row["vol_ma"]) or row["vol_ma"] <= 0:
        return PullbackDecision(False, reason="no volume MA", stamps=stamps)
    if row["volume"] < cfg.pb_vol_spike_mult * row["vol_ma"]:
        return PullbackDecision(False,
            reason=f"no volume spike (ratio={stamps['vol_ratio']} < {cfg.pb_vol_spike_mult})",
            stamps=stamps)

    # ── Determine regime ──
    up = ema_uptrend(row, cfg)

    # GAINER: clear uptrend + RSI band + RSI rising.
    if up and cfg.pb_gainer_enabled:
        if rsi_now is None or not (cfg.pb_rsi_min <= rsi_now <= cfg.pb_rsi_max):
            return PullbackDecision(False,
                reason=f"gainer RSI out of band ({rsi_now})", stamps=stamps)
        if not rsi_rising(df, cfg):
            return PullbackDecision(False, reason="gainer RSI not rising", stamps=stamps)
        stop, too_deep, dist = _bounded_wick_stop(df, entry_price, cfg)
        if too_deep:
            return PullbackDecision(False,
                reason=f"wick stop too deep ({dist:.2f}% > {cfg.pb_max_wick_stop_pct}%)",
                stamps=stamps)
        return PullbackDecision(True, regime="gainer",
            reason="gainer: uptrend + RSI rising in band + vol spike + good price",
            stop_price=stop, entry_ref_price=entry_price, stamps=stamps)

    # DIPPER: not a clear uptrend, but near 24h-low support and stabilizing.
    if not up and cfg.pb_dipper_enabled:
        if low_24h is None:
            return PullbackDecision(False, reason="dipper: no 24h low available", stamps=stamps)
        proximity_pct = (entry_price - low_24h) / low_24h * 100
        stamps["low_proximity_pct"] = round(proximity_pct, 2)
        if proximity_pct > cfg.pb_low_proximity_pct:
            return PullbackDecision(False,
                reason=f"dipper: not near 24h low ({proximity_pct:.2f}% > {cfg.pb_low_proximity_pct}%)",
                stamps=stamps)
        # Stabilizing = RSI turning up (even if below the gainer band).
        if not rsi_rising(df, cfg):
            return PullbackDecision(False, reason="dipper: RSI not turning up", stamps=stamps)
        stop, too_deep, dist = _bounded_wick_stop(df, entry_price, cfg)
        if too_deep:
            return PullbackDecision(False,
                reason=f"dipper wick stop too deep ({dist:.2f}% > {cfg.pb_max_wick_stop_pct}%)",
                stamps=stamps)
        return PullbackDecision(True, regime="dipper",
            reason="dipper: near 24h low + RSI turning up + vol spike + good price",
            stop_price=stop, entry_ref_price=entry_price, stamps=stamps)

    return PullbackDecision(False, reason="no regime matched", stamps=stamps)


# ── Session windows ─────────────────────────────────────────────────────────

def parse_windows(spec: str) -> list[tuple[int, int]]:
    """Parse 'HH:MM-HH:MM,HH:MM-HH:MM' into a list of (start_min, end_min) pairs."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or "-" not in part:
            continue
        a, b = part.split("-", 1)
        try:
            ah, am = map(int, a.split(":"))
            bh, bm = map(int, b.split(":"))
            out.append((ah * 60 + am, bh * 60 + bm))
        except ValueError:
            continue
    return out


def in_session(now_utc: datetime, cfg) -> bool:
    """Is `now_utc` within any configured session window (in the local tz offset)?"""
    windows = parse_windows(cfg.pb_session_windows)
    if not windows:
        return True  # empty = always active
    local = now_utc + timedelta(hours=cfg.pb_session_tz_offset)
    minute_of_day = local.hour * 60 + local.minute
    for start, end in windows:
        if start <= end:
            if start <= minute_of_day <= end:
                return True
        else:  # wraps midnight
            if minute_of_day >= start or minute_of_day <= end:
                return True
    return False
