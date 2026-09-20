"""
Candle Range Theory — DETECTION ONLY. Nothing here can change a trade.

WHY THIS EXISTS
---------------
Entry timing, not selection, is where the losses are. Today's live measurement:

    79.1% of entries are ALREADY LOSING the first time the guardian sees them
    entered green -> n=9,  mean net ROI +3.60, win 55.6%
    entered red   -> n=34, mean net ROI -3.37, win 26.5%

and NOTHING in the ~40-field entry context separates the two (every field's
median split came in under 1 ROI point, none monotonic). So the fix is not
another screening gate — it is a different TRIGGER.

The current trigger is distance: enter once price retraces AUTO_CALLBACK_*.
That says nothing about whether the move is over. CRT's trigger is structural:
a push beyond a level that FAILED, confirmed by a close back inside. The
`wait_*` columns already say the better price is one candle later (81% of the
time, median +0.26%); CRT is a rule for WHICH candle.

THE RULE
--------
Three candles. The first defines the range — its high is CRH, its low is CRL.
The second must sweep one extreme AND CLOSE BACK INSIDE. If it closes beyond,
the setup is dead: price is more likely to continue than to reverse. The third
is where entry would happen.

For a SHORT, the sweep is above: high > CRH and close <= CRH — a failed push
up. That is the counter-trend form, and it is the one that fits this bot,
which fades strength (RSI >= 75 on coins up 8%+). The trend-following variant
described alongside CRT does the opposite — it buys dips in an uptrend — and
would confirm the reverse of every trade taken here.

WHAT THIS IS NOT
----------------
Not validated. The source material is a walk-through of selected examples with
no test in it. This records whether the condition held at entry so the question
can be settled against this account's own trades, rather than someone's chart.
"""
from __future__ import annotations


def _block(candles: list, lo: int, hi: int) -> tuple:
    """(high, low, close) of candles[lo:hi] treated as one larger candle."""
    part = candles[lo:hi]
    if not part:
        return None, None, None
    highs = [c[1] for c in part]
    lows = [c[2] for c in part]
    return max(highs), min(lows), part[-1][3]


def detect(candles: list, group: int = 5) -> dict:
    """
    `candles` are CLOSED candles, oldest first, each (ts, high, low, close).

    `group` aggregates the entry timeframe into the range timeframe — 5 means
    five 3m candles make one 15m range candle. Aggregating costs no extra API
    call, which is why the range is built this way rather than fetched.

    Returns a flat dict, always the same keys, so a row can be stamped
    unconditionally. `swept` is None when there is not enough data — never
    False, because "no opinion" and "no sweep" must not look alike in the
    analysis later.
    """
    out = {
        "crt_ok": False,          # enough data to judge
        "crt_crh": None,
        "crt_crl": None,
        "crt_swept": None,        # None = unknown, not "no"
        "crt_side": None,         # "short" on a high sweep, "long" on a low
        "crt_penetration_pct": None,   # how far beyond the level the wick went
        "crt_close_pos": None,    # where the sweep candle closed in the range
        "crt_group": int(group),
    }
    try:
        group = max(1, int(group))
        if not candles or len(candles) < group * 2:
            return out
        rng_hi, rng_lo, _ = _block(candles, -group * 2, -group)
        sw_hi, sw_lo, sw_close = _block(candles, -group, len(candles))
        if None in (rng_hi, rng_lo, sw_hi, sw_lo, sw_close):
            return out
        if rng_hi <= rng_lo:
            return out            # a zero-height range cannot be swept

        out["crt_ok"] = True
        out["crt_crh"] = float(rng_hi)
        out["crt_crl"] = float(rng_lo)
        out["crt_close_pos"] = round(
            (float(sw_close) - rng_lo) / (rng_hi - rng_lo), 4)

        # A sweep is a wick beyond the level AND a close back inside. A close
        # BEYOND is the invalidation, not a weaker version of the same thing.
        high_swept = sw_hi > rng_hi and sw_close <= rng_hi
        low_swept = sw_lo < rng_lo and sw_close >= rng_lo

        if high_swept and not low_swept:
            out["crt_swept"] = True
            out["crt_side"] = "short"          # a failed push UP
            out["crt_penetration_pct"] = round(
                (sw_hi - rng_hi) / rng_hi * 100, 4)
        elif low_swept and not high_swept:
            out["crt_swept"] = True
            out["crt_side"] = "long"           # a failed push DOWN
            out["crt_penetration_pct"] = round(
                (rng_lo - sw_lo) / rng_lo * 100, 4)
        elif high_swept and low_swept:
            # Both extremes taken and closed inside. CRT says nothing about
            # which side wins, so this is recorded as no signal rather than
            # guessed at.
            out["crt_swept"] = False
            out["crt_side"] = "both"
        else:
            out["crt_swept"] = False
        return out
    except Exception:
        # Detection must never be able to break a scan.
        return out


def agrees(crt: dict, side: str) -> bool | None:
    """
    Did CRT confirm THIS trade's direction?

    None when there was no opinion, so a missing judgement is never counted as
    a disagreement in the analysis.
    """
    if not crt or not crt.get("crt_ok"):
        return None
    if crt.get("crt_swept") is None:
        return None
    if not crt.get("crt_swept"):
        return False
    return str(crt.get("crt_side") or "") == str(side or "").lower()
