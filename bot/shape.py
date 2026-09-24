"""
SHAPE — the path, not the point. Computed in code, recorded, never gating.

WHY THIS EXISTS
---------------
Every field in `_candidate_state` is a SCALAR describing the present bar:
rsi, atr_pct, body_pct, taper_ratio, turn_rise_pct. There is no sequence
anywhere in it. A chart is a PATH; the state is a POINT.

The operator's goal — "consolidating before the breakout rather than already
extended" — is a statement about the last N bars, and none of the ~40 entry
fields can express it. That is consistent with the central finding: 79.1% of
entries are already losing on first sight, and NOTHING in those 40 fields
separates a green start from a red one.

So this asks the two questions an eye actually answers:

    CONSOLIDATING or EXTENDED     — is price coiling, or has it already run?
    ACCELERATING or DECELERATING  — is the current leg gaining or losing pace?

WHY CODE AND NOT A MODEL
------------------------
Range contraction, bar overlap and leg velocity are arithmetic. A decision
model should never be spent on arithmetic — and a computed baseline is what
any future model-judged shape question has to beat. jev's picks already lose
with every advantage (HANDOVER, 2026-09-23), so the prior on a model reading
a numeric sequence as *shape* is modest.

EVERYTHING IS ATR-NORMALISED
----------------------------
Demo's ATR is a measured 0.715-0.751x live's for the SAME symbol (n=80 paired
here, n=13 in the handover — two independent periods). Dividing by ATR IS
that correction, so a shape score means the same thing on both instances and
can be ported without the 0.751 factor.

THE HORIZON CONSTRAINT
----------------------
Median hold is 1.3 minutes; 80% of trades close within 3. A judgement over 30
bars of 3m data describes a window ~20x the life of the position. `lookback`
is therefore deliberately short and the caller chooses the timeframe — this
must NOT silently inherit SCANNER_TIMEFRAME the way CRT did.

NOT GATING. Recorded only, scored against `adverse_pct` and `edge_ratio` at
1-3 minutes, exactly as the other triggers are.
"""
from __future__ import annotations


def _safe(v):
    try:
        f = float(v)
        return f if f == f else None          # NaN check
    except (TypeError, ValueError):
        return None


def _atr_unit(candles: list, atr_pct: float | None, ref: float) -> float | None:
    """
    One ATR in PRICE terms. Prefers the caller's atr_pct (the same figure the
    rest of the bot sizes from); falls back to mean true range over the window
    so the module still works on candles alone.
    """
    a = _safe(atr_pct)
    if a and a > 0 and ref:
        return ref * a / 100.0
    trs = []
    prev_close = None
    for _, h, l, c in candles:
        h, l, c = _safe(h), _safe(l), _safe(c)
        if None in (h, l, c):
            continue
        tr = h - l if prev_close is None else max(
            h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    return (sum(trs) / len(trs)) if trs else None


def describe(candles: list, atr_pct: float | None = None,
             lookback: int = 12) -> dict:
    """
    `candles` are CLOSED bars, oldest first, each (ts, high, low, close).

    Returns a flat dict, always the same keys, so a row can be stamped
    unconditionally. Every value is None when it cannot be computed — never a
    default that would read as a measurement.

    Keys:
      shape_ok             enough data to judge
      compression          recent half's range / earlier half's, 1.0 = flat.
                           BELOW 1 means contracting = coiling.
      overlap              mean bar-to-bar range overlap, 0-1. HIGH means
                           bars sit on top of each other = consolidation.
      extension_atr        move over the window, in ATRs. HIGH means already
                           run.
      accel                last leg's pace / previous leg's. ABOVE 1 =
                           accelerating.
      leg_atr_per_bar      current pace, ATRs per bar — the raw figure `accel`
                           is a ratio of.
      consolidating        bool: contracting AND overlapping AND not extended
      extended             bool: moved far for the noise in the window
      shape_bars           bars actually used
    """
    out = {"shape_ok": False, "compression": None, "overlap": None,
           "extension_atr": None, "accel": None, "leg_atr_per_bar": None,
           "consolidating": None, "extended": None, "shape_bars": 0}
    try:
        lookback = max(4, int(lookback))
        rows = [(t, _safe(h), _safe(l), _safe(c)) for t, h, l, c in
                (candles or [])[-lookback:]]
        rows = [r for r in rows if None not in r[1:]]
        if len(rows) < 4:
            return out
        out["shape_bars"] = len(rows)
        closes = [r[3] for r in rows]
        unit = _atr_unit(rows, atr_pct, closes[-1])
        if not unit or unit <= 0:
            return out

        half = len(rows) // 2
        early, late = rows[:half], rows[half:]

        def span(block):
            return max(r[1] for r in block) - min(r[2] for r in block)

        e_span, l_span = span(early), span(late)
        out["compression"] = round(l_span / e_span, 4) if e_span > 0 else None

        # Bar-to-bar overlap: how much of each bar's range sits inside the
        # previous bar's. Coiling price overlaps; a trend steps away.
        ov = []
        for (_, h1, l1, _), (_, h2, l2, _) in zip(rows, rows[1:]):
            lo, hi = max(l1, l2), min(h1, h2)
            r2 = h2 - l2
            if r2 > 0:
                ov.append(max(0.0, hi - lo) / r2)
        out["overlap"] = round(sum(ov) / len(ov), 4) if ov else None

        # How far it has ALREADY run, in ATRs.
        out["extension_atr"] = round(abs(closes[-1] - closes[0]) / unit, 4)

        # Pace of the last leg against the one before it.
        prev_leg = abs(closes[half] - closes[0]) / unit / max(1, half)
        last_leg = abs(closes[-1] - closes[half]) / unit / max(1, len(rows) - half)
        out["leg_atr_per_bar"] = round(last_leg, 4)
        out["accel"] = (round(last_leg / prev_leg, 4)
                        if prev_leg > 1e-9 else None)

        # The two labels. Thresholds live HERE, in code, so changing them is a
        # coefficient edit and every component stays on the row to be re-fitted.
        c_, o_, x_ = out["compression"], out["overlap"], out["extension_atr"]
        if None not in (c_, o_, x_):
            out["consolidating"] = bool(
                c_ <= COMPRESSION_MAX and o_ >= OVERLAP_MIN
                and x_ <= EXTENDED_ATR)
            out["extended"] = bool(x_ >= EXTENDED_ATR)
        out["shape_ok"] = True
        return out
    except Exception:
        # Shape must never be able to break a scan.
        return out


# Thresholds, code-owned. Starting values are PLAIN, not a fitted result:
# a window whose second half is narrower than its first, whose bars mostly
# overlap, and which has not already travelled 2 ATRs.
COMPRESSION_MAX = 0.90
OVERLAP_MIN = 0.35
EXTENDED_ATR = 2.0


def favours(shape: dict, side: str) -> bool | None:
    """
    Would this shape have ARGUED FOR the trade?

    None when there is no opinion, so a missing judgement is never scored as a
    disagreement — the same contract as `crt.agrees`.

    Deliberately side-agnostic on direction: consolidation before a move is
    the operator's stated setup whichever way it breaks. Direction is the
    scanner's job; shape only says whether the moment is right.
    """
    if not shape or not shape.get("shape_ok"):
        return None
    c = shape.get("consolidating")
    if c is None:
        return None
    return bool(c)
