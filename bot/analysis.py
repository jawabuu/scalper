"""
Trade analysis.

Turns recorded futures trades into answers about which entry conditions
actually paid, rather than which ones felt right.

A deliberate bias runs through this module: it reports sample size as
prominently as the result, and refuses to characterise a split it considers too
thin. Slicing a small set of trades by several conditions produces impressive-
looking differences that are pure noise — which is exactly how a strategy gets
overfitted to a fortnight of data. Every group therefore carries a `confidence`
label, and the summary states plainly when nothing can yet be concluded.

Nothing here fetches or trades. It reads closed-trade records and computes.
"""
from __future__ import annotations

from dataclasses import dataclass

# Below this, a group's numbers are noise dressed as signal.
MIN_USABLE = 30
MIN_INDICATIVE = 10


def confidence_for(n: int) -> str:
    if n >= MIN_USABLE:
        return "usable"
    if n >= MIN_INDICATIVE:
        return "thin"
    return "insufficient"


def _roi(t: dict) -> float | None:
    v = t.get("final_roi")
    return None if v is None else float(v)


def _realised(t: dict) -> float | None:
    """
    Prefer the NET figure. Binance's realizedPnl excludes commission, so a
    gross-only expectancy can read positive while the wallet falls — which is
    exactly what happened: +60 USDT of P&L against a 20 USDT wallet loss, the
    difference being 80 USDT of fees.
    """
    v = t.get("net_pnl_usdt")
    if v is None:
        v = t.get("realised_pnl_usdt")
    return None if v is None else float(v)


def _fees(t: dict) -> float | None:
    v = t.get("fees_usdt")
    return None if v is None else float(v)


def group_stats(trades: list[dict]) -> dict:
    """
    Performance of one group of trades.

    Expectancy (average realised P&L per trade) is the number that matters:
    a high win rate with a poor expectancy means the losers are bigger than the
    winners, which is the failure mode a win-rate-only view hides.
    """
    scored = [t for t in trades if _roi(t) is not None]
    n = len(scored)
    if n == 0:
        return {"n": 0, "confidence": "insufficient"}

    rois = [_roi(t) for t in scored]
    wins = [r for r in rois if r > 0]
    realised = [_realised(t) for t in scored if _realised(t) is not None]
    peaks = [float(t.get("peak_roi") or 0) for t in scored]
    givebacks = [float(t.get("peak_roi") or 0) - r
                 for t, r in zip(scored, rois)]

    return {
        "n": n,
        "confidence": confidence_for(n),
        "win_rate": round(len(wins) / n * 100, 1),
        "avg_roi": round(sum(rois) / n, 2),
        "total_roi": round(sum(rois), 1),
        "avg_peak_roi": round(sum(peaks) / n, 2),
        # How much of the best moment was handed back. High give-back with a
        # decent peak means the exit is the problem, not the entry.
        "avg_giveback": round(sum(givebacks) / n, 2),
        "expectancy_usdt": (round(sum(realised) / len(realised), 4)
                            if realised else None),
        "total_realised": round(sum(realised), 2) if realised else None,
        "total_fees": (round(sum(f for f in (_fees(t) for t in scored)
                                 if f is not None), 2)
                       if any(_fees(t) is not None for t in scored) else None),
        "best_roi": round(max(rois), 2),
        "worst_roi": round(min(rois), 2),
    }


@dataclass
class Bucket:
    label: str
    lo: float
    hi: float

    def holds(self, v: float | None) -> bool:
        return v is not None and self.lo <= v < self.hi


def bucket_by(trades: list[dict], value_of, buckets: list[Bucket]) -> list[dict]:
    """Split trades into buckets by some entry value and stat each."""
    out = []
    for b in buckets:
        members = [t for t in trades if b.holds(value_of(t))]
        stats = group_stats(members)
        stats["label"] = b.label
        out.append(stats)
    return out


def _stamp(t: dict, key: str):
    """Entry-condition stamps live under entry_context on the trade record."""
    ctx = t.get("entry_context") or {}
    v = ctx.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


RSI_BUCKETS = [Bucket("<50", -1, 50), Bucket("50-60", 50, 60),
               Bucket("60-70", 60, 70), Bucket("70-78", 70, 78),
               Bucket("78-85", 78, 85), Bucket("85+", 85, 999)]

DIST_BUCKETS = [Bucket("<1%", 0, 1), Bucket("1-2%", 1, 2),
                Bucket("2-3%", 2, 3), Bucket("3-5%", 3, 5),
                Bucket("5%+", 5, 999)]

ATR_BUCKETS = [Bucket("<0.3%", 0, 0.3), Bucket("0.3-0.7%", 0.3, 0.7),
               Bucket("0.7-1.5%", 0.7, 1.5), Bucket("1.5%+", 1.5, 999)]


def analyse(trades: list[dict]) -> dict:
    """
    Full report. Every section carries its own sample size so a striking
    difference across three trades is not mistaken for a finding.
    """
    trades = list(trades or [])
    overall = group_stats(trades)

    longs = [t for t in trades if t.get("side") == "long"]
    shorts = [t for t in trades if t.get("side") == "short"]

    by_reason: dict[str, dict] = {}
    for t in trades:
        by_reason.setdefault(t.get("exit_reason") or "unknown", []).append(t)
    exits = []
    for reason, group in sorted(by_reason.items()):
        s = group_stats(group)
        s["label"] = reason
        exits.append(s)

    # Was the callback set by the distance ratio or overridden by the ATR
    # floor? They measure different things — opportunity vs noise — so their
    # outcomes are worth comparing directly rather than by impression.
    by_cb: dict[str, list] = {}
    for t in trades:
        src = (t.get("entry_context") or {}).get("callback_source")
        if src:
            by_cb.setdefault(src, []).append(t)
    callback_rule = []
    for src, group in sorted(by_cb.items()):
        st = group_stats(group)
        st["label"] = src
        callback_rule.append(st)

    reentries = [t for t in trades if (t.get("entry_context") or {}).get("was_reentry")]
    fresh = [t for t in trades if not (t.get("entry_context") or {}).get("was_reentry")]

    notes: list[str] = []
    if overall.get("n", 0) < MIN_INDICATIVE:
        notes.append(
            f"Only {overall.get('n', 0)} closed trades. Nothing here supports a "
            f"conclusion yet — treat every number as provisional.")
    elif overall.get("n", 0) < MIN_USABLE:
        notes.append(
            f"{overall['n']} closed trades. Enough to spot something worth "
            f"watching, not enough to act on. Splits below will be thinner still.")
    if overall.get("expectancy_usdt") is not None and overall["expectancy_usdt"] < 0 \
            and overall.get("win_rate", 0) >= 50:
        notes.append(
            "Win rate is at or above 50% but expectancy is negative — the losers "
            "are bigger than the winners. That is an exit problem, not an entry one.")
    if overall.get("total_fees"):
        notes.append(
            f"Fees so far: {overall['total_fees']:.2f} USDT across "
            f"{overall['n']} trades. Expectancy below is NET of fees; the "
            f"exchange's own P&L figure is gross, which is why a positive "
            f"P&L can sit alongside a falling wallet balance.")
    if overall.get("avg_giveback") and overall["avg_giveback"] > 10:
        notes.append(
            f"Average give-back is {overall['avg_giveback']:.1f}% ROI from peak. "
            f"Arming the trail earlier would convert more of that into realised P&L.")

    # Would a tighter stop have cut winners short? Only the trough can say.
    winners = [t for t in trades if (_roi(t) or 0) > 0]
    troughs = [float(t.get("trough_roi")) for t in winners
               if t.get("trough_roi") is not None]
    stop_impact = []
    for level in (5.0, 10.0, 15.0, 20.0):
        hit = [v for v in troughs if v <= -level]
        stop_impact.append({
            "stop_roi": level,
            "winners_cut": len(hit),
            "of_winners": len(troughs),
            "confidence": confidence_for(len(troughs)),
        })
    if troughs:
        notes_extra = (f"{len(troughs)} winners have trough data; "
                       f"deepest dip {min(troughs):.1f}% ROI.")
    else:
        notes_extra = ("No trough data yet — it is recorded from this version "
                       "on, and is what shows whether a tighter stop would "
                       "have cut winners short.")

    return {
        "stop_impact": stop_impact,
        "trough_note": notes_extra,
        "overall": overall,
        "by_side": {"long": group_stats(longs), "short": group_stats(shorts)},
        "by_rsi": bucket_by(trades, lambda t: _stamp(t, "rsi"), RSI_BUCKETS),
        "by_distance_to_extreme": bucket_by(
            trades, lambda t: _stamp(t, "dist_to_extreme_pct"), DIST_BUCKETS),
        "by_atr": bucket_by(trades, lambda t: _stamp(t, "atr_pct"), ATR_BUCKETS),
        "by_exit_reason": exits,
        "by_callback_rule": callback_rule,
        "reentries": {"override_reentries": group_stats(reentries),
                      "fresh_entries": group_stats(fresh)},
        "notes": notes,
        "thresholds": {"min_indicative": MIN_INDICATIVE, "min_usable": MIN_USABLE},
    }
