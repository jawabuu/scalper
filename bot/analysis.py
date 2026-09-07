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


def _margin(t: dict) -> float | None:
    """
    Capital committed to the trade.

    Recorded directly from this version on. For trades closed earlier the
    field is absent, so it is derived from the two numbers that ARE present:
    margin = realised / (ROI / 100). That keeps the return-on-capital figure
    working across a history that spans the change, instead of showing a dash
    until every old trade has aged out.
    """
    v = t.get("margin")
    try:
        if v:
            return float(v)
    except (TypeError, ValueError):
        pass

    pnl = t.get("realised_pnl_usdt")
    roi = t.get("roi_from_realised")
    if roi is None:
        roi = t.get("final_roi")
    try:
        if pnl is not None and roi:
            derived = abs(float(pnl)) / (abs(float(roi)) / 100.0)
            return derived if derived > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


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
    margins = [m for m in (_margin(t) for t in scored) if m]
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
        # Return on the capital actually committed. Averaging ROI percentages
        # weights every trade equally regardless of size, so one large loss and
        # one small win can average to a healthy-looking number while the
        # account is down. This divides the money earned by the money at work.
        "capital_deployed": (round(sum(m for m in margins if m), 2)
                             if margins else None),
        "return_on_capital": (
            round(sum(realised) / sum(m for m in margins if m) * 100, 2)
            if realised and margins and sum(m for m in margins if m) else None),
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


def reconcile(trades: list[dict], wallet_now: float | None,
              wallet_start: float | None) -> dict:
    """
    Does the arithmetic close?

        sum(realised) - sum(fees)  ==  wallet change

    Three numbers previously came from three sources — ROI from prices,
    realised from the ledger or fills, fees from the ledger — so nothing
    guaranteed they agreed. This states plainly whether they do, and by how
    much they do not.
    """
    scored = [t for t in trades if t.get("realised_pnl_usdt") is not None]
    gross = sum(float(t["realised_pnl_usdt"]) for t in scored)
    fees = sum(float(t.get("fees_usdt") or 0) for t in scored)
    net = gross - fees
    out = {
        "trades": len(scored),
        "gross_pnl": round(gross, 4),
        "fees": round(fees, 4),
        "net_pnl": round(net, 4),
        "wallet_start": wallet_start,
        "wallet_now": wallet_now,
        "missing_fee_data": sum(1 for t in scored if t.get("fees_usdt") is None),
        "estimated_exits": sum(1 for t in scored
                               if t.get("pnl_source") in (None, "computed")),
    }
    if wallet_now is not None and wallet_start is not None:
        moved = wallet_now - wallet_start
        out["wallet_change"] = round(moved, 4)
        out["discrepancy"] = round(moved - net, 4)
        out["reconciles"] = abs(moved - net) <= max(1.0, abs(net) * 0.02)
    return out


# Trading sessions in UTC. Deliberately coarse: a 24-way split of a few dozen
# trades is noise, whereas four buckets can reach a usable sample. Boundaries
# follow the main centres rather than exchange hours, since crypto never closes.
SESSIONS = [
    ("Asia", 0, 8),
    ("Europe", 8, 13),
    ("EU/US overlap", 13, 17),
    ("US", 17, 24),
]


def _entry_hour(t: dict) -> int | None:
    """
    Hour of day (UTC) the position was OPENED.

    Entry time is what the hypothesis is about — whether a direction works
    better at certain times — so it beats exit time, which drifts by however
    long the trade ran.
    """
    ts = t.get("opened_at") or t.get("closed_at")
    if not ts:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).hour
    except (TypeError, ValueError, OSError):
        return None


def _time_split(trades: list[dict], label: str, members) -> dict:
    """One time bucket, split by direction so the two can be compared."""
    group = [t for t in trades if members(t)]
    out = group_stats(group)
    out["label"] = label
    for side in ("long", "short"):
        sub = [t for t in group if t.get("side") == side]
        st = group_stats(sub)
        out[side] = {"n": st.get("n", 0),
                     "win_rate": st.get("win_rate"),
                     "avg_roi": st.get("avg_roi"),
                     "expectancy_usdt": st.get("expectancy_usdt"),
                     "confidence": st.get("confidence")}
    return out


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

    # Does a direction work better at certain times? Sessions first, because
    # four buckets can reach a usable sample where twenty-four cannot.
    by_session = [
        _time_split(trades, name,
                    lambda t, lo=lo, hi=hi: (
                        (_entry_hour(t) is not None) and lo <= _entry_hour(t) < hi))
        for name, lo, hi in SESSIONS
    ]
    by_hour = [
        _time_split(trades, f"{h:02d}:00 UTC",
                    lambda t, h=h: _entry_hour(t) == h)
        for h in range(24)
    ]

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

    # ROI derived from money vs ROI derived from prices: if these disagree the
    # trade's numbers are internally inconsistent and should not be trusted.
    mismatched = 0
    for t in trades:
        a, b = t.get("final_roi"), t.get("roi_from_realised")
        if a is not None and b is not None and abs(float(a) - float(b)) > 1.0:
            mismatched += 1
    if mismatched:
        notes.append(
            f"{mismatched} trade(s) have a price-derived ROI that disagrees "
            f"with the ROI implied by the money actually received. Trust the "
            f"money: a price-based exit estimate can invent a profit.")

    return {
        "roi_mismatches": mismatched,
        "stop_impact": stop_impact,
        "trough_note": notes_extra,
        "overall": overall,
        "by_side": {"long": group_stats(longs), "short": group_stats(shorts)},
        "by_rsi": bucket_by(trades, lambda t: _stamp(t, "rsi"), RSI_BUCKETS),
        "by_distance_to_extreme": bucket_by(
            trades, lambda t: _stamp(t, "dist_to_extreme_pct"), DIST_BUCKETS),
        "by_atr": bucket_by(trades, lambda t: _stamp(t, "atr_pct"), ATR_BUCKETS),
        "by_exit_reason": exits,
        "by_session": by_session,
        "by_hour": [b for b in by_hour if b.get("n")],
        "by_callback_rule": callback_rule,
        "reentries": {"override_reentries": group_stats(reentries),
                      "fresh_entries": group_stats(fresh)},
        "notes": notes,
        "thresholds": {"min_indicative": MIN_INDICATIVE, "min_usable": MIN_USABLE},
    }
