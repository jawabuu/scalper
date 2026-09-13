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
from datetime import datetime, timezone

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


def _ctx(t: dict, key: str):
    """Read a numeric value from the trade's recorded entry context."""
    v = (t.get("entry_context") or {}).get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


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


def _verified(t: dict) -> bool:
    """
    Did the money figure come from the exchange?

    Where neither the ledger nor the fills were readable, the exit price was
    reconstructed from a guess. One such trade reported -75% ROI on a position
    whose stop was capped at -30%; another reported a flat zero. Those are
    fabrications, not estimates, and averaging them corrupts every total —
    including the wallet reconciliation, which then cannot close.
    """
    v = t.get("pnl_verified")
    if v is not None:
        return bool(v)
    # Older records have no flag; treat a computed source as unverified.
    src = t.get("pnl_source")
    if src in ("ledger", "fills"):
        return True
    if src in ("computed", "none"):
        return False
    # No provenance at all: fall back to the exit-estimate marker.
    return not t.get("exit_is_estimate")


def account_return(trades: list[dict], baseline: float | None,
                   wallet_now: float | None = None,
                   day_baseline: float | None = None,
                   day_start_ts: float | None = None) -> dict:
    """
    Account growth: net P&L divided by the balance it started from.

    Distinct from return on capital, which divides by the SUM of margins across
    sequential trades — the same money recycled, so that figure is return per
    dollar of turnover rather than growth of the account. This one compounds;
    that one measures edge per trade.

    Computed from the TRADE RECORD rather than the wallet, so it matches the
    tables below it and does not swing with unrealised P&L on open positions.
    The cost is that it excludes unverified trades and can therefore disagree
    with the actual balance — so the disagreement is reported rather than left
    to be discovered.
    """
    scored = [t for t in trades if _verified(t)
              and t.get("realised_pnl_usdt") is not None]
    net = sum((_realised(t) or 0.0) for t in scored)

    out = {"trades": len(scored), "net_pnl": round(net, 4),
           "baseline": baseline, "pct": None,
           "day_pct": None, "day_net_pnl": None, "day_trades": 0,
           "wallet_pct": None, "disagrees_with_wallet": False}

    if baseline and baseline > 0:
        out["pct"] = round(net / baseline * 100, 3)

    # Today's slice, on its own baseline.
    if day_start_ts is not None:
        today = [t for t in scored
                 if (t.get("closed_at") or 0) >= day_start_ts]
        day_net = sum((_realised(t) or 0.0) for t in today)
        out["day_trades"] = len(today)
        out["day_net_pnl"] = round(day_net, 4)
        base = day_baseline or baseline
        if base and base > 0:
            out["day_pct"] = round(day_net / base * 100, 3)

    # Does the trade record agree with the money? A gap means excluded or
    # mis-recorded trades, or unrealised P&L on something still open.
    if wallet_now is not None and baseline and baseline > 0:
        moved = wallet_now - baseline
        out["wallet_pct"] = round(moved / baseline * 100, 3)
        out["wallet_gap"] = round(moved - net, 4)
        out["disagrees_with_wallet"] = abs(moved - net) > max(1.0, abs(net) * 0.05)

    return out


# ── Execution diagnostics ──────────────────────────────────────────────────
# The aggregate tables answer "does this factor pay?". They cannot answer
# "what happened to THAT trade?", which is what a post-mortem needs. This
# section flags individual trades whose EXECUTION — not whose thesis — went
# wrong, and renders them as a block that can be copied out verbatim.
#
# The four flags come from the 2026-09-12 post-mortems:
#   stop_overshoot  the realised loss went well past the stop that was sized
#                   (D8: two trades sized at -30% both realised ~-43%)
#   dead_on_arrival never traded above entry and still lost heavily
#   breakout_entry  entered against a still-expanding move
#   thin_callback   the entry trigger was inside noise width

# How many trades the card shows and the report covers. A flagged list grows
# with every run — 76 rows is a wall, not a diagnosis. Rank by money moved and
# show the worst; the count of everything else is still reported.
DIAG_TOP_N = 10

STOP_OVERSHOOT_ROI = 5.0     # realised worse than the sized stop by this much
DEAD_LOSS_ROI = 20.0         # never green, and lost at least this much
# A callback threshold of 1.0 fired on 76 of 78 trades, including the biggest
# WINNER of the sample (STORJ +218.08 on a 0.38% callback — the same 0.38% as
# a -103.90 loser). A flag that describes the normal trade carries no
# information. 0 disables it; raise it only if the split table shows the flag
# actually separating winners from losers.
THIN_CALLBACK_PCT = 0.0      # 0 = off


def _diag_flags(t: dict, thin_callback_pct: float = THIN_CALLBACK_PCT
                ) -> list[str]:
    """Which execution problems, if any, this trade shows."""
    flags: list[str] = []
    ctx = t.get("entry_context") or {}
    roi = _roi(t)
    sized = ctx.get("sized_stop_roi")

    if roi is not None and sized:
        try:
            if roi < 0 and abs(roi) > float(sized) + STOP_OVERSHOOT_ROI:
                flags.append("stop_overshoot")
        except (TypeError, ValueError):
            pass

    peak = t.get("peak_roi")
    if roi is not None and peak is not None:
        try:
            if float(peak) <= 0.0 and roi <= -DEAD_LOSS_ROI:
                flags.append("dead_on_arrival")
        except (TypeError, ValueError):
            pass

    if ctx.get("breakout"):
        flags.append("breakout_entry")

    cb = ctx.get("callback_pct")
    if thin_callback_pct > 0:
        try:
            if cb is not None and float(cb) < thin_callback_pct:
                flags.append("thin_callback")
        except (TypeError, ValueError):
            pass

    return flags


def _flag_verdict(hits: int, total: int, net_pnl: float) -> str:
    """
    Whether a flag is telling you anything. Two ways to be useless: firing on
    nearly everything, or catching trades that made money.
    """
    if not hits:
        return "never fires"
    if total and hits / total > 0.6:
        return "fires on most trades — describes the strategy, not a fault"
    if net_pnl > 0:
        return "its catches are net POSITIVE — not a fault"
    return "specific and net negative"


def _diag_severity(row: dict) -> float:
    """
    How much this trade actually moved the account. Money first — a -60% ROI
    on a tiny position matters less than -20% on a large one — falling back to
    ROI when the money could not be read.
    """
    v = row.get("realised_pnl_usdt")
    if v is not None:
        try:
            return abs(float(v))
        except (TypeError, ValueError):
            pass
    v = row.get("final_roi")
    try:
        return abs(float(v)) / 100.0 if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _diag_row(t: dict, thin_callback_pct: float = THIN_CALLBACK_PCT) -> dict:
    """Every field a post-mortem of this trade needs, in one flat row."""
    ctx = t.get("entry_context") or {}
    roi = _roi(t)
    sized = ctx.get("sized_stop_roi")
    overshoot = None
    if roi is not None and sized:
        try:
            overshoot = round(abs(roi) - float(sized), 2) if roi < 0 else None
        except (TypeError, ValueError):
            overshoot = None
    return {
        "symbol": t.get("symbol"),
        "side": t.get("side"),
        "opened_at": t.get("opened_at"),
        "flags": _diag_flags(t, thin_callback_pct),
        "final_roi": roi,
        "peak_roi": t.get("peak_roi"),
        "trough_roi": t.get("trough_roi"),
        "roi_at_0s": t.get("roi_at_0s"),
        "signal_age_s": t.get("signal_age_s"),
        "observation_lag_s": t.get("observation_lag_s"),
        "drift_since_sizing_pct": t.get("drift_since_sizing_pct"),
        "change_24h_pct": ctx.get("change_24h_pct"),
        "body_pct": ctx.get("body_pct"),
        "taper_ratio": ctx.get("taper_ratio"),
        "tapering": ctx.get("tapering"),
        "taper_vol_ratio": ctx.get("taper_vol_ratio"),
        "taper_close_pos": ctx.get("taper_close_pos"),
        "upper_wick_pct": ctx.get("upper_wick_pct"),
        "lower_wick_pct": ctx.get("lower_wick_pct"),
        "sized_stop_roi": sized,
        "stop_overshoot_roi": overshoot,
        "realised_pnl_usdt": _realised(t),
        "fees_usdt": _fees(t),
        "rsi": ctx.get("rsi"),
        "atr_pct": ctx.get("atr_pct"),
        "recent_tr_pct": ctx.get("recent_tr_pct"),
        "dist_to_extreme_pct": ctx.get("dist_to_extreme_pct"),
        "callback_pct": ctx.get("callback_pct"),
        "callback_source": ctx.get("callback_source"),
        "ema_gap_pct": ctx.get("ema_gap_pct"),
        "htf_trend_pct": ctx.get("htf_trend_pct"),
        "breakout": ctx.get("breakout"),
        "brk_at_extreme": ctx.get("brk_at_extreme"),
        "brk_consecutive": ctx.get("brk_consecutive"),
        "brk_gap_wide": ctx.get("brk_gap_wide"),
        "brk_gap_widening": ctx.get("brk_gap_widening"),
        "exit_reason": t.get("exit_reason"),
    }


def _fmt(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def diagnostics_report(rows: list[dict], omitted: int = 0) -> str:
    """
    A plain-text block, one paragraph per flagged trade, safe to paste
    somewhere else without carrying any account identifiers.
    """
    if not rows:
        return "No flagged trades."
    head = f"EXECUTION DIAGNOSTICS — {len(rows)} shown"
    if omitted:
        head += f", {omitted} further flagged trade(s) not shown"
    out = [head + " (worst first, by money moved)", ""]
    for r in rows:
        when = ""
        if r.get("opened_at"):
            try:
                when = datetime.fromtimestamp(
                    float(r["opened_at"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            except (TypeError, ValueError, OSError):
                when = ""
        out.append(f"{r.get('symbol')} {str(r.get('side') or '').upper()} {when}")
        out.append(f"  flags        : {', '.join(r['flags']) or 'none'}")
        out.append(f"  entry        : rsi {_fmt(r['rsi'],1)}  "
                   f"atr {_fmt(r['atr_pct'],3)}%  "
                   f"recent_tr {_fmt(r['recent_tr_pct'],3)}%  "
                   f"dist {_fmt(r['dist_to_extreme_pct'])}%")
        out.append(f"  trigger      : callback {_fmt(r['callback_pct'])}% "
                   f"({r.get('callback_source') or 'n/a'})  "
                   f"signal_age {_fmt(r['signal_age_s'],1)}s  "
                   f"drift {_fmt(r['drift_since_sizing_pct'],3)}% (+ve = good fill)  "
                   f"obs_lag {_fmt(r['observation_lag_s'],2)}s")
        out.append(f"  taper        : ratio {_fmt(r['taper_ratio'],3)} "
                   f"(under 1 = pushes shrinking)  tapering {_fmt(r['tapering'])}  "
                   f"vol_ratio {_fmt(r['taper_vol_ratio'],3)}  "
                   f"close_pos {_fmt(r['taper_close_pos'],3)}")
        out.append(f"  character    : 24h change {_fmt(r['change_24h_pct'])}%  "
                   f"body {_fmt(r['body_pct'],1)}%  "
                   f"upper_wick {_fmt(r['upper_wick_pct'],1)}%  "
                   f"lower_wick {_fmt(r['lower_wick_pct'],1)}%")
        out.append(f"  structure    : breakout {_fmt(r['breakout'])} "
                   f"[extreme {_fmt(r['brk_at_extreme'])}, "
                   f"consec {_fmt(r['brk_consecutive'],0)}, "
                   f"wide {_fmt(r['brk_gap_wide'])}, "
                   f"widening {_fmt(r['brk_gap_widening'])}]  "
                   f"ema_gap {_fmt(r['ema_gap_pct'],3)}%  "
                   f"htf {_fmt(r['htf_trend_pct'],3)}%")
        out.append(f"  outcome      : roi@first-sight {_fmt(r['roi_at_0s'])}% "
                   f"(lag artefact, not entry quality)  "
                   f"peak {_fmt(r['peak_roi'])}%  "
                   f"trough {_fmt(r['trough_roi'])}%  "
                   f"final {_fmt(r['final_roi'])}%")
        out.append(f"  stop         : sized {_fmt(r['sized_stop_roi'],1)}%  "
                   f"overshoot {_fmt(r['stop_overshoot_roi'])}%  "
                   f"exit_reason {r.get('exit_reason') or 'n/a'}")
        out.append(f"  money        : realised {_fmt(r['realised_pnl_usdt'],4)}  "
                   f"fees {_fmt(r['fees_usdt'],4)}")
        out.append("")
    if omitted:
        out.append(f"({omitted} further flagged trade(s) omitted — ranked by "
                   f"money moved, so every one of them moved less than these)")
    return "\n".join(out)


def analyse(trades: list[dict]) -> dict:
    """
    Full report. Every section carries its own sample size so a striking
    difference across three trades is not mistaken for a finding.

    Trades whose P&L could not be read from the exchange are EXCLUDED from
    every statistic and reported separately, because an invented number is
    worse than a missing one.
    """
    all_trades = list(trades or [])
    unverified = [t for t in all_trades if not _verified(t)]
    trades = [t for t in all_trades if _verified(t)]
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

    # ── Which price triggered the stops ──
    # Binance's default, CONTRACT_PRICE, is the last trade on THIS book, so a
    # single wick closes the position. MARK_PRICE is a smoothed cross-venue
    # index. The change was made on evidence from one trade (BR gave back 19.7
    # ROI points to a candle whose range was exactly the give-back), so it has
    # to be checkable rather than assumed.
    #
    # Losses and gains are reported SEPARATELY: the claim is specifically that
    # mark price avoids exits on fake moves, which should show as a smaller
    # average LOSS. If it also shrinks average gains, it is triggering late on
    # real moves too, and that is the cost side of the trade-off.
    by_wt: dict[str, list[dict]] = {}
    for t in trades:
        wt = t.get("stop_working_type")
        if wt:
            by_wt.setdefault(str(wt), []).append(t)
    working_type = []
    for wt, group in sorted(by_wt.items()):
        st = group_stats(group)
        st["label"] = wt
        wins = [t for t in group if (_realised(t) or 0) > 0]
        losses = [t for t in group if (_realised(t) or 0) < 0]
        st["n_wins"] = len(wins)
        st["n_losses"] = len(losses)
        st["avg_win_usdt"] = (round(sum(_realised(t) for t in wins)/len(wins), 2)
                              if wins else None)
        st["avg_loss_usdt"] = (round(sum(_realised(t) for t in losses)/len(losses), 2)
                               if losses else None)
        st["avg_win_roi"] = (round(sum((_roi(t) or 0) for t in wins)/len(wins), 2)
                             if wins else None)
        st["avg_loss_roi"] = (round(sum((_roi(t) or 0) for t in losses)/len(losses), 2)
                              if losses else None)
        # The direct measure: how far past its sized stop a loser ran.
        ov = []
        for t in losses:
            sized = (t.get("entry_context") or {}).get("sized_stop_roi")
            r = _roi(t)
            if sized and r is not None and r < 0:
                ov.append(abs(r) - float(sized))
        st["avg_overshoot_roi"] = round(sum(ov)/len(ov), 2) if ov else None
        st["n_overshoot_measured"] = len(ov)
        working_type.append(st)

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

    # ── Regime splits: measurement only ──────────────────────────────────
    # Does a direction work better in a particular market regime? Breadth is a
    # DIRECT measure of it (how much of the market is rising), where session is
    # a proxy. Both are reported so they can be compared — they will correlate,
    # since trading hours have characteristic breadth, and if both light up
    # they may be measuring the same thing twice.
    BREADTH_BUCKETS = [("<30% up", -1, 30), ("30-50% up", 30, 50),
                       ("50-70% up", 50, 70), (">70% up", 70, 1000)]
    by_breadth = [
        _time_split(trades, name,
                    lambda t, lo=lo, hi=hi: (
                        (_ctx(t, "breadth_pct") is not None)
                        and lo <= _ctx(t, "breadth_pct") < hi))
        for name, lo, hi in BREADTH_BUCKETS
    ]

    # Is the 3-minute signal running with the coin's hourly trend or against
    # it? A short taken while the hour is still rising is a different trade
    # from one taken while the hour is falling.
    def _aligned(t, want):
        htf = _ctx(t, "htf_trend_pct")
        side = t.get("side")
        if htf is None or side not in ("long", "short"):
            return False
        with_trend = (htf > 0) if side == "long" else (htf < 0)
        return with_trend is want

    by_htf = [
        _time_split(trades, "with the hourly trend", lambda t: _aligned(t, True)),
        _time_split(trades, "against the hourly trend", lambda t: _aligned(t, False)),
    ]

    # ── What a fail-fast cutoff would have cost, and saved ───────────────
    # Two questions a cutoff needs answered before it can be chosen:
    #   1. how many WINNERS took longer than the cutoff to go green — the cost
    #   2. where the LOSERS stood at that moment — the saving
    # Neither is answerable from peak and final alone, which is why the
    # underlying fields are recorded per trade.
    winners = [t for t in trades if (_roi(t) or 0) > 0]
    losers = [t for t in trades if (_roi(t) or 0) < 0]
    fail_fast = []
    for secs, key in ((60, "roi_at_60s"), (180, "roi_at_180s"),
                      (300, "roi_at_300s")):
        # A winner is cut if it had not yet gone positive at the cutoff.
        timed = [t for t in winners if t.get("secs_to_first_positive") is not None]
        cut = [t for t in timed if float(t["secs_to_first_positive"]) > secs]
        cut_value = sum((_realised(t) or 0.0) for t in cut)

        # Losers that never went green: where were they at the cutoff?
        # peak_roi is no longer clamped at 0, so a never-green trade reports a
        # NEGATIVE peak. A truthiness test read that as "went green".
        never = [t for t in losers
                 if t.get("peak_roi") is None or float(t["peak_roi"]) <= 0.0]
        at_mark = [float(t[key]) for t in never if t.get(key) is not None]
        avg_at = round(sum(at_mark) / len(at_mark), 2) if at_mark else None
        avg_final = (round(sum((_roi(t) or 0) for t in never) / len(never), 2)
                     if never else None)
        fail_fast.append({
            "cutoff_s": secs,
            "winners_cut": len(cut),
            "of_winners": len(timed),
            "winner_value_lost": round(cut_value, 2),
            "losers_never_green": len(never),
            "avg_roi_at_cutoff": avg_at,
            "avg_final_roi": avg_final,
            # The saving per trade is the gap between where it stood at the
            # cutoff and where it ended.
            "avg_roi_saved": (round(avg_at - avg_final, 2)
                              if avg_at is not None and avg_final is not None
                              else None),
            "confidence": confidence_for(len(timed)),
        })

    # ── Breakout structure: is fading an expanding move the wrong side? ──
    # A wide EMA gap alone fits a blow-off top as well as a breakout. The
    # question is whether a WIDENING gap at a new extreme, with consecutive
    # candles carrying it, means the move is still expanding — in which case
    # the fade is on the wrong side of it.
    def _flag(t, key):
        return bool((t.get("entry_context") or {}).get(key))

    by_breakout = [
        _time_split(trades, "breakout structure", lambda t: _flag(t, "breakout")),
        _time_split(trades, "no breakout structure",
                    lambda t: (t.get("entry_context") or {}).get("breakout") is False),
    ]
    # Components separately, so it is visible WHICH part carries the signal
    # rather than only the combination.
    by_breakout_parts = [
        _time_split(trades, "at a new extreme", lambda t: _flag(t, "brk_at_extreme")),
        _time_split(trades, "gap wide", lambda t: _flag(t, "brk_gap_wide")),
        _time_split(trades, "gap widening", lambda t: _flag(t, "brk_gap_widening")),
        _time_split(trades, "3+ candles in a row",
                    lambda t: ((t.get("entry_context") or {}).get("brk_consecutive") or 0) >= 3),
    ]

    reentries = [t for t in trades if (t.get("entry_context") or {}).get("was_reentry")]
    fresh = [t for t in trades if not (t.get("entry_context") or {}).get("was_reentry")]

    notes: list[str] = []
    if unverified:
        notes.append(
            f"{len(unverified)} trade(s) EXCLUDED: their P&L could not be read "
            f"from the exchange, so the exit price was reconstructed from a "
            f"guess. Those figures are invented, not approximate — including "
            f"them would corrupt every total below.")
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

    # ── Execution diagnostics ──
    # Built from ALL trades, including unverified ones: a trade whose P&L
    # could not be read is itself worth looking at, and this section makes no
    # statistical claim that a bad number could corrupt.
    diag_rows = []
    for t in all_trades:
        row = _diag_row(t)
        if row["flags"]:
            diag_rows.append(row)
    diag_rows.sort(key=_diag_severity, reverse=True)
    diag_top = diag_rows[:DIAG_TOP_N]
    diag_omitted = max(0, len(diag_rows) - len(diag_top))
    # Counts cover EVERY flagged trade, not just the shown ones — otherwise
    # capping the list would silently shrink the tallies.
    diag_counts: dict[str, int] = {}
    for r in diag_rows:
        for f in r["flags"]:
            diag_counts[f] = diag_counts.get(f, 0) + 1

    # Per-flag winners/losers split. A flag only earns its place if what it
    # catches loses money: thin_callback fired on 76 of 78 trades and its
    # catches were net POSITIVE, which makes it a description of the strategy
    # rather than a diagnosis. This table makes that visible without having to
    # read the rows.
    diag_split: list[dict] = []
    for flag in sorted(diag_counts):
        hits = [r for r in diag_rows if flag in r["flags"]]
        pnls = [r.get("realised_pnl_usdt") for r in hits]
        pnls = [float(v) for v in pnls if v is not None]
        wins = [v for v in pnls if v > 0]
        losses = [v for v in pnls if v < 0]
        diag_split.append({
            "flag": flag,
            "hits": len(hits),
            "winners": len(wins),
            "losers": len(losses),
            "net_pnl": round(sum(pnls), 2) if pnls else None,
            "share_of_trades": (round(len(hits) / len(all_trades) * 100, 1)
                                if all_trades else None),
            "verdict": _flag_verdict(len(hits), len(all_trades),
                                     sum(pnls) if pnls else 0.0),
        })

    return {
        "unverified_trades": len(unverified),
        "execution_diagnostics": {
            "rows": diag_top,
            "total_flagged": len(diag_rows),
            "omitted": diag_omitted,
            "top_n": DIAG_TOP_N,
            "counts": diag_counts,
            "by_flag": diag_split,
            "report": diagnostics_report(diag_top, diag_omitted),
            "thresholds": {
                "stop_overshoot_roi": STOP_OVERSHOOT_ROI,
                "dead_loss_roi": DEAD_LOSS_ROI,
                "thin_callback_pct": THIN_CALLBACK_PCT,
            },
        },
        "unverified_symbols": sorted({t.get("symbol", "?") for t in unverified}),
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
        "by_breakout": by_breakout,
        "by_breakout_parts": by_breakout_parts,
        "fail_fast_impact": fail_fast,
        "by_breadth": by_breadth,
        "by_htf_alignment": by_htf,
        "by_session": by_session,
        "by_hour": [b for b in by_hour if b.get("n")],
        "by_callback_rule": callback_rule,
        "by_working_type": working_type,
        "reentries": {"override_reentries": group_stats(reentries),
                      "fresh_entries": group_stats(fresh)},
        "notes": notes,
        "thresholds": {"min_indicative": MIN_INDICATIVE, "min_usable": MIN_USABLE},
    }
