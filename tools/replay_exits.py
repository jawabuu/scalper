#!/usr/bin/env python3
"""
What would these trades have done under DIFFERENT EXIT RULES?

    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/replay_exits.py --minutes 60

The journal cannot answer this on its own. Once fail-fast cut a position at
-5% ROI, the path after that moment was never recorded — the position was
closed. So this fetches 1-minute candles forward from each entry and replays
the trade under rules that were not the ones that fired.

WHAT IT SIMULATES
  actual          what really happened, from the journal. The control.
  trail only      the adaptive trail alone. No fail-fast, no profit floor,
                  no armed trail. The callback that was actually sized for
                  that trade, taken from its entry context.
  trail + TP      the same, plus a take-profit at --tp-roi.
  TP only         take-profit and the sized ATR stop, no trailing at all.

THREE REASONS TO DISTRUST THE OUTPUT, ALL STRUCTURAL
  1. INTRA-CANDLE ORDER IS UNKNOWN. A 1m candle gives open/high/low/close
     with no sequence. When a candle would have hit BOTH a stop and a target,
     this books the STOP — always. That is pessimistic by construction, and
     it is the single largest source of backtest self-deception in this kind
     of replay. A result that survives it is worth something; one that needs
     the optimistic reading is worth nothing.
  2. A TRAILING STOP IS TICK-BY-TICK. Binance moves it on every tick; this
     can only move it once per minute, using the candle extreme. The
     simulated trail therefore lags the real one and will usually give back
     MORE than the exchange would. Direction of error is known, magnitude
     is not.
  3. NO SLIPPAGE ON THE SIMULATED EXITS. Real fills are not at the trigger
     price. Measured mark-vs-last divergence on live is a median 0.068% with
     a tail past 2%; none of that is charged here.

Everything is PRICE %, leverage divided out, net of a round-trip fee.
"""

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

FEE_PCT_ROUND_TRIP = 0.09
DEFAULT_TRADES = "logs/trades.jsonl"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def replay(candles, side, entry, callback_pct, stop_pct, tp_pct,
           use_trail=True, ff_pct=None, ff_after_bars=1, ff_peak_ceiling=None):
    """
    Walk 1m candles and return the exit price under one rule set.

    `callback_pct` trails from the best price seen. `stop_pct` is a fixed
    distance from entry. `tp_pct` is a fixed target. Any may be None.

    WHEN A CANDLE WOULD HIT BOTH, THE LOSS IS BOOKED. Intra-candle order is
    unknown and assuming the favourable one is how replays flatter themselves.
    """
    short = str(side).lower().startswith("short")
    best = entry
    # FAIL-FAST, modelled as the guardian applies it:
    #   * only after ff_after_bars have passed (GUARD_FAIL_FAST_S=60 -> 1 bar)
    #   * only while the peak has stayed at or below ff_peak_ceiling
    #     (_peak_ceiling = GUARD_BREAKEVEN_AT_ROI)
    # Both conditions matter. Modelling it as a plain stop would cut trades
    # the real rule never touches and overstate the saving.
    peak_pct = 0.0
    for bar_i, (_ts, _o, high, low, close) in enumerate(candles):
        adverse = high if short else low
        favourable = low if short else high
        fav_pct = ((entry - favourable) / entry * 100 if short
                   else (favourable - entry) / entry * 100)
        peak_pct = max(peak_pct, fav_pct)

        # 0. fail-fast, before the wider rules — it is the tightest level and
        #    the real guardian evaluates it on its own poll, not after them.
        if (ff_pct is not None and bar_i >= ff_after_bars
                and (ff_peak_ceiling is None or peak_pct <= ff_peak_ceiling)):
            hit = (entry * (1 + ff_pct / 100) if short
                   else entry * (1 - ff_pct / 100))
            if (short and adverse >= hit) or (not short and adverse <= hit):
                return hit, "fail_fast"

        # 1. the adverse extreme, checked FIRST and always
        if stop_pct is not None:
            hit = entry * (1 + stop_pct / 100) if short else entry * (1 - stop_pct / 100)
            if (short and adverse >= hit) or (not short and adverse <= hit):
                return hit, "stop"
        if use_trail and callback_pct:
            trig = best * (1 + callback_pct / 100) if short else best * (1 - callback_pct / 100)
            if (short and adverse >= trig) or (not short and adverse <= trig):
                return trig, "trail"

        # 2. only then the favourable one
        if tp_pct is not None:
            tgt = entry * (1 - tp_pct / 100) if short else entry * (1 + tp_pct / 100)
            if (short and favourable <= tgt) or (not short and favourable >= tgt):
                return tgt, "tp"

        best = min(best, favourable) if short else max(best, favourable)
    return close, "timeout"


def pnl_pct(side, entry, exit_px):
    short = str(side).lower().startswith("short")
    raw = (entry - exit_px) / entry if short else (exit_px - entry) / entry
    return raw * 100 - FEE_PCT_ROUND_TRIP


def describe(name, vals, reasons=None):
    if not vals:
        print(f"  {name:<16} (none)")
        return
    v = sorted(vals)
    n = len(v)
    line = (f"  {name:<16}{n:>5}{st.median(v):>+10.3f}{st.mean(v):>+9.3f}"
            f"{sum(v):>+10.3f}{v[max(0, n//10-1)]:>+9.3f}"
            f"{v[min(n-1, 9*n//10)]:>+9.3f}{sum(1 for x in v if x > 0)/n:>7.0%}")
    if reasons:
        from collections import Counter
        c = Counter(reasons)
        line += "   " + " ".join(f"{k}:{c[k]}" for k in sorted(c))
    print(line)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trades", default=DEFAULT_TRADES)
    p.add_argument("--minutes", type=int, default=60,
                   help="how far forward to replay (default 60)")
    p.add_argument("--tp-roi", type=float, default=15.0,
                   help="take-profit in ROI %% (default 15); converted to "
                        "price %% per trade using that trade's leverage")
    p.add_argument("--demo", action="store_true")
    # Filters, so a rule change can be replayed on EXACTLY the trades it
    # would affect. Raising GUARD_FAIL_FAST_MAX_PEAK_ROI from 3.0 to 0.1
    # stops cutting the 40% of fail-fast exits whose peak sat between them —
    # and the journal cannot say what those would have done, because
    # fail-fast closes at -5% ROI regardless of peak, so every band shows the
    # same final. Only a replay answers it.
    p.add_argument("--only-exit", default=None,
                   help="replay only trades with this exit_reason")
    p.add_argument("--peak-band", default=None,
                   help="LO,HI on peak_roi (ROI %%, as recorded)")
    p.add_argument("--ff-roi", type=float, nargs="*", default=[],
                   help="fail-fast cut levels in ROI%% to simulate, e.g. "
                        "--ff-roi 3 3.5 4 5 (current live is 5)")
    p.add_argument("--ff-peak-ceiling", type=float, default=3.0,
                   help="fail-fast only applies while peak ROI%% stayed at or "
                        "below this (GUARD_BREAKEVEN_AT_ROI, default 3)")
    p.add_argument("--limit", type=int, default=200,
                   help="most recent N trades (each costs one candle fetch)")
    args = p.parse_args()

    path = Path(args.trades)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resolve_shadow_outcomes import _exchange          # noqa: E402
    ex = _exchange(args.demo)

    trades = [t for t in _rows(path)
              if _f(t.get("leverage")) and _f(t.get("entry_price"))
              and _f(t.get("opened_at")) and t.get("side")]
    if args.only_exit:
        trades = [t for t in trades
                  if str(t.get("exit_reason")) == args.only_exit]
    if args.peak_band:
        lo, hi = (float(v) for v in args.peak_band.split(","))
        trades = [t for t in trades
                  if _f(t.get("peak_roi")) is not None
                  and lo <= _f(t["peak_roi"]) < hi]
    trades = trades[-args.limit:]
    if not trades:
        print("no replayable trades (need leverage, entry_price, opened_at, side)")
        return 1

    now = time.time()
    ripe = [t for t in trades if now - _f(t["opened_at"]) > args.minutes * 60]
    print(f"{len(ripe)} of {len(trades)} trades are old enough to replay "
          f"{args.minutes} min forward — {len(ripe)} candle fetches, "
          f"expect ~{max(1, len(ripe)//60)} min\n", file=sys.stderr)

    ORDER = ["actual", "trail only", "trail + TP", "TP only"] + [
        "fail-fast %g%%" % v for v in args.ff_roi]
    out = {k: ([], []) for k in ORDER}
    for i, t in enumerate(ripe, 1):
        if len(ripe) >= 50 and i % 25 == 0:
            print(f"  {i}/{len(ripe)} ...", file=sys.stderr)
        lev = _f(t["leverage"])
        entry = _f(t["entry_price"])
        side = t["side"]
        ctx = t.get("entry_context") or {}
        cb = _f(ctx.get("callback_pct"))
        stop_roi = _f(ctx.get("sized_stop_roi"))
        stop_pct = stop_roi / lev if stop_roi else None
        tp_pct = args.tp_roi / lev
        try:
            c = ex.fetch_ohlcv(t["symbol"], "1m",
                               int(_f(t["opened_at"]) * 1000), args.minutes)
        except Exception as e:
            print(f"  {t['symbol']}: {type(e).__name__}", file=sys.stderr)
            continue
        if not c:
            continue
        candles = [(r[0], r[1], r[2], r[3], r[4]) for r in c]

        fin = _f(t.get("final_roi"))
        if fin is not None:
            out["actual"][0].append(fin / lev - FEE_PCT_ROUND_TRIP)
            out["actual"][1].append(t.get("exit_reason") or "?")
        # FAIL-FAST VARIANTS. Current is GUARD_FAIL_FAST_LOSS_ROI, and the
        # question is whether cutting TIGHTER raises the payoff ratio:
        # 75% of winners never dip past -0.32% of price, while the cut sits
        # at -0.50%. Each variant keeps the real conditions — a 60s delay and
        # the peak ceiling — so it cannot cut trades the live rule would not.
        variants = [("trail only", dict(tp_pct=None, use_trail=True)),
                    ("trail + TP", dict(tp_pct=tp_pct, use_trail=True)),
                    ("TP only",    dict(tp_pct=tp_pct, use_trail=False))]
        ceiling = (args.ff_peak_ceiling / lev
                   if args.ff_peak_ceiling is not None else None)
        for ff_roi in args.ff_roi:
            variants.append((
                "fail-fast %g%%" % ff_roi,
                dict(tp_pct=None, use_trail=True, ff_pct=ff_roi / lev,
                     ff_peak_ceiling=ceiling)))
        for name, kw in variants:
            px, why = replay(candles, side, entry, cb, stop_pct, **kw)
            out[name][0].append(pnl_pct(side, entry, px))
            out[name][1].append(why)

    print(f"PRICE %, leverage divided out, net of {FEE_PCT_ROUND_TRIP}% fees. "
          f"{args.minutes} min forward, TP at {args.tp_roi}% ROI.\n")
    print(f"  {'rule':<16}{'n':>5}{'median':>10}{'mean':>9}{'TOTAL':>10}"
          f"{'p10':>9}{'p90':>9}{'win':>7}   exits")
    for k in ORDER:
        if out.get(k) and out[k][0]:
            describe(k, *out[k])

    # PAIRED comparison. The same trade appears under every rule, so the
    # unpaired spread hugely overstates the uncertainty: most of the variance
    # is "which trade was it", which cancels when you difference per trade.
    # Comparing the two distributions instead gave t~1.0 on a difference that
    # is much better determined than that.
    import random
    random.seed(7)
    base = out["actual"][0]
    if base:
        print("\nPAIRED vs actual — same trade under both rules, "
              "difference taken per trade")
        print(f"  {'rule':<16}{'median diff':>13}{'mean diff':>11}"
              f"{'95% CI on mean':>22}{'better':>8}")
        for k in ORDER[1:]:
            v = out[k][0]
            if len(v) != len(base):
                continue
            d = [x - y for x, y in zip(v, base)]
            boot = []
            for _ in range(20000):
                boot.append(st.mean([random.choice(d) for _ in d]))
            boot.sort()
            lo, hi = boot[500], boot[19500]
            flag = "" if lo < 0 < hi else "  *"
            print(f"  {k:<16}{st.median(d):>+13.3f}{st.mean(d):>+11.3f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}"
                  f"{sum(1 for x in d if x > 0)/len(d):>7.0%}{flag}")
        print("  * = the 95% interval excludes zero")

    print("""
READ THIS BEFORE BELIEVING ANY OF IT
  Intra-candle order is unknown; a candle hitting both stop and target books
  the STOP. Pessimistic by construction — a result that needs the other
  reading is worth nothing.
  The simulated trail moves once a minute, not tick-by-tick, so it gives back
  MORE than the exchange would. Error direction known, magnitude not.
  No slippage on simulated exits (live mark-vs-last: median 0.068%, tail >2%).
  Trades still open, or newer than the window, are excluded — so a losing
  streak that is still running is invisible here.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
