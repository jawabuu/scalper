#!/usr/bin/env python3
"""
Before/after comparison for an AUTO_CALLBACK_ATR_MULT change, WITHIN one
instance.

    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/compare_callback_mult.py

Why this exists
---------------
Every comparison of the multiplier so far has been CROSS-CONTAINER, and demo
is live data with a ~0.751 factor on ATR, a different leverage and a
different symbol set. Those confounds are larger than the effect. Splitting
within ONE instance removes all three at once.

The groups identify THEMSELVES from the data — `callback_pct / atr_pct` on
each entry, rounded. No deploy timestamp to remember, and a re-run months
later still splits correctly.

What it reports, and why each one
---------------------------------
  drift_since_sizing   does a wider callback actually fill better? It should,
                       and cross-container it did (-0.669% vs -0.264%).
  first_sight          ROI at the guardian's FIRST observation, and the
                       share underwater. The wider callback was 95%
                       underwater vs 42% — the cost side of the trade-off.
  final                the outcome. THE ONLY ONE THAT DECIDES ANYTHING.

EVERYTHING IS IN PRICE %, NOT ROI. ROI multiplies by leverage, and comparing
ROI across any leverage change silently scales the result. Reasoning in ROI
about a quantity denominated in price caused four separate wrong conclusions
in this investigation.

What it will NOT do
-------------------
Declare a winner. It prints counts, medians, means, a trimmed mean and the
spread, and says plainly when the sample is too small — which at the observed
rate of ~40 trades/day it will be for at least a week per group.
"""

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

DEFAULT_JOURNAL = "logs/trades.jsonl"

# Below this, differences between groups are not interpretable. 30 is already
# generous for a distribution with a p10-p90 spread of ~2.5% of price.
MIN_N = 30


def _f(d, k):
    try:
        v = float(d.get(k))
        return v
    except (TypeError, ValueError):
        return None


def load(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def enrich(rows: list) -> list:
    """Per-trade figures, all in PRICE %, with the multiplier each ran under."""
    out = []
    for r in rows:
        ctx = r.get("entry_context") or {}
        atr = _f(ctx, "atr_pct")
        cb = _f(ctx, "callback_pct")
        lev = _f(r, "leverage")
        if not (atr and cb and lev):
            continue
        # The callback is max(ratio-derived, ATR floor). atr_floor trades are
        # the ones the MULTIPLIER governs; ratio trades are not, and grouping
        # them by callback/atr invents phantom cohorts (the first version
        # produced groups at 0.70/0.90/1.00 that were nothing but ratio
        # trades with coincidental ATRs).
        #
        # But DROPPING them hides the change's second effect. Live 09-21:
        # at 1.25 the floor bound on 97% of trades; at 0.563 it bound on 52%
        # and AUTO_CALLBACK_RATIO took the rest. Eleven trades vanished from
        # the comparison, and a result read as "0.563 vs 1.25" would partly
        # be "ratio-source vs floor-source".
        #
        # So they are kept as their OWN cohort, labelled, never merged.
        src = (ctx.get("callback_source") or "").strip() or "unknown"
        fin = _f(r, "final_roi")
        fs = _f(r, "roi_at_first_sight")
        out.append({
            "src": src,
            # None for anything the multiplier does not govern: a number here
            # would be read as a setting when it is an accident of the data.
            "mult": round(cb / atr, 2) if src == "atr_floor" else None,
            "cb": cb,
            "lev": lev,
            "atr": atr,
            "symbol": r.get("symbol"),
            "opened_at": _f(r, "opened_at"),
            "drift": _f(r, "drift_since_sizing_pct"),
            "first_sight_px": None if fs is None else fs / lev,
            "final_px": None if fin is None else fin / lev,
            # never_green: the position never traded above entry. On the
            # 2026-09-21 exports this split winners from losers more cleanly
            # than anything else — 30% of live trades, median -0.570% of
            # price, against +0.237% for those that did go green.
            #
            # A WIDER callback fills deeper into the bounce, so the position
            # starts further underwater and needs more recovery just to reach
            # entry. Lowering the multiplier should REDUCE this rate; that is
            # the sharpest prediction of the change.
            #
            # CAVEAT: peak_roi is SAMPLED, not tracked, and under-records by a
            # median 9.55 ROI points on fast moves. A position that went green
            # between polls records peak <= 0 and is counted here. This
            # OVER-counts, and does so more when moves are fast.
            "never_green": (None if _f(r, "peak_roi") is None
                            else _f(r, "peak_roi") <= 0.0),
            "net": _f(r, "net_pnl_usdt"),
            "fees": _f(r, "fees_usdt"),
            "exit": r.get("exit_reason"),
        })
    return out


def _stats(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    n = len(vals)
    trimmed = vals[1:-1] if n > 4 else vals
    return {
        "n": n,
        "median": st.median(vals),
        "mean": st.mean(vals),
        "trimmed_mean": st.mean(trimmed),
        "p10": vals[max(0, n // 10 - 1)],
        "p90": vals[min(n - 1, 9 * n // 10)],
        "win": sum(1 for v in vals if v > 0) / n,
    }


def _lbl(k):
    return f"{k:.2f}" if isinstance(k, float) else str(k)


def report(groups: dict) -> None:
    keys = sorted(groups, key=lambda k: (isinstance(k, str), k))
    print(f"{'multiplier':<12}{'n':>5}{'ATR med':>10}{'lev':>6}"
          f"{'drift':>9}{'first_sight':>13}{'underwater':>12}")
    for k in keys:
        g = groups[k]
        fs = [d["first_sight_px"] for d in g if d["first_sight_px"] is not None]
        under = (sum(1 for v in fs if v < 0) / len(fs)) if fs else float("nan")
        print(f"{_lbl(k):<12}{len(g):>5}"
              f"{st.median([d['atr'] for d in g]):>10.3f}"
              f"{st.median([d['lev'] for d in g]):>6.0f}"
              f"{st.median([d['drift'] for d in g if d['drift'] is not None]):>9.3f}"
              f"{(st.median(fs) if fs else float('nan')):>13.3f}"
              f"{under:>11.0%}")

    print("\nNEVER_GREEN — never traded above entry. The sharpest split in the")
    print("data, and the metric most likely to move with the multiplier.")
    print(f"{'multiplier':<12}{'n':>5}{'never_green':>13}{'  their final':>15}"
          f"{'  others final':>16}")
    for k in keys:
        g = [d for d in groups[k] if d["never_green"] is not None]
        if not g:
            continue
        ng = [d for d in g if d["never_green"]]
        ok = [d for d in g if not d["never_green"]]
        f_ng = [d["final_px"] for d in ng if d["final_px"] is not None]
        f_ok = [d["final_px"] for d in ok if d["final_px"] is not None]
        print(f"{_lbl(k):<12}{len(g):>5}{len(ng) / len(g):>12.0%}"
              f"{(st.median(f_ng) if f_ng else float('nan')):>+15.3f}"
              f"{(st.median(f_ok) if f_ok else float('nan')):>+16.3f}")
    print("  (peak_roi is sampled — this OVER-counts when moves are fast)")

    print("\nFINAL OUTCOME — price %, leverage divided out. The deciding number.")
    print(f"{'multiplier':<12}{'n':>5}{'median':>9}{'mean':>9}{'trimmed':>9}"
          f"{'p10':>9}{'p90':>9}{'win':>7}")
    for k in keys:
        s = _stats([d["final_px"] for d in groups[k]])
        if not s:
            continue
        print(f"{_lbl(k):<12}{s['n']:>5}{s['median']:>+9.3f}{s['mean']:>+9.3f}"
              f"{s['trimmed_mean']:>+9.3f}{s['p10']:>+9.3f}{s['p90']:>+9.3f}"
              f"{s['win']:>7.0%}")

    print("\nECONOMICS — leverage-free: how many times its own fee a trade captures")
    for k in keys:
        g = groups[k]
        fees = sum(d["fees"] for d in g if d["fees"] is not None)
        net = sum(d["net"] for d in g if d["net"] is not None)
        if fees:
            print(f"  {_lbl(k)}: gross {net + fees:+.2f}  fees {fees:.2f}  "
                  f"-> {(net + fees) / fees:.2f}x   net {net:+.2f} USDT")

    print("\nEXITS")
    for k in keys:
        print(f"  {_lbl(k)}: {dict(Counter(d['exit'] for d in groups[k]))}")


def warn(groups: dict) -> None:
    keys = sorted(groups, key=lambda k: (isinstance(k, str), k))
    print()
    small = [k for k in keys if len(groups[k]) < MIN_N]
    if small:
        print(f"!! GROUPS UNDER {MIN_N} TRADES: "
              f"{', '.join(f'{_lbl(k)} (n={len(groups[k])})' for k in small)}")
        print("   Differences are not interpretable yet. At ~40 trades/day this")
        print("   needs roughly a week per group. Do not act on this output.")

    levs = {k: st.median([d["lev"] for d in groups[k]]) for k in keys}
    if len({round(v) for v in levs.values()}) > 1:
        print("!! LEVERAGE DIFFERS BETWEEN GROUPS — this is no longer a clean")
        print("   within-instance comparison. Price % removes the direct effect")
        print("   but not the sizing and fee differences that come with it.")

    atrs = {k: st.median([d["atr"] for d in groups[k]]) for k in keys}

    # The floor/ratio split is ENDOGENOUS. The floor binds when ATR is high
    # relative to the distance and the ratio wins when ATR is low, so "ratio"
    # is largely a label for CALM COINS. Live 2026-09-22: ratio cohort median
    # ATR 0.556% against 0.900 and 1.012 for the floor groups, and it looked
    # much the best on every outcome. That may be the market, not the rule.
    non_floor = [k for k in keys if isinstance(k, str)]
    floor = [k for k in keys if isinstance(k, float)]
    if non_floor and floor:
        fa = st.median([atrs[k] for k in floor])
        for k in non_floor:
            if fa and not (0.8 <= atrs[k] / fa <= 1.25):
                print(f"!! COHORT '{k}' IS ATR-SELECTED: median ATR "
                      f"{atrs[k]:.3f}% vs {fa:.3f}% for the floor groups "
                      f"({atrs[k] / fa:.2f}x).")
                print("   Membership DEPENDS on ATR and ATR predicts outcome, so")
                print("   this cohort's numbers are a volatility regime as much as")
                print("   a sizing rule. Do not read it as a fair comparison.")

    # Compare the FLOOR groups pairwise. The first version only ran when
    # there were exactly two cohorts, so the moment a 'ratio' cohort appeared
    # the check silently stopped firing — and it stopped firing precisely when
    # the floor groups had drifted to 1.29 vs 1.01 (1.27x), outside tolerance.
    # A guard that switches itself off as the data gets more complex is worse
    # than no guard.
    for i, a_k in enumerate(floor):
        for b_k in floor[i + 1:]:
            a, b = atrs[a_k], atrs[b_k]
            if a and b and not (0.8 <= a / b <= 1.25):
                print(f"!! MARKET REGIME DIFFERS between {_lbl(a_k)} and "
                      f"{_lbl(b_k)}: median ATR {a:.3f} vs {b:.3f} "
                      f"({a / b:.2f}x).")
                print("   These groups did not trade the same volatility. A")
                print("   difference in outcome may be the market, not the")
                print("   multiplier.")

    print("\nRemember: a wider callback is EXPECTED to fill better (drift) and")
    print("to be underwater more often at first sight. Neither is the result.")
    print("Only `final` decides, and only once both groups clear the sample bar.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--journal", default=DEFAULT_JOURNAL)
    p.add_argument("--since", type=float, default=None,
                   help="unix seconds; ignore trades opened before this")
    args = p.parse_args()

    path = Path(args.journal)
    if not path.exists():
        print(f"no journal at {path}", file=sys.stderr)
        return 1

    rows = enrich(load(path))
    if args.since:
        rows = [r for r in rows if (r["opened_at"] or 0) >= args.since]
    if not rows:
        print("no atr_floor trades with both callback_pct and atr_pct recorded")
        return 0

    # Multipliers cluster (0.74/0.75/0.76 are one setting); merge to 1
    # decimal. Non-floor sources keep their source name as the label so they
    # can never be mistaken for a multiplier setting.
    merged = {}
    for r in rows:
        key = round(r["mult"], 1) if r["mult"] is not None else r["src"]
        merged.setdefault(key, []).append(r)

    floor_keys = [k for k in merged if isinstance(k, float)]
    other = {k: v for k, v in merged.items() if not isinstance(k, float)}
    if other:
        print("NOTE: not every trade is governed by the multiplier. The callback")
        print("is max(ratio-derived, ATR floor), so lowering the floor hands")
        print("trades to AUTO_CALLBACK_RATIO instead. Those are shown as their")
        print("own cohort — a difference between floor groups may partly be a")
        print("difference in WHICH SOURCE won.\n")
        for k, v in sorted(other.items()):
            share = len(v) / len(rows)
            print(f"  source '{k}': {len(v)} trades ({share:.0%} of all), "
                  f"callback median {st.median([d['cb'] for d in v]):.3f}%")
        print()

    if len(floor_keys) < 2:
        only = floor_keys[0] if floor_keys else None
        print(f"only ONE multiplier in this journal: "
              f"{only if only is None else f'{only:.2f}'} "
              f"(n={sum(len(merged[k]) for k in floor_keys)} floor-sourced).")
        print("Nothing to compare yet — this is the BEFORE baseline.")
        print("Re-run after the change has accumulated trades.")
        report(merged)
        return 0

    report(merged)
    warn(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
