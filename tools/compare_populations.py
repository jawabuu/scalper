#!/usr/bin/env python3
"""
Would trading jev's picks have beaten trading the bot's?

    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/compare_populations.py

Why this is answerable at all
-----------------------------
Both populations are in the shadow log. Every candidate carries the bot's
decision AND jev's verdict, and the resolver labels the forward path of both.
So the comparison is HYPOTHETICAL-vs-HYPOTHETICAL, not hypothetical against
the bot's real fills — which would have been meaningless, since real trades
carry stops, trails, slippage and fees that a refused candidate never sees.

    bot only    bot ENTER, jev SKIP     what the bot trades today
    jev only    bot SKIP,  jev ENTER    what it would trade instead
    both        both ENTER              the overlap (small)
    neither     both SKIP               the rest

WHAT THIS IS NOT
----------------
It is NOT a backtest. No stop, no trail, no fail-fast, no fill model, no
slippage. It measures the PRICE PATH available after each decision, which is
the fairest like-for-like available and still an upper bound on what any
strategy could extract from it.

Two things it cannot tell you:

  CAPACITY. ENTRY_MAX_POSITIONS is 6 and the symbol cooldown is 1800s. jev's
  population is ~40x the bot's, so most of it could never be taken. A per-
  trade edge there is not an achievable edge.

  FEES. ~0.09% of price per round trip, unchanged by leverage, is subtracted
  explicitly below. On a population whose median move is ~0.25%, that is not
  a detail — it is most of the answer.
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

DEFAULT_OUT = "logs/shadow_outcomes.jsonl"

# Round-trip cost as a % of PRICE. Measured at 0.0997-0.1000% of notional on
# both instances (fee rate is identical; only notional differs), so ~0.1% in
# and out is the honest figure to subtract from a raw price move.
FEE_PCT_ROUND_TRIP = 0.09


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


def bucket(rows: list) -> dict:
    out = {"bot only": [], "jev only": [], "both": [], "neither": []}
    for r in rows:
        b = (r.get("bot_decision") or "").upper() == "ENTER"
        j = (r.get("jev_verdict") or "").upper() == "ENTER"
        out["both" if (b and j) else
            "bot only" if b else
            "jev only" if j else "neither"].append(r)
    return out


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def report(groups: dict, horizon: str) -> None:
    h = horizon
    print(f"\nHorizon {h} min — favoured_side_pct is the PRICE move in the "
          f"trade's direction.\n")
    print(f"{'population':<12}{'n':>6}{'ATR':>8}{'median':>9}{'mean':>8}"
          f"{'win':>7}{'adverse':>9}{'edge':>7}{'net of fees':>13}")
    for k in ("bot only", "both", "jev only", "neither"):
        g = groups[k]
        v = [r.get("favoured_side_pct", {}).get(h) for r in g]
        v = [x for x in v if x is not None]
        if len(v) < 5:
            continue
        adv = med([r.get("adverse_pct", {}).get(h) for r in g])
        edge = med([r.get("edge_ratio", {}).get(h) for r in g])
        atr = med([r.get("atr_pct") for r in g])
        # Net expectancy per trade: the median move less a round trip. The
        # median, not the mean, because the mean is dominated by a handful of
        # large moves that capacity limits would never let you take anyway.
        net = st.median(v) - FEE_PCT_ROUND_TRIP
        print(f"{k:<12}{len(v):>6}{(atr or 0):>8.3f}{st.median(v):>+9.3f}"
              f"{st.mean(v):>+8.3f}{sum(1 for x in v if x > 0) / len(v):>7.0%}"
              f"{(adv or 0):>9.3f}{(edge or 0):>7.3f}{net:>+13.3f}")


def warn(groups: dict) -> None:
    b, j = len(groups["bot only"]), len(groups["jev only"])
    print()
    if b and j:
        print(f"!! CAPACITY: jev's population is {j / max(b, 1):.0f}x the "
              f"bot's ({j} vs {b}).")
        print("   ENTRY_MAX_POSITIONS=6 and a 1800s symbol cooldown mean most "
              "of it")
        print("   could never be taken. A per-trade edge there is NOT an "
              "achievable edge.")
    print(f"!! FEES: {FEE_PCT_ROUND_TRIP:.2f}% of price per round trip is "
          f"subtracted in the last")
    print("   column. On a population whose median move is ~0.25%, that is "
          "most of the answer.")
    print("!! NOT A BACKTEST: no stop, no trail, no fail-fast, no fill model, "
          "no slippage.")
    print("   These are upper bounds on what any strategy could extract.")
    # Volatility selection, the trap that killed three earlier findings.
    atrs = {k: med([r.get("atr_pct") for r in v]) for k, v in groups.items()
            if len(v) >= 5}
    if "bot only" in atrs and "jev only" in atrs and atrs["bot only"]:
        ratio = (atrs["jev only"] or 0) / atrs["bot only"]
        if not (0.8 <= ratio <= 1.25):
            print(f"!! ATR DIFFERS: jev-only {atrs['jev only']:.3f}% vs "
                  f"bot-only {atrs['bot only']:.3f}% ({ratio:.2f}x).")
            print("   A calmer population shows a smaller adverse excursion "
                  "MECHANICALLY.")
            print("   Compare `edge` (dimensionless) rather than `adverse`.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outcomes", default=DEFAULT_OUT)
    p.add_argument("--horizons", default="2,5,30",
                   help="comma-separated minutes (default 2,5,30)")
    args = p.parse_args()

    path = Path(args.outcomes)
    if not path.exists():
        print(f"no outcomes file at {path}", file=sys.stderr)
        return 1
    rows = load(path)
    if not rows:
        print("no observations yet")
        return 0

    groups = bucket(rows)
    print(f"{len(rows)} observations")
    for k, v in groups.items():
        print(f"  {k:<10} {len(v):>6}")
    for h in [x.strip() for x in args.horizons.split(",") if x.strip()]:
        report(groups, h)
    warn(groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
