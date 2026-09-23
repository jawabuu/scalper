#!/usr/bin/env python3
"""
Would trading jev's picks have beaten trading the bot's?

    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/compare_jev_picks.py

THE TWO POPULATIONS ARE NOT COMPARABLE AND THIS TOOL SAYS SO LOUDLY.

  bot entries   real trades: real fills, slippage, stops, trails, fail-fast,
                fees, and an entry that only filled after a retracement.
                Everything that can go wrong, did or didn't.

  jev picks     candidates the bot SKIPPED, labelled with the forward price
                move from the DECISION price. No fill, no slippage, no stop,
                no trail, no fail-fast. An idealised path.

A naive comparison hands jev a large unearned advantage. This tool charges
the jev side for what it can (fees, and optionally a stop) and then states
plainly what remains uncharged. **A gap smaller than the uncharged costs is
not evidence of anything.**

Why the question is worth asking at all: jev says ENTER on thousands of
candidates the bot skips, and that population has a materially better
favourable-to-adverse ratio (1.119 vs 0.800 at 2 min, ATR-normalised). That
is a real signal in a population the bot does not trade. Whether it survives
execution is exactly what cannot be read off the shadow log.
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

DEFAULT_TRADES = "logs/trades.jsonl"
DEFAULT_OUTCOMES = "logs/shadow_outcomes.jsonl"

# Round-trip cost in PRICE %, both sides of the trade. Leverage-free, so it
# applies identically to a real trade and a hypothetical one.
FEE_PCT_ROUND_TRIP = 0.09

# The horizon that matches the strategy: median hold is 1.96 min, 80% of
# trades close within 3.
HORIZON = "2"


def _load(path: Path) -> list:
    if not path.exists():
        return []
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


def bot_trades(rows: list) -> list:
    """Real outcomes, in PRICE % with leverage divided out, NET of fees."""
    out = []
    for r in rows:
        lev = r.get("leverage")
        fin = r.get("final_roi")
        if not lev or fin is None:
            continue
        # final_roi is gross of fees; charge the same round trip the jev side
        # pays so the two are on one footing.
        out.append(fin / lev - FEE_PCT_ROUND_TRIP)
    return out


def jev_picks(rows: list, stop_atr: float | None) -> tuple:
    """
    Candidates the bot SKIPPED and jev said ENTER on.

    `favoured_side_pct` is signed so + means the direction was right. If
    `stop_atr` is given, a pick whose ADVERSE excursion exceeded that many
    ATRs is booked as a stop-out at that level instead of at its close —
    the single largest uncharged cost otherwise.
    """
    out, stopped = [], 0
    for r in rows:
        if r.get("jev_verdict") != "ENTER" or r.get("bot_decision") != "SKIP":
            continue
        fav = (r.get("favoured_side_pct") or {}).get(HORIZON)
        if fav is None:
            continue
        if stop_atr is not None:
            atr = r.get("atr_pct")
            adv = (r.get("adverse_pct") or {}).get(HORIZON)
            if atr and adv is not None and adv >= stop_atr * atr:
                out.append(-stop_atr * atr - FEE_PCT_ROUND_TRIP)
                stopped += 1
                continue
        out.append(fav - FEE_PCT_ROUND_TRIP)
    return out, stopped


def describe(label: str, vals: list) -> None:
    if not vals:
        print(f"{label:<22} (no data)")
        return
    v = sorted(vals)
    n = len(v)
    print(f"{label:<22}{n:>6}{st.median(v):>+10.3f}{st.mean(v):>+9.3f}"
          f"{v[max(0, n // 10 - 1)]:>+9.3f}{v[min(n - 1, 9 * n // 10)]:>+9.3f}"
          f"{sum(1 for x in v if x > 0) / n:>7.0%}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trades", default=DEFAULT_TRADES)
    p.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    p.add_argument("--stop-atr", type=float, default=1.5,
                   help="charge jev's picks a stop at N x ATR of adverse "
                        "excursion (default 1.5, matching ATR_STOP_MULT); "
                        "0 disables")
    args = p.parse_args()

    tr = _load(Path(args.trades))
    oc = _load(Path(args.outcomes))
    if not tr or not oc:
        print("need both logs/trades.jsonl and logs/shadow_outcomes.jsonl",
              file=sys.stderr)
        return 1

    bot = bot_trades(tr)
    stop = None if args.stop_atr <= 0 else args.stop_atr
    jev, stopped = jev_picks(oc, stop)

    print(f"PRICE %, leverage divided out, NET of a {FEE_PCT_ROUND_TRIP}% "
          f"round trip. Horizon {HORIZON} min.\n")
    print(f"{'':22}{'n':>6}{'median':>10}{'mean':>9}{'p10':>9}{'p90':>9}{'win':>7}")
    describe("bot's real trades", bot)
    describe("jev's picks", jev)
    if stop:
        print(f"\n  {stopped}/{len(jev)} of jev's picks ({stopped/max(len(jev),1):.0%}) "
              f"hit a {stop}x ATR stop and were booked there.")

    print("\n" + "=" * 68)
    print("WHAT THE jev SIDE IS STILL NOT CHARGED FOR")
    print("=" * 68)
    print("""
  ENTRY FILL. The bot enters on a trailing order that waits for a
  retracement. Many of these candidates would never have filled at all, and
  those that did would fill at a worse price than the decision price used
  here. This is the single largest unmodelled advantage.

  EXIT LOGIC. Real trades exit on a trail, a fail-fast at 60s, or a stop —
  all of which cut winners short as well as losers. jev's picks are marked
  at a fixed horizon, which is the best possible exit rule in hindsight.

  SLIPPAGE AND MARK-vs-LAST. Measured at a median 0.068% of price on live,
  with a tail to 2.4%. Charged on every real trade, none of the picks.

  CAPACITY. jev says ENTER on thousands of candidates; the bot takes ~40 a
  day against ENTRY_MAX_POSITIONS=6. Trading all of them is not available,
  so the comparison is against a portfolio that could not be held.

  A GAP SMALLER THAN THESE IS NOT EVIDENCE. Treat a jev advantage under
  roughly 0.2% of price per trade as unproven.
""")
    if bot and jev:
        gap = st.median(jev) - st.median(bot)
        print(f"  observed median gap: {gap:+.3f}% of price per trade")
        if abs(gap) < 0.2:
            print("  -> INSIDE the uncharged-cost band. Not evidence either way.")
        elif gap > 0:
            print("  -> larger than the band, but see the list above before "
                  "believing it.")
        else:
            print("  -> jev's picks are WORSE even before its advantages are "
                  "charged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
