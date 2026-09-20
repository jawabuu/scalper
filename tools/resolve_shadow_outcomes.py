#!/usr/bin/env python3
"""
Label shadow decisions with what the market actually did next.

Run it on the box, inside the bot container:

    docker exec $(docker ps -q -f name=scalper-pullback-1) \
        python tools/resolve_shadow_outcomes.py

    docker exec $(docker ps -q -f name=scalper-pullback-1) \
        python tools/resolve_shadow_outcomes.py --summary

Safe to re-run: already-labelled observations are skipped, and decisions too
recent to have a 60-minute outcome are left for a later run. Nothing here
touches the trading process or rewrites the decision log.

Worth running on a timer once a day or so — every run picks up whatever has
ripened since the last.

READ THE OBSERVATION COUNT, NOT THE ROW COUNT. `decision_rows_covered` is how
many log lines fed in; `observations` is the real sample size after repeated
judgements of one coin are collapsed. The first live sample was 1021 rows
across 24 symbols — a row count would have overstated it by ~40x.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.shadow_outcomes import (  # noqa: E402
    DEFAULT_IN, DEFAULT_OUT, HORIZONS_MIN, resolve, summarize,
)


def _exchange(demo: bool):
    """
    A READ-ONLY ccxt client, built here rather than borrowed from the bot so
    this can never share state with the trading path. No API keys: public
    OHLCV needs none, and not holding them means this cannot place an order
    even by accident.
    """
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    if demo:
        ex.set_sandbox_mode(True)
    return ex


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", default=DEFAULT_IN)
    p.add_argument("--out", dest="dst", default=DEFAULT_OUT)
    p.add_argument("--demo", action="store_true",
                   help="use the testnet feed (match the container you ran in)")
    p.add_argument("--summary", action="store_true",
                   help="only read what is already labelled; fetch nothing")
    p.add_argument("--horizon", type=int, default=30,
                   help=f"minutes for --summary (labelled: {HORIZONS_MIN})")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.summary:
        n = resolve(_exchange(args.demo), args.src, args.dst)
        print(f"labelled {n} new observation(s) -> {args.dst}")

    s = summarize(args.dst, horizon=args.horizon)
    print(json.dumps(s, indent=2))
    crt = s.get("by_crt_agrees") or {}
    if crt and "True" not in crt:
        print("\nNOTE: no observation yet where CRT AGREED. A trigger that "
              "never fires cannot be told apart from a good one — the "
              "positive class has to come from REFUSED candidates.",
              file=sys.stderr)
    if s.get("observations", 0) < 30:
        print("\nNOTE: under 30 observations. Read the medians as a shape "
              "check, not a result — and do not re-fit _WEIGHTS on them.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
