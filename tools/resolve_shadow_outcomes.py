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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger(__name__)

from bot.shadow_outcomes import (  # noqa: E402
    DEFAULT_IN, DEFAULT_OUT, HORIZONS_MIN, resolve, summarize,
)


def _exchange(demo: bool):
    """
    A READ-ONLY ccxt client, built here rather than borrowed from the bot so
    this can never share state with the trading path. No API keys: public
    OHLCV needs none, and not holding them means this cannot place an order
    even by accident.

    ROUTED THROUGH SOCKS_PROXY, exactly as bot/engine.py does. Without it
    Binance answers 451 "restricted location" from the VPS's region — and it
    does so on exchangeInfo during market loading, BEFORE any candle fetch, so
    every row failed identically and logged one warning each.
    """
    import ccxt
    params = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    proxy = (os.environ.get("SOCKS_PROXY") or "").strip()
    if proxy:
        params["proxies"] = {"http": proxy, "https": proxy}
        log.info(f"proxy active: {proxy}")
    else:
        log.warning("SOCKS_PROXY is not set — if this box is in a restricted "
                    "region Binance will answer 451 on the first request.")
    ex = ccxt.binanceusdm(params)
    if demo:
        ex.enable_demo_trading(True)
    # FAIL LOUDLY AND ONCE. Region and connectivity problems are fatal for the
    # whole pass, not per-row: without this each of ~2000 rows logs its own
    # warning and the run looks like 2000 unrelated failures.
    try:
        ex.load_markets()
    except Exception as e:
        raise SystemExit(
            f"cannot reach the exchange ({type(e).__name__}: {e})\n"
            f"proxy={proxy or '(unset)'} demo={demo}\n"
            f"Nothing was written. Set SOCKS_PROXY (the bot uses "
            f"http://gluetun:8888 in production) and re-run — resolve() is "
            f"idempotent, so nothing is lost.")
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
    # 2 minutes, NOT 30. Median hold is 1.96 min and 80% of trades close
    # within 3. summarize() was changed to 2 in v3.75.0 and this was missed,
    # so the CLI kept reporting the window that made the first trigger read
    # meaningless.
    p.add_argument("--horizon", type=int, default=2,
                   help=f"minutes for --summary (labelled: {HORIZONS_MIN}); "
                        f"median hold is ~2 min, do not raise this casually")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.summary:
        n = resolve(_exchange(args.demo), args.src, args.dst)
        print(f"labelled {n} new observation(s) -> {args.dst}")

    # Rows already in the outcomes file are SKIPPED, so observations written
    # before a horizon or field change keep their old shape forever. Say so
    # rather than letting a stale summary read as current.
    try:
        import json as _json
        _rows = [_json.loads(l) for l in Path(args.dst).read_text().splitlines() if l.strip()]
        _stale = [r for r in _rows if "adverse_pct" not in r
                  or "1" not in (r.get("returns_pct") or {})]
        if _stale:
            print(f"\n!! {len(_stale)} of {len(_rows)} observations were labelled "
                  f"BEFORE the current horizons/fields and cannot be compared\n"
                  f"   with newer ones. resolve() skips rows it has already "
                  f"written, so they will\n   never update. To rebuild:\n"
                  f"     mv {args.dst} {args.dst}.old\n"
                  f"     python tools/resolve_shadow_outcomes.py\n"
                  f"   The decision log is untouched and re-labelling is "
                  f"idempotent.", file=sys.stderr)
    except Exception:
        pass

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
