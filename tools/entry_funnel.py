#!/usr/bin/env python3
"""
WHERE are entries being lost, and to WHICH change?

    docker logs $(docker ps -q -f name=scalper-1) 2>&1 \
        | python tools/entry_funnel.py -

Three changes landed within days of each other and all three reduce entries:

    AUTO_SHORT_RSI_MIN=72        admits MORE (should raise entries)
    ENTRY_DEFER_S=120            a candidate must keep qualifying for 2 min
    ENTRY_ORDER_TYPE=maker_limit a post-only limit may never fill

Plus the market itself: movers per scan fell from ~40 to ~20 over the same
period, which is not a config change at all.

Counting entries tells you nothing about which. The funnel does, because each
stage has its own log line and its own attrition:

    movers        the market's offer          <- not a config change
    candidates    scanner screening
    refusals      the deterministic gates     <- where the RSI band acts
    deferred      ENTRY_DEFER_S
    placed        an order reached the exchange
    filled        maker_limit's fill rate     <- where post-only acts

A stage that loses 5% is not your problem however loud it is; a stage that
loses 60% is, however quiet.

READS THE CONTAINER LOG, so it only sees what is still in the ring buffer —
usually a few hours. It is a snapshot, not a history. Re-run it across
different windows rather than trusting one.
"""

import argparse
import re
import sys
from collections import Counter

SCAN = re.compile(r"Scan: (\d+) candidate\(s\) from (\d+) movers")
REFUSE = re.compile(r"auto-trade refusals this cycle: (.+)")
DEFERRED = re.compile(r"(\S+) DEFERRED — must keep qualifying")
DEFER_OK = re.compile(r"(\S+) defer satisfied")
PLACED = re.compile(r"(?:MAKER ENTRY|ENTRY PLACED)")
MAKER = re.compile(r"MAKER ENTRY")
FILLED = re.compile(r"FILLED — a position is open")
STALE = re.compile(r"cancelled stale entry order")
REJECT = re.compile(r"-2021")
HALT = re.compile(r"STILL HALTED")
GATE = re.compile(r"blocked by shadow gate")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logfile", nargs="?", default="-",
                   help="'-' for stdin (the usual: docker logs ... | this)")
    args = p.parse_args()
    fh = sys.stdin if args.logfile == "-" else open(args.logfile, encoding="utf-8",
                                                    errors="ignore")

    scans, movers, cands = 0, 0, 0
    reasons = Counter()
    deferred, defer_ok = set(), set()
    placed = maker = filled = stale = rejected = halted = gated = 0

    for line in fh:
        m = SCAN.search(line)
        if m:
            scans += 1
            cands += int(m.group(1))
            movers += int(m.group(2))
            continue
        m = REFUSE.search(line)
        if m:
            # The counter is per CYCLE and cycles repeat the same candidates,
            # so these are not unique symbols. Use the SHARE between reasons,
            # never the absolute number.
            for part in m.group(1).split(","):
                k, _, v = part.strip().partition("=")
                try:
                    reasons[k] += int(v)
                except ValueError:
                    pass
            continue
        m = DEFERRED.search(line)
        if m:
            deferred.add(m.group(1))
            continue
        m = DEFER_OK.search(line)
        if m:
            defer_ok.add(m.group(1))
            continue
        if MAKER.search(line):
            maker += 1
        if PLACED.search(line):
            placed += 1
        if FILLED.search(line):
            filled += 1
        if STALE.search(line):
            stale += 1
        if REJECT.search(line):
            rejected += 1
        if HALT.search(line):
            halted += 1
        if GATE.search(line):
            gated += 1

    if not scans:
        print("no 'Scan:' lines — is this the right container's log?",
              file=sys.stderr)
        return 1

    print(f"{'THE MARKET':<28}")
    print(f"  scans                    {scans}")
    print(f"  movers per scan          {movers/scans:.1f}"
          f"     (was ~40 on 2026-09-21..25)")
    print(f"  candidates per scan      {cands/scans:.1f}")
    print(f"  -> the scanner passes    {cands/max(movers,1):.0%} of movers\n")

    print(f"{'THE GATES  (share, not count — cycles repeat candidates)':<28}")
    tot = sum(reasons.values()) or 1
    for k, v in reasons.most_common(8):
        print(f"  {k:<24} {v/tot:>5.0%}")
    print()

    print(f"{'THE NEW CHANGES':<28}")
    if deferred:
        thru = len(defer_ok & deferred) or len(defer_ok)
        print(f"  ENTRY_DEFER_S")
        print(f"    started the clock      {len(deferred)}")
        print(f"    reached placement      {thru}"
              f"   -> {1-thru/len(deferred):.0%} lost here")
    else:
        print("  ENTRY_DEFER_S            no DEFERRED lines — not active, or "
              "nothing reached it")
    if maker:
        resolved = filled + stale
        print(f"  ENTRY_ORDER_TYPE=maker_limit")
        print(f"    placed                 {maker}")
        print(f"    rejected (-2021)       {rejected}")
        print(f"    filled                 {filled}")
        print(f"    stale (TTL expired)    {stale}")
        if resolved:
            print(f"    fill rate              {filled/resolved:.0%} of resolved"
                  f"   -> {1-filled/max(resolved,1):.0%} lost here")
    else:
        print("  maker_limit              no MAKER ENTRY lines — not active")
    if gated:
        print(f"  shadow gate blocked      {gated}")
    if halted:
        print(f"  !! daily halt was ACTIVE for part of this window "
              f"({halted} heartbeats) — entries were blocked for reasons "
              f"unrelated to any of the above")
    print(f"""
READ IT THIS WAY
  movers per scan is the MARKET. If it has halved, entries halve with it and
  no config change is responsible for that part.
  The gate SHARES say which deterministic rule dominates; the RSI band acts
  there, and lowering AUTO_SHORT_RSI_MIN should SHRINK rsi_band's share.
  The defer and the fill rate are the only two stages a change added, and
  each prints its own attrition.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
