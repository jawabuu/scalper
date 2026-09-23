#!/usr/bin/env python3
"""
What would SHAPE have done as a gate? Counterfactual, on trades already taken.

Two questions, deliberately separate:

  1. IS IT CORRECT?  On refused candidates (shadow rows), does a consolidating
     shape precede a smaller adverse excursion at 1-3 minutes? That is the
     operator's actual goal, and refused candidates are where the sample is.
     -> tools/resolve_shadow_outcomes.py already reports this, now split by
        shape_favours and by each raw component.

  2. WHAT WOULD IT HAVE EARNED?  On ENTERED trades, if the bot had skipped
     every entry shape did not favour, what happens to net P&L? That is this
     tool.

WHY BOTH. Question 1 is the honest test — it is measured on the 99% the bot
declines, so it is not conditioned on the trigger being tested. Question 2 is
the one that decides anything, but it is conditioned on the existing entry
rules, so a good answer here and a bad answer there means the shape is picking
up something the current gates already capture.

READ THE CAVEAT BEFORE QUOTING ANY NUMBER
-----------------------------------------
This scores a gate on trades that were ENTERED. It cannot see the trades a
shape gate would have ADDED, and it assumes every skipped trade returns
exactly zero — no fee, no slippage, no opportunity cost. It is an upper bound
on the benefit of skipping, not an estimate of the strategy.

The operator's stated interest is the reverse direction: if shape is reliable,
ADMIT more coins (relax 24h change, relax distance-to-extreme). This tool
cannot answer that. Only question 1 on refused candidates can, because those
are the coins that would be admitted.

    python tools/score_shape.py logs/trades.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _rows(path: Path) -> list:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _f(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _net(t):
    v = _f(t.get("net_pnl_usdt"))
    if v is not None:
        return v
    r, f = _f(t.get("realised_pnl_usdt")), _f(t.get("fees_usdt"))
    return None if r is None else r - (f or 0.0)


def _ctx(t, key):
    return (t.get("entry_context") or {}).get(key)


def main(path="logs/trades.jsonl"):
    p = Path(path)
    if not p.exists():
        print(f"no such file: {p}")
        return 1
    trades = [t for t in _rows(p) if _net(t) is not None]
    if not trades:
        print("no trades with a net figure")
        return 1

    with_shape = [t for t in trades if _ctx(t, "shape_ok")]
    print(f"trades: {len(trades)}   with shape recorded: {len(with_shape)}")
    if not with_shape:
        print("\nShape is not on these rows yet. It is stamped at SCAN time, so\n"
              "only trades entered after the deploy carry it. Nothing to score.")
        return 0

    base = sum(_net(t) for t in with_shape)
    print(f"\nactual net over those trades: {base:+.4f}")

    # The label, then each component as a threshold sweep. The sweep is the
    # point: a single label can fail because its thresholds are wrong rather
    # than because the idea is.
    def report(name, keep):
        kept = [t for t in with_shape if keep(t)]
        skipped = [t for t in with_shape if not keep(t)]
        if not kept or not skipped:
            return
        knet = sum(_net(t) for t in kept)
        kwins = sum(1 for t in kept if _net(t) > 0)
        print(f"  {name:34s} keep {len(kept):3d}  skip {len(skipped):3d}  "
              f"net {knet:+8.4f}  delta {knet - base:+8.4f}  "
              f"win {kwins / len(kept) * 100:4.0f}%")

    print("\nIf the bot had SKIPPED entries the rule did not favour:")
    print(f"  {'rule':34s} {'':>26s} {'net':>9s} {'vs actual':>10s}")
    report("shape_favours is True", lambda t: _ctx(t, "consolidating") is True)
    report("not extended", lambda t: _ctx(t, "extended") is not True)
    for thr in (0.7, 0.8, 0.9, 1.0):
        report(f"compression <= {thr}",
               lambda t, x=thr: (_f(_ctx(t, "compression")) or 9) <= x)
    for thr in (1.0, 1.5, 2.0, 3.0):
        report(f"extension_atr <= {thr}",
               lambda t, x=thr: (_f(_ctx(t, "extension_atr")) or 9) <= x)
    for thr in (0.8, 1.0, 1.2):
        report(f"accel >= {thr}",
               lambda t, x=thr: (_f(_ctx(t, "accel")) or 0) >= x)

    print("\nCAVEAT: scored on trades ALREADY ENTERED. It cannot see trades a\n"
          "shape gate would have ADDED, and assumes a skipped trade returns\n"
          "exactly zero. Upper bound on skipping, not an estimate of the\n"
          "strategy. For 'can I admit MORE coins', see the refused-candidate\n"
          "split in tools/resolve_shadow_outcomes.py — those are the coins\n"
          "that would be admitted, and this tool cannot reach them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
