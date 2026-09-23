#!/usr/bin/env python3
"""
Compare two instances (or two periods) WITHIN shape cohort.

WHY SHAPE IS THE ONLY FAIR SPLIT ACROSS INSTANCES
-------------------------------------------------
Demo's ATR is a measured **0.715x** live's for the SAME symbol (n=80 paired,
2026-09-20 exports) and 0.751x (n=13, 2026-09-21) — two independent periods,
same answer. So:

    by_atr          a "0.7-1.5%" demo bucket is a DIFFERENT market from the
                    live bucket of the same name
    by_change_24h   same problem, and each instance selects its own coins
    by_rsi          RSI agrees to ~1 point, so this one is fair
    SHAPE           ATR-NORMALISED, so it is fair BY CONSTRUCTION

That is the contribution. Comparing live and demo performance has been
confounded by every ATR-derived dimension; shape removes the confound rather
than correcting for it afterwards.

WHAT IT ANSWERS
---------------
Given the same shape of price action, does one instance do better? If yes, the
difference is execution, leverage or fees — not the market. If no, the gap is
in WHICH shapes each instance finds, which is a scanner question.

    python tools/compare_shape.py live.csv demo.csv

Accepts the CSV exports (EXPORT... on the dashboard) or trades.jsonl.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _f(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _load(path: Path) -> list:
    """CSV (flat ctx_ columns) or JSONL (nested entry_context) — one shape out."""
    rows = []
    if path.suffix.lower() == ".csv":
        for r in csv.DictReader(path.open()):
            rows.append({k[4:] if k.startswith("ctx_") else k: v
                         for k, v in r.items()})
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        flat = dict(t)
        flat.update(t.get("entry_context") or {})
        rows.append(flat)
    return rows


def _net(t):
    v = _f(t.get("net_pnl_usdt"))
    if v is not None:
        return v
    r, f = _f(t.get("realised_pnl_usdt")), _f(t.get("fees_usdt"))
    return None if r is None else r - (f or 0.0)


def _fee_multiple(rows):
    """
    How many times its own fee a trade captures. THE leverage-free measure:
    fees scale with notional and notional scales with leverage, so leverage
    multiplies gross AND fees equally and cancels here. Established in the
    handover's demo-vs-live section; reused so the two tools agree.
    """
    gross = sum(_f(t.get("realised_pnl_usdt")) or 0.0 for t in rows)
    fees = sum(_f(t.get("fees_usdt")) or 0.0 for t in rows)
    return (gross / fees) if fees else None


def _cohort(t):
    if not str(t.get("shape_ok", "")).lower() in ("true", "1"):
        return None
    if str(t.get("consolidating", "")).lower() == "true":
        return "consolidating"
    if str(t.get("extended", "")).lower() == "true":
        return "extended"
    return "neither"


def main(a_path, b_path, a_name="A", b_name="B"):
    A, B = _load(Path(a_path)), _load(Path(b_path))
    print(f"{a_name}: {len(A)} trades    {b_name}: {len(B)} trades\n")

    have = [t for t in A + B if _cohort(t)]
    if not have:
        print("No trades carry shape yet.\n"
              "Shape is stamped at SCAN time, so only trades entered after the\n"
              "v3.84.0 deploy have it. Nothing to compare — this is expected\n"
              "on the first run, not a fault.")
        return 0

    print(f"{'cohort':16s} {'n':>4s} {'net':>10s} {'net/trade':>10s} "
          f"{'win%':>5s} {'fee mult':>9s}   instance")
    for name, rows in ((a_name, A), (b_name, B)):
        for coh in ("consolidating", "neither", "extended"):
            g = [t for t in rows if _cohort(t) == coh]
            nets = [_net(t) for t in g if _net(t) is not None]
            if not nets:
                continue
            fm = _fee_multiple(g)
            print(f"{coh:16s} {len(nets):4d} {sum(nets):10.4f} "
                  f"{sum(nets)/len(nets):10.4f} "
                  f"{sum(1 for v in nets if v>0)/len(nets)*100:5.0f} "
                  f"{(f'{fm:.2f}x' if fm else '   -'):>9s}   {name}")

    print("\nRead it this way:")
    print("  same cohort, different result -> execution/fees/leverage, NOT the market")
    print("  cohorts differently POPULATED -> the scanner finds different shapes")
    print("\nfee multiple is leverage-free: fees scale with notional and notional")
    print("scales with leverage, so leverage cancels. Compare THAT across")
    print("instances, not net/trade.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(*sys.argv[1:]))
