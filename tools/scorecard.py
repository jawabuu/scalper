#!/usr/bin/env python3
"""
Score each MECHANISM against its own goal — not against P&L.

    docker logs --since 6h $(docker ps -q -f name=scalper-1) 2>&1 \
      | docker exec -i $(docker ps -q -f name=scalper-1) \
          python tools/scorecard.py -

WHY THIS EXISTS

Four changes are live at once (change floor, proportional distance, maker
entry, stop floor) and more are queued. P&L cannot separate them, and
waiting for enough trades to separate them means months. But each mechanism
was built to do ONE measurable thing, and most of those things are
observable long before any P&L signal:

    supply        movers per scan          target ~30
    maker entry   fee as % of notional     target 0.070%
    maker fill    filled / resolved        kill below 50%
    defer         started -> placed        diagnostic
    stop floor    median stop in % price   target >= 1.00%
    leg gate      blocked share            diagnostic
    btc regime    blocked share            diagnostic

A mechanism that misses its OWN target is broken regardless of what P&L
does. A mechanism that hits it has done its job even if P&L is flat — the
P&L question is then about whether the goal was worth having, which is a
different argument and a slower one.

DEFER vs SWEEPS

Both require a candidate to persist. At AUTO_TRADE_INTERVAL=30 with a 120s
scanner and ENTRY_DEFER_S=120, they overlap heavily:

    AUTO_STRENGTH_SWEEPS=2   qualify on 2 consecutive SCANS (~120s apart),
                             sampled at scan boundaries
    ENTRY_DEFER_S=120        keep qualifying for 120s CONTINUOUSLY, checked
                             every auto-trade cycle (~4 checks)

The defer is STRICTER: continuous qualification, not two samples. A
candidate that flickers out between scans passes sweeps and fails the defer.
So they are not redundant — but running BOTH means the sweep requirement is
almost always already satisfied by the time the defer clears, which makes
sweeps a no-op rather than a second filter. This reports the overlap so the
question can be settled from data.
"""

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

SCAN = re.compile(r"Scan: (\d+) candidate\(s\) from (\d+) movers")
DEFERRED = re.compile(r"(\S+) DEFERRED — must keep qualifying")
DEFER_OK = re.compile(r"(\S+) defer satisfied — qualified continuously for (\d+)s")
MAKER = re.compile(r"MAKER ENTRY")
FILLED = re.compile(r"FILLED — a position is open")
STALE = re.compile(r"cancelled stale entry order")
LEG = re.compile(r"leg gate \(warn\) — would BLOCK|blocked — .*bars into the leg")
BTC = re.compile(r"BTC regime \(warn\)|blocked — BTC ")
TALLY = re.compile(r"(\d+)/(\d+) blocked so far")
LIMIT = re.compile(r"limit ([0-9.]+)% \(([^)]*)\)")
STREAK = re.compile(r"strengthened on (\d+) scan")


def _verdict(ok, note=""):
    return ("  PASS" if ok else "  MISS") + (f"  {note}" if note else "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logfile", nargs="?", default="-")
    p.add_argument("--trades", default="logs/trades.jsonl")
    p.add_argument("--hours", type=float, default=24.0,
                   help="window for the trade-journal half")
    args = p.parse_args()
    fh = sys.stdin if args.logfile == "-" else open(args.logfile, encoding="utf-8",
                                                    errors="ignore")
    log = fh.read()

    print("=" * 62)
    print("MECHANISM SCORECARD — each against its OWN goal, not P&L")
    print("=" * 62)

    # ── supply ────────────────────────────────────────────────────────────
    scans = SCAN.findall(log)
    print("\nSUPPLY   goal: ~30 movers per scan (was ~40 at breadth 85,")
    print("         fell to ~15 at breadth 57; the change floor is the knob)")
    if scans:
        mv = [int(m) for _, m in scans]
        cd = [int(c) for c, _ in scans]
        print(f"  scans {len(scans)}   movers/scan {st.median(mv):.1f}"
              f"   candidates/scan {st.median(cd):.1f}"
              f"   pass {sum(cd)/max(sum(mv),1):.0%}")
        print(_verdict(st.median(mv) >= 25,
                       "" if st.median(mv) >= 25 else
                       "lower SCAN_MIN_CHANGE_PCT before touching volume"))
    else:
        print("  no Scan: lines in this window")

    # ── maker entry ───────────────────────────────────────────────────────
    print("\nMAKER ENTRY   goal: fee 0.070% of notional (taker round trip is")
    print("              0.100%); kill if fill rate < 50% of RESOLVED")
    placed = len(MAKER.findall(log))
    filled = len(FILLED.findall(log))
    stale = len(STALE.findall(log))
    resolved = filled + stale
    if placed:
        print(f"  placed {placed}   filled {filled}   stale {stale}"
              f"   pending {placed - resolved}")
        if resolved:
            r = filled / resolved
            print(f"  fill rate {r:.0%} of resolved")
            print(_verdict(r >= 0.50, "" if r >= 0.50 else
                           "below the kill threshold — check for a halt in "
                           "this window first, it cancels resting entries"))
    else:
        print("  no MAKER ENTRY lines — not active in this window")

    # ── defer vs sweeps ───────────────────────────────────────────────────
    print("\nDEFER vs SWEEPS   are they the same requirement twice?")
    started = {m for m in DEFERRED.findall(log)}
    ok = DEFER_OK.findall(log)
    print(f"  started the clock {len(started)}   reached placement {len(ok)}")
    if ok:
        waits = [int(w) for _, w in ok]
        print(f"  median wait {st.median(waits)}s"
              f"   (ENTRY_DEFER_S is the floor, scans are ~120s apart)")
    streaks = [int(x) for x in STREAK.findall(log)]
    if streaks:
        print(f"  candidates refused on streak: {len(streaks)}"
              f"   median streak {st.median(streaks):.0f}")
    print("""  READ IT THIS WAY
    The defer requires CONTINUOUS qualification over ~4 auto-trade cycles.
    AUTO_STRENGTH_SWEEPS=2 requires qualification at 2 SCAN BOUNDARIES,
    ~120s apart. The defer is strictly harder: a candidate that flickers
    out between scans passes sweeps and fails the defer.
    So with ENTRY_DEFER_S=120 and sweeps=2, SWEEPS IS A NO-OP — anything
    clearing the defer has necessarily been qualifying across two scans.
    Running sweeps=1 alongside the defer is the honest configuration;
    sweeps=2 adds nothing and makes the refusal counts harder to read.""")

    # ── proportional distance ─────────────────────────────────────────────
    print("\nPROPORTIONAL DISTANCE   goal: the limit scales with the move, so")
    print("                        lowering the change floor cannot turn an")
    print("                        extremes gate into a mid-range one")
    lims = LIMIT.findall(log)
    if lims:
        prop = [(float(v), w) for v, w in lims if "absolute" not in w]
        absn = [v for v, w in lims if "absolute" in w]
        print(f"  refusals carrying a limit: {len(lims)}"
              f"   proportional {len(prop)}   absolute {len(absn)}")
        if prop:
            print(f"  proportional limits seen: "
                  f"{min(v for v, _ in prop):.2f}% to "
                  f"{max(v for v, _ in prop):.2f}%")
        print(_verdict(bool(prop), "" if prop else
                       "only absolute limits — either no small movers were "
                       "refused yet, or the setting is not reaching the "
                       "auto-trader"))
    else:
        print("  no distance refusals in this window (the reason prints only")
        print("  when the gate FIRES, so silence is not evidence either way)")

    # ── gates in warn ─────────────────────────────────────────────────────
    for name, rx in (("LEG GATE", LEG), ("BTC REGIME", BTC)):
        hits = rx.findall(log)
        print(f"\n{name}   goal: measure the block rate before trading on it")
        if hits:
            tal = TALLY.findall(log)
            if tal:
                b, s = tal[-1]
                print(f"  {b}/{s} blocked ({int(b)/max(int(s),1):.0%})")
                if int(b) / max(int(s), 1) > 0.80:
                    print("  NOTE a gate blocking >80% cannot be moved to")
                    print("  'block' without near-halting that side.")
            else:
                print(f"  {len(hits)} lines, no running tally found")
        else:
            print("  not firing in this window")

    # ── stop floor, from the journal ──────────────────────────────────────
    print("\nSTOP FLOOR   goal: median sized stop >= 1.00% of price")
    print("             (the payoff ratio was 0.47 and needs 1.33)")
    try:
        cut = time.time() - args.hours * 3600
        rows = []
        for line in Path(args.trades).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            # A journal line that is not an object must not crash a
            # reporting tool — it has happened with partial writes.
            if not isinstance(t, dict):
                continue
            c = t.get("entry_context") or {}
            if not isinstance(c, dict):
                continue
            if not c.get("auto") or (t.get("opened_at") or 0) < cut:
                continue
            lev, stop = t.get("leverage"), c.get("sized_stop_roi")
            net = t.get("net_pnl_usdt")
            if lev and stop:
                rows.append((stop / lev, net))
        if rows:
            sp = [r[0] for r in rows]
            print(f"  n={len(rows)}   median {st.median(sp):.3f}% of price"
                  f"   under 1.00%: {sum(1 for v in sp if v < 0.999)/len(sp):.0%}")
            print(_verdict(st.median(sp) >= 0.999))
            W = [r[1] for r in rows if r[1] and r[1] > 0]
            L = [r[1] for r in rows if r[1] and r[1] <= 0]
            if W and L:
                ratio = abs(st.mean(W) / st.mean(L))
                print(f"  payoff ratio {ratio:.2f}   win {len(W)/len(rows):.0%}"
                      f"   break-even needs "
                      f"{abs(st.mean(L))/(st.mean(W)+abs(st.mean(L))):.0%}")
        else:
            print(f"  no bot trades in the last {args.hours:.0f}h")
    except FileNotFoundError:
        print(f"  {args.trades} not found")

    print("\n" + "=" * 62)
    print("A mechanism missing its OWN target is broken whatever P&L does.")
    print("One hitting its target has done its job — whether the goal was")
    print("worth having is a separate and much slower argument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
