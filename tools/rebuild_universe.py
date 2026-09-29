#!/usr/bin/env python3
"""
Reconstruct the scanner's candidate universe for a PAST moment — and prove
the reconstruction is faithful before trusting it.

    # validate against scans that actually happened (DO THIS FIRST)
    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/rebuild_universe.py --validate --samples 8

    # then, only then, rebuild an arbitrary past moment
    docker exec $(docker ps -q -f name=scalper-1) \
        python tools/rebuild_universe.py --at 2026-09-12T14:00:00Z

WHY THIS EXISTS
  The one finding that matters — shorts in a falling BTC, 65% win, p=0.0064 —
  rests on n=79, because BTC fell on only a handful of days this month. A
  backtest across a real bear stretch would turn that into thousands. History
  is the only way to get that sample.

WHY IT IS VALIDATION-FIRST
  The scanner picks from "top N movers by 24h change, above the p85 volume
  floor, across ~525 symbols". That needs the 24h TICKER SNAPSHOT as it stood
  at the time, and Binance does not serve historical ticker snapshots. This
  rebuilds both numbers from klines instead:

      24h quote volume  = sum of quote volume over the trailing 24h
      24h change        = close(T) / close(T-24h) - 1

  That is a RECONSTRUCTION, not a recording. If it cannot reproduce a week
  you have logs for, it cannot be trusted on a year you do not.

  `--validate` replays moments the bot actually scanned and compares the
  rebuilt candidate set against the symbols in shadow_decisions.jsonl. It
  reports overlap, misses and extras. **A miss rate above ~10% means the
  reconstruction is wrong and any backtest built on it is fiction.**

KNOWN SOURCES OF DRIFT, none of which this can fix
  SURVIVORSHIP   symbols delisted since are absent; symbols listed since
                 appear for periods when they did not exist. Both bias a
                 backtest toward instruments that survived.
  THE PERCENTILE The p85 floor is computed from the WHOLE universe, so a
                 partial or differently-composed fetch moves the floor and
                 changes which symbols pass — a second-order error that
                 compounds with survivorship.
  FUNDING        not modelled at all.
  THE TICKER     Binance's 24h ticker is a rolling window updated
                 continuously; a kline sum is bucketed. They disagree
                 slightly, most at the edges of the window.
"""

import argparse
import json
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MS = 1000
DAY_MS = 24 * 60 * 60 * MS


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_at(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()


def usdt_symbols(exchange) -> list:
    exchange.load_markets()
    out = []
    for sym, m in exchange.markets.items():
        if not m.get("swap") or not m.get("active"):
            continue
        if m.get("quote") != "USDT":
            continue
        out.append(sym)
    return sorted(out)


def window_stats(exchange, symbol: str, at_ts: float) -> tuple:
    """
    (quote_volume_24h, pct_change_24h) as of `at_ts`, from 1h klines.

    Uses the RAW endpoint, not fetch_ohlcv: ccxt's OHLCV drops quote volume
    and returns base volume only, which is a different number and not what
    the scanner's percentile floor is computed from.
    """
    since = int((at_ts - 25 * 3600) * MS)
    raw = exchange.fapiPublicGetKlines({
        "symbol": symbol.replace("/", "").replace(":USDT", ""),
        "interval": "1h", "startTime": since, "limit": 26})
    bars = [b for b in raw if int(b[0]) <= at_ts * MS]
    if len(bars) < 24:
        return None, None
    bars = bars[-24:]
    qv = sum(float(b[7]) for b in bars)          # index 7 = quote asset volume
    first_open, last_close = float(bars[0][1]), float(bars[-1][4])
    if first_open <= 0:
        return None, None
    return qv, (last_close / first_open - 1.0) * 100.0


def rebuild(exchange, at_ts: float, *, percentile: float, min_change: float,
            top_n: int, symbols=None, verbose=True) -> list:
    syms = symbols or usdt_symbols(exchange)
    rows = []
    for i, s in enumerate(syms, 1):
        if verbose and i % 50 == 0:
            print("  %d/%d ..." % (i, len(syms)), file=sys.stderr)
        try:
            qv, pct = window_stats(exchange, s, at_ts)
        except Exception:
            continue
        if qv is None:
            continue
        rows.append((s, qv, pct))
    if not rows:
        return []
    vols = sorted(r[1] for r in rows)
    k = (len(vols) - 1) * percentile / 100.0
    lo, hi = int(k), min(int(k) + 1, len(vols) - 1)
    floor = vols[lo] + (vols[hi] - vols[lo]) * (k - lo)
    kept = [r for r in rows if r[1] >= floor and abs(r[2]) >= min_change]
    kept.sort(key=lambda r: abs(r[2]), reverse=True)
    if verbose:
        print("  universe %d, p%g floor %.1fM, movers %d" % (
            len(rows), percentile, floor / 1e6, len(kept)), file=sys.stderr)
    return kept[:top_n]


def recorded_scans(path: Path, n: int) -> list:
    """
    (scan_ts, {symbols}) from the shadow decision log — the only record of
    WHICH symbols the scanner offered, as opposed to how many.
    """
    buckets = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ts, sym = r.get("ts"), r.get("symbol")
        if not ts or not sym:
            continue
        buckets.setdefault(int(ts // 120) * 120, set()).add(sym)
    full = [(t, s) for t, s in sorted(buckets.items()) if len(s) >= 5]
    if not full:
        return []
    step = max(1, len(full) // n)
    return full[::step][:n]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validate", action="store_true",
                   help="replay recorded scans and score the reconstruction")
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--at", help="UTC moment, e.g. 2026-09-12T14:00:00Z")
    p.add_argument("--decisions", default="logs/shadow_decisions.jsonl")
    p.add_argument("--percentile", type=float, default=85.0)
    p.add_argument("--min-change", type=float, default=8.0)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resolve_shadow_outcomes import _exchange
    ex = _exchange(args.demo)

    if args.validate:
        scans = recorded_scans(Path(args.decisions), args.samples)
        if not scans:
            print("no usable scans in %s" % args.decisions, file=sys.stderr)
            return 1
        print("Validating %d recorded scans. Rebuilt vs what the bot saw.\n"
              % len(scans))
        print("  %-22s %7s %7s %8s %8s" % ("scan", "actual", "rebuilt",
                                           "overlap", "missed"))
        misses = []
        for ts, actual in scans:
            got = {r[0] for r in rebuild(ex, ts, percentile=args.percentile,
                                         min_change=args.min_change,
                                         top_n=args.top_n, verbose=False)}
            inter = actual & got
            miss = (len(actual) - len(inter)) / max(len(actual), 1)
            misses.append(miss)
            print("  %-22s %7d %7d %8.0f%% %7.0f%%" % (
                _iso(ts), len(actual), len(got),
                100 * len(inter) / max(len(actual), 1), 100 * miss))
        m = st.median(misses)
        print("\n  median miss rate %.0f%%" % (100 * m))
        if m > 0.10:
            print("""
  ABOVE 10%% — THE RECONSTRUCTION IS NOT FAITHFUL.
  Any backtest built on it would be fiction. Likely causes, in order:
    - the 24h ticker is a ROLLING window; kline sums are bucketed
    - the percentile floor moves when the universe composition differs
    - symbols listed or delisted since shift both the floor and the ranking
  Do not proceed to --at until this is under 10%%.""")
        else:
            print("  Under 10%. The reconstruction tracks what the bot saw;\n"
                  "  survivorship and funding still apply to any backtest.")
        return 0

    if not args.at:
        print("give --at or --validate", file=sys.stderr)
        return 1
    ts = _parse_at(args.at)
    if ts > time.time() - DAY_MS / MS:
        print("--at must be at least 24h in the past (the window needs to be "
              "complete)", file=sys.stderr)
        return 1
    movers = rebuild(ex, ts, percentile=args.percentile,
                     min_change=args.min_change, top_n=args.top_n)
    print("%s — %d movers" % (_iso(ts), len(movers)))
    for s, qv, pct in movers:
        print("  %-22s %10.1fM %+8.2f%%" % (s, qv / 1e6, pct))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
