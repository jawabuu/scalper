"""
Forward returns for shadow decisions — the counterfactual the log lacks.

WHY THIS EXISTS
---------------
The shadow log records what jev and the bot each THOUGHT about a candidate.
It records nothing about what the market then did. With the bot refusing most
candidates (350 of 351 in the first real sample), almost every row is a
refusal carrying no outcome, so nothing can be learned from the bulk of the
data even in principle: agreement between two opinions is not evidence that
either was right.

Measured against forward returns instead, a refusal becomes a labelled
observation. The question stops being "did jev agree with the bot" and
becomes "when they refused, was refusing correct" — which is answerable, and
is what JEV-BRIEF.md's hypothesis actually needs.

OFF THE TRADE PATH, ENTIRELY
----------------------------
Nothing here runs in the trading process. It is a separate pass over a file
the bot has already written, and it cannot delay, gate or size anything. The
scanner row carries no price, and adding one would mean editing scanner.py —
on the trade path — so this fetches historical candles instead and derives
both the baseline and the horizons itself. Slower, and worth it.

APPEND-ONLY, SIDECAR
--------------------
Outcomes go to their own file, NEVER back into shadow_decisions.jsonl. That
log is append-only and written from a live process; rewriting rows in place
to add a column invites a torn write for no benefit. Join on (symbol, ts).

NOT INDEPENDENT BY DEFAULT
--------------------------
Before v3.71.0's dedup, the same candidate was judged repeatedly as the
scanner re-offered it — 1021 rows over 24 symbols in one sample. Labelling
each of those with its own forward return would manufacture confidence:
forty rows about one coin in one state share essentially one outcome. This
resolver therefore COLLAPSES near-duplicate rows before labelling, and
records how many it collapsed, so a reader can see the real sample size
rather than the row count.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_IN = "logs/shadow_decisions.jsonl"
DEFAULT_OUT = "logs/shadow_outcomes.jsonl"

# Minutes after the decision at which forward return is measured. Short
# because this is a scalper: a 4h return says nothing about a trade whose
# median hold is measured in candles.
HORIZONS_MIN = (15, 30, 60)

# Two rows for the same symbol inside this many seconds are treated as one
# observation. Independent of the live dedup window: this one repairs rows
# already written, including everything logged before dedup existed.
COLLAPSE_SEC = 900.0

# How stale a candle may be and still count as "the price at horizon H".
# Without a bound, a feed that stops early silently reports its LAST known
# price at every later horizon — 30m and 60m then read identically and the
# gap looks like a flat market instead of missing data.
MAX_STALENESS_S = 120.0


@dataclass
class ShadowOutcome:
    """One labelled observation. Joins to a decision row on (symbol, ts)."""
    symbol: str
    ts: float                  # ts of the FIRST row in the collapsed group
    side: str
    bot_decision: str
    jev_verdict: str
    confidence: float
    conviction: float
    looks_exhausted: float
    regime_aligned: float
    structure_intact: float
    composed_score: float
    base_price: float
    returns_pct: dict          # {"15": -0.4, "30": ...}; horizon -> % move
    favoured_side_pct: dict    # same, signed so + means the SIDE was right
    collapsed_rows: int        # how many decision rows this observation covers
    resolved_ts: float


def _load(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            log.warning(f"shadow outcomes: skipping unparseable line {i}: {e}")
    return rows


def collapse(rows: list, window: float = COLLAPSE_SEC) -> list:
    """
    Group near-duplicate decisions into one observation each.

    Keyed on (symbol, side) within a time window, keeping the EARLIEST row of
    each group — the first judgement is the one made without the benefit of
    the scanner having looked again.

    Rows with jev_verdict UNKNOWN are dropped: they carry no judgement, only
    a failed call. Everything written before v3.69.0 is such a row.
    """
    usable = [r for r in rows
              if r.get("jev_verdict") not in (None, "UNKNOWN")
              and r.get("ts") is not None]
    usable.sort(key=lambda r: r["ts"])
    groups: dict = {}
    order: list = []
    for r in usable:
        key = (r.get("symbol"), r.get("side"))
        g = groups.get(key)
        if g is not None and r["ts"] - g[-1]["ts"] <= window:
            g.append(r)
            continue
        if g is not None:
            order.append(groups.pop(key))
        groups[key] = [r]
        order.append(groups[key])
    # `order` holds live references, so late appends are already reflected.
    seen = set()
    out = []
    for g in order:
        if id(g) in seen:
            continue
        seen.add(id(g))
        out.append(g)
    return out


def _resolved_keys(path: Path) -> set:
    """(symbol, ts) already written, so a re-run is cheap and idempotent."""
    return {(r.get("symbol"), r.get("ts")) for r in _load(path)}


def resolve(exchange, in_path: str = DEFAULT_IN, out_path: str = DEFAULT_OUT,
            horizons=HORIZONS_MIN, collapse_sec: float = COLLAPSE_SEC,
            now: float | None = None) -> int:
    """
    Label every decision old enough to have an outcome. Returns rows written.

    `exchange` is any ccxt-style object with fetch_ohlcv(symbol, timeframe,
    since, limit). Injected rather than constructed so this is testable
    without a network and cannot accidentally share the trading client.

    Only groups whose LONGEST horizon has fully elapsed are resolved; the rest
    are left for a later run, so a re-run picks them up rather than writing a
    truncated observation.
    """
    now = time.time() if now is None else now
    src, dst = Path(in_path), Path(out_path)
    done = _resolved_keys(dst)
    longest = max(horizons)
    written = 0

    for group in collapse(_load(src), collapse_sec):
        head = group[0]
        key = (head.get("symbol"), head.get("ts"))
        if key in done:
            continue
        if now - head["ts"] < longest * 60:
            continue                      # not ripe; a later run gets it
        try:
            rec = _label(exchange, head, group, horizons, now)
        except Exception as e:
            log.warning(f"shadow outcomes: {head.get('symbol')} at "
                        f"{head.get('ts')} not resolved "
                        f"({type(e).__name__}: {e})")
            continue
        if rec is None:
            continue
        with open(dst, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        written += 1
    return written


def _label(exchange, head: dict, group: list, horizons, now: float):
    """One observation, or None when the candles do not cover it."""
    symbol = head["symbol"]
    ts = float(head["ts"])
    since = int((ts - 60) * 1000)
    limit = max(horizons) + 5
    candles = exchange.fetch_ohlcv(symbol, "1m", since, limit)
    if not candles:
        return None

    def close_at(target_s: float):
        # Last candle opening at or before the target, but only if it is
        # actually NEAR it. Never interpolates and never carries a stale
        # price forward: a missing minute reads as missing.
        best = None
        for c in candles:
            if c[0] / 1000.0 <= target_s:
                best = c
            else:
                break
        if best is None:
            return None
        if target_s - best[0] / 1000.0 > MAX_STALENESS_S:
            return None
        return float(best[4])

    base = close_at(ts)
    if not base:
        return None

    side = str(head.get("side") or "").lower()
    sign = -1.0 if side.startswith("short") else 1.0
    rets, favoured = {}, {}
    for h in horizons:
        px = close_at(ts + h * 60)
        if px is None:
            continue
        pct = round((px - base) / base * 100.0, 4)
        rets[str(h)] = pct
        # Signed by the side under consideration, so + always means the
        # direction was right. Without this a short's winning move reads
        # negative and every later average silently inverts.
        favoured[str(h)] = round(pct * sign, 4)
    if not rets:
        return None

    return ShadowOutcome(
        symbol=symbol, ts=ts, side=head.get("side", ""),
        bot_decision=head.get("bot_decision", ""),
        jev_verdict=head.get("jev_verdict", ""),
        confidence=float(head.get("confidence") or 0.0),
        conviction=float(head.get("conviction") or 0.0),
        looks_exhausted=float(head.get("looks_exhausted") or 0.0),
        regime_aligned=float(head.get("regime_aligned") or 0.0),
        structure_intact=float(head.get("structure_intact") or 0.0),
        composed_score=float(head.get("composed_score") or 0.0),
        base_price=base, returns_pct=rets, favoured_side_pct=favoured,
        collapsed_rows=len(group), resolved_ts=now)


def summarize(out_path: str = DEFAULT_OUT, horizon: int = 30) -> dict:
    """
    What the labelled data says. Deliberately plain: counts, medians, and the
    split by verdict. No significance testing — with samples this small it
    would dress up noise.
    """
    rows = _load(Path(out_path))
    h = str(horizon)
    vals = [(r, r.get("favoured_side_pct", {}).get(h)) for r in rows]
    vals = [(r, v) for r, v in vals if v is not None]
    if not vals:
        return {"observations": 0, "horizon_min": horizon}

    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        if not n:
            return None
        return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 4)

    out = {
        "observations": len(vals),
        "decision_rows_covered": sum(r.get("collapsed_rows", 1) for r, _ in vals),
        "horizon_min": horizon,
        "median_favoured_pct": med([v for _, v in vals]),
        "by_verdict": {},
        "by_bot_decision": {},
    }
    for field, bucket in (("jev_verdict", "by_verdict"),
                          ("bot_decision", "by_bot_decision")):
        seen = {}
        for r, v in vals:
            seen.setdefault(r.get(field, "?"), []).append(v)
        out[bucket] = {k: {"n": len(v), "median_favoured_pct": med(v)}
                       for k, v in sorted(seen.items())}
    return out
