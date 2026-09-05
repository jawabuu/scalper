"""
Candidate scanner — surfaces coins whose indicators suggest POTENTIAL setups.

This module LISTS candidates for the operator to review. It never trades, never
places an order, and never decides anything. Everything it produces is an
indicator of potential, not a signal — the operator makes the call.

Focus: top gainers and top losers. Quiet coins that have barely moved in 24h are
deliberately excluded — the strategy needs movers.

Candidate logic (both directions look for the EMA pair CONVERGING toward a
cross, i.e. the move is developing but has not yet completed):

  SHORT potential — fading an exhausted move:
    - 24h volume above the floor, and the coin is a big mover
    - RSI above short_rsi_min (overbought / exhausted)
    - EMA9 still ABOVE EMA21 but falling toward it (gap narrowing)
      -> approaching a bearish cross

  LONG potential:
    - same volume / mover filters
    - RSI above long_rsi_min
    - EMA9 still BELOW EMA21 but rising toward it (gap narrowing)
      -> approaching a bullish cross
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta


@dataclass
class ScanConfig:
    min_24h_vol_usdt: float = 50_000_000
    # "Movers only": ignore coins whose 24h change is smaller than this in
    # absolute terms. Top gainers AND top losers both qualify.
    min_abs_change_pct: float = 5.0
    short_rsi_min: float = 70.0
    long_rsi_min: float = 50.0
    # Upper bound for LONG candidates. Without it the long screen admits the
    # whole RSI range above the floor, so an overbought coin in a bearish
    # structure (arguably a short setup) surfaces as a long. Longs want
    # oversold-to-neutral and recovering, not already-hot.
    long_rsi_max: float = 65.0
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_len: int = 14
    # How many candles back to measure whether the EMA gap is narrowing.
    convergence_lookback: int = 3
    # Ignore a gap this small as "already crossed / too close to call".
    min_gap_pct: float = 0.01
    # The gap must NARROW by at least this much to count as converging. A coin
    # trending at a steady rate has an EMA gap that asymptotes to a constant,
    # drifting by rounding-level amounts (~0.002%); without this floor that
    # drift would be misread as convergence toward a cross.
    min_convergence_pct: float = 0.05


@dataclass
class Candidate:
    symbol: str
    direction: str          # "short" | "long"
    rsi: float
    ema_gap_pct: float      # (ema9 - ema21) / ema21 * 100
    gap_change_pct: float   # how much the gap narrowed over the lookback
    change_24h_pct: float
    volume_24h_usdt: float
    note: str               # human-readable why-it-surfaced
    # Where price sits in the 24h range: 0.0 = at the 24h low, 1.0 = at the high.
    # A SHORT near the high has resistance overhead and further to fall; a LONG
    # near the low has support beneath it. None when high/low are unavailable.
    range_pos_24h: float | None = None

    @property
    def range_quality(self) -> str:
        """
        How well the 24h range position supports this direction.
        Short wants price HIGH in the range; long wants it LOW.
        """
        if self.range_pos_24h is None:
            return "unknown"
        favourable = self.range_pos_24h if self.direction == "short" else (1 - self.range_pos_24h)
        if favourable >= 0.80:
            return "strong"
        if favourable >= 0.60:
            return "good"
        if favourable >= 0.40:
            return "neutral"
        return "weak"

    def as_row(self) -> dict:
        # Everything cast to native Python types — the API layer must never see
        # a numpy scalar (np.bool_ in particular is not JSON serialisable).
        return {
            "symbol": str(self.symbol),
            "direction": str(self.direction),
            "rsi": round(float(self.rsi), 1),
            "ema_gap_pct": round(float(self.ema_gap_pct), 3),
            "gap_narrowing_pct": round(float(self.gap_change_pct), 3),
            "change_24h_pct": round(float(self.change_24h_pct), 2),
            "volume_24h_usdt": round(float(self.volume_24h_usdt), 0),
            "note": str(self.note),
        }


def prepare(df: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    """Add the indicators the scanner needs."""
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=cfg.ema_fast)
    df["ema_slow"] = ta.ema(df["close"], length=cfg.ema_slow)
    df["rsi"] = ta.rsi(df["close"], length=cfg.rsi_len)
    return df


def ema_gap_pct(row) -> float:
    """Signed gap between the EMAs, as a % of the slow EMA.

    Positive = fast above slow (bullish structure).
    Negative = fast below slow (bearish structure).
    """
    if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]) or row["ema_slow"] == 0:
        return 0.0
    # float() is required, not cosmetic: numpy scalars leak into comparisons and
    # produce np.bool_, which is NOT a bool subclass and cannot be JSON encoded
    # (it surfaces as a 500 from the API).
    return float((row["ema_fast"] - row["ema_slow"]) / row["ema_slow"] * 100)


def gap_series(df: pd.DataFrame) -> pd.Series:
    return (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"] * 100


def is_converging(df: pd.DataFrame, cfg: ScanConfig) -> tuple[bool, float]:
    """
    Is the EMA gap NARROWING toward a cross?

    Compares the absolute gap now against `convergence_lookback` candles ago.
    Returns (narrowing, change_in_abs_gap) where a negative change means the
    gap shrank. The shrink must exceed min_convergence_pct to count — a coin
    trending at a steady rate has a gap that asymptotes and drifts by
    rounding-level amounts, which is not convergence toward a cross.
    """
    lb = cfg.convergence_lookback
    if len(df) < lb + 1:
        return False, 0.0
    gaps = gap_series(df)
    now, before = gaps.iloc[-1], gaps.iloc[-1 - lb]
    if pd.isna(now) or pd.isna(before):
        return False, 0.0
    change = float(abs(now) - abs(before))
    return bool(change <= -cfg.min_convergence_pct), change


def passes_market_filters(volume_24h_usdt: float, change_24h_pct: float,
                          cfg: ScanConfig) -> tuple[bool, str]:
    """Volume floor, and 'movers only' — quiet coins are excluded."""
    if volume_24h_usdt < cfg.min_24h_vol_usdt:
        return False, f"volume {volume_24h_usdt/1e6:.1f}M below {cfg.min_24h_vol_usdt/1e6:.0f}M floor"
    if abs(change_24h_pct) < cfg.min_abs_change_pct:
        return False, (f"only moved {change_24h_pct:+.2f}% in 24h "
                       f"(need +/-{cfg.min_abs_change_pct}%) — not a mover")
    return True, "mover"


def range_position_24h(price: float, high_24h: float | None,
                       low_24h: float | None) -> float | None:
    """Where `price` sits in the 24h range: 0.0 at the low, 1.0 at the high."""
    if high_24h is None or low_24h is None:
        return None
    rng = high_24h - low_24h
    if rng <= 0:
        return None
    return max(0.0, min(1.0, (price - low_24h) / rng))


def evaluate_symbol(symbol: str, df: pd.DataFrame, volume_24h_usdt: float,
                    change_24h_pct: float, cfg: ScanConfig,
                    high_24h: float | None = None,
                    low_24h: float | None = None) -> Candidate | None:
    """
    Assess one symbol. Returns a Candidate if it shows POTENTIAL, else None.

    This is a screening aid, not a trade signal.
    """
    ok, _why = passes_market_filters(volume_24h_usdt, change_24h_pct, cfg)
    if not ok:
        return None

    need = max(cfg.ema_slow, cfg.rsi_len, cfg.convergence_lookback + 1)
    if len(df) < need:
        return None

    df = prepare(df, cfg)
    row = df.iloc[-1]
    if pd.isna(row["rsi"]):
        return None

    rsi = float(row["rsi"])
    gap = ema_gap_pct(row)
    narrowing, gap_change = is_converging(df, cfg)
    rpos = range_position_24h(float(row["close"]), high_24h, low_24h)

    # NOTE: convergence is INFORMATIONAL (fade-early mode). RSI leads the screen;
    # the EMA gap and its narrowing are reported so the operator can judge how
    # far along the turn is, but they no longer gate the candidate.

    # SHORT potential: overbought inside a still-bullish structure -> fade it
    if gap > 0 and rsi >= cfg.short_rsi_min:
        return Candidate(
            symbol=symbol, direction="short", rsi=rsi, ema_gap_pct=gap,
            gap_change_pct=gap_change, change_24h_pct=change_24h_pct,
            volume_24h_usdt=volume_24h_usdt, range_pos_24h=rpos,
            note=(f"RSI {rsi:.0f} (>={cfg.short_rsi_min:.0f}) overbought; EMA9 {gap:+.2f}% "
                  f"above EMA21, gap {gap_change:+.2f}%"
                  + (" and converging" if narrowing else "")),
        )

    # LONG potential: recovering inside a still-bearish structure
    if gap < 0 and cfg.long_rsi_min <= rsi <= cfg.long_rsi_max:
        return Candidate(
            symbol=symbol, direction="long", rsi=rsi, ema_gap_pct=gap,
            gap_change_pct=gap_change, change_24h_pct=change_24h_pct,
            volume_24h_usdt=volume_24h_usdt, range_pos_24h=rpos,
            note=(f"RSI {rsi:.0f} (in {cfg.long_rsi_min:.0f}-{cfg.long_rsi_max:.0f}); EMA9 {gap:+.2f}% below "
                  f"EMA21, gap {gap_change:+.2f}%"
                  + (" and converging" if narrowing else "")),
        )

    return None


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """
    Order candidates by how close they are to a cross (smallest absolute gap
    first), since those are the most imminent. Purely a display convenience.
    """
    return sorted(candidates, key=lambda c: abs(c.ema_gap_pct))


def format_table(candidates: list[Candidate]) -> str:
    """Plain-text table for logs / CLI review."""
    if not candidates:
        return "No candidates met the criteria."
    hdr = (f"{'SYMBOL':<14}{'DIR':<7}{'RSI':>6}{'EMA GAP%':>10}"
           f"{'NARROW':>9}{'24H%':>9}{'VOL(M)':>10}")
    lines = [hdr, "-" * len(hdr)]
    for c in candidates:
        lines.append(
            f"{c.symbol:<14}{c.direction:<7}{c.rsi:>6.0f}{c.ema_gap_pct:>10.3f}"
            f"{c.gap_change_pct:>9.3f}{c.change_24h_pct:>+9.2f}"
            f"{c.volume_24h_usdt/1e6:>10.1f}"
        )
    lines.append("")
    lines.append("Indicators of POTENTIAL only — review each before acting.")
    return "\n".join(lines)


# ── Change tracking across refreshes ─────────────────────────────────────────
#
# A snapshot says a coin IS overbought. A delta says it is BECOMING more so —
# which is the actual signal. The tracker remembers the previous scan per symbol
# and reports what changed:
#
#   SHORT strengthens when RSI climbs further into overbought, and CONFIRMS when
#   EMA9 crosses DOWN through EMA21.
#   LONG strengthens when RSI climbs, and CONFIRMS when EMA9 crosses UP
#   through EMA21.

@dataclass
class Snapshot:
    rsi: float
    ema_gap_pct: float


@dataclass
class Delta:
    rsi_change: float = 0.0
    gap_change: float = 0.0
    crossed_down: bool = False   # EMA9 fell below EMA21 since last scan
    crossed_up: bool = False     # EMA9 rose above EMA21 since last scan
    is_new: bool = True          # not seen in the previous scan
    strength: str = "new"        # new | strengthening | weakening | CONFIRMED
    note: str = ""


class ScanTracker:
    """
    Holds the previous scan's markers per symbol so each refresh can report
    what MOVED. Purely observational — it never trades.
    """

    def __init__(self):
        self._prev: dict[str, Snapshot] = {}

    def diff(self, c: "Candidate") -> Delta:
        prev = self._prev.get(c.symbol)
        if prev is None:
            return Delta(is_new=True, strength="new",
                         note="first seen this session")

        d = Delta(
            rsi_change=c.rsi - prev.rsi,
            gap_change=c.ema_gap_pct - prev.ema_gap_pct,
            is_new=False,
        )
        # Cross detection: sign flip of the EMA gap between scans.
        d.crossed_down = bool(prev.ema_gap_pct > 0 >= c.ema_gap_pct)
        d.crossed_up = bool(prev.ema_gap_pct < 0 <= c.ema_gap_pct)

        if c.direction == "short":
            if d.crossed_down:
                d.strength = "CONFIRMED"
                d.note = (f"EMA9 crossed BELOW EMA21 (gap {prev.ema_gap_pct:+.2f} -> "
                          f"{c.ema_gap_pct:+.2f}); RSI {prev.rsi:.0f} -> {c.rsi:.0f}")
            elif d.rsi_change > 0:
                d.strength = "strengthening"
                d.note = (f"RSI rising {prev.rsi:.0f} -> {c.rsi:.0f} "
                          f"({d.rsi_change:+.1f}), deeper into overbought")
            else:
                d.strength = "weakening"
                d.note = f"RSI easing {prev.rsi:.0f} -> {c.rsi:.0f} ({d.rsi_change:+.1f})"
        else:  # long
            if d.crossed_up:
                d.strength = "CONFIRMED"
                d.note = (f"EMA9 crossed ABOVE EMA21 (gap {prev.ema_gap_pct:+.2f} -> "
                          f"{c.ema_gap_pct:+.2f}); RSI {prev.rsi:.0f} -> {c.rsi:.0f}")
            elif d.rsi_change > 0:
                d.strength = "strengthening"
                d.note = (f"RSI rising {prev.rsi:.0f} -> {c.rsi:.0f} "
                          f"({d.rsi_change:+.1f})")
            else:
                d.strength = "weakening"
                d.note = f"RSI easing {prev.rsi:.0f} -> {c.rsi:.0f} ({d.rsi_change:+.1f})"
        return d

    def commit(self, candidates: list["Candidate"]):
        """Record this scan's markers as the baseline for the next refresh."""
        for c in candidates:
            self._prev[c.symbol] = Snapshot(rsi=c.rsi, ema_gap_pct=c.ema_gap_pct)

    def annotate(self, candidates: list["Candidate"]) -> list[tuple["Candidate", Delta]]:
        """Pair each candidate with what changed since the last refresh."""
        return [(c, self.diff(c)) for c in candidates]


_STRENGTH_ORDER = {"CONFIRMED": 0, "strengthening": 1, "new": 2, "weakening": 3}
_RANGE_ORDER = {"strong": 0, "good": 1, "neutral": 2, "unknown": 3, "weak": 4}


def rank_with_deltas(pairs: list[tuple["Candidate", Delta]]) -> list[tuple["Candidate", Delta]]:
    """Most actionable first: confirmed crosses, then strengthening, then new."""
    return sorted(pairs, key=lambda p: (
        _STRENGTH_ORDER.get(p[1].strength, 9),
        _RANGE_ORDER.get(p[0].range_quality, 9),   # 24h-range support
        -abs(p[0].rsi - 50),
    ))


def format_table_with_deltas(pairs: list[tuple["Candidate", Delta]]) -> str:
    if not pairs:
        return "No candidates met the criteria."
    hdr = (f"{'SYMBOL':<14}{'DIR':<7}{'RSI':>6}{'dRSI':>7}{'EMA GAP%':>10}"
           f"{'24H%':>9}{'VOL(M)':>9}  {'STATUS':<14}")
    lines = [hdr, "-" * (len(hdr) + 30)]
    for c, d in pairs:
        drsi = "  —" if d.is_new else f"{d.rsi_change:+.1f}"
        lines.append(
            f"{c.symbol:<14}{c.direction:<7}{c.rsi:>6.0f}{drsi:>7}"
            f"{c.ema_gap_pct:>10.3f}{c.change_24h_pct:>+9.2f}"
            f"{c.volume_24h_usdt/1e6:>9.1f}  {d.strength:<14}{d.note}"
        )
    lines.append("")
    lines.append("Indicators of POTENTIAL only — review each before acting.")
    return "\n".join(lines)
