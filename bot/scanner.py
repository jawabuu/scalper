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

from dataclasses import dataclass, field

import pandas as pd
import pandas_ta as ta


@dataclass
class ScanConfig:
    min_24h_vol_usdt: float = 50_000_000
    # How the volume floor is applied:
    #   "absolute"   — vol must exceed min_24h_vol_usdt (correct on live)
    #   "percentile" — vol must be in the top (100 - vol_percentile)% of the
    #                  scanned universe. Scale-invariant, so it still
    #                  discriminates where reported volumes are inflated (demo
    #                  reports roughly 35x live, which makes an absolute floor
    #                  inert — every coin passes).
    volume_mode: str = "absolute"
    vol_percentile: float = 60.0
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
    # The EMA gap decides which side a coin is screened for, but a hard sign
    # test creates a blind spot at the crossover: a coin at RSI 85 whose EMA9
    # has JUST crossed below EMA21 is a confirmed roll-over — a better short
    # than one still climbing — yet it routed to the long side and failed the
    # long RSI band, vanishing from both screens. This tolerance lets a coin
    # still be screened for its direction while the gap sits just the other
    # side of zero. Expressed as a % of EMA21, same unit as the gap itself.
    ema_tolerance_pct: float = 0.15
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_len: int = 14
    atr_len: int = 14
    # How many recent candles to measure CURRENT velocity over. ATR(14) blends
    # in the quiet base that preceded a vertical move; measured on a STORJ-like
    # acceleration it reads ~1.6x low, and ~1.8x low on a blow-off. Over a
    # normal or quiet tape the two agree, so this only bites when it should.
    recent_tr_candles: int = 3
    # How many candles to scan for the taper comparison. candle_taper needs
    # FOUR candles moving with the trend inside this window before it can
    # compare the recent two against the previous two — at 6 it found them on
    # only 54 of 232 trades. Widening raises coverage without changing what is
    # compared; the cost is that the "previous two" can sit further back.
    taper_window: int = 10
    # Window for the U/V turn test, and how many candles the low must be
    # behind us before it counts as crossed rather than still forming.
    # How far back to look for the leg that created the setup.
    advance_lookback: int = 30
    # Window for the efficiency ratio (trend vs chop).
    er_lookback: int = 20
    turn_lookback: int = 10
    turn_min_bars_since: int = 2
    # How many candles to read body/wick structure over.
    shape_candles: int = 5
    # Half-width of the "at the extreme" band, as a % of the 24h high/low.
    # Was hardcoded at 0.1%. Both the STORJ and TA losses sat at 0.15% from
    # the high and so reported at_extreme=false, while AUTO_MAX_DIST_PCT was
    # recruiting candidates all the way out to 3.0%.
    extreme_band_pct: float = 0.1
    # Volatility band as ATR% of price. A coin below the floor barely moves —
    # the stop gets hit by noise before the move pays. Above the ceiling it
    # moves too erratically for a fixed-ROI stop to survive. 0 disables either.
    min_atr_pct: float = 0.0
    max_atr_pct: float = 0.0
    # The stop distance in PRICE % that the guardian will use, so the scanner
    # can express it in ATRs. At 10x, a -10% ROI stop is a 1.0% price move.
    stop_pct_for_ratio: float = 1.0
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
    # Distances to each 24h extreme, so the decision can be made from the
    # candidate list rather than only after a position is open.
    pct_above_24h_low: float | None = None
    pct_below_24h_high: float | None = None
    # Average True Range as a % of price — how much this coin actually moves
    # per candle. Used to judge whether a fixed stop distance is realistic.
    # Whether the EMA gap is NARROWING toward a cross. is_converging() has
    # always computed this; the long branch recorded the raw delta and threw
    # the verdict away, so "recovering" in its own comment was never tested.
    gap_narrowing: bool = False
    # Signed gap increasing: EMA9 GAINING on EMA21. Distinct from
    # gap_narrowing, which is absolute and cannot tell a bottom forming from a
    # top rolling over.
    gap_rising: bool = False
    # Volume across the advance that created the setup, measured over the whole
    # leg rather than the last two candles.
    advance: dict = field(default_factory=dict)
    # Trend-vs-chop: net displacement over total path travelled.
    efficiency: dict = field(default_factory=dict)
    gap_rise_pct: float = 0.0
    # Right side of a U or V: the fast EMA's low is behind us and it is rising
    # off it. Distinct from gap_narrowing, which is only a two-point shrink.
    turn: dict = field(default_factory=dict)
    atr_pct: float | None = None
    # Current velocity: mean true range of the last few candles, as % of price.
    # Distinct from atr_pct, which lags on an accelerating move.
    recent_tr_pct: float | None = None
    # How the recent candles are built: body vs wick share of the range.
    # Distinguishes a full-bodied run from a wick-heavy one of equal ATR.
    shape: dict = field(default_factory=dict)
    # Whether the pushes WITH the move are shrinking — the operator's visual
    # cue for timing a turn.
    taper: dict = field(default_factory=dict)
    # Stop distance expressed in ATRs: how many typical candle-ranges the stop
    # sits away. Below ~1 the stop is inside normal noise and likely to be hit
    # for reasons unrelated to the thesis.
    stop_vs_atr: float | None = None

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
    # Higher-timeframe EMA gap, resampled from the same candles. Recorded
    # only — nothing gates on it yet. Defaulted, so it must follow every
    # non-defaulted field in the dataclass.
    htf_trend_pct: float | None = None
    # Breakout structure components. Recorded only — nothing gates on it.
    breakout: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        # Everything cast to native Python types — the API layer must never see
        # a numpy scalar (np.bool_ in particular is not JSON serialisable).
        return {
            "symbol": str(self.symbol),
            "direction": str(self.direction),
            "rsi": round(float(self.rsi), 1),
            "ema_gap_pct": round(float(self.ema_gap_pct), 3),
            "htf_trend_pct": self.htf_trend_pct,
            "breakout": dict(self.breakout or {}),
            "gap_narrowing_pct": round(float(self.gap_change_pct), 3),
            "gap_narrowing": bool(self.gap_narrowing),
            "gap_rising": bool(self.gap_rising),
            "advance": dict(self.advance or {}),
            "efficiency": dict(self.efficiency or {}),
            "gap_rise_pct": round(float(self.gap_rise_pct or 0.0), 4),
            "turn": dict(self.turn or {}),
            "change_24h_pct": round(float(self.change_24h_pct), 2),
            "volume_24h_usdt": round(float(self.volume_24h_usdt), 0),
            # 24h range context. These were dropped when as_row() was rewritten
            # for numpy casting, which is why the range column read "n/a" — the
            # API simply never sent them.
            "range_pos_24h": (None if self.range_pos_24h is None
                              else round(float(self.range_pos_24h), 3)),
            "range_quality": str(self.range_quality),
            "pct_above_24h_low": (None if self.pct_above_24h_low is None
                                  else round(float(self.pct_above_24h_low), 2)),
            "pct_below_24h_high": (None if self.pct_below_24h_high is None
                                   else round(float(self.pct_below_24h_high), 2)),
            "atr_pct": None if self.atr_pct is None else round(float(self.atr_pct), 3),
            "recent_tr_pct": (None if self.recent_tr_pct is None
                              else round(float(self.recent_tr_pct), 3)),
            "shape": dict(self.shape or {}),
            "taper": dict(self.taper or {}),
            "stop_vs_atr": (None if self.stop_vs_atr is None
                            else round(float(self.stop_vs_atr), 2)),
            "note": str(self.note),
        }


def prepare(df: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    """Add the indicators the scanner needs."""
    df = df.copy()
    df["ema_fast"] = ta.ema(df["close"], length=cfg.ema_fast)
    df["ema_slow"] = ta.ema(df["close"], length=cfg.ema_slow)
    df["rsi"] = ta.rsi(df["close"], length=cfg.rsi_len)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=cfg.atr_len)
    # True range per candle, so current velocity can be read without waiting
    # for a 14-period average to catch up.
    prev_close = df["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return df


def recent_tr_pct(df: pd.DataFrame, candles: int = 3) -> float | None:
    """
    Mean true range of the last `candles` candles, as a % of price.

    This is the CURRENT velocity. ATR(14) is the recent average, which is a
    different thing on an accelerating move and the difference is the whole
    point: a callback floored on ATR(14) can be noise-width on a coin that is
    covering several ATRs per candle.
    """
    try:
        if df is None or "tr" not in df or len(df) < 1:
            return None
        px = float(df["close"].iloc[-1])
        if px <= 0:
            return None
        v = df["tr"].tail(max(1, int(candles))).mean()
        if pd.isna(v):
            return None
        return float(v) / px * 100
    except Exception:
        return None


def candle_shape(df, candles: int = 5) -> dict:
    """
    How the recent candles are BUILT, not how big they are.

    ATR collapses a candle to one number, so a full-bodied vertical run and a
    wick-heavy chop of the same range read identically. They are opposite
    situations for a fade: a full body is price being carried, a long wick is
    price being rejected.

    body_pct        mean |close-open| / (high-low), as %. High = directional.
    upper_wick_pct  mean upper wick / range, as %. For a SHORT this is
                    rejection overhead — the thing being faded.
    lower_wick_pct  the mirror, for a LONG.

    Measurement only — nothing gates on this.
    """
    out = {"body_pct": None, "upper_wick_pct": None, "lower_wick_pct": None}
    try:
        if df is None or len(df) < 1:
            return out
        tail = df.tail(max(1, int(candles)))
        bodies, uppers, lowers = [], [], []
        for _, r in tail.iterrows():
            hi, lo = float(r["high"]), float(r["low"])
            op, cl = float(r["open"]), float(r["close"])
            rng = hi - lo
            if rng <= 0:
                continue
            bodies.append(abs(cl - op) / rng * 100)
            uppers.append((hi - max(op, cl)) / rng * 100)
            lowers.append((min(op, cl) - lo) / rng * 100)
        if not bodies:
            return out
        out["body_pct"] = round(sum(bodies) / len(bodies), 1)
        out["upper_wick_pct"] = round(sum(uppers) / len(uppers), 1)
        out["lower_wick_pct"] = round(sum(lowers) / len(lowers), 1)
        return out
    except Exception:
        return out


def candle_taper(df, direction: str, window: int = 6) -> dict:
    """
    Is the move RUNNING OUT rather than still running?

    The operator's visual rule: timing a short, the GREEN candles get smaller;
    timing a long, the RED candles get smaller. The move keeps going but each
    push covers less ground than the last.

    This looks only at candles moving WITH the trend being faded — green for a
    short, red for a long — because a countertrend candle in the middle of a run
    says nothing about whether the pushes are weakening. It compares the mean
    body of the most recent two against the two before them.

    taper_ratio  < 1 means the pushes are shrinking (exhaustion, the fade is
                 timely); > 1 means they are growing (still expanding, which is
                 the shape STORJ, TA and GRIFFAIN all had).
    vol_ratio    the same comparison on volume. A new extreme made on less
                 volume than the push before it is the classic version of this.
    close_pos    where the trend candles CLOSE within their own range, 0-1.
                 For a short, green candles closing nearer their low means
                 buyers are failing to hold the push.

    Measurement only — nothing gates on this.
    """
    out = {"taper_ratio": None, "tapering": None, "trend_candles": 0,
           "vol_ratio": None, "close_pos": None}
    try:
        if df is None or len(df) < 4:
            return out
        want_green = direction == "short"
        rows = []
        for _, r in df.tail(max(4, int(window))).iterrows():
            op, cl = float(r["open"]), float(r["close"])
            is_green = cl > op
            if is_green != want_green:
                continue
            hi, lo = float(r["high"]), float(r["low"])
            rng = hi - lo
            rows.append({
                "body": abs(cl - op),
                "vol": float(r.get("volume") or 0.0),
                # For a short: how near the LOW did this green candle close?
                # 0 = closed at its low (push rejected), 1 = closed at its high.
                "close_pos": ((cl - lo) / rng if rng > 0 else 0.5) if want_green
                             else ((hi - cl) / rng if rng > 0 else 0.5),
            })
        out["trend_candles"] = len(rows)
        if len(rows) < 4:
            return out           # not enough pushes to compare

        recent, earlier = rows[-2:], rows[-4:-2]
        eb = sum(r["body"] for r in earlier) / 2
        rb = sum(r["body"] for r in recent) / 2
        if eb > 0:
            out["taper_ratio"] = round(rb / eb, 3)
            out["tapering"] = bool(rb < eb)
        ev = sum(r["vol"] for r in earlier) / 2
        rv = sum(r["vol"] for r in recent) / 2
        if ev > 0:
            out["vol_ratio"] = round(rv / ev, 3)
        out["close_pos"] = round(
            sum(r["close_pos"] for r in recent) / len(recent), 3)
        return out
    except Exception:
        return out


def breakout_structure(df, high_24h, low_24h, direction: str,
                       gap_pct: float, gap_change: float,
                       green_needed: int = 3,
                       min_gap_pct: float = 0.5,
                       extreme_band_pct: float = 0.1) -> dict:
    """
    Is this a BREAKOUT rather than an exhaustion?

    A wide EMA gap on its own is ambiguous — it fits a blow-off top as well as
    a breakout. What distinguishes them is whether the move is still
    EXPANDING: price making a new extreme, consecutive candles carrying it
    there, and the gap widening rather than rolling over.

    Returns the components as well as the verdict, so a split can show which
    part carries the information rather than only the combination.

    Recorded only — nothing acts on this.
    """
    out = {"at_extreme": False, "consecutive": 0, "gap_wide": False,
           "gap_widening": False, "breakout": False}
    try:
        if df is None or len(df) < green_needed + 1:
            return out
        closes = df["close"].astype(float)
        opens = df["open"].astype(float)
        price = float(closes.iloc[-1])

        # At the extreme in the direction of the move: a NEW high for an
        # upside breakout, a new low for a downside one.
        band = max(0.0, float(extreme_band_pct)) / 100.0
        if direction == "short" and high_24h:
            out["at_extreme"] = price >= float(high_24h) * (1.0 - band)
        elif direction == "long" and low_24h:
            out["at_extreme"] = price <= float(low_24h) * (1.0 + band)

        # Consecutive candles carrying the move. For an upside breakout that
        # means green closes; downside, red.
        want_green = (direction == "short")
        run = 0
        for i in range(1, green_needed + 1):
            c, o = float(closes.iloc[-i]), float(opens.iloc[-i])
            if (c > o) if want_green else (c < o):
                run += 1
            else:
                break
        out["consecutive"] = run

        out["gap_wide"] = abs(float(gap_pct)) >= min_gap_pct
        # gap_change is the change in the ABSOLUTE gap: negative means it
        # narrowed, positive means it widened.
        out["gap_widening"] = float(gap_change) > 0

        out["breakout"] = bool(out["at_extreme"]
                               and run >= green_needed
                               and out["gap_wide"]
                               and out["gap_widening"])
    except Exception:
        pass
    return out


def htf_trend(df, factor: int = 20) -> float | None:
    """
    The coin's HIGHER-TIMEFRAME trend, as a signed EMA gap percentage.

    Resampled from the candles already fetched for the 24h range, so it costs
    no extra requests. On a 3m chart a factor of 20 gives 1-hour bars.

    Answers whether the 3-minute signal is aligned with the hourly trend or
    fighting it — a short taken while the hour is still rising is a different
    proposition from one taken while the hour is falling, and nothing in the
    system currently distinguishes them.
    """
    try:
        # 24h on a 3m chart is 480 candles, and requiring factor*25 = 500 made
        # this return None on every single call — the field was recorded but
        # always empty, so the alignment table never populated. The real
        # requirement is enough resampled bars for a 21-period EMA, which the
        # len(htf) check below enforces.
        if df is None or len(df) < factor * 22:
            return None
        closes = df["close"].astype(float)
        # Take every Nth close: a cheap resample that needs no date index.
        htf = closes.iloc[::factor].reset_index(drop=True)
        if len(htf) < 22:
            return None
        fast = htf.ewm(span=9, adjust=False).mean().iloc[-1]
        slow = htf.ewm(span=21, adjust=False).mean().iloc[-1]
        if not slow:
            return None
        return round(float((fast - slow) / slow * 100), 3)
    except Exception:
        return None


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


def efficiency_ratio(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Is the move GOING somewhere, or just moving?

    Kaufman's efficiency ratio: net displacement over the sum of absolute
    candle-to-candle moves.

        near 1.0   every candle pushed the same way — a trend
        near 0.0   large candles cancelling out — chop

    A fade wants a LOW ratio: plenty of movement that goes nowhere, so a
    stretched price snaps back. A high ratio is a trend, and fading a trend
    catches the pullback and pays for the resumption.

    AKE/USDT 2026-09-16 ran 0.0165 -> 0.0283 (+68.57%) in a day with large
    alternating candles. Seven shorts were taken against it for -$94.98. The
    alternating candles look like chop at 3-minute resolution; the ratio says
    otherwise, which is the distinction no other recorded measure makes.

    `direction` carries the sign of the net move, so a high ratio can be read
    as "trending WITH the fade" or "trending AGAINST it".
    """
    out = {"efficiency": None, "er_bars": None, "er_direction": None}
    try:
        if df is None or "close" not in df or len(df) < 6:
            return out
        closes = df["close"].astype(float).tail(max(6, int(lookback)))
        if len(closes) < 6:
            return out
        net = float(closes.iloc[-1]) - float(closes.iloc[0])
        path = float(closes.diff().abs().sum())
        if path <= 0:
            return out
        out["efficiency"] = round(abs(net) / path, 3)
        out["er_bars"] = len(closes)
        out["er_direction"] = "up" if net > 0 else ("down" if net < 0 else "flat")
        return out
    except Exception:
        return out


def advance_volume(df: pd.DataFrame, direction: str = "short",
                   lookback: int = 30) -> dict:
    """
    Volume across the ADVANCE THAT CREATED THE SETUP — not the last two candles.

    taper_vol_ratio compares the most recent two trend candles against the two
    before them. That is a local burst reading taken at the moment of
    evaluation, and it answered a different question from the one asked:
    whether volume is DIMINISHING as price climbs into RSI 75. That is a
    property of the whole leg, measured BEFORE the candidate becomes eligible.

    For a SHORT the leg runs from the lowest low in the window up to the
    highest high. Its volume is split in half and compared:

        vol_trend < 1   volume falling as price rose — classic bearish
                        divergence, the exhaustion the fade is looking for
        vol_trend > 1   volume building into the high — participation, which
                        reads as a breakout rather than a top

    `peak_vol_early` is the stricter textbook shape: the heaviest bar of the
    advance sat in the FIRST half, so the later highs were made on thinner
    trade. Longs are the mirror — the decline into the low.

    Measurement only. Nothing gates on this.
    """
    out = {"adv_bars": None, "adv_vol_trend": None, "adv_price_pct": None,
           "peak_vol_early": None}
    try:
        if df is None or len(df) < 8 or "volume" not in df:
            return out
        win = df.tail(max(8, int(lookback))).reset_index(drop=True)
        short = direction == "short"
        # The leg: for a short, trough -> peak. For a long, peak -> trough.
        if short:
            i_start = int(win["low"].idxmin())
            i_end = int(win["high"].iloc[i_start:].idxmax()) if i_start < len(win)-1 else len(win)-1
        else:
            i_start = int(win["high"].idxmax())
            i_end = int(win["low"].iloc[i_start:].idxmin()) if i_start < len(win)-1 else len(win)-1
        leg = win.iloc[i_start:i_end + 1]
        if len(leg) < 4:
            return out
        out["adv_bars"] = len(leg)
        p0 = float(leg["low"].iloc[0] if short else leg["high"].iloc[0])
        p1 = float(leg["high"].iloc[-1] if short else leg["low"].iloc[-1])
        out["adv_price_pct"] = round((p1 - p0) / p0 * 100, 3) if p0 else None
        half = len(leg) // 2
        first = leg["volume"].iloc[:half].mean()
        second = leg["volume"].iloc[half:].mean()
        if first and first > 0:
            out["adv_vol_trend"] = round(float(second) / float(first), 3)
        out["peak_vol_early"] = bool(int(leg["volume"].reset_index(drop=True)
                                         .idxmax()) < half)
        return out
    except Exception:
        return out


def gap_rising(df: pd.DataFrame, cfg: ScanConfig) -> tuple[bool, float]:
    """
    Is EMA9 GAINING on EMA21 — the SIGNED gap increasing?

    is_converging() compares ABSOLUTE gaps, so it cannot tell these apart:

        -0.30% -> -0.05%   EMA9 rising toward EMA21 from underneath
        +0.10% -> +0.02%   EMA9 FALLING toward EMA21 from above

    Both shrink in absolute terms and both read as "narrowing". They are
    opposite situations: the first is a bottom forming, the second is a top
    rolling over. LAB at 2026-09-14 07:52 was the second — EMA9 0.05299 above
    EMA21 0.05296, yellow curving down toward pink — and the long gate let it
    through.

    The signed test needs no sign restriction and no threshold. A crossover in
    progress (-0.02 -> +0.02) is rising and passes, which is correct: the
    operator's objection is to converging FROM THE TOP, not to a cross.

    Returns (rising, signed_change).
    """
    try:
        gaps = gap_series(df)
        lb = max(1, int(cfg.convergence_lookback))
        if len(gaps) < lb + 1:
            return False, 0.0
        now = float(gaps.iloc[-1])
        before = float(gaps.iloc[-1 - lb])
        change = now - before
        return bool(change > float(cfg.min_convergence_pct)), round(change, 4)
    except Exception:
        return False, 0.0


def turned(df: pd.DataFrame, cfg: ScanConfig, direction: str = "long") -> dict:
    """
    Is this the RIGHT side of a U or a V — has the bottom been crossed?

    is_converging() asks only whether the gap is smaller than it was N candles
    ago. That is a two-point comparison: a collapsing market can post a smaller
    absolute gap after one pause and read as "narrowing" while EMA9 sits far
    below EMA21 and is still falling. USELESS at 18:06 was exactly that, and it
    lost 36% in ten seconds.

    What actually distinguishes a turn is that the LOW IS BEHIND US:

      * the fast EMA's minimum within the window is at least
        `turn_min_bars_since` candles back — a bottom still forming is not a
        bottom crossed
      * the fast EMA now sits above that minimum — it is rising off it, not
        flat along it

    This needs no magnitude threshold. "How close is close enough" would be a
    number invented from a couple of charts; "the low is behind us and we are
    above it" is a structural fact about the series.

    Returns bars_since_low and rise_pct alongside the verdict so the strictness
    can be judged from recorded data rather than argued about.
    """
    out = {"turned_up": False, "bars_since_low": None, "rise_pct": None}
    try:
        lb = max(3, int(cfg.turn_lookback))
        if df is None or "ema_fast" not in df or len(df) < lb:
            return out
        fast = df["ema_fast"].tail(lb).reset_index(drop=True)
        if fast.isna().any():
            return out
        # A LONG wants the right side of a U: the LOW behind it, rising off it.
        # A SHORT wants the mirror — the HIGH behind it, falling away. Computing
        # one verdict for both directions would record, on every short, whether
        # the EMA turned UP: meaningless for a fade, and the opposite of what
        # the field name implies.
        if direction == "short":
            i_ext = int(fast.idxmax())
        else:
            i_ext = int(fast.idxmin())
        bars_since = len(fast) - 1 - i_ext
        ext = float(fast.iloc[i_ext])
        now = float(fast.iloc[-1])
        out["bars_since_low"] = bars_since     # bars since the extreme
        out["rise_pct"] = (round((now - ext) / ext * 100, 3)
                           if ext else None)   # signed move off it
        if bars_since < int(cfg.turn_min_bars_since):
            return out                         # the extreme is still forming
        out["turned_up"] = bool(now < ext) if direction == "short" \
            else bool(now > ext)
        return out
    except Exception:
        return out


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
                    low_24h: float | None = None,
                    why: list | None = None) -> Candidate | None:
    """
    Assess one symbol. Returns a Candidate if it shows POTENTIAL, else None.

    `why` is an optional sink: on rejection, the reason is appended to it.
    A dozen `return None` paths were indistinguishable from outside, so a
    cross-instance comparison could only report "failed the RSI/EMA screen"
    for all of them. Live converts 28% of movers to candidates against demo's
    48%, and naming the gate is the only way to find out which one.

    This is a screening aid, not a trade signal.
    """
    def _no(reason: str):
        if why is not None:
            why.append(reason)
        return None

    ok, mkt_why = passes_market_filters(volume_24h_usdt, change_24h_pct, cfg)
    if not ok:
        return _no(f"market filter: {mkt_why}")

    need = max(cfg.ema_slow, cfg.rsi_len, cfg.convergence_lookback + 1)
    if len(df) < need:
        return _no(f"only {len(df)} candles, needs {need} "
                   f"(recently listed?)")

    df = prepare(df, cfg)
    row = df.iloc[-1]
    if pd.isna(row["rsi"]):
        return _no("RSI not computable on the latest candle")

    rsi = float(row["rsi"])
    gap = ema_gap_pct(row)
    narrowing, gap_change = is_converging(df, cfg)
    htf = htf_trend(df)
    close_px = float(row["close"])
    rtr = recent_tr_pct(df, cfg.recent_tr_candles)
    shp = candle_shape(df, cfg.shape_candles)
    tap_short = candle_taper(df, "short", cfg.taper_window)
    tap_long = candle_taper(df, "long", cfg.taper_window)
    turn_short = turned(df, cfg, "short")
    turn_long = turned(df, cfg, "long")
    rising, rise_change = gap_rising(df, cfg)
    eff = efficiency_ratio(df, cfg.er_lookback)
    adv_short = advance_volume(df, "short", cfg.advance_lookback)
    adv_long = advance_volume(df, "long", cfg.advance_lookback)
    atr_pct = None
    if "atr" in row and not pd.isna(row["atr"]) and close_px > 0:
        atr_pct = float(row["atr"]) / close_px * 100
    # Volatility band. Too quiet and the stop is hit by noise before the move
    # pays; too wild and a fixed-ROI stop cannot survive normal swings.
    if atr_pct is not None:
        if cfg.min_atr_pct and atr_pct < cfg.min_atr_pct:
            return _no(f"ATR {atr_pct:.3f}% below SCAN_MIN_ATR_PCT "
                       f"{cfg.min_atr_pct}")
        if cfg.max_atr_pct and atr_pct > cfg.max_atr_pct:
            return _no(f"ATR {atr_pct:.3f}% above SCAN_MAX_ATR_PCT "
                       f"{cfg.max_atr_pct}")
    rpos = range_position_24h(close_px, high_24h, low_24h)
    above_low = below_high = None
    if high_24h and low_24h and float(high_24h) > float(low_24h):
        above_low = (close_px - float(low_24h)) / float(low_24h) * 100
        below_high = (close_px - float(high_24h)) / float(high_24h) * 100

    # NOTE: convergence is INFORMATIONAL (fade-early mode). RSI leads the screen;
    # the EMA gap and its narrowing are reported so the operator can judge how
    # far along the turn is, but they no longer gate the candidate.

    # SHORT potential: overbought, with EMA9 above EMA21 OR just below it.
    # Checked before the long branch: within the tolerance band both could
    # match, and a high RSI is the stronger signal about direction.
    if gap > -cfg.ema_tolerance_pct and rsi >= cfg.short_rsi_min:
        return Candidate(
            symbol=symbol, direction="short", rsi=rsi, ema_gap_pct=gap,
            htf_trend_pct=htf,
            breakout=breakout_structure(df, high_24h, low_24h, "short",
                                        gap, gap_change,
                                        extreme_band_pct=cfg.extreme_band_pct),
            gap_change_pct=gap_change, gap_narrowing=narrowing, gap_rising=rising,
            gap_rise_pct=rise_change, turn=turn_short, advance=adv_short, efficiency=eff, change_24h_pct=change_24h_pct,
            volume_24h_usdt=volume_24h_usdt, range_pos_24h=rpos,
            pct_above_24h_low=above_low, pct_below_24h_high=below_high,
            atr_pct=atr_pct, recent_tr_pct=rtr, shape=shp, taper=tap_short,
            stop_vs_atr=(None if not atr_pct else
                         (cfg.stop_pct_for_ratio / atr_pct) if cfg.stop_pct_for_ratio else None),
            note=(("just crossed down — " if gap < 0 else "")
                  + f"RSI {rsi:.0f} (>={cfg.short_rsi_min:.0f}) overbought; EMA9 {gap:+.2f}% "
                  f"above EMA21, gap {gap_change:+.2f}%"
                  + (" and converging" if narrowing else "")),
        )

    # LONG potential: recovering, with EMA9 below EMA21 OR just above it.
    if gap < cfg.ema_tolerance_pct and cfg.long_rsi_min <= rsi <= cfg.long_rsi_max:
        return Candidate(
            symbol=symbol, direction="long", rsi=rsi, ema_gap_pct=gap,
            htf_trend_pct=htf,
            breakout=breakout_structure(df, high_24h, low_24h, "long",
                                        gap, gap_change,
                                        extreme_band_pct=cfg.extreme_band_pct),
            gap_change_pct=gap_change, gap_narrowing=narrowing, gap_rising=rising,
            gap_rise_pct=rise_change, turn=turn_long, advance=adv_long, efficiency=eff, change_24h_pct=change_24h_pct,
            volume_24h_usdt=volume_24h_usdt, range_pos_24h=rpos,
            pct_above_24h_low=above_low, pct_below_24h_high=below_high,
            atr_pct=atr_pct, recent_tr_pct=rtr, shape=shp, taper=tap_long,
            stop_vs_atr=(None if not atr_pct else
                         (cfg.stop_pct_for_ratio / atr_pct) if cfg.stop_pct_for_ratio else None),
            note=(("just crossed up — " if gap > 0 else "")
                  + f"RSI {rsi:.0f} (in {cfg.long_rsi_min:.0f}-{cfg.long_rsi_max:.0f}); EMA9 {gap:+.2f}% below "
                  f"EMA21, gap {gap_change:+.2f}%"
                  + (" and converging" if narrowing else "")),
        )

    # Neither branch matched: name the band it missed, with the actual
    # numbers, because this is the fallthrough that most rejections take.
    return _no(
        f"RSI {rsi:.1f} with EMA gap {gap:+.3f}% fits neither band — "
        f"short needs RSI>={cfg.short_rsi_min:.0f} and gap>"
        f"{-cfg.ema_tolerance_pct:+.2f}%; long needs "
        f"{cfg.long_rsi_min:.0f}<=RSI<={cfg.long_rsi_max:.0f} and gap<"
        f"{cfg.ema_tolerance_pct:+.2f}%")


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


def balance_directions(pairs: list[tuple["Candidate", "Delta"]],
                       max_share: float = 0.70) -> list[tuple["Candidate", "Delta"]]:
    """
    Keep both directions visible.

    Ranking alone can fill the whole list with one direction on a strongly
    trending day, hiding the handful of opposite-side setups. This caps either
    direction at `max_share` of the displayed rows WHENEVER the other direction
    has candidates — if one side has none, the other keeps the full list.

    Order within each direction is preserved, so the best candidates still lead.
    """
    if not pairs:
        return pairs
    longs = [p for p in pairs if p[0].direction == "long"]
    shorts = [p for p in pairs if p[0].direction == "short"]
    if not longs or not shorts:
        return pairs            # only one kind exists — show them all

    total = len(pairs)
    cap = max(1, int(total * max_share))
    kept_long = longs[:cap]
    kept_short = shorts[:cap]
    # Re-interleave by the original ranking so the strongest still sort first.
    keep = set(id(p) for p in kept_long + kept_short)
    return [p for p in pairs if id(p) in keep]


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
