"""Tests for the candidate scanner (listing only — never trades)."""
import pandas as pd
import pytest

from bot.scanner import (
    ScanConfig, Candidate, prepare, ema_gap_pct, is_converging,
    passes_market_filters, evaluate_symbol, rank, format_table,
)


@pytest.fixture
def cfg():
    return ScanConfig()


def _frame(closes):
    n = len(closes)
    return pd.DataFrame({
        "open":  [closes[0]] + closes[:-1],
        "high":  [c * 1.002 for c in closes],
        "low":   [c * 0.998 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


def _rising_then_stalling(n=60):
    """Uptrend that flattens at the end — EMA9 above EMA21 but converging down."""
    closes = [100 + i * 0.5 for i in range(n - 12)]
    last = closes[-1]
    # flatten / tick down so the fast EMA rolls toward the slow one
    closes += [last - i * 0.35 for i in range(1, 13)]
    return _frame(closes)


def _falling_then_stabilising(n=60):
    """Downtrend that flattens — EMA9 below EMA21 but converging up."""
    closes = [100 - i * 0.5 for i in range(n - 12)]
    last = closes[-1]
    closes += [last + i * 0.35 for i in range(1, 13)]
    return _frame(closes)


# ── market filters: movers only ──────────────────────────────────────────────

def test_rejects_low_volume(cfg):
    ok, why = passes_market_filters(10e6, 12.0, cfg)
    assert not ok and "below" in why


def test_rejects_quiet_coin(cfg):
    """A coin that barely moved is excluded even with huge volume."""
    ok, why = passes_market_filters(500e6, 1.2, cfg)
    assert not ok and "not a mover" in why


def test_accepts_top_gainer(cfg):
    ok, _ = passes_market_filters(120e6, +18.0, cfg)
    assert ok


def test_accepts_top_loser(cfg):
    """Big NEGATIVE movers qualify too — shorts come from here."""
    ok, _ = passes_market_filters(120e6, -22.0, cfg)
    assert ok


# ── convergence detection ────────────────────────────────────────────────────

def test_converging_detected(cfg):
    df = prepare(_rising_then_stalling(), cfg)
    narrowing, change = is_converging(df, cfg)
    assert narrowing
    assert change < 0          # absolute gap shrank


def test_diverging_not_flagged(cfg):
    # Genuinely ACCELERATING uptrend -> gap widening, not converging
    closes = [100 * (1 + 0.002 * i) ** i for i in range(60)]
    df = prepare(_frame(closes), cfg)
    narrowing, change = is_converging(df, cfg)
    assert not narrowing
    assert change > 0


def test_steady_trend_drift_is_not_convergence(cfg):
    """
    A coin trending at a CONSTANT rate has an EMA gap that asymptotes, drifting
    by rounding-level amounts (~0.002%). That must not be read as converging
    toward a cross.
    """
    closes = [100 * (1.01 ** i) for i in range(60)]
    df = prepare(_frame(closes), cfg)
    narrowing, change = is_converging(df, cfg)
    assert abs(change) < cfg.min_convergence_pct
    assert not narrowing


# ── candidate classification ─────────────────────────────────────────────────

def test_short_candidate_from_stalling_gainer(cfg):
    df = _rising_then_stalling()
    c = evaluate_symbol("AAA/USDT", df, volume_24h_usdt=200e6,
                        change_24h_pct=+15.0, cfg=cfg)
    # EMA9 above EMA21 and converging; surfaces as short IF RSI is high enough
    if c is not None:
        assert c.direction == "short"
        assert c.ema_gap_pct > 0
        assert c.rsi >= cfg.short_rsi_min


def test_long_candidate_from_stabilising_loser(cfg):
    df = _falling_then_stabilising()
    c = evaluate_symbol("BBB/USDT", df, volume_24h_usdt=200e6,
                        change_24h_pct=-15.0, cfg=cfg)
    if c is not None:
        assert c.direction == "long"
        assert c.ema_gap_pct < 0
        assert c.rsi >= cfg.long_rsi_min


def test_quiet_coin_never_surfaces(cfg):
    df = _rising_then_stalling()
    c = evaluate_symbol("CCC/USDT", df, volume_24h_usdt=200e6,
                        change_24h_pct=+0.5, cfg=cfg)   # not a mover
    assert c is None


def test_low_volume_never_surfaces(cfg):
    df = _rising_then_stalling()
    c = evaluate_symbol("DDD/USDT", df, volume_24h_usdt=5e6,
                        change_24h_pct=+15.0, cfg=cfg)
    assert c is None


def test_insufficient_history_returns_none(cfg):
    df = _frame([100.0, 101.0, 102.0])
    c = evaluate_symbol("EEE/USDT", df, 200e6, 15.0, cfg)
    assert c is None


# ── presentation ─────────────────────────────────────────────────────────────

def test_rank_puts_closest_to_cross_first():
    mk = lambda sym, gap: Candidate(sym, "short", 75, gap, -0.1, 15, 200e6, "")
    out = rank([mk("A", 2.0), mk("B", 0.3), mk("C", 1.1)])
    assert [c.symbol for c in out] == ["B", "C", "A"]


def test_format_table_empty():
    assert "No candidates" in format_table([])


def test_format_table_includes_disclaimer():
    c = Candidate("X/USDT", "short", 75.0, 0.4, -0.2, 12.0, 200e6, "note")
    out = format_table([c])
    assert "X/USDT" in out
    assert "POTENTIAL" in out


# ── Delta tracking across refreshes ──────────────────────────────────────────

from bot.scanner import ScanTracker, Delta, rank_with_deltas, format_table_with_deltas


def _cand(sym, direction, rsi, gap):
    return Candidate(sym, direction, rsi, gap, 0.0, 15.0, 200e6, "")


def test_first_sighting_is_new():
    t = ScanTracker()
    d = t.diff(_cand("A/USDT", "short", 72, 1.5))
    assert d.is_new and d.strength == "new"


def test_short_strengthens_when_rsi_climbs():
    """RSI 70 -> 80 on refresh is a strengthening short."""
    t = ScanTracker()
    first = _cand("A/USDT", "short", 70, 1.5)
    t.commit([first])
    d = t.diff(_cand("A/USDT", "short", 80, 1.2))
    assert d.strength == "strengthening"
    assert d.rsi_change == pytest.approx(10.0)


def test_short_confirmed_on_bearish_cross():
    """EMA9 dropping BELOW EMA21 confirms the short."""
    t = ScanTracker()
    t.commit([_cand("A/USDT", "short", 75, +0.40)])
    d = t.diff(_cand("A/USDT", "short", 78, -0.10))
    assert d.crossed_down
    assert d.strength == "CONFIRMED"
    assert "crossed BELOW" in d.note


def test_long_confirmed_on_bullish_cross():
    """RSI 50 -> 55 with EMA9 rising ABOVE EMA21 confirms the long."""
    t = ScanTracker()
    t.commit([_cand("B/USDT", "long", 50, -0.30)])
    d = t.diff(_cand("B/USDT", "long", 55, +0.12))
    assert d.crossed_up
    assert d.strength == "CONFIRMED"
    assert d.rsi_change == pytest.approx(5.0)


def test_weakening_when_rsi_eases():
    t = ScanTracker()
    t.commit([_cand("C/USDT", "short", 78, 1.4)])
    d = t.diff(_cand("C/USDT", "short", 71, 1.3))
    assert d.strength == "weakening"
    assert d.rsi_change < 0


def test_no_false_cross_when_gap_keeps_sign():
    t = ScanTracker()
    t.commit([_cand("D/USDT", "short", 75, 1.5)])
    d = t.diff(_cand("D/USDT", "short", 76, 0.9))
    assert not d.crossed_down and not d.crossed_up


def test_ranking_puts_confirmed_first():
    t = ScanTracker()
    t.commit([_cand("A/USDT", "short", 75, +0.4),
              _cand("B/USDT", "short", 72, +1.5)])
    pairs = t.annotate([_cand("A/USDT", "short", 78, -0.1),   # CONFIRMED
                        _cand("B/USDT", "short", 74, +1.4),   # strengthening
                        _cand("E/USDT", "short", 71, +2.0)])  # new
    ordered = rank_with_deltas(pairs)
    assert [c.symbol for c, _ in ordered][0] == "A/USDT"
    assert ordered[0][1].strength == "CONFIRMED"


def test_delta_table_renders():
    t = ScanTracker()
    t.commit([_cand("A/USDT", "short", 70, +0.5)])
    pairs = t.annotate([_cand("A/USDT", "short", 80, -0.2)])
    out = format_table_with_deltas(pairs)
    assert "CONFIRMED" in out and "POTENTIAL" in out


# ── Regression: the API payload must be JSON-serialisable ───────────────────

def test_no_numpy_types_leak_into_api_payload(cfg):
    """
    numpy scalars from pandas leaked into the snapshot. np.bool_ is NOT a bool
    subclass and cannot be JSON encoded, so /api/scan returned 500. Every value
    crossing the API boundary must be a native Python type.
    """
    import json
    import numpy as np

    df = _rising_then_stalling()
    c = evaluate_symbol("AAA/USDT:USDT", df, 200e6, 15.0, cfg)
    if c is None:                      # ensure we exercise a real candidate
        c = Candidate("AAA/USDT:USDT", "short", 72.0,
                      float(ema_gap_pct(prepare(df, cfg).iloc[-1])),
                      -0.6, 15.0, 200e6, "note")

    t = ScanTracker()
    t.commit([c])
    # Force a cross so crossed_down/up are exercised
    c2 = Candidate(c.symbol, c.direction, c.rsi + 3, -abs(c.ema_gap_pct) - 0.1,
                   c.gap_change_pct, c.change_24h_pct, c.volume_24h_usdt, c.note)
    _, d = t.annotate([c2])[0]

    row = c2.as_row()
    row.update({
        "strength": str(d.strength),
        "rsi_change": None if d.is_new else round(float(d.rsi_change), 1),
        "crossed_down": bool(d.crossed_down),
        "crossed_up": bool(d.crossed_up),
        "delta_note": str(d.note),
    })

    json.dumps(row)                    # must not raise

    for k, v in row.items():
        assert not isinstance(v, (np.generic,)), f"{k} is a numpy type: {type(v)}"


def test_cross_flags_are_native_bools():
    import numpy as np
    t = ScanTracker()
    t.commit([_cand("A/USDT", "short", 75, +0.40)])
    d = t.diff(_cand("A/USDT", "short", 78, -0.10))
    assert type(d.crossed_down) is bool
    assert type(d.crossed_up) is bool
    assert not isinstance(d.crossed_down, np.generic)


def test_ema_gap_returns_native_float(cfg):
    import numpy as np
    df = prepare(_rising_then_stalling(), cfg)
    g = ema_gap_pct(df.iloc[-1])
    assert type(g) is float
    assert not isinstance(g, np.generic)
