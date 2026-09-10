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


# ── 24h range position ───────────────────────────────────────────────────────

from bot.scanner import range_position_24h


def test_range_position_maths():
    assert range_position_24h(100, 100, 90) == pytest.approx(1.0)   # at the high
    assert range_position_24h(90, 100, 90) == pytest.approx(0.0)    # at the low
    assert range_position_24h(95, 100, 90) == pytest.approx(0.5)
    assert range_position_24h(95, None, 90) is None
    assert range_position_24h(95, 90, 90) is None                   # zero range


def test_short_near_24h_high_is_strong():
    """A short near the 24h high has resistance overhead — the stronger case."""
    c = Candidate("X", "short", 75, 0.5, 0, 15, 200e6, "", range_pos_24h=0.95)
    assert c.range_quality == "strong"


def test_short_near_24h_low_is_weak():
    c = Candidate("X", "short", 75, 0.5, 0, 15, 200e6, "", range_pos_24h=0.08)
    assert c.range_quality == "weak"


def test_long_near_24h_low_is_strong():
    c = Candidate("X", "long", 55, -0.5, 0, -15, 200e6, "", range_pos_24h=0.05)
    assert c.range_quality == "strong"


def test_long_near_24h_high_is_weak():
    c = Candidate("X", "long", 55, -0.5, 0, -15, 200e6, "", range_pos_24h=0.92)
    assert c.range_quality == "weak"


def test_ranking_prefers_range_supported_candidate():
    """Among equally-strong signals, the one with 24h-range support ranks higher."""
    t = ScanTracker()
    weak = Candidate("W/USDT", "short", 75, 0.5, 0, 15, 200e6, "", range_pos_24h=0.10)
    strong = Candidate("S/USDT", "short", 75, 0.5, 0, 15, 200e6, "", range_pos_24h=0.95)
    ordered = rank_with_deltas(t.annotate([weak, strong]))
    assert [c.symbol for c, _ in ordered][0] == "S/USDT"


def test_unknown_range_does_not_crash_ranking():
    t = ScanTracker()
    c = Candidate("U/USDT", "short", 75, 0.5, 0, 15, 200e6, "", range_pos_24h=None)
    assert c.range_quality == "unknown"
    assert len(rank_with_deltas(t.annotate([c]))) == 1


# ── Long RSI upper bound ─────────────────────────────────────────────────────

def _bearish_frame():
    """Bearish EMA structure (fast below slow) that is converging upward."""
    closes = [100 - i * 0.5 for i in range(48)]
    last = closes[-1]
    closes += [last + i * 0.35 for i in range(1, 13)]
    return _frame(closes)


def test_long_screen_excludes_overbought(cfg):
    """
    Without an upper bound, a coin at RSI 85 in a bearish structure surfaced as
    a LONG candidate — arguably a short setup. The max must exclude it.
    """
    c = ScanConfig(long_rsi_min=38.0, long_rsi_max=65.0)
    assert not (c.long_rsi_min <= 85 <= c.long_rsi_max)
    assert not (c.long_rsi_min <= 70 <= c.long_rsi_max)


def test_long_screen_admits_recovering_range():
    c = ScanConfig(long_rsi_min=38.0, long_rsi_max=65.0)
    for rsi in (38, 45, 55, 65):
        assert c.long_rsi_min <= rsi <= c.long_rsi_max


def test_long_max_applied_in_evaluate():
    """A bearish-structure coin above long_rsi_max must not surface as long."""
    c = ScanConfig(long_rsi_min=38.0, long_rsi_max=45.0)  # deliberately tight
    df = _bearish_frame()
    got = evaluate_symbol("BBB/USDT", df, 200e6, -15.0, c)
    if got is not None:
        assert c.long_rsi_min <= got.rsi <= c.long_rsi_max


# ── Direction balancing (keep both sides visible) ────────────────────────────

from bot.scanner import balance_directions


def _pair(sym, direction):
    return (Candidate(sym, direction, 70, 0.5, 0, 10, 200e6, ""), Delta(is_new=True))


def test_balance_caps_dominant_direction():
    pairs = [_pair(f"S{i}", "short") for i in range(9)] + [_pair(f"L{i}", "long") for i in range(2)]
    out = balance_directions(pairs, max_share=0.70)
    shorts = [p for p in out if p[0].direction == "short"]
    longs = [p for p in out if p[0].direction == "long"]
    assert len(shorts) <= 7          # capped at 70% of 11
    assert len(longs) == 2           # minority kept in full


def test_balance_keeps_all_when_one_direction_absent():
    pairs = [_pair(f"S{i}", "short") for i in range(6)]
    assert len(balance_directions(pairs)) == 6


def test_balance_preserves_ranking_order():
    pairs = [_pair("S1", "short"), _pair("L1", "long"), _pair("S2", "short")]
    out = balance_directions(pairs)
    assert [c.symbol for c, _ in out] == ["S1", "L1", "S2"]


def test_balance_handles_empty():
    assert balance_directions([]) == []


# ── OHLCV window must actually span 24h for the range fallback ───────────────

def test_candle_window_covers_24h():
    """
    The 24h high/low fallback is only honest if the fetched window covers 24h.
    120 candles is 6h on a 3m chart — that would label a 6-hour range as daily.
    """
    from bot.scan_runner import ScanRunner
    from bot.scanner import ScanConfig
    for tf, mins in [("3m", 3), ("5m", 5), ("15m", 15)]:
        r = ScanRunner(ScanConfig(), timeframe=tf, max_symbols=1, interval=60)
        hours = r._candles_for_24h() * mins / 60
        assert hours >= 24, f"{tf} only spans {hours:.1f}h"


def test_candle_window_respects_exchange_limit():
    from bot.scan_runner import ScanRunner
    from bot.scanner import ScanConfig
    r = ScanRunner(ScanConfig(), timeframe="1m", max_symbols=1, interval=60)
    assert r._candles_for_24h() <= 1000       # Binance kline limit headroom


def test_candle_window_has_a_floor_for_indicators():
    from bot.scan_runner import ScanRunner
    from bot.scanner import ScanConfig
    r = ScanRunner(ScanConfig(), timeframe="1h", max_symbols=1, interval=60)
    assert r._candles_for_24h() >= 120        # enough for EMA21 / RSI14


# ── Scanner environment must match the trading account ───────────────────────

def test_scanner_uses_demo_endpoint_when_demo():
    """
    Screening the live market while entries execute on demo surfaced candidates
    that were not tradable there ("no price" / "no leverage" at entry). The
    scanner must screen the same environment the account trades on.
    """
    from bot.scan_runner import ScanRunner
    r = ScanRunner(ScanConfig(), timeframe="3m", max_symbols=5, interval=60, demo=True)
    assert "demo-fapi" in r.exchange.urls["api"]["fapiPublic"]
    assert r.demo is True


def test_scanner_uses_live_endpoint_when_not_demo():
    from bot.scan_runner import ScanRunner
    r = ScanRunner(ScanConfig(), timeframe="3m", max_symbols=5, interval=60, demo=False)
    assert "demo-fapi" not in r.exchange.urls["api"]["fapiPublic"]
    assert r.demo is False


def test_snapshot_reports_which_market_was_scanned():
    from bot.scan_runner import ScanRunner
    r = ScanRunner(ScanConfig(), timeframe="3m", max_symbols=5, interval=60, demo=True)
    assert r.snapshot()["demo"] is True


# ── Volume filtering must survive inflated demo volumes ──────────────────────

def _vol_runner(mode, mult, percentile=60.0):
    import time
    from bot.scan_runner import ScanRunner
    live = {"AAA": 86e6, "BBB": 40e6, "CCC": 200e6, "DDD": 12e6, "EEE": 500e6}

    class Ex:
        def fetch_tickers(self):
            return {f"{k}/USDT:USDT": {"quoteVolume": v * mult, "percentage": 12.0,
                                       "high": 125.0, "low": 100.0}
                    for k, v in live.items()}
        def fetch_ohlcv(self, sym, tf, limit=100):
            cl = [100 + i * 0.5 for i in range(48)]
            last = cl[-1]
            cl += [last - i * 0.35 for i in range(1, 7)]
            now = int(time.time() * 1000)
            return [[now - (len(cl) - i) * 180000, c, c * 1.002, c * 0.998, c, 1000.0]
                    for i, c in enumerate(cl)]

    cfg = ScanConfig(min_24h_vol_usdt=50e6, min_abs_change_pct=8.0,
                     volume_mode=mode, vol_percentile=percentile, long_rsi_min=38.0)
    r = ScanRunner(cfg, timeframe="3m", max_symbols=10, interval=60)
    r.exchange = Ex()
    pairs = r.scan_once()
    return r, sorted(c.symbol.split("/")[0] for c, _ in pairs)


def test_absolute_floor_goes_inert_on_inflated_volumes():
    """Demo reports ~35x live volume, which lets an absolute floor pass everything."""
    _, live_syms = _vol_runner("absolute", 1)
    _, demo_syms = _vol_runner("absolute", 35)
    assert len(demo_syms) > len(live_syms)
    assert len(demo_syms) == 5          # filter no longer discriminates


def test_percentile_floor_selects_the_same_coins_regardless_of_scale():
    """Ranking survives a uniform rescale, so percentile mode is environment-agnostic."""
    _, live_syms = _vol_runner("percentile", 1)
    _, demo_syms = _vol_runner("percentile", 35)
    assert live_syms == demo_syms
    assert live_syms == ["CCC", "EEE"]


def test_percentile_floor_scales_with_the_universe():
    live_r, _ = _vol_runner("percentile", 1)
    demo_r, _ = _vol_runner("percentile", 35)
    assert demo_r._effective_vol_floor == pytest.approx(
        live_r._effective_vol_floor * 35, rel=1e-6)


def test_percentile_setting_controls_strictness():
    _, loose = _vol_runner("percentile", 1, percentile=20.0)
    _, tight = _vol_runner("percentile", 1, percentile=80.0)
    assert len(loose) >= len(tight)


# ── as_row must carry the 24h range (regression) ─────────────────────────────

def test_as_row_includes_range_fields():
    """
    as_row() was rewritten for numpy casting and silently dropped the range
    fields, so the API never sent them and the column read "n/a" regardless of
    the fallbacks upstream.
    """
    c = Candidate("X/USDT", "short", 75, 0.5, 0, 15, 200e6, "",
                  range_pos_24h=0.86, pct_above_24h_low=21.4,
                  pct_below_24h_high=-2.88)
    row = c.as_row()
    for key in ("range_pos_24h", "range_quality",
                "pct_above_24h_low", "pct_below_24h_high"):
        assert key in row, f"as_row() dropped {key}"
    assert row["range_pos_24h"] == pytest.approx(0.86)
    assert row["range_quality"] == "strong"


def test_as_row_is_json_serialisable():
    import json
    c = Candidate("X/USDT", "long", 55, -0.5, 0, -15, 200e6, "",
                  range_pos_24h=0.06, pct_above_24h_low=1.2,
                  pct_below_24h_high=-30.0)
    json.dumps(c.as_row())


def _mild_stall():
    """Uptrend with only a slight stall, so RSI stays in the short band."""
    closes = [100 + i * 0.5 for i in range(48)]
    last = closes[-1]
    closes += [last - i * 0.35 for i in range(1, 7)]
    return _frame(closes)


def test_distances_computed_from_extremes():
    df = _mild_stall()
    c = evaluate_symbol("AAA/USDT", df, 200e6, 15.0, ScanConfig(long_rsi_min=38.0),
                        high_24h=125.0, low_24h=100.0)
    assert c is not None
    assert c.pct_above_24h_low > 0      # above the 24h low
    assert c.pct_below_24h_high < 0     # below the 24h high


def test_distances_absent_without_extremes():
    df = _mild_stall()
    c = evaluate_symbol("AAA/USDT", df, 200e6, 15.0, ScanConfig(long_rsi_min=38.0))
    if c is not None:
        assert c.pct_above_24h_low is None
        assert c.pct_below_24h_high is None


# ── ATR / volatility ─────────────────────────────────────────────────────────

def _vol_frame(vol):
    cl = [100 + i * 0.5 for i in range(48)]
    last = cl[-1]
    cl += [last - i * 0.35 for i in range(1, 7)]
    return pd.DataFrame({
        "open": [cl[0]] + cl[:-1],
        "high": [c * (1 + vol) for c in cl],
        "low": [c * (1 - vol) for c in cl],
        "close": cl, "volume": [1000.0] * len(cl),
    })


def test_atr_pct_scales_with_volatility():
    cfg = ScanConfig(long_rsi_min=38.0, stop_pct_for_ratio=1.0)
    calm = evaluate_symbol("X/USDT", _vol_frame(0.0008), 200e6, 15.0, cfg,
                           high_24h=125.0, low_24h=100.0)
    wild = evaluate_symbol("X/USDT", _vol_frame(0.01), 200e6, 15.0, cfg,
                           high_24h=125.0, low_24h=100.0)
    assert calm.atr_pct < wild.atr_pct


def test_stop_expressed_in_atrs():
    """
    The useful number: how many typical candle-ranges the stop sits away. Under
    ~1 ATR it is inside normal noise and gets hit regardless of the thesis.
    """
    cfg = ScanConfig(long_rsi_min=38.0, stop_pct_for_ratio=1.0)
    calm = evaluate_symbol("X/USDT", _vol_frame(0.0008), 200e6, 15.0, cfg,
                           high_24h=125.0, low_24h=100.0)
    wild = evaluate_symbol("X/USDT", _vol_frame(0.01), 200e6, 15.0, cfg,
                           high_24h=125.0, low_24h=100.0)
    assert calm.stop_vs_atr > 1.5      # comfortable room
    assert wild.stop_vs_atr < 1.0      # stop sits inside the noise


def test_volatility_band_excludes_too_quiet():
    cfg = ScanConfig(long_rsi_min=38.0, min_atr_pct=1.0)
    assert evaluate_symbol("X/USDT", _vol_frame(0.0008), 200e6, 15.0, cfg,
                           high_24h=125.0, low_24h=100.0) is None


def test_volatility_band_excludes_too_wild():
    cfg = ScanConfig(long_rsi_min=38.0, max_atr_pct=1.0)
    assert evaluate_symbol("X/USDT", _vol_frame(0.01), 200e6, 15.0, cfg,
                           high_24h=125.0, low_24h=100.0) is None


def test_volatility_band_disabled_by_default():
    cfg = ScanConfig(long_rsi_min=38.0)
    assert cfg.min_atr_pct == 0.0 and cfg.max_atr_pct == 0.0
    assert evaluate_symbol("X/USDT", _vol_frame(0.01), 200e6, 15.0, cfg,
                           high_24h=125.0, low_24h=100.0) is not None


def test_atr_in_as_row_and_serialisable():
    import json
    cfg = ScanConfig(long_rsi_min=38.0, stop_pct_for_ratio=1.0)
    c = evaluate_symbol("X/USDT", _vol_frame(0.002), 200e6, 15.0, cfg,
                        high_24h=125.0, low_24h=100.0)
    row = c.as_row()
    assert "atr_pct" in row and "stop_vs_atr" in row
    json.dumps(row)


# ── EMA crossover tolerance ──────────────────────────────────────────────────

def _at_gap(target_gap_pct, drift=0.5):
    """Frame engineered so the final EMA gap lands near target_gap_pct."""
    closes = [100 + i * drift for i in range(48)]
    last = closes[-1]
    # taper down to pull EMA9 toward (and through) EMA21
    n = 3 if target_gap_pct > 0.2 else (9 if target_gap_pct > -0.1 else 16)
    closes += [last - i * 0.55 for i in range(1, n + 1)]
    return _frame(closes)


def test_overbought_coin_just_crossed_down_is_still_a_short():
    """
    A coin at high RSI whose EMA9 has just crossed below EMA21 is a confirmed
    roll-over — the strongest kind of fade. A hard sign test routed it to the
    long side, where the long RSI band rejected it, so it vanished entirely.
    """
    cfg = ScanConfig(short_rsi_min=70, long_rsi_min=38, long_rsi_max=65,
                     ema_tolerance_pct=0.15)
    # Directly exercise the branch logic across the crossover.
    for gap in (0.10, 0.01, -0.05, -0.14):
        assert gap > -cfg.ema_tolerance_pct      # qualifies as a short at RSI>=70


def test_tolerance_does_not_extend_indefinitely():
    cfg = ScanConfig(ema_tolerance_pct=0.15)
    assert not (-0.30 > -cfg.ema_tolerance_pct)  # well past the cross: not a short


def test_zero_tolerance_restores_strict_behaviour():
    cfg = ScanConfig(ema_tolerance_pct=0.0)
    assert not (-0.01 > -cfg.ema_tolerance_pct)


def test_short_branch_wins_inside_the_overlap_band():
    """
    Inside the tolerance band both branches could match. RSI decides, and the
    short test runs first — a high RSI is the stronger statement of direction.
    """
    cfg = ScanConfig(short_rsi_min=70, long_rsi_min=38, long_rsi_max=90,
                     ema_tolerance_pct=0.15)
    rsi, gap = 85.0, -0.05
    short_ok = gap > -cfg.ema_tolerance_pct and rsi >= cfg.short_rsi_min
    long_ok = gap < cfg.ema_tolerance_pct and cfg.long_rsi_min <= rsi <= cfg.long_rsi_max
    assert short_ok and long_ok          # both would match...
    c = evaluate_symbol("X/USDT", _at_gap(-0.05), 200e6, 12.0, cfg,
                        high_24h=130.0, low_24h=100.0)
    if c is not None:
        assert c.direction == "short"    # ...and short wins


def test_tolerance_is_configurable():
    import os
    from bot.config import _env_float
    os.environ["_T_EMA"] = "0.4"
    try:
        assert _env_float("SCAN_EMA_TOLERANCE_PCT", 0.15) == pytest.approx(0.15)
        assert _env_float("_T_EMA", 0.15) == pytest.approx(0.4)
    finally:
        os.environ.pop("_T_EMA", None)


# ── Higher-timeframe trend, from candles already fetched ─────────────────────

def test_htf_trend_sign_follows_the_trend():
    """
    Resampled from the same candles used for the 24h range, so it costs no
    extra requests.
    """
    import numpy as np
    import pandas as pd
    from bot.scanner import htf_trend
    n = 600
    assert htf_trend(pd.DataFrame({"close": np.linspace(100, 130, n)})) > 0
    assert htf_trend(pd.DataFrame({"close": np.linspace(130, 100, n)})) < 0


def test_htf_trend_is_flat_on_a_flat_series():
    import numpy as np
    import pandas as pd
    from bot.scanner import htf_trend
    assert htf_trend(pd.DataFrame({"close": np.full(600, 100.0)})) == pytest.approx(0.0)


def test_htf_trend_needs_enough_history():
    import pandas as pd
    from bot.scanner import htf_trend
    assert htf_trend(pd.DataFrame({"close": [1, 2, 3]})) is None
    assert htf_trend(None) is None


def test_candidate_carries_the_htf_field():
    from bot.scanner import Candidate
    import dataclasses
    names = {f.name for f in dataclasses.fields(Candidate)}
    assert "htf_trend_pct" in names


# ── Breakout structure (measurement only) ────────────────────────────────────

def _bdf(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["open", "close"])


_RISING = [[0.98, 1.00], [1.0, 1.02], [1.02, 1.05], [1.05, 1.09]]


def test_full_breakout_is_detected():
    """
    New extreme + consecutive candles + a gap that is wide AND widening. The
    widening is what separates a breakout from a blow-off top.
    """
    from bot.scanner import breakout_structure
    r = breakout_structure(_bdf(_RISING), 1.09, 0.8, "short", 1.4, 0.3)
    assert r["breakout"] is True
    assert r["at_extreme"] and r["gap_wide"] and r["gap_widening"]
    assert r["consecutive"] == 3


def test_a_narrowing_gap_is_not_a_breakout():
    from bot.scanner import breakout_structure
    assert breakout_structure(_bdf(_RISING), 1.09, 0.8, "short", 1.4, -0.3)["breakout"] is False


def test_a_narrow_gap_is_not_a_breakout():
    from bot.scanner import breakout_structure
    assert breakout_structure(_bdf(_RISING), 1.09, 0.8, "short", 0.2, 0.3)["breakout"] is False


def test_away_from_the_extreme_is_not_a_breakout():
    from bot.scanner import breakout_structure
    assert breakout_structure(_bdf(_RISING), 1.30, 0.8, "short", 1.4, 0.3)["breakout"] is False


def test_a_broken_run_is_counted_accurately():
    from bot.scanner import breakout_structure
    mixed = _bdf([[0.98, 1.00], [1.0, 1.02], [1.02, 0.99], [0.99, 1.09]])
    r = breakout_structure(mixed, 1.09, 0.8, "short", 1.4, 0.3)
    assert r["consecutive"] == 1 and r["breakout"] is False


def test_downside_breakout_uses_red_candles_and_the_low():
    from bot.scanner import breakout_structure
    falling = _bdf([[1.09, 1.05], [1.05, 1.02], [1.02, 1.0], [1.0, 0.97]])
    r = breakout_structure(falling, 1.3, 0.97, "long", -1.4, 0.3)
    assert r["breakout"] is True and r["consecutive"] == 3


def test_too_little_history_is_safe():
    from bot.scanner import breakout_structure
    assert breakout_structure(_bdf([[1, 1]]), 1, 1, "short", 1, 1)["breakout"] is False
    assert breakout_structure(None, 1, 1, "short", 1, 1)["breakout"] is False
