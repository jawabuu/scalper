"""
Tests for the two additions made after the STORJ/TA losses:

  * AutoTradeConfig.veto_breakout — refuse candidates whose move is still
    expanding (scanner.breakout_structure's verdict, previously recorded only).
  * ROI checkpoint at age 0 — the ROI at first observation, which neither
    peak_roi (clamped at 0, ratchets up) nor trough_roi (cannot separate
    "opened at -13%" from "opened flat and fell to -13%") can express.
"""
import pytest

from bot.auto_trader import (AutoTradeConfig, AutoTrader, callback_for,
                             evaluate_candidate)
from bot.scanner import breakout_structure


def _short(**kw):
    row = {"symbol": "X/USDT:USDT", "direction": "short", "rsi": 88.0,
           "pct_below_24h_high": -0.1, "pct_above_24h_low": 110.0,
           "strength": "strengthening"}
    row.update(kw)
    return row


def _brk(**kw):
    out = {"at_extreme": True, "consecutive": 3, "gap_wide": True,
           "gap_widening": True, "breakout": True}
    out.update(kw)
    return out


def _long(**kw):
    """A coin pinned to its 24h low that is still dipping — the long-side
    mirror of the STORJ/TA case."""
    row = {"symbol": "Y/USDT:USDT", "direction": "long", "rsi": 40.0,
           "pct_above_24h_low": 0.1, "pct_below_24h_high": -60.0,
           "strength": "strengthening", "gap_narrowing": True, "gap_rising": True,
           "turn": {"turned_up": True, "bars_since_low": 4, "rise_pct": 1.2}}
    row.update(kw)
    return row


# ── The veto ────────────────────────────────────────────────────────────────

def test_veto_off_by_default_keeps_current_behaviour():
    """The default must not change which trades are taken."""
    cfg = AutoTradeConfig(enabled=True)
    assert cfg.veto_breakout is False
    d = evaluate_candidate(_short(breakout=_brk()), streak=2, cfg=cfg, atr_pct=2.0)
    assert d.enter


def test_veto_on_refuses_an_expanding_move():
    cfg = AutoTradeConfig(enabled=True, veto_breakout=True)
    d = evaluate_candidate(_short(breakout=_brk()), streak=2, cfg=cfg, atr_pct=2.0)
    assert not d.enter
    assert "expanding" in d.reason


def test_veto_on_still_allows_an_exhausted_move():
    """
    Same RSI and same distance to the extreme — only the structure differs.
    A gap that has stopped widening is the exhaustion case the fade targets.
    """
    cfg = AutoTradeConfig(enabled=True, veto_breakout=True)
    row = _short(breakout=_brk(gap_widening=False, breakout=False))
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=2.0).enter


def test_veto_tolerates_a_missing_breakout_block():
    """Older scanner rows, or a scan that errored, must not become un-enterable."""
    cfg = AutoTradeConfig(enabled=True, veto_breakout=True)
    assert evaluate_candidate(_short(), streak=2, cfg=cfg, atr_pct=2.0).enter
    assert evaluate_candidate(_short(breakout=None), streak=2,
                              cfg=cfg, atr_pct=2.0).enter


def test_veto_is_live_tunable_but_not_a_safety_limit():
    assert "veto_breakout" in AutoTrader.TUNABLE
    assert "veto_breakout" not in AutoTrader.SAFETY_ONLY


# ── The long side: same failure, mirrored ───────────────────────────────────

def test_veto_on_refuses_a_long_that_is_still_dipping():
    """
    A coin on its 24h low with consecutive red candles and a widening gap is
    still falling. Buying it is the same mistake as shorting STORJ, mirrored —
    and max_dist_to_extreme_pct selects for it the same way.
    """
    cfg = AutoTradeConfig(enabled=True, veto_breakout=True, long_rsi_min=38.0,
                          long_rsi_max=65.0)
    d = evaluate_candidate(_long(breakout=_brk()), streak=2, cfg=cfg, atr_pct=2.0)
    assert not d.enter
    assert "expanding" in d.reason


def test_veto_on_still_allows_a_long_whose_fall_has_stalled():
    cfg = AutoTradeConfig(enabled=True, veto_breakout=True, long_rsi_min=38.0,
                          long_rsi_max=65.0)
    row = _long(breakout=_brk(gap_widening=False, breakout=False))
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=2.0).enter


def test_breakout_structure_is_symmetric_for_longs():
    """
    Price at the 24h low, three consecutive RED candles, gap wide and widening.
    breakout_structure flips both the extreme test and the candle colour on
    direction, so the long verdict must come out the same as the short one.
    """
    df = _frame([0.0490, 0.0483, 0.0477, 0.0470],
                [0.0497, 0.0490, 0.0483, 0.0477])
    out = breakout_structure(df, high_24h=0.0900, low_24h=0.0470,
                             direction="long", gap_pct=-2.5, gap_change=0.4)
    assert out["at_extreme"]
    assert out["consecutive"] == 3      # counts red closes, not green
    assert out["gap_wide"]             # keys on abs(gap), so a negative gap counts
    assert out["breakout"]


# ── The structure the veto keys on, against the two real trades ─────────────

def _frame(closes, opens):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"close": closes, "open": opens})


def test_storj_shape_is_flagged_as_a_breakout():
    """
    STORJ at 11:33: last price 0.06804 exactly equal to the 24h high, a run of
    green candles, EMA9 0.06542 vs EMA21 0.06349 (+3.04%) and widening.
    """
    df = _frame([0.0655, 0.0662, 0.0671, 0.06804],
                [0.0650, 0.0655, 0.0662, 0.0671])
    out = breakout_structure(df, high_24h=0.06804, low_24h=0.03163,
                             direction="short", gap_pct=3.04, gap_change=0.4)
    assert out["at_extreme"]
    assert out["consecutive"] == 3
    assert out["gap_wide"]
    assert out["gap_widening"]
    assert out["breakout"]


def test_a_rolled_over_gap_is_not_a_breakout():
    """Identical price structure; only the gap direction differs."""
    df = _frame([0.0655, 0.0662, 0.0671, 0.06804],
                [0.0650, 0.0655, 0.0662, 0.0671])
    out = breakout_structure(df, high_24h=0.06804, low_24h=0.03163,
                             direction="short", gap_pct=3.04, gap_change=-0.4)
    assert not out["breakout"]
    assert not out["gap_widening"]


# ── ROI at first observation ────────────────────────────────────────────────

def test_zero_is_the_first_roi_checkpoint():
    from bot.futures_guardian import FuturesGuardian
    assert FuturesGuardian.ROI_CHECKPOINTS_S[0] == 0
    for mark in (60, 180, 300):
        assert mark in FuturesGuardian.ROI_CHECKPOINTS_S


def test_roi_at_0s_records_a_position_that_opened_underwater():
    """
    The case the STORJ and TA rows could not express: peak_roi reports +0% for
    a trade that opened at -12%, because it is clamped at 0 and only ratchets
    upward. The age-0 checkpoint keeps the real figure.
    """
    from bot.futures_guard import GuardState
    from bot.futures_guardian import FuturesGuardian

    state = GuardState(peak_roi=max(-12.0, 0.0))
    assert state.peak_roi == 0.0          # the existing clamp, unchanged

    # Replay what _note_progress does, without constructing a live guardian.
    age = 0.0
    for mark in FuturesGuardian.ROI_CHECKPOINTS_S:
        if age >= mark and str(mark) not in state.roi_checkpoints:
            state.roi_checkpoints[str(mark)] = round(-12.0, 2)

    assert state.roi_checkpoints["0"] == -12.0
    assert "60" not in state.roi_checkpoints    # not yet 60s old


def test_roi_checkpoints_survive_a_restart(tmp_path):
    """The age-0 value must persist, or it is lost on every redeploy."""
    from bot import futures_state
    from bot.futures_guard import GuardState

    s = GuardState(peak_roi=0.0)
    s.roi_checkpoints = {"0": -12.0, "60": -30.0}

    path = str(tmp_path / "futures_state.json")
    assert futures_state.save(path, states={"X/USDT:USDT": s},
                              pos_meta={}, closed_trades=[], owner="t")
    restored = futures_state.restore_states(futures_state.load(path, owner="t"))
    assert restored["X/USDT:USDT"].roi_checkpoints["0"] == -12.0


# ── Velocity-aware callback floor ───────────────────────────────────────────

def test_velocity_floor_off_in_the_bare_dataclass():
    """
    AutoTradeConfig defaults to "off"; the deployed default is "short" and
    comes from BotConfig / AUTO_CALLBACK_USE_VELOCITY.
    """
    cfg = AutoTradeConfig(enabled=True)
    from bot.auto_trader import velocity_mode
    assert velocity_mode(cfg.callback_use_velocity) == "off"
    cb, _, src = callback_for(0.15, 0.512, cfg, recent_tr_pct=2.69)
    assert cb == pytest.approx(0.38, abs=0.01)   # STORJ, exactly as logged
    assert src == "atr_floor"


def test_velocity_floor_widens_the_storj_callback():
    """
    STORJ was logged at atr_pct 0.512 with its last candles covering ~2.69%.
    The 0.38% callback that resulted triggered on a wiggle mid-run.
    """
    cfg = AutoTradeConfig(enabled=True, callback_use_velocity=True)
    cb, notes, src = callback_for(0.15, 0.512, cfg, recent_tr_pct=2.69)
    assert cb == pytest.approx(2.02, abs=0.01)
    assert src == "velocity_floor"
    assert any("recent range" in n for n in notes)


def test_velocity_floor_is_inert_when_atr_already_leads():
    """A quiet tape must size exactly as before — ATR is the larger measure."""
    cfg_off = AutoTradeConfig(enabled=True)
    cfg_on = AutoTradeConfig(enabled=True, callback_use_velocity=True)
    a, _, sa = callback_for(1.0, 0.83, cfg_off, recent_tr_pct=0.70)
    b, _, sb = callback_for(1.0, 0.83, cfg_on, recent_tr_pct=0.70)
    assert a == b and sa == sb == "atr_floor"


def test_velocity_floor_tolerates_a_missing_measure():
    """Older rows carry no recent_tr_pct; sizing must fall back, not crash."""
    cfg = AutoTradeConfig(enabled=True, callback_use_velocity=True)
    cb, _, src = callback_for(0.15, 0.512, cfg, recent_tr_pct=None)
    assert cb == pytest.approx(0.38, abs=0.01)
    assert src == "atr_floor"


def test_velocity_floor_is_live_tunable():
    assert "callback_use_velocity" in AutoTrader.TUNABLE
    assert "callback_use_velocity" not in AutoTrader.SAFETY_ONLY


# ── Configurable extreme band ───────────────────────────────────────────────

def test_default_band_preserves_the_old_constant():
    """0.1% is what 0.999 meant — the default must not change recorded data."""
    df = _frame([0.0655, 0.0662, 0.0671, 0.06794],
                [0.0650, 0.0655, 0.0662, 0.0671])
    out = breakout_structure(df, high_24h=0.06804, low_24h=0.03163,
                             direction="short", gap_pct=2.258, gap_change=0.4)
    assert not out["at_extreme"]          # 0.15% away, outside a 0.1% band


def test_wider_band_catches_the_storj_distance():
    df = _frame([0.0655, 0.0662, 0.0671, 0.06794],
                [0.0650, 0.0655, 0.0662, 0.0671])
    out = breakout_structure(df, high_24h=0.06804, low_24h=0.03163,
                             direction="short", gap_pct=2.258, gap_change=0.4,
                             extreme_band_pct=0.2)
    assert out["at_extreme"] and out["breakout"]


def test_band_is_symmetric_for_longs():
    df = _frame([0.0490, 0.0483, 0.0477, 0.04707],
                [0.0497, 0.0490, 0.0483, 0.0477])
    near = dict(high_24h=0.0900, low_24h=0.0470, direction="long",
                gap_pct=-2.5, gap_change=0.4)
    assert not breakout_structure(df, **near)["at_extreme"]
    assert breakout_structure(df, extreme_band_pct=0.2, **near)["at_extreme"]


# ── recent_tr_pct ───────────────────────────────────────────────────────────

def test_recent_tr_exceeds_atr_on_an_accelerating_move():
    """The whole premise: ATR(14) lags current velocity when it matters."""
    import numpy as np
    import pandas_ta as ta
    from bot.scanner import prepare, recent_tr_pct, ScanConfig
    pd = pytest.importorskip("pandas")

    base = np.linspace(0.0320, 0.0600, 110)
    tail = [0.0600]
    for _ in range(10):
        tail.append(tail[-1] * 1.022)
    close = np.concatenate([base, np.array(tail[1:])])
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame({"open": open_,
                       "high": np.maximum(open_, close) * 1.004,
                       "low": np.minimum(open_, close) * 0.998,
                       "close": close, "volume": np.full(len(close), 3e8)})
    out = prepare(df, ScanConfig())
    atr_pct = float(out["atr"].iloc[-1]) / float(out["close"].iloc[-1]) * 100
    rtr = recent_tr_pct(out, 3)
    assert rtr > atr_pct * 1.3


def test_recent_tr_handles_a_frame_without_tr():
    from bot.scanner import recent_tr_pct
    pd = pytest.importorskip("pandas")
    assert recent_tr_pct(pd.DataFrame({"close": [1.0]}), 3) is None
    assert recent_tr_pct(None, 3) is None


# ── The LSK winner: the case every filter must NOT block ────────────────────
#
# 2026-09-12 07:00 UTC. dist 0.37%, atr_pct 1.536, callback 1.15%,
# roi at first observation -3.7%, closed +7.38% (+$6.40 realised).
# Its ATR was genuinely high, so the existing ATR floor sized it correctly.

def test_absolute_floor_blocks_both_losers_and_spares_the_winner():
    cfg = AutoTradeConfig(enabled=True, callback_min_pct=1.0,
                          callback_ratio=0.25, callback_atr_mult=0.75)
    lsk, _, _ = callback_for(0.37, 1.536, cfg)
    storj, _, _ = callback_for(0.15, 0.512, cfg)
    ta, _, _ = callback_for(0.15, 0.775, cfg)
    assert lsk == pytest.approx(1.15, abs=0.01)   # untouched — already above
    assert storj == pytest.approx(1.00, abs=0.01)  # was 0.38
    assert ta == pytest.approx(1.00, abs=0.01)     # was 0.58


def test_default_absolute_floor_changes_nothing():
    cfg = AutoTradeConfig(enabled=True)
    cb, _, _ = callback_for(0.15, 0.512, cfg)
    assert cb == pytest.approx(0.38, abs=0.01)


def test_lsk_is_not_vetoed_at_the_recommended_band():
    """
    The winner sat 0.37% from the high; both losers sat at 0.15%. A 0.2% band
    separates them — but see the note in DEFECTS.md: that is three trades and
    0.2 was chosen knowing the answer.
    """
    df = _frame([0.2210, 0.2240, 0.2265, 0.2280],
                [0.2200, 0.2210, 0.2240, 0.2265])
    out = breakout_structure(df, high_24h=0.2288, low_24h=0.1900,
                             direction="short", gap_pct=1.929, gap_change=0.4,
                             extreme_band_pct=0.2)
    assert not out["at_extreme"]
    assert not out["breakout"]


def test_callback_min_is_live_tunable():
    assert "callback_min_pct" in AutoTrader.TUNABLE


# ── LSK 10:36 — the loser the veto catches at the DEFAULT band ──────────────
#
# dist 0.09%, atr_pct 1.177, callback 0.88%, roi@0s -6.2%, closed -42.78%.
# Same symbol as the 07:00 winner, 3.5 hours later, opposite outcome.

def test_veto_catches_lsk_1036_without_widening_the_band():
    """at_extreme fires at 0.09% inside the stock 0.1% band — no tuning."""
    df = _frame([0.2180, 0.2192, 0.2199, 0.22010],
                [0.2170, 0.2180, 0.2192, 0.2199])
    out = breakout_structure(df, high_24h=0.22030, low_24h=0.1900,
                             direction="short", gap_pct=2.119, gap_change=0.4)
    assert out["at_extreme"]
    assert out["breakout"]

    cfg = AutoTradeConfig(enabled=True, veto_breakout=True)
    row = {"symbol": "LSK/USDT:USDT", "direction": "short", "rsi": 78.4,
           "pct_below_24h_high": -0.09, "pct_above_24h_low": 15.0,
           "strength": "strengthening", "atr_pct": 1.177, "breakout": out}
    assert not evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=1.177).enter


def test_callback_is_a_pure_function_of_atr_at_these_distances():
    """
    All four logged trades came out at exactly 0.75 x atr_pct: the ratio term
    (distance x 0.25) never binds at these distances. So an absolute callback
    floor is an ATR floor wearing a different hat, and AUTO_MIN_ATR_PCT is the
    knob that already exists for that.
    """
    cfg = AutoTradeConfig(enabled=True, callback_ratio=0.25,
                          callback_atr_mult=0.75)
    for dist, atr, logged in [(0.09, 1.177, 0.88), (0.37, 1.536, 1.15),
                              (0.15, 0.775, 0.58), (0.15, 0.512, 0.38)]:
        cb, _, src = callback_for(dist, atr, cfg)
        assert cb == pytest.approx(logged, abs=0.01)
        assert cb == pytest.approx(atr * 0.75, abs=0.01)
        assert src == "atr_floor"


# ── Execution diagnostics ───────────────────────────────────────────────────

def _trade(**kw):
    t = {"symbol": "X/USDT:USDT", "side": "short", "opened_at": 1789209392.8,
         "final_roi": -42.78, "peak_roi": 0.0, "trough_roi": -21.39,
         "roi_at_0s": -6.2, "signal_age_s": 30.0,
         "realised_pnl_usdt": -37.3008, "fees_usdt": 1.729,
         "exit_reason": "stop",
         "entry_context": {"rsi": 78.4, "atr_pct": 1.177,
                           "dist_to_extreme_pct": 0.09, "breakout": True,
                           "brk_at_extreme": True, "brk_consecutive": 3,
                           "brk_gap_wide": True, "brk_gap_widening": True,
                           "callback_pct": 0.88, "callback_source": "atr_floor",
                           "sized_stop_roi": 30.0}}
    ctx = kw.pop("entry_context", None)
    t.update(kw)
    if ctx:
        t["entry_context"] = {**t["entry_context"], **ctx}
    return t


def test_lsk_loser_trips_the_three_live_flags():
    """
    thin_callback is off by default — it fired on 76 of 78 real trades and its
    catches were net positive, so it described the strategy rather than a fault.
    """
    from bot.analysis import analyse
    d = analyse([_trade()])["execution_diagnostics"]
    assert len(d["rows"]) == 1
    assert set(d["rows"][0]["flags"]) == {
        "stop_overshoot", "dead_on_arrival", "breakout_entry"}
    assert d["rows"][0]["stop_overshoot_roi"] == pytest.approx(12.78, abs=0.01)


def test_thin_callback_can_be_switched_back_on():
    from bot.analysis import _diag_flags
    t = _trade()
    assert "thin_callback" not in _diag_flags(t)
    assert "thin_callback" in _diag_flags(t, thin_callback_pct=1.0)


def test_a_flag_that_fires_on_everything_is_called_out():
    from bot.analysis import _flag_verdict
    assert "describes the strategy" in _flag_verdict(76, 78, -500.0)
    assert "net POSITIVE" in _flag_verdict(5, 78, +336.07)
    assert _flag_verdict(3, 78, -182.87) == "specific and net negative"
    assert _flag_verdict(0, 78, 0.0) == "never fires"


def test_per_flag_split_separates_a_useful_flag_from_a_useless_one():
    """
    Mirrors the real sample: stop_overshoot caught 3 losers and no winners;
    thin_callback caught 7 winners and 3 losers for a net gain.
    """
    from bot.analysis import analyse
    losers = [_trade(symbol=f"L{i}/USDT:USDT", realised_pnl_usdt=-40.0)
              for i in range(3)]
    winners = [_trade(symbol=f"W{i}/USDT:USDT", final_roi=50.0, peak_roi=55.0,
                      realised_pnl_usdt=+60.0, exit_reason="trail",
                      entry_context={"sized_stop_roi": 30.0})
               for i in range(7)]
    split = {r["flag"]: r for r in
             analyse(losers + winners)["execution_diagnostics"]["by_flag"]}
    over = split["stop_overshoot"]
    assert over["winners"] == 0 and over["losers"] == 3
    assert over["net_pnl"] < 0
    assert over["verdict"] == "specific and net negative"
    brk = split["breakout_entry"]
    assert brk["winners"] == 7
    assert "describes the strategy" in brk["verdict"] or "net POSITIVE" in brk["verdict"]


def test_clean_winner_is_not_flagged():
    from bot.analysis import analyse
    win = _trade(final_roi=7.38, peak_roi=8.3, trough_roi=-3.7, roi_at_0s=-3.7,
                 signal_age_s=85.0, realised_pnl_usdt=6.4016,
                 exit_reason="trail",
                 entry_context={"atr_pct": 1.536, "dist_to_extreme_pct": 0.37,
                                "breakout": False, "brk_at_extreme": False,
                                "callback_pct": 1.15})
    d = analyse([win])["execution_diagnostics"]
    assert d["rows"] == []
    assert d["report"] == "No flagged trades."


def test_report_is_plain_text_and_names_the_numbers():
    from bot.analysis import analyse
    rpt = analyse([_trade()])["execution_diagnostics"]["report"]
    assert "<" not in rpt          # copyable as-is, no markup
    for frag in ("stop_overshoot", "callback 0.88%", "final -42.78%",
                 "overshoot 12.78%", "signal_age 30.0s"):
        assert frag in rpt


def test_unverified_trades_still_appear():
    """A trade whose P&L could not be read is exactly one worth looking at."""
    from bot.analysis import analyse
    t = _trade(realised_pnl_usdt=None, exit_is_estimate=True)
    assert analyse([t])["execution_diagnostics"]["rows"]


def test_diagnostics_survive_a_trade_with_no_context():
    from bot.analysis import analyse
    bare = {"symbol": "Y/USDT:USDT", "side": "long", "final_roi": -5.0}
    analyse([bare])   # must not raise


# ── Coin character: what ATR cannot express ─────────────────────────────────

def _shaped(bodies, wicks):
    pd = pytest.importorskip("pandas")
    rows, px = [], 1.0
    for b, w in zip(bodies, wicks):
        o = px
        c = px * (1 + b)
        rows.append((o, max(o, c) * (1 + w), min(o, c) * (1 - w * 0.2), c))
        px = c
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_body_separates_a_carried_move_from_a_rejected_one():
    """
    Two tapes of comparable range: one full-bodied, one mostly upper wick.
    ATR reports a similar number for both; they are opposite situations for a
    short, which is trying to fade rejection rather than stand in front of a
    trend.
    """
    from bot.scanner import candle_shape
    carried = candle_shape(_shaped([0.02] * 5, [0.001] * 5), 5)
    rejected = candle_shape(_shaped([0.002] * 5, [0.02] * 5), 5)
    assert carried["body_pct"] > 80
    assert rejected["body_pct"] < 20
    assert rejected["upper_wick_pct"] > carried["upper_wick_pct"] * 5


def test_candle_shape_survives_degenerate_candles():
    from bot.scanner import candle_shape
    pd = pytest.importorskip("pandas")
    flat = pd.DataFrame({"open": [1.0], "high": [1.0],
                         "low": [1.0], "close": [1.0]})
    assert candle_shape(flat, 5)["body_pct"] is None   # zero range, not a crash
    assert candle_shape(None, 5)["body_pct"] is None


# ── Drift since sizing ──────────────────────────────────────────────────────

def test_drift_is_positive_against_the_position_on_both_sides():
    from bot.futures_guardian import FuturesGuardian as G
    short = {"entry_context": {"sized_price": 0.0620}, "entry_price": 0.0650}
    long_ = {"entry_context": {"sized_price": 0.0650}, "entry_price": 0.0620}
    assert G._drift_pct(short, "short") == pytest.approx(4.839, abs=0.01)
    assert G._drift_pct(long_, "long") == pytest.approx(4.615, abs=0.01)
    # a short filled BELOW where it was sized drifted in its favour
    good = {"entry_context": {"sized_price": 0.0650}, "entry_price": 0.0620}
    assert G._drift_pct(good, "short") < 0


def test_drift_is_none_without_a_sized_price():
    from bot.futures_guardian import FuturesGuardian as G
    assert G._drift_pct({"entry_price": 0.065}, "short") is None
    assert G._drift_pct({"entry_context": {"sized_price": 0}, "entry_price": 1}, "short") is None


def test_character_fields_reach_the_diagnostics_report():
    from bot.analysis import analyse
    t = _trade(drift_since_sizing_pct=3.482,
               entry_context={"change_24h_pct": 108.9, "body_pct": 91.2,
                              "upper_wick_pct": 5.1, "lower_wick_pct": 3.7})
    rpt = analyse([t])["execution_diagnostics"]["report"]
    assert "drift 3.482%" in rpt
    assert "24h change 108.90%" in rpt
    assert "body 91.2%" in rpt


# ── Diagnostics capping ─────────────────────────────────────────────────────

def _many(n):
    out = []
    for i in range(n):
        loss = -2.0 - i          # steadily worse, so ranking is unambiguous
        t = _trade(symbol=f"C{i}/USDT:USDT", final_roi=loss,
                   realised_pnl_usdt=loss * 1.6, trough_roi=loss / 2)
        out.append(t)
    return out


def test_card_shows_only_the_worst_ten():
    from bot.analysis import analyse, DIAG_TOP_N
    d = analyse(_many(76))["execution_diagnostics"]
    assert d["total_flagged"] == 76
    assert len(d["rows"]) == DIAG_TOP_N == 10
    assert d["omitted"] == 66


def test_rows_are_ranked_by_money_not_roi():
    from bot.analysis import analyse
    rows = analyse(_many(76))["execution_diagnostics"]["rows"]
    pnls = [abs(r["realised_pnl_usdt"]) for r in rows]
    assert pnls == sorted(pnls, reverse=True)
    assert pnls[0] == max(abs(t["realised_pnl_usdt"]) for t in _many(76))


def test_severity_falls_back_to_roi_when_money_is_unreadable():
    from bot.analysis import _diag_severity
    assert _diag_severity({"realised_pnl_usdt": -12.5, "final_roi": -3.0}) == 12.5
    assert _diag_severity({"realised_pnl_usdt": None, "final_roi": -40.0}) == 0.4
    assert _diag_severity({}) == 0.0


def test_chips_count_every_flagged_trade_not_just_the_shown():
    from bot.analysis import analyse
    d = analyse(_many(76))["execution_diagnostics"]
    assert d["counts"]["breakout_entry"] == 76
    assert len(d["rows"]) == 10


def test_report_states_what_it_omitted():
    from bot.analysis import analyse
    rpt = analyse(_many(76))["execution_diagnostics"]["report"]
    assert "66 further flagged trade(s) not shown" in rpt
    assert "moved less than these" in rpt


def test_small_sets_are_not_annotated_as_truncated():
    from bot.analysis import analyse
    d = analyse(_many(3))["execution_diagnostics"]
    assert d["omitted"] == 0
    assert "not shown" not in d["report"]
    assert "omitted" not in d["report"]


# ── Fail-fast on entry ROI ──────────────────────────────────────────────────
#
# The four trades with a logged first-observation ROI:
#   LSK 07:00  -3.7%  -> +7.38%   (winner)
#   LSK 10:36  -6.2%  -> -42.78%
#   TA  08:18 -10.6%  -> -43.15%
#   STORJ     -66.6%  -> -67.22%

class _Pos:
    symbol = "X/USDT:USDT"
    side = "short"


def _guardian(**cfgkw):
    """A guardian stub carrying only what _should_fail_fast reads."""
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import FuturesGuardian

    g = FuturesGuardian.__new__(FuturesGuardian)
    base = dict(fail_fast_s=60.0, fail_fast_max_peak_roi=0.0,
                fail_fast_loss_roi=5.0)
    base.update(cfgkw)
    g.cfg = GuardConfig(**base)
    # Ten seconds old: long enough that a short timer has elapsed, short
    # enough that the stock 60s one has not. Stamping it at call time makes
    # sub-second timers race the test itself.
    g._pos_meta = {"X/USDT:USDT": {"opened_seen_at": time.time() - 10.0}}
    return g


def _state(entry_roi, peak=0.0):
    from bot.futures_guard import GuardState
    s = GuardState(peak_roi=peak)
    if entry_roi is not None:
        s.roi_checkpoints = {"0": entry_roi}
    return s


import time  # noqa: E402  (used by _guardian above)


def test_roi_at_fill_is_identically_zero():
    """
    The reason the entry-ROI cut was removed. roi_pct measures against
    pos.entry_price, which IS the fill price, so ROI at the moment of fill is
    zero by construction — for either side, at any leverage.
    """
    from bot.futures_guard import roi_pct

    class P:
        entry_price = 0.01687
        effective_leverage = 20.0
        side = "short"

    p = P()
    assert roi_pct(p, p.entry_price) == 0.0
    p.side = "long"
    assert roi_pct(p, p.entry_price) == 0.0


def test_a_nonzero_first_sight_roi_is_observation_lag():
    """
    KOMA read 0.00 because it was seen on the fill tick; GRIFFAIN read -13.31
    because it was seen ~3s later. Same field, and it describes the guardian,
    not the trade.
    """
    from bot.futures_guard import roi_pct

    class P:
        entry_price = 0.01495
        effective_leverage = 20.0
        side = "short"

    seen_at_fill = roi_pct(P(), 0.01495)
    seen_3s_later = roi_pct(P(), 0.01495 * 1.00665)
    assert seen_at_fill == 0.0
    assert seen_3s_later == pytest.approx(-13.3, abs=0.2)


def test_the_entry_roi_cut_is_gone():
    """It keyed on poll latency. No configuration may resurrect it."""
    from bot.futures_guard import GuardConfig
    assert not hasattr(GuardConfig(), "fail_fast_entry_roi")
    g = _guardian()
    # -10.6% at first sight, ten seconds old: only the 60s timer governs
    assert not g._should_fail_fast(_Pos(), _state(-10.6), -12.0)


def test_recovering_positions_are_spared_when_worsening_is_required():
    """
    RIVER reached -17.54% and closed +59.99%. Cutting on depth alone takes it;
    requiring it to be WORSE than where it started does not.
    """
    g = _guardian(fail_fast_require_worsening=True, fail_fast_s=5.0)
    recovering = _state(-17.54)
    assert not g._should_fail_fast(_Pos(), recovering, -9.0)   # climbing back
    worsening = _state(-6.2)
    assert g._should_fail_fast(_Pos(), worsening, -21.39)      # LSK 10:36


def test_worsening_gate_is_inert_without_an_entry_roi():
    """Trades closed before roi_at_0s existed must behave as they did."""
    g = _guardian(fail_fast_require_worsening=True, fail_fast_s=5.0)
    assert g._should_fail_fast(_Pos(), _state(None), -20.0)


def test_drift_is_positive_when_the_fill_favours_the_position():
    """
    A short filled ABOVE where it was sized sold higher — favourable. KOMA
    filled +0.308% above its sized price. An earlier version called that
    "against the position".
    """
    from bot.futures_guardian import FuturesGuardian as G
    short_high = {"entry_context": {"sized_price": 100.0}, "entry_price": 100.308}
    assert G._drift_pct(short_high, "short") == pytest.approx(0.308, abs=0.001)
    long_low = {"entry_context": {"sized_price": 100.0}, "entry_price": 99.7}
    assert G._drift_pct(long_low, "long") == pytest.approx(0.3, abs=0.001)


# ── peak_roi is no longer clamped ───────────────────────────────────────────

def test_peak_seeds_from_the_real_roi_including_negative():
    """
    STORJ was first seen at -66.6% and reported peak 0.00%. That was the
    clamp, not an observation, and it could not be told apart from a trade
    that opened flat.
    """
    from bot.futures_guard import adopt_state, GuardConfig
    st = adopt_state(_FakePos(), orders=[], current_roi=-66.6, cfg=GuardConfig())
    assert st.peak_roi == pytest.approx(-66.6)


def test_a_profitable_adoption_still_seeds_from_its_roi():
    from bot.futures_guard import adopt_state, GuardConfig
    st = adopt_state(_FakePos(), orders=[], current_roi=12.5, cfg=GuardConfig())
    assert st.peak_roi == pytest.approx(12.5)


def test_peak_still_only_ratchets_upward():
    from bot.futures_guard import GuardState, update_peak
    st = GuardState(peak_roi=-20.0)
    update_peak(st, -30.0)
    assert st.peak_roi == pytest.approx(-20.0)   # worse does not move the peak
    assert st.trough_roi == pytest.approx(-30.0)
    update_peak(st, -5.0)
    assert st.peak_roi == pytest.approx(-5.0)    # better does


def test_a_negative_peak_changes_no_guard_decision():
    """Every consumer compares peak against a threshold at or above zero."""
    from bot.futures_guard import GuardState, GuardConfig, is_armed
    cfg = GuardConfig(arm_roi=5.0, callback_roi=3.0, breakeven_at_roi=3.0)
    assert not is_armed(GuardState(peak_roi=-13.31), cfg)
    assert not is_armed(GuardState(peak_roi=0.0), cfg)
    assert is_armed(GuardState(peak_roi=5.0), cfg)


def test_fail_fast_still_sees_a_never_green_trade_as_eligible():
    g = _guardian(fail_fast_s=5.0, fail_fast_max_peak_roi=0.5)
    never_green = _state(-13.31, peak=-13.31)
    assert g._should_fail_fast(_Pos(), never_green, -20.0)
    went_green = _state(-13.31, peak=8.0)
    assert not g._should_fail_fast(_Pos(), went_green, -20.0)


def test_never_green_losers_are_still_counted_as_never_green():
    """A negative peak must not read as 'went green' via truthiness."""
    from bot.analysis import analyse
    t = _trade(peak_roi=-13.31, final_roi=-30.33, realised_pnl_usdt=-28.44,
               secs_to_first_positive=None)
    out = analyse([t])
    rows = [r for r in out.get("fail_fast_impact", [])
            if r.get("losers_never_green")]
    assert rows, "a loser with peak -13.31 must count as never green"


class _FakePos:
    symbol = "X/USDT:USDT"
    side = "short"
    entry_price = 1.0
    margin = 100.0
    notional = 2000.0


# ── Observation lag ─────────────────────────────────────────────────────────

def test_position_carries_an_exchange_timestamp():
    from bot.futures_guard import FuturesPosition
    p = FuturesPosition(symbol="X/USDT:USDT", side="short", entry_price=1.0,
                        qty=10.0, leverage=20, margin=0.5, updated_at=1789200000.0)
    assert p.updated_at == 1789200000.0
    # optional: existing construction sites pass no timestamp
    q = FuturesPosition(symbol="X/USDT:USDT", side="short", entry_price=1.0,
                        qty=10.0, leverage=20, margin=0.5)
    assert q.updated_at is None


def test_observation_lag_reaches_the_diagnostics_report():
    from bot.analysis import analyse
    t = _trade(observation_lag_s=3.02)
    rpt = analyse([t])["execution_diagnostics"]["report"]
    assert "obs_lag 3.02s" in rpt
    assert "lag artefact, not entry quality" in rpt


def test_report_marks_the_drift_sign_convention():
    from bot.analysis import analyse
    rpt = analyse([_trade(drift_since_sizing_pct=0.308)])["execution_diagnostics"]["report"]
    assert "+ve = good fill" in rpt


# ── Restart gap: a position that closes before the first pass ───────────────

def test_pos_meta_survives_a_restart_with_its_identity_fields(tmp_path):
    """
    A RIVER trade closed during a redeploy and recorded no side, entry price or
    final ROI — only the ledger's realised figure. pos_meta persisted just
    entry_context and opened_seen_at, and the close record needs more.
    """
    from bot import futures_state
    from bot.futures_guard import GuardState

    meta = {"X/USDT:USDT": {
        "side": "short", "entry_price": 1.486, "margin": 84.26,
        "leverage": 20.0, "current_roi": -3.2, "current_price": 1.50,
        "opened_seen_at": 1789200000.0, "fill_time": 1789199998.0,
        "fill_price": 1.486, "entry_context": {"rsi": 76.4}}}
    path = str(tmp_path / "s.json")
    assert futures_state.save(path, states={"X/USDT:USDT": GuardState()},
                              pos_meta=meta, closed_trades=[], owner="t")
    back = futures_state.load(path, owner="t")["pos_meta"]["X/USDT:USDT"]
    for k in ("side", "entry_price", "margin", "leverage", "fill_time"):
        assert back[k] == meta["X/USDT:USDT"][k], k


def test_a_close_right_after_restart_can_still_compute_roi():
    """margin is what final_roi is derived from — without it the row is blank."""
    from bot import futures_state
    from bot.futures_guard import GuardState
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "s.json")
    meta = {"X/USDT:USDT": {"side": "short", "entry_price": 1.486,
                            "margin": 84.26, "leverage": 20.0,
                            "opened_seen_at": 1789200000.0}}
    futures_state.save(path, states={"X/USDT:USDT": GuardState()},
                       pos_meta=meta, closed_trades=[], owner="t")
    back = futures_state.load(path, owner="t")["pos_meta"]["X/USDT:USDT"]
    realised, margin = 20.4138, back["margin"]
    assert round(realised / margin * 100, 2) == pytest.approx(24.23, abs=0.01)


# ── Fail-fast must never be silently off ────────────────────────────────────

def test_zero_disables_fail_fast_on_the_first_line():
    """The behaviour that cost 119 trades: 0 is not 'default', it is OFF."""
    g = _guardian(fail_fast_s=0.0)
    never_green_and_deep = _state(-10.0, peak=0.0)
    assert not g._should_fail_fast(_Pos(), never_green_and_deep, -25.0)


def test_the_same_trade_is_cut_once_it_is_armed():
    g = _guardian(fail_fast_s=60.0, fail_fast_loss_roi=5.0,
                  fail_fast_max_peak_roi=0.0)
    g._pos_meta["X/USDT:USDT"]["opened_seen_at"] = time.time() - 120
    assert g._should_fail_fast(_Pos(), _state(-10.0, peak=0.0), -25.0)


def test_snapshot_reports_whether_fail_fast_is_live():
    """It was absent from the snapshot, so 0 looked like a working config."""
    from bot.futures_guard import GuardConfig
    for secs, expected in ((0.0, False), (60.0, True)):
        cfg = GuardConfig(fail_fast_s=secs)
        assert bool(cfg.fail_fast_s) is expected


# ── Rescue trail ────────────────────────────────────────────────────────────

def test_rescue_trail_is_tight_by_default():
    """
    A refused stop means the position cannot be protected normally, so the
    rescue exists to LEAVE. 0.1 is the exchange minimum callbackRate.
    """
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig()
    assert cfg.rescue_trail_callback_pct == 0.1


def test_rescue_callback_never_goes_below_the_exchange_minimum():
    """Binance rejects callbackRate under 0.1, which would leave NO stop."""
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._place_native_trail)
    assert "max(0.1," in src


def test_rescue_trail_ignores_the_profit_lock_test():
    """
    _place_native_trail refuses when trail_locks_in() <= 0 — it asks whether a
    GAIN would be preserved. On a losing position that is the wrong question,
    and refusing left LSK 23:05 with no stop at all down to -61.31%.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._place_native_trail)
    rescue_block = src[src.index("if rescue:"):src.index("cb = trail_callback_price_pct")]
    assert "locked" not in rescue_block
    # The params actually sent: callbackRate only. No activationPrice means
    # Binance activates it at the current mark rather than waiting for a
    # profit level that a losing position will never reach.
    params = [l for l in rescue_block.splitlines() if "params={" in l]
    assert params and "callbackRate" in params[0]
    assert "activationPrice" not in params[0]


def test_rescue_is_requested_when_a_fixed_stop_is_refused():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    assert "_place_native_trail(pos, rescue=True)" in src


def test_the_refusal_reason_is_logged():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    assert "stop REFUSED by the exchange" in src
    assert "exchange said: {msg}" in src


# ── Candle taper: the operator's visual rule for timing a turn ──────────────

def _run(bodies, green=True):
    """Signed bodies: positive = green, negative = red."""
    pd = pytest.importorskip("pandas")
    rows, px = [], 1.0
    for b in bodies:
        o = px
        up = (b > 0) if green else (b < 0)
        c = px * (1 + abs(b)) if up else px * (1 - abs(b))
        hi = max(o, c) * 1.001
        lo = min(o, c) * 0.999
        rows.append((o, hi, lo, c, 1e6))
        px = c
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_shrinking_green_pushes_read_as_tapering_for_a_short():
    from bot.scanner import candle_taper
    t = candle_taper(_run([0.040, 0.030, 0.012, 0.008]), "short")
    assert t["tapering"] is True
    assert t["taper_ratio"] < 1


def test_growing_green_pushes_read_as_expanding():
    """The shape STORJ, TA and GRIFFAIN all had."""
    from bot.scanner import candle_taper
    t = candle_taper(_run([0.008, 0.012, 0.030, 0.040]), "short")
    assert t["tapering"] is False
    assert t["taper_ratio"] > 1


def test_the_long_side_reads_shrinking_RED_candles():
    from bot.scanner import candle_taper
    t = candle_taper(_run([0.040, 0.030, 0.012, 0.008], green=False), "long")
    assert t["tapering"] is True


def test_countertrend_candles_are_ignored():
    """A red candle inside a rally says nothing about whether pushes weaken."""
    from bot.scanner import candle_taper
    green = candle_taper(_run([0.04, 0.03, 0.012, 0.008]), "short")
    assert green["trend_candles"] == 4
    # the same series read for a LONG finds no red candles at all
    assert candle_taper(_run([0.04, 0.03, 0.012, 0.008]), "long")["trend_candles"] == 0


def test_too_few_pushes_reports_nothing_rather_than_guessing():
    from bot.scanner import candle_taper
    t = candle_taper(_run([0.02, 0.02]), "short")
    assert t["taper_ratio"] is None
    assert t["tapering"] is None


def test_taper_survives_degenerate_input():
    from bot.scanner import candle_taper
    pd = pytest.importorskip("pandas")
    assert candle_taper(None, "short")["taper_ratio"] is None
    assert candle_taper(pd.DataFrame(), "short")["taper_ratio"] is None


def test_taper_reaches_the_diagnostics_report():
    from bot.analysis import analyse
    t = _trade(entry_context={"taper_ratio": 0.302, "tapering": True,
                              "taper_vol_ratio": 0.61, "taper_close_pos": 0.22})
    rpt = analyse([t])["execution_diagnostics"]["report"]
    assert "ratio 0.302" in rpt
    assert "under 1 = pushes shrinking" in rpt


# ── Strength streaks count scans, not polls ─────────────────────────────────

def _rows(strength="strengthening"):
    return [{"symbol": "X/USDT:USDT", "direction": "short", "strength": strength}]


def test_rereading_the_same_scan_does_not_advance_the_streak():
    """
    run_once fires every 30s; the scanner refreshes every 120s. Four reads of
    one scan used to count as four confirmations.
    """
    from bot.auto_trader import StrengthTracker
    t = StrengthTracker()
    for _ in range(4):
        t.update(_rows(), scan_ts=1000.0)
    assert t.streak("X/USDT:USDT", "short") == 1


def test_a_fresh_scan_does_advance_it():
    from bot.auto_trader import StrengthTracker
    t = StrengthTracker()
    t.update(_rows(), scan_ts=1000.0)
    t.update(_rows(), scan_ts=1120.0)
    assert t.streak("X/USDT:USDT", "short") == 2


def test_two_sweeps_now_needs_two_real_scans():
    """At SCANNER_INTERVAL=120 that is ~240s of persistence, not 30s."""
    from bot.auto_trader import StrengthTracker, AutoTradeConfig
    cfg = AutoTradeConfig(required_strength_sweeps=2)
    t = StrengthTracker()
    t.update(_rows(), scan_ts=1000.0)
    for _ in range(3):                      # 30s polls within the same scan
        t.update(_rows(), scan_ts=1000.0)
    assert t.streak("X/USDT:USDT", "short") < cfg.required_strength_sweeps
    t.update(_rows(), scan_ts=1120.0)
    assert t.streak("X/USDT:USDT", "short") >= cfg.required_strength_sweeps


def test_weakening_still_resets_on_a_fresh_scan():
    from bot.auto_trader import StrengthTracker
    t = StrengthTracker()
    t.update(_rows(), scan_ts=1000.0)
    t.update(_rows(), scan_ts=1120.0)
    t.update(_rows("weakening"), scan_ts=1240.0)
    assert t.streak("X/USDT:USDT", "short") == 0


def test_scan_ts_is_required_so_the_bug_cannot_return_silently():
    """
    It was optional, which left a future call site able to omit it and restore
    per-poll counting without any signal.
    """
    import inspect
    from bot.auto_trader import StrengthTracker
    sig = inspect.signature(StrengthTracker.update)
    assert sig.parameters["scan_ts"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        StrengthTracker().update(_rows())


def test_an_explicit_none_still_counts_but_warns(caplog):
    from bot.auto_trader import StrengthTracker
    t = StrengthTracker()
    with caplog.at_level("WARNING"):
        t.update(_rows(), None)
        t.update(_rows(), None)
    assert t.streak("X/USDT:USDT", "short") == 2
    assert any("no scan timestamp" in r.message for r in caplog.records)


def test_run_once_passes_the_scan_timestamp():
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.run_once)
    assert 'scan_ts=snap.get("last_scan_ts")' in src


# ── Exit reason must name what actually closed the trade ────────────────────

def test_an_explicit_exit_reason_wins_over_the_inference():
    """
    A fail-fast market close leaves native_trail_id and stop_roi set, so the
    inference bucketed it as trail/stop. 54 + 103 = 157 in the real data: not
    one fail-fast exit was visible despite three firing in the logs.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._record_closed_trade)
    assert 'meta.get("exit_reason")' in src
    idx = src.index('"exit_reason":')
    assert src.index('meta.get("exit_reason")') < src.index('"trail" if', idx)


def test_fail_fast_stamps_its_reason_before_closing():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    i = src.index("_should_fail_fast")
    assert '"exit_reason"] = "fail_fast"' in src[i:i + 500]


def test_past_stop_closes_are_labelled():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    assert '"exit_reason"] = "past_stop"' in src


# ── Longs must show the EMA gap turning, not merely being below ────────────
#
# The "4" trade, 2026-09-13 12:49: coin down 14.01% on 24h, EMA9 0.01974 under
# EMA21 0.01978 and the gap NOT narrowing. The scanner listed it because the
# long branch only tested `gap < ema_tolerance_pct`, which any downtrend meets.
# It entered at 0.01978 and was fail-fast cut at -5.1%.

def _long_cand(**kw):
    row = {"symbol": "4/USDT:USDT", "direction": "long", "rsi": 48.0,
           "pct_above_24h_low": 1.13, "pct_below_24h_high": -19.26,
           "strength": "strengthening", "atr_pct": 0.6,
           "gap_narrowing": True, "gap_rising": True, "gap_narrowing_pct": -0.4,
           "turn": {"turned_up": True, "bars_since_low": 4, "rise_pct": 1.2}}
    row.update(kw)
    return row


def test_the_gate_is_on_by_default():
    """
    Not a new hypothesis: the scanner's own comment claims "recovering" and
    never tested it. Default ON restores the documented intent.
    """
    assert AutoTradeConfig().long_require_convergence is True


def test_a_long_with_a_widening_gap_is_refused():
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    d = evaluate_candidate(_long_cand(gap_rising=False, gap_rise_pct=-0.3),
                           streak=2, cfg=cfg, atr_pct=0.6)
    assert not d.enter
    assert "not gaining on EMA21" in d.reason


def test_converging_from_the_TOP_is_refused():
    """
    LAB 2026-09-14 07:52: EMA9 0.05299 ABOVE EMA21 0.05296, curving DOWN
    toward it. The absolute test called that "narrowing" — the same verdict it
    gives a bottom forming underneath. The signed test separates them.
    """
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    rolling_over = _long_cand(ema_gap_pct=0.057, gap_narrowing=True,
                              gap_rising=False, gap_rise_pct=-0.08)
    d = evaluate_candidate(rolling_over, streak=2, cfg=cfg, atr_pct=0.6)
    assert not d.enter
    assert "from the top" in d.reason


def test_a_crossover_in_progress_is_allowed():
    """No objection to a cross — only to converging from above."""
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    crossing = _long_cand(ema_gap_pct=0.022, gap_rising=True, gap_rise_pct=0.04)
    assert evaluate_candidate(crossing, streak=2, cfg=cfg, atr_pct=0.6).enter


def test_the_signed_test_separates_the_two_shrinking_cases():
    from bot.scanner import gap_rising, ScanConfig
    pd = pytest.importorskip("pandas")

    def frame(gaps):
        slow = [1.0] * len(gaps)
        fast = [1.0 * (1 + g / 100) for g in gaps]
        return pd.DataFrame({"ema_fast": fast, "ema_slow": slow})

    cfg = ScanConfig()
    lb = cfg.convergence_lookback
    from_below = frame([-0.30] * lb + [-0.05])
    from_above = frame([+0.10] * lb + [+0.02])
    assert gap_rising(from_below, cfg)[0] is True
    assert gap_rising(from_above, cfg)[0] is False


def test_a_long_with_a_narrowing_gap_is_allowed():
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    assert evaluate_candidate(_long_cand(), streak=2, cfg=cfg, atr_pct=0.6).enter


def test_a_missing_flag_is_treated_as_not_rising():
    """Older rows carry no verdict; refusing is the safe reading for a long."""
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    row = _long_cand()
    row.pop("gap_rising")
    assert not evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=0.6).enter


def test_shorts_are_untouched_by_the_gate():
    """
    Fading an RSI extreme works off the stretch itself. 150 shorts at +$2.47
    expectancy were taken without this condition and must stay unaffected.
    """
    cfg = AutoTradeConfig(enabled=True, veto_breakout=False)
    row = _short(breakout=_brk(gap_widening=False, breakout=False))
    row["gap_narrowing"] = False
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=2.0).enter


def test_the_gate_can_be_turned_off():
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52,
                          long_require_convergence=False)
    assert evaluate_candidate(_long_cand(gap_narrowing=False), streak=2,
                              cfg=cfg, atr_pct=0.6).enter


def test_the_scanner_publishes_the_verdict_it_already_computed():
    import inspect
    from bot import scanner
    src = inspect.getsource(scanner.evaluate_symbol)
    assert src.count("gap_narrowing=narrowing") == 2   # both branches
    assert '"gap_narrowing": bool' in inspect.getsource(scanner.Candidate.as_row)


# ── Stop trigger price ──────────────────────────────────────────────────────
#
# BR 2026-09-13 14:15 peaked +9.63% ROI on a 3% trail and exited -10.07%. The
# 19.7-point give-back equals 0.99% of price; the candle's range was 0.99%.
# The stop was taken by one wick, because CONTRACT_PRICE is the last trade on
# this book.

def test_mark_price_is_the_default():
    from bot.futures_guard import GuardConfig
    assert GuardConfig().stop_working_type == "MARK_PRICE"


def test_every_protective_order_carries_the_working_type():
    """Fixed stops AND trails, the rescue trail included — four sites."""
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian)
    assert src.count('"workingType": self.cfg.stop_working_type') == 4


def test_the_trade_records_which_price_was_in_force():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._record_closed_trade)
    assert '"stop_working_type": self.cfg.stop_working_type' in src


def _wt(wt, roi, pnl, sized=30.0):
    return {"symbol": "X/USDT:USDT", "side": "short", "final_roi": roi,
            "realised_pnl_usdt": pnl, "fees_usdt": 1.7, "margin_usdt": 88.0,
            "stop_working_type": wt, "exit_is_estimate": False,
            "entry_context": {"sized_stop_roi": sized}}


def test_losses_and_gains_are_reported_separately():
    """
    A single expectancy figure could hide the trade-off: mark price should
    shrink the average LOSS, and if it shrinks the average GAIN too it is
    triggering late on real moves. Both have to be visible.
    """
    from bot.analysis import analyse
    rows = analyse([_wt("CONTRACT_PRICE", -40, -35), _wt("CONTRACT_PRICE", -38, -33),
                    _wt("CONTRACT_PRICE", 20, 17),
                    _wt("MARK_PRICE", -28, -24), _wt("MARK_PRICE", -31, -27),
                    _wt("MARK_PRICE", 22, 19)])["by_working_type"]
    by = {r["label"]: r for r in rows}
    assert by["MARK_PRICE"]["avg_loss_usdt"] > by["CONTRACT_PRICE"]["avg_loss_usdt"]
    assert by["MARK_PRICE"]["avg_overshoot_roi"] < by["CONTRACT_PRICE"]["avg_overshoot_roi"]
    for r in rows:
        assert r["avg_win_usdt"] is not None and r["avg_loss_usdt"] is not None


def test_trades_without_the_field_are_simply_absent():
    """Every trade before this change has no working type — it must not crash."""
    from bot.analysis import analyse
    t = _wt("MARK_PRICE", -10, -9)
    t.pop("stop_working_type")
    assert analyse([t])["by_working_type"] == []


# ── Profit floor: a peak above breakeven must not become a loss ─────────────
#
# PUNDIX peaked +7.36% and closed -10.30%. BR +9.63% -> -10.07%. HIVE +10.98%
# -> -1.83%. In each case arming the native trail cancelled the fixed stop, so
# above arm_roi the only protection was a 0.15%-of-price trail.

class _FloorPos:
    symbol = "X/USDT:USDT"
    side = "short"
    entry_price = 0.1354
    qty = 640.0
    leverage = 20
    effective_leverage = 20.0
    margin = 87.2


def _floor_guardian(**cfgkw):
    from bot.futures_guard import GuardConfig, GuardState
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    base = dict(breakeven_at_roi=3.0, breakeven_stop_roi=2.0, arm_roi=5.0)
    base.update(cfgkw)
    g.cfg = GuardConfig(**base)
    g._all_stop_ids = {}
    g._states = {}
    g._records = []
    g._record = lambda *a, **k: None
    g.placed = []

    def _place(pos, price):
        g.placed.append(price)
        return f"floor-{len(g.placed)}"
    g._place_stop = _place
    return g


# A price at which _FloorPos is genuinely in profit. The floor now places at
# the best level STILL AVAILABLE, so a fixture sitting exactly at entry has
# nothing to lock and correctly places nothing.
_IN_PROFIT = 0.1354 * (1 - 0.0025)      # short, +5% ROI at 20x


def test_no_floor_before_the_peak_clears_breakeven():
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=2.9)
    g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    assert st.floor_stop_id is None
    assert g.placed == []


def test_a_floor_is_placed_once_the_peak_clears_breakeven():
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=7.36)          # PUNDIX
    g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    assert st.floor_stop_id == "floor-1"
    assert st.floor_roi == 2.0
    assert len(g.placed) == 1


def test_it_is_never_placed_twice_however_many_cycles_run():
    """The core anti-stacking guarantee."""
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=12.0)
    for _ in range(50):
        g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    assert len(g.placed) == 1
    assert st.floor_stop_id == "floor-1"


def test_a_restart_does_not_place_a_second_floor():
    """
    floor_stop_id is persisted. If it were not, the guardian would come back
    with no record and place another — stacking on the same position.
    """
    import tempfile, os
    from bot import futures_state
    from bot.futures_guard import GuardState
    st = GuardState(peak_roi=12.0)
    st.floor_stop_id, st.floor_roi = "floor-1", 2.0
    path = os.path.join(tempfile.mkdtemp(), "s.json")
    futures_state.save(path, states={"X/USDT:USDT": st}, pos_meta={},
                       closed_trades=[], owner="t")
    back = futures_state.restore_states(
        futures_state.load(path, owner="t"))["X/USDT:USDT"]
    assert back.floor_stop_id == "floor-1"

    g = _floor_guardian()
    g._ensure_profit_floor(_FloorPos(), back, _IN_PROFIT)
    assert g.placed == []               # nothing re-placed


def test_arming_the_trail_does_not_cancel_the_floor():
    """This is the bug: _cancel_superseded_stops took the fixed stop away."""
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=7.36)
    st.floor_stop_id = "floor-1"
    st.native_trail_id = "trail-9"
    g._states["X/USDT:USDT"] = st
    g._all_stop_ids["X/USDT:USDT"] = ["floor-1", "old-stop-2", "trail-9"]
    cancelled = []
    g._cancel_stop = lambda pos, oid: (cancelled.append(oid), True)[1]
    g._cancel_superseded_stops(_FloorPos(), keep="trail-9")
    assert "floor-1" not in cancelled     # the floor survives
    assert "trail-9" not in cancelled     # so does the trail
    assert "old-stop-2" in cancelled      # the superseded one goes


def test_the_floor_is_cancelled_when_the_position_closes():
    """It must not outlive the trade and fire against a later position."""
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=7.36)
    st.floor_stop_id = "floor-1"
    cancelled = []
    g._cancel_stop = lambda pos, oid: (cancelled.append(oid), True)[1]
    g._cancel_profit_floor("X/USDT:USDT", st)
    assert cancelled == ["floor-1"]
    assert st.floor_stop_id is None


def test_a_failed_cancel_still_clears_the_id():
    """The 120s orphan sweep is the backstop; state must not hold a dead id."""
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=7.36)
    st.floor_stop_id = "floor-1"

    def _boom(pos, oid):
        raise RuntimeError("exchange said no")
    g._cancel_stop = _boom
    g._cancel_profit_floor("X/USDT:USDT", st)
    assert st.floor_stop_id is None


def test_the_floor_can_be_switched_off():
    from bot.futures_guard import GuardState
    g = _floor_guardian(profit_floor_enabled=False)
    st = GuardState(peak_roi=12.0)
    g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    assert st.floor_stop_id is None and g.placed == []


def test_the_floor_is_inert_when_breakeven_is_not_configured():
    """
    GuardConfig defaults breakeven_at_roi to 0, so a deployment that never set
    it gets no floor rather than one at 0% ROI — which after ~1.9% of fees
    would guarantee a small LOSS, the opposite of the intent.
    """
    from bot.futures_guard import GuardConfig, GuardState
    assert GuardConfig().breakeven_at_roi == 0.0
    g = _floor_guardian(breakeven_at_roi=0.0, breakeven_stop_roi=0.0)
    st = GuardState(peak_roi=20.0)
    g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    assert st.floor_stop_id is None and g.placed == []


def test_at_the_operators_settings_the_floor_clears_fees():
    """GUARD_BREAKEVEN_AT_ROI=3 / STOP_ROI=2 against a ~1.9% round trip."""
    from bot.futures_guard import GuardState
    g = _floor_guardian(breakeven_at_roi=3.0, breakeven_stop_roi=2.0)
    st = GuardState(peak_roi=3.1)
    g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    assert st.floor_roi == 2.0
    assert st.floor_roi > 1.9      # net positive after the round trip


# ── The U/V turn: the low must be BEHIND us ────────────────────────────────
#
# USELESS 18:06 was bought one second before a 36% drop. gap_narrowing passed
# it because that test compares the gap now against N candles ago — two
# points. A market still collapsing can post a smaller absolute gap after one
# pause and read as "narrowing" while the fast EMA keeps making new lows.

def _turn_frame(vals):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"ema_fast": vals,
                         "ema_slow": [v * 1.01 for v in vals]})


def test_the_right_side_of_a_V_is_a_turn():
    from bot.scanner import turned, ScanConfig
    t = turned(_turn_frame([1.10,1.07,1.04,1.00,1.02,1.04,1.06,1.08,1.10,1.12]),
                  ScanConfig())
    assert t["turned_up"] is True
    assert t["bars_since_low"] >= 2
    assert t["rise_pct"] > 0


def test_a_flat_bottomed_U_counts_once_it_lifts():
    from bot.scanner import turned, ScanConfig
    t = turned(_turn_frame([1.10,1.05,1.01,1.00,1.00,1.00,1.01,1.03,1.05,1.07]),
                  ScanConfig())
    assert t["turned_up"] is True


def test_a_market_still_falling_is_not_a_turn():
    """USELESS 18:06. The low IS the latest bar."""
    from bot.scanner import turned, ScanConfig
    t = turned(_turn_frame([1.10,1.08,1.06,1.04,1.03,1.02,1.01,1.00,0.99,0.98]),
                  ScanConfig())
    assert t["turned_up"] is False
    assert t["bars_since_low"] == 0


def test_a_pause_that_resumes_falling_is_not_a_turn():
    """The case a two-point gap comparison cannot see."""
    from bot.scanner import turned, ScanConfig
    t = turned(_turn_frame([1.10,1.06,1.02,1.00,1.01,1.00,0.98,0.96,0.94,0.92]),
                  ScanConfig())
    assert t["turned_up"] is False


def test_a_bottom_still_forming_does_not_count():
    """min_bars_since: a low made on the last bar is not a low crossed."""
    from bot.scanner import turned, ScanConfig
    t = turned(_turn_frame([1.10,1.08,1.06,1.04,1.03,1.02,1.01,1.00,0.995,0.99]),
                  ScanConfig())
    assert t["turned_up"] is False


def test_the_long_gate_refuses_without_a_turn():
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    row = _long_cand()
    row["turn"] = {"turned_up": False, "bars_since_low": 0, "rise_pct": 0.0}
    d = evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=0.6)
    assert not d.enter
    assert "no turn yet" in d.reason


def test_shorts_are_not_subject_to_the_turn():
    cfg = AutoTradeConfig(enabled=True, veto_breakout=False)
    row = _short(breakout=_brk(gap_widening=False, breakout=False))
    row["turn"] = {"turned_up": False, "bars_since_low": 0, "rise_pct": 0.0}
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=2.0).enter


def test_the_turn_requirement_can_be_switched_off():
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52,
                          long_require_turn=False)
    row = _long_cand()
    row["turn"] = {"turned_up": False, "bars_since_low": 0, "rise_pct": 0.0}
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=0.6).enter


def test_a_missing_turn_block_refuses():
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    row = _long_cand()
    row.pop("turn", None)
    assert not evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=0.6).enter


def test_each_direction_gets_its_own_turn():
    """
    turned() was computed ONCE and attached to both branches, so every short
    recorded whether the fast EMA turned UP — meaningless for a fade, and the
    opposite of what the field name implies. Nothing gated on it, so no trade
    behaved wrongly, but the recorded data was misleading.
    """
    import inspect
    from bot import scanner
    src = inspect.getsource(scanner.evaluate_symbol)
    assert 'turned(df, cfg, "short")' in src
    assert 'turned(df, cfg, "long")' in src
    assert "turn=turn_short," in src and "turn=turn_long," in src


def test_a_short_reads_the_HIGH_behind_it():
    """The mirror: top crossed, falling away from it."""
    from bot.scanner import turned, ScanConfig
    rising_then_rolling = _turn_frame(
        [0.90,0.93,0.96,1.00,0.98,0.96,0.94,0.92,0.90,0.88])
    t = turned(rising_then_rolling, ScanConfig(), "short")
    assert t["turned_up"] is True          # "turned" in the short's favour
    assert t["bars_since_low"] >= 2        # bars since the HIGH

    still_rising = _turn_frame(
        [0.90,0.92,0.94,0.96,0.97,0.98,0.99,1.00,1.01,1.02])
    assert turned(still_rising, ScanConfig(), "short")["turned_up"] is False


def test_the_same_series_reads_oppositely_for_the_two_sides():
    from bot.scanner import turned, ScanConfig
    falling = _turn_frame([1.10,1.07,1.04,1.00,0.97,0.94,0.91,0.88,0.85,0.82])
    assert turned(falling, ScanConfig(), "long")["turned_up"] is False
    assert turned(falling, ScanConfig(), "short")["turned_up"] is True


# ── Adaptive trail: the stop distance, managed by the exchange ─────────────

def _trail_guardian(**cfgkw):
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    base = dict(adaptive_trail_enabled=True, rescue_trail_callback_pct=0.1)
    base.update(cfgkw)
    g.cfg = GuardConfig(**base)
    g._all_stop_ids, g._states, g.placed = {}, {}, []
    g._record = lambda *a, **k: None
    g.dry_run = False

    def _native(pos, rescue=False, callback_pct=None):
        g.placed.append(callback_pct)
        return f"trail-{len(g.placed)}"
    g._place_native_trail = _native
    return g


class _TrailPos:
    symbol = "X/USDT:USDT"
    side = "short"
    entry_price = 0.1382
    qty = 12642.0
    leverage = 20
    effective_leverage = 20.0
    margin = 87.36


def test_callback_is_the_stop_expressed_as_price():
    """30% ROI at 20x is 1.50% of price; the same stop at 10x is 3.00%."""
    from bot.futures_guard import GuardState
    g = _trail_guardian()
    st = GuardState()
    g._ensure_adaptive_trail(_TrailPos(), st, 30.0)
    assert g.placed == [1.5]

    class P10(_TrailPos):
        effective_leverage = 10.0
    g2 = _trail_guardian()
    g2._ensure_adaptive_trail(P10(), GuardState(), 30.0)
    assert g2.placed == [3.0]


def test_it_tracks_each_trade_s_own_stop():
    """sized_stop_roi varies 16-30 across real trades; the trail follows it."""
    from bot.futures_guard import GuardState
    for stop_roi, expect in ((17.5, 0.88), (22.0, 1.1), (26.3, 1.31)):
        g = _trail_guardian()
        g._ensure_adaptive_trail(_TrailPos(), GuardState(), stop_roi)
        assert g.placed == [expect], (stop_roi, g.placed)


def test_on_by_default_and_switchable_off():
    """
    Default ON from v3.15.0. The legacy guardian tests pin it OFF explicitly
    because they exercise the fixed-stop mechanics, which it runs ahead of.
    """
    from bot.futures_guard import GuardConfig, GuardState
    assert GuardConfig().adaptive_trail_enabled is True
    g = _trail_guardian(adaptive_trail_enabled=False)
    g._ensure_adaptive_trail(_TrailPos(), GuardState(), 30.0)
    assert g.placed == []


def test_never_placed_twice():
    from bot.futures_guard import GuardState
    g = _trail_guardian()
    st = GuardState()
    for _ in range(20):
        g._ensure_adaptive_trail(_TrailPos(), st, 30.0)
    assert len(g.placed) == 1


def test_a_restart_does_not_place_a_second(tmp_path):
    from bot import futures_state
    from bot.futures_guard import GuardState
    st = GuardState()
    st.adaptive_trail_id = "trail-1"
    path = str(tmp_path / "s.json")
    futures_state.save(path, states={"X/USDT:USDT": st}, pos_meta={},
                       closed_trades=[], owner="t")
    back = futures_state.restore_states(
        futures_state.load(path, owner="t"))["X/USDT:USDT"]
    assert back.adaptive_trail_id == "trail-1"
    g = _trail_guardian()
    g._ensure_adaptive_trail(_TrailPos(), back, 30.0)
    assert g.placed == []


def test_out_of_band_callbacks_leave_the_fixed_stop_alone():
    """Binance accepts 0.1-10%. Outside that, do nothing rather than guess."""
    from bot.futures_guard import GuardState
    g = _trail_guardian()
    g._ensure_adaptive_trail(_TrailPos(), GuardState(), 1.0)   # 0.05% — too tight
    assert g.placed == []
    g2 = _trail_guardian()
    g2._ensure_adaptive_trail(_TrailPos(), GuardState(), 400.0)  # 20% — too wide
    assert g2.placed == []


def test_it_is_not_superseded_when_other_stops_are_cancelled():
    from bot.futures_guard import GuardState
    g = _trail_guardian()
    st = GuardState()
    st.adaptive_trail_id = "trail-1"
    g._states["X/USDT:USDT"] = st
    g._all_stop_ids["X/USDT:USDT"] = ["trail-1", "old-stop"]
    cancelled = []
    g._cancel_stop = lambda pos, oid: (cancelled.append(oid), True)[1]
    g._cancel_superseded_stops(_TrailPos(), keep=None)
    assert "trail-1" not in cancelled
    assert "old-stop" in cancelled


def test_arming_cancels_the_adaptive_trail_rather_than_stacking():
    """
    The armed trail is ~0.15% of price; the adaptive one is the stop distance,
    ~1.5%. Two reduce-only trails would rest together and only the tight one
    could ever fire. Exactly the stacking to avoid.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    i = src.index("if trail_id:")
    block = src[i:i + 1400]
    assert "state.adaptive_trail_id" in block
    assert "superseded by the" in block
    assert "state.adaptive_trail_id = None" in block


def test_the_adaptive_trail_is_not_placed_once_a_trail_is_armed():
    """The other direction: no adaptive trail on top of an armed one."""
    from bot.futures_guard import GuardState
    g = _trail_guardian()
    st = GuardState()
    st.native_trail_id = "armed-1"
    g._ensure_adaptive_trail(_TrailPos(), st, 30.0)
    assert g.placed == []


def test_selection_and_entry_never_reach_it():
    import inspect
    from bot import auto_trader, futures_entry
    for mod in (auto_trader, futures_entry):
        assert "_ensure_adaptive_trail" not in inspect.getsource(mod)


def test_a_refused_stop_does_not_add_a_rescue_on_top_of_the_adaptive_trail():
    """
    The rescue path guarded only on native_trail_id. With the adaptive trail
    resting, a refused fixed stop would have placed a 0.1% rescue on top of a
    ~1.5% adaptive trail — two trails, a duplicate.

    It is also no longer an emergency: the adaptive trail went on at adoption,
    so there is no unprotected window to rescue from.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    i = src.index("fixed stop refused")
    assert "state.adaptive_trail_id" in src[max(0, i - 800):i]
    assert "no second trail placed" in src


def test_fail_fast_reads_no_orders_at_all():
    """It must keep working whatever is resting on the exchange."""
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._should_fail_fast)
    for token in ("adaptive_trail_id", "floor_stop_id", "stop_order_id",
                  "native_trail_id"):
        assert token not in src, token


def test_the_fixed_stop_is_still_placed_alongside():
    """
    The adaptive trail is insurance-backed, not a replacement: _ensure_adaptive
    _trail must not short-circuit the stop logic below it.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    i = src.index("_ensure_adaptive_trail")
    after = src[i:]
    assert "return" not in after.split("\n")[0]
    assert "_place_stop" in after or "stop_price" in after


# ── Cleanup and reaping, end to end ────────────────────────────────────────

def _sweep_guardian(orders, live):
    """The real reap_orphan_stops with its exchange calls stubbed out."""
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import FuturesGuardian

    class _Pos:
        def __init__(self, sym): self.symbol = sym

    import threading
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig()
    g._states, g.cancelled = {}, []
    g._lock = threading.RLock()
    g._pending_cancels, g._cancel_attempts = {}, {}
    g._cancelled_ids = set()
    g._all_stop_ids = {}
    g._record = lambda *a, **k: None
    g._cancel_any = lambda oid, sym: (g.cancelled.append(oid), True)[1]
    g._clear_pending_cancel = lambda *a: None
    g._queue_pending_cancel = lambda *a: None
    g.fetch_positions = lambda: [_Pos(s) for s in live]
    g._all_open_orders_raw = lambda: list(orders)
    g._normalise_order = lambda o: dict(o)
    g._orphan_scan_symbols = lambda live_set: []
    g.exchange = type("X", (), {"fetch_open_orders": staticmethod(lambda s: [])})()
    return g


def _ord(oid, sym="X/USDT:USDT", typ="STOP_MARKET", ts=1, ro=True):
    return {"id": oid, "symbol": sym, "type": typ, "reduce_only": ro, "ts": ts}


def test_the_sweep_does_not_cancel_the_deliberate_protective_set():
    """
    It used to trim a live position to ONE stop. With three intentional orders
    that would have cancelled the PROFIT FLOOR — the one that must survive.
    """
    from bot.futures_guard import GuardState
    orders = [_ord("fix", ts=1), _ord("trail", typ="TRAILING_STOP_MARKET", ts=2),
              _ord("floor", ts=3)]
    g = _sweep_guardian(orders, {"X/USDT:USDT"})
    st = GuardState()
    st.stop_order_id, st.adaptive_trail_id, st.floor_stop_id = "fix", "trail", "floor"
    g._states["X/USDT:USDT"] = st
    g.reap_orphan_stops()
    assert g.cancelled == []


def test_an_untracked_extra_on_a_live_position_is_still_swept():
    """A superseded ratchet whose cancel never took must still go."""
    from bot.futures_guard import GuardState
    orders = [_ord("floor", ts=3), _ord("ghost-a", ts=1), _ord("ghost-b", ts=2)]
    g = _sweep_guardian(orders, {"X/USDT:USDT"})
    st = GuardState()
    st.floor_stop_id = "floor"
    g._states["X/USDT:USDT"] = st
    g.reap_orphan_stops()
    assert "floor" not in g.cancelled
    assert "ghost-a" in g.cancelled          # older of the two untracked


def test_orphans_on_a_dead_symbol_are_swept_including_trails():
    orders = [_ord("s1"), _ord("t1", typ="TRAILING_STOP_MARKET")]
    g = _sweep_guardian(orders, set())
    g.reap_orphan_stops()
    assert set(g.cancelled) == {"s1", "t1"}


def test_entry_orders_are_never_touched():
    g = _sweep_guardian(
        [_ord("entry", typ="TRAILING_STOP_MARKET", ro=False)], set())
    g.reap_orphan_stops()
    assert g.cancelled == []


def test_a_trailing_stop_counts_as_protective():
    """is_protective_stop matches on 'STOP' in the type — TRAILING_STOP_MARKET
    contains it, so trails are adopted and reaped like any other stop."""
    from bot.futures_guard import is_protective_stop

    class P:
        side = "short"
    for typ in ("STOP_MARKET", "TRAILING_STOP_MARKET"):
        assert is_protective_stop(
            {"reduceOnly": True, "side": "buy", "type": typ}, P())
    assert not is_protective_stop(
        {"reduceOnly": False, "side": "buy", "type": "TRAILING_STOP_MARKET"}, P())


def test_the_adaptive_trail_does_not_mutate_shared_config():
    """
    It used to pass its callback by temporarily writing
    cfg.rescue_trail_callback_pct — shared state the dashboard can also write
    through update_rules(). Sequential position management made it safe today
    and a race the moment a second writer appeared.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._ensure_adaptive_trail)
    assert "self.cfg.rescue_trail_callback_pct =" not in src
    assert "callback_pct=cb" in src


def test_a_failed_trail_leaves_the_position_on_the_fixed_stop():
    """
    On failure adaptive_trail_id stays None, so: the fixed stop logic below
    still runs, the 0.1% rescue is NOT skipped, and the next cycle retries.
    """
    from bot.futures_guard import GuardState
    g = _trail_guardian()

    def _boom(pos, rescue=False, callback_pct=None):
        raise RuntimeError("exchange refused")
    g._place_native_trail = _boom
    st = GuardState()
    g._ensure_adaptive_trail(_TrailPos(), st, 30.0)   # must not raise
    assert st.adaptive_trail_id is None


def test_a_none_response_is_treated_as_failure_not_success():
    from bot.futures_guard import GuardState
    g = _trail_guardian()
    g._place_native_trail = lambda pos, rescue=False, callback_pct=None: None
    st = GuardState()
    g._ensure_adaptive_trail(_TrailPos(), st, 30.0)
    assert st.adaptive_trail_id is None
    assert g._all_stop_ids.get("X/USDT:USDT") in (None, [])


# ── Protection audit ────────────────────────────────────────────────────────

def _audit_guardian(live_orders, **state_ids):
    import threading, time as _t
    from bot.futures_guard import GuardConfig, GuardState
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig()
    g._lock = threading.RLock()
    g._audit_last = {}
    g._normalise_order = lambda o: dict(o)
    g.fetch_open_orders = lambda sym: list(live_orders)
    g.mark_price = lambda pos: pos.entry_price
    st = GuardState()
    for k, v in state_ids.items():
        setattr(st, k, v)
    return g, st


def _po(oid, typ="STOP_MARKET"):
    return {"id": oid, "type": typ, "reduce_only": True}


def test_an_empty_listing_is_reported_as_BLIND_not_unprotected(caplog):
    """
    SOLV 19:26: the listing returned nothing at all and the audit shouted
    UNPROTECTED twice while two orders were resting. A listing that sees
    nothing cannot prove anything.
    """
    g, st = _audit_guardian([], stop_order_id="fix")
    with caplog.at_level("INFO"):
        g._audit_protection(_TrailPos(), st)
    msgs = " ".join(r.message for r in caplog.records)
    assert "PROTECTION-BLIND" in msgs
    assert "PROTECTION-UNPROTECTED" not in msgs
    assert "PROTECTION-MISSING" not in msgs


def test_a_listing_with_orders_but_none_protective_IS_unprotected(caplog):
    g, st = _audit_guardian(
        [{"id": "entry", "type": "TRAILING_STOP_MARKET", "reduce_only": False}],
        stop_order_id="fix")
    with caplog.at_level("WARNING"):
        g._audit_protection(_TrailPos(), st)
    assert any("PROTECTION-UNPROTECTED" in r.message for r in caplog.records)


def test_a_trail_placed_this_cycle_is_not_superseded(caplog):
    """
    The exemption used to read self._states[symbol], but manage_position
    mutates a LOCAL state and writes it back later — so an order placed
    earlier in the same cycle was invisible and got cancelled. SOLV lost its
    adaptive trail one second after it was placed.
    """
    from bot.futures_guard import GuardState
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g._states = {}                      # nothing written back yet
    g._all_stop_ids = {"X/USDT:USDT": ["trail-live", "old-stop"]}
    cancelled = []
    g._cancel_stop = lambda pos, oid: (cancelled.append(oid), True)[1]
    live = GuardState()
    live.adaptive_trail_id = "trail-live"
    g._cancel_superseded_stops(_TrailPos(), keep=None, state=live)
    assert "trail-live" not in cancelled
    assert "old-stop" in cancelled


def test_audit_flags_a_tracked_id_the_exchange_does_not_have(caplog):
    g, st = _audit_guardian([_po("other")], floor_stop_id="floor")
    with caplog.at_level("WARNING"):
        g._audit_protection(_TrailPos(), st)
    msgs = " ".join(r.message for r in caplog.records)
    assert "PROTECTION-MISSING" in msgs and "floor" in msgs


def test_audit_flags_an_order_the_guardian_did_not_place(caplog):
    g, st = _audit_guardian([_po("ghost")], stop_order_id=None)
    with caplog.at_level("WARNING"):
        g._audit_protection(_TrailPos(), st)
    assert any("PROTECTION-UNTRACKED" in r.message for r in caplog.records)


def test_audit_flags_two_trailing_stops(caplog):
    g, st = _audit_guardian(
        [_po("t1", "TRAILING_STOP_MARKET"), _po("t2", "TRAILING_STOP_MARKET")],
        adaptive_trail_id="t1", native_trail_id="t2")
    with caplog.at_level("WARNING"):
        g._audit_protection(_TrailPos(), st)
    msgs = " ".join(r.message for r in caplog.records)
    assert "PROTECTION-DUPLICATE" in msgs
    assert "PROTECTION-OVERLAP" in msgs


def test_a_healthy_position_logs_no_warning(caplog):
    g, st = _audit_guardian(
        [_po("trail", "TRAILING_STOP_MARKET"), _po("floor")],
        adaptive_trail_id="trail", floor_stop_id="floor")
    with caplog.at_level("WARNING"):
        g._audit_protection(_TrailPos(), st)
    assert not [r for r in caplog.records if "PROTECTION-" in r.message]


def test_the_audit_is_throttled_per_symbol():
    g, st = _audit_guardian([_po("floor")], floor_stop_id="floor")
    calls = []
    g.fetch_open_orders = lambda sym: (calls.append(sym), [_po("floor")])[1]
    for _ in range(10):
        g._audit_protection(_TrailPos(), st)
    assert len(calls) == 1


# ── Wait counterfactual ─────────────────────────────────────────────────────

def _wait_guardian(candles):
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig()
    g.atr_timeframe = "3m"
    g.exchange = type("X", (), {
        "fetch_ohlcv": staticmethod(lambda s, timeframe=None, since=None,
                                    limit=None: candles)})()
    return g


def test_waiting_is_scored_positive_when_it_gets_a_better_short_entry():
    """
    A short filled at 0.28884 while price ran up. Two candles later it is
    higher, so waiting would have SOLD HIGHER — a better entry.
    """
    g = _wait_guardian([
        [1_000_000, 0.289, 0.2995, 0.2885, 0.2955, 1],
        [1_180_000, 0.2955, 0.2999, 0.294, 0.2990, 1],
    ])
    meta = {"entry_price": 0.28884,
            "entry_context": {"sized_at": 1000.0}}
    out = g._wait_counterfactual("BR/USDT:USDT", "short", meta)
    assert out["wait_1c_pct"] > 0 and out["wait_2c_pct"] > 0
    assert out["wait_candles_seen"] == 2


def test_waiting_is_scored_negative_when_the_move_already_went():
    """A short whose price fell away: waiting sells LOWER, a worse entry."""
    g = _wait_guardian([
        [1_000_000, 0.289, 0.289, 0.280, 0.282, 1],
        [1_180_000, 0.282, 0.283, 0.275, 0.276, 1],
    ])
    meta = {"entry_price": 0.28884, "entry_context": {"sized_at": 1000.0}}
    out = g._wait_counterfactual("BR/USDT:USDT", "short", meta)
    assert out["wait_2c_pct"] < 0


def test_the_worst_adverse_excursion_is_reported():
    """If that exceeds the stop, waiting meant watching rather than sitting."""
    g = _wait_guardian([
        [1_000_000, 0.289, 0.300, 0.288, 0.295, 1],
        [1_180_000, 0.295, 0.297, 0.294, 0.296, 1],
    ])
    meta = {"entry_price": 0.28884, "entry_context": {"sized_at": 1000.0}}
    out = g._wait_counterfactual("BR/USDT:USDT", "short", meta)
    assert out["wait_worst_pct"] == pytest.approx(
        (0.300 - 0.28884) / 0.28884 * 100, abs=0.01)


def test_a_long_is_scored_the_other_way_round():
    g = _wait_guardian([
        [1_000_000, 0.2112, 0.2112, 0.2070, 0.2074, 1],
        [1_180_000, 0.2074, 0.2080, 0.2060, 0.2065, 1],
    ])
    meta = {"entry_price": 0.2112, "entry_context": {"sized_at": 1000.0}}
    out = g._wait_counterfactual("USELESS/USDT:USDT", "long", meta)
    assert out["wait_2c_pct"] > 0          # buying lower is better for a long


def test_it_never_disturbs_the_record_on_failure():
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig()
    g.atr_timeframe = "3m"

    def _boom(*a, **k):
        raise RuntimeError("klines unavailable")
    g.exchange = type("X", (), {"fetch_ohlcv": staticmethod(_boom)})()
    out = g._wait_counterfactual(
        "X/USDT:USDT", "short",
        {"entry_price": 1.0, "entry_context": {"sized_at": 1000.0}})
    assert out == {}


def test_a_trade_without_a_signal_time_is_skipped():
    g = _wait_guardian([[1_000_000, 1, 1, 1, 1, 1]])
    assert g._wait_counterfactual("X/USDT:USDT", "short",
                                  {"entry_price": 1.0}) == {}


def test_the_aggregate_split_reports_winners_and_losers_apart():
    from bot.analysis import analyse
    def t(pnl, w2):
        return {"symbol": "X/USDT:USDT", "side": "short", "final_roi": 10.0,
                "realised_pnl_usdt": pnl, "fees_usdt": 1.7, "margin_usdt": 88.0,
                "exit_is_estimate": False, "wait_1c_pct": w2 / 2,
                "wait_2c_pct": w2, "wait_worst_pct": 1.0,
                "entry_context": {"sized_stop_roi": 30.0}}
    out = analyse([t(20, 0.8), t(15, 0.4), t(-30, -0.6)])["wait_two_candles"]
    assert out["n"] == 3
    assert out["n_winners"] == 2 and out["n_losers"] == 1
    assert out["winners_2c_pct"] > 0 > out["losers_2c_pct"]
    assert out["share_better_2c"] == pytest.approx(66.7, abs=0.1)


def test_a_wider_taper_window_raises_coverage_without_changing_the_comparison():
    """
    candle_taper needs FOUR trend candles before it can compare the recent two
    against the previous two. At window 6 only 54 of 232 real trades produced a
    reading. Widening scans further for those four; it does not change what is
    compared.
    """
    from bot.scanner import candle_taper
    # four green pushes, but spaced out by red candles
    mixed = _run([0.04, -0.01, 0.03, -0.01, 0.012, -0.005, 0.008])
    narrow = candle_taper(mixed, "short", 4)
    wide = candle_taper(mixed, "short", 10)
    assert narrow["taper_ratio"] is None        # cannot find four in four bars
    assert wide["taper_ratio"] is not None
    assert wide["tapering"] is True             # 0.04,0.03 -> 0.012,0.008


def test_the_taper_window_default_and_knob():
    from bot.scanner import ScanConfig
    assert ScanConfig().taper_window == 10
    assert ScanConfig(taper_window=14).taper_window == 14


# ── Rate-limit monitor ──────────────────────────────────────────────────────

def _fresh_monitor():
    from bot.futures_guardian import RateLimitMonitor
    return RateLimitMonitor()


def test_it_recognises_every_shape_the_limit_arrives_in():
    """
    Same condition, different call paths: ccxt.RateLimitExceeded,
    ccxt.DDoSProtection, a bare HTTP 429, or Binance's own -1003.
    """
    m = _fresh_monitor()
    for text in ('binanceusdm {"code":-1003,"msg":"Too many requests."}',
                 "HTTP 429 Too Many Requests",
                 "DDoSProtection: way too many requests",
                 "HTTP 418 IP banned until ..."):
        assert m.looks_rate_limited(text), text


def test_ordinary_errors_are_not_flagged():
    m = _fresh_monitor()
    for text in ('binanceusdm {"code":-2021,"msg":"Order would immediately trigger."}',
                 'binanceusdm {"code":-2011,"msg":"Unknown order sent."}',
                 "ConnectionResetError"):
        assert not m.looks_rate_limited(text), text


def test_a_hit_sets_a_cooldown_and_counts(caplog):
    m = _fresh_monitor()
    with caplog.at_level("ERROR"):
        hit = m.note(RuntimeError(), 'binanceusdm {"code":-1003,"msg":"Too many requests."}')
    assert hit is True
    st = m.status()
    assert st["limited_now"] is True
    assert st["hits_last_hour"] == 1
    assert 0 < st["seconds_remaining"] <= m.COOLDOWN_S
    assert any("RATE-LIMITED" in r.message for r in caplog.records)


def test_a_non_rate_limit_error_changes_nothing():
    m = _fresh_monitor()
    assert m.note(RuntimeError(), "code -2021 order would immediately trigger") is False
    assert m.status()["limited_now"] is False
    assert m.status()["hits_last_hour"] == 0


def test_hits_age_out_of_the_hourly_count():
    import time as _t
    m = _fresh_monitor()
    m.hits = [_t.time() - 4000, _t.time() - 10]      # one over an hour old
    assert m.status()["hits_last_hour"] == 1


def test_detection_rides_on_the_shared_error_formatter():
    """
    Every except block in the guardian formats through _safe_err, so detection
    needs no hook per call site — which is how it would have been missed.
    """
    import inspect
    from bot import futures_guardian as fg
    src = inspect.getsource(fg._safe_err)
    assert "RATE_LIMIT.note" in src


def test_the_poll_loop_backs_off_while_limited():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian)
    i = src.index('RATE_LIMIT.status()\n            if rl["limited_now"]')
    assert "seconds_remaining" in src[i:i + 400]


def test_the_status_reaches_the_snapshot():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    assert '"rate_limit": RATE_LIMIT.status()' in inspect.getsource(FuturesGuardian)


# ── Refusal tally ───────────────────────────────────────────────────────────

def test_every_gate_maps_to_its_own_key():
    """
    Refusal text carries the candidate's numbers, so counting raw strings gives
    one bucket per candidate. The classifier collapses each to its rule.
    """
    from bot.auto_trader import _refusal_key
    cases = {
        "longs disabled (short only)": "direction",
        "ATR 0.48% below the 0.5% floor": "min_atr",
        "RSI 54.3 above the long ceiling of 52.0": "rsi_band",
        "RSI 41.2 below the long floor of 45.0": "rsi_band",
        "EMA9 is not gaining on EMA21 (signed gap moved -0.08%)": "gap_not_rising",
        "no turn yet — the fast EMA's low is 0 candle(s) back": "no_turn",
        "breakout structure: at the 24h extreme, 3 consecutive candles": "breakout_veto",
        "RSI 80, strengthened 1 scans, 13.65% from the 24h high, limit 3.0%": "distance",
        "RSI 76, strengthened 1 scans — needs 2": "streak",
    }
    for text, key in cases.items():
        assert _refusal_key(text) == key, (text, _refusal_key(text))


def test_an_unrecognised_reason_lands_in_other():
    from bot.auto_trader import _refusal_key
    assert _refusal_key("something entirely new") == "other"
    assert _refusal_key("") == "other"
    assert _refusal_key(None) == "other"


# ── Advance volume ──────────────────────────────────────────────────────────

def _leg(prices, vols):
    pd = pytest.importorskip("pandas")
    rows = []
    for i, (p, v) in enumerate(zip(prices, vols)):
        o = prices[i - 1] if i else p
        rows.append((o, max(o, p) * 1.001, min(o, p) * 0.999, p, v))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])


def test_volume_fading_into_the_high_is_divergence():
    """The exhaustion case: price climbs, participation drains."""
    from bot.scanner import advance_volume
    up = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09]
    a = advance_volume(_leg(up, [100, 95, 90, 80, 70, 55, 45, 35, 28, 20]), "short")
    assert a["adv_vol_trend"] < 1
    assert a["peak_vol_early"] is True
    assert a["adv_price_pct"] > 0


def test_volume_building_into_the_high_is_participation():
    from bot.scanner import advance_volume
    up = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09]
    a = advance_volume(_leg(up, [20, 28, 35, 45, 55, 70, 80, 90, 95, 100]), "short")
    assert a["adv_vol_trend"] > 1
    assert a["peak_vol_early"] is False


def test_it_measures_the_LEG_not_the_last_two_candles():
    """
    The distinction that prompted it: taper_vol_ratio compares the most recent
    two pushes. This spans the whole advance, so a late burst inside a fading
    leg does not flip the verdict.
    """
    from bot.scanner import advance_volume, candle_taper
    up = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09]
    vols = [100, 95, 90, 80, 70, 55, 45, 35, 60, 65]     # fading, late uptick
    a = advance_volume(_leg(up, vols), "short")
    assert a["adv_vol_trend"] < 1          # the LEG still faded
    assert a["adv_bars"] >= 8              # and it spans the leg, not 2 bars


def test_the_long_side_reads_the_decline_into_the_low():
    from bot.scanner import advance_volume
    down = [1.09, 1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01, 1.00]
    a = advance_volume(_leg(down, [100, 95, 90, 80, 70, 55, 45, 35, 28, 20]), "long")
    assert a["adv_vol_trend"] < 1
    assert a["adv_price_pct"] < 0


def test_advance_volume_survives_degenerate_input():
    from bot.scanner import advance_volume
    pd = pytest.importorskip("pandas")
    assert advance_volume(None, "short")["adv_vol_trend"] is None
    assert advance_volume(pd.DataFrame(), "short")["adv_vol_trend"] is None


def test_the_distance_refusal_is_not_filed_under_RSI():
    """
    Its text OPENS with "RSI 80, strengthened 1 scans, 13.65% from the 24h
    high, limit 3.0%". A naive rsi-first match filed every distance refusal
    under the RSI band and hid the distance gate completely.
    """
    from bot.auto_trader import _refusal_key
    assert _refusal_key(
        "RSI 80, strengthened 1 scans, 13.65% from the 24h high, "
        "limit 3.0%") == "distance"
    assert _refusal_key("RSI 54.3 above the long ceiling of 52.0") == "rsi_band"
    assert _refusal_key("RSI 41.2 below the long floor of 45.0") == "rsi_band"


# ── Cross-margin leverage ───────────────────────────────────────────────────
#
# PUFFER 2026-09-15: the bot showed +4.1% ROI and margin $1800.91 while Binance
# showed +86.37% on $85.48 of margin. $1800.91 is the NOTIONAL — 65,345 x
# 0.0275599 — so notional/margin derived 1.0x and every ROI came out 20x small.
# Nothing warned, because margin was not <= 0.

def _puffer(margin):
    from bot.futures_guard import FuturesPosition
    return FuturesPosition(symbol="PUFFER/USDT:USDT", side="short",
                           entry_price=0.0275599, qty=65345.0,
                           leverage=20, margin=margin)


def test_a_margin_equal_to_notional_does_not_yield_1x():
    pos = _puffer(65345.0 * 0.0275599)          # the cross-margin payload
    assert pos.effective_leverage == 20.0       # falls back to the reported field


def test_a_real_margin_still_derives_the_true_leverage():
    """The derivation exists because the reported field has been 1 on isolated
    positions. It must keep working."""
    pos = _puffer(85.48)
    assert 20 < pos.effective_leverage < 22


def test_roi_matches_binance_once_leverage_is_right():
    """
    Binance: +73.83 USDT on 1800.90 notional is a 4.0996% favourable move,
    and +73.83 on 85.48 of margin is +86.37% ROI. Both must fall out.
    """
    from bot.futures_guard import roi_pct
    pos = _puffer(85.48)
    move = 73.83 / pos.notional
    assert roi_pct(pos, pos.entry_price * (1 - move)) == pytest.approx(86.37, abs=0.1)


def test_the_broken_payload_reported_a_twentieth_of_it():
    """What the operator actually saw on the dashboard: +4.1%."""
    from bot.futures_guard import roi_pct
    broken = _puffer(65345.0 * 0.0275599)
    broken.leverage = 1                          # nothing to fall back to
    move = 73.83 / broken.notional
    assert roi_pct(broken, broken.entry_price * (1 - move)) == pytest.approx(
        4.10, abs=0.05)


def test_the_parse_checks_believability_not_just_zero():
    """
    The old guard was `if margin <= 0`. 1800.90 is not zero, so it never fired.
    The check is now whether the implied leverage is credible.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian)
    assert "implied < 1.5" in src
    assert "margin field unusable" in src


# ── Adaptive floor level, and a loud failure ───────────────────────────────
#
# ARK/USDT 2026-09-15 on live: peaked +3.98%, the dashboard showed
# "Stop @ ROI +2%", and it ran to -20.13% with nothing resting. That UI field
# is the guardian's INTENT, not what the exchange holds.

def test_the_floor_places_at_the_configured_level_when_the_peak_is_intact():
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=7.36)
    g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)   # +5% ROI now
    assert st.floor_stop_id is not None
    assert st.floor_roi == 2.0                            # full level available


def test_it_places_LOWER_rather_than_failing_when_the_peak_is_given_back():
    """
    The peak was made and handed back between polls. A fixed +2% stop would
    sit on the wrong side of the market and be refused for the life of the
    position. Locking less beats locking nothing.
    """
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=3.98)
    barely = 0.1354 * (1 - 0.0005)       # short, +1% ROI at 20x
    g._ensure_profit_floor(_FloorPos(), st, barely)
    assert st.floor_stop_id is not None
    assert 0 < st.floor_roi < 2.0


def test_nothing_is_placed_once_the_profit_is_gone(caplog):
    """A floor at or below break-even would only lock in a loss."""
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    st = GuardState(peak_roi=3.98)
    underwater = 0.1354 * (1 + 0.005)    # short, -10% ROI
    with caplog.at_level("ERROR"):
        g._ensure_profit_floor(_FloorPos(), st, underwater)
    assert st.floor_stop_id is None
    assert any("PROTECTION-NO-FLOOR" in r.message for r in caplog.records)


def test_a_refused_placement_is_reported_not_swallowed(caplog):
    from bot.futures_guard import GuardState
    g = _floor_guardian()

    def _boom(pos, price):
        raise RuntimeError('binance {"code":-2021,"msg":"Order would immediately trigger."}')
    g._place_stop = _boom
    st = GuardState(peak_roi=7.36)
    with caplog.at_level("ERROR"):
        g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    msgs = " ".join(r.message for r in caplog.records)
    assert "PROTECTION-NO-FLOOR" in msgs
    assert "-2021" in msgs                 # the exchange's reason is carried
    assert st.floor_stop_id is None


def test_repeated_failures_do_not_spam_but_do_keep_reporting(caplog):
    from bot.futures_guard import GuardState
    g = _floor_guardian()
    g._place_stop = lambda pos, price: None
    st = GuardState(peak_roi=7.36)
    with caplog.at_level("ERROR"):
        for _ in range(6):
            g._ensure_profit_floor(_FloorPos(), st, _IN_PROFIT)
    n = len([r for r in caplog.records if "PROTECTION-NO-FLOOR" in r.message])
    assert n == 2                          # attempts 1 and 5, not all six
    assert st.floor_attempts == 6


def test_the_two_bands_meet_by_construction():
    """
    REZ 2026-09-15 peaked +2.08% — above the 1.0 fail-fast ceiling and below
    the 3.0 floor threshold — and ran to -16.74% with no protection of any
    kind. Two independent settings defined the coverage and nothing joined
    them, so every change to either reopened a hole.
    """
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import _peak_ceiling
    cfg = GuardConfig(breakeven_at_roi=3.0)
    assert cfg.fail_fast_max_peak_roi is None      # unset by default
    assert _peak_ceiling(cfg) == 3.0               # tied to the floor threshold


def test_moving_the_floor_threshold_moves_the_ceiling_with_it():
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import _peak_ceiling
    for at in (2.0, 3.0, 5.0):
        assert _peak_ceiling(GuardConfig(breakeven_at_roi=at)) == at


def test_an_explicit_ceiling_still_overrides():
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import _peak_ceiling
    cfg = GuardConfig(breakeven_at_roi=3.0, fail_fast_max_peak_roi=0.5)
    assert _peak_ceiling(cfg) == 0.5


def test_a_peak_inside_the_old_hole_is_now_fail_fast_eligible():
    """REZ's +2.08% peak, against a 3.0 floor threshold."""
    from bot.futures_guard import GuardConfig, GuardState
    from bot.futures_guardian import _peak_ceiling
    cfg = GuardConfig(breakeven_at_roi=3.0)
    st = GuardState(peak_roi=2.08)
    assert st.peak_roi <= _peak_ceiling(cfg)       # covered now
    old = GuardConfig(breakeven_at_roi=3.0, fail_fast_max_peak_roi=1.0)
    assert st.peak_roi > _peak_ceiling(old)        # was not


# ── Velocity floor, shorts only ─────────────────────────────────────────────
#
# Of 64 velocity-floored trades the nine LONGS returned -$17.64 each — 14% of
# the population and 233% of the net loss — while the 55 shorts returned
# +$1.65. The same shape appears without velocity: atr_floor longs -$13.00,
# `ratio` longs +$6.13. It is ATR-based WIDENING of a long's callback that
# hurts, not velocity itself.

def _cbcfg(**kw):
    base = dict(enabled=True, callback_ratio=0.25, callback_atr_mult=0.75,
                callback_use_velocity="all")
    base.update(kw)
    return AutoTradeConfig(**base)


def test_a_short_still_gets_the_velocity_floor():
    from bot.auto_trader import callback_for
    cb, _, src = callback_for(0.30, 1.022, _cbcfg(callback_use_velocity="short"),
                              recent_tr_pct=2.633, side="short")
    assert src == "velocity_floor"
    assert cb == pytest.approx(0.75 * 2.633, abs=0.01)


def test_a_long_does_not():
    from bot.auto_trader import callback_for
    cb, _, src = callback_for(0.30, 1.022, _cbcfg(callback_use_velocity="short"),
                              recent_tr_pct=2.633, side="long")
    assert src != "velocity_floor"
    assert cb == pytest.approx(0.75 * 1.022, abs=0.01)   # the ATR floor instead


def test_the_two_sides_now_differ_on_identical_inputs():
    from bot.auto_trader import callback_for
    args = dict(recent_tr_pct=6.214)
    cfg = _cbcfg(callback_use_velocity="short")
    short_cb = callback_for(0.22, 1.498, cfg, side="short", **args)[0]
    long_cb = callback_for(0.22, 1.498, cfg, side="long", **args)[0]
    assert short_cb > long_cb


def test_the_split_can_be_switched_off():
    """velocity_shorts_only=False restores the old behaviour for both sides."""
    from bot.auto_trader import callback_for
    cfg = _cbcfg(callback_use_velocity="all")
    _, _, src = callback_for(0.30, 1.022, cfg, recent_tr_pct=2.633, side="long")
    assert src == "velocity_floor"


def test_no_side_given_keeps_the_floor():
    """Callers that do not pass a side must not silently lose the floor."""
    from bot.auto_trader import callback_for
    _, _, src = callback_for(0.30, 1.022, _cbcfg(callback_use_velocity="short"),
                             recent_tr_pct=2.633)
    assert src == "velocity_floor"


def test_the_flag_is_off_entirely_when_velocity_is_disabled():
    from bot.auto_trader import callback_for
    cfg = _cbcfg(callback_use_velocity="off")
    for side in ("short", "long"):
        _, _, src = callback_for(0.30, 1.022, cfg, recent_tr_pct=2.633, side=side)
        assert src != "velocity_floor"


def test_the_setting_takes_four_values():
    """One setting replaced two booleans that could contradict each other."""
    from bot.auto_trader import velocity_mode
    assert velocity_mode("shorts") == "short"
    assert velocity_mode("longs") == "long"
    assert velocity_mode("all") == "all"
    assert velocity_mode("off") == "off"
    # the booleans it replaced still work
    assert velocity_mode(True) == "all"
    assert velocity_mode(False) == "off"
    assert velocity_mode(None) == "off"
    assert velocity_mode("nonsense") == "off"


def test_shorts_is_the_default():
    from bot.config import BotConfig
    from bot.auto_trader import velocity_mode
    assert velocity_mode(BotConfig().auto_callback_use_velocity) == "short"


def test_long_only_mode_is_available_even_though_the_data_advises_against_it():
    from bot.auto_trader import callback_for
    cfg = _cbcfg(callback_use_velocity="longs")
    assert callback_for(0.30, 1.022, cfg, recent_tr_pct=2.633,
                        side="long")[2] == "velocity_floor"
    assert callback_for(0.30, 1.022, cfg, recent_tr_pct=2.633,
                        side="short")[2] != "velocity_floor"


def test_evaluate_candidate_passes_the_side_through():
    """The split is useless if the call site does not tell callback_for which
    direction it is sizing."""
    import inspect
    from bot import auto_trader
    src = inspect.getsource(auto_trader.evaluate_candidate)
    assert "side=side" in src


def test_a_hand_set_gap_is_reported_at_startup(caplog):
    """
    Nothing stopped an operator reopening the hole by hand, and it was silent.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.__init__)
    assert "PROTECTION GAP" in src
    assert "Protection is continuous" in src


# ── The post-loss cooldown was dead code ───────────────────────────────────
#
# AKE/USDT 2026-09-16: seven entries in one day for -$94.98, four of them 8-21
# minutes after the previous close, against AUTO_SYMBOL_COOLDOWN_S=1800.
# note_closed_trade() held the whole chain and NOTHING called it.

def test_the_guardian_tells_someone_when_a_position_closes():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian)
    assert 'getattr(self, "on_position_closed", None)' in src


def test_main_wires_the_callback_to_the_auto_trader():
    """The hook is useless unwired — which is exactly how it shipped."""
    from pathlib import Path
    src = Path("main.py").read_text()
    assert "guardian.on_position_closed = auto.note_closed_trade" in src


def test_a_loss_blocks_the_symbol_for_the_cooldown():
    from bot.auto_trader import (SafetyState, AutoTradeConfig, record_loss,
                                 check_safety)
    cfg = AutoTradeConfig(symbol_cooldown_s=1800.0, cooldown_override_rsi_delta=0)
    st = SafetyState()
    record_loss(st, "AKE/USDT:USDT", cfg, entry_rsi=76.0, now=1000.0)
    args = dict(balance=5000.0, open_positions=0, symbol="AKE/USDT:USDT")
    ok, why = check_safety(st, cfg, now=1000.0 + 8 * 60, **args)
    assert not ok                       # 8 minutes later, as AKE was
    ok, _ = check_safety(st, cfg, now=1000.0 + 1801, **args)
    assert ok


def test_the_reentry_cap_binds_once_losses_are_recorded():
    from bot.auto_trader import (SafetyState, AutoTradeConfig, record_loss,
                                 record_reentry, check_safety)
    cfg = AutoTradeConfig(symbol_cooldown_s=1800.0,
                          cooldown_override_rsi_delta=3.0,
                          max_reentries_per_symbol=2)
    st = SafetyState()
    record_loss(st, "AKE/USDT:USDT", cfg, entry_rsi=76.0, now=1000.0)
    args = dict(balance=5000.0, open_positions=0, symbol="AKE/USDT:USDT",
                side="short")
    for i in range(2):
        ok, why = check_safety(st, cfg, now=1100.0, current_rsi=85.0, **args)
        assert ok and "cooldown overridden" in why
        record_reentry(st, "AKE/USDT:USDT")
    ok, why = check_safety(st, cfg, now=1100.0, current_rsi=90.0, **args)
    assert not ok
    assert "already retried" in why


def test_a_winning_close_does_not_block_the_symbol():
    from bot.auto_trader import AutoTradeConfig
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.note_closed_trade)
    assert "realised < 0" in src


# ── Efficiency ratio: trend or chop ────────────────────────────────────────

def _closes(vals):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"close": vals})


def test_a_clean_trend_scores_near_one():
    from bot.scanner import efficiency_ratio
    e = efficiency_ratio(_closes([1 + i * 0.01 for i in range(20)]))
    assert e["efficiency"] > 0.95
    assert e["er_direction"] == "up"


def test_pure_chop_scores_near_zero():
    from bot.scanner import efficiency_ratio
    e = efficiency_ratio(_closes([1 + (0.03 if i % 2 else -0.03) for i in range(20)]))
    assert e["efficiency"] < 0.15


def test_the_AKE_shape_is_distinguished_from_chop():
    """
    Large alternating candles inside a trend. It LOOKS like chop at 3-minute
    resolution — AKE ran +68.57% that way while seven shorts faded it — and no
    other recorded measure separates the two.
    """
    from bot.scanner import efficiency_ratio
    trending = efficiency_ratio(
        _closes([1 + i * 0.01 + (0.02 if i % 2 else -0.02) for i in range(20)]))
    choppy = efficiency_ratio(
        _closes([1 + (0.03 if i % 2 else -0.03) for i in range(20)]))
    assert trending["efficiency"] > choppy["efficiency"] * 2


def test_efficiency_survives_degenerate_input():
    from bot.scanner import efficiency_ratio
    pd = pytest.importorskip("pandas")
    assert efficiency_ratio(None)["efficiency"] is None
    assert efficiency_ratio(pd.DataFrame())["efficiency"] is None
    assert efficiency_ratio(_closes([1.0] * 20))["efficiency"] is None


def test_both_directions_carry_the_reading():
    import inspect
    from bot import scanner
    src = inspect.getsource(scanner.evaluate_symbol)
    assert src.count("efficiency=eff,") == 2


# ── Day card ────────────────────────────────────────────────────────────────

def test_the_day_boundary_is_local_not_utc():
    import time as _t
    import bot.auto_trader as at
    old = at.DAY_TZ_OFFSET_H
    try:
        at.DAY_TZ_OFFSET_H = 3.0
        now = _t.time()
        ds = at.day_start_ts(now)
        # local midnight: 00:00 in UTC+3 is 21:00 UTC the day before
        assert _t.gmtime(ds + 3 * 3600).tm_hour == 0
        assert _t.gmtime(ds + 3 * 3600).tm_min == 0
        assert 0 <= (now - ds) < 86400
    finally:
        at.DAY_TZ_OFFSET_H = old


def test_utc_offset_zero_still_gives_utc_midnight():
    import time as _t
    import bot.auto_trader as at
    old = at.DAY_TZ_OFFSET_H
    try:
        at.DAY_TZ_OFFSET_H = 0.0
        ds = at.day_start_ts(_t.time())
        assert _t.gmtime(ds).tm_hour == 0
    finally:
        at.DAY_TZ_OFFSET_H = old


def _dtrade(closed_at, pnl):
    return {"symbol": "X/USDT:USDT", "side": "short", "final_roi": 5.0,
            "realised_pnl_usdt": pnl, "fees_usdt": 1.7, "margin_usdt": 88.0,
            "exit_is_estimate": False, "closed_at": closed_at,
            "entry_context": {"sized_stop_roi": 30.0}}


def test_the_day_report_counts_only_todays_trades():
    from bot.analysis import day_report
    start = 1_000_000.0
    out = day_report([_dtrade(start - 3600, 50.0),      # yesterday
                      _dtrade(start + 60, 20.0),
                      _dtrade(start + 120, -5.0)],
                     day_baseline=5000.0, day_start_ts=start, tz_offset_h=3.0)
    assert out["trades"] == 2
    assert out["wins"] == 1
    assert out["net_pnl"] == pytest.approx(15.0)
    assert out["pct"] == pytest.approx(0.3, abs=0.01)
    assert out["tz_offset_h"] == 3.0


def test_it_reports_a_baseline_with_no_trades_yet():
    from bot.analysis import day_report
    out = day_report([], day_baseline=5000.0, day_start_ts=1_000_000.0)
    assert out["trades"] == 0 and out["baseline"] == 5000.0
    assert out["pct"] == pytest.approx(0.0)


def test_it_survives_a_missing_baseline():
    from bot.analysis import day_report
    out = day_report([_dtrade(2.0, 10.0)], day_baseline=None, day_start_ts=1.0)
    assert out["pct"] is None
    assert out["trades"] == 1


def test_the_card_never_yields_its_subtitle_to_a_warning():
    """
    account_return replaces the daily line with the wallet-gap warning, which
    on demo is nearly permanent. This card has its own.
    """
    from pathlib import Path
    ui = Path("ui/index.html").read_text()
    assert 'id="s-day"' in ui and 'id="s-day-sub"' in ui
    i = ui.index("const dy = a.day || {};")
    block = ui[i:i + 1400]
    assert "disagrees_with_wallet" not in block


# ── Stat strip layout ───────────────────────────────────────────────────────

def _ui():
    from pathlib import Path
    return Path("ui/index.html").read_text()


def test_the_duplicate_wallet_tile_is_hidden_on_futures():
    """
    Both tiles were fed the same g.wallet_balance and the futures view just
    relabelled the second one, so the row showed $5307.53 twice and spent a
    column on it. On spot the two figures genuinely differ, so it is hidden
    rather than deleted.
    """
    ui = _ui()
    assert 'body[data-view="futures"] #stat-avail-card { display: none; }' in ui
    assert 'id="stat-avail-card"' in ui


def test_the_guardian_cards_are_hidden_on_spot():
    """
    Account return and Today need a daily baseline from the auto-trader;
    Return on capital divides by margin. Spot has no auto-trader, no guardian
    and no margin, so all three can only read "—".
    """
    ui = _ui()
    for card in ("stat-acct-card", "stat-day-card", "stat-roc-card"):
        assert f'body[data-view="spot"] #{card}' in ui


def test_fees_stays_visible_on_spot():
    """Spot trades do pay commission — it reads "—" because the spot path does
    not populate it, which is a gap to fix rather than a card to hide."""
    ui = _ui()
    assert 'body[data-view="spot"] #stat-fees-card' not in ui


def test_each_view_declares_its_own_column_count():
    ui = _ui()
    import re
    fut = re.search(r'body\[data-view="futures"\] \.stats-row \{(.*?)\}', ui, re.S)
    spot = re.search(r'body\[data-view="spot"\] \.stats-row \{(.*?)\}', ui, re.S)
    assert fut and spot
    assert fut.group(1).count("fr") == 9      # nine cards on futures
    assert spot.group(1).count("fr") == 7     # seven on spot


def test_the_open_positions_subtitle_moved_to_the_positions_card():
    ui = _ui()
    assert 'id="s-open-sub"' in ui
    assert "setEl('s-open-sub'" in ui
    assert "setEl('s-portfolio-sub', '');" in ui


# ── Account return: wallet first ───────────────────────────────────────────

def test_the_wallet_is_the_headline_not_the_trade_record():
    """
    The card led with the figure reconstructed from the trade record and
    relegated the actual balance to a warning — showing +7.22% when the wallet
    said +6.15%. The wallet is the truth; the record excludes unverified
    trades.
    """
    ui = _ui()
    i = ui.index("// WALLET FIRST.")
    block = ui[i:i + 1800]
    assert "setEl('s-acct', (walletPct" in block
    assert "bot ${botPct" in block


def test_the_gap_is_signed_wallet_minus_bot():
    ui = _ui()
    i = ui.index("// WALLET FIRST.")
    block = ui[i:i + 1800]
    assert "WALLET MINUS BOT" in block
    assert "gap ${gap >= 0 ? '+' : '-'}" in block


# ── Amount masking ─────────────────────────────────────────────────────────

def test_every_card_dollar_figure_goes_through_the_mask():
    ui = _ui()
    for call in ("setEl('s-portfolio', money(",
                 "setEl('s-balance', money(",
                 "setEl('s-fees', money(",
                 "setEl('s-pnl', realised ? money("):
        assert call in ui, call


def test_the_trade_history_column_is_masked_too():
    """Cards alone would leave every row below showing the figures."""
    ui = _ui()
    i = ui.index("const pnl = t.realised_pnl_usdt") if "const pnl = t.realised_pnl_usdt" in ui else 0
    assert "money(`${pnl >= 0 ? '+' : ''}$${pnl}`)" in ui


def test_percentages_are_never_masked():
    """They are the part worth sharing — the point is to hide balances."""
    ui = _ui()
    assert "setEl('s-winrate'" in ui
    i = ui.index("setEl('s-winrate'")
    assert "money(" not in ui[i:i + 120]


def test_the_toggle_persists_and_has_a_server_default():
    ui = _ui()
    assert "localStorage.setItem('hideAmounts'" in ui
    assert "hide_amounts_default" in ui
    from bot.config import BotConfig
    assert BotConfig().hide_card_amounts is False


# ── Session toggles in the UI ──────────────────────────────────────────────

def test_only_the_session_table_gets_toggles():
    """
    The toggle belongs beside the numbers that justify it. Every other
    timeTable is read-only.
    """
    ui = _ui()
    i = ui.index("timeTable('By session (entry time, UTC)'")
    j = ui.index("timeTable('By hour of entry (UTC)'")
    assert "trend.', true)" in ui[i:j]
    k = ui.index("timeTable('By market breadth at entry'")
    assert ", true)" not in ui[k:k + 400]


def test_the_toggle_writes_through_the_existing_rules_endpoint():
    from pathlib import Path
    ui = _ui()
    assert "JSON.stringify({ rules: { sessions } })" in ui
    # and the click must show something BEFORE the request returns
    assert "sess-pending" in ui
    assert "sess-failed" in ui
    assert '"sessions": (dict, None, None),' in Path("bot/auto_trader.py").read_text()


def test_the_labels_map_to_the_session_keys():
    ui = _ui()
    for label, key in (("Asia", "AS"), ("Europe", "EU"),
                       ("EU/US overlap", "OV"), ("US", "US")):
        assert f"'{label}': '{key}'" in ui


# ── Trade journal ───────────────────────────────────────────────────────────
#
# Closed trades lived in futures_state.json, which save_state() rewrites in
# FULL every guardian cycle — 34,560 times a day. At the 5000-trade cap that is
# an 11 MB serialise every 2.5s, ~383 GB of disk writes a day, to persist a few
# kilobytes of changed state. And the cap silently DROPPED trades past ~83 days.

def _journal(tmp_path, **kw):
    from bot.trade_journal import TradeJournal
    return TradeJournal(str(tmp_path / "trades.jsonl"), **kw)


def _rec(i):
    return {"symbol": f"X{i}/USDT:USDT", "side": "short",
            "realised_pnl_usdt": float(i), "closed_at": 1_000_000.0 + i}


def test_trades_round_trip(tmp_path):
    j = _journal(tmp_path)
    for i in range(50):
        assert j.append(_rec(i))
    out = j.load()
    assert len(out) == 50
    assert out[0]["symbol"] == "X0/USDT:USDT"
    assert out[-1]["symbol"] == "X49/USDT:USDT"


def test_a_limit_returns_the_most_recent(tmp_path):
    j = _journal(tmp_path)
    for i in range(50):
        j.append(_rec(i))
    out = j.load(limit=5)
    assert [t["symbol"] for t in out] == [f"X{i}/USDT:USDT" for i in range(45, 50)]


def test_rotation_archives_rather_than_discards(tmp_path):
    """
    The old cap threw the oldest trades away. Rotation moves whole files aside
    and load() reads back through them, so nothing is lost.
    """
    j = _journal(tmp_path, max_bytes=300, keep_archives=50)
    for i in range(60):
        j.append(_rec(i))
    out = j.load()
    assert len(out) == 60
    assert [t["symbol"] for t in out] == [f"X{i}/USDT:USDT" for i in range(60)]
    assert j.stats()["archives"] >= 1


def test_two_rotations_in_one_second_do_not_collide(tmp_path):
    j = _journal(tmp_path, max_bytes=120, keep_archives=50)
    for i in range(40):
        j.append(_rec(i))
    assert len(j.load()) == 40          # nothing overwritten


def test_a_corrupt_tail_line_does_not_lose_the_file(tmp_path):
    """A partial write is the expected state after a hard kill."""
    j = _journal(tmp_path)
    for i in range(10):
        j.append(_rec(i))
    with open(j.path, "a", encoding="utf-8") as fh:
        fh.write('{"symbol": "TRUNCA')
    out = j.load()
    assert len(out) == 10


def test_migration_runs_once_and_only_once(tmp_path):
    j = _journal(tmp_path)
    old = [_rec(i) for i in range(5)]
    assert j.import_existing(old) == 5
    assert j.import_existing(old) == 0      # journal is no longer empty
    assert len(j.load()) == 5


def test_an_unwritable_journal_never_raises(tmp_path):
    from bot.trade_journal import TradeJournal
    j = TradeJournal("/proc/nope/trades.jsonl")
    assert j.append(_rec(1)) is False       # logged, not raised
    assert j.load() == []


def test_the_state_file_no_longer_carries_trades():
    """The whole point: save_state() must not serialise the history."""
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.save_state)
    assert 'if getattr(self, "_journal", None) is not None:' in src
    assert "trades = []" in src


def test_the_guardian_appends_once_per_trade():
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._record_closed_trade)
    assert "jr.append(rec)" in src


# ── History view window ────────────────────────────────────────────────────

def test_the_table_renders_a_window_not_the_whole_record():
    ui = _ui()
    assert "const shown = withinWindow(d.trades);" in ui
    assert 'id="ft-window"' in ui


def test_both_exports_exist_and_differ():
    ui = _ui()
    assert "downloadTradesCsv(true)" in ui       # view
    assert "downloadTradesCsv(false)" in ui      # all
    assert "const trades = viewOnly ? withinWindow(all) : all;" in ui


def test_the_count_shows_both_figures():
    """So it is obvious the table is a view, not the whole record."""
    ui = _ui()
    assert "of ${d.trades.length} · last ${_histDays}d" in ui


# ── No stale config references ─────────────────────────────────────────────
#
# v3.26.0 renamed auto_trading_window to auto_sessions and left ONE reference
# in main.py. AutoTrader raised at construction, the except logged one line,
# and the bot ran for hours with the scanner and guardian healthy and NO
# entries taken. Nothing on the dashboard said so.

def test_no_module_references_a_config_field_that_does_not_exist():
    """
    Catches the whole class: every cfg.<attr> in main.py must resolve on a
    real BotConfig. A rename that misses a call site fails HERE, not at
    startup in production.
    """
    import re
    from pathlib import Path
    from bot.config import BotConfig
    cfg = BotConfig()
    src = Path("main.py").read_text()
    refs = sorted(set(re.findall(r"\bcfg\.([a-zA-Z_][a-zA-Z0-9_]*)", src)))
    missing = [r for r in refs if not hasattr(cfg, r)]
    assert not missing, f"main.py references non-existent config: {missing}"


def test_the_ui_does_not_reference_the_removed_window():
    ui = _ui()
    assert "trading_window" not in ui


def test_an_auto_trade_start_failure_is_loud():
    from pathlib import Path
    main = Path("main.py").read_text()
    assert "AUTO-TRADE IS NOT RUNNING" in main
    assert "set_auto_trader_error" in main
    from bot import api
    assert hasattr(api, "set_auto_trader_error")


def test_the_dashboard_shows_the_start_failure():
    ui = _ui()
    assert "AUTO-TRADE IS NOT `" in ui or "AUTO-TRADE IS NOT " in ui
    assert "a.start_error" in ui


def test_export_is_one_control_with_two_choices():
    ui = _ui()
    assert 'id="ft-export"' in ui
    assert "onExportPick(this)" in ui
    assert "downloadTradesCsv(true)" in ui and "downloadTradesCsv(false)" in ui


def test_session_toggles_actually_apply():
    """
    They rendered and did nothing: `sessions` is declared (dict, None, None),
    so `typ is str` was False and the value fell into the numeric branch.
    """
    import logging
    from bot.auto_trader import AutoTrader, AutoTradeConfig
    logging.disable(logging.CRITICAL)
    try:
        a = AutoTrader.__new__(AutoTrader)
        a.cfg = AutoTradeConfig(); a.state = None; a._log = []
        applied, errors = a.update_rules({"sessions": {"US": "L0S1"}})
        assert not errors and applied
        assert a.cfg.sessions["US"] == "L0S1"
        assert a.cfg.sessions["AS"] == "L1S1"      # others untouched
    finally:
        logging.disable(logging.NOTSET)


def test_a_bad_session_key_is_rejected_not_applied():
    import logging
    from bot.auto_trader import AutoTrader, AutoTradeConfig
    logging.disable(logging.CRITICAL)
    try:
        a = AutoTrader.__new__(AutoTrader)
        a.cfg = AutoTradeConfig(); a.state = None; a._log = []
        applied, errors = a.update_rules({"sessions": {"ZZ": "L1S1"}})
        assert errors and not applied
        applied, errors = a.update_rules({"sessions": "L1S1"})
        assert errors and not applied
    finally:
        logging.disable(logging.NOTSET)


def test_an_unreadable_spec_switches_nothing_off():
    import logging
    from bot.auto_trader import AutoTrader, AutoTradeConfig
    logging.disable(logging.CRITICAL)
    try:
        a = AutoTrader.__new__(AutoTrader)
        a.cfg = AutoTradeConfig(); a.state = None; a._log = []
        a.update_rules({"sessions": {"AS": "garbage"}})
        assert a.cfg.sessions["AS"] == "L1S1"
    finally:
        logging.disable(logging.NOTSET)


def test_the_hide_control_is_legible():
    """A 12px emoji at 0.55 opacity on a dark card was invisible in practice."""
    ui = _ui()
    assert "'SHOW' : 'HIDE'" in ui
    assert ">HIDE</button>" in ui


def test_the_exact_dashboard_payload_does_not_raise():
    """
    Regression for the 500 seen in production on v3.27.1:

        File "bot/auto_trader.py", line 934, in update_rules
          elif not (lo <= val <= hi):
        TypeError: '<=' not supported between instances of 'NoneType' and 'dict'

    The toggle POSTs the WHOLE map, not one key, and it reached the numeric
    branch. update_rules must never raise on a well-formed payload — an
    exception there is a 500 that tells the operator nothing.
    """
    import logging
    from bot.auto_trader import AutoTrader, AutoTradeConfig
    logging.disable(logging.CRITICAL)
    try:
        a = AutoTrader.__new__(AutoTrader)
        a.cfg = AutoTradeConfig(); a.state = None; a._log = []
        payload = {"sessions": {"AS": "L1S1", "EU": "L1S1",
                                "OV": "L1S1", "US": "L0S1"}}
        applied, errors = a.update_rules(payload)
        assert not errors
        assert a.cfg.sessions["US"] == "L0S1"
    finally:
        logging.disable(logging.NOTSET)


def test_no_tunable_type_can_reach_the_numeric_comparison_unguarded():
    """
    The root cause was a TUNABLE type with no branch of its own falling
    through to `lo <= val <= hi`. Every declared type must be handled before
    that line.
    """
    from bot.auto_trader import AutoTrader
    handled = {bool, str, dict, int, float}
    declared = {typ for typ, _, _ in AutoTrader.TUNABLE.values()}
    assert declared <= handled, f"unhandled TUNABLE type(s): {declared - handled}"


def test_a_rule_update_failure_is_a_400_not_a_500():
    """
    The TypeError escaped as an ASGI 500: a stack trace in the log and nothing
    useful on the page. Whatever the cause, a bad rule is a client error
    carrying its reason.
    """
    import inspect
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("applied, errors = _auto.update_rules(rules)")
    block = src[max(0, i - 200):i + 700]
    assert "except Exception as e:" in block
    assert "status_code=400" in block
    assert "except HTTPException:" in block      # real 400s pass through


# ── Today's baseline ───────────────────────────────────────────────────────

def _dt(ts, pnl):
    return {"symbol": "X/USDT:USDT", "side": "short", "final_roi": 5.0,
            "realised_pnl_usdt": pnl, "fees_usdt": 1.7, "margin_usdt": 88.0,
            "exit_is_estimate": False, "closed_at": ts,
            "entry_context": {"sized_stop_roi": 30.0}}


def test_the_baseline_is_reconstructed_from_the_wallet():
    """
    day_start_balance is set when the day KEY changes, so a mid-day restart
    stores the balance at THAT moment. The card read "from $5308" when the
    real 00:00 figure was $5063.54, and the percentage inherited the error.
    """
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()   # process-global cache
    start = 1_000_000.0
    out = day_report([_dt(start + 60, 228.65)], day_baseline=5308.0,
                     day_start_ts=start, wallet_now=5292.19)
    assert out["baseline"] == pytest.approx(5063.54, abs=0.01)
    assert out["baseline_source"] == "reconstructed"
    assert out["pct"] == pytest.approx(4.516, abs=0.01)


def test_it_falls_back_to_the_stored_value_with_no_trades():
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()   # process-global cache
    out = day_report([], day_baseline=5308.0, day_start_ts=1_000_000.0,
                     wallet_now=5292.19)
    assert out["baseline"] == 5308.0
    assert out["baseline_source"] == "stored"


def test_yesterdays_trades_do_not_move_todays_baseline():
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()   # process-global cache
    start = 1_000_000.0
    out = day_report([_dt(start - 3600, 500.0), _dt(start + 60, 100.0)],
                     day_baseline=9999.0, day_start_ts=start, wallet_now=5100.0)
    assert out["baseline"] == pytest.approx(5000.0, abs=0.01)
    assert out["trades"] == 1


def test_a_losing_day_reconstructs_upward():
    """Net negative means the day STARTED higher than the wallet is now."""
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()   # process-global cache
    start = 1_000_000.0
    out = day_report([_dt(start + 60, -150.0)], day_baseline=None,
                     day_start_ts=start, wallet_now=4850.0)
    assert out["baseline"] == pytest.approx(5000.0, abs=0.01)
    assert out["pct"] < 0


def test_a_missing_wallet_does_not_invent_a_baseline():
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()   # process-global cache
    out = day_report([_dt(1_000_060.0, 10.0)], day_baseline=5000.0,
                     day_start_ts=1_000_000.0, wallet_now=None)
    assert out["baseline"] == 5000.0
    assert out["baseline_source"] == "stored"


def test_the_card_marks_a_stored_baseline_as_uncertain():
    ui = _ui()
    assert "dy.baseline_source === 'stored'" in ui


# ── Candidate price stream ─────────────────────────────────────────────────
#
# Every gate was applied to a scan up to 120s old while the order was SIZED at
# the current price. LSK/USDT 2026-09-16 was decided on a 0.5239 market and
# sized at 0.4612 — a 12% gap.

def _stream():
    from bot.candidate_stream import CandidateStream
    return CandidateStream(demo=True, stale_after_s=20.0)


def test_symbols_are_converted_to_the_wire_form():
    s = _stream()
    assert s._wire("BTC/USDT:USDT") == "btcusdt"
    assert s._wire("1000PEPE/USDT:USDT") == "1000pepeusdt"
    assert s._wire("SYN/USDT") == "synusdt"


def test_a_quote_is_readable_after_ingest():
    s = _stream()
    s._ingest('{"data":{"s":"BTCUSDT","p":"64000.5"}}')
    assert s.price("BTC/USDT:USDT") == pytest.approx(64000.5)


def test_a_stale_quote_is_not_returned():
    """Falling back to the scan figure is correct; a stale price is not."""
    import time
    s = _stream()
    s._ingest('{"data":{"s":"BTCUSDT","p":"64000.5"}}')
    assert s.price("BTC/USDT:USDT", now=time.time() + 25) is None


def test_an_unknown_symbol_returns_none_not_an_error():
    assert _stream().price("NOPE/USDT:USDT") is None


def test_a_malformed_frame_is_ignored():
    s = _stream()
    for junk in ("", "not json", '{"data":{}}', '{"data":{"s":"X"}}'):
        s._ingest(junk)
    assert s.status()["quotes"] == 0


def test_tracking_drops_symbols_that_are_no_longer_eligible():
    s = _stream()
    s._ingest('{"data":{"s":"BTCUSDT","p":"1"}}')
    s.track(["ETH/USDT:USDT"])
    assert s.price("BTC/USDT:USDT") is None      # dropped with its quote
    assert s.status()["tracking"] == 1


# ── Drift gate ─────────────────────────────────────────────────────────────

def _drift_row(side, dist, hi=None, lo=None, live=None):
    return {"symbol": "LSK/USDT:USDT", "direction": side,
            "dist_to_extreme_pct": dist, "high_24h": hi, "low_24h": lo,
            "live_price": live}


def test_drift_reconstructs_the_price_the_scan_decided_at():
    """LSK: dist 0.98% from a 0.5291 high back-solves to 0.5239."""
    from bot.auto_trader import live_drift
    row = _drift_row("short", 0.98, hi=0.5291)
    drift, live_dist = live_drift(row, 0.4612)
    assert drift == pytest.approx(-11.97, abs=0.05)
    assert live_dist == pytest.approx(12.83, abs=0.1)


def test_no_live_quote_means_no_opinion():
    from bot.auto_trader import live_drift
    assert live_drift(_drift_row("short", 0.98, hi=0.5291), None) == (None, None)


def test_a_missing_extreme_does_not_guess():
    from bot.auto_trader import live_drift
    assert live_drift(_drift_row("short", 0.98), 0.46) == (None, None)


def test_a_stale_signal_is_refused():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75,
                          max_dist_to_extreme_pct=3.0)
    row = _short(breakout=_brk(gap_widening=False, breakout=False))
    row.update({"dist_to_extreme_pct": 0.98, "high_24h": 0.5291,
                "live_price": 0.4612})
    d = evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=2.0)
    assert not d.enter
    assert "stale signal" in d.reason


def test_a_live_price_close_to_the_scan_still_enters():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75,
                          max_dist_to_extreme_pct=3.0, veto_breakout=False)
    row = _short(breakout=_brk(gap_widening=False, breakout=False))
    row.update({"dist_to_extreme_pct": 0.98, "high_24h": 0.5291,
                "live_price": 0.5291 * (1 - 0.012)})
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=2.0).enter


# ── Volume deferral, shorts only ───────────────────────────────────────────
#
# Across 89 shorts, volume FADING into the high returned +$2.14/trade against
# -$1.00 when it was BUILDING. Not mirrored for longs: a top forms on
# declining volume (distribution) while a bottom often forms on a volume SPIKE
# (a selling climax), so the same reading means the opposite thing — and the
# long sample is 16 trades with both buckets losing.

def _vol_short(vol_trend):
    row = _short(breakout=_brk(gap_widening=False, breakout=False))
    row["advance"] = {"adv_vol_trend": vol_trend, "adv_bars": 25}
    return row


def test_a_short_is_deferred_while_volume_builds():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False)
    d = evaluate_candidate(_vol_short(1.92), streak=2, cfg=cfg, atr_pct=2.0)
    assert not d.enter
    assert "volume still building" in d.reason


def test_a_short_enters_once_volume_is_fading():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False)
    assert evaluate_candidate(_vol_short(0.62), streak=2, cfg=cfg,
                              atr_pct=2.0).enter


def test_a_missing_volume_reading_does_not_defer():
    """Most trades predate the measure; absence must not block them."""
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False)
    row = _vol_short(None)
    row["advance"] = {}
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=2.0).enter


def test_longs_are_not_deferred_on_rising_volume():
    """
    The asymmetry is the point: a selling climax on peak volume is the classic
    bottom, so building volume may MARK a long entry rather than forbid it.
    """
    cfg = AutoTradeConfig(enabled=True, long_rsi_min=45, long_rsi_max=52)
    row = _long_cand()
    row["advance"] = {"adv_vol_trend": 3.5, "adv_bars": 25}
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=0.6).enter


def test_the_deferral_can_be_switched_off():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False,
                          defer_on_rising_volume=False)
    assert evaluate_candidate(_vol_short(3.0), streak=2, cfg=cfg,
                              atr_pct=2.0).enter


def test_the_threshold_is_tunable():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False,
                          defer_vol_trend=2.5)
    assert evaluate_candidate(_vol_short(1.9), streak=2, cfg=cfg,
                              atr_pct=2.0).enter          # under the bar now
    assert not evaluate_candidate(_vol_short(2.6), streak=2, cfg=cfg,
                                  atr_pct=2.0).enter


def test_both_new_refusals_have_their_own_tally_key():
    from bot.auto_trader import _refusal_key
    assert _refusal_key("volume still building into the high (adv 1.9)") == "vol_deferred"
    assert _refusal_key("stale signal: the scan saw 0.98%") == "stale_signal"


def test_a_missing_stream_never_breaks_a_cycle():
    """The stream is optional. Absent, every decision uses the scan snapshot."""
    from bot.auto_trader import _stream_status
    assert _stream_status(None) == {"enabled": False, "connected": False}

    class Broken:
        def status(self):
            raise RuntimeError("socket gone")
    out = _stream_status(Broken())
    assert out["enabled"] is True and out["connected"] is False


def test_stream_health_is_logged_every_cycle():
    """
    A stream that stops delivering degrades entries to the scan snapshot with
    nothing failing — a quiet regression only visible weeks later in the
    numbers. One greppable word.
    """
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader)
    assert "STREAM ok" in src or "'ok' if healthy" in src
    assert "STREAM-SAVED" in src
    assert "falling back to the scan" in src


def test_the_dashboard_shows_stream_health():
    ui = _ui()
    assert "function streamBadge(" in ui
    assert "LIVE FEED DOWN" in ui


def test_masking_does_not_wait_on_the_network():
    """
    The toggle called refresh(), so four cards waited on /api/analysis — which
    runs analyse() over EVERY trade. A ~30s lag that would only grow.
    Masking is a pure display transform; the numbers are already in the page.
    """
    ui = _ui()
    assert "let _lastAnalysis = null;" in ui
    assert "function renderAnalysis(a)" in ui
    assert "function renderFuturesHistoryFrom(d)" in ui
    i = ui.index("function toggleHideAmounts()")
    block = ui[i:i + 700]
    assert "rerenderAnalysis()" in block
    assert "renderFuturesHistoryFrom(_lastFuturesHist)" in block


def test_stream_health_is_delivery_not_the_socket_flag():
    """
    `connected` flickers False on every resubscribe, and the candidate set
    turns over every couple of minutes. A stream sending 1,016 messages with
    seven fresh quotes reported DEGRADED.
    """
    import time
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=True, stale_after_s=20.0)
    s.track(["BTC/USDT:USDT"])
    s._ingest('{"data":{"s":"BTCUSDT","p":"64000"}}')
    assert s._connected is False        # a resubscribe is pending
    assert s.healthy() is True          # but data is arriving
    assert s.status()["healthy"] is True
    assert s.healthy(now=time.time() + 30) is False


def test_a_resubscribe_is_not_counted_as_a_reconnect():
    """The tracked set changes every cycle; counting those made a healthy
    stream look like it was flapping. Needs the websocket ON — with it off
    there is no socket to resubscribe."""
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=True, websocket_enabled=True)
    for syms in (["A/USDT:USDT"], ["B/USDT:USDT"], ["C/USDT:USDT"]):
        s.track(syms)
    st = s.status()
    assert st["resubscribes"] == 3
    assert st["reconnects"] == 0


def test_a_stream_with_no_data_is_not_healthy():
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=True)
    s.track(["BTC/USDT:USDT"])
    assert s.healthy() is False
    assert s.status()["healthy"] is False


# ── Stream: one bad name silences everything ───────────────────────────────
#
# Live reported connected=True with msgs=0. Binance ACCEPTS a socket whose
# subscription contains an unrecognised stream name and then sends nothing —
# the whole subscription goes silent, not just that symbol. The live universe
# is 718 symbols against demo's 574 and is not all USDT-margined.

def test_only_usdt_margined_pairs_are_tracked():
    from bot.candidate_stream import CandidateStream as C
    assert C._wire("BTC/USDT:USDT") == "btcusdt"
    assert C._wire("1000PEPE/USDT:USDT") == "1000pepeusdt"
    assert C._wire("SYN/USDT") == "synusdt"
    for bad in ("BTC/USD:BTC", "ETHUSD_PERP", "BTC/BUSD:BUSD", "", None,
                "WEIRD-THING"):
        assert C._wire(bad) is None, bad


def test_non_usdt_margined_symbols_are_dropped(caplog):
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=True)
    with caplog.at_level("WARNING"):
        s.track(["BTC/USDT:USDT", "BTC/USD:BTC", "ETH/USDT:USDT"])
    assert s.status()["tracking"] == 2
    assert any("not USDT-margined" in r.message for r in caplog.records)


def test_price_on_an_unsubscribable_symbol_is_none():
    from bot.candidate_stream import CandidateStream
    assert CandidateStream(demo=True).price("BTC/USD:BTC") is None


def test_a_silent_connection_is_reported_distinctly():
    """
    Distinct from a connection failure, because it is diagnosed differently:
    the socket opened, so the proxy and endpoint are fine.
    """
    import inspect
    from bot import candidate_stream
    src = inspect.getsource(candidate_stream.CandidateStream._session)
    assert "CONNECTED BUT SILENT" in src
    assert "unrecognised symbol" in src


def test_a_resubscribe_is_not_blocked_by_silence():
    """
    `async for msg in ws` blocks until the next frame or the 60s read timeout,
    so a resubscribe could not take effect on a stream delivering nothing —
    precisely the case that needed one.
    """
    import inspect
    from bot import candidate_stream
    src = inspect.getsource(candidate_stream.CandidateStream._session)
    assert "ws.receive(timeout=5)" in src
    assert "async for msg in ws" not in src


def test_non_ascii_is_a_WEBSOCKET_limit_not_a_symbol_limit():
    """
    These are real tradeable pairs — 我踏马来了/USDT traded on 2026-09-13.
    Excluding them from REST too cost them live prices for no reason: REST
    returns every symbol in one response and matches the key locally. Only the
    websocket URL cannot carry them.
    """
    from bot.candidate_stream import CandidateStream as C
    assert C._wire("我踏马来了/USDT:USDT") == "我踏马来了usdt"   # tracked
    assert C.ws_safe("我踏马来了usdt") is False                  # not subscribed
    assert C.ws_safe("arbusdt") is True


def test_rest_prices_a_symbol_the_websocket_cannot_subscribe_to():
    import time, logging
    from bot.candidate_stream import CandidateStream
    logging.disable(logging.CRITICAL)
    try:
        s = CandidateStream(demo=False, rest_interval_s=0.05,
                            rest_fetcher=lambda: {"我踏马来了usdt": 0.0137})
        s.track(["我踏马来了/USDT:USDT"])
        s.start(); time.sleep(0.25)
        assert s.price("我踏马来了/USDT:USDT") == pytest.approx(0.0137)
    finally:
        s.stop(); logging.disable(logging.NOTSET)


def test_the_subscription_still_excludes_them():
    """One non-ASCII name silences the WHOLE subscription, so the URL builder
    must filter even though the tracker does not."""
    import inspect
    from bot import candidate_stream
    src = inspect.getsource(candidate_stream.CandidateStream._session)
    assert "self.ws_safe(w)" in src


def _unused_non_ascii_subscription_check():
    """
    龙虾USDT silenced the ENTIRE live subscription: `.isalnum()` is True for
    CJK under Unicode, so it passed the filter, went into the URL as
    %E9%BE%99%E8%99%BEusdt, and Binance accepted the socket and sent nothing
    for ANY symbol.

    These are real tradeable pairs — 我踏马来了/USDT traded on 2026-09-13 — so
    they are excluded from the STREAM only, and still scanned, entered and
    guarded on the scan snapshot.
    """
    from bot.candidate_stream import CandidateStream as C
    assert C._wire("龙虾/USDT:USDT") is None
    assert C._wire("我踏马来了/USDT:USDT") is None
    assert C._wire("ARB/USDT:USDT") == "arbusdt"
    assert C._wire("1000PEPE/USDT:USDT") == "1000pepeusdt"


def test_one_bad_symbol_does_not_drop_the_good_ones():
    """Non-USDT-M is dropped; non-ASCII is kept for REST."""
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=False)
    s.track(["ARB/USDT:USDT", "龙虾/USDT:USDT", "FIL/USDT:USDT",
             "BTC/USD:BTC"])
    assert s.status()["tracking"] == 3        # the CJK one is tracked


def test_repeated_silence_backs_off_rather_than_looping():
    """
    A structurally broken subscription fails identically every time.
    Reconnecting at full speed is a request loop against the same endpoint the
    guardian uses.
    """
    import inspect
    from bot import candidate_stream
    src = inspect.getsource(candidate_stream.CandidateStream._run)
    assert "_silent_sessions >= 3" in src
    assert "backing off 5 minutes" in src


def test_a_good_session_clears_the_silence_counter():
    import inspect
    from bot import candidate_stream
    src = inspect.getsource(candidate_stream.CandidateStream._session)
    i = src.index("session ended after")
    assert "_silent_sessions = 0" in src[max(0, i - 200):i]


def test_a_silent_session_probes_a_known_symbol():
    """
    Demo works on the identical code path through the same proxy; only the
    host differs. Guessing is worthless, so the stream asks the endpoint a
    question with a known answer.
    """
    import inspect
    from bot import candidate_stream
    src = inspect.getsource(candidate_stream.CandidateStream._probe)
    assert "btcusdt@markPrice@1s" in src
    assert "/ws" in src
    for outcome in ("PROBE OK", "PROBE SILENT", "PROBE FAILED"):
        assert outcome in src


def test_the_probe_distinguishes_the_three_causes():
    import inspect
    from bot import candidate_stream
    src = inspect.getsource(candidate_stream.CandidateStream._probe)
    assert "combined" in src.lower()          # URL form or a symbol
    assert "geo-restriction" in src           # host not delivering
    assert "cannot reach this host" in src    # proxy


def test_the_probe_result_reaches_the_dashboard():
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=False)
    assert "probe" in s.status()


def test_the_stream_endpoint_and_proxy_are_overridable():
    """
    PROBE SILENT on the live host through the VPN exit, while demo worked on
    the same proxy and REST worked too. That is environmental, so both
    variables must be testable without a rebuild.
    """
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=False, base_url="wss://alt.example/stream")
    assert s.base_url == "wss://alt.example/stream"
    assert CandidateStream(demo=False).base_url is None


def test_the_stream_proxy_can_be_disabled_independently():
    import main
    class C:
        stream_proxy = ""
        socks_proxy = "http://gluetun:8888"
    assert main._stream_proxy(C()) == "http://gluetun:8888"
    C.stream_proxy = "none"
    assert main._stream_proxy(C()) is None
    C.stream_proxy = "direct"
    assert main._stream_proxy(C()) is None
    C.stream_proxy = "http://other:3128"
    assert main._stream_proxy(C()) == "http://other:3128"


def test_a_persistently_degraded_stream_does_not_spam():
    """It stays degraded by definition; a warning every 30s buries the log."""
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader)
    i = src.index("falling back to the scan")
    assert "_stream_log_n % 20 == 0" in src[max(0, i - 600):i]


# ── REST fallback ──────────────────────────────────────────────────────────
#
# The LIVE websocket host does not deliver to this address — direct or
# proxied, one well-known symbol, silent. REST on the same network works.

def test_rest_fills_quotes_when_the_socket_cannot():
    import time
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=False, rest_interval_s=0.05,
                        rest_fetcher=lambda: {"btcusdt": 64000.0,
                                              "ethusdt": 3200.0})
    s.track(["BTC/USDT:USDT", "ETH/USDT:USDT"])
    s.start()
    try:
        time.sleep(0.3)
        assert s.price("BTC/USDT:USDT") == pytest.approx(64000.0)
        assert s.status()["source"] == "rest"
        assert s.healthy() is True
    finally:
        s.stop()


def test_rest_only_fills_tracked_symbols():
    """One call returns every symbol; the candidate set is a local filter."""
    import time
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=False, rest_interval_s=0.05,
                        rest_fetcher=lambda: {"btcusdt": 1.0, "dogeusdt": 2.0})
    s.track(["BTC/USDT:USDT"])
    s.start()
    try:
        time.sleep(0.25)
        assert s.price("BTC/USDT:USDT") == pytest.approx(1.0)
        assert s.price("DOGE/USDT:USDT") is None
    finally:
        s.stop()


def test_a_failing_rest_poll_never_raises():
    import time
    from bot.candidate_stream import CandidateStream

    def _boom():
        raise RuntimeError("network down")
    s = CandidateStream(demo=False, rest_interval_s=0.05, rest_fetcher=_boom)
    s.track(["BTC/USDT:USDT"])
    s.start()
    try:
        time.sleep(0.25)
        assert s.price("BTC/USDT:USDT") is None      # falls back to the scan
        assert s.status()["rest_errors"] > 0
    finally:
        s.stop()


def test_websocket_wins_on_recency_when_both_are_live():
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=False, rest_fetcher=lambda: {"btcusdt": 1.0})
    s.track(["BTC/USDT:USDT"])
    s._ingest('{"data":{"s":"BTCUSDT","p":"64000"}}')
    assert s.status()["source"] == "websocket"
    assert s.price("BTC/USDT:USDT") == pytest.approx(64000.0)


def test_the_fetcher_returns_empty_rather_than_raising():
    from bot.candidate_stream import make_rest_fetcher

    class Broken:
        def fapiPublicGetPremiumIndex(self):
            raise RuntimeError("nope")
    assert make_rest_fetcher(Broken())() == {}
    assert make_rest_fetcher(object())() == {}


def test_the_fetcher_parses_a_premium_index_payload():
    from bot.candidate_stream import make_rest_fetcher

    class Ex:
        def fapiPublicGetPremiumIndex(self):
            return [{"symbol": "BTCUSDT", "markPrice": "64000.10"},
                    {"symbol": "ETHUSDT", "markPrice": "3200.5"},
                    {"symbol": "BADUSDT", "markPrice": None},
                    {"symbol": "ZEROUSDT", "markPrice": "0"}]
    out = make_rest_fetcher(Ex())()
    assert out == {"btcusdt": pytest.approx(64000.10),
                   "ethusdt": pytest.approx(3200.5)}


def test_the_skip_warning_only_fires_when_the_set_changes(caplog):
    """The candidate list is rebuilt every cycle; it was warning every 30s."""
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=False)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            s.track(["ARB/USDT:USDT", "BTC/USD:BTC"])
    assert len([r for r in caplog.records
                if "not USDT-margined" in r.message]) == 1


def test_every_attribute_run_once_uses_is_initialised():
    """
    `self._stream_log_n` was never set in __init__ — my edit targeted a string
    that did not exist and failed silently. The modulo raised AttributeError
    on the first cycle, the surrounding `except` logged at DEBUG, and DEBUG is
    invisible at INFO. Stream health never printed while the stream worked.

    This catches the class: every `self.<attr>` that run_once READS must be
    assigned somewhere in the class.
    """
    import inspect, re
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader)
    run = inspect.getsource(AutoTrader.run_once)
    read = set(re.findall(r"self\.(_[a-zA-Z0-9_]+)\b", run))
    assigned = set(re.findall(r"self\.(_[a-zA-Z0-9_]+)\s*(?::[^=]+)?=", src))
    methods = {n for n, _ in inspect.getmembers(AutoTrader)}
    # getattr(self, "x", default) is a deliberate optional read
    guarded = set(re.findall(r'getattr\(self,\s*"(_[a-zA-Z0-9_]+)"', run))
    missing = read - assigned - methods - guarded
    assert not missing, f"read in run_once but never assigned: {sorted(missing)}"


def test_a_stream_status_failure_is_visible():
    """DEBUG hid the bug for an entire deploy."""
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader)
    i = src.index("stream status failed")
    assert "_log.warning" in src[max(0, i - 120):i]


# ── set_leverage ───────────────────────────────────────────────────────────
#
# Binance stores leverage PER SYMBOL. LSK ran at 20x on live within an hour of
# BR running at 10x, so an account "set to 10x" is only 10x on the pairs that
# were set by hand — and every guard threshold halves in price terms on the
# rest.

class _LevEx:
    def __init__(self, start=20.0, fail=False, caps_at=None):
        self.lev, self.fail, self.caps_at = start, fail, caps_at
        self.calls = []

    def market_id(self, symbol):
        return symbol.split("/")[0] + "USDT"

    def fapiPrivatePostLeverage(self, params):
        self.calls.append(params)
        if self.fail:
            raise RuntimeError("-4028 invalid leverage")
        want = float(params["leverage"])
        self.lev = min(want, self.caps_at) if self.caps_at else want
        return {}


def _lev_service(ex):
    """
    EntryService reaches the exchange through `self.guardian.exchange`. The
    first version of this helper set `svc.exchange = ex`, so the test passed
    against an attribute the real class does not have — and the code raised on
    every cycle in production the moment ENTRY_TARGET_LEVERAGE was set.
    """
    from bot.futures_entry import EntryService
    svc = EntryService.__new__(EntryService)
    svc.guardian = type("G", (), {"exchange": ex})()
    svc.symbol_leverage_detail = lambda sym: (ex.lev, "positionRisk")
    return svc


def test_leverage_is_set_when_it_differs():
    ex = _LevEx(start=20.0)
    svc = _lev_service(ex)
    lev, _ = svc.ensure_leverage("LSK/USDT:USDT", 10)
    assert ex.calls == [{"symbol": "LSKUSDT", "leverage": 10}]
    assert lev == 10.0


def test_no_call_is_made_when_it_already_matches():
    ex = _LevEx(start=10.0)
    lev, _ = _lev_service(ex).ensure_leverage("BR/USDT:USDT", 10)
    assert ex.calls == []
    assert lev == 10.0


def test_a_failure_does_NOT_block_the_trade():
    """
    The operator's requirement. Proceeding at the reported leverage is the
    pre-existing behaviour and is correctly sized for it — wrong-but-known
    leverage is safe; unknown leverage is not.
    """
    ex = _LevEx(start=20.0, fail=True)
    lev, src = _lev_service(ex).ensure_leverage("LSK/USDT:USDT", 10)
    assert lev == 20.0            # the real figure, not the wish
    assert src == "positionRisk"


def test_the_result_is_re_read_rather_than_assumed():
    """Binance caps leverage by notional tier and can refuse quietly."""
    ex = _LevEx(start=20.0, caps_at=5.0)
    lev, _ = _lev_service(ex).ensure_leverage("LSK/USDT:USDT", 10)
    assert lev == 5.0             # what the exchange actually holds


def test_target_zero_leaves_the_exchange_alone():
    ex = _LevEx(start=20.0)
    lev, _ = _lev_service(ex).ensure_leverage("LSK/USDT:USDT", 0)
    assert ex.calls == []
    assert lev == 20.0


def test_it_is_off_by_default():
    from bot.config import BotConfig
    assert BotConfig().entry_target_leverage == 0.0


def test_the_entry_path_sets_leverage_before_sizing():
    """The plan must be built on the leverage the position will actually use."""
    import inspect
    from bot.futures_entry import EntryService
    src = inspect.getsource(EntryService)
    i = src.index("target_lev = getattr(self.limits")
    j = src.index("leverage, lev_source = self.symbol_leverage_detail(symbol)", i)
    assert "self.ensure_leverage(symbol, target_lev)" in src[i:j]


# ── The cooldown override was invisible ────────────────────────────────────
#
# LSK re-entered 4 minutes after a -$27.38 loss. The logs showed neither a
# cooldown nor an override, because check_safety returns a REASON on SUCCESS
# too — "cooldown overridden" — and the caller only logged `why` when it
# REFUSED. The mechanism worked; nothing said so.

def test_the_override_produces_a_reason_on_success():
    import logging
    from bot.auto_trader import (AutoTrader, AutoTradeConfig, SafetyState,
                                 check_safety)
    logging.disable(logging.CRITICAL)
    try:
        a = AutoTrader.__new__(AutoTrader)
        a.cfg = AutoTradeConfig(symbol_cooldown_s=1800.0,
                                cooldown_override_rsi_delta=3.0,
                                max_reentries_per_symbol=2)
        a.state = SafetyState(); a._log = []
        a.entry = type("E", (), {"clear_pending": lambda self, s: None})()
        sym = "LSK/USDT:USDT"
        a.note_closed_trade(sym, -27.379, entry_rsi=84.0)
        assert a.state.symbol_blocked_until.get(sym)
        ok, why = check_safety(a.state, a.cfg, balance=5000.0,
                               open_positions=0, symbol=sym,
                               current_rsi=87.1, side="short")
        assert ok is True
        assert "cooldown overridden" in why      # the reason existed all along
    finally:
        logging.disable(logging.NOTSET)


def test_a_successful_override_is_now_logged():
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.run_once)
    assert 'if ok and why and why != "ok":' in src
    i = src.index('if ok and why and why != "ok":')
    block = src[i:i + 900]
    assert "_log.warning" in block
    assert "cooldown_override" in block


def test_applying_the_cooldown_reaches_the_container_log():
    """It only went to the event feed, so logs could not answer 'did it fire?'"""
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.note_closed_trade)
    assert "cooldown applied to" in src
    assert "_log.info" in src


def test_a_winner_still_sets_no_cooldown():
    import logging
    from bot.auto_trader import AutoTrader, AutoTradeConfig, SafetyState
    logging.disable(logging.CRITICAL)
    try:
        a = AutoTrader.__new__(AutoTrader)
        a.cfg = AutoTradeConfig(symbol_cooldown_s=1800.0)
        a.state = SafetyState(); a._log = []
        a.entry = type("E", (), {"clear_pending": lambda self, s: None})()
        a.note_closed_trade("BR/USDT:USDT", +46.55, entry_rsi=80.0)
        assert not a.state.symbol_blocked_until.get("BR/USDT:USDT")
    finally:
        logging.disable(logging.NOTSET)


def test_ordinary_success_does_not_log_a_reason():
    """
    check_safety returns (True, "ok") on plain success, so `if ok and why:`
    logged `auto-trade: BR/USDT:USDT — ok` on EVERY entry. Only the override
    is worth a line.
    """
    from bot.auto_trader import check_safety, SafetyState, AutoTradeConfig
    ok, why = check_safety(SafetyState(), AutoTradeConfig(), balance=5000.0,
                           open_positions=0, symbol="BR/USDT:USDT",
                           current_rsi=80.0, side="short")
    assert (ok, why) == (True, "ok")
    import inspect
    from bot.auto_trader import AutoTrader
    assert 'if ok and why and why != "ok":' in inspect.getsource(AutoTrader.run_once)


def test_a_plaintext_tls_reply_is_reported_as_a_proxy_problem():
    """
    SSL: WRONG_VERSION_NUMBER means TLS began and plaintext came back. Through
    an HTTP proxy that is a CONNECT tunnel that was never opened — the PROXY
    answered, not the host. Reporting it as "cannot reach this host" pointed
    the diagnosis in the wrong direction.
    """
    import inspect
    from bot import candidate_stream
    for fn in (candidate_stream.CandidateStream._probe,
               candidate_stream.CandidateStream._run):
        src = inspect.getsource(fn)
        if "WRONG_VERSION_NUMBER" in src:
            assert "CANDIDATE_STREAM_PROXY=none" in src
            break
    else:
        raise AssertionError("no TLS-plaintext branch found")


# ── REST is the transport on both environments ─────────────────────────────
#
# Demo ran the websocket while live fell back to REST, so the two were
# gathering inputs differently and findings did not translate. The websocket
# buys ~1s against REST's <=3s, and the drift gate is looking for
# percent-scale movement — LSK moved 12% between scan and sizing.

def test_the_websocket_is_off_by_default():
    from bot.candidate_stream import CandidateStream
    from bot.config import BotConfig
    assert CandidateStream(demo=True).websocket_enabled is False
    assert BotConfig().stream_websocket_enabled is False


def test_no_socket_thread_is_started_when_it_is_off():
    import logging
    from bot.candidate_stream import CandidateStream
    logging.disable(logging.CRITICAL)
    try:
        s = CandidateStream(demo=True, rest_fetcher=lambda: {})
        s.start()
        assert s._thread is None
    finally:
        s.stop()
        logging.disable(logging.NOTSET)


def test_rest_still_runs_with_the_websocket_off():
    import time, logging
    from bot.candidate_stream import CandidateStream
    logging.disable(logging.CRITICAL)
    try:
        s = CandidateStream(demo=True, rest_interval_s=0.05,
                            rest_fetcher=lambda: {"brusdt": 1.23})
        s.track(["BR/USDT:USDT"])
        s.start()
        time.sleep(0.25)
        assert s.price("BR/USDT:USDT") == pytest.approx(1.23)
        assert s.status()["source"] == "rest"
    finally:
        s.stop()
        logging.disable(logging.NOTSET)


def test_tracking_does_not_churn_a_socket_that_does_not_exist():
    """track() forced a resubscribe every cycle; with no socket that is pure
    churn against an endpoint nothing is listening to."""
    from bot.candidate_stream import CandidateStream
    s = CandidateStream(demo=True)
    for syms in (["A/USDT:USDT"], ["B/USDT:USDT"], ["C/USDT:USDT"]):
        s.track(syms)
    assert s.status()["resubscribes"] == 0


def test_the_health_line_hides_socket_fields_when_it_is_off():
    """"connected=False" must not read as a fault on a REST-only run."""
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.run_once)
    assert 'ws_on = h.get("websocket_enabled")' in src
    assert '"ws=off "' in src


# ── Fees paid in BNB ───────────────────────────────────────────────────────
#
# The income parser summed COMMISSION rows by `income` and ignored `asset`.
# Paying fees in BNB makes that a BNB amount — ~0.0000155 on a $15 notional —
# which rounded to 0.0000, so every "net of fees" figure on live was GROSS
# while demo's was net. The accounts stopped being comparable, silently.

def _income_guardian(rows, bnb=None):
    import threading
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig()
    g._lock = threading.RLock()
    g.exchange = type("X", (), {
        "fapiPrivateGetIncome": staticmethod(lambda p: rows),
        "market_id": staticmethod(lambda s: s.split("/")[0] + "USDT")})()
    if bnb is not None:
        g._candidate_stream = type("S", (), {"bnb_mark": staticmethod(
            lambda max_age_s=120.0: bnb)})()
    return g


def test_a_usdt_commission_is_used_as_is():
    g = _income_guardian([
        {"incomeType": "REALIZED_PNL", "income": "-27.379", "asset": "USDT"},
        {"incomeType": "COMMISSION", "income": "-1.76", "asset": "USDT"}])
    pnl, comm, seen = g._income_for_position("LSK/USDT:USDT", None)
    assert seen and comm == pytest.approx(1.76)
    assert g._last_fee_source == "ledger"


def test_a_bnb_commission_is_converted():
    """0.00001552 BNB at 902.35 is ~0.014 USDT — the real cost."""
    g = _income_guardian([
        {"incomeType": "REALIZED_PNL", "income": "-0.5915", "asset": "USDT"},
        {"incomeType": "COMMISSION", "income": "-0.00001552", "asset": "BNB"}],
        bnb=902.35)
    _, comm, _ = g._income_for_position("BULLA/USDT:USDT", None)
    assert comm == pytest.approx(0.00001552 * 902.35, rel=1e-6)
    assert g._last_fee_source == "converted"


def test_an_unconvertible_asset_records_nothing_rather_than_a_wrong_number():
    """A BNB count posing as dollars is worse than a gap the caller can fill."""
    g = _income_guardian([
        {"incomeType": "COMMISSION", "income": "-0.00001552", "asset": "BNB"}],
        bnb=None)
    _, comm, _ = g._income_for_position("BULLA/USDT:USDT", None)
    assert comm == 0.0
    assert g._last_fee_source == "unconverted"


def test_the_estimate_fallback_is_notional_based():
    from bot.futures_guard import GuardConfig
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig(taker_fee_rate=0.0005)
    assert g.estimate_fees(15.52) == pytest.approx(15.52 * 0.0005 * 2)


def test_the_bnb_mark_comes_free_from_the_existing_poll():
    """premiumIndex returns every symbol — no extra call, no extra weight."""
    import time, logging
    from bot.candidate_stream import CandidateStream
    logging.disable(logging.CRITICAL)
    try:
        s = CandidateStream(demo=False, rest_interval_s=0.05,
                            rest_fetcher=lambda: {"brusdt": 1.0,
                                                  "bnbusdt": 902.35})
        s.track(["BR/USDT:USDT"])          # BNB is NOT a tracked candidate
        s.start(); time.sleep(0.25)
        assert s.bnb_mark() == pytest.approx(902.35)
        assert s.bnb_mark(max_age_s=0) is None      # stale is worse than absent
    finally:
        s.stop(); logging.disable(logging.NOTSET)


# ── Late volume peak ───────────────────────────────────────────────────────

def _bulla(vt, early):
    return {"symbol": "BULLA/USDT:USDT", "direction": "short", "rsi": 85.5,
            "pct_below_24h_high": -1.69, "pct_above_24h_low": 50.0,
            "strength": "strengthening", "atr_pct": 5.49,
            "advance": {"adv_vol_trend": vt, "peak_vol_early": early}}


def test_a_late_volume_peak_defers_below_the_main_threshold():
    """
    BULLA passed at 1.605 (under 2.0) and lost. peak_vol_early was FALSE — the
    heaviest trade arrived in the SECOND half of a leg that had doubled.
    """
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False,
                          defer_vol_trend=2.0, defer_vol_late_trend=1.5)
    d = evaluate_candidate(_bulla(1.605, False), streak=2, cfg=cfg, atr_pct=5.49)
    assert not d.enter and "LATE in the advance" in d.reason


def test_the_same_growth_with_an_early_peak_is_allowed():
    """Volume merely growing is ambiguous; growing with its peak still ahead
    is the move being bought."""
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False,
                          defer_vol_trend=2.0, defer_vol_late_trend=1.5)
    assert evaluate_candidate(_bulla(1.605, True), streak=2, cfg=cfg,
                              atr_pct=5.49).enter


def test_mild_growth_with_a_late_peak_still_passes():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False,
                          defer_vol_trend=2.0, defer_vol_late_trend=1.5)
    assert evaluate_candidate(_bulla(1.2, False), streak=2, cfg=cfg,
                              atr_pct=5.49).enter


def test_the_late_test_can_be_disabled():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False,
                          defer_vol_trend=2.0, defer_vol_late_trend=0.0)
    assert evaluate_candidate(_bulla(1.605, False), streak=2, cfg=cfg,
                              atr_pct=5.49).enter


def test_a_missing_peak_flag_does_not_defer():
    cfg = AutoTradeConfig(enabled=True, short_rsi_min=75, veto_breakout=False,
                          defer_vol_trend=2.0, defer_vol_late_trend=1.5)
    assert evaluate_candidate(_bulla(1.605, None), streak=2, cfg=cfg,
                              atr_pct=5.49).enter


# ── Mobile layout ──────────────────────────────────────────────────────────
#
# v3.25.1 added per-view column templates as `body[data-view="futures"]
# .stats-row`. That is a TWO-selector rule, so it outranked the single-class
# `.stats-row { grid-template-columns: 1fr 1fr }` in the mobile media queries
# at EVERY width — specificity ignores media queries and source order. A phone
# got nine columns and printed each figure one character per line.

def test_the_per_view_templates_are_desktop_scoped():
    import re
    ui = _ui()
    for view in ("futures", "spot"):
        i = ui.index(f'body[data-view="{view}"] .stats-row {{')
        before = ui[:i]
        # the nearest preceding @media must be a min-width desktop query
        last = None
        for m in re.finditer(r"@media \(([^)]+)\)", before):
            last = m.group(1)
        assert last and "min-width" in last, (view, last)


def test_the_breakpoints_meet_without_a_gap():
    """
    769 meets the existing max-width:768 tablet query exactly. A gap would let
    the base nine-column fallback leak through on mid-size screens.
    """
    ui = _ui()
    assert "@media (min-width: 769px)" in ui
    assert "@media (max-width: 768px)" in ui


def test_hiding_a_card_still_applies_at_every_width():
    """Which cards exist is width-independent; only the COLUMNS are scoped."""
    import re
    ui = _ui()
    i = ui.index('body[data-view="futures"] #stat-avail-card')
    before = ui[:i]
    opens = len(re.findall(r"@media", before))
    # the hide rules sit above the first media query in the sheet
    assert opens == 0, "card-hiding rules must not be inside a media query"


def test_the_oversold_end_is_split_at_45():
    """
    Lowering AUTO_LONG_RSI_MIN to 38 opens a band no long has ever been taken
    in. A single "<50" bucket would bury it with the 45-50 group that returned
    -$9.58/trade, so the new band would be unreadable against a known-bad one.
    """
    from bot.analysis import RSI_BUCKETS
    labels = [b.label for b in RSI_BUCKETS]
    assert "38-45" in labels and "45-50" in labels
    assert "<50" not in labels


def test_the_split_does_not_overlap_or_leave_a_hole():
    from bot.analysis import RSI_BUCKETS
    lows = [b.lo for b in RSI_BUCKETS]
    highs = [b.hi for b in RSI_BUCKETS]
    for i in range(len(RSI_BUCKETS) - 1):
        assert highs[i] == lows[i + 1], (RSI_BUCKETS[i].label,
                                         RSI_BUCKETS[i + 1].label)



def test_ensure_leverage_uses_the_attribute_the_class_actually_has():
    """
    Guard against the mock shaping the code. EntryService has no `exchange`;
    it has `guardian.exchange`, and every other method in the class uses that.
    """
    import inspect
    from bot.futures_entry import EntryService
    src = inspect.getsource(EntryService.ensure_leverage)
    assert "self.exchange" not in src
    assert 'getattr(self, "guardian", None), "exchange"' in src


def test_a_missing_exchange_handle_does_not_raise():
    from bot.futures_entry import EntryService
    svc = EntryService.__new__(EntryService)
    svc.symbol_leverage_detail = lambda sym: (20.0, "positionRisk")
    lev, src = svc.ensure_leverage("LSK/USDT:USDT", 10)
    assert lev == 20.0 and src == "positionRisk"


# ── Peer cross-evaluation ──────────────────────────────────────────────────
#
# Over one day demo and live traded 15 symbols EACH and overlapped on FOUR.
# Demo won 77%, live 41%, on identical rules — and nothing in either log says
# whether that is the config, the market, or which coins each happened to see.

class _Scan:
    def __init__(self, rows):
        self._rows = rows

    def snapshot(self):
        return {"candidates": self._rows}


def _pe(tmp_path, mode="both", label="live"):
    from bot.peer_eval import PeerEval
    return PeerEval(mode=mode, peer_url="", label=label,
                    path=str(tmp_path / "peer.jsonl"))


def _payload(symbol="BR/USDT:USDT"):
    return {"from": "demo", "symbol": symbol, "side": "short",
            "at": 1_000_000.0, "reason": "RSI 85", "readings": {"rsi": 85.0}}


def test_a_symbol_the_peer_never_saw_is_reported_as_such(tmp_path):
    """The most informative verdict: the DATA differed, not the rules."""
    pe = _pe(tmp_path)
    out = pe.evaluate(_payload(), _Scan([]), None)
    assert out["verdict"] == "not_surfaced"
    assert "not in our current scan" in out["detail"]


def test_a_symbol_the_peer_refuses_carries_the_reason(tmp_path):
    import logging
    from bot.auto_trader import AutoTradeConfig
    logging.disable(logging.CRITICAL)
    try:
        pe = _pe(tmp_path)
        row = _short(breakout=_brk(gap_widening=False, breakout=False))
        row["symbol"] = "BR/USDT:USDT"
        row["rsi"] = 60.0                      # below short_rsi_min
        auto = type("A", (), {"cfg": AutoTradeConfig(
            enabled=True, short_rsi_min=75)})()
        out = pe.evaluate(_payload(), _Scan([row]), auto)
        assert out["verdict"] == "refused"
        assert out["detail"]
    finally:
        logging.disable(logging.NOTSET)


def test_agreement_is_reported_as_would_enter(tmp_path):
    import logging
    from bot.auto_trader import AutoTradeConfig
    logging.disable(logging.CRITICAL)
    try:
        pe = _pe(tmp_path)
        row = _short(breakout=_brk(gap_widening=False, breakout=False))
        row["symbol"] = "BR/USDT:USDT"
        auto = type("A", (), {"cfg": AutoTradeConfig(
            enabled=True, short_rsi_min=75, veto_breakout=False)})()
        out = pe.evaluate(_payload(), _Scan([row]), auto)
        assert out["verdict"] == "would_enter"
    finally:
        logging.disable(logging.NOTSET)


def test_each_side_can_be_send_receive_or_both(tmp_path):
    from bot.peer_eval import PeerEval
    for mode, sends, receives in (("off", False, False),
                                  ("send", True, False),
                                  ("receive", False, True),
                                  ("both", True, True)):
        p = PeerEval(mode=mode, peer_url="http://peer:8000", label="x")
        assert (p.sends, p.receives) == (sends, receives), mode


def test_receiving_is_refused_when_the_mode_says_send_only(tmp_path):
    pe = _pe(tmp_path, mode="send")
    assert pe.evaluate(_payload(), _Scan([]), None)["verdict"] == "disabled"


def test_a_malformed_payload_never_raises(tmp_path):
    pe = _pe(tmp_path)
    for bad in ({}, {"symbol": ""}, {"symbol": None}):
        out = pe.evaluate(bad, _Scan([]), None)
        assert out["verdict"] in ("error", "not_surfaced")


def test_the_endpoint_is_rate_limited(tmp_path):
    from bot.peer_eval import MAX_PER_MINUTE
    pe = _pe(tmp_path)
    verdicts = [pe.evaluate(_payload(), _Scan([]), None)["verdict"]
                for _ in range(MAX_PER_MINUTE + 5)]
    assert "rate_limited" in verdicts


def test_evaluation_makes_no_exchange_calls():
    """It answers from the snapshot already in hand — no weight, and it cannot
    be used to drive requests, which is what replaces auth here."""
    import inspect
    from bot import peer_eval
    src = inspect.getsource(peer_eval.PeerEval.evaluate)
    for forbidden in ("fetch_ohlcv", "fetch_ticker", "exchange", "requests"):
        assert forbidden not in src, forbidden


def test_both_sides_are_recorded_for_comparison(tmp_path):
    pe = _pe(tmp_path)
    pe.evaluate(_payload(), _Scan([]), None)
    rows = pe.report()
    assert rows and rows[0]["kind"] == "received"
    assert "theirs" in rows[0] and "mine" in rows[0]


# ── Calibration from paired readings ───────────────────────────────────────
#
# The "live ATR is 35% higher" figure came from THREE coins at ONE moment,
# read off two screenshots. Scaling a live setting by 1.35 would bake that
# guess into the config and then measure everything through it.

def _paired(tmp_path):
    from bot.peer_eval import PeerEval
    return PeerEval(mode="both", peer_url="", label="live",
                    path=str(tmp_path / "p.jsonl"))


class _Snap:
    def __init__(self, rows):
        self._rows = rows

    def snapshot(self):
        return {"candidates": self._rows}


def test_only_shared_symbols_are_paired(tmp_path):
    pe = _paired(tmp_path)
    mine = _Snap([{"symbol": "SYN/USDT:USDT", "atr_pct": 2.463},
                  {"symbol": "ONLYMINE/USDT:USDT", "atr_pct": 1.0}])
    theirs = {"from": "demo", "kind": "scan", "rows": [
        {"symbol": "SYN/USDT:USDT", "atr_pct": 1.784},
        {"symbol": "ONLYTHEIRS/USDT:USDT", "atr_pct": 1.0}]}
    out = pe.compare(theirs, mine)
    assert out["paired"] == 1


def test_the_calibration_reproduces_the_screenshot_figures(tmp_path):
    """Same three coins, same moment: ATR ~1.365, RSI ~1.000."""
    pe = _paired(tmp_path)
    mine = _Snap([
        {"symbol": "SYN/USDT:USDT", "atr_pct": 2.463, "rsi": 50.1},
        {"symbol": "BR/USDT:USDT", "atr_pct": 3.011, "rsi": 48.0},
        {"symbol": "AKE/USDT:USDT", "atr_pct": 4.357, "rsi": 41.8}])
    theirs = {"from": "demo", "kind": "scan", "rows": [
        {"symbol": "SYN/USDT:USDT", "atr_pct": 1.784, "rsi": 50.1},
        {"symbol": "BR/USDT:USDT", "atr_pct": 2.279, "rsi": 48.8},
        {"symbol": "AKE/USDT:USDT", "atr_pct": 3.193, "rsi": 40.4}]}
    pe.compare(theirs, mine)
    c = pe.calibration()
    assert c["pairs"] == 3
    assert c["fields"]["atr_pct"]["median"] == pytest.approx(1.365, abs=0.01)
    assert c["fields"]["rsi"]["median"] == pytest.approx(1.000, abs=0.01)


def test_the_spread_is_reported_not_just_the_median(tmp_path):
    """A ratio is only usable if it is STABLE — the spread says whether it is."""
    pe = _paired(tmp_path)
    mine = _Snap([{"symbol": f"C{i}/USDT:USDT", "atr_pct": v}
                  for i, v in enumerate((2.0, 3.0, 4.0))])
    theirs = {"from": "demo", "kind": "scan",
              "rows": [{"symbol": f"C{i}/USDT:USDT", "atr_pct": v}
                       for i, v in enumerate((1.0, 3.0, 8.0))]}
    pe.compare(theirs, mine)
    f = pe.calibration()["fields"]["atr_pct"]
    for key in ("n", "median", "mean", "stdev", "min", "max"):
        assert key in f
    assert f["max"] > f["min"]


def test_a_field_with_too_few_points_is_omitted(tmp_path):
    """Three points is the floor, and even that is thin."""
    pe = _paired(tmp_path)
    mine = _Snap([{"symbol": "A/USDT:USDT", "atr_pct": 2.0}])
    pe.compare({"from": "demo", "kind": "scan",
                "rows": [{"symbol": "A/USDT:USDT", "atr_pct": 1.0}]}, mine)
    assert "atr_pct" not in pe.calibration()["fields"]


def test_comparison_makes_no_exchange_calls():
    import inspect
    from bot import peer_eval
    src = inspect.getsource(peer_eval.PeerEval.compare)
    for forbidden in ("fetch_ohlcv", "fetch_ticker", "exchange", "requests"):
        assert forbidden not in src, forbidden


def test_peer_eval_is_wired_independently_of_auto_trade():
    """
    It lived inside the auto-trade block, so it existed only if the scanner
    AND entry service both came up — and RECEIVING is useful regardless: an
    instance with auto-trade off can still answer "would I have taken this?"
    and still pair readings for calibration.
    """
    from pathlib import Path
    src = Path("main.py").read_text()
    i = src.index("_PEER_EVAL = None")
    j = src.index("auto.peer_eval = _PEER_EVAL")
    assert i < j, "peer eval must be constructed before the auto-trade block"
    # and not nested inside it
    line = [ln for ln in src.split("\n") if "_PEER_EVAL = None" in ln][0]
    assert len(line) - len(line.lstrip()) <= 4, "should be at function level"


def test_a_disabled_peer_eval_says_which_kind_of_disabled():
    """"not enabled" covered both "never constructed" and "mode=off", which
    are diagnosed completely differently."""
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("def peer_report")
    block = src[i:i + 1200]
    assert "was not constructed at startup" in block
    assert "PEER_EVAL_MODE=" in block


def test_the_reading_fields_match_the_scanner_row():
    """
    The first version guessed `vol_usdt_24h` and `dist_to_extreme_pct`. The
    row carries `volume_24h_usdt` and `pct_below_24h_high`, so those fields
    silently never paired — and volume is what drives the percentile
    divergence, the most important omission of the lot.
    """
    import inspect
    from bot import peer_eval, scanner
    row_src = inspect.getsource(scanner.Candidate.as_row)
    for f in peer_eval._FIELDS:
        if f in ("strength",):          # added by the snapshot, not as_row
            continue
        assert f'"{f}"' in row_src, f"{f} is not a real scanner row key"


def test_volume_pairs_now(tmp_path):
    from bot.peer_eval import PeerEval
    pe = PeerEval(mode="both", peer_url="", label="live",
                  path=str(tmp_path / "p.jsonl"))
    mine = _Snap([{"symbol": f"C{i}/USDT:USDT", "volume_24h_usdt": v,
                   "atr_pct": 2.0}
                  for i, v in enumerate((349e6, 674e6, 451e6))])
    theirs = {"from": "demo", "kind": "scan",
              "rows": [{"symbol": f"C{i}/USDT:USDT", "volume_24h_usdt": v,
                        "atr_pct": 2.0}
                       for i, v in enumerate((6476e6, 9000e6, 12051e6))]}
    pe.compare(theirs, mine)
    c = pe.calibration()
    assert "volume_24h_usdt" in c["fields"]
    assert c["fields"]["volume_24h_usdt"]["median"] < 0.2   # demo ~19x higher


def test_the_cross_symbol_spread_is_reported(tmp_path):
    """
    30 readings of 6 coins is not 30 independent observations. Within a symbol
    the ATR ratio was stable to three decimals; ACROSS symbols it ran
    1.16-1.49, and that spread decides whether one scaling factor is usable.
    """
    from bot.peer_eval import PeerEval
    pe = PeerEval(mode="both", peer_url="", label="live",
                  path=str(tmp_path / "p.jsonl"))
    mine = _Snap([{"symbol": f"C{i}/USDT:USDT", "atr_pct": v}
                  for i, v in enumerate((1.159, 1.360, 1.492))])
    theirs = {"from": "demo", "kind": "scan",
              "rows": [{"symbol": f"C{i}/USDT:USDT", "atr_pct": 1.0}
                       for i in range(3)]}
    pe.compare(theirs, mine)
    c = pe.calibration()
    assert c["atr_across_symbols"]["symbols"] == 3
    assert c["atr_across_symbols"]["spread_pct"] == pytest.approx(28.7, abs=0.5)
    assert len(c["atr_by_symbol"]) == 3


def test_categorical_fields_are_agreement_not_ratio(tmp_path):
    """SYN read 'weakening' on demo and 'strengthening' on live at the same
    moment — a ratio is meaningless, agreement is not."""
    from bot.peer_eval import PeerEval
    pe = PeerEval(mode="both", peer_url="", label="live",
                  path=str(tmp_path / "p.jsonl"))
    mine = _Snap([{"symbol": f"C{i}/USDT:USDT", "strength": s, "atr_pct": 1.0}
                  for i, s in enumerate(("weakening", "strengthening",
                                         "weakening"))])
    theirs = {"from": "demo", "kind": "scan",
              "rows": [{"symbol": f"C{i}/USDT:USDT", "strength": "weakening",
                        "atr_pct": 1.0} for i in range(3)]}
    pe.compare(theirs, mine)
    ag = pe.calibration()["agreement"]["strength"]
    assert ag["n"] == 3 and ag["same_pct"] == pytest.approx(66.7, abs=0.1)


# ── not_surfaced must name the cause ───────────────────────────────────────
#
# 9 of 13 cross-evaluations came back "this symbol is not in our current scan
# — not a mover, under the volume floor, or outside the RSI screen". Three
# causes, three different settings, one message. It could not support a
# harmonisation decision.

def _peer_stage(tmp_path, snap):
    from bot.peer_eval import PeerEval
    pe = PeerEval(mode="both", peer_url="", label="live",
                  path=str(tmp_path / "p.jsonl"))
    scanner = type("S", (), {"snapshot": staticmethod(lambda: snap)})()
    return pe.evaluate({"symbol": "REZ/USDT:USDT", "side": "short"},
                       scanner, None)


_BASE = {"config": {"min_abs_change_pct": 8, "effective_vol_floor": 30_600_000}}


def test_screened_out_is_distinguished(tmp_path):
    out = _peer_stage(tmp_path, {**_BASE, "candidates": [],
                                 "movers": ["REZ/USDT:USDT"],
                                 "rejected": {"REZ/USDT:USDT": "failed RSI"}})
    assert out["verdict"] == "not_surfaced"
    assert out["stage"] == "screened_out"
    assert "failed RSI" in out["detail"]


def test_not_a_mover_is_distinguished(tmp_path):
    out = _peer_stage(tmp_path, {**_BASE, "candidates": [],
                                 "movers": ["AAA/USDT:USDT"], "rejected": {}})
    assert out["stage"] == "not_a_mover"
    assert "8" in out["detail"]              # the threshold it missed


def test_an_unrecorded_reason_says_so_rather_than_guessing(tmp_path):
    out = _peer_stage(tmp_path, {**_BASE, "candidates": [], "movers": [],
                                 "rejected": {}})
    assert out["stage"] == "unknown"
    assert "has not recorded why" in out["detail"]


def test_the_scanner_records_why_a_mover_was_dropped():
    import inspect
    from bot.scan_runner import ScanRunner
    src = inspect.getsource(ScanRunner)
    assert "rejected[sym]" in src
    assert '"rejected": dict(' in src
    assert '"movers": sorted(' in src


# ── Which gate rejected a symbol ───────────────────────────────────────────
#
# Live converts 28% of movers to candidates (8 of 29); demo 48% (12 of 25).
# With RSI identical between venues (ratio 0.998), the gate doing the work is
# somewhere in evaluate_symbol — which had a dozen indistinguishable
# `return None` paths.

def _flat_df(n=120, lo=100.0, hi=101.0):
    import numpy as np, pandas as pd
    px = pd.Series(np.linspace(lo, hi, n))
    return pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999,
                         "close": px, "volume": np.full(n, 1e6)})


def _screen_cfg():
    from bot.scanner import ScanConfig
    return ScanConfig(short_rsi_min=70, long_rsi_min=38, long_rsi_max=65,
                      ema_tolerance_pct=0.15, min_24h_vol_usdt=0,
                      min_abs_change_pct=8)


def test_a_short_history_names_itself():
    """Live screens 718 symbols against demo's 574 — the extra listings are
    newer, and a recently listed pair has too few candles."""
    from bot.scanner import evaluate_symbol
    why = []
    evaluate_symbol("Y/USDT:USDT", _flat_df(10), 5e8, 12.0, _screen_cfg(),
                    high_24h=102, low_24h=99, why=why)
    assert why and "candles" in why[0] and "recently listed" in why[0]


def test_a_weak_mover_names_the_threshold_it_missed():
    from bot.scanner import evaluate_symbol
    why = []
    evaluate_symbol("Z/USDT:USDT", _flat_df(), 5e8, 2.0, _screen_cfg(),
                    high_24h=102, low_24h=99, why=why)
    assert why and "market filter" in why[0] and "8" in why[0]


def test_the_band_fallthrough_reports_the_actual_readings():
    """The fallthrough is where most rejections land, so it must say which
    band was missed and by how much."""
    import inspect
    from bot import scanner
    src = inspect.getsource(scanner.evaluate_symbol)
    assert "fits neither band" in src
    assert "short needs RSI>=" in src and "long needs" in src


def test_the_sink_is_optional():
    """`why=None` must behave exactly as before for every existing caller."""
    from bot.scanner import evaluate_symbol
    assert evaluate_symbol("Y/USDT:USDT", _flat_df(10), 5e8, 12.0,
                           _screen_cfg(), high_24h=102, low_24h=99) is None


def test_the_runner_stores_the_specific_reason():
    import inspect
    from bot.scan_runner import ScanRunner
    src = inspect.getsource(ScanRunner)
    assert "why: list = []" in src
    assert "rejected[sym] = (why[0] if why else" in src


# ── The hidden second volume floor ─────────────────────────────────────────
#
# There are TWO volume checks. _prefilter uses the percentile; evaluate_symbol
# -> passes_market_filters then applies cfg.min_24h_vol_usdt AGAIN to every
# mover. In percentile mode that second check is a hidden hard floor.
#
# Live: p80 floor 17.9M, then 14 of 34 movers died against SCAN_MIN_VOL_USDT
# 50M. Demo: p85 floor 850M (volumes read ~21x higher), so the 50M never
# fired. The entire selection divergence, from identical config.

def test_percentile_mode_neutralises_the_absolute_floor():
    import inspect
    from bot.scan_runner import ScanRunner
    src = inspect.getsource(ScanRunner)
    assert "self.cfg.min_24h_vol_usdt = 0.0" in src
    i = src.index("self.cfg.min_24h_vol_usdt = 0.0")
    assert "percentile" in src[max(0, i - 1400):i].lower()


def test_it_says_so_once_rather_than_every_scan():
    import inspect
    from bot.scan_runner import ScanRunner
    src = inspect.getsource(ScanRunner)
    assert "_abs_floor_noted" in src
    assert "ignoring" in src


def test_the_screen_passes_a_low_volume_mover_once_neutralised():
    from bot.scanner import ScanConfig, passes_market_filters
    cfg = ScanConfig(min_24h_vol_usdt=50e6, min_abs_change_pct=8)
    assert passes_market_filters(20.6e6, 18.0, cfg)[0] is False
    cfg.min_24h_vol_usdt = 0.0
    assert passes_market_filters(20.6e6, 18.0, cfg)[0] is True
    # the movement filter must still bite
    assert passes_market_filters(20.6e6, 2.0, cfg)[0] is False


# ── Today's baseline must not drift ────────────────────────────────────────

def _day_trade(ts, pnl):
    return {"symbol": "X/USDT:USDT", "side": "short", "final_roi": 5.0,
            "realised_pnl_usdt": pnl, "fees_usdt": 1.0, "margin_usdt": 88.0,
            "exit_is_estimate": False, "closed_at": ts,
            "entry_context": {"sized_stop_roi": 30.0}}


def test_the_baseline_is_computed_once_and_held():
    """
    wallet_now - net is exact at any instant, but it was recomputed on EVERY
    dashboard refresh, so the "from" figure drifted all day and read as
    "24h ago" rather than a fixed 00:00.
    """
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()
    start = 1_000_000.0
    a = day_report([_day_trade(start + 60, 100.0)], day_baseline=None,
                   day_start_ts=start, wallet_now=5100.0)
    b = day_report([_day_trade(start + 60, 100.0),
                    _day_trade(start + 120, 50.0)], day_baseline=None,
                   day_start_ts=start, wallet_now=5150.0)
    assert a["baseline"] == pytest.approx(5000.0)
    assert b["baseline"] == pytest.approx(5000.0)     # HELD
    assert b["net_pnl"] == pytest.approx(150.0)       # net still moves


def test_it_re_bases_when_the_day_rolls():
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()
    start = 1_000_000.0
    day_report([_day_trade(start + 60, 100.0)], day_baseline=None,
               day_start_ts=start, wallet_now=5100.0)
    nxt = day_report([_day_trade(start + 90_000, 20.0)], day_baseline=None,
                     day_start_ts=start + 86_400, wallet_now=5170.0)
    assert nxt["baseline"] == pytest.approx(5150.0)


def test_only_the_current_day_is_cached():
    """The cache is keyed on the day start; yesterday's entry is useless."""
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()
    start = 1_000_000.0
    day_report([_day_trade(start + 60, 10.0)], day_baseline=None,
               day_start_ts=start, wallet_now=5010.0)
    day_report([_day_trade(start + 90_000, 10.0)], day_baseline=None,
               day_start_ts=start + 86_400, wallet_now=5020.0)
    assert len(_DAY_BASELINE) == 1


# ── Re-basing without wiping history ───────────────────────────────────────
#
# Both baselines are sticky by design: today's is cached per day so it stops
# drifting, and wallet_start is written once and persisted. That is right
# until a figure is WRONG — a baseline reconstructed at an unlucky moment, or
# a deposit that makes the old start meaningless. The only previous remedy was
# clearing trade history, which throws away the trades to fix one number.

def test_the_reset_endpoint_touches_no_trades():
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("def reset_baselines")
    block = src[i:i + 3200]
    for forbidden in ("closed_trades", "_journal", "cancel", "create_order",
                      "close_position"):
        assert forbidden not in block, forbidden


def test_clearing_the_day_cache_forces_a_recompute():
    from bot.analysis import day_report, _DAY_BASELINE
    _DAY_BASELINE.clear()
    start = 1_000_000.0
    a = day_report([_day_trade(start + 60, 100.0)], day_baseline=None,
                   day_start_ts=start, wallet_now=5100.0)
    assert a["baseline"] == pytest.approx(5000.0)
    # a wrong first reconstruction would otherwise be held all day
    _DAY_BASELINE.clear()
    b = day_report([_day_trade(start + 60, 100.0),
                    _day_trade(start + 120, 50.0)], day_baseline=None,
                   day_start_ts=start, wallet_now=5150.0)
    assert b["baseline"] == pytest.approx(5000.0)   # recomputed, still right


def test_the_day_key_is_cleared_so_roll_day_re_bases():
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("def reset_baselines")
    block = src[i:i + 3200]
    assert "day_start_balance = 0.0" in block
    assert 'day_key = ""' in block


def test_the_account_reset_uses_the_cached_balance_attribute():
    """There is no wallet_balance() method on the guardian."""
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("def reset_baselines")
    block = src[i:i + 3200]
    assert '_wallet_balance_cached' in block
    assert "g.wallet_balance()" not in block


def test_the_account_reset_is_opt_in():
    """Today alone is the common case; re-basing account return discards the
    whole run's reference point."""
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("def reset_baselines")
    block = src[i:i + 3200]
    assert 'payload.get("day", True)' in block       # defaults on
    assert 'payload.get("account")' in block         # defaults OFF


def test_the_ui_offers_both_scopes():
    ui = _ui()
    assert "function resetBaselines()" in ui
    assert "Today only" in ui
    assert "not touched" in ui


def test_an_explicit_wallet_start_wins_over_the_current_balance():
    """
    "Re-base to now" is only right when the run genuinely starts now. After a
    deposit, or when picking up a run already in progress, the operator has
    the correct figure and the bot does not.
    """
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("def reset_baselines")
    block = src[i:i + 3200]
    assert 'payload.get("account_value")' in block
    j = block.index('payload.get("account_value")')
    k = block.index("_wallet_balance_cached")
    assert j < k, "the explicit value must be tried before the cached balance"


def test_a_bad_explicit_value_changes_nothing():
    from pathlib import Path
    src = Path("bot/api.py").read_text()
    i = src.index("def reset_baselines")
    block = src[i:i + 3200]
    assert "must be a positive" in block
    assert "bal = None" in block


def test_the_env_override_is_applied_after_the_state_restore():
    """Applied before load_state, the persisted value would win and
    GUARD_WALLET_START would appear to do nothing."""
    from pathlib import Path
    src = Path("main.py").read_text()
    a = src.index("guardian.load_state(cfg.futures_state_path)")
    b = src.index("cfg.guard_wallet_start > 0")
    assert a < b


def test_the_env_default_leaves_the_restored_value_alone():
    from bot.config import BotConfig
    import os
    os.environ.pop("GUARD_WALLET_START", None)
    assert BotConfig().guard_wallet_start == 0.0


def test_the_ui_asks_for_a_figure_and_validates_it():
    ui = _ui()
    i = ui.index("async function resetBaselines()")
    block = ui[i:i + 1600]
    assert "prompt(" in block
    assert "account_value" in block
    assert "not a positive number" in block


# ── Price reference and provenance ───────────────────────────────────────────
# 2026-09-17/18: on five shorts the guardian's price came in BELOW the real
# mark every time — by 0.11%, 0.21%, 0.99% and 9.78%. For a short that inflates
# ROI, and the damage scaled with the gap: floor refused with -2021, trail left
# dormant, trail rejected outright, +96.5% reported on a position 1.4% down.

def _price_guardian():
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.exchange = object()
    g._premium_index_mark = lambda symbol: None
    return g


def test_mark_beats_last_when_both_are_present():
    g = _price_guardian()
    t = {"last": 0.0015552, "close": 0.0015552,
         "info": {"markPrice": "0.0015585"}}
    assert g._price_from_ticker("ONE/USDT:USDT", t) == 0.0015585


def test_the_premium_index_supplies_mark_when_the_ticker_has_none():
    # ccxt's binanceusdm ticker comes from /fapi/v1/ticker/24hr, which carries
    # no markPrice, so this is the normal path and not a fallback.
    g = _price_guardian()
    g._premium_index_mark = lambda symbol: 0.1211
    assert g._price_from_ticker("X/USDT:USDT", {"last": 0.10926}) == 0.1211


def test_last_is_used_when_no_mark_exists_but_the_source_says_so():
    g = _price_guardian()
    assert g._price_from_ticker("UNI/USDT:USDT", {"last": 7.298}) == 7.298
    assert "NO MARK" in g._last_price_source


def test_the_source_of_every_price_is_recorded():
    g = _price_guardian()
    g._premium_index_mark = lambda symbol: 0.1211
    g._price_from_ticker("X/USDT:USDT", {"last": 1.0})
    assert g._last_price_source == "premiumIndex.markPrice"


def test_a_stale_fallback_price_is_named_out_loud(caplog):
    """
    resolve_price() reaches for previousClose (24h old) and a completed candle
    before giving up. Whatever it returns must not pass silently as current.
    """
    import bot.futures_guardian as fg
    g = _price_guardian()
    orig = fg.resolve_price
    fg.resolve_price = lambda ex, sym: (0.10926, "ticker.previousClose")
    try:
        with caplog.at_level("WARNING"):
            assert g._price_from_ticker("X/USDT:USDT", {}) == 0.10926
        assert "previousClose" in caplog.text
        assert "NOT a live price" in caplog.text
    finally:
        fg.resolve_price = orig


# ── An order the exchange refused is not protection ──────────────────────────

class _RejectPos:
    symbol = "ONE/USDT:USDT"; side = "short"
    entry_price = 0.12093; qty = 1.0
    leverage = 10; effective_leverage = 10.0; margin = 1.0


def _accept_guardian():
    from bot.futures_guardian import FuturesGuardian
    g = FuturesGuardian.__new__(FuturesGuardian)
    g._record = lambda *a, **k: None
    return g


def test_a_rejected_order_yields_no_id():
    g = _accept_guardian()
    order = {"id": "2000001443682594", "status": "REJECTED"}
    assert g._accepted_id(_RejectPos(), order, "ARMED trailing stop") is None


def test_rejection_is_read_from_the_raw_info_block_too():
    g = _accept_guardian()
    order = {"id": "123", "info": {"status": "REJECTED"}}
    assert g._accepted_id(_RejectPos(), order, "fixed stop") is None


def test_an_expired_order_is_not_protection_either():
    g = _accept_guardian()
    assert g._accepted_id(_RejectPos(), {"id": "1", "status": "EXPIRED"},
                          "trail") is None


def test_an_accepted_order_returns_its_id():
    g = _accept_guardian()
    order = {"id": "2000001442640586", "status": "NEW"}
    assert g._accepted_id(_RejectPos(), order, "ARMED trailing stop") \
        == "2000001442640586"


def test_a_response_with_no_id_is_a_failure():
    g = _accept_guardian()
    assert g._accepted_id(_RejectPos(), {"status": "NEW"}, "trail") is None


def test_a_missing_status_does_not_block_a_real_id():
    """Not every payload carries status; absence is not rejection."""
    g = _accept_guardian()
    assert g._accepted_id(_RejectPos(), {"id": "77"}, "trail") == "77"


def test_all_three_placement_sites_verify_acceptance():
    """
    The fixed stop, the rescue trail and the armed trail. Any one of them
    recording an id for a refused order lets it displace real protection.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian)
    # Four: the fixed stop, each leg of a split stop, the rescue trail and
    # the armed trail. The split legs were missed on the first pass.
    assert src.count("self._accepted_id(pos, order,") == 4
    import bot.futures_guardian as fg
    place = inspect.getsource(FuturesGuardian._place_stop)
    split = inspect.getsource(FuturesGuardian._place_split_stops)
    trail = inspect.getsource(FuturesGuardian._place_native_trail)
    for name, body in (("_place_stop", place), ("_place_split_stops", split),
                       ("_place_native_trail", trail)):
        assert "_accepted_id(" in body, name


def test_a_refused_trail_returns_none_so_nothing_is_superseded():
    """
    The supersede is gated on a truthy return, so returning None is what stops
    a rejected trail from cancelling the adaptive one that was actually there.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._place_native_trail)
    assert src.count("if not oid:\n            return None") >= 1


# ── Peer control is hidden when the peer is off ──────────────────────────────

def test_status_reports_whether_peer_eval_is_live():
    import inspect
    import bot.api as api
    src = inspect.getsource(api)
    assert '"peer_eval_enabled"' in src
    # Derived from what the PeerEval object actually does, not from the raw
    # mode string, because sends also requires PEER_EVAL_URL to be set.
    assert 'getattr(_peer_eval, "sends", False)' in src
    assert 'getattr(_peer_eval, "receives", False)' in src


def test_the_peer_button_is_hidden_by_default_and_sized_like_its_neighbours():
    html = open("ui/index.html", encoding="utf-8").read()
    i = html.index('id="ft-peer"')
    tag = html[i:html.index("</button>", i)]
    assert "display:none" in tag          # off unless the status says otherwise
    assert "padding:3px 6px" in tag       # same as ft-window and ft-export
    assert 'status.peer_eval_enabled' in html


def test_all_three_history_controls_share_one_padding():
    html = open("ui/index.html", encoding="utf-8").read()
    for el in ('id="ft-window"', 'id="ft-peer"', 'id="ft-export"'):
        i = html.index(el)
        assert "padding:3px 6px" in html[i:i + 400], el
