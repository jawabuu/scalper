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
