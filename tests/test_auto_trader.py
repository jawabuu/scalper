"""Tests for unattended auto-trading: entry rules, callback sizing, safety limits."""
import pytest

from bot.auto_trader import (
    AutoTradeConfig, SafetyState, StrengthTracker,
    distance_to_extreme, callback_for, evaluate_candidate,
    check_safety, record_entry, record_loss, roll_day,
    MIN_CALLBACK_PCT, MAX_CALLBACK_PCT,
)


@pytest.fixture
def cfg():
    return AutoTradeConfig(enabled=True)


def _short(**kw):
    row = {"symbol": "X/USDT:USDT", "direction": "short", "rsi": 80.0,
           "pct_below_24h_high": -2.0, "pct_above_24h_low": 40.0,
           "strength": "strengthening"}
    row.update(kw)
    return row


def _long(**kw):
    row = {"symbol": "Y/USDT:USDT", "direction": "long", "rsi": 55.0,
           "pct_above_24h_low": 2.0, "pct_below_24h_high": -30.0,
           "strength": "strengthening"}
    row.update(kw)
    return row


# ── Entry rules ──────────────────────────────────────────────────────────────

def test_short_meeting_all_rules_enters(cfg):
    d = evaluate_candidate(_short(), streak=2, cfg=cfg, atr_pct=0.5)
    assert d.enter and d.side == "short"
    assert d.callback_pct == pytest.approx(1.0)     # half of 2%


def test_long_meeting_all_rules_enters(cfg):
    d = evaluate_candidate(_long(), streak=2, cfg=cfg, atr_pct=0.5)
    assert d.enter and d.callback_pct == pytest.approx(1.0)


def test_short_rsi_floor_is_strict(cfg):
    assert not evaluate_candidate(_short(rsi=77.9), streak=2, cfg=cfg).enter
    assert evaluate_candidate(_short(rsi=78.0), streak=2, cfg=cfg, atr_pct=0.5).enter


def test_long_rsi_floor_is_strict(cfg):
    assert not evaluate_candidate(_long(rsi=47.9), streak=2, cfg=cfg).enter
    assert evaluate_candidate(_long(rsi=48.0), streak=2, cfg=cfg, atr_pct=0.5).enter


def test_requires_consecutive_strengthening(cfg):
    assert not evaluate_candidate(_short(), streak=1, cfg=cfg).enter
    assert evaluate_candidate(_short(), streak=2, cfg=cfg, atr_pct=0.5).enter


def test_must_be_near_the_relevant_extreme(cfg):
    assert not evaluate_candidate(_short(pct_below_24h_high=-8.0), streak=2, cfg=cfg).enter
    assert not evaluate_candidate(_long(pct_above_24h_low=8.0), streak=2, cfg=cfg).enter


def test_uses_the_extreme_that_matches_direction(cfg):
    """A short measures against the 24h HIGH, a long against the LOW."""
    assert distance_to_extreme(_short(), "short") == pytest.approx(2.0)
    assert distance_to_extreme(_short(), "long") == pytest.approx(40.0)


def test_missing_range_refuses_rather_than_guessing(cfg):
    row = _short(pct_below_24h_high=None)
    d = evaluate_candidate(row, streak=2, cfg=cfg)
    assert not d.enter and "range unavailable" in d.reason


# ── Callback sizing ──────────────────────────────────────────────────────────

def test_callback_is_half_the_distance(cfg):
    cb, notes, src = callback_for(3.0, atr_pct=0.2, cfg=cfg)
    assert cb == pytest.approx(1.5) and not notes


def test_callback_floored_at_atr(cfg):
    """A callback inside one candle's range would be hit by noise alone."""
    cb, notes, src = callback_for(0.4, atr_pct=0.7, cfg=cfg)
    assert cb == pytest.approx(0.52, abs=0.01)
    assert any("ATR" in n for n in notes)


def test_callback_respects_exchange_minimum(cfg):
    cb, _, src = callback_for(0.05, atr_pct=None, cfg=cfg)
    assert cb == pytest.approx(MIN_CALLBACK_PCT)


def test_callback_respects_exchange_maximum(cfg):
    cb, notes, src = callback_for(20.0, atr_pct=None, cfg=cfg)
    assert cb == pytest.approx(MAX_CALLBACK_PCT)
    assert any("maximum" in n for n in notes)


# ── Streak tracking ──────────────────────────────────────────────────────────

def test_streak_builds_over_consecutive_scans():
    t = StrengthTracker()
    for expected in (1, 2, 3):
        t.update([_short(strength="strengthening")])
        assert t.streak("X/USDT:USDT", "short") == expected


def test_streak_resets_when_weakening():
    t = StrengthTracker()
    t.update([_short(strength="strengthening")])
    t.update([_short(strength="weakening")])
    assert t.streak("X/USDT:USDT", "short") == 0


def test_streak_cleared_when_candidate_drops_out():
    t = StrengthTracker()
    t.update([_short(strength="strengthening")])
    t.update([])                      # no longer a candidate
    assert t.streak("X/USDT:USDT", "short") == 0


def test_confirmed_counts_towards_the_streak():
    t = StrengthTracker()
    t.update([_short(strength="CONFIRMED")])
    t.update([_short(strength="CONFIRMED")])
    assert t.streak("X/USDT:USDT", "short") == 2


# ── Safety limits ────────────────────────────────────────────────────────────

def test_daily_loss_limit_halts_trading(cfg):
    st = roll_day(SafetyState(), 100.0)
    ok, why = check_safety(st, cfg, balance=94.0, open_positions=0, symbol="X")
    assert not ok and "daily loss limit" in why
    # stays halted even if the balance recovers within the same day
    ok, _ = check_safety(st, cfg, balance=100.0, open_positions=0, symbol="X")
    assert not ok


def test_position_cap_blocks_entries(cfg):
    st = roll_day(SafetyState(), 100.0)
    ok, why = check_safety(st, cfg, balance=100.0,
                           open_positions=cfg.max_open_positions, symbol="X")
    assert not ok and "position limit" in why


def test_symbol_cooldown_after_a_loss(cfg):
    st = roll_day(SafetyState(), 100.0)
    st = record_loss(st, "X", cfg)
    ok, why = check_safety(st, cfg, balance=100.0, open_positions=0, symbol="X")
    assert not ok and "cooldown" in why
    # a different symbol is unaffected
    ok, _ = check_safety(st, cfg, balance=100.0, open_positions=0, symbol="Y")
    assert ok


def test_trade_rate_cap(cfg):
    import time
    st = roll_day(SafetyState(), 100.0)
    now = time.time()
    for _ in range(cfg.max_trades_per_hour):
        record_entry(st, now)
    ok, why = check_safety(st, cfg, balance=100.0, open_positions=0,
                           symbol="X", now=now)
    assert not ok and "trades in the last hour" in why


def test_old_entries_fall_out_of_the_rate_window(cfg):
    import time
    st = roll_day(SafetyState(), 100.0)
    now = time.time()
    for _ in range(cfg.max_trades_per_hour):
        record_entry(st, now - 4000)        # over an hour ago
    ok, _ = check_safety(st, cfg, balance=100.0, open_positions=0,
                         symbol="X", now=now)
    assert ok


def test_new_utc_day_resets_the_halt(cfg):
    st = roll_day(SafetyState(), 100.0)
    check_safety(st, cfg, balance=90.0, open_positions=0, symbol="X")
    assert st.halted_reason
    st.day_key = "1970-01-01"             # simulate the day rolling over
    st = roll_day(st, 90.0)
    assert st.halted_reason is None
    assert st.day_start_balance == 90.0   # rebaselined


def test_disabled_by_default():
    assert AutoTradeConfig().enabled is False


# ── Live rule editing ────────────────────────────────────────────────────────

def _trader():
    from bot.auto_trader import AutoTrader
    a = AutoTrader.__new__(AutoTrader)
    a.cfg = AutoTradeConfig()
    a._log = []
    return a


def test_entry_rules_are_live_editable():
    a = _trader()
    applied, errors = a.update_rules({
        "short_rsi_min": 75, "long_rsi_min": 52,
        "required_strength_sweeps": 3, "max_dist_to_extreme_pct": 2.0,
        "callback_ratio": 0.4,
    })
    assert not errors
    assert a.cfg.short_rsi_min == 75
    assert a.cfg.required_strength_sweeps == 3
    assert a.cfg.callback_ratio == pytest.approx(0.4)


def test_safety_limits_cannot_be_changed_from_the_dashboard():
    """
    The daily stop, cooldown and rate cap bound a bad run. The moment you most
    want to relax them from a dashboard is right after a halt fires — so they
    stay env-only.
    """
    a = _trader()
    # max_trades_per_hour deliberately moved to the tunable tier: it is a rate
    # limit, and the loss limits below are what actually bound the damage.
    for key in ("daily_loss_limit_pct", "symbol_cooldown_s",
                "max_open_positions", "max_reentries_per_symbol"):
        applied, errors = a.update_rules({key: 999})
        assert applied == {}
        assert any("safety limit" in e for e in errors)
    assert a.cfg.daily_loss_limit_pct == 5.0


def test_out_of_range_values_are_rejected():
    a = _trader()
    for bad in ({"short_rsi_min": 150}, {"required_strength_sweeps": 0},
                {"callback_ratio": 5.0}, {"max_dist_to_extreme_pct": 0.0}):
        applied, errors = a.update_rules(bad)
        assert applied == {} and errors


def test_a_bad_value_leaves_all_rules_unchanged():
    """Nothing is applied if any value is invalid — no half-changed ruleset."""
    a = _trader()
    before = (a.cfg.short_rsi_min, a.cfg.long_rsi_min)
    applied, errors = a.update_rules({"short_rsi_min": 75, "long_rsi_min": 999})
    assert applied == {} and errors
    assert (a.cfg.short_rsi_min, a.cfg.long_rsi_min) == before


def test_edited_rules_take_effect_on_the_next_decision():
    a = _trader()
    row = {"symbol": "X/USDT:USDT", "direction": "short", "rsi": 76.0,
           "pct_below_24h_high": -2.0, "strength": "strengthening"}
    assert not evaluate_candidate(row, streak=2, cfg=a.cfg).enter    # 76 < 78
    a.update_rules({"short_rsi_min": 75})
    assert evaluate_candidate(row, streak=2, cfg=a.cfg, atr_pct=0.5).enter


# ── Cooldown override on a genuinely stronger signal ─────────────────────────

from bot.auto_trader import record_reentry


def test_stronger_signal_overrides_the_cooldown(cfg):
    """
    A coin that stopped you out and has pushed FURTHER into the extreme is a
    stronger fade than the one that failed — a bare timer discards that.
    """
    st = roll_day(SafetyState(), 1000.0)
    st = record_loss(st, "X", cfg, entry_rsi=78.0)
    ok, why = check_safety(st, cfg, balance=1000.0, open_positions=0,
                           symbol="X", current_rsi=85.0, side="short")
    assert ok and "cooldown overridden" in why


def test_marginal_improvement_does_not_override(cfg):
    st = roll_day(SafetyState(), 1000.0)
    st = record_loss(st, "X", cfg, entry_rsi=78.0)
    ok, _ = check_safety(st, cfg, balance=1000.0, open_positions=0,
                         symbol="X", current_rsi=79.0, side="short")
    assert not ok          # under the 3-point delta


def test_weaker_signal_stays_blocked(cfg):
    st = roll_day(SafetyState(), 1000.0)
    st = record_loss(st, "X", cfg, entry_rsi=78.0)
    ok, _ = check_safety(st, cfg, balance=1000.0, open_positions=0,
                         symbol="X", current_rsi=72.0, side="short")
    assert not ok


def test_reentries_are_capped_however_strong_the_signal(cfg):
    """Re-entering a coin that keeps running against you is how a fade dies."""
    st = roll_day(SafetyState(), 1000.0)
    st = record_loss(st, "X", cfg, entry_rsi=78.0)
    for _ in range(cfg.max_reentries_per_symbol):
        record_reentry(st, "X")
    ok, why = check_safety(st, cfg, balance=1000.0, open_positions=0,
                           symbol="X", current_rsi=95.0, side="short")
    assert not ok and "already retried" in why


def test_reentry_counts_reset_next_day(cfg):
    st = roll_day(SafetyState(), 1000.0)
    record_reentry(st, "X")
    st.day_key = "1970-01-01"
    st = roll_day(st, 1000.0)
    assert st.reentries_today == {}


def test_strict_timer_when_override_disabled():
    cfg = AutoTradeConfig(cooldown_override_rsi_delta=0.0)
    st = roll_day(SafetyState(), 1000.0)
    st = record_loss(st, "X", cfg, entry_rsi=78.0)
    ok, _ = check_safety(st, cfg, balance=1000.0, open_positions=0,
                         symbol="X", current_rsi=95.0, side="short")
    assert not ok


def test_reentry_cap_is_not_dashboard_tunable():
    from bot.auto_trader import AutoTrader
    a = AutoTrader.__new__(AutoTrader); a.cfg = AutoTradeConfig(); a._log = []
    applied, errors = a.update_rules({"max_reentries_per_symbol": 99})
    assert applied == {} and any("safety limit" in e for e in errors)


# ── Rate cap moved to the tunable tier ───────────────────────────────────────

def test_trades_per_hour_is_live_editable():
    """
    A rate limit, not a loss limit — the daily stop and position cap already
    bound the damage, so this one is tunable from the dashboard.
    """
    a = _trader()
    applied, errors = a.update_rules({"max_trades_per_hour": 20})
    assert not errors and a.cfg.max_trades_per_hour == 20


def test_trades_per_hour_can_be_disabled_live():
    a = _trader()
    a.update_rules({"max_trades_per_hour": 0})
    assert a.cfg.max_trades_per_hour == 0
    st = roll_day(SafetyState(), 1000.0)
    import time
    now = time.time()
    for _ in range(50):
        record_entry(st, now)
    ok, _ = check_safety(st, a.cfg, balance=1000.0, open_positions=0,
                         symbol="X", now=now)
    assert ok


def test_trades_per_hour_is_bounded():
    a = _trader()
    applied, errors = a.update_rules({"max_trades_per_hour": 500})
    assert applied == {} and errors


def test_loss_limits_remain_env_only():
    """Moving the rate cap must not have loosened the real safety limits."""
    a = _trader()
    for key in ("daily_loss_limit_pct", "symbol_cooldown_s",
                "max_open_positions", "max_reentries_per_symbol"):
        applied, errors = a.update_rules({key: 999})
        assert applied == {} and any("safety limit" in e for e in errors)


# ── The halt must not lag behind the limit ───────────────────────────────────

def _at(balance_start):
    from bot.auto_trader import AutoTrader
    a = AutoTrader.__new__(AutoTrader)
    a.cfg = AutoTradeConfig(enabled=True, daily_loss_limit_pct=5.0)
    a.state = SafetyState()
    a._log = []
    roll_day(a.state, balance_start)
    return a


def test_halt_fires_without_any_candidates():
    """
    The drawdown was only evaluated inside the per-candidate loop, so a scan
    with no candidates performed no check and the halt lagged — one run reached
    -9.4% against a 5% limit.
    """
    a = _at(4494.25)
    a._check_daily_drawdown(4260.0)          # -5.2%
    assert a.state.halted_reason
    assert "5.2%" in a.state.halted_reason


def test_halt_not_triggered_inside_the_limit():
    a = _at(4494.25)
    a._check_daily_drawdown(4300.0)          # -4.3%
    assert a.state.halted_reason is None


def test_halt_is_not_re_reported_once_set():
    a = _at(4494.25)
    a._check_daily_drawdown(4260.0)
    first = a.state.halted_reason
    a._check_daily_drawdown(4000.0)
    assert a.state.halted_reason == first
    assert len([r for r in a._log if r["action"] == "halted"]) == 1


def test_halt_blocks_entries_once_set():
    a = _at(4494.25)
    a._check_daily_drawdown(4260.0)
    ok, why = check_safety(a.state, a.cfg, balance=4260.0,
                           open_positions=0, symbol="X")
    assert not ok and "daily loss limit" in why


def test_no_baseline_means_no_false_halt():
    from bot.auto_trader import AutoTrader
    a = AutoTrader.__new__(AutoTrader)
    a.cfg = AutoTradeConfig(daily_loss_limit_pct=5.0)
    a.state = SafetyState()          # day_start_balance still 0
    a._log = []
    a._check_daily_drawdown(100.0)
    assert a.state.halted_reason is None


# ── Which rule set the callback ──────────────────────────────────────────────

def test_callback_source_reports_the_ratio(cfg):
    _, _, src = callback_for(3.0, atr_pct=0.2, cfg=cfg)
    assert src == "ratio"


def test_callback_source_reports_the_atr_floor(cfg):
    """
    At a low ratio the ATR floor governs nearly every entry, so the setting
    being tuned no longer drives the result. Recording which rule applied makes
    that visible in the outcomes instead of having to reason about it.
    """
    _, _, src = callback_for(0.4, atr_pct=0.7, cfg=cfg)
    assert src == "atr_floor"


def test_callback_source_reports_exchange_bounds(cfg):
    assert callback_for(0.05, atr_pct=None, cfg=cfg)[2] == "exchange_min"
    assert callback_for(20.0, atr_pct=None, cfg=cfg)[2] == "exchange_max"


def test_decision_carries_the_callback_source(cfg):
    d = evaluate_candidate(_short(), streak=2, cfg=cfg, atr_pct=0.5)
    assert d.enter and d.callback_source in ("ratio", "atr_floor",
                                             "exchange_min", "exchange_max")


def test_low_ratio_shifts_control_to_the_floor():
    """Demonstrates the shift the operator observed when moving 0.5 -> 0.1."""
    loose = AutoTradeConfig(callback_ratio=0.5, callback_atr_mult=0.75)
    tight = AutoTradeConfig(callback_ratio=0.1, callback_atr_mult=0.75)
    assert callback_for(2.0, atr_pct=0.5, cfg=loose)[2] == "ratio"
    assert callback_for(2.0, atr_pct=0.5, cfg=tight)[2] == "atr_floor"


# ── Longs need a ceiling, shorts deliberately do not ─────────────────────────

def _band_cfg(lo=45.0, hi=50.0):
    return AutoTradeConfig(long_rsi_min=lo, long_rsi_max=hi, short_rsi_min=75.0,
                           max_dist_to_extreme_pct=3.0, required_strength_sweeps=2)


def _long_row(rsi):
    return {"symbol": "X/USDT:USDT", "direction": "long", "rsi": rsi,
            "atr_pct": 0.5, "pct_above_24h_low": 1.0,
            "pct_below_24h_high": -1.0, "range_pos_24h": 0.1}


def test_long_above_the_ceiling_is_skipped():
    """
    Split by entry RSI within one session's longs — same regime, differing only
    by RSI — the 50-60 band lost 13 USDT per trade over 11 trades while 45-50
    made money. The auto-trader had a floor but no ceiling.
    """
    cfg = _band_cfg()
    d = evaluate_candidate(_long_row(58), streak=2, cfg=cfg, atr_pct=0.5)
    assert not d.enter and "above the long ceiling" in d.reason


def test_long_inside_the_band_is_taken():
    cfg = _band_cfg()
    for rsi in (45, 48, 50):
        assert evaluate_candidate(_long_row(rsi), streak=2, cfg=cfg,
                                  atr_pct=0.5).enter


def test_long_below_the_floor_is_still_skipped():
    cfg = _band_cfg()
    d = evaluate_candidate(_long_row(42), streak=2, cfg=cfg, atr_pct=0.5)
    assert not d.enter and "below the long floor" in d.reason


def test_ceiling_of_zero_disables_it():
    cfg = _band_cfg(hi=0.0)
    assert evaluate_candidate(_long_row(64), streak=2, cfg=cfg, atr_pct=0.5).enter


def test_shorts_have_no_ceiling():
    """The thesis is that more overbought is a better fade — do not cap it."""
    cfg = _band_cfg()
    row = {"symbol": "X/USDT:USDT", "direction": "short", "rsi": 92,
           "atr_pct": 0.5, "pct_above_24h_low": 20.0,
           "pct_below_24h_high": -1.0, "range_pos_24h": 0.95}
    assert evaluate_candidate(row, streak=2, cfg=cfg, atr_pct=0.5).enter


def test_ceiling_is_live_editable():
    a = _trader()
    applied, errors = a.update_rules({"long_rsi_max": 52})
    assert not errors and a.cfg.long_rsi_max == 52


# ── Direction lever ──────────────────────────────────────────────────────────

def _dir_cfg(mode):
    return AutoTradeConfig(directions=mode, long_rsi_min=45, long_rsi_max=52,
                           short_rsi_min=75, max_dist_to_extreme_pct=3.0,
                           required_strength_sweeps=2)


def _dir_row(side, rsi):
    return {"symbol": "X/USDT:USDT", "direction": side, "rsi": rsi,
            "atr_pct": 0.5, "pct_above_24h_low": 1.0,
            "pct_below_24h_high": -1.0,
            "range_pos_24h": 0.1 if side == "long" else 0.9}


def test_all_permits_both_sides():
    cfg = _dir_cfg("all")
    assert evaluate_candidate(_dir_row("long", 48), 2, cfg, atr_pct=0.5).enter
    assert evaluate_candidate(_dir_row("short", 80), 2, cfg, atr_pct=0.5).enter


def test_long_only_blocks_shorts():
    cfg = _dir_cfg("long")
    assert evaluate_candidate(_dir_row("long", 48), 2, cfg, atr_pct=0.5).enter
    d = evaluate_candidate(_dir_row("short", 80), 2, cfg, atr_pct=0.5)
    assert not d.enter and "disabled" in d.reason


def test_short_only_blocks_longs():
    cfg = _dir_cfg("short")
    assert evaluate_candidate(_dir_row("short", 80), 2, cfg, atr_pct=0.5).enter
    d = evaluate_candidate(_dir_row("long", 48), 2, cfg, atr_pct=0.5)
    assert not d.enter and "disabled" in d.reason


def test_direction_is_live_editable():
    a = _trader()
    applied, errors = a.update_rules({"directions": "short"})
    assert not errors and a.cfg.directions == "short"


def test_invalid_direction_is_rejected():
    a = _trader()
    applied, errors = a.update_rules({"directions": "sideways"})
    assert applied == {} and "must be one of" in errors[0]


def test_direction_value_is_normalised():
    a = _trader()
    a.update_rules({"directions": "  SHORT "})
    assert a.cfg.directions == "short"


# ── Live edits last for the instance, not beyond it ──────────────────────────

def test_live_edit_holds_while_running():
    a = _trader()
    a.update_rules({"directions": "short"})
    assert a.cfg.directions == "short"
    # and keeps holding across further evaluations
    a.update_rules({"long_rsi_max": 52})
    assert a.cfg.directions == "short"


def test_a_fresh_instance_takes_the_declared_config():
    """
    The compose file is the declared configuration and must win on restart. A
    dashboard change is session tuning, not a persisted override.
    """
    a = _trader()
    a.update_rules({"directions": "short"})
    b = _trader()                       # as after a redeploy
    assert b.cfg.directions == "all"


def test_rules_are_not_written_to_the_state_file():
    import inspect
    from bot import futures_state
    assert "auto_rules" not in inspect.getsource(futures_state.save)
