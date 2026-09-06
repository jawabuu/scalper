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
    cb, notes = callback_for(3.0, atr_pct=0.2, cfg=cfg)
    assert cb == pytest.approx(1.5) and not notes


def test_callback_floored_at_atr(cfg):
    """A callback inside one candle's range would be hit by noise alone."""
    cb, notes = callback_for(0.4, atr_pct=0.7, cfg=cfg)
    assert cb == pytest.approx(0.52, abs=0.01)
    assert any("ATR" in n for n in notes)


def test_callback_respects_exchange_minimum(cfg):
    cb, _ = callback_for(0.05, atr_pct=None, cfg=cfg)
    assert cb == pytest.approx(MIN_CALLBACK_PCT)


def test_callback_respects_exchange_maximum(cfg):
    cb, notes = callback_for(20.0, atr_pct=None, cfg=cfg)
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
    for key in ("daily_loss_limit_pct", "symbol_cooldown_s",
                "max_trades_per_hour", "max_open_positions"):
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
