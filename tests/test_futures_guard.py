"""Tests for the futures position guardian (Phase 1)."""
import pytest

from bot.futures_guard import (
    is_armed,
    FuturesPosition, GuardState, GuardConfig,
    roi_pct, price_for_roi, stop_side,
    desired_stop_roi, update_peak, should_replace_stop, evaluate,
)


@pytest.fixture
def cfg():
    # Option A: stop -7% ROI, arm +15% ROI, callback 10% ROI
    return GuardConfig(initial_stop_roi=7.0, arm_roi=15.0,
                       callback_roi=10.0, min_stop_move_roi=1.0).validate()


@pytest.fixture
def long_pos():
    # 100 margin at 10x = 1000 notional, entry 100.0
    return FuturesPosition("BTC/USDT", "long", 100.0, 10.0, 10, 100.0)


@pytest.fixture
def short_pos():
    return FuturesPosition("BTC/USDT", "short", 100.0, 10.0, 10, 100.0)


# ── ROI maths ────────────────────────────────────────────────────────────────

def test_roi_long(long_pos):
    # +1% price move at 10x = +10% ROI
    assert roi_pct(long_pos, 101.0) == pytest.approx(10.0)
    # -0.7% price = -7% ROI (the initial stop level)
    assert roi_pct(long_pos, 99.3) == pytest.approx(-7.0)


def test_roi_short(short_pos):
    # Short profits when price FALLS: -1% price = +10% ROI
    assert roi_pct(short_pos, 99.0) == pytest.approx(10.0)
    # Price rising hurts a short
    assert roi_pct(short_pos, 100.7) == pytest.approx(-7.0)


def test_price_for_roi_roundtrip(long_pos, short_pos):
    for pos in (long_pos, short_pos):
        for target in (-7.0, 0.0, 5.0, 15.0, 40.0):
            p = price_for_roi(pos, target)
            assert roi_pct(pos, p) == pytest.approx(target)


def test_stop_side(long_pos, short_pos):
    assert stop_side(long_pos) == "sell"
    assert stop_side(short_pos) == "buy"


# ── The invariant that prevents the manual-setup flaw ────────────────────────

def test_config_rejects_callback_wider_than_arm():
    """
    The manual setup had activation +15% ROI with a 2% PRICE callback (=20% ROI),
    so arming put the stop at -5% ROI — a winning trade exiting at a loss. The
    guardian must refuse such a config.
    """
    bad = GuardConfig(initial_stop_roi=7.0, arm_roi=15.0, callback_roi=20.0)
    with pytest.raises(AssertionError, match="must be LESS than arm_roi"):
        bad.validate()


def test_arming_locks_in_profit(cfg, long_pos):
    """With callback < arm, the first armed stop is strictly positive ROI."""
    state = GuardState()
    price = price_for_roi(long_pos, 15.0)      # exactly at the arm threshold
    state, stop_price, reason = evaluate(long_pos, price, state, cfg)
    assert state.armed
    assert state.stop_roi == pytest.approx(5.0)      # 15 - 10
    assert roi_pct(long_pos, stop_price) == pytest.approx(5.0)
    assert stop_price > long_pos.entry_price          # above entry for a long


# ── Stop level progression ───────────────────────────────────────────────────

def test_initial_stop_before_arming(cfg, long_pos):
    state = GuardState()
    price = price_for_roi(long_pos, 3.0)   # in profit but below the arm level
    state, stop_price, reason = evaluate(long_pos, price, state, cfg)
    assert not state.armed
    assert state.stop_roi == pytest.approx(-7.0)
    assert "initial protective stop" in reason


def test_trailing_ratchets_up_only(cfg, long_pos):
    state = GuardState()
    # Run to +40% ROI
    state, _, _ = evaluate(long_pos, price_for_roi(long_pos, 40.0), state, cfg)
    assert state.stop_roi == pytest.approx(30.0)
    # Pull back to +20% ROI — peak stays 40, stop must NOT loosen
    state, stop_price, reason = evaluate(long_pos, price_for_roi(long_pos, 20.0), state, cfg)
    assert state.peak_roi == pytest.approx(40.0)
    assert state.stop_roi == pytest.approx(30.0)
    assert stop_price is None
    assert reason == "stop unchanged"


def test_trailing_works_for_short(cfg, short_pos):
    """Short side: profit comes from falling price; stop must sit ABOVE entry."""
    state = GuardState()
    price = price_for_roi(short_pos, 30.0)   # short is up 30% ROI
    assert price < short_pos.entry_price      # price fell
    state, stop_price, reason = evaluate(short_pos, price, state, cfg)
    assert state.armed
    assert state.stop_roi == pytest.approx(20.0)
    # For a short, a profitable stop sits BELOW entry price
    assert stop_price < short_pos.entry_price
    assert roi_pct(short_pos, stop_price) == pytest.approx(20.0)


def test_short_initial_stop_above_entry(cfg, short_pos):
    state = GuardState()
    state, stop_price, _ = evaluate(short_pos, short_pos.entry_price, state, cfg)
    # -7% ROI on a short = price moved UP against it
    assert stop_price > short_pos.entry_price
    assert roi_pct(short_pos, stop_price) == pytest.approx(-7.0)


def test_min_move_prevents_stop_spam(cfg, long_pos):
    state = GuardState()
    state, _, _ = evaluate(long_pos, price_for_roi(long_pos, 30.0), state, cfg)
    assert state.stop_roi == pytest.approx(20.0)
    # Tiny improvement (< min_stop_move_roi) must not trigger a replace
    state, stop_price, reason = evaluate(long_pos, price_for_roi(long_pos, 30.5), state, cfg)
    assert stop_price is None
    assert reason == "stop unchanged"
    # A larger improvement does
    state, stop_price, _ = evaluate(long_pos, price_for_roi(long_pos, 32.0), state, cfg)
    assert stop_price is not None
    assert state.stop_roi == pytest.approx(22.0)


def test_stop_never_loosens_on_reversal(cfg, short_pos):
    state = GuardState()
    state, _, _ = evaluate(short_pos, price_for_roi(short_pos, 50.0), state, cfg)
    locked = state.stop_roi
    assert locked == pytest.approx(40.0)
    # Market reverses hard against the short
    state, stop_price, _ = evaluate(short_pos, price_for_roi(short_pos, -5.0), state, cfg)
    assert state.stop_roi == pytest.approx(locked)   # unchanged
    assert stop_price is None


# ── Scaling with margin size ─────────────────────────────────────────────────

def test_roi_thresholds_scale_with_margin(cfg):
    """
    -7% ROI is 7 USDT on 100 margin and 0.70 USDT on 10 margin — the ROI figure
    is margin-relative, so the same config scales automatically.
    """
    lev = 10
    for margin in (10.0, 100.0, 1000.0):
        # Keep the position internally consistent: notional = margin x leverage
        entry, qty = 100.0, (margin * lev) / 100.0
        pos = FuturesPosition("X/USDT", "long", entry, qty, lev, margin)
        assert pos.effective_leverage == pytest.approx(lev)
        stop_price = price_for_roi(pos, -7.0)
        pnl = (stop_price - entry) / entry * pos.notional
        assert pnl == pytest.approx(-0.07 * margin)


# ── Safety: entry orders must never be touched ───────────────────────────────

def test_trailing_stop_ENTRY_order_is_not_protective(short_pos):
    """
    The operator opens shorts with a trailing-stop ENTRY order so the market
    confirms direction before filling. That order is NOT reduce-only and must
    never be classified as protection (or the guardian would cancel it and
    destroy the entry method).
    """
    entry_order = {
        "id": "999", "side": "sell", "type": "TRAILING_STOP_MARKET",
        "reduceOnly": False,          # ← entry, not protection
        "stopPrice": 2.49,
    }
    from bot.futures_guard import is_protective_stop, adoptable_stop
    assert not is_protective_stop(entry_order, short_pos)
    assert adoptable_stop([entry_order], short_pos) is None


def test_reduce_only_stop_is_protective(short_pos):
    from bot.futures_guard import is_protective_stop
    protective = {
        "id": "1", "side": "buy",     # buy closes a short
        "type": "STOP_MARKET", "reduceOnly": True, "stopPrice": 2.5175,
    }
    assert is_protective_stop(protective, short_pos)


def test_wrong_side_order_ignored(short_pos):
    from bot.futures_guard import is_protective_stop
    # A sell reduce-only order does not close a SHORT (buy does)
    wrong = {"id": "2", "side": "sell", "type": "STOP_MARKET",
             "reduceOnly": True, "stopPrice": 2.4}
    assert not is_protective_stop(wrong, short_pos)


def test_non_stop_order_ignored(long_pos):
    from bot.futures_guard import is_protective_stop
    tp = {"id": "3", "side": "sell", "type": "TAKE_PROFIT_LIMIT",
          "reduceOnly": True, "price": 110.0}
    # TAKE_PROFIT_LIMIT has no "STOP" in the type -> not adopted as the stop
    assert not is_protective_stop(tp, long_pos)


# ── Adoption ─────────────────────────────────────────────────────────────────

def test_adopt_existing_stop(cfg, short_pos):
    from bot.futures_guard import adopt_state, roi_pct
    existing = {"id": "77", "side": "buy", "type": "STOP_MARKET",
                "reduceOnly": True, "stopPrice": price_for_roi(short_pos, -7.0)}
    st = adopt_state(short_pos, [existing], current_roi=3.0, cfg=cfg)
    assert st.stop_order_id == "77"
    assert st.stop_roi == pytest.approx(-7.0)
    assert st.peak_roi == pytest.approx(3.0)   # seeded from current
    assert not st.armed


def test_adopt_unprotected_position(cfg, long_pos):
    from bot.futures_guard import adopt_state
    st = adopt_state(long_pos, [], current_roi=0.0, cfg=cfg)
    assert st.stop_roi is None        # no stop -> guardian will place one
    assert st.stop_order_id is None


def test_adopt_already_profitable_seeds_peak_and_arms(cfg, short_pos):
    """A position discovered already past the arm threshold arms immediately."""
    from bot.futures_guard import adopt_state
    st = adopt_state(short_pos, [], current_roi=25.0, cfg=cfg)
    assert st.peak_roi == pytest.approx(25.0)
    assert st.armed
    # Next evaluation should want a stop at 25 - 10 = +15% ROI
    st, stop_price, why = evaluate(short_pos, price_for_roi(short_pos, 25.0), st, cfg)
    assert st.stop_roi == pytest.approx(15.0)


def test_adopt_ignores_entry_order_and_places_own_stop(cfg, short_pos):
    """With only an ENTRY order resting, the position counts as unprotected."""
    from bot.futures_guard import adopt_state
    entry = {"id": "999", "side": "sell", "type": "TRAILING_STOP_MARKET",
             "reduceOnly": False, "stopPrice": 2.49}
    st = adopt_state(short_pos, [entry], current_roi=0.0, cfg=cfg)
    assert st.stop_roi is None
    st, stop_price, why = evaluate(short_pos, short_pos.entry_price, st, cfg)
    assert stop_price is not None
    assert "initial protective stop" in why
