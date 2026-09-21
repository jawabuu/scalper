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


# ── Breakeven step ───────────────────────────────────────────────────────────

def _be_cfg(at=3.0, stop=0.0):
    return GuardConfig(initial_stop_roi=10, arm_roi=5, callback_roi=3,
                       breakeven_at_roi=at, breakeven_stop_roi=stop)


def test_stop_stays_at_initial_below_the_breakeven_trigger():
    cfg = _be_cfg()
    assert desired_stop_roi(GuardState(peak_roi=2.9), cfg) == pytest.approx(-10.0)


def test_stop_moves_to_breakeven_once_triggered():
    """
    A trade that showed a real gain should not run all the way back to its
    initial stop. Two losers peaking above +3% gave back 28.5% of ROI doing so.
    """
    cfg = _be_cfg()
    assert desired_stop_roi(GuardState(peak_roi=3.0), cfg) == pytest.approx(0.0)
    assert desired_stop_roi(GuardState(peak_roi=4.9), cfg) == pytest.approx(0.0)


def test_trail_still_takes_over_above_the_arm_level():
    """The step sits BELOW the trail — winner behaviour above arm is unchanged."""
    cfg = _be_cfg()
    assert desired_stop_roi(GuardState(peak_roi=5.0), cfg) == pytest.approx(2.0)
    assert desired_stop_roi(GuardState(peak_roi=20.0), cfg) == pytest.approx(17.0)


def test_breakeven_level_is_configurable_for_fees():
    """Exiting at exactly 0% ROI still loses the round-trip fee."""
    cfg = _be_cfg(stop=2.0)
    assert desired_stop_roi(GuardState(peak_roi=4.0), cfg) == pytest.approx(2.0)


def test_step_disabled_by_default():
    cfg = GuardConfig(initial_stop_roi=10, arm_roi=5, callback_roi=3)
    assert cfg.breakeven_at_roi == 0.0
    assert desired_stop_roi(GuardState(peak_roi=4.0), cfg) == pytest.approx(-10.0)


def test_breakeven_must_sit_below_the_arm_level():
    with pytest.raises(AssertionError):
        _be_cfg(at=5.0).validate()


def test_breakeven_stop_must_sit_below_its_trigger():
    with pytest.raises(AssertionError):
        _be_cfg(at=3.0, stop=4.0).validate()


def test_step_respects_the_atr_derived_initial_stop():
    """The override still applies below the trigger."""
    cfg = _be_cfg()
    assert desired_stop_roi(GuardState(peak_roi=1.0), cfg,
                            initial_stop_override=7.5) == pytest.approx(-7.5)
    assert desired_stop_roi(GuardState(peak_roi=3.5), cfg,
                            initial_stop_override=7.5) == pytest.approx(0.0)


# ── Volatility floor on the armed trail's callback (v3.72.0) ────────────────
#
# STG 2026-09-20 is why this exists. At 19.8x a 3% ROI target produced a 0.15%
# callback — a QUARTER of the coin's 0.604% ATR and a sixth of its 0.916%
# recent range — while the ENTRY path had independently chosen 0.45% for the
# same coin seconds earlier. Peak +23.2% ROI at 14:55:56; closed at -8.08% at
# 14:55:59, three seconds later, on an ordinary candle.
#
# The only bound before this was Binance's 0.1%-5%, which describes what the
# EXCHANGE accepts, not what the market does.

from bot.futures_guard import (                                    # noqa: E402
    trail_vol_floor_pct, trail_callback_price_pct, callback_roi_at,
    effective_arm_roi, is_armed,
)


def _vcfg(**kw):
    base = dict(arm_roi=5.0, trail_callback_roi=3.0,
                trail_callback_atr_mult=0.75, min_trail_lock_roi=2.0)
    base.update(kw)
    return GuardConfig(**base)


def test_the_stg_callback_is_no_longer_inside_the_coins_own_movement():
    cfg = _vcfg()
    atr, rtr, lev = 0.604, 0.916, 19.7876
    assert trail_callback_price_pct(lev, cfg) == pytest.approx(0.15, abs=0.01)
    floored = trail_callback_price_pct(lev, cfg, atr, rtr)
    assert floored > atr, "the callback must sit OUTSIDE ATR, not inside it"
    assert floored == pytest.approx(0.69, abs=0.01)   # 0.75 x 0.916


def test_the_floor_takes_the_larger_of_atr_and_recent_range():
    # ATR(14) understates an accelerating move, and the scanner selects
    # accelerating moves. Mirrors the entry path's _callback_for.
    cfg = _vcfg()
    assert trail_vol_floor_pct(cfg, 0.4, 1.2) == pytest.approx(0.9)
    assert trail_vol_floor_pct(cfg, 1.2, 0.4) == pytest.approx(0.9)


def test_no_volatility_known_means_no_floor_invented():
    # A floor guessed from nothing is a number with no meaning behind it.
    cfg = _vcfg()
    assert trail_vol_floor_pct(cfg, None, None) == 0.0
    assert trail_callback_price_pct(20, cfg) == trail_callback_price_pct(
        20, cfg, None, None)


def test_the_floor_never_NARROWS_a_callback():
    # A calm coin must keep the ROI-derived callback, not be pulled down to a
    # tiny ATR. The floor is a floor, never a target.
    cfg = _vcfg(trail_callback_roi=20.0)
    wide = trail_callback_price_pct(10, cfg)
    assert trail_callback_price_pct(10, cfg, 0.05, 0.05) == wide


def test_the_floor_can_be_switched_off_for_the_old_behaviour():
    cfg = _vcfg(trail_callback_atr_mult=0.0)
    assert trail_callback_price_pct(19.8, cfg, 0.604, 0.916) == pytest.approx(
        0.15, abs=0.01)


def test_the_floor_still_respects_the_exchange_maximum():
    # Binance caps callbackRate at 5%; a wild coin must not produce a rejected
    # order, which would leave the position with no trail at all.
    cfg = _vcfg()
    assert trail_callback_price_pct(10, cfg, 40.0, None) <= 5.0


def test_a_floored_trail_arms_LATER_instead_of_being_refused():
    """
    The interaction that matters. A noise-width callback costs more ROI at
    high leverage than the +5% arm level, so the pre-v3.72.0 code refused to
    arm at all and kept the fixed stop — stripping the trail from exactly the
    volatile positions that most need one.
    """
    cfg = _vcfg()
    lev, atr = 19.7876, 0.604
    give_back = callback_roi_at(lev, cfg, atr, 0.916)
    arm_at = effective_arm_roi(cfg, lev, atr, 0.916)
    assert give_back > cfg.arm_roi, "the premise: give-back exceeds the arm level"
    assert arm_at == pytest.approx(give_back + cfg.min_trail_lock_roi)
    # STG peaked at +23.2%, so under this rule it WOULD have armed.
    assert is_armed(GuardState(peak_roi=23.2), cfg, lev, atr, 0.916)
    assert not is_armed(GuardState(peak_roi=5.0), cfg, lev, atr, 0.916)


def test_arming_later_still_locks_in_at_least_the_minimum():
    cfg = _vcfg()
    lev, atr, rtr = 19.7876, 0.604, 0.916
    arm_at = effective_arm_roi(cfg, lev, atr, rtr)
    locked = arm_at - callback_roi_at(lev, cfg, atr, rtr)
    assert locked >= cfg.min_trail_lock_roi


def test_a_calm_coin_arms_at_the_normal_level():
    # Deferral must not become the default — it applies only where the floor
    # actually binds.
    cfg = _vcfg()
    assert effective_arm_roi(cfg, 10.0, 0.05, 0.05) == pytest.approx(cfg.arm_roi)


def test_without_leverage_the_arm_level_is_unchanged():
    # Callers that do not know a position's leverage keep the old behaviour.
    cfg = _vcfg()
    assert effective_arm_roi(cfg) == pytest.approx(cfg.arm_roi)


def test_the_give_back_report_matches_the_callback_actually_sent():
    # trail_locks_in and callback_roi_at must take the SAME volatility as the
    # callback they describe, or a floored trail is reported as locking in the
    # unfloored amount — a wrong number that looks checked.
    cfg = _vcfg()
    lev, atr = 19.7876, 0.604
    cb = trail_callback_price_pct(lev, cfg, atr, None)
    assert callback_roi_at(lev, cfg, atr, None) == pytest.approx(cb * lev)


def test_higher_leverage_no_longer_buys_a_tighter_price_trail():
    """
    The coupling this corrects: ROI/leverage alone made the PRICE trail
    tighter as leverage rose (0.60% at 5x, 0.15% at 20x), so the higher the
    leverage the more certainly the trail sat inside noise.
    """
    cfg = _vcfg()
    atr = 0.604
    unfloored = [trail_callback_price_pct(L, cfg) for L in (5, 10, 20)]
    assert unfloored == sorted(unfloored, reverse=True), \
        "the premise: without a floor, more leverage means a tighter trail"
    assert unfloored[-1] < atr, "and at 20x it lands inside the coin's ATR"

    widths = [trail_callback_price_pct(L, cfg, atr, None) for L in (5, 10, 20)]
    # The width is max(roi/leverage, floor): the ROI term still dominates at
    # low leverage, and the floor takes over exactly where the old behaviour
    # went inside noise. What matters is that it stops shrinking.
    assert widths[1] == widths[2], f"floor should pin the high end: {widths}"
    assert all(w >= 0.75 * atr for w in widths)


def test_arm_at_entry_must_use_the_DEFERRED_level_not_the_configured_one():
    """
    v3.73.1. Arm-at-entry derives its activation from the arm level and places
    the trail BEFORE any peak exists. Using cfg.arm_roi there meant a trail
    activating at +5% that then gave back 7.5% — engaging BELOW entry, the
    exact failure deferral exists to prevent — so _place_native_trail refused
    it and GUARD_ARM_AT_ENTRY silently became a no-op on every coin where the
    floor binds. On the live config (10x, mult 1.25, AUTO_MIN_ATR_PCT=0.5)
    that is EVERY coin.
    """
    cfg = _vcfg(trail_callback_atr_mult=1.25)
    lev, atr = 10.0, 0.5
    give_back = callback_roi_at(lev, cfg, atr, None)
    assert give_back > cfg.arm_roi, "the premise, on the live config"
    arm_at = effective_arm_roi(cfg, lev, atr, None)
    assert arm_at - give_back >= cfg.min_trail_lock_roi, \
        "activating at the deferred level must still lock in profit"


def test_with_the_floor_off_the_callback_is_pure_roi_over_leverage():
    """
    The default since v3.73.4, and what BOTH containers now run. The floor is
    dead code unless someone deliberately sets the multiplier.
    """
    cfg = _vcfg(trail_callback_atr_mult=0.0)
    for atr in (0.3, 0.6, 1.5, 3.0):
        assert trail_callback_price_pct(10, cfg, atr, None) == pytest.approx(0.30)
    assert effective_arm_roi(cfg, 10, 1.5, None) == pytest.approx(cfg.arm_roi)
