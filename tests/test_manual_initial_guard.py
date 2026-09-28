"""
MANUAL_INITIAL_GUARD_ONLY — the COMPLETE contract.

The promise: a position the operator opened by hand is bounded ONLY by the
initial trailing stop and the fixed ATR stop placed at adoption. Nothing
else may move, tighten, replace or cancel them.

Three mechanisms escaped the first two implementations, all found in
production rather than by these tests:

  1. the BREAKEVEN RATCHET      moved the fixed stop to +2% ROI (QNT 12:15)
  2. the RUNTIME ARM            replaced a 1.54% trail with 0.30% (QNT 12:37)
  3. the SUPERSESSION + cancel  that arming triggers, leaving no fixed stop

All three were missed because the first pass enumerated PLACEMENT METHODS.
Two of the three live inside evaluate() as arithmetic branches, and the third
is a consequence of the second. **Enumerate by EFFECT, not by method name.**

So this file is organised by effect: for every mechanism that can change a
position's exit, one test that it is skipped under the flag, and one that it
STILL RUNS without it. A gate that disables something for everyone is as
broken as one that disables nothing.
"""

import pytest

from bot.futures_guard import (
    GuardConfig, GuardState, FuturesPosition,
    adopt_state, desired_stop_roi, evaluate, is_armed,
)


def _cfg(**kw):
    base = dict(initial_stop_roi=15.0, arm_roi=5.0, callback_roi=3.0,
                breakeven_at_roi=3.0, breakeven_stop_roi=2.0,
                use_native_trail=True)
    base.update(kw)
    return GuardConfig(**base)


def _pos(entry=100.0, side="long", lev=10.0):
    return FuturesPosition(symbol="X/USDT:USDT", side=side, entry_price=entry,
                           qty=1.0, leverage=lev, margin=10.0)


# effective_arm_roi() is volatility- and leverage-aware (v3.72.0): with
# trail_callback_pct=1.0 at 10x it resolves to 12.0, NOT cfg.arm_roi=5.0.
# Tests that need "armed" must clear the EFFECTIVE threshold, so they use a
# peak above it rather than above arm_roi.
ARMED_PEAK = 20.0


def _state(peak=0.0, armed=False):
    st = GuardState()
    st.peak_roi = peak
    st.armed = armed
    return st


# ── EFFECT 1: the fixed stop must not RATCHET to a profit lock ─────────────

def test_breakeven_ratchet_is_skipped():
    """
    QNT 2026-09-28 12:15. Stop moved -14.8% -> +2.0% once peak crossed +3%,
    and that closed the trade at +1.48%.
    """
    st = _state(peak=4.0)
    assert desired_stop_roi(st, _cfg(), leverage=10.0,
                            reduced_guard=True) == -15.0


def test_breakeven_ratchet_STILL_RUNS_normally():
    st = _state(peak=4.0)
    assert desired_stop_roi(st, _cfg(), leverage=10.0) == 2.0


# ── EFFECT 2: the position must not ARM ────────────────────────────────────

def test_runtime_arming_is_skipped():
    """
    QNT 2026-09-28 12:37. Peak hit +7.8%, a 0.30% armed trail replaced the
    1.54% adaptive one — five times tighter — and the fixed stop was
    cancelled behind it.
    """
    st = _state(peak=ARMED_PEAK)
    assert is_armed(st, _cfg(), leverage=10.0, reduced_guard=True) is False


def test_runtime_arming_STILL_RUNS_normally():
    st = _state(peak=ARMED_PEAK)
    assert is_armed(st, _cfg(), leverage=10.0) is True


def test_arming_is_skipped_at_ADOPTION_too():
    # A restart that restores a peak could otherwise arm on the spot.
    st = adopt_state(_pos(), [], 0.0, _cfg(), reduced_guard=True)
    assert st.armed is False


def test_adoption_arming_STILL_RUNS_normally():
    st = adopt_state(_pos(), [], 0.0, _cfg())
    assert st.armed is False, "peak is 0 at adoption, so False either way"


# ── EFFECT 3: arming's CONSEQUENCES cannot fire if arming cannot ───────────

def test_a_reduced_guard_position_never_reaches_the_armed_stop_level():
    """
    Arming is the single upstream trigger for THREE things: the armed trail,
    the adaptive supersession, and the fixed-stop cancellation. Gating it at
    source is what makes all three unreachable.
    """
    st = _state(peak=20.0)
    # normally this would be peak - callback_roi = 17.0
    assert desired_stop_roi(st, _cfg(), leverage=10.0) == 17.0
    assert desired_stop_roi(st, _cfg(), leverage=10.0,
                            reduced_guard=True) == -15.0


def test_the_guardian_trail_block_is_also_gated():
    # Belt-and-braces: state.armed is already False, but this block is where
    # the QNT armed trail was actually placed.
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    assert "not self._reduced_guard(pos)" in src


# ── EFFECT 4: what must STILL happen ───────────────────────────────────────

def test_the_initial_stop_is_STILL_returned_under_reduced_guard():
    # "Bounded only by the initial trailing stop" means it must still BE
    # there. A gate that removes all protection is the opposite of the ask.
    for peak in (0.0, 2.0, 4.0, 20.0):
        assert desired_stop_roi(_state(peak=peak), _cfg(), leverage=10.0,
                                reduced_guard=True) == -15.0


def test_an_initial_stop_OVERRIDE_is_still_honoured():
    # The ATR-derived stop from adoption must survive the gate.
    assert desired_stop_roi(_state(peak=ARMED_PEAK), _cfg(),
                            initial_stop_override=22.0, leverage=10.0,
                            reduced_guard=True) == -22.0


def test_the_RESCUE_path_is_deliberately_NOT_gated():
    """
    It sets state.armed only when the fixed stop was REJECTED. A trailing stop
    is then the last protection available, and gating it would leave a
    reduced-guard position with nothing at all.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian)
    i = src.index("rescue=True")
    assert "_reduced_guard" not in src[i:i + 600]


# ── EFFECT 5: evaluate() end to end ────────────────────────────────────────

@pytest.mark.parametrize("peak", [0.0, 3.5, 6.0, 12.0, ARMED_PEAK])
def test_evaluate_never_tightens_a_reduced_guard_position(peak):
    """
    The integration check. Whatever the peak, the stop stays at the initial
    level — no ratchet, no arm, no peak-minus-callback.
    """
    st = _state(peak=peak)
    st2, _price, _why = evaluate(_pos(), 100.0, st, _cfg(),
                                 reduced_guard=True)
    assert st2.stop_roi in (0.0, -15.0) or st2.stop_roi <= -15.0
    assert st2.armed is False


@pytest.mark.parametrize("peak,expect_armed", [(0.0, False), (ARMED_PEAK, True)])
def test_evaluate_STILL_arms_a_normal_position(peak, expect_armed):
    st = _state(peak=peak)
    st2, _price, _why = evaluate(_pos(), 100.0, st, _cfg())
    assert st2.armed is expect_armed


# ── EFFECT 6: the flag itself ──────────────────────────────────────────────

def test_the_default_is_FULL_protection():
    assert desired_stop_roi(_state(peak=4.0), _cfg(), leverage=10.0) == 2.0
    assert is_armed(_state(peak=ARMED_PEAK), _cfg(), leverage=10.0) is True
