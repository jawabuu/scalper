"""
AUTO_MAX_DIST_PROPORTIONAL — keep the setup GEOMETRY when the change floor
drops.

AUTO_MAX_DIST_PCT is absolute, so its meaning depends on the range:

    a 15% mover, 3% limit -> the top 20% of the range   (an extreme)
    a  4% mover, 3% limit -> the top 75% of the range   (mid-range)

SCAN_MIN_CHANGE_PCT=8 was an operator judgement in a high-breadth market,
not a measurement. Breadth fell ~85 -> ~57 and movers ~40 -> ~15, and at
ch>=8 only 40 coins in the whole 525-symbol universe now qualify — the
volume floor is NOT binding, the change floor is. Restoring supply means
lowering it, and doing that with an absolute distance limit would silently
convert an extremes gate into a mid-range one.

The gate is currently EXACT: of 339 entered trades, 158 of 159 longs sat
below range_pos 0.2 and all 222 shorts above 0.8. These tests pin the
property that keeps that true.
"""

import pytest

from bot.auto_trader import AutoTradeConfig, effective_max_dist


def _cfg(**kw):
    base = dict(max_dist_to_extreme_pct=3.0, max_dist_proportional=True,
                max_dist_range_ratio=0.2)
    base.update(kw)
    return AutoTradeConfig(**base)


@pytest.mark.parametrize("change,expect", [
    (15.0, 3.0),     # 0.2 x 15 = 3.0, equals the cap -> unchanged
    (20.0, 3.0),     # above the cap -> capped, never widened
    (40.0, 3.0),
    (10.0, 2.0),
    (8.0, 1.6),
    (5.0, 1.0),
    (4.0, 0.8),
    (3.0, 0.6),
])
def test_the_limit_scales_with_the_move(change, expect):
    got, why = effective_max_dist(_cfg(), {"change_24h_pct": change})
    assert abs(got - expect) < 1e-9, why


def test_it_can_only_TIGHTEN_never_widen():
    """
    Proportional mode takes the tighter of the two, so anything that passes
    today still passes. A gate change that ADMITS new candidates while a
    change floor is also moving would make both unreadable.
    """
    for ch in (1, 5, 15, 50, 300):
        got, _ = effective_max_dist(_cfg(), {"change_24h_pct": ch})
        assert got <= 3.0


def test_a_NEGATIVE_change_uses_its_magnitude():
    # Shorts carry a positive change, longs a negative one; the geometry is
    # the same either way.
    a, _ = effective_max_dist(_cfg(), {"change_24h_pct": -10.0})
    b, _ = effective_max_dist(_cfg(), {"change_24h_pct": 10.0})
    assert a == b == 2.0


@pytest.mark.parametrize("bad", [None, "", "n/a", 0, 0.0, [], {}])
def test_an_UNUSABLE_change_FALLS_BACK_to_the_absolute_limit(bad):
    """
    A missing or zero 24h change must not produce a zero limit, which would
    refuse every candidate silently — the failure mode that cost four hours
    when the daily halt went quiet.
    """
    got, why = effective_max_dist(_cfg(), {"change_24h_pct": bad})
    assert got == 3.0
    assert "absolute" in why


def test_OFF_by_default_and_unchanged_when_off():
    assert AutoTradeConfig().max_dist_proportional is False
    from bot.config import BotConfig
    c = BotConfig()
    assert c.auto_max_dist_proportional is False
    assert c.auto_max_dist_range_ratio == 0.2
    got, why = effective_max_dist(
        _cfg(max_dist_proportional=False), {"change_24h_pct": 4.0})
    assert got == 3.0 and why == "absolute"


def test_the_ratio_is_configurable():
    got, _ = effective_max_dist(
        _cfg(max_dist_range_ratio=0.1), {"change_24h_pct": 10.0})
    assert got == 1.0


def test_ratio_0_2_REPRODUCES_todays_geometry_at_the_old_floor():
    # The default must be a no-op for the movers the bot already trades, or
    # enabling it would be a behaviour change disguised as a safety net.
    got, _ = effective_max_dist(_cfg(), {"change_24h_pct": 15.0})
    assert got == 3.0


def test_the_reason_string_explains_WHICH_limit_applied():
    _, why = effective_max_dist(_cfg(), {"change_24h_pct": 5.0})
    assert "0.20" in why and "5.0% move" in why


def test_the_LIVE_RECHECK_uses_the_same_limit():
    """
    Scan and live re-check must agree, or a candidate could be refused on the
    scan figure and admitted on the live one.
    """
    import inspect
    from bot import auto_trader as at
    src = inspect.getsource(at.evaluate_candidate)
    assert "if live_dist > limit:" in src
    assert src.index("limit, limit_why = effective_max_dist") < \
        src.index("if live_dist > limit:")
