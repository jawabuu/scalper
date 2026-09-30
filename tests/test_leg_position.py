"""
LEG_GATE_MODE — "middle of the move", the operator's hypothesis, tested.

LONGS by adv_bars, forward outcome at 2 and 10 minutes:

    band      n      2min    win     10min    win
    4-6     214    +0.000    47%    +0.002    50%
    6-9     340    -0.009    47%    -0.020    48%
    9-14    416    +0.043    55%    +0.113    58%   <- the band
    14+     720    -0.023    46%    +0.005    50%

241/416 at 10 min, p=0.0007, and +0.113% clears the ~0.070% fee.

The hypothesis was originally about VOLUME. That was tested first and
failed: longs are flat across every adv_vol_trend band at n=1690, and the
ordering REVERSES between the 10- and 30-minute horizons. The intuition was
right; the variable was position in the leg.

`leg_position_allows` is pure, so the decision surface is tested
EXHAUSTIVELY — a wrong boundary here silently costs every long.
"""

import pytest

from bot.auto_trader import AutoTradeConfig, leg_position_allows


@pytest.mark.parametrize("bars,allow", [
    (0, False), (3, False), (5, False), (8, False),     # too early
    (9, True), (11, True), (14, True),                  # the band, INCLUSIVE
    (15, False), (30, False), (200, False),             # too late
])
def test_the_decision_surface(bars, allow):
    got, why = leg_position_allows("long", bars)
    assert got is allow, f"{bars} bars: {why}"


def test_both_boundaries_are_INCLUSIVE():
    # 9 and 14 are IN the measured band. An exclusive boundary would drop
    # the two edges of the only cell that clears the fee.
    assert leg_position_allows("long", 9)[0] is True
    assert leg_position_allows("long", 14)[0] is True
    assert leg_position_allows("long", 8)[0] is False
    assert leg_position_allows("long", 15)[0] is False


def test_SHORTS_PASS_THROUGH_untouched():
    """
    The band was measured on LONGS (n=1690 with outcomes). No equivalent
    short analysis exists, so shorts must not inherit a threshold that was
    never tested on them.
    """
    for bars in (0, 3, 9, 14, 40, None):
        allow, why = leg_position_allows("short", bars)
        assert allow is True
        assert "longs only" in why


@pytest.mark.parametrize("bad", [None, "", "n/a", [], {}])
def test_an_UNKNOWN_leg_length_ALLOWS_the_trade(bad):
    """
    The scanner returns None when the leg is under 4 bars or the window is
    short. A missing reading must never silently halt trading — that failure
    mode cost four hours when the daily halt went quiet.
    """
    allow, why = leg_position_allows("long", bad)
    assert allow is True
    assert "unknown" in why or "unreadable" in why


def test_it_never_raises_on_a_junk_side():
    for side in ("", None, "LONG", "Buy", 7):
        allow, why = leg_position_allows(side, 11)
        assert isinstance(allow, bool) and isinstance(why, str)


def test_custom_bands_are_honoured():
    assert leg_position_allows("long", 20, lo=18, hi=25)[0] is True
    assert leg_position_allows("long", 11, lo=18, hi=25)[0] is False


def test_refusals_carry_the_EVIDENCE_not_just_the_rule():
    # "too early" and "too late" fail for different measured reasons, and a
    # future reader should see which.
    _, early = leg_position_allows("long", 7)
    _, late = leg_position_allows("long", 40)
    assert "too early" in early and "n=340" in early
    assert "too late" in late and "n=720" in late


def test_the_default_is_OFF_and_the_band_matches_the_measurement():
    from bot.config import BotConfig
    c = BotConfig()
    assert c.leg_gate_mode == "off"
    assert (c.leg_bars_min, c.leg_bars_max) == (9, 14)
    assert AutoTradeConfig().leg_gate_mode == "off"


def test_warn_mode_does_not_skip_the_entry():
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.run_once)
    i = src.index("leg gate (warn)")
    assert "continue" not in src[i:i + 400]
    assert 'if _leg_mode in ("warn", "block")' in src
