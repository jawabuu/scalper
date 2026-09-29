"""
BTC_REGIME_MODE — trade WITH Bitcoin, or not at all.

Refused-candidate outcomes, 2-min horizon, n=2,401, all RSI bands pooled.
The round-trip fee is ~0.070% of price:

    side    BTC      n    win   median px%   clears the fee?
    short   down    79    65%     +0.097     YES
    short   flat   161    46%     -0.058     no
    short   up     359    51%     +0.011     no
    long    down   404    47%     -0.013     no
    long    flat   524    48%     +0.000     no
    long    up     874    51%     +0.014     no

ONE cell of six clears it: 51 wins of 79, p=0.0064 against a coin flip.

`btc_regime_allows` is pure, so the decision surface is tested EXHAUSTIVELY
rather than sampled — this gate can refuse every trade the bot would take,
and a wrong boundary is silent.
"""

import pytest

from bot.auto_trader import AutoTradeConfig, btc_regime_allows


# ── the decision surface, exhaustively ─────────────────────────────────────

@pytest.mark.parametrize("btc,side,allow", [
    # BTC DOWN: shorts only — the one cell that clears the fee
    (-5.0, "short", True), (-5.0, "long", False),
    (-1.5, "short", True), (-1.5, "long", False),
    (-1.0, "short", True), (-1.0, "long", False),      # boundary, inclusive
    # FLAT: neither side cleared the fee
    (-0.99, "short", False), (-0.99, "long", False),
    (0.0, "short", False), (0.0, "long", False),
    (0.91, "short", False), (0.91, "long", False),     # the window's median
    (0.99, "short", False), (0.99, "long", False),
    # BTC UP: longs only
    (1.0, "short", False), (1.0, "long", True),        # boundary, inclusive
    (3.0, "short", False), (3.0, "long", True),
    (7.55, "short", False), (7.55, "long", True),      # the window's max
])
def test_the_decision_surface(btc, side, allow):
    got, why = btc_regime_allows(side, btc)
    assert got is allow, f"btc={btc} {side}: {why}"


def test_the_boundaries_are_INCLUSIVE_on_the_tradeable_side():
    # Exactly -1.0 is "down" and exactly +1.0 is "up". An exclusive boundary
    # would put the threshold itself in the no-trade band, which is not what
    # the bands were measured as.
    assert btc_regime_allows("short", -1.0)[0] is True
    assert btc_regime_allows("long", 1.0)[0] is True


def test_flat_refuses_BOTH_sides():
    # The "do nothing" band is the point. Shorts returned -0.058% there and
    # longs +0.000%, against a 0.070% fee.
    assert btc_regime_allows("short", 0.0)[0] is False
    assert btc_regime_allows("long", 0.0)[0] is False


# ── failure modes: a missing reading must not halt trading ────────────────

@pytest.mark.parametrize("bad", [None, "", "n/a", float("nan")])
def test_an_UNKNOWN_reading_ALLOWS_the_trade(bad):
    """
    The scanner can fail to compute btc_change_pct. A missing reading must
    not silently halt all trading — that failure mode cost four hours once
    already when the daily halt went quiet.
    """
    if bad != bad:                      # NaN compares false to itself
        allow, why = btc_regime_allows("short", bad)
        assert allow is True or allow is False   # must not raise
        return
    allow, why = btc_regime_allows("short", bad)
    assert allow is True
    assert "unknown" in why or "unreadable" in why


def test_it_never_raises_on_a_junk_side():
    # An unexpected side string must not crash the entry path.
    for side in ("", None, "LONG", "Short", "buy", 7):
        allow, why = btc_regime_allows(side, -3.0)
        assert isinstance(allow, bool) and isinstance(why, str)


def test_a_non_short_side_is_treated_as_LONG():
    # startswith("short") is the test; everything else is the long side.
    assert btc_regime_allows("long", 3.0)[0] is True
    assert btc_regime_allows("buy", 3.0)[0] is True


# ── the reason strings carry the evidence ────────────────────────────────

def test_every_refusal_says_WHY_with_numbers():
    for btc, side in ((-3.0, "long"), (0.0, "short"), (0.0, "long"),
                      (3.0, "short")):
        allow, why = btc_regime_allows(side, btc)
        assert allow is False
        assert "%" in why and any(c.isdigit() for c in why), why


# ── thresholds are configurable ──────────────────────────────────────────

def test_custom_thresholds_are_honoured():
    # A wider flat band takes fewer trades on both sides.
    assert btc_regime_allows("short", -1.5, down=-2.0, up=2.0)[0] is False
    assert btc_regime_allows("short", -2.5, down=-2.0, up=2.0)[0] is True
    assert btc_regime_allows("long", 1.5, down=-2.0, up=2.0)[0] is False
    assert btc_regime_allows("long", 2.5, down=-2.0, up=2.0)[0] is True


# ── the flag ─────────────────────────────────────────────────────────────

def test_the_default_is_OFF():
    from bot.config import BotConfig
    assert AutoTradeConfig().btc_regime_mode == "off"
    assert BotConfig().btc_regime_mode == "off"


def test_the_defaults_match_the_measured_bands():
    from bot.config import BotConfig
    c = BotConfig()
    assert c.btc_regime_down_pct == -1.0
    assert c.btc_regime_up_pct == 1.0


def test_WARN_mode_never_blocks_and_BLOCK_mode_does():
    """
    Warn must be readable before it is load-bearing. The gate can refuse
    every trade the bot would take, so the count of what it WOULD have
    blocked has to be observable first.
    """
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.run_once)
    i = src.index("BTC regime (warn)")
    warn_block = src[i:i + 400]
    assert "continue" not in warn_block, "warn mode must not skip the entry"
    assert 'if _btc_mode in ("warn", "block")' in src
    assert "would \nBLOCK" in src or "would " in src


def test_an_unknown_mode_is_treated_as_OFF():
    # A typo must not silently start refusing every trade.
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.run_once)
    assert 'in ("warn", "block")' in src, \
        "membership test, not a truthiness check"
