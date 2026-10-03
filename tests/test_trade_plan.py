"""
bot/trade_plan.py — side, entry, stop, targets from THIS coin's structure.

The replay that motivated it (199 live entries, 240 min forward):

    rule                median      mean     TOTAL    win
    actual              +0.057    -0.153    -30.39    53%
    tiered 1%/2% SL2    +0.614    -0.196    -38.99    62%

Tiering gave the BEST median, win rate and paired difference of anything
tested — and the WORST total, because a FLAT 2% stop fired on 34% of trades.
A stop that fires on a third of trades is not marking "this trade is wrong",
it is marking "price moved a bit".

`build_plan` is PURE — no exchange, no state, no clock — so a plan can be
rebuilt from a shadow log months later and scored against what happened.
"""

import pytest

from bot.trade_plan import PlanConfig, TradePlan, build_plan


def _row(to_high=2.0, to_low=15.0, atr=1.0, sym="X/USDT:USDT"):
    """
    The row as the SCANNER builds it: distances in %, not prices.

    The first version read row["price"], which does not exist — 338 of 338
    plans refused with "no entry price" and the generator never ran. The
    structure is already on the row as pct_below_24h_high / pct_above_24h_low,
    which distance_to_extreme uses today.
    """
    return {"symbol": sym, "pct_below_24h_high": to_high,
            "pct_above_24h_low": to_low, "atr_pct": atr}


def test_the_row_keys_are_the_ones_the_SCANNER_actually_emits():
    # Pins the field names. Reading a key that does not exist fails SILENTLY
    # as a refusal, which is indistinguishable from "no good setups".
    import inspect
    from bot import trade_plan
    src = inspect.getsource(trade_plan.build_plan)
    assert 'row.get("pct_below_24h_high")' in src
    assert 'row.get("pct_above_24h_low")' in src
    assert "row.get(\"price\")" not in src, "a scanner row has no price key"


# ── the stop comes from the extreme the setup FADED ───────────────────────

def test_a_SHORT_is_wrong_when_price_reclaims_the_24h_high():
    p = build_plan(_row(to_high=1.8, to_low=16.7, atr=1.0), "short")
    assert p.refused is None, p.refused
    # 1.8% to the high plus a 0.5 ATR buffer = 2.3% beyond where it sits
    assert abs(p.stop_pct - 2.3) < 1e-6
    assert "24h high" in p.basis


def test_a_LONG_is_wrong_when_price_loses_the_24h_low():
    p = build_plan(_row(to_high=19.6, to_low=2.2, atr=1.0), "long")
    assert p.refused is None, p.refused
    assert abs(p.stop_pct - 2.7) < 1e-6
    assert "24h low" in p.basis


def test_the_stop_distance_is_PER_COIN_not_a_constant():
    """
    The whole point. A flat 2% fired on 34% of trades; 'wrong' is 0.4% away
    on one coin and 3% on another because the coins are different.
    """
    near = build_plan(_row(to_high=0.45, to_low=18.0, atr=0.5), "short")
    far = build_plan(_row(to_high=2.7, to_low=16.0, atr=0.5), "short")
    assert near.refused is None and far.refused is None
    assert near.stop_pct < far.stop_pct
    # 0.71% vs 2.9% on the SAME 24h range — the distance comes from where
    # price sits, not from a constant.
    assert abs(near.stop_pct - far.stop_pct) > 1.5


def test_the_ATR_buffer_keeps_noise_off_the_stop():
    tight = build_plan(_row(to_high=0.9, to_low=17.0, atr=0.2), "short")
    loose = build_plan(_row(to_high=0.9, to_low=17.0, atr=2.0), "short")
    assert loose.stop_pct > tight.stop_pct, "more noise -> more room"


# ── targets are fractions of the ROOM the thesis predicts ────────────────

def test_targets_are_fractions_of_the_room_ahead():
    p = build_plan(_row(to_high=1.8, to_low=16.7, atr=1.0), "short")
    assert abs(p.target_pcts[0][0] - 16.7 * 0.25) < 1e-3
    assert abs(p.target_pcts[1][0] - 16.7 * 0.50) < 1e-3


def test_absolute_levels_appear_ONLY_when_an_entry_is_supplied():
    # The plan is built in percentages; prices are derived. A scanner row has
    # no price, and recomputing one from a stale quote would be worse.
    bare = build_plan(_row(), "short")
    assert bare.stop == 0.0 and bare.stop_pct > 0
    priced = build_plan(_row(), "short", entry=0.3714)
    assert priced.stop > 0.3714


def test_target_sizes_are_carried_so_the_plan_can_be_TIERED():
    p = build_plan(_row(to_high=1.8, to_low=16.7, atr=1.0), "short")
    assert p.refused is None, p.refused
    assert [f for _t, f in p.targets] == [0.34, 0.33]
    assert sum(f for _t, f in p.targets) < 1.0, "a runner must remain"


# ── refusals: a bad plan is refused, never resized ───────────────────────

def test_NO_ROOM_is_refused():
    # A short with the 24h low right beneath it has nowhere to go.
    p = build_plan(_row(to_high=22.0, to_low=0.2, atr=1.0), "short")
    assert p.refused and "room" in p.refused


def test_a_stop_INSIDE_THE_NOISE_is_refused_not_used():
    """
    The structure saying 'the stop is very close' is not a licence to use it:
    inside the noise floor it fires on nothing in particular. That is the
    0.4% callback that cost QNT 19 ROI points.
    """
    p = build_plan(_row(to_high=0.04, to_low=18.0, atr=0.01), "short")
    assert p.refused and "noise floor" in p.refused


def test_a_stop_BEYOND_THE_CAP_is_refused():
    # entry 105 against a 110 high is a 5% structural stop — past the 3.5%
    # cap, so the plan is refused rather than quietly taken at 10x the risk.
    p = build_plan(_row(to_high=4.5, to_low=14.0, atr=0.5), "short")
    assert p.refused and "cap" in p.refused


def test_the_cap_EXPOSES_a_real_tension_and_says_so():
    """
    AUTO_MAX_DIST_PCT=3 admits entries up to 3% from the extreme, so a stop
    placed BEYOND that extreme is necessarily 3%+. At 3.5% the risk per trade
    is SEVEN TIMES the ~0.5% ATR stop used today — a plan-based trade must be
    sized from its own stop, or it is a much bigger bet wearing a new name.
    """
    import inspect
    from bot import trade_plan
    src = inspect.getsource(trade_plan.PlanConfig)
    assert "SEVEN TIMES" in src and "AUTO_MAX_DIST_PCT" in src


def test_a_POOR_reward_to_risk_is_refused():
    cfg = PlanConfig(min_rr=50.0)
    p = build_plan(_row(to_high=1.8, to_low=16.7, atr=1.0), "short", cfg)
    assert p.refused and "reward/risk" in p.refused


def test_rr_is_computed_from_the_LAST_target():
    p = build_plan(_row(to_high=1.8, to_low=16.7, atr=1.0), "short")
    assert p.refused is None, p.refused
    assert abs(p.rr - p.reward_pct / p.stop_pct) < 1e-9


@pytest.mark.parametrize("row", [
    {"symbol": "X"},
    {"symbol": "X", "pct_below_24h_high": None, "pct_above_24h_low": 10},
    {"symbol": "X", "pct_below_24h_high": "n/a", "pct_above_24h_low": 10},
])
def test_unusable_input_is_REFUSED_not_guessed(row):
    p = build_plan(row, "short")
    assert p.refused and p.targets == []


def test_a_missing_atr_does_not_crash():
    p = build_plan(_row(to_high=1.8, to_low=16.7, atr=None), "short")
    assert isinstance(p, TradePlan)


# ── it is a proposal, and must be auditable ──────────────────────────────

def test_as_dict_round_trips_everything_needed_to_score_it_later():
    p = build_plan(_row(to_high=1.8, to_low=16.7, atr=1.0), "short")
    d = p.as_dict()
    for k in ("symbol", "side", "entry", "stop", "targets", "stop_pct",
              "reward_pct", "rr", "basis", "room_pct", "refused"):
        assert k in d


def test_build_plan_is_PURE():
    # No exchange, no clock, no state — so a plan can be rebuilt from a
    # shadow log months later and scored against what actually happened.
    import inspect
    src = inspect.getsource(build_plan)
    for forbidden in ("time.", "self.", "fetch", "requests", "random"):
        assert forbidden not in src, forbidden


def test_the_default_mode_is_OFF():
    from bot.config import BotConfig
    from bot.auto_trader import AutoTradeConfig
    assert BotConfig().trade_plan_mode == "off"
    assert AutoTradeConfig().trade_plan_mode == "off"


def test_LOG_mode_changes_nothing():
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader.run_once)
    i = src.index("PLAN refused")
    assert "continue" not in src[i - 600:i + 400]
