"""
tools/replay_exits.py — what would these trades have done under other exits?

The journal cannot answer this: once fail-fast cut a position, the path after
that moment was never recorded. These tests pin the PESSIMISTIC choices that
keep the replay honest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import replay_exits as re          # noqa: E402


def _c(ts, o, h, l, c):
    return (ts, o, h, l, c)


def test_a_candle_hitting_BOTH_stop_and_target_books_the_STOP():
    """
    Intra-candle order is unknown. Assuming the favourable extreme came first
    is the single largest source of self-deception in this kind of replay, so
    the loss is always booked.
    """
    # short from 100: stop at 102, target at 98, one candle spans both
    px, why = re.replay([_c(0, 100, 103, 97, 99)], "short", 100.0,
                        callback_pct=None, stop_pct=2.0, tp_pct=2.0)
    assert why == "stop" and px == 102.0


def test_the_same_holds_for_a_LONG():
    px, why = re.replay([_c(0, 100, 103, 97, 101)], "long", 100.0,
                        callback_pct=None, stop_pct=2.0, tp_pct=2.0)
    assert why == "stop" and px == 98.0


def test_a_target_hit_with_no_stop_breach_is_taken():
    px, why = re.replay([_c(0, 100, 100.5, 97, 98)], "short", 100.0,
                        callback_pct=None, stop_pct=5.0, tp_pct=2.0)
    assert why == "tp" and px == 98.0


def test_the_trail_follows_the_BEST_price_not_the_close():
    # short: price falls to 95 then recovers. A 1% trail from 95 triggers at
    # 95.95, which the same candle's high of 97 reaches.
    px, why = re.replay([_c(0, 100, 97, 95, 96)], "short", 100.0,
                        callback_pct=1.0, stop_pct=None, tp_pct=None)
    # on the FIRST candle best is still entry, so the trail sits at 101 and
    # does not trigger; the next candle trails from 95
    assert why in ("trail", "timeout")


def test_a_trail_that_never_triggers_returns_the_LAST_CLOSE():
    px, why = re.replay([_c(0, 100, 100, 98, 98), _c(1, 98, 98, 96, 96)],
                        "short", 100.0, callback_pct=5.0, stop_pct=None,
                        tp_pct=None)
    assert why == "timeout" and px == 96


def test_pnl_is_in_PRICE_percent_and_NET_of_fees():
    # short 100 -> 99 is +1.0% gross, minus the round trip
    v = re.pnl_pct("short", 100.0, 99.0)
    assert abs(v - (1.0 - re.FEE_PCT_ROUND_TRIP)) < 1e-9


def test_a_long_and_a_short_of_equal_size_are_symmetric():
    a = re.pnl_pct("short", 100.0, 99.0)
    b = re.pnl_pct("long", 100.0, 101.0)
    assert abs(a - b) < 1e-9


def test_use_trail_False_ignores_the_callback_entirely():
    # TP-only mode must not exit on a trail even with a callback passed
    px, why = re.replay([_c(0, 100, 101, 99.5, 99.6)], "short", 100.0,
                        callback_pct=0.1, stop_pct=None, tp_pct=5.0,
                        use_trail=False)
    assert why == "timeout"


def test_the_comparison_against_actual_is_PAIRED(capsys, tmp_path):
    """
    The same trade appears under every rule, so most of the variance is "which
    trade was it" and cancels on differencing. Comparing the two
    DISTRIBUTIONS gave t~1.0 on a difference that is far better determined
    than that — the unpaired spread was measuring the wrong thing.
    """
    import inspect
    src = inspect.getsource(re.main)
    assert "PAIRED vs actual" in src
    assert "x - y for x, y in zip" in src, "must difference PER TRADE"


def test_the_paired_block_flags_intervals_that_exclude_zero():
    import inspect
    src = inspect.getsource(re.main)
    assert "the 95% interval excludes zero" in src


def test_the_replay_can_be_filtered_to_one_exit_reason_and_peak_band():
    """
    The journal cannot say what a fail-fast trade would have done uncut:
    fail-fast closes at -5% ROI regardless of peak, so every peak band shows
    the same final (-0.55% to -0.61% across all four, 2026-09-26). Only a
    replay restricted to the affected subset answers it.
    """
    import inspect
    src = inspect.getsource(re.main)
    assert '--only-exit' in src and '--peak-band' in src
    assert 'str(t.get("exit_reason")) == args.only_exit' in src
    assert 'lo <= _f(t["peak_roi"]) < hi' in src


# ── FAIL-FAST VARIANTS: test the cut offline, on LIVE's own trades ─────────

def _bars(entry, lows, highs=None):
    highs = highs or [e * 1.001 for e in lows]
    return [(i, entry, h, l, l) for i, (l, h) in enumerate(zip(lows, highs))]


def test_fail_fast_waits_for_the_DELAY_before_cutting():
    """
    GUARD_FAIL_FAST_S=60 means the cut cannot fire on the first minute.
    Modelling it as a plain stop would cut trades the live rule never
    touches and overstate the saving.
    """
    # dips hard on bar 0, recovers after
    c = _bars(100.0, [99.0, 100.5, 101.0])
    px, why = re.replay(c, "long", 100.0, callback_pct=None, stop_pct=None,
                        tp_pct=None, ff_pct=0.5, ff_after_bars=1)
    assert why != "fail_fast", "bar 0 is inside the 60s delay"


def test_fail_fast_fires_after_the_delay():
    c = _bars(100.0, [99.9, 99.0, 99.0])
    px, why = re.replay(c, "long", 100.0, callback_pct=None, stop_pct=None,
                        tp_pct=None, ff_pct=0.5, ff_after_bars=1)
    assert why == "fail_fast"
    assert abs(px - 99.5) < 1e-6


def test_the_PEAK_CEILING_blocks_the_cut():
    """
    _peak_ceiling ties fail-fast to GUARD_BREAKEVEN_AT_ROI: a trade that has
    been meaningfully green is NOT fail-fast's business. Ignoring that would
    cut winners the live rule protects.
    """
    # goes +1% on bar 0 (above a 0.3% ceiling), then dips
    c = [(0, 100.0, 101.0, 100.0, 101.0), (1, 101.0, 101.0, 99.0, 99.0)]
    px, why = re.replay(c, "long", 100.0, callback_pct=None, stop_pct=None,
                        tp_pct=None, ff_pct=0.5, ff_after_bars=1,
                        ff_peak_ceiling=0.3)
    assert why != "fail_fast", "peak exceeded the ceiling"


def test_without_a_ceiling_the_same_trade_IS_cut():
    c = [(0, 100.0, 101.0, 100.0, 101.0), (1, 101.0, 101.0, 99.0, 99.0)]
    _px, why = re.replay(c, "long", 100.0, callback_pct=None, stop_pct=None,
                         tp_pct=None, ff_pct=0.5, ff_after_bars=1,
                         ff_peak_ceiling=None)
    assert why == "fail_fast"


def test_fail_fast_is_checked_BEFORE_the_wider_rules():
    # It is the tightest level, and the guardian evaluates it on its own
    # poll. Checking the stop first would attribute the exit wrongly.
    import inspect
    src = inspect.getsource(re.replay)
    assert src.index('"fail_fast"') < src.index('"stop"')


def test_the_variants_are_configurable_from_the_cli():
    import inspect
    src = inspect.getsource(re.main)
    assert "--ff-roi" in src and "--ff-peak-ceiling" in src
    assert "ff_pct=ff_roi / lev" in src, "ROI must be converted to price %"


# ── RUNNER MODE: widen the trail once ahead and never deeply red ──────────

def test_runner_mode_widens_the_trail_above_the_threshold():
    """
    Continuation on 312 demo trades is a near-constant ~1/3 at every level
    (5->10: 53%, 10->20: 38%, 20->40: 30%, 40->60: 38%). A power-law tail is
    the condition under which letting winners run beats banking them.
    """
    # rises to +3%, then retraces 1.2% from the high
    c = [(0, 100.0, 103.0, 100.0, 103.0), (1, 103.0, 103.0, 101.8, 101.8)]
    _px, why = re.replay(c, "long", 100.0, callback_pct=1.0, stop_pct=None,
                         tp_pct=None)
    assert why == "trail", "a 1% callback fires on a 1.2% retrace"
    _px, why = re.replay(c, "long", 100.0, callback_pct=1.0, stop_pct=None,
                         tp_pct=None, runner_at_pct=2.0, runner_mult=2.0)
    assert why != "trail", "at 2x the callback is 2%, so it survives"


def test_runner_mode_does_NOT_apply_below_the_threshold():
    c = [(0, 100.0, 101.0, 100.0, 101.0), (1, 101.0, 101.0, 99.8, 99.8)]
    _px, why = re.replay(c, "long", 100.0, callback_pct=1.0, stop_pct=None,
                         tp_pct=None, runner_at_pct=5.0, runner_mult=2.0)
    assert why == "trail", "peak never reached the runner threshold"


def test_a_trade_that_went_DEEPLY_RED_is_not_a_runner():  # noqa: D401
    """
    Of 27 trades peaking >= +20% ROI, only 2 ever dipped below -5% and NONE
    below -10%. A trade that goes deeply red is not a future big winner, so
    it must not be granted extra room.
    """
    # dips to -6% first, then rallies to +3%, then retraces
    c = [(0, 100.0, 100.0, 94.0, 94.0),
         (1, 94.0, 110.0, 94.0, 110.0),
         (2, 110.0, 110.0, 101.0, 101.0)]
    _px, why = re.replay(c, "long", 100.0, callback_pct=8.0, stop_pct=None,
                         tp_pct=None, runner_at_pct=2.0, runner_mult=2.0,
                         runner_max_trough_pct=5.0)
    assert why == "trail", "the -6% dip disqualifies it from runner mode"


def test_the_trough_filter_can_be_disabled():
    # A WIDE callback so the dip itself does not trigger the trail — with a
    # 1% callback a -6% dip exits on bar 0 before any rally can happen.
    c = [(0, 100.0, 100.0, 94.0, 94.0),
         (1, 94.0, 110.0, 94.0, 110.0),
         (2, 110.0, 110.0, 92.0, 92.0)]
    # best = 110. At 1x the trigger is 101.2; at 2x it is 92.4. A retrace to
    # 92 clears BOTH, so the two branches are distinguished by the LABEL.
    _px, why = re.replay(c, "long", 100.0, callback_pct=8.0, stop_pct=None,
                         tp_pct=None, runner_at_pct=2.0, runner_mult=2.0,
                         runner_max_trough_pct=None)
    assert why == "runner", "filter off -> the -6% dip does not disqualify it"
    _px, why = re.replay(c, "long", 100.0, callback_pct=8.0, stop_pct=None,
                         tp_pct=None, runner_at_pct=2.0, runner_mult=2.0,
                         runner_max_trough_pct=5.0)
    assert why == "trail", "filter on -> the dip disqualifies it"


def test_runner_exits_are_LABELLED_separately():
    # So the exit mix shows how often the wider trail actually mattered.
    c = [(0, 100.0, 105.0, 100.0, 105.0), (1, 105.0, 105.0, 102.0, 102.0)]
    _px, why = re.replay(c, "long", 100.0, callback_pct=1.0, stop_pct=None,
                         tp_pct=None, runner_at_pct=2.0, runner_mult=2.0)
    assert why == "runner"


def test_shorts_are_handled_symmetrically():
    # best = 97 (a +3% short gain). At 2x the callback is 2.0%, so the
    # trigger is 97 * 1.02 = 98.94 — the retrace must clear THAT, not 1%.
    c = [(0, 100.0, 100.0, 97.0, 97.0), (1, 97.0, 99.5, 97.0, 99.5)]
    _px, why = re.replay(c, "short", 100.0, callback_pct=1.0, stop_pct=None,
                         tp_pct=None, runner_at_pct=2.0, runner_mult=2.0)
    assert why == "runner"


def test_the_cli_exposes_all_three_knobs():
    import inspect
    src = inspect.getsource(re.main)
    for flag in ("--runner-at", "--runner-mult", "--runner-max-trough"):
        assert flag in src
    assert "runner_at_pct=at_roi / lev" in src, "ROI must convert to price %"


# ── TIERED: a POSITION trade, not a scalp ─────────────────────────────────

def _long(bars):
    """bars = [(high, low)] -> candle tuples at entry 100."""
    return [(i, 100.0, h, l, l) for i, (h, l) in enumerate(bars)]


def test_tiers_fill_IN_ORDER_and_bank_their_fractions():
    """
    Every other rule here is all-or-nothing, which is why widening a trail
    never helped: it risks the WHOLE position to capture a one-in-three
    continuation. Tiering banks most of it BEFORE that risk arises.
    """
    c = _long([(100.5, 100.0), (101.5, 101.0), (102.5, 102.0)])
    pnl, why = re.replay_tiered(c, "long", 100.0, stop_pct=2.0,
                                tiers=[(1.0, 0.34), (2.0, 0.33)])
    assert "tier" in why
    # 0.34 at +1%, 0.33 at +2%, 0.33 running to a +2% close, each slice
    # charged the 0.09% round trip:
    #   0.34*(1.00-0.09) + 0.33*(2.00-0.09) + 0.33*(2.00-0.09) = 1.570
    assert abs(pnl - 1.570) < 0.01, pnl


def test_EVERY_SLICE_is_charged_the_full_round_trip_fee():
    """
    Partial exits mean MORE round trips. A tiered rule that charged the fee
    once would look better than it is — and fees are the dominant cost here,
    at 0.070% against a 0.1175% median gross move.
    """
    import inspect
    src = inspect.getsource(re.replay_tiered)
    assert src.count("pnl_pct(side, entry") >= 3


def test_the_stop_takes_the_WHOLE_remainder():
    c = _long([(100.0, 97.9)])
    pnl, why = re.replay_tiered(c, "long", 100.0, stop_pct=2.0,
                                tiers=[(1.0, 0.34), (2.0, 0.33)])
    assert why == "stop"
    assert pnl < -2.0, "full size at -2% plus fees"


def test_a_stop_AFTER_a_tier_is_labelled_and_smaller():
    # The banked tier must survive the stop, or tiering would be pointless.
    c = _long([(101.5, 100.0), (100.0, 97.9)])
    pnl, why = re.replay_tiered(c, "long", 100.0, stop_pct=2.0,
                                tiers=[(1.0, 0.34)])
    assert "stop after 1 tier" in why
    assert pnl > -2.0, "the +1% slice offsets part of the loss"


def test_the_stop_is_checked_BEFORE_the_tiers():
    # A bar that spans both must count as the stop. Taking profit on a bar
    # that also breached the stop would flatter every result.
    import inspect
    src = inspect.getsource(re.replay_tiered)
    assert src.index('return realised, ("stop"') < src.index("while pending")


def test_the_runner_only_trails_ONCE_A_TIER_HAS_FILLED():
    """
    Before any tier fills, the structural stop is the whole protection — by
    design. Trailing from the start would make this just another scalp.
    """
    import inspect
    src = inspect.getsource(re.replay_tiered)
    assert "if trail_pct and filled and remaining > 0:" in src


def test_all_tiers_filling_closes_the_position():
    c = _long([(103.0, 100.0)])
    pnl, why = re.replay_tiered(c, "long", 100.0, stop_pct=2.0,
                                tiers=[(1.0, 0.5), (2.0, 0.5)])
    assert why == "all tiers"


def test_shorts_mirror_longs():
    c = [(0, 100.0, 100.0, 98.5, 98.5)]
    pnl, why = re.replay_tiered(c, "short", 100.0, stop_pct=2.0,
                                tiers=[(1.0, 0.5)])
    assert "tier" in why and pnl > 0


def test_the_cli_defaults_state_the_STRUCTURAL_stop():
    import inspect
    src = inspect.getsource(re.main)
    assert "--tier-stop" in src and "4x the current risk" in src
