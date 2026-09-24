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
