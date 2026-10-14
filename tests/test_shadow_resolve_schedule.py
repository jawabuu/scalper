"""
The outcome resolver must run on a SCHEDULE, not on a human remembering.

It had not run in TEN DAYS. Staleness here is INVISIBLE: the summary looks
healthy, the counts rise, and the window quietly stops moving. On 2026-10-04
the decision log ran to 10-04 01:52 while the newest observation was 09-26
18:14 — every analysis for a week rested on five days of data from BEFORE the
bug fixes.
"""

import time

from bot.auto_trader import AutoTrader, AutoTradeConfig
from bot.shadow_outcomes import collapse, resolve


class _Ex:
    def __init__(self): self.calls = 0
    def fetch_ohlcv(self, *a, **k):
        self.calls += 1
        return []


def _at(**kw):
    a = AutoTrader.__new__(AutoTrader)
    a.cfg = AutoTradeConfig(**kw)
    a._resolve_day = None
    a.guardian = type("G", (), {"exchange": _Ex()})()
    return a


def test_it_is_OFF_by_default():
    from bot.config import BotConfig
    assert BotConfig().shadow_resolve_on_rollover is False
    assert AutoTradeConfig().shadow_resolve_on_rollover is False
    assert _at()._resolve_outcomes_on_rollover() is False


def test_it_runs_ONCE_PER_DAY_not_once_per_cycle():
    # run_once fires every 30s. Resolving on each would be thousands of
    # exchange calls an hour.
    a = _at(shadow_resolve_on_rollover=True)
    now = time.time()
    assert a._resolve_outcomes_on_rollover(now) is True
    assert a._resolve_outcomes_on_rollover(now + 60) is False
    assert a._resolve_outcomes_on_rollover(now + 3600) is False


def test_a_NEW_DAY_runs_it_again():
    a = _at(shadow_resolve_on_rollover=True)
    now = time.time()
    a._resolve_outcomes_on_rollover(now)
    # The previous run's thread clears this when it finishes; clearing it
    # here keeps the test about the DAY KEY rather than thread timing.
    a._resolve_running = False
    assert a._resolve_outcomes_on_rollover(now + 26 * 3600) is True


def test_an_UNFINISHED_run_BLOCKS_the_next_night():
    """
    The thread is a daemon with no supervisor. A stalled run would otherwise
    be joined by a second the next night, and a third after that — which is
    how a hang becomes a pile-up. The operator hit the hang and turned the
    scheduled job off; this is why it is safe to turn back on.
    """
    a = _at(shadow_resolve_on_rollover=True)
    now = time.time()
    a._resolve_outcomes_on_rollover(now)
    a._resolve_running = True                  # pretend it is still going
    assert a._resolve_outcomes_on_rollover(now + 26 * 3600) is False


def test_the_running_flag_is_cleared_even_on_a_CRASH():
    # Otherwise one bad night blocks every future run.
    import inspect
    from bot.auto_trader import AutoTrader
    src = inspect.getsource(AutoTrader._resolve_outcomes_on_rollover)
    assert "finally:" in src
    assert "self._resolve_running = False" in src


def test_resolve_takes_a_WALL_CLOCK_budget_not_just_a_count():
    """
    SHADOW_RESOLVE_MAX bounds how MANY calls are made; it does nothing about
    how LONG one takes. A single stalled request hung the whole run.
    """
    import inspect
    src = inspect.getsource(resolve)
    assert "budget_s" in src
    assert "if deadline and time.time() > deadline:" in src
    assert "break" in src


def test_a_partial_pass_is_not_a_failure():
    # The rest stays ripe for the next run, so stopping early is safe.
    import inspect
    src = inspect.getsource(resolve)
    i = src.index("budget of")
    assert "stays ripe" in src[i:i + 400]


def test_the_budget_is_configurable():
    from bot.config import BotConfig
    assert BotConfig().shadow_resolve_budget_s == 180.0


def test_it_runs_even_when_AUTO_TRADE_IS_DISABLED():
    """
    The resolver is a REPORTING job. Auto-trade being paused is exactly when
    a stale dataset would go unnoticed longest.
    """
    import inspect
    src = inspect.getsource(AutoTrader.run_once)
    assert (src.index("_resolve_outcomes_on_rollover")
            < src.index("if not self.cfg.enabled"))


def test_it_NEVER_raises_into_the_trading_loop():
    a = _at(shadow_resolve_on_rollover=True)
    a.guardian = None                       # worst case
    assert a._resolve_outcomes_on_rollover() is False


def test_resolve_is_BOUNDED_and_takes_the_OLDEST_first():
    """
    One exchange call per observation. An unbounded run on a backlog of tens
    of thousands would compete with the guardian and the scanner for the same
    rate limit. Oldest-first keeps the resolved period CONTIGUOUS rather than
    pocked with holes no analysis can reason about.
    """
    import inspect
    src = inspect.getsource(resolve)
    assert "if limit is not None and len(ripe) > limit:" in src
    assert 'ripe.sort(key=lambda g: g[0]["ts"])' in src
    assert "ripe = ripe[:limit]" in src


def test_the_cap_is_configurable():
    from bot.config import BotConfig
    assert BotConfig().shadow_resolve_max == 500
    assert _at(shadow_resolve_max=50).cfg.shadow_resolve_max == 50


def test_the_cli_exposes_the_same_limit():
    import pathlib
    t = (pathlib.Path(__file__).resolve().parents[1]
         / "tools" / "resolve_shadow_outcomes.py").read_text()
    assert "--limit" in t and "limit=args.limit" in t
