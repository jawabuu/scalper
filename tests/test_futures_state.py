"""Persistence of restart-critical futures state."""
import json
import os
import tempfile
import time

import pytest

from bot import futures_state
from bot.futures_guard import GuardConfig, GuardState
from bot.futures_guardian import FuturesGuardian


@pytest.fixture
def path():
    return os.path.join(tempfile.mkdtemp(), "fstate.json")


def _guardian(path):
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig(atr_stop_mult=1.5, atr_stop_min_roi=4, atr_stop_max_roi=30)
    g.exchange = None
    g.dry_run = True
    g.demo = True
    g.atr_timeframe = "3m"
    g._init_runtime_state()
    g.risk_pct = 1.0
    g.state_path = path
    return g


def _populate(g):
    st = GuardState()
    st.peak_roi, st.armed, st.native_trail_id = 41.05, True, "t9"
    g._states["BR/USDT:USDT"] = st
    g._pos_meta["BR/USDT:USDT"] = {
        "entry_context": {"sized_stop_roi": 6.67, "rsi": 78},
        "opened_seen_at": time.time() - 600,
    }
    g._closed_trades = [{"symbol": "X/USDT:USDT", "final_roi": 5.0}]
    return g


def test_sized_stop_survives_a_restart(path):
    """
    Losing this made the guardian recompute ATR at discovery and place a stop
    2.8x wider than the position was sized for.
    """
    _populate(_guardian(path)).save_state()
    g2 = _guardian(path)
    g2.load_state(path)
    ctx = g2._pos_meta["BR/USDT:USDT"]["entry_context"]
    assert ctx["sized_stop_roi"] == pytest.approx(6.67)


def test_peak_roi_survives_a_restart(path):
    """Otherwise the trail's high-water mark resets and gives back the run."""
    _populate(_guardian(path)).save_state()
    g2 = _guardian(path)
    g2.load_state(path)
    st = g2._states["BR/USDT:USDT"]
    assert st.peak_roi == pytest.approx(41.05)
    assert st.armed is True
    assert st.native_trail_id == "t9"


def test_trade_history_survives_a_restart(path):
    _populate(_guardian(path)).save_state()
    g2 = _guardian(path)
    g2.load_state(path)
    assert len(g2._closed_trades) == 1


def test_daily_halt_is_not_reset_by_restarting(path):
    """
    A restart used to rebase the daily-loss baseline, so redeploying while down
    handed back a fresh allowance. The halt must survive.
    """
    g = _populate(_guardian(path))
    g._safety_snapshot = lambda: {
        "day_start_balance": 4729.0, "day_key": "2026-09-06",
        "halted_reason": "daily loss limit hit",
        "symbol_blocked_until": {"LDO/USDT:USDT": time.time() + 900},
        "reentries_today": {"LDO/USDT:USDT": 2},
    }
    g.save_state()

    g2 = _guardian(path)
    g2.load_state(path)
    from bot.auto_trader import AutoTrader, AutoTradeConfig, SafetyState
    a = AutoTrader.__new__(AutoTrader)
    a.cfg = AutoTradeConfig()
    a.state = SafetyState()
    a._log = []
    a.restore_safety(g2._restored_safety)

    assert a.state.halted_reason == "daily loss limit hit"
    assert a.state.day_start_balance == pytest.approx(4729.0)
    assert "LDO/USDT:USDT" in a.state.symbol_blocked_until
    assert a.state.reentries_today["LDO/USDT:USDT"] == 2


def test_missing_file_starts_clean(path):
    g = _guardian(path)
    g.load_state(path)          # nothing written yet
    assert g._states == {}


def test_corrupt_file_does_not_block_startup(path):
    with open(path, "w") as fh:
        fh.write("{not json")
    g = _guardian(path)
    g.load_state(path)          # must not raise
    assert g._states == {}


def test_schema_mismatch_is_ignored(path):
    with open(path, "w") as fh:
        json.dump({"schema": 999, "states": {"X": {}}}, fh)
    assert futures_state.load(path) == {}


def test_write_is_atomic(path):
    """A crash mid-write must not leave a truncated file."""
    _populate(_guardian(path)).save_state()
    with open(path) as fh:
        json.load(fh)           # parses cleanly
    leftovers = [f for f in os.listdir(os.path.dirname(path)) if f.startswith(".fstate-")]
    assert leftovers == []


def test_persistence_disabled_when_no_path():
    g = _guardian("")
    g._states["X"] = GuardState()
    g.save_state()              # must be a no-op, not an error


# ── Diagnostics for the sized-stop handoff ───────────────────────────────────

def test_writability_is_verified_at_startup(path):
    g = _guardian(path)
    assert g.verify_state_path() is True


def test_unwritable_path_is_reported_not_silent():
    g = _guardian("/proc/nope/definitely/not/writable/s.json")
    assert g.verify_state_path() is False


def test_disabled_persistence_is_reported():
    g = _guardian("")
    assert g.verify_state_path() is False


def test_cap_log_is_not_repeated_every_cycle(path):
    """
    Margin drifts with unrealised PnL, so the cap recomputes each cycle and
    logged an almost-identical line every few seconds.
    """
    from bot.futures_guard import FuturesPosition
    g = _guardian(path)
    g._wallet_balance_cached = 4428.0
    for margin in (492.0, 492.4, 491.6, 492.2):
        pos = FuturesPosition("ROSE/USDT:USDT", "short", 0.007395, 1330629.0, 20, margin)
        g._cap_stop_to_budget(pos, 17.6)
    capped = [a for a in g._actions if a["action"] == "stop_capped"]
    assert len(capped) == 1, f"logged {len(capped)} times for a stable level"


def test_cap_log_repeats_when_the_level_moves_materially(path):
    from bot.futures_guard import FuturesPosition
    g = _guardian(path)
    g._wallet_balance_cached = 4428.0
    for margin in (492.0, 900.0):        # a real change in the capped level
        pos = FuturesPosition("ROSE/USDT:USDT", "short", 0.007395, 1330629.0, 20, margin)
        g._cap_stop_to_budget(pos, 17.6)
    assert len([a for a in g._actions if a["action"] == "stop_capped"]) == 2


# ── Startup reset flag ───────────────────────────────────────────────────────

def _seed(path):
    futures_state.save(
        path, states={}, pos_meta={},
        closed_trades=[{"symbol": f"T{i}", "final_roi": 1.0} for i in range(5)],
        safety={"day_start_balance": 4494.25, "halted_reason": "daily loss limit hit"})
    data = json.load(open(path))
    data["states"] = {"BR/USDT:USDT": {"peak_roi": 12.0, "armed": True}}
    data["pos_meta"] = {"BR/USDT:USDT": {"entry_context": {"sized_stop_roi": 6.67}}}
    futures_state._atomic_write(path, data)


def test_history_reset_drops_trades_but_keeps_positions(path):
    """
    Contaminated trade history is the reason to reset — open positions should
    keep the stop they were sized for, and the daily baseline should stand.
    """
    _seed(path)
    futures_state.reset(path, "history")
    data = futures_state.load(path)
    assert data["closed_trades"] == []
    assert "BR/USDT:USDT" in data["states"]
    assert data["pos_meta"]["BR/USDT:USDT"]["entry_context"]["sized_stop_roi"] == 6.67
    assert data["safety"]["day_start_balance"] == pytest.approx(4494.25)


def test_history_reset_preserves_the_halt(path):
    """Resetting the record must not hand back a fresh daily allowance."""
    _seed(path)
    futures_state.reset(path, "history")
    assert futures_state.load(path)["safety"]["halted_reason"]


def test_all_reset_clears_everything(path):
    _seed(path)
    futures_state.reset(path, "all")
    assert not os.path.exists(path)
    assert futures_state.load(path) == {}


def test_reset_archives_rather_than_deletes(path):
    _seed(path)
    backup = futures_state.reset(path, "all")
    assert backup and os.path.exists(backup)
    with open(backup) as fh:
        assert len(json.load(fh)["closed_trades"]) == 5


def test_unset_or_invalid_mode_does_nothing(path):
    _seed(path)
    for mode in ("", None, "yes", "true"):
        futures_state.reset(path, mode)
    assert len(futures_state.load(path)["closed_trades"]) == 5


def test_reset_with_no_file_is_harmless(path):
    assert futures_state.reset(path, "all") == ""
