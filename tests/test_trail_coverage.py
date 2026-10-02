"""
A reduceOnly trail must cover the WHOLE position.

USELESS 2026-10-02 05:22. A maker limit for 7937 contracts had filled 392
(5%) when the guardian armed. The armed trail went on for 392, the adaptive
trail was cancelled as "superseded", and stop_order_id was None. The position
grew past 1200 contracts covered by a trail that could close 392 of them, and
sailed from a +13.25% peak down through a +7.25% stop that could not close it.

TWO failures, and the second only exists because of the first:

  1. the trail was PLACED while the position was 5% filled
  2. nothing noticed afterwards that the position had outgrown it

Fix 1 lives in `_place_native_trail` — the SHARED helper — because the first
attempt guarded `_arm_at_entry` only and the RUNTIME arm path walked straight
past it. Guarding one call site and missing its sibling is the same error
that left MANUAL_INITIAL_GUARD_ONLY incomplete twice.
"""

import inspect

from bot.futures_guard import GuardState
from bot.futures_guardian import FuturesGuardian


# ── 1: do not place a trail for a position that is still filling ──────────

def test_the_fill_guard_is_in_the_SHARED_helper():
    src = inspect.getsource(FuturesGuardian._place_native_trail)
    assert "NOT placing a native trail" in src
    assert "pos.qty < float(sized_qty) * 0.9" in src


def test_the_guard_runs_BEFORE_anything_is_sized():
    # Sizing first and returning later would still have computed a callback
    # from a corrupted leverage.
    src = inspect.getsource(FuturesGuardian._place_native_trail)
    assert src.index("sized_qty") < src.index("lev = pos.effective_leverage")


def test_a_missing_sized_qty_does_not_block_placement():
    # Operator-opened positions and restarts have no handoff. They must still
    # get a trail.
    src = inspect.getsource(FuturesGuardian._place_native_trail)
    assert "if sized_qty and pos.qty and" in src


def test_the_reason_explains_WHY_a_fraction_is_dangerous():
    src = inspect.getsource(FuturesGuardian._place_native_trail)
    assert "reduceOnly" in src and "leave the rest" in src


# ── 2: notice afterwards if the position outgrows the trail ───────────────

class _Pos:
    symbol = "USELESS/USDT:USDT"
    def __init__(self, qty): self.qty = qty


class _Ex:
    def __init__(self, boom=False): self.boom, self.cancelled = boom, []
    def cancel_order(self, oid, sym):
        if self.boom: raise RuntimeError("-2011")
        self.cancelled.append(oid)


def _g(ex):
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.exchange = ex
    g._pos_meta = {}
    g._pending_cancels = {}
    import threading
    g._lock = threading.RLock()
    g._record = lambda *a, **k: None
    return g


def _state(tid="t-1", placed=392.0):
    st = GuardState()
    st.native_trail_id = tid
    st.native_trail_qty = placed
    st.armed = True
    return st


def test_a_trail_that_no_longer_covers_the_position_is_CANCELLED():
    ex = _Ex(); g = _g(ex); st = _state()
    assert g._trail_covers_position(_Pos(1200.0), st) is False
    assert ex.cancelled == ["t-1"]
    assert st.native_trail_id is None
    assert st.armed is False, "must re-arm so a correct trail replaces it"


def test_full_coverage_is_left_alone():
    ex = _Ex(); g = _g(ex); st = _state(placed=7937.0)
    assert g._trail_covers_position(_Pos(7937.0), st) is True
    assert ex.cancelled == []


def test_a_SMALL_growth_is_tolerated():
    # Rounding and partial reductions move qty slightly; cancelling on every
    # wobble would churn protective orders.
    ex = _Ex(); g = _g(ex); st = _state(placed=1000.0)
    assert g._trail_covers_position(_Pos(1030.0), st) is True
    assert ex.cancelled == []


def test_no_trail_or_no_recorded_qty_is_a_NO_OP():
    g = _g(_Ex())
    assert g._trail_covers_position(_Pos(500.0), GuardState()) is True
    st = _state(placed=0.0)
    assert g._trail_covers_position(_Pos(500.0), st) is True


def test_a_FAILED_cancel_is_QUEUED_not_forgotten():
    """
    Neither the listing nor the cancel result is trustworthy on this venue,
    so the bot's own queue is the only durable record of "this order should
    not exist".
    """
    ex = _Ex(boom=True); g = _g(ex); st = _state()
    g._trail_covers_position(_Pos(1200.0), st)
    assert "t-1" in g._pending_cancels.get("USELESS/USDT:USDT", [])
    assert st.native_trail_id is None


def test_the_check_runs_every_poll():
    src = inspect.getsource(FuturesGuardian.manage_position)
    assert "_trail_covers_position" in src
