"""
Tests for the futures guardian exchange layer.

Uses a fake exchange so the safety properties are verified as behaviour, not
assumed: reduce-only on every order, place-then-cancel ordering, and never
cancelling a non-reduce-only (entry) order.
"""
import pytest

from bot.futures_guard import GuardConfig, price_for_roi, FuturesPosition
from bot.futures_guardian import FuturesGuardian


class FakeExchange:
    """Minimal stand-in recording every call the guardian makes."""

    def __init__(self, positions=None, orders=None, price=100.0):
        self._positions = positions or []
        self._orders = orders or []
        self._price = price
        self.created = []
        self.cancelled = []
        self.call_log = []
        self._next_id = 1000
        self.fail_next_create = False

    # -- reads
    def fetch_positions(self):
        return self._positions

    def fetch_open_orders(self, symbol):
        return [o for o in self._orders if o.get("symbol", symbol) == symbol]

    def fetch_ticker(self, symbol):
        return {"last": self._price}

    # -- precision helpers
    def price_to_precision(self, symbol, price):
        # Realistic tick precision — 4dp is far too coarse for sub-dollar coins
        # and introduces rounding error larger than the values under test.
        return f"{float(price):.8f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.3f}"

    # -- writes
    def create_order(self, symbol, type, side, amount, price=None, params=None):
        if self.fail_next_create:
            self.fail_next_create = False
            raise RuntimeError("exchange rejected order")
        self._next_id += 1
        oid = str(self._next_id)
        rec = {"id": oid, "symbol": symbol, "type": type, "side": side,
               "amount": amount, "params": params or {}}
        self.created.append(rec)
        self.call_log.append(("create", oid))
        return rec

    def cancel_order(self, order_id, symbol):
        self.cancelled.append((order_id, symbol))
        self.call_log.append(("cancel", order_id))

    def set_sandbox_mode(self, on):
        pass


def _guardian(fake, dry_run=False, cfg=None):
    g = FuturesGuardian.__new__(FuturesGuardian)     # bypass ccxt construction
    g.cfg = (cfg or GuardConfig()).validate()
    g.testnet = True
    g.dry_run = dry_run
    g.poll_interval = 1.0
    g.exchange = fake
    g._states = {}
    import threading
    g._lock = threading.RLock()
    g._last_cycle_ts = 0.0
    g._last_error = None
    g._actions = []
    return g


def _raw_pos(side="short", entry=100.0, contracts=10.0, lev=10, margin=100.0):
    return {"symbol": "DOGE/USDT:USDT", "side": side, "entryPrice": entry,
            "contracts": contracts, "leverage": lev, "initialMargin": margin}


# ── position parsing ─────────────────────────────────────────────────────────

def test_fetch_positions_parses_short():
    g = _guardian(FakeExchange(positions=[_raw_pos("short")]))
    ps = g.fetch_positions()
    assert len(ps) == 1
    assert ps[0].side == "short"
    assert ps[0].leverage == 10
    assert ps[0].margin == pytest.approx(100.0)


def test_fetch_positions_skips_zero_size():
    g = _guardian(FakeExchange(positions=[_raw_pos(contracts=0)]))
    assert g.fetch_positions() == []


# ── every order must be reduce-only ──────────────────────────────────────────

def test_placed_stop_is_reduce_only():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    assert fake.created, "guardian should have placed a protective stop"
    for o in fake.created:
        assert o["params"].get("reduceOnly") is True
        assert o["type"] == "STOP_MARKET"
        assert o["side"] == "buy"      # buy closes a short


def test_initial_stop_at_configured_roi():
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    expected = price_for_roi(pos, -g.cfg.initial_stop_roi)
    got = float(fake.created[0]["params"]["stopPrice"])
    assert got == pytest.approx(expected, rel=1e-3)
    assert got > pos.entry_price      # for a short, the stop sits above entry


# ── entry orders must never be cancelled ─────────────────────────────────────

def test_trailing_stop_entry_order_never_cancelled():
    """
    A resting non-reduce-only trailing-stop ENTRY order must be invisible to the
    guardian — cancelling it would destroy the operator's entry method.
    """
    entry_order = {"id": "ENTRY-1", "symbol": "DOGE/USDT:USDT", "side": "sell",
                   "type": "TRAILING_STOP_MARKET", "reduceOnly": False,
                   "stopPrice": 99.0}
    fake = FakeExchange(positions=[_raw_pos("short")], orders=[entry_order],
                        price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    assert ("ENTRY-1", "DOGE/USDT:USDT") not in fake.cancelled
    assert all(oid != "ENTRY-1" for oid, _ in fake.cancelled)


# ── adoption ─────────────────────────────────────────────────────────────────

def test_adopts_existing_protective_stop_without_duplicating():
    pos = FuturesPosition("DOGE/USDT:USDT", "short", 100.0, 10.0, 10, 100.0)
    existing = {"id": "OLD-1", "symbol": "DOGE/USDT:USDT", "side": "buy",
                "type": "STOP_MARKET", "reduceOnly": True,
                "stopPrice": price_for_roi(pos, -7.0)}
    fake = FakeExchange(positions=[_raw_pos("short")], orders=[existing],
                        price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    # Stop already at the right level -> nothing new placed
    assert fake.created == []
    assert g._states["DOGE/USDT:USDT"].stop_order_id == "OLD-1"


# ── place-then-cancel ordering ───────────────────────────────────────────────

def test_replacement_places_before_cancelling():
    """The new stop must be resting before the old one is removed."""
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()                       # initial stop
    first_id = fake.created[0]["id"]

    # Price falls -> short is deep in profit -> trail arms and ratchets
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 40.0)
    g.run_cycle()

    seq = [k for k, _ in fake.call_log]
    assert "create" in seq and "cancel" in seq
    # The replacement create must precede the cancel of the superseded order
    last_create = max(i for i, (k, _) in enumerate(fake.call_log) if k == "create")
    cancel_idx = min(i for i, (k, _) in enumerate(fake.call_log) if k == "cancel")
    assert last_create < cancel_idx
    assert fake.cancelled[0][0] == first_id


def test_failed_placement_leaves_old_stop_intact():
    """If placing the new stop fails, the old one must NOT be cancelled."""
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    old_id = fake.created[0]["id"]

    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 40.0)
    fake.fail_next_create = True
    g.run_cycle()

    assert fake.cancelled == [], "old stop must survive a failed replacement"
    assert g._states["DOGE/USDT:USDT"].stop_order_id == old_id


# ── dry run ──────────────────────────────────────────────────────────────────

def test_dry_run_sends_nothing():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake, dry_run=True)
    g.run_cycle()
    assert fake.created == []
    assert fake.cancelled == []
    # but it still tracks what it would have done
    assert g._states["DOGE/USDT:USDT"].stop_roi == pytest.approx(-7.0)


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_state_cleared_when_position_closes():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    assert "DOGE/USDT:USDT" in g._states
    fake._positions = []               # position closed externally
    g.run_cycle()
    assert "DOGE/USDT:USDT" not in g._states


def test_trailing_ratchets_over_cycles():
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    pos = g.fetch_positions()[0]

    fake._price = price_for_roi(pos, 20.0)
    g.run_cycle()
    assert g._states[pos.symbol].armed
    assert g._states[pos.symbol].stop_roi == pytest.approx(10.0)

    fake._price = price_for_roi(pos, 45.0)
    g.run_cycle()
    assert g._states[pos.symbol].stop_roi == pytest.approx(35.0)

    # Reversal must NOT loosen the stop
    fake._price = price_for_roi(pos, 5.0)
    before = g._states[pos.symbol].stop_roi
    g.run_cycle()
    assert g._states[pos.symbol].stop_roi == pytest.approx(before)


def test_long_position_stop_below_entry():
    fake = FakeExchange(positions=[_raw_pos("long", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    stop = float(fake.created[0]["params"]["stopPrice"])
    assert stop < 100.0
    assert fake.created[0]["side"] == "sell"   # sell closes a long


# ── Regression: leverage must be derived, not trusted ────────────────────────

def test_leverage_derived_when_field_reports_one():
    """
    Binance returned leverage=1 on an isolated 10x position, which made every
    ROI 10x too small and every stop 10x too far away (a -7% ROI stop became a
    -7% PRICE move = -70% of margin). Leverage must be derived from
    notional/margin, not taken from the field.
    """
    raw = {"symbol": "X/USDT:USDT", "side": "long", "entryPrice": 0.1189,
           "contracts": 420.5, "leverage": 1,          # ← wrong, as observed
           "initialMargin": 5.0}                        # 50 notional / 5 = 10x
    g = _guardian(FakeExchange(positions=[raw]))
    pos = g.fetch_positions()[0]
    assert pos.effective_leverage == pytest.approx(10.0, rel=1e-3)


def test_stop_is_proportionate_to_margin_not_size():
    """-7% ROI must risk 7% of MARGIN, not 7% of position size."""
    raw = {"symbol": "X/USDT:USDT", "side": "long", "entryPrice": 0.1189,
           "contracts": 420.5, "leverage": 1, "initialMargin": 5.0}
    fake = FakeExchange(positions=[raw], price=0.1189)
    g = _guardian(fake)
    g.run_cycle()
    stop = float(fake.created[0]["params"]["stopPrice"])
    pos = g.fetch_positions()[0]
    loss = (stop - pos.entry_price) / pos.entry_price * pos.notional
    assert loss == pytest.approx(-0.07 * pos.margin, rel=1e-2)   # 7% of margin
    # and NOT 7% of the price / notional
    assert abs(loss) < 0.1 * pos.margin


def test_roi_matches_binance_convention():
    """A 2% price move at 10x must read as 20% ROI, not 2%."""
    from bot.futures_guard import roi_pct
    raw = {"symbol": "X/USDT:USDT", "side": "long", "entryPrice": 100.0,
           "contracts": 0.5, "leverage": 1, "initialMargin": 5.0}  # 50/5 = 10x
    g = _guardian(FakeExchange(positions=[raw]))
    pos = g.fetch_positions()[0]
    assert roi_pct(pos, 102.0) == pytest.approx(20.0)


def test_margin_reconstructed_when_missing():
    raw = {"symbol": "X/USDT:USDT", "side": "short", "entryPrice": 100.0,
           "contracts": 1.0, "leverage": 10}          # no margin field
    g = _guardian(FakeExchange(positions=[raw]))
    pos = g.fetch_positions()[0]
    assert pos.margin == pytest.approx(10.0)          # 100 notional / 10x
    assert pos.effective_leverage == pytest.approx(10.0)
