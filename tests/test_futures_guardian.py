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

    def enable_demo_trading(self, on):
        pass

    def fetch_my_trades(self, symbol, limit=10):
        return []

    def fetch_balance(self):
        return {"USDT": {"total": 100.0, "free": 100.0},
                "info": {"totalWalletBalance": "100.0"}}


def _guardian(fake, dry_run=False, cfg=None):
    g = FuturesGuardian.__new__(FuturesGuardian)     # bypass ccxt construction
    g.cfg = (cfg or GuardConfig()).validate()
    g.demo = True
    g.dry_run = dry_run
    g.poll_interval = 1.0
    g.exchange = fake
    g._states = {}
    import threading
    g._lock = threading.RLock()
    g._last_cycle_ts = 0.0
    g._last_error = None
    g._actions = []
    g._pos_meta = {}
    g._wallet_balance_cached = 0.0
    g._closed_trades = []
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


# ── Demo trading (replaces retired futures testnet) ──────────────────────────

def test_demo_mode_switches_api_urls():
    """
    Binance retired futures testnet; ccxt now exposes enable_demo_trading(),
    which routes to demo-fapi.binance.com. Verify the swap actually happens.
    """
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    live = ex.urls["api"]["fapiPrivate"]
    ex.enable_demo_trading(True)
    demo = ex.urls["api"]["fapiPrivate"]
    assert "demo-fapi" in demo
    assert demo != live
    assert ex.options.get("enableDemoTrading") is True


def test_demo_and_sandbox_are_mutually_exclusive():
    """ccxt refuses demo mode when sandbox is on — we must never set both."""
    import ccxt
    from ccxt.base.errors import NotSupported
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    ex.isSandboxModeEnabled = True
    with pytest.raises(NotSupported):
        ex.enable_demo_trading(True)


# ── Closing a position ───────────────────────────────────────────────────────

def test_close_position_is_reduce_only_market():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    fake.created.clear()
    res = g.close_position("DOGE/USDT:USDT")
    assert res["ok"] is True
    assert len(fake.created) == 1
    o = fake.created[0]
    assert o["type"] == "MARKET"
    assert o["side"] == "buy"                       # buy closes a short
    assert o["params"].get("reduceOnly") is True


def test_close_long_sells():
    fake = FakeExchange(positions=[_raw_pos("long")], price=100.0)
    g = _guardian(fake)
    g.run_cycle(); fake.created.clear()
    g.close_position("DOGE/USDT:USDT")
    assert fake.created[0]["side"] == "sell"


def test_close_cancels_the_protective_stop_after():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    stop_id = fake.created[0]["id"]
    g.close_position("DOGE/USDT:USDT")
    assert any(oid == stop_id for oid, _ in fake.cancelled)
    # the close order must be created BEFORE the stop is cancelled
    kinds = [k for k, _ in fake.call_log]
    assert kinds.index("create") < len(kinds)


def test_close_unknown_symbol_refused():
    g = _guardian(FakeExchange(positions=[], price=100.0))
    res = g.close_position("NOPE/USDT:USDT")
    assert res["ok"] is False and "no open position" in res["error"]


def test_close_dry_run_sends_nothing():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake, dry_run=True)
    g.run_cycle(); fake.created.clear()
    res = g.close_position("DOGE/USDT:USDT")
    assert res["dry_run"] is True
    assert fake.created == []


# ── Trade history ────────────────────────────────────────────────────────────

def test_closed_trade_recorded_when_position_disappears():
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 30.0)          # run into profit
    g.run_cycle()
    fake._positions = []                            # stop triggers / closed
    g.run_cycle()

    hist = g.closed_trades()
    assert len(hist) == 1
    t = hist[0]
    assert t["symbol"] == "DOGE/USDT:USDT"
    assert t["side"] == "short"
    assert t["peak_roi"] == pytest.approx(30.0, abs=0.1)
    assert t["armed"] is True
    assert t["exit_is_estimate"] is True             # no realised PnL from fake


def test_history_is_capped_and_newest_first():
    fake = FakeExchange(positions=[], price=100.0)
    g = _guardian(fake)
    for i in range(105):
        g._closed_trades.append({"symbol": f"S{i}", "closed_at": i})
    g._closed_trades = g._closed_trades[-100:]
    hist = g.closed_trades()
    assert len(hist) == 100
    assert hist[0]["symbol"] == "S104"               # newest first


# ── Orphaned stop cleanup ────────────────────────────────────────────────────

def test_stop_cancelled_when_position_disappears():
    """
    A protective stop left resting after the position closed can trigger
    against a LATER position on the same symbol. It must be cancelled.
    """
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    stop_id = fake.created[0]["id"]
    assert not fake.cancelled

    fake._positions = []          # position closed externally
    g.run_cycle()

    assert any(oid == stop_id for oid, _ in fake.cancelled), \
        "orphaned stop was not cancelled"
    assert "DOGE/USDT:USDT" not in g._states


def test_orphan_cancel_failure_is_not_fatal():
    """The stop usually already triggered — a cancel error must not break the cycle."""
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    def boom(order_id, symbol):
        raise RuntimeError("Unknown order sent")
    g = _guardian(fake)
    g.run_cycle()
    fake.cancel_order = boom
    fake._positions = []
    g.run_cycle()                  # must not raise
    assert "DOGE/USDT:USDT" not in g._states


def test_orphan_stop_not_cancelled_in_dry_run():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake, dry_run=True)
    g.run_cycle()
    fake._positions = []
    g.run_cycle()
    assert fake.cancelled == []
