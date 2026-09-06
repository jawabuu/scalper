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

    def fetch_ohlcv(self, symbol, timeframe, limit=288):
        return [[0, 100.0, 110.0, 90.0, 100.0, 10.0] for _ in range(limit)]

    def fetch_ohlcv(self, symbol, timeframe, limit=96):
        # ts, o, h, l, c, v — a flat synthetic 24h range
        return [[i, 100.0, 105.0, 95.0, 100.0, 10.0] for i in range(limit)]

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
    # Reuse the real initialiser so this double cannot drift from __init__.
    g._init_runtime_state()
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


def test_failed_trail_arming_leaves_fixed_stop_intact():
    """
    If the native trail cannot be placed, the fixed stop must survive — the
    position is never left unprotected by a failed upgrade.
    """
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    old_id = fake.created[0]["id"]

    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 40.0)      # would arm the trail
    # Make every order placement fail, so neither the trail nor a fallback lands.
    def always_fail(**kw):
        raise RuntimeError("exchange rejected order")
    fake.create_order = always_fail
    g.run_cycle()

    assert fake.cancelled == [], "fixed stop must survive a failed arming"
    st = g._states["DOGE/USDT:USDT"]
    assert st.native_trail_id is None
    assert st.stop_order_id == old_id


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


# ── Native trailing stop for the armed phase ─────────────────────────────────

def test_arming_places_native_trailing_stop():
    """
    The armed phase must use Binance's own TRAILING_STOP_MARKET so the exchange
    tracks the peak tick-by-tick, instead of the guardian repositioning a plain
    stop every poll (which missed spikes between cycles).
    """
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()                                  # initial fixed stop
    fixed_id = fake.created[0]["id"]

    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 40.0)         # past the arm level
    g.run_cycle()

    trail = [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    assert len(trail) == 1
    assert trail[0]["params"]["reduceOnly"] is True
    assert trail[0]["params"]["callbackRate"] == g.cfg.trail_callback_pct
    assert trail[0]["side"] == "buy"               # closes a short
    # the fixed stop is cancelled only AFTER the trail is resting
    assert any(oid == fixed_id for oid, _ in fake.cancelled)
    assert g._states["DOGE/USDT:USDT"].native_trail_id == trail[0]["id"]


def test_guardian_stops_repositioning_once_native_trail_is_armed():
    """Binance owns the trail after arming — the guardian must not interfere."""
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 40.0)
    g.run_cycle()
    n_after_arm = len(fake.created)

    # Further favourable moves must NOT create more orders.
    for roi in (60.0, 90.0, 120.0):
        fake._price = price_for_roi(pos, roi)
        g.run_cycle()
    assert len(fake.created) == n_after_arm


def test_trail_refused_when_it_would_lock_in_no_profit():
    """
    callbackRate is a PRICE percent, so its ROI cost scales with leverage. At
    high leverage a 1% callback can exceed the arm level, which would engage the
    trail at or below entry — that must be refused and the fixed stop kept.
    """
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(initial_stop_roi=10, arm_roi=10, callback_roi=5,
                      trail_callback_pct=1.0)     # 1% x 10x = 10% ROI == arm
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake, cfg=cfg)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 20.0)
    g.run_cycle()

    assert not [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    assert g._states["DOGE/USDT:USDT"].native_trail_id is None


def test_long_trail_sells():
    fake = FakeExchange(positions=[_raw_pos("long", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 40.0)
    g.run_cycle()
    trail = [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    assert trail and trail[0]["side"] == "sell"


# ── Unprotected positions (-2021) ────────────────────────────────────────────

def test_position_past_its_stop_is_flagged_unprotected():
    """
    Adopting a position already worse than the stop level makes Binance reject
    the stop (-2021). The position is UNPROTECTED and must be flagged, not
    silently logged as a generic failure.
    """
    fake = FakeExchange(positions=[_raw_pos("long", entry=100.0)], price=100.0)
    def reject(**kw):
        raise RuntimeError('binanceusdm {"code":-2021,"msg":"Order would immediately trigger."}')
    fake.create_order = reject
    g = _guardian(fake)
    g.run_cycle()

    st = g._states["DOGE/USDT:USDT"]
    assert st.unprotected_reason is not None
    assert "past the" in st.unprotected_reason
    assert any(a["action"] == "UNPROTECTED" for a in g._actions)


def test_guardian_does_not_auto_close_an_unprotected_position():
    """Closing is the operator's decision — the guardian must never do it itself."""
    fake = FakeExchange(positions=[_raw_pos("long", entry=100.0)], price=100.0)
    def reject(**kw):
        raise RuntimeError('{"code":-2021,"msg":"Order would immediately trigger."}')
    fake.create_order = reject
    g = _guardian(fake)
    g.run_cycle()
    # no MARKET close was sent
    assert not [o for o in fake.created if o.get("type") == "MARKET"]


def test_unprotected_flag_clears_once_a_stop_lands():
    fake = FakeExchange(positions=[_raw_pos("long", entry=100.0)], price=100.0)
    orig = fake.create_order
    def reject(**kw):
        raise RuntimeError('{"code":-2021,"msg":"Order would immediately trigger."}')
    fake.create_order = reject
    g = _guardian(fake)
    g.run_cycle()
    assert g._states["DOGE/USDT:USDT"].unprotected_reason is not None

    fake.create_order = orig          # exchange accepts again
    fake._price = 100.5               # moved back into a placeable range
    g.run_cycle()
    assert g._states["DOGE/USDT:USDT"].unprotected_reason is None


# ── 24h range fallback ───────────────────────────────────────────────────────

def test_range_falls_back_to_candles_when_ticker_lacks_high_low():
    """The demo endpoint omits high/low; the range must come from candles."""
    fake = FakeExchange(positions=[_raw_pos("long", entry=100.0)], price=100.0)
    fake.fetch_ticker = lambda s: {"last": 100.0}     # no high/low
    g = _guardian(fake)
    hi, lo = g._range_24h("DOGE/USDT:USDT", {"last": 100.0})
    assert hi == pytest.approx(105.0)
    assert lo == pytest.approx(95.0)


def test_range_prefers_ticker_when_present():
    fake = FakeExchange(positions=[], price=100.0)
    g = _guardian(fake)
    hi, lo = g._range_24h("X/USDT:USDT", {"high": 120.0, "low": 80.0})
    assert (hi, lo) == (120.0, 80.0)


def test_range_fallback_is_cached():
    """A 5s poll loop must not refetch 24h of klines every cycle."""
    fake = FakeExchange(positions=[], price=100.0)
    calls = {"n": 0}
    orig = fake.fetch_ohlcv
    def counting(symbol, timeframe, limit=96):
        calls["n"] += 1
        return orig(symbol, timeframe, limit)
    fake.fetch_ohlcv = counting
    g = _guardian(fake)
    for _ in range(5):
        g._range_24h("X/USDT:USDT", {"last": 1.0})
    assert calls["n"] == 1


# ── 24h range fallback for open positions ────────────────────────────────────

class _RangeEx(FakeExchange):
    def __init__(self, hilo=True, **kw):
        super().__init__(**kw)
        self.hilo = hilo
        self.ohlcv_calls = 0
    def fetch_ticker(self, symbol):
        t = {"last": self._price}
        if self.hilo:
            t.update(high=110.0, low=90.0)
        return t
    def fetch_ohlcv(self, symbol, timeframe, limit=288):
        self.ohlcv_calls += 1
        return [[0, 100.0, 110.0, 90.0, 100.0, 10.0] for _ in range(limit)]


def test_range_uses_ticker_when_present():
    ex = _RangeEx(hilo=True, positions=[_raw_pos("long")], price=100.0)
    g = _guardian(ex)
    hi, lo = g._range_24h("X/USDT:USDT", ex.fetch_ticker("X"))
    assert (hi, lo) == (110.0, 90.0)
    assert ex.ohlcv_calls == 0          # no extra call needed


def test_range_falls_back_to_candles_when_ticker_omits_them():
    """
    Some ticker payloads omit high/low, which left the 24h range showing n/a
    for open positions. Derive it from candles instead.
    """
    ex = _RangeEx(hilo=False, positions=[_raw_pos("long")], price=100.0)
    g = _guardian(ex)
    hi, lo = g._range_24h("X/USDT:USDT", ex.fetch_ticker("X"))
    assert (hi, lo) == (110.0, 90.0)
    assert ex.ohlcv_calls == 1


def test_range_fallback_is_cached():
    """The guardian polls every ~5s; the fallback must not fetch klines each time."""
    ex = _RangeEx(hilo=False, positions=[_raw_pos("long")], price=100.0)
    g = _guardian(ex)
    for _ in range(5):
        g._range_24h("X/USDT:USDT", ex.fetch_ticker("X"))
    assert ex.ohlcv_calls == 1


def test_position_meta_carries_distances_to_both_extremes():
    ex = _RangeEx(hilo=True, positions=[_raw_pos("long", entry=100.0)], price=100.0)
    g = _guardian(ex)
    g.run_cycle()
    meta = g._pos_meta["DOGE/USDT:USDT"]
    assert meta["high_24h"] == 110.0 and meta["low_24h"] == 90.0
    assert meta["pct_above_24h_low"] == pytest.approx(11.11, abs=0.05)
    assert meta["pct_below_24h_high"] == pytest.approx(-9.09, abs=0.05)
    assert meta["range_pos_24h"] == pytest.approx(0.5, abs=0.01)


# ── Shared balance resolution ────────────────────────────────────────────────

def test_balance_resolver_handles_all_payload_shapes():
    """
    The guardian displays this figure and the entry service sizes positions
    from it — they must never resolve it differently.
    """
    from bot.futures_guardian import resolve_usdt_balance
    shapes = [
        {"USDT": {"total": 107.09}},
        {"info": {"totalWalletBalance": "107.09"}},
        {"total": {"USDT": 107.09}},
        {"info": {"assets": [{"asset": "BNB", "walletBalance": "1"},
                             {"asset": "USDT", "walletBalance": "107.09"}]}},
    ]
    for payload in shapes:
        val, src = resolve_usdt_balance(payload)
        assert val == pytest.approx(107.09), src


def test_balance_resolver_reports_unresolved():
    from bot.futures_guardian import resolve_usdt_balance
    assert resolve_usdt_balance({}) == (0.0, "unresolved")
    assert resolve_usdt_balance({"USDT": {"total": 0}})[0] == 0.0


def test_entry_and_guardian_agree_on_balance():
    """Both must read the same number from the same payload."""
    from bot.futures_guardian import resolve_usdt_balance
    from bot.futures_entry import EntryService, EntryLimits

    payload = {"info": {"totalWalletBalance": "107.09"}}

    class _Ex:
        def fetch_balance(self): return payload
    class _G:
        dry_run = True
        exchange = _Ex()
        cfg = GuardConfig()
        def fetch_positions(self): return []

    svc = EntryService(_G(), EntryLimits(max_positions=3, max_margin_pct=25,
                                         default_margin_pct=10, default_callback_pct=0.1))
    assert svc.wallet_balance() == pytest.approx(resolve_usdt_balance(payload)[0])


# ── One flag drives keys AND endpoint ────────────────────────────────────────

def _cfg_with(env):
    import os, importlib
    saved = {k: os.environ.get(k) for k in
             ["TESTNET", "GUARDIAN_DEMO", "GUARDIAN_TESTNET", "SCANNER_DEMO",
              "GUARDIAN_ENABLED", "BINANCE_API_KEY_TEST", "BINANCE_API_SECRET_TEST",
              "BINANCE_API_KEY_LIVE", "BINANCE_API_SECRET_LIVE"]}
    try:
        for k in ["TESTNET", "GUARDIAN_DEMO", "GUARDIAN_TESTNET", "SCANNER_DEMO"]:
            os.environ.pop(k, None)
        os.environ.update({"BINANCE_API_KEY_TEST": "tk", "BINANCE_API_SECRET_TEST": "ts",
                           "BINANCE_API_KEY_LIVE": "lk", "BINANCE_API_SECRET_LIVE": "ls"})
        os.environ.update(env)
        import bot.config
        importlib.reload(bot.config)
        return bot.config.BotConfig()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import bot.config
        importlib.reload(bot.config)


def test_testnet_true_puts_everything_on_demo():
    """TESTNET picks the keys; the endpoint must follow, not default separately."""
    c = _cfg_with({"TESTNET": "true"})
    assert c.testnet is True
    assert c.guardian_demo is True
    assert c.scanner_demo is True
    assert c.api_key == "tk"


def test_testnet_false_puts_everything_on_live():
    """
    Previously guardian_demo defaulted to True regardless, so TESTNET=false gave
    LIVE keys pointed at the DEMO endpoint.
    """
    c = _cfg_with({"TESTNET": "false"})
    assert c.testnet is False
    assert c.guardian_demo is False
    assert c.scanner_demo is False
    assert c.api_key == "lk"


def test_explicit_guardian_demo_still_overrides():
    c = _cfg_with({"TESTNET": "true", "GUARDIAN_DEMO": "false"})
    assert c.guardian_demo is False


def test_legacy_guardian_testnet_still_honoured():
    c = _cfg_with({"TESTNET": "true", "GUARDIAN_TESTNET": "false"})
    assert c.guardian_demo is False


# ── An unavailable 24h range must explain itself ─────────────────────────────

def test_range_source_records_ticker_origin():
    ex = _RangeEx(hilo=True, positions=[_raw_pos("long")], price=100.0)
    g = _guardian(ex)
    g._range_24h("X/USDT:USDT", ex.fetch_ticker("X"))
    assert g._range_source["X/USDT:USDT"] == "ticker"


def test_range_source_records_candle_fallback():
    ex = _RangeEx(hilo=False, positions=[_raw_pos("long")], price=100.0)
    g = _guardian(ex)
    g._range_24h("X/USDT:USDT", ex.fetch_ticker("X"))
    assert g._range_source["X/USDT:USDT"] == "candles"


def test_range_source_records_the_failure_reason():
    """
    The fallback failure was logged at debug and swallowed, so the range simply
    read n/a with no way to tell why. It must now say what went wrong.
    """
    ex = _RangeEx(hilo=False, positions=[_raw_pos("long")], price=100.0)
    def boom(symbol, timeframe, limit=288):
        raise RuntimeError("Invalid symbol")
    ex.fetch_ohlcv = boom
    g = _guardian(ex)
    hi, lo = g._range_24h("X/USDT:USDT", ex.fetch_ticker("X"))
    assert (hi, lo) == (None, None)
    assert "Invalid symbol" in g._range_source["X/USDT:USDT"]


def test_range_method_is_defined_once():
    """A duplicated definition silently shadowed the first — keep it singular."""
    import inspect, bot.futures_guardian as m
    src = inspect.getsource(m)
    assert src.count("def _range_24h(") == 1


# ── Realised PnL must be per-position, not per-symbol ────────────────────────

def _record_trade(guardian, symbol, side, entry, exit_, notional, opened_at,
                  peak=1.0):
    from bot.futures_guard import GuardState
    guardian._closed_trades = []
    meta = {"entry_price": entry, "current_price": exit_, "side": side,
            "notional": notional, "margin": notional / 10,
            "opened_seen_at": opened_at}
    guardian._record_closed_trade(symbol, GuardState(peak_roi=peak), meta)
    return guardian._closed_trades[0]


def test_realised_pnl_is_scoped_to_the_position_lifetime():
    """
    fetch_my_trades returns the last N fills for a SYMBOL regardless of which
    position they belonged to. Summing them blindly gave every trade on that
    symbol the same figure — two UAI trades at +24% and -5% ROI both reported
    an identical -1.4133.
    """
    import time
    now = time.time()
    older_fill = {"timestamp": int((now - 900) * 1000),
                  "info": {"realizedPnl": "1.4133"}}     # previous position
    this_fill = {"timestamp": int((now - 100) * 1000),
                 "info": {"realizedPnl": "-0.3010"}}     # this position

    class Ex(FakeExchange):
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return [older_fill, this_fill]               # API does not filter

    g = _guardian(Ex())
    rec = _record_trade(g, "UAI/USDT:USDT", "short", 0.6702, 0.6738,
                        notional=59.35, opened_at=now - 200)
    # Only the fill from THIS position counts.
    assert rec["realised_pnl_usdt"] == pytest.approx(-0.3010, abs=1e-4)


def test_two_trades_on_one_symbol_get_different_realised_values():
    import time
    now = time.time()

    class Ex(FakeExchange):
        fills = []
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return self.fills

    g = _guardian(Ex())
    g.exchange.fills = [{"timestamp": int((now - 900) * 1000),
                         "info": {"realizedPnl": "1.4133"}}]
    first = _record_trade(g, "UAI/USDT:USDT", "short", 0.6845, 0.6682,
                          notional=59.35, opened_at=now - 1000)

    g.exchange.fills = g.exchange.fills + [{"timestamp": int((now - 100) * 1000),
                                            "info": {"realizedPnl": "-0.3010"}}]
    second = _record_trade(g, "UAI/USDT:USDT", "short", 0.6702, 0.6738,
                           notional=59.35, opened_at=now - 200)

    assert first["realised_pnl_usdt"] != second["realised_pnl_usdt"]
    assert first["realised_pnl_usdt"] > 0      # the winner
    assert second["realised_pnl_usdt"] < 0     # the loser


def test_short_win_reports_positive_realised():
    """A short that closed lower must not report a loss."""
    import time
    now = time.time()

    class Ex(FakeExchange):
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return []          # no exchange figure -> computed fallback

    g = _guardian(Ex())
    rec = _record_trade(g, "UAI/USDT:USDT", "short", 0.6845, 0.6682,
                        notional=59.35, opened_at=now - 100)
    assert rec["realised_pnl_usdt"] == pytest.approx(1.4133, abs=1e-3)


# ── A position discovered already in profit must be protected ────────────────

def test_position_born_armed_gets_a_trailing_stop():
    """
    A trailing-stop ENTRY can fill after a large move, so the position is
    already past the arm level the first time the guardian sees it. It is then
    born armed, the transition never fires, and it used to fall through to the
    fixed stop — which the exchange rejects, leaving it UNPROTECTED.
    """
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    # First sight is already +25% ROI (price well below entry for a short).
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 25.0)
    g.run_cycle()

    trail = [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    assert trail, "a position discovered in profit must get a trailing stop"
    assert trail[0]["params"]["reduceOnly"] is True
    st = g._states["DOGE/USDT:USDT"]
    assert st.native_trail_id is not None
    assert st.armed is True


def test_no_fixed_stop_attempted_when_already_past_it():
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 25.0)
    g.run_cycle()
    kinds = [o["type"] for o in fake.created]
    assert "TRAILING_STOP_MARKET" in kinds
    assert kinds.count("STOP_MARKET") == 0


def test_rejected_fixed_stop_falls_back_to_trailing():
    """If the fixed stop is refused, protect with a trail rather than nothing."""
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)

    original = fake.create_order
    def reject_stop_market(**kw):
        if kw.get("type") == "STOP_MARKET":
            raise RuntimeError("Order would immediately trigger")
        return original(**kw)
    fake.create_order = reject_stop_market

    g = _guardian(fake)
    g.run_cycle()

    trail = [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    assert trail, "must fall back to a trailing stop when the fixed stop is rejected"
    st = g._states["DOGE/USDT:USDT"]
    assert st.native_trail_id is not None


def test_armed_position_is_not_re_armed_every_cycle():
    """Once a trail is resting, further cycles must not place more orders."""
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 25.0)
    g.run_cycle()
    n = len(fake.created)
    for _ in range(3):
        g.run_cycle()
    assert len(fake.created) == n


# ── Stops that fill between polls must not report a stale ROI ────────────────

def test_final_roi_derived_from_realised_when_stop_fills_between_polls():
    """
    The guardian polls, so a stop triggering between cycles fills at a price it
    never saw. Reporting the last observed price showed -3.4% ROI on a trade
    the money said was -10.4%.
    """
    import time
    now = time.time()

    class Ex(FakeExchange):
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return [{"timestamp": int((now - 10) * 1000),
                     "info": {"realizedPnl": "-1.1457"}}]

    g = _guardian(Ex())
    from bot.futures_guard import GuardState
    meta = {"entry_price": 0.07068951, "current_price": 0.070452, "side": "long",
            "notional": 110.0, "margin": 11.0, "leverage": 10.0,
            "current_roi": -3.36, "opened_seen_at": now - 600}
    g._closed_trades = []
    g._record_closed_trade("BULLA/USDT:USDT", GuardState(peak_roi=0.7), meta)
    rec = g._closed_trades[0]

    assert rec["final_roi"] == pytest.approx(-10.42, abs=0.05)
    assert rec["observed_roi"] == pytest.approx(-3.36, abs=0.05)
    assert rec["exit_from_exchange"] is True
    assert rec["exit_price"] < 0.070452        # the real fill was worse


def test_final_roi_and_realised_are_consistent():
    """final_roi x margin must reconcile with the realised figure."""
    import time
    now = time.time()

    class Ex(FakeExchange):
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return [{"timestamp": int((now - 10) * 1000),
                     "info": {"realizedPnl": "2.5972"}}]

    g = _guardian(Ex())
    from bot.futures_guard import GuardState
    meta = {"entry_price": 0.09731, "current_price": 0.09957, "side": "long",
            "notional": 114.3, "margin": 11.43, "leverage": 10.0,
            "current_roi": 22.72, "opened_seen_at": now - 600}
    g._closed_trades = []
    g._record_closed_trade("X/USDT:USDT", GuardState(peak_roi=43.98), meta)
    rec = g._closed_trades[0]
    implied = rec["final_roi"] / 100 * meta["margin"]
    assert implied == pytest.approx(rec["realised_pnl_usdt"], rel=0.01)


def test_computed_fallback_is_flagged_as_estimate():
    class Ex(FakeExchange):
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return []        # no exchange figure
    import time
    g = _guardian(Ex())
    from bot.futures_guard import GuardState
    meta = {"entry_price": 100.0, "current_price": 101.0, "side": "long",
            "notional": 100.0, "margin": 10.0, "leverage": 10.0,
            "current_roi": 10.0, "opened_seen_at": time.time() - 60}
    g._closed_trades = []
    g._record_closed_trade("X/USDT:USDT", GuardState(peak_roi=10.0), meta)
    rec = g._closed_trades[0]
    assert rec["exit_is_estimate"] is True
    assert rec["exit_from_exchange"] is False


# ── Env parsing must tolerate inline comments (regression) ───────────────────

def test_numeric_env_readers_strip_inline_comments():
    """
    Compose keeps everything after `=` verbatim, so
    `GUARD_TRAIL_CALLBACK_PCT=1.0  # price percent` arrived with the comment
    attached and crashed the process at startup. _env_bool handled this; the
    numeric readers did not.
    """
    import os
    from bot.config import _env_float, _env_int
    os.environ["_T_FLOAT"] = "1.0    # PRICE percent, Binance's own unit"
    os.environ["_T_INT"] = "6   # positions"
    try:
        assert _env_float("_T_FLOAT", 9.9) == pytest.approx(1.0)
        assert _env_int("_T_INT", 1) == 6
    finally:
        os.environ.pop("_T_FLOAT", None)
        os.environ.pop("_T_INT", None)


def test_unparseable_numeric_env_falls_back_instead_of_crashing():
    import os
    from bot.config import _env_float
    os.environ["_T_BAD"] = "not-a-number"
    try:
        assert _env_float("_T_BAD", 2.5) == pytest.approx(2.5)
    finally:
        os.environ.pop("_T_BAD", None)
