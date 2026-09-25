"""
Tests for the futures guardian exchange layer.

Uses a fake exchange so the safety properties are verified as behaviour, not
assumed: reduce-only on every order, place-then-cancel ordering, and never
cancelling a non-reduce-only (entry) order.
"""
import time
import pytest

from dataclasses import replace
from bot.futures_guard import GuardConfig, GuardState, price_for_roi, FuturesPosition
from bot.futures_guardian import FuturesGuardian


class FakeExchange:
    """Minimal stand-in recording every call the guardian makes."""

    def __init__(self, positions=None, orders=None, price=100.0):
        self._positions = positions or []
        self._orders = orders or []
        self._algo = []
        # Set True to have the NEXT trailing stop return an id without resting.
        self.reject_next_algo = False
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
        # Conditional and trailing stops rest in the ALGO book, not the
        # unified one. The fake has to model that: the guardian now CONFIRMS
        # an armed trail is resting there before cancelling live protection,
        # because the algo endpoint returns 200 with an id and rejects the
        # order asynchronously (牛来 2026-09-20). A fake with no algo book
        # would make every confirmation fail and hide the behaviour under test.
        if self.reject_next_algo and str(type).upper() == "TRAILING_STOP_MARKET":
            # Binance accepts the CALL and refuses the ORDER: an id comes
            # back, nothing rests. This is the 牛来 2026-09-20 shape.
            self.reject_next_algo = False
            return rec
        if str(type).upper() in ("TRAILING_STOP_MARKET", "STOP_MARKET",
                                 "TAKE_PROFIT_MARKET"):
            self._algo.append({
                "algoId": oid,
                "symbol": str(symbol).split(":")[0].replace("/", "").upper(),
                "algoType": "CONDITIONAL", "type": type, "side": side})
        return rec

    def fapiPrivateGetOpenAlgoOrders(self, params=None):
        return list(self._algo)

    def reject_algo(self, order_id):
        """Simulate Binance accepting the CALL and refusing the ORDER."""
        self._algo = [o for o in self._algo
                      if str(o.get("algoId")) != str(order_id)]

    def cancel_order(self, order_id, symbol):
        self._algo = [o for o in self._algo
                      if str(o.get("algoId")) != str(order_id)]
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


def _guardian(fake, dry_run=False, cfg=None, adaptive=False,
              arm_at_entry=False, vol_floor=False):
    """
    These tests exercise the FIXED-STOP mechanics: placement, ratcheting,
    replacement, arming, the refusal fallback. The adaptive trail (v3.15.0,
    default ON in production) places an order BEFORE any of that and changes
    which id each step receives, so it is disabled here unless a test asks for
    it. Adaptive-trail behaviour has its own tests in test_breakout_veto.py —
    placement, supersession by arming, the skipped rescue, sweep exemption and
    the failure path.

    Arm-at-entry (v3.47.0, default ON in production) is disabled for the same
    reason and by the same rule: it places the armed trail at ADOPTION, before
    the fixed stop, so `created[0]` is a TRAILING_STOP_MARKET and the poll-arm
    block downstream becomes a no-op. Turning it on here would not test more,
    it would test something else — these are the poll-arm path's tests, and
    that path still runs whenever GUARD_ARM_AT_ENTRY is false or the arm level
    cannot be derived. Arm-at-entry has its own coverage below and in
    test_breakout_veto.py.
    """
    g = FuturesGuardian.__new__(FuturesGuardian)     # bypass ccxt construction
    cfg = cfg or GuardConfig()
    if not adaptive:
        cfg = replace(cfg, adaptive_trail_enabled=False)
    if not arm_at_entry:
        cfg = replace(cfg, arm_at_entry=False)
    if not vol_floor:
        # The volatility floor (v3.72.0, default ON in production) widens the
        # armed callback to sit outside the coin's movement, which changes
        # both the callback sent and WHEN a position arms. Disabled here for
        # the same reason as the two above: these are the fixed-stop path's
        # tests, and the fake exchange's synthetic candles produce a ~10% ATR
        # that no real coin carries. Floor behaviour has its own tests in
        # test_futures_guard.py.
        cfg = replace(cfg, trail_callback_atr_mult=0.0)
    g.cfg = cfg.validate()
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
    for _ in range(g.MISSING_CONFIRMATIONS):
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
    for _ in range(g.MISSING_CONFIRMATIONS):
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
    for _ in range(g.MISSING_CONFIRMATIONS):
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
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()              # must not raise
    assert "DOGE/USDT:USDT" not in g._states


def test_orphan_stop_not_cancelled_in_dry_run():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake, dry_run=True)
    g.run_cycle()
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
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
    high leverage a 1% callback can exceed the arm level, which would engage
    the trail at or below entry — that must be refused and the fixed stop kept.

    v3.72.0: the give-back is now measured against the position's ACTUAL peak
    rather than cfg.arm_roi. At a peak sitting exactly on the arm level the
    refusal still holds, which is what this pins.
    """
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(initial_stop_roi=10, arm_roi=10, callback_roi=5,
                      trail_callback_pct=1.0)     # 1% x 10x = 10% ROI == arm
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake, cfg=cfg)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 10.0)          # peak == arm level
    g.run_cycle()

    assert not [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    assert g._states["DOGE/USDT:USDT"].native_trail_id is None


def test_a_peak_well_past_the_arm_level_DOES_get_its_trail():
    """
    The counterpart, and a behaviour change. The old code measured the lock
    against cfg.arm_roi, so a position that had run to +20% ROI was refused a
    trail that would have locked in +10% — it judged against the arm level
    instead of the profit actually in hand. Measuring against the peak fixes
    that, and it is the same change that lets a volatility-floored callback
    arm once the position has earned the room.
    """
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(initial_stop_roi=10, arm_roi=10, callback_roi=5,
                      trail_callback_pct=1.0)     # 10% ROI give-back at 10x
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake, cfg=cfg)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 20.0)          # peak far past the arm
    g.run_cycle()

    assert [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"], \
        "a +20% peak with a 10% give-back locks in +10% — that trail is viable"


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


# ── Per-order quantity cap (-4005) ───────────────────────────────────────────

def test_oversized_stop_is_split_across_the_order_cap():
    """
    A low-priced coin needs a huge contract count, so a legitimate position can
    exceed Binance's per-ORDER cap. One oversized stop is rejected with -4005,
    leaving the position unprotected — split it instead.
    """
    class Ex(FakeExchange):
        def market(self, symbol):
            return {"limits": {"amount": {"max": 200000.0}}}

    fake = Ex(positions=[_raw_pos("short", entry=0.01303, contracts=652939.0)],
              price=0.01303)
    g = _guardian(fake)
    g.run_cycle()

    stops = [o for o in fake.created if o["type"] == "STOP_MARKET"]
    assert len(stops) >= 4                       # 652939 / 200000
    assert all(o["params"]["reduceOnly"] for o in stops)
    assert sum(o["amount"] for o in stops) == pytest.approx(652939.0, rel=0.01)


def test_normal_size_still_uses_a_single_stop():
    class Ex(FakeExchange):
        def market(self, symbol):
            return {"limits": {"amount": {"max": 200000.0}}}

    fake = Ex(positions=[_raw_pos("short", entry=100.0, contracts=10.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    assert len([o for o in fake.created if o["type"] == "STOP_MARKET"]) == 1


def test_missing_market_limits_does_not_block():
    class Ex(FakeExchange):
        def market(self, symbol):
            raise RuntimeError("no market info")
    fake = Ex(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    assert fake.created, "must still place a stop when limits are unknown"


# ── Leverage-independent trail callback ──────────────────────────────────────

def test_trail_callback_gives_same_roi_at_any_leverage():
    """
    A fixed PRICE callback gives a different ROI give-back at 10x than at 20x,
    so the same config behaved differently per account.
    """
    from bot.futures_guard import GuardConfig, trail_callback_price_pct
    cfg = GuardConfig(arm_roi=8.0, callback_roi=5.0, trail_callback_roi=5.0)
    for lev in (5, 10, 20):
        px = trail_callback_price_pct(lev, cfg)
        assert px * lev == pytest.approx(5.0, abs=0.6)


def test_the_volatility_floor_is_OFF_by_default():
    """
    v3.73.4. The floor was introduced in v3.72.0 from one trade (STG) and a
    mechanism argument, then tested against 395 historical trail exits and NOT
    supported: median outcome improves monotonically as the callback gets
    TIGHTER relative to ATR, and the loss rate is flat ~10% for tight
    callbacks against 26% for wide ones.

    Off by default so neither container needs the env var at all.
    """
    cfg = _cfg_with({})
    assert cfg.guard_trail_callback_atr_mult == 0.0


def test_the_floor_is_NOT_chained_to_the_entry_multiplier():
    """
    v3.72.1 defaulted this to AUTO_CALLBACK_ATR_MULT. That was a mistake: the
    entry multiplier is the independent variable of a running experiment with
    a pre-committed control, so chaining silently coupled an EXIT-side setting
    to it and gave the comparison three differences instead of one.
    """
    cfg = _cfg_with({"AUTO_CALLBACK_ATR_MULT": "1.25"})
    assert cfg.auto_callback_atr_mult == pytest.approx(1.25)
    assert cfg.guard_trail_callback_atr_mult == 0.0


def test_an_explicit_guardian_multiplier_still_wins():
    cfg = _cfg_with({"AUTO_CALLBACK_ATR_MULT": "1.25",
                     "GUARD_TRAIL_CALLBACK_ATR_MULT": "0.5"})
    assert cfg.guard_trail_callback_atr_mult == pytest.approx(0.5)


def test_the_floor_can_be_disabled_even_when_the_entry_path_uses_one():
    cfg = _cfg_with({"AUTO_CALLBACK_ATR_MULT": "1.25",
                     "GUARD_TRAIL_CALLBACK_ATR_MULT": "0"})
    assert cfg.guard_trail_callback_atr_mult == 0.0


def test_trail_callback_clamped_to_exchange_range():
    from bot.futures_guard import GuardConfig, trail_callback_price_pct
    cfg = GuardConfig(trail_callback_roi=5.0)
    assert trail_callback_price_pct(100, cfg) >= 0.1     # very high leverage
    cfg2 = GuardConfig(trail_callback_roi=200.0)
    assert trail_callback_price_pct(1, cfg2) <= 5.0      # very low leverage


def test_price_callback_still_honoured_when_roi_not_set():
    from bot.futures_guard import GuardConfig, trail_callback_price_pct
    cfg = GuardConfig(trail_callback_pct=1.0, trail_callback_roi=0.0)
    assert trail_callback_price_pct(10, cfg) == pytest.approx(1.0)


# ── Closing must also respect the per-order cap ──────────────────────────────

def test_close_splits_across_the_order_cap():
    """
    Closing sent the whole position as one MARKET order, which a large position
    on a low-priced coin exceeds (-4005) — so closing from the dashboard failed
    and had to be done on Binance.
    """
    class Ex(FakeExchange):
        def market(self, symbol):
            return {"limits": {"amount": {"max": 200000.0}}}

    fake = Ex(positions=[_raw_pos("short", entry=0.01303, contracts=652939.0)],
              price=0.01303)
    g = _guardian(fake)
    g.run_cycle()
    fake.created.clear()

    res = g.close_position("DOGE/USDT:USDT")
    assert res["ok"] is True
    closes = [o for o in fake.created if o["type"] == "MARKET"]
    assert len(closes) >= 4
    assert all(o["params"]["reduceOnly"] for o in closes)
    assert sum(o["amount"] for o in closes) == pytest.approx(652939.0, rel=0.01)


def test_close_reports_partial_when_a_chunk_fails():
    """A failure mid-close must not be reported as a clean close."""
    class Ex(FakeExchange):
        def market(self, symbol):
            return {"limits": {"amount": {"max": 200000.0}}}

    fake = Ex(positions=[_raw_pos("short", entry=0.01303, contracts=652939.0)],
              price=0.01303)
    g = _guardian(fake)
    g.run_cycle()
    fake.created.clear()

    calls = {"n": 0}
    original = fake.create_order
    def fail_third(**kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("rejected")
        return original(**kw)
    fake.create_order = fail_third

    res = g.close_position("DOGE/USDT:USDT")
    assert res["ok"] is False
    assert res["partially_closed_qty"] > 0


def test_normal_close_is_a_single_order():
    class Ex(FakeExchange):
        def market(self, symbol):
            return {"limits": {"amount": {"max": 200000.0}}}
    fake = Ex(positions=[_raw_pos("short", entry=100.0, contracts=10.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle(); fake.created.clear()
    g.close_position("DOGE/USDT:USDT")
    assert len([o for o in fake.created if o["type"] == "MARKET"]) == 1


# ── Sized stop must survive a delayed fill ───────────────────────────────────

def test_guardian_honours_the_stop_the_position_was_sized_for():
    """
    A trailing-stop ENTRY rests before filling, so the guardian first sees the
    position minutes after the entry sized it. Recomputing ATR then gave a very
    different stop — a position sized for 12.9% received 24.4%, nearly 2x the
    intended risk. Sharing the ATR cache only helps inside its TTL.
    """
    from bot.futures_guard import GuardConfig, FuturesPosition

    class VolatileEx(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, limit=96):
            p = 0.2628
            return [[0, p, p * 1.0065, p * 0.9935, p, 10.0] for _ in range(limit)]

    cfg = GuardConfig(initial_stop_roi=10, arm_roi=8, callback_roi=5,
                      atr_stop_mult=1.5, atr_stop_min_roi=4, atr_stop_max_roi=30)
    g = _guardian(VolatileEx(), cfg=cfg)
    pos = FuturesPosition("FORM/USDT:USDT", "long", 0.262799, 13570, 20, 178.29)

    recomputed = g.effective_stop_roi(pos)
    g.note_entry_context("FORM/USDT:USDT", {"sized_stop_roi": 12.9})
    honoured = g.effective_stop_roi(pos)

    assert recomputed != pytest.approx(12.9, abs=0.5)
    assert honoured == pytest.approx(12.9)


def test_risk_matches_budget_across_a_delayed_fill():
    from bot.futures_guard import GuardConfig, FuturesPosition

    class VolatileEx(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, limit=96):
            p = 0.2628
            return [[0, p, p * 1.0065, p * 0.9935, p, 10.0] for _ in range(limit)]

    cfg = GuardConfig(atr_stop_mult=1.5, atr_stop_min_roi=4, atr_stop_max_roi=30)
    g = _guardian(VolatileEx(), cfg=cfg)
    wallet, margin = 4341.0, 337.58
    g.note_entry_context("FORM/USDT:USDT", {"sized_stop_roi": 12.86})
    pos = FuturesPosition("FORM/USDT:USDT", "long", 0.262799, 13570, 20, margin)
    loss = margin * g.effective_stop_roi(pos) / 100
    assert loss == pytest.approx(wallet * 0.01, rel=0.03)


def test_falls_back_to_atr_when_no_sized_stop_recorded():
    """Manually opened positions have no sized stop — ATR still applies."""
    from bot.futures_guard import GuardConfig, FuturesPosition
    cfg = GuardConfig(atr_stop_mult=1.5, atr_stop_min_roi=4, atr_stop_max_roi=30)
    g = _guardian(FakeExchange(), cfg=cfg)
    pos = FuturesPosition("X/USDT:USDT", "short", 100.0, 10.0, 10, 100.0)
    assert 4.0 <= g.effective_stop_roi(pos) <= 30.0


def test_sized_stop_ignored_when_atr_sizing_is_off():
    from bot.futures_guard import GuardConfig, FuturesPosition
    cfg = GuardConfig(initial_stop_roi=10, atr_stop_mult=0.0)
    g = _guardian(FakeExchange(), cfg=cfg)
    g.note_entry_context("X/USDT:USDT", {"sized_stop_roi": 25.0})
    pos = FuturesPosition("X/USDT:USDT", "short", 100.0, 10.0, 10, 100.0)
    assert g.effective_stop_roi(pos) == pytest.approx(10.0)


# ── ATR must be measured on the traded timeframe ─────────────────────────────

def test_atr_uses_the_configured_timeframe():
    """
    ATR was computed on 15m candles while the strategy trades 3m. A 15m ATR is
    ~2.2x a 3m ATR, so stops were more than twice as wide as the traded chart
    justified — FORM got a 24.4% ROI stop where 10.9% was right.
    """
    seen = {}

    class TFEx(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, limit=120):
            seen.setdefault("atr_tf", timeframe)
            p = 100.0
            return [[0, p, p * 1.002, p * 0.998, p, 10.0] for _ in range(limit)]

    g = _guardian(TFEx())
    g.atr_timeframe = "3m"
    g.atr_pct("X/USDT:USDT")
    assert seen["atr_tf"] == "3m"


def test_atr_timeframe_change_scales_the_stop():
    class TFEx(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, limit=120):
            # wider candles on the longer timeframe, as in a real market
            half = 0.002 if timeframe == "3m" else 0.0045
            p = 100.0
            return [[0, p, p * (1 + half), p * (1 - half), p, 10.0] for _ in range(limit)]

    from bot.futures_guard import GuardConfig, FuturesPosition
    cfg = GuardConfig(atr_stop_mult=1.5, atr_stop_min_roi=1, atr_stop_max_roi=99)
    pos = FuturesPosition("X/USDT:USDT", "long", 100.0, 10.0, 20, 100.0)

    g3 = _guardian(TFEx(), cfg=cfg); g3.atr_timeframe = "3m"
    g15 = _guardian(TFEx(), cfg=cfg); g15.atr_timeframe = "15m"
    assert g15.effective_stop_roi(pos) > g3.effective_stop_roi(pos)


def test_trail_log_reports_the_rate_actually_sent():
    """
    The log printed the raw config value, so a correctly-placed 0.25% trail was
    reported as 1.0% — which made a working trail look misconfigured.
    """
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(arm_roi=8, callback_roi=5,
                      trail_callback_pct=1.0, trail_callback_roi=5.0)
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0, lev=20, margin=100.0)],
                        price=100.0)
    g = _guardian(fake, cfg=cfg)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 25.0)
    g.run_cycle()
    armed = [a for a in g._actions if a["action"] == "trail_armed"]
    assert armed, "trail should have armed"
    assert "1.0%" not in armed[0]["detail"]
    assert "ROI" in armed[0]["detail"]


# ── Risk invariant backstop ──────────────────────────────────────────────────

def test_risk_overshoot_is_detected_and_recorded():
    """
    Sizing and stop placement happen in different components at different
    times. When they disagree the position is silently oversized — LDO risked
    2.8x its budget. This makes it loud at stop-placement time.
    """
    fake = FakeExchange(positions=[_raw_pos("short", entry=0.4311,
                                            contracts=32900.0, lev=20,
                                            margin=709.33)],
                        price=0.4311)
    g = _guardian(fake)
    g.risk_pct = 1.0
    g._wallet_balance_cached = 4729.0
    from bot.futures_guard import FuturesPosition
    pos = FuturesPosition("DOGE/USDT:USDT", "short", 0.4311, 32900.0, 20, 709.33)
    g._check_risk_invariant(pos, 18.82)

    over = g._risk_overshoots.get("DOGE/USDT:USDT")
    assert over is not None
    assert over["ratio"] == pytest.approx(2.8, abs=0.1)
    assert any(a["action"] == "RISK_OVERSHOOT" for a in g._actions)


def test_correctly_sized_position_raises_no_overshoot():
    from bot.futures_guard import FuturesPosition
    g = _guardian(FakeExchange())
    g.risk_pct = 1.0
    g._wallet_balance_cached = 4729.0
    pos = FuturesPosition("X/USDT:USDT", "short", 0.4311, 11500.0, 20, 251.28)
    g._check_risk_invariant(pos, 18.82)
    assert g._risk_overshoots == {}


def test_invariant_is_silent_without_a_risk_budget():
    """Manually opened positions have no budget — no false alarms."""
    from bot.futures_guard import FuturesPosition
    g = _guardian(FakeExchange())
    g._wallet_balance_cached = 4729.0
    pos = FuturesPosition("X/USDT:USDT", "short", 0.4311, 32900.0, 20, 709.33)
    g._check_risk_invariant(pos, 18.82)
    assert g._risk_overshoots == {}


def test_small_overshoot_within_tolerance_is_not_flagged():
    from bot.futures_guard import FuturesPosition
    g = _guardian(FakeExchange())
    g.risk_pct = 1.0
    g._wallet_balance_cached = 4729.0
    # 10% over budget — inside the 1.25x tolerance
    pos = FuturesPosition("X/USDT:USDT", "short", 0.4311, 12000.0, 20, 276.0)
    g._check_risk_invariant(pos, 18.82)
    assert g._risk_overshoots == {}


# ── Stop is capped to the risk budget ────────────────────────────────────────

def _budget_guardian(wallet=4729.0, risk_pct=1.0, min_roi=4.0):
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(atr_stop_mult=1.5, atr_stop_min_roi=min_roi, atr_stop_max_roi=30)
    g = _guardian(FakeExchange(), cfg=cfg)
    g.risk_pct = risk_pct
    g._wallet_balance_cached = wallet
    return g


def test_oversized_position_gets_a_tighter_stop():
    """
    Budget and actual margin uniquely determine the widest acceptable stop, so
    a sizing/stop disagreement can be corrected rather than merely reported.
    """
    from bot.futures_guard import FuturesPosition
    g = _budget_guardian()
    pos = FuturesPosition("LDO/USDT:USDT", "short", 0.4311, 32900.0, 20, 709.33)
    capped = g._cap_stop_to_budget(pos, 18.82)
    assert capped == pytest.approx(6.67, abs=0.05)
    assert pos.margin * capped / 100 == pytest.approx(47.29, abs=0.5)
    assert any(a["action"] == "stop_capped" for a in g._actions)


def test_correctly_sized_position_is_untouched():
    from bot.futures_guard import FuturesPosition
    g = _budget_guardian()
    pos = FuturesPosition("X/USDT:USDT", "short", 0.4311, 11500.0, 20, 251.28)
    assert g._cap_stop_to_budget(pos, 18.82) == pytest.approx(18.82)


def test_cap_never_widens_a_stop():
    """Only ever tightens — a generous budget must not loosen protection."""
    from bot.futures_guard import FuturesPosition
    g = _budget_guardian(wallet=100000.0)
    pos = FuturesPosition("X/USDT:USDT", "short", 0.4311, 11500.0, 20, 251.28)
    assert g._cap_stop_to_budget(pos, 10.0) == pytest.approx(10.0)


def test_cap_is_floored_at_the_minimum_stop():
    """A stop tighter than the floor sits inside the noise — refuse to go there."""
    from bot.futures_guard import FuturesPosition
    g = _budget_guardian(min_roi=4.0)
    huge = FuturesPosition("X/USDT:USDT", "short", 0.4311, 200000.0, 20, 4000.0)
    capped = g._cap_stop_to_budget(huge, 25.0)
    assert capped == pytest.approx(4.0)      # floored, not 1.2%


def test_manual_position_without_budget_is_not_capped():
    from bot.futures_guard import FuturesPosition
    g = _guardian(FakeExchange())
    g._wallet_balance_cached = 4729.0        # no risk_pct set
    pos = FuturesPosition("X/USDT:USDT", "short", 0.4311, 32900.0, 20, 709.33)
    assert g._cap_stop_to_budget(pos, 18.82) == pytest.approx(18.82)


def test_capped_stop_is_actually_placed():
    """The cap must reach the order, not just the log."""
    fake = FakeExchange(positions=[_raw_pos("short", entry=0.4311,
                                            contracts=32900.0, lev=20,
                                            margin=709.33)], price=0.4311)
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(atr_stop_mult=1.5, atr_stop_min_roi=4, atr_stop_max_roi=30)
    g = _guardian(fake, cfg=cfg)
    g.risk_pct = 1.0
    g.run_cycle()
    stops = [o for o in fake.created if o["type"] == "STOP_MARKET"]
    assert stops
    trigger = float(stops[0]["params"]["stopPrice"])
    # a 6.7% ROI stop at 20x is ~0.33% of price, not ~0.95%
    move = abs(trigger - 0.4311) / 0.4311 * 100
    assert move < 0.6, f"stop placed {move:.2f}% away — cap did not reach the order"


# ── Past the stop: winning vs losing are different situations ────────────────

def _rejecting_exchange(positions, price):
    class Ex(FakeExchange):
        def create_order(self, **kw):
            if kw.get("type") == "STOP_MARKET":
                raise RuntimeError("Order would immediately trigger")
            return FakeExchange.create_order(self, **kw)
    return Ex(positions=positions, price=price)


def test_position_past_its_stop_in_LOSS_is_closed():
    """
    The trailing fallback assumed 'past the stop' meant in profit. A position
    past it in the LOSING direction trails below an already-underwater price and
    protects nothing — BR ran to 3.3x its budget that way. The stop level has
    been breached, so close.
    """
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(initial_stop_roi=10, arm_roi=8, callback_roi=5,
                      atr_stop_mult=0.0, close_if_past_stop=True)
    fake = _rejecting_exchange([_raw_pos("long", entry=100.0)], 100.0)
    g = _guardian(fake, cfg=cfg)
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, -25.0)      # well past a -10% stop
    g.run_cycle()

    closes = [o for o in fake.created if o["type"] == "MARKET"]
    trails = [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    assert closes, "a breached stop must close the position"
    assert not trails, "must not trail a losing position past its stop"
    assert any(a["action"] == "closed_past_stop" for a in g._actions)


def test_position_past_its_stop_in_PROFIT_is_trailed():
    """The original case remains: gains past the stop are protected by trailing."""
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(initial_stop_roi=10, arm_roi=8, callback_roi=5,
                      trail_callback_roi=5.0, atr_stop_mult=0.0,
                      close_if_past_stop=True)
    fake = _rejecting_exchange([_raw_pos("short", entry=100.0)], 100.0)
    g = _guardian(fake, cfg=cfg)
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 25.0)       # in profit
    g.run_cycle()

    trails = [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    closes = [o for o in fake.created if o["type"] == "MARKET"]
    assert trails, "a profitable position past its stop should be trailed"
    assert not closes, "must not close a winning position"


def test_breach_can_be_left_open_when_configured():
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(initial_stop_roi=10, atr_stop_mult=0.0,
                      close_if_past_stop=False)
    fake = _rejecting_exchange([_raw_pos("long", entry=100.0)], 100.0)
    g = _guardian(fake, cfg=cfg)
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, -25.0)
    g.run_cycle()
    assert not [o for o in fake.created if o["type"] == "MARKET"]
    assert any(a["action"] == "UNPROTECTED" for a in g._actions)


# ── A transient reply must not look like a close ─────────────────────────────

def test_single_missing_cycle_does_not_close_or_cancel():
    """
    BR was recorded closed at -13.89% while still open, its stop cancelled as
    orphaned, then re-adopted and closed AGAIN at -23.62%. One position, two
    trade records, loss growing across the gap — caused by trusting a single
    fetch_positions reply.
    """
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23614)], price=0.23614)
    g = _guardian(fake)
    g.run_cycle()
    stop_id = fake.created[0]["id"]

    fake._positions = []            # one transient empty reply
    g.run_cycle()

    assert "DOGE/USDT:USDT" in g._states, "state wiped on a single absence"
    assert fake.cancelled == [], "protective stop cancelled on a single absence"
    assert g.closed_trades() == [], "phantom closed trade recorded"


def test_position_reappearing_clears_the_missing_count():
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23614)], price=0.23614)
    g = _guardian(fake)
    g.run_cycle()
    raw = fake._positions
    fake._positions = []
    g.run_cycle()
    assert g._missing_counts.get("DOGE/USDT:USDT") == 1
    fake._positions = raw           # it was there all along
    g.run_cycle()
    assert "DOGE/USDT:USDT" not in g._missing_counts
    assert g.closed_trades() == []


def test_close_is_recorded_once_confirmed():
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23614)], price=0.23614)
    g = _guardian(fake)
    g.run_cycle()
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    assert len(g.closed_trades()) == 1
    assert "DOGE/USDT:USDT" not in g._states


def test_no_duplicate_trade_records_for_one_position():
    """Two records for one position is the signature of the phantom close."""
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23614)], price=0.23614)
    g = _guardian(fake)
    g.run_cycle()
    raw = fake._positions
    for _ in range(2):              # flicker, then return
        fake._positions = []
        g.run_cycle()
        fake._positions = raw
        g.run_cycle()
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    symbols = [t["symbol"] for t in g.closed_trades()]
    assert symbols.count("DOGE/USDT:USDT") == 1, f"recorded {len(symbols)} times"


# ── Which backend answered ───────────────────────────────────────────────────

def test_instance_identity_reports_the_futures_environment():
    """
    Two instances behind separate hostnames look identical in the UI. A swapped
    nginx upstream then serves one dashboard while showing the other bot's
    data, which is only detectable from the numbers being wrong.
    """
    import bot.api as api
    prev = api._guardian

    class G:
        demo = True
        state_owner = "demo:abc12345"

    try:
        api._guardian = G()
        ident = api._instance_identity()
        assert ident["futures_env"] == "demo"
        assert ident["state_owner"] == "demo:abc12345"
        assert ident["host"]
    finally:
        api._guardian = prev


def test_instance_identity_distinguishes_live():
    import bot.api as api
    prev = api._guardian

    class G:
        demo = False
        state_owner = "live:xyz98765"

    try:
        api._guardian = G()
        assert api._instance_identity()["futures_env"] == "live"
    finally:
        api._guardian = prev


def test_instance_identity_without_a_guardian_is_unknown():
    import bot.api as api
    prev = api._guardian
    try:
        api._guardian = None
        assert api._instance_identity()["futures_env"] == "unknown"
    finally:
        api._guardian = prev


# ── The unprotected window after a fill ──────────────────────────────────────

def test_polls_faster_while_an_entry_order_rests():
    """
    Between an entry filling and the guardian seeing it, the position has no
    stop. At 20x a 0.5% move in 5s is already -10% ROI, so the window is where
    a sharp move does its damage.
    """
    g = _guardian(FakeExchange())
    g.poll_interval = 5.0
    g.pending_poll_interval = 1.0

    class Entry:
        def export_placed_orders(self): return {"X/USDT:USDT": [{"id": "E1"}]}
        def bot_placed_orders(self, sym): return [{"id": "E1"}]
    g._entry_service = Entry()
    assert g._has_pending_entries() is True


def test_normal_interval_when_nothing_is_pending():
    g = _guardian(FakeExchange())

    class Entry:
        def export_placed_orders(self): return {}
        def bot_placed_orders(self, sym): return []
    g._entry_service = Entry()
    assert g._has_pending_entries() is False


def test_no_entry_service_means_normal_polling():
    g = _guardian(FakeExchange())
    g._entry_service = None
    assert g._has_pending_entries() is False


def test_position_past_its_stop_on_discovery_is_closed_not_left_open():
    """
    A sharp move (or fill slippage) can leave a position already past its stop
    the first time the guardian sees it. It must close, not sit unprotected.
    """
    from bot.futures_guard import GuardConfig

    class Ex(FakeExchange):
        def create_order(self, **kw):
            if kw.get("type") == "STOP_MARKET":
                raise RuntimeError("Order would immediately trigger")
            return FakeExchange.create_order(self, **kw)

    cfg = GuardConfig(initial_stop_roi=10, atr_stop_mult=0.0,
                      close_if_past_stop=True)
    fake = Ex(positions=[_raw_pos("long", entry=100.0)], price=100.0)
    g = _guardian(fake, cfg=cfg)
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, -20.0)      # gapped past the stop
    g.run_cycle()
    assert [o for o in fake.created if o["type"] == "MARKET"]
    assert any(a["action"] == "closed_past_stop" for a in g._actions)


# ── Protective stops must not accumulate ─────────────────────────────────────

class _FlakyCancelEx(FakeExchange):
    """Cancels fail for a while, as observed on ON."""
    def __init__(self, fail_n=2, **kw):
        super().__init__(**kw)
        self.fail_cancels = fail_n
    def cancel_order(self, order_id, symbol):
        if self.fail_cancels > 0:
            self.fail_cancels -= 1
            raise RuntimeError("cancel failed (transient)")
        return FakeExchange.cancel_order(self, order_id, symbol)


def _ratchet_cfg():
    from bot.futures_guard import GuardConfig
    return GuardConfig(initial_stop_roi=10, arm_roi=5, callback_roi=3,
                       atr_stop_mult=0.0, use_native_trail=False,
                       min_stop_move_roi=0.5)


def test_failed_cancels_do_not_orphan_stops_permanently():
    """
    Only the latest stop id was tracked, so a failed cancel orphaned that stop
    forever. Three protective stops were left resting on one position.
    """
    fake = _FlakyCancelEx(fail_n=2, positions=[_raw_pos("long", entry=0.17129)],
                          price=0.17129)
    g = _guardian(fake, cfg=_ratchet_cfg())
    g.run_cycle()
    pos = g.fetch_positions()[0]
    for roi in (6.0, 9.0, 12.0):
        fake._price = price_for_roi(pos, roi)
        g.run_cycle()
    # the failed ones are still known, so they can be swept later
    assert g._all_stop_ids.get("DOGE/USDT:USDT")


def test_all_known_stops_cancelled_when_the_position_closes():
    fake = _FlakyCancelEx(fail_n=2, positions=[_raw_pos("long", entry=0.17129)],
                          price=0.17129)
    g = _guardian(fake, cfg=_ratchet_cfg())
    g.run_cycle()
    pos = g.fetch_positions()[0]
    for roi in (6.0, 9.0, 12.0):
        fake._price = price_for_roi(pos, roi)
        g.run_cycle()

    fake.fail_cancels = 0
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    # more than one cancel attempted: every stop ever placed, not just the last
    assert len(fake.cancelled) > 1
    assert "DOGE/USDT:USDT" not in g._all_stop_ids


def test_current_stop_is_not_cancelled_by_the_sweep():
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.17129)], price=0.17129)
    g = _guardian(fake, cfg=_ratchet_cfg())
    g.run_cycle()
    current = g._states["DOGE/USDT:USDT"].stop_order_id
    assert current not in [oid for oid, _ in fake.cancelled]


def test_native_trail_id_also_cleaned_up_on_close():
    from bot.futures_guard import GuardConfig
    cfg = GuardConfig(initial_stop_roi=10, arm_roi=5, callback_roi=3,
                      trail_callback_roi=3.0, atr_stop_mult=0.0)
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake, cfg=cfg)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 25.0)
    g.run_cycle()
    trail_id = g._states["DOGE/USDT:USDT"].native_trail_id
    assert trail_id
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    assert any(oid == trail_id for oid, _ in fake.cancelled)


# ── A failed cancel at arming must be retried, not deferred to close ─────────

class _FailFirstCancelEx(FakeExchange):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.fail_next_cancel = True
    def cancel_order(self, order_id, symbol):
        if self.fail_next_cancel:
            self.fail_next_cancel = False
            raise RuntimeError("cancel failed at arming")
        return FakeExchange.cancel_order(self, order_id, symbol)


def _armed_cfg():
    from bot.futures_guard import GuardConfig
    return GuardConfig(initial_stop_roi=5, arm_roi=5, callback_roi=3,
                       trail_callback_roi=3.0, use_native_trail=True,
                       atr_stop_mult=0.0)


def _arm(fake, adaptive=False):
    g = _guardian(fake, cfg=_armed_cfg(), adaptive=adaptive)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    fake._price = price_for_roi(pos, 8.0)
    g.run_cycle()
    return g


def test_fixed_stop_orphaned_at_arming_is_retried_next_cycle():
    """
    manage_position returned early once the native trail was set, so a fixed
    stop whose cancel failed at arming sat untouched until the position closed
    — still able to fire at a level the trade had left behind.
    """
    fake = _FailFirstCancelEx(positions=[_raw_pos("long", entry=0.23411)],
                              price=0.23411)
    g = _arm(fake)
    assert fake.cancelled == [], "cancel was expected to fail at arming"
    g.run_cycle()
    assert fake.cancelled, "orphaned fixed stop was not retried while armed"


def test_retry_does_not_cancel_the_active_trail():
    fake = _FailFirstCancelEx(positions=[_raw_pos("long", entry=0.23411)],
                              price=0.23411)
    g = _arm(fake)
    trail_id = g._states["DOGE/USDT:USDT"].native_trail_id
    for _ in range(3):
        g.run_cycle()
    assert trail_id not in [oid for oid, _ in fake.cancelled]
    assert g._states["DOGE/USDT:USDT"].native_trail_id == trail_id


def test_sweep_is_a_noop_when_nothing_is_superseded():
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23411)], price=0.23411)
    g = _arm(fake)
    n = len(fake.cancelled)
    for _ in range(3):
        g.run_cycle()
    assert len(fake.cancelled) == n


def test_trail_id_is_tracked_for_close_cleanup():
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23411)], price=0.23411)
    g = _arm(fake)
    trail_id = g._states["DOGE/USDT:USDT"].native_trail_id
    assert trail_id in g._all_stop_ids.get("DOGE/USDT:USDT", [])


# ── Orphaned protective stops ────────────────────────────────────────────────

class _OrphanEx(FakeExchange):
    """Mirrors an observed account: 3 positions, 11 resting orders."""
    def __init__(self):
        super().__init__(price=1.0)
        self.orders = [
            ("EPIC/USDT:USDT", "STOP_MARKET", True),
            ("CYS/USDT:USDT", "TRAILING_STOP_MARKET", False),
            ("WLD/USDT:USDT", "STOP_MARKET", True),
            ("TRADOOR/USDT:USDT", "STOP_MARKET", True),
            ("ON/USDT:USDT", "STOP_MARKET", True),
            ("APR/USDT:USDT", "STOP_MARKET", True),
            ("TAKE/USDT:USDT", "STOP_MARKET", True),
            ("APR/USDT:USDT", "STOP_MARKET", True),
            ("COTI/USDT:USDT", "STOP_MARKET", True),
            ("APR/USDT:USDT", "STOP_MARKET", True),
            ("ON/USDT:USDT", "STOP_MARKET", True),
        ]
        self.cancelled = []
    def fetch_open_orders(self, symbol=None):
        return [{"id": f"o{i}", "symbol": s, "type": t, "reduceOnly": ro}
                for i, (s, t, ro) in enumerate(self.orders)
                if symbol is None or s == symbol]
    def fetch_positions(self, symbols=None):
        return [{"symbol": s, "side": "long", "entryPrice": 1.0,
                 "contracts": 100.0, "leverage": 20, "initialMargin": 5.0}
                for s in ("EPIC/USDT:USDT", "TRADOOR/USDT:USDT", "CYS/USDT:USDT")]
    def cancel_order(self, oid, symbol):
        self.cancelled.append((symbol, str(oid)))


def test_orphaned_stops_are_swept():
    """
    A reduce-only stop with no position cannot protect anything, but can fire
    against a FUTURE position on the same symbol. Eight accumulated across five
    symbols, three of them on one symbol.
    """
    ex = _OrphanEx()
    g = _guardian(ex)
    res = g.reap_orphan_stops()
    assert len(res) == 8
    swept = {s for s, _ in res}
    assert swept == {"WLD/USDT:USDT", "ON/USDT:USDT", "APR/USDT:USDT",
                     "TAKE/USDT:USDT", "COTI/USDT:USDT"}


def test_stops_protecting_real_positions_are_kept():
    ex = _OrphanEx()
    g = _guardian(ex)
    g.reap_orphan_stops()
    kept = {s for s, _ in ex.cancelled}
    assert "EPIC/USDT:USDT" not in kept
    assert "TRADOOR/USDT:USDT" not in kept


def test_entry_orders_are_never_swept_by_this():
    """CYS's non-reduce-only entry is not this sweep's business."""
    ex = _OrphanEx()
    g = _guardian(ex)
    g.reap_orphan_stops()
    assert "CYS/USDT:USDT" not in {s for s, _ in ex.cancelled}


def test_non_stop_reduce_only_orders_are_left_alone():
    class Ex(_OrphanEx):
        def __init__(self):
            super().__init__()
            self.orders = [("XYZ/USDT:USDT", "LIMIT", True)]
    ex = Ex()
    g = _guardian(ex)
    assert g.reap_orphan_stops() == []


def test_sweep_survives_an_unknown_order_reply():
    class Ex(_OrphanEx):
        def cancel_order(self, oid, symbol):
            raise RuntimeError('binanceusdm {"code":-2011,"msg":"Unknown order sent."}')
    g = _guardian(Ex())
    assert g.reap_orphan_stops() == []      # nothing recorded, no exception


def test_sweep_can_be_disabled():
    ex = _OrphanEx()
    g = _guardian(ex)
    g.sweep_orphan_stops = False
    g.run_cycle()
    assert not any(s.startswith("APR") for s, _ in ex.cancelled)


# ── The sweep must not depend on the account-wide order list ─────────────────

class _BlindAccountWideEx(FakeExchange):
    """The account-wide list omits conditional orders; per-symbol finds them."""
    def __init__(self):
        super().__init__(price=1.0)
        self.cancelled = []
        self.all_orders = {
            "WLD/USDT:USDT":  [("o3", "STOP_MARKET", True)],
            "TAKE/USDT:USDT": [("o6", "STOP_MARKET", True)],
            "APR/USDT:USDT":  [("o7", "STOP_MARKET", True), ("o9", "STOP_MARKET", True)],
            "COTI/USDT:USDT": [("o8", "STOP_MARKET", True)],
            "ON/USDT:USDT":   [("o10", "STOP_MARKET", True)],
            "EPIC/USDT:USDT": [("o1", "STOP_MARKET", True)],
        }
    def fetch_open_orders(self, symbol=None):
        if symbol is None:
            return []
        return [{"id": i, "symbol": symbol, "type": t, "reduceOnly": ro}
                for i, t, ro in self.all_orders.get(symbol, [])]
    def fetch_positions(self, symbols=None):
        return [{"symbol": "EPIC/USDT:USDT", "side": "long", "entryPrice": 1.0,
                 "contracts": 100.0, "leverage": 20, "initialMargin": 5.0}]
    def cancel_order(self, oid, symbol):
        self.cancelled.append((symbol, str(oid)))


def _blind_guardian():
    ex = _BlindAccountWideEx()
    g = _guardian(ex)
    g._closed_trades = [{"symbol": s} for s in
                        ("ON/USDT:USDT", "APR/USDT:USDT", "COTI/USDT:USDT",
                         "TAKE/USDT:USDT", "WLD/USDT:USDT")]
    return g, ex


def test_orphans_found_per_symbol_when_account_wide_is_blind():
    """
    Six orphans survived a sweep that relied on the account-wide call alone.
    Closed-trade symbols are where stops get left behind, so those are queried
    directly.
    """
    g, ex = _blind_guardian()
    res = g.reap_orphan_stops()
    assert len(res) == 6
    assert {s.split("/")[0] for s, _ in res} == {"WLD", "TAKE", "APR", "COTI", "ON"}


def test_symbol_with_a_position_is_still_skipped():
    g, ex = _blind_guardian()
    res = g.reap_orphan_stops()
    assert "EPIC/USDT:USDT" not in {s for s, _ in res}


def test_scan_list_excludes_live_symbols():
    g, _ = _blind_guardian()
    syms = g._orphan_scan_symbols({"APR/USDT:USDT"})
    assert "APR/USDT:USDT" not in syms
    assert "ON/USDT:USDT" in syms


def test_duplicate_ids_across_paths_are_not_cancelled_twice():
    class BothEx(_BlindAccountWideEx):
        def fetch_open_orders(self, symbol=None):
            if symbol is None:
                return [{"id": "o7", "symbol": "APR/USDT:USDT",
                         "type": "STOP_MARKET", "reduceOnly": True}]
            return _BlindAccountWideEx.fetch_open_orders(self, symbol)
    ex = BothEx()
    g = _guardian(ex)
    g._closed_trades = [{"symbol": "APR/USDT:USDT"}]
    res = g.reap_orphan_stops()
    assert [oid for _, oid in res].count("o7") == 1


# ── Surplus stops on a LIVE position ─────────────────────────────────────────

class _SurplusEx(FakeExchange):
    """One APR position carrying four protective stops."""
    def __init__(self):
        super().__init__(price=1.0)
        self.cancelled = []
        self.book = {
            "APR/USDT:USDT": [("e", True, 1200), ("f", True, 800),
                              ("g", True, 300), ("h", True, 200)],
            "WLD/USDT:USDT": [("m", True, 350)],
        }
    def fetch_open_orders(self, symbol=None):
        if symbol is None:
            return []
        return [{"id": i, "symbol": symbol, "type": "STOP_MARKET",
                 "reduceOnly": ro, "timestamp": ts}
                for i, ro, ts in self.book.get(symbol, [])]
    def fetch_positions(self, symbols=None):
        return [{"symbol": "APR/USDT:USDT", "side": "long", "entryPrice": 0.194,
                 "contracts": 22106, "leverage": 20, "initialMargin": 214.0}]
    def cancel_order(self, oid, symbol):
        self.cancelled.append((symbol, str(oid)))


def _surplus_guardian():
    ex = _SurplusEx()
    g = _guardian(ex)
    g._closed_trades = [{"symbol": s} for s in ex.book]
    return g, ex


def test_surplus_stops_on_a_live_position_are_trimmed():
    """
    Four stops on one position: only the newest can be current, the rest are
    superseded ratchets that can still fire at a stale level.
    """
    g, ex = _surplus_guardian()
    res = g.reap_orphan_stops()
    apr = sorted(o for s, o in res if s.startswith("APR"))
    assert apr == ["f", "g", "h"]


def test_the_newest_stop_is_kept():
    g, ex = _surplus_guardian()
    g.reap_orphan_stops()
    assert "e" not in [o for _, o in ex.cancelled]


def test_a_single_stop_on_a_position_is_untouched():
    class OneEx(_SurplusEx):
        def __init__(self):
            super().__init__()
            self.book = {"APR/USDT:USDT": [("e", True, 1200)]}
    g = _guardian(OneEx())
    g._closed_trades = [{"symbol": "APR/USDT:USDT"}]
    assert g.reap_orphan_stops() == []


def test_reconciliation_separates_surplus_from_protecting():
    g, _ = _surplus_guardian()
    r = g.reconcile_orders()
    assert len(r["protecting"]) == 1
    assert len(r["duplicate_stops"]) == 3
    assert len(r["orphan_stops"]) == 1
    assert r["account_wide_count"] == 0 and r["per_symbol_count"] > 0


# ── The sweep must not depend on bot memory ──────────────────────────────────

class _RawOnlyEx(FakeExchange):
    """
    The realistic worst case: bot memory empty after a history reset, the
    unified account-wide call blind, only the raw endpoint returning data.
    """
    def __init__(self):
        super().__init__(price=1.0)
        self.cancelled = []
        self.raw = [
            {"orderId": "a", "symbol": "ONUSDT",
             "origType": "TRAILING_STOP_MARKET", "reduceOnly": "false", "time": 1000},
            {"orderId": "b", "symbol": "ONUSDT",
             "origType": "STOP_MARKET", "reduceOnly": "true", "time": 900},
            {"orderId": "c", "symbol": "WLDUSDT",
             "origType": "STOP_MARKET", "reduceOnly": "true", "time": 350},
            {"orderId": "e", "symbol": "APRUSDT",
             "origType": "STOP_MARKET", "reduceOnly": "true", "time": 1200},
            {"orderId": "f", "symbol": "APRUSDT",
             "origType": "STOP_MARKET", "reduceOnly": "true", "time": 800},
        ]
        self.markets_by_id = {f"{b}USDT": [{"symbol": f"{b}/USDT:USDT"}]
                              for b in ("ON", "WLD", "APR")}
    def fetch_open_orders(self, symbol=None):
        return []
    def fapiPrivateGetOpenOrders(self, params=None):
        return self.raw
    def fetch_positions(self, symbols=None):
        return [{"symbol": "APR/USDT:USDT", "side": "long", "entryPrice": 0.194,
                 "contracts": 22106, "leverage": 20, "initialMargin": 214.0}]
    def cancel_order(self, oid, symbol):
        self.cancelled.append((symbol, str(oid)))


def _raw_guardian():
    ex = _RawOnlyEx()
    g = _guardian(ex)
    g._closed_trades = []          # memory wiped, as after a history reset
    return g, ex


def test_sweep_works_with_no_bot_memory():
    """
    The sweep built its symbol list from bot memory, which a history reset or
    a failed persist empties — so it queried nothing and swept nothing while
    orders accumulated on the exchange.
    """
    g, ex = _raw_guardian()
    ids = sorted(o for _, o in g.reap_orphan_stops())
    assert ids == ["b", "c", "f"]


def test_reduce_only_as_the_string_false_is_not_treated_as_true():
    """bool("false") is True — this would cancel entry orders."""
    g, ex = _raw_guardian()
    ids = [o for _, o in g.reap_orphan_stops()]
    assert "a" not in ids, "a non-reduce-only entry order was cancelled"


def test_raw_symbol_ids_are_mapped_to_unified_symbols():
    g, _ = _raw_guardian()
    o = g._normalise_order({"orderId": "x", "symbol": "WLDUSDT",
                            "origType": "STOP_MARKET", "reduceOnly": "true"})
    assert o["symbol"] == "WLD/USDT:USDT"


def test_unmapped_symbol_falls_back_to_a_sensible_guess():
    g, _ = _raw_guardian()
    o = g._normalise_order({"orderId": "x", "symbol": "NEWCOINUSDT",
                            "origType": "STOP_MARKET", "reduceOnly": "true"})
    assert o["symbol"] == "NEWCOIN/USDT:USDT"


def test_reconciliation_also_uses_the_raw_listing():
    g, _ = _raw_guardian()
    r = g.reconcile_orders()
    assert r["account_wide_count"] == 5
    assert len(r["orphan_stops"]) == 2          # b, c
    assert len(r["duplicate_stops"]) == 1       # f
    assert len(r["stale_entries"]) == 1         # a


# ── ccxt's warning gate must not silence the order listing ───────────────────

class _WarnGateEx(FakeExchange):
    """
    Mimics ccxt: fetch_open_orders() without a symbol raises a rate-limit
    WARNING as an ExchangeError unless acknowledged. The request never leaves
    the process, so the result looks like an account with no orders.
    """
    def __init__(self):
        super().__init__(price=1.0)
        self.options = {"defaultType": "future"}        # option deliberately absent
        self.urls = {"api": {"fapiPrivate": "https://demo-fapi.binance.com/fapi/v1"}}
        self.sent = []
    def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
        if symbol is None:
            ack = (self.options.get("fetchOpenOrders") or {}).get("warnWithoutSymbol")
            if ack is not False:
                raise RuntimeError("ExchangeError: fetchOpenOrders() WARNING: ...")
            self.sent.append("account-wide")
            return [{"id": "z1", "symbol": "ON/USDT:USDT",
                     "type": "STOP_MARKET", "reduceOnly": True}]
        return []
    def fapiPrivateGetOpenOrders(self, params=None):
        return []
    def fetch_positions(self, symbols=None):
        return []


def test_warning_gate_is_acknowledged_before_listing():
    """
    The gate made every account-wide listing return nothing, which read as a
    clean account and left the orphan sweep with no input in any version.
    """
    ex = _WarnGateEx()
    g = _guardian(ex)
    rows = g._all_open_orders_raw()
    assert ex.sent == ["account-wide"], "the request was never sent"
    assert len(rows) == 1


def test_gate_is_fixed_even_if_options_are_replaced():
    ex = _WarnGateEx()
    g = _guardian(ex)
    ex.options = {"defaultType": "future"}       # something wiped it
    assert len(g._all_open_orders_raw()) == 1


def test_orphan_sweep_sees_orders_once_the_gate_is_open():
    ex = _WarnGateEx()
    g = _guardian(ex)
    res = g.reap_orphan_stops()
    assert [oid for _, oid in res] == ["z1"]


# ── -2011 means the order is gone, not still resting ─────────────────────────

def test_unknown_order_is_a_successful_cancel():
    """
    -2011 "Unknown order sent" means the order does not exist — already
    filled, cancelled, or from a previous run. Treating it as a failure made
    the guardian log "it is still resting" about an order that was gone, and
    retry it forever.
    """
    g = _guardian(FakeExchange())
    assert g._order_already_gone('binanceusdm {"code":-2011,"msg":"Unknown order sent."}')
    assert g._order_already_gone("Unknown order sent.")
    assert not g._order_already_gone('{"code":-1021,"msg":"Timestamp ahead."}')


def test_minus_2011_is_not_treated_as_a_successful_cancel():
    """
    -2011 was assumed to mean the order is gone. An order that was STILL
    RESTING on the exchange returned -2011 to a cancel, so treating it as
    success dropped it from tracking and orphaned it permanently.
    """
    from bot.futures_guard import FuturesPosition

    class GoneEx(FakeExchange):
        def cancel_order(self, order_id, symbol):
            raise RuntimeError('binanceusdm {"code":-2011,"msg":"Unknown order sent."}')

    g = _guardian(GoneEx())
    pos = FuturesPosition("CYS/USDT:USDT", "long", 1.0, 10.0, 20, 100.0)
    assert g._cancel_stop(pos, "1000000196042886") is False


def test_failed_cancel_is_queued_for_retry():
    from bot.futures_guard import FuturesPosition

    class GoneEx(FakeExchange):
        def cancel_order(self, order_id, symbol):
            raise RuntimeError('{"code":-2011}')

    g = _guardian(GoneEx())
    pos = FuturesPosition("CYS/USDT:USDT", "long", 1.0, 10.0, 20, 100.0)
    g._cancel_stop(pos, "old-1")
    assert "old-1" in g._pending_cancels.get("CYS/USDT:USDT", [])


def test_genuine_cancel_failure_is_still_reported():
    from bot.futures_guard import FuturesPosition

    class BadEx(FakeExchange):
        def cancel_order(self, order_id, symbol):
            raise RuntimeError('{"code":-1000,"msg":"An unknown error occurred."}')

    g = _guardian(BadEx())
    pos = FuturesPosition("CYS/USDT:USDT", "long", 1.0, 10.0, 20, 100.0)
    assert g._cancel_stop(pos, "x") is False


# ── Do not pay 40x weight for a listing that returns nothing ─────────────────

class _EmptyListingEx(FakeExchange):
    def __init__(self):
        super().__init__(price=1.0)
        self.wide_calls = 0
        self.options = {"defaultType": "future",
                        "fetchOpenOrders": {"warnWithoutSymbol": False}}
        self.urls = {"api": {"fapiPrivate": "x"}}
    def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
        if symbol is None:
            self.wide_calls += 1
        return []
    def fapiPrivateGetOpenOrders(self, params=None):
        self.wide_calls += 1
        return []


def test_account_wide_listing_backs_off_when_always_empty():
    """
    The account-wide call costs 40x request weight and returns nothing on demo
    futures, so paying it every sweep buys nothing.
    """
    ex = _EmptyListingEx()
    g = _guardian(ex)
    for _ in range(10):
        g._all_open_orders_raw()
    # 2 calls per attempt, and attempts stop after the limit
    assert ex.wide_calls == 2 * g.EMPTY_LISTING_LIMIT


def test_backoff_expires_and_reprobes():
    ex = _EmptyListingEx()
    g = _guardian(ex)
    for _ in range(10):
        g._all_open_orders_raw()
    before = ex.wide_calls
    g._last_wide_probe -= g.WIDE_PROBE_INTERVAL_S + 1
    g._all_open_orders_raw()
    assert ex.wide_calls > before


def test_a_non_empty_reply_resets_the_counter():
    class SometimesEx(_EmptyListingEx):
        def fapiPrivateGetOpenOrders(self, params=None):
            self.wide_calls += 1
            return [{"orderId": "1", "symbol": "ONUSDT",
                     "origType": "STOP_MARKET", "reduceOnly": "true"}]
    g = _guardian(SometimesEx())
    for _ in range(8):
        g._all_open_orders_raw()
    assert g._empty_listings == 0


# ── Pending-cancel queue ─────────────────────────────────────────────────────

class _FlakyCancelOnceEx(FakeExchange):
    """Cancels fail until `heal` is set — mimics an unreliable venue."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.heal = False
        self.cancelled = []
    def cancel_order(self, order_id, symbol):
        if not self.heal:
            raise RuntimeError('{"code":-2011,"msg":"Unknown order sent."}')
        self.cancelled.append((symbol, str(order_id)))


def test_queue_survives_the_position_closing():
    """
    The position is gone but the order may not be, and with order listing
    unavailable nothing else will ever discover it.
    """
    fake = _FlakyCancelOnceEx(positions=[_raw_pos("long", entry=1.0)], price=1.0)
    g = _guardian(fake)
    g.run_cycle()
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    assert g._pending_cancels, "nothing queued after a failed close-time cancel"


def test_drain_retries_and_succeeds_later():
    fake = _FlakyCancelOnceEx(positions=[_raw_pos("long", entry=1.0)], price=1.0)
    g = _guardian(fake)
    g.run_cycle()
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    fake.heal = True
    assert g.drain_pending_cancels() >= 1
    assert not g._pending_cancels


def test_drain_gives_up_after_the_attempt_limit():
    fake = _FlakyCancelOnceEx(price=1.0)
    g = _guardian(fake)
    g._pending_cancels["X/USDT:USDT"] = ["stuck-1"]
    for _ in range(g.MAX_CANCEL_ATTEMPTS + 5):
        g.drain_pending_cancels()
    assert g._cancel_attempts["stuck-1"] >= g.MAX_CANCEL_ATTEMPTS


def test_queue_is_persisted_and_restored():
    import os
    import tempfile
    from bot import futures_state
    path = os.path.join(tempfile.mkdtemp(), "s.json")
    futures_state.save(path, states={}, pos_meta={}, closed_trades=[],
                       pending_cancels={"REZ/USDT:USDT": ["1000000196098048"]})
    g = _guardian(FakeExchange())
    g.state_owner = ""
    g.load_state(path)
    assert g._pending_cancels["REZ/USDT:USDT"] == ["1000000196098048"]


# ── Conditional stops live in the ALGO book ──────────────────────────────────

class _AlgoBookEx(FakeExchange):
    """
    The real venue: trailing and conditional stops are placed via
    POST /fapi/v1/algoOrder and held in a SEPARATE book. They never appear in
    /fapi/v1/openOrders, and a regular cancel returns -2011 for them.
    """
    def __init__(self):
        super().__init__(price=1.0)
        self.cancelled = []
        self.options = {"defaultType": "future",
                        "fetchOpenOrders": {"warnWithoutSymbol": False}}
        self.urls = {"api": {"fapiPrivate": "x"}}
        self.algo = [
            {"algoId": "A1", "symbol": "REZUSDT", "algoType": "STOP_MARKET",
             "reduceOnly": "true", "bookTime": 1000},
            {"algoId": "A2", "symbol": "CATIUSDT", "algoType": "STOP_MARKET",
             "reduceOnly": "true", "bookTime": 900},
            {"algoId": "A3", "symbol": "CATIUSDT", "algoType": "STOP_MARKET",
             "reduceOnly": "true", "bookTime": 100},
            {"algoId": "A4", "symbol": "CYSUSDT",
             "algoType": "TRAILING_STOP_MARKET", "reduceOnly": "false",
             "bookTime": 800},
        ]
        self.markets_by_id = {f"{b}USDT": [{"symbol": f"{b}/USDT:USDT"}]
                              for b in ("REZ", "CATI", "CYS")}
    def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
        return []
    def fapiPrivateGetOpenOrders(self, params=None):
        return []
    def fapiPrivateGetOpenAlgoOrders(self, params=None):
        return {"orders": self.algo}
    def fapiPrivateDeleteAlgoOrder(self, params=None):
        aid = str((params or {}).get("algoId"))
        if not any(a["algoId"] == aid for a in self.algo):
            raise RuntimeError("not found")
        self.algo = [a for a in self.algo if a["algoId"] != aid]
        self.cancelled.append(aid)
        return {"success": True}
    def cancel_order(self, oid, symbol):
        raise RuntimeError('binanceusdm {"code":-2011,"msg":"Unknown order sent."}')
    def fetch_positions(self, symbols=None):
        return []


def test_algo_book_orders_are_discovered():
    """
    Every listing returned zero while stops were plainly resting, because the
    stops were algo orders and the bot only queried the regular book.
    """
    g = _guardian(_AlgoBookEx())
    assert len(g._all_open_orders_raw()) == 4


def test_algo_orders_normalise_with_algoId_and_algoType():
    g = _guardian(_AlgoBookEx())
    o = g._normalise_order({"algoId": "A9", "symbol": "REZUSDT",
                            "algoType": "STOP_MARKET", "reduceOnly": "true"})
    assert o["id"] == "A9"
    assert o["symbol"] == "REZ/USDT:USDT"
    assert o["type"] == "STOP_MARKET"
    assert o["reduce_only"] is True


def test_orphaned_algo_stops_are_cancelled():
    ex = _AlgoBookEx()
    g = _guardian(ex)
    res = g.reap_orphan_stops()
    assert sorted(o for _, o in res) == ["A1", "A2", "A3"]


def test_algo_entry_order_is_not_cancelled_by_the_stop_sweep():
    ex = _AlgoBookEx()
    g = _guardian(ex)
    g.reap_orphan_stops()
    assert "A4" in [a["algoId"] for a in ex.algo]


def test_cancel_any_falls_through_to_the_algo_book():
    """-2011 from the regular book means 'not here', not 'does not exist'."""
    ex = _AlgoBookEx()
    g = _guardian(ex)
    assert g._cancel_any("A1", "REZ/USDT:USDT") is True
    assert "A1" in ex.cancelled


def test_cancel_any_reports_failure_when_neither_book_has_it():
    ex = _AlgoBookEx()
    g = _guardian(ex)
    assert g._cancel_any("NOPE", "REZ/USDT:USDT") is False


# ── -2011 semantics: which book was searched ─────────────────────────────────

def test_minus_2011_from_both_books_means_gone():
    """
    -2011 from the REGULAR book only means "not in this book" — these are algo
    orders. -2011 from the ALGO book too means it exists in neither, so it must
    stop being retried. A successful cancel followed by a -2011 retry was
    re-queueing orders that no longer existed.
    """
    class BothMissEx(_AlgoBookEx):
        def fapiPrivateDeleteAlgoOrder(self, params=None):
            raise RuntimeError('{"code":-2011,"msg":"Unknown order sent."}')
    g = _guardian(BothMissEx())
    assert g._cancel_any("GONE-1", "REZ/USDT:USDT") is True
    assert "GONE-1" in g._cancelled_ids


def test_confirmed_cancel_is_never_retried():
    ex = _AlgoBookEx()
    g = _guardian(ex)
    assert g._cancel_any("A1", "REZ/USDT:USDT") is True
    before = len(ex.cancelled)
    assert g._cancel_any("A1", "REZ/USDT:USDT") is True
    assert len(ex.cancelled) == before, "a confirmed cancel was retried"


def test_drain_skips_ids_already_confirmed():
    ex = _AlgoBookEx()
    g = _guardian(ex)
    g._cancelled_ids.add("A1")
    g._pending_cancels["REZ/USDT:USDT"] = ["A1"]
    g.drain_pending_cancels()
    assert not g._pending_cancels


# ── The income ledger is authoritative ───────────────────────────────────────

class _IncomeEx(FakeExchange):
    """One trade opened and closed at the SAME price, paying 2.07 in fees."""
    def market_id(self, s):
        return s.split("/")[0] + "USDT"
    def fapiPrivateGetIncome(self, params=None):
        return [{"incomeType": "REALIZED_PNL", "income": "0.00000000"},
                {"incomeType": "COMMISSION", "income": "-1.03487468"},
                {"incomeType": "COMMISSION", "income": "-1.03487471"}]


def test_income_ledger_reports_true_pnl_and_fees():
    """
    Reconstructing the exit from the stop level invented a profit: a trade that
    opened and closed at the same price (0 P&L, -2.07 fees) was recorded as
    +4.32.
    """
    g = _guardian(_IncomeEx())
    pnl, comm, found = g._income_for_position("SOLV/USDT:USDT", None)
    assert found
    assert pnl == pytest.approx(0.0)
    assert comm == pytest.approx(2.0697, abs=0.001)


def test_income_absent_is_reported_not_guessed():
    g = _guardian(FakeExchange())
    pnl, comm, found = g._income_for_position("X/USDT:USDT", None)
    assert found is False


def test_income_lookup_survives_an_error():
    class BadEx(FakeExchange):
        def market_id(self, s):
            return "XUSDT"
        def fapiPrivateGetIncome(self, params=None):
            raise RuntimeError("boom")
    g = _guardian(BadEx())
    assert g._income_for_position("X/USDT:USDT", None) == (0.0, 0.0, False)


# ── Fee capture, end to end ──────────────────────────────────────────────────

class _LedgerOnlyEx(FakeExchange):
    """Fills unavailable; only the income ledger has the numbers."""
    def market_id(self, s):
        return "SOLVUSDT"
    def fetch_my_trades(self, symbol, since=None, limit=50):
        raise RuntimeError("fills unavailable")
    def fapiPrivateGetIncome(self, params=None):
        return [{"incomeType": "REALIZED_PNL", "income": "0.00000000"},
                {"incomeType": "COMMISSION", "income": "-1.03487468"},
                {"incomeType": "COMMISSION", "income": "-1.03487471"}]


def _close_one(ex):
    g = _guardian(ex)
    g.run_cycle()
    ex._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    return g


def test_closed_trade_carries_fees():
    """
    The record never included fees_usdt at all — an edit targeted a line that
    did not match and silently did nothing, so the field was absent and the UI
    always showed a dash.
    """
    g = _close_one(_LedgerOnlyEx(positions=[_raw_pos("short", entry=0.00479)],
                                 price=0.00479))
    t = g.closed_trades()[0]
    assert "fees_usdt" in t
    assert t["fees_usdt"] == pytest.approx(2.0697, abs=0.001)


def test_net_pnl_is_recorded():
    g = _close_one(_LedgerOnlyEx(positions=[_raw_pos("short", entry=0.00479)],
                                 price=0.00479))
    t = g.closed_trades()[0]
    assert t["net_pnl_usdt"] == pytest.approx(-2.0697, abs=0.001)
    assert t["pnl_source"] == "ledger"


def test_ledger_is_consulted_even_when_fills_fail():
    """
    The ledger lookup sat inside the same try as the fills, so a fill failure
    skipped the authoritative source exactly when it was most needed.
    """
    g = _close_one(_LedgerOnlyEx(positions=[_raw_pos("short", entry=0.00479)],
                                 price=0.00479))
    assert g.closed_trades()[0]["pnl_source"] == "ledger"


def test_fees_do_not_leak_between_trades():
    """
    _last_trade_fees is instance state. Without a reset, a trade whose lookup
    failed inherited the PREVIOUS trade's commission — a plausible wrong number.
    """
    class NoDataEx(FakeExchange):
        def market_id(self, s):
            return "XUSDT"
        def fetch_my_trades(self, symbol, since=None, limit=50):
            raise RuntimeError("no fills")
        def fapiPrivateGetIncome(self, params=None):
            return []
    g = _guardian(_LedgerOnlyEx(positions=[_raw_pos("short", entry=0.00479)],
                                price=0.00479))
    g.run_cycle()
    g.exchange._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    assert g.closed_trades()[0]["fees_usdt"] is not None

    # a second position whose data is unavailable must NOT inherit the first
    # fee. closed_trades() is newest-first, so index 0 is the second trade.
    g.exchange = NoDataEx(positions=[_raw_pos("long", entry=1.0)], price=1.0)
    g.run_cycle()
    g.exchange._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    assert len(g.closed_trades()) == 2
    assert g.closed_trades()[0]["fees_usdt"] is None, "stale fee leaked"
    assert g.closed_trades()[1]["fees_usdt"] == pytest.approx(2.0697, abs=0.001)


# ── The income window must cover the ENTRY fill ──────────────────────────────

def test_entry_commission_is_inside_the_income_window():
    """
    The window started at opened_seen_at — when the guardian first SAW the
    position. The entry commission is charged at the fill, which precedes that
    poll, so only the EXIT side was captured. Every fee figure was halved and
    the wallet never reconciled: 5.22 captured against 10.40 actually paid.
    """
    import time
    now = time.time()

    class BothSidesEx(FakeExchange):
        def market_id(self, s):
            return "AEROUSDT"
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return []
        def fapiPrivateGetIncome(self, params=None):
            start = (params or {}).get("startTime")
            rows = [
                {"incomeType": "COMMISSION", "income": "-1.26",
                 "time": int((now - 300) * 1000)},          # entry fill
                {"incomeType": "REALIZED_PNL", "income": "-24.5864",
                 "time": int(now * 1000)},
                {"incomeType": "COMMISSION", "income": "-1.26",
                 "time": int(now * 1000)},                  # exit fill
            ]
            return [r for r in rows if not start or r["time"] >= start]

    fake = BothSidesEx(positions=[_raw_pos("short", entry=0.6156)], price=0.6156)
    g = _guardian(fake)
    g.run_cycle()
    g._pos_meta["DOGE/USDT:USDT"]["entry_context"] = {"sized_at": now - 300}
    g._pos_meta["DOGE/USDT:USDT"]["opened_seen_at"] = now - 30
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()

    t = g.closed_trades()[0]
    assert t["fees_usdt"] == pytest.approx(2.52, abs=0.01), "entry side missed"
    assert t["net_pnl_usdt"] == pytest.approx(-27.11, abs=0.01)


def test_window_falls_back_when_no_placement_stamp():
    """A manually opened position has no sized_at; opened_seen_at still works."""
    import time
    now = time.time()

    class Ex(FakeExchange):
        def market_id(self, s):
            return "XUSDT"
        def fetch_my_trades(self, symbol, since=None, limit=50):
            return []
        def fapiPrivateGetIncome(self, params=None):
            return [{"incomeType": "COMMISSION", "income": "-1.00",
                     "time": int(now * 1000)}]

    fake = Ex(positions=[_raw_pos("long", entry=1.0)], price=1.0)
    g = _guardian(fake)
    g.run_cycle()
    fake._positions = []
    for _ in range(g.MISSING_CONFIRMATIONS):
        g.run_cycle()
    assert g.closed_trades()[0]["fees_usdt"] == pytest.approx(1.0)


# ── D5: sizing must use REALISED equity ──────────────────────────────────────

def test_wallet_balance_is_preferred_over_margin_balance():
    """
    ccxt maps USDT.total to marginBalance = wallet + unrealised PnL. Preferring
    it made the risk budget swing with open-position marks: one jump of +74% in
    34 seconds scaled every subsequent entry 1.74x.
    """
    from bot.futures_guardian import resolve_usdt_balance
    val, src = resolve_usdt_balance({
        "USDT": {"total": 7221.24, "free": 3000.0},
        "info": {"totalWalletBalance": "4148.61",
                 "assets": [{"asset": "USDT", "walletBalance": "4148.61"}]}})
    assert val == pytest.approx(4148.61)
    assert "totalWalletBalance" in src


def test_a_balance_spike_is_not_used_for_sizing():
    class BalEx(FakeExchange):
        seq = [4148.61, 4148.61, 7221.24, 4150.0]
        def __init__(self):
            super().__init__(price=1.0)
            self.i = 0
        def fetch_balance(self):
            v = self.seq[min(self.i, len(self.seq) - 1)]
            self.i += 1
            return {"info": {"totalWalletBalance": str(v)}}
        def fetch_positions(self, symbols=None):
            return []
    g = _guardian(BalEx())
    for _ in range(3):
        g.run_cycle()
    assert g._wallet_balance_cached == pytest.approx(4148.61), "spike was used"


# ── D3: untracked positions ──────────────────────────────────────────────────

def test_a_position_with_no_state_is_reported_and_adopted():
    """One ran to +70% ROI untracked, unprotected, absent from the history."""
    fake = FakeExchange(positions=[_raw_pos("short", entry=1.0)], price=1.0)
    g = _guardian(fake)
    g.run_cycle()
    assert "DOGE/USDT:USDT" in g._states
    assert any(a.get("action") == "untracked_adopted" for a in g._actions)


# ── D2: a position row is never discarded silently ───────────────────────────

def test_missing_side_is_recovered_from_positionAmt():
    """
    On one-way mode the direction lives in the SIGN of positionAmt. Discarding
    the row when ccxt fails to derive it leaves a real position unguarded.
    """
    class OddEx(FakeExchange):
        def fetch_positions(self, symbols=None):
            return [{"symbol": "PUFFER/USDT:USDT", "side": None,
                     "contracts": 156526.0, "entryPrice": 0.31, "leverage": 20,
                     "initialMargin": 153.94,
                     "info": {"positionAmt": "-156526", "entryPrice": "0.31",
                              "positionSide": "BOTH", "symbol": "PUFFERUSDT"}}]
    pos = _guardian(OddEx()).fetch_positions()
    assert len(pos) == 1 and pos[0].side == "short"


# ── D6: destructive action requires sight ────────────────────────────────────

def test_repeated_failures_make_the_guardian_blind():
    g = _guardian(FakeExchange())
    assert not g.is_blind
    for _ in range(g.BLIND_AFTER_FAILURES):
        g.note_api_failure("fetch_positions")
    assert g.is_blind
    g.note_api_success()
    assert not g.is_blind


# ── Fail-fast needs BOTH conditions ──────────────────────────────────────────

def _ff_guardian(age, roi, was_green=False):
    from bot.futures_guard import GuardConfig, price_for_roi
    import time as _t
    cfg = GuardConfig(fail_fast_s=180, fail_fast_max_peak_roi=0.0,
                      fail_fast_loss_roi=5.0, atr_stop_mult=0.0,
                      initial_stop_roi=25)
    fake = FakeExchange(positions=[_raw_pos("long", entry=1.0)], price=1.0)
    g = _guardian(fake, cfg=cfg)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    if was_green:
        fake._price = price_for_roi(pos, 4.0)
        g.run_cycle()
    g._pos_meta["DOGE/USDT:USDT"]["opened_seen_at"] = _t.time() - age
    return g, pos, g._states["DOGE/USDT:USDT"]


def test_fail_fast_fires_when_both_conditions_hold():
    g, pos, st = _ff_guardian(age=300, roi=-8)
    assert g._should_fail_fast(pos, st, -8) is True


def test_fail_fast_waits_for_the_timeout():
    g, pos, st = _ff_guardian(age=60, roi=-8)
    assert g._should_fail_fast(pos, st, -8) is False


def test_fail_fast_ignores_a_merely_slow_trade():
    """A trade at -2% is slow, not failing — cutting it pays a fee for nothing."""
    g, pos, st = _ff_guardian(age=300, roi=-2)
    assert g._should_fail_fast(pos, st, -2) is False


def test_fail_fast_leaves_a_trade_that_went_green():
    g, pos, st = _ff_guardian(age=300, roi=-8, was_green=True)
    assert g._should_fail_fast(pos, st, -8) is False


def test_fail_fast_off_by_default():
    from bot.futures_guard import GuardConfig
    assert GuardConfig().fail_fast_s == 0


# ── Arm-at-entry, through a full cycle ───────────────────────────────────────
# The fixture disables it so the poll-arm tests above keep testing the
# poll-arm path. These turn it on and check the WHOLE sequence, which is what
# was missing when GUARD_ARM_AT_ENTRY was first shipped: 15 tests failed on
# order sequence and were reasoned about rather than run.

def _armed_guardian(fake, **kw):
    return _guardian(fake, arm_at_entry=True, **kw)


def test_arm_at_entry_places_the_trail_before_the_fixed_stop():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _armed_guardian(fake)
    g.run_cycle()
    types = [o["type"] for o in fake.created]
    assert types, "nothing was placed"
    assert types[0] == "TRAILING_STOP_MARKET"
    assert "STOP_MARKET" in types, "the fixed stop must still be placed"


def test_the_armed_trail_carries_an_activate_price_at_the_arm_level():
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0, lev=10)],
                        price=100.0)
    g = _armed_guardian(fake)
    g.run_cycle()
    trails = [o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"]
    armed = [o for o in trails if "activatePrice" in (o["params"] or {})]
    assert armed, "the armed trail must send an activatePrice"
    act = float(armed[0]["params"]["activatePrice"])
    # short, +arm_roi ROI sits BELOW entry
    assert act < 100.0
    expected = 100.0 * (1 - g.cfg.arm_roi / 10 / 100)
    assert abs(act - expected) / expected < 0.001


def test_a_long_arms_above_entry_through_the_cycle():
    fake = FakeExchange(positions=[_raw_pos("long", entry=100.0, lev=10)],
                        price=100.0)
    g = _armed_guardian(fake)
    g.run_cycle()
    armed = [o for o in fake.created
             if o["type"] == "TRAILING_STOP_MARKET"
             and "activatePrice" in (o["params"] or {})]
    assert armed
    assert float(armed[0]["params"]["activatePrice"]) > 100.0


def test_every_order_it_places_is_reduce_only():
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _armed_guardian(fake)
    g.run_cycle()
    assert fake.created
    for o in fake.created:
        assert o["params"].get("reduceOnly") is True, o["type"]


def test_the_trail_is_not_placed_twice_across_cycles():
    """One trail per role. The poll-arm block must see it and stand down."""
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _armed_guardian(fake)
    g.run_cycle()
    first = len([o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"])
    g.run_cycle()
    g.run_cycle()
    after = len([o for o in fake.created if o["type"] == "TRAILING_STOP_MARKET"])
    assert after == first, f"placed {after - first} extra trail(s)"


def test_turning_it_off_restores_the_poll_arm_sequence():
    """The old path has to stay reachable, or the switch is a lie."""
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _guardian(fake, arm_at_entry=False)
    g.run_cycle()
    assert fake.created
    assert fake.created[0]["type"] == "STOP_MARKET"


def test_a_trail_that_cannot_be_placed_does_not_stop_the_fixed_stop():
    """Arm-at-entry is never fatal — the position still gets its stop."""
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _armed_guardian(fake)
    real = g._place_native_trail

    def _boom(pos, **kw):
        raise Exception("binance -2021")
    g._place_native_trail = _boom
    g.run_cycle()
    assert any(o["type"] == "STOP_MARKET" for o in fake.created)


def test_arm_at_entry_still_places_the_initial_fixed_stop():
    """
    REGRESSION. The early return in manage_position was written for stop ->
    trail -> supersede. Arm-at-entry inverts that order, so taking it on the
    first cycle skipped the stop placement entirely and left the position with
    two trails and no fixed stop. 龙虾 2026-09-19 03:25:08 shows the SIZED
    handoff line, both trails, and no "initial protective stop" line.
    """
    fake = FakeExchange(positions=[_raw_pos("short")], price=100.0)
    g = _armed_guardian(fake)
    g.run_cycle()
    types = [o["type"] for o in fake.created]
    assert "STOP_MARKET" in types, f"no fixed stop placed; got {types}"
    assert types[0] == "TRAILING_STOP_MARKET", "the trail still goes on first"


def test_the_handover_only_happens_once_the_trail_replaces_the_stop():
    """
    The flag, not the trail id, is what hands protection over — otherwise any
    resting trail suppresses the stop.
    """
    from bot.futures_guard import GuardState
    st = GuardState(native_trail_id="t1")
    assert st.armed_replaced_stop is False


def test_the_replacement_flag_survives_a_restart():
    import inspect
    import bot.futures_state as fs
    src = inspect.getsource(fs)
    assert '"armed_replaced_stop": s.armed_replaced_stop' in src
    assert 'st.armed_replaced_stop = bool(raw.get("armed_replaced_stop"))' in src


# ── A rejected armed trail must not cost the position its protection ─────────
# 牛来 2026-09-20 06:55:37. The algo endpoint returns 200 with an id and
# refuses the order AFTERWARDS — every TRAIL-RESPONSE in the logs reads
# status=None, so _accepted_id, written for exactly this, passes everything.
# One second later the adaptive trail was cancelled as "superseded" in favour
# of an order Binance had rejected, and the position held NOTHING for 27
# seconds while five profit-floor attempts were refused with -2021.

def test_a_rejected_armed_trail_keeps_the_adaptive_trail():
    from bot.futures_guard import price_for_roi
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23411)],
                        price=0.23411)
    g = _guardian(fake, cfg=_armed_cfg(), adaptive=True)
    g.run_cycle()                                   # adopt + adaptive trail
    st = g._states["DOGE/USDT:USDT"]
    adaptive = st.adaptive_trail_id
    assert adaptive, "fixture must have an adaptive trail to protect"
    fake.reject_next_algo = True                    # the arming attempt fails
    fake._price = price_for_roi(g.fetch_positions()[0], 8.0)
    g.run_cycle()
    assert st.adaptive_trail_id == adaptive, \
        "live protection was cancelled for an order that does not exist"
    assert adaptive not in [o for o, _ in fake.cancelled]


def test_an_ADOPTED_position_records_its_DERIVED_stop_and_callback():
    """
    An auto entry supplies sized_stop_roi and callback_pct via
    note_entry_context; an adopted one has neither. The six manual trades of
    2026-09-25 recorded both as None, so replay_exits.py — which reads
    callback_pct to reconstruct the trail — could not replay them at all.
    That is the one comparison MANUAL_INITIAL_GUARD_ONLY exists to make.
    """
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake, adaptive=True)
    pos = g.fetch_positions()[0]
    g._pos_meta.setdefault(pos.symbol, {})["entry_context"] = {
        "captured": "observed", "atr_pct": 1.0}
    st = GuardState()
    g._ensure_adaptive_trail(pos, st, stop_roi=20.0)
    ctx = g._pos_meta[pos.symbol]["entry_context"]
    assert ctx["sized_stop_roi"] == 20.0
    assert ctx["callback_pct"] == 2.0          # 20 / 10x leverage
    assert ctx["callback_source"] == "adaptive_derived"


def test_an_EXACT_handoff_is_never_overwritten_by_an_observation():
    # note_entry_context carries what the entry path actually sized from.
    # An observation must not clobber it.
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake, adaptive=True)
    pos = g.fetch_positions()[0]
    g._pos_meta.setdefault(pos.symbol, {})["entry_context"] = {
        "auto": True, "sized_stop_roi": 33.0, "callback_pct": 1.11}
    g._ensure_adaptive_trail(pos, GuardState(), stop_roi=20.0)
    ctx = g._pos_meta[pos.symbol]["entry_context"]
    assert ctx["sized_stop_roi"] == 33.0 and ctx["callback_pct"] == 1.11


# ── PROTECTION-DEAD: re-place an order the exchange has killed ─────────────

def _dead_setup(status, name="adaptive", oid="a-1"):
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    pos = g.fetch_positions()[0]
    st = GuardState()
    st.adaptive_trail_id = oid
    st.stop_order_id = "f-1"
    g._algo_status = lambda sym: status
    g._replace_if_dead(pos, st, name, oid)
    return st


def test_an_order_CONFIRMED_dead_is_dropped_so_it_is_re_placed():
    """
    BTW/USDT 2026-09-25. The listing went 5 -> 4 -> 3 orders while the
    position was open; PROTECTION-MISSING fired twice and nothing acted. The
    position ran to +7.8% ROI and closed at -6.6% — 14.4 points of give-back
    against a 3% callback — with the guardian still holding all five ids.
    """
    st = _dead_setup({"a-1": "CANCELED"})
    assert st.adaptive_trail_id is None
    assert st._replacements == 1


def test_MERE_ABSENCE_from_the_listing_is_NOT_enough():
    """
    That listing reports "0 protective order(s)" on positions that
    demonstrably have five — it cannot see the reduceOnly flag. Acting on
    absence alone would place duplicate stops on a healthy position.
    """
    st = _dead_setup({})                     # algo book says nothing
    assert st.adaptive_trail_id == "a-1"


def test_an_order_STILL_LIVE_is_left_alone():
    st = _dead_setup({"a-1": "NEW"})
    assert st.adaptive_trail_id == "a-1"


def test_replacement_is_CAPPED():
    # An order the exchange keeps refusing must not be re-placed on every
    # audit for the life of the position.
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    pos = g.fetch_positions()[0]
    st = GuardState()
    g._algo_status = lambda sym: {"a-1": "REJECTED"}
    for _ in range(10):
        st.adaptive_trail_id = "a-1"
        g._replace_if_dead(pos, st, "adaptive", "a-1")
    assert st._replacements == g.MAX_REPLACEMENTS


def test_an_unreadable_algo_book_changes_nothing():
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    pos = g.fetch_positions()[0]
    st = GuardState()
    st.adaptive_trail_id = "a-1"
    def boom(sym): raise RuntimeError("503")
    g._algo_status = boom
    g._replace_if_dead(pos, st, "adaptive", "a-1")
    assert st.adaptive_trail_id == "a-1"


# ── MANUAL_INITIAL_GUARD_ONLY — hand-opened trades, initial guard only ─────

def _guard_with_ctx(ctx, flag=True):
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.cfg.manual_initial_guard_only = flag
    pos = g.fetch_positions()[0]
    # opened_seen_at must survive: fail-fast needs an age, and a helper that
    # wipes it would make every fail-fast test pass for the wrong reason.
    meta = dict(g._pos_meta.get(pos.symbol) or {})
    # A real timestamp, not 0.0: the guardian guards with `if not opened`,
    # so epoch zero reads as MISSING and every fail-fast test would pass for
    # the wrong reason.
    meta["opened_seen_at"] = time.time() - 300
    if ctx is not None:
        meta["entry_context"] = ctx
    else:
        meta.pop("entry_context", None)
    g._pos_meta[pos.symbol] = meta
    return g, pos


def test_a_MANUAL_position_runs_with_the_initial_guard_only():
    g, pos = _guard_with_ctx({"sized_price": 100.0})      # no `auto`
    assert g._is_manual(pos) is True
    assert g._reduced_guard(pos) is True


def test_a_BOT_position_is_untouched():
    """
    The whole point is to change nothing for bot trades. `auto: True` is
    stamped on every auto entry.
    """
    g, pos = _guard_with_ctx({"auto": True, "sized_price": 100.0})
    assert g._is_manual(pos) is False
    assert g._reduced_guard(pos) is False


def test_a_MISSING_entry_context_is_UNKNOWN_not_manual():
    """
    Absence of a context is a missed handoff, not a hand-opened trade.
    Reading it as manual would silently drop fail-fast and the profit floor
    from a bot position — a far worse failure than the feature is worth.
    """
    g, pos = _guard_with_ctx(None)
    assert g._is_manual(pos) is False
    assert g._reduced_guard(pos) is False


def test_the_flag_OFF_leaves_even_a_manual_position_fully_guarded():
    g, pos = _guard_with_ctx({"sized_price": 100.0}, flag=False)
    assert g._is_manual(pos) is True, "still recognised as manual"
    assert g._reduced_guard(pos) is False, "but nothing changes with the flag off"


def test_fail_fast_is_skipped_for_a_reduced_guard_position():
    g, pos = _guard_with_ctx({"sized_price": 100.0})
    st = GuardState()
    st.peak_roi = 0.0
    assert g._should_fail_fast(pos, st, current_roi=-20.0) is False


def test_fail_fast_STILL_FIRES_for_a_bot_position():
    # The gate must be the manual check, not a blanket disable.
    g, pos = _guard_with_ctx({"auto": True, "sized_price": 100.0})
    st = GuardState()
    st.peak_roi = 0.0
    st.roi_checkpoints = {"0": 0.0}
    g.cfg.fail_fast_s = 60
    g.cfg.fail_fast_loss_roi = 5.0
    assert g._should_fail_fast(pos, st, current_roi=-20.0) is True


def test_the_default_is_OFF():
    from bot.config import BotConfig
    assert BotConfig().manual_initial_guard_only is False


def test_an_ASYNCHRONOUSLY_REJECTED_order_is_dropped_from_the_state():
    """
    PHA demo 2026-09-21 10:18. The guardian logged "FLOOR TRAIL resting
    id=...309", TRAIL-RESPONSE showed the activatePrice accepted and kept, and
    every PROTECTION line listed it as held. The algo HISTORY said:

        armed  ...298  FINISHED   <- actually closed the position
        FLOOR  ...309  REJECTED   <- never rested
        floor  ...532  REJECTED   <- profit floor, also refused

    _accepted_id cannot catch this: the POST returns 200 with a real id and
    the status flips afterwards.
    """
    from bot.futures_guard import GuardState
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    st = GuardState()
    st.native_trail_id = "armed-1"
    st.floor_trail_id = "floor-1"
    st.stop_order_id = "fixed-1"
    g._algo_status = lambda sym: {"armed-1": "FINISHED",
                                  "floor-1": "REJECTED",
                                  "fixed-1": "NEW"}
    g._verify_protection(g.fetch_positions()[0], st)
    assert st.floor_trail_id is None, "a REJECTED order must not be held"
    assert st.native_trail_id == "armed-1", "a live one must survive"
    assert st.stop_order_id == "fixed-1"


def test_verification_says_NOTHING_when_the_algo_book_is_unreadable():
    # Silence beats guessing: an empty reply must not look like a rejection
    # and drop live protection.
    from bot.futures_guard import GuardState
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    st = GuardState()
    st.native_trail_id = "armed-1"
    g._algo_status = lambda sym: {}
    g._verify_protection(g.fetch_positions()[0], st)
    assert st.native_trail_id == "armed-1"


def test_an_id_absent_from_the_history_is_left_alone():
    # Absent is not refused — the history has a retention window.
    from bot.futures_guard import GuardState
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    st = GuardState()
    st.native_trail_id = "armed-1"
    g._algo_status = lambda sym: {"something-else": "REJECTED"}
    g._verify_protection(g.fetch_positions()[0], st)
    assert st.native_trail_id == "armed-1"


def test_every_dead_algo_status_is_recognised():
    from bot.futures_guardian import FuturesGuardian
    assert {"REJECTED", "EXPIRED", "CANCELED", "CANCELLED"} <= \
        FuturesGuardian._DEAD_ALGO


def test_a_missing_profit_floor_is_VISIBLE_not_just_logged():
    """
    PHA 2026-09-21 08:29: the floor was refused (-2021 "would immediately
    trigger") and the position ran ~2 minutes with no break-even floor. The
    guardian logged PROTECTION-NO-FLOOR at ERROR and the dashboard showed
    NOTHING, because snapshot() only ever exposed unprotected_reason and this
    path never set it.
    """
    from bot.futures_guard import GuardState
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    st = GuardState(peak_roi=6.0)
    pos = g.fetch_positions()[0]
    g._floor_unavailable(pos, st, current=3.2, wanted=2.0,
                         err="-2021: Order would immediately trigger.")
    assert st.floor_missing_reason, "the condition must be on the state"
    assert st.unprotected_reason is None, \
        "a missing floor is NOT an unprotected position — the stop is resting"


def test_the_missing_floor_reaches_the_dashboard_payload():
    from bot.futures_guard import GuardState
    fake = FakeExchange(positions=[_raw_pos("short", entry=100.0)], price=100.0)
    g = _guardian(fake)
    g.run_cycle()
    sym = "DOGE/USDT:USDT"
    g._states[sym].floor_missing_reason = "no break-even floor (peak +6.0%)"
    snap = g.snapshot()
    assert snap["states"][sym]["floor_missing_reason"] == \
        "no break-even floor (peak +6.0%)"
    assert "floor_attempts" in snap["states"][sym]


def test_an_unconfirmed_trail_does_not_claim_it_replaced_the_stop():
    """
    armed_replaced_stop suppresses the fixed stop. Setting it for a trail that
    never rested would leave the position with neither.
    """
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23411)],
                        price=0.23411)
    from bot.futures_guard import price_for_roi
    g = _guardian(fake, cfg=_armed_cfg(), adaptive=True)
    g.run_cycle()
    st = g._states["DOGE/USDT:USDT"]
    fake.reject_next_algo = True
    fake._price = price_for_roi(g.fetch_positions()[0], 8.0)
    g.run_cycle()
    assert st.armed_replaced_stop is False


def test_a_confirmed_trail_still_supersedes_normally():
    """The fix must not break arming when the order really is resting."""
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.23411)],
                        price=0.23411)
    g = _arm(fake, adaptive=True)
    st = g._states["DOGE/USDT:USDT"]
    assert st.native_trail_id
    assert st.adaptive_trail_id is None, "confirmed arming should supersede"


def test_a_lagging_book_keeps_the_trail_recorded_and_retries():
    """
    The book can lag a placement by a cycle. Discarding the id then would mean
    arming never sticks — so the trail is RECORDED either way and only the
    cancels wait for confirmation.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian.manage_position)
    i = src.index("if trail_id:")
    j = src.index("if trail_id and superseded_ok:")
    assert "state.native_trail_id = trail_id" in src[i:j], \
        "the id must be recorded before the confirmation gate"


def test_confirmation_failure_is_treated_like_rejection():
    """An unreadable book and a refused order must reach the same safe end."""
    from bot.futures_guardian import FuturesGuardian

    class _Boom(FakeExchange):
        def fapiPrivateGetOpenAlgoOrders(self, params=None):
            raise RuntimeError("algo book unavailable")
    fake = _Boom(positions=[_raw_pos("long", entry=0.23411)], price=0.23411)
    g = _arm(fake, adaptive=True)
    st = g._states["DOGE/USDT:USDT"]
    assert st.adaptive_trail_id, "protection must be kept when unconfirmable"


# ── The superseded sweep must not eat the armed trail ────────────────────────
# _cancel_superseded_stops protected floor_stop_id and adaptive_trail_id but
# NOT native_trail_id. Arm-at-entry places the armed trail at adoption, it
# lands in _all_stop_ids, and the very next fixed-stop placement swept it —
# 4 seconds later on LSK live 2026-09-20 08:03:23, and on demo at 07:50:12.
# Both instances, every position. Arm-at-entry never survived its own first
# cycle, which is why the poll-driven ratchet went on doing all the work.
#
# Same class as the SOLV case the method's own docstring records: a sweep that
# did not know about a protection added after it was written.

def test_the_sweep_keeps_the_armed_trail():
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.38075)],
                        price=0.38075)
    g = _guardian(fake, arm_at_entry=True, adaptive=True)
    g.run_cycle()
    st = g._states["DOGE/USDT:USDT"]
    assert st.native_trail_id, "arm-at-entry placed nothing"
    assert st.native_trail_id not in [o for o, _ in fake.cancelled], \
        "the armed trail was swept by the fixed-stop placement"


def test_all_three_protections_coexist_after_the_first_cycle():
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.38075)],
                        price=0.38075)
    g = _guardian(fake, arm_at_entry=True, adaptive=True)
    g.run_cycle()
    st = g._states["DOGE/USDT:USDT"]
    assert st.adaptive_trail_id and st.native_trail_id and st.stop_order_id
    assert not fake.cancelled, f"nothing should be cancelled: {fake.cancelled}"


def test_every_poll_independent_protection_is_in_the_protected_set():
    """
    The sweep has now lost the adaptive trail once (SOLV) and the armed trail
    once (LSK). Both were added to state after the sweep was written.
    """
    import inspect
    from bot.futures_guardian import FuturesGuardian
    src = inspect.getsource(FuturesGuardian._cancel_superseded_stops)
    i = src.index("protected = {")
    block = src[i:src.index("for oid in ids", i)]
    for field in ("floor_stop_id", "adaptive_trail_id", "native_trail_id"):
        assert field in block, f"{field} can be swept"


# ── The native trail replaces the ratchet ────────────────────────────────────
# Measured on LSK live 2026-09-20: the ratchet exited at +6.64% ROI after FIVE
# cancel/replace cycles. One native trail at the same 3% ROI callback would
# have exited at +6.89% — 0.025% apart, with ONE order and no polling.
#
# Each ratchet step is also a window where the old stop is cancelled and the
# new one is unconfirmed, and a placement can return an id without resting
# (牛来 2026-09-20). Removing it removes risk as well as churn.

def _run_up(ratchet, rois=(6, 8, 10, 12, 14)):
    from dataclasses import replace
    from bot.futures_guard import price_for_roi
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.38075)],
                        price=0.38075)
    g = _guardian(fake, cfg=replace(_armed_cfg(), ratchet_enabled=ratchet))
    g.run_cycle()
    pos = g.fetch_positions()[0]
    for roi in rois:
        fake._price = price_for_roi(pos, roi)
        g.run_cycle()
    stops = [o["id"] for o in fake.created if o["type"] == "STOP_MARKET"]
    return g, fake, stops


def test_the_ratchet_is_off_by_default():
    from bot.futures_guard import GuardConfig
    assert GuardConfig().ratchet_enabled is False


def test_a_resting_trail_stops_the_repositioning():
    _, fake, stops = _run_up(ratchet=False)
    assert len(stops) == 1, f"repositioned {len(stops) - 1} time(s) anyway"
    assert len(fake.cancelled) <= 1


def test_the_ratchet_still_works_when_switched_on():
    """The old behaviour has to stay reachable, or the switch is a lie."""
    _, fake, stops = _run_up(ratchet=True)
    assert len(stops) > 1, "the ratchet did not reposition"
    assert len(fake.cancelled) > 1


def test_turning_the_ratchet_off_does_not_cost_the_fixed_stop():
    """
    REGRESSION, found by running the cycle rather than by the suite: the
    supersede sweep cancelled the fixed stop while state.stop_order_id still
    named it — the position lost its stop AND the record disagreed with the
    exchange. The sweep exists to retry a cancel that failed at ARMING, so it
    must only run when the trail actually replaced something.
    """
    from dataclasses import replace
    from bot.futures_guard import price_for_roi
    fake = FakeExchange(positions=[_raw_pos("long", entry=0.38075)],
                        price=0.38075)
    g = _guardian(fake, arm_at_entry=True, adaptive=True)
    g.cfg = replace(g.cfg, ratchet_enabled=False)
    g.run_cycle()
    pos = g.fetch_positions()[0]
    for roi in (2, 4, 6, 8, 10):
        fake._price = price_for_roi(pos, roi)
        g.run_cycle()
    st = g._states["DOGE/USDT:USDT"]
    assert st.stop_order_id, "no fixed stop on record"
    assert st.stop_order_id not in [o for o, _ in fake.cancelled], \
        "the fixed stop was swept while the state still claimed it"
    assert st.adaptive_trail_id and st.native_trail_id
