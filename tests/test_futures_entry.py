"""
Tests for operator-initiated futures entry.

The guardrails are asserted as behaviour: an entry cannot be opened without a
valid unexpired token, cannot exceed the size cap, cannot double-open a symbol,
and cannot bypass the position limit — even if the caller tries to skip preview.
"""
import time
import pytest

from bot.futures_entry import (
    EntryService, EntryLimits, compute_size, order_side_for, validate_request,
    CONFIRM_TTL_S,
)
from bot.futures_guard import GuardConfig
from bot.futures_guardian import FuturesGuardian


class FakeExchange:
    def __init__(self, balance=100.0, positions=None, price=2.0, leverage=10):
        self._balance = balance
        self._positions = positions or []
        self._price = price
        self._leverage = leverage
        self.created = []
        self.fail_next = False
        # Any symbol under test is tradable on this fake account.
        self.markets = {}

    def load_markets(self):
        return self.markets

    def market_exists(self, sym):
        return True

    def fetch_balance(self):
        return {"USDT": {"total": self._balance, "free": self._balance}}

    def fetch_positions(self, symbols=None):
        if symbols:
            return [{"symbol": symbols[0], "leverage": self._leverage,
                     "side": None, "contracts": 0, "entryPrice": 0}]
        return self._positions

    def fetch_ticker(self, symbol):
        return {"last": self._price}

    def price_to_precision(self, symbol, p):
        return f"{float(p):.6f}"

    def amount_to_precision(self, symbol, a):
        return f"{float(a):.3f}"

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("exchange rejected")
        rec = {"id": "E-1", "symbol": symbol, "type": type, "side": side,
               "amount": amount, "params": params or {}}
        self.created.append(rec)
        return rec

    def set_sandbox_mode(self, on):
        pass


def _svc(fake, dry_run=False, limits=None):
    g = FuturesGuardian.__new__(FuturesGuardian)
    g.cfg = GuardConfig().validate()
    g.exchange = fake
    g.dry_run = dry_run
    g.testnet = True
    import threading
    g._lock = threading.RLock()
    g._states = {}
    g._actions = []
    g._last_cycle_ts = 0.0
    g._last_error = None
    return EntryService(g, limits or EntryLimits())


def _raw_pos(symbol="DOGE/USDT:USDT"):
    return {"symbol": symbol, "side": "short", "entryPrice": 2.0,
            "contracts": 100.0, "leverage": 10, "initialMargin": 20.0}


# ── sizing maths ─────────────────────────────────────────────────────────────

def test_compute_size():
    margin, notional, qty = compute_size(100.0, 10.0, 10.0, 2.0)
    assert margin == pytest.approx(10.0)      # 10% of 100
    assert notional == pytest.approx(100.0)   # x10 leverage
    assert qty == pytest.approx(50.0)         # 100 / 2.0


def test_order_side_mapping():
    assert order_side_for("long") == "buy"
    assert order_side_for("short") == "sell"
    with pytest.raises(ValueError):
        order_side_for("sideways")


# ── validation ───────────────────────────────────────────────────────────────

def test_rejects_oversized_margin():
    errs = validate_request(side="short", margin_pct=40.0, callback_pct=0.1,
                            wallet_balance=100, open_positions=0,
                            symbol_has_position=False, limits=EntryLimits())
    assert any("exceeds" in e for e in errs)


def test_rejects_callback_below_minimum():
    errs = validate_request(side="short", margin_pct=10.0, callback_pct=0.01,
                            wallet_balance=100, open_positions=0,
                            symbol_has_position=False, limits=EntryLimits())
    assert any("below" in e for e in errs)


def test_rejects_when_symbol_already_has_position():
    errs = validate_request(side="short", margin_pct=10.0, callback_pct=0.1,
                            wallet_balance=100, open_positions=0,
                            symbol_has_position=True, limits=EntryLimits())
    assert any("already open" in e for e in errs)


def test_rejects_at_position_limit():
    errs = validate_request(side="short", margin_pct=10.0, callback_pct=0.1,
                            wallet_balance=100, open_positions=1,
                            symbol_has_position=False, limits=EntryLimits(max_positions=1))
    assert any("position limit" in e for e in errs)


# ── two-step flow ────────────────────────────────────────────────────────────

def test_preview_sends_no_order():
    fake = FakeExchange()
    svc = _svc(fake)
    res = svc.preview("DOGE/USDT:USDT", "short")
    assert res["ok"]
    assert fake.created == [], "preview must not place an order"
    assert res["plan"]["token"]


def test_preview_shows_projected_stop():
    """The operator sees the downside before confirming, not after."""
    svc = _svc(FakeExchange())
    plan = svc.preview("DOGE/USDT:USDT", "short")["plan"]
    assert plan["projected_stop_roi"] == pytest.approx(-7.0)
    # short -> protective stop sits ABOVE entry
    assert plan["projected_stop_price"] > plan["ref_price"]


def test_execute_requires_valid_token():
    svc = _svc(FakeExchange())
    res = svc.execute("not-a-real-token")
    assert not res["ok"]
    assert any("expired or already used" in e for e in res["errors"])


def test_execute_places_trailing_stop_entry():
    fake = FakeExchange()
    svc = _svc(fake)
    token = svc.preview("DOGE/USDT:USDT", "short")["plan"]["token"]
    res = svc.execute(token)
    assert res["ok"]
    o = fake.created[0]
    assert o["type"] == "TRAILING_STOP_MARKET"
    assert o["side"] == "sell"                     # sell opens a short
    assert o["params"]["callbackRate"] == pytest.approx(0.1)
    assert o["params"]["reduceOnly"] is False      # this is an ENTRY, not protection


def test_token_is_single_use():
    fake = FakeExchange()
    svc = _svc(fake)
    token = svc.preview("DOGE/USDT:USDT", "short")["plan"]["token"]
    assert svc.execute(token)["ok"]
    second = svc.execute(token)
    assert not second["ok"]
    assert len(fake.created) == 1, "a replayed token must not open a second position"


def test_token_expires():
    svc = _svc(FakeExchange())
    token = svc.preview("DOGE/USDT:USDT", "short")["plan"]["token"]
    svc._pending[token].created_at = time.time() - (CONFIRM_TTL_S + 1)
    res = svc.execute(token)
    assert not res["ok"]


# ── re-validation at execute time ────────────────────────────────────────────

def test_position_opened_between_preview_and_execute_blocks_entry():
    """
    The account can change after the preview. Limits are re-checked at execute,
    so a position opened in between blocks the entry.
    """
    fake = FakeExchange()
    svc = _svc(fake)
    token = svc.preview("DOGE/USDT:USDT", "short")["plan"]["token"]
    fake._positions = [_raw_pos("DOGE/USDT:USDT")]      # appeared meanwhile
    res = svc.execute(token)
    assert not res["ok"]
    assert fake.created == []


def test_hitting_position_limit_between_steps_blocks_entry():
    fake = FakeExchange()
    svc = _svc(fake, limits=EntryLimits(max_positions=1))
    token = svc.preview("DOGE/USDT:USDT", "short")["plan"]["token"]
    fake._positions = [_raw_pos("OTHER/USDT:USDT")]     # different symbol, at limit
    res = svc.execute(token)
    assert not res["ok"]
    assert fake.created == []


# ── dry run ──────────────────────────────────────────────────────────────────

def test_dry_run_sends_nothing():
    fake = FakeExchange()
    svc = _svc(fake, dry_run=True)
    token = svc.preview("DOGE/USDT:USDT", "short")["plan"]["token"]
    res = svc.execute(token)
    assert res["ok"] and res["dry_run"]
    assert fake.created == []


# ── failure handling ─────────────────────────────────────────────────────────

def test_exchange_rejection_reported_not_raised():
    fake = FakeExchange()
    fake.fail_next = True
    svc = _svc(fake)
    token = svc.preview("DOGE/USDT:USDT", "short")["plan"]["token"]
    res = svc.execute(token)
    assert not res["ok"]
    assert any("RuntimeError" in e for e in res["errors"])


def test_zero_balance_blocks_preview():
    svc = _svc(FakeExchange(balance=0.0))
    res = svc.preview("DOGE/USDT:USDT", "short")
    assert not res["ok"]


def test_long_entry_buys():
    fake = FakeExchange()
    svc = _svc(fake)
    token = svc.preview("DOGE/USDT:USDT", "long")["plan"]["token"]
    svc.execute(token)
    assert fake.created[0]["side"] == "buy"


# ── Regression: leverage misread as 1 mis-sized the entry ────────────────────

class _LevEx:
    """Fake exchange reproducing the observed UNI case."""
    def __init__(self, lev_field=1, info_lev="10", step_int=True):
        self.lev_field, self.info_lev, self.step_int = lev_field, info_lev, step_int
        self.created = []
    def fetch_balance(self):
        return {"USDT": {"total": 107.09}, "info": {"totalWalletBalance": "107.09"}}
    markets = {"UNI/USDT:USDT": {}, "X/USDT:USDT": {}, "DOGE/USDT:USDT": {},
               "AAA/USDT:USDT": {}, "BBB/USDT:USDT": {}}
    def load_markets(self): return self.markets
    def fetch_ticker(self, s): return {"last": 7.055}
    def price_to_precision(self, s, p): return f"{float(p):.4f}"
    def amount_to_precision(self, s, a):
        return str(int(float(a))) if self.step_int else f"{float(a):.3f}"
    def create_order(self, **kw):
        self.created.append(kw); return {"id": "E1"}
    def fetch_positions(self, syms=None):
        return [{"symbol": (syms or ["X"])[0], "leverage": self.lev_field,
                 "info": {"leverage": self.info_lev}}]


class _LevGuardian:
    dry_run = True
    def __init__(self, ex):
        from bot.futures_guard import GuardConfig
        self.exchange = ex
        self.cfg = GuardConfig(initial_stop_roi=10.0, arm_roi=15.0, callback_roi=10.0)
    def fetch_positions(self): return []


def _lev_svc(ex, max_positions=3):
    from bot.futures_entry import EntryService, EntryLimits
    return EntryService(_LevGuardian(ex), EntryLimits(
        max_positions=max_positions, max_margin_pct=25,
        default_margin_pct=10, default_callback_pct=0.1))


def test_prefers_raw_info_leverage_over_parsed_field():
    """
    ccxt reported leverage=1 on an isolated 10x position, which sized the entry
    at a tenth of the intended notional (then floored to the minimum lot,
    producing 1 UNI / 0.70 USDT margin instead of ~15 UNI / ~10.6 USDT).
    """
    svc = _lev_svc(_LevEx(lev_field=1, info_lev="10"))
    plan = svc.preview(symbol="UNI/USDT:USDT", side="long")["plan"]
    assert plan["leverage"] == pytest.approx(10.0)
    assert plan["qty"] == pytest.approx(15.0)
    assert plan["margin_usdt"] == pytest.approx(10.58, abs=0.05)
    assert plan["notional_usdt"] == pytest.approx(105.8, abs=0.5)


def test_unresolvable_leverage_refuses_rather_than_assuming_1x():
    """Silently defaulting to 1x mis-sizes the order — must refuse instead."""
    svc = _lev_svc(_LevEx(lev_field=None, info_lev=None))
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["ok"] is False
    assert any("leverage" in e for e in res["errors"])


def test_reports_post_rounding_size_and_warns_on_big_shrink():
    """Plan must report what will ACTUALLY open, not the pre-rounding request."""
    # Price high enough that an integer lot step shaves a lot off.
    ex = _LevEx(lev_field=1, info_lev="10")
    ex.fetch_ticker = lambda s: {"last": 40.0}     # 26.7 -> 26 lots
    svc = _lev_svc(ex)
    res = svc.preview(symbol="X/USDT:USDT", side="long")
    plan = res["plan"]
    assert plan["qty"] == float(int(plan["qty"]))                  # integer lots
    assert plan["notional_usdt"] == pytest.approx(plan["qty"] * 40.0, rel=1e-6)
    assert plan["margin_usdt"] == pytest.approx(plan["notional_usdt"] / 10.0, rel=1e-6)


# ── Demo-endpoint failures must be diagnosable ───────────────────────────────

def test_symbol_not_on_trading_account_is_named():
    """
    The scanner screens the LIVE market while entries execute on the trading
    account (often demo), whose symbol list can differ. That mismatch must say
    so, not surface as a vague "no price".
    """
    ex = _LevEx()
    ex.markets = {"ETH/USDT:USDT": {}}          # UNI absent here
    svc = _lev_svc(ex)
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["ok"] is False
    assert any("not tradable" in e for e in res["errors"])


def test_symbol_alias_is_resolved():
    """A scanner symbol X/USDT:USDT should match a market listed as X/USDT."""
    ex = _LevEx()
    ex.markets = {"UNI/USDT": {}}
    svc = _lev_svc(ex)
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["ok"] is True
    assert res["plan"]["symbol"] == "UNI/USDT"


def test_unavailable_market_list_does_not_block():
    """The exchange is the authority — an unknown market list must not refuse."""
    ex = _LevEx()
    ex.markets = {}
    svc = _lev_svc(ex)
    assert svc.preview(symbol="UNI/USDT:USDT", side="long")["ok"] is True


def test_ticker_failure_reports_the_real_error():
    ex = _LevEx()
    def boom(s): raise RuntimeError("Invalid symbol")
    ex.fetch_ticker = boom
    svc = _lev_svc(ex)
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["ok"] is False
    assert any("Invalid symbol" in e for e in res["errors"])


def test_assumed_leverage_used_only_when_exchange_reports_none():
    """
    Demo can report no leverage for a symbol with no open position. An
    operator-DECLARED value is acceptable; a silent default is not.
    """
    from bot.futures_entry import EntryService, EntryLimits
    ex = _LevEx(lev_field=None, info_lev=None)
    svc = EntryService(_LevGuardian(ex), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0))
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["ok"] is True
    assert res["plan"]["leverage"] == pytest.approx(10.0)


def test_without_declaration_missing_leverage_still_refuses():
    ex = _LevEx(lev_field=None, info_lev=None)
    svc = _lev_svc(ex)                      # assumed_leverage defaults to 0
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["ok"] is False
    assert any("leverage" in e for e in res["errors"])


# ── Assumed leverage must be labelled, never silent ──────────────────────────

def test_assumed_leverage_default_lets_demo_entries_size():
    """Demo reports no leverage for a symbol with no position; 10x default unblocks it."""
    from bot.futures_entry import EntryService, EntryLimits
    ex = _LevEx(lev_field=None, info_lev=None)
    svc = EntryService(_LevGuardian(ex), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0))
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["ok"] is True
    assert res["plan"]["leverage"] == pytest.approx(10.0)


def test_assumed_leverage_is_flagged_in_the_plan_and_warnings():
    """A guess must be visible before confirming, not discovered after the fill."""
    from bot.futures_entry import EntryService, EntryLimits
    ex = _LevEx(lev_field=None, info_lev=None)
    svc = EntryService(_LevGuardian(ex), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0))
    res = svc.preview(symbol="UNI/USDT:USDT", side="long")
    assert res["plan"]["leverage_assumed"] is True
    assert res["plan"]["leverage_source"] == "assumed"
    assert any("ASSUMED" in w for w in res.get("warnings", []))


def test_reported_leverage_is_not_flagged_as_assumed():
    ex = _LevEx(lev_field=1, info_lev="10")     # exchange does report it
    svc = _lev_svc(ex)
    plan = svc.preview(symbol="UNI/USDT:USDT", side="long")["plan"]
    assert plan["leverage"] == pytest.approx(10.0)
    assert plan["leverage_assumed"] is False
    assert plan["leverage_source"] == "info.leverage"


# ── ccxt filters out zero-size positions, hiding leverage ────────────────────

def test_raw_position_risk_recovers_leverage_ccxt_filtered_out():
    """
    ccxt's fetch_positions_risk drops rows with entryPrice <= 0, so a symbol
    with no open position loses its leverage — on LIVE as well as demo. The raw
    positionRisk endpoint still reports it and must be preferred over assuming.
    """
    from bot.futures_entry import EntryService, EntryLimits

    class Ex(_LevEx):
        def __init__(self):
            super().__init__(lev_field=None, info_lev=None)
            self.raw_called = False
        def market_id(self, s): return "UNIUSDT"
        def fetch_positions(self, symbols=None): return []      # ccxt filtered
        def fapiPrivateV3GetPositionRisk(self, params):
            self.raw_called = True
            return [{"symbol": "UNIUSDT", "leverage": "20", "entryPrice": "0.0"}]

    ex = Ex()
    svc = EntryService(_LevGuardian(ex), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0))
    lev, src = svc.symbol_leverage_detail("UNI/USDT:USDT")
    assert ex.raw_called
    assert lev == pytest.approx(20.0)          # the REAL value, not the assumed 10
    assert src == "fapiPrivateV3GetPositionRisk"


def test_assumption_only_used_when_raw_endpoint_also_fails():
    from bot.futures_entry import EntryService, EntryLimits

    class Ex(_LevEx):
        def __init__(self):
            super().__init__(lev_field=None, info_lev=None)
        def market_id(self, s): return "UNIUSDT"
        def fetch_positions(self, symbols=None): return []
        def fapiPrivateV3GetPositionRisk(self, params):
            raise RuntimeError("endpoint unavailable")

    svc = EntryService(_LevGuardian(Ex()), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0))
    lev, src = svc.symbol_leverage_detail("UNI/USDT:USDT")
    assert lev == pytest.approx(10.0) and src == "assumed"
