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
