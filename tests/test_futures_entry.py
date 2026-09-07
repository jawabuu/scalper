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
    # Wording widened when resting entry orders started counting as committed.
    assert any("already exists on this symbol" in e for e in errs)


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

    def atr_pct(self, symbol):
        """Mirrors the real guardian: ATR% from candles, single source."""
        try:
            raw = self.exchange.fetch_ohlcv(symbol, "15m", limit=96)
        except Exception:
            return None
        if not raw or len(raw) < 15:
            return None
        trs, prev = [], None
        for _ts, _o, h, l, c, _v in raw:
            tr = h - l
            if prev is not None:
                tr = max(tr, abs(h - prev), abs(l - prev))
            trs.append(tr)
            prev = c
        if not trs or not prev:
            return None
        return (sum(trs[-14:]) / min(len(trs), 14)) / prev * 100

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


# ── Volatility-scaled stops and sizing ───────────────────────────────────────

def _atr_ex(half_range, price=100.0, balance=107.0, lev="10"):
    class Ex(_LevEx):
        markets = {"X/USDT:USDT": {}}
        def __init__(self):
            super().__init__(lev_field=None, info_lev=lev)
        def load_markets(self): return self.markets
        def market_id(self, s): return "XUSDT"
        def fetch_balance(self): return {"info": {"totalWalletBalance": str(balance)}}
        def fetch_ticker(self, s): return {"last": price}
        def amount_to_precision(self, s, a): return f"{float(a):.4f}"
        def fetch_ohlcv(self, s, tf, limit=96):
            return [[0, price, price * (1 + half_range), price * (1 - half_range),
                     price, 10.0] for _ in range(limit)]
    return Ex()


def _atr_svc(ex, mult=1.5, risk_pct=1.0):
    from bot.futures_entry import EntryService, EntryLimits
    return EntryService(_LevGuardian(ex), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0,
        atr_stop_mult=mult, risk_pct=risk_pct))


def test_volatile_coin_gets_wider_stop_and_smaller_position():
    calm = _atr_svc(_atr_ex(0.00175)).preview(symbol="X/USDT:USDT", side="long")["plan"]
    wild = _atr_svc(_atr_ex(0.01)).preview(symbol="X/USDT:USDT", side="long")["plan"]
    assert abs(wild["projected_stop_roi"]) > abs(calm["projected_stop_roi"])
    assert wild["margin_usdt"] < calm["margin_usdt"]


def test_dollar_risk_is_constant_across_volatility():
    """The point of ATR sizing: the money at risk does not change with the coin."""
    plans = [_atr_svc(_atr_ex(h)).preview(symbol="X/USDT:USDT", side="long")["plan"]
             for h in (0.00175, 0.0035, 0.01)]
    losses = [p["projected_stop_loss_usdt"] for p in plans]
    for l in losses:
        assert l == pytest.approx(losses[0], rel=0.02)


def test_atr_sizing_respects_the_margin_cap():
    """A very calm coin would want a huge position — the wallet cap still binds."""
    plan = _atr_svc(_atr_ex(0.00005)).preview(symbol="X/USDT:USDT", side="long")["plan"]
    assert plan["margin_usdt"] <= 107.0 * 0.25 + 0.01


def test_disabled_by_default_keeps_flat_margin():
    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(_LevGuardian(_atr_ex(0.01)), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0))   # atr_stop_mult = 0
    plan = svc.preview(symbol="X/USDT:USDT", side="long")["plan"]
    assert plan["margin_usdt"] == pytest.approx(10.7, rel=0.02)   # flat 10%


def test_stop_roi_is_clamped():
    """A freak ATR must not produce an absurd stop in either direction."""
    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(_LevGuardian(_atr_ex(0.05)), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0,
        atr_stop_mult=1.5, risk_pct=1.0,
        atr_stop_min_roi=4.0, atr_stop_max_roi=30.0))
    plan = svc.preview(symbol="X/USDT:USDT", side="long")["plan"]
    assert 4.0 <= abs(plan["projected_stop_roi"]) <= 30.0


# ── Entry must not create a position larger than the order cap ───────────────

def test_entry_size_trimmed_to_the_per_order_cap():
    """
    A position above the per-ORDER cap cannot have a single stop or close order
    placed. The entry must never create one.
    """
    class Ex(_LevEx):
        markets = {"X/USDT:USDT": {}}
        def __init__(self):
            super().__init__(lev_field=None, info_lev="20")
        def load_markets(self): return self.markets
        def market_id(self, s): return "XUSDT"
        def market(self, s): return {"limits": {"amount": {"max": 200000.0}}}
        def fetch_balance(self): return {"info": {"totalWalletBalance": "1700.0"}}
        def fetch_ticker(self, s): return {"last": 0.01303}
        def amount_to_precision(self, s, a): return f"{float(a):.0f}"

    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(_LevGuardian(Ex()), EntryLimits(
        max_positions=6, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=20.0))
    plan = svc.preview(symbol="X/USDT:USDT", side="short")["plan"]
    assert plan["qty"] <= 200000.0, "entry sized beyond the per-order cap"


def test_entry_unaffected_when_below_the_cap():
    class Ex(_LevEx):
        markets = {"X/USDT:USDT": {}}
        def __init__(self):
            super().__init__(lev_field=None, info_lev="10")
        def load_markets(self): return self.markets
        def market_id(self, s): return "XUSDT"
        def market(self, s): return {"limits": {"amount": {"max": 200000.0}}}
        def fetch_balance(self): return {"info": {"totalWalletBalance": "107.0"}}
        def fetch_ticker(self, s): return {"last": 7.055}

    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(_LevGuardian(Ex()), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=10.0))
    plan = svc.preview(symbol="X/USDT:USDT", side="short")["plan"]
    assert plan["qty"] == pytest.approx(15.0, abs=1.0)


# ── Sizing and stop must never disagree (regression) ─────────────────────────

def test_refuses_rather_than_flat_sizing_when_atr_unavailable():
    """
    The entry silently fell back to flat sizing when its ATR lookup failed,
    while the guardian still placed an ATR-scaled stop. A full-size position
    behind a -30% ROI stop cost 3x the configured risk on one trade.
    """
    class NoAtrGuardian(_LevGuardian):
        def atr_pct(self, symbol):
            return None            # transient failure

    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(NoAtrGuardian(_atr_ex(0.0035)), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=20.0,
        atr_stop_mult=1.5, risk_pct=1.0))
    res = svc.preview(symbol="X/USDT:USDT", side="long")
    assert res["ok"] is False
    assert any("ATR is unavailable" in e for e in res["errors"])


def test_flat_sizing_still_allowed_when_atr_sizing_is_off():
    """With volatility sizing disabled there is no mismatch to worry about."""
    class NoAtrGuardian(_LevGuardian):
        def atr_pct(self, symbol):
            return None

    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(NoAtrGuardian(_atr_ex(0.0035)), EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=20.0))   # atr_stop_mult = 0
    assert svc.preview(symbol="X/USDT:USDT", side="long")["ok"] is True


def test_entry_and_guardian_use_the_same_atr():
    """One source, one cache — two implementations could disagree."""
    ex = _atr_ex(0.0035)
    g = _LevGuardian(ex)
    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(g, EntryLimits(
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=20.0,
        atr_stop_mult=1.5, risk_pct=1.0))
    assert svc._atr_pct("X/USDT:USDT") == pytest.approx(g.atr_pct("X/USDT:USDT"))


def test_risk_is_honoured_when_the_stop_is_clamped_to_max():
    """
    A stop clamped to ATR_STOP_MAX_ROI must still size so the loss equals
    ENTRY_RISK_PCT — the clamp changes the stop, not the money at risk.
    """
    from bot.futures_entry import EntryService, EntryLimits
    svc = EntryService(_LevGuardian(_atr_ex(0.05)), EntryLimits(   # very volatile
        max_positions=3, max_margin_pct=25, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=20.0,
        atr_stop_mult=1.5, risk_pct=1.0,
        atr_stop_min_roi=4.0, atr_stop_max_roi=30.0))
    plan = svc.preview(symbol="X/USDT:USDT", side="long")["plan"]
    assert abs(plan["projected_stop_roi"]) == pytest.approx(30.0)
    # loss at that stop must be ~1% of the wallet, not 30% of a flat position
    assert plan["projected_stop_loss_usdt"] == pytest.approx(107.0 * 0.01, rel=0.05)


# ── Resting entry orders must block duplicates ───────────────────────────────

class _PendingEx(_LevEx):
    markets = {"BR/USDT:USDT": {}}
    def __init__(self):
        super().__init__(lev_field=None, info_lev="20")
        self.orders = []
    def load_markets(self): return self.markets
    def market(self, s): return {"limits": {"amount": {"max": 1e12}}}
    def market_id(self, s): return "BRUSDT"
    def fetch_balance(self): return {"info": {"totalWalletBalance": "4000.0"}}
    def fetch_ticker(self, s): return {"last": 0.23614}
    def amount_to_precision(self, s, a): return f"{float(a):.0f}"
    def fetch_open_orders(self, s): return self.orders
    def fetch_ohlcv(self, s, tf, limit=120):
        p = 0.23614
        return [[0, p, p * 1.002, p * 0.998, p, 10.0] for _ in range(limit)]
    def fetch_positions(self, symbols=None):
        return [{"symbol": "BR/USDT:USDT", "info": {"leverage": "20"}}] if symbols else []


def _pending_svc(ex):
    from bot.futures_entry import EntryService, EntryLimits
    return EntryService(_LevGuardian(ex), EntryLimits(
        max_positions=6, max_margin_pct=15, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=20.0,
        atr_stop_mult=1.5, risk_pct=1.0))


def test_resting_entry_order_blocks_a_second_entry():
    """
    A trailing-stop entry rests until price reaches it, so it is not a
    position. The duplicate guard only checked positions, so the same candidate
    re-qualified each scan and stacked orders — several then filled together,
    producing a position several times the intended size.
    """
    ex = _PendingEx()
    ex.orders = [{"id": "E1", "reduceOnly": False, "symbol": "BR/USDT:USDT"}]
    res = _pending_svc(ex).preview(symbol="BR/USDT:USDT", side="long")
    assert res["ok"] is False
    assert any("resting entry order" in e for e in res["errors"])


def test_protective_orders_do_not_block_entry():
    """Reduce-only stops are protection, not commitment — they must not block."""
    ex = _PendingEx()
    ex.orders = [{"id": "S1", "reduceOnly": True, "symbol": "BR/USDT:USDT"}]
    assert _pending_svc(ex).preview(symbol="BR/USDT:USDT", side="long")["ok"] is True


def test_reduce_only_read_from_info_when_absent_at_top_level():
    ex = _PendingEx()
    ex.orders = [{"id": "S1", "info": {"reduceOnly": "true"}, "symbol": "BR/USDT:USDT"}]
    assert _pending_svc(ex).preview(symbol="BR/USDT:USDT", side="long")["ok"] is True


def test_no_orders_means_entry_allowed():
    ex = _PendingEx()
    ex.orders = []
    assert _pending_svc(ex).preview(symbol="BR/USDT:USDT", side="long")["ok"] is True


def test_repeated_scans_place_only_one_order():
    ex = _PendingEx()
    svc = _pending_svc(ex)
    placed = 0
    for _ in range(4):
        r = svc.preview(symbol="BR/USDT:USDT", side="long")
        if r.get("ok"):
            svc.execute(r["plan"]["token"])
            ex.orders.append({"id": f"E{placed}", "reduceOnly": False,
                              "symbol": "BR/USDT:USDT"})
            placed += 1
    assert placed == 1, f"{placed} orders stacked for one candidate"


def test_local_guard_blocks_duplicate_when_exchange_query_is_stale():
    """
    Binance showed three BR entries filling in the same second. The exchange
    query is the primary guard, but a stale or failing reply must not allow a
    second order — the cost is a position several times the intended size.
    """
    ex = _PendingEx()
    ex.fetch_open_orders = lambda s: []      # exchange reports nothing (stale)
    svc = _pending_svc(ex)
    first = svc.preview(symbol="BR/USDT:USDT", side="long")
    assert first["ok"] is True
    svc.execute(first["plan"]["token"])
    second = svc.preview(symbol="BR/USDT:USDT", side="long")
    assert second["ok"] is False


def test_local_guard_expires():
    ex = _PendingEx()
    ex.fetch_open_orders = lambda s: []
    svc = _pending_svc(ex)
    svc._note_pending("BR/USDT:USDT")
    assert svc._recently_placed("BR/USDT:USDT")
    svc._recent_entries["BR/USDT:USDT"] -= svc.RECENT_ENTRY_TTL_S + 1
    assert not svc._recently_placed("BR/USDT:USDT")


def test_local_guard_cleared_on_close():
    ex = _PendingEx()
    svc = _pending_svc(ex)
    svc._note_pending("BR/USDT:USDT")
    svc.clear_pending("BR/USDT:USDT")
    assert not svc._recently_placed("BR/USDT:USDT")


# ── Stale entry orders must be reaped ────────────────────────────────────────

class _ReapEx(_PendingEx):
    def __init__(self):
        super().__init__()
        self.cancelled = []
    def cancel_order(self, oid, symbol):
        self.cancelled.append((str(oid), symbol))
        self.orders = [o for o in self.orders if str(o.get("id")) != str(oid)]


def _reap_svc(ex):
    from bot.futures_entry import EntryService, EntryLimits
    g = _LevGuardian(ex)
    g.dry_run = False
    return EntryService(g, EntryLimits(
        max_positions=6, max_margin_pct=15, default_margin_pct=10,
        default_callback_pct=0.1, assumed_leverage=20.0,
        atr_stop_mult=1.5, risk_pct=1.0))


def test_unfilled_entry_order_is_cancelled_after_ttl():
    """
    A GTC entry that never fills blocks its symbol from being traded again and,
    if it eventually triggers, opens a position sized for conditions long past.
    """
    import time
    ex = _ReapEx()
    svc = _reap_svc(ex)
    ex.orders = [{"id": "E1", "reduceOnly": False, "symbol": "BR/USDT:USDT"}]
    svc._placed_orders = {"BR/USDT:USDT": [{"id": "E1",
                                            "placed_at": time.time() - 1800}]}
    svc.reap_stale_entry_orders(ttl_s=900, symbols_with_positions=set())
    assert ("E1", "BR/USDT:USDT") in ex.cancelled


def test_fresh_entry_order_is_left_alone():
    import time
    ex = _ReapEx()
    svc = _reap_svc(ex)
    ex.orders = [{"id": "E1", "reduceOnly": False, "symbol": "BR/USDT:USDT"}]
    svc._placed_orders = {"BR/USDT:USDT": [{"id": "E1", "placed_at": time.time()}]}
    svc.reap_stale_entry_orders(ttl_s=900, symbols_with_positions=set())
    assert ex.cancelled == []


def test_leftover_entry_cancelled_once_a_position_exists():
    """Otherwise it can trigger and ADD to the position — stacking again."""
    import time
    ex = _ReapEx()
    svc = _reap_svc(ex)
    ex.orders = [{"id": "E1", "reduceOnly": False, "symbol": "BR/USDT:USDT"}]
    svc._placed_orders = {"BR/USDT:USDT": [{"id": "E1", "placed_at": time.time()}]}
    svc.reap_stale_entry_orders(ttl_s=900,
                                symbols_with_positions={"BR/USDT:USDT"})
    assert ("E1", "BR/USDT:USDT") in ex.cancelled


def test_orders_the_bot_did_not_place_are_never_cancelled():
    """
    The operator's own trailing-stop entries are non-reduce-only too.
    Cancelling one would destroy their entry method.
    """
    import time
    ex = _ReapEx()
    svc = _reap_svc(ex)
    ex.orders = [{"id": "MANUAL-1", "reduceOnly": False, "symbol": "BR/USDT:USDT"}]
    svc._placed_orders = {}                      # bot placed nothing
    svc.reap_stale_entry_orders(ttl_s=0, symbols_with_positions={"BR/USDT:USDT"})
    assert ex.cancelled == []


def test_filled_order_is_forgotten_not_cancelled():
    """
    A FILL is proven by the position existing — not by the order being absent
    from fetch_open_orders, which does not reliably report conditional orders.
    """
    import time
    ex = _ReapEx()
    svc = _reap_svc(ex)
    ex.orders = []
    svc._placed_orders = {"BR/USDT:USDT": [{"id": "E1",
                                            "placed_at": time.time() - 60}]}
    svc.reap_stale_entry_orders(ttl_s=900,
                                symbols_with_positions={"BR/USDT:USDT"})
    assert svc.bot_placed_orders("BR/USDT:USDT") == []


def test_placed_orders_survive_a_restart_round_trip():
    ex = _ReapEx()
    svc = _reap_svc(ex)
    svc._placed_orders = {"BR/USDT:USDT": [{"id": "E1", "placed_at": 123.0}]}
    exported = svc.export_placed_orders()
    svc2 = _reap_svc(_ReapEx())
    svc2.import_placed_orders(exported)
    assert svc2.bot_placed_orders("BR/USDT:USDT")[0]["id"] == "E1"


def test_reaped_symbol_becomes_tradable_again():
    import time
    ex = _ReapEx()
    svc = _reap_svc(ex)
    ex.orders = [{"id": "E1", "reduceOnly": False, "symbol": "BR/USDT:USDT"}]
    svc._placed_orders = {"BR/USDT:USDT": [{"id": "E1",
                                            "placed_at": time.time() - 1800}]}
    assert svc.preview(symbol="BR/USDT:USDT", side="long")["ok"] is False
    svc.reap_stale_entry_orders(ttl_s=900, symbols_with_positions=set())
    svc.clear_pending("BR/USDT:USDT")
    assert svc.preview(symbol="BR/USDT:USDT", side="long")["ok"] is True


# ── ENTRY_ORDER_TTL_S bounds ─────────────────────────────────────────────────

def _ttl_with(value):
    import os
    import importlib
    saved = os.environ.get("ENTRY_ORDER_TTL_S")
    os.environ.update({"BINANCE_API_KEY_TEST": "k", "BINANCE_API_SECRET_TEST": "s"})
    try:
        if value is None:
            os.environ.pop("ENTRY_ORDER_TTL_S", None)
        else:
            os.environ["ENTRY_ORDER_TTL_S"] = value
        import bot.config
        importlib.reload(bot.config)
        return bot.config.BotConfig().entry_order_ttl_s
    finally:
        if saved is None:
            os.environ.pop("ENTRY_ORDER_TTL_S", None)
        else:
            os.environ["ENTRY_ORDER_TTL_S"] = saved
        import bot.config
        importlib.reload(bot.config)


def test_ttl_zero_is_floored_not_honoured():
    """0 would cancel every entry order on the next cycle, making entry impossible."""
    assert _ttl_with("0") == pytest.approx(60.0)


def test_negative_ttl_is_floored():
    assert _ttl_with("-60") == pytest.approx(60.0)


def test_short_ttl_is_floored():
    assert _ttl_with("30") == pytest.approx(60.0)


def test_normal_values_pass_through():
    assert _ttl_with("300") == pytest.approx(300.0)
    assert _ttl_with("1800") == pytest.approx(1800.0)


def test_unset_uses_the_default():
    assert _ttl_with(None) == pytest.approx(900.0)


def test_inline_comment_is_stripped():
    assert _ttl_with("600   # 10 minutes") == pytest.approx(600.0)


# ── The duplicate guard must not expire on a timer ───────────────────────────

def test_guard_holds_beyond_the_old_timer_window():
    """
    TRIA and VELVET each received a second entry order ~35 minutes after the
    first, matching the old 15-minute guard expiry. The block must last as long
    as the bot believes an order is resting, not for a fixed period.
    """
    import time
    ex = _PendingEx()
    ex.fetch_open_orders = lambda s: []          # exchange query blind
    svc = _pending_svc(ex)
    first = svc.preview(symbol="BR/USDT:USDT", side="long")
    svc.execute(first["plan"]["token"])

    # advance well past the old TTL
    svc._recent_entries["BR/USDT:USDT"] -= svc.RECENT_ENTRY_TTL_S + 1
    assert svc.preview(symbol="BR/USDT:USDT", side="long")["ok"] is False


def test_reaping_the_order_releases_the_symbol():
    """The reaper, not a clock, is what makes a symbol tradable again."""
    import time
    ex = _ReapEx()
    ex.fetch_open_orders = lambda s: []
    svc = _reap_svc(ex)
    first = svc.preview(symbol="BR/USDT:USDT", side="long")
    svc.execute(first["plan"]["token"])
    assert svc.preview(symbol="BR/USDT:USDT", side="long")["ok"] is False

    for rows in svc._placed_orders.values():
        for r in rows:
            r["placed_at"] = time.time() - 3600
    svc.reap_stale_entry_orders(ttl_s=600, symbols_with_positions=set())
    svc.clear_pending("BR/USDT:USDT")
    assert svc.preview(symbol="BR/USDT:USDT", side="long")["ok"] is True


def test_a_filled_order_also_releases_the_symbol():
    """The position appearing is what proves the fill and frees the symbol."""
    import time
    ex = _ReapEx()
    ex.fetch_open_orders = lambda s: []
    svc = _reap_svc(ex)
    svc._placed_orders = {"BR/USDT:USDT": [{"id": "E1",
                                            "placed_at": time.time() - 30}]}
    svc.reap_stale_entry_orders(ttl_s=600,
                                symbols_with_positions={"BR/USDT:USDT"})
    assert svc.bot_placed_orders("BR/USDT:USDT") == []


# ── Untracked stale orders ───────────────────────────────────────────────────

def test_untracked_stale_order_is_cancelled():
    """
    Orders placed before tracking existed rest forever and can fill hours
    later at a size and price that no longer apply — the cause of several
    recent losses.
    """
    import time
    ex = _ReapEx()
    old_ms = int((time.time() - 7200) * 1000)
    ex.orders = [{"id": "OLD-1", "reduceOnly": False, "symbol": "TRIA/USDT:USDT",
                  "timestamp": old_ms}]
    svc = _reap_svc(ex)
    svc._placed_orders = {}                       # no record of it
    svc.reap_untracked_entry_orders(ttl_s=600, symbols=["TRIA/USDT:USDT"])
    assert ("OLD-1", "TRIA/USDT:USDT") in ex.cancelled


def test_untracked_recent_order_is_left_alone():
    import time
    ex = _ReapEx()
    ex.orders = [{"id": "NEW-1", "reduceOnly": False, "symbol": "TRIA/USDT:USDT",
                  "timestamp": int(time.time() * 1000)}]
    svc = _reap_svc(ex)
    svc.reap_untracked_entry_orders(ttl_s=600, symbols=["TRIA/USDT:USDT"])
    assert ex.cancelled == []


def test_untracked_sweep_never_touches_protective_orders():
    import time
    ex = _ReapEx()
    old_ms = int((time.time() - 7200) * 1000)
    ex.orders = [{"id": "STOP-1", "reduceOnly": True, "symbol": "TRIA/USDT:USDT",
                  "timestamp": old_ms}]
    svc = _reap_svc(ex)
    svc.reap_untracked_entry_orders(ttl_s=600, symbols=["TRIA/USDT:USDT"])
    assert ex.cancelled == []


def test_untracked_sweep_releases_the_symbol():
    import time
    ex = _ReapEx()
    old_ms = int((time.time() - 7200) * 1000)
    ex.orders = [{"id": "OLD-1", "reduceOnly": False, "symbol": "BR/USDT:USDT",
                  "timestamp": old_ms}]
    svc = _reap_svc(ex)
    svc._note_pending("BR/USDT:USDT")
    svc.reap_untracked_entry_orders(ttl_s=600, symbols=["BR/USDT:USDT"])
    assert not svc._recently_placed("BR/USDT:USDT")


def test_order_without_a_timestamp_is_not_cancelled():
    """No age means no basis to judge — leave it rather than guess."""
    ex = _ReapEx()
    ex.orders = [{"id": "X-1", "reduceOnly": False, "symbol": "BR/USDT:USDT"}]
    svc = _reap_svc(ex)
    svc.reap_untracked_entry_orders(ttl_s=600, symbols=["BR/USDT:USDT"])
    assert ex.cancelled == []


# ── Absence from fetch_open_orders is not proof of filling ───────────────────

class _BlindEx(_ReapEx):
    """fetch_open_orders never reports the conditional order, as observed live."""
    def fetch_open_orders(self, s):
        return []


def test_stale_order_cancelled_even_when_query_is_blind():
    """
    The reaper treated a missing order as filled and silently forgot it, so
    nothing was ever cancelled and the symbol was unblocked while the order
    still rested on the exchange.
    """
    import time
    ex = _BlindEx()
    svc = _reap_svc(ex)
    svc._placed_orders = {"TRIA/USDT:USDT": [{"id": "E1",
                                              "placed_at": time.time() - 3600}]}
    svc.reap_stale_entry_orders(ttl_s=600, symbols_with_positions=set())
    assert ("E1", "TRIA/USDT:USDT") in ex.cancelled
    assert svc.bot_placed_orders("TRIA/USDT:USDT") == []


def test_young_order_is_kept_tracked_when_query_is_blind():
    """It must keep blocking duplicates rather than being forgotten."""
    import time
    ex = _BlindEx()
    svc = _reap_svc(ex)
    svc._placed_orders = {"TRIA/USDT:USDT": [{"id": "E1",
                                              "placed_at": time.time() - 60}]}
    svc.reap_stale_entry_orders(ttl_s=600, symbols_with_positions=set())
    assert ex.cancelled == []
    assert len(svc.bot_placed_orders("TRIA/USDT:USDT")) == 1
    assert svc._recently_placed("TRIA/USDT:USDT")


def test_position_appearing_clears_the_tracked_order():
    import time
    ex = _BlindEx()
    svc = _reap_svc(ex)
    svc._placed_orders = {"TRIA/USDT:USDT": [{"id": "E1",
                                              "placed_at": time.time() - 60}]}
    svc.reap_stale_entry_orders(ttl_s=600,
                                symbols_with_positions={"TRIA/USDT:USDT"})
    assert svc.bot_placed_orders("TRIA/USDT:USDT") == []


def test_unknown_order_reply_is_not_logged_as_a_failure():
    """-2011 means it really is gone; that is a clean outcome, not an error."""
    import time
    ex = _BlindEx()
    def boom(oid, symbol):
        raise RuntimeError("binanceusdm {\"code\":-2011,\"msg\":\"Unknown order sent.\"}")
    ex.cancel_order = boom
    svc = _reap_svc(ex)
    svc._placed_orders = {"TRIA/USDT:USDT": [{"id": "E1",
                                              "placed_at": time.time() - 3600}]}
    svc.reap_stale_entry_orders(ttl_s=600, symbols_with_positions=set())
    assert svc.bot_placed_orders("TRIA/USDT:USDT") == []
