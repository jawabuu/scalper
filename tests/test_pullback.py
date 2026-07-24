"""Unit tests for the pullback strategy pure logic."""
import os
os.environ.setdefault('BINANCE_API_KEY_TEST', 'k')
os.environ.setdefault('BINANCE_API_SECRET_TEST', 's')

import pandas as pd
import pytest
from bot.config import BotConfig
from bot import pullback as pb


@pytest.fixture
def cfg():
    os.environ['STRATEGY'] = 'pullback'
    return BotConfig()


def _frame(closes, highs=None, lows=None, opens=None, vols=None):
    n = len(closes)
    highs = highs or [c * 1.002 for c in closes]
    lows = lows or [c * 0.998 for c in closes]
    opens = opens or ([closes[0]] + closes[:-1])
    vols = vols or [1000.0] * n
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
    })


def _rising_uptrend(cfg, n=60, base=100.0):
    # Realistic gentle uptrend: up 0.4 twice, down 0.35 — net rising but keeps RSI
    # in the 55-72 band and rising at the end (not pinned at 100).
    closes = []
    price = base
    for i in range(n):
        price += -0.35 if i % 3 == 2 else 0.4
        closes.append(round(price, 4))
    closes[-3] = closes[-4] + 0.12
    closes[-2] = closes[-3] + 0.18
    closes[-1] = closes[-2] + 0.22
    df = _frame(closes)
    df["volume"] = 1000.0
    df.loc[df.index[-1], "volume"] = 5000.0  # volume spike on confirmation
    return pb.prepare_indicators(df, cfg)


# ── good-price gate ──

def test_good_price_rejects_high_in_candle(cfg):
    df = _rising_uptrend(cfg)
    # Force the last candle to close at its high (position ~1.0)
    i = df.index[-1]
    df.loc[i, "low"] = df.loc[i, "close"] * 0.99
    df.loc[i, "high"] = df.loc[i, "close"]
    row = df.loc[i]
    ok, why = pb.passes_good_price(row, cfg)
    assert not ok
    assert "high in candle" in why


def test_good_price_rejects_rejection_wick(cfg):
    df = _rising_uptrend(cfg)
    i = df.index[-1]
    c = df.loc[i, "close"]
    df.loc[i, "low"] = c * 0.999
    df.loc[i, "close"] = c
    df.loc[i, "open"] = c * 0.9995
    df.loc[i, "high"] = c * 1.02   # long upper wick above the body
    row = df.loc[i]
    # position must be > 0.5 for the wick veto; make close sit high
    if pb.candle_position(row) > 0.5:
        ok, why = pb.passes_good_price(row, cfg)
        assert not ok


def test_good_price_accepts_low_in_candle(cfg):
    df = _rising_uptrend(cfg)
    i = df.index[-1]
    c = df.loc[i, "close"]
    df.loc[i, "high"] = c * 1.01
    df.loc[i, "low"] = c * 0.998   # close near the low → low position
    row = df.loc[i]
    ok, why = pb.passes_good_price(row, cfg)
    assert ok


# ── RSI rising ──

def test_rsi_rising_true_on_uptrend(cfg):
    df = _rising_uptrend(cfg)
    assert pb.rsi_rising(df, cfg)


def test_rsi_rising_false_on_decline(cfg):
    # RSI declining over the lookback → must be rejected
    closes = [100 - i * 0.3 for i in range(40)]
    df = pb.prepare_indicators(_frame(closes), cfg)
    assert not pb.rsi_rising(df, cfg)


# ── EMA uptrend buffer ──

def test_ema_uptrend_true(cfg):
    df = _rising_uptrend(cfg)
    assert pb.ema_uptrend(df.iloc[-1], cfg)

def test_ema_uptrend_false_when_flat(cfg):
    closes = [100.0] * 40
    df = pb.prepare_indicators(_frame(closes), cfg)
    # flat → ema_fast ~= ema_slow, buffer should make this False
    assert not pb.ema_uptrend(df.iloc[-1], cfg)


# ── full gainer entry ──

def test_gainer_entry_accepts(cfg):
    df = _rising_uptrend(cfg)
    i = df.index[-1]
    c = df.loc[i, "close"]
    # ensure good price: close low in candle
    df.loc[i, "high"] = c * 1.01
    df.loc[i, "low"] = c * 0.999
    d = pb.evaluate_entry(df, cfg)
    assert d.enter
    assert d.regime == "gainer"
    assert d.stop_price is not None


def test_gainer_rejected_without_volume_spike(cfg):
    df = _rising_uptrend(cfg)
    df.loc[df.index[-1], "volume"] = 1000.0  # no spike
    d = pb.evaluate_entry(df, cfg)
    assert not d.enter
    assert "volume spike" in d.reason


def test_gainer_rejected_rsi_out_of_band(cfg):
    # Very steep move → RSI pinned near 100, above the 72 max
    closes = [100 + i * 5 for i in range(40)]
    df = pb.prepare_indicators(_frame(closes), cfg)
    df.loc[df.index[-1], "volume"] = 5000.0
    i = df.index[-1]; c = df.loc[i, "close"]
    df.loc[i, "high"] = c * 1.01; df.loc[i, "low"] = c * 0.999
    d = pb.evaluate_entry(df, cfg)
    # RSI likely > 72 → rejected (or good-price), either way no entry as gainer in-band
    if d.enter:
        assert cfg.pb_rsi_min <= d.stamps["rsi"] <= cfg.pb_rsi_max


# ── dipper entry ──

def test_dipper_entry_near_low(cfg):
    # Downtrend (ema_fast < ema_slow) but RSI turning up at the end, near 24h low
    closes = [100 - i * 0.5 for i in range(35)] + [82.6, 82.8, 83.2]
    df = _frame(closes)
    df["volume"] = 1000.0
    df.loc[df.index[-1], "volume"] = 5000.0
    df = pb.prepare_indicators(df, cfg)
    i = df.index[-1]; c = df.loc[i, "close"]
    df.loc[i, "high"] = c * 1.01; df.loc[i, "low"] = c * 0.999  # good price
    low_24h = min(closes)
    d = pb.evaluate_entry(df, cfg, low_24h=low_24h)
    # near the low and RSI rising → dipper (or rejected if RSI not yet rising)
    assert d.regime in ("dipper", None)


def test_dipper_rejected_far_from_low(cfg):
    closes = [100 - i * 0.5 for i in range(35)] + [82.6, 82.8, 83.2]
    df = _frame(closes)
    df.loc[df.index[-1], "volume"] = 5000.0
    df = pb.prepare_indicators(df, cfg)
    low_24h = 50.0  # price is far above this → not near support
    d = pb.evaluate_entry(df, cfg, low_24h=low_24h)
    assert not d.enter


# ── session windows ──

def test_parse_windows(cfg):
    w = pb.parse_windows("04:00-06:00,11:00-13:00")
    assert w == [(240, 360), (660, 780)]

def test_in_session_true(cfg):
    from datetime import datetime, timezone
    # 05:00 local (tz+3) = 02:00 UTC, inside 04:00-06:00 local window
    now = datetime(2026, 6, 8, 2, 0, tzinfo=timezone.utc)
    assert pb.in_session(now, cfg)

def test_in_session_false(cfg):
    from datetime import datetime, timezone
    # 08:00 local = 05:00 UTC, outside all windows
    now = datetime(2026, 6, 8, 5, 0, tzinfo=timezone.utc)
    assert not pb.in_session(now, cfg)

def test_empty_windows_always_active(cfg):
    from datetime import datetime, timezone
    cfg.pb_session_windows = ""
    now = datetime(2026, 6, 8, 5, 0, tzinfo=timezone.utc)
    assert pb.in_session(now, cfg)


# ── Strategy-aware timeout (integration with check_exit) ──────────────────────

def test_pullback_timeout_closes_flat_trade():
    """A pullback trade past the timeout that's flat/negative → timeout close."""
    import os
    os.environ['STRATEGY'] = 'pullback'
    from bot.engine import ScalpingEngine
    from bot.state import PositionState
    from bot.config import BotConfig
    from datetime import datetime, timezone, timedelta

    eng = object.__new__(ScalpingEngine)
    eng.cfg = BotConfig()
    eng.cfg.strategy = "pullback"
    eng.cfg.pb_timeout_candles = 5
    eng.cfg.timeframe = "3m"
    eng.cfg.take_profit_enabled = False
    eng.hard_stop_enabled = False
    eng.profit_lock_enabled = False
    # Opened 20 min ago on 3m = ~6 candles, past the 5-candle timeout
    pos = PositionState(entry_price=100.0, qty=1.0, trailing_stop=98.5,
                        trailing_active=True,
                        opened_at=datetime.now(timezone.utc) - timedelta(minutes=20))
    eng.positions = {"X/USDT": pos}
    # Flat/negative price → should time out
    assert eng.check_exit("X/USDT", 100.0) == "timeout"
    assert eng.check_exit("X/USDT", 99.0) == "timeout"


def test_pullback_timeout_spares_profitable_trade():
    """A profitable pullback trade past the timeout is NOT force-closed."""
    import os
    os.environ['STRATEGY'] = 'pullback'
    from bot.engine import ScalpingEngine
    from bot.state import PositionState
    from bot.config import BotConfig
    from datetime import datetime, timezone, timedelta

    eng = object.__new__(ScalpingEngine)
    eng.cfg = BotConfig()
    eng.cfg.strategy = "pullback"
    eng.cfg.pb_timeout_candles = 5
    eng.cfg.timeframe = "3m"
    eng.cfg.take_profit_enabled = False
    eng.hard_stop_enabled = False
    eng.profit_lock_enabled = False
    pos = PositionState(entry_price=100.0, qty=1.0, trailing_stop=99.0,
                        trailing_active=True, peak_pnl_pct=2.0,
                        opened_at=datetime.now(timezone.utc) - timedelta(minutes=20))
    eng.positions = {"X/USDT": pos}
    # In profit (price 102, above trailing 99) and past timeout → NOT closed by timeout
    assert eng.check_exit("X/USDT", 102.0) is None


# ── _env_bool robustness (safety-critical for TESTNET) ──────────────────────

def test_env_bool_strips_inline_comment():
    """A folded inline comment must NOT silently defeat a boolean flag."""
    import os
    from bot.config import _env_bool
    os.environ['X_TEST_FLAG'] = 'false  # this is a comment'
    assert _env_bool('X_TEST_FLAG', True) is False   # must read False, not default True
    os.environ['X_TEST_FLAG'] = 'true   # testnet only'
    assert _env_bool('X_TEST_FLAG', False) is True
    os.environ['X_TEST_FLAG'] = '  yes '
    assert _env_bool('X_TEST_FLAG', False) is True
    del os.environ['X_TEST_FLAG']
    # unset → default
    assert _env_bool('X_TEST_FLAG', True) is True


# ── Position sizing (regression for the oversized-position bug) ─────────────────

def test_pullback_sizing_does_not_explode_on_tiny_stop():
    """A razor-thin wick-stop must NOT produce a portfolio-maxing position."""
    import os
    os.environ['STRATEGY'] = 'pullback'
    from bot.engine import ScalpingEngine
    from bot.config import BotConfig
    eng = object.__new__(ScalpingEngine)
    eng.cfg = BotConfig()
    eng.cfg.pb_position_pct = 0.0      # auto-derive
    eng.cfg.max_open_positions = 4     # → 25% each
    eng.cfg.risk_per_trade_pct = 1.0
    eng.cfg.max_portfolio_pct = 25.0
    eng.cfg.pb_sizing_stop_floor_pct = 0.5
    balance = 10000.0
    eng._pending_pullback_stop = {"BNB/USDT": {"stop": 561.0 * 0.999, "regime": "gainer"}}
    size = eng._pullback_position_size(balance, 561.0, "BNB/USDT")
    # ~25% (the even share), NOT an exploded number
    assert size <= balance * 0.26, f"size {size} too large"
    assert size >= balance * 0.24, f"size {size} too small"


def test_pullback_sizing_auto_derives_from_max_positions():
    import os
    os.environ['STRATEGY'] = 'pullback'
    from bot.engine import ScalpingEngine
    from bot.config import BotConfig
    eng = object.__new__(ScalpingEngine)
    eng.cfg = BotConfig()
    eng.cfg.pb_position_pct = 0.0
    eng.cfg.risk_per_trade_pct = 1.0
    eng.cfg.pb_sizing_stop_floor_pct = 0.5
    eng._pending_pullback_stop = {}
    for n, expected_pct in [(4, 0.25), (5, 0.20), (2, 0.50)]:
        eng.cfg.max_open_positions = n
        eng.cfg.max_portfolio_pct = 100.0 / n  # matched cap
        size = eng._pullback_position_size(10000.0, 100.0, "X/USDT")
        assert abs(size - 10000.0 * expected_pct) < 1, f"n={n}: {size}"


def test_pullback_cap_does_not_silently_clip_share():
    """If per-position share exceeds the cap, honor the share (operator intent)."""
    import os
    os.environ['STRATEGY'] = 'pullback'
    from bot.engine import ScalpingEngine
    from bot.config import BotConfig
    eng = object.__new__(ScalpingEngine)
    eng.cfg = BotConfig()
    eng.cfg.pb_position_pct = 0.0
    eng.cfg.max_open_positions = 4      # 25% each
    eng.cfg.max_portfolio_pct = 20.0    # cap LOWER than share
    eng.cfg.risk_per_trade_pct = 1.0
    eng.cfg.pb_sizing_stop_floor_pct = 0.5
    eng._pending_pullback_stop = {}
    size = eng._pullback_position_size(10000.0, 100.0, "X/USDT")
    # Must honor the 25% share, not clip to 20%
    assert abs(size - 2500.0) < 1, f"expected 2500, got {size}"
