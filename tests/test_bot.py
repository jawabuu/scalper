"""
Unit tests — no exchange connectivity required.
"""

import pytest
import pandas as pd
import numpy as np

from bot.config import BotConfig
from bot.risk import RiskManager


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY_TEST", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET_TEST", "test_secret")
    monkeypatch.setenv("TESTNET", "true")
    return BotConfig()


@pytest.fixture
def risk(cfg):
    return RiskManager(cfg)


def make_df(n=100, trend="up") -> pd.DataFrame:
    """Synthetic OHLCV dataframe."""
    np.random.seed(42)
    close = np.cumprod(1 + np.random.normal(0.0003 if trend == "up" else -0.0003, 0.003, n)) * 100
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    open_ = np.roll(close, 1)
    volume = np.random.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


# ── Config tests ─────────────────────────────────────────────────────────────

def test_config_defaults(cfg):
    assert cfg.testnet is True
    assert cfg.api_key == "test_key"
    assert cfg.trailing_stop_pct < cfg.take_profit_pct
    assert cfg.rsi_min < cfg.rsi_max


def test_testnet_picks_test_credentials(monkeypatch):
    monkeypatch.setenv("TESTNET", "true")
    monkeypatch.setenv("BINANCE_API_KEY_TEST", "key_test")
    monkeypatch.setenv("BINANCE_API_SECRET_TEST", "secret_test")
    monkeypatch.setenv("BINANCE_API_KEY_LIVE", "key_live")
    monkeypatch.setenv("BINANCE_API_SECRET_LIVE", "secret_live")
    cfg = BotConfig()
    assert cfg.api_key == "key_test"
    assert cfg.api_secret == "secret_test"


def test_live_picks_live_credentials(monkeypatch):
    monkeypatch.setenv("TESTNET", "false")
    monkeypatch.setenv("BINANCE_API_KEY_TEST", "key_test")
    monkeypatch.setenv("BINANCE_API_SECRET_TEST", "secret_test")
    monkeypatch.setenv("BINANCE_API_KEY_LIVE", "key_live")
    monkeypatch.setenv("BINANCE_API_SECRET_LIVE", "secret_live")
    cfg = BotConfig()
    assert cfg.api_key == "key_live"
    assert cfg.api_secret == "secret_live"
    assert cfg.testnet is False


def test_config_validation_fails_without_key(monkeypatch):
    monkeypatch.setenv("TESTNET", "true")
    monkeypatch.setenv("BINANCE_API_KEY_TEST", "")
    monkeypatch.setenv("BINANCE_API_SECRET_TEST", "")
    with pytest.raises(AssertionError, match="BINANCE_API_KEY_TEST"):
        BotConfig().validate()


def test_config_validation_fails_bad_stops(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("TRAILING_STOP_PCT", "3.0")
    monkeypatch.setenv("TAKE_PROFIT_PCT", "1.0")
    with pytest.raises(AssertionError):
        BotConfig().validate()


# ── Risk manager tests ────────────────────────────────────────────────────────

def test_position_size_within_cap(risk):
    # With $1000 balance, 1% risk, max 30% cap
    size = risk.position_size_usdt(balance=1000, atr=0.5, price=50)
    assert size <= 300, "Must not exceed max_portfolio_pct cap"
    assert size > 0


def test_position_size_zero_atr(risk):
    size = risk.position_size_usdt(balance=1000, atr=0, price=50)
    assert size == 0


def test_position_size_scales_with_atr(risk):
    low_vol = risk.position_size_usdt(1000, atr=0.1, price=50)
    high_vol = risk.position_size_usdt(1000, atr=2.0, price=50)
    assert low_vol > high_vol, "Higher ATR should produce smaller position"


# ── Entry filter tests ────────────────────────────────────────────────────────

def test_indicator_computation_runs(cfg):
    from bot.engine import ScalpingEngine
    # We only need compute_indicators, not a live exchange
    engine = object.__new__(ScalpingEngine)
    engine.cfg = cfg
    df = make_df(150)
    result = engine.compute_indicators(df)
    for col in ["ema20", "ema50", "rsi", "adx", "vwap", "atr"]:
        assert col in result.columns, f"Missing column: {col}"


def test_passes_entry_filter_short_df(cfg):
    from bot.engine import ScalpingEngine
    engine = object.__new__(ScalpingEngine)
    engine.cfg = cfg
    df = make_df(30)  # too short
    df = engine.compute_indicators(df)
    assert engine.passes_entry_filter(df) is False


# ── Position state tests ──────────────────────────────────────────────────────

def test_position_state_defaults():
    from bot.state import PositionState
    p = PositionState(entry_price=100, qty=0.5, trailing_stop=99)
    assert p.candles_held == 0
    assert p.opened_at is not None


# ── OCO config tests ──────────────────────────────────────────────────────────

def test_oco_stop_wider_than_trailing(cfg):
    assert cfg.oco_stop_pct > cfg.trailing_stop_pct, \
        "OCO backstop must be wider than trailing stop or it fires first"


def test_oco_disabled_via_env(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY_TEST", "k")
    monkeypatch.setenv("BINANCE_API_SECRET_TEST", "s")
    monkeypatch.setenv("OCO_ENABLED", "false")
    cfg = BotConfig()
    assert cfg.oco_enabled is False


# ── Recovery tests ────────────────────────────────────────────────────────────

def test_position_state_has_oco_field():
    from bot.state import PositionState
    p = PositionState(entry_price=100, qty=1.0, trailing_stop=99)
    assert hasattr(p, "oco_order_list_id")
    assert p.oco_order_list_id is None


def test_trailing_stop_always_tighter_than_oco(cfg):
    """Core safety invariant: trailing fires before OCO while bot is running."""
    entry = 100.0
    trailing_stop_price = entry * (1 - cfg.trailing_stop_pct / 100)
    oco_stop_price      = entry * (1 - cfg.oco_stop_pct / 100)
    assert trailing_stop_price > oco_stop_price, \
        "Trailing stop must be above (tighter than) OCO stop"


def test_close_position_cancels_oco(cfg):
    """_close_position must attempt OCO cancellation before selling."""
    from unittest.mock import MagicMock, call
    from bot.engine import ScalpingEngine
    from bot.state import PositionState
    from bot.trade_log import TradeLog

    engine = object.__new__(ScalpingEngine)
    engine.cfg = cfg
    engine.trade_log = TradeLog()

    cancel_calls = []
    sell_calls = []

    engine.cancel_oco = lambda sym, lid: cancel_calls.append((sym, lid))
    engine.place_sell = lambda sym, qty: sell_calls.append((sym, qty)) or {"id": "x"}

    engine.positions = {
        "SOL/USDT": PositionState(
            entry_price=80.0, qty=10.0, trailing_stop=79.0,
            oco_order_list_id="oco-123"
        )
    }

    engine._close_position("SOL/USDT", 82.0, "take_profit")

    assert cancel_calls == [("SOL/USDT", "oco-123")], "OCO must be cancelled"
    assert sell_calls == [("SOL/USDT", 10.0)], "Position must be sold"
    assert "SOL/USDT" not in engine.positions, "Position must be removed"


# ── PositionStore tests ───────────────────────────────────────────────────────

def test_store_persists_and_loads(tmp_path):
    from bot.store import PositionStore
    from bot.state import PositionState

    path = tmp_path / "positions.json"
    store = PositionStore(path)

    pos = PositionState(entry_price=100.0, qty=1.5, trailing_stop=99.0)
    store["BTC/USDT"] = pos

    # New instance loads from disk
    store2 = PositionStore(path)
    assert "BTC/USDT" in store2
    assert store2["BTC/USDT"].entry_price == 100.0
    assert store2["BTC/USDT"].qty == 1.5


def test_store_delete_flushes(tmp_path):
    from bot.store import PositionStore
    from bot.state import PositionState

    path = tmp_path / "positions.json"
    store = PositionStore(path)
    store["ETH/USDT"] = PositionState(entry_price=2000.0, qty=0.5, trailing_stop=1980.0)
    del store["ETH/USDT"]

    store2 = PositionStore(path)
    assert "ETH/USDT" not in store2


def test_store_update_stop(tmp_path):
    from bot.store import PositionStore
    from bot.state import PositionState

    path = tmp_path / "positions.json"
    store = PositionStore(path)
    store["SOL/USDT"] = PositionState(entry_price=80.0, qty=10.0, trailing_stop=79.0)
    store.update_stop("SOL/USDT", 81.5)

    store2 = PositionStore(path)
    assert store2["SOL/USDT"].trailing_stop == 81.5


def test_store_survives_parameter_change(tmp_path):
    """Positions written under one config load correctly under different config."""
    from bot.store import PositionStore
    from bot.state import PositionState

    path = tmp_path / "positions.json"
    # Written under Profile B (min_volume 1M, mid-cap coin)
    store = PositionStore(path)
    store["XLM/USDT"] = PositionState(entry_price=0.25, qty=4000.0, trailing_stop=0.248)

    # Loaded after switching to Profile A (min_volume 5M — XLM might not qualify)
    # Store is config-agnostic: it just loads what was written
    store2 = PositionStore(path)
    assert "XLM/USDT" in store2, "Position must survive a config change"
    assert store2["XLM/USDT"].qty == 4000.0
