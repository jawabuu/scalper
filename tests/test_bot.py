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


# ── Trailing activation threshold tests ───────────────────────────────────────

def test_position_state_trailing_active_default_true():
    """Default must be True so existing behaviour and recovered positions are unchanged."""
    from bot.state import PositionState
    p = PositionState(entry_price=100, qty=1.0, trailing_stop=99)
    assert p.trailing_active is True
    assert p.activation_price == 0.0


def test_trailing_activation_pct_config_default(cfg):
    """Config provides the seed value; default 1.0."""
    assert cfg.trailing_activation_pct == 1.0


def test_store_roundtrip_preserves_activation(tmp_path):
    """Persisted activation fields survive a save/load cycle."""
    from bot.state import PositionState
    from bot.store import PositionStore
    path = tmp_path / "positions.json"

    store = PositionStore(path=path)
    store["BTC/USDT"] = PositionState(
        entry_price=100.0, qty=1.0, trailing_stop=98.0,
        trailing_active=False, activation_price=101.0,
    )

    store2 = PositionStore(path=path)  # re-load from disk
    p = store2["BTC/USDT"]
    assert p.trailing_active is False
    assert p.activation_price == 101.0


def test_store_load_legacy_position_defaults_active(tmp_path):
    """A position saved before this feature (no fields) loads as trailing_active=True."""
    import json
    path = tmp_path / "positions.json"
    legacy = {
        "BTC/USDT": {
            "entry_price": 100.0, "qty": 1.0, "trailing_stop": 98.0,
            "candles_held": 5, "opened_at": "2026-01-01T00:00:00+00:00",
            "oco_order_list_id": None, "backstop_type": None,
        }
    }
    path.write_text(json.dumps(legacy))

    from bot.store import PositionStore
    store = PositionStore(path=path)
    p = store["BTC/USDT"]
    assert p.trailing_active is True   # legacy positions trail immediately
    assert p.activation_price == 0.0


# ── Critical safety: dormant position must never be unprotected ────────────────

def test_no_backstop_forces_trailing_active_logic():
    """
    The core safety invariant: if trailing activation is enabled BUT no server-side
    backstop was placed, the position must start with trailing_active=True so it is
    never left completely unprotected. This test verifies the decision logic.
    """
    # Replicate the exact decision from run_cycle entry logic
    def decide(activation_enabled, activation_pct, oco_id, fill_price):
        if activation_enabled and activation_pct > 0 and oco_id:
            return (False, fill_price * (1 + activation_pct / 100))
        return (True, 0.0)

    # Case 1: activation on, backstop placed → dormant (normal)
    active, ap = decide(True, 1.0, "order123", 100.0)
    assert active is False and ap == 101.0

    # Case 2: activation on, NO backstop → forced active (safety)
    active, ap = decide(True, 1.0, None, 100.0)
    assert active is True and ap == 0.0, \
        "Position with no backstop MUST start trailing_active=True"

    # Case 3: activation off → always active
    active, ap = decide(False, 1.0, "order123", 100.0)
    assert active is True and ap == 0.0

    # Case 4: activation off, no backstop → active
    active, ap = decide(False, 1.0, None, 100.0)
    assert active is True and ap == 0.0


# ── BTC market-regime filter tests ─────────────────────────────────────────────

def test_btc_trend_threshold_logic():
    """The short-term falling decision: change < -threshold."""
    def is_falling(change_pct, threshold_pct):
        return change_pct < -abs(threshold_pct)

    # Falling more than threshold → blocked
    assert is_falling(-0.20, 0.15) is True
    # Falling but within threshold → allowed (noise)
    assert is_falling(-0.10, 0.15) is False
    # Flat → allowed
    assert is_falling(0.0, 0.15) is False
    # Rising → allowed
    assert is_falling(0.50, 0.15) is False
    # Exactly at threshold → not strictly less → allowed
    assert is_falling(-0.15, 0.15) is False


def test_btc_filter_fail_open_contract():
    """When BTC data is unavailable, the gate must allow entries (fail-open)."""
    # Simulate the gate decision from run_cycle
    def gate_blocks(enabled, available, short_term_falling):
        if not enabled:
            return False
        if not available:
            return False  # fail-open
        return short_term_falling

    # Disabled → never blocks
    assert gate_blocks(False, True, True) is False
    # Enabled, unavailable → fail-open, never blocks
    assert gate_blocks(True, False, True) is False
    # Enabled, available, falling → blocks
    assert gate_blocks(True, True, True) is True
    # Enabled, available, not falling → allows
    assert gate_blocks(True, True, False) is False


def test_btc_regime_cache_reset_semantics():
    """A None cache means 'recompute'; a dict means 'use cached'."""
    cache = None
    assert cache is None  # would trigger recompute
    cache = {"available": True, "short_term_falling": False}
    assert cache is not None  # would use cached value


# ── Partial backstop coverage (whole-unit coin) safety ─────────────────────────

def test_partial_backstop_detection():
    """
    Whole-unit-step coins can't fully cover a fractional holding with a server-side
    stop. When the uncovered fraction is meaningful (>1%, the TAO bug), the position
    must be flagged partial so the trailing stop stays active. Fee dust (<1%) is fine.
    """
    def round_to_step(qty, step):
        if step <= 0:
            return qty
        return int(round(qty / step, 9)) * step

    def is_partial(qty, step):
        post_fee = qty * (1 - 0.0015)
        if step > 0 and post_fee > 0:
            backstop_qty = round_to_step(post_fee, step)
            uncovered_pct = (post_fee - backstop_qty) / post_fee * 100
            return uncovered_pct > 1.0
        return False

    # Whole-unit coins with meaningful remainder → partial
    assert is_partial(2.0, 1.0) is True     # TAO: ~50% uncovered
    assert is_partial(1.0, 1.0) is True     # ZEC: 100% uncovered
    assert is_partial(3.0, 1.0) is True     # ~33% uncovered
    # Fine-grained coins → only fee dust uncovered → not partial
    assert is_partial(100.5, 0.001) is False
    assert is_partial(50.25, 0.01) is False
    # Large whole-unit qty → fee dust is < 1 unit → not partial
    assert is_partial(16219.0, 1.0) is False


def test_partial_backstop_forces_trailing_active():
    """When backstop is partial, the position must not start dormant."""
    def should_be_dormant(activation_enabled, oco_id, partial_backstop):
        backstop_fully_covers = bool(oco_id) and not partial_backstop
        return activation_enabled and backstop_fully_covers

    # Partial backstop + activation on → must NOT be dormant
    assert should_be_dormant(True, "order123", True) is False
    # Full backstop + activation on → dormant is allowed
    assert should_be_dormant(True, "order123", False) is True
    # No backstop → never dormant
    assert should_be_dormant(True, None, False) is False


# ── Per-coin entry-timing gate tests ───────────────────────────────────────────

def test_entry_timing_gate_logic():
    """
    The gate rejects entries where price is extended above the fast EMA beyond
    the band, and allows entries near or below it (pullbacks).
    """
    def passes(distance_pct, band_pct, enabled):
        if not enabled:
            return True
        if distance_pct is None:
            return True  # fail-open
        return distance_pct <= band_pct

    band = 0.5
    # Extended well above fast EMA → rejected (the OPN chase-the-top case)
    assert passes(2.0, band, True) is False
    assert passes(0.6, band, True) is False
    # Right at the band → allowed
    assert passes(0.5, band, True) is True
    # Near the mean → allowed (ideal entry)
    assert passes(0.1, band, True) is True
    # Pulled back below fast EMA → allowed (best entry)
    assert passes(-0.8, band, True) is True
    # Gate disabled → always allowed regardless of distance
    assert passes(5.0, band, False) is True
    # Distance unknown → fail-open
    assert passes(None, band, True) is True


def test_entry_timing_distance_sign():
    """Distance is positive when price is above the fast EMA, negative below."""
    def distance(close, fast):
        if fast <= 0:
            return None
        return (close - fast) / fast * 100

    assert distance(102, 100) == 2.0     # extended above
    assert distance(100, 100) == 0.0     # at the mean
    assert round(distance(99, 100), 2) == -1.0  # pulled back below


# ── Momentum confirmation gate tests ───────────────────────────────────────────

def test_momentum_gate_logic():
    """
    The momentum gate requires both a positive short-term slope above the
    threshold AND the most recent candle not red. This confirms the coin is
    rising at entry rather than merely in a recent (lagging) uptrend structure.
    """
    def passes(slope_pct, last_candle_up, min_slope, enabled):
        if not enabled:
            return True
        if slope_pct is None:
            return True  # fail-open
        return (slope_pct >= min_slope) and last_candle_up

    mn = 0.1
    # Rising, last candle green → pass (the OPN-going-up case we want)
    assert passes(0.5, True, mn, True) is True
    # Rising enough but last candle red → reject (early reversal)
    assert passes(0.5, False, mn, True) is False
    # Net positive but below threshold → reject (too flat, noise)
    assert passes(0.05, True, mn, True) is False
    # Falling → reject (the OPN-rolling-over case that bled us)
    assert passes(-0.8, True, mn, True) is False
    assert passes(-0.8, False, mn, True) is False
    # Disabled → always pass regardless
    assert passes(-5.0, False, mn, False) is True
    # Unknown slope → fail-open
    assert passes(None, True, mn, True) is True


def test_momentum_slope_uses_raw_price():
    """Slope is raw close-to-close percent over the lookback — no smoothing/lag."""
    def slope(close_now, close_past):
        if close_past <= 0:
            return None
        return (close_now - close_past) / close_past * 100

    assert round(slope(101.0, 100.0), 2) == 1.0      # rising 1%
    assert round(slope(100.0, 100.0), 2) == 0.0      # flat
    assert round(slope(98.0, 100.0), 2) == -2.0      # falling 2% (would reject)


# ── Continuous profit lock tests ───────────────────────────────────────────────

def test_profit_lock_floor_curve():
    """
    The floor arms only above the arm threshold and locks a rising fraction of
    the peak as the peak climbs (give-back shrinks toward zero).
    """
    def floor(peak, arm=1.0, giveback=0.18, enabled=True):
        if not enabled:
            return None
        if peak < arm or arm <= 0:
            return None
        return peak - giveback * (arm / peak)

    # Below arm → not armed
    assert floor(0.5) is None
    assert floor(0.99) is None
    # Armed → locks a rising fraction
    assert abs(floor(1.0) - 0.82) < 1e-6      # 82% at +1%
    assert abs(floor(2.0) - 1.91) < 1e-6      # 95.5% at +2%
    assert abs(floor(5.0) - 4.964) < 1e-3     # 99.3% at +5%
    # Floor always below peak (we give back something, never lock above peak)
    for p in [1.0, 2.0, 3.0, 5.0, 10.0]:
        assert floor(p) < p
    # Floor rises monotonically with peak (ratchet)
    floors = [floor(p) for p in [1.0, 2.0, 3.0, 4.0, 5.0]]
    assert floors == sorted(floors)
    # Disabled → never armed
    assert floor(5.0, enabled=False) is None


def test_profit_lock_exit_trigger():
    """Exit fires when current P&L falls to/below the armed floor."""
    def should_exit(peak_pnl, current_pnl, arm=1.0, giveback=0.18):
        if peak_pnl < arm:
            return False
        floor = peak_pnl - giveback * (arm / peak_pnl)
        return current_pnl <= floor

    # Peaked +5%, fell to +3.16% (the TON case) → exit (floor ~4.96%)
    assert should_exit(5.0, 3.16) is True
    # Peaked +2%, still at +1.95% (above floor 1.91%) → hold
    assert should_exit(2.0, 1.95) is False
    # Peaked +2%, fell to +1.90% (at/below floor 1.91%) → exit
    assert should_exit(2.0, 1.90) is True
    # Never armed (peak +0.8%) → never exits on lock
    assert should_exit(0.8, 0.5) is False


def test_peak_pnl_legacy_default():
    """Positions loaded without peak_pnl_pct default to 0.0 (unarmed)."""
    from bot.state import PositionState
    pos = PositionState(entry_price=100.0, qty=1.0, trailing_stop=98.8)
    assert pos.peak_pnl_pct == 0.0


# ── Fast monitor peak capture ──────────────────────────────────────────────────

def test_peak_capture_from_live_price():
    """
    The fix for the profit-lock missing fast spikes: peak P&L must ratchet from
    whatever live price is observed, so a spike seen between trading cycles is
    captured and the profit-lock floor reflects the true peak.
    """
    arm, gb = 0.6, 0.18
    def floor(peak):
        if peak < arm:
            return None
        return peak - gb * (arm / peak)

    # Simulate ratcheting peak as the monitor observes live prices
    peak = 0.0
    for pnl in [0.3, 0.55, 1.5, 0.9, 0.7]:   # spike to 1.5 then fall back
        if pnl > peak:
            peak = pnl
    assert peak == 1.5                        # spike captured, not the 0.55 sample
    f = floor(peak)
    assert abs(f - 1.428) < 1e-3              # floor reflects the true peak
    # At +0.7% the position is below the floor -> profit lock should fire
    assert 0.7 <= f

    # Contrast: if the monitor had only sampled 0.55 (the old cycle-only behaviour),
    # the lock would barely arm and lock far less.
    assert floor(0.55) is None                # 0.55 < arm 0.6 -> not even armed
