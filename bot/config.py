"""
Bot configuration — loaded from environment variables with safe defaults.
All risk parameters are intentionally conservative for capital preservation.
"""

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def _resolve_credentials() -> tuple[bool, str, str]:
    """
    Resolve credentials based on TESTNET flag (default: true).
    Reads BINANCE_API_KEY_TEST / BINANCE_API_KEY_LIVE (and secrets)
    and returns (testnet, api_key, api_secret).
    """
    testnet = _env_bool("TESTNET", True)
    suffix = "TEST" if testnet else "LIVE"
    api_key = _env(f"BINANCE_API_KEY_{suffix}")
    api_secret = _env(f"BINANCE_API_SECRET_{suffix}")
    return testnet, api_key, api_secret


@dataclass
class BotConfig:
    # ── Credentials (resolved together so suffix is consistent) ────────
    testnet: bool = field(default_factory=lambda: _resolve_credentials()[0])
    api_key: str = field(default_factory=lambda: _resolve_credentials()[1])
    api_secret: str = field(default_factory=lambda: _resolve_credentials()[2])

    # ── Market / timeframe ──────────────────────────────────────────────
    timeframe: str = field(default_factory=lambda: _env("TIMEFRAME", "5m"))
    max_symbols: int = field(default_factory=lambda: _env_int("MAX_SYMBOLS", 20))
    min_volume_usdt: float = field(default_factory=lambda: _env_float("MIN_VOLUME_USDT", 5_000_000))
    max_spread_pct: float = field(default_factory=lambda: _env_float("MAX_SPREAD_PCT", 0.08))
    symbol_cache_ttl: int = field(default_factory=lambda: _env_int("SYMBOL_CACHE_TTL", 300))
    blacklist: list = field(default_factory=lambda: _env("BLACKLIST", "").split(","))

    # ── Entry filters ───────────────────────────────────────────────────
    adx_min: float = field(default_factory=lambda: _env_float("ADX_MIN", 25.0))
    rsi_min: float = field(default_factory=lambda: _env_float("RSI_MIN", 50.0))
    rsi_max: float = field(default_factory=lambda: _env_float("RSI_MAX", 65.0))

    # ── Risk / exits ────────────────────────────────────────────────────
    trailing_stop_pct: float = field(default_factory=lambda: _env_float("TRAILING_STOP_PCT", 0.8))
    # Initial value for the trailing-stop activation threshold (UI-toggled, in-memory).
    # This ONLY sets the default percentage — the feature is enabled/disabled from the UI.
    # When active, a new position's trailing stop does not engage until price first
    # reaches entry * (1 + this%). Until then the server-side stop-market is the only stop.
    trailing_activation_pct: float = field(default_factory=lambda: _env_float("TRAILING_ACTIVATION_PCT", 1.0))

    # ── BTC market-regime filter (UI-toggled, in-memory) ────────────────
    # These ONLY set default values — the filter is enabled/disabled from the UI.
    # When active, new entries are skipped if BTC's short-term trend is falling:
    # i.e. BTC's current price is below its price BTC_TREND_LOOKBACK candles ago
    # by more than BTC_TREND_THRESHOLD_PCT. Open positions are never affected.
    # The slow EMA20/50 regime on BTC is logged for context but not enforced.
    btc_trend_lookback: int = field(default_factory=lambda: _env_int("BTC_TREND_LOOKBACK", 3))
    btc_trend_threshold_pct: float = field(default_factory=lambda: _env_float("BTC_TREND_THRESHOLD_PCT", 0.15))

    # ── Entry-timing gate (per-coin, UI-toggled) ────────────────────────
    # Avoids chasing a coin that has spiked above its short-term mean (the
    # whipsaw cause): only enter when price is within ENTRY_TIMING_BAND_PCT
    # above the fast EMA (length ENTRY_TIMING_EMA_LEN). DEFAULT ON — this
    # targets the core whipsaw problem. The fast-EMA distance is logged on
    # every entry regardless of whether the gate is enforced.
    entry_timing_ema_len: int = field(default_factory=lambda: _env_int("ENTRY_TIMING_EMA_LEN", 9))
    entry_timing_band_pct: float = field(default_factory=lambda: _env_float("ENTRY_TIMING_BAND_PCT", 0.8))

    # ── Momentum confirmation (per-coin, short-term direction, DEFAULT ON) ──
    # Confirms a coin is actually rising RIGHT NOW at entry, not merely in a
    # recent uptrend structure (which lagging EMA/RSI/ADX filters can still show
    # well into a decline — the OPN-rolling-over case). Uses RAW PRICE slope over
    # the last MOMENTUM_LOOKBACK candles (no smoothing — avoids lag). Requires the
    # current close to be above the close N candles ago by at least
    # MOMENTUM_MIN_SLOPE_PCT, and the most recent candle not to be red.
    momentum_lookback: int = field(default_factory=lambda: _env_int("MOMENTUM_LOOKBACK", 2))
    momentum_min_slope_pct: float = field(default_factory=lambda: _env_float("MOMENTUM_MIN_SLOPE_PCT", 0.1))

    # ── Profit lock (continuous, peak-tracking, DEFAULT ON) ─────────────
    # Once a position's P&L crosses PROFIT_LOCK_ARM_PCT, a profit floor arms and
    # ratchets up with the peak P&L, locking a rising fraction of the gain. The
    # give-back (peak minus floor) starts at PROFIT_LOCK_GIVEBACK_PCT at the arm
    # point and shrinks as the peak climbs, so big winners are locked tightly
    # (~99%) while small winners keep a little room. Sits alongside the trailing
    # stop; the position exits at whichever triggers first. Locks scalping gains
    # that the looser 1.2% trailing stop would otherwise give back.
    profit_lock_arm_pct: float = field(default_factory=lambda: _env_float("PROFIT_LOCK_ARM_PCT", 1.0))
    profit_lock_giveback_pct: float = field(default_factory=lambda: _env_float("PROFIT_LOCK_GIVEBACK_PCT", 0.18))
    take_profit_pct: float = field(default_factory=lambda: _env_float("TAKE_PROFIT_PCT", 1.5))
    # When disabled the trailing stop is the sole exit — lets winners run indefinitely.
    # Take profit then only affects the OCO backstop price (server-side safety net).
    take_profit_enabled: bool = field(default_factory=lambda: _env_bool("TAKE_PROFIT_ENABLED", True))
    max_open_positions: int = field(default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 3))
    max_hold_candles: int = field(default_factory=lambda: _env_int("MAX_HOLD_CANDLES", 12))
    risk_per_trade_pct: float = field(default_factory=lambda: _env_float("RISK_PER_TRADE_PCT", 1.0))
    max_portfolio_pct: float = field(default_factory=lambda: _env_float("MAX_PORTFOLIO_PCT", 30.0))
    min_trade_usdt: float = field(default_factory=lambda: _env_float("MIN_TRADE_USDT", 11.0))

    # ── OCO backstop (server-side safety net when bot is down) ────────
    # Set wider than trailing_stop_pct so it only fires if the bot is dead.
    # e.g. trailing=0.8%, oco_stop=2.0% — trailing always fires first.
    oco_stop_pct: float = field(default_factory=lambda: _env_float("OCO_STOP_PCT", 2.0))
    oco_enabled: bool = field(default_factory=lambda: _env_bool("OCO_ENABLED", True))

    # ── Stop-limit fallback (for pairs that don't support OCO) ──────────
    # Placed at entry * (1 - (trailing_stop_pct + stop_limit_offset_pct)%).
    # The offset pushes the stop trigger just below the trailing stop so the
    # in-memory trailing stop always fires first while the bot is running.
    # The stop-limit only triggers if the bot dies and price gaps down past
    # the trailing stop level before the bot can recover.
    #
    # stop trigger  = entry * (1 - (trailing_stop_pct + stop_limit_offset_pct))
    # limit price   = stop trigger * (1 - stop_limit_fill_buffer_pct)
    #
    # Example: trailing=1.2%, offset=0.05%, fill_buffer=0.1%
    #   stop trigger = entry * (1 - 1.25%) — just below trailing stop
    #   limit price  = stop trigger * (1 - 0.1%) — ensures fill in fast drops
    stop_limit_offset_pct: float = field(default_factory=lambda: _env_float("STOP_LIMIT_OFFSET_PCT", 0.05))
    stop_limit_fill_buffer_pct: float = field(default_factory=lambda: _env_float("STOP_LIMIT_FILL_BUFFER_PCT", 0.1))

    # ── Cooldown ────────────────────────────────────────────────────────
    # Number of candles to wait before re-entering a manually closed symbol.
    # Prevents the bot immediately re-buying something you just closed.
    manual_close_cooldown_candles: int = field(default_factory=lambda: _env_int("MANUAL_CLOSE_COOLDOWN_CANDLES", 3))

    # ── Trading hours ───────────────────────────────────────────────────
    # Restrict new entries to specific UTC hours. Open positions continue
    # to be managed (trailing stop, exits) outside trading hours.
    # Format: "HH:MM" 24hr UTC. Leave empty for unrestricted trading.
    # Example: TRADING_HOURS_START=08:00 TRADING_HOURS_END=20:00
    trading_hours_start: str = field(default_factory=lambda: _env("TRADING_HOURS_START", ""))
    trading_hours_end: str = field(default_factory=lambda: _env("TRADING_HOURS_END", ""))

    # ── Proxy ───────────────────────────────────────────────────────────
    # SOCKS5 proxy for ccxt — use socks5h:// so DNS resolves through proxy too.
    # Locally: ssh -D 1080 -N user@vps → set SOCKS_PROXY=socks5h://localhost:1080
    # Production: gluetun sidecar → set SOCKS_PROXY=socks5h://gluetun:1080
    # Leave empty to connect directly (testnet, unrestricted regions).
    socks_proxy: str = field(default_factory=lambda: _env("SOCKS_PROXY", ""))

    # ── Timing ──────────────────────────────────────────────────────────
    poll_interval: int = field(default_factory=lambda: _env_int("POLL_INTERVAL", 60))

    def validate(self):
        suffix = "TEST" if self.testnet else "LIVE"
        assert self.api_key, f"BINANCE_API_KEY_{suffix} must be set"
        assert self.api_secret, f"BINANCE_API_SECRET_{suffix} must be set"
        assert 0 < self.trailing_stop_pct < self.take_profit_pct, \
            "trailing_stop_pct must be less than take_profit_pct"
        assert 0 < self.risk_per_trade_pct <= 5, \
            "risk_per_trade_pct should be 0–5% for conservative trading"
        return self
