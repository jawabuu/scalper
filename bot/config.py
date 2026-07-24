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
    # Strip inline comments and whitespace before matching. Some env/compose
    # tooling folds a trailing `# comment` into the value; without this, a line
    # like `TESTNET=false  # note` would fail the exact match and silently fall
    # back to the default — a real hazard for a safety-critical flag. Stripping
    # makes boolean parsing robust regardless of how the value was supplied.
    raw = os.environ.get(key, "")
    val = raw.split("#", 1)[0].strip().lower()
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
    profit_lock_arm_pct: float = field(default_factory=lambda: _env_float("PROFIT_LOCK_ARM_PCT", 0.6))
    profit_lock_giveback_pct: float = field(default_factory=lambda: _env_float("PROFIT_LOCK_GIVEBACK_PCT", 0.12))

    # ── Fast peak/exit monitor ──────────────────────────────────────────
    # Independent of the main trading cycle, a lightweight loop fetches the live
    # price for each open position every MONITOR_INTERVAL seconds, ratchets the
    # peak P&L, and checks the trailing stop / profit lock. This makes the engine
    # see price spikes the way the dashboard does, so the profit lock captures the
    # true peak instead of only what the slow trading cycle happened to sample.
    monitor_interval: float = field(default_factory=lambda: _env_float("MONITOR_INTERVAL", 7.0))

    # ── Hard stop-loss ──────────────────────────────────────────────────
    # Cut a losing position at a fixed P&L (e.g. -0.5%) rather than waiting for
    # the looser trailing stop. The downside mirror of the profit lock: bounds
    # give-back on the loss side. Checked before the trailing stop and regardless
    # of trailing-active state.
    hard_stop_enabled: bool = field(default_factory=lambda: _env_bool("HARD_STOP_ENABLED", True))
    hard_stop_pct: float = field(default_factory=lambda: _env_float("HARD_STOP_PCT", 0.5))

    # ── Smart re-entry guard ────────────────────────────────────────────
    # After a RED close on a coin, refuse to re-enter it at a price higher than
    # the loss exit — avoids chasing a just-lost coin back up into the same move.
    reentry_guard_enabled: bool = field(default_factory=lambda: _env_bool("REENTRY_GUARD_ENABLED", True))
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
    # Drop the still-forming last candle from OHLCV so entry/confirmation logic
    # reads only CLOSED candles (correct for both strategies; essential for the
    # pullback good-price and RSI-rising gates). Default on.
    drop_incomplete_candle: bool = field(default_factory=lambda: _env_bool("DROP_INCOMPLETE_CANDLE", True))

    # ══════════════════════════════════════════════════════════════════════
    # STRATEGY SELECTOR
    # "breakout" = the original five-filter continuation system (default).
    # "pullback" = mean-reversion / good-price scalp with gainer & dipper regimes.
    # The two are mutually exclusive per running instance; deploy separate
    # containers to run both concurrently.
    # ══════════════════════════════════════════════════════════════════════
    strategy: str = field(default_factory=lambda: _env("STRATEGY", "breakout").lower())

    # ── Pullback strategy parameters ────────────────────────────────────
    # Universal "good entry price" gate (BOTH regimes): never enter high in the
    # candle, never at the tip of a rejection (upper-wick) spike.
    pb_candle_pos_max: float = field(default_factory=lambda: _env_float("PB_CANDLE_POS_MAX", 0.5))
    pb_upper_wick_max: float = field(default_factory=lambda: _env_float("PB_UPPER_WICK_MAX", 0.40))

    # Strict, load-bearing gates (trusted from experience — do not flex).
    pb_ema_fast: int = field(default_factory=lambda: _env_int("PB_EMA_FAST", 9))
    pb_ema_slow: int = field(default_factory=lambda: _env_int("PB_EMA_SLOW", 21))
    pb_ema_buffer_pct: float = field(default_factory=lambda: _env_float("PB_EMA_BUFFER_PCT", 0.05))
    pb_rsi_min: float = field(default_factory=lambda: _env_float("PB_RSI_MIN", 55.0))
    pb_rsi_max: float = field(default_factory=lambda: _env_float("PB_RSI_MAX", 72.0))
    pb_rsi_rising_lookback: int = field(default_factory=lambda: _env_int("PB_RSI_RISING_LOOKBACK", 3))
    # Volume FLOOR (coarse veto): current volume must be at least this % of the
    # recent volume MA, else the move is treated as collapsed/dead. NOT a spike
    # requirement. Tunable; default 40%.
    pb_vol_floor_pct: float = field(default_factory=lambda: _env_float("PB_VOL_FLOOR_PCT", 40.0))
    pb_vol_ma_len: int = field(default_factory=lambda: _env_int("PB_VOL_MA_LEN", 5))

    # Tunable knobs (loosen on testnet for more data flow).
    pb_low_proximity_pct: float = field(default_factory=lambda: _env_float("PB_LOW_PROXIMITY_PCT", 3.0))
    pb_max_wick_stop_pct: float = field(default_factory=lambda: _env_float("PB_MAX_WICK_STOP_PCT", 1.5))
    pb_min_volume_usdt: float = field(default_factory=lambda: _env_float("PB_MIN_VOLUME_USDT", 10_000_000))

    # Regime enable flags — run gainer, dipper, or both.
    pb_gainer_enabled: bool = field(default_factory=lambda: _env_bool("PB_GAINER_ENABLED", True))
    pb_dipper_enabled: bool = field(default_factory=lambda: _env_bool("PB_DIPPER_ENABLED", True))

    # Exits (strictly as specified — no profit-lock/trailing machinery).
    pb_tp_pct: float = field(default_factory=lambda: _env_float("PB_TP_PCT", 1.0))
    pb_tp_use_ma10: bool = field(default_factory=lambda: _env_bool("PB_TP_USE_MA10", True))
    pb_price_ma_len: int = field(default_factory=lambda: _env_int("PB_PRICE_MA_LEN", 10))
    # Position sizing: per-position share of the portfolio. Default 0 = AUTO-DERIVE
    # as an even split across max_open_positions (4 positions → 25% each → full even
    # deployment). Set a positive value to override with a fixed fraction.
    pb_position_pct: float = field(default_factory=lambda: _env_float("PB_POSITION_PCT", 0.0))
    pb_sizing_stop_floor_pct: float = field(default_factory=lambda: _env_float("PB_SIZING_STOP_FLOOR_PCT", 0.5))
    pb_timeout_candles: int = field(default_factory=lambda: _env_int("PB_TIMEOUT_CANDLES", 5))

    # Session windows (UTC+3) — comma-separated HH:MM-HH:MM ranges. Empty = always.
    # Default: the four windows from the spec.
    pb_session_windows: str = field(default_factory=lambda: _env(
        "PB_SESSION_WINDOWS",
        "23:00-23:15,04:00-06:00,11:00-13:00,15:00-17:00"))
    pb_session_tz_offset: int = field(default_factory=lambda: _env_int("PB_SESSION_TZ_OFFSET", 3))

    def validate(self):
        suffix = "TEST" if self.testnet else "LIVE"
        assert self.api_key, f"BINANCE_API_KEY_{suffix} must be set"
        assert self.api_secret, f"BINANCE_API_SECRET_{suffix} must be set"

        # ── Cross-wire safety ──────────────────────────────────────────────
        # Prevent an instance intended for one environment accidentally using the
        # other's credentials. If running LIVE, the LIVE keys must be present AND
        # must not be identical to the TEST keys (a common copy-paste misconfig
        # that would point a "testnet" container at real funds, or vice versa).
        test_key = _env("BINANCE_API_KEY_TEST")
        live_key = _env("BINANCE_API_KEY_LIVE")
        if test_key and live_key:
            assert test_key != live_key, (
                "BINANCE_API_KEY_TEST and BINANCE_API_KEY_LIVE are identical — "
                "refusing to start to avoid a testnet/live cross-wire. Check your env."
            )
        if not self.testnet:
            # Running live: ensure we're actually using the live key, not a stray test key.
            assert self.api_key == live_key, (
                "TESTNET=false but the resolved API key is not the LIVE key — "
                "refusing to start (possible cross-wire)."
            )
        else:
            assert self.api_key == test_key or not live_key, (
                "TESTNET=true but the resolved API key matches the LIVE key — "
                "refusing to start (possible cross-wire)."
            )

        assert self.strategy in ("breakout", "pullback"), \
            f"STRATEGY must be 'breakout' or 'pullback', got {self.strategy!r}"
        assert 0 < self.risk_per_trade_pct <= 5, \
            "risk_per_trade_pct should be 0–5% for conservative trading"
        if self.strategy == "breakout":
            assert 0 < self.trailing_stop_pct < self.take_profit_pct, \
                "trailing_stop_pct must be less than take_profit_pct"
        else:  # pullback
            assert self.pb_rsi_min < self.pb_rsi_max, "pb_rsi_min must be < pb_rsi_max"
            assert 0 < self.pb_candle_pos_max <= 1.0, "pb_candle_pos_max in (0,1]"
            assert self.pb_gainer_enabled or self.pb_dipper_enabled, \
                "at least one pullback regime (gainer/dipper) must be enabled"
        return self
