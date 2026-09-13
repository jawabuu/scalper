"""
Bot configuration — loaded from environment variables with safe defaults.
All risk parameters are intentionally conservative for capital preservation.
"""

import os
from dataclasses import dataclass, field


def _strip_inline_comment(raw: str) -> str:
    """
    Drop a trailing `# comment` and surrounding whitespace.

    Compose `environment:` entries keep everything after the `=` verbatim, so a
    line like `GUARD_TRAIL_CALLBACK_PCT=1.0  # price percent` arrives with the
    comment attached. _env_bool already handled this; the numeric readers did
    not, and crashed the process at startup instead.
    """
    return (raw or "").split("#", 1)[0].strip()


def _env(key: str, default: str = "") -> str:
    """
    Read a string setting, treating an EMPTY value as unset.

    Compose's `- VAR` form (no `=`) passes the variable through from the host,
    and when the host does not define it the container receives an empty
    string. Honouring that literally disabled persistence on an instance whose
    compose listed `- FUTURES_STATE_PATH` with no value, so an empty value
    falls back to the default instead.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    cleaned = _strip_inline_comment(raw)
    return cleaned if cleaned else default


def _env_float(key: str, default: float) -> float:
    raw = _strip_inline_comment(os.environ.get(key, ""))
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        import logging
        logging.getLogger("config").warning(
            f"{key}={raw!r} is not a number — falling back to {default}")
        return float(default)


def _env_int(key: str, default: int) -> int:
    raw = _strip_inline_comment(os.environ.get(key, ""))
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        import logging
        logging.getLogger("config").warning(
            f"{key}={raw!r} is not a number — falling back to {default}")
        return int(default)


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

    # ── Futures position guardian ───────────────────────────────────────
    # Protects futures positions the OPERATOR opens. Never opens one. Every
    # order it places is reduce-only. DRY RUN defaults to TRUE — it must be
    # switched off deliberately before it sends a single real order.
    guardian_enabled: bool = field(default_factory=lambda: _env_bool("GUARDIAN_ENABLED", False))
    guardian_dry_run: bool = field(default_factory=lambda: _env_bool("GUARDIAN_DRY_RUN", True))
    # Binance retired futures testnet in favour of "demo trading" (separate
    # credentials, routes to demo-fapi.binance.com, mirrors live market data).
    # GUARDIAN_DEMO is the current name; GUARDIAN_TESTNET is still honoured.
    # Follows TESTNET by default so ONE flag drives both halves: TESTNET picks
    # which API keys are used, and this picks which endpoint they are sent to.
    # Letting them default independently allowed live keys to be pointed at the
    # demo endpoint (or vice versa) — a confusing auth failure at best.
    # GUARDIAN_DEMO / GUARDIAN_TESTNET still override explicitly.
    guardian_demo: bool = field(default_factory=lambda: _env_bool(
        "GUARDIAN_DEMO", _env_bool("GUARDIAN_TESTNET", _env_bool("TESTNET", True))))
    guardian_poll_interval: float = field(default_factory=lambda: _env_float("GUARDIAN_POLL_INTERVAL", 5.0))
    # Faster polling while an entry order is resting. A filled position has no
    # stop until the guardian notices it, and that window is where a sharp move
    # does its damage.
    guardian_pending_poll_interval: float = field(default_factory=lambda: max(
        0.5, _env_float("GUARDIAN_PENDING_POLL_INTERVAL", 1.0)))
    # Persists restart-critical futures state: sized stops, position peaks, the
    # daily-loss baseline, cooldowns and trade history. Empty disables it.
    futures_state_path: str = field(default_factory=lambda: _env(
        "FUTURES_STATE_PATH", "/app/logs/futures_state.json"))
    # One-shot cleanup on startup: "history" drops closed trades only (keeps
    # open positions and the daily baseline); "all" drops everything. The old
    # file is archived, not deleted. Leave UNSET in normal operation — it
    # applies on every restart while it is set.
    futures_state_reset: str = field(default_factory=lambda: _env("FUTURES_STATE_RESET", ""))
    # How many closed trades to retain for analysis.
    max_closed_trades: int = field(default_factory=lambda: _env_int(
        "MAX_CLOSED_TRADES", 5000))
    # Thresholds in ROI% (Binance UI convention: PnL / margin * 100).
    guard_initial_stop_roi: float = field(default_factory=lambda: _env_float("GUARD_INITIAL_STOP_ROI", 7.0))
    guard_arm_roi: float = field(default_factory=lambda: _env_float("GUARD_ARM_ROI", 15.0))
    guard_callback_roi: float = field(default_factory=lambda: _env_float("GUARD_CALLBACK_ROI", 10.0))
    # Armed phase uses Binance's native TRAILING_STOP_MARKET. callbackRate is a
    # PRICE percentage (Binance's own unit), so at 10x a 1.0% callback gives
    # back 10% ROI. Must be smaller than GUARD_ARM_ROI in ROI terms or arming
    # would engage the trail at/below entry — checked per position at arm time.
    guard_trail_callback_pct: float = field(default_factory=lambda: _env_float("GUARD_TRAIL_CALLBACK_PCT", 1.0))
    guard_use_native_trail: bool = field(default_factory=lambda: _env_bool("GUARD_USE_NATIVE_TRAIL", True))
    # Preferred: express the trail give-back in ROI%. It is converted to the
    # price percentage Binance wants using each position's own leverage, so the
    # behaviour is identical at 10x and 20x. 0 = fall back to the price percent.
    guard_trail_callback_roi: float = field(default_factory=lambda: _env_float("GUARD_TRAIL_CALLBACK_ROI", 0.0))
    # Volatility-scaled stop + sizing. 0 disables (fixed stop, flat % margin).
    # When set, stop = mult x ATR and the position is sized so the loss at that
    # stop equals ENTRY_RISK_PCT of the wallet — constant risk across coins.
    atr_stop_mult: float = field(default_factory=lambda: _env_float("ATR_STOP_MULT", 0.0))
    # Candle timeframe for ATR. Defaults to the scanner timeframe so stops are
    # sized on the chart being traded — a 15m ATR is ~2.2x a 3m ATR.
    atr_timeframe: str = field(default_factory=lambda: _env(
        "ATR_TIMEFRAME", _env("SCANNER_TIMEFRAME", "3m")))
    atr_stop_min_roi: float = field(default_factory=lambda: _env_float("ATR_STOP_MIN_ROI", 4.0))
    atr_stop_max_roi: float = field(default_factory=lambda: _env_float("ATR_STOP_MAX_ROI", 30.0))
    # Close a position discovered ALREADY past its stop. The loss level has been
    # breached, so closing is what the stop was for; trailing it instead protects
    # nothing. Set false to only report and leave it open.
    guard_close_if_past_stop: bool = field(default_factory=lambda: _env_bool("GUARD_CLOSE_IF_PAST_STOP", True))
    # Move the stop to breakeven once peak ROI reaches this. Sits below the
    # trail, so arm/give-back behaviour above arm_roi is unchanged. 0 disables.
    guard_breakeven_at_roi: float = field(default_factory=lambda: _env_float("GUARD_BREAKEVEN_AT_ROI", 0.0))
    # Where the stop goes at that point. Exactly 0 is entry price, which still
    # loses the round-trip fee (~2% ROI at 20x), so a small positive value is
    # closer to true breakeven.
    guard_breakeven_stop_roi: float = field(default_factory=lambda: _env_float("GUARD_BREAKEVEN_STOP_ROI", 0.0))
    entry_risk_pct: float = field(default_factory=lambda: _env_float("ENTRY_RISK_PCT", 1.0))

    # ── Unattended auto-trading ─────────────────────────────────────────
    # OFF by default and also toggleable from the dashboard. Opens leveraged
    # positions without supervision, so the safety limits below are not
    # optional extras — they are what makes it survivable.
    auto_trade_enabled: bool = field(default_factory=lambda: _env_bool("AUTO_TRADE_ENABLED", False))
    auto_interval: int = field(default_factory=lambda: _env_int("AUTO_TRADE_INTERVAL", 30))
    auto_max_dist_pct: float = field(default_factory=lambda: _env_float("AUTO_MAX_DIST_PCT", 3.0))
    # Absolute floor on the entry callback, as a % of price. The exchange
    # minimum is far below noise width on a mover: the two losers of 2026-09-12
    # triggered on 0.38% and 0.58% retraces, while the winner that morning had
    # 1.15%. Unlike the velocity floor this depends on no measure the bot has
    # not been recording. 0 keeps the exchange minimum.
    auto_callback_min_pct: float = field(default_factory=lambda: _env_float(
        "AUTO_CALLBACK_MIN_PCT", 0.0))
    # Restrict the timed fail-fast to positions that are WORSE than where they
    # were first seen, so a recovering position is never cut on depth alone.
    guard_fail_fast_require_worsening: bool = field(default_factory=lambda: _env_bool(
        "GUARD_FAIL_FAST_REQUIRE_WORSENING", False))
    # Longs only: require the EMA gap to be narrowing. Default TRUE — the
    # condition the scanner's own comment already claimed to apply.
    auto_long_require_convergence: bool = field(default_factory=lambda: _env_bool(
        "AUTO_LONG_REQUIRE_CONVERGENCE", True))
    auto_callback_use_velocity: bool = field(default_factory=lambda: _env_bool(
        "AUTO_CALLBACK_USE_VELOCITY", False))
    auto_recent_tr_candles: int = field(default_factory=lambda: _env_int(
        "AUTO_RECENT_TR_CANDLES", 3))
    scan_extreme_band_pct: float = field(default_factory=lambda: _env_float(
        "SCAN_EXTREME_BAND_PCT", 0.1))
    # Veto candidates whose move is still expanding. Default OFF — this changes
    # which trades are taken, so it is opt-in.
    auto_veto_breakout: bool = field(default_factory=lambda: _env_bool(
        "AUTO_VETO_BREAKOUT", False))
    auto_strength_sweeps: int = field(default_factory=lambda: _env_int("AUTO_STRENGTH_SWEEPS", 2))
    auto_long_rsi_min: float = field(default_factory=lambda: _env_float("AUTO_LONG_RSI_MIN", 48.0))
    # Ceiling for longs. 0 disables it (the previous behaviour, where longs
    # inherited the scanner's much wider band).
    auto_long_rsi_max: float = field(default_factory=lambda: _env_float("AUTO_LONG_RSI_MAX", 0.0))
    # "all", "long" or "short" — disable one side without redeploying.
    auto_directions: str = field(default_factory=lambda: (
        _env("AUTO_DIRECTIONS", "all").strip().lower() or "all"))
    # Whether the daily-loss halt is active. Disabling it removes the only
    # automatic brake on a bad day.
    auto_daily_halt_enabled: bool = field(default_factory=lambda: _env_bool(
        "AUTO_DAILY_HALT_ENABLED", True))
    # Restrict ENTRIES to a UTC window, e.g. "05:00-11:00". Exits are never
    # restricted. Empty trades around the clock.
    auto_trading_window: str = field(default_factory=lambda: _env(
        "AUTO_TRADING_WINDOW", "").strip())
    # Minimum ATR% to enter. 0 disables.
    auto_min_atr_pct: float = field(default_factory=lambda: _env_float(
        "AUTO_MIN_ATR_PCT", 0.0))
    guard_fail_fast_s: float = field(default_factory=lambda: _env_float(
        "GUARD_FAIL_FAST_S", 0.0))
    guard_fail_fast_max_peak_roi: float = field(default_factory=lambda: _env_float(
        "GUARD_FAIL_FAST_MAX_PEAK_ROI", 0.0))
    guard_fail_fast_loss_roi: float = field(default_factory=lambda: _env_float(
        "GUARD_FAIL_FAST_LOSS_ROI", 5.0))
    # MARK_PRICE or CONTRACT_PRICE. Default MARK_PRICE: contract price is the
    # last trade on this book, so a wick closes the position.
    guard_stop_working_type: str = field(default_factory=lambda: (
        _env("GUARD_STOP_WORKING_TYPE", "MARK_PRICE") or "MARK_PRICE").upper())
    # Price-% callback for the rescue trail used when a fixed stop is refused.
    # Tight by design — a refused stop means "get out", not "ride it". 0.1 is
    # the exchange minimum.
    guard_rescue_trail_callback_pct: float = field(default_factory=lambda: _env_float(
        "GUARD_RESCUE_TRAIL_CALLBACK_PCT", 0.1))
    # Start with SPOT trading halted. Useful when a container is redeployed
    # mid-session and you want to inspect before it can act.
    kill_switch_on_start: bool = field(default_factory=lambda: _env_bool(
        "KILL_SWITCH_ON_START", False))
    auto_short_rsi_min: float = field(default_factory=lambda: _env_float("AUTO_SHORT_RSI_MIN", 78.0))
    auto_callback_ratio: float = field(default_factory=lambda: _env_float("AUTO_CALLBACK_RATIO", 0.5))
    auto_callback_atr_mult: float = field(default_factory=lambda: _env_float("AUTO_CALLBACK_ATR_MULT", 0.75))
    auto_daily_loss_limit_pct: float = field(default_factory=lambda: _env_float("AUTO_DAILY_LOSS_LIMIT_PCT", 5.0))
    auto_symbol_cooldown_s: float = field(default_factory=lambda: _env_float("AUTO_SYMBOL_COOLDOWN_S", 1800.0))
    auto_max_trades_per_hour: int = field(default_factory=lambda: _env_int("AUTO_MAX_TRADES_PER_HOUR", 6))
    # Re-enter a cooled-down symbol when the signal has pushed this much
    # further into the extreme than the entry that failed. 0 = strict timer.
    auto_cooldown_override_rsi: float = field(default_factory=lambda: _env_float("AUTO_COOLDOWN_OVERRIDE_RSI", 3.0))
    auto_max_reentries_per_symbol: int = field(default_factory=lambda: _env_int("AUTO_MAX_REENTRIES_PER_SYMBOL", 2))

    # ── Operator-initiated futures entry (UI button) ────────────────────
    # The ONLY component that can OPEN a position. Off by default; honours
    # GUARDIAN_DRY_RUN. All limits are re-checked server-side at execute time.
    futures_entry_enabled: bool = field(default_factory=lambda: _env_bool("FUTURES_ENTRY_ENABLED", False))
    entry_max_positions: int = field(default_factory=lambda: _env_int("ENTRY_MAX_POSITIONS", 1))
    entry_default_margin_pct: float = field(default_factory=lambda: _env_float("ENTRY_DEFAULT_MARGIN_PCT", 10.0))
    entry_max_margin_pct: float = field(default_factory=lambda: _env_float("ENTRY_MAX_MARGIN_PCT", 25.0))
    entry_default_callback_pct: float = field(default_factory=lambda: _env_float("ENTRY_DEFAULT_CALLBACK_PCT", 0.1))
    # Fallback leverage when the exchange reports none. Declaring it is an
    # explicit statement of what is set on Binance — not a silent default.
    # Demo does not report leverage for a symbol with no open position, which
    # blocked entries entirely. Default to 10x — it must match what is actually
    # configured on Binance, so the preview labels the value as ASSUMED to make
    # a mismatch visible before confirming.
    entry_assumed_leverage: float = field(default_factory=lambda: _env_float("ENTRY_ASSUMED_LEVERAGE", 10.0))
    # How long an unfilled entry order may rest before the bot cancels it. A
    # GTC order that never fills blocks its symbol and, if it eventually
    # triggers, opens a position sized for conditions long past.
    # Also cancel stale entry orders the bot has NO record of (e.g. placed
    # before tracking existed). Leave false if you place entries by hand —
    # those would be cancelled too.
    entry_reap_untracked: bool = field(default_factory=lambda: _env_bool("ENTRY_REAP_UNTRACKED", False))
    # Cancel reduce-only stops resting on symbols with no open position. Such
    # an order cannot protect anything but can fire against a FUTURE position
    # on the same symbol. Defaults ON: a protective stop with nothing to
    # protect is stale whoever placed it.
    guard_sweep_orphan_stops: bool = field(default_factory=lambda: _env_bool("GUARD_SWEEP_ORPHAN_STOPS", True))
    entry_order_ttl_s: float = field(default_factory=lambda: max(
        60.0, _env_float("ENTRY_ORDER_TTL_S", 900.0)))

    # ── Candidate scanner (read-only futures market screen) ─────────────
    # Surfaces potential long/short candidates for operator review. Uses PUBLIC
    # futures market data only — no keys, no trading permissions.
    scanner_enabled: bool = field(default_factory=lambda: _env_bool("SCANNER_ENABLED", False))
    scanner_interval: int = field(default_factory=lambda: _env_int("SCANNER_INTERVAL", 120))
    scanner_timeframe: str = field(default_factory=lambda: _env("SCANNER_TIMEFRAME", "5m"))
    scanner_max_symbols: int = field(default_factory=lambda: _env_int("SCANNER_MAX_SYMBOLS", 40))
    # Scan the same environment the account trades on, so every candidate is
    # actually tradable. Defaults to follow the guardian; override if needed.
    scanner_demo: bool = field(default_factory=lambda: _env_bool(
        "SCANNER_DEMO", _env_bool("GUARDIAN_DEMO",
        _env_bool("GUARDIAN_TESTNET", _env_bool("TESTNET", True)))))
    scan_min_vol_usdt: float = field(default_factory=lambda: _env_float("SCAN_MIN_VOL_USDT", 50_000_000))
    # "absolute" or "percentile". Demo reports inflated volumes, which makes an
    # absolute floor inert — percentile mode stays meaningful in both.
    scan_volume_mode: str = field(default_factory=lambda: _env("SCAN_VOLUME_MODE", "absolute").lower())
    scan_vol_percentile: float = field(default_factory=lambda: _env_float("SCAN_VOL_PERCENTILE", 60.0))
    # Volatility band, ATR as % of price. 0 disables. A coin below the floor is
    # too quiet for the stop to survive noise-free; above the ceiling a fixed
    # stop sits inside normal swings.
    scan_min_atr_pct: float = field(default_factory=lambda: _env_float("SCAN_MIN_ATR_PCT", 0.0))
    scan_max_atr_pct: float = field(default_factory=lambda: _env_float("SCAN_MAX_ATR_PCT", 0.0))
    scan_min_change_pct: float = field(default_factory=lambda: _env_float("SCAN_MIN_CHANGE_PCT", 5.0))
    scan_short_rsi_min: float = field(default_factory=lambda: _env_float("SCAN_SHORT_RSI_MIN", 70.0))
    scan_long_rsi_min: float = field(default_factory=lambda: _env_float("SCAN_LONG_RSI_MIN", 50.0))
    scan_long_rsi_max: float = field(default_factory=lambda: _env_float("SCAN_LONG_RSI_MAX", 65.0))
    # How far past the EMA crossover a coin can sit and still be screened for
    # its direction. Catches the just-crossed roll-over that a hard sign test
    # made invisible. 0 restores the strict behaviour.
    scan_ema_tolerance_pct: float = field(default_factory=lambda: _env_float("SCAN_EMA_TOLERANCE_PCT", 0.15))

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
    # Good-price gate. The upper-wick veto is the PRIMARY filter (avoid spike
    # rejections). candle_pos_max is now a high BACKSTOP (default 0.90) that only
    # catches the extreme top — a clean high close with no wick passes.
    pb_candle_pos_max: float = field(default_factory=lambda: _env_float("PB_CANDLE_POS_MAX", 0.90))
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

        # Keys and endpoint must describe the same environment. Sending demo
        # credentials to the live endpoint (or the reverse) fails in ways that
        # are hard to read, so say so plainly at startup.
        if self.guardian_enabled and self.testnet != self.guardian_demo:
            import logging
            logging.getLogger("config").warning(
                f"ENVIRONMENT MISMATCH: TESTNET={self.testnet} selects the "
                f"{'TEST/demo' if self.testnet else 'LIVE'} API keys, but "
                f"GUARDIAN_DEMO={self.guardian_demo} points them at the "
                f"{'demo' if self.guardian_demo else 'live'} endpoint. "
                f"Unset GUARDIAN_DEMO to follow TESTNET, or make them agree."
            )

        raw_ttl = _env_float("ENTRY_ORDER_TTL_S", 900.0)
        if raw_ttl < 60.0:
            import logging
            logging.getLogger("config").warning(
                f"ENTRY_ORDER_TTL_S={raw_ttl:g} is below the 60s floor and has "
                f"been raised to 60. A shorter life would cancel entry orders "
                f"before they could plausibly fill, and 0 or less would cancel "
                f"every one on the next cycle.")

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
