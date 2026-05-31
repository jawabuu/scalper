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
    take_profit_pct: float = field(default_factory=lambda: _env_float("TAKE_PROFIT_PCT", 1.5))
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

    # ── Cooldown ────────────────────────────────────────────────────────
    # Number of candles to wait before re-entering a manually closed symbol.
    # Prevents the bot immediately re-buying something you just closed.
    manual_close_cooldown_candles: int = field(default_factory=lambda: _env_int("MANUAL_CLOSE_COOLDOWN_CANDLES", 3))

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
