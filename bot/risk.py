"""
Risk manager — ATR-scaled position sizing.

Position size = (balance * risk_pct) / (atr_multiplier * atr_in_usdt)
Capped at max_portfolio_pct of total balance per trade.
"""

import logging
import ccxt

from .config import BotConfig

log = logging.getLogger("risk")


class RiskManager:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self._balance_cache: float = 0
        self._balance_ts: float = 0

    def get_available_usdt(self, exchange: ccxt.Exchange) -> float:
        try:
            balance = exchange.fetch_balance()
            usdt = balance.get("USDT", {}).get("free", 0) or 0
            return float(usdt)
        except Exception as e:
            log.warning(f"Balance fetch failed: {e}")
            return self._balance_cache

    def position_size_usdt(
        self,
        balance: float,
        atr: float,
        price: float,
        atr_multiplier: float = 1.5,
    ) -> float:
        """
        Kelly-lite sizing: risk a fixed % of balance, scaled by ATR volatility.
        A higher ATR (more volatile) shrinks the position; lower ATR grows it.
        """
        if atr <= 0 or price <= 0:
            return 0

        # Dollar value of ATR-based risk per unit
        risk_per_unit = atr * atr_multiplier

        # Units to risk at our allowed dollar risk
        dollar_risk = balance * (self.cfg.risk_per_trade_pct / 100)
        units = dollar_risk / risk_per_unit
        position_usdt = units * price

        # Hard cap: never put more than max_portfolio_pct in one trade
        max_usdt = balance * (self.cfg.max_portfolio_pct / 100)
        position_usdt = min(position_usdt, max_usdt)

        log.debug(
            f"Sizing: balance={balance:.2f} atr={atr:.6f} price={price:.6f} "
            f"→ {position_usdt:.2f} USDT"
        )
        return round(position_usdt, 2)
