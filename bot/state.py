from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PositionState:
    entry_price: float
    qty: float
    trailing_stop: float
    candles_held: int = 0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # ID of the server-side backstop order (OCO or stop-market).
    # None if placement failed or was not attempted.
    oco_order_list_id: str | None = None
    # Type of backstop placed: "oco", "stop_market", or None
    backstop_type: str | None = None
    # Whether the in-memory trailing stop is active yet.
    # Defaults True (immediate trailing — original behaviour). When the activation
    # threshold feature is on, new positions start False and flip to True once price
    # first reaches the activation threshold. Once True it never reverts.
    trailing_active: bool = True
    # The price level at which trailing activates (entry * (1 + activation_pct/100)).
    # 0 means "already active / no threshold" (the default immediate-trailing case).
    activation_price: float = 0.0
    # Highest P&L percent this position has reached, for the continuous profit lock.
    # Ratchets up only; the locked profit floor is derived from this peak.
    peak_pnl_pct: float = 0.0
    # ── Live market values — the single source of truth ──────────────────
    # The fast monitor updates these from the live price each pass. The dashboard
    # READS these rather than fetching its own ticker, so the displayed price/P&L
    # and the values the engine acts on are always the same number. 0.0 until the
    # first monitor pass populates them (the dashboard falls back to entry price).
    current_price: float = 0.0
    pnl_pct: float = 0.0
    pnl_usdt: float = 0.0
    # Wall-clock time of the last live-price update, for staleness display.
    last_price_ts: float = 0.0
