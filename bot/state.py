from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


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
