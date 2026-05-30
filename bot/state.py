from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PositionState:
    entry_price: float
    qty: float
    trailing_stop: float
    candles_held: int = 0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # ID of the OCO order placed as a backstop when the bot enters.
    # Cancelled and re-evaluated on recovery; None if OCO placement failed.
    oco_order_list_id: str | None = None
