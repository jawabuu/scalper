"""
Persistent position store.

Wraps a dict[str, PositionState] and writes positions.json to disk on every
mutation (add, update, delete). On startup the engine loads this file as the
primary source of truth for recovery — no inference from Binance balances or
OCO orders needed.

File location: logs/positions.json (same volume mount as bot.log, so it
survives container restarts as long as the volume is intact).
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from .state import PositionState

log = logging.getLogger("store")

STORE_PATH = Path("logs/positions.json")


def _serialize(positions: dict[str, PositionState]) -> dict:
    out = {}
    for sym, pos in positions.items():
        out[sym] = {
            "entry_price":        pos.entry_price,
            "qty":                pos.qty,
            "trailing_stop":      pos.trailing_stop,
            "candles_held":       pos.candles_held,
            "opened_at":          pos.opened_at.isoformat(),
            "oco_order_list_id":  pos.oco_order_list_id,
            "backstop_type":      pos.backstop_type,
            "trailing_active":    pos.trailing_active,
            "activation_price":   pos.activation_price,
            "peak_pnl_pct":       pos.peak_pnl_pct,
        }
    return out


def _deserialize(data: dict) -> dict[str, PositionState]:
    positions = {}
    for sym, d in data.items():
        try:
            positions[sym] = PositionState(
                entry_price=float(d["entry_price"]),
                qty=float(d["qty"]),
                trailing_stop=float(d["trailing_stop"]),
                candles_held=int(d.get("candles_held", 0)),
                opened_at=datetime.fromisoformat(d["opened_at"]),
                oco_order_list_id=d.get("oco_order_list_id"),
                backstop_type=d.get("backstop_type"),
                trailing_active=d.get("trailing_active", True),
                activation_price=float(d.get("activation_price", 0.0)),
                peak_pnl_pct=float(d.get("peak_pnl_pct", 0.0)),
            )
        except Exception as e:
            log.warning(f"Skipping malformed position record for {sym}: {e}")
    return positions


class PositionStore:
    """
    Dict-like wrapper around positions that flushes to disk on every write.
    Thread-safe: engine thread writes, API thread reads via snapshot().
    """

    def __init__(self, path: Path = STORE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, PositionState] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self):
        if not self._path.exists():
            log.info(f"No position store found at {self._path} — starting fresh.")
            return
        try:
            raw = json.loads(self._path.read_text())
            self._data = _deserialize(raw)
            if self._data:
                log.info(f"Loaded {len(self._data)} position(s) from {self._path}: {list(self._data)}")
            else:
                log.info("Position store exists but is empty.")
        except Exception as e:
            log.error(f"Could not load position store: {e} — starting fresh.")
            self._data = {}

    def _flush(self):
        """Write current state to disk. Called after every mutation."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_serialize(self._data), indent=2))
            tmp.replace(self._path)  # atomic on POSIX
        except Exception as e:
            log.error(f"Could not persist positions: {e}")

    # ── Dict interface ───────────────────────────────────────────────────

    def __contains__(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __getitem__(self, symbol: str) -> PositionState:
        with self._lock:
            return self._data[symbol]

    def __setitem__(self, symbol: str, pos: PositionState):
        with self._lock:
            self._data[symbol] = pos
            self._flush()

    def __delitem__(self, symbol: str):
        with self._lock:
            del self._data[symbol]
            self._flush()

    def get(self, symbol: str, default=None):
        with self._lock:
            return self._data.get(symbol, default)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def items(self):
        with self._lock:
            return list(self._data.items())

    def values(self):
        with self._lock:
            return list(self._data.values())

    def clear(self):
        with self._lock:
            self._data.clear()
            self._flush()

    def snapshot(self) -> dict[str, PositionState]:
        """Thread-safe copy for API reads."""
        with self._lock:
            return dict(self._data)

    def update_stop(self, symbol: str, new_stop: float):
        """Update trailing stop in-place and flush."""
        with self._lock:
            self._data[symbol].trailing_stop = new_stop
            self._flush()

    def increment_candles(self, symbol: str):
        """Increment candles_held in-place and flush."""
        with self._lock:
            self._data[symbol].candles_held += 1
            self._flush()

    def set_candles(self, symbol: str, value: int):
        """Set candles_held to an absolute value (true time-based count) and flush."""
        with self._lock:
            if symbol in self._data:
                self._data[symbol].candles_held = int(value)
                self._flush()
