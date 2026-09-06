"""
Futures state persistence.

The guardian and auto-trader keep everything in memory, so a redeploy loses it.
Some of that is merely inconvenient (trade history, strength streaks); some is a
genuine risk regression:

  * The stop a position was SIZED for. Without it the guardian recomputes ATR
    at discovery and can place a stop far wider than the position was sized
    for — a 2.8x risk overshoot was traced to exactly this.
  * Each position's PEAK ROI. The trail's high-water mark resets, so a position
    that ran to +40% and pulled back to +20% restarts its peak at 20 and gives
    back the difference.
  * The daily-loss baseline. Restarting while down 4% rebases it, handing back
    a fresh 5% of rope — the halt is meant to stop a bad day, not restart it.
  * Symbol cooldowns and re-entry counts, both of which exist to bound a losing
    sequence.

Written atomically (temp file + rename) so a crash mid-write cannot leave a
truncated file that fails to load. Loading is best-effort: a corrupt or absent
file starts clean rather than blocking startup, because being unable to read
yesterday's state is not a reason to refuse to trade today.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time

log = logging.getLogger("futures_state")

SCHEMA = 1


def _atomic_write(path: str, payload: dict) -> bool:
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".fstate-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    except Exception as e:
        log.warning(f"could not persist futures state to {path}: {e}")
        return False


def save(path: str, *, states: dict, pos_meta: dict, closed_trades: list,
         safety: dict | None = None) -> bool:
    """Persist the state that matters across a restart."""
    payload = {
        "schema": SCHEMA,
        "saved_at": time.time(),
        # Only the fields that change behaviour on reload — not caches, which
        # are cheap to rebuild and stale by definition.
        "states": {
            sym: {
                "peak_roi": s.peak_roi,
                "armed": s.armed,
                "stop_roi": s.stop_roi,
                "stop_order_id": s.stop_order_id,
                "native_trail_id": s.native_trail_id,
            }
            for sym, s in (states or {}).items()
        },
        "pos_meta": {
            sym: {"entry_context": (m or {}).get("entry_context") or {},
                  "opened_seen_at": (m or {}).get("opened_seen_at")}
            for sym, m in (pos_meta or {}).items()
        },
        "closed_trades": list(closed_trades or [])[-200:],
        "safety": safety or {},
    }
    return _atomic_write(path, payload)


def load(path: str) -> dict:
    """
    Read persisted state. Returns {} when absent or unreadable — a missing or
    corrupt file must not stop the bot starting.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        if data.get("schema") != SCHEMA:
            log.warning(f"futures state schema {data.get('schema')} != {SCHEMA}; ignoring")
            return {}
        age = time.time() - float(data.get("saved_at") or 0)
        log.info(f"Restored futures state from {path} "
                 f"({len(data.get('states') or {})} position(s), "
                 f"{len(data.get('closed_trades') or [])} closed trade(s), "
                 f"{age/60:.0f}m old)")
        return data
    except Exception as e:
        log.warning(f"could not read futures state from {path}: {e}")
        return {}


def restore_states(data: dict):
    """Rebuild GuardState objects from persisted form."""
    from .futures_guard import GuardState
    out = {}
    for sym, raw in (data.get("states") or {}).items():
        try:
            st = GuardState()
            st.peak_roi = float(raw.get("peak_roi") or 0.0)
            st.armed = bool(raw.get("armed"))
            st.stop_roi = raw.get("stop_roi")
            st.stop_order_id = raw.get("stop_order_id")
            st.native_trail_id = raw.get("native_trail_id")
            out[sym] = st
        except Exception as e:
            log.debug(f"could not restore state for {sym}: {e}")
    return out
