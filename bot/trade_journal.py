"""
Append-only journal of closed trades.

Closed trades used to live inside futures_state.json, which `save_state()`
rewrites in full every guardian cycle — every 2.5 seconds. That is 34,560 full
serialisations a day of a file whose trade history only ever grows:

    331 trades    0.73 MB  ->   25 GB written per day
    1,800 trades  3.99 MB  ->  138 GB per day
    5,000 trades 11.07 MB  ->  383 GB per day

The trades are append-only and are never read during a cycle. They were in
there only because it was one file.

The cap that kept it from getting worse — DEFAULT_MAX_CLOSED_TRADES = 5000 —
silently DROPPED the oldest trades, about 83 days at 60 trades a day. A run
would quietly stop being able to answer questions about its own first month.

This module writes one JSON object per line when a trade closes, and nothing
at all in between. Rotation moves whole files aside rather than discarding
rows, so history is archived instead of lost.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import logging

log = logging.getLogger("trade_journal")

# One line per trade; a day at 60 trades/day is ~130 KB.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_KEEP_ARCHIVES = 12


class TradeJournal:
    """JSON-lines trade log with size-based rotation."""

    def __init__(self, path: str,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 keep_archives: int = DEFAULT_KEEP_ARCHIVES):
        self.path = Path(path) if path else None
        self.max_bytes = int(max_bytes)
        self.keep_archives = int(keep_archives)
        self._warned = False

    # ── writing ────────────────────────────────────────────────────────────
    def append(self, rec: dict) -> bool:
        """
        Add one closed trade. Called once per trade, not once per cycle.

        A failure here must never stop the guardian: the trade has already
        happened, and the in-memory list still has it.
        """
        if not self.path:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(rec, default=str, separators=(",", ":"))
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._rotate_if_needed()
            return True
        except Exception as e:
            if not self._warned:
                log.error(f"trade journal unwritable at {self.path}: {e} — "
                          f"history will not survive a restart")
                self._warned = True
            return False

    def _rotate_if_needed(self):
        try:
            if self.path.stat().st_size < self.max_bytes:
                return
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            # ALWAYS suffixed and zero-padded, so a plain lexical sort matches
            # write order. Naming the first archive "trades-STAMP.jsonl" and
            # the next "trades-STAMP-1.jsonl" sorts the SECOND one first — "-"
            # precedes "." — which silently reordered history on read.
            n = 0
            while True:
                archive = self.path.with_name(
                    f"{self.path.stem}-{stamp}-{n:03d}.jsonl")
                if not archive.exists():
                    break
                n += 1
            self.path.rename(archive)
            log.warning(f"trade journal rotated to {archive.name}")
            self._prune_archives()
        except Exception as e:
            log.warning(f"trade journal rotation failed: {e}")

    def _prune_archives(self):
        """Keep the newest N archives. Rotation ARCHIVES; it does not delete
        history until there is a lot of it."""
        try:
            olds = sorted(self.path.parent.glob(f"{self.path.stem}-*.jsonl"))
            for p in olds[:-self.keep_archives] if self.keep_archives else []:
                p.unlink()
                log.info(f"pruned old trade archive {p.name}")
        except Exception:
            pass

    # ── reading ────────────────────────────────────────────────────────────
    def load(self, limit: int | None = None) -> list[dict]:
        """
        Read trades back, oldest first. `limit` returns the most recent N,
        which is what a dashboard wants without paying for the whole file.
        """
        if not self.path:
            return []
        try:
            # Read back through archives when the live file alone cannot
            # satisfy the request. Without this, the dashboard would show
            # almost nothing for a while after every rotation.
            files = [self.path] if self.path.exists() else []
            need = int(limit) if limit else None
            if need is None or sum(1 for f in files
                                   for _ in open(f, encoding="utf-8")) < need:
                archives = sorted(
                    self.path.parent.glob(f"{self.path.stem}-*.jsonl"))
                files = archives + files
            lines = []
            for f in files:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        lines.extend(fh.readlines())
                except Exception:
                    continue
            if limit:
                lines = lines[-int(limit):]
            out = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    # One corrupt line must not lose the file. A partial write
                    # at the tail is the expected case after a hard kill.
                    continue
            return out
        except Exception as e:
            log.error(f"trade journal unreadable at {self.path}: {e}")
            return []

    def import_existing(self, trades: list[dict]) -> int:
        """
        One-time migration of trades already inside futures_state.json.

        Only runs when the journal is empty, so it cannot duplicate on a
        restart.
        """
        if not self.path or not trades:
            return 0
        try:
            if self.path.exists() and self.path.stat().st_size > 0:
                return 0
            n = 0
            for rec in trades:
                if self.append(rec):
                    n += 1
            if n:
                log.warning(
                    f"migrated {n} trade(s) from the state file into "
                    f"{self.path.name}. They are no longer rewritten every "
                    f"guardian cycle.")
            return n
        except Exception as e:
            log.error(f"trade journal migration failed: {e}")
            return 0

    def stats(self) -> dict:
        out = {"path": str(self.path) if self.path else None,
               "bytes": 0, "archives": 0}
        try:
            if not self.path:
                return out
            # Archives are counted whether or not the LIVE file exists: right
            # after a rotation it does not, until the next trade closes.
            out["archives"] = len(list(
                self.path.parent.glob(f"{self.path.stem}-*.jsonl")))
            if self.path.exists():
                out["bytes"] = self.path.stat().st_size
        except Exception:
            pass
        return out
