"""
Cross-evaluation between two instances.

When one instance enters a trade it asks the other: "would YOU have taken
this?" The peer answers from its OWN scan snapshot and its OWN config, and
both sides are written to a JSONL file.

WHY IT ANSWERS SOMETHING THE TRADE LOGS CANNOT. Over one day demo and live
traded 15 symbols each and overlapped on FOUR. Demo won 77% and live 41% on
identical rules. Nothing in either log says whether that is the config, the
market, or which coins each happened to see — and the charts are near-identical
in shape, so it is unlikely to be the strategy.

The answer separates into three cases, and the peer can distinguish all three:

    not_surfaced  the peer's scanner never produced this symbol at all —
                  it was not a mover, or sat under the volume floor. The
                  DATA differed, not the rules.
    refused       the peer had it and its rules said no, WITH the reason.
                  The rules differed, on the same opportunity.
    would_enter   the peer agrees. Divergence is then execution: fills,
                  timing, position slots.

NO EXCHANGE CALLS. The peer answers from the snapshot it already holds, so
this costs no API weight and cannot be used to drive requests. It is also the
truer comparison: the snapshot is what that instance ACTUALLY had in hand at
that moment, not a reconstruction.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("peer_eval")

# Evaluation is cheap, but a peer that floods is still a peer that floods.
MAX_PER_MINUTE = 60


class PeerEval:
    """
    Sending, receiving, or both.

    Failure is always silent from the trading loop's point of view: this is
    instrumentation, and instrumentation that can break a trade is worse than
    no instrumentation.
    """

    def __init__(self, mode: str = "both", peer_url: str = "",
                 label: str = "", path: str = "", timeout_s: float = 4.0):
        self.mode = (mode or "off").strip().lower()
        self.peer_url = (peer_url or "").rstrip("/")
        self.label = label or "unknown"
        self.path = Path(path) if path else None
        self.timeout_s = float(timeout_s)
        self._lock = threading.RLock()
        self._recent: list[float] = []
        self.sent = 0
        self.received = 0
        self.errors = 0

    @property
    def sends(self) -> bool:
        return self.mode in ("send", "both") and bool(self.peer_url)

    @property
    def receives(self) -> bool:
        return self.mode in ("receive", "both")

    # ── sending ────────────────────────────────────────────────────────────
    def notify_entry(self, symbol: str, side: str, row: dict, decision_reason: str):
        """
        Tell the peer we entered. Fire-and-forget on its own thread: the
        trading loop must never wait on the other container.
        """
        if not self.sends:
            return
        payload = {
            "from": self.label,
            "symbol": symbol,
            "side": side,
            "at": time.time(),
            "reason": decision_reason,
            "readings": _readings(row),
        }
        t = threading.Thread(target=self._post, args=(payload,),
                             name="peer-eval-send", daemon=True)
        t.start()

    def _post(self, payload: dict):
        try:
            import requests
            r = requests.post(f"{self.peer_url}/api/peer/evaluate",
                              json=payload, timeout=self.timeout_s)
            body = r.json() if r.ok else {"error": f"HTTP {r.status_code}"}
            with self._lock:
                self.sent += 1
            self.record({"kind": "sent", "at": time.time(),
                         "mine": payload, "theirs": body})
        except Exception as e:
            with self._lock:
                self.errors += 1
            # One line, not one per failure: a peer that is down stays down.
            if self.errors in (1, 10) or self.errors % 100 == 0:
                log.warning(f"peer eval send failed ({self.errors}): "
                            f"{str(e)[:120]}")

    # ── receiving ──────────────────────────────────────────────────────────
    def _rate_ok(self) -> bool:
        now = time.time()
        with self._lock:
            self._recent = [t for t in self._recent if now - t < 60.0]
            if len(self._recent) >= MAX_PER_MINUTE:
                return False
            self._recent.append(now)
            return True

    def evaluate(self, payload: dict, scanner, auto) -> dict:
        """
        Answer "would you have taken this?" from our own snapshot and config.

        Never raises: a malformed payload from the peer returns an error
        object rather than propagating into the API thread.
        """
        if not self.receives:
            return {"verdict": "disabled", "label": self.label}
        if not self._rate_ok():
            return {"verdict": "rate_limited", "label": self.label}
        try:
            symbol = str(payload.get("symbol") or "")
            side = str(payload.get("side") or "")
            if not symbol:
                return {"verdict": "error", "detail": "no symbol"}

            snap = scanner.snapshot() if scanner is not None else {}
            rows = {r.get("symbol"): r for r in (snap.get("candidates") or [])}
            row = rows.get(symbol)

            out = {"label": self.label, "symbol": symbol, "side": side,
                   "at": time.time()}
            if row is None:
                # The most informative answer: our pipeline never saw it.
                out.update({
                    "verdict": "not_surfaced",
                    "detail": ("this symbol is not in our current scan — not a "
                               "mover, under the volume floor, or outside the "
                               "RSI screen"),
                    "scan_size": len(rows),
                })
                self.record({"kind": "received", "at": time.time(),
                             "theirs": payload, "mine": out})
                return out

            out["readings"] = _readings(row)
            if auto is None:
                out["verdict"] = "no_auto_trader"
                return out

            from bot.auto_trader import evaluate_candidate
            streak = 99   # strength streak is per-instance history, not a rule
            d = evaluate_candidate(row, streak=streak, cfg=auto.cfg,
                                   atr_pct=row.get("atr_pct"))
            out.update({
                "verdict": "would_enter" if d.enter else "refused",
                "detail": d.reason,
            })
            self.record({"kind": "received", "at": time.time(),
                         "theirs": payload, "mine": out})
            with self._lock:
                self.received += 1
            return out
        except Exception as e:
            with self._lock:
                self.errors += 1
            log.warning(f"peer eval failed: {str(e)[:160]}")
            return {"verdict": "error", "detail": str(e)[:160],
                    "label": self.label}

    # ── calibration ────────────────────────────────────────────────────────
    def notify_scan(self, rows: list[dict]):
        """
        Send our whole candidate list so the peer can pair readings with its
        own for the SAME symbol at the SAME moment.

        Entry notifications are too sparse to calibrate anything — they fire a
        few times an hour. Every shared candidate on every scan gives dozens of
        paired observations, which is what is needed before scaling a setting
        by a ratio.
        """
        if not self.sends or not rows:
            return
        payload = {"from": self.label, "at": time.time(), "kind": "scan",
                   "rows": [{"symbol": r.get("symbol"), **_readings(r)}
                            for r in rows if r.get("symbol")]}
        t = threading.Thread(target=self._post_scan, args=(payload,),
                             name="peer-eval-scan", daemon=True)
        t.start()

    def _post_scan(self, payload: dict):
        try:
            import requests
            requests.post(f"{self.peer_url}/api/peer/compare",
                          json=payload, timeout=self.timeout_s)
        except Exception:
            with self._lock:
                self.errors += 1

    def compare(self, payload: dict, scanner) -> dict:
        """Pair the peer's readings with ours and record every overlap."""
        if not self.receives:
            return {"paired": 0, "detail": "disabled"}
        if not self._rate_ok():
            return {"paired": 0, "detail": "rate_limited"}
        try:
            snap = scanner.snapshot() if scanner is not None else {}
            mine = {r.get("symbol"): r for r in (snap.get("candidates") or [])}
            theirs = {r.get("symbol"): r for r in (payload.get("rows") or [])}
            paired = 0
            for sym, t in theirs.items():
                m = mine.get(sym)
                if m is None:
                    continue
                self.record({"kind": "pair", "at": time.time(), "symbol": sym,
                             "peer": payload.get("from"),
                             "theirs": {k: v for k, v in t.items()
                                        if k != "symbol"},
                             "mine": _readings(m)})
                paired += 1
            return {"paired": paired, "mine": len(mine), "theirs": len(theirs)}
        except Exception as e:
            return {"paired": 0, "detail": str(e)[:120]}

    def calibration(self, limit: int = 5000) -> dict:
        """
        Per-field ratio between the two instances, over every paired reading.

        This is what a scaling factor must come from. Three coins off a
        screenshot gave 1.32-1.38 for ATR and could not say whether that is
        stable, coin-dependent, or drifting with the hour.
        """
        import statistics as st
        rows = [r for r in self.report(limit=limit) if r.get("kind") == "pair"]
        fields = ("atr_pct", "rsi", "ema_gap_pct", "dist_to_extreme_pct",
                  "vol_usdt_24h", "recent_tr_pct", "change_24h_pct")
        out = {"pairs": len(rows), "by_symbol": {}, "fields": {}}
        for f in fields:
            ratios = []
            for r in rows:
                a = (r.get("theirs") or {}).get(f)
                b = (r.get("mine") or {}).get(f)
                try:
                    a, b = float(a), float(b)
                except (TypeError, ValueError):
                    continue
                if abs(a) < 1e-9:
                    continue
                ratios.append(b / a)
            if len(ratios) >= 3:
                out["fields"][f] = {
                    "n": len(ratios),
                    "median": round(st.median(ratios), 4),
                    "mean": round(st.mean(ratios), 4),
                    "stdev": round(st.pstdev(ratios), 4),
                    "min": round(min(ratios), 4),
                    "max": round(max(ratios), 4),
                }
        syms = {}
        for r in rows:
            syms[r.get("symbol")] = syms.get(r.get("symbol"), 0) + 1
        out["by_symbol"] = dict(sorted(syms.items(), key=lambda kv: -kv[1])[:20])
        return out

    # ── recording ──────────────────────────────────────────────────────────
    def record(self, entry: dict):
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str,
                                    separators=(",", ":")) + "\n")
        except Exception:
            pass

    def report(self, limit: int = 500) -> list[dict]:
        if not self.path or not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()[-int(limit):]
            out = []
            for ln in lines:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def status(self) -> dict:
        with self._lock:
            return {"mode": self.mode, "label": self.label,
                    "peer": self.peer_url or None,
                    "sends": self.sends, "receives": self.receives,
                    "sent": self.sent, "received": self.received,
                    "errors": self.errors}


# The readings that decide an entry, so a disagreement can be traced to the
# number it came from rather than just to a verdict.
_FIELDS = ("rsi", "atr_pct", "dist_to_extreme_pct", "ema_gap_pct",
           "change_24h_pct", "vol_usdt_24h", "strength", "turned_up",
           "gap_rising", "breakout", "efficiency", "recent_tr_pct",
           "callback_pct", "streak")


def _readings(row: dict) -> dict:
    out = {}
    for k in _FIELDS:
        if k in (row or {}):
            out[k] = row[k]
    adv = (row or {}).get("advance") or {}
    for k in ("adv_vol_trend", "peak_vol_early", "adv_price_pct"):
        if k in adv:
            out[k] = adv[k]
    return out
