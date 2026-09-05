"""
Scan runner — wires the pure scanner logic to live Binance USD-M futures data.

READ-ONLY. Uses public market endpoints only (tickers + OHLCV). It holds no API
keys, has no trading permissions, and cannot place, modify or cancel an order.
Its entire output is a list of candidates for the operator to look at.

Efficiency note: fetching OHLCV per symbol is the expensive call, so symbols are
pre-filtered on the cheap bulk ticker feed (volume + 24h move) first, and only
the survivors get an OHLCV fetch.
"""
from __future__ import annotations

import logging
import threading
import time

import ccxt
import pandas as pd

from .scanner import (
    ScanConfig, Candidate, ScanTracker, Delta,
    evaluate_symbol, rank_with_deltas,
)

log = logging.getLogger("scan_runner")


class ScanRunner:
    def __init__(self, cfg: ScanConfig, timeframe: str = "5m",
                 max_symbols: int = 40, socks_proxy: str | None = None,
                 interval: int = 120):
        self.cfg = cfg
        self.timeframe = timeframe
        self.max_symbols = max_symbols
        self.interval = interval
        self.tracker = ScanTracker()

        params: dict = {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        if socks_proxy:
            params["proxies"] = {"http": socks_proxy, "https": socks_proxy}
        # No apiKey/secret — public data only, cannot trade.
        self.exchange = ccxt.binanceusdm(params)

        self._lock = threading.Lock()
        self._last_results: list[tuple[Candidate, Delta]] = []
        self._last_scan_ts: float = 0.0
        self._last_error: str | None = None
        self._universe_size: int = 0

    # ── data ────────────────────────────────────────────────────────────────

    def _prefilter(self) -> list[tuple[str, float, float]]:
        """
        Cheap bulk pass: keep only symbols that clear the volume floor and are
        genuine movers. Returns (symbol, quote_volume, pct_change).
        """
        tickers = self.exchange.fetch_tickers()
        out = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT:USDT") and not sym.endswith("/USDT"):
                continue
            qv = t.get("quoteVolume") or 0.0
            pct = t.get("percentage")
            if pct is None or qv < self.cfg.min_24h_vol_usdt:
                continue
            if abs(pct) < self.cfg.min_abs_change_pct:
                continue
            out.append((sym, float(qv), float(pct)))
        # Biggest movers first, then cap the OHLCV workload.
        out.sort(key=lambda r: abs(r[2]), reverse=True)
        return out[: self.max_symbols]

    def _ohlcv(self, symbol: str) -> pd.DataFrame | None:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=120)
        except Exception as e:
            log.debug(f"OHLCV failed for {symbol}: {e}")
            return None
        if not raw or len(raw) < 3:
            return None
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        # Drop the still-forming candle so indicators use closed data only.
        if len(df) > 1:
            df = df.iloc[:-1]
        return df

    # ── scan ────────────────────────────────────────────────────────────────

    def scan_once(self) -> list[tuple[Candidate, Delta]]:
        candidates: list[Candidate] = []
        try:
            movers = self._prefilter()
        except Exception as e:
            with self._lock:
                self._last_error = f"ticker fetch failed: {e}"
            log.warning(self._last_error)
            return self._last_results

        self._universe_size = len(movers)
        for sym, qv, pct in movers:
            df = self._ohlcv(sym)
            if df is None:
                continue
            try:
                c = evaluate_symbol(sym, df, qv, pct, self.cfg)
            except Exception as e:
                log.debug(f"evaluate failed for {sym}: {e}")
                continue
            if c is not None:
                candidates.append(c)

        pairs = rank_with_deltas(self.tracker.annotate(candidates))
        self.tracker.commit(candidates)

        with self._lock:
            self._last_results = pairs
            self._last_scan_ts = time.time()
            self._last_error = None
        log.info(f"Scan: {len(candidates)} candidate(s) from {len(movers)} movers")
        return pairs

    def run_forever(self):
        while True:
            try:
                self.scan_once()
            except Exception as e:
                log.warning(f"scan cycle error: {e}")
            time.sleep(self.interval)

    def start_background(self):
        t = threading.Thread(target=self.run_forever, daemon=True, name="scanner")
        t.start()
        log.info(f"Scanner started (every {self.interval}s, tf={self.timeframe})")
        return t

    # ── API surface ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            rows = []
            for c, d in self._last_results:
                row = c.as_row()
                row.update({
                    "strength": d.strength,
                    "rsi_change": None if d.is_new else round(d.rsi_change, 1),
                    "crossed_down": d.crossed_down,
                    "crossed_up": d.crossed_up,
                    "delta_note": d.note,
                })
                rows.append(row)
            return {
                "candidates": rows,
                "last_scan_ts": self._last_scan_ts,
                "last_scan_ago_s": (time.time() - self._last_scan_ts) if self._last_scan_ts else None,
                "universe_size": self._universe_size,
                "error": self._last_error,
                "config": {
                    "min_24h_vol_usdt": self.cfg.min_24h_vol_usdt,
                    "min_abs_change_pct": self.cfg.min_abs_change_pct,
                    "short_rsi_min": self.cfg.short_rsi_min,
                    "long_rsi_min": self.cfg.long_rsi_min,
                    "timeframe": self.timeframe,
                },
            }
