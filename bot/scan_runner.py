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
    evaluate_symbol, rank_with_deltas, balance_directions,
)

log = logging.getLogger("scan_runner")


class ScanRunner:
    def __init__(self, cfg: ScanConfig, timeframe: str = "5m",
                 max_symbols: int = 40, socks_proxy: str | None = None,
                 interval: int = 120, demo: bool = False):
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
        self.demo = demo
        if demo:
            # Screen the SAME environment the account trades on. Screening the
            # live market while entries execute on demo surfaced candidates that
            # were not tradable there, and produced "no price" / "no leverage"
            # errors at entry time.
            try:
                self.exchange.enable_demo_trading(True)
                log.info("Scanner using DEMO market data (demo-fapi.binance.com)")
            except AttributeError:
                log.warning("ccxt has no enable_demo_trading(); scanner stays on "
                            "live market data — candidates may not be tradable "
                            "on a demo account.")
                self.demo = False

        self._lock = threading.Lock()
        self._last_results: list[tuple[Candidate, Delta]] = []
        self._last_scan_ts: float = 0.0
        self._last_error: str | None = None
        self._universe_size: int = 0
        self._last_rejected: dict = {}
        self._abs_floor_noted: bool = False
        self._last_movers: set = set()
        self._effective_vol_floor: float = 0.0
        # Regime measures, derived from the ticker set already fetched.
        self._breadth_pct: float | None = None
        self._breadth_counts: tuple = (0, 0)
        self._btc_change_pct: float | None = None
        # How many symbols needed the candle fallback for their 24h range —
        # a high count means the ticker feed is not supplying high/low.
        self._range_misses: int = 0
        self._last_duration_s: float = 0.0

    # ── data ────────────────────────────────────────────────────────────────

    def _prefilter(self) -> list[tuple[str, float, float]]:
        """
        Cheap bulk pass: keep only symbols that clear the volume floor and are
        genuine movers. Returns (symbol, quote_volume, pct_change).
        """
        tickers = self.exchange.fetch_tickers()
        self._log_instrument_classes(
            [s for s in tickers
             if s.endswith("/USDT:USDT") or s.endswith("/USDT")])

        # Percentile mode: derive the volume floor from the universe itself, so
        # the filter keeps working where absolute volumes are not comparable to
        # live (demo inflates them).
        vol_floor = self.cfg.min_24h_vol_usdt
        if self.cfg.volume_mode == "percentile":
            vols = []
            for sym, t in tickers.items():
                if not (sym.endswith("/USDT:USDT") or sym.endswith("/USDT")):
                    continue
                if self._excluded(sym) or self._wrong_class(sym):
                    continue      # keep them out of the percentile sample too
                qv = t.get("quoteVolume")
                if qv:
                    vols.append(float(qv))
            if vols:
                vols.sort()
                k = (len(vols) - 1) * self.cfg.vol_percentile / 100.0
                lo, hi = int(k), min(int(k) + 1, len(vols) - 1)
                vol_floor = vols[lo] + (vols[hi] - vols[lo]) * (k - lo)
                # NEUTRALISE the absolute floor for the rest of the scan.
                #
                # There are TWO volume checks. This prefilter uses the
                # percentile; evaluate_symbol -> passes_market_filters then
                # applies cfg.min_24h_vol_usdt AGAIN to every mover. In
                # percentile mode that second check is a hidden hard floor.
                #
                # It hit only live: demo's quote volumes read ~21x higher, so
                # its p85 floor was 850M and every mover cleared 50M by a wide
                # margin. Live's p80 floor was 17.9M, and 14 of 34 movers died
                # against the 50M — the entire selection divergence, from a
                # setting both of us believed the percentile had overridden.
                #
                # Percentile mode means the percentile decides. Full stop.
                if self.cfg.min_24h_vol_usdt:
                    if not getattr(self, "_abs_floor_noted", False):
                        log.warning(
                            f"percentile mode: ignoring "
                            f"SCAN_MIN_VOL_USDT="
                            f"{self.cfg.min_24h_vol_usdt/1e6:.0f}M — the p"
                            f"{self.cfg.vol_percentile:.0f} floor "
                            f"({vol_floor/1e6:.1f}M) is the only volume gate.")
                        self._abs_floor_noted = True
                    self.cfg.min_24h_vol_usdt = 0.0
                log.info(f"Volume floor (p{self.cfg.vol_percentile:.0f}) = "
                         f"{vol_floor/1e6:.1f}M across {len(vols)} symbols")
        self._effective_vol_floor = vol_floor

        # Market breadth, computed from tickers we have already fetched, so it
        # costs nothing. A direct measure of regime rather than a proxy: if
        # most of the market is falling, fading highs runs with the tide; if
        # most is rising, it runs against it.
        up = down = 0
        btc_pct = None
        for sym, t in tickers.items():
            pc = t.get("percentage")
            if pc is None:
                continue
            if sym.startswith("BTC/"):
                btc_pct = float(pc)
            if float(pc) > 0:
                up += 1
            elif float(pc) < 0:
                down += 1
        total = up + down
        self._breadth_pct = round(up / total * 100, 1) if total else None
        self._breadth_counts = (up, down)
        self._btc_change_pct = btc_pct

        out = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT:USDT") and not sym.endswith("/USDT"):
                continue
            if self._excluded(sym) or self._wrong_class(sym):
                continue
            qv = t.get("quoteVolume") or 0.0
            pct = t.get("percentage")
            if pct is None or qv < vol_floor:
                continue
            if abs(pct) < self.cfg.min_abs_change_pct:
                continue
            out.append((sym, float(qv), float(pct),
                        t.get("high"), t.get("low")))
        # Biggest movers first, then cap the OHLCV workload.
        out.sort(key=lambda r: abs(r[2]), reverse=True)
        return out[: self.max_symbols]

    def _underlying_type(self, sym: str) -> str | None:
        """Binance's exchangeInfo `underlyingType`, via the ccxt market."""
        try:
            info = (self.exchange.market(sym) or {}).get("info") or {}
        except Exception:
            return None
        v = info.get("underlyingType")
        return str(v) if v not in (None, "", []) else None

    def _log_instrument_classes(self, symbols) -> None:
        """
        One-time inventory of the instrument classes in the universe.

        Printed once per process so the operator can see EXACTLY which value
        separates TradFi perps from crypto perps, rather than anyone guessing.
        SNXX 2026-09-18 returned -4411 "Please sign TradFi-Perps agreement"
        and nothing in the scan distinguished it beforehand.
        """
        if getattr(self, "_classes_logged", False):
            return
        self._classes_logged = True
        seen: dict = {}
        for sym in symbols:
            k = self._underlying_type(sym) or "(none)"
            ex, n = seen.get(k, (sym, 0))
            seen[k] = (ex, n + 1)
        if not seen:
            return
        log.info("instrument classes in the scan universe: " + " | ".join(
            f"underlyingType={k}: {n} symbol(s), e.g. {ex}"
            for k, (ex, n) in sorted(seen.items(), key=lambda kv: -kv[1][1])))
        allowed = tuple(getattr(self.cfg, "allowed_underlying", ()) or ())
        if allowed:
            log.info(f"SCAN_ALLOWED_UNDERLYING={','.join(allowed)} — only "
                     f"those classes are tradeable; symbols with no "
                     f"underlyingType are allowed through.")

    def _wrong_class(self, sym: str) -> bool:
        """
        True when the symbol belongs to an instrument class we do not trade.

        Fails OPEN: no configured list, or no underlyingType on the market,
        and the symbol passes. Getting this wrong in the other direction would
        empty the universe silently.
        """
        allowed = tuple(getattr(self.cfg, "allowed_underlying", ()) or ())
        if not allowed:
            return False
        ut = self._underlying_type(sym)
        if ut is None:
            return False
        return ut.upper() not in tuple(a.upper() for a in allowed)

    def _excluded(self, sym: str) -> bool:
        """
        Symbols the scan must never surface.

        Binance's TradFi perps share the ticker feed and pass the volume and
        RSI filters like any mover, but entries come back -4411 "Please sign
        TradFi-Perps agreement contract fapi" — an account permission, not a
        transient error. A rejection opens no position, so it triggers no
        cooldown, so the symbol is re-attempted every scan indefinitely.
        SNXX 2026-09-18 14:14:56 and again at 14:15:36.

        Matched on the BASE so the operator writes SNXX, not SNXX/USDT:USDT.
        """
        ex = tuple(getattr(self.cfg, "exclude_symbols", ()) or ())
        if not ex:
            return False
        base = sym.split("/")[0].upper()
        return base in ex or sym.upper() in ex

    def _candles_for_24h(self) -> int:
        """
        How many candles of this timeframe span 24h.

        The 24h high/low fallback is only truthful if the fetched window
        actually covers 24 hours — 120 candles is just 6h on a 3m chart, which
        would present a 6-hour range as if it were the daily one.
        """
        tf = (self.timeframe or "5m").lower().strip()
        try:
            if tf.endswith("m"):
                mins = int(tf[:-1])
            elif tf.endswith("h"):
                mins = int(tf[:-1]) * 60
            elif tf.endswith("d"):
                mins = int(tf[:-1]) * 1440
            else:
                mins = 5
        except ValueError:
            mins = 5
        needed = int(24 * 60 / max(mins, 1)) + 2
        # Enough for the indicators regardless, and within Binance's limit.
        return max(120, min(needed, 1000))

    def _ohlcv(self, symbol: str) -> pd.DataFrame | None:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, self.timeframe,
                                            limit=self._candles_for_24h())
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
        started = time.time()
        candidates: list[Candidate] = []
        self._range_misses = 0
        try:
            movers = self._prefilter()
        except Exception as e:
            with self._lock:
                self._last_error = f"ticker fetch failed: {e}"
            log.warning(self._last_error)
            return self._last_results

        self._universe_size = len(movers)
        # Why each symbol did NOT become a candidate. Without this, peer
        # cross-evaluation can only say "not in our current scan", which
        # covers three completely different causes — not a mover, under the
        # volume floor, or screened out on RSI — and they are fixed by three
        # different settings.
        rejected: dict = {}
        mover_syms = {m[0] for m in movers}
        for sym, qv, pct, hi, lo in movers:
            df = self._ohlcv(sym)
            if df is None:
                rejected[sym] = "no candles returned"
                continue
            try:
                # Some ticker payloads omit high/low; derive them from the
                # candles instead. The fetch window is sized to cover 24h
                # (see _candles_for_24h), so this is a true daily range rather
                # than whatever happened to be in a short buffer.
                h = float(hi) if hi else None
                l = float(lo) if lo else None
                if h is None or l is None:
                    try:
                        h = h if h is not None else float(df["high"].max())
                        l = l if l is not None else float(df["low"].min())
                        self._range_misses += 1
                    except Exception as e:
                        # Was silently swallowed, leaving the range as "n/a"
                        # with no way to tell why.
                        log.warning(f"{sym}: 24h range fallback failed: "
                                    f"{type(e).__name__}: {e}")
                if h is None or l is None:
                    log.warning(f"{sym}: no 24h high/low available "
                                f"(ticker high={hi!r} low={lo!r})")
                why: list = []
                c = evaluate_symbol(sym, df, qv, pct, self.cfg,
                                    high_24h=h, low_24h=l, why=why)
            except Exception as e:
                log.debug(f"evaluate failed for {sym}: {e}")
                rejected[sym] = f"evaluate failed: {type(e).__name__}"
                continue
            if c is not None:
                candidates.append(c)
            else:
                rejected[sym] = (why[0] if why else "screened out")

        pairs = balance_directions(rank_with_deltas(self.tracker.annotate(candidates)))
        self.tracker.commit(candidates)

        with self._lock:
            self._last_results = pairs
            self._last_rejected = rejected
            self._last_movers = mover_syms
            self._last_scan_ts = time.time()
            self._last_duration_s = self._last_scan_ts - started
            self._last_error = None
        log.info(f"Scan: {len(candidates)} candidate(s) from {len(movers)} movers "
                 f"in {self._last_duration_s:.1f}s")
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
                    "strength": str(d.strength),
                    "rsi_change": None if d.is_new else round(float(d.rsi_change), 1),
                    "crossed_down": bool(d.crossed_down),
                    "crossed_up": bool(d.crossed_up),
                    "delta_note": str(d.note),
                })
                rows.append(row)
            # Regime context travels with the snapshot so every entry can be
        # stamped with the market conditions it was taken in.
        return {
            "breadth_pct": self._breadth_pct,
            "breadth_up": self._breadth_counts[0],
            "breadth_down": self._breadth_counts[1],
            "btc_change_pct": self._btc_change_pct,
                "candidates": rows,
                "last_scan_ts": self._last_scan_ts,
                "last_scan_ago_s": (time.time() - self._last_scan_ts) if self._last_scan_ts else None,
                "universe_size": self._universe_size,
                # Why each mover did NOT become a candidate, so peer
                # cross-evaluation can name the cause instead of saying
                # "not in our current scan" for three different reasons.
                "rejected": dict(getattr(self, "_last_rejected", {})),
                "movers": sorted(getattr(self, "_last_movers", set())),
                "demo": self.demo,
                "range_fallbacks": self._range_misses,
                "scan_duration_s": round(self._last_duration_s, 1),
                "interval_s": self.interval,
                "error": self._last_error,
                "config": {
                    "min_24h_vol_usdt": self.cfg.min_24h_vol_usdt,
                    "volume_mode": self.cfg.volume_mode,
                    "vol_percentile": self.cfg.vol_percentile,
                    "effective_vol_floor": round(self._effective_vol_floor, 0),
                    "min_abs_change_pct": self.cfg.min_abs_change_pct,
                    "short_rsi_min": self.cfg.short_rsi_min,
                    "long_rsi_min": self.cfg.long_rsi_min,
                    "long_rsi_max": self.cfg.long_rsi_max,
                    "timeframe": self.timeframe,
                },
            }
