"""
Live prices for the handful of symbols that are currently eligible.

WHAT THIS FIXES. The scanner runs every SCANNER_INTERVAL (120s) and the
auto-trader reads its snapshot every AUTO_TRADE_INTERVAL (30s). Every gate —
RSI band, distance to the extreme, ATR, the turn, the breakout veto — is
therefore applied to the market as it was up to two minutes ago, while the
order is SIZED at the current price. LSK/USDT on 2026-09-16 was decided on a
0.5239 market and sized at 0.4612: a 12% gap, and every threshold that
admitted it had been true of a price that no longer existed.

WHAT IT DOES NOT FIX. The entry order is a TRAILING_STOP_MARKET, which Binance
manages tick-by-tick on its own servers. The TRIGGER was never stale. This
makes the DECISION live; the exchange still owns the fire.

SCOPE. Only symbols the scanner has already marked eligible — ten to twenty,
not the 574 it screens. The scanner stays the selector; the stream only says
whether the market still looks the way the scanner found it.

DEGRADATION. Every consumer treats "no quote" and "stale quote" as "carry on
with the scan figure". A stream that dies must cost accuracy, never entries.
"""
from __future__ import annotations

import json
import logging
import threading
import time

log = logging.getLogger("candidate_stream")

WS_BASE = "wss://fstream.binance.com/stream"
WS_BASE_DEMO = "wss://fstream.binancefuture.com/stream"

# A quote older than this is not used. Two 3m candles of silence on a symbol
# that is supposed to be moving means the stream is not healthy.
DEFAULT_STALE_AFTER_S = 20.0


class Quote:
    __slots__ = ("price", "at", "volume", "prev_volume")

    def __init__(self, price: float, at: float):
        self.price = price
        self.at = at
        self.volume = 0.0
        self.prev_volume = 0.0

    def age(self, now: float | None = None) -> float:
        return (now or time.time()) - self.at


class CandidateStream:
    """
    Maintains live marks for a changing set of symbols.

    Thread-safe and non-blocking: the auto-trader asks for a price and either
    gets a fresh one or does not. It never waits on the socket.
    """

    def __init__(self, demo: bool = True,
                 stale_after_s: float = DEFAULT_STALE_AFTER_S,
                 proxy: str | None = None):
        self.demo = demo
        self.stale_after_s = float(stale_after_s)
        self.proxy = proxy
        self._quotes: dict[str, Quote] = {}
        self._want: set[str] = set()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = False
        self._connected_at = 0.0
        self._reconnects = 0
        self._messages = 0
        self._last_msg_at = 0.0
        self._last_error = ""

    # ── symbols ────────────────────────────────────────────────────────────
    @staticmethod
    def _wire(symbol: str) -> str:
        """BTC/USDT:USDT -> btcusdt, the form Binance's stream expects."""
        return symbol.split(":")[0].replace("/", "").lower()

    def track(self, symbols: list[str]):
        """
        Replace the tracked set. Called each time the scanner publishes, so
        the stream follows eligibility rather than accumulating symbols.
        """
        want = {self._wire(s) for s in symbols if s}
        with self._lock:
            if want == self._want:
                return
            added = want - self._want
            dropped = self._want - want
            self._want = want
            # Prune against the NEW SET, not the diff. Diffing only drops
            # symbols that were previously wanted, so a quote that arrived for
            # anything else — a late frame after a resubscribe, say — would sit
            # in the map forever and could be read as live for a symbol no
            # longer eligible.
            for key in [k for k in self._quotes if k not in want]:
                self._quotes.pop(key, None)
        if added or dropped:
            log.info(f"stream tracking {len(want)} symbol(s) "
                     f"(+{len(added)} -{len(dropped)})")
            # The subscription is rebuilt on reconnect; forcing one is simpler
            # and cheaper than managing SUBSCRIBE/UNSUBSCRIBE frames for a set
            # that turns over every couple of minutes anyway.
            self._reconnect()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="candidate-stream",
                                        daemon=True)
        self._thread.start()
        log.info("candidate stream started")

    def stop(self):
        self._stop.set()

    def _reconnect(self):
        self._connected = False   # the run loop notices and rebuilds

    def _run(self):
        import asyncio
        while not self._stop.is_set():
            try:
                asyncio.run(self._session())
            except Exception as e:
                self._last_error = str(e)[:200]
                log.warning(f"candidate stream session ended: {e}")
            if self._stop.is_set():
                break
            self._reconnects += 1
            # Backoff, capped: a stream that cannot connect must not become a
            # request storm against the same endpoint the guardian uses.
            delay = min(30.0, 2.0 * min(self._reconnects, 10))
            time.sleep(delay)

    async def _session(self):
        import aiohttp
        with self._lock:
            want = sorted(self._want)
        if not want:
            await self._idle()
            return
        streams = "/".join(f"{s}@markPrice@1s" for s in want)
        base = WS_BASE_DEMO if self.demo else WS_BASE
        url = f"{base}?streams={streams}"
        timeout = aiohttp.ClientTimeout(total=None, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.ws_connect(url, proxy=self.proxy,
                                       heartbeat=30) as ws:
                self._connected = True
                self._connected_at = time.time()
                log.info(f"candidate stream connected: {len(want)} symbol(s)")
                async for msg in ws:
                    if self._stop.is_set() or not self._connected:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    self._ingest(msg.data)
                self._connected = False

    async def _idle(self):
        """Nothing eligible: wait rather than hammering a connect."""
        import asyncio
        for _ in range(20):
            if self._stop.is_set():
                return
            with self._lock:
                if self._want:
                    return
            await asyncio.sleep(0.5)

    def _ingest(self, raw: str):
        try:
            msg = json.loads(raw)
            data = msg.get("data") or msg
            sym = str(data.get("s") or "").lower()
            price = data.get("p")
            if not sym or price is None:
                return
            now = time.time()
            with self._lock:
                q = self._quotes.get(sym)
                if q is None:
                    self._quotes[sym] = Quote(float(price), now)
                else:
                    q.price = float(price)
                    q.at = now
                self._messages += 1
                self._last_msg_at = now
        except Exception:
            # A malformed frame is not worth a log line per tick.
            pass

    # ── reading ────────────────────────────────────────────────────────────
    def price(self, symbol: str, now: float | None = None) -> float | None:
        """
        Live mark, or None when there is no fresh quote.

        None is the normal answer during the first seconds of tracking a new
        symbol, and the correct answer when the stream is unhealthy. Callers
        fall back to the scan figure.
        """
        key = self._wire(symbol)
        now = now or time.time()
        with self._lock:
            q = self._quotes.get(key)
            if q is None:
                return None
            if q.age(now) > self.stale_after_s:
                return None
            return q.price

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            fresh = sum(1 for q in self._quotes.values()
                        if q.age(now) <= self.stale_after_s)
            return {
                "connected": self._connected,
                "tracking": len(self._want),
                "quotes": len(self._quotes),
                "fresh": fresh,
                "messages": self._messages,
                "reconnects": self._reconnects,
                "last_message_age_s": (round(now - self._last_msg_at, 1)
                                       if self._last_msg_at else None),
                "last_error": self._last_error,
            }
