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
                 proxy: str | None = None,
                 base_url: str | None = None,
                 rest_fetcher=None,
                 rest_interval_s: float = 3.0,
                 websocket_enabled: bool = False):
        self.demo = demo
        self.stale_after_s = float(stale_after_s)
        # The live host accepted the socket and delivered nothing, on a clean
        # ASCII subscription, while demo worked through the SAME proxy — and a
        # single-symbol probe on /ws/ was silent too. That points at the exit
        # IP rather than at anything here: Binance blocks datacenter and VPN
        # ranges on the live market-data hosts more readily than on testnet,
        # while leaving REST permitted.
        #
        # Both variables are therefore settable without a rebuild, so the
        # alternatives can be tried directly: a different endpoint, or no
        # proxy at all.
        self.proxy = proxy
        self.base_url = (base_url or "").strip() or None
        # REST FALLBACK. The live websocket host does not deliver to this
        # address — direct or proxied, one well-known symbol, silent. REST on
        # the same network works, and one call to /fapi/v1/premiumIndex
        # returns the mark price for EVERY symbol, so the candidate set is a
        # local filter rather than N requests.
        #
        # At 3s that is ~200 weight/min against a 2400 limit, and it is 40x
        # fresher than the 120s scan. The drift gate is looking for
        # percent-scale movement — LSK moved 12% — so seconds are ample.
        # Websocket is preferred when available; this is what makes the live
        # container useful anyway.
        # WEBSOCKET IS OPT-IN. REST is the transport on BOTH environments so
        # findings translate: a measurement taken on demo means the same thing
        # on live only if the inputs were gathered the same way.
        #
        # It buys ~1s freshness against REST's <=3s, and the drift gate is
        # looking for percent-scale movement — LSK moved 12% between scan and
        # sizing. Two seconds does not change that verdict. Against it: a
        # permanently churning reconnect loop wherever a proxy will not tunnel
        # wss://, a probe every few minutes, and a failure mode with no REST
        # equivalent (one unrecognised symbol silences the whole subscription,
        # where one REST call returns every symbol).
        self.websocket_enabled = bool(websocket_enabled)
        self.rest_fetcher = rest_fetcher
        self.rest_interval_s = float(rest_interval_s)
        self._rest_thread: threading.Thread | None = None
        self._rest_polls = 0
        self._rest_errors = 0
        self._source = "none"
        self._skipped_last: set = set()
        # Latest BNB/USDT mark, harvested from the same poll.
        self._bnb_mark: float | None = None
        self._bnb_mark_at: float = 0.0
        self._quotes: dict[str, Quote] = {}
        self._want: set[str] = set()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = False
        self._connected_at = 0.0
        self._reconnects = 0
        # A deliberate resubscribe is not a fault. The candidate set turns
        # over every couple of minutes, so counting those as reconnects made
        # a healthy stream look like it was flapping.
        self._resubscribing = False
        self._resubscribes = 0
        # Sessions that opened and delivered nothing. A subscription that is
        # structurally wrong fails the same way every time, so reconnecting
        # at full speed is a pointless request loop.
        self._silent_sessions = 0
        self._probe_result = ""
        self._messages = 0
        self._last_msg_at = 0.0
        self._last_error = ""

    # ── symbols ────────────────────────────────────────────────────────────
    @staticmethod
    def _wire(symbol: str) -> str | None:
        """
        BTC/USDT:USDT -> btcusdt, the form Binance's stream expects.

        Returns None for anything that is not a USDT-margined futures pair.
        One unrecognised name makes Binance ACCEPT the socket and send
        NOTHING — the whole subscription goes silent, not just that symbol —
        so a name that cannot be trusted must never reach the URL. The live
        universe is 718 symbols against demo's 574 and is not all USDT-M.
        """
        try:
            base = str(symbol or "").split(":")[0].replace("/", "").upper()
            if not base.endswith("USDT"):
                return None
            # MUST BE ASCII. `.isalnum()` is True for CJK under Unicode, so a
            # symbol like 龙虾USDT passed the previous check, went into the
            # stream URL percent-encoded as %E9%BE%99%E8%99%BEusdt, and
            # silenced the ENTIRE subscription — Binance accepted the socket
            # and sent nothing for any symbol. These are real, tradeable
            # pairs: 我踏马来了/USDT traded on 2026-09-13. They are excluded
            # from the STREAM only, and still scanned, entered and guarded
            # normally on the scan snapshot.
            if not base.isascii():
                return None
            stem = base[:-4]
            if not stem or not stem.replace("_", "").isalnum():
                return None
            return base.lower()
        except Exception:
            return None

    def track(self, symbols: list[str]):
        """
        Replace the tracked set. Called each time the scanner publishes, so
        the stream follows eligibility rather than accumulating symbols.
        """
        wired = {s: self._wire(s) for s in symbols if s}
        want = {w for w in wired.values() if w}
        skipped = [s for s, w in wired.items() if not w]
        # Only when the set CHANGES. The candidate list is rebuilt every
        # cycle, so an unchanged skip list was logging the same warning every
        # 30 seconds.
        if skipped and set(skipped) != self._skipped_last:
            self._skipped_last = set(skipped)
            log.warning(
                f"stream skipping {len(skipped)} symbol(s) it cannot "
                f"subscribe to (non-ASCII or not USDT-M): "
                f"{', '.join(skipped[:6])}"
                f"{'...' if len(skipped) > 6 else ''}. One unrecognised name "
                f"silences the ENTIRE subscription, so they are left out.")
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
            # that turns over every couple of minutes anyway. No socket, no
            # resubscribe — REST reads self._want directly.
            if self.websocket_enabled:
                self._reconnect()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if self.rest_fetcher is not None:
            self._rest_thread = threading.Thread(
                target=self._rest_loop, name="candidate-rest", daemon=True)
            self._rest_thread.start()
            log.info(f"candidate REST fallback every {self.rest_interval_s:g}s "
                     f"(used whenever the websocket has no fresh quote)")
        if not self.websocket_enabled:
            log.info("candidate stream: websocket DISABLED, REST only "
                     "(CANDIDATE_WEBSOCKET_ENABLED=true to turn it on)")
            return
        self._thread = threading.Thread(target=self._run, name="candidate-stream",
                                        daemon=True)
        self._thread.start()
        log.info("candidate stream started (websocket + REST)")

    def stop(self):
        self._stop.set()

    def _reconnect(self):
        """Rebuild the subscription because the tracked set changed."""
        self._resubscribing = True
        self._resubscribes += 1
        self._connected = False   # the run loop notices and rebuilds

    def _run(self):
        import asyncio
        while not self._stop.is_set():
            try:
                asyncio.run(self._session())
            except Exception as e:
                txt = str(e)
                self._last_error = txt[:200]
                if "WRONG_VERSION_NUMBER" in txt or "record layer" in txt:
                    log.error(
                        f"candidate stream: TLS got a PLAINTEXT reply from "
                        f"{'the proxy' if self.proxy else 'the host'} "
                        f"({txt[:90]}). A wss:// connection through an HTTP "
                        f"proxy needs a CONNECT tunnel; answering in the clear "
                        f"means it was not opened. Try "
                        f"CANDIDATE_STREAM_PROXY=none.")
                else:
                    log.warning(f"candidate stream session ended: {e}")
            if self._stop.is_set():
                break
            if self._resubscribing and not self._silent_sessions:
                # Expected: reconnect at once, do not count it as a failure.
                self._resubscribing = False
                continue
            if self._silent_sessions >= 3:
                # Three silent sessions is a broken subscription, not bad
                # luck. Back off hard rather than reconnecting every few
                # seconds against an endpoint the guardian also uses.
                self._resubscribing = False
                log.error(
                    f"candidate stream silent {self._silent_sessions} "
                    f"session(s) in a row — backing off 5 minutes. Entries "
                    f"continue on the scan snapshot.")
                for _ in range(300):
                    if self._stop.is_set():
                        return
                    time.sleep(1.0)
                self._silent_sessions = 0
                continue
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
        base = self.base_url or (WS_BASE_DEMO if self.demo else WS_BASE)
        url = f"{base}?streams={streams}"
        timeout = aiohttp.ClientTimeout(total=None, sock_read=60)
        # The endpoint and the first few stream names, because "connected but
        # no messages" is otherwise undiagnosable: a single unrecognised
        # stream name makes Binance accept the socket and send nothing.
        log.info(f"candidate stream connecting to {base} "
                 f"({len(want)} stream(s): {', '.join(want[:4])}"
                 f"{'...' if len(want) > 4 else ''})"
                 + (f" via proxy {self.proxy}" if self.proxy else ""))
        before = self._messages
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.ws_connect(url, proxy=self.proxy,
                                       heartbeat=30) as ws:
                self._connected = True
                self._connected_at = time.time()
                log.info(f"candidate stream connected: {len(want)} symbol(s)")
                while not self._stop.is_set() and self._connected:
                    # receive() with a timeout rather than `async for`: a
                    # resubscribe request could not take effect while no
                    # messages were arriving, because the iterator blocks
                    # until the next frame or the 60s read timeout — exactly
                    # the case on a stream that is delivering nothing.
                    try:
                        msg = await ws.receive(timeout=5)
                    except Exception:
                        continue
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._ingest(msg.data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                      aiohttp.WSMsgType.CLOSING,
                                      aiohttp.WSMsgType.ERROR):
                        self._last_error = (
                            f"socket closed: code={ws.close_code} "
                            f"{str(msg.data)[:120]}")
                        log.warning(f"candidate stream {self._last_error}")
                        break
                self._connected = False
        got = self._messages - before
        if got == 0:
            self._silent_sessions += 1
            # PROBE. Demo works on the identical code path through the same
            # proxy; only the host differs. Guessing from here is worthless,
            # so ask the endpoint a question with a known answer: one
            # well-known symbol on the SINGLE-stream URL. The result separates
            # the three candidate causes without another deploy.
            await self._probe()
            # Distinct from a connection failure and diagnosed differently.
            self._last_error = (
                f"connected to {base} but received NO messages")
            log.error(
                f"candidate stream CONNECTED BUT SILENT: {len(want)} stream(s) "
                f"on {base}, zero messages. Usually an unrecognised symbol in "
                f"the subscription — Binance accepts the socket and sends "
                f"nothing. Streams: {', '.join(want[:8])}"
                f"{'...' if len(want) > 8 else ''}")
        else:
            self._silent_sessions = 0
            log.info(f"candidate stream session ended after {got} message(s)")

    def _rest_loop(self):
        """
        Poll mark prices for the tracked set. Runs alongside the websocket and
        only fills gaps: _ingest keeps the newest timestamp either way, so a
        healthy socket simply wins on recency.
        """
        while not self._stop.is_set():
            try:
                with self._lock:
                    want = set(self._want)
                if want:
                    marks = self.rest_fetcher() or {}
                    now = time.time()
                    # premiumIndex returns EVERY symbol, so BNBUSDT is already
                    # in hand. Keeping it costs no call and no weight, and it
                    # is what converts a BNB-denominated commission into USDT.
                    bnb = marks.get("bnbusdt")
                    if bnb:
                        self._bnb_mark, self._bnb_mark_at = float(bnb), now
                    hit = 0
                    with self._lock:
                        for wire, px in marks.items():
                            if wire not in want:
                                continue
                            q = self._quotes.get(wire)
                            if q is None:
                                self._quotes[wire] = Quote(float(px), now)
                            else:
                                q.price = float(px)
                                q.at = now
                            hit += 1
                        if hit:
                            self._rest_polls += 1
                            self._last_msg_at = now
                            if self._source != "websocket":
                                self._source = "rest"
            except Exception as e:
                self._rest_errors += 1
                if self._rest_errors in (1, 10) or self._rest_errors % 100 == 0:
                    log.warning(f"candidate REST poll failed "
                                f"({self._rest_errors}): {str(e)[:140]}")
            self._stop.wait(self.rest_interval_s)

    async def _probe(self):
        """
        Subscribe to btcusdt alone on the /ws/ endpoint for a few seconds.

            messages arrive  -> the host and proxy are fine; the COMBINED
                                /stream?streams= form or one of the symbols
                                in it is the problem
            silent           -> the host itself is not delivering to us:
                                geo-restriction on the live endpoint, or a
                                connection-rate block. Demo would still work,
                                which is exactly what is observed.
            handshake fails  -> the proxy cannot reach this host at all
        """
        import aiohttp
        base = (self.base_url or (WS_BASE_DEMO if self.demo else WS_BASE)).replace("/stream", "/ws")
        url = f"{base}/btcusdt@markPrice@1s"
        log.warning(f"candidate stream PROBE: {url}")
        try:
            timeout = aiohttp.ClientTimeout(total=20, sock_read=10)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.ws_connect(url, proxy=self.proxy,
                                           heartbeat=15) as ws:
                    for _ in range(6):
                        try:
                            msg = await ws.receive(timeout=3)
                        except Exception:
                            continue
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            log.warning(
                                "candidate stream PROBE OK: the host and proxy "
                                "are fine. The combined /stream?streams= URL "
                                "or a symbol inside it is what is silent.")
                            self._probe_result = "single-stream works"
                            return
                    log.error(
                        "candidate stream PROBE SILENT: btcusdt alone on /ws/ "
                        "also returned nothing. The endpoint is not delivering "
                        "to this IP — geo-restriction on the LIVE host or a "
                        "connection-rate block. Demo uses a different host, "
                        "which is why it works.")
                    self._probe_result = "host silent for btcusdt too"
        except Exception as e:
            txt = str(e)
            if "WRONG_VERSION_NUMBER" in txt or "record layer" in txt:
                # The client began TLS and got back plaintext. Through an HTTP
                # proxy that means the CONNECT tunnel was never opened and the
                # proxy answered in the clear — an error page, a block notice.
                # It is the PROXY, not the host.
                log.error(
                    "candidate stream PROBE: TLS got a plaintext reply "
                    "(WRONG_VERSION_NUMBER). The proxy is answering instead of "
                    "opening a CONNECT tunnel to this host — set "
                    "CANDIDATE_STREAM_PROXY=none to bypass it for the stream.")
                self._probe_result = "proxy did not tunnel (plaintext reply)"
            else:
                log.error(f"candidate stream PROBE FAILED to connect: "
                          f"{txt[:160]} — the proxy cannot reach this host.")
                self._probe_result = f"probe connect failed: {txt[:80]}"

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
                self._source = "websocket"
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
        if not key:
            return None
        now = now or time.time()
        with self._lock:
            q = self._quotes.get(key)
            if q is None:
                return None
            if q.age(now) > self.stale_after_s:
                return None
            return q.price

    def bnb_mark(self, max_age_s: float = 120.0) -> float | None:
        """
        BNB/USDT, for converting commissions paid in BNB.

        Stale is worse than absent: a price two minutes old is fine for a
        fee that is fractions of a cent, but an hour-old one silently
        misstates every fee in the tables.
        """
        with self._lock:
            if not self._bnb_mark:
                return None
            if time.time() - self._bnb_mark_at > max_age_s:
                return None
            return self._bnb_mark

    def healthy(self, now: float | None = None) -> bool:
        """
        Is the stream DELIVERING? Not "is the socket flag set".

        `connected` flickers False on every resubscribe, and the tracked set
        changes every couple of minutes — so a stream sending 1,016 messages a
        minute with seven fresh quotes was reporting DEGRADED. What matters to
        a caller is whether a usable quote exists, which is what this asks.
        """
        now = now or time.time()
        with self._lock:
            if not self._last_msg_at:
                return False
            if now - self._last_msg_at > self.stale_after_s:
                return False
            return any(q.age(now) <= self.stale_after_s
                       for q in self._quotes.values())

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            fresh = sum(1 for q in self._quotes.values()
                        if q.age(now) <= self.stale_after_s)
            return {
                "healthy": self.healthy(now),
                "websocket_enabled": self.websocket_enabled,
                "connected": self._connected,
                "resubscribes": self._resubscribes,
                "silent_sessions": self._silent_sessions,
                "probe": self._probe_result,
                "source": self._source,
                "rest_polls": self._rest_polls,
                "rest_errors": self._rest_errors,
                "tracking": len(self._want),
                "quotes": len(self._quotes),
                "fresh": fresh,
                "messages": self._messages,
                "reconnects": self._reconnects,
                "last_message_age_s": (round(now - self._last_msg_at, 1)
                                       if self._last_msg_at else None),
                "last_error": self._last_error,
            }


def make_rest_fetcher(exchange):
    """
    A fetcher over ccxt that returns {wire_symbol: mark_price}.

    ONE call for every symbol — /fapi/v1/premiumIndex with no argument —
    rather than one per candidate. At a 3s interval that is roughly 200 weight
    a minute against a 2400 limit.

    Returns {} on any failure. The caller treats an empty result as "no quote",
    which falls back to the scan figure, so a REST outage costs accuracy and
    never entries.
    """
    def _fetch() -> dict:
        try:
            fn = getattr(exchange, "fapiPublicGetPremiumIndex", None)
            if fn is None:
                return {}
            rows = fn()
            if isinstance(rows, dict):
                rows = [rows]
            out = {}
            for r in rows or []:
                sym = str(r.get("symbol") or "").lower()
                px = r.get("markPrice")
                if not sym or px is None:
                    continue
                try:
                    val = float(px)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    out[sym] = val
            return out
        except Exception:
            return {}
    return _fetch
