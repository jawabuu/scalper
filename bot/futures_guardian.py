"""
Futures guardian — exchange layer.

Connects the pure decision logic in futures_guard.py to a live Binance USD-M
futures account. It protects positions the OPERATOR opened; it never opens one.

SAFETY PROPERTIES (each enforced in code, not convention):

  1. Every order it places is reduceOnly=True. A reduce-only order can only
     shrink or close an existing position — it can never open one, and never
     increase exposure.
  2. It only ever cancels orders that pass is_protective_stop(): reduce-only,
     closing side, stop-typed. A trailing-stop ENTRY order (not reduce-only) is
     invisible to it and can never be cancelled.
  3. Stop replacement is PLACE-THEN-CANCEL: the new stop is resting before the
     old one is removed. A brief moment with two stops is harmless (both are
     reduce-only; the first to trigger closes the position and the other is
     cleaned up). A moment with ZERO stops on a leveraged position is not.
  4. DRY_RUN mode logs every intended action without sending it.
  5. Credential cross-wire check: refuses to start if testnet and live keys are
     identical, or if the resolved key does not match the declared environment.

It sets no leverage and no margin mode — the operator configures those on
Binance. The guardian only reads what is there.
"""
from __future__ import annotations

import logging
import threading
import time

import ccxt

from .futures_state import identity as futures_state_identity
from .futures_guard import (
    FuturesPosition, GuardState, GuardConfig,
    roi_pct, price_for_roi, stop_side,
    adopt_state, is_protective_stop, evaluate,
    callback_roi_at, trail_locks_in, is_armed, atr_stop_roi,
    trail_callback_price_pct,
)

log = logging.getLogger("futures_guardian")


class RateLimitMonitor:
    """
    Notices when Binance is rate-limiting us, and says so loudly.

    `enableRateLimit: True` is client-side PACING — ccxt spaces requests to
    stay under the published limit. It does nothing when a limit is actually
    hit: there was no retry, no backoff, and no signal anywhere that it had
    happened. Binance escalates repeated 429s to a temporary IP ban (418), so
    silently continuing at the same rate is the worst response available.

    Detection is by error CODE and message rather than exception class, because
    the same condition arrives as ccxt.RateLimitExceeded, ccxt.DDoSProtection
    or a bare HTTP error depending on the call path.

        429    too many requests
        418    IP banned (an escalated 429)
        -1003  TOO_MANY_REQUESTS, Binance's own weight error
    """

    MARKERS = ("429", "418", "-1003", "too many requests",
               "rate limit", "ratelimit", "way too many requests",
               "ip banned", "ddosprotection")
    COOLDOWN_S = 60.0

    def __init__(self):
        self.hits: list[float] = []
        self.last_at: float | None = None
        self.last_msg: str = ""
        self.cooldown_until: float = 0.0

    @classmethod
    def looks_rate_limited(cls, text: str) -> bool:
        t = (text or "").lower()
        return any(m in t for m in cls.MARKERS)

    def note(self, err, text: str = "") -> bool:
        """Record a rate-limit error. Returns True if it was one."""
        blob = f"{type(err).__name__} {text}".lower()
        if not self.looks_rate_limited(blob):
            return False
        now = time.time()
        self.hits.append(now)
        self.hits = [h for h in self.hits if now - h < 3600]
        self.last_at, self.last_msg = now, text[:200]
        self.cooldown_until = now + self.COOLDOWN_S
        log.error(
            f"RATE-LIMITED by the exchange ({len(self.hits)} time(s) in the "
            f"last hour): {text[:160]} — backing off for "
            f"{self.COOLDOWN_S:.0f}s. Repeated 429s escalate to an IP ban.")
        return True

    def status(self) -> dict:
        now = time.time()
        self.hits = [h for h in self.hits if now - h < 3600]
        return {
            "limited_now": now < self.cooldown_until,
            "seconds_remaining": max(0, round(self.cooldown_until - now)),
            "hits_last_hour": len(self.hits),
            "last_at": self.last_at,
            "last_message": self.last_msg,
        }


RATE_LIMIT = RateLimitMonitor()


def _peak_ceiling(cfg) -> float:
    """
    The fail-fast peak ceiling actually in force.

    Defaults to breakeven_at_roi so the fail-fast band and the profit-floor
    band MEET. Every version where these were independent left a hole, and a
    position landing in it had no protection of any kind.
    """
    v = getattr(cfg, "fail_fast_max_peak_roi", None)
    if v is not None:
        return float(v)
    return float(getattr(cfg, "breakeven_at_roi", 0.0) or 0.0)


def _safe_err(err) -> str:
    """Exchange error as code + message, never the signed request URL."""
    from bot.futures_entry import _safe_err as _f
    try:
        text = _f(err)
    except Exception:
        text = f"{type(err).__name__}"
    # Every except block in the guardian formats through here, so detection
    # rides along automatically rather than needing a hook per call site.
    try:
        RATE_LIMIT.note(err, text)
    except Exception:
        pass
    return text


def resolve_usdt_balance(bal: dict) -> tuple[float, str]:
    """
    Total USDT wallet balance from a ccxt futures balance payload.

    Single source of truth: the guardian displays this figure and the entry
    service sizes positions from it, so they must never resolve it differently.
    Several shapes are tried because the payload differs between live and demo.
    Returns (value, source) so the origin can be logged when a figure surprises.
    """
    candidates: list[tuple[str, object]] = []

    # ORDER MATTERS. Sizing must use REALISED equity.
    #
    # ccxt maps USDT.total to marginBalance = walletBalance + unrealised PnL,
    # so preferring it made the risk budget swing with the mark price of every
    # open position: winning trades inflated the budget and sized the NEXT
    # position larger, precisely when exposure was already highest. One
    # observed jump of +74% in 34 seconds scaled every subsequent entry 1.74x.
    #
    # totalWalletBalance excludes unrealised PnL and is the figure "0.5% of
    # the wallet" is supposed to mean.
    info = (bal or {}).get("info") or {}
    if isinstance(info, dict):
        candidates.append(("info.totalWalletBalance", info.get("totalWalletBalance")))
        assets = info.get("assets")
        if isinstance(assets, list):
            for a in assets:
                if (a or {}).get("asset") == "USDT":
                    candidates.append(("assets[USDT].walletBalance", a.get("walletBalance")))

    usdt = (bal or {}).get("USDT") or {}
    if isinstance(usdt, dict):
        # Fallbacks only: these include unrealised PnL.
        candidates.append(("USDT.total (margin balance)", usdt.get("total")))
        candidates.append(("USDT.free", usdt.get("free")))

    total_map = (bal or {}).get("total") or {}
    if isinstance(total_map, dict):
        candidates.append(("total.USDT", total_map.get("USDT")))

    if isinstance(info, dict):
        candidates.append(("info.availableBalance", info.get("availableBalance")))

    for source, raw in candidates:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val, source
    return 0.0, "unresolved"


def fee_reserve_basis(assets_seen: dict, basis: dict,
                      live_prices: dict | None = None) -> dict:
    """
    Value a non-USDT balance at what it COST, not at today's mark.

    WHY NOT THE LIVE PRICE. BNB is held to pay fees, not as a position. Marking
    it to market turns BNB's price into account performance: on a $500 reserve
    a 10% BNB move is +/-$50, which is larger than a day of trading and would
    appear in the return card as if the bot had earned it. At a fixed basis the
    reserve falls ONLY as it is spent, which is the thing actually being
    measured.

    It also removes the failure mode entirely. Valuing at the mark means a
    momentary price-lookup failure drops the asset from the total — a $500
    reserve reading as a $500 loss from a network blip. A basis is recorded
    once and then never needs a lookup again.

    `basis` is the stored asset -> price map and is UPDATED IN PLACE the first
    time an asset is seen with a usable live price. An asset already carrying a
    basis keeps it: re-basing on a later price would reintroduce exactly the
    drift this exists to prevent.

    Returns the price map to value with. Stablecoins are 1.0 and never stored.
    """
    out = {"USDT": 1.0, "BUSD": 1.0, "USDC": 1.0, "FDUSD": 1.0}
    live = dict(live_prices or {})
    for name in (assets_seen or {}):
        up = str(name).upper()
        if up in out:
            continue
        if up in basis:
            out[up] = float(basis[up])          # cost basis wins, always
            continue
        px = live.get(up)
        try:
            px = float(px) if px is not None else None
        except (TypeError, ValueError):
            px = None
        if px and px > 0:
            basis[up] = px                      # recorded once, then fixed
            out[up] = px
    return out


def resolve_account_value(bal: dict, prices: dict | None = None) -> tuple:
    """
    The whole account in USDT: every asset's walletBalance at its own price.

    DISTINCT FROM THE SIZING BALANCE, on purpose.

    Sizing must use USDT ALONE. ENTRY_RISK_PCT means a fraction of the capital
    that can absorb a loss, and BNB held to pay fees cannot take a trading
    loss — counting it would size every position against money that is not at
    risk.

    Account VALUE is the other question: "am I up or down". Fees paid in BNB
    leave the BNB balance and never touch USDT, so a USDT-only figure cannot
    see them at all. On an account funded 4500 USDT + 500 BNB, every fee is
    invisible to the USDT wallet and the trade record drifts from it forever —
    part of what the header reports as wallet_gap.

    `prices` maps asset -> USDT price; USDT and the stablecoins are 1.0.
    An asset with no price is REPORTED, not silently dropped, because a
    missing price understates the account rather than failing loudly.

    Returns (value, per_asset, unpriced).
    """
    info = (bal or {}).get("info") or {}
    assets = info.get("assets") if isinstance(info, dict) else None
    if not isinstance(assets, list):
        return 0.0, {}, []
    prices = dict(prices or {})
    for stable in ("USDT", "BUSD", "USDC", "FDUSD"):
        prices.setdefault(stable, 1.0)
    total = 0.0
    per: dict = {}
    unpriced: list = []
    for a in assets:
        if not isinstance(a, dict):
            continue
        name = str(a.get("asset") or "").upper()
        try:
            amt = float(a.get("walletBalance") or 0)
        except (TypeError, ValueError):
            continue
        if not name or amt == 0:
            continue
        rate = prices.get(name)
        if rate is None:
            unpriced.append(name)
            per[name] = {"amount": amt, "usdt": None}
            continue
        usd = amt * float(rate)
        per[name] = {"amount": amt, "usdt": round(usd, 8)}
        total += usd
    return round(total, 8), per, unpriced


def resolve_price(exchange, symbol: str) -> tuple[float, str]:
    """
    Current price for a symbol, from whichever field the endpoint provides.

    Single source of truth for pricing, shared by the guardian and the entry
    service. Payload shape varies between live and demo — some responses carry
    no `last` at all — so several fields are tried before falling back to the
    most recent candle close. Returns (price, source); price is 0.0 if nothing
    could be read, with the reason in the source string.
    """
    try:
        t = exchange.fetch_ticker(symbol) or {}
    except Exception as e:
        return 0.0, f"ticker error: {type(e).__name__}: {e}"

    for field in ("last", "close", "markPrice", "mark", "previousClose"):
        try:
            v = float(t.get(field))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v, f"ticker.{field}"

    # Some payloads only carry the book.
    try:
        bid, ask = float(t.get("bid")), float(t.get("ask"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2, "ticker.bid/ask mid"
    except (TypeError, ValueError):
        pass

    info = t.get("info") or {}
    for field in ("markPrice", "lastPrice", "indexPrice"):
        try:
            v = float(info.get(field))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v, f"ticker.info.{field}"

    # Last resort: the most recent candle close.
    try:
        raw = exchange.fetch_ohlcv(symbol, "1m", limit=2)
        if raw:
            v = float(raw[-1][4])
            if v > 0:
                return v, "ohlcv close"
    except Exception as e:
        return 0.0, f"no price field; ohlcv fallback failed: {e}"

    return 0.0, "no usable price field in ticker"


def _r2(v):
    return None if v is None else round(float(v), 2)


def _r3(v):
    return None if v is None else round(float(v), 3)


class FuturesGuardian:
    def __init__(self, guard_cfg: GuardConfig, *, api_key: str, api_secret: str,
                 demo: bool = True, dry_run: bool = True, atr_timeframe: str = "3m",
                 poll_interval: float = 5.0, socks_proxy: str | None = None):
        self.cfg = guard_cfg.validate()
        if self.cfg.fail_fast_s:
            log.info(
                f"Fail-fast ARMED: cut after {self.cfg.fail_fast_s:.0f}s when "
                f"peak <= {_peak_ceiling(self.cfg):+.1f}% ROI and "
                f"current <= {-abs(self.cfg.fail_fast_loss_roi):.1f}% ROI")
            # The two bands must MEET. A position peaking above the fail-fast
            # ceiling but below the floor threshold gets NEITHER, and nothing
            # in the code prevents an operator opening that gap by hand.
            ceiling = _peak_ceiling(self.cfg)
            at = float(self.cfg.breakeven_at_roi or 0.0)
            if at and ceiling < at:
                log.error(
                    f"PROTECTION GAP: a position peaking between "
                    f"+{ceiling:.1f}% and +{at:.1f}% ROI gets NEITHER fail-fast "
                    f"(peak too high) NOR a profit floor (peak too low). REZ "
                    f"peaked +2.08% in exactly this band and ran to -16.74%. "
                    f"Unset GUARD_FAIL_FAST_MAX_PEAK_ROI to tie the two "
                    f"together, or raise it to {at:.1f}.")
            elif at:
                log.info(
                    f"Protection is continuous: fail-fast covers peaks up to "
                    f"+{ceiling:.1f}% ROI, the profit floor takes over at "
                    f"+{at:.1f}%.")
        else:
            log.warning(
                "Fail-fast DISABLED (GUARD_FAIL_FAST_S is 0 or unset) — losing "
                "trades will run to their stop. Set it to enable early cuts.")
        self.demo = demo
        # ATR must be measured on the timeframe actually traded.
        self.atr_timeframe = atr_timeframe or "3m"
        # Identifies which environment owns the state file, so a demo and a
        # live container sharing a volume cannot load each other's state.
        self.state_owner = futures_state_identity(demo, api_key)
        self.dry_run = dry_run
        self.poll_interval = poll_interval

        params: dict = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                # ccxt REFUSES fetchOpenOrders() without a symbol until this is
                # acknowledged, raising what looks like an ExchangeError but is
                # only a rate-limit warning. Without it the account-wide order
                # listing never runs, so the orphan sweep had no input at all.
                # The higher weight is acceptable: it runs once every 120s.
                "fetchOpenOrders": {"warnWithoutSymbol": False},
            },
        }
        if socks_proxy:
            params["proxies"] = {"http": socks_proxy, "https": socks_proxy}
        self.exchange = ccxt.binanceusdm(params)
        if demo:
            # ccxt removed sandbox/testnet support for binanceusdm; Binance now
            # offers "demo trading", which routes to demo-fapi.binance.com with
            # separate demo credentials. enable_demo_trading() swaps the API
            # URLs. Note ccxt refuses demo mode if sandbox mode is also on, so
            # the two must never both be set.
            try:
                self.exchange.enable_demo_trading(True)
                log.info("Futures DEMO trading enabled (demo-fapi.binance.com)")
            except AttributeError:
                # Older ccxt without demo support — fall back to the legacy call.
                log.warning("ccxt has no enable_demo_trading(); falling back to "
                            "set_sandbox_mode(). Upgrade ccxt if this fails.")
                self.exchange.set_sandbox_mode(True)

        self._init_runtime_state()

        mode = "DRY RUN (no orders sent)" if dry_run else "LIVE (places real orders)"
        log.warning(
            f"Futures guardian starting — {'DEMO' if demo else 'LIVE ACCOUNT'} — {mode} | "
            f"stop {-self.cfg.initial_stop_roi:+.0f}% ROI, arm +{self.cfg.arm_roi:.0f}% ROI, "
            f"native trail {self.cfg.trail_callback_pct:.2f}% price"
        )

    def _init_runtime_state(self):
        """
        All mutable runtime state in one place.

        __init__ calls this, and so can any construction path that bypasses it
        (test doubles). Keeping the attribute list in a single method stops the
        two from drifting apart and producing AttributeErrors that get silently
        swallowed by the per-position exception handler.
        """
        # symbol -> GuardState
        self._states: dict[str, GuardState] = {}
        self._lock = threading.RLock()
        self._last_cycle_ts: float = 0.0
        self._last_error: str | None = None
        self._wallet_balance_cached: float = 0.0
        # Raw fetch_balance payload, for account_value(). Empty until the
        # first poll, which is why account_value degrades to USDT-only.
        self._last_balance_payload: dict = {}
        # Cost basis per non-USDT asset, restored from state on load.
        self._asset_basis: dict = {}
        self._actions: list[dict] = []   # recent actions, for the dashboard
        self._pos_meta: dict[str, dict] = {}   # symbol -> sizing snapshot
        self._closed_trades: list[dict] = []   # futures trade history
        self._journal = None                  # set by attach_journal()
        # symbol -> (high, low, fetched_at). Some ticker payloads omit high/low,
        # so the 24h range is derived from candles as a fallback and cached —
        # the guardian polls every few seconds and must not refetch 24h of
        # klines that often.
        self._range_cache: dict[str, tuple[float, float, float]] = {}
        # symbol -> where the 24h range came from, or why it is missing. Shown
        # in the snapshot so an "n/a" is diagnosable without server logs.
        self._range_source: dict[str, str] = {}
        self._risk_overshoots: dict[str, dict] = {}
        self._capped_stop_reported: dict[str, float] = {}
        self._listing_warned: bool = False
        self.max_closed_trades: int = getattr(
            self, "max_closed_trades", self.DEFAULT_MAX_CLOSED_TRADES)
        self._empty_listings: int = 0
        self._balance_outliers: int = 0
        self._untracked_reported: dict = {}
        self._dropped_reported: dict = {}
        # Consecutive API failures across ALL endpoints. Each call used to
        # handle its own failure locally, so the bot had no concept of "I
        # cannot currently see the exchange" — and cancelled a filled order
        # two minutes after failing to read the order book at all.
        self._api_failures: int = 0
        self._blind_since: float | None = None
        self._pending_cancels: dict[str, list] = {}
        # Last protection audit per symbol, so a healthy position
        # logs at most every PROTECTION_AUDIT_INTERVAL_S.
        self._audit_last: dict[str, float] = {}
        self._cancel_attempts: dict[str, int] = {}
        self._last_trade_fees: float | None = None
        # Ids confirmed cancelled or confirmed absent from both books. Nothing
        # here is ever retried — a successful cancel followed by a -2011 retry
        # was re-queueing orders that no longer existed.
        self._cancelled_ids: set = set()
        # Wallet at the first observation, so the reconciliation has a baseline
        # to measure against. Persisted with the rest of the state.
        self.wallet_start: float | None = getattr(self, "wallet_start", None)
        self._last_wide_probe: float = 0.0
        self._missing_counts: dict[str, int] = {}
        self._stop_source_reported: set = set()
        self._atr_cache: dict[str, tuple[float, float]] = {}   # sym -> (atr%, ts)
        # Extra order ids when a stop had to be split across the per-order cap.
        self._split_stop_ids: dict[str, list[str]] = {}
        # EVERY protective stop id placed per symbol, not just the current one.
        # Ratcheting replaces a stop each time it moves; a failed cancel used to
        # orphan that stop permanently, because only the latest id was tracked.
        # Three stops were left resting on one position for exactly that reason.
        self._all_stop_ids: dict[str, list[str]] = {}
        # Where restart-critical state is persisted. Empty disables it.
        self.state_path: str = getattr(self, "state_path", "")
        self.state_owner: str = getattr(self, "state_owner", "")
        self._entry_service = getattr(self, "_entry_service", None)
        # How long an unfilled entry order may rest before it is cancelled.
        self.entry_order_ttl_s: float = getattr(self, "entry_order_ttl_s", 900.0)
        # Poll interval used while an entry order is resting and could fill at
        # any moment. Shrinks the unprotected window after a fill.
        self.pending_poll_interval: float = getattr(
            self, "pending_poll_interval", 1.0)
        self.reap_untracked: bool = getattr(self, "reap_untracked", False)
        self.sweep_orphan_stops: bool = getattr(self, "sweep_orphan_stops", True)
        self._last_untracked_sweep: float = 0.0
        self.untracked_sweep_interval_s: float = 120.0

    # ── reading ─────────────────────────────────────────────────────────────

    def note_api_failure(self, where: str, err=None):
        """Record a failed exchange read and enter read-only if they pile up."""
        self._api_failures = getattr(self, "_api_failures", 0) + 1
        if self._api_failures == self.BLIND_AFTER_FAILURES:
            self._blind_since = time.time()
            log.error(
                f"BLIND: {self._api_failures} consecutive exchange failures "
                f"(latest at {where}). Suspending entries, cancellations and "
                f"reaping until reads succeed. Positions are still monitored.")

    def note_api_success(self):
        if getattr(self, "_api_failures", 0):
            if getattr(self, "_blind_since", None):
                log.warning(
                    f"Exchange reads recovered after "
                    f"{time.time() - self._blind_since:.0f}s blind — resuming.")
            self._api_failures = 0
            self._blind_since = None

    @property
    def is_blind(self) -> bool:
        """True when the bot cannot currently see the exchange."""
        return getattr(self, "_api_failures", 0) >= self.BLIND_AFTER_FAILURES

    def _log_dropped_position(self, raw, why: str):
        """Report a position row discarded during normalisation."""
        try:
            info = (raw or {}).get("info") or {}
            sym = (raw or {}).get("symbol") or info.get("symbol") or "?"
            key = f"{sym}:{why}"
            if self._dropped_reported.get(key):
                return
            self._dropped_reported[key] = True
            recovered = "RECOVERED" in why
            log.warning(
                f"POSITION ROW {'REPAIRED' if recovered else 'DISCARDED'} "
                f"{sym}: {why}. "
                f"positionAmt={info.get('positionAmt')!r} "
                f"entryPrice={info.get('entryPrice')!r} "
                f"positionSide={info.get('positionSide')!r} "
                f"ccxt side={(raw or {}).get('side')!r} "
                f"contracts={(raw or {}).get('contracts')!r}. "
                + ("" if recovered
                   else "If a position IS open on this symbol it is UNGUARDED."))
        except Exception as e:
            # Do NOT swallow: a silent reporter is how the original defect
            # stayed invisible.
            log.warning(f"could not report a discarded position row: {e}")

    def fetch_positions(self) -> list[FuturesPosition]:
        """Open positions with non-zero size, normalised to FuturesPosition."""
        try:
            raw = self.exchange.fetch_positions()
            self.note_api_success()
        except Exception as e:
            self.note_api_failure("fetch_positions", e)
            raise
        out = []
        for p in raw:
            try:
                contracts = float(p.get("contracts") or 0)
                if contracts == 0:
                    # Never drop a position silently: an unguarded position is
                    # the worst failure the system has, and this path used to
                    # produce no output at all.
                    self._log_dropped_position(p, "zero contracts")
                    continue
                side = (p.get("side") or "").lower()
                if side not in ("long", "short"):
                    # RECOVER rather than discard. On one-way mode Binance
                    # reports positionSide BOTH and the direction lives in the
                    # SIGN of positionAmt; if ccxt fails to derive it, throwing
                    # the row away leaves a real position unguarded.
                    amt = ((p.get("info") or {}).get("positionAmt")
                           or p.get("contracts"))
                    try:
                        amt = float(amt)
                    except (TypeError, ValueError):
                        amt = 0.0
                    if amt > 0:
                        side = "long"
                    elif amt < 0:
                        side = "short"
                    if side in ("long", "short"):
                        self._log_dropped_position(
                            p, f"side missing from ccxt — RECOVERED as {side}")
                    else:
                        self._log_dropped_position(p, "unrecognised side")
                        continue
                entry = float(p.get("entryPrice") or 0)
                if entry <= 0:
                    # Never drop a position silently: an unguarded position is
                    # the worst failure the system has, and this path used to
                    # produce no output at all.
                    self._log_dropped_position(p, "no entry price")
                    continue
                info = p.get("info") or {}
                # The `leverage` field has been observed to come back as 1 on
                # isolated positions, which silently scales every ROI and stop
                # distance by the leverage factor. Try several sources, but the
                # authoritative value is derived from notional/margin below.
                lev_raw = (p.get("leverage") or info.get("leverage")
                           or info.get("marginRatio") and None)
                try:
                    lev = int(float(lev_raw)) if lev_raw else 1
                except (TypeError, ValueError):
                    lev = 1

                notional = entry * abs(contracts)
                margin_raw = (p.get("initialMargin") or info.get("initialMargin")
                              or p.get("collateral") or info.get("isolatedMargin"))
                try:
                    margin = float(margin_raw) if margin_raw else 0.0
                except (TypeError, ValueError):
                    margin = 0.0
                # A CROSS-margin position reports isolatedMargin as 0, and the
                # unified `collateral` field has been observed carrying the
                # NOTIONAL instead of the margin. That gives notional/margin =
                # 1.0, so every ROI reads 20x too small and every threshold
                # sits 20x too wide — PUFFER showed +4.1% where Binance showed
                # +86.37%, and nothing warned because margin was not <= 0.
                #
                # So the check is not "is margin missing" but "is the leverage
                # it implies believable".
                implied = (notional / margin) if margin > 0 else 0.0
                if margin <= 0 or implied < 1.5:
                    reconstructed = notional / max(lev, 1)
                    log.warning(
                        f"{p.get('symbol')}: margin field unusable "
                        f"({margin_raw!r} -> {margin:.4f}, implying {implied:.2f}x "
                        f"leverage on a {notional:.2f} notional). Reconstructing "
                        f"{reconstructed:.4f} from leverage={lev}. Cross-margin "
                        f"positions report this differently — verify ROI against "
                        f"the Binance UI."
                    )
                    margin = reconstructed

                # Binance reports updateTime in ms on positionRisk. On a
                # freshly opened position that is the fill time.
                upd = None
                try:
                    raw = (p.get("info") or {}).get("updateTime") or p.get("timestamp")
                    if raw:
                        upd = float(raw) / 1000.0
                except (TypeError, ValueError):
                    upd = None

                pos = FuturesPosition(
                    symbol=p["symbol"], side=side, entry_price=entry,
                    qty=abs(contracts), leverage=max(lev, 1), margin=margin,
                    updated_at=upd,
                )
                # Sanity-check the derived leverage against the reported one.
                eff = pos.effective_leverage
                if lev > 1 and abs(eff - lev) / lev > 0.2:
                    log.warning(
                        f"{pos.symbol}: derived leverage {eff:.1f}x differs from "
                        f"reported {lev}x — using derived (notional/margin)."
                    )
                out.append(pos)
            except Exception as e:
                log.debug(f"skipping unparseable position {p.get('symbol')}: {e}")
        return out

    def fetch_open_orders(self, symbol: str) -> list[dict]:
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            log.warning(f"fetch_open_orders failed for {symbol}: {e}")
            return []

    def mark_price(self, pos: FuturesPosition) -> float | None:
        return self._price_from_ticker(pos.symbol, self._ticker(pos.symbol))

    # Divergence worth a line, and how often to say it. The profit floor sits
    # 0.1% from entry at 20x, so a gap of that order makes it unplaceable.
    PRICE_DIVERGENCE_PCT = 0.05
    PRICE_LOG_INTERVAL_S = 30.0
    MARK_CACHE_TTL_S = 1.0

    def _price_from_ticker(self, symbol: str, t: dict | None) -> float | None:
        """
        The price every ROI, peak, floor and trail level is computed from.

        This MUST be the reference the exchange triggers on. Stops go on with
        workingType=MARK_PRICE, so this returns MARK.

        It used to return ticker `last`, and on 2026-09-17/18 the guardian's
        price came in BELOW the real mark on five shorts out of five — by
        0.11%, 0.21%, 0.99% and 9.78%. For a short that inflates ROI, and the
        damage scaled with the gap: a small one had the floor refused with
        -2021, a larger one left the trail resting with an activation price it
        never reached, and the largest had the trail rejected outright while
        the guardian reported +96.5% ROI on a position that was 1.4% DOWN.

        Returns the price; the source is recorded on self._last_price_source
        so callers can log where a number actually came from.
        """
        mark, last, src = self._mark_and_last(symbol, t)

        if mark is not None and last is not None and mark > 0 and last > 0:
            div = abs(mark - last) / mark * 100
            if div >= self.PRICE_DIVERGENCE_PCT:
                self._throttled(
                    f"div:{symbol}",
                    lambda: log.info(
                        f"PRICE-DIVERGENCE {symbol}: mark={mark} last={last} "
                        f"({(mark - last) / mark * 100:+.3f}%). Levels are "
                        f"computed from mark, which is what stops trigger on."))

        if mark is not None and mark > 0:
            self._last_price_source = src
            return mark

        if last is not None and last > 0:
            # Acting on last is better than going blind, but every level this
            # cycle is then computed against a reference the exchange does NOT
            # trigger on, which is how the -2021 refusals happened.
            self._last_price_source = "ticker.last (NO MARK)"
            self._throttled(
                f"nomark:{symbol}",
                lambda: log.warning(
                    f"{symbol}: no mark price available — using last={last}. "
                    f"Stop levels this cycle may be refused."))
            return last

        # Last resort. resolve_price() reaches for previousClose (24h old) and
        # a completed candle close before giving up, so whatever it returns is
        # named out loud rather than silently treated as the current price.
        price, source = resolve_price(self.exchange, symbol)
        self._last_price_source = source
        if price <= 0:
            log.warning(f"{symbol}: could not read a price ({source})")
            return None
        if source not in ("ticker.last", "ticker.close", "ticker.markPrice",
                          "ticker.mark", "ticker.info.markPrice",
                          "ticker.info.lastPrice", "ticker.bid/ask mid"):
            log.warning(
                f"{symbol}: price {price} came from {source}, which is NOT a "
                f"live price. ROI, peak and every stop level this cycle are "
                f"derived from it.")
        return price

    def _throttled(self, key: str, emit):
        seen = self.__dict__.setdefault("_log_seen", {})
        now = time.time()
        if now - seen.get(key, 0.0) >= self.PRICE_LOG_INTERVAL_S:
            seen[key] = now
            emit()

    def _mark_and_last(self, symbol: str, t: dict | None):
        """(mark, last, source) from one ticker payload."""
        def _num(d, *fields):
            for f in fields:
                try:
                    v = float((d or {}).get(f))
                except (TypeError, ValueError):
                    continue
                if v > 0:
                    return v
            return None

        info = (t or {}).get("info") or {}
        last = _num(t, "last", "close") or _num(info, "lastPrice")
        mark = _num(t, "markPrice", "mark")
        src = "ticker.markPrice"
        if mark is None:
            mark = _num(info, "markPrice")
            src = "ticker.info.markPrice"
        if mark is None:
            mark = self._premium_index_mark(symbol)
            src = "premiumIndex.markPrice"
        return mark, last, src

    def _premium_index_mark(self, symbol: str) -> float | None:
        """
        Mark from the premium index. ccxt's binanceusdm ticker comes from
        /fapi/v1/ticker/24hr, which carries no markPrice, so this is the normal
        path rather than a fallback. Cached briefly because the guardian polls
        every 2.5s and may hold several positions; briefly, because a stale
        mark is the very thing this change exists to avoid.
        """
        cache = self.__dict__.setdefault("_mark_cache", {})
        now = time.time()
        hit = cache.get(symbol)
        if hit and now - hit[0] < self.MARK_CACHE_TTL_S:
            return hit[1]
        for name in ("fetchMarkPrice", "fetch_mark_price"):
            fn = getattr(self.exchange, name, None)
            if fn is None:
                continue
            try:
                row = fn(symbol) or {}
                v = float((row.get("info") or {}).get("markPrice")
                          or row.get("markPrice") or 0)
                if v > 0:
                    cache[symbol] = (now, v)
                    return v
            except Exception as e:
                log.debug(f"{name} failed for {symbol}: {e}")
        fn = getattr(self.exchange, "fapiPublicGetPremiumIndex", None)
        if fn is not None:
            try:
                row = fn({"symbol": symbol.split(":")[0].replace("/", "")}) or {}
                if isinstance(row, list):
                    row = row[0] if row else {}
                v = float(row.get("markPrice") or 0)
                if v > 0:
                    cache[symbol] = (now, v)
                    return v
            except Exception as e:
                log.debug(f"premiumIndex failed for {symbol}: {e}")
        cache[symbol] = (now, None)
        return None

    RANGE_CACHE_TTL_S = 300.0
    # Consecutive cycles a position must be absent before it counts as closed.
    MISSING_CONFIRMATIONS = 3
    # Cancels are retried well past the position's life: an order that reports
    # -2011 may still be resting, so give it many chances before abandoning it.
    MAX_CANCEL_ATTEMPTS = 60
    # How many closed trades to keep. Analysis needs the whole run, not a
    # window; at ~100 trades a day this holds well over a month.
    DEFAULT_MAX_CLOSED_TRADES = 5000
    # Guard against a single bad balance reading scaling position size.
    BALANCE_JUMP_TOLERANCE = 0.25
    BALANCE_JUMP_CONFIRMATIONS = 3
    # Consecutive failures before the bot treats itself as unable to see the
    # exchange and stops taking destructive action.
    BLIND_AFTER_FAILURES = 3
    # Seconds of slack before the order-placed stamp when querying the income
    # ledger, so the entry fill's commission is inside the window.
    INCOME_LOOKBACK_PAD_S = 120.0
    # Give up on the expensive account-wide listing after this many empties,
    # then re-probe on a slow clock in case the environment starts reporting.
    EMPTY_LISTING_LIMIT = 5
    WIDE_PROBE_INTERVAL_S = 1800.0
    # Candle timeframe for ATR. Must match the timeframe the strategy trades:
    # a 15m ATR is ~2.2x a 3m ATR, so stops sized off 15m were more than twice
    # as wide as the 3m chart justified (FORM: 24.4% ROI where 10.9% was right).
    atr_timeframe: str = "3m"

    def _range_24h(self, symbol: str, ticker: dict | None) -> tuple[float | None, float | None]:
        """
        24h high/low for a symbol.

        Prefers the ticker (one field lookup, no extra call). Falls back to
        deriving them from 24h of candles when the ticker omits them — which the
        demo endpoint does — and caches the result so a 5-second poll loop does
        not refetch klines every cycle.
        """
        hi = (ticker or {}).get("high")
        lo = (ticker or {}).get("low")
        try:
            if hi and lo and float(hi) > float(lo):
                self._range_source[symbol] = "ticker"
                return float(hi), float(lo)
        except (TypeError, ValueError):
            pass

        cached = self._range_cache.get(symbol)
        if cached and (time.time() - cached[2]) < self.RANGE_CACHE_TTL_S:
            self._range_source[symbol] = "candles (cached)"
            return cached[0], cached[1]

        try:
            # 96 x 15m = 24h, one request, coarse enough for a daily range.
            raw = self.exchange.fetch_ohlcv(symbol, "15m", limit=96)
            if raw:
                highs = [r[2] for r in raw if r[2] is not None]
                lows = [r[3] for r in raw if r[3] is not None]
                if highs and lows:
                    h, l = float(max(highs)), float(min(lows))
                    self._range_cache[symbol] = (h, l, time.time())
                    self._range_source[symbol] = "candles"
                    return h, l
            self._range_source[symbol] = "empty candle response"
            log.warning(f"{symbol}: 24h range fallback returned no candles")
        except Exception as e:
            # WARNING, not debug: this was swallowed silently and the range just
            # showed "n/a" with no way to tell why.
            self._range_source[symbol] = f"{type(e).__name__}: {e}"
            log.warning(f"{symbol}: 24h range fallback failed: {type(e).__name__}: {e}")
        return None, None

    def atr_pct(self, symbol: str) -> float | None:
        """
        ATR as a % of price, from the same cached candle window used for the
        24h range. Cached because the guardian polls every few seconds and this
        needs a klines call.
        """
        now = time.time()
        cached = self._atr_cache.get(symbol)
        if cached and (now - cached[1]) < self.RANGE_CACHE_TTL_S:
            return cached[0]
        try:
            raw = self.exchange.fetch_ohlcv(symbol, self.atr_timeframe, limit=120)
            if not raw or len(raw) < 15:
                return None
            trs = []
            prev_close = None
            for _ts, _o, h, l, c, _v in raw:
                if None in (h, l, c):
                    continue
                tr = h - l
                if prev_close is not None:
                    tr = max(tr, abs(h - prev_close), abs(l - prev_close))
                trs.append(tr)
                prev_close = c
            if not trs or not prev_close:
                return None
            atr = sum(trs[-14:]) / min(len(trs), 14)
            pct = atr / prev_close * 100
            self._atr_cache[symbol] = (pct, now)
            return pct
        except Exception as e:
            log.warning(f"{symbol}: ATR lookup failed: {type(e).__name__}: {e}")
            return None

    def _check_risk_invariant(self, pos: FuturesPosition, stop_roi: float):
        """
        Assert the loss at the placed stop matches the configured risk budget.

        Sizing and stop placement are computed in different components at
        different times. When they disagree the position is silently oversized
        — one trade risked 2.8x its budget before anyone noticed. This is the
        backstop: it cannot prevent the mismatch, but it makes it loud the
        moment a stop is placed rather than when the loss lands.
        """
        budget_pct = getattr(self, "risk_pct", 0.0)
        wallet = self._wallet_balance_cached
        if not budget_pct or wallet <= 0 or not pos.margin:
            return
        budget = wallet * budget_pct / 100.0
        loss = pos.margin * abs(stop_roi) / 100.0
        if budget > 0 and loss > budget * 1.25:
            msg = (f"RISK OVERSHOOT: stop at {-abs(stop_roi):.1f}% ROI on "
                   f"{pos.margin:.2f} margin risks {loss:.2f} USDT, but the "
                   f"budget is {budget:.2f} ({budget_pct}% of {wallet:.2f}) "
                   f"— {loss/budget:.1f}x. Sizing and stop disagree.")
            log.error(f"{pos.symbol}: {msg}")
            self._record(pos.symbol, "RISK_OVERSHOOT",
                         f"{loss:.2f} at risk vs {budget:.2f} budget "
                         f"({loss/budget:.1f}x)")
            with self._lock:
                self._risk_overshoots[pos.symbol] = {
                    "loss_at_stop": round(loss, 2), "budget": round(budget, 2),
                    "ratio": round(loss / budget, 2), "stop_roi": round(stop_roi, 2),
                }

    # Ages at which the position's ROI is sampled, in seconds. 0 captures the
    # ROI at FIRST OBSERVATION: for a bot-placed entry the pending poll runs
    # every second, so this is the position's ROI before the market has had
    # time to move it — i.e. the fill-versus-trigger gap expressed in ROI.
    # peak_roi cannot carry this (it is clamped at 0 and only ratchets up) and
    # neither can trough_roi (it cannot separate "opened at -13%" from "opened
    # flat and fell to -13%").
    ROI_CHECKPOINTS_S = (0, 60, 180, 300)

    @staticmethod
    def _drift_pct(meta: dict, side: str) -> float | None:
        """
        Adverse price movement between sizing and the actual entry, as a % of
        the sized price. Positive is always against the position, whichever
        side it is, so the two directions can be pooled.
        """
        try:
            ctx = meta.get("entry_context") or {}
            sized = ctx.get("sized_price")
            entry = meta.get("entry_price")
            if not sized or not entry:
                return None
            sized, entry = float(sized), float(entry)
            if sized <= 0:
                return None
            move = (entry - sized) / sized * 100.0
            # Sign so that POSITIVE is favourable to the position. A short
            # filled ABOVE where it was sized sold higher, which is better; a
            # long filled BELOW bought cheaper. An earlier version had this
            # backwards and reported a good fill as "against the position".
            return round(move if str(side).lower() == "short" else -move, 3)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _should_fail_fast(self, pos, state, current_roi: float) -> bool:
        """
        Cut a trade that never went green and is now losing.

        The loss floor and the never-green test are always required. Time
        alone would cut trades that are merely slow; the floor targets "going
        wrong" rather than "not going right yet".

        One further gate, off by default:

        fail_fast_require_worsening  cut only what is worse than where it
                                   started, so a recovering position is left
                                   alone regardless of depth.
        """
        cfg = self.cfg
        if not getattr(cfg, "fail_fast_s", 0):
            return False
        if state.peak_roi > _peak_ceiling(cfg):
            return False                      # it has been green — leave it
        floor = getattr(cfg, "fail_fast_loss_roi", 5.0)
        if current_roi > -abs(floor):
            return False                      # losing, but not badly

        # ROI at FIRST OBSERVATION. peak_roi cannot stand in for this: it is
        # clamped at 0, so a position first seen at -10.6% records peak 0.00
        # and looks identical to one that opened flat. The age-0 checkpoint
        # keeps the real figure.
        entry_roi = state.roi_checkpoints.get("0")

        # NOTE: an earlier version cut immediately when the age-0 checkpoint
        # was far underwater. That was wrong. roi_pct measures against
        # pos.entry_price, which IS the fill price, so ROI at the moment of
        # fill is identically zero. A non-zero age-0 reading therefore measures
        # how LATE the guardian looked, not how the trade opened — KOMA read
        # 0.00 because it was seen on the fill tick, GRIFFAIN -13.31 because it
        # was seen ~3s later. Cutting on it would cut on poll latency.

        # A position that is recovering is going the other way, whatever its
        # absolute ROI. RIVER reached -17.54% and closed +59.99%; cutting on
        # depth alone would have taken it. Only cut what is getting WORSE than
        # where it started.
        if getattr(cfg, "fail_fast_require_worsening", False):
            if entry_roi is not None and current_roi > entry_roi:
                return False

        opened = (self._pos_meta.get(pos.symbol, {}) or {}).get("opened_seen_at")
        if not opened:
            return False
        return (time.time() - float(opened)) >= cfg.fail_fast_s

    def _note_progress(self, pos, state, current_roi: float):
        """Record when the trade first went green, and its ROI at fixed ages."""
        now = time.time()
        if state.first_positive_at is None and current_roi > 0:
            state.first_positive_at = now
        opened = (self._pos_meta.get(pos.symbol, {}) or {}).get("opened_seen_at")
        if not opened:
            return
        age = now - float(opened)
        for mark in self.ROI_CHECKPOINTS_S:
            key = str(mark)
            if age >= mark and key not in state.roi_checkpoints:
                state.roi_checkpoints[key] = round(current_roi, 2)

    def _capture_entry_context(self, pos: FuturesPosition, price: float,
                               range_pos: float | None) -> dict:
        """
        Snapshot the conditions a position was entered under.

        Recorded once, on first sight, and never updated — the point is what was
        true AT ENTRY, so later analysis can ask which conditions actually paid.
        External entries are captured approximately (first observation rather
        than the true fill moment); auto-trade entries supply exact values via
        note_entry_context().
        """
        ctx = {
            "side": pos.side,
            "atr_pct": self.atr_pct(pos.symbol),
            "range_pos_24h": range_pos,
            "leverage": pos.effective_leverage,
            "margin": pos.margin,
            "captured": "observed",
        }
        meta = self._pos_meta.get(pos.symbol, {})
        hi, lo = meta.get("high_24h"), meta.get("low_24h")
        try:
            if hi and lo:
                ctx["dist_to_extreme_pct"] = (
                    abs((price - float(lo)) / float(lo) * 100) if pos.side == "long"
                    else abs((price - float(hi)) / float(hi) * 100))
        except (TypeError, ValueError):
            pass
        return ctx

    def note_entry_context(self, symbol: str, ctx: dict):
        """
        Record exact entry conditions from whoever opened the position.

        The auto-trader knows the RSI, distance and re-entry status it acted on;
        the guardian can only observe after the fact. Exact beats observed.
        """
        with self._lock:
            meta = self._pos_meta.setdefault(symbol, {})
            existing = meta.get("entry_context") or {}
            merged = {**existing, **ctx, "captured": "exact"}
            meta["entry_context"] = merged
        log.info(f"{symbol}: entry context recorded {ctx}")

    def effective_stop_roi(self, pos: FuturesPosition) -> float:
        """
        The initial stop distance to use for this position.

        With atr_stop_mult set, the stop scales with the coin's own volatility
        rather than being a constant — a fixed stop sits inside the noise on a
        volatile coin and needlessly far away on a calm one.
        """
        if not self.cfg.atr_stop_mult:
            return self.cfg.initial_stop_roi

        # Prefer the stop the position was actually SIZED for.
        #
        # A trailing-stop entry rests before it fills, so the guardian first
        # sees the position minutes after the entry sized it. Recomputing ATR
        # then can give a very different answer — a position sized for a 12.9%
        # stop received a 24.4% one, nearly 2x the intended risk. Sharing the
        # ATR cache only helps inside its TTL; the sized stop must be carried
        # with the position instead.
        with self._lock:
            sized = (self._pos_meta.get(pos.symbol, {})
                     .get("entry_context") or {}).get("sized_stop_roi")
        if sized:
            if pos.symbol not in self._stop_source_reported:
                self._stop_source_reported.add(pos.symbol)
                log.info(f"{pos.symbol}: stop {float(sized):.1f}% ROI from the "
                         f"SIZED handoff (entry and guardian agree)")
            return float(sized)

        if pos.symbol not in self._stop_source_reported:
            self._stop_source_reported.add(pos.symbol)
            log.warning(
                f"{pos.symbol}: no sized stop recorded — deriving from a fresh "
                f"ATR read. This position was opened before the handoff existed, "
                f"or its state was lost. The budget cap is the safety net."
            )

        a = self.atr_pct(pos.symbol)
        roi = atr_stop_roi(a, pos.effective_leverage, self.cfg)
        if roi is None:
            return self.cfg.initial_stop_roi
        return roi

    def _cap_stop_to_budget(self, pos: FuturesPosition, stop_roi: float) -> float:
        """
        Tighten a stop that would risk more than the configured budget.

        The budget and the position's ACTUAL margin uniquely determine the
        widest acceptable stop: loss = margin x stop_roi, so the cap is simply
        budget / margin. When sizing and stop placement disagree, this recovers
        the stop the position was really sized for rather than trusting an ATR
        reading taken at a different moment.

        Tightening is always safe in risk terms, but a tighter stop sits closer
        to the noise, so it is floored at atr_stop_min_roi. If even that floor
        exceeds the budget the position is simply too large for it, which is
        reported rather than silently accepted.
        """
        budget_pct = getattr(self, "risk_pct", 0.0)
        wallet = self._wallet_balance_cached
        if not budget_pct or wallet <= 0 or not pos.margin:
            return stop_roi

        budget = wallet * budget_pct / 100.0
        max_stop_roi = budget / pos.margin * 100.0
        if stop_roi <= max_stop_roi * 1.05:
            return stop_roi          # already within budget

        floor = self.cfg.atr_stop_min_roi or 0.0
        capped = max(max_stop_roi, floor)
        if capped >= stop_roi:
            return stop_roi

        # Margin drifts with unrealised PnL, so this recomputes every cycle and
        # would otherwise log an almost-identical line every few seconds. Report
        # it once, and again only if the level moves meaningfully.
        prev = self._capped_stop_reported.get(pos.symbol)
        if prev is None or abs(prev - capped) >= 0.5:
            log.warning(
                f"{pos.symbol}: stop {stop_roi:.1f}% ROI on {pos.margin:.2f} margin "
                f"would risk {pos.margin * stop_roi / 100:.2f} vs a {budget:.2f} "
                f"budget — tightening to {capped:.1f}% ROI "
                f"({pos.margin * capped / 100:.2f} at risk)."
            )
            self._record(pos.symbol, "stop_capped",
                         f"{stop_roi:.1f}% -> {capped:.1f}% ROI to hold risk at "
                         f"{budget:.2f} USDT")
            self._capped_stop_reported[pos.symbol] = capped
        if capped > max_stop_roi * 1.05:
            log.error(
                f"{pos.symbol}: even the {floor:.1f}% minimum stop risks "
                f"{pos.margin * capped / 100:.2f} vs a {budget:.2f} budget — "
                f"the position is too large for the risk setting."
            )
        return capped
        if abs(roi - self.cfg.initial_stop_roi) > 0.5:
            log.info(f"{pos.symbol}: ATR {a:.2f}% -> initial stop {roi:.0f}% ROI "
                     f"(fixed default would be {self.cfg.initial_stop_roi:.0f}%)")
        return roi

    def _ticker(self, symbol: str) -> dict | None:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            log.warning(f"ticker failed for {symbol}: {e}")
            return None

    # ── writing (the only two methods that mutate the account) ──────────────

    def market_max_qty(self, symbol: str) -> float | None:
        """
        The per-order quantity cap that applies to a MARKET-executing order.

        Binance publishes two caps: LOT_SIZE (limit orders) and
        MARKET_LOT_SIZE (market orders), and the market one is usually far
        smaller. Stops, trailing stops and entries here all execute as market,
        so the MARKET cap governs — but only LOT_SIZE was being read, so orders
        sized against it were rejected with -4005 "Quantity greater than max
        quantity".

        Returns the smaller of the two that are published.
        """
        try:
            m = self.exchange.market(symbol) or {}
            limits = m.get("limits") or {}
            caps = []
            for key in ("market", "amount"):
                mx = ((limits.get(key) or {}).get("max"))
                if mx:
                    caps.append(float(mx))
            # Fall back to the raw filter if ccxt did not surface it.
            if not caps:
                for f in ((m.get("info") or {}).get("filters") or []):
                    if f.get("filterType") in ("MARKET_LOT_SIZE", "LOT_SIZE"):
                        try:
                            caps.append(float(f.get("maxQty")))
                        except (TypeError, ValueError):
                            pass
            return min(caps) if caps else None
        except Exception:
            return None

    def _place_stop(self, pos: FuturesPosition, stop_price: float) -> str | None:
        """
        Place a reduce-only STOP_MARKET that closes `pos` at `stop_price`.

        reduceOnly=True is non-negotiable: it makes it impossible for this order
        to open or enlarge a position, whatever else goes wrong.
        """
        price_str = self.exchange.price_to_precision(pos.symbol, stop_price)
        side = stop_side(pos)

        # A position can be larger than the per-order cap. Placing one oversized
        # stop is rejected outright, so split it into several reduce-only stops
        # at the same trigger — together they still close the whole position.
        max_qty = self.market_max_qty(pos.symbol)
        if max_qty and pos.qty > max_qty:
            return self._place_split_stops(pos, stop_price, max_qty, side)

        qty_str = self.exchange.amount_to_precision(pos.symbol, pos.qty)

        if self.dry_run:
            log.info(f"[DRY RUN] would place {side} STOP_MARKET reduceOnly "
                     f"{qty_str} {pos.symbol} trigger={price_str}")
            return f"dry-{int(time.time()*1000)}"

        order = self.exchange.create_order(
            symbol=pos.symbol, type="STOP_MARKET", side=side,
            amount=float(qty_str), price=None,
            params={"stopPrice": float(price_str), "reduceOnly": True,
                    "workingType": self.cfg.stop_working_type},
        )
        oid = self._accepted_id(pos, order, "fixed stop")
        if not oid:
            return None
        log.info(f"Placed stop for {pos.symbol}: {side} STOP_MARKET "
                 f"trigger={price_str} id={oid}")
        return oid

    def _place_split_stops(self, pos: FuturesPosition, stop_price: float,
                           max_qty: float, side: str) -> str | None:
        """Place several reduce-only stops when one would exceed the order cap."""
        remaining = pos.qty
        ids: list[str] = []
        price_str = self.exchange.price_to_precision(pos.symbol, stop_price)
        n = 0
        while remaining > 0 and n < 20:
            chunk = min(remaining, max_qty)
            qty_str = self.exchange.amount_to_precision(pos.symbol, chunk)
            if float(qty_str) <= 0:
                break
            if self.dry_run:
                log.info(f"[DRY RUN] would place {side} STOP_MARKET reduceOnly "
                         f"{qty_str} {pos.symbol} trigger={price_str} (split)")
                ids.append(f"dry-split-{n}")
            else:
                order = self.exchange.create_order(
                    symbol=pos.symbol, type="STOP_MARKET", side=side,
                    amount=float(qty_str), price=None,
                    params={"stopPrice": float(price_str), "reduceOnly": True,
                    "workingType": self.cfg.stop_working_type},
                )
                # Same exposure as the single stop: a refused split leg must
                # not be recorded as though it were resting.
                split_id = self._accepted_id(pos, order, "split stop")
                if split_id:
                    ids.append(split_id)
            remaining -= float(qty_str)
            n += 1
        if not ids:
            return None
        log.warning(
            f"{pos.symbol}: position {pos.qty:g} exceeds the per-order cap "
            f"{max_qty:g} — placed {len(ids)} split stops at {price_str}")
        self._split_stop_ids[pos.symbol] = ids
        return ids[0]

    # Statuses that mean the exchange did NOT take the order. Binance answers
    # 200 with the order body either way, so create_order returning normally
    # is not acceptance.
    DEAD_STATUSES = {"REJECTED", "EXPIRED", "CANCELED", "CANCELLED"}

    def _confirm_resting(self, pos: FuturesPosition, order_id: str,
                         what: str) -> bool:
        """
        Is this order ACTUALLY in the algo book?

        `_accepted_id` decides from `order["status"]`, and on the algo endpoint
        that field is empty on every response — every TRAIL-RESPONSE in the
        logs reads `status=None`. Rejection there is ASYNCHRONOUS: the call
        returns 200 with an id, and the order is refused afterwards. So the
        guard written to catch exactly this passes everything.

        牛来 2026-09-20 06:55:37: the armed trail came back with an id, was
        logged as "locks in ~+2% ROI", and Binance's order history shows it
        REJECTED. One second later the adaptive trail was cancelled as
        "superseded" in favour of an order that did not exist. The position
        held nothing for 27 seconds until a floor finally landed at +1.1%,
        after five -2021 refusals, having peaked at +7.3%.

        Returns True only on POSITIVE confirmation. An unreadable book returns
        False, which keeps the existing protection in place — being unable to
        confirm and being rejected must lead to the same safe outcome.
        """
        if not order_id:
            return False
        try:
            wanted = str(order_id)
            unified = str(getattr(pos, "symbol", "") or "")
            base = unified.split(":")[0].replace("/", "").upper()
            for o in (self._algo_orders() or []):
                if not isinstance(o, dict):
                    continue
                sym = str(o.get("symbol") or "").upper()
                if sym and sym != base and sym != unified.upper():
                    continue
                for key in ("algoId", "orderId", "id", "clientAlgoId"):
                    if str(o.get(key) or "") == wanted:
                        return True
            log.error(
                f"{pos.symbol}: {what} id={wanted} is NOT in the algo book — "
                f"the exchange accepted the call and refused the order. "
                f"Existing protection is being KEPT.")
            self._record(pos.symbol, "order_not_resting", f"{what} {wanted}")
            return False
        except Exception as e:
            log.warning(
                f"{pos.symbol}: could not confirm {what} is resting "
                f"({_safe_err(e)}) — keeping existing protection.")
            return False

    def _accepted_id(self, pos: FuturesPosition, order: dict, what: str) -> str | None:
        """
        The order id, but only if the exchange actually took the order.

        OP 03:16:27 and 牛来 04:23:07 were both logged as "ARMED native
        trailing stop ... locks in ~+2% ROI" with an id. Binance's own order
        history shows both REJECTED. The guardian then cancelled the adaptive
        trail as "superseded" in favour of an order that did not exist, and the
        position ran unprotected until a floor landed seconds later.

        Returning None here makes every caller treat it as a failed placement,
        which is what they already do for an exception.
        """
        oid = str((order or {}).get("id")
                  or (order or {}).get("orderId") or "")
        status = str((order or {}).get("status")
                     or ((order or {}).get("info") or {}).get("status")
                     or "").upper()
        if status in self.DEAD_STATUSES:
            log.error(
                f"{pos.symbol}: {what} was {status} by the exchange — "
                f"NOT recording it as protection (id={oid or 'none'}). "
                f"The position is not protected by this order.")
            self._record(pos.symbol, "order_not_accepted",
                         f"{what} {status}")
            return None
        if not oid:
            log.error(f"{pos.symbol}: {what} returned no order id "
                      f"(status={status or 'unknown'}) — treating as failed.")
            return None
        if status and status not in ("NEW", "OPEN", "PARTIALLY_FILLED",
                                     "FILLED", "ACCEPTED", "WORKING"):
            log.warning(f"{pos.symbol}: {what} accepted with unexpected "
                        f"status {status} (id={oid}).")
        return oid

    # _activation_now was removed in v3.54.0. It computed an activation just
    # past the mark so a trail would be "live at once", and it never worked —
    # it sent the field as `activationPrice`, which the algo endpoint ignores.
    # Now the key is right it would be actively HARMFUL: Binance requires a
    # BUY trail's activation at or below the current price, so a value just
    # above would be REJECTED rather than quietly dropped. "Activate now" is
    # expressed by OMITTING the field, which is Binance's own default.

    def _activation_at_roi(self, pos: FuturesPosition, roi: float):
        """
        The price at which the position reaches `roi`, as an activatePrice.

        Binance activates a BUY trail when price <= activatePrice and a SELL
        trail when price >= it, and price_for_roi gives exactly the price where
        the ROI is reached — below entry for a short, above for a long. So the
        level lands on the side the exchange requires for either direction,
        with no epsilon.

        Derived from the ENTRY price, which does not move, so unlike a value
        computed from a mark read moments earlier it cannot be outrun.
        """
        if not pos.entry_price or pos.entry_price <= 0:
            return None
        try:
            raw = price_for_roi(pos, float(roi))
        except Exception:
            return None
        if not raw or raw <= 0:
            return None
        try:
            return float(self.exchange.price_to_precision(pos.symbol, raw))
        except Exception:
            return raw

    def _log_trail_response(self, pos: FuturesPosition,
                            params: dict, order: dict) -> None:
        """
        Compare the activation SENT against the one the exchange kept.

        A silent substitution is the dangerous case: the order is accepted, so
        nothing raises, and the trail rests at a level nobody asked for. That
        is what the wrong field name caused for every trail between v3.44 and
        v3.54.
        """
        try:
            info = (order or {}).get("info") or {}
            sent = params.get("activatePrice")
            got = info.get("activatePrice") or info.get("activationPrice")
            rate = info.get("priceRate") or info.get("callbackRate")
            log.info(
                f"TRAIL-RESPONSE {pos.symbol}: "
                f"id={info.get('orderId') or (order or {}).get('id')} "
                f"status={info.get('status')} "
                f"activatePrice sent={sent} kept={got} "
                f"callbackRate sent={params.get('callbackRate')} kept={rate} "
                f"workingType={info.get('workingType')}")
            if sent and got:
                try:
                    d = (float(got) - float(sent)) / float(sent) * 100
                except (TypeError, ValueError, ZeroDivisionError):
                    return
                if abs(d) > 0.01:
                    log.error(
                        f"TRAIL-ACTIVATION-IGNORED {pos.symbol}: asked for "
                        f"{sent}, the exchange kept {got} ({d:+.3f}%). The "
                        f"trail is NOT resting where it was placed.")
        except Exception as e:
            log.debug(f"{pos.symbol}: trail response log failed: {e}")

    def _create_trail_order(self, pos: FuturesPosition, side: str,
                            qty: float, cb: float,
                            activation: float | None = None,
                            activate_now: bool = True):
        """
        Place the trail, activating it immediately where the exchange allows.

        If the explicit activationPrice is refused, retry WITHOUT one. That is
        exactly the previous behaviour, so this can never leave a position with
        less protection than before — only with a trail that is live sooner.
        """
        params = {"callbackRate": cb, "reduceOnly": True,
                  "workingType": self.cfg.stop_working_type}
        # THE FIELD IS `activatePrice`, NOT `activationPrice`.
        #
        # ccxt maps activationPrice for POST /fapi/v1/order, then routes
        # conditional linear-swap orders to POST /fapi/v1/algoOrder
        # (binance.py:6888) — a different endpoint with a different schema.
        # Binance ignores unrecognised parameters rather than rejecting them,
        # so every trail since v3.44 was accepted with the activation silently
        # defaulted to the current price. Verified 2026-09-18 on COTI demo:
        # activatePrice=0.02181 against a mark of 0.020771 was kept EXACTLY,
        # +5.0022%, with reduceOnly=True and workingType=MARK_PRICE — the same
        # combination that had been substituted eight times running under the
        # other spelling. The response has always used `activatePrice`; that
        # was the clue.
        #
        # Only an explicit LEVEL is sent. "Activate now" is omitted on
        # purpose: Binance requires a BUY trail's activation to sit AT OR
        # BELOW the current price and a SELL trail's at or above, so an
        # already-satisfied activation cannot be expressed. Omitting it makes
        # Binance default to the current price, which is exactly what
        # "activate now" means — and now that the key is recognised, sending
        # one on the wrong side would be REJECTED rather than ignored.
        act = activation if activation else None
        if act:
            params["activatePrice"] = act
        # Log what is SENT, not just what comes back. G/USDT 2026-09-18
        # 14:46:43 asked for activation 0.009818 (+5% ROI) and the exchange
        # reported 0.0098795, which is mark — accepted, but with a DIFFERENT
        # activation, and no rejection to explain it. Without the request
        # beside the response there is no way to tell whether the value was
        # ignored, clamped, or never sent.
        log.info(f"TRAIL-REQUEST {pos.symbol}: side={side} qty={qty} "
                 f"params={params}")
        try:
            order = self.exchange.create_order(
                symbol=pos.symbol, type="TRAILING_STOP_MARKET", side=side,
                amount=qty, price=None, params=params)
            self._log_trail_response(pos, params, order)
            return order
        except Exception as e:
            if not act:
                raise
            log.warning(
                f"{pos.symbol}: trail refused with activatePrice={act} "
                f"({_safe_err(e)}) — retrying without one. Binance will then "
                f"derive one from its own latest price, which may leave the "
                f"trail dormant.")
            params.pop("activatePrice", None)
            log.info(f"TRAIL-REQUEST {pos.symbol} (retry): side={side} "
                     f"qty={qty} params={params}")
            order = self.exchange.create_order(
                symbol=pos.symbol, type="TRAILING_STOP_MARKET", side=side,
                amount=qty, price=None, params=params)
            self._log_trail_response(pos, params, order)
            return order

    def _place_native_trail(self, pos: FuturesPosition,
                            rescue: bool = False,
                            callback_pct: float | None = None,
                            activation: float | None = None) -> str | None:
        """
        Place Binance's own TRAILING_STOP_MARKET for the armed phase.

        Once this is resting the exchange tracks the peak continuously, tick by
        tick, so the trail no longer depends on how often the guardian polls —
        the sampling gap that let a spike go unnoticed disappears.

        callbackRate is a PRICE percentage, so its ROI cost scales with the
        position's leverage; that is why the lock-in check happens here rather
        than at startup.
        """
        lev = pos.effective_leverage
        if rescue:
            # RESCUE trail: the fixed stop was refused and this is the only
            # protection the position will get. Its job is to CAP THE LOSS, not
            # to lock in profit, so it uses its own wider callback and skips
            # the lock-in test entirely — that test asks whether the trail
            # would preserve a gain, which is the wrong question when the
            # position is already losing. Without an activationPrice Binance
            # activates it immediately at the current mark and trails from
            # there, so it fires on a `rescue_callback_pct` adverse move.
            # callback_pct lets a caller size this explicitly. The adaptive
            # trail used to pass its value by temporarily MUTATING
            # cfg.rescue_trail_callback_pct — which is shared state the
            # dashboard can also write through update_rules(), so it was a
            # race waiting for a second writer.
            cb = max(0.1, float(callback_pct
                                if callback_pct is not None
                                else self.cfg.rescue_trail_callback_pct))
            qty_str = self.exchange.amount_to_precision(pos.symbol, pos.qty)
            side = stop_side(pos)
            if self.dry_run:
                log.info(f"[DRY RUN] would place RESCUE {side} "
                         f"TRAILING_STOP_MARKET reduceOnly {qty_str} "
                         f"{pos.symbol} callbackRate={cb}%")
                return f"dry-rescue-{int(time.time()*1000)}"
            order = self._create_trail_order(
                pos, side, float(qty_str), cb)
            oid = self._accepted_id(pos, order, "RESCUE trailing stop")
            if not oid:
                return None
            self._log_trail_activation(pos, order, oid, cb, "RESCUE trail")
            log.warning(
                f"{pos.symbol}: RESCUE trailing stop placed — activates now, "
                f"closes on a {cb}% adverse move ({cb * lev:.0f}% ROI at "
                f"{lev:.0f}x). This caps the loss; it does not lock in profit. "
                f"id={oid}")
            return oid

        cb = trail_callback_price_pct(lev, self.cfg)
        locked = trail_locks_in(lev, self.cfg)
        if locked <= 0:
            log.warning(
                f"{pos.symbol}: arming a {cb}% trail at {lev:.0f}x gives back "
                f"{callback_roi_at(lev, self.cfg):.0f}% ROI, but the arm level is "
                f"+{self.cfg.arm_roi:.0f}% — the trail would engage at "
                f"{locked:+.0f}% ROI (at or below entry). Keeping the fixed stop "
                f"instead. Raise GUARD_ARM_ROI or lower GUARD_TRAIL_CALLBACK_PCT."
            )
            return None

        qty_str = self.exchange.amount_to_precision(pos.symbol, pos.qty)
        side = stop_side(pos)
        if self.dry_run:
            log.info(f"[DRY RUN] would place {side} TRAILING_STOP_MARKET reduceOnly "
                     f"{qty_str} {pos.symbol} callbackRate={cb}%")
            return f"dry-trail-{int(time.time()*1000)}"

        order = self._create_trail_order(pos, side, float(qty_str), cb,
                                         activation=activation)
        oid = self._accepted_id(pos, order, "ARMED trailing stop")
        if not oid:
            return None
        self._log_trail_activation(pos, order, oid, cb, "ARMED trail")
        log.info(
            f"{pos.symbol}: ARMED native trailing stop (callback {cb}% price = "
            f"{callback_roi_at(lev, self.cfg):.0f}% ROI at {lev:.0f}x), locks in "
            f"~{locked:+.0f}% ROI. id={oid}"
        )
        return oid

    def _log_trail_activation(self, pos: FuturesPosition, order: dict,
                              oid: str, cb: float, what: str):
        """
        Record the activation price Binance assigned, and how far it sits from
        the mark we hold.

        An explicit activationPrice is now sent (GUARD_TRAIL_ACTIVATE_NOW), so
        this is the confirmation that it landed — and still catches the case
        where the exchange refused it and the retry let Binance derive its own.
        A BUY trail (a short) activates at price <= activationPrice, so one BELOW
        the mark means the trail rests DORMANT until price falls that far —
        and if the move reverses first it never activates at all, while the
        guardian holds an id and the exchange really is holding the order.

        ONE 2026-09-17 19:56:49 shows that shape directly: activation
        0.0015552 against a mark of at least 0.0015585, 0.21% away.

        ONE 17:00 is the case this line exists for. Its trail was accepted,
        Finished, and still gave back 47 ROI points on a 3% callback — which
        no explanation so far covers. The gap recorded here says whether the
        trail was live when it was placed or waiting to be.
        """
        info = (order or {}).get("info") or {}
        try:
            act = float(info.get("activatePrice")
                        or info.get("activationPrice") or 0)
        except (TypeError, ValueError):
            act = 0.0
        mark = self.mark_price(pos)
        if act > 0 and mark and mark > 0:
            gap = (act - mark) / mark * 100
            # "buy" closes a short and activates at price <= act, so an
            # activation below the mark is not yet reachable.
            live = act >= mark if stop_side(pos) == "buy" else act <= mark
            log.info(
                f"TRAIL-ACTIVATION {pos.symbol}: {what} id={oid} "
                f"activation={act} mark={mark} ({gap:+.3f}%) callback={cb}% "
                f"-> {'LIVE now' if live else 'DORMANT until price reaches it'}"
                f" | src={getattr(self, '_last_price_source', '?')}")
            if not live:
                # An arm-at-entry trail is dormant ON PURPOSE — it waits for
                # the arm level. Only an UNINTENDED dormancy is an alarm, or
                # this cries wolf on every entry.
                deliberate = bool(getattr(self.cfg, "arm_at_entry", False)) \
                    and what.startswith("ARMED")
                if deliberate:
                    log.info(
                        f"{pos.symbol}: {what} is dormant by design — it arms "
                        f"when price reaches {act} ({abs(gap):.3f}% away).")
                else:
                    log.warning(
                        f"{pos.symbol}: {what} is resting but NOT yet active "
                        f"— it protects nothing until price moves "
                        f"{abs(gap):.3f}% further in profit.")
        else:
            log.info(
                f"TRAIL-ACTIVATION {pos.symbol}: {what} id={oid} callback={cb}%"
                f" — exchange returned no activation price to check.")

    def _arm_at_entry(self, pos: FuturesPosition, state) -> None:
        """
        Place the armed trail NOW, dormant, with its activation at the arm-ROI
        price — so the exchange arms it tick by tick instead of the guardian
        noticing on a poll.

        The guardian polls every 2.5s. ROI can reach +10% and fall back to +2%
        between two polls and the arm never fires, because arming was an event
        the guardian had to WITNESS. Binance sees every tick. Handing it the
        level turns a missed observation into a resting order.

        Dormancy is the mechanism here, not the bug it was elsewhere: the trail
        protects nothing until price reaches the arm level, which is exactly
        when the old code would have placed it. It is strictly earlier, never
        later.

        No stacking: one trail per role. This writes native_trail_id, and the
        arm block in manage_position is already gated on `not
        state.native_trail_id`, so it becomes a no-op — including its
        supersede of the adaptive trail. The adaptive trail therefore LIVES,
        which is a gain: every supersede discarded the extreme Binance had
        been tracking on it, and that extreme is the peak the guardian cannot
        see. Both are reduceOnly and close the same direction; whichever fires
        first closes the position and the other is rejected harmlessly.
        """
        if not getattr(self.cfg, "arm_at_entry", False):
            return                              # GUARD_ARM_AT_ENTRY=false
        if not self.cfg.use_native_trail:
            return
        if state.native_trail_id:
            return                              # one trail per role
        if getattr(state, "armed", False):
            return                  # already past arm_roi — let the arm block
                                    # place it live rather than dormant here
        arm_roi = float(getattr(self.cfg, "arm_roi", 0) or 0)
        if arm_roi <= 0:
            return
        activation = self._activation_at_roi(pos, arm_roi)
        if not activation:
            log.warning(f"{pos.symbol}: cannot derive the arm level from "
                        f"entry {pos.entry_price} — leaving the trail to the "
                        f"poll-driven arm block.")
            return
        try:
            trail_id = self._place_native_trail(pos, activation=activation)
        except Exception as e:
            # Never fatal. The poll-driven arm block is still there and will
            # place the trail the old way if this fails.
            log.warning(f"{pos.symbol}: arm-at-entry failed ({_safe_err(e)}) "
                        f"— falling back to arming on a poll.")
            self._record(pos.symbol, "arm_at_entry_failed", str(e))
            return
        if not trail_id:
            return
        state.native_trail_id = trail_id
        self._all_stop_ids.setdefault(pos.symbol, []).append(trail_id)
        log.warning(
            f"{pos.symbol}: ARMED AT ENTRY — trail resting with activation at "
            f"{activation} (+{arm_roi:g}% ROI). It is DORMANT BY DESIGN and "
            f"arms the moment the exchange sees that price, with no poll "
            f"needed. id={trail_id}")
        self._record(pos.symbol, "armed_at_entry",
                     f"activation {activation} (+{arm_roi:g}% ROI)")

    def _ensure_adaptive_trail(self, pos: FuturesPosition, state, stop_roi: float):
        """
        A native trail at THIS position's stop distance, placed at adoption.

        callbackRate = stop_roi / leverage, i.e. the stop expressed as a price
        percentage. Across 158 recorded trades that lands between 0.69% and
        2.53% — inside Binance's 0.1-10% band with no clamping.

        Placed ONCE, guarded the same way as the profit floor: it returns if an
        id is already held, the id is persisted so a restart cannot place a
        second, and the level does not ratchet on this side.
        """
        if not getattr(self.cfg, "adaptive_trail_enabled", False):
            return
        if state.adaptive_trail_id or state.native_trail_id:
            return
        lev = pos.effective_leverage or 1.0
        cb = round(abs(stop_roi) / lev, 2)
        if cb < 0.1 or cb > 10:
            log.warning(f"{pos.symbol}: adaptive trail {cb}% outside the "
                        f"0.1-10% band — leaving the fixed stop alone")
            return
        try:
            oid = self._place_native_trail(pos, rescue=True, callback_pct=cb)
            if not oid:
                return
            state.adaptive_trail_id = oid
            self._all_stop_ids.setdefault(pos.symbol, []).append(oid)
            log.warning(
                f"{pos.symbol}: ADAPTIVE TRAIL placed, callback {cb}% of price "
                f"= {abs(stop_roi):.1f}% ROI at {lev:.0f}x. Exchange-managed, "
                f"tick-by-tick — it does not depend on the guardian's poll.")
            self._record(pos.symbol, "adaptive_trail",
                         f"trail {cb}% price / {abs(stop_roi):.1f}% ROI")
        except Exception as e:
            log.error(f"{pos.symbol}: adaptive trail failed, fixed stop "
                      f"remains: {_safe_err(e)}")

    PROTECTION_AUDIT_INTERVAL_S = 30.0

    def _audit_protection(self, pos: FuturesPosition, state):
        """
        Reconcile what the guardian THINKS is protecting this position against
        what the exchange actually has resting, and say so in one greppable
        line per symbol.

            docker logs $C | grep PROTECTION

        Anomalies are logged at WARNING with a leading tag so each class can be
        counted on its own:

            PROTECTION-UNPROTECTED   nothing protective resting at all
            PROTECTION-MISSING       a tracked id the exchange does not report
            PROTECTION-UNTRACKED     a protective order the guardian did not place
            PROTECTION-DUPLICATE     more than one trailing stop on one position
            PROTECTION-OVERLAP       adaptive trail and armed trail together

        Throttled per symbol: a healthy position logs at most every 30s.
        """
        now = time.time()
        last = self._audit_last.get(pos.symbol, 0.0)
        if now - last < self.PROTECTION_AUDIT_INTERVAL_S:
            return
        self._audit_last[pos.symbol] = now

        tracked = {
            "fixed": state.stop_order_id,
            "adaptive": getattr(state, "adaptive_trail_id", None),
            "armed": state.native_trail_id,
            "floor": getattr(state, "floor_stop_id", None),
        }
        tracked_ids = {v for v in tracked.values() if v}

        try:
            # fetch_open_orders is the UNIFIED book, which structurally cannot
            # contain what this audits: conditional and trailing stops are ALGO
            # orders in a separate book. _open_orders_multi already knows that
            # ("they never appear in /fapi/v1/openOrders"); the audit did not,
            # so it read a book that can never hold them and reported
            # PROTECTION-BLIND on every cycle. WLD 2026-09-18 07:15:20-24 shows
            # all three in one pass: unified 0, raw fapi 0, algo 2.
            rows = list(self.fetch_open_orders(pos.symbol) or [])
            try:
                unified = str(getattr(pos, "symbol", "") or "")
                base = unified.split(":")[0].replace("/", "").upper()
                for o in (self._algo_orders() or []):
                    if not isinstance(o, dict):
                        continue
                    # Algo rows carry the WIRE symbol (ONEUSDT), not the
                    # unified one (ONE/USDT:USDT). Compare on the stripped form
                    # so the filter does not silently drop everything — an
                    # over-strict match here would look exactly like the bug
                    # being fixed.
                    sym = str(o.get("symbol") or "").upper()
                    if not sym or sym == unified.upper() or sym == base:
                        rows.append(o)
            except Exception as e:
                log.debug(f"{pos.symbol}: algo book unavailable to audit: {e}")
            live = [self._normalise_order(o) for o in rows]
        except Exception as e:
            log.warning(f"PROTECTION {pos.symbol}: could not list orders "
                        f"({_safe_err(e)}) — audit skipped this cycle")
            return

        prot = [o for o in live
                if o.get("reduce_only") and "STOP" in (o.get("type") or "").upper()]
        # An id present in ANY book is present. Classification is a separate
        # question from existence, and conflating them produced a false alarm
        # the moment the audit started seeing the algo book: G/USDT
        # 2026-09-18 08:38:39 reported "2 order(s) but NONE is protective" and
        # both tracked ids MISSING, while both were resting fine.
        #
        # Binance's algo rows do not always carry reduceOnly, so they fail the
        # protective test above. That test stays STRICT on purpose — entries
        # here are TRAILING_STOP orders too, and widening it would let the
        # sweep treat an entry as a stop and cancel it.
        all_ids = {o["id"] for o in live if o.get("id")}
        live_ids = {o["id"] for o in prot if o.get("id")}
        unclassified = [o for o in live
                        if o.get("id") and o["id"] not in live_ids
                        and o["id"] in tracked_ids]
        if unclassified:
            log.debug(
                f"{pos.symbol}: {len(unclassified)} tracked order(s) in the "
                f"listing could not be classified as protective (algo rows "
                f"often omit reduceOnly): "
                + ", ".join(f"{o['id']}:{o.get('type') or '?'}"
                            for o in unclassified))
        trails = [o for o in prot
                  if "TRAILING" in (o.get("type") or "").upper()]

        held = ", ".join(f"{k}={v}" for k, v in tracked.items() if v) or "NONE"
        log.info(f"PROTECTION {pos.symbol}: roi={roi_pct(pos, self.mark_price(pos) or pos.entry_price):+.1f}% "
                 f"| guardian holds [{held}] | exchange has {len(prot)} "
                 f"protective order(s), {len(trails)} trailing")

        if not live:
            # The listing returned NOTHING AT ALL — not "your orders are gone"
            # but "I cannot see any orders on this symbol". The account-wide
            # probe has been observed returning empty while orders demonstrably
            # rested. Claiming UNPROTECTED here cries wolf on every cycle.
            log.info(f"PROTECTION-BLIND {pos.symbol}: the order listing "
                     f"returned nothing at all, so the audit cannot confirm "
                     f"anything. guardian holds [{held}].")
            return
        if not prot:
            # Only an alarm when the tracked orders are genuinely ABSENT. If
            # they are in the listing but unclassifiable, that is a field-shape
            # problem in the algo payload, not an unprotected position.
            tracked_present = [o for o in tracked_ids if o in all_ids]
            if tracked_present:
                log.info(
                    f"PROTECTION {pos.symbol}: the listing returned "
                    f"{len(live)} order(s); {len(tracked_present)} of the "
                    f"guardian's are present but carry no reduceOnly flag, so "
                    f"they cannot be confirmed protective. Not treating this "
                    f"as unprotected. guardian holds [{held}].")
            else:
                log.warning(f"PROTECTION-UNPROTECTED {pos.symbol}: the listing "
                            f"returned {len(live)} order(s) but NONE is "
                            f"protective. guardian holds [{held}].")
        for name, oid in tracked.items():
            # Existence, not classification — see all_ids above.
            if oid and oid not in all_ids:
                log.warning(f"PROTECTION-MISSING {pos.symbol}: tracked {name} "
                            f"{oid} is not in the exchange's open orders — it "
                            f"filled, was cancelled, or the listing is blind.")
        for o in prot:
            if o.get("id") and o["id"] not in tracked_ids:
                log.warning(f"PROTECTION-UNTRACKED {pos.symbol}: protective "
                            f"order {o['id']} ({o.get('type')}) is resting but "
                            f"the guardian did not place it.")
        if len(trails) > 1:
            log.warning(f"PROTECTION-DUPLICATE {pos.symbol}: {len(trails)} "
                        f"trailing stops resting: "
                        f"{[t.get('id') for t in trails]}. Exactly one is "
                        f"intended.")
        if tracked["adaptive"] and tracked["armed"]:
            # Under arm-at-entry BOTH trails are meant to rest: the armed one
            # is placed dormant at entry, so there is no arm event to
            # supersede the adaptive one — and not superseding it is the point,
            # since every supersede discarded the extreme Binance had tracked.
            if getattr(self.cfg, "arm_at_entry", False):
                log.info(f"PROTECTION {pos.symbol}: adaptive and armed trails "
                         f"both resting, as arm-at-entry intends "
                         f"({tracked['adaptive']}, {tracked['armed']}).")
            else:
                log.warning(f"PROTECTION-OVERLAP {pos.symbol}: adaptive trail "
                            f"{tracked['adaptive']} and armed trail "
                            f"{tracked['armed']} are both held — arming should "
                            f"have superseded the adaptive one.")

    def _wait_counterfactual(self, symbol: str, side: str, meta: dict) -> dict:
        """
        What would WAITING have done?

        The entry is a trailing stop that fires on a retrace, so it can fill on
        a noise wiggle inside a fast move — BR filled 0.13% off a candle low and
        ran 3.5% against it in six seconds. The operator's proposal is to wait
        two candles after the signal and then enter at market.

        This answers that from history instead of changing behaviour. At close,
        it pulls the candles covering the signal and reports where price was
        one and two candles later, and how far it travelled against the
        position in between.

            wait_1c_pct / wait_2c_pct   price vs the ACTUAL fill, signed so
                                        POSITIVE means waiting got a better
                                        entry
            wait_worst_pct              the worst adverse excursion between the
                                        signal and the 2-candle mark, as a % of
                                        the fill. If that exceeds the stop
                                        distance, waiting would have watched
                                        the move happen rather than sat in it.

        Measurement only. One klines call per closed trade, off the hot path,
        and any failure returns empty rather than disturbing the record.
        """
        out = {}
        try:
            ctx = meta.get("entry_context") or {}
            sized_at = ctx.get("sized_at")
            entry = meta.get("entry_price")
            if not sized_at or not entry:
                return out
            tf = getattr(self, "atr_timeframe", "3m")
            secs = 180 if tf.endswith("m") and tf[:-1] == "3" else 180
            since = int((float(sized_at) - secs) * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=tf,
                                              since=since, limit=4)
            after = [c for c in (ohlcv or []) if c[0] / 1000.0 >= float(sized_at)]
            if not after:
                return out
            entry = float(entry)
            short = str(side).lower() == "short"

            def vs_fill(price):
                # positive = a BETTER entry than the one actually taken
                d = (price - entry) / entry * 100.0
                return round(d if short else -d, 3)

            if len(after) >= 1:
                out["wait_1c_pct"] = vs_fill(float(after[0][4]))
            if len(after) >= 2:
                out["wait_2c_pct"] = vs_fill(float(after[1][4]))
            window = after[:2]
            if window:
                # worst move AGAINST the position over the wait
                adverse = [(max(float(c[2]) for c in window) - entry) / entry * 100.0
                           if short else
                           (entry - min(float(c[3]) for c in window)) / entry * 100.0]
                out["wait_worst_pct"] = round(max(adverse), 3)
            out["wait_candles_seen"] = len(after)
            return out
        except Exception as e:
            log.debug(f"{symbol}: wait counterfactual unavailable: "
                      f"{_safe_err(e)}")
            return out

    def _cancel_profit_floor(self, symbol: str, state):
        """
        Drop the floor when the position closes.

        reap_orphan_stops() would take it within 120s anyway — a reduce-only
        stop on a symbol with no position. Cancelling it here closes that
        window so nothing rests against a later position on the same symbol.
        """
        oid = getattr(state, "floor_stop_id", None)
        if not oid:
            return
        try:
            pos = type("P", (), {"symbol": symbol})()
            self._cancel_stop(pos, oid)
        except Exception as e:
            log.debug(f"{symbol}: floor cancel deferred to the sweep: "
                      f"{_safe_err(e)}")
        finally:
            state.floor_stop_id = None
            state.floor_roi = None

    def _ensure_profit_floor(self, pos: FuturesPosition, state, price: float):
        """
        Place a hard STOP_MARKET at breakeven once the position has been far
        enough ahead, and never move or cancel it while the position lives.

        A peak above breakeven_at_roi must not become a loss. Arming the native
        trail used to cancel the fixed stop, so above arm_roi the only
        protection was a 0.15%-of-price trail — narrower than one candle on
        these coins. PUNDIX peaked +7.36% and closed -10.30%; BR +9.63% ->
        -10.07%; HIVE +10.98% -> -1.83%.

        Placed ONCE and never repositioned. Three things prevent stacking:
          * it returns immediately if state.floor_stop_id is already set
          * floor_stop_id is PERSISTED, so a restart does not re-place it
          * the level is fixed, so there is no ratchet to re-issue it
        It is exempt from _cancel_superseded_stops (which runs while the
        position is OPEN) but NOT from reap_orphan_stops (which runs when the
        symbol has NO position), so it cannot outlive the trade.
        """
        if not getattr(self.cfg, "profit_floor_enabled", True):
            return
        if state.floor_stop_id:
            return                      # already placed — never a second one
        at = getattr(self.cfg, "breakeven_at_roi", 0.0)
        level = getattr(self.cfg, "breakeven_stop_roi", 0.0)
        if not at or state.peak_roi < at:
            return

        # ADAPTIVE LEVEL. Placing the floor at a fixed +2% ROI assumes price is
        # still above it. If the peak was made and given back between two 2.5s
        # polls, a stop at +2% sits on the wrong side of the market: Binance
        # refuses it with -2021 "would immediately trigger", and every retry
        # for the life of the position fails identically. ARK peaked +3.98%,
        # showed "Stop @ ROI +2%" on the dashboard, and ran to -20% with
        # nothing resting.
        #
        # So the floor is placed at the best level STILL AVAILABLE: the
        # configured one when the peak is intact, less when it is not, and
        # never above the current price. It locks what is actually there
        # rather than failing to lock anything.
        current = roi_pct(pos, price)
        buffer = max(0.2, abs(level) * 0.25)
        usable = min(level, current - buffer)
        if usable <= 0:
            # Nothing positive left to protect. Fail-fast owns this case; a
            # floor at or below break-even would only lock in a loss.
            self._floor_unavailable(pos, state, current, level)
            return
        try:
            floor_price = price_for_roi(pos, usable)
            oid = self._place_stop(pos, floor_price)
            if not oid:
                # A None return is a FAILURE, not a quiet no-op. This path was
                # silent, so a floor that never went on looked identical to one
                # that did.
                self._floor_unavailable(pos, state, current, usable)
                return
            state.floor_stop_id = oid
            state.floor_roi = usable
            state.floor_attempts = 0
            self._all_stop_ids.setdefault(pos.symbol, []).append(oid)
            log.warning(
                f"{pos.symbol}: PROFIT FLOOR placed at +{usable:.1f}% ROI "
                f"@ {floor_price} (peak reached +{state.peak_roi:.1f}%). "
                f"This order is not cancelled while the position is open.")
            self._record(pos.symbol, "profit_floor",
                         f"floor at +{usable:.1f}% ROI after peak "
                         f"+{state.peak_roi:.1f}%")
        except Exception as e:
            self._floor_unavailable(pos, state, current, usable, _safe_err(e))

    def _floor_unavailable(self, pos, state, current: float, wanted: float,
                           err: str = ""):
        """
        The floor could not be placed. Say so LOUDLY.

        Previously this logged once at error and moved on, leaving
        floor_stop_id as None. The dashboard kept showing "Stop @ ROI +2%",
        because that field is the guardian's INTENT, not what the exchange
        holds — so a position with no floor at all looked protected.
        """
        state.floor_attempts = getattr(state, "floor_attempts", 0) + 1
        if state.floor_attempts in (1, 5) or state.floor_attempts % 40 == 0:
            log.error(
                f"PROTECTION-NO-FLOOR {pos.symbol}: peak reached "
                f"+{state.peak_roi:.1f}% but NO profit floor is resting "
                f"(attempt {state.floor_attempts}, wanted +{wanted:.1f}% ROI, "
                f"currently {current:+.1f}%). This position is NOT protected "
                f"at break-even." + (f" exchange said: {err}" if err else ""))
            self._record(pos.symbol, "no_profit_floor",
                         f"peak +{state.peak_roi:.1f}%, no floor resting")

    def _cancel_superseded_stops(self, pos: FuturesPosition, keep: str | None,
                                 state=None):
        """
        Cancel protective stops for this symbol other than the current one.

        `state` MUST be the live object manage_position is working on. It used
        to read self._states[symbol], but manage_position mutates a LOCAL state
        and only writes it back later — so a trail or floor placed earlier in
        the same cycle was invisible here and got cancelled as superseded.

        SOLV 19:26:49 lost its adaptive trail to exactly that, one second after
        it was placed.
        """
        ids = list(self._all_stop_ids.get(pos.symbol) or [])
        st_ = state if state is not None else self._states.get(pos.symbol)
        # Never superseded: the profit floor is the guarantee that a position
        # which has been well ahead does not close at a loss, and the trails
        # are the protection that does not depend on the guardian's poll.
        #
        # native_trail_id was MISSING here, and arm-at-entry made that fatal.
        # The armed trail is placed at adoption, lands in _all_stop_ids, and
        # the very next fixed-stop placement swept it — 4 seconds later on
        # LSK live 2026-09-20 08:03:23, 4 seconds on demo 07:50:12. Both
        # instances, every position. Arm-at-entry has therefore never survived
        # its own first cycle, which is why the poll-driven ratchet has gone
        # on doing all the work.
        #
        # Same class of bug as SOLV below: a sweep that did not know about a
        # protection added after it was written.
        protected = {i for i in (
            getattr(st_, "floor_stop_id", None),
            getattr(st_, "adaptive_trail_id", None),
            getattr(st_, "native_trail_id", None),
        ) if i} if st_ else set()
        for oid in ids:
            if keep and oid == keep:
                continue
            if oid in protected:
                continue
            if self._cancel_stop(pos, oid):
                self._all_stop_ids[pos.symbol] = [
                    i for i in self._all_stop_ids.get(pos.symbol, []) if i != oid]

    @staticmethod
    def _order_already_gone(err) -> bool:
        """
        -2011 "Unknown order sent" means the order does NOT exist — already
        filled, already cancelled, or from a previous run. It is a SUCCESSFUL
        outcome for a cancel, not a failure, and must not be reported as
        "still resting".
        """
        t = str(err).lower()
        return "-2011" in t or "unknown order" in t

    def _queue_pending_cancel(self, symbol: str, order_id: str):
        """
        Remember a cancel that did not take, so it survives the position
        closing and a container restart.

        Neither order listing nor the cancel result is trustworthy on this
        venue, so the only durable record of "this order should not exist" is
        the bot's own queue.
        """
        with self._lock:
            q = self._pending_cancels.setdefault(symbol, [])
            if order_id not in q:
                q.append(order_id)

    def _clear_pending_cancel(self, symbol: str, order_id: str):
        with self._lock:
            q = self._pending_cancels.get(symbol) or []
            self._pending_cancels[symbol] = [i for i in q if i != order_id]
            if not self._pending_cancels[symbol]:
                self._pending_cancels.pop(symbol, None)
        self._cancel_attempts.pop(order_id, None)

    def drain_pending_cancels(self) -> int:
        """
        Retry every cancel that has not yet succeeded, including for symbols
        with no open position. Runs on the sweep clock.
        """
        with self._lock:
            work = {k: list(v) for k, v in self._pending_cancels.items()}
        done = 0
        for symbol, ids in work.items():
            for oid in ids:
                if oid in self._cancelled_ids:
                    self._clear_pending_cancel(symbol, oid)
                    continue
                n = self._cancel_attempts.get(oid, 0)
                if n >= self.MAX_CANCEL_ATTEMPTS:
                    continue
                try:
                    self.exchange.cancel_order(oid, symbol)
                    log.warning(f"{symbol}: pending cancel of {oid} SUCCEEDED "
                                f"on attempt {n + 1}")
                    self._record(symbol, "pending_cancel_ok", f"id={oid}")
                    self._clear_pending_cancel(symbol, oid)
                    done += 1
                except Exception as e:
                    ok, gone = self._cancel_algo_order(oid, symbol)
                    if ok or gone:
                        self._cancelled_ids.add(oid)
                        self._clear_pending_cancel(symbol, oid)
                        done += 1
                        continue
                    self._cancel_attempts[oid] = n + 1
                    if self._cancel_attempts[oid] == self.MAX_CANCEL_ATTEMPTS:
                        log.error(
                            f"{symbol}: giving up on cancelling {oid} after "
                            f"{self.MAX_CANCEL_ATTEMPTS} attempts ({e}). If it "
                            f"is still on the exchange, cancel it by hand.")
                        self._record(symbol, "cancel_abandoned", f"id={oid}")
        return done

    def _cancel_stop(self, pos: FuturesPosition, order_id: str) -> bool:
        """
        Cancel a stop the guardian is managing.

        Callers must only pass IDs of orders that satisfied is_protective_stop().
        """
        if self.dry_run:
            log.info(f"[DRY RUN] would cancel order {order_id} on {pos.symbol}")
            return True
        try:
            self.exchange.cancel_order(order_id, pos.symbol)
            log.info(f"Cancelled superseded stop {order_id} on {pos.symbol}")
            self._clear_pending_cancel(pos.symbol, order_id)
            return True
        except Exception as e:
            # -2011 "Unknown order sent" is NOT proof the order is gone here.
            # An order that was still resting on the exchange returned -2011 to
            # a cancel, so treating it as success dropped it from tracking and
            # orphaned it permanently. Count the attempts instead and keep it
            # queued; a retry costs one request and may succeed.
            # These stops are ALGO orders, so a regular cancel misses them.
            ok, gone = self._cancel_algo_order(order_id, pos.symbol)
            if ok or gone:
                self._cancelled_ids.add(order_id)
                self._clear_pending_cancel(pos.symbol, order_id)
                return True
            n = self._cancel_attempts.get(order_id, 0) + 1
            self._cancel_attempts[order_id] = n
            gone = self._order_already_gone(e)
            if n == 1 or n % 20 == 0:
                log.warning(
                    f"cancel {order_id} on {pos.symbol} failed (attempt {n}): {e}"
                    + ("  [-2011 is not reliable on this venue; still queued]"
                       if gone else ""))
            self._queue_pending_cancel(pos.symbol, order_id)
            return False

    # ── per-position management ─────────────────────────────────────────────

    def _record(self, symbol: str, action: str, detail: str):
        entry = {"ts": time.time(), "symbol": symbol,
                 "action": action, "detail": detail}
        with self._lock:
            self._actions.append(entry)
            self._actions = self._actions[-50:]

    def manage_position(self, pos: FuturesPosition):
        price = self.mark_price(pos)
        if price is None:
            return
        t = self._ticker(pos.symbol) or {}
        hi, lo = self._range_24h(pos.symbol, t)
        current_roi = roi_pct(pos, price)
        range_pos = None
        dist_low = dist_high = None
        try:
            if hi and lo and float(hi) > float(lo):
                hi_f, lo_f = float(hi), float(lo)
                range_pos = (price - lo_f) / (hi_f - lo_f)
                # How far price has travelled from each extreme, as a %.
                dist_low = (price - lo_f) / lo_f * 100
                dist_high = (price - hi_f) / hi_f * 100
        except (TypeError, ValueError):
            pass

        with self._lock:
            self._pos_meta[pos.symbol] = {
                "margin": pos.margin, "notional": pos.notional,
                "leverage": pos.effective_leverage, "side": pos.side,
                "entry_price": pos.entry_price, "current_price": price,
                "current_roi": current_roi,
                "high_24h": float(hi) if hi else None,
                "low_24h": float(lo) if lo else None,
                "range_pos_24h": range_pos,
                "pct_above_24h_low": dist_low,
                "pct_below_24h_high": dist_high,
                "opened_seen_at": self._pos_meta.get(pos.symbol, {}).get(
                    "opened_seen_at", time.time()),
                # Exchange-side fill time, captured once. opened_seen_at minus
                # this is the guardian's observation lag — the thing the age-0
                # ROI checkpoint was accidentally measuring.
                "fill_time": self._pos_meta.get(pos.symbol, {}).get(
                    "fill_time", pos.updated_at),
                "fill_price": self._pos_meta.get(pos.symbol, {}).get(
                    "fill_price", pos.entry_price),
                # Entry conditions, captured once and preserved, so closed
                # trades can later be analysed by what they were entered on.
                "entry_context": self._pos_meta.get(pos.symbol, {}).get(
                    "entry_context") or self._capture_entry_context(pos, price, range_pos),
            }

        with self._lock:
            state = self._states.get(pos.symbol)

        if state is None:
            # First sight: adopt any protective stop already resting so we
            # manage it rather than stacking a second one on top.
            orders = self.fetch_open_orders(pos.symbol)
            state = adopt_state(pos, orders, roi_pct(pos, price), self.cfg)
            self._record(pos.symbol, "adopted",
                         f"{pos.side} @ {pos.entry_price} | "
                         f"existing stop: {state.stop_roi is not None}")

        prev_order_id = state.stop_order_id
        prev_stop_roi = state.stop_roi
        was_armed = state.armed
        # Instrumentation for a fail-fast rule: when the position first turned
        # positive, and where it stood at fixed ages. Recorded only — nothing
        # acts on these yet, but without them any cutoff would be chosen blind.
        self._note_progress(pos, state, current_roi)

        if self._should_fail_fast(pos, state, current_roi):
            # Mark it before closing: _record_closed_trade reads meta, and the
            # position may be gone by the next cycle.
            self._pos_meta.setdefault(pos.symbol, {})["exit_reason"] = "fail_fast"
            log.warning(
                f"FAIL-FAST {pos.symbol}: no positive peak after "
                f"{self.cfg.fail_fast_s:.0f}s and ROI {current_roi:+.1f}% — "
                f"closing at market rather than waiting for the stop.")
            self._record(pos.symbol, "fail_fast",
                         f"peak {state.peak_roi:+.1f}%, ROI {current_roi:+.1f}%")
            try:
                self.close_position(pos.symbol)
            except Exception as e:
                log.warning(f"{pos.symbol}: fail-fast close failed: {e}")
            return

        _stop_roi_used = self._cap_stop_to_budget(pos, self.effective_stop_roi(pos))
        stop_roi_used = _stop_roi_used
        self._check_risk_invariant(pos, _stop_roi_used)
        _peak_before = state.peak_roi
        state, stop_price, reason = evaluate(
            pos, price, state, self.cfg,
            initial_stop_override=_stop_roi_used)

        # peak_roi is MONOTONIC, so one bad price pins it for the life of the
        # position and every later decision is taken against it. 牛来 reached a
        # recorded peak of +96.5% while actually 1.4% down. Record the price
        # and the field it came from at the moment the peak moves, so a phantom
        # can be traced to its source instead of inferred from order history.
        if state.peak_roi > _peak_before + 0.01:
            log.info(
                f"PEAK {pos.symbol}: {_peak_before:+.1f}% -> "
                f"{state.peak_roi:+.1f}% ROI | price={price} "
                f"src={getattr(self, '_last_price_source', '?')} "
                f"entry={pos.entry_price} lev={pos.effective_leverage:.1f}x")

        # ── Armed phase: Binance owns the trail ──────────────────────────────
        # Once a native trailing stop is resting the exchange tracks the peak
        # continuously, so the guardian must NOT keep repositioning stops — it
        # only watches. This is what removes the polling gap.
        # Before anything else: once the position has been far enough ahead it
        # gets a hard floor that nothing below cancels.
        self._ensure_adaptive_trail(pos, state, _stop_roi_used)
        self._arm_at_entry(pos, state)
        self._ensure_profit_floor(pos, state, price)
        self._audit_protection(pos, state)

        # A trail armed AT ENTRY has not replaced anything yet, so the initial
        # fixed stop still has to be placed. This early return was written for
        # the other order of events — stop first, trail later, trail supersedes
        # stop — and arm-at-entry inverts it. Taken on the first cycle it skips
        # the stop placement below entirely, leaving the position with two
        # trails and no fixed stop.
        #
        # Observed on 龙虾 2026-09-19 03:25:08: the SIZED handoff line appears,
        # both trails are placed, and the "initial protective stop at -10.3%
        # ROI" line never does.
        #
        # `armed_replaced_stop` is set only where the trail genuinely takes a
        # fixed stop's place, so the guard now means what it always intended:
        # the trail owns protection BECAUSE it replaced the stop.
        # BINANCE OWNS THE TRAIL once one is resting AND a fixed stop already
        # exists. The second clause is what keeps v3.56.0's fix intact: the
        # initial protective stop still has to be placed on the first cycle,
        # because a trail armed AT ENTRY has replaced nothing yet.
        #
        # After that, repositioning is pointless and costly. Measured on LSK
        # live 2026-09-20: the ratchet fired at +6.64% ROI after FIVE
        # cancel/replace cycles; a single native trail at the same 3% ROI
        # callback would have exited at +6.89% — 0.025% apart, with ONE order
        # and no polling. Each ratchet step is also a window where the old
        # stop is cancelled and the new one is unconfirmed, and a placement
        # can return an id without resting (牛来 2026-09-20).
        #
        # The ratchet predates working native trails. It was the only trailing
        # the bot had; it is now a third answer to a question the exchange
        # already answers tick-by-tick.
        _trail_owns = bool(state.native_trail_id) and (
            getattr(state, "armed_replaced_stop", False)
            or bool(state.stop_order_id))
        if not getattr(self.cfg, "ratchet_enabled", False) and _trail_owns:
            # Binance owns the trail, so no repositioning.
            #
            # The supersede sweep runs ONLY when the trail actually replaced a
            # fixed stop. Its purpose is to retry a cancel that FAILED at
            # arming, so a stale stop does not sit at a level the trade has
            # long left behind.
            #
            # Under arm-at-entry the trail replaced NOTHING: the fixed stop is
            # live protection that is meant to coexist with it. Sweeping there
            # cancelled the fixed stop while state.stop_order_id still named
            # it — the position lost the stop AND the record disagreed with
            # the exchange. Caught by running the cycle, not by the suite.
            if getattr(state, "armed_replaced_stop", False):
                self._cancel_superseded_stops(pos, keep=state.native_trail_id,
                                              state=state)
            with self._lock:
                self._states[pos.symbol] = state
            return

        # ── Transition: arm the native trail, replacing the fixed stop ───────
        # Arm whenever the position IS armed and no trail is resting yet — not
        # only on the transition. A position opened by a trailing-stop ENTRY can
        # be deep in profit the first time the guardian sees it, in which case
        # it is born armed and the transition never fires. That left it falling
        # through to the fixed stop, which the exchange rejects as "would
        # trigger immediately", looping UNPROTECTED.
        if self.cfg.use_native_trail and state.armed and not state.native_trail_id:
            trail_id = None
            try:
                trail_id = self._place_native_trail(pos)
            except Exception as e:
                log.error(f"FAILED to arm native trail for {pos.symbol}: {e} — "
                          f"keeping the fixed stop")
                self._record(pos.symbol, "trail_failed", str(e))

            # GATE THE CANCELS, NOT THE RECORD.
            #
            # The response cannot tell us whether the order took: status is
            # empty on every algo reply and rejection arrives afterwards.
            # 牛来 2026-09-20 06:55:37 came back with an id, was logged as
            # protection, and Binance rejected it — one second later the
            # adaptive trail was cancelled as "superseded" in favour of
            # nothing, and the position held no trail for 27 seconds.
            #
            # But the book can also simply LAG a placement by a cycle, and
            # discarding the id then would mean arming never sticks. So the
            # trail stays recorded either way — a phantom id is caught by the
            # audit — and only the CANCELS wait for positive confirmation.
            # Unconfirmed means keep what is already protecting the position
            # and try the supersede again next cycle.
            superseded_ok = bool(trail_id) and self._confirm_resting(
                pos, trail_id, "ARMED trailing stop")
            if trail_id and not superseded_ok:
                log.warning(
                    f"{pos.symbol}: the armed trail is not visible in the algo "
                    f"book yet — KEEPING the adaptive trail and the fixed stop "
                    f"until it is. No protection is removed on an unconfirmed "
                    f"placement.")

            if trail_id:
                # RECORDED regardless of confirmation. A phantom id is caught
                # by the audit; discarding a real one because the book lagged
                # a cycle would mean arming never sticks at all.
                state.native_trail_id = trail_id
                self._all_stop_ids.setdefault(pos.symbol, [])
                if trail_id not in self._all_stop_ids[pos.symbol]:
                    self._all_stop_ids[pos.symbol].append(trail_id)

            if trail_id and superseded_ok:
                # The armed trail SUPERSEDES the adaptive one — it is tighter
                # (GUARD_TRAIL_CALLBACK_ROI, typically 0.15% of price, against
                # the adaptive trail's stop-distance ~1.5%), so the wide one
                # can never fire first and would only rest as a duplicate.
                # Cancel it here rather than leaving it to the orphan sweep.
                if state.adaptive_trail_id:
                    if self._cancel_stop(pos, state.adaptive_trail_id):
                        self._all_stop_ids[pos.symbol] = [
                            i for i in self._all_stop_ids.get(pos.symbol, [])
                            if i != state.adaptive_trail_id]
                        log.info(
                            f"{pos.symbol}: adaptive trail superseded by the "
                            f"armed trail — cancelled, one trail resting.")
                    state.adaptive_trail_id = None
                # Place-then-cancel, same as everywhere else: the trail is
                # resting before the fixed stop is removed.
                if prev_order_id:
                    if self._cancel_stop(pos, prev_order_id):
                        # Drop it from the sweep list so it is not re-cancelled
                        # on every subsequent cycle.
                        self._all_stop_ids[pos.symbol] = [
                            i for i in self._all_stop_ids.get(pos.symbol, [])
                            if i != prev_order_id]
                    else:
                        # Reached only for a genuine failure: an order that is
                        # already gone now reports success (-2011 handling in
                        # _cancel_stop), so this really does mean still resting.
                        log.error(
                            f"{pos.symbol}: armed the trail but could NOT cancel "
                            f"the fixed stop {prev_order_id} — it is still "
                            f"resting and will be retried each cycle.")
                        self._record(pos.symbol, "stop_cancel_failed",
                                     f"id={prev_order_id} left resting at arming")
                else:
                    log.warning(
                        f"{pos.symbol}: armed with NO fixed stop id on record "
                        f"(state lost, or the order listing could not see it). "
                        f"Any existing stop is untracked.")
                state.stop_order_id = None
                # The trail has now REPLACED a fixed stop, which is what lets
                # manage_position hand protection over to it. Set ONLY on the
                # confirmed path: an unconfirmed trail has replaced nothing,
                # and claiming otherwise would suppress the fixed stop.
                state.armed_replaced_stop = True
                # Report the callbackRate actually sent, not the raw config
                # value — printing trail_callback_pct made a correctly-placed
                # 0.25% trail look like a 1.0% one.
                self._record(pos.symbol, "trail_armed",
                             f"native trailing stop, callback "
                             f"{trail_callback_price_pct(pos.effective_leverage, self.cfg):.2f}% price "
                             f"({callback_roi_at(pos.effective_leverage, self.cfg):.0f}% ROI)")
                with self._lock:
                    self._states[pos.symbol] = state
                return
            # Falling through means the trail could not be armed (e.g. the
            # lock-in check failed) — carry on managing the fixed stop.

        if stop_price is not None:
            stop_price = float(self.exchange.price_to_precision(pos.symbol, stop_price))
            # PLACE-THEN-CANCEL: never leave the position unprotected.
            new_id = None
            try:
                new_id = self._place_stop(pos, stop_price)
                state.unprotected_reason = None
            except Exception as e:
                msg = str(e)
                if "-2021" in msg or "immediately trigger" in msg.lower():
                    # The position is already worse than its stop level, so the
                    # stop cannot be placed at all. This is NOT a transient
                    # error: the position is unprotected until the operator acts.
                    # Deliberately not auto-closing — that is the operator's call.
                    cur = roi_pct(pos, price)
                    # Log what the exchange actually said, with the numbers
                    # needed to explain it. Without these a rejection at -2.0%
                    # ROI on a position whose stop was 28 points away could not
                    # be diagnosed at all.
                    log.error(
                        f"{pos.symbol}: stop REFUSED by the exchange — "
                        f"stop_price={stop_price} mark={price} "
                        f"side={pos.side} entry={pos.entry_price} "
                        f"roi={cur:+.1f}% intended_stop={-abs(stop_roi_used):+.1f}% ROI "
                        f"| exchange said: {msg}")
                    # NOTE: the comment that used to sit here assumed a refused
                    # stop meant the position was IN PROFIT. LSK 23:07 was
                    # refused at -2.0% ROI, so that is false. Fall back to
                    # a trailing stop rather than leaving it naked.
                    # Which side of the stop are we on? Past it in the WINNING
                    # direction means gains to protect — trail. Past it in the
                    # LOSING direction means the stop level has been breached
                    # and the position should already be closed.
                    breached = cur <= -abs(stop_roi_used)
                    if breached:
                        log.error(
                            f"{pos.symbol}: already at {cur:+.1f}% ROI, past its "
                            f"{-abs(stop_roi_used):+.1f}% stop — the stop level has "
                            f"been breached."
                        )
                        if self.cfg.close_if_past_stop:
                            self._pos_meta.setdefault(
                                pos.symbol, {})["exit_reason"] = "past_stop"
                            res = self.close_position(pos.symbol)
                            self._record(
                                pos.symbol, "closed_past_stop",
                                f"discovered at {cur:+.1f}% ROI, past the "
                                f"{-abs(stop_roi_used):+.1f}% stop — closed "
                                f"({'ok' if res.get('ok') else res.get('error')})")
                            state.unprotected_reason = None
                            with self._lock:
                                self._states[pos.symbol] = state
                            return
                        state.unprotected_reason = (
                            f"at {cur:+.1f}% ROI, already past its "
                            f"{-abs(stop_roi_used):+.1f}% stop and NOT closed "
                            f"(close_if_past_stop is off)")
                        self._record(pos.symbol, "UNPROTECTED",
                                     state.unprotected_reason)
                        with self._lock:
                            self._states[pos.symbol] = state
                        return

                    trail_id = None
                    if state.adaptive_trail_id:
                        # The adaptive trail is already resting at this
                        # position's own stop distance, placed at adoption and
                        # managed by the exchange. A refused fixed stop is then
                        # not an emergency — there is no unprotected window to
                        # rescue from, and a second trail would be a duplicate.
                        log.info(
                            f"{pos.symbol}: fixed stop refused, but the "
                            f"adaptive trail is already resting — no rescue "
                            f"needed, no second trail placed.")
                        state.unprotected_reason = None
                        with self._lock:
                            self._states[pos.symbol] = state
                        return
                    if self.cfg.use_native_trail and not state.native_trail_id:
                        try:
                            # rescue=True: this is the only protection this
                            # position will get, so it must cap the loss rather
                            # than try to preserve a gain that does not exist.
                            trail_id = self._place_native_trail(pos, rescue=True)
                        except Exception as te:
                            log.error(f"{pos.symbol}: trailing fallback failed: {te}")
                    if trail_id:
                        state.native_trail_id = trail_id
                        state.stop_order_id = None
                        state.armed = True
                        state.unprotected_reason = None
                        log.warning(
                            f"{pos.symbol}: fixed stop rejected at {cur:+.1f}% ROI "
                            f"— protected with a RESCUE trailing stop instead "
                            f"(caps further loss; not a profit lock)."
                        )
                        self._record(pos.symbol, "trail_fallback",
                                     f"fixed stop rejected at {cur:+.1f}% ROI; "
                                     f"trailing stop placed")
                        with self._lock:
                            self._states[pos.symbol] = state
                        return

                    # Report the stop ACTUALLY used. Printing the config value
                    # made a message read "at -5.7% ROI, past the -10% stop",
                    # which is self-contradictory and hid that the stop had been
                    # tightened by the budget cap.
                    state.unprotected_reason = (
                        f"already at {cur:+.1f}% ROI, past the "
                        f"{-abs(stop_roi_used):+.1f}% stop (config default "
                        f"{-self.cfg.initial_stop_roi:+.0f}%) — exchange rejected "
                        f"the stop, and the trailing fallback also failed"
                    )
                    log.error(
                        f"{pos.symbol}: UNPROTECTED — {state.unprotected_reason}. "
                        f"Close it or accept the risk; the guardian will not "
                        f"close a position on its own."
                    )
                    self._record(pos.symbol, "UNPROTECTED", state.unprotected_reason)
                else:
                    log.error(f"FAILED to place stop for {pos.symbol}: {e} — "
                              f"leaving the existing stop in place")
                    self._record(pos.symbol, "place_failed", msg)
                # evaluate() optimistically recorded the new stop level before
                # the order was sent. Roll it back so state reflects what is
                # ACTUALLY resting — otherwise should_replace_stop sees the
                # level as already achieved and never retries, leaving the
                # position unprotected indefinitely.
                state.stop_roi = prev_stop_roi
                with self._lock:
                    self._states[pos.symbol] = state
                return   # keep old state/order; do not cancel anything

            if prev_order_id and new_id:
                self._cancel_stop(pos, prev_order_id)

            if new_id:
                self._all_stop_ids.setdefault(pos.symbol, [])
                if new_id not in self._all_stop_ids[pos.symbol]:
                    self._all_stop_ids[pos.symbol].append(new_id)
            state.stop_order_id = new_id
            # Sweep any earlier stops whose cancel did not take. Harmless
            # individually, but they accumulate and can fire against a LATER
            # position on the same symbol.
            self._cancel_superseded_stops(pos, keep=new_id, state=state)
            log.info(f"{pos.symbol}: {reason} | ROI now {roi_pct(pos, price):+.1f}% "
                     f"| stop @ {stop_price}")
            self._record(pos.symbol, "stop_set",
                         f"{reason} (stop {state.stop_roi:+.1f}% ROI @ {stop_price})")

        with self._lock:
            self._states[pos.symbol] = state

    # ── cycle ───────────────────────────────────────────────────────────────

    def load_state(self, path: str):
        """Restore state from a previous run. Best-effort; never blocks startup."""
        from . import futures_state
        self.state_path = path
        data = futures_state.load(path, owner=self.state_owner)
        if not data:
            return
        restored = futures_state.restore_states(data)
        with self._lock:
            self._states.update(restored)
            for sym, m in (data.get("pos_meta") or {}).items():
                meta = self._pos_meta.setdefault(sym, {})
                if m.get("entry_context"):
                    meta["entry_context"] = m["entry_context"]
                # Restore anything the close record needs. manage_position
                # overwrites these within a poll, so they only matter for a
                # position that closes before the first pass after a restart —
                # which previously produced a row with no side, entry or ROI.
                for k in ("opened_seen_at", "side", "entry_price", "margin",
                          "leverage", "current_roi", "current_price",
                          "fill_time", "fill_price"):
                    if m.get(k) is not None:
                        meta[k] = m[k]
            if not self._closed_trades:
                # The journal is the source of truth. The state file is read
                # only to MIGRATE trades written before the journal existed —
                # once, because import_existing refuses a non-empty journal.
                jr = getattr(self, "_journal", None)
                from_state = list(data.get("closed_trades") or [])
                if jr is not None:
                    if from_state:
                        jr.import_existing(from_state)
                    self._closed_trades = jr.load(limit=self.max_closed_trades)
                    if not self._closed_trades and from_state:
                        self._closed_trades = from_state
                else:
                    self._closed_trades = from_state
        self._restored_safety = data.get("safety") or {}
        self._restored_placed_orders = data.get("placed_orders") or {}
        try:
            self._asset_basis = dict(data.get("asset_basis") or {})
        except Exception:
            self._asset_basis = {}
        # A baseline written before v3.61.1 is in USDT-only units. Comparing it
        # against an account-value wallet_now invents a gain of exactly the
        # reserve, so it is migrated once rather than left to mislead.
        self._wallet_start_basis = str(data.get("wallet_start_basis") or "")
        if data.get("wallet_start") and self.wallet_start is None:
            self.wallet_start = float(data["wallet_start"])
        with self._lock:
            for sym, ids in (data.get("stop_ids") or {}).items():
                cur = self._all_stop_ids.setdefault(sym, [])
                for oid in ids:
                    if oid not in cur:
                        cur.append(oid)
        with self._lock:
            for sym, ids in (data.get("pending_cancels") or {}).items():
                q = self._pending_cancels.setdefault(sym, [])
                for oid in ids:
                    if oid not in q:
                        q.append(oid)
        if data.get("pending_cancels"):
            n = sum(len(v) for v in data["pending_cancels"].values())
            log.warning(f"restored {n} order(s) still awaiting cancellation "
                        f"from a previous run — they will be retried")
        if data.get("stop_ids"):
            log.info(f"restored {sum(len(v) for v in data['stop_ids'].values())} "
                     f"tracked stop id(s) across "
                     f"{len(data['stop_ids'])} symbol(s)")
        for sym, st in restored.items():
            log.info(f"{sym}: restored peak {st.peak_roi:+.1f}% ROI, "
                     f"armed={st.armed}, trail={bool(st.native_trail_id)}")

    def verify_state_path(self) -> bool:
        """
        Confirm the state file can actually be written.

        Writes a throwaway probe file beside the real one. The previous version
        proved writability by SAVING EMPTY STATE to the real path — and ran
        after load_state, so every startup read the file and then immediately
        overwrote it with nothing. Restored trades, position records and the
        daily-loss baseline were destroyed on disk each boot.
        """
        if not self.state_path:
            log.warning("Futures state persistence DISABLED (no path set) — "
                        "sized stops and the daily-loss baseline will not "
                        "survive a restart.")
            return False
        import os
        probe = f"{self.state_path}.probe"
        try:
            os.makedirs(os.path.dirname(probe) or ".", exist_ok=True)
            with open(probe, "w") as fh:
                fh.write("ok")
            os.unlink(probe)
            log.info(f"Futures state persistence OK -> {self.state_path}")
            return True
        except Exception as e:
            log.error(
                f"Futures state NOT writable at {self.state_path}: {e} — sized "
                f"stops and the daily-loss baseline will be lost on restart. "
                f"Check the volume mount is writable.")
            return False

    def attach_journal(self, path: str, max_bytes: int | None = None,
                       keep_archives: int | None = None):
        """
        Give the guardian an append-only trade journal.

        Without one it falls back to keeping trades in the state file, which
        still works — it is just the arrangement that rewrites 11 MB every
        2.5 seconds and drops history past the cap.
        """
        try:
            from bot.trade_journal import (TradeJournal, DEFAULT_MAX_BYTES,
                                           DEFAULT_KEEP_ARCHIVES)
            self._journal = TradeJournal(
                path,
                max_bytes=max_bytes or DEFAULT_MAX_BYTES,
                keep_archives=(DEFAULT_KEEP_ARCHIVES if keep_archives is None
                               else keep_archives))
            log.info(f"Trade journal at {path} — trades are appended once each "
                     f"instead of rewritten with the state file every cycle.")
        except Exception as e:
            log.error(f"trade journal unavailable ({e}); falling back to the "
                      f"state file")
            self._journal = None

    def journal_stats(self) -> dict:
        jr = getattr(self, "_journal", None)
        return jr.stats() if jr is not None else {}

    def save_state(self):
        if not self.state_path:
            return
        from . import futures_state
        with self._lock:
            states = dict(self._states)
            meta = dict(self._pos_meta)
            trades = list(self._closed_trades)
        entry = getattr(self, "_entry_service", None)
        placed = entry.export_placed_orders() if entry is not None else {}
        with self._lock:
            stop_ids = {k: list(v) for k, v in self._all_stop_ids.items()}
            pending = {k: list(v) for k, v in self._pending_cancels.items()}
        # Trades go to the journal, so they are NOT rewritten here. At the old
        # 5000-trade cap this call serialised 11 MB every 2.5 seconds — 383 GB
        # of disk writes a day to persist a few kilobytes of changed state.
        if getattr(self, "_journal", None) is not None:
            trades = []
        futures_state.save(self.state_path, states=states, pos_meta=meta,
                           closed_trades=trades, owner=self.state_owner,
                           placed_orders=placed, stop_ids=stop_ids,
                           pending_cancels=pending,
                           wallet_start=self.wallet_start,
                           asset_basis=getattr(self, "_asset_basis", {}),
                           wallet_start_basis=getattr(
                               self, "_wallet_start_basis", ""),
                           safety=getattr(self, "_safety_snapshot", lambda: {})())

    def run_cycle(self):
        try:
            positions = self.fetch_positions()
        except Exception as e:
            self._last_error = f"fetch_positions failed: {e}"
            log.warning(self._last_error)
            return

        try:
            b = self.exchange.fetch_balance()
            # Keep the raw payload: account_value() needs the assets[] array,
            # and re-fetching it would double the balance calls.
            self._last_balance_payload = b
            val, source = resolve_usdt_balance(b)
            if val <= 0:
                log.warning("Could not resolve a positive USDT futures balance")
            elif abs(val - self._wallet_balance_cached) > 0.01:
                log.info(f"Futures wallet balance {val:.2f} USDT (from {source})")
            # A single reading must not be able to scale position size. A jump
            # beyond this fraction between polls is treated as suspect: the
            # last good value is kept and the reading is logged rather than
            # acted on. Real deposits and withdrawals settle after
            # BALANCE_JUMP_CONFIRMATIONS consecutive readings agree.
            prev = self._wallet_balance_cached
            if prev and val > 0:
                ratio = val / prev
                if ratio > (1 + self.BALANCE_JUMP_TOLERANCE) or ratio < (1 - self.BALANCE_JUMP_TOLERANCE):
                    self._balance_outliers += 1
                    if self._balance_outliers < self.BALANCE_JUMP_CONFIRMATIONS:
                        log.warning(
                            f"Balance reading {val:.2f} is {(ratio - 1) * 100:+.1f}% "
                            f"from {prev:.2f} ({source}) — IGNORED for sizing "
                            f"({self._balance_outliers}/{self.BALANCE_JUMP_CONFIRMATIONS}). "
                            f"A bad reading would scale every new position.")
                        val = prev
                    else:
                        log.warning(
                            f"Balance {val:.2f} confirmed after "
                            f"{self._balance_outliers} readings — accepting.")
                        self._balance_outliers = 0
                else:
                    self._balance_outliers = 0
            self._wallet_balance_cached = val
            if (self.wallet_start and val > 0
                    and getattr(self, "_wallet_start_basis", "") != "account"):
                # One-time migration. The stored baseline is USDT-only; the
                # cards now compare it against USDT + reserve, which invents a
                # gain of exactly the reserve. Add the reserve once.
                try:
                    res = float(self.account_value().get("reserve") or 0.0)
                except Exception:
                    res = 0.0
                if res:
                    prev = self.wallet_start
                    self.wallet_start = float(prev) + res
                    log.warning(
                        f"Migrated the reconciliation baseline to account-value "
                        f"units: {prev:.2f} -> {self.wallet_start:.2f} "
                        f"(+{res:.2f} fee reserve). Without this the account "
                        f"return read high by exactly the reserve.")
                self._wallet_start_basis = "account"
                self.save_state()
            if self.wallet_start is None and val > 0:
                # ACCOUNT-VALUE units, matching what the cards read. Recording
                # it in USDT-only units while wallet_now included the reserve
                # produced a phantom gain of exactly the reserve: live showed
                # ACCOUNT RETURN +3.01% and gap +$2.86 on a $2.86 BNB balance.
                av = self.account_value().get("value") or val
                self.wallet_start = float(av)
                self._wallet_start_basis = "account"
                log.info(f"Reconciliation baseline: account value "
                         f"{av:.2f} USDT (wallet {val:.2f} + reserve "
                         f"{av - val:.2f})")
        except Exception as e:
            log.debug(f"balance fetch failed: {e}")

        live_symbols = {p.symbol for p in positions}

        # BACKSTOP: is there a position on the exchange we are not guarding?
        #
        # Every other consistency check is internal — they compare the bot's
        # view against itself. This is the only one that asks the exchange.
        # An unguarded position is the worst failure the system has (one ran
        # to +70% ROI untracked, unprotected, and absent from the history),
        # and it can arrive by several routes: a filled order mistaken for a
        # cancelled one, a row dropped in normalisation, or a restart that
        # lost state. This catches all of them regardless of cause.
        with self._lock:
            tracked = set(self._states)
        untracked = live_symbols - tracked
        for sym in sorted(untracked):
            if self._untracked_reported.get(sym):
                continue
            self._untracked_reported[sym] = True
            log.error(
                f"UNTRACKED POSITION {sym} is open on the exchange but the "
                f"guardian has no state for it — it is unprotected. Adopting "
                f"it now and placing a stop.")
            self._record(sym, "untracked_adopted", "found open but not guarded")
        for sym in list(self._untracked_reported):
            if sym not in live_symbols:
                self._untracked_reported.pop(sym, None)
        with self._lock:
            tracked = len(self._states)
        if tracked and not positions:
            # Every tracked position vanishing at once is far more likely to be
            # an API hiccup than a simultaneous close.
            log.warning(
                f"fetch_positions returned NO positions while tracking {tracked} "
                f"— treating as a transient reply, not a mass close.")

        # Forget state for positions that have closed (stopped out or closed by
        # the operator) so a future position on the same symbol starts fresh.
        with self._lock:
            # A symbol missing from ONE fetch_positions response is not proof it
            # closed. A transient or partial reply used to trigger the full
            # close path: a phantom trade recorded, state wiped, and the
            # protective stop cancelled as orphaned — leaving a position that
            # was still open now unprotected. One position was recorded closed
            # twice while its loss grew from 141 to 240 USDT. Require the
            # absence to repeat before believing it.
            gone = []
            for sym in list(self._states):
                if sym in live_symbols:
                    self._missing_counts.pop(sym, None)
                    continue
                n = self._missing_counts.get(sym, 0) + 1
                self._missing_counts[sym] = n
                if n < self.MISSING_CONFIRMATIONS:
                    log.warning(
                        f"{sym}: absent from the position list "
                        f"({n}/{self.MISSING_CONFIRMATIONS}) — holding state and "
                        f"its stop until confirmed. A transient reply must not "
                        f"look like a close."
                    )
                    continue
                gone.append((sym, self._states[sym], self._pos_meta.get(sym, {})))
                self._missing_counts.pop(sym, None)
            for sym, st, meta in gone:
                self._record_closed_trade(sym, st, meta)
                # Drop the floor now rather than leaving it to the 120s sweep.
                self._cancel_profit_floor(sym, st)
                if getattr(st, "adaptive_trail_id", None):
                    try:
                        self._cancel_stop(
                            type("P", (), {"symbol": sym})(),
                            st.adaptive_trail_id)
                    except Exception:
                        pass
                    st.adaptive_trail_id = None
                log.info(f"{sym}: position gone — clearing guard state")
                self._record(sym, "closed", "position no longer open")
                # Tell whoever is trading that this symbol closed. The
                # auto-trader's entire post-loss cooldown chain hung off
                # note_closed_trade() and NOTHING called it: record_loss never
                # ran, symbol_blocked_until was never set, the 30-minute
                # cooldown never applied, and max_reentries_per_symbol never
                # bound. AKE was re-entered seven times in one day for -$94.98,
                # four of them within 8-21 minutes of the previous close.
                cb = getattr(self, "on_position_closed", None)
                if cb:
                    try:
                        # The realised figure is computed INSIDE
                        # _record_closed_trade, not carried on meta, so read it
                        # from the record that call just produced.
                        rec = getattr(self, "_last_closed_rec", None) or {}
                        if rec.get("symbol") != sym:
                            rec = {}
                        cb(sym,
                           rec.get("realised_pnl_usdt"),
                           (rec.get("entry_context") or {}).get("rsi"))
                    except Exception as e:
                        log.warning(f"{sym}: close callback failed: "
                                    f"{_safe_err(e)}")
                del self._states[sym]
                self._pos_meta.pop(sym, None)
                self._capped_stop_reported.pop(sym, None)
                self._stop_source_reported.discard(sym)
                self._missing_counts.pop(sym, None)

        # Cancel any protective stop left resting after the position closed.
        # An orphaned reduce-only stop is not harmless: if a NEW position is
        # later opened on the same symbol, that stale order can trigger against
        # it at a level chosen for the old trade. Done outside the state lock
        # because it makes network calls.
        for sym, st, _meta in gone:
            ids = set(self._all_stop_ids.pop(sym, []) or [])
            if st.stop_order_id:
                ids.add(st.stop_order_id)
            if st.native_trail_id:
                ids.add(st.native_trail_id)
            for oid in ids:
                if not self._cancel_orphan_stop(sym, oid):
                    # Keep it queued: the position is gone but the order may
                    # not be, and nothing else will ever discover it.
                    self._queue_pending_cancel(sym, oid)

        for pos in positions:
            try:
                self.manage_position(pos)
            except Exception as e:
                log.warning(f"manage_position failed for {pos.symbol}: {e}")

        # Reap entry orders this bot placed that should no longer rest: stale
        # unfilled ones (which block their symbol and would open a position
        # sized for conditions that have passed), and any left over on a symbol
        # that now has a position (which would add to it).
        entry = getattr(self, "_entry_service", None)
        if entry is not None:
            try:
                entry.reap_stale_entry_orders(self.entry_order_ttl_s, live_symbols)
            except Exception as e:
                log.warning(f"entry-order reap failed: {e}")

            # Sweep for orders the bot has no record of. Anything placed before
            # tracking existed rests indefinitely and can fill hours later.
            sweep_due = self._untracked_due()
            if self._pending_cancels:
                try:
                    self.drain_pending_cancels()
                except Exception as e:
                    log.warning(f"pending-cancel drain failed: {e}")
            if self.sweep_orphan_stops and sweep_due:
                try:
                    self.reap_orphan_stops()
                except Exception as e:
                    log.warning(f"orphan-stop sweep failed: {e}")

            if self.reap_untracked and sweep_due:
                try:
                    entry.reap_untracked_entry_orders(
                        self.entry_order_ttl_s, self._reap_scan_symbols())
                except Exception as e:
                    log.warning(f"untracked reap failed: {e}")

        self._last_cycle_ts = time.time()
        self._last_error = None
        self.save_state()

    def _record_closed_trade(self, symbol: str, state: GuardState, meta: dict):
        """
        Log a position that has disappeared from the account.

        The exit price is the LAST OBSERVED mark price, not the actual fill —
        the guardian polls, so a stop that triggered between cycles filled at a
        price it never saw. Binance's realised PnL is fetched where available
        and preferred; otherwise the figures here are an approximation and are
        labelled as such.
        """
        entry = meta.get("entry_price")
        last = meta.get("current_price")
        side = meta.get("side")
        notional = meta.get("notional") or 0.0

        # Compute the result from our own record first. This is direction-aware
        # and cannot come out sign-inverted: a short profits when price falls.
        computed = None
        if entry and last and side and notional:
            move = ((last - entry) / entry) if side == "long" else ((entry - last) / entry)
            computed = move * notional

        # Binance's realizedPnl is the exchange's own figure, so prefer
        # it — but only when it AGREES IN SIGN with the direction-aware figure.
        # A short that closed in profit was being reported as a loss because the
        # exchange value was taken on trust.
        realised = None
        pnl_source = "none"
        ledger_pnl = None
        # How much of the position the fills we can see actually account for.
        # Declared HERE, not inside the try below, because a fill-lookup
        # failure must leave these false rather than undefined — the ordering
        # test reads them either way.
        fills_cover = None
        fills_complete = False
        # Reset per close. This is instance state, and leaving it set meant a
        # trade whose fill lookup failed inherited the PREVIOUS trade's
        # commission — a wrong number that looks entirely plausible.
        self._last_trade_fees = None
        try:
            # Scope the fills to THIS position's lifetime. fetch_my_trades
            # returns the last N fills for the symbol regardless of which
            # position they belonged to, so summing them blindly gave every
            # trade on a symbol the SAME realised figure — two UAI trades with
            # +24% and -5% ROI both reported an identical -1.4133.
            opened_at = meta.get("opened_seen_at")
            since_ms = int(opened_at * 1000) if opened_at else None
            trades = self.exchange.fetch_my_trades(symbol, since=since_ms, limit=50)

            scoped = []
            for t in trades:
                ts = t.get("timestamp")
                if since_ms and ts and ts < since_ms:
                    continue        # belongs to an earlier position
                scoped.append(t)

            pnl = sum(float((t.get("info") or {}).get("realizedPnl") or 0)
                      for t in scoped)
            # Binance's realizedPnl EXCLUDES commission — that is a separate
            # field on the same fill. Summing it alone gives a GROSS figure,
            # which is why a positive total P&L can sit alongside a falling
            # wallet balance. Capture the commission so net is reportable.
            fees = 0.0
            for t in scoped:
                info = t.get("info") or {}
                try:
                    fees += abs(float(info.get("commission") or
                                      (t.get("fee") or {}).get("cost") or 0))
                except (TypeError, ValueError):
                    pass
            self._last_trade_fees = round(fees, 6)

            # Do these fills account for the WHOLE position? A close that
            # arrives in several parts, or a query that reads too early, gives
            # a partial sum — and a partial sum is no more trustworthy than a
            # partial ledger. Coverage is what separates "authoritative" from
            # "as much as had landed when we looked".
            try:
                close_side = "buy" if side == "short" else "sell"
                closed_qty = sum(
                    float(t.get("amount") or 0) for t in scoped
                    if str(t.get("side") or "").lower() == close_side)
                pos_qty = 0.0
                if entry and entry > 0:
                    pos_qty = float(meta.get("notional") or 0) / entry
                if pos_qty > 0:
                    fills_cover = closed_qty / pos_qty
                    fills_complete = fills_cover >= 0.99
            except Exception as e:
                log.debug(f"{symbol}: fill coverage check failed: {e}")

            if scoped and not pnl:
                log.debug(f"{symbol}: {len(scoped)} fill(s) in scope, all zero realisedPnl")
            if pnl:
                if computed is None or (pnl >= 0) == (computed >= 0):
                    realised = pnl
                    pnl_source = "fills"
                else:
                    log.warning(
                        f"{symbol}: exchange realisedPnl {pnl:+.4f} disagrees in sign "
                        f"with the {side} result computed from entry/exit "
                        f"({computed:+.4f}) — using the computed value."
                    )
                    realised = computed
                    pnl_source = "computed"
        except Exception as e:
            log.debug(f"realised PnL lookup failed for {symbol}: {e}")

        # The income ledger is queried INDEPENDENTLY of the fills. It used to
        # sit inside the same try, so a fill-lookup failure skipped it — losing
        # the authoritative source exactly when the fallback was needed.
        try:
            # The window must start when the ORDER WAS PLACED, not when the
            # guardian first saw the position. The entry commission is charged
            # at the fill, which precedes the first poll — starting at
            # opened_seen_at filtered it out and captured only the EXIT side,
            # halving every fee figure and leaving the wallet unreconciled.
            ctx = meta.get("entry_context") or {}
            placed_at = ctx.get("sized_at")
            opened_at = meta.get("opened_seen_at")
            floor_ts = placed_at or opened_at
            if floor_ts:
                # A small margin in case the fill preceded the recorded stamp.
                income_since = int((float(floor_ts) - self.INCOME_LOOKBACK_PAD_S)
                                   * 1000)
            else:
                income_since = None
            led_pnl, led_comm, led_found = self._income_for_position(
                symbol, income_since)
            if led_found:
                if led_comm:
                    self._last_trade_fees = round(led_comm, 6)
                ledger_pnl = led_pnl
                log.info(f"{symbol}: ledger realised {led_pnl:+.4f}, "
                         f"fees {led_comm:.4f}")
            else:
                log.warning(
                    f"{symbol}: income ledger returned nothing — P&L and fees "
                    f"fall back to fills or a price estimate, which can be wrong.")
        except Exception as e:
            log.warning(f"{symbol}: income ledger lookup failed: {e}")

        if realised is None and computed is not None:
            realised = computed
            pnl_source = "computed"

        # Three sources, in order of what they actually are:
        #
        #   fills   — the executed trades. Authoritative WHEN COMPLETE.
        #   ledger  — the exchange's own accounting. Authoritative EVENTUALLY;
        #             the income endpoint can still be filling in when this
        #             runs, seconds after the close.
        #   computed — the exit reconstructed from the stop level. An estimate,
        #             and the one that invents money.
        #
        # a9b1d41 put the ledger above the others because a price ESTIMATE
        # invented +4.32 on a trade opened and closed at the same price. That
        # reasoning stands and is why `computed` is still last. But it also put
        # the ledger above the FILLS, and those are not an estimate — WLD
        # 2026-09-18 10:16 summed to +0.2684 from the fills, Binance's own
        # order detail reported Total PNL 0.26840000, and the wallet moved
        # +0.26. The ledger returned +0.0934, roughly a third, which is what a
        # partially-populated income query looks like on a multi-fill close.
        # Taking it cost that trade 0.175 USDT of recorded profit.
        #
        # The original bug cannot return through fills: a trade opened and
        # closed at the same price sums to zero in the fill records, correctly.
        # Only the estimate could invent the +4.32.
        if ledger_pnl is not None:
            disagrees = realised is not None and abs(realised - ledger_pnl) > 0.01
            if pnl_source == "fills" and fills_complete:
                # Complete fills outrank the ledger. Still say so — that line
                # is what surfaced this, and silencing it would hide the next
                # disagreement of a different kind.
                if disagrees:
                    log.warning(
                        f"{symbol}: fills P&L {realised:+.4f} disagrees with the "
                        f"income ledger {ledger_pnl:+.4f} — using the FILLS "
                        f"(they cover {fills_cover:.0%} of the position; the "
                        f"ledger can still be filling in). Check wallet_gap.")
            else:
                if disagrees:
                    why = ("fills covered only "
                           f"{fills_cover:.0%} of the position"
                           if fills_cover is not None and pnl_source == "fills"
                           else f"{pnl_source} is not the fill record")
                    log.warning(
                        f"{symbol}: {pnl_source} P&L {realised:+.4f} disagrees "
                        f"with the income ledger {ledger_pnl:+.4f} — using the "
                        f"ledger ({why}).")
                realised = ledger_pnl
                pnl_source = "ledger"

        # When the exchange reports realised PnL it reflects the ACTUAL fill.
        # The guardian polls, so a stop that triggered between cycles filled at
        # a price it never observed — final_roi and exit_price computed from the
        # last observed price then understate the move. Derive both from the
        # realised figure instead, which is why a stopped-out trade could show
        # -3% ROI while the money said -10%.
        margin = meta.get("margin") or 0.0
        leverage = meta.get("leverage") or 0.0
        final_roi = _r2(meta.get("current_roi"))
        exit_price = last
        exit_from_exchange = False

        if realised is not None and realised != computed and margin > 0:
            final_roi = round(realised / margin * 100.0, 2)
            if entry and leverage > 0:
                move = (final_roi / 100.0) / leverage
                exit_price = (entry * (1 + move) if side == "long"
                              else entry * (1 - move))
            exit_from_exchange = True
            observed = _r2(meta.get("current_roi"))
            if observed is not None and abs(final_roi - observed) > 1.0:
                log.info(
                    f"{symbol}: exit reconstructed from realised PnL — "
                    f"{final_roi:+.2f}% ROI (last observed was {observed:+.2f}%, "
                    f"the stop filled between polls)"
                )
            # What the protective orders SHOULD have delivered, against what
            # the wallet says. A native trail gives back its callback from the
            # peak and no more, so peak minus callback is the floor this exit
            # ought to have respected.
            #
            # ONE 17:00 UTC is why this exists: peak +31.8%, an ARMED trail
            # accepted with a 3% ROI callback, and a close at -15.18%. That is
            # 47 ROI points past where the trail could account for, and nothing
            # in the record said which order filled or at what level. Without
            # this line the next one is diagnosed the same way — from order
            # history, days later.
            try:
                st = self._states.get(symbol)
                peak = getattr(st, "peak_roi", None) if st else None
                if peak is not None and peak > 0:
                    give_back = peak - final_roi
                    cb_roi = callback_roi_at(leverage, self.cfg) if leverage else None
                    resting = {k: v for k, v in (
                        ("fixed", getattr(st, "stop_order_id", None)),
                        ("floor", getattr(st, "floor_stop_id", None)),
                        ("trail", getattr(st, "native_trail_id", None)),
                        ("adaptive", getattr(st, "adaptive_trail_id", None)),
                    ) if v}
                    line = (f"GIVE-BACK {symbol}: peak {peak:+.1f}% -> final "
                            f"{final_roi:+.2f}% = {give_back:.1f} ROI points"
                            + (f" against a {cb_roi:.0f}% callback" if cb_roi else "")
                            + f" | resting: {resting or 'nothing on record'}"
                            + f" | exit~{exit_price}")
                    # A trail cannot give back more than its callback plus the
                    # slippage on one market fill. Materially more means the
                    # trail was not doing the work, and that is worth an alarm
                    # rather than a line buried at INFO.
                    if cb_roi and give_back > cb_roi * 2 + 5:
                        log.warning(line + " — MORE THAN THE TRAIL CAN EXPLAIN")
                    else:
                        log.info(line)
            except Exception as e:
                log.debug(f"{symbol}: give-back line failed: {e}")

        rec = {
            "symbol": symbol,
            "side": meta.get("side"),
            "entry_price": entry,
            "exit_price": exit_price,
            "margin_usdt": meta.get("margin"),
            "leverage": meta.get("leverage"),
            "peak_roi": round(state.peak_roi, 2),
            "trough_roi": round(state.trough_roi, 2),
            # Seconds from first sighting to the first positive ROI. None means
            # it never went green — the case a fail-fast rule targets.
            "secs_to_first_positive": (
                round(state.first_positive_at - float(meta["opened_seen_at"]), 1)
                if state.first_positive_at and meta.get("opened_seen_at") else None),
            # ROI at first sight. NOT a property of the entry: ROI against the
            # fill price is zero at the fill, so any non-zero value here is the
            # market moving during the guardian's observation lag.
            "roi_at_first_sight": state.roi_checkpoints.get("0"),
            "roi_at_0s": state.roi_checkpoints.get("0"),   # kept for old rows
            # Seconds between the exchange filling the entry and the guardian
            # first seeing the position. The honest measure of that lag.
            "observation_lag_s": (
                round(meta["opened_seen_at"] - float(meta["fill_time"]), 2)
                if meta.get("opened_seen_at") and meta.get("fill_time")
                else None),
            "fill_price": meta.get("fill_price"),
            # Which price triggered this trade's stops. Recorded so the
            # MARK_PRICE change can be compared against CONTRACT_PRICE rather
            # than assumed to have helped.
            "stop_working_type": self.cfg.stop_working_type,
            # What waiting two candles after the signal would have done.
            # Recorded only — nothing waits.
            **self._wait_counterfactual(symbol, meta.get("side"), meta),
            "roi_at_60s": state.roi_checkpoints.get("60"),
            "roi_at_180s": state.roi_checkpoints.get("180"),
            "roi_at_300s": state.roi_checkpoints.get("300"),
            "final_roi": final_roi,
            # ROI DERIVED from the same figure as the money, so the two can
            # never disagree. Previously ROI came from entry/exit prices while
            # realised came from the ledger, and one trade reported +4.18% ROI
            # against a true result of zero P&L.
            "roi_from_realised": (
                round(realised / margin * 100, 2)
                if realised is not None and margin else None),
            "net_roi": (
                round((realised - self._last_trade_fees) / margin * 100, 2)
                if realised is not None and margin
                and self._last_trade_fees is not None else None),
            "pnl_source": pnl_source,
            # Whether the money figure came from the EXCHANGE at all. When
            # neither the income ledger nor the fills are available, the
            # fallback reconstructs the exit from a price it has to guess —
            # which produced a -75% ROI on a position whose stop was capped at
            # -30%, and a flat 0 on another. Such numbers are not merely
            # imprecise, they are invented, so they are marked and excluded
            # from every total rather than silently averaged in.
            "pnl_verified": pnl_source in ("ledger", "fills"),
            "observed_roi": _r2(meta.get("current_roi")),
            "exit_from_exchange": exit_from_exchange,
            "stop_roi": None if state.stop_roi is None else round(state.stop_roi, 2),
            "armed": state.armed,
            # The capital actually committed. Needed to compute return on
            # capital: averaging ROI percentages weights a large trade the same
            # as a small one, so one big loss can hide behind two small wins.
            "margin": round(margin, 4) if margin else None,
            "realised_pnl_usdt": None if realised is None else round(realised, 4),
            # Commission from the income ledger. GROSS realised P&L excludes
            # it, which is why a positive P&L can sit beside a falling wallet.
            "fee_source": getattr(self, "_last_fee_source", "ledger"),
            "fees_usdt": (None if self._last_trade_fees is None
                          else round(self._last_trade_fees, 4)),
            "net_pnl_usdt": (
                round(realised - self._last_trade_fees, 4)
                if realised is not None and self._last_trade_fees is not None
                else None),
            "exit_is_estimate": not exit_from_exchange,
            "entry_context": meta.get("entry_context") or {},
            # How far price DRIFTED between sizing and the fill, signed so
            # positive is always against the position. Elapsed time is only a
            # proxy for this: a calm coin can rest for many minutes and be the
            # same trade, a vertical one is a different trade in seconds.
            "drift_since_sizing_pct": self._drift_pct(meta, meta.get("side")),
            # Seconds between the entry being SIZED and the position first
            # being seen. A TRAILING_STOP_MARKET entry rests until price
            # retraces, so the conditions that sized the callback can be many
            # minutes stale by the time it fills — STORJ rested 305s while
            # price ran. Computed from fields already recorded.
            "signal_age_s": (
                round(meta["opened_seen_at"] - (meta.get("entry_context") or {})["sized_at"], 1)
                if meta.get("opened_seen_at")
                and (meta.get("entry_context") or {}).get("sized_at")
                else None),
            # An EXPLICIT reason wins. The fallback below infers from state,
            # which cannot see a market close the guardian made itself: a
            # fail-fast leaves both native_trail_id and stop_roi set, so every
            # fail-fast exit was bucketed as "stop" or "trail" and its effect
            # was unmeasurable in the exit-reason table.
            "exit_reason": (meta.get("exit_reason")
                            or ("trail" if state.native_trail_id
                                else ("stop" if state.stop_roi is not None
                                      else "unknown"))),
            "opened_at": meta.get("opened_seen_at"),
            "closed_at": time.time(),
        }
        # The journal is the durable record. Appended ONCE per trade rather
        # than rewritten with the whole state file every 2.5s cycle.
        jr = getattr(self, "_journal", None)
        if jr is not None:
            jr.append(rec)
        with self._lock:
            self._closed_trades.append(rec)
            self._last_closed_rec = rec
            # A 100-trade cap silently truncated the record: a 24-hour run kept
            # reporting exactly 100 trades while far more had closed, so every
            # split was computed on a moving window rather than the whole
            # sample. Data collection is the point, so the ceiling is now high
            # enough not to bind in practice and configurable if it ever does.
            if len(self._closed_trades) > self.max_closed_trades:
                self._closed_trades = self._closed_trades[-self.max_closed_trades:]
        log.info(
            f"CLOSED {symbol} {rec['side']} peak={rec['peak_roi']:+.1f}% ROI "
            f"final={rec['final_roi']}% realised="
            f"{'n/a' if realised is None else f'{realised:+.4f}'}"
        )

    def close_position(self, symbol: str) -> dict:
        """
        Close an open position at market with a REDUCE-ONLY order.

        Reduce-only means this can only ever shrink or flatten the position; it
        cannot open or reverse one, whatever size is passed. The resting
        protective stop is cancelled afterwards, not before, so the position is
        never left unprotected while the close is in flight.
        """
        positions = {p.symbol: p for p in self.fetch_positions()}
        pos = positions.get(symbol)
        if pos is None:
            return {"ok": False, "error": f"no open position on {symbol}"}

        side = stop_side(pos)          # the side that closes this position

        # Split across the per-order quantity cap, same as the protective stop.
        # A single oversized MARKET order is rejected with -4005, which is why
        # closing from the dashboard failed and had to be done on Binance.
        max_qty = self.market_max_qty(symbol)
        chunks: list[float] = []
        remaining = pos.qty
        if max_qty and remaining > max_qty:
            while remaining > 0 and len(chunks) < 20:
                c = float(self.exchange.amount_to_precision(symbol,
                                                            min(remaining, max_qty)))
                if c <= 0:
                    break
                chunks.append(c)
                remaining -= c
        else:
            chunks = [float(self.exchange.amount_to_precision(symbol, pos.qty))]

        qty = sum(chunks)
        if self.dry_run:
            log.info(f"[DRY RUN] would close {symbol}: {side} MARKET reduceOnly "
                     f"{qty} in {len(chunks)} order(s)")
            self._record(symbol, "close_dry_run", f"{side} MARKET {qty}")
            return {"ok": True, "dry_run": True, "symbol": symbol,
                    "side": side, "qty": qty, "message": "Dry run — no order sent."}

        ids: list[str] = []
        for c in chunks:
            try:
                order = self.exchange.create_order(
                    symbol=symbol, type="MARKET", side=side, amount=c,
                    price=None, params={"reduceOnly": True},
                )
                ids.append(str(order.get("id") or order.get("orderId") or ""))
            except Exception as e:
                log.error(f"close failed for {symbol}: {e}")
                self._record(symbol, "close_failed", str(e))
                # Report partial success honestly rather than claiming a clean
                # close — some of the position may already be flat.
                return {"ok": False, "error": str(e),
                        "partially_closed_qty": sum(chunks[:len(ids)]),
                        "order_ids": ids}

        oid = ids[0] if ids else ""
        if len(ids) > 1:
            log.warning(f"{symbol}: closed in {len(ids)} orders (per-order cap)")
        log.warning(f"CLOSED {symbol} by operator: {side} MARKET reduceOnly qty={qty} id={oid}")
        self._record(symbol, "closed_by_operator", f"{side} MARKET {qty} id={oid}")

        # Remove the now-redundant protective stop.
        with self._lock:
            st = self._states.get(symbol)
        if st and st.stop_order_id:
            self._cancel_stop(pos, st.stop_order_id)

        return {"ok": True, "dry_run": False, "symbol": symbol,
                "side": side, "qty": qty, "order_id": oid}

    def _endpoint_hint(self) -> str:
        """Best-effort description of the API host actually in use."""
        try:
            urls = getattr(self.exchange, "urls", {}) or {}
            api = urls.get("api")
            if isinstance(api, dict):
                for key in ("fapiPrivate", "fapiPublic", "future", "private"):
                    if api.get(key):
                        return str(api[key])
                return str(next(iter(api.values()), ""))
            return str(api or "")
        except Exception:
            return "unknown"

    def diagnose_order_listing(self) -> dict:
        """
        Report what every candidate order-listing call actually returns.

        Balance and position calls succeed while order listings come back
        empty, so the question is which specific endpoint disagrees with the
        exchange UI. Guessing has been wrong repeatedly; this reports the URL
        used, the count, and the raw error for each candidate so the answer is
        observed rather than inferred.
        """
        urls = getattr(self.exchange, "urls", {}) or {}
        api = urls.get("api") if isinstance(urls.get("api"), dict) else {}
        out = {
            "demo_flag": self.demo,
            "demo_enabled_in_ccxt": bool(
                (getattr(self.exchange, "options", {}) or {}).get("enableDemoTrading")),
            "fapiPrivate_url": api.get("fapiPrivate", "?"),
            "private_url": api.get("private", "?"),
            "default_type": (getattr(self.exchange, "options", {}) or {}).get("defaultType"),
            "attempts": [],
        }
        try:
            out["positions"] = len(self.exchange.fetch_positions() or [])
        except Exception as e:
            out["positions"] = f"error: {e}"

        candidates = [
            ("fetch_open_orders()", lambda: self.exchange.fetch_open_orders()),
            ("fetch_open_orders(type=future)",
             lambda: self.exchange.fetch_open_orders(None, None, None,
                                                     {"type": "future"})),
            ("fapiPrivateGetOpenOrders",
             lambda: self.exchange.fapiPrivateGetOpenOrders()),
            ("fapiPrivateGetOpenOrders(recvWindow)",
             lambda: self.exchange.fapiPrivateGetOpenOrders({"recvWindow": 10000})),
        ]
        for name, call in candidates:
            row = {"call": name}
            try:
                got = call() or []
                row["count"] = len(got)
                row["sample"] = [
                    {k: v for k, v in (o.items() if isinstance(o, dict) else [])
                     if k in ("orderId", "id", "symbol", "origType", "type",
                              "reduceOnly", "status")}
                    for o in list(got)[:3]
                ]
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            out["attempts"].append(row)
        return out

    def _algo_orders(self) -> list:
        """Open ALGO orders — the book conditional and trailing stops live in."""
        for name in ("fapiPrivateGetOpenAlgoOrders", "fapiPrivateGetAllAlgoOrders"):
            fn = getattr(self.exchange, name, None)
            if fn is None:
                continue
            try:
                res = fn() or []
            except Exception as e:
                log.debug(f"{name} failed: {e}")
                continue
            # Binance wraps the list; accept either shape.
            if isinstance(res, dict):
                res = (res.get("orders") or res.get("data")
                       or res.get("algoOrders") or [])
            if res:
                return list(res)
        return []

    def _income_for_position(self, symbol: str, since_ms: int | None) -> tuple:
        """
        Realised P&L and commission from Binance's INCOME ledger.

        This is the authoritative record of what a position actually paid or
        earned. It matters because the fallback — reconstructing the exit from
        the stop level — can invent a profit: one trade opened and closed at
        the SAME price (true result: 0 P&L, -2.07 USDT in fees) and was
        recorded as +4.32 USDT because the estimated exit was never checked
        against the ledger.

        Returns (realised_pnl, commission, found).
        """
        fn = getattr(self.exchange, "fapiPrivateGetIncome", None)
        if fn is None:
            self._last_funding = 0.0
            return 0.0, 0.0, False
        try:
            params = {"symbol": self.exchange.market_id(symbol), "limit": 200}
            if since_ms:
                params["startTime"] = int(since_ms)
            rows = fn(params) or []
        except Exception as e:
            log.debug(f"{symbol}: income lookup failed: {e}")
            self._last_funding = 0.0
            return 0.0, 0.0, False

        pnl = comm = fund = 0.0
        unconverted = 0.0
        fee_src = ""
        seen = False
        for r in rows:
            if not isinstance(r, dict):
                continue
            kind = str(r.get("incomeType") or "").upper()
            try:
                val = float(r.get("income") or 0)
            except (TypeError, ValueError):
                continue
            if kind == "REALIZED_PNL":
                pnl += val
                seen = True
            elif kind == "FUNDING_FEE":
                # Binance breaks Realized PNL into Closing PNL + Funding Fee +
                # Trading Fee, and this component was never read. Scalps rarely
                # straddle a funding stamp so it is usually zero — but when it
                # lands it moved the wallet and nothing recorded it, which is
                # one of the contributions to wallet_gap.
                #
                # SIGNED, not abs(): funding is received as often as paid.
                fund += val
                seen = True
            elif kind == "COMMISSION":
                # The ASSET matters. Paying fees in BNB makes `income` a BNB
                # amount, and summing it as USDT reported 0.0000 on every live
                # trade — so every "net of fees" figure on that account was
                # actually GROSS while demo's was net. The two stopped being
                # comparable, silently.
                asset = str(r.get("asset") or "USDT").upper()
                if asset in ("USDT", "BUSD", "USDC", ""):
                    comm += abs(val)
                    fee_src = fee_src or "ledger"
                else:
                    rate = self._fee_asset_rate(asset)
                    if rate:
                        comm += abs(val) * rate
                        fee_src = "converted"
                    else:
                        # Better to record NOTHING than a BNB count posing as
                        # dollars. The caller falls back to an estimate.
                        unconverted += abs(val)
                        fee_src = fee_src or "unconverted"
                seen = True
        if unconverted:
            log.warning(
                f"{symbol}: {unconverted:.8f} of commission in a non-USDT "
                f"asset could not be converted — fees are understated for "
                f"this trade.")
        self._last_fee_source = fee_src or "ledger"
        if fund:
            log.info(f"{symbol}: funding {fund:+.6f} folded into realised "
                     f"(closing {pnl:+.6f})")
        self._last_funding = round(fund, 8)
        # Binance's own position card sums Closing PNL + Funding Fee into
        # Realized PNL, so the realised figure carries funding too. Keeping it
        # out was why a funded position moved the wallet by more than the
        # trade record explained.
        return round(pnl + fund, 8), round(comm, 8), seen

    def estimate_fees(self, notional: float) -> float:
        """
        Fallback when the ledger fee cannot be used: notional x rate x 2 legs.

        Approximate by construction — it assumes taker on both legs, which is
        right for a trailing entry and a market stop but wrong for anything
        that rests. Only used when the ledger gives nothing usable, and the
        trade records fee_source="estimated" so the figure is never mistaken
        for a measurement.
        """
        try:
            rate = float(getattr(self.cfg, "taker_fee_rate", 0.0005) or 0.0005)
            return round(abs(float(notional)) * rate * 2.0, 8)
        except Exception:
            return 0.0

    def _fee_asset_rate(self, asset: str) -> float | None:
        """
        Price of a fee asset in USDT. BNB comes free with the candidate
        poller — premiumIndex returns every symbol, so BNBUSDT is already in
        hand and needs no extra call.
        """
        if asset == "BNB":
            st = getattr(self, "_candidate_stream", None)
            if st is not None:
                try:
                    return st.bnb_mark()
                except Exception:
                    return None
        return None

    def _cancel_any(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order that may live in EITHER book.

        Conditional and trailing stops are algo orders, so a regular cancel
        returns -2011 for them — which means "not in the book I searched",
        not "does not exist". Always try both before concluding anything.
        """
        if self.dry_run:
            log.info(f"[DRY RUN] would cancel {order_id} on {symbol}")
            return True
        if order_id in self._cancelled_ids:
            return True                       # already done; never retry
        try:
            self.exchange.cancel_order(order_id, symbol)
            self._cancelled_ids.add(order_id)
            return True
        except Exception as e:
            ok, gone = self._cancel_algo_order(order_id, symbol)
            if ok or gone:
                # Cancelled, or absent from BOTH books — either way it is not
                # resting and must not be queued for another attempt.
                self._cancelled_ids.add(order_id)
                return True
            log.debug(f"{symbol}: cancel {order_id} failed in both books: {e}")
            return False

    def _cancel_algo_order(self, order_id: str, symbol: str) -> tuple:
        """
        Cancel via the ALGO book. Returns (cancelled, definitely_gone).

        -2011 from the REGULAR book only means "not in the book I searched" —
        these are algo orders. But -2011 from the ALGO book too means the order
        exists in neither, so it really is gone and must stop being retried.
        """
        fn = getattr(self.exchange, "fapiPrivateDeleteAlgoOrder", None)
        if fn is None:
            return False, False
        try:
            fn({"algoId": order_id})
            log.info(f"Cancelled ALGO order {order_id} on {symbol}")
            return True, True
        except Exception as e:
            gone = self._order_already_gone(e)
            log.debug(f"algo cancel {order_id} failed: {e}")
            return False, gone

    def _all_open_orders_raw(self) -> list:
        """
        Every open order on the account, from the exchange directly.

        The unified fetch_open_orders() call with no symbol has been observed
        returning nothing on this account while per-symbol queries return
        orders, so the raw endpoint is tried as well. Asking the exchange for
        the full list is the only approach that does not depend on the bot
        remembering which symbols to check — and that memory is cleared by a
        history reset or a failed persist.
        """
        # Belt and braces: ccxt raises a rate-limit WARNING as an ExchangeError
        # unless this is acknowledged, and the request never leaves the
        # process. Setting it in the constructor is not enough if anything
        # replaces options later, and the failure mode is silent — an empty
        # order list that looks like a clean account.
        try:
            opts = self.exchange.options
            if not isinstance(opts.get("fetchOpenOrders"), dict):
                opts["fetchOpenOrders"] = {}
            opts["fetchOpenOrders"]["warnWithoutSymbol"] = False
        except Exception:
            pass

        # The account-wide listing carries 40x the normal request weight. On
        # Binance demo futures it returns nothing no matter what, so paying
        # that weight every sweep buys exactly nothing. After a run of empty
        # replies, stop asking and retry only occasionally — cleanup relies on
        # the bot's persisted record of what it placed, not on discovery.
        now = time.time()
        skip_wide = (self._empty_listings >= self.EMPTY_LISTING_LIMIT
                     and now - self._last_wide_probe < self.WIDE_PROBE_INTERVAL_S)
        if skip_wide:
            log.debug(f"skipping account-wide order listing "
                      f"({self._empty_listings} empty replies; retry in "
                      f"{self.WIDE_PROBE_INTERVAL_S/60:.0f}m)")
            return []
        self._last_wide_probe = now

        rows = []
        for label, call in (
                ("unified", lambda: self.exchange.fetch_open_orders()),
                ("raw fapi", lambda: self.exchange.fapiPrivateGetOpenOrders()),
                # Conditional and trailing stops are ALGO orders on Binance
                # futures — placed via POST /fapi/v1/algoOrder and held in a
                # SEPARATE book. They never appear in /fapi/v1/openOrders,
                # which is why every listing returned zero while orders were
                # plainly resting on the exchange.
                ("algo", lambda: self._algo_orders()),
        ):
            try:
                got = call() or []
                log.info(f"open-order listing via {label}: {len(got)} order(s)")
                for o in got:
                    if isinstance(o, dict):
                        rows.append(o)
            except Exception as e:
                # Log the MESSAGE, not just the type. "ExchangeError" alone
                # hides exactly what Binance said, which is the only useful
                # part when two order listings disagree with the UI.
                log.warning(f"open-order listing via {label} FAILED: "
                            f"{type(e).__name__}: {e}")

        if rows:
            self._empty_listings = 0
        else:
            self._empty_listings += 1
            if self._empty_listings == self.EMPTY_LISTING_LIMIT:
                log.warning(
                    f"account-wide order listing has returned nothing "
                    f"{self._empty_listings} times; it costs 40x request "
                    f"weight, so it will now be probed only every "
                    f"{self.WIDE_PROBE_INTERVAL_S/60:.0f} minutes. Cleanup "
                    f"continues from the bot's own persisted record of the "
                    f"stops it placed.")
        return rows

    def _normalise_order(self, o: dict) -> dict:
        """Accept either a ccxt order or a raw Binance one."""
        info = o.get("info") if isinstance(o.get("info"), dict) else o
        sym = o.get("symbol") or ""
        if sym and "/" not in sym:
            # raw Binance gives BRUSDT; map it back to the unified symbol
            try:
                sym = self.exchange.markets_by_id[sym][0]["symbol"]
            except Exception:
                base = sym[:-4] if sym.endswith("USDT") else sym
                sym = f"{base}/USDT:USDT"
        # Careful: the raw endpoint returns the STRING "false", and
        # bool("false") is True. Coerce explicitly — getting this wrong would
        # classify an entry order as a protective stop and cancel it, breaking
        # the guarantee that the bot never touches an entry it did not place.
        ro = o.get("reduceOnly")
        if ro is None:
            ro = info.get("reduceOnly")
        if isinstance(ro, str):
            ro = ro.strip().lower() == "true"
        ro = bool(ro)
        return {
            # Algo orders identify themselves with algoId, not orderId.
            "id": str(o.get("id") or info.get("orderId")
                      or info.get("algoId") or o.get("algoId") or ""),
            "symbol": sym,
            "type": (o.get("type") or info.get("origType")
                     or info.get("type") or info.get("algoType")
                     or info.get("strategyType") or "").upper(),
            "reduce_only": ro,
            "ts": o.get("timestamp") or float(info.get("time") or 0),
            "qty": o.get("amount") or info.get("origQty"),
        }

    def _orphan_scan_symbols(self, live: set) -> list:
        """
        Symbols worth checking for orphaned stops: anywhere the bot has traded
        or is looking. Symbols with a CLOSED trade are the likeliest source —
        that is exactly when a stop is left behind.
        """
        syms = set()
        with self._lock:
            syms.update(self._states.keys())
            syms.update(self._pos_meta.keys())
            syms.update(self._all_stop_ids.keys())
            for t in self._closed_trades[-60:]:
                if t.get("symbol"):
                    syms.add(t["symbol"])
        entry = getattr(self, "_entry_service", None)
        if entry is not None:
            try:
                syms.update(entry.export_placed_orders().keys())
            except Exception:
                pass
        scanner = getattr(self, "_scanner", None)
        if scanner is not None:
            try:
                for row in (scanner.snapshot().get("candidates") or []):
                    if row.get("symbol"):
                        syms.add(row["symbol"])
            except Exception:
                pass
        return sorted(syms - live)

    def reconcile_orders(self) -> dict:
        """
        Compare every resting order against actual positions.

        Exists so a mismatch is visible in the dashboard rather than requiring
        the exchange UI to be read by hand. Classifies each order as
        protecting a real position, an orphaned stop, or a stale entry.
        """
        out = {"positions": [], "protecting": [], "orphan_stops": [],
               "stale_entries": [], "account_wide_count": 0,
               "per_symbol_count": 0, "error": None,
               # Which account/endpoint this reflects. When an order listing
               # disagrees with the exchange UI, the first question is whether
               # both are looking at the same place.
               "endpoint": self._endpoint_hint(),
               "demo": self.demo}
        try:
            live = {p.symbol: p for p in self.fetch_positions()}
        except Exception as e:
            out["error"] = f"positions: {e}"
            return out
        out["positions"] = [
            {"symbol": sym, "side": p.side, "qty": p.qty,
             "margin": round(p.margin, 2)} for sym, p in live.items()]

        orders, seen = [], set()
        for raw in self._all_open_orders_raw():
            o = self._normalise_order(raw)
            if o["id"] and o["id"] not in seen:
                seen.add(o["id"]); orders.append(o); out["account_wide_count"] += 1
        for sym in sorted(live) + self._orphan_scan_symbols(set(live)):
            try:
                for raw in (self.exchange.fetch_open_orders(sym) or []):
                    o = self._normalise_order(raw)
                    o["symbol"] = o["symbol"] or sym
                    if o["id"] and o["id"] not in seen:
                        seen.add(o["id"]); orders.append(o)
                        out["per_symbol_count"] += 1
            except Exception:
                pass

        for o in orders:
            ro = o["reduce_only"]
            sym = o["symbol"]
            row = dict(o)
            if not ro:
                row["has_position"] = sym in live
                out["stale_entries"].append(row)
            elif sym in live:
                out["protecting"].append(row)
            else:
                out["orphan_stops"].append(row)

        # A position rests up to three protective orders ON PURPOSE — fixed
        # stop, adaptive trail, profit floor — so "extras" are only surplus if
        # the guardian is NOT holding an id for them. Anything it tracks is
        # deliberate; the rest are superseded ratchets whose cancel never took.
        out["duplicate_stops"] = []
        by_sym: dict = {}
        for r in out["protecting"]:
            by_sym.setdefault(r["symbol"], []).append(r)
        keep = []
        for sym, rows in by_sym.items():
            st = self._states.get(sym)
            deliberate = {i for i in (
                getattr(st, "stop_order_id", None),
                getattr(st, "native_trail_id", None),
                getattr(st, "adaptive_trail_id", None),
                getattr(st, "floor_stop_id", None),
            ) if i} if st else set()
            tracked = [r for r in rows if r.get("id") in deliberate]
            rest = [r for r in rows if r.get("id") not in deliberate]
            rest.sort(key=lambda r: r.get("ts") or 0, reverse=True)
            keep.extend(tracked)
            if rest:
                keep.append(rest[0])
                out["duplicate_stops"].extend(rest[1:])
        out["protecting"] = keep
        return out

    def reap_orphan_stops(self) -> list:
        """
        Cancel reduce-only stops resting on symbols with NO open position.

        A reduce-only stop with nothing to reduce serves no purpose and cannot
        protect anything — but it CAN fire against a future position on the
        same symbol, closing it at a level chosen for a trade that already
        ended. Eight such orders accumulated across five symbols, three of them
        on one symbol.

        Unlike the entry reaper this does not need to know who placed the
        order: a protective stop with no position is unambiguously stale
        whoever created it. Non-reduce-only orders are never touched, so the
        operator's own entries are safe.
        """
        try:
            live = {p.symbol for p in self.fetch_positions()}
        except Exception as e:
            log.warning(f"orphan-stop sweep could not list positions: {_safe_err(e)}")
            return []

        # Collect from BOTH paths and de-duplicate by order id. The
        # account-wide call is cheaper, but conditional (stop / trailing)
        # orders have been observed missing from it, so every symbol the bot
        # has touched is also queried directly. Six orphans survived a sweep
        # that relied on the account-wide call alone.
        orders, seen_ids = [], set()
        account_wide = 0
        for raw in self._all_open_orders_raw():
            o = self._normalise_order(raw)
            if o["id"] and o["id"] not in seen_ids:
                seen_ids.add(o["id"]); orders.append(o); account_wide += 1

        per_symbol = 0
        # Per-symbol is now a top-up, not the primary source: live symbols may
        # carry SURPLUS stops, and remembered symbols are still worth checking.
        for sym in sorted(live) + self._orphan_scan_symbols(live):
            try:
                for raw in (self.exchange.fetch_open_orders(sym) or []):
                    o = self._normalise_order(raw)
                    o["symbol"] = o["symbol"] or sym
                    if o["id"] and o["id"] not in seen_ids:
                        seen_ids.add(o["id"]); orders.append(o); per_symbol += 1
            except Exception as e:
                log.debug(f"{sym}: per-symbol order list failed: {e}")

        log.info(f"orphan-stop sweep: {account_wide} order(s) account-wide, "
                 f"{per_symbol} more per-symbol, {len(orders)} total")
        # An empty result is now the NORMAL, healthy state: everything the bot
        # placed has been cleaned up. It used to mean the listing was broken,
        # because conditional stops live in the algo book and only the regular
        # book was queried. Warn only if orders are still queued for
        # cancellation while the listing claims there is nothing there — that
        # combination is genuinely contradictory.
        with self._lock:
            queued = sum(len(v) for v in self._pending_cancels.values())
        if not orders and queued and not self._listing_warned:
            self._listing_warned = True
            log.warning(
                f"Order listing reports nothing, yet {queued} order(s) are "
                f"still queued for cancellation. Either they are gone and the "
                f"queue is stale, or a book is not being read. Both listings "
                f"and the algo book were tried.")

        cancelled = []
        surplus: dict = {}
        for o in orders or []:
            if not o["reduce_only"]:
                continue                     # an entry order; not ours to judge
            sym = o["symbol"]
            if not sym:
                continue
            if sym in live:
                # A live position keeps its most recent stop; earlier ones are
                # superseded ratchets that can still fire at a stale level.
                surplus.setdefault(sym, []).append(o)
                continue
            otype = o["type"]
            if "STOP" not in otype and "TRAILING" not in otype:
                continue                     # not a protective stop
            oid = o["id"]
            if self._cancel_any(oid, sym):
                log.warning(f"{sym}: cancelled ORPHANED protective stop {oid} "
                            f"— no open position to protect")
                self._record(sym, "orphan_stop_swept", f"id={oid} (no position)")
                cancelled.append((sym, oid))
                self._clear_pending_cancel(sym, oid)
            else:
                self._queue_pending_cancel(sym, oid)
        # Now trim any live position down to its INTENTIONAL protective set.
        #
        # This used to keep exactly one stop and cancel the rest, on the
        # assumption that extras were superseded ratchets whose cancel never
        # took. That assumption no longer holds: a position now rests up to
        # three orders ON PURPOSE — the fixed stop, the adaptive trail, and the
        # profit floor — each with a distinct job and its own id in state.
        #
        # Keeping only the newest would have cancelled the PROFIT FLOOR, the
        # one order that must survive while the position is open, and the
        # adaptive trail placed at adoption. Anything the guardian is holding
        # an id for is deliberate and is never surplus.
        for sym, rows in surplus.items():
            st = self._states.get(sym)
            deliberate = {i for i in (
                getattr(st, "stop_order_id", None),
                getattr(st, "native_trail_id", None),
                getattr(st, "adaptive_trail_id", None),
                getattr(st, "floor_stop_id", None),
            ) if i} if st else set()
            rows = [r for r in rows if r.get("id") not in deliberate]
            if len(rows) < 2:
                continue
            rows.sort(key=lambda r: r.get("ts") or 0, reverse=True)
            for o in rows[1:]:
                otype = o["type"]
                if "STOP" not in otype and "TRAILING" not in otype:
                    continue
                oid = o["id"]
                if self._cancel_any(oid, sym):
                    log.warning(f"{sym}: cancelled SURPLUS protective stop {oid} "
                                f"— position already has a newer one")
                    self._record(sym, "surplus_stop_swept", f"id={oid}")
                    cancelled.append((sym, oid))
                else:
                    self._queue_pending_cancel(sym, oid)

        if cancelled:
            log.warning(f"orphan-stop sweep cancelled {len(cancelled)} order(s)")
        return cancelled

    def _cancel_orphan_stop(self, symbol: str, order_id: str) -> bool:
        """
        Cancel a stop left behind by a closed position. Returns True only on a
        confirmed cancel.

        The order has often already triggered (that is why the position
        closed), so an error here is common. But -2011 has been observed on
        orders that were STILL RESTING, so a failure is queued for retry rather
        than assumed benign — nothing else will ever discover it.
        """
        if self.dry_run:
            log.info(f"[DRY RUN] would cancel orphaned stop {order_id} on {symbol}")
            return True
        try:
            self.exchange.cancel_order(order_id, symbol)
            log.info(f"Cancelled orphaned stop {order_id} on {symbol} "
                     f"(position already closed)")
            self._record(symbol, "orphan_stop_cancelled", f"id={order_id}")
            return True
        except Exception as e:
            # Often the stop is what closed the position — but -2011 has been
            # seen on orders still resting, so report failure and let the
            # caller queue it.
            ok, gone = self._cancel_algo_order(order_id, symbol)
            if ok or gone:
                self._cancelled_ids.add(order_id)
                if ok:
                    self._record(symbol, "orphan_stop_cancelled",
                                 f"id={order_id} (algo)")
                return True
            log.debug(f"orphaned stop {order_id} on {symbol} not cancellable: {e}")
            return False

    def account_value(self, bal: dict | None = None) -> dict:
        """
        The whole account in USDT, with the fee reserve at COST.

        This is what the return cards should read. The USDT balance alone
        cannot see a fee paid in BNB — LSK 2026-09-20 proved it in isolation:
        the USDT wallet moved -0.2273, exactly the closing PnL, while the
        0.0395 fee left the BNB balance. That 0.0395 is the entire wallet_gap
        on that trade.

        Non-USDT assets are valued at the basis recorded the first time a
        price was seen, never at the live mark — see fee_reserve_basis.

        Never raises, and never reports a smaller account because a lookup
        failed: on any error it falls back to the USDT balance and says so in
        `source`, so a blip reads as "degraded" rather than as a loss.
        """
        out = {"value": 0.0, "usdt": 0.0, "reserve": 0.0,
               "per_asset": {}, "unpriced": [], "source": "account_value"}
        try:
            if bal is None:
                bal = self._last_balance_payload or {}
            usdt, _ = resolve_usdt_balance(bal)
            # A payload we cannot parse must not resolve to ZERO — that is the
            # "blip reads as a loss" failure this whole design exists to
            # avoid. Fall back to the last good cached balance.
            if not usdt or float(usdt) <= 0:
                usdt = getattr(self, "_wallet_balance_cached", 0.0) or 0.0
            out["usdt"] = float(usdt or 0.0)
            if not hasattr(self, "_asset_basis"):
                self._asset_basis = {}
            info = (bal or {}).get("info") or {}
            assets = info.get("assets") if isinstance(info, dict) else None
            seen = {str((a or {}).get("asset") or "").upper(): True
                    for a in (assets or []) if isinstance(a, dict)}
            prices = fee_reserve_basis(
                seen, self._asset_basis, self._fee_asset_prices())
            # Build the total as USDT + NON-USDT, never by summing the whole
            # assets[] array.
            #
            # v3.61.0 summed everything and shipped a doubling bug: demo read
            # +100.00% with gap +$5000 on a 5000 wallet, i.e. wallet_now came
            # back as 10000. Any payload that lists USDT twice, or carries a
            # total row alongside the per-asset rows, double-counts the margin
            # balance — which is the largest number there.
            #
            # resolve_usdt_balance is the one resolver sizing already trusts,
            # so taking USDT from it and adding only the reserve makes the
            # total impossible to double-count by construction.
            _, per, unpriced = resolve_account_value(bal, prices)
            if not per:
                # No usable assets array. The USDT figure is still correct, but
                # say so — a silent "account_value" would imply the reserve was
                # measured and found to be zero.
                out["source"] = "usdt-only (no assets array)"
                out["value"] = out["usdt"]
                out["reserve"] = 0.0
                return out
            stable = ("USDT", "BUSD", "USDC", "FDUSD")
            reserve = 0.0
            for name, d in (per or {}).items():
                if str(name).upper() in stable:
                    continue
                v = (d or {}).get("usdt")
                if v:
                    reserve += float(v)
            out.update(per_asset=per, unpriced=unpriced)
            out["reserve"] = round(reserve, 8)
            out["value"] = round(out["usdt"] + reserve, 8)
            if not out["value"]:
                out["source"] = "usdt-only (no assets array)"
                out["value"] = out["usdt"]
                out["reserve"] = 0.0
                return out
            if unpriced:
                out["source"] = f"account_value (unpriced: {','.join(unpriced)})"
            return out
        except Exception as e:
            log.debug(f"account_value failed ({e}) — falling back to USDT")
            out["value"] = out["usdt"] or (
                getattr(self, "_wallet_balance_cached", 0.0) or 0.0)
            out["source"] = "usdt-only (degraded)"
            return out

    def _fee_asset_prices(self) -> dict:
        """Live prices for basis-setting only. Used ONCE per asset."""
        prices = {}
        for name in list(getattr(self, "_asset_basis", {}) or {}) + ["BNB"]:
            if name in prices or name == "USDT":
                continue
            try:
                rate = self._fee_asset_rate(name)
                if rate:
                    prices[name] = float(rate)
            except Exception:
                continue
        return prices

    def closed_trades(self) -> list[dict]:
        with self._lock:
            return list(reversed(self._closed_trades))

    def _untracked_due(self) -> bool:
        """Sweeping costs one call per symbol, so it runs on its own slower clock."""
        now = time.time()
        if now - self._last_untracked_sweep < self.untracked_sweep_interval_s:
            return False
        self._last_untracked_sweep = now
        return True

    def _reap_scan_symbols(self) -> list:
        """
        Symbols worth sweeping: whatever the bot has touched, plus anything the
        scanner is currently looking at.
        """
        syms = set()
        entry = getattr(self, "_entry_service", None)
        if entry is not None:
            syms.update(entry.export_placed_orders().keys())
        scanner = getattr(self, "_scanner", None)
        if scanner is not None:
            try:
                for row in (scanner.snapshot().get("candidates") or []):
                    if row.get("symbol"):
                        syms.add(row["symbol"])
            except Exception:
                pass
        with self._lock:
            syms.update(self._states.keys())
            syms.update(self._pos_meta.keys())
        return sorted(syms)

    def _has_pending_entries(self) -> bool:
        entry = getattr(self, "_entry_service", None)
        if entry is None:
            return False
        try:
            return any(entry.bot_placed_orders(sym)
                       for sym in list(entry.export_placed_orders().keys()))
        except Exception:
            return False

    def run_forever(self):
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                log.warning(f"guardian cycle error: {e}")

            # Between an entry FILLING and the guardian first seeing it, the
            # position has no stop at all. On a sharp move that window is long
            # enough to blow past the stop level entirely — at 20x, a 0.5% move
            # in 5s is already -10% ROI. Poll faster while an entry order is
            # resting, so a new position is protected sooner.
            interval = self.poll_interval
            if self._has_pending_entries():
                interval = min(interval, self.pending_poll_interval)
            # If the exchange rate-limited us, stretch the poll rather than
            # continuing at the rate that caused it. Protection is already
            # resting on the exchange, so a slower loop costs observation, not
            # safety.
            rl = RATE_LIMIT.status()
            if rl["limited_now"]:
                time.sleep(max(interval, min(rl["seconds_remaining"], 15)))
            else:
                time.sleep(interval)

    def start_background(self):
        t = threading.Thread(target=self.run_forever, daemon=True,
                             name="futures-guardian")
        t.start()
        return t

    # ── API surface ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            states = {
                sym: {
                    "peak_roi": round(s.peak_roi, 2),
                    "trough_roi": round(s.trough_roi, 2),
                    "armed": s.armed,
                    "stop_roi": None if s.stop_roi is None else round(s.stop_roi, 2),
                    "stop_order_id": s.stop_order_id,
                    "native_trail_id": s.native_trail_id,
                    "unprotected_reason": s.unprotected_reason,
                    # Sizing, so a surprising position size is visible rather
                    # than something to reconstruct from the exchange UI.
                    "margin_usdt": round(self._pos_meta.get(sym, {}).get("margin", 0.0), 2),
                    "notional_usdt": round(self._pos_meta.get(sym, {}).get("notional", 0.0), 2),
                    "leverage": round(self._pos_meta.get(sym, {}).get("leverage", 0.0), 1),
                    "side": self._pos_meta.get(sym, {}).get("side"),
                    "current_roi": _r2(self._pos_meta.get(sym, {}).get("current_roi")),
                    "entry_price": self._pos_meta.get(sym, {}).get("entry_price"),
                    "current_price": self._pos_meta.get(sym, {}).get("current_price"),
                    "range_pos_24h": _r3(self._pos_meta.get(sym, {}).get("range_pos_24h")),
                    "pct_above_24h_low": _r2(self._pos_meta.get(sym, {}).get("pct_above_24h_low")),
                    "pct_below_24h_high": _r2(self._pos_meta.get(sym, {}).get("pct_below_24h_high")),
                    "high_24h": self._pos_meta.get(sym, {}).get("high_24h"),
                    "low_24h": self._pos_meta.get(sym, {}).get("low_24h"),
                    "range_source": self._range_source.get(sym, "unknown"),
                    "risk_overshoot": self._risk_overshoots.get(sym),
                }
                for sym, s in self._states.items()
            }
            actions = list(reversed(self._actions[-20:]))
        return {
            "enabled": True,
            "dry_run": self.dry_run,
            "demo": self.demo,
            "states": states,
            "recent_actions": actions,
            "last_cycle_ago_s": (time.time() - self._last_cycle_ts) if self._last_cycle_ts else None,
            "error": self._last_error,
            "wallet_balance": self._wallet_balance_cached,
            "config": {
                "initial_stop_roi": self.cfg.initial_stop_roi,
                "arm_roi": self.cfg.arm_roi,
                "callback_roi": self.cfg.callback_roi,
                "trail_callback_pct": self.cfg.trail_callback_pct,
                "use_native_trail": self.cfg.use_native_trail,
                # Fail-fast was invisible here, so a value of 0 — which
                # disables it outright on the first line of the predicate —
                # looked identical to a working configuration. 119 trades ran
                # with it silently off.
                "fail_fast_s": self.cfg.fail_fast_s,
                "fail_fast_max_peak_roi": _peak_ceiling(self.cfg),
                "fail_fast_loss_roi": self.cfg.fail_fast_loss_roi,
                "fail_fast_enabled": bool(self.cfg.fail_fast_s),
            },
            "rate_limit": RATE_LIMIT.status(),
            # What is ACTUALLY resting, per symbol, so the dashboard can stop
            # showing the configured floor level for a position that has none.
            "floor_state": {
                sym: {
                    "floor_roi": getattr(st, "floor_roi", None),
                    "has_floor": bool(getattr(st, "floor_stop_id", None)),
                    "failed_attempts": getattr(st, "floor_attempts", 0),
                    "peak_roi": st.peak_roi,
                }
                for sym, st in (self._states or {}).items()
            },
        }
