"""
Scalping bot engine — momentum + trend filtering with trailing stop-loss.

Entry:    ADX > 25, RSI 50-65, EMA20 > EMA50, close > VWAP
Exit:     Trailing stop (in-memory, updates every cycle) OR take-profit OR timeout

Safety:   On entry an OCO order is placed on Binance with a wider fixed stop
          (OCO_STOP_PCT, default 2%) as a server-side backstop.
          - While the bot is running the trailing stop always fires first
            because it is tighter than the OCO stop.
          - If the bot dies the OCO protects the position until restart.
          - On startup the bot recovers open positions from Binance, cancels
            stale OCOs, and resumes normal trailing-stop management.
"""

import time
import logging
import threading
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timezone

import ccxt
import pandas as pd
import pandas_ta as ta

from .config import BotConfig
from .risk import RiskManager
from .state import PositionState
from .trade_log import TradeLog, ClosedTrade
from .store import PositionStore

log = logging.getLogger("engine")


class ScalpingEngine:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.exchange = self._init_exchange()
        self.risk = RiskManager(cfg)
        self.positions = PositionStore()
        self._symbol_cache: list[str] = []
        self._cache_ts: float = 0
        self._cooldown: dict[str, float] = {}  # symbol → earliest re-entry timestamp
        self.kill_switch: bool = False
        # Trailing-stop activation threshold — in-memory only, UI-controlled.
        # enabled=False means immediate trailing (original behaviour).
        # pct seeds from config but is adjustable live via the UI.
        self.trailing_activation_enabled: bool = False
        self.trailing_activation_pct: float = cfg.trailing_activation_pct
        # BTC market-regime filter — in-memory only, UI-controlled.
        # enabled=False means BTC trend is computed and logged but never blocks entries.
        self.btc_filter_enabled: bool = False
        self.btc_trend_lookback: int = cfg.btc_trend_lookback
        self.btc_trend_threshold_pct: float = cfg.btc_trend_threshold_pct
        # Per-cycle cache of the BTC regime so we compute it at most once per cycle.
        self._btc_regime_cache: dict | None = None
        # Entry-timing gate (per-coin) — skips entries extended above the fast EMA.
        # Default ON: combined with momentum confirmation, this enforces
        # "rising BUT not over-extended" — rejecting both downswings (momentum)
        # and local-top chases (this gate). Fully toggleable via the UI.
        self.entry_timing_enabled: bool = True
        self.entry_timing_ema_len: int = cfg.entry_timing_ema_len
        self.entry_timing_band_pct: float = cfg.entry_timing_band_pct
        # Momentum confirmation (per-coin short-term direction) — DEFAULT ON.
        # Confirms the coin is rising right now using raw-price slope (no lag).
        self.momentum_enabled: bool = True
        self.momentum_lookback: int = cfg.momentum_lookback
        self.momentum_min_slope_pct: float = cfg.momentum_min_slope_pct
        # Continuous profit lock (peak-tracking) — DEFAULT ON. Once P&L crosses
        # the arm threshold, a profit floor ratchets up with the peak P&L and
        # locks a rising fraction of the gain. Sits alongside the trailing stop.
        self.profit_lock_enabled: bool = True
        self.profit_lock_arm_pct: float = cfg.profit_lock_arm_pct
        self.profit_lock_giveback_pct: float = cfg.profit_lock_giveback_pct
        # Hard stop-loss: cut a losing position at a fixed P&L rather than waiting
        # for the trailing stop. Downside mirror of the profit lock.
        self.hard_stop_enabled: bool = cfg.hard_stop_enabled
        self.hard_stop_pct: float = cfg.hard_stop_pct
        # Smart re-entry guard: after a RED close on a coin, don't re-enter it at a
        # price HIGHER than the loss exit (avoids chasing a loser back up). Records
        # the last exit price + whether it was a loss, per symbol.
        self.reentry_guard_enabled: bool = cfg.reentry_guard_enabled
        self._last_exit: dict[str, dict] = {}  # symbol → {"price": float, "was_loss": bool}
        self.trade_log = TradeLog()
        self.last_balance: float = 0.0
        self.last_cycle_ts: float = 0.0
        # Serializes all position mutations so the fast monitor loop and the main
        # trading cycle never race on the same position (e.g. both trying to close).
        self._pos_lock = threading.RLock()
        # Fast peak/exit monitor interval (seconds). Far shorter than the main
        # trading cycle so the engine sees price spikes the way the dashboard does,
        # ratcheting peak P&L and triggering the profit lock / trailing stop promptly.
        self.monitor_interval: float = getattr(cfg, "monitor_interval", 7.0)

    # ------------------------------------------------------------------
    # Exchange setup
    # ------------------------------------------------------------------

    def _init_exchange(self) -> ccxt.binance:
        params: dict = {
            "apiKey": self.cfg.api_key,
            "secret": self.cfg.api_secret,
            "options": {"defaultType": "spot"},
            "enableRateLimit": True,
        }

        if self.cfg.socks_proxy:
            # Route all ccxt traffic through the SOCKS5 proxy.
            # socks5h:// ensures DNS also resolves through the proxy (no leaks).
            # Requires requests[socks] (PySocks) — included in requirements.txt.
            params["proxies"] = {
                "http":  self.cfg.socks_proxy,
                "https": self.cfg.socks_proxy,
            }
            log.info(f"🔀 Proxy active: {self.cfg.socks_proxy}")
        else:
            log.info("🔀 No proxy configured — connecting directly")

        exchange = ccxt.binance(params)

        if self.cfg.testnet:
            exchange.set_sandbox_mode(True)
            log.info("🟡 Testnet mode active — endpoint: testnet.binance.vision")
        else:
            log.info("🔴 LIVE trading mode active")

        exchange.load_markets()
        return exchange

    # ------------------------------------------------------------------
    # Startup recovery — Option B
    # Reads open balances from Binance, reconstructs PositionState for
    # each coin we hold, cancels any stale OCO orders, and lets the
    # trailing stop take over from the current price.
    # ------------------------------------------------------------------

    def recover_positions(self):
        """
        Primary recovery: load positions.json written by PositionStore.
        This is immune to parameter changes between restarts — the stored
        positions reflect exactly what the bot entered, regardless of current
        config values.

        Secondary sanity check: verify each stored position still has a real
        balance on Binance (catches the edge case where positions.json is
        stale after a manual sell). Any position that has no corresponding
        Binance balance is discarded and removed from the store.
        """
        stored = list(self.positions.items())  # loaded by PositionStore.__init__

        if not stored:
            log.info("No stored positions to recover.")
            return

        log.info(f"Verifying {len(stored)} stored position(s) against Binance balance...")

        try:
            balance_data = self.exchange.fetch_balance()
        except Exception as e:
            log.warning(f"Recovery: could not fetch balance for sanity check: {e}. Trusting store as-is.")
            log.info(f"Resumed {len(stored)} position(s) from store (unverified).")
            return

        discarded = 0
        for symbol, pos in stored:
            coin = symbol.replace("/USDT", "")
            amounts = balance_data.get(coin, {})
            if not isinstance(amounts, dict):
                amounts = {}
            free = float(amounts.get("free") or 0)
            used = float(amounts.get("used") or 0)  # locked in open orders (e.g. stop-market)
            total = free + used

            # Fetch current price for notional check
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker["last"] or pos.entry_price)
            except Exception:
                price = pos.entry_price

            notional = total * price

            if notional < self.cfg.min_trade_usdt * 0.5:
                # Balance gone — position was likely closed manually or OCO fired
                log.warning(
                    f"Recovery: {symbol} has no balance on Binance "
                    f"(free={free} used={used} total={total} notional={notional:.2f}) — discarding from store."
                )
                del self.positions[symbol]
                discarded += 1
            else:
                # Check if the stored backstop order is still active on Binance.
                # If it is, keep it — don't cancel valid protection.
                # Only cancel if the order is gone (filled/cancelled) or unknown.
                backstop_still_active = False
                if pos.oco_order_list_id:
                    try:
                        open_orders = self.exchange.fetch_open_orders(symbol)
                        open_ids = {str(o.get("id") or "") for o in open_orders}
                        open_list_ids = {str(o.get("orderListId") or "") for o in open_orders}
                        backstop_still_active = (
                            pos.oco_order_list_id in open_ids or
                            pos.oco_order_list_id in open_list_ids
                        )
                        if backstop_still_active:
                            log.info(f"Recovery: {symbol} backstop order {pos.oco_order_list_id} still active — keeping")
                        else:
                            # Backstop is gone — cancel any other stale orders and
                            # bot will re-place on next cycle via normal management
                            log.info(f"Recovery: {symbol} backstop {pos.oco_order_list_id} no longer active — clearing")
                            pos.oco_order_list_id = None
                            pos.backstop_type = None
                            self.positions[symbol] = pos
                    except Exception as e:
                        log.warning(f"Recovery: could not check orders for {symbol}: {e}")

                # Reconcile stored qty against the actual Binance balance.
                #
                # `total` = free + used, where `used` is coins locked by open orders
                # (e.g. the active stop-market backstop). The DELIVERABLE balance — what
                # we can actually sell — is at most `total`. We never want the stored qty
                # to exceed what Binance reports, because that inflates the displayed
                # position value and portfolio total (the TAO 2.0-vs-0.998 discrepancy).
                #
                # Rules:
                #   - Binance MORE than stored (>5%): excess is unmanaged, keep stored qty.
                #   - Binance LESS than stored (>5%): trust Binance, adjust down.
                #   - Within 5%: snap to the actual total so the display is exact.
                if total > pos.qty * 1.05:
                    log.warning(
                        f"Recovery: {symbol} Binance balance ({total}) exceeds stored qty "
                        f"({pos.qty}) by >5% — managing stored qty only. "
                        f"Excess {total - pos.qty:.6f} units are unmanaged."
                    )
                    # qty stays as stored — don't update
                elif total < pos.qty * 0.95:
                    log.warning(
                        f"Recovery: {symbol} qty adjusted down {pos.qty} → {total} "
                        f"(partial sell, dust, or stored qty was stale)"
                    )
                    pos.qty = total
                    self.positions[symbol] = pos

                log.warning(
                    f"RECOVERED {symbol}: entry={pos.entry_price:.6f} "
                    f"stop={pos.trailing_stop:.6f} qty={pos.qty} candles={pos.candles_held} "
                    f"backstop={'active' if backstop_still_active else 'none'}"
                )

        active = len(self.positions)
        log.info(
            f"Recovery complete: {active} active, {discarded} discarded."
        )

    def _cancel_open_orders(self, symbol: str):
        """Cancel all open orders for a symbol (clears stale OCOs)."""
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order["id"], symbol)
                    log.info(f"Cancelled stale order {order['id']} for {symbol}")
                except Exception as e:
                    log.warning(f"Could not cancel order {order['id']} for {symbol}: {e}")
        except Exception as e:
            log.warning(f"Could not fetch open orders for {symbol}: {e}")

    # ------------------------------------------------------------------
    # Symbol selection
    # ------------------------------------------------------------------

    def get_candidate_symbols(self) -> list[str]:
        now = time.time()
        if now - self._cache_ts < self.cfg.symbol_cache_ttl:
            return self._symbol_cache

        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            log.warning(f"Ticker fetch failed: {e}")
            return self._symbol_cache

        candidates = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT"):
                continue
            if sym in self.cfg.blacklist:
                continue
            volume_usdt = (t.get("quoteVolume") or 0)
            bid, ask = t.get("bid") or 0, t.get("ask") or 0
            spread_pct = ((ask - bid) / ask * 100) if ask > 0 else 999
            # Filter out stablecoins and near-stable assets by 24h price change.
            # Any coin passing ADX/RSI filters will have moved well above 0.5%.
            price_change = abs(t.get("percentage") or 0)
            if price_change < 0.5:
                log.debug(f"Skipping {sym}: 24h change {price_change:.2f}% — likely stablecoin")
                continue
            if volume_usdt >= self.cfg.min_volume_usdt and spread_pct <= self.cfg.max_spread_pct:
                candidates.append((sym, volume_usdt))

        candidates.sort(key=lambda x: x[1], reverse=True)
        self._symbol_cache = [s for s, _ in candidates[:self.cfg.max_symbols]]
        self._cache_ts = now
        log.info(f"Symbol universe: {self._symbol_cache}")
        return self._symbol_cache

    # ------------------------------------------------------------------
    # OHLCV + indicators
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, self.cfg.timeframe, limit=150)
        except Exception as e:
            log.debug(f"OHLCV fetch failed for {symbol}: {e}")
            return None

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df

    def compute_btc_regime(self) -> dict:
        """
        Compute BTC's market regime once per cycle and cache it.

        Returns a dict with:
          - short_term_falling: bool  → BTC down > threshold over the lookback window
          - short_term_change_pct: float → % change over the lookback window
          - regime_bullish: bool | None → EMA20 > EMA50 on BTC (slow context); None if unknown
          - available: bool → False if the BTC fetch failed (callers should fail-open)

        Fail-open contract: if BTC data cannot be fetched, available=False and
        short_term_falling=False, so a transient error never blocks trading.
        """
        if self._btc_regime_cache is not None:
            return self._btc_regime_cache

        regime = {
            "short_term_falling": False,
            "short_term_change_pct": 0.0,
            "regime_bullish": None,
            "available": False,
        }

        df = self.fetch_ohlcv("BTC/USDT")
        if df is None or len(df) < 51:
            log.debug("BTC regime: fetch failed or insufficient data — failing open (entries allowed)")
            self._btc_regime_cache = regime
            return regime

        try:
            closes = df["close"]
            current = float(closes.iloc[-1])
            lookback = max(1, int(self.btc_trend_lookback))
            # Guard against a lookback longer than the data we have
            if lookback >= len(closes):
                lookback = len(closes) - 1
            past = float(closes.iloc[-1 - lookback])
            change_pct = (current - past) / past * 100 if past > 0 else 0.0

            ema20 = float(ta.ema(closes, length=20).iloc[-1])
            ema50 = float(ta.ema(closes, length=50).iloc[-1])

            regime["short_term_change_pct"] = round(change_pct, 4)
            regime["short_term_falling"] = change_pct < -abs(self.btc_trend_threshold_pct)
            regime["regime_bullish"] = ema20 > ema50
            regime["available"] = True
        except Exception as e:
            log.debug(f"BTC regime computation error: {e} — failing open")
            # leave defaults (available=False, not falling)

        self._btc_regime_cache = regime
        return regime

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema20"] = ta.ema(df["close"], length=20)
        df["ema50"] = ta.ema(df["close"], length=50)
        # Fast EMA for the per-coin entry-timing gate (default length 9).
        # Use the runtime length if set, else fall back to the configured default.
        fast_len = getattr(self, "entry_timing_ema_len", None) or self.cfg.entry_timing_ema_len
        df["ema_fast"] = ta.ema(df["close"], length=fast_len)
        df["rsi"]   = ta.rsi(df["close"], length=14)
        adx_df      = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"]   = adx_df["ADX_14"]
        df["+di"]   = adx_df["DMP_14"]
        df["vwap"]  = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["atr"]   = ta.atr(df["high"], df["low"], df["close"], length=14)
        return df

    def passes_entry_filter(self, df: pd.DataFrame) -> bool:
        if len(df) < 60:
            return False
        row = df.iloc[-1]
        cond = {
            "ema_aligned": row["ema20"] > row["ema50"],
            "rsi_range":   self.cfg.rsi_min <= row["rsi"] <= self.cfg.rsi_max,
            "adx_strong":  row["adx"] > self.cfg.adx_min,
            "di_bullish":  row["+di"] > 20,
            "above_vwap":  row["close"] > row["vwap"],
        }
        passed = all(cond.values())
        if passed:
            log.debug(
                f"Entry filter PASSED — "
                f"ema={row['ema20']:.4f}/{row['ema50']:.4f} "
                f"rsi={row['rsi']:.1f} adx={row['adx']:.1f} "
                f"close={row['close']:.6f} vwap={row['vwap']:.6f}"
            )
        else:
            failed = [k for k, v in cond.items() if not v]
            vals = {
                "ema20": round(row["ema20"], 6), "ema50": round(row["ema50"], 6),
                "rsi": round(row["rsi"], 2), "adx": round(row["adx"], 2),
                "di+": round(row["+di"], 2),
                "close": round(row["close"], 6), "vwap": round(row["vwap"], 6),
            }
            log.debug(f"Entry filter FAILED {failed} — {vals}")
        return passed

    def entry_timing_distance_pct(self, df: pd.DataFrame) -> float | None:
        """
        How far current price sits ABOVE the fast EMA, as a percent.
        Positive = price above the fast EMA (extended); negative = below (pulled back).
        Returns None if the fast EMA isn't available.
        """
        row = df.iloc[-1]
        fast = row.get("ema_fast")
        if fast is None or fast != fast or fast <= 0:  # NaN-safe
            return None
        return (row["close"] - fast) / fast * 100

    def passes_entry_timing(self, df: pd.DataFrame) -> tuple[bool, float | None]:
        """
        Per-coin entry-timing gate. Returns (passes, distance_pct).

        A quality scalp entry is near the short-term mean, not chasing a spike.
        We allow entry only when price is within the band ABOVE the fast EMA.
        If price is pulled back to/below the fast EMA, that's an ideal entry too
        (distance <= band always passes). Only an OVER-extended price is rejected.

        When the gate is disabled, this always passes but still returns the
        distance so it can be logged for analysis.
        """
        distance = self.entry_timing_distance_pct(df)
        if not self.entry_timing_enabled:
            return True, distance
        if distance is None:
            # Can't assess — fail open (allow), consistent with other gates.
            return True, distance
        passes = distance <= self.entry_timing_band_pct
        return passes, distance

    def momentum_slope_pct(self, df: pd.DataFrame) -> float | None:
        """
        Raw-price slope over the last `momentum_lookback` candles, as a percent.
        Uses close prices directly (no moving average) to avoid lag.
        Positive = price higher than N candles ago. None if insufficient data.
        """
        lookback = max(1, int(self.momentum_lookback))
        if len(df) <= lookback:
            return None
        current = float(df["close"].iloc[-1])
        past = float(df["close"].iloc[-1 - lookback])
        if past <= 0:
            return None
        return (current - past) / past * 100

    def passes_momentum(self, df: pd.DataFrame) -> tuple[bool, float | None]:
        """
        Short-term direction confirmation. Returns (passes, slope_pct).

        Confirms the coin is actually rising AT ENTRY — not merely in a recent
        uptrend structure that lagging filters still report during a decline.
        Requires BOTH:
          - raw-price slope over the lookback window >= momentum_min_slope_pct
          - the most recent candle is not red (close >= open)
        The 'not red' check prevents entering on a net-positive window whose final
        candle has already turned down (early reversal).

        When disabled, always passes but still returns slope for logging.
        """
        slope = self.momentum_slope_pct(df)
        if not self.momentum_enabled:
            return True, slope
        if slope is None:
            return True, slope  # fail-open, consistent with other gates
        last = df.iloc[-1]
        last_candle_up = float(last["close"]) >= float(last["open"])
        passes = (slope >= self.momentum_min_slope_pct) and last_candle_up
        return passes, slope

    def profit_lock_floor_pct(self, peak_pnl_pct: float) -> float | None:
        """
        Continuous profit-lock floor, in P&L percent, derived from the peak P&L
        the position has reached.

        Returns None if the lock hasn't armed (peak below the arm threshold).
        Once armed, the floor = peak - give_back, where give_back starts at
        profit_lock_giveback_pct at the arm point and SHRINKS as the peak climbs:

            give_back = giveback_pct * (arm_pct / peak)

        So a +1% peak locks ~82% of the gain (small cushion to let it develop),
        while a +5% peak locks ~99% (give back almost nothing on a real winner).
        The floor ratchets up only, because peak_pnl_pct only ever increases.
        """
        if not self.profit_lock_enabled:
            return None
        arm = self.profit_lock_arm_pct
        if peak_pnl_pct < arm or arm <= 0:
            return None
        give_back = self.profit_lock_giveback_pct * (arm / peak_pnl_pct)
        return peak_pnl_pct - give_back

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def get_precision(self, symbol: str) -> tuple[int, int]:
        market = self.exchange.market(symbol)
        return market["precision"]["price"], market["precision"]["amount"]

    def round_amount(self, amount: float, precision: int) -> float:
        d = Decimal(str(amount)).quantize(Decimal(10) ** -int(precision), rounding=ROUND_DOWN)
        return float(d)

    def round_price(self, price: float, precision: int) -> float:
        d = Decimal(str(price)).quantize(Decimal(10) ** -int(precision), rounding=ROUND_DOWN)
        return float(d)

    def _lot_step_size(self, symbol: str) -> float:
        """Return the MARKET_LOT_SIZE stepSize for a symbol, or 0 if unavailable."""
        try:
            market = self.exchange.market(symbol)
            filters = market.get("info", {}).get("filters", [])
            for f in filters:
                if f.get("filterType") == "MARKET_LOT_SIZE":
                    return float(f.get("stepSize", 0))
            # Fall back to LOT_SIZE if MARKET_LOT_SIZE not present
            for f in filters:
                if f.get("filterType") == "LOT_SIZE":
                    return float(f.get("stepSize", 0))
        except Exception:
            pass
        return 0

    def round_to_step(self, qty: float, step: float) -> float:
        """Round qty down to the nearest lot step size."""
        if step <= 0:
            return qty
        from decimal import Decimal, ROUND_DOWN
        step_d = Decimal(str(step))
        qty_d  = Decimal(str(qty))
        return float((qty_d // step_d) * step_d)

    def place_buy(self, symbol: str, usdt_amount: float, current_price: float) -> dict | None:
        _, amount_prec = self.get_precision(symbol)
        raw_qty = usdt_amount / current_price

        # Round to lot step size first (MARKET_LOT_SIZE filter), then to precision
        step = self._lot_step_size(symbol)
        if step > 0:
            qty = self.round_to_step(raw_qty, step)
        else:
            qty = self.round_amount(raw_qty, amount_prec)

        if qty <= 0:
            log.debug(f"Skipping {symbol}: allocation ${usdt_amount:.2f} too small for lot size (step={step})")
            return None
        # Also check min notional (Binance rejects orders below ~$10 equivalent)
        if qty * current_price < self.cfg.min_trade_usdt:
            log.debug(f"Skipping {symbol}: notional ${qty * current_price:.2f} below minimum")
            return None
        try:
            order = self.exchange.create_market_buy_order(symbol, qty)
            log.info(f"BUY {symbol} qty={qty} ~{usdt_amount:.2f} USDT")
            return order
        except Exception as e:
            err = str(e)
            if "MARKET_LOT_SIZE" in err or "LOT_SIZE" in err:
                # Coin's lot size is incompatible with our allocation — not an error,
                # just skip this symbol. Happens with very low-price coins where
                # stepSize doesn't divide evenly into our position size.
                log.debug(f"Skipping {symbol}: lot size incompatible — {e}")
            else:
                log.error(f"Buy order failed for {symbol}: {e}")
            return None

    def _oco_supported(self, symbol: str) -> bool:
        """Check whether Binance supports OCO orders for this market."""
        try:
            market = self.exchange.market(symbol)
            order_types = [t.upper() for t in (market.get("orderTypes") or [])]
            return "OCO" in order_types
        except Exception:
            return False

    def place_oco(self, symbol: str, qty: float, entry_price: float) -> str | None:
        """
        Place an OCO (One-Cancels-the-Other) order as a server-side backstop.

        - Take-profit limit: entry * (1 + take_profit_pct)   ← same as bot TP
        - Stop-loss:         entry * (1 - oco_stop_pct)      ← WIDER than trailing stop
          e.g. trailing=0.8%, oco=2.0% → trailing always fires first while bot runs.

        Returns the orderListId string on success, None on failure.
        OCO is not available on all Binance spot pairs or on the testnet — if
        unsupported, logs at DEBUG and returns None. Trailing stop remains active.
        """
        if not self.cfg.oco_enabled:
            return None

        if not self._oco_supported(symbol):
            log.debug(f"OCO not supported for {symbol} — trailing stop only")
            return None

        price_prec, amount_prec = self.get_precision(symbol)
        qty = self.round_amount(qty, amount_prec)

        tp_price   = self.round_price(entry_price * (1 + self.cfg.take_profit_pct / 100), price_prec)
        stop_price = self.round_price(entry_price * (1 - self.cfg.oco_stop_pct / 100), price_prec)
        # Limit price slightly below stop to ensure fill
        stop_limit = self.round_price(stop_price * 0.999, price_prec)

        try:
            resp = self.exchange.create_order(
                symbol=symbol,
                type="OCO",
                side="sell",
                amount=qty,
                price=tp_price,
                params={
                    "stopPrice": stop_price,
                    "stopLimitPrice": stop_limit,
                    "stopLimitTimeInForce": "GTC",
                },
            )
            list_id = str(resp.get("orderListId") or resp.get("id") or "")
            log.info(
                f"OCO placed for {symbol}: tp={tp_price} stop={stop_price} "
                f"listId={list_id}"
            )
            return list_id or None
        except Exception as e:
            log.warning(
                f"OCO placement failed for {symbol} (bot trailing stop still active): {e}"
            )
            return None

    def _min_price(self, symbol: str) -> float:
        """Return the minimum allowed price for a symbol from PRICE_FILTER."""
        try:
            market = self.exchange.market(symbol)
            filters = market.get("info", {}).get("filters", [])
            for f in filters:
                if f.get("filterType") == "PRICE_FILTER":
                    return float(f.get("minPrice", 0))
        except Exception:
            pass
        return 0

    def place_stop_limit(self, symbol: str, qty: float, entry_price: float) -> str | None:
        """
        Place a stop-market sell order as a fallback for pairs that don't support OCO.

        Uses STOP_LOSS (stop-market) not STOP_LOSS_LIMIT — guarantees fill even in
        fast gap-downs since it executes at market price when triggered.

        Stop trigger = current_price * (1 - (trailing_stop_pct + stop_limit_offset_pct)%)
          → calculated from current market price to stay within PERCENT_PRICE_BY_SIDE band
          → sits just below the trailing stop so in-memory trailing always fires first

        Returns the order ID string on success, None on failure.
        """
        price_prec, amount_prec = self.get_precision(symbol)
        # Use lot step rounding — more accurate than decimal precision for stop orders
        step = self._lot_step_size(symbol)
        if step > 0:
            qty = self.round_to_step(qty, step)
        else:
            qty = self.round_amount(qty, amount_prec)

        if qty <= 0:
            log.warning(f"Stop-market qty rounded to 0 for {symbol} — skipping")
            return None

        # Binance PERCENT_PRICE_BY_SIDE validates sell stop orders against the bid price.
        # Fetch ticker once and use bid as reference — this keeps the stop within
        # the allowed band even when last price and bid diverge (common on testnet).
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = float(ticker.get("last") or entry_price)
            bid = float(ticker.get("bid") or 0)
            reference_price = bid if bid > 0 else current_price
        except Exception:
            current_price = entry_price
            reference_price = entry_price
        total_stop_pct = self.cfg.trailing_stop_pct + self.cfg.stop_limit_offset_pct
        # Ensure minimum price precision of 4 decimal places — prevents rounding
        # to whole numbers on low-precision testnet markets (e.g. XRP precision=0)
        effective_prec = max(int(price_prec), 4)
        stop_price = self.round_price(reference_price * (1 - total_stop_pct / 100), effective_prec)

        log.info(
            f"Stop-market calc for {symbol}: entry={entry_price} bid={reference_price:.6f} "
            f"stop_trigger={stop_price} (trailing={self.cfg.trailing_stop_pct}% + offset={self.cfg.stop_limit_offset_pct}%)"
        )

        # Clamp to minimum price — prevents rejection on low-price coins
        min_price = self._min_price(symbol)
        if min_price > 0:
            stop_price = max(stop_price, min_price)

        try:
            # Use STOP_LOSS (stop-market) — guarantees fill at market price when
            # triggered, unlike STOP_LOSS_LIMIT which may not fill in fast drops.
            order = self.exchange.create_order(
                symbol=symbol,
                type="STOP_LOSS",
                side="sell",
                amount=qty,
                params={"stopPrice": stop_price},
            )
            order_id = str(order.get("id") or "")
            log.info(
                f"Stop-market placed for {symbol}: trigger={stop_price} "
                f"id={order_id}"
            )
            return order_id or None
        except Exception as e:
            err = str(e)
            if "PERCENT_PRICE_BY_SIDE" in err:
                # Log the filter values from the market to understand the band
                try:
                    market = self.exchange.market(symbol)
                    filters = market.get("info", {}).get("filters", [])
                    ppbs = [f for f in filters if f.get("filterType") == "PERCENT_PRICE_BY_SIDE"]
                    log.warning(
                        f"Stop-market rejected for {symbol} (PERCENT_PRICE_BY_SIDE): "
                        f"stop={stop_price} filter={ppbs} — trailing stop only"
                    )
                except Exception:
                    log.warning(
                        f"Stop-market rejected for {symbol}: stop price outside Binance "
                        f"allowed band (PERCENT_PRICE_BY_SIDE) — trailing stop only"
                    )
            elif "market lot size" in err.lower() or "LOT_SIZE" in err:
                log.warning(f"Stop-market rejected for {symbol}: lot size issue — {e}")
            else:
                log.warning(f"Stop-market placement failed for {symbol}: {e}")
            return None

    def cancel_oco(self, symbol: str, list_id: str | None):
        """Cancel a previously placed OCO or stop-limit order before the bot exits."""
        if not list_id:
            return
        try:
            # Try as OCO first, then as plain order (stop-limit fallback)
            try:
                self.exchange.cancel_order(list_id, symbol, params={"orderListId": list_id})
                log.info(f"OCO cancelled for {symbol} id={list_id}")
            except Exception:
                self.exchange.cancel_order(list_id, symbol)
                log.info(f"Stop-limit cancelled for {symbol} id={list_id}")
        except Exception as e:
            # May already be filled or cancelled — log and move on
            log.debug(f"Order cancel for {symbol} id={list_id}: {e}")

    def place_sell(self, symbol: str, qty: float) -> dict | None:
        try:
            order = self.exchange.create_market_sell_order(symbol, qty)
            log.info(f"SELL {symbol} qty={qty} id={order.get('id','?')}")
            return order
        except Exception as e:
            log.error(f"Sell order failed for {symbol}: {e}")
            return None  # caller checks for None to detect failure

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def update_trailing_stop(self, sym: str, current_price: float):
        pos = self.positions[sym]

        # Store the live market values — this is the single source of truth that
        # the dashboard reads, so the displayed price/P&L and the values the engine
        # acts on are always identical.
        pos.current_price = current_price
        pos.pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        pos.pnl_usdt = (current_price - pos.entry_price) * pos.qty
        pos.last_price_ts = time.time()

        # Track the peak P&L this position has reached, for the profit lock.
        # Ratchets up only. Computed every pass regardless of trailing-active
        # state so the lock can arm even during a dormant trailing phase.
        current_pnl_pct = pos.pnl_pct
        if current_pnl_pct > pos.peak_pnl_pct:
            pos.peak_pnl_pct = current_pnl_pct
        self.positions[sym] = pos

        # Activation threshold: if this position's trailing stop is not yet active,
        # check whether price has reached the activation level. Until it does, the
        # trailing stop does not move and check_exit will not use it — the server-side
        # stop-market is the only protection during this phase.
        if not pos.trailing_active:
            if pos.activation_price > 0 and current_price >= pos.activation_price:
                pos.trailing_active = True
                # Seed the trailing stop from the current price now that it's active.
                pos.trailing_stop = current_price * (1 - self.cfg.trailing_stop_pct / 100)
                self.positions[sym] = pos
                log.info(
                    f"{sym} trailing stop ACTIVATED at {current_price:.6f} "
                    f"(reached threshold {pos.activation_price:.6f}); "
                    f"stop set to {pos.trailing_stop:.6f}"
                )
            # Not yet at threshold — do not trail.
            return

        new_stop = current_price * (1 - self.cfg.trailing_stop_pct / 100)
        if new_stop > pos.trailing_stop:
            self.positions.update_stop(sym, new_stop)
            log.debug(f"{sym} trailing stop raised to {new_stop:.6f}")

    def check_exit(self, sym: str, current_price: float) -> str | None:
        pos = self.positions[sym]
        current_pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100

        # Hard stop-loss — checked FIRST and regardless of trailing-active state.
        # Cuts a losing position at a fixed P&L rather than waiting for the looser
        # trailing stop or riding it down hoping for recovery. This is the downside
        # mirror of the profit lock: it bounds give-back on the loss side.
        if self.hard_stop_enabled and current_pnl_pct <= -self.hard_stop_pct:
            return "hard_stop"

        # Take profit — only checked when enabled.
        # When disabled the trailing stop is the sole profit exit,
        # allowing winners to run as far as momentum carries them.
        if self.cfg.take_profit_enabled:
            tp = pos.entry_price * (1 + self.cfg.take_profit_pct / 100)
            if current_price >= tp:
                return "take_profit"
        # Only use the trailing stop as an exit once it has activated.
        # Before activation, the server-side stop-market is the protection.
        if pos.trailing_active and current_price <= pos.trailing_stop:
            return "trailing_stop"
        # Continuous profit lock — exits at whichever triggers first vs the trail.
        # Once the peak P&L has armed the lock, the floor ratchets up with the peak.
        # If current P&L falls back to that locked floor, take the guaranteed gain
        # rather than letting the looser trailing stop give it back.
        floor_pct = self.profit_lock_floor_pct(pos.peak_pnl_pct)
        if floor_pct is not None:
            if current_pnl_pct <= floor_pct:
                return "profit_lock"
        if pos.candles_held >= self.cfg.max_hold_candles:
            return "timeout"
        return None

    def _close_position(self, sym: str, price: float, reason: str):
        """Cancel OCO, market-sell, record to trade log, remove from state.
        If the sell fails the position is kept — it will be retried next cycle.
        """
        pos = self.positions[sym]

        # Cancel server-side OCO first so it doesn't race with our sell
        self.cancel_oco(sym, pos.oco_order_list_id)

        # Fetch actual available balance — may differ from stored qty due to
        # partial fills, dust, or exchange rounding on the original buy
        try:
            balance_data = self.exchange.fetch_balance()
            coin = sym.replace("/USDT", "")
            actual_qty = float(balance_data.get(coin, {}).get("free") or 0)
            if actual_qty > 0 and abs(actual_qty - pos.qty) / max(pos.qty, 1e-9) > 0.01:
                log.warning(
                    f"Sell qty adjusted {pos.qty} → {actual_qty} for {sym} "
                    f"(stored qty differed from actual balance)"
                )
                pos.qty = actual_qty
                self.positions[sym] = pos  # persist corrected qty
        except Exception as e:
            log.warning(f"Could not verify sell qty for {sym}: {e} — using stored qty")
            actual_qty = pos.qty

        sell_qty = actual_qty if actual_qty > 0 else pos.qty
        order = self.place_sell(sym, sell_qty)

        if order is None:
            # Sell failed — keep position in store, retry next cycle
            log.error(
                f"Sell FAILED for {sym} — position kept, will retry next cycle. "
                f"Check Binance manually if this persists."
            )
            return

        pnl_pct  = (price - pos.entry_price) / pos.entry_price * 100
        pnl_usdt = (price - pos.entry_price) * sell_qty
        give_back = pos.peak_pnl_pct - pnl_pct
        log.info(
            f"CLOSED {sym} reason={reason} pnl={pnl_pct:+.2f}% "
            f"peak={pos.peak_pnl_pct:+.2f}% gaveback={give_back:.2f}%"
        )

        # Record this exit for the smart re-entry guard: remember the exit price
        # and whether the trade was a loss, so the entry logic can refuse to buy
        # the same coin back at a higher price after a red close.
        self._last_exit[sym] = {"price": price, "was_loss": pnl_pct < 0}

        # Apply re-entry cooldown for manual closes so the bot doesn't
        # immediately buy back something the user just intentionally closed.
        if reason == "manual":
            # Candle duration from TIMEFRAME string (e.g. "5m" → 300s, "15m" → 900s)
            tf = self.cfg.timeframe.lower()
            if tf.endswith("m"):
                candle_secs = int(tf[:-1]) * 60
            elif tf.endswith("h"):
                candle_secs = int(tf[:-1]) * 3600
            else:
                candle_secs = 300  # fallback: 5m
            cooldown_secs = candle_secs * self.cfg.manual_close_cooldown_candles
            self._cooldown[sym] = time.time() + cooldown_secs
            log.info(f"{sym} cooldown: no re-entry for {self.cfg.manual_close_cooldown_candles} candles (~{cooldown_secs//60}min)")

        self.trade_log.append(ClosedTrade(
            symbol=sym,
            entry_price=pos.entry_price,
            exit_price=price,
            qty=sell_qty,
            pnl_pct=round(pnl_pct, 4),
            pnl_usdt=round(pnl_usdt, 4),
            reason=reason,
            opened_at=pos.opened_at,
        ))
        del self.positions[sym]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def close_all_positions(self, reason: str = "kill_switch"):
        """Cancel all OCOs and immediately market-sell all open positions."""
        with self._pos_lock:
            for sym in list(self.positions.keys()):
                pos = self.positions[sym]
                try:
                    df = self.fetch_ohlcv(sym)
                    price = df["close"].iloc[-1] if df is not None else pos.entry_price
                except Exception:
                    price = pos.entry_price
                self._close_position(sym, price, reason)

    def run_cycle(self):
        self.last_cycle_ts = time.time()
        # Clear the per-cycle BTC regime cache so it's recomputed fresh this cycle.
        self._btc_regime_cache = None

        if self.kill_switch:
            if self.positions:
                log.warning("Kill switch active — closing all positions")
                self.close_all_positions()
            return

        symbols = self.get_candidate_symbols()
        balance = self.risk.get_available_usdt(self.exchange)
        self.last_balance = balance
        log.info(f"Available USDT: {balance:.2f}")

        # --- Manage existing positions ---
        # Fetch full balance once per cycle for server-side close detection
        try:
            live_balance = self.exchange.fetch_balance()
        except Exception as e:
            log.warning(f"Balance fetch failed — skipping server-side close check: {e}")
            live_balance = None

        for sym in list(self.positions.keys()):
            pos = self.positions[sym]

            # ── Server-side close detection ──────────────────────────────
            # If the stop-limit or OCO fired while the bot was running/restarting,
            # the coin balance will be zero. Detect and record cleanly.
            if live_balance is not None:
                coin = sym.replace("/USDT", "")
                coin_data = live_balance.get(coin) or {}
                if not isinstance(coin_data, dict):
                    coin_data = {}
                coin_free   = float(coin_data.get("free")  or 0)
                coin_used   = float(coin_data.get("used")  or 0)
                coin_total  = coin_free + coin_used  # used = locked in open orders
                notional = coin_total * pos.entry_price
                if notional < self.cfg.min_trade_usdt * 0.5:
                    # Balance is gone — server-side order must have fired.
                    # Fetch actual fill price from recent trade history for accuracy.
                    exit_price = pos.entry_price
                    try:
                        trades = self.exchange.fetch_my_trades(sym, limit=5)
                        if trades:
                            # Most recent trade is the server-side fill
                            last_trade = sorted(trades, key=lambda t: t["timestamp"])[-1]
                            exit_price = float(last_trade.get("price") or pos.entry_price)
                            log.info(f"Server-side fill price for {sym}: {exit_price} (from trade history)")
                    except Exception as e:
                        log.debug(f"Could not fetch trade history for {sym}: {e} — using ticker")
                        try:
                            ticker = self.exchange.fetch_ticker(sym)
                            exit_price = float(ticker["last"] or pos.entry_price)
                        except Exception:
                            exit_price = pos.entry_price

                    pnl_pct  = (exit_price - pos.entry_price) / pos.entry_price * 100
                    pnl_usdt = (exit_price - pos.entry_price) * pos.qty
                    log.warning(
                        f"SERVER-SIDE CLOSE detected for {sym}: "
                        f"free={coin_free} locked={coin_used} total={coin_total} notional={notional:.2f} "
                        f"exit={exit_price:.6f} pnl={pnl_pct:+.2f}% — recording and removing from store."
                    )
                    self.trade_log.append(ClosedTrade(
                        symbol=sym,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        qty=pos.qty,
                        pnl_pct=round(pnl_pct, 4),
                        pnl_usdt=round(pnl_usdt, 4),
                        reason="server_side",
                        opened_at=pos.opened_at,
                    ))
                    # Apply cooldown — prevents immediate re-entry on the same cycle
                    # that could double-up position if detection misfires
                    from datetime import timezone as _tz
                    candle_secs = 300  # 5m default
                    self._cooldown[sym] = time.time() + candle_secs
                    log.info(f"{sym} server-side cooldown: no re-entry for 1 candle")
                    del self.positions[sym]
                    continue

                # ── Live qty reconciliation ──────────────────────────────
                # Keep the in-memory qty honest against the actual Binance balance.
                # This prevents the dashboard from showing an inflated position value
                # if the holding changed outside the bot's normal flow (partial fill,
                # manual action, exchange adjustment). Only adjusts DOWN to what's
                # actually held — never invents coins. Uses total (free+used) so an
                # active backstop locking the coin doesn't look like a reduced balance.
                if coin_total > 0 and pos.qty > 0:
                    divergence = abs(coin_total - pos.qty) / pos.qty
                    if coin_total < pos.qty * 0.95 and divergence > 0.01:
                        log.warning(
                            f"{sym} qty reconciled {pos.qty} → {coin_total} "
                            f"(live balance below stored qty by {divergence*100:.1f}%)"
                        )
                        pos.qty = coin_total
                        self.positions[sym] = pos

            # ── Normal cycle management ───────────────────────────────────
            df = self.fetch_ohlcv(sym)
            if df is None:
                continue
            price = df["close"].iloc[-1]
            with self._pos_lock:
                # Re-check the position still exists — the fast monitor may have
                # closed it between the start of this iteration and acquiring the lock.
                if sym not in self.positions:
                    continue
                self.update_trailing_stop(sym, price)
                reason = self.check_exit(sym, price)
                if reason:
                    self._close_position(sym, price, reason)
                else:
                    self.positions.increment_candles(sym)

        # --- Look for new entries ---
        if len(self.positions) >= self.cfg.max_open_positions:
            return

        # Trading hours check — only restricts new entries, not position management
        if self.cfg.trading_hours_start and self.cfg.trading_hours_end:
            from datetime import datetime, timezone, time as dt_time
            now_utc = datetime.now(timezone.utc).time()
            try:
                start = dt_time(*map(int, self.cfg.trading_hours_start.split(":")))
                end   = dt_time(*map(int, self.cfg.trading_hours_end.split(":")))
                # Handle overnight windows e.g. 22:00–06:00
                if start <= end:
                    in_window = start <= now_utc <= end
                else:
                    in_window = now_utc >= start or now_utc <= end
                if not in_window:
                    log.debug(
                        f"Outside trading hours ({self.cfg.trading_hours_start}–"
                        f"{self.cfg.trading_hours_end} UTC) — skipping new entries"
                    )
                    return
            except ValueError as e:
                log.warning(f"Invalid trading hours format: {e} — trading unrestricted")

        # BTC market-regime check — compute once for this cycle (cached).
        # Always computed so it can be logged on entries; only ENFORCED when enabled.
        btc = self.compute_btc_regime()
        if self.btc_filter_enabled:
            if not btc["available"]:
                # Fail-open: if BTC data is unavailable, allow entries but warn.
                log.warning(
                    "BTC filter enabled but BTC trend unavailable — failing open "
                    "(entries allowed this cycle)"
                )
            elif btc["short_term_falling"]:
                log.info(
                    f"BTC short-term trend falling ({btc['short_term_change_pct']:+.2f}% "
                    f"over {self.btc_trend_lookback} candles, threshold "
                    f"-{abs(self.btc_trend_threshold_pct):.2f}%) — skipping new entries this cycle"
                )
                return

        for sym in symbols:
            if sym in self.positions:
                continue
            if len(self.positions) >= self.cfg.max_open_positions:
                break

            # Skip if symbol is in manual-close cooldown
            cooldown_until = self._cooldown.get(sym, 0)
            if time.time() < cooldown_until:
                remaining = int(cooldown_until - time.time())
                log.debug(f"{sym} skipped — manual close cooldown ({remaining}s remaining)")
                continue

            df = self.fetch_ohlcv(sym)
            if df is None:
                continue

            df = self.compute_indicators(df)
            if not self.passes_entry_filter(df):
                continue

            # Per-coin entry-timing gate — avoid chasing a coin extended above its
            # short-term mean (the whipsaw cause). Distance is captured for logging
            # whether or not the gate is enforced.
            timing_ok, timing_distance = self.passes_entry_timing(df)
            if not timing_ok:
                log.info(
                    f"{sym} skipped by entry-timing gate: price {timing_distance:+.2f}% "
                    f"above EMA{self.entry_timing_ema_len} (band {self.entry_timing_band_pct:.2f}%) "
                    f"— too extended, waiting for pullback"
                )
                continue

            # Momentum confirmation — is the coin actually rising right now?
            # Raw-price slope over the short lookback; slope captured for logging
            # whether or not the gate is enforced.
            momentum_ok, momentum_slope = self.passes_momentum(df)
            if not momentum_ok:
                slope_str = f"{momentum_slope:+.2f}%" if momentum_slope is not None else "n/a"
                log.info(
                    f"{sym} skipped by momentum gate: {self.momentum_lookback}-candle "
                    f"slope {slope_str} (min {self.momentum_min_slope_pct:.2f}%) "
                    f"or last candle red — not confirmed rising"
                )
                continue

            price = df["close"].iloc[-1]

            # Smart re-entry guard — after a RED close on this coin, refuse to buy
            # it back at a price HIGHER than the loss exit. This stops the bot from
            # chasing a coin it just lost on back up into the same rolling-over move
            # (the repeated-HOME churn pattern). A green prior close, or a re-entry
            # at/below the prior exit, is allowed.
            if self.reentry_guard_enabled:
                last = self._last_exit.get(sym)
                if last and last["was_loss"] and price > last["price"]:
                    log.info(
                        f"{sym} skipped by re-entry guard: last close was a loss at "
                        f"{last['price']:.6f}, would re-enter higher at {price:.6f} "
                        f"— not chasing a loser back up"
                    )
                    continue

            atr   = df["atr"].iloc[-1]
            alloc = self.risk.position_size_usdt(balance, atr, price)

            if alloc < self.cfg.min_trade_usdt:
                log.debug(f"{sym} allocation {alloc:.2f} below minimum, skipping")
                continue

            order = self.place_buy(sym, alloc, price)
            if not order:
                continue

            fill_price    = order.get("average") or price
            qty           = order.get("filled") or alloc / fill_price
            trailing_stop = fill_price * (1 - self.cfg.trailing_stop_pct / 100)

            # Place server-side backstop — OCO preferred, stop-market as fallback.
            # Binance's filled qty in the order response is pre-fee — the actual
            # balance is slightly less after trading fees (typically 0.1%).
            # Apply a 0.15% buffer and round DOWN to lot step to ensure we never
            # try to sell more than we actually hold.
            _, amount_prec = self.get_precision(sym)
            step = self._lot_step_size(sym)
            # The deliverable balance after fees (Binance deducts the trading fee from
            # the asset received unless paid in BNB). A server-side stop can sell at
            # most this much, floored to the lot step.
            post_fee_balance = qty * (1 - 0.0015)
            if step > 0:
                backstop_qty = self.round_to_step(post_fee_balance, step)
            else:
                backstop_qty = self.round_amount(post_fee_balance, amount_prec)

            # Detect MEANINGFUL partial coverage. Normal fee dust leaves a tiny sliver
            # uncovered (~0.1-0.15% of the position) which the in-memory trailing stop
            # mops up at exit — that is fine. The dangerous case is the whole-unit coin
            # where flooring to a step=1.0 lot leaves a large fraction exposed: e.g.
            # hold ~1.997 TAO, the stop covers only 1.0, leaving ~0.997 (~50% of the
            # position) unprotected — the TAO 2-bought/1-stopped bug. We flag partial
            # when the uncovered amount exceeds 1% of the holding (well above fee dust).
            # When partial is True the trailing stop must stay ACTIVE to cover the rest.
            if step > 0 and post_fee_balance > 0:
                uncovered_pct = (post_fee_balance - backstop_qty) / post_fee_balance * 100
                partial_backstop = uncovered_pct > 1.0
            else:
                partial_backstop = False
            log.debug(
                f"Backstop qty for {sym}: filled={qty} step={step} "
                f"post_fee={post_fee_balance:.6f} backstop_qty={backstop_qty} "
                f"partial={partial_backstop}"
            )

            backstop_type = None
            oco_id = self.place_oco(sym, backstop_qty, fill_price)
            if oco_id:
                backstop_type = "oco"
            else:
                if not self.cfg.oco_enabled:
                    log.info(f"OCO disabled — placing stop-market for {sym}")
                else:
                    log.debug(f"OCO returned None for {sym} — trying stop-market fallback")
                oco_id = self.place_stop_limit(sym, backstop_qty, fill_price)
                if oco_id:
                    backstop_type = "stop_market"

            if oco_id:
                log.info(f"✅ Server-side backstop confirmed for {sym}: type={backstop_type} id={oco_id}")
            else:
                log.warning(
                    f"⚠️  No server-side backstop placed for {sym} — "
                    f"trailing stop is the only protection"
                )

            # Determine trailing activation for this position based on the current
            # in-memory UI setting. If enabled, the trailing stop stays dormant until
            # price reaches entry * (1 + activation_pct%); until then the server-side
            # backstop is the protection.
            #
            # CRITICAL SAFETY RULES — a dormant trailing stop relies on the server-side
            # backstop. The position may only start dormant if the backstop FULLY covers
            # it. We force the trailing stop active immediately when either:
            #   (a) no backstop was placed (oco_id is None), or
            #   (b) the backstop only PARTIALLY covers the position (whole-unit coins
            #       where a fractional remainder can't be covered by a step=1.0 stop —
            #       e.g. hold 1.998 TAO, stop only covers 1.0, leaving 0.998 exposed).
            # In both cases starting dormant would leave coins unprotected.
            backstop_fully_covers = bool(oco_id) and not partial_backstop
            if self.trailing_activation_enabled and self.trailing_activation_pct > 0 and backstop_fully_covers:
                trailing_active = False
                activation_price = fill_price * (1 + self.trailing_activation_pct / 100)
            else:
                trailing_active = True
                activation_price = 0.0
                if self.trailing_activation_enabled and not backstop_fully_covers:
                    reason = "no server-side backstop" if not oco_id else \
                             f"backstop only covers {backstop_qty} of {qty} units"
                    log.warning(
                        f"{sym}: {reason} — forcing trailing stop ACTIVE immediately "
                        f"(overriding activation threshold) to avoid an unprotected position."
                    )

            with self._pos_lock:
                self.positions[sym] = PositionState(
                    entry_price=fill_price,
                    qty=qty,
                    trailing_stop=trailing_stop,
                    oco_order_list_id=oco_id,
                    backstop_type=backstop_type,
                    trailing_active=trailing_active,
                    activation_price=activation_price,
                    current_price=fill_price,
                    last_price_ts=time.time(),
                )
            # Stamp the BTC regime on every entry so we can later correlate
            # win-rate against BTC trend, whether or not the filter is enforced.
            btc_regime = self.compute_btc_regime()
            if btc_regime["available"]:
                btc_state = (
                    f"BTC[{btc_regime['short_term_change_pct']:+.2f}% "
                    f"{'falling' if btc_regime['short_term_falling'] else 'stable/up'}, "
                    f"regime={'bull' if btc_regime['regime_bullish'] else 'bear'}]"
                )
            else:
                btc_state = "BTC[unavailable]"

            timing_state = (
                f"EMA{self.entry_timing_ema_len}_dist="
                f"{timing_distance:+.2f}%" if timing_distance is not None else "EMA_dist=n/a"
            )
            momentum_state = (
                f"slope{self.momentum_lookback}="
                f"{momentum_slope:+.2f}%" if momentum_slope is not None else "slope=n/a"
            )

            log.info(
                f"ENTERED {sym} @ {fill_price:.6f} "
                f"trailing_stop={trailing_stop:.6f} "
                f"oco_stop={fill_price * (1 - self.cfg.oco_stop_pct / 100):.6f} "
                f"oco_id={oco_id} "
                f"trailing_active={trailing_active} "
                f"{btc_state} {timing_state} {momentum_state}"
                + (f" activation_price={activation_price:.6f}" if not trailing_active else "")
            )

    def monitor_positions_once(self):
        """
        One pass of the fast monitor: for each open position, fetch the LIVE
        ticker price (the same source the dashboard uses), ratchet the peak P&L,
        and check the trailing stop / profit lock. Runs far more often than the
        main trading cycle so fast spikes are captured and the profit lock fires
        near the true peak. All position mutation happens under the position lock
        so it never races the main cycle.
        """
        # Snapshot the symbols so we don't hold the lock during network calls.
        symbols = list(self.positions.keys())

        for sym in symbols:
            # Fetch the live price OUTSIDE the lock (network I/O can be slow).
            try:
                ticker = self.exchange.fetch_ticker(sym)
                price = float(ticker["last"] or 0)
            except Exception as e:
                log.debug(f"Monitor: ticker fetch failed for {sym}: {e}")
                continue
            if price <= 0:
                continue
            with self._pos_lock:
                # The position may have closed (main cycle or a prior monitor pass)
                # while we were fetching — re-check before acting.
                if sym not in self.positions:
                    continue
                self.update_trailing_stop(sym, price)
                reason = self.check_exit(sym, price)
                if reason:
                    log.info(f"Monitor triggered exit for {sym}: {reason}")
                    self._close_position(sym, price, reason)

    def _refresh_balance(self):
        """Update the engine's available-USDT figure (single source of truth)."""
        try:
            raw = self.exchange.fetch_balance()
            free_usdt = raw.get("USDT", {}).get("free")
            if free_usdt is not None:
                self.last_balance = float(free_usdt)
        except Exception as e:
            log.debug(f"Monitor: balance refresh failed: {e}")

    def _monitor_loop(self):
        """Daemon loop running the fast peak/exit monitor."""
        log.info(f"Fast monitor started (interval {self.monitor_interval}s).")
        while True:
            try:
                # Keep available USDT fresh every pass, whether flat or in trades,
                # so the dashboard's portfolio total reads one consistent source.
                self._refresh_balance()
                if not self.kill_switch and self.positions:
                    self.monitor_positions_once()
            except Exception as e:
                log.exception(f"Error in monitor loop: {e}")
            time.sleep(self.monitor_interval)

    def run(self):
        log.info("Bot started. Press Ctrl+C to stop.")
        self.recover_positions()
        # Seed the balance once up front so the dashboard shows a real figure
        # immediately, before the first monitor pass refreshes it.
        self._refresh_balance()
        # Start the fast peak/exit monitor as a daemon thread so it runs
        # independently of (and far more often than) the main trading cycle.
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                log.info("Shutdown requested.")
                break
            except Exception as e:
                log.exception(f"Unexpected error in run_cycle: {e}")
            time.sleep(self.cfg.poll_interval)
