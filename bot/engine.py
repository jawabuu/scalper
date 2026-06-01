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
        self.trade_log = TradeLog()
        self.last_balance: float = 0.0
        self.last_cycle_ts: float = 0.0

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

            # Fetch current price for notional check
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker["last"] or pos.entry_price)
            except Exception:
                price = pos.entry_price

            notional = free * price

            if notional < self.cfg.min_trade_usdt * 0.5:
                # Balance gone — position was likely closed manually or OCO fired
                log.warning(
                    f"Recovery: {symbol} has no balance on Binance "
                    f"(free={free} notional={notional:.2f}) — discarding from store."
                )
                del self.positions[symbol]
                discarded += 1
            else:
                # Cancel any open OCO — trailing stop takes over
                self._cancel_open_orders(symbol)

                # If Binance shows MORE than the stored qty, trust the store.
                # The excess could be a manual top-up, a testnet quirk, or a
                # partial fill discrepancy — we only manage what the bot entered.
                # If Binance shows LESS (partial sell, dust), trust Binance.
                if free > pos.qty * 1.05:
                    log.warning(
                        f"Recovery: {symbol} Binance balance ({free}) exceeds stored qty "
                        f"({pos.qty}) by >5% — managing stored qty only. "
                        f"Excess {free - pos.qty:.6f} units are unmanaged."
                    )
                    # qty stays as stored — don't update
                elif free < pos.qty * 0.95:
                    log.warning(
                        f"Recovery: {symbol} qty adjusted down {pos.qty} → {free} "
                        f"(partial sell or dust detected)"
                    )
                    pos.qty = free
                    self.positions[symbol] = pos

                log.warning(
                    f"RECOVERED {symbol}: entry={pos.entry_price:.6f} "
                    f"stop={pos.trailing_stop:.6f} qty={pos.qty} candles={pos.candles_held}"
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

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema20"] = ta.ema(df["close"], length=20)
        df["ema50"] = ta.ema(df["close"], length=50)
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
        qty = self.round_amount(qty, amount_prec)

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
        new_stop = current_price * (1 - self.cfg.trailing_stop_pct / 100)
        if new_stop > pos.trailing_stop:
            self.positions.update_stop(sym, new_stop)
            log.debug(f"{sym} trailing stop raised to {new_stop:.6f}")

    def check_exit(self, sym: str, current_price: float) -> str | None:
        pos = self.positions[sym]
        # Take profit — only checked when enabled.
        # When disabled the trailing stop is the sole profit exit,
        # allowing winners to run as far as momentum carries them.
        if self.cfg.take_profit_enabled:
            tp = pos.entry_price * (1 + self.cfg.take_profit_pct / 100)
            if current_price >= tp:
                return "take_profit"
        if current_price <= pos.trailing_stop:
            return "trailing_stop"
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
        log.info(f"CLOSED {sym} reason={reason} pnl={pnl_pct:+.2f}%")

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
                coin_free = float(
                    (live_balance.get(coin) or {}).get("free") or 0
                )
                notional = coin_free * pos.entry_price
                if notional < self.cfg.min_trade_usdt * 0.5:
                    # Balance is gone — server-side order must have fired
                    try:
                        ticker = self.exchange.fetch_ticker(sym)
                        exit_price = float(ticker["last"] or pos.entry_price)
                    except Exception:
                        exit_price = pos.entry_price
                    pnl_pct  = (exit_price - pos.entry_price) / pos.entry_price * 100
                    pnl_usdt = (exit_price - pos.entry_price) * pos.qty
                    log.warning(
                        f"SERVER-SIDE CLOSE detected for {sym}: "
                        f"balance={coin_free} notional={notional:.2f} "
                        f"pnl={pnl_pct:+.2f}% — recording and removing from store."
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
                    # Cancel any remaining orders for this symbol
                    self._cancel_open_orders(sym)
                    del self.positions[sym]
                    continue

            # ── Normal cycle management ───────────────────────────────────
            df = self.fetch_ohlcv(sym)
            if df is None:
                continue
            price = df["close"].iloc[-1]
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

            price = df["close"].iloc[-1]
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
            qty_after_fees = qty * (1 - 0.0015)  # 0.15% buffer covers standard + BNB fees
            backstop_qty = self.round_to_step(qty_after_fees, step) if step > 0                 else self.round_amount(qty_after_fees, amount_prec)
            log.debug(f"Backstop qty for {sym}: filled={qty} after_fees={qty_after_fees:.6f} rounded={backstop_qty}")

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

            self.positions[sym] = PositionState(
                entry_price=fill_price,
                qty=qty,
                trailing_stop=trailing_stop,
                oco_order_list_id=oco_id,
                backstop_type=backstop_type,
            )
            log.info(
                f"ENTERED {sym} @ {fill_price:.6f} "
                f"trailing_stop={trailing_stop:.6f} "
                f"oco_stop={fill_price * (1 - self.cfg.oco_stop_pct / 100):.6f} "
                f"oco_id={oco_id}"
            )

    def run(self):
        log.info("Bot started. Press Ctrl+C to stop.")
        self.recover_positions()
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                log.info("Shutdown requested.")
                break
            except Exception as e:
                log.exception(f"Unexpected error in run_cycle: {e}")
            time.sleep(self.cfg.poll_interval)
