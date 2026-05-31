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

    def place_buy(self, symbol: str, usdt_amount: float, current_price: float) -> dict | None:
        _, amount_prec = self.get_precision(symbol)
        qty = self.round_amount(usdt_amount / current_price, amount_prec)
        if qty <= 0:
            # Allocation too small relative to coin price and lot size precision.
            # Common for high-price coins (BNB, BTC) with small balance — not an error.
            log.debug(f"Skipping {symbol}: allocation ${usdt_amount:.2f} too small for lot size (qty rounds to 0)")
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

    def cancel_oco(self, symbol: str, list_id: str | None):
        """Cancel a previously placed OCO order before the bot executes its own exit."""
        if not list_id:
            return
        try:
            self.exchange.cancel_order(list_id, symbol, params={"orderListId": list_id})
            log.info(f"OCO cancelled for {symbol} listId={list_id}")
        except Exception as e:
            # May already be filled or cancelled — log and move on
            log.debug(f"OCO cancel for {symbol} listId={list_id}: {e}")

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
        for sym in list(self.positions.keys()):
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

            # Place OCO backstop — wider than trailing stop, lives on Binance servers
            oco_id = self.place_oco(sym, qty, fill_price)

            self.positions[sym] = PositionState(
                entry_price=fill_price,
                qty=qty,
                trailing_stop=trailing_stop,
                oco_order_list_id=oco_id,
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
