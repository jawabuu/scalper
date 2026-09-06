"""
Auto-trade decision logic.

Pure functions: given a candidate, its recent behaviour and the account state,
decide whether to open a position and with what trailing callback. No exchange
calls here, so every rule and every safety limit is unit-testable.

The operator's rules:

  1. Price within `max_dist_to_extreme_pct` of the relevant 24h extreme —
     the 24h low for a long, the 24h high for a short.
  2. The candidate must have STRENGTHENED on the last N consecutive scans
     (RSI moving further in the trade's favour), not merely appeared once.
  3. RSI above `long_rsi_min` for longs, above `short_rsi_min` for shorts.
  4. The trailing callback is HALF the distance to that extreme — so a short
     2% below the 24h high trails at 1%. This ties the stop to structure
     rather than a fixed number.

Bounds that the operator's rules do not cover, added deliberately:

  * Binance accepts a callbackRate between 0.1% and 5%; the half-distance can
    fall outside that range and is clamped.
  * A callback inside one ATR is inside the coin's normal candle range and
    would be hit for reasons unrelated to the thesis, so it is floored at a
    multiple of ATR.
  * Automation runs unattended, so it also honours a daily loss limit, a
    per-symbol cooldown after a loss, and a trade-rate cap. Without these a
    bad run compounds with nobody watching.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Binance trailing-stop callbackRate limits.
MIN_CALLBACK_PCT = 0.1
MAX_CALLBACK_PCT = 5.0


@dataclass
class AutoTradeConfig:
    enabled: bool = False

    # ── Entry rules ─────────────────────────────────────────────────────
    max_dist_to_extreme_pct: float = 3.0   # within 3% of the 24h low/high
    required_strength_sweeps: int = 2      # strengthened on N consecutive scans
    long_rsi_min: float = 48.0
    short_rsi_min: float = 78.0

    # ── Trailing callback ───────────────────────────────────────────────
    callback_ratio: float = 0.5            # half the distance to the extreme
    callback_atr_mult: float = 0.75        # floor: callback >= this x ATR
    callback_min_pct: float = MIN_CALLBACK_PCT
    callback_max_pct: float = MAX_CALLBACK_PCT

    # ── Safety limits for unattended running ────────────────────────────
    daily_loss_limit_pct: float = 5.0      # halt for the day at -5% of wallet
    symbol_cooldown_s: float = 1800.0      # 30 min before retrying a loser
    # A cooldown on a pure timer discards real information: a coin that stopped
    # you out and has since pushed FURTHER into the extreme is a stronger fade
    # than the one that failed. So the cooldown can be overridden when the
    # signal has genuinely improved by this much RSI versus the failed entry.
    # 0 disables the override (strict timer).
    cooldown_override_rsi_delta: float = 3.0
    # But the loop still has to be bounded. Re-entering a coin that keeps
    # running against you is how a fade strategy dies, so a symbol can only be
    # retried this many times per day however good the signal looks.
    max_reentries_per_symbol: int = 2
    max_trades_per_hour: int = 6
    max_open_positions: int = 3


@dataclass
class AutoDecision:
    enter: bool
    symbol: str = ""
    side: str = ""
    callback_pct: float | None = None
    reason: str = ""
    notes: list[str] = field(default_factory=list)


def distance_to_extreme(candidate_row: dict, side: str) -> float | None:
    """
    How far price is from the extreme that matters for this direction, as a
    positive percentage. Long -> above the 24h low. Short -> below the 24h high.
    """
    if side == "long":
        v = candidate_row.get("pct_above_24h_low")
    else:
        v = candidate_row.get("pct_below_24h_high")
    if v is None:
        return None
    return abs(float(v))


def callback_for(distance_pct: float, atr_pct: float | None,
                 cfg: AutoTradeConfig) -> tuple[float, list[str]]:
    """
    Trailing callback = half the distance to the extreme, bounded.

    Returns (callback_pct, notes) where notes explain any clamping, so a
    surprising value on a filled order can be traced.
    """
    notes: list[str] = []
    cb = distance_pct * cfg.callback_ratio

    if atr_pct and cfg.callback_atr_mult:
        floor = atr_pct * cfg.callback_atr_mult
        if cb < floor:
            notes.append(
                f"callback {cb:.2f}% was inside {cfg.callback_atr_mult:g}x ATR "
                f"({atr_pct:.2f}%) — raised to {floor:.2f}%")
            cb = floor

    if cb < cfg.callback_min_pct:
        notes.append(f"callback raised to the {cfg.callback_min_pct}% exchange minimum")
        cb = cfg.callback_min_pct
    elif cb > cfg.callback_max_pct:
        notes.append(f"callback capped at the {cfg.callback_max_pct}% exchange maximum")
        cb = cfg.callback_max_pct

    return round(cb, 2), notes


def evaluate_candidate(row: dict, streak: int, cfg: AutoTradeConfig,
                       atr_pct: float | None = None) -> AutoDecision:
    """
    Apply the entry rules to one scanner candidate.

    `streak` is how many consecutive scans this candidate has strengthened for.
    """
    symbol = row.get("symbol", "")
    side = row.get("direction", "")
    if side not in ("long", "short"):
        return AutoDecision(False, symbol, side, reason=f"unknown direction {side!r}")

    rsi = row.get("rsi")
    if rsi is None:
        return AutoDecision(False, symbol, side, reason="no RSI")
    floor = cfg.long_rsi_min if side == "long" else cfg.short_rsi_min
    if float(rsi) < floor:
        return AutoDecision(False, symbol, side,
                            reason=f"RSI {rsi} below the {side} floor of {floor}")

    if streak < cfg.required_strength_sweeps:
        return AutoDecision(False, symbol, side,
                            reason=f"strengthened on {streak} scan(s), "
                                   f"needs {cfg.required_strength_sweeps}")

    dist = distance_to_extreme(row, side)
    if dist is None:
        return AutoDecision(False, symbol, side,
                            reason="24h range unavailable — cannot size the callback")
    if dist > cfg.max_dist_to_extreme_pct:
        extreme = "24h low" if side == "long" else "24h high"
        return AutoDecision(False, symbol, side,
                            reason=f"{dist:.2f}% from the {extreme}, "
                                   f"limit {cfg.max_dist_to_extreme_pct}%")

    cb, notes = callback_for(dist, atr_pct, cfg)
    extreme = "24h low" if side == "long" else "24h high"
    return AutoDecision(
        True, symbol, side, callback_pct=cb,
        reason=(f"RSI {rsi}, strengthened {streak} scans, {dist:.2f}% from the "
                f"{extreme} -> {cb}% callback"),
        notes=notes,
    )


# ── Streak tracking ─────────────────────────────────────────────────────────

class StrengthTracker:
    """
    Counts consecutive scans on which a candidate strengthened.

    A single strengthening scan is noise; the operator's rule is that a setup
    must build over successive sweeps before it is acted on.
    """

    def __init__(self):
        self._streaks: dict[str, int] = {}

    def update(self, rows: list[dict]) -> dict[str, int]:
        seen = set()
        for row in rows:
            key = f"{row.get('symbol')}:{row.get('direction')}"
            seen.add(key)
            strength = row.get("strength")
            if strength in ("strengthening", "CONFIRMED"):
                self._streaks[key] = self._streaks.get(key, 0) + 1
            else:
                self._streaks[key] = 0
        # A candidate that dropped out of the list loses its streak.
        for key in list(self._streaks):
            if key not in seen:
                del self._streaks[key]
        return dict(self._streaks)

    def streak(self, symbol: str, side: str) -> int:
        return self._streaks.get(f"{symbol}:{side}", 0)


# ── Safety gate ─────────────────────────────────────────────────────────────

import time as _time


@dataclass
class SafetyState:
    """Running state the safety limits are evaluated against."""
    day_start_balance: float = 0.0
    day_key: str = ""
    recent_entry_times: list[float] = field(default_factory=list)
    symbol_blocked_until: dict[str, float] = field(default_factory=dict)
    # RSI at the entry that failed, per symbol — the bar a re-entry must beat.
    failed_entry_rsi: dict[str, float] = field(default_factory=dict)
    reentries_today: dict[str, int] = field(default_factory=dict)
    halted_reason: str | None = None


def _day_key(now: float) -> str:
    return _time.strftime("%Y-%m-%d", _time.gmtime(now))


def roll_day(state: SafetyState, balance: float, now: float | None = None) -> SafetyState:
    """Reset the daily baseline (and any halt) when the UTC day changes."""
    now = now or _time.time()
    key = _day_key(now)
    if state.day_key != key:
        state.day_key = key
        state.day_start_balance = balance
        state.halted_reason = None
        state.reentries_today = {}
    if state.day_start_balance <= 0:
        state.day_start_balance = balance
    return state


def check_safety(state: SafetyState, cfg: AutoTradeConfig, *, balance: float,
                 open_positions: int, symbol: str,
                 current_rsi: float | None = None, side: str = "short",
                 now: float | None = None) -> tuple[bool, str]:
    """
    Whether an automated entry is permitted right now.

    These limits exist because nobody is watching. A losing run with no daily
    stop compounds; a symbol that just stopped out is the one most likely to
    stop out again; and an unbounded trade rate can burn the account in fees
    alone if the entry rules start matching too often.
    """
    now = now or _time.time()

    if state.halted_reason:
        return False, state.halted_reason

    if cfg.daily_loss_limit_pct and state.day_start_balance > 0:
        drawdown = (state.day_start_balance - balance) / state.day_start_balance * 100
        if drawdown >= cfg.daily_loss_limit_pct:
            state.halted_reason = (
                f"daily loss limit hit: down {drawdown:.1f}% from "
                f"{state.day_start_balance:.2f} — auto-trade halted until "
                f"tomorrow (UTC) or a manual reset"
            )
            return False, state.halted_reason

    if open_positions >= cfg.max_open_positions:
        return False, f"at the {cfg.max_open_positions}-position limit"

    blocked_until = state.symbol_blocked_until.get(symbol, 0)
    if now < blocked_until:
        # The cooldown can be overridden by a genuinely stronger signal, but
        # only up to the per-symbol re-entry cap.
        prior = state.failed_entry_rsi.get(symbol)
        improved = (
            cfg.cooldown_override_rsi_delta
            and prior is not None
            and current_rsi is not None
            and abs(current_rsi - prior) >= cfg.cooldown_override_rsi_delta
            and ((current_rsi > prior) if side == "short" else (current_rsi > prior))
        )
        used = state.reentries_today.get(symbol, 0)
        if improved and used < cfg.max_reentries_per_symbol:
            return True, (f"cooldown overridden: RSI {current_rsi:.0f} vs "
                          f"{prior:.0f} at the failed entry "
                          f"(re-entry {used + 1}/{cfg.max_reentries_per_symbol})")
        if improved:
            return False, (f"{symbol} signal improved but already retried "
                           f"{used}/{cfg.max_reentries_per_symbol} times today")
        return False, (f"{symbol} in cooldown for another "
                       f"{int(blocked_until - now)}s after a loss")

    cutoff = now - 3600
    state.recent_entry_times = [t for t in state.recent_entry_times if t >= cutoff]
    if cfg.max_trades_per_hour and len(state.recent_entry_times) >= cfg.max_trades_per_hour:
        return False, (f"{len(state.recent_entry_times)} trades in the last hour, "
                       f"limit {cfg.max_trades_per_hour}")

    return True, "ok"


def record_entry(state: SafetyState, now: float | None = None) -> SafetyState:
    state.recent_entry_times.append(now or _time.time())
    return state


def record_loss(state: SafetyState, symbol: str, cfg: AutoTradeConfig,
                entry_rsi: float | None = None,
                now: float | None = None) -> SafetyState:
    """
    Block a symbol after a losing trade, remembering the RSI it failed at so a
    later re-entry can be judged against it rather than on a bare timer.
    """
    now = now or _time.time()
    if cfg.symbol_cooldown_s:
        state.symbol_blocked_until[symbol] = now + cfg.symbol_cooldown_s
    if entry_rsi is not None:
        state.failed_entry_rsi[symbol] = float(entry_rsi)
    return state


def record_reentry(state: SafetyState, symbol: str) -> SafetyState:
    state.reentries_today[symbol] = state.reentries_today.get(symbol, 0) + 1
    return state


# ── Runner ──────────────────────────────────────────────────────────────────

import logging as _logging

_log = _logging.getLogger("auto_trader")


class AutoTrader:
    """
    Drives unattended entries from scanner output.

    Deliberately does NOT own its own market data or order placement — it reuses
    the scanner's candidates and the entry service's preview/execute path, so an
    automated entry passes through exactly the same guardrails and sizing as a
    manual one. Automation adds a decision, not a second code path.
    """

    def __init__(self, cfg: AutoTradeConfig, scanner, entry_service, guardian):
        self.cfg = cfg
        self.scanner = scanner
        self.entry = entry_service
        self.guardian = guardian
        self.tracker = StrengthTracker()
        self.state = SafetyState()
        self._log: list[dict] = []
        self._last_run: float = 0.0

    # -- reporting -------------------------------------------------------
    def _record(self, action: str, detail: str, symbol: str = ""):
        self._log.append({"ts": _time.time(), "action": action,
                          "symbol": symbol, "detail": detail})
        self._log = self._log[-40:]

    def safety_snapshot(self) -> dict:
        """SafetyState in a persistable form — the daily halt must survive a restart."""
        return {
            "day_start_balance": self.state.day_start_balance,
            "day_key": self.state.day_key,
            "recent_entry_times": list(self.state.recent_entry_times),
            "symbol_blocked_until": dict(self.state.symbol_blocked_until),
            "failed_entry_rsi": dict(self.state.failed_entry_rsi),
            "reentries_today": dict(self.state.reentries_today),
            "halted_reason": self.state.halted_reason,
        }

    def restore_safety(self, data: dict):
        """
        Reinstate the safety counters. Without this a restart rebases the daily
        loss baseline, so a bad day could be reset simply by redeploying.
        """
        if not data:
            return
        s = self.state
        s.day_start_balance = float(data.get("day_start_balance") or 0.0)
        s.day_key = data.get("day_key") or ""
        s.recent_entry_times = list(data.get("recent_entry_times") or [])
        s.symbol_blocked_until = dict(data.get("symbol_blocked_until") or {})
        s.failed_entry_rsi = dict(data.get("failed_entry_rsi") or {})
        s.reentries_today = dict(data.get("reentries_today") or {})
        s.halted_reason = data.get("halted_reason")
        if s.halted_reason:
            _log.warning(f"Auto-trade remains HALTED after restart: {s.halted_reason}")

    def snapshot(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "halted_reason": self.state.halted_reason,
            "day_start_balance": round(self.state.day_start_balance, 2),
            "trades_last_hour": len(self.state.recent_entry_times),
            "cooldowns": {k: int(v - _time.time())
                          for k, v in self.state.symbol_blocked_until.items()
                          if v > _time.time()},
            "last_run_ago_s": (_time.time() - self._last_run) if self._last_run else None,
            "recent": list(reversed(self._log[-15:])),
            "config": {
                "max_dist_to_extreme_pct": self.cfg.max_dist_to_extreme_pct,
                "required_strength_sweeps": self.cfg.required_strength_sweeps,
                "long_rsi_min": self.cfg.long_rsi_min,
                "short_rsi_min": self.cfg.short_rsi_min,
                "callback_ratio": self.cfg.callback_ratio,
                "daily_loss_limit_pct": self.cfg.daily_loss_limit_pct,
                "max_open_positions": self.cfg.max_open_positions,
                "max_trades_per_hour": self.cfg.max_trades_per_hour,
                "callback_atr_mult": self.cfg.callback_atr_mult,
                "cooldown_override_rsi_delta": self.cfg.cooldown_override_rsi_delta,
                "max_reentries_per_symbol": self.cfg.max_reentries_per_symbol,
                "symbol_cooldown_s": self.cfg.symbol_cooldown_s,
            },
            # Re-entries used today, so the cap is visible before it bites.
            "reentries_today": dict(self.state.reentries_today),
        }

    def set_enabled(self, on: bool) -> dict:
        self.cfg.enabled = bool(on)
        _log.warning(f"AUTO-TRADE {'ENABLED' if on else 'DISABLED'} by operator")
        self._record("toggle", "enabled" if on else "disabled")
        return self.snapshot()

    # Entry rules are live-tunable; SAFETY limits are deliberately not.
    # The daily loss stop, per-symbol cooldown and trade-rate cap exist to
    # bound a bad run, and the moment you most want to relax them from a
    # dashboard — right after a halt fires — is exactly when you should not.
    # Those stay env-only so changing them is a conscious redeploy.
    TUNABLE = {
        "max_dist_to_extreme_pct": (float, 0.1, 50.0),
        "required_strength_sweeps": (int, 1, 10),
        "long_rsi_min": (float, 0.0, 100.0),
        "short_rsi_min": (float, 0.0, 100.0),
        "callback_ratio": (float, 0.05, 2.0),
        "callback_atr_mult": (float, 0.0, 5.0),
        # How much stronger a signal must be to override a cooldown. This is an
        # entry-quality judgement, so it is tunable; the re-entry CAP is not.
        "cooldown_override_rsi_delta": (float, 0.0, 50.0),
    }
    SAFETY_ONLY = {"daily_loss_limit_pct", "symbol_cooldown_s",
                   "max_trades_per_hour", "max_open_positions",
                   "max_reentries_per_symbol"}

    def update_rules(self, payload: dict) -> tuple[dict, list[str]]:
        """
        Apply live edits to the entry rules. Returns (applied, errors).

        Nothing is applied if any value is invalid, so a bad edit cannot leave
        the rules half-changed while the bot is trading on them.
        """
        applied: dict = {}
        errors: list[str] = []

        for key, raw in (payload or {}).items():
            if key in self.SAFETY_ONLY:
                errors.append(
                    f"{key} is a safety limit and can only be changed via the "
                    f"environment, not from the dashboard")
                continue
            if key not in self.TUNABLE:
                continue
            typ, lo, hi = self.TUNABLE[key]
            try:
                val = typ(raw)
            except (TypeError, ValueError):
                errors.append(f"{key} must be a {typ.__name__}")
                continue
            if not (lo <= val <= hi):
                errors.append(f"{key} must be between {lo} and {hi}")
                continue
            applied[key] = val

        if errors:
            return {}, errors

        for key, val in applied.items():
            setattr(self.cfg, key, val)
        if applied:
            _log.warning(f"Auto-trade rules updated: {applied}")
            self._record("rules_updated", ", ".join(f"{k}={v}" for k, v in applied.items()))
        return applied, []

    def reset_halt(self) -> dict:
        self.state.halted_reason = None
        self._record("reset", "halt cleared by operator")
        return self.snapshot()

    # -- main loop -------------------------------------------------------
    def run_once(self):
        self._last_run = _time.time()
        if not self.cfg.enabled:
            return

        snap = self.scanner.snapshot()
        rows = snap.get("candidates") or []
        self.tracker.update(rows)

        try:
            balance = self.entry.wallet_balance()
            positions = self.guardian.fetch_positions()
        except Exception as e:
            _log.warning(f"auto-trade: account read failed: {e}")
            return

        roll_day(self.state, balance)
        open_syms = {p.symbol for p in positions}

        for row in rows:
            symbol = row.get("symbol", "")
            side = row.get("direction", "")
            if symbol in open_syms:
                continue

            streak = self.tracker.streak(symbol, side)
            decision = evaluate_candidate(
                row, streak, self.cfg, atr_pct=row.get("atr_pct"))
            if not decision.enter:
                continue

            ok, why = check_safety(self.state, self.cfg, balance=balance,
                                   open_positions=len(positions), symbol=symbol,
                                   current_rsi=row.get("rsi"), side=side)
            if not ok:
                _log.info(f"auto-trade: {symbol} qualified but blocked — {why}")
                self._record("blocked", why, symbol)
                continue

            # Route through the SAME preview/execute path a manual entry uses,
            # so every guardrail (margin cap, leverage resolution, duplicate
            # position check) applies identically.
            prev = self.entry.preview(symbol=symbol, side=side,
                                      callback_pct=decision.callback_pct)
            if not prev.get("ok"):
                errs = "; ".join(prev.get("errors", []))
                _log.warning(f"auto-trade: {symbol} preview refused — {errs}")
                self._record("preview_refused", errs, symbol)
                continue

            token = (prev.get("plan") or {}).get("token")
            res = self.entry.execute(token) if token else {"ok": False,
                                                           "error": "no token"}
            if res.get("ok"):
                record_entry(self.state)
                if "cooldown overridden" in why:
                    record_reentry(self.state, symbol)
                # Remember the RSI this entry was taken at, so if it fails the
                # next attempt is judged against it.
                self.state.failed_entry_rsi.setdefault(symbol, row.get("rsi") or 0.0)
                # Hand the guardian the exact conditions this entry acted on,
                # so the trade can later be analysed by what triggered it.
                try:
                    self.guardian.note_entry_context(symbol, {
                        "rsi": row.get("rsi"),
                        "atr_pct": row.get("atr_pct"),
                        "dist_to_extreme_pct": abs(
                            row.get("pct_above_24h_low") if side == "long"
                            else row.get("pct_below_24h_high")),
                        "range_pos_24h": row.get("range_pos_24h"),
                        "ema_gap_pct": row.get("ema_gap_pct"),
                        "strength": row.get("strength"),
                        "streak": streak,
                        "callback_pct": decision.callback_pct,
                        "was_reentry": "cooldown overridden" in why,
                        "auto": True,
                    })
                except Exception as e:
                    _log.debug(f"could not record entry context for {symbol}: {e}")
                positions.append(object())     # count it toward the cap now
                note = " | ".join(decision.notes) if decision.notes else ""
                _log.warning(
                    f"AUTO-ENTRY {side.upper()} {symbol}: {decision.reason}"
                    f"{' (' + note + ')' if note else ''}"
                    f"{' [DRY RUN]' if res.get('dry_run') else ''}")
                self._record("entered",
                             f"{side} · {decision.reason}"
                             + (f" · {note}" if note else ""), symbol)
            else:
                self._record("execute_failed", str(res.get("error")), symbol)

    def note_closed_trade(self, symbol: str, realised: float | None,
                          entry_rsi: float | None = None):
        """Apply the post-loss cooldown when a position closes down."""
        if realised is not None and realised < 0:
            record_loss(self.state, symbol, self.cfg,
                        entry_rsi=entry_rsi or self.state.failed_entry_rsi.get(symbol))
            self._record("cooldown", f"loss on {symbol}; blocked for "
                                     f"{int(self.cfg.symbol_cooldown_s)}s", symbol)
