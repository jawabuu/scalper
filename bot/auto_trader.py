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
    # Floor the entry callback on CURRENT velocity rather than ATR(14) when
    # velocity is the larger of the two. Off by default: it changes the price
    # at which entries trigger.
    # LONGS ONLY: require the EMA gap to be NARROWING toward a cross before
    # buying. The scanner's long branch is commented "recovering, with EMA9
    # below EMA21 OR just above it", but it only tested `gap <
    # ema_tolerance_pct` — satisfied by any downtrend however deep. A coin
    # down 14% with EMA9 under EMA21 and the gap still widening is not
    # recovering; it is falling.
    #
    # Default ON. Not a new hypothesis — it is the condition the code already
    # claimed to apply. Shorts are untouched: fading an RSI extreme works off
    # the stretch itself and does not need the EMAs to have turned. A long has
    # no equivalent, because "oversold" bounds nothing.
    # Longs must also show the TURN, not merely a shrinking gap.
    # Defer a SHORT while volume is still expanding into the high. Shorts
    # only — see evaluate_candidate() for why longs are not mirrored.
    defer_on_rising_volume: bool = True
    defer_vol_trend: float = 1.0
    # Lower bar when the heaviest volume bar sat in the SECOND half of the
    # advance. 0 disables this second test.
    defer_vol_late_trend: float = 1.5
    long_require_turn: bool = True
    long_require_convergence: bool = True
    short_require_turn: bool = False
    short_require_convergence: bool = False
    # off | all | short | long. Two booleans said the same thing twice and
    # could contradict each other; one setting cannot.
    #   short  velocity floor on shorts only (the default)
    #   all    both directions — the behaviour that reversed
    #   long   longs only
    #   off    disabled
    # Accepts the old booleans too: True -> "all", False -> "off".
    callback_use_velocity: object = "off"
    # Refuse candidates the scanner flagged as an EXPANDING move rather than an
    # exhausted one (price at the 24h extreme, consecutive candles carrying it,
    # EMA gap wide AND widening). scanner.breakout_structure() has computed this
    # on every candidate all along and nothing acted on it. OFF by default so it
    # can be A/B'd against the recorded verdict rather than assumed.
    veto_breakout: bool = False
    max_dist_to_extreme_pct: float = 3.0   # within 3% of the 24h low/high
    required_strength_sweeps: int = 2      # strengthened on N consecutive scans
    long_rsi_min: float = 48.0
    # Upper bound for longs. The auto-trader had a floor but no ceiling, so it
    # inherited the scanner's much wider band. Within a single session's longs
    # — same regime, differing only by entry RSI — the 45-50 band was the only
    # profitable one; 50-60 lost 13 USDT per trade across 11 trades. 0 disables
    # the ceiling.
    long_rsi_max: float = 0.0
    # Which directions may be traded: "all", "long" or "short". A fade
    # strategy's two halves can behave very differently in a given regime, so
    # being able to disable one without redeploying is worth having.
    directions: str = "all"
    # Whether the daily-loss halt is active at all. Disabling it removes the
    # only automatic brake on a bad day, so it is deliberately visible in the
    # dashboard rather than hidden in a config file.
    daily_halt_enabled: bool = True
    # Restrict entries to a window of the UTC day, e.g. "05:00-11:00". Empty
    # means trade around the clock. Exits are NEVER restricted — a position
    # opened inside the window is managed normally until it closes.
    # Per-session, per-direction switches. "L1S1" = longs and shorts both on.
    # Keys are the SESSIONS defined in analysis.py, on UTC hours — independent
    # of DAY_TZ_OFFSET_H, which only moves the daily accounting boundary.
    sessions: dict = field(default_factory=lambda: {
        "AS": "L1S1", "EU": "L1S1", "OV": "L1S1", "US": "L1S1"})
    # Minimum ATR to enter at all. A coin that barely moves cannot clear its
    # own round-trip fee: 68 entries below 0.3% ATR won 26.5% and lost 747
    # USDT, while 0.7-1.5% won 82.4% and made 285.
    min_atr_pct: float = 0.0
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
    # Which rule actually set the callback: the distance ratio, or the ATR
    # floor overriding it. Recorded so the two can be compared on outcomes
    # instead of impressions.
    callback_source: str = "ratio"


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


def velocity_mode(value) -> str:
    """
    Normalise AUTO_CALLBACK_USE_VELOCITY to one of off | all | short | long.

    Accepts the booleans it replaced so an existing deployment keeps working:
    True means both directions, False means disabled.
    """
    if value is True:
        return "all"
    if value is False or value is None:
        return "off"
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on", "all", "both"):
        return "all"
    if v in ("short", "shorts"):
        return "short"
    if v in ("long", "longs"):
        return "long"
    return "off"


def callback_for(distance_pct: float, atr_pct: float | None,
                 cfg: AutoTradeConfig,
                 recent_tr_pct: float | None = None,
                 side: str | None = None) -> tuple[float, list[str], str]:
    """
    Trailing callback = half the distance to the extreme, bounded.

    Returns (callback_pct, notes) where notes explain any clamping, so a
    surprising value on a filled order can be traced.
    """
    notes: list[str] = []
    source = "ratio"
    cb = distance_pct * cfg.callback_ratio

    # The floor is meant to keep the callback outside noise. ATR(14) is the
    # wrong yardstick for that on an accelerating move: STORJ was logged at
    # atr_pct 0.512 while its last candles were covering ~2.7%, so the floor
    # came out at 0.38% and the entry triggered on a wiggle, then the move
    # resumed 3.3% against it within seconds. Taking the larger of the two
    # keeps the floor honest when velocity is rising and changes nothing when
    # it is not — over a normal tape the two measures agree.
    vol_pct = atr_pct
    # SHORT ONLY by default. On 64 velocity-floored trades the nine LONGS
    # returned -$17.64 each — 14% of the population and 233% of the net loss —
    # while the 55 shorts returned +$1.65. The same pattern shows without
    # velocity: atr_floor longs returned -$13.00 while `ratio` longs returned
    # +$6.13, so it is ATR-based WIDENING of a long's callback that hurts, not
    # velocity as such. A long that waits for a deeper retrace enters after the
    # bounce has run.
    mode = velocity_mode(cfg.callback_use_velocity)
    use_velocity = mode != "off" and (
        mode == "all" or not side or side == mode)

    if use_velocity and recent_tr_pct and atr_pct:
        if recent_tr_pct > atr_pct:
            vol_pct = recent_tr_pct
            notes.append(
                f"velocity {recent_tr_pct:.2f}% exceeds ATR {atr_pct:.2f}% — "
                f"floor taken from recent range")
    elif use_velocity and recent_tr_pct and not atr_pct:
        vol_pct = recent_tr_pct

    if vol_pct and cfg.callback_atr_mult:
        floor = vol_pct * cfg.callback_atr_mult
        if cb < floor:
            notes.append(
                f"callback {cb:.2f}% was inside {cfg.callback_atr_mult:g}x "
                f"{'recent range' if vol_pct != atr_pct else 'ATR'} "
                f"({vol_pct:.2f}%) — raised to {floor:.2f}%")
            cb = floor
            source = ("velocity_floor"
                      if use_velocity and recent_tr_pct
                      and atr_pct and recent_tr_pct > atr_pct else "atr_floor")

    if cb < cfg.callback_min_pct:
        notes.append(f"callback raised to the {cfg.callback_min_pct}% exchange minimum")
        cb = cfg.callback_min_pct
        source = "exchange_min"
    elif cb > cfg.callback_max_pct:
        notes.append(f"callback capped at the {cfg.callback_max_pct}% exchange maximum")
        cb = cfg.callback_max_pct
        source = "exchange_max"

    return round(cb, 2), notes, source


def _stream_status(stream) -> dict:
    """Stream health for the dashboard. Absent is a normal state, not an error."""
    if stream is None:
        return {"enabled": False, "connected": False}
    try:
        out = stream.status()
        out["enabled"] = True
        return out
    except Exception as e:
        return {"enabled": True, "connected": False, "last_error": str(e)[:120]}


def live_drift(row: dict, live_price: float | None) -> tuple[float | None, float | None]:
    """
    How far the market has moved since the scan that produced this candidate.

    Returns (drift_pct, live_dist_to_extreme_pct). Both None when there is no
    live quote — the caller then proceeds on the scan figures, as it always
    did.

    The scan's own price is not recorded directly, but it is recoverable: the
    candidate carries dist_to_extreme_pct and the 24h extreme it was measured
    against, so the price the DECISION was made at can be reconstructed. LSK on
    2026-09-16 back-solved to 0.5239 while the order was sized at 0.4612.
    """
    try:
        if live_price is None or live_price <= 0:
            return None, None
        dist = row.get("dist_to_extreme_pct")
        side = row.get("direction") or row.get("side")
        hi, lo = row.get("high_24h"), row.get("low_24h")
        extreme = hi if side == "short" else lo
        if dist is None or not extreme:
            return None, None
        extreme = float(extreme)
        scan_price = (extreme * (1 - float(dist) / 100.0) if side == "short"
                      else extreme * (1 + float(dist) / 100.0))
        if scan_price <= 0:
            return None, None
        drift = (live_price - scan_price) / scan_price * 100.0
        live_dist = (abs(extreme - live_price) / extreme * 100.0)
        return round(drift, 3), round(live_dist, 3)
    except Exception:
        return None, None


def _refusal_key(reason: str) -> str:
    """
    Collapse a refusal reason to the RULE that produced it.

    The full text carries the candidate's numbers, so counting raw strings
    would produce one bucket per candidate. These keys let a cycle summary
    show which gate actually does the work — and whether the others are
    decoration.
    """
    r = (reason or "").lower()
    # ORDER MATTERS. The distance refusal reads "RSI 80, strengthened 1 scans,
    # 13.65% from the 24h high, limit 3.0%" — it opens with RSI, so a naive
    # "rsi" check first would file every distance refusal under the RSI band
    # and hide the distance gate entirely. Specific markers precede general.
    for marker, key in (
        ("volume still building", "vol_deferred"),
        ("heaviest volume came late", "vol_deferred_late"),
        ("stale signal", "stale_signal"),
        ("disabled (", "direction"),          # "longs disabled (short only)"
        ("gaining on ema21", "gap_not_rising"),
        ("no turn yet", "no_turn"),
        ("breakout structure", "breakout_veto"),
        ("limit", "distance"),                # "... 13.65% from the 24h high, limit 3%"
        ("from the 24h", "distance"),
        ("ceiling", "rsi_band"),              # "RSI 54.3 above the long ceiling"
        ("floor of", "rsi_band"),             # "RSI 41.2 below the long floor of"
        ("strengthen", "streak"),             # "... strengthened 1 scans — needs 2"
        ("atr", "min_atr"),                   # "ATR 0.48% below the 0.5% floor"
        ("rsi", "rsi_band"),
    ):
        if marker in r:
            return key
    return "other"


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

    allowed = (cfg.directions or "all").strip().lower()
    if allowed in ("long", "short") and side != allowed:
        return AutoDecision(False, symbol, side,
                            reason=f"{side}s disabled ({allowed} only)")

    if cfg.min_atr_pct:
        a = atr_pct if atr_pct is not None else row.get("atr_pct")
        if a is None or float(a) < cfg.min_atr_pct:
            return AutoDecision(False, symbol, side,
                                reason=f"ATR {a}% below the {cfg.min_atr_pct}% "
                                       f"floor — too quiet to clear fees")

    rsi = row.get("rsi")
    if rsi is None:
        return AutoDecision(False, symbol, side, reason="no RSI")
    floor = cfg.long_rsi_min if side == "long" else cfg.short_rsi_min
    if float(rsi) < floor:
        return AutoDecision(False, symbol, side,
                            reason=f"RSI {rsi} below the {side} floor of {floor}")

    # Longs also need a CEILING. A long taken in the middle of the RSI range is
    # neither oversold enough to bounce nor strong enough to trend, and that
    # band was the only losing one when longs were split by entry RSI within a
    # single session. Shorts are unbounded above by design — the whole thesis
    # is that more overbought is a better fade.
    if side == "long" and cfg.long_rsi_max and float(rsi) > cfg.long_rsi_max:
        return AutoDecision(False, symbol, side,
                            reason=f"RSI {rsi} above the long ceiling of "
                                   f"{cfg.long_rsi_max}")

    # A wide EMA gap fits a blow-off top and a breakout equally well; what
    # separates them is whether the move is still expanding. Note this veto is
    # in tension with max_dist_to_extreme_pct, which SELECTS for price at the
    # extreme — so the distance filter recruits exactly these candidates and
    # callback_for() then gives them the loosest trigger, because the callback
    # is proportional to distance from the extreme and collapses to its ATR
    # floor at zero distance.
    if side == "long" and cfg.long_require_turn:
        # The RIGHT side of a U or V. gap_narrowing alone is a two-point
        # shrink: a market still collapsing can post a smaller gap than N
        # candles ago and pass, which is how USELESS 18:06 was bought one
        # second before a 36% drop. This asks whether the low is BEHIND us.
        turn = row.get("turn") or {}
        if not turn.get("turned_up"):
            return AutoDecision(
                False, symbol, side,
                reason=(f"no turn yet — the fast EMA's low is "
                        f"{turn.get('bars_since_low')} candle(s) back "
                        f"(rise {turn.get('rise_pct')}%). Buying the left side "
                        f"of a U is buying a falling market."))

    if side == "short" and cfg.short_require_turn:
        # The mirror of the long turn test. turned(..., "short") asks whether
        # the HIGH is behind us: the fast EMA's extreme within the window is at
        # least turn_min_bars_since candles back, and price is off it. A peak
        # still forming is not a peak crossed.
        #
        # This is the "confirm the downturn" screen. Without it a short needed
        # only a high RSI, and RSI stays high for the whole of a trend.
        turn = row.get("turn") or {}
        if not turn.get("turned_up"):
            return AutoDecision(
                False, symbol, side,
                reason=(f"no turn yet — the high is not behind us "
                        f"(bars since extreme "
                        f"{turn.get('bars_since_low')}, move off it "
                        f"{turn.get('rise_pct')}%)"))

    if side == "short" and cfg.short_require_convergence:
        # SIGNED: EMA9 must be falling back TOWARD EMA21 from above. A gap
        # still widening is a trend, not a rollover.
        if not row.get("gap_narrowing"):
            return AutoDecision(
                False, symbol, side,
                reason=(f"EMA9 is not converging on EMA21 "
                        f"(gap {row.get('ema_gap_pct')}%, moved "
                        f"{row.get('gap_rise_pct')}%) — a widening gap is a "
                        f"trend, not a rollover"))

    if side == "long" and cfg.long_require_convergence:
        # SIGNED, not absolute. A gap going +0.10% -> +0.02% shrinks just as
        # much as one going -0.30% -> -0.05%, but the first is EMA9 FALLING
        # toward EMA21 from above — a top rolling over — and the second is
        # EMA9 rising toward it from underneath. Only the second is a long.
        # A crossover in progress (-0.02 -> +0.02) is rising and passes.
        if not row.get("gap_rising"):
            return AutoDecision(
                False, symbol, side,
                reason=(f"EMA9 is not gaining on EMA21 "
                        f"(signed gap moved {row.get('gap_rise_pct')}% from "
                        f"{row.get('ema_gap_pct')}%) — converging from the top "
                        f"is a rollover, not a recovery"))

    if cfg.veto_breakout:
        brk = row.get("breakout") or {}
        if brk.get("breakout"):
            return AutoDecision(
                False, symbol, side,
                reason=(f"breakout structure: at the 24h extreme, "
                        f"{brk.get('consecutive', 0)} consecutive candles, EMA "
                        f"gap wide and widening — still expanding, not exhausted"))

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

    # LIVE RE-CHECK. Everything above was decided on the scan's market, which
    # can be two minutes old. If a live quote says the price has moved past
    # the distance limit since then, the candidate no longer qualifies —
    # refusing here is the difference between fading an extreme and chasing
    # something that has already gone.
    live_px = row.get("live_price")
    drift_pct, live_dist = live_drift(row, live_px)
    if live_dist is not None and cfg.max_dist_to_extreme_pct:
        if live_dist > cfg.max_dist_to_extreme_pct:
            return AutoDecision(
                False, symbol, side,
                reason=(f"stale signal: the scan saw {dist:.2f}% from the 24h "
                        f"{'high' if side == 'short' else 'low'}, live is "
                        f"{live_dist:.2f}% (limit {cfg.max_dist_to_extreme_pct}%, "
                        f"drift {drift_pct:+.2f}%)"))

    # VOLUME DEFERRAL, SHORTS ONLY. Volume still building into the high means
    # the move is being bought, not exhausted — the fade is early. Across 89
    # shorts, volume fading into the high returned +$2.14/trade against -$1.00
    # when it was building.
    #
    # NOT mirrored for longs. A top forms on declining volume (distribution);
    # a bottom often forms on a volume SPIKE (a selling climax). The same
    # reading means the opposite thing, and the long sample is 16 trades with
    # both buckets losing — it says nothing either way. Recorded, not acted on.
    if side == "short" and cfg.defer_on_rising_volume:
        adv = (row.get("advance") or {})
        vt = adv.get("adv_vol_trend")
        early = adv.get("peak_vol_early")
        if vt is not None:
            vt = float(vt)
            # TWO ways to defer, because the two readings say different
            # things. adv_vol_trend is HOW MUCH volume grew; peak_vol_early is
            # WHERE the heaviest bar sat.
            #
            # BULLA 2026-09-16 passed the first test at 1.605 (under 2.0) and
            # lost: peak_vol_early was FALSE, so the heaviest trade arrived in
            # the SECOND half of a leg that had doubled. Volume merely growing
            # is ambiguous; volume growing with its peak still ahead is the
            # move being bought, which is what a fade must not stand in front
            # of.
            if vt >= cfg.defer_vol_trend:
                return AutoDecision(
                    False, symbol, side,
                    reason=(f"volume still building into the high "
                            f"(adv_vol_trend {vt:.2f} >= "
                            f"{cfg.defer_vol_trend}) — deferring, not "
                            f"refusing: the move is being bought, so the fade "
                            f"is early"))
            if (early is False and cfg.defer_vol_late_trend
                    and vt >= cfg.defer_vol_late_trend):
                return AutoDecision(
                    False, symbol, side,
                    reason=(f"heaviest volume came LATE in the advance "
                            f"(adv_vol_trend {vt:.2f} >= "
                            f"{cfg.defer_vol_late_trend}, peak_vol_early "
                            f"false) — the buying has not peaked yet"))

    cb, notes, cb_source = callback_for(dist, atr_pct, cfg,
                                       recent_tr_pct=row.get("recent_tr_pct"),
                                       side=side)
    extreme = "24h low" if side == "long" else "24h high"
    return AutoDecision(
        True, symbol, side, callback_pct=cb,
        reason=(f"RSI {rsi}, strengthened {streak} scans, {dist:.2f}% from the "
                f"{extreme} -> {cb}% callback ({cb_source})"),
        notes=notes, callback_source=cb_source,
    )


# ── Streak tracking ─────────────────────────────────────────────────────────

class StrengthTracker:
    """
    Counts consecutive SCANS on which a candidate strengthened.

    A single strengthening scan is noise; the operator's rule is that a setup
    must build over successive sweeps before it is acted on. "Scan" means a
    fresh scanner pass, not a fresh read of the last one — see update().
    """

    def __init__(self):
        self._streaks: dict[str, int] = {}
        # The scan these streaks were last advanced on, so a repeated read of
        # an unchanged snapshot cannot inflate them.
        self._last_scan_ts: float | None = None

    def update(self, rows: list[dict], scan_ts: float | None) -> dict[str, int]:
        """
        Count one scan. `scan_ts` is the scanner's `last_scan_ts`.

        run_once() fires every AUTO_TRADE_INTERVAL (30s) but the scanner only
        refreshes every SCANNER_INTERVAL (120s), so without this guard the same
        candidate list was counted four times and a "2 scan" requirement was
        satisfied 30 seconds after a single scan — by the SAME data, which is
        no confirmation at all. Re-reading an unchanged snapshot must not count.

        `scan_ts` is REQUIRED, deliberately. It was optional at first, which
        left a future call site able to omit it and silently restore the bug
        this method exists to prevent. Passing None still counts the call — a
        caller with genuinely no timestamp has nothing to dedupe on — but it
        now has to be written out, and it is logged.
        """
        if scan_ts is None:
            _log.warning(
                "StrengthTracker.update called with no scan timestamp — "
                "streaks will advance per CALL, not per scan.")
        if scan_ts is not None:
            if self._last_scan_ts is not None and scan_ts == self._last_scan_ts:
                return dict(self._streaks)      # same scan, already counted
            self._last_scan_ts = scan_ts

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
    day_started_at: float = 0.0
    # When the baseline was captured, and how. A baseline taken AT the
    # rollover is the real 00:00 balance; one taken at a mid-day restart is
    # just "the balance when the process woke up" and must not be presented as
    # the day's starting figure.
    # The day's high-water mark. The daily loss limit trails this, not the
    # opening balance, so a day that gives back its gains stops.
    day_peak_balance: float = 0.0
    day_baseline_at: float = 0.0
    day_baseline_source: str = ""      # rollover | restart | operator
    day_key: str = ""
    recent_entry_times: list[float] = field(default_factory=list)
    symbol_blocked_until: dict[str, float] = field(default_factory=dict)
    # RSI at the entry that failed, per symbol — the bar a re-entry must beat.
    failed_entry_rsi: dict[str, float] = field(default_factory=dict)
    reentries_today: dict[str, int] = field(default_factory=dict)
    halted_reason: str | None = None


# Hours east of UTC that the trading "day" rolls over on. The daily baseline
# and the halt reset on this boundary, so an operator in UTC+3 sees a day that
# starts at their midnight rather than at 03:00.
DAY_TZ_OFFSET_H: float = 0.0


def _day_key(now: float, offset_h: float | None = None) -> str:
    off = DAY_TZ_OFFSET_H if offset_h is None else offset_h
    return _time.strftime("%Y-%m-%d", _time.gmtime(now + off * 3600.0))


def day_start_ts(now: float, offset_h: float | None = None) -> float:
    """Unix time of the most recent local midnight."""
    off = (DAY_TZ_OFFSET_H if offset_h is None else offset_h) * 3600.0
    shifted = now + off
    midnight = shifted - (shifted % 86400.0)
    return midnight - off


# How soon after local midnight a rollover still counts as capturing the true
# 00:00 balance. The guardian polls every few seconds, so a running bot rolls
# within one cycle; anything later means the process was not up at midnight.
BASELINE_FRESH_S = 300.0


def roll_day(state: SafetyState, balance: float, now: float | None = None) -> SafetyState:
    """Reset the daily baseline (and any halt) when the LOCAL day changes."""
    now = now or _time.time()
    key = _day_key(now)
    if state.day_key != key:
        state.day_key = key
        state.day_start_balance = balance
        state.day_peak_balance = balance      # the peak resets with the day
        started = day_start_ts(now)
        state.day_started_at = started
        state.day_baseline_at = now
        # Crossing midnight WHILE RUNNING gives the true 00:00 balance. Waking
        # up mid-day gives whatever the wallet holds now, which is not the
        # day's start and must not be shown as one.
        state.day_baseline_source = (
            "rollover" if (now - started) <= BASELINE_FRESH_S else "restart")
        state.halted_reason = None
        state.reentries_today = {}
    if state.day_start_balance <= 0:
        state.day_start_balance = balance
        state.day_baseline_at = now
        if not state.day_baseline_source:
            state.day_baseline_source = "restart"
    return state


def parse_window(text: str):
    """
    Parse "HH:MM-HH:MM" into (start_minutes, end_minutes) from UTC midnight.

    A window that wraps past midnight (e.g. "22:00-04:00") is supported: the
    end being less than the start means it spans the day boundary.
    Returns None for an empty or unparseable value, meaning no restriction.
    """
    if not text:
        return None
    try:
        a, b = str(text).strip().split("-", 1)
        def mins(v):
            h, m = (v.strip().split(":") + ["0"])[:2]
            h, m = int(h), int(m)
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError(v)
            return h * 60 + m
        return mins(a), mins(b)
    except Exception:
        return None


def in_window(window, now_utc=None) -> bool:
    """Is the current UTC time inside the window? True when unrestricted."""
    if window is None:
        return True
    from datetime import datetime, timezone
    now = now_utc or datetime.now(timezone.utc)
    cur = now.hour * 60 + now.minute
    start, end = window
    if start == end:
        return True                      # a zero-width window means no limit
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end     # wraps past midnight


# UTC hour ranges, matching analysis.SESSIONS so the toggles line up with the
# table they are read from.
SESSION_HOURS = (("AS", 0, 8), ("EU", 8, 13), ("OV", 13, 17), ("US", 17, 24))
SESSION_NAMES = {"AS": "Asia", "EU": "Europe", "OV": "EU/US overlap", "US": "US"}


def session_key(now: float | None = None) -> str:
    """Which session a moment falls in, by UTC hour."""
    h = _time.gmtime(now or _time.time()).tm_hour
    for key, lo, hi in SESSION_HOURS:
        if lo <= h < hi:
            return key
    return "US"


def parse_session(spec) -> tuple[bool, bool]:
    """
    "L1S1" -> (long_on, short_on). Anything unreadable means BOTH ON.

    Defaulting to on matters: a typo in an env var must not silently stop
    trading, which is the failure an operator discovers hours later.
    """
    try:
        t = str(spec or "").strip().upper()
        if not t:
            return True, True
        lg = "L0" not in t
        sh = "S0" not in t
        return lg, sh
    except Exception:
        return True, True


def session_allows(cfg, side: str, now: float | None = None) -> tuple[bool, str]:
    key = session_key(now)
    spec = (getattr(cfg, "sessions", None) or {}).get(key)
    lg, sh = parse_session(spec)
    on = lg if side == "long" else sh
    if on:
        return True, ""
    return False, (f"{side}s are switched off for the "
                   f"{SESSION_NAMES.get(key, key)} session "
                   f"({spec or 'L1S1'}, UTC hours)")


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

    # The halt must respect the switch in BOTH directions. Disabling it used to
    # leave an existing halted_reason in place, so trading stayed blocked while
    # the dashboard said the halt was off — and this second, independent check
    # below re-set the halt even when disabled. Both cost data-collection time
    # that only shows up when someone comes back and finds it stopped.
    # Outside the trading window nothing new is opened. Checked first because
    # it is the cheapest test and the most common reason to decline.
    # A session gate rather than one window. Both directions used to share a
    # single time range, but the two behave differently within the same hours:
    # across 329 trades, US shorts returned +7.91% avg ROI while US longs
    # returned -2.72%, and Asia was the only session where both sides were
    # positive. One window cannot express that; two overlapping time gates
    # would be a second place for a coverage hole to open.
    ok_session, session_why = session_allows(cfg, side, now)
    if not ok_session:
        return False, session_why

    halt_on = getattr(cfg, "daily_halt_enabled", True)
    if not halt_on and state.halted_reason:
        state.halted_reason = None

    if state.halted_reason:
        return False, state.halted_reason

    # The limit trails the day's HIGH-WATER MARK, not the opening balance.
    #
    # Measured from the open, a day that runs +3% and then bleeds back to
    # breakeven has "lost nothing" and keeps trading, having given back the
    # entire day. Measured from the peak, the same day stops once it hands
    # back the limit. On a day that only ever falls, peak == open and the two
    # are identical — so this is never LOOSER than the old behaviour, only
    # tighter on days that were ahead.
    #
    # The peak is seeded at the open and updated on every check, so it cannot
    # start below the balance the day began with.
    if halt_on and cfg.daily_loss_limit_pct and state.day_start_balance > 0:
        if balance > 0:
            state.day_peak_balance = max(
                float(state.day_peak_balance or 0.0),
                float(state.day_start_balance), float(balance))
        peak = float(state.day_peak_balance or state.day_start_balance)
        drawdown = (peak - balance) / peak * 100
        if drawdown >= cfg.daily_loss_limit_pct:
            # As a RETURN, so "-3.0% on the day" reads the way it should.
            from_open = (balance - state.day_start_balance) / state.day_start_balance * 100
            state.halted_reason = (
                f"daily loss limit hit: down {drawdown:.1f}% from today's "
                f"high {peak:.2f} (open {state.day_start_balance:.2f}, "
                f"{from_open:+.1f}% on the day) — auto-trade halted until "
                f"tomorrow or a manual reset"
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
        # Which RULE refused candidates last cycle, for the dashboard.
        self._last_refusals: dict[str, int] = {}
        # Optional live-price feed, set by main.py. None means every decision
        # uses the scan snapshot, which is the behaviour this replaced.
        self.stream = None
        # Throttle for the stream health line. It was NEVER INITIALISED: the
        # modulo raised AttributeError on the first cycle, the surrounding
        # `except` logged at DEBUG, and DEBUG is invisible at INFO — so stream
        # health silently never printed while the stream itself worked.
        self._stream_log_n: int = 0
        # Cross-evaluation peer, set by main.py. None = off.
        self.peer_eval = None
        # Shadow decision logger (bot/shadow_decision.py), set by main.py.
        # None = off. Advisory only — it never gates, sizes or delays a trade.
        self.shadow = None

    # -- reporting -------------------------------------------------------
    # Rejections that will NEVER succeed on a retry. Narrow on purpose:
    # -2021 (would immediately trigger) and margin errors are transient and
    # must stay retryable.
    PERMANENT_REJECTIONS = ("-4411", "agreement", "not authorized",
                            "not permitted", "permission")

    # The market fields that identify an INSTRUMENT CLASS rather than a
    # symbol. Binance groups TradFi perps (tokenised equities) apart from
    # crypto perps here, so one rejection can teach the whole class instead of
    # the operator maintaining a list that a new listing defeats.
    CLASS_FIELDS = ("underlyingType", "underlyingSubType", "contractType",
                    "marginAsset", "quoteAsset")

    @classmethod
    def _is_permanent_rejection(cls, detail: str) -> bool:
        d = (detail or "").lower()
        return any(k.lower() in d for k in cls.PERMANENT_REJECTIONS)

    def _market_class(self, symbol: str):
        """
        A hashable fingerprint of the symbol's instrument class.

        Read from ccxt's market info, which carries Binance's own exchangeInfo
        fields. Returns None when nothing distinguishing is available — and
        None must never match, or one rejection would block everything.
        """
        ex = getattr(self, "exchange", None) or getattr(
            getattr(self, "entry", None), "exchange", None)
        if ex is None:
            ex = getattr(getattr(getattr(self, "entry", None), "guardian",
                                 None), "exchange", None)
        if ex is None:
            return None
        try:
            info = (ex.market(symbol) or {}).get("info") or {}
        except Exception:
            return None
        key = tuple(
            (f, str(info.get(f)))
            for f in self.CLASS_FIELDS
            if info.get(f) not in (None, "", [])
        )
        return key or None

    def _class_is_blocked(self, symbol: str) -> bool:
        if not self._blocked_classes:
            return False
        k = self._market_class(symbol)
        return bool(k) and k in self._blocked_classes

    def _record(self, action: str, detail: str, symbol: str = ""):
        self._log.append({"ts": _time.time(), "action": action,
                          "symbol": symbol, "detail": detail})
        self._log = self._log[-40:]

    # NOTE: live rule edits are deliberately NOT persisted. The compose file is
    # the declared configuration and must win on every restart; a dashboard
    # change is session tuning that lasts as long as the instance does.
    def safety_snapshot(self) -> dict:
        """SafetyState in a persistable form — the daily halt must survive a restart."""
        return {
            "day_start_balance": self.state.day_start_balance,
            "day_started_at": getattr(self.state, "day_started_at", 0.0),
            "day_peak_balance": getattr(self.state, "day_peak_balance", 0.0),
            "day_baseline_at": getattr(self.state, "day_baseline_at", 0.0),
            "day_baseline_source": getattr(self.state, "day_baseline_source", ""),
            "day_key": self.state.day_key,
            "recent_entry_times": list(self.state.recent_entry_times),
            "symbol_blocked_until": dict(self.state.symbol_blocked_until),
            "failed_entry_rsi": dict(self.state.failed_entry_rsi),
            "reentries_today": dict(self.state.reentries_today),
            "day_start_balance": self.state.day_start_balance,
            "day_started_at": getattr(self.state, "day_started_at", 0.0),
            "day_baseline_at": getattr(self.state, "day_baseline_at", 0.0),
            "day_baseline_source": getattr(self.state, "day_baseline_source", ""),
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
        s.day_started_at = float(data.get("day_started_at") or 0.0)
        s.day_peak_balance = float(data.get("day_peak_balance") or 0.0)
        s.day_baseline_at = float(data.get("day_baseline_at") or 0.0)
        s.day_baseline_source = str(data.get("day_baseline_source") or "")
        s.day_key = data.get("day_key") or ""
        s.recent_entry_times = list(data.get("recent_entry_times") or [])
        s.symbol_blocked_until = dict(data.get("symbol_blocked_until") or {})
        s.failed_entry_rsi = dict(data.get("failed_entry_rsi") or {})
        s.reentries_today = dict(data.get("reentries_today") or {})
        s.halted_reason = data.get("halted_reason")
        if s.halted_reason and not getattr(self.cfg, "daily_halt_enabled", True):
            _log.warning(
                f"Discarding a persisted halt because the daily halt is "
                f"disabled: {s.halted_reason}")
            s.halted_reason = None
        elif s.halted_reason:
            _log.warning(f"Auto-trade remains HALTED after restart: {s.halted_reason}")

    def snapshot(self) -> dict:
        return {
            "enabled": self.cfg.enabled,
            "halted_reason": self.state.halted_reason,
            # Why each visible candidate was NOT entered. Rule-level refusals
            # used to be silent, so a dashboard full of candidates with no
            # entries gave no way to tell a broken bot from rules that simply
            # do not match the current market.
            "skip_reasons": dict(getattr(self, "_skip_reasons", {})),
            # Entries the exchange refused. Counted so lost opportunities are
            # visible without reading the exchange's own order list.
            "rejections": getattr(self, "_rejections", 0),
            "day_start_balance": round(self.state.day_start_balance, 2),
            "trades_last_hour": len(self.state.recent_entry_times),
            "cooldowns": {k: int(v - _time.time())
                          for k, v in self.state.symbol_blocked_until.items()
                          if v > _time.time()},
            "last_run_ago_s": (_time.time() - self._last_run) if self._last_run else None,
            "recent": list(reversed(self._log[-15:])),
            "config": {
                "callback_min_pct": self.cfg.callback_min_pct,
                "long_require_turn": self.cfg.long_require_turn,
                "last_refusals": dict(getattr(self, "_last_refusals", {})),
                "stream": (_stream_status(getattr(self, "stream", None))),
                "long_require_convergence": self.cfg.long_require_convergence,
                "short_require_turn": self.cfg.short_require_turn,
                "short_require_convergence": self.cfg.short_require_convergence,
                "callback_use_velocity": velocity_mode(
                    self.cfg.callback_use_velocity),
                "veto_breakout": self.cfg.veto_breakout,
                "max_dist_to_extreme_pct": self.cfg.max_dist_to_extreme_pct,
                "required_strength_sweeps": self.cfg.required_strength_sweeps,
                "long_rsi_min": self.cfg.long_rsi_min,
                "long_rsi_max": self.cfg.long_rsi_max,
                "directions": self.cfg.directions,
                "daily_halt_enabled": self.cfg.daily_halt_enabled,
                "sessions": dict(self.cfg.sessions or {}),
                "session_now": session_key(),
                "min_atr_pct": self.cfg.min_atr_pct,
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
            "day_start_balance": self.state.day_start_balance,
            "day_started_at": getattr(self.state, "day_started_at", 0.0),
            "day_baseline_at": getattr(self.state, "day_baseline_at", 0.0),
            "day_baseline_source": getattr(self.state, "day_baseline_source", ""),
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
        # Entry quality, not a safety limit — tunable so the A/B can be run
        # from the dashboard without a redeploy.
        "callback_min_pct": (float, 0.1, 5.0),
        "long_require_turn": (bool, None, (True, False)),
        "defer_on_rising_volume": (bool, None, (True, False)),
        "defer_vol_trend": (float, 0.5, 5.0),
        "defer_vol_late_trend": (float, 0.0, 5.0),
        "long_require_convergence": (bool, None, (True, False)),
        "short_require_turn": (bool, None, (True, False)),
        "short_require_convergence": (bool, None, (True, False)),
        "callback_use_velocity": (str, None, ("off", "all", "short", "long")),
        "veto_breakout": (bool, None, (True, False)),
        "max_dist_to_extreme_pct": (float, 0.1, 50.0),
        "required_strength_sweeps": (int, 1, 10),
        "long_rsi_min": (float, 0.0, 100.0),
        "long_rsi_max": (float, 0.0, 100.0),
        # A string enum rather than a numeric range: the third element is the
        # set of allowed values instead of an upper bound.
        "directions": (str, None, ("all", "long", "short")),
        "daily_halt_enabled": (bool, None, (True, False)),
        "sessions": (dict, None, None),
        "min_atr_pct": (float, 0.0, 10.0),
        "short_rsi_min": (float, 0.0, 100.0),
        "callback_ratio": (float, 0.05, 2.0),
        "callback_atr_mult": (float, 0.0, 5.0),
        # How much stronger a signal must be to override a cooldown. This is an
        # entry-quality judgement, so it is tunable; the re-entry CAP is not.
        "cooldown_override_rsi_delta": (float, 0.0, 50.0),
        # A rate limit, not a loss limit: the daily loss stop and position cap
        # already bound the damage, so this is tunable. 0 disables it.
        "max_trades_per_hour": (int, 0, 200),
    }
    SAFETY_ONLY = {"daily_loss_limit_pct", "symbol_cooldown_s",
                   "max_open_positions", "max_reentries_per_symbol"}

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

            # Booleans need explicit parsing: bool("false") is True, so casting
            # a form value through bool() would accept "false" as enabled and
            # any typo as enabled too.
            # Free-text settings (no allowed-value set and no bounds) are
            # validated by their own parser, not by a numeric range.
            # Session switches arrive as a dict, so they cannot go through the
            # string or numeric paths. Declared as (dict, None, None), `typ is
            # str` was False and the value fell into the numeric branch, where
            # it failed — the toggles rendered and did nothing.
            if typ is dict:
                if not isinstance(raw, dict):
                    errors.append(f"{key} must be an object like "
                                  f'{{"AS": "L1S1", "EU": "L0S1"}}')
                    continue
                cleaned, bad = {}, []
                for k, v in raw.items():
                    kk = str(k).strip().upper()
                    vv = str(v).strip().upper()
                    if kk not in ("AS", "EU", "OV", "US"):
                        bad.append(kk)
                        continue
                    # Anything unreadable resolves to BOTH ON rather than
                    # silently switching a session off.
                    lg, sh = parse_session(vv)
                    cleaned[kk] = f"L{1 if lg else 0}S{1 if sh else 0}"
                if bad:
                    errors.append(f"unknown session key(s): {', '.join(bad)}")
                    continue
                merged = dict(getattr(self.cfg, key, {}) or {})
                merged.update(cleaned)
                setattr(self.cfg, key, merged)
                applied[key] = merged
                continue

            if typ is str and lo is None and hi is None:
                val = str(raw).strip()
                setattr(self.cfg, key, val)
                applied[key] = val
                continue

            if typ is bool:
                if isinstance(raw, bool):
                    val = raw
                else:
                    text = str(raw).strip().lower()
                    if text in ("true", "1", "yes", "on"):
                        val = True
                    elif text in ("false", "0", "no", "off"):
                        val = False
                    else:
                        errors.append(f"{key} must be true or false")
                        continue
                setattr(self.cfg, key, val)
                applied[key] = val
                if key == "daily_halt_enabled" and val is False:
                    # getattr: update_rules is reachable before state exists.
                    st = getattr(self, "state", None)
                    if st is not None and st.halted_reason:
                        _log.warning(
                            f"daily halt disabled — clearing the active halt "
                            f"({st.halted_reason})")
                        st.halted_reason = None
                        self._record("reset", "halt cleared: feature disabled")
                continue

            try:
                val = typ(raw)
            except (TypeError, ValueError):
                errors.append(f"{key} must be a {typ.__name__}")
                continue
            if isinstance(hi, (tuple, list, set)):
                val = str(val).strip().lower()
                if val not in hi:
                    errors.append(
                        f"{key} must be one of "
                        f"{', '.join(sorted(str(x) for x in hi))}")
                    continue
            elif not (lo <= val <= hi):
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

    def reset_halt(self, balance: float | None = None) -> dict:
        """
        Clear a daily-loss halt and REBASE the baseline to the current balance.

        Clearing the reason alone was useless: the drawdown is measured from
        day_start_balance, which had not changed, so the very next cycle
        recomputed the same figure and halted again. The halt appeared to clear
        and then immediately returned.

        Rebasing gives the operator one further allowance measured from here,
        which is a bounded and explicit decision rather than a no-op.
        """
        was = self.state.halted_reason
        self.state.halted_reason = None
        bal = balance
        if bal is None:
            try:
                bal = self.entry.wallet_balance()
            except Exception:
                bal = None
        if bal and bal > 0:
            old_base = self.state.day_start_balance
            self.state.day_start_balance = float(bal)
            _log.warning(
                f"Halt cleared and baseline REBASED {old_base:.2f} -> {bal:.2f}. "
                f"The next {self.cfg.daily_loss_limit_pct:g}% is measured from "
                f"here, so this grants one further allowance.")
            self._record("reset", f"halt cleared, baseline rebased to {bal:.2f}")
        else:
            _log.warning(
                "Halt cleared but the balance could not be read, so the "
                "baseline is unchanged — the halt will re-fire on the next "
                "cycle. Retry once the balance is available.")
            self._record("reset", "halt cleared (baseline NOT rebased)")
        if was:
            _log.warning(f"previous halt reason was: {was}")
        return self.snapshot()

    # -- main loop -------------------------------------------------------
    def run_once(self):
        self._last_run = _time.time()
        if not self.cfg.enabled:
            return

        # Which RULE refused a candidate was never logged — only "a position
        # already exists", which is not a rule at all. So there was no way to
        # tell whether one gate was doing all the work and the rest were
        # decoration. Tallied per cycle rather than one line per candidate,
        # which would be hundreds an hour.
        refusals: dict[str, int] = {}

        snap = self.scanner.snapshot()
        # Follow eligibility rather than accumulating symbols: the stream
        # tracks what the scanner currently considers a candidate, which is
        # ten to twenty of the 574 it screens.
        st = getattr(self, "stream", None)
        if st is not None:
            try:
                st.track([r.get("symbol") for r in
                          (snap.get("candidates") or [])])
            except Exception as e:
                _log.debug(f"stream track failed: {e}")

        # Send the candidate list for calibration. Entry notifications alone
        # are far too sparse to measure a ratio from — a handful an hour
        # against dozens of paired candidate readings per hour here.
        pe = getattr(self, "peer_eval", None)
        if pe is not None:
            try:
                pe.notify_scan(snap.get("candidates") or [])
            except Exception:
                pass
        rows = snap.get("candidates") or []
        # Pass the scan timestamp so a repeated read of the same snapshot does
        # not advance the streaks.
        self.tracker.update(rows, scan_ts=snap.get("last_scan_ts"))

        try:
            balance = self.entry.wallet_balance()
            positions = self.guardian.fetch_positions()
        except Exception as e:
            _log.warning(f"auto-trade: account read failed: {e}")
            return

        roll_day(self.state, balance)

        # Evaluate the daily loss EVERY cycle, not only when a candidate is
        # being considered. Previously the drawdown was checked inside the
        # per-candidate loop, so a scan with no candidates performed no check
        # and the halt could lag well past its limit.
        self._check_daily_drawdown(balance)

        # Reasons are rebuilt each pass so they always describe the CURRENT
        # candidate list rather than accumulating stale entries.
        self._skip_reasons = {}
        # Symbols the exchange has permanently refused this session. Process
        # scoped on purpose: a restart clears it, so signing the agreement
        # takes effect without a config change.
        if not hasattr(self, "_blocked"):
            self._blocked: set[str] = set()
        if not hasattr(self, "_blocked_classes"):
            # Instrument classes, not symbols. One -4411 teaches the whole
            # class, so a newly listed TradFi perp is skipped without ever
            # being attempted.
            self._blocked_classes: set = set()
        if not hasattr(self, "_rejections"):
            self._rejections = 0
        open_syms = {p.symbol for p in positions}
        # Also skip symbols with a resting entry order. The entry service
        # refuses these too, but checking here avoids a pointless preview and
        # keeps the log readable.
        for row in rows:
            sym = row.get("symbol", "")
            if sym not in open_syms:
                try:
                    if self.entry.pending_entry_orders(sym):
                        open_syms.add(sym)
                except Exception:
                    pass

        for row in rows:
            symbol = row.get("symbol", "")
            side = row.get("direction", "")
            if symbol in open_syms:
                self._skip_reasons[symbol] = "position or order already open"
                continue
            if symbol in self._blocked or self._class_is_blocked(symbol):
                # Permanently refused earlier this session, or of the same
                # instrument class as something that was. Not a strategy
                # decision, so it stays out of the refusal counter.
                self._blocked.add(symbol)      # cache, so the lookup is once
                self._skip_reasons[symbol] = (
                    "blocked: the exchange refused this instrument class for "
                    "a reason retrying cannot fix")
                continue

            streak = self.tracker.streak(symbol, side)
            # A live quote when the stream has one; None otherwise, and the
            # candidate is judged on the scan figures exactly as before.
            st = getattr(self, "stream", None)
            if st is not None:
                try:
                    row["live_price"] = st.price(symbol)
                except Exception:
                    row["live_price"] = None

            # Record when the live quote is what decided it, so the value of
            # the stream is measurable rather than assumed.
            _pre_live = row.get("live_price")

            decision = evaluate_candidate(
                row, streak, self.cfg, atr_pct=row.get("atr_pct"))
            if not decision.enter:
                # Record WHY. Rule-level refusals were previously silent, so a
                # dashboard full of candidates with no entries gave no clue
                # whether the bot was broken or the rules simply did not match.
                self._skip_reasons[symbol] = decision.reason
                if "stale signal" in (decision.reason or ""):
                    _log.warning(
                        f"STREAM-SAVED {symbol}: {decision.reason}. On the "
                        f"scan figures alone this would have been entered.")
                refusals[_refusal_key(decision.reason)] = refusals.get(
                    _refusal_key(decision.reason), 0) + 1
                # The refusal arm of the shadow log (JEV-BRIEF.md §1): without
                # this, there is no counterfactual for what jev would have
                # done with a candidate the bot itself skipped. Fire-and-
                # forget, same pattern as peer_eval below — never affects the
                # decision already made.
                sh = getattr(self, "shadow", None)
                if sh is not None:
                    try:
                        sh.decide_async(symbol, side, row,
                                       bot_decision="SKIP", snap=snap)
                    except Exception:
                        pass
                continue
            self._skip_reasons.pop(symbol, None)

            ok, why = check_safety(self.state, self.cfg, balance=balance,
                                   open_positions=len(positions), symbol=symbol,
                                   current_rsi=row.get("rsi"), side=side)
            # "ok" is what check_safety returns on ORDINARY success, so a
            # bare truthiness test logged a useless line on every entry. Only
            # the override is worth a line.
            if ok and why and why != "ok":
                # check_safety returns a REASON on success too, and the only
                # one it produces is "cooldown overridden". That was discarded,
                # so a symbol re-entered minutes after a loss left no trace at
                # all: LSK re-entered 4 minutes after a -$27.38 loss and the
                # logs showed neither a cooldown nor an override.
                _log.warning(f"auto-trade: {symbol} — {why}")
                self._record("cooldown_override", why, symbol)
            if not ok:
                self._skip_reasons[symbol] = why
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
                self._skip_reasons[symbol] = errs
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
                        "recent_tr_pct": row.get("recent_tr_pct"),
                        "gap_narrowing": row.get("gap_narrowing"),
                        "gap_rising": row.get("gap_rising"),
                        "efficiency": (row.get("efficiency") or {}).get("efficiency"),
                        "er_direction": (row.get("efficiency") or {}).get("er_direction"),
                        "adv_vol_trend": (row.get("advance") or {}).get("adv_vol_trend"),
                        "adv_bars": (row.get("advance") or {}).get("adv_bars"),
                        "adv_price_pct": (row.get("advance") or {}).get("adv_price_pct"),
                        "peak_vol_early": (row.get("advance") or {}).get("peak_vol_early"),
                        "gap_rise_pct": row.get("gap_rise_pct"),
                        "turned_up": (row.get("turn") or {}).get("turned_up"),
                        "bars_since_low": (row.get("turn") or {}).get("bars_since_low"),
                        "turn_rise_pct": (row.get("turn") or {}).get("rise_pct"),
                        # A coin up 108% in 24h is not the same trade as one
                        # up 17%. The scanner has always computed this; it was
                        # never recorded against the trade it produced.
                        "change_24h_pct": row.get("change_24h_pct"),
                        "taper_ratio": (row.get("taper") or {}).get("taper_ratio"),
                        "tapering": (row.get("taper") or {}).get("tapering"),
                        "taper_vol_ratio": (row.get("taper") or {}).get("vol_ratio"),
                        "taper_close_pos": (row.get("taper") or {}).get("close_pos"),
                        "taper_trend_candles": (row.get("taper") or {}).get("trend_candles"),
                        "body_pct": (row.get("shape") or {}).get("body_pct"),
                        "upper_wick_pct": (row.get("shape") or {}).get("upper_wick_pct"),
                        "lower_wick_pct": (row.get("shape") or {}).get("lower_wick_pct"),
                        "dist_to_extreme_pct": abs(
                            row.get("pct_above_24h_low") if side == "long"
                            else row.get("pct_below_24h_high")),
                        "range_pos_24h": row.get("range_pos_24h"),
                        "ema_gap_pct": row.get("ema_gap_pct"),
                        # Regime context at entry. Recorded ONLY — nothing
                        # gates on these yet. The point is to find out whether
                        # they predict which direction works before acting on
                        # a guess about which threshold matters.
                        "htf_trend_pct": row.get("htf_trend_pct"),
                        # Breakout structure at entry. The question is whether
                        # a wide, WIDENING gap at a new extreme means the move
                        # is still expanding — in which case fading it is the
                        # wrong side. Recorded only.
                        "breakout": (row.get("breakout") or {}).get("breakout"),
                        "brk_at_extreme": (row.get("breakout") or {}).get("at_extreme"),
                        "brk_consecutive": (row.get("breakout") or {}).get("consecutive"),
                        "brk_gap_wide": (row.get("breakout") or {}).get("gap_wide"),
                        "brk_gap_widening": (row.get("breakout") or {}).get("gap_widening"),
                        "breadth_pct": snap.get("breadth_pct"),
                        "btc_change_pct": snap.get("btc_change_pct"),
                        "strength": row.get("strength"),
                        "streak": streak,
                        "callback_pct": decision.callback_pct,
                        "callback_source": decision.callback_source,
                        "was_reentry": "cooldown overridden" in why,
                        # The exchange's own order id. Added so the shadow
                        # decision log (bot/shadow_decision.py, JEV-BRIEF.md
                        # §5) can join a jev verdict recorded BEFORE this
                        # order existed to the closed-trade record it later
                        # becomes — no other field here is a stable join key.
                        "entry_order_id": res.get("order_id"),
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
                # Ask the other instance whether it would have taken this. Own
                # thread, everything swallowed — instrumentation must never
                # affect a trade.
                pe = getattr(self, "peer_eval", None)
                if pe is not None:
                    try:
                        pe.notify_entry(symbol, side, row, decision.reason)
                    except Exception:
                        pass
                # The ENTER arm of the shadow log (JEV-BRIEF.md §1) — same
                # candidate, same fire-and-forget pattern as peer_eval just
                # above, now carrying the real order id so /api/shadow can
                # join this verdict to the trade it turns into.
                sh = getattr(self, "shadow", None)
                if sh is not None:
                    try:
                        sh.decide_async(symbol, side, row,
                                       bot_decision="ENTER", snap=snap,
                                       entry_order_id=res.get("order_id"))
                    except Exception:
                        pass
                self._record("entered",
                             f"{side} · {decision.reason}"
                             + (f" · {note}" if note else ""), symbol)
            else:
                # execute() returns `errors` (a list); reading `error` recorded
                # the literal string "None", so every rejection showed up blank
                # in the dashboard and could only be found on the exchange.
                detail = "; ".join(res.get("errors") or []) or str(
                    res.get("error") or "unknown")
                self._record("execute_failed", detail, symbol)
                self._skip_reasons[symbol] = f"exchange rejected: {detail}"
                self._rejections += 1
                _log.warning(f"auto-trade: {symbol} entry rejected — {detail}")
                # A PERMANENT rejection must not be retried. -4411 is an
                # account permission (TradFi perps need their own signed
                # agreement), and a rejection opens no position, so it
                # triggers no cooldown — SNXX was re-attempted every scan.
                # Blocked for this process only: restart clears it, so a
                # signed agreement takes effect without touching config.
                if self._is_permanent_rejection(detail):
                    self._blocked.add(symbol)
                    klass = self._market_class(symbol)
                    if klass:
                        self._blocked_classes.add(klass)
                        _log.error(
                            f"auto-trade: {symbol} BLOCKED, and so is every "
                            f"symbol of the same instrument class — the "
                            f"exchange refused it for a reason retrying "
                            f"cannot fix ({detail}). class="
                            + ", ".join(f"{k}={v}" for k, v in klass))
                    else:
                        _log.error(
                            f"auto-trade: {symbol} BLOCKED for this session "
                            f"({detail}). No instrument-class fields were "
                            f"available, so only this symbol is blocked — "
                            f"others of the same kind will each be refused "
                            f"once before they are caught.")

        if refusals:
            top = ", ".join(f"{k}={v}" for k, v in
                            sorted(refusals.items(), key=lambda kv: -kv[1]))
            _log.info(f"auto-trade refusals this cycle: {top}")
            self._last_refusals = dict(refusals)

        # STREAM HEALTH, once a cycle. A stream that silently stops delivering
        # degrades entries back to the scan snapshot without anything failing —
        # exactly the kind of quiet regression that is only noticed weeks later
        # in the numbers. One greppable word: STREAM.
        st = getattr(self, "stream", None)
        if st is not None:
            try:
                h = st.status()
                # Delivery, not the socket flag — see CandidateStream.healthy
                healthy = h.get("healthy")
                # Socket fields only when a socket is meant to exist, so
                # "connected=False" cannot read as a fault on a REST-only run.
                ws_on = h.get("websocket_enabled")
                ws_part = (
                    f"connected={h.get('connected')} "
                    f"msgs={h.get('messages')} "
                    f"reconnects={h.get('reconnects')} " if ws_on else "ws=off ")
                msg = (
                    f"STREAM {'ok' if healthy else 'DEGRADED'}: "
                    f"src={h.get('source')} "
                    f"{ws_part}"
                    f"rest={h.get('rest_polls')}/{h.get('rest_errors')}e "
                    f"tracking={h.get('tracking')} "
                    f"fresh={h.get('fresh')}/{h.get('quotes')} "
                    f"last_msg={h.get('last_message_age_s')}s")
                if healthy:
                    if self._stream_log_n % 20 == 0:
                        _log.info(msg)
                else:
                    # Throttled. A stream that is environmentally blocked
                    # stays degraded, and a warning every 30s buries
                    # everything else in the log. First occurrence, then
                    # roughly every ten minutes.
                    if self._stream_log_n % 20 == 0:
                        _log.warning(
                            msg + " — entries are falling back to the scan "
                            "snapshot, which can be up to one scanner interval "
                            "old"
                            + (f". last error: {h['last_error']}"
                               if h.get("last_error") else "")
                            + (f" | probe: {h['probe']}"
                               if h.get("probe") else ""))
                self._stream_log_n += 1
            except Exception as e:
                # WARNING, not DEBUG. A failure here means stream health is
                # unreportable, which is exactly the state that hid this bug.
                _log.warning(f"stream status failed: {e}")

    def _check_daily_drawdown(self, balance: float):
        """
        Halt as soon as the limit is breached, independent of candidate flow.

        This bounds NEW entries only — it cannot unwind positions already open,
        so the realised drawdown can still exceed the limit by roughly the
        exposure outstanding when it fires. That is inherent, but the halt
        should at least not be late.
        """
        if not self.cfg.daily_halt_enabled:
            return
        if self.state.halted_reason or not self.cfg.daily_loss_limit_pct:
            return
        start = self.state.day_start_balance
        if start <= 0:
            return
        dd = (start - balance) / start * 100
        if dd >= self.cfg.daily_loss_limit_pct:
            self.state.halted_reason = (
                f"daily loss limit hit: down {dd:.1f}% from {start:.2f} — "
                f"auto-trade halted until tomorrow (UTC) or a manual reset")
            _log.error(f"AUTO-TRADE HALTED: {self.state.halted_reason}")
            self._record("halted", self.state.halted_reason)

    def note_closed_trade(self, symbol: str, realised: float | None,
                          entry_rsi: float | None = None):
        """Apply the post-loss cooldown when a position closes down."""
        try:
            self.entry.clear_pending(symbol)
        except Exception:
            pass
        if realised is not None and realised < 0:
            record_loss(self.state, symbol, self.cfg,
                        entry_rsi=entry_rsi or self.state.failed_entry_rsi.get(symbol))
            self._record("cooldown", f"loss on {symbol}; blocked for "
                                     f"{int(self.cfg.symbol_cooldown_s)}s", symbol)
            # Also to the container log. This only reached the event feed, so
            # "did the cooldown apply?" was unanswerable from logs.
            _log.info(
                f"auto-trade: cooldown applied to {symbol} after a "
                f"{realised:+.2f} loss — blocked for "
                f"{int(self.cfg.symbol_cooldown_s)}s"
                + (f", unless RSI rises "
                   f"{self.cfg.cooldown_override_rsi_delta:g} points"
                   if self.cfg.cooldown_override_rsi_delta else ""))
