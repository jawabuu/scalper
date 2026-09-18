#!/usr/bin/env python3
"""
Binance momentum scalping bot — entrypoint.
Starts the FastAPI dashboard in a background thread, then runs the bot loop.
"""

import logging
from pathlib import Path
import threading
import os
import sys

from bot import BotConfig, ScalpingEngine, __version__
from bot.api import run_api


def _stream_proxy(cfg):
    """
    Proxy for the candidate stream.

    Unset inherits SOCKS_PROXY. The literal "none" goes DIRECT — worth trying,
    because the live websocket host was silent through the VPN exit while REST
    on the same proxy worked fine.
    """
    v = (getattr(cfg, "stream_proxy", "") or "").strip()
    if v.lower() in ("none", "direct", "off"):
        return None
    return v or (cfg.socks_proxy or None)


def setup_logging():
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)-8s %(name)-10s %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/bot.log"),
        ],
    )
    if level_name != "INFO":
        logging.getLogger("main").info(f"Log level set to {level_name}")


if __name__ == "__main__":
    setup_logging()
    log = logging.getLogger("main")

    try:
        cfg = BotConfig().validate()
        # The trading day rolls over on the operator's local midnight, not
        # UTC's. Set before anything reads a day key.
        import bot.auto_trader as _at
        _at.DAY_TZ_OFFSET_H = float(cfg.day_tz_offset_h or 0.0)
        if cfg.day_tz_offset_h:
            log.info(f"Trading day rolls over at 00:00 UTC"
                     f"{cfg.day_tz_offset_h:+g} "
                     f"({-cfg.day_tz_offset_h % 24:02.0f}:00 UTC)")
    except AssertionError as e:
        log.error(f"Config error: {e}")
        sys.exit(1)

    tp_info = f"tp={cfg.take_profit_pct}%" if cfg.take_profit_enabled else "tp=disabled"
    log.info(
        f"🚀 Scalper v{__version__} — testnet={cfg.testnet} tf={cfg.timeframe} "
        f"stop={cfg.trailing_stop_pct}% {tp_info} "
        f"risk={cfg.risk_per_trade_pct}%/trade"
    )

    engine = ScalpingEngine(cfg)
    # Handles the auto-trader needs; stay None when their subsystem is off.
    guardian = None
    entry_service = None
    scan_runner = None

    # Explicit feature banner — makes it unambiguous from the logs whether the
    # optional subsystems are on, instead of silence when they are off.
    log.info(
        f"Features: guardian={'ON' if cfg.guardian_enabled else 'off'}"
        f"{' (DRY RUN)' if cfg.guardian_enabled and cfg.guardian_dry_run else ''} | "
        f"scanner={'ON' if cfg.scanner_enabled else 'off'}"
        f"{' [demo market]' if (cfg.scanner_enabled and cfg.scanner_demo) else ''} | "
        f"futures env={'DEMO' if cfg.guardian_demo else 'LIVE'} "
        f"(keys={'TEST' if cfg.testnet else 'LIVE'})"
    )

    # Start API server in background (daemon thread — dies with main process)
    # Optional futures position guardian. Protects positions the operator opens
    # on futures; never opens one. Defaults to DRY RUN.
    if cfg.guardian_enabled:
      try:
        from bot.futures_guard import GuardConfig
        from bot.futures_guardian import FuturesGuardian
        from bot.api import set_guardian
        gcfg = GuardConfig(
            initial_stop_roi=cfg.guard_initial_stop_roi,
            arm_roi=cfg.guard_arm_roi,
            callback_roi=cfg.guard_callback_roi,
            trail_callback_pct=cfg.guard_trail_callback_pct,
            use_native_trail=cfg.guard_use_native_trail,
            trail_callback_roi=cfg.guard_trail_callback_roi,
            atr_stop_mult=cfg.atr_stop_mult,
            atr_stop_min_roi=cfg.atr_stop_min_roi,
            atr_stop_max_roi=cfg.atr_stop_max_roi,
            close_if_past_stop=cfg.guard_close_if_past_stop,
            breakeven_at_roi=cfg.guard_breakeven_at_roi,
            breakeven_stop_roi=cfg.guard_breakeven_stop_roi,
        fail_fast_s=cfg.guard_fail_fast_s,
        fail_fast_max_peak_roi=cfg.guard_fail_fast_max_peak_roi,
        fail_fast_loss_roi=cfg.guard_fail_fast_loss_roi,
        rescue_trail_callback_pct=cfg.guard_rescue_trail_callback_pct,
        trail_activate_now=cfg.guard_trail_activate_now,
        trail_activation_eps_pct=cfg.guard_trail_activation_eps_pct,
        stop_working_type=cfg.guard_stop_working_type,
        profit_floor_enabled=cfg.guard_profit_floor_enabled,
        taker_fee_rate=cfg.guard_taker_fee_rate,
        adaptive_trail_enabled=cfg.guard_adaptive_trail_enabled,
        fail_fast_require_worsening=cfg.guard_fail_fast_require_worsening,
        ).validate()
        guardian = FuturesGuardian(
            gcfg,
            api_key=cfg.api_key, api_secret=cfg.api_secret,
            demo=cfg.guardian_demo,
            dry_run=cfg.guardian_dry_run,
            atr_timeframe=cfg.atr_timeframe,
            poll_interval=cfg.guardian_poll_interval,
            socks_proxy=cfg.socks_proxy or None,
        )
        # The guardian needs the risk budget to police the sizing/stop invariant.
        guardian.risk_pct = cfg.entry_risk_pct
        guardian.max_closed_trades = cfg.max_closed_trades
        guardian.pending_poll_interval = cfg.guardian_pending_poll_interval
        guardian.sweep_orphan_stops = cfg.guard_sweep_orphan_stops
        # Restore before the first cycle so discovered positions keep the stop
        # they were sized for and their peak ROI, instead of being re-derived.
        if cfg.futures_state_path:
            if cfg.futures_state_reset:
                from bot import futures_state as _fs
                _fs.reset(cfg.futures_state_path, cfg.futures_state_reset)
                log.warning(
                    "FUTURES_STATE_RESET is set — this runs on EVERY restart "
                    "while present. Remove it from the environment once the "
                    "cleanup has happened.")
            # Attach BEFORE load_state so the restore path can migrate any
            # trades still sitting in the state file.
            guardian.attach_journal(
                cfg.trade_journal_path
                or str(Path(cfg.futures_state_path).with_name("trades.jsonl")),
                max_bytes=cfg.trade_journal_max_mb * 1024 * 1024,
                keep_archives=cfg.trade_journal_keep)
            guardian.load_state(cfg.futures_state_path)
            # An explicit start overrides whatever was restored. Applied AFTER
            # load_state, or the persisted value would win and the setting
            # would appear to do nothing.
            if cfg.guard_wallet_start > 0:
                prev = guardian.wallet_start
                guardian.wallet_start = float(cfg.guard_wallet_start)
                log.warning(
                    f"GUARD_WALLET_START: account-return baseline set to "
                    f"{cfg.guard_wallet_start:.2f}"
                    + (f" (was {prev:.2f})" if prev else ""))
            guardian.verify_state_path()
        set_guardian(guardian)
        guardian.start_background()

        # Operator-initiated entry depends on the guardian (shares its exchange
        # connection and dry-run flag), so it can only exist alongside it.
        if cfg.futures_entry_enabled:
            from bot.futures_entry import EntryService, EntryLimits
            from bot.api import set_entry_service
            entry_service = EntryService(guardian, EntryLimits(
                max_positions=cfg.entry_max_positions,
                max_margin_pct=cfg.entry_max_margin_pct,
                default_margin_pct=cfg.entry_default_margin_pct,
                default_callback_pct=cfg.entry_default_callback_pct,
                assumed_leverage=cfg.entry_assumed_leverage,
                target_leverage=cfg.entry_target_leverage,
                atr_stop_mult=cfg.atr_stop_mult,
                atr_stop_min_roi=cfg.atr_stop_min_roi,
                atr_stop_max_roi=cfg.atr_stop_max_roi,
                risk_pct=cfg.entry_risk_pct,
            ))
            set_entry_service(entry_service)
            # The guardian drives the reap loop and owns persistence.
            guardian._entry_service = entry_service
            guardian.entry_order_ttl_s = cfg.entry_order_ttl_s
            guardian.reap_untracked = cfg.entry_reap_untracked
            entry_service.import_placed_orders(
                getattr(guardian, "_restored_placed_orders", {}))
            log.warning(
                f"Futures ENTRY enabled — max {cfg.entry_max_positions} position(s), "
                f"margin {cfg.entry_default_margin_pct}% (cap {cfg.entry_max_margin_pct}%), "
                f"{'DRY RUN' if cfg.guardian_dry_run else 'LIVE ORDERS'}"
            )
      except Exception as e:
        log.error(f"Guardian failed to start: {e}", exc_info=True)

    # Optional read-only candidate scanner (public futures market data only —
    # no keys, cannot trade). Surfaces potential setups for operator review.
    if cfg.scanner_enabled:
      try:
        from bot.scanner import ScanConfig
        from bot.scan_runner import ScanRunner
        from bot.api import set_scanner
        scan_cfg = ScanConfig(
            min_24h_vol_usdt=cfg.scan_min_vol_usdt,
            min_abs_change_pct=cfg.scan_min_change_pct,
            short_rsi_min=cfg.scan_short_rsi_min,
            long_rsi_min=cfg.scan_long_rsi_min,
            long_rsi_max=cfg.scan_long_rsi_max,
            ema_tolerance_pct=cfg.scan_ema_tolerance_pct,
            volume_mode=cfg.scan_volume_mode,
            vol_percentile=cfg.scan_vol_percentile,
            min_atr_pct=cfg.scan_min_atr_pct,
            max_atr_pct=cfg.scan_max_atr_pct,
            recent_tr_candles=cfg.auto_recent_tr_candles,
            taper_window=cfg.auto_taper_window,
            er_lookback=cfg.auto_er_lookback,
            turn_lookback=cfg.auto_turn_lookback,
            turn_min_bars_since=cfg.auto_turn_min_bars_since,
            extreme_band_pct=cfg.scan_extreme_band_pct,
            # Express the guardian's stop in ATRs: at L leverage a -R% ROI stop
            # is an R/L % price move.
            stop_pct_for_ratio=(cfg.guard_initial_stop_roi /
                                max(cfg.entry_assumed_leverage, 1.0)),
        )
        scan_runner = ScanRunner(
            scan_cfg,
            timeframe=cfg.scanner_timeframe,
            max_symbols=cfg.scanner_max_symbols,
            socks_proxy=cfg.socks_proxy or None,
            interval=cfg.scanner_interval,
            demo=cfg.scanner_demo,
        )
        if guardian is not None:
            guardian._scanner = scan_runner     # widens the untracked sweep
        set_scanner(scan_runner)
        scan_runner.start_background()
      except Exception as e:
        log.error(f"Scanner failed to start: {e}", exc_info=True)

    # Cross-evaluation, wired INDEPENDENTLY of auto-trade.
    #
    # It used to live inside the auto-trade block, so it existed only if the
    # scanner AND the entry service both came up — and RECEIVING is useful
    # regardless: an instance with auto-trade off can still answer "would I
    # have taken this?" and still pair readings for calibration.
    _PEER_EVAL = None
    try:
        from bot.peer_eval import PeerEval
        from bot.api import set_peer_eval
        _label = (cfg.peer_eval_label
                  or ("demo" if cfg.guardian_demo else "live"))
        _PEER_EVAL = PeerEval(
            mode=cfg.peer_eval_mode,
            peer_url=cfg.peer_eval_url,
            label=_label,
            path=str(Path(cfg.futures_state_path).with_name(
                "peer-eval.jsonl")))
        set_peer_eval(_PEER_EVAL)
        if _PEER_EVAL.sends or _PEER_EVAL.receives:
            log.warning(
                f"Peer eval: mode={_PEER_EVAL.mode} as '{_label}'"
                + (f" -> {cfg.peer_eval_url}" if _PEER_EVAL.sends
                   else " (receive only)"))
        else:
            log.info(f"Peer eval: off (PEER_EVAL_MODE={cfg.peer_eval_mode!r})")
    except Exception as e:
        log.error(f"peer eval failed to start: {e}", exc_info=True)

    # Unattended auto-trading. Requires BOTH the scanner (for candidates) and
    # the entry service (for guardrailed order placement); without either it
    # cannot run, so it is only started when both are present.
    # Constructed whenever its dependencies exist, so the dashboard toggle works
    # even if it starts disabled. The AutoTradeConfig.enabled flag gates action.
    if True:
        try:
            from bot.auto_trader import AutoTradeConfig, AutoTrader
            from bot.api import set_auto_trader
            if scan_runner and entry_service and guardian:
                auto_cfg = AutoTradeConfig(
                    enabled=cfg.auto_trade_enabled,
                    max_dist_to_extreme_pct=cfg.auto_max_dist_pct,
                    veto_breakout=cfg.auto_veto_breakout,
                    long_require_convergence=cfg.auto_long_require_convergence,
                    long_require_turn=cfg.auto_long_require_turn,
                    defer_on_rising_volume=cfg.auto_defer_rising_volume,
                    defer_vol_trend=cfg.auto_defer_vol_trend,
                    defer_vol_late_trend=cfg.auto_defer_vol_late_trend,
                    callback_use_velocity=cfg.auto_callback_use_velocity,
                    **({"callback_min_pct": cfg.auto_callback_min_pct}
                       if cfg.auto_callback_min_pct > 0 else {}),
                    required_strength_sweeps=cfg.auto_strength_sweeps,
                    long_rsi_min=cfg.auto_long_rsi_min,
                    long_rsi_max=cfg.auto_long_rsi_max,
                    directions=cfg.auto_directions,
                    daily_halt_enabled=cfg.auto_daily_halt_enabled,
                    sessions=cfg.auto_sessions,
                    min_atr_pct=cfg.auto_min_atr_pct,
                    short_rsi_min=cfg.auto_short_rsi_min,
                    callback_ratio=cfg.auto_callback_ratio,
                    callback_atr_mult=cfg.auto_callback_atr_mult,
                    daily_loss_limit_pct=cfg.auto_daily_loss_limit_pct,
                    symbol_cooldown_s=cfg.auto_symbol_cooldown_s,
                    max_trades_per_hour=cfg.auto_max_trades_per_hour,
                    cooldown_override_rsi_delta=cfg.auto_cooldown_override_rsi,
                    max_reentries_per_symbol=cfg.auto_max_reentries_per_symbol,
                    max_open_positions=cfg.entry_max_positions,
                )
                auto = AutoTrader(auto_cfg, scan_runner, entry_service, guardian)
                # The guardian owns the state file; give it a way to snapshot
                # the auto-trader's safety counters alongside its own state.
                guardian._safety_snapshot = auto.safety_snapshot
                # Without this the post-loss cooldown is dead code.
                guardian.on_position_closed = auto.note_closed_trade
                auto.restore_safety(getattr(guardian, "_restored_safety", {}))
                if cfg.stream_enabled:
                    try:
                        from bot.candidate_stream import CandidateStream
                        from bot.candidate_stream import make_rest_fetcher
                        # REST is the fallback because the LIVE websocket host
                        # does not deliver to this address, while REST on the
                        # same network does. The websocket still wins when it
                        # works — _ingest keeps the newest timestamp either
                        # way, so a healthy socket simply beats the poller on
                        # recency.
                        fetcher = None
                        try:
                            fetcher = make_rest_fetcher(guardian.exchange)
                        except Exception as e:
                            log.warning(f"REST mark fetcher unavailable: {e}")
                        auto.stream = CandidateStream(
                            demo=cfg.guardian_demo,
                            stale_after_s=cfg.stream_stale_after_s,
                            proxy=_stream_proxy(cfg),
                            base_url=cfg.stream_url or None,
                            rest_fetcher=fetcher,
                            rest_interval_s=cfg.stream_rest_interval_s,
                            websocket_enabled=cfg.stream_websocket_enabled)
                        auto.stream.start()
                        # The guardian converts BNB-denominated commissions
                        # with the BNB mark this poller already harvests.
                        guardian._candidate_stream = auto.stream
                    except Exception as e:
                        log.error(f"candidate stream unavailable ({e}) — "
                                  f"entries will use the scan snapshot")
                        auto.stream = None
                auto.peer_eval = _PEER_EVAL
                set_auto_trader(auto)

                def _auto_loop():
                    import time as _t
                    while True:
                        try:
                            auto.run_once()
                        except Exception as e:
                            # A raise here abandons the WHOLE cycle, so one bad
                            # symbol stops every other candidate being
                            # considered. It logged at WARNING with no stack
                            # and repeated silently every 30s — an
                            # AttributeError from ENTRY_TARGET_LEVERAGE blocked
                            # entries entirely and looked like a quiet market.
                            log.error(f"auto-trade cycle error: {e}",
                                      exc_info=True)
                            log.error(
                                "NO ENTRIES WILL BE TAKEN until this is fixed. "
                                "The scanner and guardian are unaffected, so "
                                "the bot will look healthy.")
                        _t.sleep(cfg.auto_interval)

                threading.Thread(target=_auto_loop, daemon=True,
                                 name="auto-trader").start()
                log.warning(
                    f"Auto-trade wired: {'ENABLED' if cfg.auto_trade_enabled else 'off'} "
                    f"(toggle from the dashboard) — every {cfg.auto_interval}s")
            else:
                log.info("Auto-trade not wired: needs both the scanner and "
                         "futures entry to be enabled")
        except Exception as e:
            # LOUD. v3.26.0 renamed a config field and left one reference
            # behind; auto-trade raised at construction and the bot ran for
            # hours with the scanner and guardian working normally and NO
            # entries being taken. Nothing in the dashboard said so — it was
            # one ERROR line in a startup log nobody re-reads.
            log.error(f"Auto-trade failed to start: {e}", exc_info=True)
            log.error(
                "AUTO-TRADE IS NOT RUNNING. The scanner and guardian are "
                "unaffected, so the bot will look healthy and open NOTHING. "
                "Fix the error above and restart.")
            try:
                from bot.api import set_auto_trader_error
                set_auto_trader_error(str(e))
            except Exception:
                pass

    run_api(engine, host="0.0.0.0", port=8000)

    engine.run()
