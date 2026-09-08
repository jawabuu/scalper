#!/usr/bin/env python3
"""
Binance momentum scalping bot — entrypoint.
Starts the FastAPI dashboard in a background thread, then runs the bot loop.
"""

import logging
import threading
import os
import sys

from bot import BotConfig, ScalpingEngine, __version__
from bot.api import run_api


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
            guardian.load_state(cfg.futures_state_path)
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
                    required_strength_sweeps=cfg.auto_strength_sweeps,
                    long_rsi_min=cfg.auto_long_rsi_min,
                    long_rsi_max=cfg.auto_long_rsi_max,
                    directions=cfg.auto_directions,
                    daily_halt_enabled=cfg.auto_daily_halt_enabled,
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
                auto.restore_safety(getattr(guardian, "_restored_safety", {}))
                set_auto_trader(auto)

                def _auto_loop():
                    import time as _t
                    while True:
                        try:
                            auto.run_once()
                        except Exception as e:
                            log.warning(f"auto-trade cycle error: {e}")
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
            log.error(f"Auto-trade failed to start: {e}", exc_info=True)

    run_api(engine, host="0.0.0.0", port=8000)

    engine.run()
