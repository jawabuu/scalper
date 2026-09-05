#!/usr/bin/env python3
"""
Binance momentum scalping bot — entrypoint.
Starts the FastAPI dashboard in a background thread, then runs the bot loop.
"""

import logging
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

    # Explicit feature banner — makes it unambiguous from the logs whether the
    # optional subsystems are on, instead of silence when they are off.
    log.info(
        f"Features: guardian={'ON' if cfg.guardian_enabled else 'off'}"
        f"{' (DRY RUN)' if cfg.guardian_enabled and cfg.guardian_dry_run else ''} | "
        f"scanner={'ON' if cfg.scanner_enabled else 'off'}"
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
        ).validate()
        guardian = FuturesGuardian(
            gcfg,
            api_key=cfg.api_key, api_secret=cfg.api_secret,
            testnet=cfg.guardian_testnet,
            dry_run=cfg.guardian_dry_run,
            poll_interval=cfg.guardian_poll_interval,
            socks_proxy=cfg.socks_proxy or None,
        )
        set_guardian(guardian)
        guardian.start_background()

        # Operator-initiated entry depends on the guardian (shares its exchange
        # connection and dry-run flag), so it can only exist alongside it.
        if cfg.futures_entry_enabled:
            from bot.futures_entry import EntryService, EntryLimits
            from bot.api import set_entry_service
            set_entry_service(EntryService(guardian, EntryLimits(
                max_positions=cfg.entry_max_positions,
                max_margin_pct=cfg.entry_max_margin_pct,
                default_margin_pct=cfg.entry_default_margin_pct,
                default_callback_pct=cfg.entry_default_callback_pct,
            )))
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
        )
        runner = ScanRunner(
            scan_cfg,
            timeframe=cfg.scanner_timeframe,
            max_symbols=cfg.scanner_max_symbols,
            socks_proxy=cfg.socks_proxy or None,
            interval=cfg.scanner_interval,
        )
        set_scanner(runner)
        runner.start_background()
      except Exception as e:
        log.error(f"Scanner failed to start: {e}", exc_info=True)

    run_api(engine, host="0.0.0.0", port=8000)

    engine.run()
