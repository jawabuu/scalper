#!/usr/bin/env python3
"""
Binance momentum scalping bot — entrypoint.
Starts the FastAPI dashboard in a background thread, then runs the bot loop.
"""

import logging
import os
import sys

from bot import BotConfig, ScalpingEngine
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
        f"Starting — testnet={cfg.testnet} tf={cfg.timeframe} "
        f"stop={cfg.trailing_stop_pct}% {tp_info} "
        f"risk={cfg.risk_per_trade_pct}%/trade"
    )

    engine = ScalpingEngine(cfg)

    # Start API server in background (daemon thread — dies with main process)
    run_api(engine, host="0.0.0.0", port=8000)

    engine.run()
