"""Background alert scanner — runs once a day at market close (16:00 WIB, Monday–Friday).

IDX trading sessions: 09:00–16:00 WIB (UTC+7).
The scheduler fires at market close to scan daily closing anomalies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings

logger = logging.getLogger(__name__)

WIB = timezone(timedelta(hours=7))

scheduler = AsyncIOScheduler()


async def _run_alert_scan() -> None:
    now_wib = datetime.now(WIB)
    weekday = now_wib.weekday()  # 0=Mon … 6=Sun

    if weekday >= 5:
        logger.debug("Skipping alert scan — weekend (WIB: %s)", now_wib)
        return

    logger.info("Scheduled market-close alert scan and price sync starting (WIB: %s)", now_wib)

    from app.db.database import get_session_factory

    async with get_session_factory()() as db:
        # 1. Sync fresh closing prices into PostgreSQL stock_daily_prices
        from app.clients.cached_sectors import CachedSectorsClient
        from app.db.stock_prices import sync_all_daily_prices

        try:
            sectors = CachedSectorsClient(db)
            sync_res = await sync_all_daily_prices(db, sectors)
            logger.info("Market close daily price sync complete: %s", sync_res)
        except Exception:
            logger.exception("Market close daily price sync failed")

        # 2. Run market close anomaly alerts scan
        from app.agents.alert import AlertAgent

        agent = AlertAgent(db)
        try:
            result = await agent.run()
            logger.info("Scheduled scan result: %s", result)
        except Exception:
            logger.exception("Scheduled alert scan failed")


async def _run_daily_stock_insights() -> None:
    now_wib = datetime.now(WIB)
    logger.info(
        "Scheduled daily AI stock insights starting at 06:00 WIB (current WIB: %s)", now_wib
    )

    from app.agents.stock_insights import generate_all_stock_insights
    from app.db.database import get_session_factory

    async with get_session_factory()() as db:
        try:
            results = await generate_all_stock_insights(db)
            logger.info(
                "Daily AI stock insights successfully generated for %d tickers", len(results)
            )
        except Exception:
            logger.exception("Scheduled daily AI stock insights failed")


def start_scheduler() -> None:
    # Cron job: Daily alert scan once a day at market close (16:00 WIB, Mon–Fri)
    scheduler.add_job(
        _run_alert_scan,
        "cron",
        day_of_week="mon-fri",
        hour=settings.alert_scan_hour,
        minute=settings.alert_scan_minute,
        timezone=WIB,
        id="alert_scan",
        replace_existing=True,
    )
    # Cron job: Daily AI stock insight generation at 06:00 WIB (UTC+7)
    scheduler.add_job(
        _run_daily_stock_insights,
        "cron",
        hour=6,
        minute=0,
        timezone=WIB,
        id="daily_stock_insights",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started — alert scan at %02d:%02d WIB (Mon–Fri), AI insights at 06:00 WIB",
        settings.alert_scan_hour,
        settings.alert_scan_minute,
    )


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
