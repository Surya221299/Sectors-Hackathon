"""Database helper for reading, writing, and synchronizing stock_daily_prices."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.cached_sectors import CachedSectorsClient
from app.clients.sectors import bare_symbol
from app.config import settings

logger = logging.getLogger(__name__)


async def get_db_daily_prices(db: AsyncSession, ticker: str) -> list[dict[str, Any]]:
    """Retrieve daily prices from PostgreSQL sorted chronologically."""
    clean = bare_symbol(ticker).upper()
    try:
        res = await db.execute(
            text(
                "SELECT date, open, high, low, close, volume FROM stock_daily_prices "
                "WHERE ticker = :t ORDER BY date ASC"
            ),
            {"t": clean},
        )
        rows = res.mappings().all()
        return [
            {
                "date": str(r["date"]),
                "open": float(r["open"]) if r["open"] is not None else None,
                "high": float(r["high"]) if r["high"] is not None else None,
                "low": float(r["low"]) if r["low"] is not None else None,
                "close": float(r["close"]),
                "volume": int(r["volume"]) if r["volume"] is not None else None,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("Failed to fetch daily prices for %s from DB: %s", clean, exc)
        return []


async def upsert_stock_daily_prices(
    db: AsyncSession, ticker: str, raw_prices: list[dict[str, Any]]
) -> int:
    """Upsert daily prices into PostgreSQL stock_daily_prices table."""
    if not raw_prices:
        return 0

    clean = bare_symbol(ticker).upper()
    upsert_sql = text(
        "INSERT INTO stock_daily_prices (ticker, date, open, high, low, close, volume) "
        "VALUES (:ticker, :date, :open, :high, :low, :close, :volume) "
        "ON CONFLICT (ticker, date) DO UPDATE SET "
        "  open = EXCLUDED.open, "
        "  high = EXCLUDED.high, "
        "  low = EXCLUDED.low, "
        "  close = EXCLUDED.close, "
        "  volume = EXCLUDED.volume"
    )

    params = []
    for p in raw_prices:
        p_date = p.get("date")
        p_close = p.get("close")
        if not p_date or p_close is None:
            continue

        parsed_date = date.fromisoformat(str(p_date)) if isinstance(p_date, str) else p_date
        params.append(
            {
                "ticker": clean,
                "date": parsed_date,
                "open": float(p["open"]) if p.get("open") is not None else None,
                "high": float(p["high"]) if p.get("high") is not None else None,
                "low": float(p["low"]) if p.get("low") is not None else None,
                "close": float(p_close),
                "volume": int(p["volume"]) if p.get("volume") is not None else None,
            }
        )

    if params:
        await db.execute(upsert_sql, params)
        await db.commit()
        logger.info("Batch upserted %d daily price points into DB for %s", len(params), clean)

    return len(params)


async def sync_daily_prices_for_ticker(
    db: AsyncSession, sectors: CachedSectorsClient, ticker: str
) -> list[dict[str, Any]]:
    """Fetch daily prices from Sectors API and upsert into PostgreSQL."""
    clean = bare_symbol(ticker).upper()
    try:
        raw_prices = cast(list[dict[str, Any]], await sectors.get_daily_prices(clean))
        if raw_prices:
            normalized = [
                {
                    "date": str(p["date"]),
                    "open": float(p["open"]) if p.get("open") is not None else None,
                    "high": float(p["high"]) if p.get("high") is not None else None,
                    "low": float(p["low"]) if p.get("low") is not None else None,
                    "close": float(p["close"]),
                    "volume": int(p["volume"]) if p.get("volume") is not None else None,
                }
                for p in raw_prices
            ]
            try:
                await upsert_stock_daily_prices(db, clean, raw_prices)
            except Exception as exc:
                logger.warning("Could not persist daily prices to DB for %s: %s", clean, exc)
            return normalized
    except Exception as exc:
        logger.error("Failed to fetch daily prices for %s: %s", clean, exc)

    return await get_db_daily_prices(db, clean)


async def sync_all_daily_prices(db: AsyncSession, sectors: CachedSectorsClient) -> dict[str, int]:
    """Sync all tracked tickers daily prices into PostgreSQL."""
    results: dict[str, int] = {}
    for ticker in settings.tracked_tickers:
        prices = cast(list[dict[str, Any]], await sectors.get_daily_prices(ticker))
        count = await upsert_stock_daily_prices(db, ticker, prices)
        results[ticker] = count

    logger.info("Daily price sync complete for all %d tickers", len(results))
    return results
