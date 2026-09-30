from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.stock_insights import generate_stock_insights_for_ticker
from app.api.deps import get_sectors
from app.clients.cached_sectors import CachedSectorsClient
from app.clients.sectors import bare_symbol
from app.db.database import get_db
from app.db.scores import latest_scores
from app.db.stock_insights import latest_stock_insights, stock_insights_generated_today
from app.models.schemas import (
    Fundamentals,
    InsightChip,
    PricePoint,
    StockDetail,
    StockInsightsResponse,
    StockListResponse,
    StockSummary,
)

router = APIRouter()


def pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 2)


def ratio(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def to_summary(row: dict[str, Any], score_info: dict[str, Any] | None = None) -> StockSummary:
    q = row["query_values"]
    ticker = bare_symbol(row["symbol"])
    return StockSummary(
        ticker=ticker,
        name=row["company_name"],
        sector=q["sector"],
        sub_sector=q["sub_sector"],
        price=q["last_close_price"],
        change_pct=pct(q.get("daily_close_change")),
        market_cap=q.get("market_cap"),
        recommendation=score_info.get("recommendation") if score_info else None,
        overall_score=score_info.get("overall_score") if score_info else None,
    )


async def tracked_rows(sectors: CachedSectorsClient) -> list[dict[str, Any]]:
    screener = cast(dict[str, Any], await sectors.list_companies())
    return cast(list[dict[str, Any]], screener["results"])


@router.get("/stocks", response_model=StockListResponse)
async def list_stocks(
    db: AsyncSession = Depends(get_db),
    sectors: CachedSectorsClient = Depends(get_sectors),
) -> StockListResponse:
    rows = await tracked_rows(sectors)
    try:
        scores = await latest_scores(db)
        score_map = {s["ticker"]: s for s in scores}
    except Exception:
        score_map = {}
    return StockListResponse(
        stocks=[to_summary(r, score_map.get(bare_symbol(r["symbol"]))) for r in rows]
    )


@router.get("/stock/{ticker}", response_model=StockDetail)
async def get_stock(
    ticker: str,
    db: AsyncSession = Depends(get_db),
    sectors: CachedSectorsClient = Depends(get_sectors),
) -> StockDetail:
    ticker = bare_symbol(ticker)
    rows = await tracked_rows(sectors)
    row = next((r for r in rows if bare_symbol(r["symbol"]) == ticker), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"{ticker} is not tracked")
    q = row["query_values"]
    latest_close_date = q.get("latest_close_date")

    # Priority 1: Check PostgreSQL `stock_daily_prices` table in database
    db_rows: Any = []
    try:
        db_prices = await db.execute(
            text(
                "SELECT date, open, high, low, close, volume FROM stock_daily_prices "
                "WHERE ticker = :t ORDER BY date ASC"
            ),
            {"t": ticker},
        )
        db_rows = db_prices.mappings().all()
    except Exception:
        db_rows = []

    daily: list[dict[str, Any]]
    if db_rows:
        db_last_date = str(db_rows[-1]["date"])
        # If DB data is behind the latest close date from the market, auto-sync fresh prices
        if latest_close_date and db_last_date < str(latest_close_date):
            from app.db.stock_prices import sync_daily_prices_for_ticker

            daily = await sync_daily_prices_for_ticker(db, sectors, ticker)
        else:
            daily = [
                {
                    "date": str(r["date"]),
                    "open": float(r["open"]) if r["open"] is not None else None,
                    "high": float(r["high"]) if r["high"] is not None else None,
                    "low": float(r["low"]) if r["low"] is not None else None,
                    "close": float(r["close"]),
                    "volume": int(r["volume"]) if r["volume"] is not None else None,
                }
                for r in db_rows
            ]
    else:
        from app.db.stock_prices import sync_daily_prices_for_ticker

        daily = await sync_daily_prices_for_ticker(db, sectors, ticker)

    scores_list: list[dict[str, Any]] = []
    try:
        scores_list = await latest_scores(db)
    except Exception:
        scores_list = []
    score_info = next((s for s in scores_list if s["ticker"] == ticker), None)

    # Fetch stored AI insights for this ticker from PostgreSQL if available
    insight_chips: list[InsightChip] | None = None
    try:
        insight_rows = await latest_stock_insights(db, ticker)
        chips = [InsightChip(label=r["label"], text=r["content"]) for r in insight_rows]
        if score_info and score_info.get("reasoning"):
            chips.insert(0, InsightChip(label="Scoring Agent", text=score_info["reasoning"]))
        if chips:
            insight_chips = chips
    except Exception:
        if score_info and score_info.get("reasoning"):
            insight_chips = [InsightChip(label="Scoring Agent", text=score_info["reasoning"])]
        else:
            insight_chips = None

    q = row["query_values"]
    return StockDetail(
        **to_summary(row, score_info).model_dump(),
        fundamentals=Fundamentals(
            pe=ratio(q.get("pe_ttm")),
            pb=ratio(q.get("pb_mrq")),
            roe_pct=pct(q.get("roe_ttm")),
            der=ratio(q.get("der_mrq")),
            dividend_yield_pct=pct(q.get("yield_ttm")),
        ),
        week52_high=q.get("52_w_high_price"),
        week52_low=q.get("52_w_low_price"),
        prices=[
            PricePoint(
                date=str(d["date"]),
                open=float(d["open"]) if d.get("open") is not None else None,
                high=float(d["high"]) if d.get("high") is not None else None,
                low=float(d["low"]) if d.get("low") is not None else None,
                close=float(d["close"]),
                volume=int(d["volume"]) if d.get("volume") is not None else None,
            )
            for d in daily
        ],
        insights=insight_chips,
    )


@router.get("/stock/{ticker}/insights", response_model=StockInsightsResponse)
async def get_stock_insights(
    ticker: str,
    db: AsyncSession = Depends(get_db),
) -> StockInsightsResponse:
    clean_ticker = bare_symbol(ticker).upper()

    try:
        # If not generated today, generate now and persist to PostgreSQL
        if not await stock_insights_generated_today(db, clean_ticker):
            await generate_stock_insights_for_ticker(db, clean_ticker)
        rows = await latest_stock_insights(db, clean_ticker)
        if not rows:
            await generate_stock_insights_for_ticker(db, clean_ticker)
            rows = await latest_stock_insights(db, clean_ticker)
    except Exception:
        try:
            rows = await latest_stock_insights(db, clean_ticker)
        except Exception:
            rows = []

    chips = [InsightChip(label=r["label"], text=r["content"]) for r in rows]
    try:
        scores_list = await latest_scores(db)
        score_info = next((s for s in scores_list if s["ticker"] == clean_ticker), None)
        if score_info and score_info.get("reasoning"):
            chips.insert(0, InsightChip(label="Scoring Agent", text=score_info["reasoning"]))
    except Exception:
        pass

    if not chips:
        raise HTTPException(
            status_code=404,
            detail=f"Insights not available for {clean_ticker}",
        )

    gen_date = str(rows[0]["generated_date"]) if rows else "today"
    return StockInsightsResponse(
        ticker=clean_ticker,
        generated_date=gen_date,
        insights=chips,
    )
