"""Alert Agent — monitors Sectors data for significant price/volume/sentiment moves.

Three-node LangGraph pipeline (deterministic, no LLM tool-calling):
  1. fetch_market_data  — pull top movers + news via CachedSectorsClient
  2. detect_anomalies   — apply threshold rules to flag anomalies
  3. generate_alerts    — write alert records to PostgreSQL

Data access: Sectors REST API (periodic polling) through the two-layer cache.
Triggered by POST /api/alerts/scan or a scheduled job.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.cached_sectors import CachedSectorsClient
from app.clients.sectors import bare_symbol
from app.config import settings

logger = logging.getLogger(__name__)

PRICE_CHANGE_HIGH = 0.05
PRICE_CHANGE_MEDIUM = 0.03
NEGATIVE_SENTIMENT_TAGS = {"Bearish", "Lawsuit", "Scandal", "Fraud", "Default", "Downgrade"}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AlertState(TypedDict):
    top_movers: list[dict[str, Any]]
    most_traded: list[dict[str, Any]]
    news_articles: list[dict[str, Any]]
    anomalies: list[dict[str, Any]]
    alerts_created: int


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AlertAgent:
    """LangGraph alert agent — deterministic anomaly detection.

    Nodes:
      1. fetch_market_data  — get top gainers/losers + latest news
      2. detect_anomalies   — rule-based threshold checks
      3. generate_alerts    — persist alert rows to PostgreSQL
    """

    def __init__(self, db: AsyncSession, device_id: str | None = None) -> None:
        self._db = db
        self._device_id = device_id
        self._client = CachedSectorsClient(db)

    # -- Node 1: fetch_market_data -------------------------------------------

    async def _fetch_market_data(self, state: AlertState) -> dict[str, Any]:
        movers: list[dict[str, Any]] = []
        try:
            data = await self._client.get_top_companies()
            if isinstance(data, dict):
                for category in ("top_gainers", "top_losers"):
                    period_data = data.get(category, {})
                    if isinstance(period_data, dict):
                        entries = period_data.get("1d", [])
                    elif isinstance(period_data, list):
                        entries = period_data
                    else:
                        entries = []
                    for entry in entries:
                        movers.append({**entry, "_category": category})
        except Exception as exc:
            logger.warning("Failed to fetch top companies: %s", exc)

        traded: list[dict[str, Any]] = []
        try:
            traded_data = await self._client.get_most_traded()
            if isinstance(traded_data, dict):
                latest_date = max(traded_data.keys()) if traded_data else ""
                entries = traded_data.get(latest_date, [])
                if isinstance(entries, list):
                    traded = entries
        except Exception as exc:
            logger.warning("Failed to fetch most traded: %s", exc)

        articles: list[dict[str, Any]] = []
        try:
            news_data = await self._client.get_news()
            results = news_data.get("results", []) if isinstance(news_data, dict) else []
            if isinstance(results, list):
                articles = results
        except Exception as exc:
            logger.warning("Failed to fetch news: %s", exc)

        logger.info(
            "Fetched %d movers, %d most-traded, %d news articles",
            len(movers),
            len(traded),
            len(articles),
        )
        return {"top_movers": movers, "most_traded": traded, "news_articles": articles}

    # -- Node 2: detect_anomalies --------------------------------------------

    async def _synthesize_alert_message(
        self,
        ticker: str,
        name: str,
        alert_type: str,
        severity: str,
        context_data: dict[str, Any],
    ) -> str:
        """Generate a high-density, informative analytical alert for investors.
        Uses Google Gemini when available; falls back to structured institutional templates.
        """
        if settings.gemini_api_key and not settings.use_mock_data:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=settings.gemini_api_key)
                system_instruction = (
                    "You are an expert equity research analyst for the Indonesia Stock Exchange (IDX). "
                    "Synthesize a clear, highly informative, and analytical market alert notification (2-3 sentences, 40-75 words) "
                    "for retail investors regarding an IDX stock anomaly. "
                    "Explain: 1) What the specific quantitative anomaly is. 2) The market/institutional context behind it. "
                    "3) Actionable takeaway or key support/resistance levels for investors to watch. "
                    "Do NOT use markdown headers, prefixes, or disclaimers. Output only the synthesized paragraph."
                )
                prompt = (
                    f"Ticker: {ticker} ({name})\n"
                    f"Alert Type: {alert_type} (Severity: {severity})\n"
                    f"Details: {context_data}\n"
                )
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=settings.gemini_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.3,
                            max_output_tokens=250,
                        ),
                    ),
                    timeout=4.0,
                )
                text_out = (response.text or "").strip()
                if len(text_out) > 40:
                    return text_out
            except Exception as exc:
                logger.debug("LLM alert synthesis skipped/failed: %s", exc)

        # Deterministic Institutional Fallback Templates
        news_articles = context_data.get("news", [])
        news_snippet = ""
        if news_articles:
            titles = [a.get("title", "") for a in news_articles if a.get("title")]
            if titles:
                news_snippet = f" Catalyst is reinforced by {len(news_articles)} negative news headline(s), including '{titles[0]}'."
            else:
                news_snippet = f" Catalyst is accompanied by {len(news_articles)} negative news headline(s)."

        if alert_type == "price_spike":
            direction = context_data.get("direction", "up")
            change = context_data.get("change", 0.0)
            price = context_data.get("price", 0)
            action = "surged up by" if direction == "up" else "dropped down by"
            takeaway = (
                "Strong momentum indicates aggressive buying interest; watch whether trading volume sustains a breakout above upper resistance."
                if direction == "up"
                else "Elevated selling pressure triggers a volatility flag; monitor nearby support zones to gauge whether the dip is being absorbed."
            )
            return (
                f"{name} ({ticker}) {action} {change:.1%} to Rp {price:,.0f}, triggering a {severity}-severity price breakout on the IDX. "
                f"{takeaway}{news_snippet}"
            )

        if alert_type == "volume_surge":
            volume = context_data.get("volume", 0)
            price = context_data.get("price", 0)
            return (
                f"{name} ({ticker}) entered the IDX top most-traded list with exceptional trading liquidity of {volume:,.0f} lots at Rp {price:,.0f}. "
                "An abnormal volume surge typically indicates institutional rebalancing, block accumulation, or heavy liquidity turnover. "
                f"Investors should track institutional foreign net flow and observe whether price action confirms a directional trend.{news_snippet}"
            )

        if alert_type == "sentiment_shift":
            titles = context_data.get("titles", [])
            count = context_data.get("count", 1)
            example = f" (e.g., '{titles[0]}')" if titles else ""
            return (
                f"{ticker} recorded a significant sentiment shift with {count} negative news headline(s) detected{example}. "
                "Negative regulatory or media scrutiny can induce heightened short-term price volatility. "
                "Investors are advised to review fundamental solvency and debt ratios rather than reacting solely to headline sentiment."
            )

        return f"{name} ({ticker}): {alert_type} detected at severity {severity}."

    async def _detect_anomalies(self, state: AlertState) -> dict[str, Any]:
        anomalies: list[dict[str, Any]] = []
        seen_tickers: set[str] = set()

        for mover in state["top_movers"]:
            symbol = mover.get("symbol", "")
            ticker = bare_symbol(symbol)
            if not ticker or ticker in seen_tickers or ticker not in settings.tracked_tickers:
                continue

            change = abs(mover.get("price_change", 0.0))
            price = mover.get("last_close_price", 0)
            name = mover.get("name", ticker)
            direction = "up" if mover.get("price_change", 0) > 0 else "down"

            if change >= PRICE_CHANGE_HIGH:
                severity = "high"
            elif change >= PRICE_CHANGE_MEDIUM:
                severity = "medium"
            else:
                continue

            seen_tickers.add(ticker)
            msg = await self._synthesize_alert_message(
                ticker,
                name,
                "price_spike",
                severity,
                {"direction": direction, "change": change, "price": price},
            )
            anomalies.append(
                {
                    "ticker": ticker,
                    "alert_type": "price_spike",
                    "severity": severity,
                    "message": msg,
                }
            )

        for entry in state["most_traded"]:
            symbol = entry.get("symbol", "")
            ticker = bare_symbol(symbol)
            if not ticker or ticker not in settings.tracked_tickers or ticker in seen_tickers:
                continue
            volume = entry.get("volume", 0)
            price = entry.get("price", 0)
            name = entry.get("company_name", ticker)
            seen_tickers.add(ticker)
            msg = await self._synthesize_alert_message(
                ticker,
                name,
                "volume_surge",
                "medium",
                {"volume": volume, "price": price},
            )
            anomalies.append(
                {
                    "ticker": ticker,
                    "alert_type": "volume_surge",
                    "severity": "medium",
                    "message": msg,
                }
            )

        news_by_ticker: dict[str, list[dict[str, Any]]] = {}
        for article in state["news_articles"]:
            tags = set(article.get("tags", []))
            if not tags & NEGATIVE_SENTIMENT_TAGS:
                continue
            for sym in article.get("symbols", []):
                t = bare_symbol(sym)
                if t in settings.tracked_tickers:
                    news_by_ticker.setdefault(t, []).append(article)

        for ticker, articles in news_by_ticker.items():
            if ticker in seen_tickers:
                for a in anomalies:
                    if a["ticker"] == ticker:
                        titles = [art.get("title", "") for art in articles if art.get("title")]
                        title_str = f", including '{titles[0]}'" if titles else ""
                        a["message"] += (
                            f" Catalyst is reinforced by {len(articles)} negative news headline(s){title_str}."
                        )
                        break
                continue

            titles = [a.get("title", "") for a in articles[:3]]
            severity = "high" if len(articles) >= 3 else "medium"
            msg = await self._synthesize_alert_message(
                ticker,
                ticker,
                "sentiment_shift",
                severity,
                {"count": len(articles), "titles": titles, "news": articles},
            )
            anomalies.append(
                {
                    "ticker": ticker,
                    "alert_type": "sentiment_shift",
                    "severity": severity,
                    "message": msg,
                }
            )

        logger.info("Detected %d anomalies", len(anomalies))
        return {"anomalies": anomalies}

    # -- Node 3: generate_alerts ---------------------------------------------

    async def _generate_alerts(self, state: AlertState) -> dict[str, Any]:
        created = 0
        now = datetime.now(UTC)

        device_filter = "AND device_id = :device_id" if self._device_id else "AND device_id IS NULL"

        for anomaly in state["anomalies"]:
            existing = await self._db.execute(
                text(
                    "SELECT id FROM alerts "
                    "WHERE ticker = :ticker AND alert_type = :alert_type "
                    f"AND created_at > :since {device_filter}"
                ),
                {
                    "ticker": anomaly["ticker"],
                    "alert_type": anomaly["alert_type"],
                    "since": now.replace(hour=0, minute=0, second=0, microsecond=0),
                    "device_id": self._device_id,
                },
            )
            if existing.first() is not None:
                logger.debug(
                    "Skipping duplicate alert for %s/%s", anomaly["ticker"], anomaly["alert_type"]
                )
                continue

            await self._db.execute(
                text(
                    "INSERT INTO alerts "
                    "(device_id, ticker, alert_type, severity, message, created_at) "
                    "VALUES (:device_id, :ticker, :alert_type, :severity, :message, :created_at)"
                ),
                {
                    "device_id": self._device_id,
                    "ticker": anomaly["ticker"],
                    "alert_type": anomaly["alert_type"],
                    "severity": anomaly["severity"],
                    "message": anomaly["message"],
                    "created_at": now,
                },
            )
            created += 1

        if created:
            await self._db.commit()

        logger.info("Created %d new alerts", created)
        return {"alerts_created": created}

    # -- Graph ---------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(AlertState)
        graph.add_node("fetch_market_data", self._fetch_market_data)
        graph.add_node("detect_anomalies", self._detect_anomalies)
        graph.add_node("generate_alerts", self._generate_alerts)
        graph.set_entry_point("fetch_market_data")
        graph.add_edge("fetch_market_data", "detect_anomalies")
        graph.add_edge("detect_anomalies", "generate_alerts")
        graph.add_edge("generate_alerts", END)
        return graph.compile()

    # -- Public API ----------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Execute the full alert pipeline and return a summary."""
        graph = self._build_graph()
        initial: AlertState = {
            "top_movers": [],
            "most_traded": [],
            "news_articles": [],
            "anomalies": [],
            "alerts_created": 0,
        }
        result = await graph.ainvoke(initial)
        return {
            "anomalies_detected": len(result["anomalies"]),
            "alerts_created": result["alerts_created"],
        }
