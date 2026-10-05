"""Chatbot Agent — LangGraph workflow with Sectors MCP / REST tool-calling.

Three-node graph: classify_question -> retrieve_context -> generate_response.
The LLM autonomously picks which Sectors tools to query based on the user's
question — this is the agentic tool-use behavior.

Two data modes controlled by ``settings.use_mcp``:
  - **MCP** (demo): connects to the hosted Sectors MCP server — live data,
    burns API credits per question.
  - **REST** (development default): uses CachedSectorsClient with two-layer
    cache (L1 memory + L2 PostgreSQL) — saves credits.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import END, StateGraph
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import cache_get, cache_set
from app.clients.cached_sectors import CachedSectorsClient
from app.clients.sectors import bare_symbol
from app.config import settings

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 4000
MAX_TOOL_ROUNDS = 3

MCP_TOOL_WHITELIST = {
    # Core Fundamentals & Reports
    "fetch-company-report",
    # Daily Price & Transactions (documentation & server runtime names)
    "fetch-daily-transaction",
    "fetch-daily-price",
    # Screener & Companies (documentation & server runtime names)
    "fetch-companies-by-subsector",
    "fetch-companies",
    # Subsectors (documentation & server runtime names)
    "get-subsectors",
    "fetch-subsectors",
    # Market Movers & Rankings
    "fetch-most-traded-stocks",
    "fetch-companies-top-changes",
    "fetch-index-daily",
    "fetch-idx-market-cap",
    "fetch-subsector-report",
    # News & Regulatory Filings
    "fetch-news",
    "fetch-filings",
    # Foreign Flow & Institutional Activity
    "fetch-foreign-flow",
    # Extended Financials, Corporate Actions & Ownership
    "fetch-corporate-actions",
    "fetch-quarterly-financials",
    "fetch-shareholders-composition",
}

TICKER_PARAM_NAMES = {"symbol", "symbols", "ticker"}

MCP_DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "fetch-company-report": {"sections": ["overview", "valuation"]},
    "fetch-companies-top-changes": {"periods": ["1d"], "n_stock": 5},
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ChatState(TypedDict):
    user_message: str
    chat_history: list[dict[str, str]]
    question_type: str
    entities: list[str]
    context: list[dict[str, Any]]
    response: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """\
You are a financial question classifier for Indonesian stock market (IDX) analysis.

Classify the user's message into one of these types:
- portfolio: About the user's stock portfolio, holdings, portfolio risk, or stock outlook
  without a ticker specified (e.g. "My Stock Outlook", "My portfolio Risks",
  "Why Did My Stock Move?", "What stocks do I have in my portfolio?")
- recommendation: About recommended stocks, top picks, scoring rankings, or best stocks to buy
  (e.g. "Recommended Stocks", "Top 3 stocks", "What stocks do you recommend?")
- single_stock: About a specific stock (e.g. "How is BBCA performing?")
- comparison: Comparing two or more stocks (e.g. "Compare BBCA vs BMRI")
- sector: About a sector/industry (e.g. "How is the banking sector?")
- market: About the overall market or IHSG index
- general: General finance question, greeting, or unclear

Extract any stock tickers mentioned (uppercase IDX tickers like BBCA, TLKM).

Respond ONLY with valid JSON:
{"question_type": "...", "entities": ["TICKER1", "TICKER2"]}"""

RETRIEVAL_PROMPT = """\
You are a data retrieval agent for Indonesian stock market analysis.
Use the available tools to fetch the Sectors data and internal app data needed to answer.

Rules:
- For portfolio questions (e.g. "My Stock Outlook", "My portfolio Risks"):
  1. Call get_my_portfolio to retrieve the user's current holdings and average buy price.
  2. If the user owns stocks, call get_company_report, get_daily_prices, or get_news for those
     held tickers so you can evaluate their performance, outlook, and risk.
- For recommendation questions (e.g. "Recommended Stocks", "Top stocks"):
  1. Call get_top_recommended_stocks to get the top 3 AI-scored stocks from the database.
  2. You may also fetch sector reports or market movers if helpful.
- For single stock investment thesis, buy/sell questions, or outlook (e.g. "Why should I buy BMRI?", "Should I buy BBCA?", "Is BBRI a buy?"):
  1. If internal database tools are available, call get_stock_ai_score and/or get_stock_ai_insights to fetch Invelio's proprietary 5-pillar score, recommendation status, and equity thesis.
  2. Call get_company_report to retrieve valuation (P/E, PBV, ROE, dividend yield, market cap rank).
  3. Call get_insider_transactions, get_foreign_flow, or get_news to uncover institutional signals, insider accumulation, and recent catalysts.
- For single stock questions, fetch the company report and optionally news or insider transactions.
- For comparisons, fetch company reports for each ticker.
- For sector questions, fetch the sector/subsector report.
- For market questions, fetch market index data and top movers.
- For foreign capital flow, institutional accumulation, or net foreign buy/sell, fetch foreign flow.
- For corporate actions, dividends schedule, or stock splits, fetch corporate actions.
- For quarterly financial performance, fetch quarterly financials.
- Call multiple tools when comparing stocks.
- IDX tickers do NOT include the .JK suffix (e.g. use BBCA, not BBCA.JK).
- Do NOT fabricate data — only return what the tools provide.

Question type: {question_type}
Detected tickers: {entities}"""

RESPONSE_PROMPT = """\
You are Invelio, an intelligent AI financial research assistant for the Indonesian stock market (IDX).

Your answers MUST be grounded in the retrieved data below.
Never fabricate financial figures. If data is missing or insufficient, state it clearly.

Language Rule:
- Always respond in the SAME LANGUAGE as the user's inquiry (e.g., respond in English if asked in English, Bahasa Indonesia if asked in Indonesian).

Response Structure & Directness:
- CRITICAL: Never start with generic data preambles like "Based on the latest Sectors data, PT [Company] closed at Rp X on Date..." or raw dates.
- Answer the user's question directly from the very first sentence.

Specific Query Types:
1. Stock Thesis & "Why Buy / Should I Buy / Is [Ticker] a Good Buy?" Queries:
   - Immediate Opening: Open directly with the core investment thesis answering why investors are considering or bullish on this stock (e.g. "Here is the key investment thesis and primary reasons why investors consider buying PT Bank Mandiri (BMRI):" or in Indonesian "Berikut adalah tesis investasi utama dan alasan investor mempertimbangkan membeli PT Bank Mandiri (BMRI):").
   - Key Investment Catalysts ("Why Buy"): Structure the arguments into clear bold bullet points synthesizing retrieved data:
     * Valuation & Profitability: Forward P/E, P/BV vs sector median, ROE capital efficiency, strong balance sheet.
     * Dividend Yield & Capital Return: High dividend yield (e.g. >5% yield), consistent shareholder payouts.
     * Institutional & Insider Confidence: Insider accumulation (executives/directors purchasing shares), sovereign/institutional backing (Danantara tags), or foreign institutional capital flow.
     * Industry Dominance & Scale: Market cap ranking, digital banking or loan growth, competitive moat.
     * Invelio AI Score & Recommendation: If available in retrieved data, explicitly cite Invelio's quantitative score (0-100) and BUY/HOLD recommendation status.
   - Key Risks to Monitor: Provide a brief, balanced section (1-2 concise bullet points) detailing key risk factors to watch (e.g. interest rate cycle, credit quality/NPLs, macroeconomic headwinds).
   - Disclaimer: Do NOT start with defensive apologies ("As an AI assistant, I do not provide financial advice..."). Present the objective analysis first, and place a compact 1-line disclaimer at the very end: "*Disclaimer: For research and educational purposes only; not personalized financial advice.*"

2. Portfolio Queries:
   - If the user has holdings, summarize positions (ticker, shares, average price) and analyze outlook and risks based on retrieved data.
   - If empty, warmly inform them that their portfolio currently has no tracked holdings, and invite them to add stocks in the Portfolio tab.

3. Top Recommendations Queries:
   - Present Top 3 AI recommendations clearly with rank, ticker, company name, overall score, recommendation status (BUY/HOLD/SELL), and key reasoning.
   - Explain that recommendations are powered by Invelio's 5-pillar multi-factor model (Fundamental 30%, Macro 15%, Sector 20%, Risk 15%, Sentiment 20%).

4. General Single Stock / Sector / Comparison Queries:
   - Use clean, structured bullet points.
   - Format Indonesian Rupiah cleanly (e.g. Rp 4,030, Rp 372.4T).
   - Always keep numbers accurate from retrieved data.

Formatting & Heading Mandate:
- CRITICAL: NEVER use markdown heading hashtags (#, ##, ###) for titles, subtitles, or section names anywhere in your response.
- The mobile UI cannot render markdown header tags and will display raw "###".
- Always use bold text with a colon instead (e.g., "**Tesis Investasi Utama:**", "**Risiko yang Perlu Dipantau:**", "**Key Investment Thesis:**").

[Retrieved Data]
{context}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _extract_text(content: Any) -> str:
    """Normalize LLM content to a plain string (Gemini returns a list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


def _get_llm(temperature: float = 0.0, streaming: bool = False) -> BaseChatModel:
    if settings.chatbot_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.chatbot_model,
            temperature=temperature,
            google_api_key=settings.gemini_api_key,
            streaming=streaming,
            max_retries=1,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.chatbot_model,
        temperature=temperature,
        api_key=SecretStr(settings.openai_api_key) if settings.openai_api_key else None,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# REST tools — wraps CachedSectorsClient (development mode)
# ---------------------------------------------------------------------------


def _validate_ticker(ticker: str) -> str | None:
    """Return normalized ticker if tracked, else None."""
    t = ticker.upper().removesuffix(".JK")
    return t if t in settings.tracked_tickers else None


def _build_rest_tools(client: CachedSectorsClient) -> list[Any]:
    @tool
    async def get_company_report(ticker: str) -> str:
        """Get a company's financial report: overview, valuation
        (PE, PBV, ROE, DER), market cap, dividend yield.

        Tracked: BBCA BBRI BMRI BBNI TLKM ASII UNVR ICBP AMRT ANTM."""
        t = _validate_ticker(ticker)
        if t is None:
            return f"Ticker {ticker!r} is not tracked."
        data = await client.get_company_report(t)
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_daily_prices(ticker: str) -> str:
        """Get 90-day daily OHLCV price history for a stock.

        Tracked: BBCA BBRI BMRI BBNI TLKM ASII UNVR ICBP AMRT ANTM."""
        t = _validate_ticker(ticker)
        if t is None:
            return f"Ticker {ticker!r} is not tracked."
        data = await client.get_daily_prices(t)
        if isinstance(data, list) and len(data) > 10:
            summary = {
                "total_days": len(data),
                "latest_10": data[-10:],
                "oldest": data[0] if data else None,
            }
            return _truncate(json.dumps(summary, default=str))
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_stock_list() -> str:
        """Get all tracked IDX stocks with sector, sub-sector, and latest price."""
        data = await client.list_companies()
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_most_traded() -> str:
        """Get today's most actively traded stocks by volume."""
        data = await client.get_most_traded()
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_top_movers() -> str:
        """Get today's top gainers and top losers in the IDX market."""
        data = await client.get_top_companies()
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_market_index() -> str:
        """Get the IHSG (Jakarta Composite Index) data and recent trend."""
        data = await client.get_ihsg()
        if isinstance(data, list) and len(data) > 10:
            summary = {"total_days": len(data), "latest_10": data[-10:]}
            return _truncate(json.dumps(summary, default=str))
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_sector_report(sub_sector: str) -> str:
        """Get performance report for an IDX sub-sector.

        Valid sub-sectors: banks, telecommunication, basic-materials, food-beverage,
        food-staples-retailing, multi-sector-holdings, nondurable-household-products."""
        data = await client.get_sector_report(sub_sector)
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_news(ticker: str = "") -> str:
        """Get latest news articles. Pass a ticker for stock-specific
        news, or empty string for all market news."""
        if ticker:
            t = _validate_ticker(ticker)
            if t is None:
                return f"Ticker {ticker!r} is not tracked."
            data = await client.get_news(t)
        else:
            data = await client.get_news(None)
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_insider_transactions(ticker: str = "") -> str:
        """Get insider transactions (buy/sell) for a stock.
        Pass a ticker or empty for all."""
        if ticker:
            t = _validate_ticker(ticker)
            if t is None:
                return f"Ticker {ticker!r} is not tracked."
            data = await client.get_news_filings(t)
        else:
            data = await client.get_news_filings(None)
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_foreign_flow(ticker: str = "IHSG") -> str:
        """Get foreign institutional investor fund flow (net foreign buy/sell in IDR)
        for a stock or the overall IHSG market.

        Tracked: BBCA BBRI BMRI BBNI TLKM ASII UNVR ICBP AMRT ANTM, or IHSG."""
        target = "IHSG"
        if ticker and ticker.upper() != "IHSG":
            t = _validate_ticker(ticker)
            if t is None:
                return f"Ticker {ticker!r} is not tracked."
            target = t
        data = await client.get_foreign_flow(target)
        return _truncate(json.dumps(data, default=str))

    @tool
    async def get_idx_market_cap() -> str:
        """Get historical total market capitalization for the Indonesian Stock Exchange (IDX)."""
        data = await client.get_idx_total()
        if isinstance(data, list) and len(data) > 10:
            summary = {"total_days": len(data), "latest_10": data[-10:]}
            return _truncate(json.dumps(summary, default=str))
        return _truncate(json.dumps(data, default=str))

    return [
        get_company_report,
        get_daily_prices,
        get_stock_list,
        get_most_traded,
        get_top_movers,
        get_market_index,
        get_idx_market_cap,
        get_sector_report,
        get_news,
        get_insider_transactions,
        get_foreign_flow,
    ]


# ---------------------------------------------------------------------------
# Internal Database Tools (Portfolio & AI Recommendations)
# ---------------------------------------------------------------------------


def _build_internal_tools(db: AsyncSession | None, device_id: str | None) -> list[Any]:
    @tool
    async def get_my_portfolio() -> str:
        """Get the user's current stock portfolio holdings (tickers, stock names, total shares,
        average buy price, and total invested) for this device from the database.
        Use this when the user asks about 'my stock', 'my portfolio', or portfolio risk/outlook."""
        if db is None or not device_id:
            return "No portfolio data available (database session or device ID not available)."
        try:
            rows = await db.execute(
                text(
                    "SELECT ticker, stock_name, shares, price_per_share, total_invested, buy_date "
                    "FROM user_holdings WHERE device_id = :device ORDER BY ticker, buy_date ASC"
                ),
                {"device": device_id},
            )
            records = rows.fetchall()
            if not records:
                return (
                    "User's portfolio is currently empty (0 stocks held). "
                    "The user has not added any holdings yet."
                )

            holdings_map: dict[str, dict[str, Any]] = {}
            for r in records:
                ticker = r.ticker
                shares = float(r.shares)
                total_invested = float(r.total_invested)
                if ticker not in holdings_map:
                    holdings_map[ticker] = {
                        "ticker": ticker,
                        "stock_name": r.stock_name,
                        "total_shares": 0.0,
                        "total_invested": 0.0,
                        "lots_count": 0,
                    }
                holdings_map[ticker]["total_shares"] += shares
                holdings_map[ticker]["total_invested"] += total_invested
                holdings_map[ticker]["lots_count"] += 1

            summary = []
            for h in holdings_map.values():
                avg_price = (
                    round(h["total_invested"] / h["total_shares"], 2)
                    if h["total_shares"] > 0
                    else 0.0
                )
                summary.append(
                    {
                        "ticker": h["ticker"],
                        "stock_name": h["stock_name"],
                        "total_shares": h["total_shares"],
                        "avg_buy_price": avg_price,
                        "total_invested_idr": round(h["total_invested"], 2),
                        "lots_count": h["lots_count"],
                    }
                )
            return _truncate(json.dumps({"portfolio_holdings": summary}, default=str))
        except Exception as exc:
            logger.warning("Failed to fetch user portfolio: %s", exc)
            return f"Error retrieving user portfolio: {exc}"

    @tool
    async def get_top_recommended_stocks() -> str:
        """Get the top 3 AI-recommended Indonesian stocks from the database scoring engine.
        Returns rank, ticker, company name, overall score (0-100), recommendation (BUY/HOLD/SELL),
        component scores, and analytical reasoning.
        Use this when the user asks for 'recommended stocks', 'top picks', or 'best stocks'."""
        if db is None:
            return "Recommendations unavailable (database not connected)."
        try:
            from app.db.scores import latest_scores

            rows = await latest_scores(db)
            if not rows:
                return (
                    "No stock recommendations available yet in the database. "
                    "The scoring agent has not produced any scores."
                )

            top3 = []
            for idx, r in enumerate(rows[:3]):
                top3.append(
                    {
                        "rank": idx + 1,
                        "ticker": r.get("ticker"),
                        "name": r.get("name"),
                        "overall_score": r.get("overall_score"),
                        "recommendation": r.get("recommendation"),
                        "reasoning": r.get("reasoning"),
                        "scores": {
                            "fundamental": r.get("fundamental_score"),
                            "macro": r.get("macro_score"),
                            "sector": r.get("sector_score"),
                            "risk": r.get("risk_score"),
                            "sentiment": r.get("sentiment_score"),
                        },
                    }
                )
            return _truncate(json.dumps({"top_recommendations": top3}, default=str))
        except Exception as exc:
            logger.warning("Failed to fetch recommended stocks: %s", exc)
            return f"Error retrieving recommendations: {exc}"

    @tool
    async def get_stock_ai_score(ticker: str) -> str:
        """Get Invelio's proprietary AI quantitative score (0-100), recommendation (BUY/HOLD/SELL),
        5-pillar component scores (Fundamental, Macro, Sector, Risk, Sentiment), and equity thesis
        for a specific tracked IDX ticker (e.g. BMRI, BBCA, BBRI)."""
        if db is None:
            return "Stock score unavailable (database not connected)."
        clean = _validate_ticker(ticker)
        if clean is None:
            return f"Ticker {ticker!r} is not tracked."
        try:
            from app.db.scores import latest_scores

            rows = await latest_scores(db)
            match = next((r for r in rows if r.get("ticker") == clean), None)
            if not match:
                return f"No score recorded yet for {clean}."
            return _truncate(
                json.dumps(
                    {
                        "ticker": match.get("ticker"),
                        "name": match.get("name"),
                        "overall_score": match.get("overall_score"),
                        "recommendation": match.get("recommendation"),
                        "thesis": match.get("reasoning"),
                        "component_scores": {
                            "fundamental": match.get("fundamental_score"),
                            "macro": match.get("macro_score"),
                            "sector": match.get("sector_score"),
                            "risk": match.get("risk_score"),
                            "sentiment": match.get("sentiment_score"),
                        },
                    },
                    default=str,
                )
            )
        except Exception as exc:
            logger.warning("Failed to fetch stock score for %s: %s", clean, exc)
            return f"Error retrieving stock score: {exc}"

    @tool
    async def get_stock_ai_insights(ticker: str) -> str:
        """Get Invelio's daily 4-card AI equity research insights (Technical Analysis,
        Fundamentals, Market Sentiment, and Outlook & Risks) for a specific tracked IDX stock."""
        if db is None:
            return "Stock insights unavailable (database not connected)."
        clean = _validate_ticker(ticker)
        if clean is None:
            return f"Ticker {ticker!r} is not tracked."
        try:
            from app.db.stock_insights import latest_stock_insights

            insights = await latest_stock_insights(db, clean)
            if not insights:
                return f"No structured AI insights available yet for {clean}."
            return _truncate(
                json.dumps(
                    [
                        {
                            "analysis_type": i.get("analysis_type"),
                            "label": i.get("label"),
                            "content": i.get("content"),
                        }
                        for i in insights
                    ],
                    default=str,
                )
            )
        except Exception as exc:
            logger.warning("Failed to fetch stock insights for %s: %s", clean, exc)
            return f"Error retrieving stock insights: {exc}"

    return [
        get_my_portfolio,
        get_top_recommended_stocks,
        get_stock_ai_score,
        get_stock_ai_insights,
    ]


# ---------------------------------------------------------------------------
# MCP tools — connects to hosted Sectors MCP server (demo mode)
# ---------------------------------------------------------------------------


def _guard_mcp_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any] | str:
    """Validate ticker and inject cost-saving defaults for MCP tool calls.

    Returns the cleaned args dict, or an error string if the ticker is rejected.
    """
    guarded = dict(args)

    for param in TICKER_PARAM_NAMES:
        if param not in guarded:
            continue
        raw = guarded[param]
        if isinstance(raw, str):
            tickers = [t.strip() for t in raw.split(",")]
        elif isinstance(raw, list):
            tickers = raw
        else:
            continue
        cleaned = [t.upper().removesuffix(".JK") for t in tickers if t.strip()]
        allowed_set = set(settings.tracked_tickers) | (
            {"IHSG"} if tool_name in {"fetch-foreign-flow", "fetch-index-daily"} else set()
        )
        rejected = [t for t in cleaned if t not in allowed_set]
        if rejected:
            allowed = ", ".join(settings.tracked_tickers)
            return f"Ticker {', '.join(rejected)} not tracked. Use: {allowed}"
        guarded[param] = (
            ",".join(f"{t}.JK" if t != "IHSG" else t for t in cleaned)
            if isinstance(raw, str)
            else cleaned
        )

    defaults = MCP_DEFAULT_ARGS.get(tool_name, {})
    for key, value in defaults.items():
        if key not in guarded:
            guarded[key] = value

    if tool_name == "fetch-company-report" and "sections" in guarded:
        if isinstance(guarded["sections"], str):
            guarded["sections"] = [s.strip() for s in guarded["sections"].split(",") if s.strip()]
    if tool_name == "fetch-companies-top-changes" and "periods" in guarded:
        if isinstance(guarded["periods"], str):
            guarded["periods"] = [p.strip() for p in guarded["periods"].split(",") if p.strip()]

    return guarded


def _wrap_mcp_tool(original: Any) -> Any:
    """Wrap an MCP tool with ticker validation and default-arg injection."""
    name = getattr(original, "name", "")
    description = getattr(original, "description", "")
    args_schema = getattr(original, "args_schema", None)

    async def guarded_coro(**kwargs: Any) -> Any:
        result = _guard_mcp_args(name, kwargs)
        if isinstance(result, str):
            return result
        return await original.ainvoke(result)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,  # type: ignore[arg-type]
        coroutine=guarded_coro,
    )


def _mcp_cache_key(tool_name: str, args: dict[str, Any]) -> tuple[str, int] | None:
    """Map an MCP tool call to (cache_key, ttl). Returns None if not cacheable."""

    def _ticker(args: dict[str, Any]) -> str:
        for p in TICKER_PARAM_NAMES:
            raw = args.get(p, "")
            if raw:
                t = raw.split(",")[0].strip() if isinstance(raw, str) else raw[0]
                return bare_symbol(t)
        return ""

    mapping: dict[str, tuple[str, int]] = {
        "fetch-company-report": (
            f"company_report:{_ticker(args)}",
            settings.cache_ttl_fundamentals,
        ),
        # Daily price & transactions (documentation & server runtime names)
        "fetch-daily-transaction": (
            f"daily_prices:{_ticker(args)}",
            settings.cache_ttl_prices,
        ),
        "fetch-daily-price": (
            f"daily_prices:{_ticker(args)}",
            settings.cache_ttl_prices,
        ),
        # Screener & companies (documentation & server runtime names)
        "fetch-companies-by-subsector": (
            "companies_list",
            settings.cache_ttl_prices,
        ),
        "fetch-companies": (
            "companies_list",
            settings.cache_ttl_prices,
        ),
        "fetch-most-traded-stocks": (
            "most_traded",
            settings.cache_ttl_prices,
        ),
        "fetch-companies-top-changes": (
            "top_companies",
            settings.cache_ttl_prices,
        ),
        # Market index & market cap
        "fetch-index-daily": (
            "ihsg",
            settings.cache_ttl_market_index,
        ),
        "fetch-idx-market-cap": (
            "idx_total",
            settings.cache_ttl_market_index,
        ),
        # Sector & subsector reports (documentation & server runtime names)
        "fetch-subsector-report": (
            f"sector_report:{args.get('sub_sector', args.get('subsector', ''))}",
            settings.cache_ttl_sector_reports,
        ),
        "get-subsectors": (
            "subsectors",
            settings.cache_ttl_sector_reports,
        ),
        "fetch-subsectors": (
            "subsectors",
            settings.cache_ttl_sector_reports,
        ),
        # News & Filings
        "fetch-news": (
            f"news:{_ticker(args)}" if _ticker(args) else "news:all",
            settings.cache_ttl_news,
        ),
        "fetch-filings": (
            f"news_filings:{_ticker(args)}" if _ticker(args) else "news_filings:all",
            settings.cache_ttl_sector_reports,
        ),
        # Foreign flow & institutional analysis
        "fetch-foreign-flow": (
            f"foreign_flow:{_ticker(args) or 'IHSG'}",
            settings.cache_ttl_sector_reports,
        ),
        # Corporate actions & dividends
        "fetch-corporate-actions": (
            f"corporate_actions:{_ticker(args)}",
            settings.cache_ttl_fundamentals,
        ),
        # Quarterly financials
        "fetch-quarterly-financials": (
            f"quarterly_financials:{_ticker(args)}",
            settings.cache_ttl_fundamentals,
        ),
        # Shareholders composition
        "fetch-shareholders-composition": (
            f"shareholders:{_ticker(args)}",
            settings.cache_ttl_fundamentals,
        ),
    }
    return mapping.get(tool_name)


def _wrap_mcp_tool_cached(original: Any, db: AsyncSession | None) -> Any:
    """Wrap an MCP tool with ticker validation, default-arg injection, and L1/L2 cache."""
    name = getattr(original, "name", "")
    description = getattr(original, "description", "")
    args_schema = getattr(original, "args_schema", None)

    async def cached_coro(**kwargs: Any) -> Any:
        result = _guard_mcp_args(name, kwargs)
        if isinstance(result, str):
            return result

        cache_info = _mcp_cache_key(name, result)
        if cache_info is not None:
            cache_key, ttl = cache_info
            mcp_key = f"mcp:{cache_key}"
            hit = await cache_get(mcp_key, db)
            if hit is not None:
                logger.debug("MCP cache hit: %s", mcp_key)
                return hit.get("result", hit)
            data = await original.ainvoke(result)
            data_str = str(data).lower()
            if (
                "validation error" not in data_str
                and "invalid arguments" not in data_str
                and "error" not in data_str[:100]
            ):
                await cache_set(mcp_key, {"result": data}, ttl, db)
            return data

        return await original.ainvoke(result)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,  # type: ignore[arg-type]
        coroutine=cached_coro,
    )


@asynccontextmanager
async def _mcp_tools(db: AsyncSession | None = None) -> Any:
    """Connect to Sectors MCP server and yield cache-wrapped LangChain tools."""
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = settings.sectors_mcp_url
    auth_token = settings.sectors_api_key.strip()
    bearer = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"
    headers = {"Authorization": bearer} if auth_token else {}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            all_tools = await load_mcp_tools(session)
            tools = [
                _wrap_mcp_tool_cached(t, db) for t in all_tools if t.name in MCP_TOOL_WHITELIST
            ]
            logger.info(
                "Loaded %d/%d MCP tools from Sectors (cache-enabled)", len(tools), len(all_tools)
            )
            yield tools


# ---------------------------------------------------------------------------
# Unified tool loader
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _get_tools(db: AsyncSession | None = None, device_id: str | None = None) -> Any:
    """Yield tools from MCP (demo) or REST+cache (development), along with internal DB tools."""
    internal_tools = _build_internal_tools(db, device_id)
    if settings.use_mcp:
        logger.info("Chatbot mode: MCP (live Sectors data, cache-enabled)")
        async with _mcp_tools(db) as tools:
            yield internal_tools + tools
    else:
        logger.info("Chatbot mode: REST + cache")
        client = CachedSectorsClient(db)
        rest_tools = _build_rest_tools(client)
        yield internal_tools + rest_tools


def _detect_user_language(text: str) -> str:
    """Detect if the user query is predominantly Indonesian or English."""
    id_words = {
        "kenapa",
        "harus",
        "beli",
        "saham",
        "apakah",
        "bagus",
        "turun",
        "anjlok",
        "rugi",
        "bagaimana",
        "rekomendasi",
        "portofolio",
        "dividen",
        "apa",
        "alasan",
        "layak",
        "untung",
        "prospek",
        "kinerja",
        "analisis",
        "berapa",
        "investasi",
        "saya",
        "ini",
    }
    tokens = set(text.lower().replace("?", " ").replace("!", " ").replace(".", " ").split())
    if tokens & id_words:
        return "id"
    return "en"


def _synthesize_analyst_directive(
    user_message: str, question_type: str, entities: list[str]
) -> str:
    """Generate dynamic analytical reasoning instructions tailored specifically to the user's inquiry.

    This ensures Gemini does not output static, defensive, or boilerplate company profiles,
    but instead critically interprets the retrieved Sectors & Invelio AI data to directly
    answer the user's specific investment angle (e.g. 'Why Buy', 'Why did it drop', 'Dividend outlook').
    """
    msg = user_message.lower()
    ticker_str = ", ".join(entities) if entities else "the requested stock"
    lang = _detect_user_language(user_message)
    lang_instruction = (
        "LANGUAGE MANDATE: You MUST respond in fluent, professional Bahasa Indonesia."
        if lang == "id"
        else "LANGUAGE MANDATE: You MUST respond in clear, professional English."
    )

    # 1. Buy Thesis / Why Buy / Should I Buy
    buy_signals = [
        "why buy",
        "why should i buy",
        "should i buy",
        "kenapa beli",
        "kenapa harus beli",
        "layak beli",
        "alasan beli",
        "prospek beli",
        "beli atau tidak",
        "rekomendasi beli",
        "worth buying",
        "good buy",
        "is it a buy",
        "reasons to buy",
        "apakah layak",
        "apakah bagus",
        "rekomendasi",
        "alasan untuk beli",
    ]
    if any(s in msg for s in buy_signals):
        if lang == "id":
            return (
                f"\n\n[MANDATORY ANALYTICAL TASK: BUY THESIS & INVESTMENT REASONS FOR {ticker_str}]\n"
                f"{lang_instruction}\n"
                f'User Inquiry: "{user_message}"\n'
                "Instruksi Analis: Anda adalah Senior Equity Research Analyst Invelio. Pengguna menanyakan ALASAN MEMBELI / TESIS INVESTASI.\n"
                "FORMAT WAJIB: DILARANG KERAS menggunakan tanda pagar markdown (###, ##, #) untuk judul atau pemisah bagian apapun karena aplikasi akan menampilkan tanda pagar mentah. Gunakan format teks tebal (**Judul Bagian:**).\n"
                "DILARANG memberikan pembukaan basa-basi atau profil umum perusahaan ('Bank Mandiri didirikan tahun...').\n"
                "Olah seluruh data Sectors dan Invelio yang terambil di atas menjadi TESIS INVESTASI BERBOBOT dan PERSUASIF:\n"
                "1. **Buka Langsung dengan Inti Tesis di Kalimat Pertama:** (Contoh: 'Berikut adalah tesis investasi utama dan alasan mengapa investor mempertimbangkan untuk membeli PT [Perusahaan] ([Ticker]):')\n"
                "2. **Kelompokkan Bukti Menjadi Poin Katalis Tebal (Bold Bullet Points):**\n"
                "   - **Valuasi Menarik & Rasio Keuangan:** Bandingkan P/E saat ini, forward P/E, dan PBV terhadap median industri. Jelaskan mengapa angka ini murah/menarik.\n"
                "   - **Profitabilitas & Efisiensi Modal:** Paparkan Return on Equity (ROE) dan margin laba yang kuat.\n"
                "   - **Imbal Hasil Dividen (Dividend Yield):** Sorot persentase dividend yield (>5%) dan rekam jejak dividen tunai.\n"
                "   - **Dukungan Institusi & Sinyal Smart Money:** Sebutkan keterlibatan investor strategis/Danantara, akumulasi orang dalam (insider buying), atau net foreign buy.\n"
                "   - **Skor Kuantitatif Invelio AI:** Jika ada di data, cantumkan skor keseluruhan (0-100) dan rekomendasi BUY/HOLD dari model multi-faktor Invelio.\n"
                "3. **Risiko yang Perlu Dipantau:** 1-2 risiko realistis (misal tekanan makro/outflow asing) agar analisis berimbang.\n"
                "4. **Disclaimer Singkat di Akhir:** Taruh 1 baris disclaimer edukasi di paling bawah: '*Disclaimer: For research and educational purposes only; not personalized financial advice.*'"
            )
        else:
            return (
                f"\n\n[MANDATORY ANALYTICAL TASK: BUY THESIS & INVESTMENT REASONS FOR {ticker_str}]\n"
                f"{lang_instruction}\n"
                f'User Inquiry: "{user_message}"\n'
                "Analyst Directive: You are a Senior Equity Research Analyst for Invelio. The user is asking WHY they should BUY / invest in this stock.\n"
                "CRITICAL FORMATTING MANDATE: NEVER use markdown heading hashtags (###, ##, #) for any headers or section titles. Use bold text (**Section Title:**) instead.\n"
                "Do NOT provide a generic boilerplate company history, passive definitions, or defensive apologies.\n"
                "Synthesize the retrieved Sectors and Invelio data into a compelling, data-backed Bull Case Investment Thesis:\n"
                "1. **Open Directly with the Core Thesis in Sentence 1:** (e.g., 'Here is the key investment thesis and primary reasons why investors consider buying PT [Company] ([Ticker]):')\n"
                "2. **Structure Evidence into Bold Catalysts:**\n"
                "   - **Valuation & Multiples:** Compare current P/E, forward P/E, and PBV against sector peers to demonstrate value discount.\n"
                "   - **Profitability & Capital Efficiency:** Highlight Return on Equity (ROE) and capital efficiency.\n"
                "   - **Dividend Yield & Cash Returns:** Emphasize dividend yield percentage (>5%) and payout consistency.\n"
                "   - **Institutional & Smart Money Backing:** Cite insider accumulation, sovereign investment (Danantara), or institutional flows.\n"
                "   - **Invelio AI Model Verdict:** Explicitly cite Invelio's proprietary quantitative score (0-100) and BUY/HOLD recommendation.\n"
                "3. **Key Risks to Monitor:** 1-2 concise, balanced risks (e.g. macro headwinds, rates) for professional balance.\n"
                "4. **Compact Disclaimer at End:** '*Disclaimer: For research and educational purposes only; not personalized financial advice.*'"
            )

    # 2. Price drop / Sell / Negative performance
    drop_signals = [
        "why drop",
        "why fell",
        "why down",
        "kenapa turun",
        "anjlok",
        "kenapa merah",
        "turun hari ini",
        "rugi",
        "cut loss",
    ]
    if any(s in msg for s in drop_signals):
        return (
            f"\n\n[MANDATORY ANALYTICAL TASK: PRICE DROP & VOLATILITY ANALYSIS FOR {ticker_str}]\n"
            f"{lang_instruction}\n"
            "Analyze the retrieved daily price changes, foreign capital flow (net foreign sell/outflow), "
            "and negative news sentiment to clearly explain the market catalysts behind the drop and key support levels. "
            "Do NOT use markdown hashtags (###). Use bold text for sections."
        )

    # 3. Dividend inquiry
    dividend_signals = ["dividend", "dividen", "yield", "payout", "jadwal dividen"]
    if any(s in msg for s in dividend_signals):
        return (
            f"\n\n[MANDATORY ANALYTICAL TASK: DIVIDEND & INCOME ANALYSIS FOR {ticker_str}]\n"
            f"{lang_instruction}\n"
            "Focus directly on {ticker_str}'s dividend yield (TTM), historical dividend track record, "
            "cash flow stability, and corporate action schedules from the retrieved data. "
            "Do NOT use markdown hashtags (###). Use bold text for sections."
        )

    # 4. Multi-stock comparison
    if len(entities) >= 2 or question_type == "comparison":
        return (
            f"\n\n[MANDATORY ANALYTICAL TASK: HEAD-TO-HEAD COMPARISON FOR {ticker_str}]\n"
            f"{lang_instruction}\n"
            "Compare the tickers head-to-head across Valuation (P/E, PBV), Profitability (ROE), "
            "Dividend Yield, Market Cap scale, and Invelio AI Scores. Give a clear analytical verdict on which stock suits what investor profile. "
            "Do NOT use markdown hashtags (###). Use bold text for sections."
        )

    # Default single stock
    if entities or question_type == "single_stock":
        return (
            f"\n\n{lang_instruction}\nDirectly analyze {ticker_str} using the retrieved data above."
        )

    return f"\n\n{lang_instruction}"


def _clean_markdown_headings(text: str) -> str:
    """Convert or remove markdown heading hashtags (#, ##, ###) so chat responses never contain raw ###."""
    if not text:
        return ""

    def _replace_header(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if content.startswith("**") and content.endswith("**"):
            return content
        return f"**{content}**"

    # Convert lines starting with '#' followed by heading text into bold text
    cleaned = re.sub(r"(?m)^#{1,6}\s+(.+)$", _replace_header, text)
    # Strip any remaining stray hash sequences
    cleaned = re.sub(r"#{2,6}", "", cleaned)
    return cleaned


async def _stream_clean_headings(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Filter markdown heading hashtags (#, ##, ###) on the fly during SSE streaming."""
    line_buf = ""
    in_heading = False
    checking_line_start = True

    async for chunk in chunks:
        pos = 0
        while pos < len(chunk):
            if in_heading:
                nl_idx = chunk.find("\n", pos)
                if nl_idx != -1:
                    line_buf += chunk[pos : nl_idx + 1]
                    yield _clean_markdown_headings(line_buf)
                    line_buf = ""
                    in_heading = False
                    checking_line_start = True
                    pos = nl_idx + 1
                else:
                    line_buf += chunk[pos:]
                    pos = len(chunk)
            elif checking_line_start:
                nl_idx = chunk.find("\n", pos)
                segment = chunk[pos : nl_idx + 1] if nl_idx != -1 else chunk[pos:]
                line_buf += segment
                stripped = line_buf.lstrip()
                if stripped.startswith("#"):
                    in_heading = True
                    if nl_idx != -1:
                        yield _clean_markdown_headings(line_buf)
                        line_buf = ""
                        in_heading = False
                        checking_line_start = True
                    pos = nl_idx + 1 if nl_idx != -1 else len(chunk)
                elif stripped:
                    yield line_buf
                    line_buf = ""
                    checking_line_start = False
                    if nl_idx != -1:
                        checking_line_start = True
                    pos = nl_idx + 1 if nl_idx != -1 else len(chunk)
                elif nl_idx != -1:
                    yield line_buf
                    line_buf = ""
                    checking_line_start = True
                    pos = nl_idx + 1
                else:
                    pos = len(chunk)
            else:
                nl_idx = chunk.find("\n", pos)
                if nl_idx != -1:
                    yield chunk[pos : nl_idx + 1]
                    checking_line_start = True
                    pos = nl_idx + 1
                else:
                    yield chunk[pos:]
                    pos = len(chunk)

    if line_buf:
        yield _clean_markdown_headings(line_buf)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ChatbotAgent:
    """LangGraph chatbot with Sectors tool-calling (MCP or REST).

    Nodes:
      1. classify_question  — categorize the question + extract tickers
      2. retrieve_context   — LLM picks which Sectors tools to call (agentic)
      3. generate_response  — produce a grounded answer from retrieved data
    """

    def __init__(self, db: AsyncSession | None = None, device_id: str | None = None) -> None:
        self._db = db
        self._device_id = device_id
        self._tools: list[Any] = []
        self._tool_map: dict[str, Any] = {}

    # -- Node implementations -----------------------------------------------

    async def _classify_question(self, state: ChatState) -> dict[str, Any]:
        llm = _get_llm(temperature=0.0)
        result = await llm.ainvoke(
            [
                SystemMessage(content=CLASSIFY_PROMPT),
                HumanMessage(content=state["user_message"]),
            ]
        )
        try:
            parsed = json.loads(_extract_text(result.content))
            return {
                "question_type": parsed.get("question_type", "general"),
                "entities": parsed.get("entities", []),
            }
        except (json.JSONDecodeError, AttributeError, TypeError):
            return {"question_type": "general", "entities": []}

    async def _retrieve_context(self, state: ChatState) -> dict[str, Any]:
        llm_with_tools = _get_llm(temperature=0.0).bind_tools(self._tools)
        system = RETRIEVAL_PROMPT.format(
            question_type=state["question_type"],
            entities=state["entities"],
        )
        messages: list[Any] = [
            SystemMessage(content=system),
            HumanMessage(content=state["user_message"]),
        ]

        context_parts: list[dict[str, Any]] = []
        for _ in range(MAX_TOOL_ROUNDS):
            result = await llm_with_tools.ainvoke(messages)
            messages.append(result)

            if not result.tool_calls:
                break

            for tc in result.tool_calls:
                fn = self._tool_map.get(tc["name"])
                if fn is None:
                    tool_output = f"Unknown tool: {tc['name']}"
                else:
                    try:
                        tool_output = await fn.ainvoke(tc["args"])
                        tool_output = _truncate(str(tool_output))
                    except Exception as exc:
                        logger.warning("Tool %s failed: %s", tc["name"], exc)
                        tool_output = f"Error fetching data: {exc}"

                context_parts.append({"tool": tc["name"], "args": tc["args"], "data": tool_output})
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))

        return {"context": context_parts}

    def _format_context(self, context: list[dict[str, Any]]) -> str:
        parts = []
        for item in context:
            args_str = (
                ", ".join(f"{k}={v!r}" for k, v in item["args"].items()) if item["args"] else ""
            )
            data_str = str(item["data"])
            # Filter out ugly technical validation errors so they do not pollute LLM context
            if "validation error" in data_str.lower() or "not match schema" in data_str.lower():
                continue
            parts.append(f"[{item['tool']}({args_str})]\n{data_str}")
        return "\n\n".join(parts) if parts else "(no data retrieved)"

    def _build_response_messages(self, state: ChatState) -> list[Any]:
        context_text = self._format_context(state.get("context", []))
        directive = _synthesize_analyst_directive(
            state["user_message"],
            state.get("question_type", ""),
            state.get("entities", []),
        )
        system = RESPONSE_PROMPT.format(context=context_text)
        if directive:
            system = f"{system}\n\n{directive}"

        history_msgs: list[Any] = []
        for msg in (state.get("chat_history") or [])[-6:]:
            if msg["role"] == "user":
                history_msgs.append(HumanMessage(content=msg["content"]))
            else:
                history_msgs.append(AIMessage(content=msg["content"]))

        user_content = state["user_message"]
        if directive and "MANDATORY ANALYTICAL TASK" in directive:
            user_content = (
                f"{user_content}\n\n"
                f"[Analyst Reminder: Synthesize the investment thesis directly answering this inquiry using the data above.]"
            )

        return [
            SystemMessage(content=system),
            *history_msgs,
            HumanMessage(content=user_content),
        ]

    async def _generate_response(self, state: ChatState) -> dict[str, Any]:
        llm = _get_llm(temperature=0.3)
        messages = self._build_response_messages(state)
        result = await llm.ainvoke(messages)
        return {"response": _clean_markdown_headings(_extract_text(result.content))}

    # -- Graph ---------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(ChatState)
        graph.add_node("classify_question", self._classify_question)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_response", self._generate_response)
        graph.set_entry_point("classify_question")
        graph.add_edge("classify_question", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
        graph.add_edge("generate_response", END)
        return graph.compile()

    # -- Public API ----------------------------------------------------------

    async def run(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        """Execute the full pipeline and return the complete response."""
        async with _get_tools(self._db, self._device_id) as tools:
            self._tools = tools
            self._tool_map = {t.name: t for t in tools}
            graph = self._build_graph()
            initial: ChatState = {
                "user_message": message,
                "chat_history": history or [],
                "question_type": "",
                "entities": [],
                "context": [],
                "response": "",
            }
            result = await graph.ainvoke(initial)
            return cast(str, result["response"])

    async def stream(
        self, message: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        """Run classify + retrieve, then stream the final response tokens."""
        async with _get_tools(self._db, self._device_id) as tools:
            self._tools = tools
            self._tool_map = {t.name: t for t in tools}

            state: ChatState = {
                "user_message": message,
                "chat_history": history or [],
                "question_type": "",
                "entities": [],
                "context": [],
                "response": "",
            }

            classified = await self._classify_question(state)
            state["question_type"] = classified["question_type"]
            state["entities"] = classified["entities"]

            retrieved = await self._retrieve_context(state)
            state["context"] = retrieved["context"]

            messages = self._build_response_messages(state)
            llm = _get_llm(temperature=0.3, streaming=True)

            async def _raw_token_generator() -> AsyncIterator[str]:
                async for chunk in llm.astream(messages):
                    text = _extract_text(chunk.content) if chunk.content else ""
                    if text:
                        yield text

            async for cleaned_chunk in _stream_clean_headings(_raw_token_generator()):
                yield cleaned_chunk
