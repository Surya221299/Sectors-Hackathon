"""Tests for the Chatbot Agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.chatbot import (
    MCP_TOOL_WHITELIST,
    ChatbotAgent,
    ChatState,
    _build_rest_tools,
    _guard_mcp_args,
    _mcp_cache_key,
    _truncate,
    _wrap_mcp_tool,
    _wrap_mcp_tool_cached,
)

# -- Unit: _truncate ---------------------------------------------------------


def test_truncate_short_string():
    assert _truncate("short") == "short"


def test_truncate_long_string():
    result = _truncate("x" * 5000, max_chars=100)
    assert len(result) < 5000
    assert result.endswith("[truncated]")


def test_truncate_exact_boundary():
    text = "a" * 4000
    assert _truncate(text) == text


# -- Unit: _classify_question ------------------------------------------------


@pytest.mark.asyncio
async def test_classify_single_stock():
    agent = ChatbotAgent()

    mock_response = MagicMock()
    mock_response.content = json.dumps({"question_type": "single_stock", "entities": ["BBCA"]})

    with patch("app.agents.chatbot._get_llm") as mock_llm:
        llm_instance = AsyncMock()
        llm_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = llm_instance

        state: ChatState = {
            "user_message": "How is BBCA performing?",
            "chat_history": [],
            "question_type": "",
            "entities": [],
            "context": [],
            "response": "",
        }
        result = await agent._classify_question(state)

    assert result["question_type"] == "single_stock"
    assert "BBCA" in result["entities"]


@pytest.mark.asyncio
async def test_classify_comparison():
    agent = ChatbotAgent()

    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {"question_type": "comparison", "entities": ["BBCA", "BMRI"]}
    )

    with patch("app.agents.chatbot._get_llm") as mock_llm:
        llm_instance = AsyncMock()
        llm_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = llm_instance

        state: ChatState = {
            "user_message": "Compare BBCA and BMRI",
            "chat_history": [],
            "question_type": "",
            "entities": [],
            "context": [],
            "response": "",
        }
        result = await agent._classify_question(state)

    assert result["question_type"] == "comparison"
    assert result["entities"] == ["BBCA", "BMRI"]


@pytest.mark.asyncio
async def test_classify_handles_invalid_json():
    agent = ChatbotAgent()

    mock_response = MagicMock()
    mock_response.content = "not valid json at all"

    with patch("app.agents.chatbot._get_llm") as mock_llm:
        llm_instance = AsyncMock()
        llm_instance.ainvoke.return_value = mock_response
        mock_llm.return_value = llm_instance

        state: ChatState = {
            "user_message": "hello there",
            "chat_history": [],
            "question_type": "",
            "entities": [],
            "context": [],
            "response": "",
        }
        result = await agent._classify_question(state)

    assert result["question_type"] == "general"
    assert result["entities"] == []


# -- Unit: _format_context ---------------------------------------------------


def test_format_context_empty():
    agent = ChatbotAgent()
    assert agent._format_context([]) == "(no data retrieved)"


def test_format_context_with_data():
    agent = ChatbotAgent()
    ctx = [
        {"tool": "fetch-company-report", "args": {"symbol": "BBCA"}, "data": '{"pe": 20}'},
    ]
    result = agent._format_context(ctx)
    assert "fetch-company-report" in result
    assert "BBCA" in result


# -- Unit: _build_response_messages ------------------------------------------


def test_build_response_messages_includes_history():
    agent = ChatbotAgent()
    state: ChatState = {
        "user_message": "What about BBCA?",
        "chat_history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "question_type": "single_stock",
        "entities": ["BBCA"],
        "context": [],
        "response": "",
    }
    msgs = agent._build_response_messages(state)
    # system + 2 history + 1 user = 4
    assert len(msgs) == 4


def test_build_response_messages_limits_history():
    agent = ChatbotAgent()
    big_history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
    state: ChatState = {
        "user_message": "latest",
        "chat_history": big_history,
        "question_type": "general",
        "entities": [],
        "context": [],
        "response": "",
    }
    msgs = agent._build_response_messages(state)
    # system + 6 history (capped) + 1 user = 8
    assert len(msgs) == 8


# -- Unit: _guard_mcp_args ---------------------------------------------------


def test_guard_passes_tracked_ticker():
    result = _guard_mcp_args("fetch-company-report", {"symbol": "BBCA.JK"})
    assert isinstance(result, dict)
    assert result["symbol"] == "BBCA.JK"
    assert result["sections"] == "overview,valuation"


def test_guard_rejects_untracked_ticker():
    result = _guard_mcp_args("fetch-company-report", {"symbol": "GOTO"})
    assert isinstance(result, str)
    assert "not tracked" in result


def test_guard_injects_defaults():
    result = _guard_mcp_args("fetch-companies-top-changes", {})
    assert isinstance(result, dict)
    assert result["periods"] == "1d"
    assert result["n_stock"] == 5


def test_guard_does_not_override_explicit_args():
    result = _guard_mcp_args("fetch-companies-top-changes", {"periods": "7d", "n_stock": 3})
    assert isinstance(result, dict)
    assert result["periods"] == "7d"
    assert result["n_stock"] == 3


def test_guard_normalizes_ticker_case():
    result = _guard_mcp_args("fetch-daily-transaction", {"symbol": "bbca"})
    assert isinstance(result, dict)
    assert result["symbol"] == "BBCA.JK"


def test_guard_passes_tools_without_ticker():
    result = _guard_mcp_args("get-subsectors", {"some_param": "value"})
    assert isinstance(result, dict)
    assert result["some_param"] == "value"


def test_guard_foreign_flow_allows_ihsg_and_tracked():
    res_bbca = _guard_mcp_args("fetch-foreign-flow", {"symbol": "bbca"})
    assert isinstance(res_bbca, dict)
    assert res_bbca["symbol"] == "BBCA.JK"

    res_ihsg = _guard_mcp_args("fetch-foreign-flow", {"symbol": "IHSG"})
    assert isinstance(res_ihsg, dict)
    assert res_ihsg["symbol"] == "IHSG"


def test_mcp_cache_keys_coverage():
    # Documentation and server runtime tool names
    assert _mcp_cache_key("fetch-daily-transaction", {"symbol": "BBCA"}) == (
        "daily_prices:BBCA",
        300,
    )
    assert _mcp_cache_key("fetch-daily-price", {"symbol": "BBCA"}) == ("daily_prices:BBCA", 300)
    assert _mcp_cache_key("fetch-companies-by-subsector", {}) == ("companies_list", 300)
    assert _mcp_cache_key("fetch-companies", {}) == ("companies_list", 300)
    assert _mcp_cache_key("get-subsectors", {}) == ("subsectors", 3600)
    assert _mcp_cache_key("fetch-subsectors", {}) == ("subsectors", 3600)
    assert _mcp_cache_key("fetch-foreign-flow", {"symbol": "BBCA"}) == ("foreign_flow:BBCA", 3600)
    assert _mcp_cache_key("fetch-idx-market-cap", {}) == ("idx_total", 600)
    assert _mcp_cache_key("fetch-corporate-actions", {"symbol": "BBCA"}) == (
        "corporate_actions:BBCA",
        86400,
    )


def test_rest_tools_contain_foreign_flow_and_market_cap():
    mock_client = MagicMock()
    tools = _build_rest_tools(mock_client)
    tool_names = {t.name for t in tools}
    assert "get_foreign_flow" in tool_names
    assert "get_idx_market_cap" in tool_names
    assert "get_company_report" in tool_names


def test_mcp_tool_whitelist_coverage():
    expected_tools = {
        "fetch-daily-transaction",
        "fetch-daily-price",
        "fetch-companies-by-subsector",
        "fetch-companies",
        "get-subsectors",
        "fetch-subsectors",
        "fetch-foreign-flow",
        "fetch-idx-market-cap",
        "fetch-corporate-actions",
        "fetch-quarterly-financials",
        "fetch-shareholders-composition",
    }
    assert expected_tools.issubset(MCP_TOOL_WHITELIST)
    assert "fetch-daily-close" not in MCP_TOOL_WHITELIST


@pytest.mark.asyncio
async def test_wrap_mcp_tool_normalizes_and_rejects():
    mock_original = MagicMock()
    mock_original.name = "fetch-company-report"
    mock_original.description = "Test company report"
    mock_original.args_schema = None
    mock_original.ainvoke = AsyncMock(return_value='{"data": "ok"}')

    wrapped = _wrap_mcp_tool(mock_original)
    assert wrapped.name == "fetch-company-report"

    # Tracked ticker is normalized and forwarded
    res = await wrapped.ainvoke({"symbol": "bbca"})
    assert res == '{"data": "ok"}'
    mock_original.ainvoke.assert_awaited_once_with(
        {"symbol": "BBCA.JK", "sections": "overview,valuation"}
    )

    # Untracked ticker is rejected without calling original
    mock_original.ainvoke.reset_mock()
    res_reject = await wrapped.ainvoke({"symbol": "GOTO"})
    assert "not tracked" in res_reject
    mock_original.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_wrap_mcp_tool_cached_l1_hit():
    mock_original = MagicMock()
    mock_original.name = "fetch-foreign-flow"
    mock_original.description = "Test foreign flow"
    mock_original.args_schema = None
    mock_original.ainvoke = AsyncMock(return_value='{"net_foreign": 1000000}')

    wrapped = _wrap_mcp_tool_cached(mock_original, db=None)

    # First call: cache miss, calls original
    res1 = await wrapped.ainvoke({"symbol": "bbca"})
    assert "1000000" in str(res1)
    assert mock_original.ainvoke.call_count == 1

    # Second call: cache hit, original not called again
    res2 = await wrapped.ainvoke({"symbol": "bbca"})
    assert "1000000" in str(res2)
    assert mock_original.ainvoke.call_count == 1
