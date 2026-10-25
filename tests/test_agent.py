"""Routing, validation and graph tests - no API calls needed."""
from types import SimpleNamespace

import pytest

from src.agent.router import extract_tickers, route
from src.validation.answer_validator import validate_numbers


@pytest.mark.parametrize("question, must_have, must_not_have", [
    ("What is Apple's RSI?", {"get_technical_indicators"}, {"search_sec_filings", "get_recent_news"}),
    ("What risks did Microsoft report?", {"search_sec_filings"}, {"get_technical_indicators"}),
    ("Why did Nvidia's stock move recently?", {"get_price_summary", "get_recent_news"}, set()),
    ("Compare Apple's technical trend and business risks",
     {"get_technical_indicators", "search_sec_filings"}, set()),
])
def test_routing(question, must_have, must_not_have):
    tools = set(route(question)["tools"])
    assert must_have <= tools
    assert not (must_not_have & tools)


def test_technical_query_skips_rag():
    assert "search_sec_filings" not in route("Show the MACD and SMA for TSLA")["tools"]


def test_vague_question_gets_all_tools():
    assert len(route("Tell me about AMZN")["tools"]) == 4


@pytest.mark.parametrize("question, expected", [
    ("What is Apple's RSI?", ["AAPL"]),
    ("Compare MSFT and $NVDA", ["MSFT", "NVDA"]),
    ("Is the SEC filing for TSLA risky?", ["TSLA"]),
])
def test_extract_tickers(question, expected):
    assert extract_tickers(question) == expected


def test_validator_accepts_tool_numbers():
    tool_out = [{"rsi_14": 63.42, "close": 231.5, "signals": {"trend": "uptrend"}}]
    v = validate_numbers("RSI is 63.42 and the close was $231.50.", tool_out)
    assert v["passed"] and v["numbers_checked"] == 2


def test_validator_flags_invented_numbers():
    v = validate_numbers("RSI is 71.9 today.", [{"rsi_14": 63.42}])
    assert not v["passed"] and v["unverified"] == ["71.9"]


def test_validator_ignores_integers():
    assert validate_numbers("The 14-day RSI over 30 days in 2026", [{}])["passed"]


class _Block(SimpleNamespace):
    pass


class _FakeMessages:
    """Scripted Claude: first asks for the RSI tool, then answers using its output."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(stop_reason="tool_use", content=[
                _Block(type="tool_use", id="t1", name="get_technical_indicators", input={"ticker": "AAPL"})])
        return SimpleNamespace(stop_reason="end_turn", content=[
            _Block(type="text", text="## Summary\nAAPL RSI is 55.55.")])


def test_graph_end_to_end(monkeypatch):
    from src.agent import graph

    fake = _FakeMessages()
    monkeypatch.setattr(graph, "_client", SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    monkeypatch.setitem(graph.TOOL_FUNCTIONS, "get_technical_indicators",
                        lambda ticker: {"ticker": ticker, "rsi_14": 55.55})

    result = graph.build_graph().invoke({"question": "What is Apple's RSI?"})

    assert [t["name"] for t in result["tool_log"]] == ["get_technical_indicators"]
    assert result["answer"].endswith("55.55.")
    assert result["validation"]["passed"]
    # Router restricted the tools offered to Claude
    offered = {t["name"] for t in fake.calls[0]["tools"]}
    assert "search_sec_filings" not in offered
    # Second call carried the tool result back
    assert fake.calls[1]["messages"][-1]["content"][0]["type"] == "tool_result"
