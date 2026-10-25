"""Tool schemas exposed to Claude and the Python functions that execute them."""
from __future__ import annotations

from src.tools.filing_search import filing_search
from src.tools.market_data import get_price_history, price_summary
from src.tools.news_search import get_news
from src.tools.technical_indicators import summarize


def _indicators(ticker: str) -> dict:
    return {"ticker": ticker.upper(), **summarize(get_price_history(ticker, period="1y"))}


_TICKER = {"type": "string", "description": "US stock ticker symbol, e.g. AAPL"}

TOOL_SCHEMAS = {
    "get_price_summary": {
        "name": "get_price_summary",
        "description": "Latest close, % change, high/low and average volume over the last N trading days, "
                       "plus 52-week range and data freshness.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": _TICKER,
                           "days": {"type": "integer", "description": "Trading-day window, default 30"}},
            "required": ["ticker"],
        },
    },
    "get_technical_indicators": {
        "name": "get_technical_indicators",
        "description": "Latest SMA20/50, EMA20, RSI14, MACD, Bollinger Bands and ATR14 from daily data, "
                       "with trend/momentum/volatility labels.",
        "input_schema": {"type": "object", "properties": {"ticker": _TICKER}, "required": ["ticker"]},
    },
    "search_sec_filings": {
        "name": "search_sec_filings",
        "description": "Semantic search over the company's latest 10-K and 10-Q (Risk Factors and MD&A sections). "
                       "Returns passages with filing type, date and section for citation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": _TICKER,
                "query": {"type": "string", "description": "What to look for, e.g. 'supply chain risks'"},
                "section": {"type": "string", "enum": ["Risk Factors", "MD&A"],
                            "description": "Optional section filter"},
            },
            "required": ["ticker", "query"],
        },
    },
    "get_recent_news": {
        "name": "get_recent_news",
        "description": "Recent dated news headlines and summaries for the company.",
        "input_schema": {"type": "object", "properties": {"ticker": _TICKER}, "required": ["ticker"]},
    },
}

TOOL_FUNCTIONS = {
    "get_price_summary": lambda ticker, days=30: price_summary(ticker, days),
    "get_technical_indicators": _indicators,
    "search_sec_filings": lambda ticker, query, section=None: filing_search(ticker, query, section),
    "get_recent_news": lambda ticker: {"ticker": ticker.upper(), "articles": get_news(ticker)},
}
