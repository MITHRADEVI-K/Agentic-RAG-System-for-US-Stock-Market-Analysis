"""Deterministic intent routing: decides which tools the agent may use.

Running this before the LLM keeps routing cheap, predictable and testable -
e.g. a pure RSI question never gets the filing-search tool.
"""
from __future__ import annotations

import re

COMPANY_NAMES = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
}
STOPWORDS = {"I", "A", "THE", "AND", "OR", "RSI", "MACD", "SMA", "EMA", "ATR", "US", "USA",
             "SEC", "MD", "CEO", "AI", "ETF", "IPO", "EPS", "PE", "Q", "K", "WHAT", "WHY", "HOW", "IS"}

TECHNICAL = r"\b(rsi|macd|sma|ema|moving average|bollinger|atr|indicator|technical|trend|momentum|overbought|oversold|volatil|support|resistance)"
PRICE = r"\b(price|close|volume|move|moved|movement|perform|return|gain|drop|fell|rose|rall|52.?week|chart|last \d+ days)"
FILINGS = r"\b(risk|10-?k|10-?q|filing|annual report|md&a|management.s discussion|revenue|segment|guidance|competition|business|fundamental|supply chain|regulat)"
NEWS = r"\b(news|headline|event|announce|why did|recent(ly)?|latest|happen)"

ALL_TOOLS = ["get_price_summary", "get_technical_indicators", "search_sec_filings", "get_recent_news"]


def extract_tickers(question: str) -> list[str]:
    found = []
    for name, tick in COMPANY_NAMES.items():
        if re.search(rf"\b{name}\b", question, re.I):
            found.append(tick)
    for tok in re.findall(r"\$?\b[A-Z]{1,5}\b", question):
        tok = tok.lstrip("$")
        if tok not in STOPWORDS:
            found.append(tok)
    return list(dict.fromkeys(found))


def route(question: str) -> dict:
    """Return intent labels and the subset of tools the agent may call."""
    q = question.lower()
    intents = []
    if re.search(TECHNICAL, q):
        intents.append("technical")
    if re.search(PRICE, q):
        intents.append("price")
    if re.search(FILINGS, q):
        intents.append("filings")
    if re.search(NEWS, q):
        intents.append("news")
    if not intents or re.search(r"\b(analy[sz]e|overview|full|complete|compare)\b", q):
        intents = ["technical", "price", "filings", "news"]

    tools = []
    if "price" in intents or "technical" in intents:
        tools.append("get_price_summary")
    if "technical" in intents:
        tools.append("get_technical_indicators")
    if "filings" in intents:
        tools.append("search_sec_filings")
    if "news" in intents:
        tools.append("get_recent_news")
    return {"intents": intents, "tickers": extract_tickers(question), "tools": tools}
