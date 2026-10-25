"""Agent-facing wrapper around filing retrieval."""
from __future__ import annotations

from src.rag.retriever import index_ticker, indexed_tickers, search_filings


def filing_search(ticker: str, query: str, section: str | None = None, k: int = 5) -> dict:
    ticker = ticker.upper()
    if ticker not in indexed_tickers():
        index_ticker(ticker, log=lambda *_: None)  # index on first use
    hits = search_filings(query, ticker=ticker, section=section, k=k)
    if not hits:
        return {"ticker": ticker, "passages": [], "note": "No matching filing passages found."}
    return {"ticker": ticker, "passages": hits}
