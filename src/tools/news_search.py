"""Recent company news headlines (via yfinance, no API key needed)."""
from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf


def _parse_item(item: dict) -> dict | None:
    # yfinance has returned two shapes over time: flat, and nested under "content".
    c = item.get("content", item)
    title = c.get("title")
    if not title:
        return None
    published = c.get("pubDate") or c.get("displayTime")
    if not published and item.get("providerPublishTime"):
        published = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc).isoformat()
    url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or ""
    publisher = (c.get("provider") or {}).get("displayName") or c.get("publisher") or ""
    return {
        "title": title,
        "publisher": publisher,
        "published": (published or "")[:10],
        "url": url,
        "summary": (c.get("summary") or "")[:400],
        "related_tickers": item.get("relatedTickers", []),
    }


def get_news(ticker: str, limit: int = 8) -> list[dict]:
    """Return up to `limit` recent dated headlines that mention `ticker`."""
    ticker = ticker.upper()
    try:
        raw = yf.Ticker(ticker).news or []
        if not raw:  # Ticker.news is sometimes empty; Search is more reliable
            raw = yf.Search(ticker, news_count=20).news or []
    except Exception as exc:
        return [{"error": f"News provider failed: {exc}"}]
    raw = [i for i in raw if not i.get("relatedTickers") or ticker in i["relatedTickers"]]
    items = [p for p in (_parse_item(i) for i in raw) if p]
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:limit]
