"""Fetch and validate daily OHLCV data.

Uses yfinance (free, no API key). Prices are delayed, not real-time, and
yfinance is an unofficial Yahoo wrapper - the freshness info returned with
every result makes that explicit to the user.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

PROVIDER = "Yahoo Finance via yfinance (delayed)"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class MarketDataError(Exception):
    """Raised when market data cannot be retrieved or fails validation."""


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise and validate an OHLCV frame: dedupe, sort, drop bad rows."""
    if df is None or df.empty:
        raise MarketDataError("No price data returned")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MarketDataError(f"Missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
    if df.empty:
        raise MarketDataError("All rows failed validation")
    return df


def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Return cleaned daily OHLCV for `ticker` over a yfinance period string."""
    ticker = ticker.upper().strip()
    try:
        raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    except Exception as exc:  # network / provider failure
        raise MarketDataError(f"Provider error for {ticker}: {exc}") from exc
    if raw is None or raw.empty:
        raise MarketDataError(f"No data for '{ticker}' - is the ticker valid?")
    return clean_ohlcv(raw)


def freshness(df: pd.DataFrame) -> dict:
    """Describe how current the data is so answers can disclose it."""
    last = df.index[-1].to_pydatetime()
    age_days = (datetime.now() - last).days
    return {
        "provider": PROVIDER,
        "last_bar_date": last.strftime("%Y-%m-%d"),
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "is_stale": age_days > 4,  # allows for weekends + a holiday
    }


def price_summary(ticker: str, days: int = 30) -> dict:
    """Structured price snapshot used by the agent's market-data tool."""
    df = get_price_history(ticker, period="1y")
    window = df.tail(days)
    first, last = window["Close"].iloc[0], window["Close"].iloc[-1]
    return {
        "ticker": ticker.upper(),
        "last_close": round(float(last), 2),
        "change_pct_window": round(float((last / first - 1) * 100), 2),
        "window_days": len(window),
        "window_high": round(float(window["High"].max()), 2),
        "window_low": round(float(window["Low"].min()), 2),
        "avg_volume_window": int(window["Volume"].mean()),
        "52w_high": round(float(df["High"].max()), 2),
        "52w_low": round(float(df["Low"].min()), 2),
        "freshness": freshness(df),
    }
