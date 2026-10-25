"""Technical indicators computed with pandas, plus plain-language signals.

Implemented directly (not via a library) so every formula is visible and
testable; tests compare the results against the `ta` package as a reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


def wilder_smooth(values: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing: seed with the SMA of the first `window` values,
    then avg[i] = (avg[i-1] * (window - 1) + value[i]) / window."""
    vals = values.dropna()
    out = pd.Series(np.nan, index=values.index)
    if len(vals) < window:
        return out
    avg = vals.iloc[:window].mean()
    out[vals.index[window - 1]] = avg
    for idx, v in vals.iloc[window:].items():
        avg = (avg * (window - 1) + v) / window
        out[idx] = avg
    return out


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0), window)
    avg_loss = wilder_smooth(-delta.clip(upper=0), window)
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + num_std * std, "bb_lower": mid - num_std * std})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1, skipna=False)
    tr.iloc[0] = df["High"].iloc[0] - df["Low"].iloc[0]
    return wilder_smooth(tr, window)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with all indicator columns added."""
    out = df.copy()
    close = out["Close"]
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["ema_20"] = ema(close, 20)
    out["rsi_14"] = rsi(close, 14)
    out = out.join(macd(close)).join(bollinger(close))
    out["atr_14"] = atr(out, 14)
    return out


def _r(x) -> float | None:
    return None if pd.isna(x) else round(float(x), 2)


def summarize(df: pd.DataFrame) -> dict:
    """Latest indicator values plus rule-based trend/momentum/volatility labels."""
    ind = add_indicators(df)
    last = ind.iloc[-1]
    close = last["Close"]

    if pd.notna(last["sma_50"]):
        if close > last["sma_20"] > last["sma_50"]:
            trend = "uptrend (price > SMA20 > SMA50)"
        elif close < last["sma_20"] < last["sma_50"]:
            trend = "downtrend (price < SMA20 < SMA50)"
        else:
            trend = "mixed / sideways"
    else:
        trend = "insufficient history for SMA50"

    r = last["rsi_14"]
    if pd.isna(r):
        momentum = "insufficient history"
    elif r >= 70:
        momentum = "overbought (RSI >= 70)"
    elif r <= 30:
        momentum = "oversold (RSI <= 30)"
    else:
        momentum = "neutral RSI"
    if pd.notna(last["macd_hist"]):
        momentum += ", MACD " + ("above" if last["macd_hist"] > 0 else "below") + " signal"

    atr_pct = last["atr_14"] / close * 100 if pd.notna(last["atr_14"]) else None
    if atr_pct is None:
        volatility = "insufficient history"
    else:
        volatility = "high" if atr_pct > 3 else "moderate" if atr_pct > 1.5 else "low"
        volatility += f" (ATR = {atr_pct:.2f}% of price)"

    return {
        "as_of": ind.index[-1].strftime("%Y-%m-%d"),
        "close": _r(close),
        "sma_20": _r(last["sma_20"]),
        "sma_50": _r(last["sma_50"]),
        "ema_20": _r(last["ema_20"]),
        "rsi_14": _r(r),
        "macd": _r(last["macd"]),
        "macd_signal": _r(last["macd_signal"]),
        "macd_hist": _r(last["macd_hist"]),
        "bb_upper": _r(last["bb_upper"]),
        "bb_lower": _r(last["bb_lower"]),
        "atr_14": _r(last["atr_14"]),
        "signals": {"trend": trend, "momentum": momentum, "volatility": volatility},
    }
