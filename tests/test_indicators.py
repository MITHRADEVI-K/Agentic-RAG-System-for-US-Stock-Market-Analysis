"""Our indicator implementations vs the `ta` library as a reference."""
import numpy as np
import pandas as pd
import pytest
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from src.tools import technical_indicators as ti


def _close_enough(a: pd.Series, b: pd.Series, start: int = 60):
    a, b = a.iloc[start:], b.iloc[start:]  # skip warm-up where seeding differs
    np.testing.assert_allclose(a.values, b.values, rtol=1e-3, atol=1e-3)


def test_sma(ohlcv):
    _close_enough(ti.sma(ohlcv["Close"], 20), SMAIndicator(ohlcv["Close"], 20).sma_indicator())


def test_ema(ohlcv):
    _close_enough(ti.ema(ohlcv["Close"], 20), EMAIndicator(ohlcv["Close"], 20).ema_indicator())


def test_rsi(ohlcv):
    # `ta` seeds RSI with an EMA from the first bar instead of Wilder's SMA seed;
    # the two converge once the seed has decayed.
    _close_enough(ti.rsi(ohlcv["Close"], 14), RSIIndicator(ohlcv["Close"], 14).rsi(), start=120)


def test_rsi_textbook_seed():
    # First RSI value = 100 - 100 / (1 + avg gain / avg loss) over the first 14 changes.
    close = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                       45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28])
    d = close.diff().dropna()
    expected = 100 - 100 / (1 + d.clip(lower=0).mean() / (-d.clip(upper=0)).mean())
    assert ti.rsi(close).iloc[-1] == pytest.approx(expected)


def test_rsi_bounds(ohlcv):
    r = ti.rsi(ohlcv["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1, 40, dtype=float))
    assert ti.rsi(close).iloc[-1] == pytest.approx(100)


def test_macd(ohlcv):
    ref = MACD(ohlcv["Close"])
    ours = ti.macd(ohlcv["Close"])
    _close_enough(ours["macd"], ref.macd())
    _close_enough(ours["macd_signal"], ref.macd_signal())


def test_bollinger(ohlcv):
    ref = BollingerBands(ohlcv["Close"], 20, 2)
    ours = ti.bollinger(ohlcv["Close"])
    _close_enough(ours["bb_upper"], ref.bollinger_hband())
    _close_enough(ours["bb_lower"], ref.bollinger_lband())


def test_atr(ohlcv):
    ref = AverageTrueRange(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], 14).average_true_range()
    _close_enough(ti.atr(ohlcv, 14), ref)


def test_summarize_structure(ohlcv):
    s = ti.summarize(ohlcv)
    for key in ("close", "rsi_14", "sma_20", "sma_50", "macd", "atr_14", "signals"):
        assert key in s
    assert set(s["signals"]) == {"trend", "momentum", "volatility"}


def test_summarize_short_history(ohlcv):
    s = ti.summarize(ohlcv.head(25))
    assert s["sma_50"] is None
    assert "insufficient" in s["signals"]["trend"]
