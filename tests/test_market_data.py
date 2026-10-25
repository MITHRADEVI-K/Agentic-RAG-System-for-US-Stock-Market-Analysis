from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.tools.market_data import MarketDataError, clean_ohlcv, freshness


def test_removes_duplicates_and_sorts(ohlcv):
    messy = pd.concat([ohlcv.iloc[5:], ohlcv.iloc[:5], ohlcv.iloc[[10]]])
    cleaned = clean_ohlcv(messy)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()
    assert len(cleaned) == len(ohlcv)


def test_drops_invalid_rows(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[3, bad.columns.get_loc("Close")] = None
    bad.iloc[4, bad.columns.get_loc("Close")] = -5
    assert len(clean_ohlcv(bad)) == len(ohlcv) - 2


def test_empty_raises():
    with pytest.raises(MarketDataError):
        clean_ohlcv(pd.DataFrame())


def test_missing_column_raises(ohlcv):
    with pytest.raises(MarketDataError):
        clean_ohlcv(ohlcv.drop(columns=["Volume"]))


def test_stale_detection(ohlcv):
    old = ohlcv.copy()
    assert freshness(old)["is_stale"]  # synthetic data is from 2025
    recent = ohlcv.tail(1).copy()
    recent.index = [pd.Timestamp(datetime.now() - timedelta(days=1)).normalize()]
    assert not freshness(recent)["is_stale"]
