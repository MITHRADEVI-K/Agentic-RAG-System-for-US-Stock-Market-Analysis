import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV (random walk), 200 business days."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1.5, 200))
    high = close + rng.uniform(0.1, 2, 200)
    low = close - rng.uniform(0.1, 2, 200)
    open_ = low + (high - low) * rng.uniform(0, 1, 200)
    idx = pd.bdate_range("2025-01-01", periods=200)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1e6, 5e6, 200)}, index=idx)
