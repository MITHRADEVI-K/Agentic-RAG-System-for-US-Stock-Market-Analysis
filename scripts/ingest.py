"""Pre-index SEC filings so the first agent query isn't slow.

Usage:  python scripts/ingest.py AAPL MSFT NVDA
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_TICKERS
from src.rag.retriever import index_ticker

if __name__ == "__main__":
    for t in sys.argv[1:] or DEFAULT_TICKERS:
        print(f"Indexing {t} ...")
        print(f"{t}: {index_ticker(t)} chunks indexed")
