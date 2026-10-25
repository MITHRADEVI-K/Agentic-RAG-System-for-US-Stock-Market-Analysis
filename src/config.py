"""Central configuration loaded from environment variables (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
for _d in (RAW_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")

# SEC requires a descriptive User-Agent with contact email on every request.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
COLLECTION_NAME = "sec_filings"
