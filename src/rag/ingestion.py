"""Download 10-K / 10-Q filings from SEC EDGAR and extract key sections."""
from __future__ import annotations

import json
import re
import time

import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR, SEC_USER_AGENT

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# (section name, start heading regex, end heading regex) per form type.
SECTIONS = {
    "10-K": [
        ("Risk Factors", r"item\s*1a\.?\s*[\-–—:]?\s*risk\s+factors", r"item\s*1b\.?|item\s*1c\.?|item\s*2\.?\s*[\-–—:]?\s*properties"),
        ("MD&A", r"item\s*7\.?\s*[\-–—:]?\s*management.s\s+discussion", r"item\s*7a\.?|item\s*8\.?\s*[\-–—:]?\s*financial"),
    ],
    "10-Q": [
        ("MD&A", r"item\s*2\.?\s*[\-–—:]?\s*management.s\s+discussion", r"item\s*3\.?\s*[\-–—:]?\s*quantitative|item\s*4\.?\s*[\-–—:]?\s*controls"),
        ("Risk Factors", r"item\s*1a\.?\s*[\-–—:]?\s*risk\s+factors", r"item\s*2\.?\s*[\-–—:]?\s*unregistered|item\s*5\.?|item\s*6\.?\s*[\-–—:]?\s*exhibits"),
    ],
}


class SECError(Exception):
    pass


def _get(url: str) -> requests.Response:
    if not SEC_USER_AGENT:
        raise SECError("Set SEC_USER_AGENT in .env (e.g. 'Your Name your@email.com') - SEC requires it")
    time.sleep(0.15)  # stay well under SEC's 10 requests/second limit
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    if resp.status_code != 200:
        raise SECError(f"SEC returned {resp.status_code} for {url}")
    return resp


def ticker_to_cik(ticker: str) -> int:
    cache = RAW_DIR / "company_tickers.json"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        data = _get(TICKER_MAP_URL).json()
        cache.write_text(json.dumps(data))
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise SECError(f"Ticker {ticker} not found in SEC company list")


def latest_filings(ticker: str, forms=("10-K", "10-Q"), per_form: int = 1) -> list[dict]:
    """Metadata for the most recent filings of each requested form type."""
    cik = ticker_to_cik(ticker)
    recent = _get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    found, counts = [], {f: 0 for f in forms}
    for i, form in enumerate(recent["form"]):
        if form in counts and counts[form] < per_form:
            acc = recent["accessionNumber"][i]
            found.append({
                "ticker": ticker.upper(),
                "cik": cik,
                "form": form,
                "accession": acc,
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "url": ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""), doc=recent["primaryDocument"][i]),
            })
            counts[form] += 1
    return found


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="display: none"]'):
        hidden.decompose()  # inline XBRL header blocks
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def extract_sections(text: str, form: str) -> dict[str, str]:
    """Pull named sections out of filing text.

    Headings appear in the table of contents too, so for each section we try
    every start match and keep the longest span - the real section body.
    """
    out = {}
    for name, start_pat, end_pat in SECTIONS.get(form, []):
        best = ""
        for m in re.finditer(start_pat, text, flags=re.I):
            end = re.search(end_pat, text[m.end():], flags=re.I)
            body = text[m.start(): m.end() + end.start()] if end else ""
            if len(body) > len(best):
                best = body
        if len(best) > 500:
            out[name] = best.strip()
    return out


def fetch_filing_sections(meta: dict) -> dict[str, str]:
    """Download one filing (cached on disk) and return its key sections."""
    cache = RAW_DIR / f"{meta['ticker']}_{meta['form']}_{meta['accession']}.txt"
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        text = html_to_text(_get(meta["url"]).text)
        cache.write_text(text, encoding="utf-8")
    return extract_sections(text, meta["form"])
