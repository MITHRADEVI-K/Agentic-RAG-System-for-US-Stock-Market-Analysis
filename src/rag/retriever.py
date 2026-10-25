"""Index filings into ChromaDB and retrieve passages with source metadata."""
from __future__ import annotations

from src.rag.chunking import chunk_text
from src.rag.embeddings import get_collection
from src.rag.ingestion import fetch_filing_sections, latest_filings


def index_ticker(ticker: str, forms=("10-K", "10-Q"), log=print) -> int:
    """Download, chunk and embed the latest filings for a ticker. Idempotent."""
    col = get_collection()
    total = 0
    for meta in latest_filings(ticker, forms):
        sections = fetch_filing_sections(meta)
        if not sections:
            log(f"  {meta['form']} {meta['filing_date']}: no sections extracted")
            continue
        for section, text in sections.items():
            chunks = chunk_text(text)
            ids = [f"{meta['accession']}|{section}|{i}" for i in range(len(chunks))]
            metadatas = [{
                "ticker": meta["ticker"],
                "form": meta["form"],
                "filing_date": meta["filing_date"],
                "report_date": meta["report_date"],
                "section": section,
                "url": meta["url"],
                "chunk": i,
            } for i in range(len(chunks))]
            col.upsert(ids=ids, documents=chunks, metadatas=metadatas)
            total += len(chunks)
            log(f"  {meta['form']} {meta['filing_date']} / {section}: {len(chunks)} chunks")
    return total


def search_filings(query: str, ticker: str | None = None, section: str | None = None,
                   form: str | None = None, k: int = 5) -> list[dict]:
    """Semantic search over indexed filings, filtered by metadata."""
    col = get_collection()
    filters = [{key: val} for key, val in
               (("ticker", ticker and ticker.upper()), ("section", section), ("form", form)) if val]
    where = filters[0] if len(filters) == 1 else {"$and": filters} if filters else None
    res = col.query(query_texts=[query], n_results=k, where=where)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "source": f"{meta['ticker']} {meta['form']} filed {meta['filing_date']} "
                      f"(period {meta['report_date']}), {meta['section']}",
            "url": meta["url"],
            "relevance": round(1 - dist, 3),
            **{k2: meta[k2] for k2 in ("ticker", "form", "filing_date", "report_date", "section")},
        })
    return hits


def indexed_tickers() -> list[str]:
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    return sorted({m["ticker"] for m in metas})
