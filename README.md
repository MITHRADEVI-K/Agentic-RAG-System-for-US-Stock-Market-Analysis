# Agentic RAG System for US Stock Market Analysis

An AI agent that analyses US stocks by combining **market data**, **technical indicators**, **SEC filing retrieval (RAG)** and **news**, then writes a cited, numerically validated report.

Built with **LangGraph** (orchestration), **Claude** (Anthropic API), **ChromaDB** (vector search), **SEC EDGAR**, **yfinance**, and **Streamlit**.

> Educational project, not investment advice. Price data is delayed.

## Architecture

```
            ┌──────────┐
 question → │  route   │  deterministic intent + ticker extraction → allowed tool subset
            └────┬─────┘
                 ▼
            ┌──────────┐  tool calls   ┌──────────┐
            │  agent   │ ────────────► │  tools   │  price · indicators · SEC RAG · news
            │ (Claude) │ ◄──────────── │          │
            └────┬─────┘   results     └──────────┘
                 ▼ final answer
            ┌──────────┐
            │ validate │  every decimal in the answer must match a tool output
            └──────────┘
```

- **RAG retrieves knowledge** (10-K/10-Q Risk Factors and MD&A, chunked with ticker, form, filing date, period and section metadata).
- **Tools retrieve numbers**: the LLM never computes or recalls prices or indicators itself.
- **Router** limits which tools are available, so a pure RSI question never triggers filing search.
- **Validator** flags any number in the answer that didn't come from a tool.

## Project structure

```
app/dashboard.py              Streamlit UI (charts + agent tab)
scripts/ingest.py             Pre-index SEC filings
src/agent/graph.py            LangGraph workflow (route → agent ⇄ tools → validate)
src/agent/router.py           Intent / ticker routing
src/agent/tool_registry.py    Tool schemas for Claude + their Python implementations
src/agent/prompts.py          System prompt
src/tools/                    market_data, technical_indicators, news_search, filing_search
src/rag/                      ingestion (EDGAR), chunking, embeddings (Chroma), retriever
src/validation/               numeric answer validation
tests/                        pytest suite (no API key needed)
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then add ANTHROPIC_API_KEY and SEC_USER_AGENT
```

## Run

```powershell
pytest                                   # unit tests
python scripts/ingest.py AAPL MSFT NVDA  # index filings (one-time, ~1-2 min)
python -m src.agent.graph "What is Apple's RSI?"
streamlit run app/dashboard.py
```

## Example questions

| Question | Tools routed |
|---|---|
| What is Apple's RSI? | price + indicators |
| What risks did Microsoft report? | SEC filing RAG |
| Why did Nvidia's stock move recently? | price + news |
| Analyze AAPL: technical trends, news and risks | all four |

## Limitations

- yfinance data is delayed and unofficial; swap `src/tools/market_data.py` for Alpha Vantage / Finnhub / Polygon for production use.
- Section extraction from filings is regex-based and can miss unusual layouts.
- Only the latest 10-K and 10-Q are indexed per ticker.
- The numeric validator checks decimals only; it can't verify claims in prose.

## Roadmap

Intraday data, multi-agent analysts, news sentiment, fundamental ratios and peer comparison, backtesting, an automated eval set, caching and monitoring.
