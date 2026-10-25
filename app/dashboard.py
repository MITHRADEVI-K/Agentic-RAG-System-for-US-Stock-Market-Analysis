"""Streamlit dashboard. Run with:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.agent.graph import run_analysis
from src.config import DEFAULT_TICKERS
from src.tools.market_data import MarketDataError, freshness, get_price_history
from src.tools.news_search import get_news
from src.tools.technical_indicators import add_indicators, summarize

st.set_page_config(page_title="Agentic Stock Analysis", layout="wide")
st.title("Agentic RAG Stock Analysis")
st.caption("Educational analysis, not investment advice. Prices are delayed.")

with st.sidebar:
    ticker = st.text_input("Ticker", value=DEFAULT_TICKERS[0]).upper().strip()
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y"], index=2)


@st.cache_data(ttl=900)
def load(ticker: str, period: str):
    return get_price_history(ticker, period)


tab_chart, tab_agent = st.tabs(["Chart & indicators", "Ask the agent"])

with tab_chart:
    try:
        df = load(ticker, period)
    except MarketDataError as exc:
        st.error(str(exc))
        st.stop()

    fr = freshness(df)
    summary = summarize(df)
    if fr["is_stale"]:
        st.warning(f"Data may be stale - last bar is {fr['last_bar_date']}.")
    st.caption(f"Provider: {fr['provider']} - last bar {fr['last_bar_date']} - retrieved {fr['retrieved_at_utc']}")

    c = st.columns(5)
    c[0].metric("Close", summary["close"])
    c[1].metric("RSI 14", summary["rsi_14"])
    c[2].metric("SMA 20", summary["sma_20"])
    c[3].metric("SMA 50", summary["sma_50"])
    c[4].metric("ATR 14", summary["atr_14"])
    st.write(" | ".join(f"**{k.title()}:** {v}" for k, v in summary["signals"].items()))

    ind = add_indicators(df)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=ind.index, open=ind["Open"], high=ind["High"], low=ind["Low"],
                                 close=ind["Close"], name="Price"), row=1, col=1)
    for col_name in ("sma_20", "sma_50", "bb_upper", "bb_lower"):
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col_name], name=col_name, line={"width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi_14"], name="RSI 14"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["macd_hist"], name="MACD hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["macd"], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["macd_signal"], name="Signal"), row=3, col=1)
    fig.update_layout(height=750, xaxis_rangeslider_visible=False, margin={"t": 20})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent news")
    for n in get_news(ticker, limit=6):
        if "error" in n:
            st.warning(n["error"])
        else:
            st.markdown(f"- **{n['published']}** [{n['title']}]({n['url']}) - {n['publisher']}")

with tab_agent:
    question = st.text_area(
        "Question",
        value=f"Analyze {ticker} for the last 30 days. What are its technical trends, important news, and major risks?",
    )
    if st.button("Run analysis", type="primary"):
        with st.spinner("Agent is routing, calling tools and writing the analysis..."):
            result = run_analysis(question)
        if result.get("error"):
            st.error(result["error"])
        else:
            st.markdown(result["answer"])
            v = result.get("validation", {})
            if v.get("passed"):
                st.success(f"Numeric check passed ({v['numbers_checked']} numbers matched tool outputs).")
            else:
                st.warning(f"Numbers not found in tool outputs: {v.get('unverified')}")
        with st.expander("Routing and tool calls"):
            st.json(result.get("routing", {}))
            for t in result.get("tool_log", []):
                st.markdown(f"**{t['name']}** `{t['input']}`")
                st.json(t["output"], expanded=False)
