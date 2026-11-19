from __future__ import annotations
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from market_analysis import prepare_market_data, analyze_market_data
from agent import MarketResearchAgent
from reporting import make_markdown_report, make_json_report

load_dotenv()
st.set_page_config(page_title="MarketMind AI", page_icon="📊", layout="wide")
st.markdown(
    """<style>
    :root { color-scheme: light; }
    .stApp { background: #f5f7fb; color: #1f2937; }
    section[data-testid="stSidebar"] {
        background-color: #eef2f7;
        color: #1f2937;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1f2937;
    }
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stMetric"],
    .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li {
        color: #1f2937;
    }
    .stApp input, .stApp textarea, .stApp select {
        color: #1f2937;
        background-color: #ffffff;
    }
    .block-container { max-width: 1320px; padding-top: 2rem; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf2;
        border-radius: 12px;
        padding: 14px;
    }
    .stApp [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        line-height: 1.2;
        white-space: nowrap;
    }
    .stApp [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    .stApp [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }
    </style>""",
    unsafe_allow_html=True,
)
st.title("📊 MarketMind AI")
st.caption("An AI-assisted educational market research agent powered by Markov-chain analysis and Monte Carlo resampling.")
with st.sidebar:
    st.header("Research settings")
    uploaded=st.file_uploader("Upload historical CSV",type=["csv"],help="Required columns: Date and Close")
    horizon=st.slider("Scenario horizon (trading days)",10,252,30,10)
    paths=st.slider("Simulation paths",100,2000,500,100)
    threshold_pct=st.slider("State threshold (%)",0.1,5.0,0.5,0.1,help="Daily returns above this value are Up; returns below its negative are Down.")
    st.caption(f"State thresholds: Up > +{threshold_pct:.1f}%, Down < −{threshold_pct:.1f}%, otherwise Stable.")
if uploaded is None:
    st.info("Upload a CSV containing Date and Close columns to start the agent.")
    st.markdown("Example: `Date,Close` followed by dated closing-price rows. Extra columns are ignored.")
    st.stop()
try:
    raw=pd.read_csv(BytesIO(uploaded.getvalue()))
    data=prepare_market_data(raw,threshold_pct=threshold_pct)
except Exception as e:
    st.error(f"Could not prepare this file: {e}"); st.stop()
with st.spinner("Agent is running the analysis workflow…"):
    try: result=MarketResearchAgent().run(data,horizon,paths,threshold_pct=threshold_pct)
    except Exception as e: st.error(f"Analysis failed: {e}"); st.stop()
s=result['summary']; m=s['simulation']
st.caption(f"Source: {uploaded.name} · {len(data):,} valid price observations · {data.Date.min():%d %b %Y} to {data.Date.max():%d %b %Y}")
cols=st.columns(4)
cols[0].metric("Latest close",f"{s['latest_close']:,.2f}")
cols[1].metric("Historical total return",f"{s['total_return_pct']:+.2f}%")
cols[2].metric("Annualized volatility",f"{s['annualized_volatility_pct']:.2f}%" if s['annualized_volatility_pct'] is not None else "N/A")
cols[3].metric("Maximum drawdown",f"{s['max_drawdown_pct']:.2f}%")
st.caption("Changing the threshold recalculates return states, transition probabilities, and scenarios. Price-based metrics above remain tied to the uploaded prices.")
state_counts=s["state_counts"]
state_percentages=result["analysis"]["metrics"]["state_percentages"]
state_cols=st.columns(3)
for col,state in zip(state_cols,("Up","Stable","Down")):
    col.metric(f"{state} days",f"{state_counts[state]:,}",f"{state_percentages[state]:.1f}% of returns")
tabs=st.tabs(["Agent research","Historical data","Method & limitations"])
with tabs[0]:
    st.subheader("Agent workflow")
    for i,step in enumerate(result['agent_steps'],1): st.write(f"**{i}.** {step}")
    st.subheader("Research explanation")
    st.write(result['commentary'])
    st.subheader(f"{m['horizon_trading_days']}-day illustrative scenario distribution")
    st.warning("Simulation output is not a price forecast, trading signal, or investment recommendation.")
    st.line_chart(result['scenarios'])
    a,b,c=st.columns(3); a.metric("10th percentile",f"{m['10th_percentile_end']:,.2f}"); b.metric("Median",f"{m['median_end']:,.2f}"); c.metric("90th percentile",f"{m['90th_percentile_end']:,.2f}")
    st.subheader("Markov transition probabilities")
    st.dataframe(result['analysis']['transition_probabilities'].style.format('{:.1%}'),use_container_width=True)
    report=make_markdown_report(result,uploaded.name)
    st.download_button("Download research report (Markdown)",report,file_name="marketmind_research_report.md",mime="text/markdown")
    st.download_button("Download computed summary (JSON)",make_json_report(result),file_name="marketmind_summary.json",mime="application/json")
with tabs[1]:
    st.subheader("Historical closing price"); st.line_chart(data.set_index('Date')['Close'])
    st.subheader("Daily returns (%)"); st.line_chart(data.set_index('Date')['Return'])
    st.dataframe(data.tail(200),use_container_width=True)
    st.download_button("Download cleaned data",data.to_csv(index=False).encode(),"marketmind_cleaned_data.csv","text/csv")
with tabs[2]:
    st.markdown("""### How the agent works
1. **Data validation:** parses dates and positive closing prices, removes invalid rows and duplicate dates.
2. **Historical metrics:** computes total return, CAGR, annualized volatility and maximum drawdown.
3. **Markov chain:** classifies each daily return using the adjustable symmetric threshold and estimates next-state probabilities.
4. **Monte Carlo:** resamples observed returns conditional on simulated states to produce percentile paths.
5. **Research explanation:** summarizes computed outputs. If `GEMINI_API_KEY` is configured, Gemini can phrase the explanation; all calculations remain local Python.

### Important limitations
This is a learning and research project, not financial advice. Historical results and simulated scenarios do not predict future performance. The model omits many real-world factors and can fail when market conditions change. Do not use it as the sole basis for financial decisions.
""")
