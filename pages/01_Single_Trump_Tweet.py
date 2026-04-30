# -*- coding: utf-8 -*-
"""Single Trump Tweet - per-row inspector."""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "backtest_results_v2.csv"

st.set_page_config(page_title="Single Trump Tweet", layout="wide")
st.title("Single Trump Tweet")

WINDOWS = ["10m", "30m", "45m", "60m", "2h", "4h", "6h", "12h", "1d", "2d"]
DECISION_LABEL = {1: "BUY", -1: "SHORT", 0: "IGNORE"}
SECTOR_NAMES = {
    "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
    "SPY": "S&P 500 (broad market)",
}


@st.cache_data(show_spinner="Loading...")
def load(src):
    df = pd.read_csv(src)
    df["tweet_time"] = pd.to_datetime(df["tweet_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    if "t0_bar" in df.columns:
        df["t0_bar"] = pd.to_datetime(df["t0_bar"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    return df


with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Override with a different CSV (optional)", type=["csv"])

if upload is not None:
    df_all = load(upload)
    src_caption = f"Using uploaded file ({upload.name})."
elif DEFAULT_RESULTS.exists():
    df_all = load(str(DEFAULT_RESULTS))
    src_caption = f"Using default: {DEFAULT_RESULTS.name}."
else:
    st.error(f"`{DEFAULT_RESULTS.name}` not found. Run the backtester or upload a CSV in the sidebar.")
    st.stop()

in_rth = (
    df_all["t0_bar"].notna()
    & ((df_all["t0_bar"] - df_all["tweet_time"]).dt.total_seconds().between(0, 60))
)
df = df_all[in_rth].reset_index(drop=False).rename(columns={"index": "orig_row"})

st.caption(f"{src_caption}  |  Rows in file: {len(df_all):,}  |  In-RTH (sampleable): {len(df):,}")

if len(df) == 0:
    st.error("No tweets in this file were posted during market hours.")
    st.stop()

sel_col1, sel_col2 = st.columns([3, 1])
with sel_col1:
    row_idx = st.number_input(
        "Row index (0 = first in-RTH tweet)",
        min_value=0, max_value=len(df) - 1, value=0, step=1,
        help=f"Pick any row from 0 to {len(df) - 1}.",
    )
with sel_col2:
    st.write("")
    if st.button("Random row", use_container_width=True):
        st.session_state["row_idx_override"] = int(np.random.randint(0, len(df)))
        st.rerun()

if "row_idx_override" in st.session_state:
    row_idx = st.session_state.pop("row_idx_override")

row = df.iloc[int(row_idx)]

ticker     = str(row.get("ticker", "")).strip()
sector     = SECTOR_NAMES.get(ticker, "(unknown sector)")
decision   = row.get("decision")
decision_s = DECISION_LABEL.get(int(decision), "-") if pd.notna(decision) else "-"
score      = row.get("importance_score_1_to_10")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tweet time (ET)", str(row["tweet_time"])[:19] if pd.notna(row["tweet_time"]) else "-")
c2.metric("ETF ticker", ticker if ticker else "-", help=sector)
c3.metric("Recommendation", decision_s)
c4.metric("Importance", f"{int(score)}/10" if pd.notna(score) else "-")

st.markdown(f"**Sector:** {sector}")
if "category" in df.columns and pd.notna(row.get("category")):
    st.markdown(f"**LLM category:** {row['category']}")
st.caption(f"Original row in CSV: {int(row.get('orig_row', row_idx))}")

st.markdown("### Tweet")
st.write(row.get("content") or "-")

st.markdown(f"### Post-tweet performance for {ticker}")

table_rows = []
for w in WINDOWS:
    r   = row.get(f"ret_post_{w}")
    cr  = row.get(f"car_raw_post_{w}")
    ct  = row.get(f"car_trade_post_{w}")
    table_rows.append({
        "Horizon": w,
        f"{ticker} return (%)": r * 100 if pd.notna(r) else np.nan,
        "CAR vs SPY (%)":       cr * 100 if pd.notna(cr) else np.nan,
        "CAR x decision (%)":   ct * 100 if pd.notna(ct) else np.nan,
    })
post_table = pd.DataFrame(table_rows)

st.dataframe(
    post_table.style.format({
        f"{ticker} return (%)": "{:+.4f}",
        "CAR vs SPY (%)":       "{:+.4f}",
        "CAR x decision (%)":   "{:+.4f}",
    }, na_rep="-"),
    hide_index=True, use_container_width=True,
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=post_table["Horizon"], y=post_table[f"{ticker} return (%)"], name=f"{ticker} return", mode="lines+markers"))
fig.add_trace(go.Scatter(x=post_table["Horizon"], y=post_table["CAR vs SPY (%)"],        name="CAR (vs SPY)",      mode="lines+markers"))
fig.add_trace(go.Scatter(x=post_table["Horizon"], y=post_table["CAR x decision (%)"],    name="CAR x decision",    mode="lines+markers"))
fig.add_hline(y=0, line_dash="dot", line_color="gray")
fig.update_layout(
    xaxis_title="Horizon (post-tweet)",
    yaxis_title="Return (%)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=380, margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw row data"):
    st.dataframe(row.to_frame(name="value"), use_container_width=True)
