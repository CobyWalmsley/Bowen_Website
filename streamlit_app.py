# -*- coding: utf-8 -*-
"""Trump Backtest - Home page."""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Trump Backtest", layout="wide")
st.title("Trump Backtest")

HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "backtest_results_v2.csv"

WINDOWS = [("10m", 10), ("30m", 30), ("45m", 45), ("60m", 60),
           ("2h", 120), ("4h", 240), ("6h", 360), ("12h", 720),
           ("1d", 1440), ("2d", 2880)]
WINDOW_LABELS = [w[0] for w in WINDOWS]
SCORE_COL = "importance_score_1_to_10"

SECTOR_NAMES = {
    "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
    "SPY": "S&P 500 (broad market)",
}


@st.cache_data(show_spinner="Loading results...")
def load_results(src):
    df = pd.read_csv(src)
    df["tweet_time"] = pd.to_datetime(df["tweet_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    if "t0_bar" in df.columns:
        df["t0_bar"] = pd.to_datetime(df["t0_bar"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    return df


with st.sidebar:
    st.header("Data source")
    upload = st.file_uploader("Backtest results CSV", type=["csv"], key="results_upload")
    if upload is not None:
        df = load_results(upload)
        st.caption(f"Using uploaded file ({upload.name})")
    elif DEFAULT_RESULTS.exists():
        df = load_results(DEFAULT_RESULTS)
        st.caption(f"Using default: {DEFAULT_RESULTS.name}")
    else:
        st.error(f"No file uploaded and {DEFAULT_RESULTS.name} not found.")
        st.stop()

if "t0_bar" not in df.columns:
    df["t0_bar"] = pd.NaT

with st.sidebar:
    st.header("Filters")
    if SCORE_COL in df.columns:
        avail_scores = sorted(int(s) for s in df[SCORE_COL].dropna().unique() if not pd.isna(s))
    else:
        avail_scores = []
    selected_scores = st.multiselect("Importance score (out of 10)", avail_scores, default=avail_scores)
    portfolio_window = st.selectbox("Portfolio holding window", WINDOW_LABELS, index=WINDOW_LABELS.index("1d"))
    drop_zero = st.checkbox("Exclude decision == 0 (ignore) from portfolio", value=True)

mask = pd.Series(True, index=df.index)
if selected_scores and SCORE_COL in df.columns:
    mask &= df[SCORE_COL].isin(selected_scores)
df_f = df[mask].copy()
df_valid = df_f.dropna(subset=["t0_bar"])

st.caption(f"Total: {len(df):,}  |  After filters: {len(df_f):,}  |  With market data: {len(df_valid):,}")

# 1) AVERAGE RETURN + CAR
st.subheader("Average return and CAR across horizons")
ret_means = [df_valid[f"ret_post_{w}"].mean() * 100 for w in WINDOW_LABELS]
car_raw_means = [df_valid[f"car_raw_post_{w}"].mean() * 100 for w in WINDOW_LABELS]
car_trade_means = [df_valid[f"car_trade_post_{w}"].mean() * 100 for w in WINDOW_LABELS]
sample_n = [df_valid[f"ret_post_{w}"].notna().sum() for w in WINDOW_LABELS]

fig_avg = go.Figure()
fig_avg.add_trace(go.Scatter(x=WINDOW_LABELS, y=ret_means, name="Raw return (ETF only)", mode="lines+markers"))
fig_avg.add_trace(go.Scatter(x=WINDOW_LABELS, y=car_raw_means, name="CAR (ETF - SPY)", mode="lines+markers"))
fig_avg.add_trace(go.Scatter(x=WINDOW_LABELS, y=car_trade_means, name="CAR x decision", mode="lines+markers"))
fig_avg.update_layout(xaxis_title="Horizon (post-tweet)", yaxis_title="Mean return (pct)",
                     hovermode="x unified", height=380, margin=dict(l=10, r=10, t=10, b=10),
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_avg, use_container_width=True)

with st.expander("Sample size per horizon"):
    st.dataframe(pd.DataFrame({"Horizon": WINDOW_LABELS, "N (non-NaN)": sample_n}),
                 hide_index=True, use_container_width=True)

# 2) PORTFOLIO EQUITY CURVE
st.subheader(f"Portfolio equity curve - each trade held for {portfolio_window}")
ret_col = f"ret_post_{portfolio_window}"
car_trade_col = f"car_trade_post_{portfolio_window}"

port = df_f.dropna(subset=[ret_col, "tweet_time", "decision"]).sort_values("tweet_time").copy()
if drop_zero:
    port = port[port["decision"] != 0]

if len(port) == 0:
    st.warning("No tweets meet the filters with a valid return for this window.")
else:
    port["pnl_raw"] = port[ret_col] * port["decision"]
    port["pnl_car"] = port[car_trade_col]
    port["equity_raw"] = (1 + port["pnl_raw"].fillna(0)).cumprod()
    port["equity_car"] = (1 + port["pnl_car"].fillna(0)).cumprod()

    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(x=port["tweet_time"], y=port["equity_raw"], name="Raw PnL (decision x return)", mode="lines"))
    fig_port.add_trace(go.Scatter(x=port["tweet_time"], y=port["equity_car"], name="CAR PnL (decision x CAR)", mode="lines"))
    fig_port.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig_port.update_layout(xaxis_title="Tweet time", yaxis_title="Equity (start = 1.0)",
                          hovermode="x unified", height=380, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_port, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Trades executed", f"{len(port):,}")
    c2.metric("Final raw PnL", f"{(port['equity_raw'].iloc[-1] - 1)*100:+.2f}%")
    c3.metric("Final CAR PnL", f"{(port['equity_car'].iloc[-1] - 1)*100:+.2f}%")

# 3) RETURN DISTRIBUTION
st.subheader("Return distribution")
dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
with dcol1:
    dist_window = st.selectbox("Horizon", WINDOW_LABELS, index=WINDOW_LABELS.index("1d"), key="dist_window")
with dcol2:
    dist_norm = st.selectbox("Normalize", ["Counts", "Probability density"], index=1, key="dist_norm")
with dcol3:
    nbins = st.slider("Bins", min_value=10, max_value=80, value=30, step=5)

histnorm = None if dist_norm == "Counts" else "probability density"
ret_col_d = f"ret_post_{dist_window}"
car_col_d = f"car_raw_post_{dist_window}"

dist_df = df_valid.dropna(subset=[ret_col_d, car_col_d]).copy()
if SCORE_COL in dist_df.columns:
    dist_df["score"] = dist_df[SCORE_COL].astype("Int64").astype(str)
    score_order = [str(s) for s in sorted(dist_df[SCORE_COL].dropna().astype(int).unique())]
else:
    dist_df["score"] = "all"
    score_order = ["all"]

dist_df["ret_pct"] = dist_df[ret_col_d] * 100
dist_df["car_pct"] = dist_df[car_col_d] * 100

if len(dist_df) == 0:
    st.warning("No data at this horizon.")
else:
    h1, h2 = st.columns(2)
    with h1:
        fig_r = px.histogram(dist_df, x="ret_pct", color="score", category_orders={"score": score_order},
                            nbins=nbins, histnorm=histnorm, barmode="overlay", opacity=0.55,
                            labels={"ret_pct": "Raw return (pct)", "score": "Importance"})
        fig_r.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_r.update_layout(title=f"Raw return @ {dist_window}", height=400,
                           margin=dict(l=10, r=10, t=50, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_r, use_container_width=True)
    with h2:
        fig_c = px.histogram(dist_df, x="car_pct", color="score", category_orders={"score": score_order},
                            nbins=nbins, histnorm=histnorm, barmode="overlay", opacity=0.55,
                            labels={"car_pct": "CAR (pct)", "score": "Importance"})
        fig_c.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_c.update_layout(title=f"CAR @ {dist_window}", height=400,
                           margin=dict(l=10, r=10, t=50, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_c, use_container_width=True)

    def stats(col, lbl):
        g = dist_df.groupby("score")[col].agg(
            N="count",
            mean_pct=lambda s: s.mean() * 100,
            median_pct=lambda s: s.median() * 100,
            std_pct=lambda s: s.std() * 100,
            pct_pos=lambda s: (s > 0).mean() * 100,
        ).round(3)
        g.columns = [f"{lbl} {c}" for c in ["N", "mean", "median", "std", "% pos"]]
        return g

    summary = pd.concat([stats(ret_col_d, "Raw"), stats(car_col_d, "CAR")], axis=1).reset_index()
    summary = summary.rename(columns={"score": "Importance"})
    with st.expander("Per-score summary stats"):
        st.dataframe(summary, hide_index=True, use_container_width=True)

# 4) IMPORTANCE SCORE FREQUENCY
st.subheader("Importance score distribution")
if SCORE_COL in df.columns:
    sc = df[SCORE_COL].dropna().astype(int).value_counts().sort_index()
    fig_sc = px.bar(x=sc.index.astype(str), y=sc.values,
                   labels={"x": "Importance score", "y": "Tweet count"}, text=sc.values)
    fig_sc.update_traces(textposition="outside")
    fig_sc.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig_sc, use_container_width=True)

# 5) ETF UNIVERSE
st.subheader("ETF universe")
etf_counts = df["ticker"].dropna().value_counts().rename_axis("Ticker").reset_index(name="Recommendations")
left, right = st.columns([1, 2], gap="large")
with left:
    st.dataframe(etf_counts, hide_index=True, use_container_width=True, height=360)
with right:
    fig_etf = px.bar(etf_counts.sort_values("Recommendations", ascending=False),
                    x="Ticker", y="Recommendations", text="Recommendations")
    fig_etf.update_traces(textposition="outside")
    fig_etf.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_etf, use_container_width=True)

# 6) PER-ETF AVERAGE RETURN + CAR
st.subheader("Per-ETF average return and CAR")
st.caption("Filtered to a single ETF. Sidebar score filter still applies.")

etf_options = sorted(df_valid["ticker"].dropna().unique())
if not etf_options:
    st.info("No ETFs in the filtered set.")
else:
    selected_etf = st.selectbox("ETF", etf_options,
                                format_func=lambda t: f"{t} - {SECTOR_NAMES.get(t, '?')}",
                                key="per_etf_selector")
    etf_df = df_valid[df_valid["ticker"] == selected_etf]
    sector_label = SECTOR_NAMES.get(selected_etf, "?")

    rm = [etf_df[f"ret_post_{w}"].mean() * 100 for w in WINDOW_LABELS]
    cm = [etf_df[f"car_raw_post_{w}"].mean() * 100 for w in WINDOW_LABELS]
    tm = [etf_df[f"car_trade_post_{w}"].mean() * 100 for w in WINDOW_LABELS]
    nm = [etf_df[f"ret_post_{w}"].notna().sum() for w in WINDOW_LABELS]

    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(x=WINDOW_LABELS, y=rm, name="Raw return", mode="lines+markers"))
    fig_e.add_trace(go.Scatter(x=WINDOW_LABELS, y=cm, name="CAR (vs SPY)", mode="lines+markers"))
    fig_e.add_trace(go.Scatter(x=WINDOW_LABELS, y=tm, name="CAR x decision", mode="lines+markers"))
    fig_e.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_e.update_layout(title=f"{selected_etf} - {sector_label} ({len(etf_df)} tweets)",
                       xaxis_title="Horizon (post-tweet)", yaxis_title="Mean return (pct)",
                       hovermode="x unified", height=380, margin=dict(l=10, r=10, t=50, b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_e, use_container_width=True)

    with st.expander("Sample size per horizon"):
        st.dataframe(pd.DataFrame({"Horizon": WINDOW_LABELS, "N (non-NaN)": nm}),
                     hide_index=True, use_container_width=True)
