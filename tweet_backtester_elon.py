# -*- coding: utf-8 -*-
"""
Elon Musk tweet event-study backtester.

Same method as the Trump backtester, but the Elon CSV's `Time` column was
mangled by Excel into MM:SS.0 format with no hour info. We instead recover
each tweet's exact UTC timestamp from the Twitter snowflake encoded in
`post_id` (millisecond precision), which is more accurate anyway.

Outputs `backtest_results_elon.csv` and `.xlsx` with the same schema as the
Trump results.
"""

from pathlib import Path
import pandas as pd
import numpy as np

HERE       = Path(__file__).parent
TWEETS_CSV = HERE / "elon_final.csv"
PRICES_CSV = HERE / "ETFs_1min.csv"
OUT_CSV    = HERE / "backtest_results_elon.csv"
OUT_XLSX   = HERE / "backtest_results_elon.xlsx"

WINDOWS = [
    ("10m",   10), ("30m",   30), ("45m",   45), ("60m",   60),
    ("2h",   120), ("4h",   240), ("6h",   360),
    ("12h",  720), ("1d", 1440), ("2d", 2880),
]
LONG_THRESHOLD_MIN = 390   # >= 6.5h => allowed to cross sessions

TWITTER_EPOCH_MS = 1288834974657   # Twitter snowflake epoch (2010-11-04)

# ----- LOAD MARKET DATA -------------------------------------------------------
print("Loading minute prices...")
prices = pd.read_csv(PRICES_CSV)
prices["ts_event"] = pd.to_datetime(prices["ts_event"], utc=True).dt.tz_convert("America/New_York")
prices = prices.set_index("ts_event").sort_index()
prices = prices.groupby(prices.index.date, group_keys=False).apply(lambda g: g.ffill())
print(f"  {len(prices):,} bars, {prices.index.normalize().nunique()} trading days, "
      f"{prices.index.min().date()} -> {prices.index.max().date()}")

# ----- LOAD TWEETS ------------------------------------------------------------
print("Loading Elon tweets...")
tweets = pd.read_csv(TWEETS_CSV)

# Recover tweet_time from Twitter snowflake post_id (ms precision).
# post_id was stored in Excel as float (scientific notation) - we lose the
# bottom ~5 bits of precision but the timestamp lives in the upper 42 bits
# so this is fine for our purposes.
pids = pd.to_numeric(tweets["post_id"], errors="coerce").to_numpy()
pids_int = np.where(np.isnan(pids), 0, pids).astype(np.int64)
ts_ms = (pids_int >> 22) + TWITTER_EPOCH_MS
tweet_dt = pd.to_datetime(ts_ms, unit="ms", utc=True, errors="coerce")
# Mask back to NaT where post_id was missing
tweet_dt = tweet_dt.where(~pd.isna(pids))
tweets["tweet_time"] = tweet_dt.tz_convert("America/New_York")

tweets = tweets.rename(columns={
    "ticker classification": "ticker",
    "recommended_action":    "decision",
})
print(f"  {len(tweets):,} tweets, "
      f"{tweets['tweet_time'].min()} -> {tweets['tweet_time'].max()}")

# ----- HELPERS ----------------------------------------------------------------
def first_bar_at_or_after(t):
    pos = prices.index.searchsorted(t, side="left")
    return int(pos) if pos < len(prices) else None

def last_bar_at_or_before(t):
    pos = prices.index.searchsorted(t, side="right") - 1
    return int(pos) if pos >= 0 else None

def window_return(t0_pos, target_pos, col):
    if t0_pos is None or target_pos is None:
        return np.nan
    p0 = prices[col].iat[t0_pos]
    p1 = prices[col].iat[target_pos]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return p1 / p0 - 1

# ----- COMPUTE ----------------------------------------------------------------
print("Computing returns...")
records = []
n_outside_rth = 0
n_no_data     = 0

for _, row in tweets.iterrows():
    tweet_t   = row["tweet_time"]
    ticker    = row["ticker"]
    decision  = row["decision"]
    rec = {
        "post_id":  row.get("post_id"),
        "tweet_time": tweet_t,
        "ticker":   ticker,
        "decision": decision,
        "content":  row.get("content"),
        "importance_score_1_to_10": row.get("importance_score_1_to_10"),
        "confidence_score_1_to_10": row.get("confidence_score_1_to_10"),
        "category": row.get("category"),
        "relevance_classification": row.get("relevance_classification"),
    }

    def all_nan():
        for label, _ in WINDOWS:
            for d in ("post", "pre"):
                rec[f"ret_{d}_{label}"]       = np.nan
                rec[f"car_raw_{d}_{label}"]   = np.nan
                rec[f"car_trade_{d}_{label}"] = np.nan
        rec["t0_bar"] = pd.NaT

    if pd.isna(tweet_t) or ticker not in prices.columns:
        all_nan(); records.append(rec); n_no_data += 1; continue

    # ceil can raise on the DST fall-back ambiguous hour (e.g. 2025-11-02 01:xx ET).
    # Pass ambiguous='NaT' so we silently skip those edge tweets rather than crash.
    try:
        t0_nominal = tweet_t.ceil("min", ambiguous="NaT", nonexistent="shift_forward")
    except Exception:
        t0_nominal = pd.NaT
    if pd.isna(t0_nominal):
        all_nan(); records.append(rec); n_no_data += 1; continue
    if t0_nominal < prices.index[0]:
        all_nan(); records.append(rec); n_no_data += 1; continue

    t0_pos = first_bar_at_or_after(t0_nominal)
    if t0_pos is None:
        all_nan(); records.append(rec); n_no_data += 1; continue

    t0_ts   = prices.index[t0_pos]
    t0_date = t0_ts.date()
    rec["t0_bar"] = t0_ts
    if t0_ts > t0_nominal:
        n_outside_rth += 1

    for label, mins in WINDOWS:
        is_long = mins >= LONG_THRESHOLD_MIN
        offset  = pd.Timedelta(minutes=mins)

        post_pos = first_bar_at_or_after(tweet_t + offset)
        if (post_pos is None or post_pos <= t0_pos
            or (not is_long and prices.index[post_pos].date() != t0_date)):
            ret_post = np.nan; ret_spy_p = np.nan
        else:
            ret_post  = window_return(t0_pos, post_pos, ticker)
            ret_spy_p = window_return(t0_pos, post_pos, "SPY")

        pre_pos = last_bar_at_or_before(tweet_t - offset)
        if (pre_pos is None or pre_pos >= t0_pos
            or (not is_long and prices.index[pre_pos].date() != t0_date)):
            ret_pre = np.nan; ret_spy_b = np.nan
        else:
            ret_pre   = window_return(pre_pos, t0_pos, ticker)
            ret_spy_b = window_return(pre_pos, t0_pos, "SPY")

        car_raw_p = ret_post - ret_spy_p if pd.notna(ret_post) and pd.notna(ret_spy_p) else np.nan
        car_raw_b = ret_pre  - ret_spy_b if pd.notna(ret_pre)  and pd.notna(ret_spy_b) else np.nan
        car_tr_p  = car_raw_p * decision if pd.notna(car_raw_p) and pd.notna(decision) else np.nan
        car_tr_b  = car_raw_b * decision if pd.notna(car_raw_b) and pd.notna(decision) else np.nan

        rec[f"ret_post_{label}"]       = ret_post
        rec[f"car_raw_post_{label}"]   = car_raw_p
        rec[f"car_trade_post_{label}"] = car_tr_p
        rec[f"ret_pre_{label}"]        = ret_pre
        rec[f"car_raw_pre_{label}"]    = car_raw_b
        rec[f"car_trade_pre_{label}"]  = car_tr_b

    records.append(rec)

out = pd.DataFrame.from_records(records)

desc = ["post_id", "tweet_time", "t0_bar", "ticker", "decision",
        "importance_score_1_to_10", "confidence_score_1_to_10",
        "category", "relevance_classification", "content"]
post_cols, pre_cols = [], []
for label, _ in WINDOWS:
    post_cols += [f"ret_post_{label}",  f"car_raw_post_{label}",  f"car_trade_post_{label}"]
    pre_cols  += [f"ret_pre_{label}",   f"car_raw_pre_{label}",   f"car_trade_pre_{label}"]
out = out[desc + post_cols + pre_cols]

out.to_csv(OUT_CSV, index=False)
out_xlsx = out.copy()
for c in ("tweet_time", "t0_bar"):
    out_xlsx[c] = pd.to_datetime(out_xlsx[c]).dt.tz_localize(None)
out_xlsx.to_excel(OUT_XLSX, index=False)

print()
print(f"Tweets processed:        {len(out):,}")
print(f"  no data (pre-2023 / missing ticker): {n_no_data}")
print(f"  posted outside RTH (t0 snapped forward): {n_outside_rth}")
print(f"\nNon-NaN counts per post-tweet window:")
for label, _ in WINDOWS:
    print(f"  {label:<4}: {out[f'ret_post_{label}'].notna().sum():>4}  "
          f"car_raw mean = {out[f'car_raw_post_{label}'].mean()*100:+.4f}%  "
          f"car_trade mean = {out[f'car_trade_post_{label}'].mean()*100:+.4f}%")
print(f"\nWrote: {OUT_CSV.name} and {OUT_XLSX.name}")
label, _ in WINDOWS:
    print(f"  {label:<4}: {out[f'ret_post_{label}'].notna().sum():>4}  "
          f"car_raw mean = {out[f'car_raw_post_{label}'].mean()*100:+.4f}%  "
          f"car_trade mean = {out[f'car_trade_post_{label}'].mean()*100:+.4f}%")
print(f"\nWrote: {OUT_CSV.name} and {OUT_XLSX.name}")
de_rth}")
print(f"\nNon-NaN counts per post-tweet window:")
for label, _ in WINDOWS:
    print(f"  {label:<4}: {out[f'ret_post_{label}'].notna().sum():>4}  "
          f"car_raw mean = {out[f'car_raw_post_{label}'].mean()*100:+.4f}%  "
          f"car_trade mean = {out[f'car_trade_post_{label}'].mean()*100:+.4f}%")
print(f"\nWrote: {OUT_CSV.name} and {OUT_XLSX.name}")
