# -*- coding: utf-8 -*-
"""
Trump-tweet event-study backtester.

Inputs
------
TWEETS_CSV : per-tweet data (Date, Time, content, ticker classification,
             recommended_action, ...)
PRICES_CSV : 1-minute closes for sector SPDRs + SPY, indexed by ts_event in ET.

Output
------
A wide DataFrame (saved to .csv and .xlsx) with one row per tweet and these
columns per window W in {10m, 30m, 45m, 60m, 2h, 4h, 6h, 12h, 1d, 2d} for
both directions (post-tweet and pre-tweet):

    ret_<dir>_<W>          raw cumulative return of the tweet's ticker
    car_raw_<dir>_<W>      ticker return − SPY return     (raw abnormal return)
    car_trade_<dir>_<W>    car_raw × decision             (signed by trade side)

Methodology
-----------
* t0  = first market minute strictly after the tweet timestamp
        (rounds away the tweet's seconds → no look-ahead). If the tweet falls
        outside RTH, t0 snaps to the next session's 09:30 bar.
* target = first market minute ≥ tweet_time + W   (post-tweet)
           last market minute ≤ tweet_time − W    (pre-tweet)
* Intraday windows (W < 6.5h: 10m..6h):  if target lands on a different
  trading day than t0, set NaN  (e.g. tweet at 3:55 PM + 30 min crosses
  the close → NaN).
* Day-scale windows (W ≥ 6.5h: 12h, 1d, 2d):  cross-session is allowed.
* Returns computed as close[target] / close[t0] − 1, on price columns that
  have been forward-filled WITHIN each trading day (so stale prints aren't
  carried across overnight gaps).
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────── CONFIG ──────────────────────────────────────────────────
HERE       = Path(__file__).parent
TWEETS_CSV = HERE / "batch-main.csv"
PRICES_CSV = HERE / "ETFs_1min.csv"
OUT_CSV    = HERE / "backtest_results_v2.csv"
OUT_XLSX   = HERE / "backtest_results_v2.xlsx"

# (label, minutes)  — ordered shortest → longest
WINDOWS = [
    ("10m",   10),
    ("30m",   30),
    ("45m",   45),
    ("60m",   60),
    ("2h",   120),
    ("4h",   240),
    ("6h",   360),
    ("12h",  720),
    ("1d",  1440),
    ("2d",  2880),
]
LONG_THRESHOLD_MIN = 390   # >= 6.5h ⇒ allowed to cross sessions

# ─────────────────── LOAD MARKET DATA ────────────────────────────────────────
print("Loading minute prices...")
prices = pd.read_csv(PRICES_CSV)
prices["ts_event"] = pd.to_datetime(prices["ts_event"], utc=True).dt.tz_convert("America/New_York")
prices = prices.set_index("ts_event").sort_index()

# Forward-fill within each trading day (don't carry stale prices overnight).
prices = prices.groupby(prices.index.date, group_keys=False).apply(lambda g: g.ffill())

idx_arr  = prices.index.values            # numpy datetime64[ns, ET] for searchsorted
print(f"  {len(prices):,} bars, {prices.index.normalize().nunique()} trading days, "
      f"{prices.index.min().date()} → {prices.index.max().date()}")

# ─────────────────── LOAD TWEETS ─────────────────────────────────────────────
print("Loading tweets...")
tweets = pd.read_csv(TWEETS_CSV)
tweets["tweet_time"] = pd.to_datetime(
    tweets["Date"] + " " + tweets["Time"], errors="coerce"
).dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
tweets = tweets.rename(columns={
    "ticker classification": "ticker",
    "recommended_action":    "decision",
})
print(f"  {len(tweets):,} tweets, "
      f"{tweets['tweet_time'].min()} → {tweets['tweet_time'].max()}")

# ─────────────────── HELPERS ─────────────────────────────────────────────────
def first_bar_at_or_after(t):
    """Index location of first bar with timestamp >= t. None if past data end."""
    pos = prices.index.searchsorted(t, side="left")
    return int(pos) if pos < len(prices) else None

def last_bar_at_or_before(t):
    """Index location of last bar with timestamp <= t. None if before data start."""
    pos = prices.index.searchsorted(t, side="right") - 1
    return int(pos) if pos >= 0 else None

def window_return(t0_pos, target_pos, col):
    """close[target] / close[t0] - 1, with NaN guards."""
    if t0_pos is None or target_pos is None:
        return np.nan
    p0 = prices[col].iat[t0_pos]
    p1 = prices[col].iat[target_pos]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return p1 / p0 - 1

# ─────────────────── PER-TWEET COMPUTATION ───────────────────────────────────
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

    # Skip if essentials missing
    if pd.isna(tweet_t) or ticker not in prices.columns:
        for label, _ in WINDOWS:
            for d in ("post", "pre"):
                rec[f"ret_{d}_{label}"]       = np.nan
                rec[f"car_raw_{d}_{label}"]   = np.nan
                rec[f"car_trade_{d}_{label}"] = np.nan
        rec["t0_bar"] = pd.NaT
        records.append(rec)
        n_no_data += 1
        continue

    # Round the tweet up to the next minute (kills the seconds → no lookahead)
    t0_nominal = tweet_t.ceil("min")

    # Drop tweets that fall before the market-data window: snapping to the
    # first available bar would invent a t0 unrelated to the tweet.
    if t0_nominal < prices.index[0]:
        for label, _ in WINDOWS:
            for d in ("post", "pre"):
                rec[f"ret_{d}_{label}"]       = np.nan
                rec[f"car_raw_{d}_{label}"]   = np.nan
                rec[f"car_trade_{d}_{label}"] = np.nan
        rec["t0_bar"] = pd.NaT
        records.append(rec)
        n_no_data += 1
        continue

    t0_pos = first_bar_at_or_after(t0_nominal)
    if t0_pos is None:
        # past end of price data
        for label, _ in WINDOWS:
            for d in ("post", "pre"):
                rec[f"ret_{d}_{label}"]       = np.nan
                rec[f"car_raw_{d}_{label}"]   = np.nan
                rec[f"car_trade_{d}_{label}"] = np.nan
        rec["t0_bar"] = pd.NaT
        records.append(rec)
        n_no_data += 1
        continue

    t0_ts   = prices.index[t0_pos]
    t0_date = t0_ts.date()
    rec["t0_bar"] = t0_ts

    # Track whether tweet was outside RTH (t0 had to snap forward in time)
    if t0_ts > t0_nominal:
        n_outside_rth += 1

    # ----- compute every window in both directions ---------------------------
    for label, mins in WINDOWS:
        is_long = mins >= LONG_THRESHOLD_MIN
        offset  = pd.Timedelta(minutes=mins)

        # ---- POST: target = first bar >= tweet_t + W ------------------------
        post_pos = first_bar_at_or_after(tweet_t + offset)
        # Reject if: no bar / different day on intraday window /
        # collapses onto t0 (e.g. tweet 8pm + 12h both snap to next open)
        if (post_pos is None
            or post_pos <= t0_pos
            or (not is_long and prices.index[post_pos].date() != t0_date)):
            ret_post  = np.nan
            ret_spy_p = np.nan
        else:
            ret_post  = window_return(t0_pos, post_pos, ticker)
            ret_spy_p = window_return(t0_pos, post_pos, "SPY")

        # ---- PRE: target = last bar <= tweet_t - W --------------------------
        pre_pos = last_bar_at_or_before(tweet_t - offset)
        if (pre_pos is None
            or pre_pos >= t0_pos
            or (not is_long and prices.index[pre_pos].date() != t0_date)):
            ret_pre   = np.nan
            ret_spy_b = np.nan
        else:
            # return FROM (W ago) TO t0
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

# Tidy column order: descriptive cols, then post-windows grouped, then pre-windows
desc = ["post_id", "tweet_time", "t0_bar", "ticker", "decision",
        "importance_score_1_to_10", "confidence_score_1_to_10",
        "category", "relevance_classification", "content"]
post_cols, pre_cols = [], []
for label, _ in WINDOWS:
    post_cols += [f"ret_post_{label}",  f"car_raw_post_{label}",  f"car_trade_post_{label}"]
    pre_cols  += [f"ret_pre_{label}",   f"car_raw_pre_{label}",   f"car_trade_pre_{label}"]
out = out[desc + post_cols + pre_cols]

# ─────────────────── SAVE ────────────────────────────────────────────────────
out.to_csv(OUT_CSV, index=False)
# xlsx — convert tz-aware to tz-naive for Excel compatibility
out_xlsx = out.copy()
for c in ("tweet_time", "t0_bar"):
    out_xlsx[c] = pd.to_datetime(out_xlsx[c]).dt.tz_localize(None)
out_xlsx.to_excel(OUT_XLSX, index=False)

# ─────────────────── REPORT ──────────────────────────────────────────────────
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
se)

# ----- REPORT -----------------------------------------------------------------
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
          f"car_trade mean = {out[f'car_trade_post_{label}'].mean()*100:+.4f}%")

print(f"\nWrote: {OUT_CSV.name} and {OUT_XLSX.name}")
ide_rth}")
print(f"\nNon-NaN counts per post-tweet window:")
for label, _ in WINDOWS:
    print(f"  {label:<4}: {out[f'ret_post_{label}'].notna().sum():>4}  "
          f"car_raw mean = {out[f'car_raw_post_{label}'].mean()*100:+.4f}%  "
          f"car_trade mean = {out[f'car_trade_post_{label}'].mean()*100:+.4f}%")
print(f"\nWrote: {OUT_CSV.name} and {OUT_XLSX.name}")
ame}")
