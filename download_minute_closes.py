"""
Databento — 1-minute closing prices for 22 US stocks/ETFs
Date range: 2023-03-28 through 2025-12-10 (regular trading hours only)

Output: closes_1min.csv  (rows = ET timestamp, columns = symbols, values = close)

Usage
-----
    pip install databento pandas
    export DATABENTO_API_KEY="db-XXXXXXXX..."     # or paste below
    python download_minute_closes.py

The script (1) prices the request via metadata.get_cost so you can see the
charge against your $125 credit BEFORE downloading, (2) waits for you to
confirm, then (3) streams the data and writes a tidy wide-format CSV.
"""

import os
import sys
import pandas as pd
import databento as db

# ───────────────────────── CONFIG ──────────────────────────────────────────────
API_KEY = os.getenv("DATABENTO_API_KEY", "db-PASTE-YOUR-KEY-HERE")

SYMBOLS = [
    # 12 stocks
    "OXY", "LHX", "BG", "EIX", "JNJ", "MU",
    "TPL", "LLY", "CME", "WBD", "KEY", "GOOGL",
    # 10 sector SPDR ETFs
    "XLI", "XLU", "XLE", "XLV", "XLF",
    "XLP", "XLB", "XLC", "XLY", "XLK",
]

# DBEQ.BASIC = Databento's consolidated US-equities feed.
# Covers stocks AND ETFs from any listing exchange (NYSE, Nasdaq, Arca, etc.),
# and is the cheapest dataset that supports ohlcv-1m for this symbol mix.
DATASET = "DBEQ.BASIC"
SCHEMA  = "ohlcv-1m"

START = "2023-03-28"
END   = "2025-12-11"          # END is EXCLUSIVE — use day after your last day
OUT   = "closes_1min.csv"

# ───────────────────────── 1. CLIENT ──────────────────────────────────────────
client = db.Historical(API_KEY)

# ───────────────────────── 2. PRICE THE REQUEST ───────────────────────────────
print(f"Pricing {len(SYMBOLS)} symbols × ohlcv-1m × "
      f"{START} → {END} on {DATASET}...")

cost = client.metadata.get_cost(
    dataset=DATASET,
    symbols=SYMBOLS,
    schema=SCHEMA,
    start=START,
    end=END,
    stype_in="raw_symbol",
)
print(f"Estimated cost: ${cost:.4f}   (you have $125 in free credits)")

if input("Proceed with download? [y/N] ").strip().lower() != "y":
    sys.exit("Aborted by user.")

# ───────────────────────── 3. DOWNLOAD ────────────────────────────────────────
print("Downloading...")
store = client.timeseries.get_range(
    dataset=DATASET,
    symbols=SYMBOLS,
    schema=SCHEMA,
    start=START,
    end=END,
    stype_in="raw_symbol",
)

df = store.to_df()            # ts_event-indexed (UTC), one row per symbol-minute
print(f"Got {len(df):,} bars across {df['symbol'].nunique()} symbols.")

# ───────────────────────── 4. RTH FILTER + PIVOT ──────────────────────────────
# Bars are labeled by interval START. Regular session = 09:30..15:59 ET
# (that's 390 bars/day per symbol; the 15:59 bar covers the closing minute).
df.index = df.index.tz_convert("America/New_York")
df = df.between_time("09:30", "15:59")

wide = (
    df.reset_index()
      .pivot_table(index="ts_event", columns="symbol", values="close")
      .sort_index()
)

# Reorder columns to match SYMBOLS order, drop any that returned no data
ordered = [s for s in SYMBOLS if s in wide.columns]
missing = [s for s in SYMBOLS if s not in wide.columns]
wide = wide[ordered]

wide.to_csv(OUT)
print(f"Wrote {len(wide):,} rows × {len(wide.columns)} symbols → {OUT}")
if missing:
    print(f"WARNING: no data returned for: {missing}")
