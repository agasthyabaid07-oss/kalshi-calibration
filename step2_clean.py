"""
STEP 2: Turn the raw downloads into one clean table for analysis.

HOW TO RUN (in IDLE): open step2_clean.py and press F5.
Needs: data/markets_raw.csv, data/snapshots_raw.csv, data/series.csv (from step 1)
Makes: data/analysis.csv

One row in analysis.csv = one market at one snapshot time, with:
    prob     -> the market's implied probability (the mid price)
    outcome  -> 1 if the market resolved YES, 0 if NO
    plus category, spread, volume, hours before close, and close date.

WHY WE USE THE MID PRICE (not the last trade):
    The last trade can be hours old. The mid price, (bid + ask) / 2, is what
    the market thought at that exact moment. If the bid or ask is missing,
    or the spread is very wide, the "price" is unreliable, so we drop it.
"""

import os
import pandas as pd

# ---------------------------------------------------------------- SETTINGS
MAX_SPREAD = 0.10   # drop snapshots where ask - bid is more than 10 cents
MIN_VOLUME = 10     # drop markets where fewer than 10 contracts ever traded
# -------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

markets = pd.read_csv(os.path.join(DATA_DIR, "markets_raw.csv"))
snaps = pd.read_csv(os.path.join(DATA_DIR, "snapshots_raw.csv"))
series_path = os.path.join(DATA_DIR, "series.csv")
try:
    series = pd.read_csv(series_path)
except (FileNotFoundError, pd.errors.EmptyDataError):
    series = pd.DataFrame(columns=["series_ticker", "category"])

print("Raw markets:", len(markets), "| raw snapshots:", len(snaps))

# Drop the placeholder rows for markets that had no price history
no_history = snaps[snaps["hours_before_close"].isna()]["ticker"].nunique()
print("Markets with no usable price history:", no_history)
snaps = snaps.dropna(subset=["hours_before_close"])
snaps = snaps.drop_duplicates(subset=["ticker", "hours_before_close"])

# Join everything together
df = snaps.merge(markets, on="ticker", how="inner", suffixes=("", "_mkt"))
df = df.merge(series[["series_ticker", "category"]], on="series_ticker", how="left")
df["category"] = df["category"].fillna("Unknown")

# Keep a log of how many rows each filter removes (put this in your write-up!)
log = [("Snapshots joined to markets", len(df))]

df = df[df["result"].isin(["yes", "no"])]
df["outcome"] = (df["result"] == "yes").astype(int)
log.append(("Clear YES/NO result", len(df)))

df = df.dropna(subset=["yes_bid", "yes_ask"])
df = df[(df["yes_bid"] > 0) & (df["yes_ask"] < 1) & (df["yes_ask"] > df["yes_bid"])]
log.append(("Valid bid and ask", len(df)))

df["spread"] = df["yes_ask"] - df["yes_bid"]
df = df[df["spread"] <= MAX_SPREAD]
log.append(("Spread <= {:.0f} cents".format(MAX_SPREAD * 100), len(df)))

df = df[df["volume"] >= MIN_VOLUME]
log.append(("Volume >= {}".format(MIN_VOLUME), len(df)))

df["prob"] = (df["yes_bid"] + df["yes_ask"]) / 2
df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")
df["hours_before_close"] = df["hours_before_close"].astype(int)

# Liquidity groups: split markets into three equal-sized groups by volume
df["liquidity"] = pd.qcut(df["volume"].rank(method="first"), 3,
                          labels=["Low volume", "Mid volume", "High volume"])

cols = ["ticker", "event_ticker", "series_ticker", "category", "yes_sub_title",
        "close_time", "hours_before_close", "yes_bid", "yes_ask", "spread",
        "prob", "last_trade", "volume", "liquidity", "outcome"]
df = df[cols].sort_values(["close_time", "ticker", "hours_before_close"])
df.to_csv(os.path.join(DATA_DIR, "analysis.csv"), index=False)

print("\nFilter log (rows remaining after each step):")
for name, n in log:
    print("   {:<32} {:>8}".format(name, n))
print("\nUnique markets:", df["ticker"].nunique(),
      "| unique events:", df["event_ticker"].nunique())
print("Rows per snapshot time:")
print(df.groupby("hours_before_close").size().to_string())
print("\nSTEP 2 DONE. Saved data/analysis.csv. Next: run step3_calibration.py")
