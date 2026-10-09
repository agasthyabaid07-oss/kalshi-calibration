"""
STEP 1: Download settled Kalshi markets and their price history.

HOW TO RUN (in IDLE):
    File -> Open -> step1_collect.py, then press F5.

WHAT IT DOES:
    1. Downloads the list of Kalshi "series" (to know each market's category).
    2. Downloads settled markets (both recent and archived ones).
    3. For each market, downloads hourly prices before it closed and records
       the price at several "snapshot" times (e.g. 24 hours before close).
    4. Saves everything into the "data" folder next to this script.

It is SAFE to stop and re-run: markets already downloaded are skipped.
No account, no API key and no money are needed. It only reads public data.

First run tip: keep MAX_MARKETS small (e.g. 300) to check everything works,
then raise it (3000-10000) for the real study. Bigger = slower.
"""

import os
import json
import time
import requests
import pandas as pd

# ---------------------------------------------------------------- SETTINGS
MAX_MARKETS = 300            # how many settled markets to collect
SNAPSHOT_HOURS = [1, 24, 168]  # take a price 1h, 24h and 7 days before close
MIN_HOURS_OPEN = 30          # skip markets open for less than this many hours
PER_SERIES_CAP = 25          # at most this many markets from any one series
EXCLUDE_CATEGORIES = ["Crypto"]  # categories to leave out
RANDOM_SEED = 7              # change to get a different random set of series
SLEEP_SECONDS = 0.15        # pause between requests (be polite to the API)
BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
# -------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MARKETS_FILE = os.path.join(DATA_DIR, "markets_raw.csv")
SNAPSHOTS_FILE = os.path.join(DATA_DIR, "snapshots_raw.csv")
SERIES_FILE = os.path.join(DATA_DIR, "series.csv")

session = requests.Session()


def api_get(path, params=None, retries=5):
    """Call the Kalshi API and return the JSON. Retries if rate-limited."""
    url = BASE_URL + path
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print("   network problem:", e, "- retrying")
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            time.sleep(SLEEP_SECONDS)
            return r.json()
        if r.status_code == 429:          # too many requests: wait and retry
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        print("   API error", r.status_code, r.text[:200])
        time.sleep(2 ** attempt)
    return None


def to_dollars(value):
    """Kalshi sends prices as strings like '0.5600'. Turn them into floats.
    Older responses used whole cents (56), so convert those too."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v / 100 if v > 1.0 else v


def field(d, name):
    """Read a price field whether it is called 'close' or 'close_dollars'."""
    if d is None:
        return None
    if name in d:
        return d[name]
    return d.get(name + "_dollars")


def parse_time(s):
    return pd.Timestamp(s).tz_convert("UTC") if s else None


# ------------------------------------------------------------ 1. SERIES
def download_series():
    print("Downloading series list (for categories)...")
    data = api_get("/series") or {}
    rows = [{"series_ticker": s.get("ticker"),
             "series_title": s.get("title"),
             "category": s.get("category")}
            for s in data.get("series", [])]
    df = pd.DataFrame(rows)
    df.to_csv(SERIES_FILE, index=False)
    print("   got", len(df), "series")
    return df


# ------------------------------------------------------------ 2. MARKETS
def keep_market(m):
    """Only simple yes/no markets that have finished with a clear result,
    and that were open long enough to have a price 24 hours before closing.
    (Many Kalshi markets only last a few hours - those are skipped.)"""
    if (m.get("market_type", "binary") != "binary"
            or m.get("result") not in ("yes", "no")
            or m.get("mve_collection_ticker")):
        return False
    try:
        hours_open = (parse_time(m["close_time"]) - parse_time(m["open_time"])).total_seconds() / 3600
    except Exception:
        return False
    return hours_open >= MIN_HOURS_OPEN


def download_markets(series):
    """Go through the series one by one (in a random but repeatable order)
    and collect settled markets from each. Taking at most PER_SERIES_CAP
    markets per series gives a good mix of topics instead of thousands of
    near-identical contracts from one series."""
    print("Downloading settled markets, series by series...")
    s = series.dropna(subset=["series_ticker"])
    s = s[~s["category"].isin(EXCLUDE_CATEGORIES)]
    s = s.sample(frac=1, random_state=RANDOM_SEED)   # shuffle
    found = {}
    tried = 0
    for _, row in s.iterrows():
        if len(found) >= MAX_MARKETS:
            break
        tried += 1
        st = row["series_ticker"]
        taken = 0
        for path, params in [("/markets", {"series_ticker": st, "status": "settled", "limit": 1000}),
                             ("/historical/markets", {"series_ticker": st, "limit": 1000})]:
            data = api_get(path, params) or {}
            for m in data.get("markets", []):
                if taken >= PER_SERIES_CAP or len(found) >= MAX_MARKETS:
                    break
                if keep_market(m) and m["ticker"] not in found:
                    m["_series_ticker"] = st
                    found[m["ticker"]] = m
                    taken += 1
        if tried % 20 == 0:
            print("   checked {} series, {} usable markets so far".format(tried, len(found)))
    print("   checked {} series in total".format(tried))

    rows = []
    for m in list(found.values())[:MAX_MARKETS]:
        rows.append({
            "ticker": m["ticker"],
            "series_ticker": m["_series_ticker"],
            "event_ticker": m.get("event_ticker"),
            "yes_sub_title": m.get("yes_sub_title"),
            "result": m.get("result"),
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
            "volume": float(m.get("volume_fp") or m.get("volume") or 0),
            "open_interest": float(m.get("open_interest_fp") or m.get("open_interest") or 0),
            "last_price": to_dollars(field(m, "last_price")),
        })
    df = pd.DataFrame(rows)
    df.to_csv(MARKETS_FILE, index=False)
    print("   saved", len(df), "markets to", MARKETS_FILE)
    return df


# ------------------------------------------------- 3. SERIES FOR EACH MARKET
def attach_series(markets, series):
    """Each event ticker looks like 'KXCPI-26SEP'. The part before the first
    dash is usually the series. If that guess is wrong, ask the API."""
    known = set(series["series_ticker"].dropna()) if len(series) else set()
    cache = {}
    out = []
    for ev in markets["event_ticker"]:
        guess = str(ev).split("-")[0]
        if guess in known:
            out.append(guess)
            continue
        if ev not in cache:
            data = api_get("/events/" + str(ev)) or {}
            cache[ev] = (data.get("event") or {}).get("series_ticker", guess)
        out.append(cache[ev])
    markets["series_ticker"] = out
    return markets


# --------------------------------------------------------- 4. CANDLESTICKS
def get_candles(series_ticker, ticker, start_ts, end_ts):
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60}
    data = api_get("/series/{}/markets/{}/candlesticks".format(series_ticker, ticker), params)
    if not data or not data.get("candlesticks"):
        data = api_get("/historical/markets/{}/candlesticks".format(ticker), params)
    return (data or {}).get("candlesticks", [])


def snapshot_rows(market, candles):
    """For each snapshot time, take the last hourly candle ending at or
    before that time and record the bid, ask and last trade price."""
    close_ts = int(parse_time(market["close_time"]).timestamp())
    candles = sorted(candles, key=lambda c: c["end_period_ts"])
    rows = []
    for h in SNAPSHOT_HOURS:
        target = close_ts - h * 3600
        usable = [c for c in candles if c["end_period_ts"] <= target]
        if not usable:
            continue
        c = usable[-1]
        if target - c["end_period_ts"] > 6 * 3600:   # too stale, skip
            continue
        price = c.get("price") or {}
        rows.append({
            "ticker": market["ticker"],
            "hours_before_close": h,
            "candle_ts": c["end_period_ts"],
            "yes_bid": to_dollars(field(c.get("yes_bid"), "close")),
            "yes_ask": to_dollars(field(c.get("yes_ask"), "close")),
            "last_trade": to_dollars(field(price, "close") or field(price, "previous")),
            "open_interest": float(c.get("open_interest_fp") or c.get("open_interest") or 0),
        })
    return rows


def download_snapshots(markets):
    done = set()
    if os.path.exists(SNAPSHOTS_FILE):
        old = pd.read_csv(SNAPSHOTS_FILE)
        done = set(old["ticker"])
        print("Resuming:", len(done), "markets already have prices")
    todo = markets[~markets["ticker"].isin(done)]
    print("Downloading price history for", len(todo), "markets...")

    max_h = max(SNAPSHOT_HOURS)
    buffer = []
    for i, (_, m) in enumerate(todo.iterrows(), start=1):
        close_ts = int(parse_time(m["close_time"]).timestamp())
        candles = get_candles(m["series_ticker"], m["ticker"],
                              close_ts - (max_h + 7) * 3600, close_ts)
        rows = snapshot_rows(m, candles)
        if not rows:   # remember we tried, so a re-run skips it
            rows = [{"ticker": m["ticker"], "hours_before_close": None}]
        buffer.extend(rows)
        if i % 25 == 0 or i == len(todo):
            pd.DataFrame(buffer).to_csv(SNAPSHOTS_FILE, mode="a", index=False,
                                        header=not os.path.exists(SNAPSHOTS_FILE))
            buffer = []
            print("   {}/{} markets done".format(i, len(todo)))


# ------------------------------------------------------------------ MAIN
if __name__ == "__main__":
    start = time.time()
    series = download_series()
    markets = download_markets(series)
    if len(markets) == 0:
        print("No markets downloaded. Check your internet connection.")
    else:
        download_snapshots(markets)
        print("\nSTEP 1 DONE in {:.1f} minutes. Next: run step2_clean.py"
              .format((time.time() - start) / 60))
