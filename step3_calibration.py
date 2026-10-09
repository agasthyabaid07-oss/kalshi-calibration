"""
STEP 3: The core study. Are Kalshi prices accurate probabilities?

HOW TO RUN (in IDLE): open step3_calibration.py and press F5.
Needs: data/analysis.csv (from step 2)
Makes: charts in the "figures" folder and numbers in results/summary.txt

WHAT IT COMPUTES:
  1. Calibration table: group contracts by price (0-10c, 10-20c, ...) and
     compare the average price with how often they actually resolved YES.
  2. Confidence intervals, so you know which gaps are real and which are noise.
  3. Brier score (+ its 3-part breakdown) and log loss: accuracy scores.
  4. Favorite-longshot test: are cheap contracts overpriced?
  5. Recalibration slope: one number summarising the bias (1.0 = perfect).
  6. The same checks split by category, liquidity and time before close.

IMPORTANT STATISTICS NOTE (mention this in interviews!):
  Markets inside the same event are linked. If "CPI above 3.0%" is YES, then
  "CPI above 2.5%" is YES too. Treating them as independent would make our
  confidence intervals too narrow. So we use a "cluster bootstrap": we
  resample whole EVENTS, not single markets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- SETTINGS
MAIN_HORIZON = 24        # main analysis uses the price 24h before close
N_BINS = 10              # 10 price buckets of 10 cents each
N_BOOTSTRAP = 1000       # more = more precise intervals, but slower
MIN_ROWS_PER_GROUP = 50  # skip categories with too few markets
SEED = 42
# -------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)
report = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    report.append(line)


# ------------------------------------------------------- statistics tools
def wilson_interval(yes, n, z=1.96):
    """95% confidence interval for a YES frequency (works well for small n
    and for frequencies near 0 or 1, unlike the textbook formula)."""
    if n == 0:
        return (np.nan, np.nan)
    p = yes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)


def brier(prob, outcome):
    return np.mean((prob - outcome) ** 2)


def log_loss(prob, outcome):
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return -np.mean(outcome * np.log(p) + (1 - outcome) * np.log(1 - p))


def murphy_decomposition(df, n_bins=N_BINS):
    """Brier = reliability - resolution + uncertainty.
    reliability: calibration error (lower = better)
    resolution:  how informative the prices are (higher = better)
    uncertainty: how unpredictable the events were (fixed by the data)"""
    base = df["outcome"].mean()
    bins = pd.cut(df["prob"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby(bins, observed=True).agg(n=("outcome", "size"),
                                            p=("prob", "mean"),
                                            o=("outcome", "mean"))
    N = len(df)
    rel = (g["n"] * (g["p"] - g["o"]) ** 2).sum() / N
    res = (g["n"] * (g["o"] - base) ** 2).sum() / N
    unc = base * (1 - base)
    return rel, res, unc


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def recalibration_fit(prob, outcome):
    """Fit: P(YES) = sigmoid(a + b * logit(price)) with Newton's method.
    b = 1 and a = 0 means perfectly calibrated.
    b > 1 means reality is MORE extreme than prices: longshots are
          overpriced and favorites underpriced (favorite-longshot bias)."""
    X = np.column_stack([np.ones(len(prob)), logit(prob)])
    y = outcome.astype(float)
    w = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (y - mu)
        hess = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w  # [a, b]


def cluster_bootstrap(df, stat_fn, n=N_BOOTSTRAP):
    """Resample whole events with replacement, recompute the statistic each
    time, and return the 2.5% and 97.5% percentiles = a 95% interval."""
    groups = {k: v.index.to_numpy() for k, v in df.groupby("event_ticker")}
    keys = np.array(list(groups.keys()))
    stats = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in pick])
        sample = df.loc[idx]
        try:
            stats.append(stat_fn(sample))
        except Exception:
            continue
    return np.percentile(stats, [2.5, 97.5])


def calibration_table(df, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    df = df.assign(bin=pd.cut(df["prob"], edges, include_lowest=True))
    rows = []
    for b, g in df.groupby("bin", observed=True):
        n, yes = len(g), g["outcome"].sum()
        lo, hi = wilson_interval(yes, n)
        rows.append({"price_bucket": "{:.0f}-{:.0f}c".format(max(b.left, 0) * 100, b.right * 100),
                     "n": n,
                     "avg_price": g["prob"].mean(),
                     "yes_rate": yes / n,
                     "ci_low": lo, "ci_high": hi,
                     "gap": yes / n - g["prob"].mean()})
    return pd.DataFrame(rows)


def plot_calibration(ax, table, label=None, color=None):
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")
    yerr = [(table["yes_rate"] - table["ci_low"]).clip(lower=0),
            (table["ci_high"] - table["yes_rate"]).clip(lower=0)]
    ax.errorbar(table["avg_price"], table["yes_rate"], yerr=yerr, fmt="o-",
                capsize=3, label=label, color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Market price (implied probability)")
    ax.set_ylabel("Fraction that resolved YES")
    ax.grid(alpha=0.3)


# ================================================================ ANALYSIS
all_df = pd.read_csv(os.path.join(DATA_DIR, "analysis.csv"))
if len(all_df) == 0:
    raise SystemExit("analysis.csv is empty. Re-run step 1 (with more markets), then step 2.")
if (all_df["hours_before_close"] == MAIN_HORIZON).sum() == 0:
    best = all_df["hours_before_close"].value_counts().idxmax()
    print("No rows at {}h before close - using {}h instead.".format(MAIN_HORIZON, best))
    MAIN_HORIZON = int(best)
df = all_df[all_df["hours_before_close"] == MAIN_HORIZON].reset_index(drop=True)

say("=" * 64)
say("KALSHI CALIBRATION STUDY - price {}h before close".format(MAIN_HORIZON))
say("=" * 64)
say("Markets: {} | Events: {} | Overall YES rate: {:.1%} | Avg price: {:.1%}".format(
    len(df), df["event_ticker"].nunique(), df["outcome"].mean(), df["prob"].mean()))

# ---- 1. calibration table + chart
table = calibration_table(df)
table.to_csv(os.path.join(RES_DIR, "calibration_table.csv"), index=False)
say("\n1. CALIBRATION TABLE")
say(table.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
plot_calibration(ax1, table, label="Kalshi ({}h before close)".format(MAIN_HORIZON))
ax1.set_title("Do Kalshi prices match real-world frequencies?")
ax1.legend(loc="upper left")
ax2.bar(table["price_bucket"], table["n"], color="steelblue")
ax2.set_title("Markets per price bucket")
ax2.tick_params(axis="x", rotation=60)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "1_calibration.png"), dpi=150)
plt.close(fig)

# ---- 2. accuracy scores
b = brier(df["prob"].values, df["outcome"].values)
ll = log_loss(df["prob"].values, df["outcome"].values)
rel, res, unc = murphy_decomposition(df)
b_ci = cluster_bootstrap(df, lambda s: brier(s["prob"].values, s["outcome"].values))
say("\n2. ACCURACY SCORES")
say("Brier score: {:.4f}  (95% CI {:.4f} to {:.4f}); always guessing 50% scores 0.25".format(b, *b_ci))
say("   Reliability (calibration error): {:.4f}".format(rel))
say("   Resolution (information):         {:.4f}".format(res))
say("   Uncertainty (difficulty):         {:.4f}".format(unc))
say("Log loss: {:.4f}  (always guessing 50% scores 0.693)".format(ll))

# ---- 3. favorite-longshot test
say("\n3. FAVORITE-LONGSHOT TEST (gap = actual YES rate - average price)")
for name, mask in [("Longshots (price < 15c)", df["prob"] < 0.15),
                   ("Middle (15c-85c)", (df["prob"] >= 0.15) & (df["prob"] <= 0.85)),
                   ("Favorites (price > 85c)", df["prob"] > 0.85)]:
    sub = df[mask].reset_index(drop=True)
    if len(sub) < 20:
        say("   {:<26} too few markets ({})".format(name, len(sub)))
        continue
    gap = sub["outcome"].mean() - sub["prob"].mean()
    lo, hi = cluster_bootstrap(sub, lambda s: s["outcome"].mean() - s["prob"].mean())
    verdict = "significant" if (lo > 0 or hi < 0) else "not significant"
    say("   {:<26} n={:>6}  gap={:+.3f}  95% CI [{:+.3f}, {:+.3f}]  {}".format(
        name, len(sub), gap, lo, hi, verdict))

# ---- 4. recalibration slope
a_hat, b_hat = recalibration_fit(df["prob"].values, df["outcome"].values)
slope_ci = cluster_bootstrap(df, lambda s: recalibration_fit(s["prob"].values, s["outcome"].values)[1],
                             n=min(N_BOOTSTRAP, 500))
say("\n4. RECALIBRATION SLOPE (1.0 = perfect; >1 = favorite-longshot bias; <1 = prices too extreme)")
say("   intercept a = {:+.3f}, slope b = {:.3f}  (95% CI {:.3f} to {:.3f})".format(a_hat, b_hat, *slope_ci))

# ---- 5. breakdowns
def breakdown(data, column, title, filename):
    say("\n{}".format(title))
    groups = [(k, g) for k, g in data.groupby(column, observed=True) if len(g) >= MIN_ROWS_PER_GROUP]
    if not groups:
        say("   no group has {}+ markets yet - collect more data".format(MIN_ROWS_PER_GROUP))
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i, (k, g) in enumerate(groups):
        g = g.reset_index(drop=True)
        t = calibration_table(g, n_bins=5)
        plot_calibration(ax, t, label="{} (n={})".format(k, len(g)), color=colors[i % 10])
        bs = brier(g["prob"].values, g["outcome"].values)
        r, _, _ = murphy_decomposition(g, n_bins=5)
        say("   {:<20} n={:>6}  Brier={:.4f}  calibration error={:.4f}".format(str(k), len(g), bs, r))
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l != "Perfect calibration"]
    ax.legend([h for h, _ in keep], [l for _, l in keep], fontsize=8, loc="upper left")
    ax.set_title(title.split(". ", 1)[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close(fig)

breakdown(df, "category", "5. BY CATEGORY", "2_by_category.png")
breakdown(df, "liquidity", "6. BY LIQUIDITY (trading volume)", "3_by_liquidity.png")
# Fair time comparison: only markets that have a price at EVERY snapshot
# time, so each row describes the same set of questions.
n_horizons = all_df["hours_before_close"].nunique()
counts = all_df.groupby("ticker")["hours_before_close"].nunique()
common = counts[counts == n_horizons].index
same_markets = all_df[all_df["ticker"].isin(common)].reset_index(drop=True)
say("\n(Time comparison uses only the {} markets priced at all {} snapshot times.)".format(
    len(common), n_horizons))
breakdown(same_markets, "hours_before_close", "7. BY TIME BEFORE CLOSE (hours, same markets)", "4_by_horizon.png")

with open(os.path.join(RES_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(report))
print("\nSTEP 3 DONE. Charts are in 'figures', numbers in 'results/summary.txt'.")
print("Next: run step4_backtest.py")
