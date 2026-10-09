"""
STEP 4: Could anyone actually make money from the mispricing?

HOW TO RUN (in IDLE): open step4_backtest.py and press F5.
Needs: data/analysis.csv (from step 2)
Makes: figures/5_equity_curve.png and results/backtest.txt

THE IDEA:
  If cheap "longshot" contracts are overpriced, betting AGAINST them should
  win on average. Betting against YES = buying NO. We test two strategies:
    A) Fade longshots:  YES price < threshold  -> buy NO
    B) Back favorites:  YES price > 1 - threshold -> buy YES

WE MAKE IT REALISTIC:
  * We pay the ASK (the price you really get when buying right now), not the
    mid price. Buying NO costs (1 - YES bid).
  * We pay Kalshi's taker fee: round_up(0.07 x contracts x P x (1 - P)),
    where P is the price paid. (Check Kalshi's fee page for any changes.)
  * We choose the threshold using only the OLDER 60% of markets (training),
    then judge it on the NEWER 40% it has never seen (testing). Picking the
    best threshold on all the data would be cheating ("overfitting").
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- SETTINGS
HORIZON = 24                  # trade at the price 24h before close
CONTRACTS_PER_TRADE = 100     # size of each bet
FEE_RATE = 0.07               # Kalshi general taker fee multiplier
THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
TRAIN_FRACTION = 0.6
# -------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

report = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    report.append(line)


def kalshi_fee(price, contracts):
    """Taker fee in dollars for one order, rounded UP to the next cent."""
    raw = FEE_RATE * contracts * price * (1 - price)
    return math.ceil(round(raw * 100, 6)) / 100


def make_trades(df, strategy, threshold):
    """Return one row per trade with its cost and profit in dollars."""
    if strategy == "fade_longshots":
        pick = df[df["prob"] < threshold].copy()
        pick["price_paid"] = 1 - pick["yes_bid"]          # NO ask
        pick["won"] = pick["outcome"] == 0
    else:  # back_favorites
        pick = df[df["prob"] > 1 - threshold].copy()
        pick["price_paid"] = pick["yes_ask"]
        pick["won"] = pick["outcome"] == 1
    C = CONTRACTS_PER_TRADE
    pick["cost"] = pick["price_paid"] * C
    pick["fee"] = [kalshi_fee(p, C) for p in pick["price_paid"]]
    pick["pnl"] = np.where(pick["won"], C, 0) - pick["cost"] - pick["fee"]
    return pick


def summarise(trades):
    """Key numbers. The t-statistic uses profit per EVENT, because markets
    in the same event are linked and are not independent bets."""
    if len(trades) == 0:
        return None
    by_event = trades.groupby("event_ticker")["pnl"].sum()
    n_ev = len(by_event)
    t_stat = (by_event.mean() / (by_event.std(ddof=1) / np.sqrt(n_ev))) if n_ev > 2 and by_event.std() > 0 else np.nan
    equity = trades.sort_values("close_time")["pnl"].cumsum()
    drawdown = (equity - equity.cummax()).min()
    return {"trades": len(trades), "events": n_ev,
            "win_rate": trades["won"].mean(),
            "total_pnl": trades["pnl"].sum(),
            "fees": trades["fee"].sum(),
            "return_on_capital": trades["pnl"].sum() / trades["cost"].sum(),
            "pnl_per_contract_cents": 100 * trades["pnl"].sum() / (len(trades) * CONTRACTS_PER_TRADE),
            "t_stat": t_stat,
            "max_drawdown": drawdown}


def fmt(s):
    if s is None:
        return "no trades"
    return ("trades={trades:>5} events={events:>5} win={win_rate:6.1%} "
            "PnL=${total_pnl:>9.2f} fees=${fees:>7.2f} ROC={return_on_capital:+6.2%} "
            "per-contract={pnl_per_contract_cents:+.2f}c t={t_stat:+.2f} maxDD=${max_drawdown:.2f}").format(**s)


# ================================================================ BACKTEST
df = pd.read_csv(os.path.join(DATA_DIR, "analysis.csv"))
df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")
if len(df) and (df["hours_before_close"] == HORIZON).sum() == 0:
    HORIZON = int(df["hours_before_close"].value_counts().idxmax())
    print("No rows at the chosen horizon - using {}h before close instead.".format(HORIZON))
df = df[df["hours_before_close"] == HORIZON].sort_values("close_time").reset_index(drop=True)
if len(df) < 50:
    raise SystemExit("Only {} rows - collect more markets in step 1 first.".format(len(df)))

split_time = df["close_time"].quantile(TRAIN_FRACTION)
train = df[df["close_time"] <= split_time]
test = df[df["close_time"] > split_time]

say("=" * 64)
say("BACKTEST - trading {}h before close, {} contracts per trade".format(HORIZON, CONTRACTS_PER_TRADE))
say("Training: {} markets up to {:%Y-%m-%d} | Testing: {} markets after".format(
    len(train), split_time, len(test)))
say("=" * 64)

fig, ax = plt.subplots(figsize=(10, 5))
for strategy in ["fade_longshots", "back_favorites"]:
    say("\nSTRATEGY:", strategy.replace("_", " ").upper())
    say("Training results by threshold:")
    best_t, best_score = None, -np.inf
    for t in THRESHOLDS:
        s = summarise(make_trades(train, strategy, t))
        say("   threshold {:.2f}: {}".format(t, fmt(s)))
        if s and s["trades"] >= 20 and s["return_on_capital"] > best_score:
            best_t, best_score = t, s["return_on_capital"]
    if best_t is None:
        say("   not enough trades in training data to pick a threshold")
        continue
    test_trades = make_trades(test, strategy, best_t)
    s = summarise(test_trades)
    say("Chosen threshold (from training only): {:.2f}".format(best_t))
    say("OUT-OF-SAMPLE TEST: {}".format(fmt(s)))
    if s:
        if s["total_pnl"] > 0 and s["t_stat"] > 2:
            say("   -> Profitable after fees and statistically significant. Check it hard for errors!")
        elif s["total_pnl"] > 0:
            say("   -> Profitable but NOT statistically significant (could be luck).")
        else:
            say("   -> Not profitable after fees and spreads. An honest, valid finding.")
        eq = test_trades.sort_values("close_time")
        ax.plot(eq["close_time"], eq["pnl"].cumsum(),
                label="{} (threshold {:.2f})".format(strategy.replace("_", " "), best_t))

ax.axhline(0, color="gray", lw=1)
ax.set_title("Out-of-sample profit after fees and spreads")
ax.set_ylabel("Cumulative profit ($)")
ax.set_xlabel("Market close date")
ax.grid(alpha=0.3)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "5_equity_curve.png"), dpi=150)
plt.close(fig)

# Cost breakdown: how much of the raw edge do spread and fees eat?
say("\nCOST BREAKDOWN on all data, fade longshots at 0.10:")
tr = make_trades(df, "fade_longshots", 0.10)
if len(tr):
    C = CONTRACTS_PER_TRADE
    mid_cost = (1 - tr["prob"]) * C
    gross_at_mid = (np.where(tr["won"], C, 0) - mid_cost).sum()
    spread_cost = (tr["cost"] - mid_cost).sum()
    say("   Profit if you could trade at the mid price: ${:.2f}".format(gross_at_mid))
    say("   minus half-spread paid:                      ${:.2f}".format(spread_cost))
    say("   minus Kalshi fees:                           ${:.2f}".format(tr["fee"].sum()))
    say("   = Real profit:                               ${:.2f}".format(tr["pnl"].sum()))

with open(os.path.join(RES_DIR, "backtest.txt"), "w") as f:
    f.write("\n".join(report))
print("\nSTEP 4 DONE. See results/backtest.txt and figures/5_equity_curve.png")
