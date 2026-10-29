import pandas as pd
import matplotlib.pyplot as plt
import config

config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(config.snapshot_path(), index_col="Date", parse_dates=True)
close = df["Close"]
ret = close.pct_change().dropna()          # simple daily return: (today - yesterday) / yesterday

# ---- Plot 1: price over time ----
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(close.index, close.values)
ax.set_title("Nifty 50 Close price")
ax.set_ylabel("Index level")
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "eda_price.png", dpi=120)
plt.close(fig)

# ---- Plot 2: return distribution ----
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(ret.values, bins=100)
ax.set_title("Distribution of daily returns")
ax.set_xlabel("Daily return")
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "eda_return_hist.png", dpi=120)
plt.close(fig)

# ---- Plot 3: rolling volatility (21 trading days is about 1 month) ----
vol = ret.rolling(21).std()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(vol.index, vol.values)
ax.set_title("21-day rolling volatility of daily returns")
ax.set_ylabel("Std of daily return")
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "eda_rolling_vol.png", dpi=120)
plt.close(fig)

# ---- Numbers ----
print("=== RETURN STATS ===")
print("Mean: %.5f | Std: %.5f" % (ret.mean(), ret.std()))
print("Skewness: %.3f" % ret.skew())
print("Excess kurtosis: %.3f (normal distribution = 0)" % ret.kurt())
print("Share of up days: %.3f" % (ret > 0).mean())

print("\n=== ROLLING VOLATILITY ===")
print("Min: %.5f | Median: %.5f | Max: %.5f" % (vol.min(), vol.median(), vol.max()))
print("Date of max volatility:", vol.idxmax().date())

print("\n=== PRICE RANGE: FIRST 70% vs LAST 30% OF DATES ===")
cut = int(len(close) * 0.7)
print("First 70%%: min %.0f, max %.0f  (ends %s)" % (close[:cut].min(), close[:cut].max(), close.index[cut - 1].date()))
print("Last 30%%:  min %.0f, max %.0f  (starts %s)" % (close[cut:].min(), close[cut:].max(), close.index[cut].date()))
print("\nPlots saved in:", config.RESULTS_DIR)