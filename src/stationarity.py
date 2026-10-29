import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import config

df = pd.read_csv(config.snapshot_path(), index_col="Date", parse_dates=True)
close = df["Close"]
ret = close.pct_change().dropna()

def adf_report(name, series):
    stat, pvalue, *_ = adfuller(series.dropna())
    print("%-18s ADF statistic: %8.3f | p-value: %.4f" % (name, stat, pvalue))

print("=== ADF TEST (null hypothesis: series is NON-stationary) ===")
adf_report("Close price", close)
adf_report("Daily returns", ret)

# ---- ACF / PACF plots ----
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
plot_acf(ret, lags=40, ax=axes[0, 0], title="ACF of returns")
plot_pacf(ret, lags=40, ax=axes[0, 1], title="PACF of returns")
plot_acf(ret ** 2, lags=40, ax=axes[1, 0], title="ACF of SQUARED returns")
plot_pacf(ret ** 2, lags=40, ax=axes[1, 1], title="PACF of SQUARED returns")
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "eda_acf_pacf.png", dpi=120)
plt.close(fig)

print("\nSaved:", config.RESULTS_DIR / "eda_acf_pacf.png")