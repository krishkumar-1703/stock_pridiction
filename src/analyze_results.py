import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from split import get_splits

BLOCK = 20        # block length in days (errors are not independent: volatility clusters)
N_BOOT = 5000


def block_bootstrap_means(d, rng):
    n = len(d)
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]
    return d[idx].mean(axis=1)


def main():
    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    _, _, test = get_splits(data)
    preds = pd.read_csv(config.RESULTS_DIR / "test_predictions.csv", index_col="Date", parse_dates=True)
    assert preds.index.equals(test.index)

    y = preds["actual_ret"].values
    rf_name = f"Random Forest (seed {config.RF_SEEDS[0]})"
    lstm_name = f"LSTM (seed {config.LSTM_SEEDS[0]})"
    models = {
        "Naive": preds["naive"],
        rf_name: preds[f"rf_s{config.RF_SEEDS[0]}"],
        lstm_name: preds[f"lstm_s{config.LSTM_SEEDS[0]}"],
    }

    # ---- Figure 1: predicted vs actual PRICE (full test period, then last 60 days) ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(test.index, test["next_close"], color="black", label="actual next-day close")
    for name, p in models.items():
        axes[0].plot(test.index, test["close"] * (1 + p), linewidth=1, label=name)
    axes[0].set_title("Test period: predicted vs actual next-day close (x = forecast date)")
    axes[0].legend()
    sub = test.iloc[-60:]
    axes[1].plot(sub.index, sub["next_close"], color="black", marker="o", markersize=3, label="actual")
    for name, p in models.items():
        axes[1].plot(sub.index, sub["close"] * (1 + p.iloc[-60:]), linewidth=1, label=name)
    axes[1].set_title("Zoom: last 60 test days")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "test_pred_vs_actual.png", dpi=120)
    plt.close(fig)

    # ---- Figure 2: predicted vs actual RETURN ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, name in zip(axes, [rf_name, lstm_name]):
        ax.scatter(models[name], y, s=6, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("predicted next-day return")
        ax.set_ylabel("actual next-day return")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "test_return_scatter.png", dpi=120)
    plt.close(fig)

    # ---- Figure 3: residuals, and error over time ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in models.items():
        res = y - p.values
        axes[0].hist(res, bins=60, histtype="step", label=name)
        sq = pd.Series(res ** 2, index=test.index)
        roll = sq.rolling(63).mean() ** 0.5
        axes[1].plot(roll.index, roll.values, label=name)
    axes[0].set_title("Residuals (actual - predicted return)")
    axes[0].legend()
    axes[1].set_title("Rolling 63-day return RMSE (error over time)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "test_residuals_and_error.png", dpi=120)
    plt.close(fig)

    # ---- Paired block bootstrap vs naive (test set) ----
    rng = np.random.default_rng(config.RANDOM_SEED)
    se_naive = (y - preds["naive"].values) ** 2
    comparisons = {
        "Train-mean return": preds["train_mean"].values,
        "Moving average": preds["moving_avg"].values,
        "Linear Regression": preds["linear_reg"].values,
        rf_name: models[rf_name].values,
        lstm_name: models[lstm_name].values,
    }
    rows = []
    for name, p in comparisons.items():
        d = se_naive - (y - p) ** 2               # positive = model beats naive that day
        boot = block_bootstrap_means(d, rng)
        base = se_naive.mean()
        rows.append({"model": name,
                     "mse_reduction_pct": d.mean() / base * 100,
                     "ci95_low_pct": np.percentile(boot, 2.5) / base * 100,
                     "ci95_high_pct": np.percentile(boot, 97.5) / base * 100,
                     "share_resamples_not_better": float((boot <= 0).mean())})
    table = pd.DataFrame(rows).set_index("model")
    print("=== MSE REDUCTION VS NAIVE ON TEST (positive = better than naive) ===")
    print(table.round(3).to_string())
    print("\nA 95% interval that includes 0 means the difference cannot be told apart from chance.")
    table.to_csv(config.RESULTS_DIR / "significance_vs_naive.csv")
    print("Saved 3 plots and significance_vs_naive.csv")


if __name__ == "__main__":
    main()