import numpy as np
import pandas as pd
import config


def main():
    raw = pd.read_csv(config.snapshot_path(), index_col="Date", parse_dates=True)
    feats = pd.read_csv(config.PROCESSED_DIR / "features.csv", index_col="Date", parse_dates=True)

    data = feats.copy()
    data["close"] = raw["Close"].loc[feats.index]
    # shift(-1) pulls TOMORROW's close onto today's row. The last row gets NaN (no tomorrow yet).
    data["next_close"] = raw["Close"].shift(-1).loc[feats.index]
    data[config.TARGET_COL] = data["next_close"] / data["close"] - 1

    n_before = len(data)
    data = data.dropna()
    data.to_csv(config.DATASET_PATH)

    feature_cols = [c for c in data.columns if c not in config.NON_FEATURE_COLS]

    print("Rows before:", n_before, "| after:", len(data))
    print("First date:", data.index.min().date(), "| Last date:", data.index.max().date())
    print("Number of features:", len(feature_cols))

    # ---- Alignment test: today's target must equal tomorrow's ret_lag0 ----
    t = data[config.TARGET_COL].iloc[:-1].values
    r = data["ret_lag0"].iloc[1:].values
    print("\nAlignment test (target at t == ret_lag0 at t+1):", bool(np.allclose(t, r)))

    print("\n=== TARGET (next-day return) ===")
    y = data[config.TARGET_COL]
    print("Mean: %.5f | Std: %.5f" % (y.mean(), y.std()))
    print("Share of up days: %.3f" % (y > 0).mean())

    print("\n=== LAST 3 ROWS ===")
    print(data[["close", "next_close", config.TARGET_COL]].tail(3))

    print("\n=== FEATURE vs TARGET CORRELATION (top 5 by size) ===")
    corr = data[feature_cols].corrwith(y)
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(5).round(4))
    print("Saved to:", config.DATASET_PATH)


if __name__ == "__main__":
    main()