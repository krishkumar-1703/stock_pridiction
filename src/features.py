import numpy as np
import pandas as pd
import config


def build_features(df):
    """Build relative (stationary) features. Every value at row t uses data up to day t only."""
    close = df["Close"]
    ret = close.pct_change()

    feats = pd.DataFrame(index=df.index)

    # Today's return and lagged returns
    for k in range(0, 5):
        feats[f"ret_lag{k}"] = ret.shift(k)

    # Multi-day returns (momentum)
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_10d"] = close.pct_change(10)

    # Price relative to moving averages (ratio, not raw level)
    for n in (5, 10, 20, 50):
        feats[f"close_sma{n}"] = close / close.rolling(n).mean() - 1

    # Volatility
    feats["vol_5"] = ret.rolling(5).std()
    feats["vol_21"] = ret.rolling(21).std()

    # Intraday shape
    feats["hl_range"] = (df["High"] - df["Low"]) / close
    feats["oc_change"] = (close - df["Open"]) / df["Open"]

    # RSI (Wilder's smoothing, 14 days)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    feats["rsi_14"] = 100 - 100 / (1 + rs)

    return feats


def main():
    df = pd.read_csv(config.snapshot_path(), index_col="Date", parse_dates=True)

    full = build_features(df)
    feats = full.dropna()   # drops the first ~49 warm-up rows (the 50-day average needs 50 days)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "features.csv"
    feats.to_csv(out)

    print("Rows before dropna:", len(full), "| after:", len(feats))
    print("First date:", feats.index.min().date(), "| Last date:", feats.index.max().date())
    print("NaN left:", int(feats.isna().sum().sum()), "| inf left:", int(np.isinf(feats.values).sum()))
    print("\nFeature columns:", list(feats.columns))
    print("\n=== FEATURE SUMMARY ===")
    print(feats.describe().T[["mean", "std", "min", "max"]].round(4))

    # ---- Look-ahead test ----
    # Rebuild features using only the first 2000 rows. If any feature at those rows
    # changes compared with the full-data version, it was using future information.
    part = build_features(df.iloc[:2000])
    same = np.allclose(full.iloc[:2000].values, part.values, equal_nan=True)
    print("\nLook-ahead test (features unchanged when future rows are removed):", same)
    print("Saved to:", out)


if __name__ == "__main__":
    main()