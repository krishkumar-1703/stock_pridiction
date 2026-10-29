import pandas as pd
import config

df = pd.read_csv(config.snapshot_path(), index_col="Date", parse_dates=True)

print("=== SHAPE & DTYPES ===")
print(df.shape)
print(df.dtypes)

print("\n=== ORDERING & DUPLICATES ===")
print("Dates sorted ascending:", df.index.is_monotonic_increasing)
print("Duplicate dates:", df.index.duplicated().sum())
gaps = df.index.to_series().diff().dt.days
print("Largest gaps between rows (days):")
print(gaps.sort_values(ascending=False).head(5))

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== SANITY CHECKS ===")
print("High < Low rows:", (df["High"] < df["Low"]).sum())
print("Close outside [Low, High]:", ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum())
print("Non-positive prices:", (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum())
print("Zero-volume days:", (df["Volume"] == 0).sum())

print("\n=== SUMMARY STATS ===")
print(df.describe().round(2))

print("\n=== OUTLIERS (daily returns) ===")
ret = df["Close"].pct_change().dropna()
print("Mean daily return: %.5f | Std: %.5f" % (ret.mean(), ret.std()))
print("Days with |return| > 5%:", (ret.abs() > 0.05).sum())
print("5 biggest down days:")
print(ret.nsmallest(5).round(4))
print("5 biggest up days:")
print(ret.nlargest(5).round(4))
