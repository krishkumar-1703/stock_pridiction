import pandas as pd
import yfinance as yf
import config

def main():
    # yfinance treats `end` as EXCLUSIVE, so add one day to include END_DATE
    end_exclusive = (pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        config.TICKER,
        start=config.START_DATE,
        end=end_exclusive,
        auto_adjust=True,   # prices adjusted for splits/dividends; fixed for the whole project
        progress=False,
    )

    if df.empty:
        raise SystemExit("Download returned no data. Check internet/ticker and retry.")

    # Newer yfinance versions return two-level column names even for one ticker; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.snapshot_path())

    print("Ticker:", config.TICKER)
    print("Saved to:", config.snapshot_path())
    print("Rows:", len(df))
    print("First date:", df.index.min().date(), "| Last date:", df.index.max().date())
    print(df.head(3))
    print(df.tail(3))

if __name__ == "__main__":
    main()
    
