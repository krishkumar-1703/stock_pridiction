import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

import config
from features import build_features
from lstm_model import LSTMRegressor, predict_returns


def download_live():
    """Latest daily data, same source, start date and price adjustment as the frozen snapshot."""
    df = yf.download(config.TICKER, start=config.START_DATE, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Download returned no data. Check the internet connection and retry.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def drop_incomplete_day(df):
    """Nifty trades 9:15-15:30 IST. Drop today's row if the session may still be running."""
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if df.index[-1].date() == now.date() and (now.hour * 60 + now.minute) < 15 * 60 + 45:
        return df.iloc[:-1], True
    return df, False


def load_models():
    lr = joblib.load(config.MODELS_DIR / "linear_regression.joblib")
    rf = joblib.load(config.MODELS_DIR / "random_forest.joblib")
    b = torch.load(config.LSTM_PATH, weights_only=True)
    net = LSTMRegressor(b["n_features"], b["hidden"], b["dropout"])
    net.load_state_dict(b["state_dict"])
    net.eval()
    return lr, rf, net, b


def make_forecast(df, models):
    """Next-session return and price from every model, using only data up to the last row of df."""
    lr, rf, net, b = models
    cols = b["features"]
    assert list(cols) == list(rf["features"]) == list(lr["features"])

    feats = build_features(df).dropna()
    asof = feats.index[-1]
    close_t = float(df["Close"].loc[asof])
    last = feats[cols].iloc[[-1]]

    rets = {"Naive": 0.0}
    rets[f"Moving average (N={lr['ma_window']})"] = float(feats["ret_lag0"].iloc[-lr["ma_window"]:].mean())
    rets["Linear Regression"] = float(lr["model"].predict(lr["scaler"].transform(last))[0])
    rets["Random Forest"] = float(rf["model"].predict(last)[0])

    window = feats[cols].iloc[-b["lookback"]:].values
    scaled = (window - np.array(b["scaler_mean"])) / np.array(b["scaler_scale"])
    X = scaled[None, :, :].astype(np.float32)
    rets["LSTM"] = float(predict_returns(net, X, b["y_mean"], b["y_std"])[0])

    table = pd.DataFrame({"pred_return": pd.Series(rets)})
    table["pred_close"] = close_t * (1 + table["pred_return"])
    return asof, close_t, table


def main():
    df = download_live()
    df, dropped = drop_incomplete_day(df)
    print("Live rows:", len(df), "| last complete day:", df.index[-1].date(),
          "| dropped an unfinished day:", dropped)

    # Does the live pipeline reproduce the frozen features on the days they share?
    saved = pd.read_csv(config.PROCESSED_DIR / "features.csv", index_col="Date", parse_dates=True)
    live_feats = build_features(df).dropna()
    common = saved.index.intersection(live_feats.index)
    diff = float(np.abs(saved.loc[common].values - live_feats.loc[common].values).max())
    print("Shared days:", len(common), "| max feature difference vs frozen snapshot: %.2e" % diff)
    print("Live features match the snapshot:", diff < 1e-6)

    models = load_models()
    asof, close_t, table = make_forecast(df, models)
    print("\nForecast made from the close of", asof.date(), "| close = %.2f" % close_t)
    print("Prediction is for the NEXT trading session.")
    print(table.round(6).to_string())
    print("\nModel estimates only. Not financial advice.")


if __name__ == "__main__":
    main()