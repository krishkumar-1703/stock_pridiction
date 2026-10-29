import sys
import numpy as np
import pandas as pd
import config
from features import build_features

results = []


def check(name, condition, detail=""):
    ok = bool(condition)
    results.append(ok)
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""))


def section(title):
    print(f"\n--- {title} ---")


def check_files():
    section("Folders and files")
    for d in [config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.RESULTS_DIR]:
        check(f"folder exists: {d.name}", d.is_dir())
    for name in ["config.py", "download_data.py", "inspect_data.py", "eda.py",
                 "stationarity.py", "features.py", "target.py", "verify_project.py"]:
        check(f"src/{name} exists", (config.ROOT / "src" / name).is_file())
    req = config.ROOT / "requirements.txt"
    check("requirements.txt exists and is not empty", req.is_file() and req.stat().st_size > 0)


def check_raw():
    section("Raw data")
    snap = config.snapshot_path()
    check("snapshot file exists", snap.is_file())
    if not snap.is_file():
        return None
    raw = pd.read_csv(snap, index_col="Date", parse_dates=True)
    check("dates sorted ascending", raw.index.is_monotonic_increasing)
    check("no duplicate dates", not raw.index.duplicated().any())
    check("no missing OHLC values", raw[["Open", "High", "Low", "Close"]].isna().sum().sum() == 0)
    check("High >= Low on every row", (raw["High"] >= raw["Low"]).all())
    gap = (pd.Timestamp(config.END_DATE) - raw.index.max()).days
    check("last date within 5 days before END_DATE", 0 <= gap <= 5,
          f"last date {raw.index.max().date()}")
    return raw


def check_features(raw):
    section("Features")
    path = config.PROCESSED_DIR / "features.csv"
    check("features.csv exists", path.is_file())
    if not path.is_file() or raw is None:
        return
    feats = pd.read_csv(path, index_col="Date", parse_dates=True)
    check("no NaN in features", feats.isna().sum().sum() == 0)
    check("no inf in features", not np.isinf(feats.values).any())
    scale_cols = [c for c in feats.columns if c != "rsi_14"]
    check("all features are scale-free (|value| < 1, except RSI)",
          feats[scale_cols].abs().max().max() < 1)
    check("RSI stays within 0-100", feats["rsi_14"].between(0, 100).all())
    full = build_features(raw)
    rebuilt = full.dropna()
    same = feats.shape == rebuilt.shape and np.allclose(feats.values, rebuilt.values)
    check("saved features match a fresh rebuild", same)
    part = build_features(raw.iloc[:2000])
    check("look-ahead test (features unchanged when future rows removed)",
          np.allclose(full.iloc[:2000].values, part.values, equal_nan=True))


def check_dataset(raw):
    section("Dataset and target")
    check("dataset.csv exists", config.DATASET_PATH.is_file())
    if not config.DATASET_PATH.is_file() or raw is None:
        return
    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    feature_cols = [c for c in data.columns if c not in config.NON_FEATURE_COLS]
    check("16 feature columns", len(feature_cols) == 16, f"found {len(feature_cols)}")
    check("no price-level columns among features",
          not any(c in feature_cols for c in ["Open", "High", "Low", "Close", "close", "next_close"]))
    check("no NaN in dataset", data.isna().sum().sum() == 0)
    t = data[config.TARGET_COL].iloc[:-1].values
    r = data["ret_lag0"].iloc[1:].values
    check("target alignment (target at t == ret_lag0 at t+1)", np.allclose(t, r))
    expected = raw["Close"].shift(-1).loc[data.index] / raw["Close"].loc[data.index] - 1
    check("target recomputed from raw prices matches", np.allclose(data[config.TARGET_COL], expected))
    check("dataset ends one trading day before raw data ends", data.index.max() == raw.index[-2],
          f"dataset ends {data.index.max().date()}")


def check_split():
    section("Chronological split")
    paths = [config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH]
    for p in paths:
        check(f"{p.name} exists", p.is_file())
    if not all(p.is_file() for p in paths) or not config.DATASET_PATH.is_file():
        return
    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    train, val, test = [pd.read_csv(p, index_col="Date", parse_dates=True) for p in paths]
    check("split rows add up to the full dataset", len(train) + len(val) + len(test) == len(data))
    check("train ends before validation starts", train.index.max() < val.index.min())
    check("validation ends before test starts", val.index.max() < test.index.min())
    check("every split is sorted by date",
          train.index.is_monotonic_increasing and val.index.is_monotonic_increasing
          and test.index.is_monotonic_increasing)
    check("train is about 70% of the data", abs(len(train) / len(data) - config.TRAIN_FRAC) < 0.01)


def check_baselines():
    section("Baselines")
    import joblib
    from metrics import evaluate

    # Metric sanity tests on a tiny example with known answers
    toy = pd.DataFrame({"close": [100.0, 100.0], "next_close": [110.0, 90.0],
                        config.TARGET_COL: [0.10, -0.10]})
    perfect = evaluate(toy, [0.10, -0.10])
    check("metrics: perfect forecast gives zero error and 100% direction",
          np.isclose(perfect["rmse_ret"], 0) and np.isclose(perfect["rmse_price"], 0)
          and perfect["dir_acc"] == 100)
    check("metrics: zero-change forecast has no direction score",
          np.isnan(evaluate(toy, [0.0, 0.0])["dir_acc"]))
    check("metrics: one right call out of two gives 50%", evaluate(toy, [0.1, 0.1])["dir_acc"] == 50)

    model_path = config.MODELS_DIR / "linear_regression.joblib"
    table_path = config.RESULTS_DIR / "baselines_validation.csv"
    check("linear_regression.joblib exists", model_path.is_file())
    check("baselines_validation.csv exists", table_path.is_file())
    if not model_path.is_file() or not config.TRAIN_PATH.is_file():
        return
    bundle = joblib.load(model_path)
    train = pd.read_csv(config.TRAIN_PATH, index_col="Date", parse_dates=True)
    check("scaler was fit on TRAIN only (its means equal the train feature means)",
          np.allclose(bundle["scaler"].mean_, train[bundle["features"]].mean().values))
    check("model uses 16 relative features and no price columns",
          len(bundle["features"]) == 16
          and not any(c in config.NON_FEATURE_COLS for c in bundle["features"]))


def check_random_forest():
    section("Random Forest")
    import joblib
    model_path = config.MODELS_DIR / "random_forest.joblib"
    pred_path = config.RESULTS_DIR / "rf_val_predictions.csv"
    check("random_forest.joblib exists", model_path.is_file())
    check("rf_val_predictions.csv exists", pred_path.is_file())
    if not model_path.is_file() or not pred_path.is_file() or not config.VAL_PATH.is_file():
        return
    bundle = joblib.load(model_path)
    val = pd.read_csv(config.VAL_PATH, index_col="Date", parse_dates=True)
    preds = pd.read_csv(pred_path, index_col="Date", parse_dates=True)
    check("model uses 16 relative features and no price columns",
          len(bundle["features"]) == 16
          and not any(c in config.NON_FEATURE_COLS for c in bundle["features"]))
    check("predictions cover exactly the validation dates", preds.index.equals(val.index))


def check_lstm():
    section("LSTM")
    import torch
    from lstm_model import make_windows

    X = np.arange(100, dtype=float).reshape(50, 2)
    w = make_windows(X, [10], 5)
    check("windows: last step is day t and first step is day t-4",
          np.array_equal(w[0, -1], X[10]) and np.array_equal(w[0, 0], X[6]))

    pred_path = config.RESULTS_DIR / "lstm_val_predictions.csv"
    check("lstm.pt exists", config.LSTM_PATH.is_file())
    check("lstm_val_predictions.csv exists", pred_path.is_file())
    if not config.LSTM_PATH.is_file() or not pred_path.is_file() or not config.TRAIN_PATH.is_file():
        return
    bundle = torch.load(config.LSTM_PATH, weights_only=True)
    train = pd.read_csv(config.TRAIN_PATH, index_col="Date", parse_dates=True)
    val = pd.read_csv(config.VAL_PATH, index_col="Date", parse_dates=True)
    preds = pd.read_csv(pred_path, index_col="Date", parse_dates=True)
    feats = bundle["features"]
    check("model uses 16 relative features and no price columns",
          len(feats) == 16 and not any(c in config.NON_FEATURE_COLS for c in feats))
    check("scaler was fit on TRAIN only (stored means equal train feature means)",
          np.allclose(bundle["scaler_mean"], train[feats].mean().values))
    check("target scaling uses train statistics only",
          np.isclose(bundle["y_mean"], train[config.TARGET_COL].mean())
          and np.isclose(bundle["y_std"], train[config.TARGET_COL].std(ddof=0)))
    check("predictions cover exactly the validation dates", preds.index.equals(val.index))


def check_final():
    section("Final test evaluation")
    pred_path = config.RESULTS_DIR / "test_predictions.csv"
    table_path = config.RESULTS_DIR / "final_test_table.csv"
    check("test_predictions.csv exists", pred_path.is_file())
    check("final_test_table.csv exists", table_path.is_file())
    if not pred_path.is_file() or not config.TEST_PATH.is_file():
        return
    test = pd.read_csv(config.TEST_PATH, index_col="Date", parse_dates=True)
    preds = pd.read_csv(pred_path, index_col="Date", parse_dates=True)
    check("test predictions cover exactly the test dates", preds.index.equals(test.index))
    check("actual returns in the predictions match the test targets",
          np.allclose(preds["actual_ret"], test[config.TARGET_COL]))
    check("no NaN in any prediction column", preds.isna().sum().sum() == 0)
    check("5 Random Forest and 5 LSTM seed columns are present",
          all(f"rf_s{s}" in preds.columns for s in config.RF_SEEDS)
          and all(f"lstm_s{s}" in preds.columns for s in config.LSTM_SEEDS))


def check_analysis():
    section("Result plots and significance")
    for name in ["test_pred_vs_actual.png", "test_return_scatter.png",
                 "test_residuals_and_error.png", "significance_vs_naive.csv"]:
        check(f"results/{name} exists", (config.RESULTS_DIR / name).is_file())


def check_dashboard():
    section("Forecast module and dashboard")
    forecast_ok = (config.ROOT / "src" / "forecast.py").is_file()
    check("src/forecast.py exists", forecast_ok)
    check("src/app.py exists", (config.ROOT / "src" / "app.py").is_file())
    pred_path = config.RESULTS_DIR / "test_predictions.csv"
    if not forecast_ok or not pred_path.is_file() or not config.snapshot_path().is_file():
        return
    from forecast import load_models, make_forecast

    raw = pd.read_csv(config.snapshot_path(), index_col="Date", parse_dates=True)
    test = pd.read_csv(config.TEST_PATH, index_col="Date", parse_dates=True)
    preds = pd.read_csv(pred_path, index_col="Date", parse_dates=True)
    last = test.index[-1]

    # Rebuild the forecast for the last test day from the frozen snapshot (no internet needed)
    asof, close_t, table = make_forecast(raw.loc[:last], load_models())
    check("forecast as-of date is the last test date", asof == last)
    row = preds.loc[last]
    ma_name = [i for i in table.index if i.startswith("Moving average")][0]
    pairs = [("Naive", "naive"), (ma_name, "moving_avg"), ("Linear Regression", "linear_reg"),
             ("Random Forest", f"rf_s{config.RF_SEEDS[0]}"), ("LSTM", f"lstm_s{config.LSTM_SEEDS[0]}")]
    for model_name, col in pairs:
        check(f"forecast module reproduces the saved test prediction: {model_name}",
              np.isclose(table.loc[model_name, "pred_return"], row[col], atol=1e-6, rtol=0))


def main():
    check_files()
    raw = check_raw()
    check_features(raw)
    check_dataset(raw)
    check_split()
    check_baselines()
    check_random_forest()
    check_lstm()
    check_final()
    check_analysis()
    check_dashboard()

    passed = sum(results)
    print(f"\n{passed} of {len(results)} checks passed.")
    if passed == len(results):
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - fix the [FAIL] lines above before moving on")
        sys.exit(1)


if __name__ == "__main__":
    main()