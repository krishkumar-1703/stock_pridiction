import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from split import get_splits
from metrics import evaluate, directional_accuracy
from train_rf import make_rf
from train_lstm import train_one
from lstm_model import make_windows, predict_returns

# RULE: this is the ONLY script that scores the test set.
# Nothing is tuned or changed after seeing its output.
TOL = 2e-6


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    train, val, test = get_splits(data)
    feature_cols = [c for c in data.columns if c not in config.NON_FEATURE_COLS]
    n_train, n_val, n_test = len(train), len(val), len(test)
    L = config.LOOKBACK
    y_test = test[config.TARGET_COL].values
    print("Test rows:", n_test, "| from", test.index.min().date(), "to", test.index.max().date())

    # ---- Frozen settings, read from the files saved earlier ----
    lr_bundle = joblib.load(config.MODELS_DIR / "linear_regression.joblib")
    rf_bundle = joblib.load(config.MODELS_DIR / "random_forest.joblib")
    assert lr_bundle["features"] == feature_cols and rf_bundle["features"] == feature_cols
    ma_n = lr_bundle["ma_window"]
    depth, leaf = rf_bundle["max_depth"], rf_bundle["min_samples_leaf"]
    print(f"Frozen settings: MA window {ma_n} | RF depth {depth}, min leaf {leaf} | "
          f"LSTM lookback {L}, seeds {config.LSTM_SEEDS}")

    preds_test = {}

    # ---- Baselines ----
    preds_test["naive"] = np.zeros(n_test)
    preds_test["train_mean"] = np.full(n_test, float(train[config.TARGET_COL].mean()))
    preds_test["moving_avg"] = data["ret_lag0"].rolling(ma_n).mean().loc[test.index].values
    preds_test["linear_reg"] = lr_bundle["model"].predict(lr_bundle["scaler"].transform(test[feature_cols]))
    print("NaN in moving-average predictions:", int(np.isnan(preds_test["moving_avg"]).sum()))

    # ---- Random Forest, 5 seeds ----
    print("\n=== RANDOM FOREST: reproduce validation, then predict test ===")
    X_train, y_train = train[feature_cols], train[config.TARGET_COL].values
    rf_test, rf_val_rmse = {}, {}
    for seed in config.RF_SEEDS:
        m = make_rf(depth, leaf, seed).fit(X_train, y_train)
        rf_val_rmse[seed] = evaluate(val, m.predict(val[feature_cols]))["rmse_ret"]
        rf_test[seed] = m.predict(test[feature_cols])
    saved_rf = pd.read_csv(config.RESULTS_DIR / "rf_validation.csv", index_col=0)["rmse_ret"].iloc[0]
    now = rf_val_rmse[config.RF_SEEDS[0]]
    print("seed %d: validation RMSE now %.6f | saved %.6f | reproduced: %s"
          % (config.RF_SEEDS[0], now, saved_rf, abs(now - saved_rf) < TOL))

    # ---- LSTM, 5 seeds (same preparation as train_lstm.py) ----
    print("\n=== LSTM: reproduce validation, then predict test ===")
    scaler = StandardScaler().fit(train[feature_cols])          # TRAIN only
    X_all = scaler.transform(data[feature_cols])
    y_train_raw = train[config.TARGET_COL].values
    y_mean, y_std = float(y_train_raw.mean()), float(y_train_raw.std())
    y_all = ((data[config.TARGET_COL].values - y_mean) / y_std).astype(np.float32)

    train_pos = np.arange(L - 1, n_train)
    val_pos = np.arange(n_train, n_train + n_val)
    test_pos = np.arange(n_train + n_val, n_train + n_val + n_test)
    Xtr, ytr = make_windows(X_all, train_pos, L), y_all[train_pos]
    Xva, yva = make_windows(X_all, val_pos, L), y_all[val_pos]
    Xte = make_windows(X_all, test_pos, L)
    print("Test windows:", Xte.shape, "| alignment check:",
          bool(np.allclose(Xte[0, -1], X_all[n_train + n_val])))

    saved_l = pd.read_csv(config.RESULTS_DIR / "lstm_seeds_validation.csv").set_index("seed")["rmse_ret"]
    lstm_test = {}
    for seed in config.LSTM_SEEDS:
        model, hist, best_epoch = train_one(seed, Xtr, ytr, Xva, yva, len(feature_cols))
        now = evaluate(val, predict_returns(model, Xva, y_mean, y_std))["rmse_ret"]
        lstm_test[seed] = predict_returns(model, Xte, y_mean, y_std)
        print("seed %d: best epoch %2d | validation RMSE now %.6f | saved %.6f | reproduced: %s"
              % (seed, best_epoch, now, saved_l[seed], abs(now - saved_l[seed]) < TOL))

    # ---- Score the TEST set ----
    results = {
        "Naive (tomorrow = today)": [evaluate(test, preds_test["naive"])],
        "Train-mean return (constant)": [evaluate(test, preds_test["train_mean"])],
        f"Moving average of returns (N={ma_n})": [evaluate(test, preds_test["moving_avg"])],
        "Linear Regression": [evaluate(test, preds_test["linear_reg"])],
        "Random Forest (mean of 5 seeds)": [evaluate(test, rf_test[s]) for s in config.RF_SEEDS],
        "LSTM (mean of 5 seeds)": [evaluate(test, lstm_test[s]) for s in config.LSTM_SEEDS],
    }
    mean_tbl = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in results.items()}).T
    std_tbl = pd.DataFrame({k: pd.DataFrame(v).std(ddof=0) for k, v in results.items()}).T

    print("\n=== TEST: PRICE LEVEL (the table format from the project brief) ===")
    print(mean_tbl[["rmse_price", "mae_price", "mape_price", "dir_acc"]].round(4).to_string())
    print("\n=== TEST: RETURN LEVEL ===")
    print(mean_tbl[["rmse_ret", "mae_ret", "r2_ret", "r2_price"]].round(6).to_string())

    print("\n=== SEED SPREAD ON TEST (std over 5 seeds) ===")
    print(std_tbl.loc[["Random Forest (mean of 5 seeds)", "LSTM (mean of 5 seeds)"],
                      ["rmse_ret", "rmse_price", "dir_acc"]].round(6).to_string())

    print("\n=== PER-SEED TEST RESULTS ===")
    for label, d in [("Random Forest", rf_test), ("LSTM", lstm_test)]:
        for s, p in d.items():
            m = evaluate(test, p)
            print(f"{label:<14} seed {s}: rmse_ret {m['rmse_ret']:.6f} | r2 {m['r2_ret']:.4f} "
                  f"| dir_acc {m['dir_acc']:.1f}%")

    print("\n=== DIRECTION REFERENCES (test, % of days correct) ===")
    refs = {"Always up": np.ones(n_test), "Same direction as today": test["ret_lag0"].values}
    for name, p in refs.items():
        print(f"{name:<26} {directional_accuracy(y_test, p):.1f}%")
    print("Sampling noise on a direction score over %d days: about +/- %.1f points"
          % (n_test, 1.96 * np.sqrt(0.25 / n_test) * 100))

    # ---- Save ----
    mean_tbl.to_csv(config.RESULTS_DIR / "final_test_table.csv")
    std_tbl.to_csv(config.RESULTS_DIR / "final_test_table_seed_std.csv")
    out = pd.DataFrame({"actual_ret": y_test}, index=test.index)
    for k, v in preds_test.items():
        out[k] = v
    for s in config.RF_SEEDS:
        out[f"rf_s{s}"] = rf_test[s]
    for s in config.LSTM_SEEDS:
        out[f"lstm_s{s}"] = lstm_test[s]
    out.to_csv(config.RESULTS_DIR / "test_predictions.csv")
    print("\nSaved final tables and test predictions.")


if __name__ == "__main__":
    main()