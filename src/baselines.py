import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import config
from split import get_splits
from metrics import evaluate, directional_accuracy, r2


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    train, val, test = get_splits(data)          # test is NOT used in this file
    feature_cols = [c for c in data.columns if c not in config.NON_FEATURE_COLS]
    y_val = val[config.TARGET_COL].values

    # ---- 1. Naive: predicted return = 0 ----
    pred_naive = np.zeros(len(val))

    # ---- 2. Moving average of returns; window chosen on VALIDATION ----
    print("=== MOVING-AVERAGE WINDOW CHOICE (validation return RMSE) ===")
    best_n, best_score, pred_ma = None, np.inf, None
    for n in config.MA_WINDOWS:
        # rolling mean over ALL past returns (including train days), read at validation dates
        pred = data["ret_lag0"].rolling(n).mean().loc[val.index].values
        score = evaluate(val, pred)["rmse_ret"]
        print(f"window {n:>2}: {score:.6f}")
        if score < best_score:
            best_n, best_score, pred_ma = n, score, pred
    print("Chosen window:", best_n)

    # ---- 3. Linear Regression; scaler fit on TRAIN only ----
    scaler = StandardScaler().fit(train[feature_cols])
    X_train = scaler.transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    lr = LinearRegression().fit(X_train, train[config.TARGET_COL].values)
    pred_lr = lr.predict(X_val)

    print("\n=== LINEAR REGRESSION ===")
    print("Train R2 on returns: %.4f" % r2(train[config.TARGET_COL].values, lr.predict(X_train)))
    coefs = pd.Series(lr.coef_, index=feature_cols)
    print("Largest standardized coefficients:")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(5).round(5))

    # ---- Results on VALIDATION ----
    rows = {
        "Naive (tomorrow = today)": evaluate(val, pred_naive),
        f"Moving average of returns (N={best_n})": evaluate(val, pred_ma),
        "Linear Regression": evaluate(val, pred_lr),
    }
    table = pd.DataFrame(rows).T

    print("\n=== VALIDATION: RETURN LEVEL ===")
    print(table[["rmse_ret", "mae_ret", "r2_ret", "dir_acc"]].round(6).to_string())
    print("\n=== VALIDATION: PRICE LEVEL ===")
    print(table[["rmse_price", "mae_price", "mape_price", "r2_price"]].round(4).to_string())

    print("\n=== DIRECTION REFERENCES (validation, % of days correct) ===")
    refs = {"Always up": np.ones(len(val)), "Same direction as today": val["ret_lag0"].values}
    for name, p in refs.items():
        print(f"{name:<26} {directional_accuracy(y_val, p):.1f}%")

    # ---- Save ----
    table.to_csv(config.RESULTS_DIR / "baselines_validation.csv")
    preds = pd.DataFrame({"actual_ret": y_val, "naive": pred_naive,
                          "moving_avg": pred_ma, "linear_reg": pred_lr}, index=val.index)
    preds.to_csv(config.RESULTS_DIR / "baselines_val_predictions.csv")
    joblib.dump({"scaler": scaler, "model": lr, "features": feature_cols, "ma_window": best_n},
                config.MODELS_DIR / "linear_regression.joblib")
    print("\nSaved results and model.")


if __name__ == "__main__":
    main()