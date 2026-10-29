import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import config
from split import get_splits
from metrics import evaluate


def make_rf(max_depth, min_leaf, seed):
    return RandomForestRegressor(n_estimators=300, max_depth=max_depth,
                                 min_samples_leaf=min_leaf, max_features=0.5,
                                 n_jobs=-1, random_state=seed)


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    train, val, test = get_splits(data)          # test is NOT used in this file
    feature_cols = [c for c in data.columns if c not in config.NON_FEATURE_COLS]
    X_train, y_train = train[feature_cols], train[config.TARGET_COL].values
    X_val, y_val = val[feature_cols], val[config.TARGET_COL].values

    naive = evaluate(val, np.zeros(len(val)))
    print("Reference: naive validation return RMSE = %.6f" % naive["rmse_ret"])

    # ---- Tune on VALIDATION only ----
    print("\n=== GRID SEARCH (validation return RMSE) ===")
    best = None
    for depth in config.RF_DEPTHS:
        for leaf in config.RF_MIN_LEAFS:
            model = make_rf(depth, leaf, config.RF_SEEDS[0]).fit(X_train, y_train)
            score = evaluate(val, model.predict(X_val))["rmse_ret"]
            print(f"max_depth={depth} min_samples_leaf={leaf:>3}: {score:.6f}")
            if best is None or score < best[0]:
                best = (score, depth, leaf)
    _, best_depth, best_leaf = best
    print(f"Chosen: max_depth={best_depth}, min_samples_leaf={best_leaf}")

    # ---- Seed variance for the chosen setting ----
    print("\n=== SEED VARIANCE (chosen setting) ===")
    rmses, dirs, main_model = [], [], None
    for seed in config.RF_SEEDS:
        model = make_rf(best_depth, best_leaf, seed).fit(X_train, y_train)
        m = evaluate(val, model.predict(X_val))
        rmses.append(m["rmse_ret"])
        dirs.append(m["dir_acc"])
        print(f"seed {seed}: rmse_ret {m['rmse_ret']:.6f} | dir_acc {m['dir_acc']:.1f}%")
        if main_model is None:
            main_model = model
    print("RMSE mean %.6f | std %.6f" % (np.mean(rmses), np.std(rmses)))
    print("Direction mean %.1f%% | std %.1f" % (np.mean(dirs), np.std(dirs)))

    # ---- Main model results on validation ----
    pred_val = main_model.predict(X_val)
    res = evaluate(val, pred_val)
    print("\n=== MAIN MODEL (seed %d) ON VALIDATION ===" % config.RF_SEEDS[0])
    print("Return level: rmse %.6f | mae %.6f | r2 %.4f | dir_acc %.1f%%"
          % (res["rmse_ret"], res["mae_ret"], res["r2_ret"], res["dir_acc"]))
    print("Price level:  rmse %.2f | mae %.2f | mape %.4f%% | r2 %.4f"
          % (res["rmse_price"], res["mae_price"], res["mape_price"], res["r2_price"]))

    # ---- Feature importance ----
    imp = pd.Series(main_model.feature_importances_, index=feature_cols)
    perm = permutation_importance(main_model, X_val, y_val, n_repeats=10,
                                  scoring="neg_mean_squared_error",
                                  random_state=config.RANDOM_SEED, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=feature_cols)
    print("\n=== TOP 5 IMPURITY IMPORTANCE ===")
    print(imp.sort_values(ascending=False).head(5).round(4))
    print("\n=== TOP 5 PERMUTATION IMPORTANCE (validation, increase in MSE) ===")
    print(perm_imp.sort_values(ascending=False).head(5).map(lambda v: f"{v:.3e}"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    imp.sort_values().plot.barh(ax=axes[0], title="Impurity importance (train)")
    perm_imp.sort_values().plot.barh(ax=axes[1], title="Permutation importance (validation)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "rf_feature_importance.png", dpi=120)
    plt.close(fig)

    # ---- Save ----
    pd.DataFrame({"actual_ret": y_val, "random_forest": pred_val}, index=val.index) \
        .to_csv(config.RESULTS_DIR / "rf_val_predictions.csv")
    pd.DataFrame({"Random Forest": res}).T.to_csv(config.RESULTS_DIR / "rf_validation.csv")
    joblib.dump({"model": main_model, "features": feature_cols,
                 "max_depth": best_depth, "min_samples_leaf": best_leaf},
                config.MODELS_DIR / "random_forest.joblib")
    print("\nSaved model, predictions and importance plot.")


if __name__ == "__main__":
    main()