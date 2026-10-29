import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import config
from split import get_splits
from metrics import evaluate
from lstm_model import LSTMRegressor, make_windows, predict_returns, set_seed


def train_one(seed, Xtr, ytr, Xva, yva, n_features):
    set_seed(seed)
    model = LSTMRegressor(n_features, config.LSTM_HIDDEN, config.LSTM_DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR,
                           weight_decay=config.LSTM_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    gen = torch.Generator().manual_seed(seed)

    hist = {"train": [], "val": []}
    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    n = len(Xtr_t)
    for epoch in range(1, config.LSTM_MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=gen)
        for i in range(0, n, config.LSTM_BATCH):
            idx = order[i:i + config.LSTM_BATCH]
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(Xtr_t), ytr_t).item()
            va_loss = loss_fn(model(Xva_t), yva_t).item()
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val, best_epoch, bad = va_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.LSTM_PATIENCE:
                break

    model.load_state_dict(best_state)      # go back to the best epoch
    return model, hist, best_epoch


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    train, val, test = get_splits(data)          # test is NOT used in this file
    feature_cols = [c for c in data.columns if c not in config.NON_FEATURE_COLS]
    n_train, n_val, L = len(train), len(val), config.LOOKBACK

    # Scalers are fit on TRAIN only, then applied to every row
    scaler = StandardScaler().fit(train[feature_cols])
    X_all = scaler.transform(data[feature_cols])
    y_train_raw = train[config.TARGET_COL].values
    y_mean, y_std = float(y_train_raw.mean()), float(y_train_raw.std())
    y_all = ((data[config.TARGET_COL].values - y_mean) / y_std).astype(np.float32)

    train_pos = np.arange(L - 1, n_train)              # first L-1 train rows lack a full window
    val_pos = np.arange(n_train, n_train + n_val)
    Xtr, ytr = make_windows(X_all, train_pos, L), y_all[train_pos]
    Xva, yva = make_windows(X_all, val_pos, L), y_all[val_pos]

    print("Train windows:", Xtr.shape, "| ends", train.index[-1].date())
    print("Validation windows:", Xva.shape, "| starts", val.index[0].date())
    ok = (np.allclose(Xva[0, -1], X_all[n_train]) and np.allclose(Xva[0, 0], X_all[n_train - L + 1]))
    print("Window alignment check:", bool(ok))

    print("\nReference: naive validation return RMSE = %.6f" % evaluate(val, np.zeros(n_val))["rmse_ret"])
    rf_path = config.RESULTS_DIR / "rf_validation.csv"
    if rf_path.is_file():
        rf = pd.read_csv(rf_path, index_col=0)
        print("Reference: Random Forest validation return RMSE = %.6f" % rf["rmse_ret"].iloc[0])
    print("Reference: predicting the train mean gives validation loss %.4f (scaled units)"
          % np.mean(yva ** 2))

    # ---- Train with several seeds ----
    print("\n=== SEEDS ===")
    rows, main_run = [], None
    for seed in config.LSTM_SEEDS:
        model, hist, best_epoch = train_one(seed, Xtr, ytr, Xva, yva, len(feature_cols))
        pred_ret = predict_returns(model, Xva, y_mean, y_std)
        m = evaluate(val, pred_ret)
        rows.append({"seed": seed, "best_epoch": best_epoch, "epochs_run": len(hist["val"]),
                     "train_loss": hist["train"][best_epoch - 1], "val_loss": hist["val"][best_epoch - 1],
                     "rmse_ret": m["rmse_ret"], "r2_ret": m["r2_ret"], "dir_acc": m["dir_acc"]})
        print(f"seed {seed}: best epoch {best_epoch:>3} of {len(hist['val']):>3} run | "
              f"train loss {rows[-1]['train_loss']:.4f} | val loss {rows[-1]['val_loss']:.4f} | "
              f"rmse_ret {m['rmse_ret']:.6f} | r2 {m['r2_ret']:.4f} | dir_acc {m['dir_acc']:.1f}%")
        if main_run is None:
            main_run = (seed, model, hist, best_epoch, pred_ret, m)

    table = pd.DataFrame(rows)
    print("\nRMSE mean %.6f | std %.6f" % (table["rmse_ret"].mean(), table["rmse_ret"].std(ddof=0)))
    print("Direction mean %.1f%% | std %.1f" % (table["dir_acc"].mean(), table["dir_acc"].std(ddof=0)))
    table.to_csv(config.RESULTS_DIR / "lstm_seeds_validation.csv", index=False)

    # ---- Main model (first seed) ----
    seed, model, hist, best_epoch, pred_ret, res = main_run
    print("\n=== MAIN MODEL (seed %d) ON VALIDATION ===" % seed)
    print("Return level: rmse %.6f | mae %.6f | r2 %.4f | dir_acc %.1f%%"
          % (res["rmse_ret"], res["mae_ret"], res["r2_ret"], res["dir_acc"]))
    print("Price level:  rmse %.2f | mae %.2f | mape %.4f%% | r2 %.4f"
          % (res["rmse_price"], res["mae_price"], res["mape_price"], res["r2_price"]))

    # ---- Training curves ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["train"], label="train loss")
    ax.plot(hist["val"], label="validation loss")
    ax.axhline(np.mean(yva ** 2), color="red", linestyle=":", label="predict train mean (validation)")
    ax.axvline(best_epoch - 1, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch (0-based)")
    ax.set_ylabel("MSE on scaled target")
    ax.set_title(f"LSTM training curves (seed {seed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "lstm_training_curves.png", dpi=120)
    plt.close(fig)

    # ---- Save ----
    pd.DataFrame({"actual_ret": val[config.TARGET_COL].values, "lstm": pred_ret}, index=val.index) \
        .to_csv(config.RESULTS_DIR / "lstm_val_predictions.csv")
    torch.save({"state_dict": model.state_dict(), "n_features": len(feature_cols),
                "hidden": config.LSTM_HIDDEN, "dropout": config.LSTM_DROPOUT, "lookback": L,
                "features": feature_cols, "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(), "y_mean": y_mean, "y_std": y_std},
               config.LSTM_PATH)
    print("\nSaved model, predictions, seed table and training curves.")


if __name__ == "__main__":
    main()