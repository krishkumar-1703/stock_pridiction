import numpy as np
import config


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mape(y, p):
    # Percentage error. Breaks down when the true value is near zero (so we only use it on prices).
    return float(np.mean(np.abs((y - p) / y)) * 100)


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


def directional_accuracy(actual_ret, pred_ret):
    """Percent of days where predicted up/down matches actual up/down."""
    return float(np.mean((np.asarray(pred_ret) > 0) == (np.asarray(actual_ret) > 0)) * 100)


def evaluate(df, pred_ret):
    """df needs columns: close, next_close, target_ret. pred_ret is the predicted next-day return."""
    p_ret = np.asarray(pred_ret, dtype=float)
    y_ret = df[config.TARGET_COL].values
    y_price = df["next_close"].values
    p_price = df["close"].values * (1 + p_ret)
    return {
        "rmse_ret": rmse(y_ret, p_ret),
        "mae_ret": mae(y_ret, p_ret),
        "r2_ret": r2(y_ret, p_ret),
        "rmse_price": rmse(y_price, p_price),
        "mae_price": mae(y_price, p_price),
        "mape_price": mape(y_price, p_price),
        "r2_price": r2(y_price, p_price),
        # a forecast of exactly zero change has no direction
        "dir_acc": np.nan if np.all(p_ret == 0) else directional_accuracy(y_ret, p_ret),
    }