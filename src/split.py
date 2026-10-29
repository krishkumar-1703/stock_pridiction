import pandas as pd
import config


def get_splits(data):
    """Cut a dataset into train / validation / test by date order. Used by every later phase."""
    n = len(data)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    train = data.iloc[:n_train]
    val = data.iloc[n_train:n_train + n_val]
    test = data.iloc[n_train + n_val:]
    return train, val, test


def describe_split(name, part):
    y = part[config.TARGET_COL]
    buy_hold = part["next_close"].iloc[-1] / part["close"].iloc[0] - 1
    print(f"\n=== {name.upper()} ===")
    print("Rows:", len(part), "| From", part.index.min().date(), "to", part.index.max().date())
    print("Mean daily return: %.5f | Std: %.5f" % (y.mean(), y.std()))
    print("Share of up days: %.3f" % (y > 0).mean())
    print("Close range: %.0f to %.0f" % (part["close"].min(), part["close"].max()))
    print("Index return over the period (buy and hold): %.1f%%" % (buy_hold * 100))
    print("Mean vol_21: %.5f" % part["vol_21"].mean())


def main():
    data = pd.read_csv(config.DATASET_PATH, index_col="Date", parse_dates=True)
    train, val, test = get_splits(data)

    for name, part, path in [("train", train, config.TRAIN_PATH),
                             ("validation", val, config.VAL_PATH),
                             ("test", test, config.TEST_PATH)]:
        part.to_csv(path)
        describe_split(name, part)

    print("\n=== ORDERING ===")
    print("Train ends before validation starts:", train.index.max() < val.index.min())
    print("Validation ends before test starts:", val.index.max() < test.index.min())
    print("Rows add up:", len(train) + len(val) + len(test) == len(data), f"({len(data)} total)")


if __name__ == "__main__":
    main()