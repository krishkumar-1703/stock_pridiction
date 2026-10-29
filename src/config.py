from pathlib import Path

# ---- Paths (work no matter where you run the script from) ----
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ---- Data settings ----
TICKER = "^NSEI"              # Nifty 50 index. Alternatives: "SPY", "RELIANCE.NS"
START_DATE = "2010-01-01"
END_DATE = "2026-10-09"       # last day we WANT included (frozen snapshot)

# ---- Reproducibility ----
RANDOM_SEED = 42

def snapshot_path():
    safe = TICKER.replace("^", "").replace(".", "_")
    return RAW_DIR / f"{safe}_snapshot.csv"


# ---- Target / column names ----
TARGET_COL = "target_ret"
NON_FEATURE_COLS = ["close", "next_close", "target_ret"]   # kept for evaluation only, never model inputs
DATASET_PATH = PROCESSED_DIR / "dataset.csv"

# ---- Chronological split (by date order, never random) ----
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15          # the remaining 15% is the test set
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
# ---- Baselines ----
MA_WINDOWS = [5, 10, 20, 50]

# ---- Random Forest ----
RF_DEPTHS = [3, 5, 8]
RF_MIN_LEAFS = [20, 50, 100]
RF_SEEDS = [42, 43, 44, 45, 46]     # the first seed is the main model

# ---- LSTM ----
LOOKBACK = 20
LSTM_HIDDEN = 32
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_WEIGHT_DECAY = 1e-4
LSTM_BATCH = 64
LSTM_MAX_EPOCHS = 100
LSTM_PATIENCE = 10
LSTM_SEEDS = [42, 43, 44, 45, 46]     # the first seed is the main model
LSTM_PATH = MODELS_DIR / "lstm.pt"