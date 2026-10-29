import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                          # (batch, lookback, hidden)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)   # use the last day only


def make_windows(X, positions, lookback):
    """For each row position p, return rows p-lookback+1 ... p (the last row is day p itself)."""
    return np.stack([X[p - lookback + 1: p + 1] for p in positions]).astype(np.float32)


def predict_returns(model, X, y_mean, y_std):
    """Run the model and convert its scaled output back into real next-day returns."""
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(X)).numpy()
    return scaled.astype(float) * y_std + y_mean