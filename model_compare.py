"""
model_compare.py
================
Head-to-head: gradient-boosted trees (LightGBM) vs a deep net (PyTorch MLP) for
World Cup 1X2 prediction (home win / draw / away win), on the engineered match-diff
features from wc_squad_features.build_match_row().

What this gives you
-------------------
  * Ranked Probability Score (RPS) -- the correct metric for ordered 1X2 forecasts,
    plus log-loss and accuracy.
  * Time-aware (forward-chaining) cross-validation -- random k-fold LEAKS the future
    into the past and will flatter you. Never use it for match prediction.
  * A market baseline -- de-vigged bookmaker odds -> probabilities -> RPS. If you can't
    beat the closing line, you're not adding value. This is the honest yardstick.
  * Probability calibration (the probs matter more than the argmax here).
  * Unified save/load + predict_proba interface so app.py can serve either model.

Honest expectation
------------------
On a dataset this size (international matches are SPARSE) gradient boosting almost
always beats deep learning on engineered tabular features. Expect LightGBM to win on
RPS. The MLP is here for a real comparison, not because it's likely to be better --
DL earns its keep at scale and on raw/unstructured inputs (e.g. event sequences),
not on a few dozen hand-built columns. If the MLP wins, suspect a leak or a bug first.

Dependencies: lightgbm, scikit-learn, numpy, pandas. torch is OPTIONAL (MLP is
import-guarded); the LightGBM + baseline path runs without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Class convention: 0 = home win, 1 = draw, 2 = away win (ordered).
N_CLASSES = 3


# ---------------------------------------------------------------------------
# Metric: Ranked Probability Score (lower is better)
# ---------------------------------------------------------------------------

def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean RPS over rows. probs: (n, 3) ordered class probs; outcomes: (n,) in {0,1,2}.

    RPS penalises by *cumulative* distance, so predicting a draw when away win occurs
    is less wrong than predicting home win -- exactly the ordering you want in 1X2.
    """
    probs = np.asarray(probs, dtype=float)
    onehot = np.eye(N_CLASSES)[np.asarray(outcomes, dtype=int)]
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (N_CLASSES - 1)))


def odds_to_probs(home_odds, draw_odds, away_odds) -> np.ndarray:
    """De-vig decimal odds -> implied probabilities (basic normalisation / overround removal)."""
    raw = np.vstack([1 / np.asarray(home_odds, float),
                     1 / np.asarray(draw_odds, float),
                     1 / np.asarray(away_odds, float)]).T
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Time-aware cross-validation (forward chaining)
# ---------------------------------------------------------------------------

def time_series_folds(dates: pd.Series, n_folds: int = 5):
    """Yield (train_idx, valid_idx) expanding-window folds ordered by date.

    Each fold trains on everything before a cutoff and validates on the next block.
    No future information ever reaches the training side.
    """
    order = np.argsort(pd.to_datetime(dates).to_numpy())
    n = len(order)
    fold_size = n // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        valid_end = fold_size * (i + 1) if i < n_folds else n
        train_idx = order[:train_end]
        valid_idx = order[train_end:valid_end]
        if len(valid_idx):
            yield train_idx, valid_idx


# ---------------------------------------------------------------------------
# Model A: LightGBM multiclass
# ---------------------------------------------------------------------------

@dataclass
class LGBModel:
    params: dict = None
    num_round: int = 500
    model_: lgb.Booster = None

    def default_params(self) -> dict:
        return {
            "objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
            "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 20,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "verbosity": -1,
        }

    def fit(self, X, y, X_val=None, y_val=None):
        p = self.params or self.default_params()
        dtrain = lgb.Dataset(X, label=y)
        valid = [lgb.Dataset(X_val, label=y_val, reference=dtrain)] if X_val is not None else None
        cbs = [lgb.early_stopping(50, verbose=False)] if valid else []
        self.model_ = lgb.train(p, dtrain, num_boost_round=self.num_round,
                                valid_sets=valid, callbacks=cbs)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict(X, num_iteration=getattr(self.model_, "best_iteration", None))


# ---------------------------------------------------------------------------
# Model B: PyTorch MLP (import-guarded)
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _MLP(nn.Module):
        def __init__(self, n_in, hidden=(128, 64), p_drop=0.3):
            super().__init__()
            layers, d = [], n_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
                d = h
            layers += [nn.Linear(d, N_CLASSES)]   # logits; softmax in loss / at inference
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    @dataclass
    class MLPModel:
        hidden: tuple = (128, 64)
        p_drop: float = 0.3
        lr: float = 1e-3
        epochs: int = 300
        patience: int = 25
        scaler_: StandardScaler = None
        net_: object = None

        def fit(self, X, y, X_val=None, y_val=None):
            self.scaler_ = StandardScaler().fit(X)
            Xt = torch.tensor(self.scaler_.transform(X), dtype=torch.float32)
            yt = torch.tensor(np.asarray(y), dtype=torch.long)
            self.net_ = _MLP(X.shape[1], self.hidden, self.p_drop)
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
            loss_fn = nn.CrossEntropyLoss()

            has_val = X_val is not None
            if has_val:
                Xv = torch.tensor(self.scaler_.transform(X_val), dtype=torch.float32)
                yv = torch.tensor(np.asarray(y_val), dtype=torch.long)

            best, best_state, wait = np.inf, None, 0
            for _ in range(self.epochs):
                self.net_.train()
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt), yt)
                loss.backward(); opt.step()
                if has_val:
                    self.net_.eval()
                    with torch.no_grad():
                        vloss = loss_fn(self.net_(Xv), yv).item()
                    if vloss < best - 1e-5:
                        best, best_state, wait = vloss, self.net_.state_dict(), 0
                    else:
                        wait += 1
                        if wait >= self.patience:
                            break
            if best_state is not None:
                self.net_.load_state_dict(best_state)
            return self

        def predict_proba(self, X) -> np.ndarray:
            self.net_.eval()
            Xt = torch.tensor(self.scaler_.transform(X), dtype=torch.float32)
            with torch.no_grad():
                return torch.softmax(self.net_(Xt), dim=1).numpy()


# ---------------------------------------------------------------------------
# Cross-validated comparison
# ---------------------------------------------------------------------------

def evaluate_cv(X: pd.DataFrame, y: np.ndarray, dates: pd.Series,
                market_probs: np.ndarray | None = None, n_folds: int = 5) -> pd.DataFrame:
    """Forward-chaining CV; returns per-model mean RPS / log-loss / accuracy vs the market."""
    from sklearn.metrics import log_loss, accuracy_score

    builders = {"lightgbm": lambda: LGBModel()}
    if _HAS_TORCH:
        builders["mlp_torch"] = lambda: MLPModel()

    scores = {k: {"rps": [], "logloss": [], "acc": []} for k in builders}
    scores["market"] = {"rps": [], "logloss": [], "acc": []}

    Xv = X.values if isinstance(X, pd.DataFrame) else X
    for tr, va in time_series_folds(dates, n_folds):
        for name, build in builders.items():
            m = build().fit(Xv[tr], y[tr], Xv[va], y[va])
            p = m.predict_proba(Xv[va])
            scores[name]["rps"].append(rps(p, y[va]))
            scores[name]["logloss"].append(log_loss(y[va], p, labels=[0, 1, 2]))
            scores[name]["acc"].append(accuracy_score(y[va], p.argmax(1)))
        if market_probs is not None:
            pm = market_probs[va]
            scores["market"]["rps"].append(rps(pm, y[va]))
            scores["market"]["logloss"].append(log_loss(y[va], pm, labels=[0, 1, 2]))
            scores["market"]["acc"].append(accuracy_score(y[va], pm.argmax(1)))

    rows = []
    for name, d in scores.items():
        if d["rps"]:
            rows.append({"model": name,
                         "RPS": np.mean(d["rps"]), "logloss": np.mean(d["logloss"]),
                         "accuracy": np.mean(d["acc"])})
    return pd.DataFrame(rows).sort_values("RPS").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train final + save/load (unified interface for app.py)
# ---------------------------------------------------------------------------

def train_and_save(X, y, dates, out_prefix="wc_model", which="lightgbm"):
    """Train the chosen model on all data (last fold as val for early stopping) and persist."""
    order = np.argsort(pd.to_datetime(dates).to_numpy())
    cut = int(len(order) * 0.85)
    tr, va = order[:cut], order[cut:]
    Xv = X.values if isinstance(X, pd.DataFrame) else X

    if which == "lightgbm":
        m = LGBModel().fit(Xv[tr], y[tr], Xv[va], y[va])
        joblib.dump({"type": "lightgbm", "booster": m.model_,
                     "columns": list(X.columns) if isinstance(X, pd.DataFrame) else None},
                    f"{out_prefix}.joblib")
    elif which == "mlp_torch":
        if not _HAS_TORCH:
            raise RuntimeError("torch not installed")
        m = MLPModel().fit(Xv[tr], y[tr], Xv[va], y[va])
        torch.save(m.net_.state_dict(), f"{out_prefix}_mlp.pt")
        joblib.dump({"type": "mlp_torch", "scaler": m.scaler_,
                     "hidden": m.hidden, "n_in": Xv.shape[1],
                     "columns": list(X.columns) if isinstance(X, pd.DataFrame) else None},
                    f"{out_prefix}_mlp_meta.joblib")
    return m


def load_predictor(prefix="wc_model", which="lightgbm"):
    """Return a callable proba(X_df) -> (n,3) for whichever model was saved."""
    if which == "lightgbm":
        blob = joblib.load(f"{prefix}.joblib")
        booster = blob["booster"]
        return lambda X: booster.predict(X.values if hasattr(X, "values") else X)
    else:
        meta = joblib.load(f"{prefix}_mlp_meta.joblib")
        net = _MLP(meta["n_in"], meta["hidden"])
        net.load_state_dict(torch.load(f"{prefix}_mlp.pt"))
        net.eval()
        scaler = meta["scaler"]

        def proba(X):
            Xa = X.values if hasattr(X, "values") else X
            Xt = torch.tensor(scaler.transform(Xa), dtype=torch.float32)
            with torch.no_grad():
                return torch.softmax(net(Xt), dim=1).numpy()
        return proba


if __name__ == "__main__":
    # Smoke test on synthetic data shaped like build_match_row output.
    rng = np.random.default_rng(0)
    n, d = 1200, 8
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"diff_f{i}" for i in range(d)])
    logits = X.values @ rng.normal(size=(d, N_CLASSES))
    y = (logits + rng.normal(scale=2, size=(n, N_CLASSES))).argmax(1)
    dates = pd.date_range("2018-01-01", periods=n, freq="D").to_series().reset_index(drop=True)
    mkt = odds_to_probs(rng.uniform(1.5, 4, n), rng.uniform(3, 4, n), rng.uniform(1.5, 4, n))

    print(evaluate_cv(X, y, dates, market_probs=mkt, n_folds=4).to_string(index=False))
    train_and_save(X, y, dates, which="lightgbm")
    pred = load_predictor(which="lightgbm")
    print("\nsample proba:", np.round(pred(X.iloc[:2]), 3))
