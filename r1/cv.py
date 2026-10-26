"""Fold construction, scoring and result I/O shared by all model families."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results_r1"


def inner_folds(train_pids, k):
    """Grouped k-fold over participants: sorted order, contiguous nearly-equal chunks, no shuffling."""
    return [np.asarray(c) for c in np.array_split(np.sort(np.asarray(train_pids)), k) if len(c)]


def per_participant_rmse(y_true, y_pred, pid):
    err2 = (np.asarray(y_pred) - np.asarray(y_true)) ** 2
    return pd.Series(err2).groupby(np.asarray(pid)).mean() ** 0.5


def selection_score(rmse_by_pid):
    """Mean over participants of their held-out RMSE (each participant held out exactly once)."""
    return float(np.mean(list(rmse_by_pid.values())))


def run_seed(*parts):
    """Deterministic 32-bit seed from (dataset, model, outer fold, inner fold|'refit', trial, seed index, retry)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def outer_dir(dataset, model, outer_pid, root=RESULTS_ROOT):
    d = Path(root) / dataset / model / f"outer_{int(outer_pid):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def save_predictions(path, panel_df, rows, y_pred):
    sub = panel_df.iloc[rows]
    pd.DataFrame({"row": rows, "pid": sub["pid"].to_numpy(), "seg": sub["seg"].to_numpy(),
                  "t": sub["t"].to_numpy(), "y_true": sub["y"].to_numpy(), "y_pred": np.asarray(y_pred)}
                 ).to_csv(path, index=False)
