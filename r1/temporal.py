"""Nested selection and training for the temporal family.

GPU-resident windows: features (causally filtered, standardised on the training participants of the fit) are kept
on the GPU; a window ending at row r is X[r-L+1 : r+1] (always inside one segment because only rows with
t >= L-1 are used). Training rows use a fixed stride; early stopping uses a dedicated participant; scoring uses
the evaluation grid. One fresh Optuna study per (dataset, model, outer fold); inner grouped 3-fold CV."""
import copy
import math

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn

from .cv import inner_folds, per_participant_rmse, run_seed, selection_score
from .preprocess import CUTOFF_GRID, FoldScaler
from .temporal_models import (GRUModel, LSTMModel, PatchTSTModel, TCNModel, TFTModel, patch_count,
                              tcn_receptive_field)

MODELS = ["LSTM", "GRU", "TCN", "TFT", "PATCHTST"]
WINDOWS = [30, 60, 90, 120, 180, 240]
CUTOFF_CHOICES = [str(c) for c in CUTOFF_GRID]
# cut rules (a)+(b) applied uniformly to all temporal models and datasets.
SETTINGS = {"trials": 25, "stride": 8, "max_epochs": 60, "patience": 10, "inner_k": 3, "bf16": True,
            "lr_patience": 7}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # speed only


def _cutoff(c):
    return None if c == "None" else float(c)


def suggest(trial, model):
    p = {"cutoff": trial.suggest_categorical("cutoff", CUTOFF_CHOICES),
         "L": trial.suggest_categorical("L", WINDOWS)}
    if model in ("LSTM", "GRU"):
        p.update(hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 96, 128]),
                 num_layers=trial.suggest_int("num_layers", 1, 3),
                 dropout=trial.suggest_float("dropout", 0.1, 0.5),
                 lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                 batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                 weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    elif model == "TCN":
        p.update(kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
                 num_levels=trial.suggest_int("num_levels", 4, 9),
                 base_channels=trial.suggest_categorical("base_channels", [32, 48, 64]),
                 dropout=trial.suggest_float("dropout", 0.0, 0.5),
                 lr=trial.suggest_float("lr", 5e-5, 1e-3, log=True),
                 batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                 weight_decay=0.0)
        # receptive field >= L: raise levels to the smallest valid value (every trial valid)
        while tcn_receptive_field(p["num_levels"], p["kernel_size"]) < p["L"]:
            p["num_levels"] += 1
    elif model == "TFT":
        p.update(hidden_size=trial.suggest_categorical("hidden_size", [32, 48, 64, 96]),
                 num_heads=trial.suggest_categorical("num_heads", [2, 4]),
                 dropout=trial.suggest_float("dropout", 0.1, 0.5),
                 lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                 batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                 weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
    else:
        p.update(patch_len=trial.suggest_categorical("patch_len", [12, 16, 20, 24]),
                 d_model=trial.suggest_categorical("d_model", [32, 48, 64, 96, 128]),
                 nhead=trial.suggest_categorical("nhead", [4, 8]),
                 num_layers=trial.suggest_int("num_layers", 2, 4),
                 channel_mode=trial.suggest_categorical("channel_mode", ["CI", "CD"]),
                 dropout=trial.suggest_float("dropout", 0.1, 0.3),
                 lr=trial.suggest_float("lr", 1e-4, 1e-3, log=True),
                 batch_size=trial.suggest_categorical("batch_size", [32, 64]),
                 weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
        # >= 2 patches: lower the patch length to the largest valid value (every trial valid)
        valid = [q for q in [12, 16, 20, 24] if patch_count(p["L"], q)[0] >= 2]
        if p["patch_len"] not in valid:
            p["patch_len"] = max(valid)
    return p


def build(model, p, n_features):
    if model == "LSTM":
        return LSTMModel(n_features, p["hidden_size"], p["num_layers"], p["dropout"])
    if model == "GRU":
        return GRUModel(n_features, p["hidden_size"], p["num_layers"], p["dropout"])
    if model == "TCN":
        return TCNModel(n_features, p["num_levels"], p["base_channels"], p["kernel_size"], p["dropout"])
    if model == "TFT":
        return TFTModel(n_features, p["hidden_size"], p["num_heads"], p["dropout"])
    return PatchTSTModel(n_features, p["L"], p["patch_len"], p["d_model"], p["nhead"], p["num_layers"],
                         p["dropout"], p["channel_mode"])


class _Windows:
    """GPU tensors for one fit: standardised X and y, plus window gathering by end row."""

    def __init__(self, X, y, train_rows):
        self.sx = FoldScaler().fit(X[train_rows])
        self.sy = FoldScaler().fit(y[train_rows].reshape(-1, 1))
        self.X = torch.tensor(self.sx.transform(X), dtype=torch.float32, device=DEVICE)
        self.y = torch.tensor(self.sy.transform(y.reshape(-1, 1)).ravel(), dtype=torch.float32, device=DEVICE)

    def batch(self, ends, L):
        idx = ends[:, None] + torch.arange(-L + 1, 1, device=DEVICE)[None, :]
        return self.X[idx], self.y[ends]


def _predict(net, W, ends, L, bs=512):  # smaller kernels: avoids Windows GPU timeouts (TDR)
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=SETTINGS["bf16"] and DEVICE.type == "cuda"):
        for i in range(0, len(ends), bs):
            xb, _ = W.batch(ends[i:i + bs], L)
            out.append(net(xb).float())
    return torch.cat(out) if out else torch.empty(0, device=DEVICE)


def train_and_predict(model, p, X, y, df, train_pids, es_pid, score_rows, seed):
    """Train on train_pids minus es_pid (early stopping on es_pid's evaluation-grid rows); return predictions
    (original units) for score_rows. Returns (y_pred, epochs_run) or (None, epochs) if training diverged."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    L = p["L"]
    pid, t = df["pid"].to_numpy(), df["t"].to_numpy()
    fit_pids = [q for q in train_pids if q != es_pid]
    fit_rows = np.isin(pid, train_pids)  # scaler fitted on all training participants incl. the ES participant
    W = _Windows(X, y, fit_rows)
    tr_ends = np.flatnonzero(np.isin(pid, fit_pids) & (t >= L - 1))
    tr_ends = tr_ends[(t[tr_ends] - (L - 1)) % SETTINGS["stride"] == 0]
    es_ends = np.flatnonzero((pid == es_pid) & (t >= 240))
    tr_ends = torch.tensor(tr_ends, device=DEVICE)
    es_ends_t = torch.tensor(es_ends, device=DEVICE)
    sc_ends = torch.tensor(np.asarray(score_rows), device=DEVICE)
    net = build(model, p, X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=SETTINGS["lr_patience"], min_lr=1e-6)
    crit = nn.MSELoss()
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best, best_state, wait, ep = math.inf, None, 0, 0
    use_bf16 = SETTINGS["bf16"] and DEVICE.type == "cuda"
    for ep in range(1, SETTINGS["max_epochs"] + 1):
        net.train()
        perm = tr_ends[torch.randperm(len(tr_ends), device=DEVICE, generator=gen)]
        for i in range(0, len(perm), p["batch_size"]):
            xb, yb = W.batch(perm[i:i + p["batch_size"]], L)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = net(xb)
            loss = crit(out.float(), yb)
            if not torch.isfinite(loss):
                return None, ep
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        val = crit(_predict(net, W, es_ends_t, L), W.y[es_ends_t]).item()
        if not math.isfinite(val):
            return None, ep
        sched.step(val)
        if val < best:
            best, best_state, wait = val, copy.deepcopy(net.state_dict()), 0
        else:
            wait += 1
            if wait >= SETTINGS["patience"]:
                break
    net.load_state_dict(best_state)
    yp = _predict(net, W, sc_ends, L).cpu().numpy()
    return W.sy.inverse(yp.reshape(-1, 1)).ravel(), ep


def _run_with_retries(model, p, X, y, df, train_pids, es_pid, rows, seed_parts):
    """re-run a failed/diverged training with the next retry index, at most twice."""
    for retry in range(3):
        yp, ep = train_and_predict(model, p, X, y, df, train_pids, es_pid, rows, run_seed(*seed_parts, retry))
        if yp is not None:
            return yp, ep, retry
    return None, ep, 3


def nested_outer_fold(panel, model, outer_pid, filtered_cache, n_trials=None, seeds=(0, 1, 2), log=print, study_dir=None):
    df = panel.df
    pid = df["pid"].to_numpy()
    y = df["y"].to_numpy(float)
    grid = panel.eval_mask()
    train_pids = sorted(int(q) for q in panel.pids if q != outer_pid)
    folds = inner_folds(train_pids, SETTINGS["inner_k"])
    events = []

    def _log_event(e):
        e = {"trial": e["trial"], "fold": e["fold"], "seed": e.get("seed", ""), "epochs": e["epochs"], "retries": e["retries"]}
        events.append(e)
        if study_dir is not None:
            pd.DataFrame([e]).to_csv(study_dir / "training_events.csv", mode="a", index=False,
                                     header=not (study_dir / "training_events.csv").exists())

    def objective(trial):
        p = suggest(trial, model)
        X = filtered_cache[_cutoff(p["cutoff"])]
        held = {}
        for fi, f in enumerate(folds):
            inner_train = [q for q in train_pids if q not in set(f.tolist())]
            es = inner_train[0]  # first (sorted) inner-training participant
            rows = np.flatnonzero(np.isin(pid, f) & grid)
            yp, ep, retry = _run_with_retries(model, p, X, y, df, inner_train, es, rows,
                                              (panel.name, model, outer_pid, fi, trial.number, 0))
            _log_event({"trial": trial.number, "fold": fi, "epochs": ep, "retries": retry})
            if yp is None:
                raise optuna.TrialPruned()
            held.update(per_participant_rmse(y[rows], yp, pid[rows]).to_dict())
            trial.report(selection_score(held), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("resolved_params", p)
        return selection_score(held)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    budget = n_trials or SETTINGS["trials"]
    storage = f"sqlite:///{(study_dir / 'study.db').as_posix()}" if study_dir is not None else None
    study = optuna.create_study(direction="minimize", storage=storage, study_name="study", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10, seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    if len(finished) < budget:  # resume an interrupted study (orchestration only)
        study.optimize(objective, n_trials=budget - len(finished))
    trials = study.trials_dataframe()
    done = trials[trials["state"] == "COMPLETE"].sort_values("number")
    best_so_far = pd.DataFrame({"trial": done["number"], "best_so_far": done["value"].cummin()})
    ranked = sorted([tr for tr in study.trials if tr.state == optuna.trial.TrialState.COMPLETE], key=lambda tr: tr.value)
    X_rows = np.flatnonzero((pid == outer_pid) & grid)
    es_outer = train_pids[list(panel.pids).index(outer_pid) % len(train_pids)]
    preds, chosen = {}, None
    for tr in ranked:  # next-best inner configuration if the refit keeps failing
        p = tr.user_attrs["resolved_params"]
        X = filtered_cache[_cutoff(p["cutoff"])]
        preds = {}
        for s in seeds:
            yp, ep, retry = _run_with_retries(model, p, X, y, df, train_pids, es_outer, X_rows,
                                              (panel.name, model, outer_pid, "refit", -1, s))
            _log_event({"trial": f"refit{tr.number}", "fold": "outer", "seed": s, "epochs": ep, "retries": retry})
            if yp is None:
                break
            preds[s] = yp
        if len(preds) == len(seeds):
            chosen = dict(p, trial=tr.number, inner_score=float(tr.value))
            break
    chosen["n_complete"] = int((trials["state"] == "COMPLETE").sum())
    chosen["n_pruned"] = int((trials["state"] == "PRUNED").sum())
    chosen["es_outer"] = int(es_outer)
    return chosen, trials, best_so_far, pd.DataFrame(events), X_rows, preds
