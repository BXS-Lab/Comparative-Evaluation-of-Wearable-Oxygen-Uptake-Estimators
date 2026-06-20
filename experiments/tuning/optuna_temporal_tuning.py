"""
Phase 2: Optuna Hyperparameter Tuning for Temporal Models

For each model, with filter_hz and seq_length FIXED from Phase 1 (best_configs.json),
tunes model architecture and training HPs using full 16-fold LOPO.

Usage:
    python optuna_temporal_tuning.py --model lstm --output ../../results/optuna_temporal
    python optuna_temporal_tuning.py --model gru --output ../../results/optuna_temporal
    python optuna_temporal_tuning.py --model tcn --output ../../results/optuna_temporal
    python optuna_temporal_tuning.py --model tft --output ../../results/optuna_temporal
    python optuna_temporal_tuning.py --model patchtst --output ../../results/optuna_temporal
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Data helpers (same as filter_seqlen_sweep.py)
# ---------------------------------------------------------------------------

def butter_lowpass(data, cutoff_hz, fs=1.0, order=2):
    nyq = 0.5 * fs
    nc = cutoff_hz / nyq
    if nc >= 1.0 or nc <= 0.0:
        return data.copy()
    b, a = butter(order, nc, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~(df == -1).any(axis=1)].reset_index(drop=True)
    pidx = df["ID"].to_numpy(int)
    weight = df["Weight"].to_numpy(float)
    vo2 = df["VO2"].to_numpy(float) / weight
    participants = np.unique(pidx)
    N = len(df)
    X = np.zeros((N, 6), dtype=float)
    X[:, 0] = df["BF"].values
    X[:, 1] = df["VE"].values
    X[:, 2] = df["ACC"].values
    X[:, 3] = df["HR"].values
    X[:, 4] = df["CAD"].values
    for pid in participants:
        idx = pidx == pid
        hr_pi = X[idx, 3]
        dhr = np.zeros_like(hr_pi)
        if len(hr_pi) > 1:
            dhr[1:] = np.diff(hr_pi)
        X[idx, 5] = dhr
    return pidx, X, vo2, participants


def apply_lowpass(X, pidx, participants, cutoff_hz):
    if cutoff_hz is None:
        return X.copy()
    X_out = X.copy()
    for pid in participants:
        idx = pidx == pid
        pdata = X_out[idx]
        mu = pdata.mean(axis=0, keepdims=True)
        X_out[idx] = butter_lowpass(pdata - mu, cutoff_hz) + mu
    return X_out


def create_sequences(X, y, pidx, seq_length):
    participants = np.unique(pidx)
    X_seqs, y_seqs, pidx_seqs = [], [], []
    for pid in participants:
        idx = np.where(pidx == pid)[0]
        Xp, yp = X[idx], y[idx]
        for i in range(seq_length, len(Xp)):
            X_seqs.append(Xp[i - seq_length : i])
            y_seqs.append(yp[i - 1])
            pidx_seqs.append(pid)
    return np.array(X_seqs), np.array(y_seqs), np.array(pidx_seqs)


class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32),
                                nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32),
                                nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
    def forward(self, x):
        return x[:, :, :-self.s].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_out, n_out, k, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1, nn.ReLU(), nn.Dropout(dropout),
                                 self.conv2, self.chomp2, nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size=6, num_channels=None, kernel_size=3, dropout=0.3):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 128, 128, 128]
        layers = []
        for i, outc in enumerate(num_channels):
            inc = input_size if i == 0 else num_channels[i - 1]
            d = 2 ** i
            layers.append(TemporalBlock(inc, outc, kernel_size, 1, d, (kernel_size - 1) * d, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(num_channels[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        return self.fc(self.network(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.var_sel = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_size, input_size), nn.Softmax(dim=-1))
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.grn = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout))
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        w = self.var_sel(x)
        h, _ = self.lstm(self.embed(x * w))
        a, _ = self.attn(h, h, h)
        return self.fc(self.ln(a + self.grn(a))[:, -1, :]).squeeze(-1)


class PatchTSTModel_CD(nn.Module):
    """Channel-Dependent (current) PatchTST — flattens all channels per patch."""
    def __init__(self, input_size=6, seq_len=120, patch_len=24, stride=12,
                 d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Sequential(nn.Linear(patch_len * input_size, d_model),
                                         nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        B = x.shape[0]
        patches = [x[:, i:i+self.patch_len, :].reshape(B, -1)
                   for i in range(0, x.shape[1] - self.patch_len + 1, self.stride)]
        x = self.patch_embed(torch.stack(patches, 1)) + self.pos_embed
        return self.fc(self.transformer(x).mean(1)).squeeze(-1)


class PatchTSTModel_CI(nn.Module):
    """Channel-Independent PatchTST — processes each channel through shared transformer."""
    def __init__(self, input_size=6, seq_len=120, patch_len=24, stride=12,
                 d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Sequential(nn.Linear(patch_len, d_model),
                                         nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        B, L, C = x.shape
        # Process each channel independently: (B, L, C) -> (B*C, L, 1)
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)
        patches = [x[:, i:i+self.patch_len, :].reshape(B * C, -1)
                   for i in range(0, L - self.patch_len + 1, self.stride)]
        x = self.patch_embed(torch.stack(patches, 1)) + self.pos_embed
        out = self.transformer(x).mean(1)  # (B*C, d_model)
        out = out.reshape(B, C, -1).mean(1)  # (B, d_model) — avg across channels
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tcn_channels(seq_length, kernel_size=3, base_ch=32, max_ch=128):
    needed = (seq_length - 1) / (2 * (kernel_size - 1))
    num_levels = max(3, math.ceil(math.log2(needed + 1)))
    return [min(max_ch, base_ch * (2 ** min(i, 2))) for i in range(num_levels)]


def tcn_rf(num_levels, kernel_size=3):
    return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)


def train_model(model, tr_loader, va_loader, epochs=100, lr=0.001, patience=15, wd=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.5, patience=7, min_lr=1e-6)
    crit = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        vl /= max(1, len(va_loader))
        sched.step(vl)
        if vl < best_loss:
            best_loss, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    loader = DataLoader(TSDataset(X, np.zeros(len(X))), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Suggest HPs per model
# ---------------------------------------------------------------------------

def suggest_params(trial, model_type):
    if model_type == "lstm":
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 96, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
    elif model_type == "gru":
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 96, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
    elif model_type == "tcn":
        return {
            "num_levels": trial.suggest_int("num_levels", 4, 9),
            "base_channels": trial.suggest_categorical("base_channels", [32, 48, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            # NO weight_decay for TCN (conflicts with weight_norm)
        }
    elif model_type == "tft":
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 48, 64, 96]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
    elif model_type == "patchtst":
        return {
            "d_model": trial.suggest_categorical("d_model", [32, 48, 64, 96, 128]),
            "nhead": trial.suggest_categorical("nhead", [4, 8]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "patch_len": trial.suggest_categorical("patch_len", [12, 16, 20, 24]),
            "channel_mode": trial.suggest_categorical("channel_mode", ["CI", "CD"]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
    else:
        raise ValueError(f"Unknown model: {model_type}")


def build_model_from_params(model_type, params, seq_length):
    if model_type == "lstm":
        return LSTMModel(6, params["hidden_size"], params["num_layers"], params["dropout"])
    elif model_type == "gru":
        return GRUModel(6, params["hidden_size"], params["num_layers"], params["dropout"])
    elif model_type == "tcn":
        nl = params["num_levels"]
        bc = params["base_channels"]
        ks = params["kernel_size"]
        # Check RF
        rf = tcn_rf(nl, ks)
        if rf < seq_length:
            return None  # invalid — will be caught by caller
        channels = [min(128, bc * (2 ** min(i, 2))) for i in range(nl)]
        return TCNModel(6, channels, ks, params["dropout"])
    elif model_type == "tft":
        hs = params["hidden_size"]
        nh = params["num_heads"]
        if hs % nh != 0:
            return None
        return TFTModel(6, hs, nh, params["dropout"])
    elif model_type == "patchtst":
        dm = params["d_model"]
        nh = params["nhead"]
        pl = params["patch_len"]
        st = max(4, pl // 2)
        num_patches = (seq_length - pl) // st + 1
        if dm % nh != 0 or num_patches < 2:
            return None
        if params["channel_mode"] == "CI":
            return PatchTSTModel_CI(6, seq_length, pl, st, dm, nh, params["num_layers"], params["dropout"])
        else:
            return PatchTSTModel_CD(6, seq_length, pl, st, dm, nh, params["num_layers"], params["dropout"])
    return None


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

# Global cache for filtered + sequenced data
_data_cache = {}


def get_data(input_csv, filter_hz, seq_length):
    key = (input_csv, filter_hz, seq_length)
    if key not in _data_cache:
        pidx, X_raw, y, participants = load_raw(input_csv)
        X_filt = apply_lowpass(X_raw, pidx, participants, filter_hz)
        X_seq, y_seq, pidx_seq = create_sequences(X_filt, y, pidx, seq_length)
        _data_cache[key] = (X_seq, y_seq, pidx_seq, np.unique(pidx_seq))
    return _data_cache[key]


def objective(trial, model_type, filter_hz, seq_length, input_csv):
    params = suggest_params(trial, model_type)

    model_template = build_model_from_params(model_type, params, seq_length)
    if model_template is None:
        return float("inf")  # invalid config

    X_seq, y_seq, pidx_seq, participants = get_data(input_csv, filter_hz, seq_length)

    batch_size = params["batch_size"]
    lr = params["lr"]
    wd = params.get("weight_decay", 0.0)

    rmses = []
    for fold_idx, test_pid in enumerate(participants):
        train_mask = pidx_seq != test_pid
        test_mask = ~train_mask

        train_pids = participants[participants != test_pid]
        val_pid = train_pids[fold_idx % len(train_pids)]
        val_mask = (pidx_seq == val_pid) & train_mask
        actual_train = train_mask & ~val_mask

        X_tr, y_tr = X_seq[actual_train], y_seq[actual_train]
        X_va, y_va = X_seq[val_mask], y_seq[val_mask]
        X_te, y_te = X_seq[test_mask], y_seq[test_mask]

        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_tr_n = sc_X.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        y_tr_n = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
        X_va_n = sc_X.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)
        y_va_n = sc_y.transform(y_va.reshape(-1, 1)).ravel()
        X_te_n = sc_X.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

        model = build_model_from_params(model_type, params, seq_length)
        tr_loader = DataLoader(TSDataset(X_tr_n, y_tr_n), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(TSDataset(X_va_n, y_va_n), batch_size=batch_size, shuffle=False)

        model = train_model(model, tr_loader, va_loader, epochs=100, lr=lr, patience=15, wd=wd)
        y_pred = sc_y.inverse_transform(predict(model, X_te_n).reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        rmses.append(rmse)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Report for pruning
        trial.report(np.mean(rmses), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(rmses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["lstm", "gru", "tcn", "tft", "patchtst"])
    parser.add_argument("--input", default="../../data/TB-File-01.csv")
    parser.add_argument("--output", default="../../results/optuna_temporal")
    parser.add_argument("--best-configs", default="../../results/filter_seqlen_sweep/best_configs.json")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    model_type = args.model
    out_dir = os.path.join(args.output, model_type.upper())
    os.makedirs(out_dir, exist_ok=True)

    # Load Phase 1 best configs
    with open(args.best_configs) as f:
        best_configs = json.load(f)

    cfg = best_configs[model_type]
    filter_hz = cfg["filter_hz"]
    seq_length = cfg["seq_length"]

    n_trials = args.n_trials
    if n_trials is None:
        n_trials = 100 if model_type in ["patchtst", "tft"] else 75

    print(f"{'='*70}")
    print(f"Optuna Tuning: {model_type.upper()}")
    print(f"  filter_hz={filter_hz}, seq_length={seq_length}")
    print(f"  n_trials={n_trials}")
    print(f"  device={DEVICE}")
    print(f"{'='*70}")

    db_path = os.path.join(out_dir, "study.db")
    study = optuna.create_study(
        study_name=f"vo2_{model_type}_optimization",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10, seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    obj_fn = lambda trial: objective(trial, model_type, filter_hz, seq_length, args.input)
    study.optimize(obj_fn, n_trials=n_trials)

    # Save results
    best = study.best_trial
    print(f"\nBest trial #{best.number}: RMSE={best.value:.4f}")
    print(f"  Params: {best.params}")

    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"params": best.params, "rmse": best.value,
                    "filter_hz": filter_hz, "seq_length": seq_length}, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(out_dir, "all_trials.csv"), index=False)

    # Plots
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "optimization_history.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "param_importances.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
