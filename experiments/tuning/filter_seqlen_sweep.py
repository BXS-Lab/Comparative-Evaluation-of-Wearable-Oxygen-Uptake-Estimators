"""
Phase 1: Filter Cutoff + Sequence Length Sweep for Temporal Models

Three-stage separable design:
  Stage A: Filter sweep at fixed seq_length=120 (10 cutoffs x 5 models = 50 configs)
  Stage B: Seq_length sweep at best filter per model (9 seq_lengths x 5 models = 45 configs)
  Stage C: Joint verification (top 3 filter x top 3 seq x 5 models = 45 configs)
  Total: 140 configs, each with 16-fold LOPO, 25 epochs

Usage:
    python filter_seqlen_sweep.py --input ../../data/TB-File-01.csv --output ../../results/filter_seqlen_sweep
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Data helpers
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
        pdata_filt = butter_lowpass(pdata - mu, cutoff_hz)
        X_out[idx] = pdata_filt + mu
    return X_out


def create_sequences(X, y, pidx, seq_length):
    participants = np.unique(pidx)
    X_seqs, y_seqs, pidx_seqs = [], [], []
    for pid in participants:
        idx = np.where(pidx == pid)[0]
        Xp, yp = X[idx], y[idx]
        for i in range(seq_length, len(Xp)):
            X_seqs.append(Xp[i - seq_length : i])
            y_seqs.append(yp[i - 1])  # contemporaneous target
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
# Model definitions (minimal copies — avoid importing from train_comprehensive
# which prints on import and has side effects)
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
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_out, n_out, k, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                 self.conv2, self.chomp2, self.relu2, self.drop2)
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
        for i in range(len(num_channels)):
            d = 2 ** i
            inc = input_size if i == 0 else num_channels[i - 1]
            outc = num_channels[i]
            layers.append(TemporalBlock(inc, outc, kernel_size, stride=1,
                                        dilation=d, padding=(kernel_size - 1) * d, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(num_channels[-1], 32), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        return self.fc(out[:, :, -1]).squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.var_sel = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_size, input_size), nn.Softmax(dim=-1))
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.grn = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout))
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        w = self.var_sel(x)
        x = self.embed(x * w)
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        g = self.grn(a)
        f = self.ln(a + g)
        return self.fc(f[:, -1, :]).squeeze(-1)


class PatchTSTModel(nn.Module):
    def __init__(self, input_size=6, seq_len=120, patch_len=24, stride=12,
                 d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Sequential(nn.Linear(patch_len * input_size, d_model),
                                         nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        B = x.shape[0]
        patches = []
        for i in range(0, x.shape[1] - self.patch_len + 1, self.stride):
            patches.append(x[:, i : i + self.patch_len, :].reshape(B, -1))
        patches = torch.stack(patches, dim=1)
        x = self.patch_embed(patches) + self.pos_embed
        out = self.transformer(x)
        return self.fc(out.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# TCN channel auto-scaler
# ---------------------------------------------------------------------------

def get_tcn_channels(seq_length, kernel_size=3, base_ch=32, max_ch=128):
    needed = (seq_length - 1) / (2 * (kernel_size - 1))
    num_levels = max(3, math.ceil(math.log2(needed + 1)))
    channels = []
    for i in range(num_levels):
        ch = min(max_ch, base_ch * (2 ** min(i, 2)))
        channels.append(ch)
    rf = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
    return channels, rf


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

import copy


def train_model(model, train_loader, val_loader, epochs=25, lr=0.001, patience=8):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    crit = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += crit(model(xb), yb).item()
        vl /= max(1, len(val_loader))
        sched.step(vl)

        if vl < best_loss:
            best_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    ds = TSDataset(X, np.zeros(len(X)))
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_name, seq_length, input_size=6):
    if model_name == "lstm":
        return LSTMModel(input_size=input_size)
    elif model_name == "gru":
        return GRUModel(input_size=input_size)
    elif model_name == "tcn":
        channels, rf = get_tcn_channels(seq_length)
        return TCNModel(input_size=input_size, num_channels=channels)
    elif model_name == "tft":
        return TFTModel(input_size=input_size)
    elif model_name == "patchtst":
        pl = min(24, max(8, seq_length // 5))
        st = max(4, pl // 2)
        num_patches = (seq_length - pl) // st + 1
        if num_patches < 2:
            return None  # invalid config
        return PatchTSTModel(input_size=input_size, seq_len=seq_length,
                             patch_len=pl, stride=st)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Single config evaluation (16-fold LOPO)
# ---------------------------------------------------------------------------

def evaluate_config(model_name, X_filtered, y, pidx, participants, seq_length,
                    epochs=25, patience=8):
    X_seq, y_seq, pidx_seq = create_sequences(X_filtered, y, pidx, seq_length)
    if len(y_seq) == 0:
        return float("inf"), 0.0, 0.0, []

    all_participants = np.unique(pidx_seq)
    rmses, corrs, r2s = [], [], []

    batch_size = 64
    if seq_length >= 360:
        batch_size = 16 if model_name == "tft" else 32

    for fold_idx, test_pid in enumerate(all_participants):
        train_mask = pidx_seq != test_pid
        test_mask = ~train_mask

        # Validation: deterministic
        train_pids = all_participants[all_participants != test_pid]
        val_pid = train_pids[fold_idx % len(train_pids)]
        val_mask = (pidx_seq == val_pid) & train_mask
        actual_train = train_mask & ~val_mask

        X_tr, y_tr = X_seq[actual_train], y_seq[actual_train]
        X_va, y_va = X_seq[val_mask], y_seq[val_mask]
        X_te, y_te_raw = X_seq[test_mask], y_seq[test_mask]

        # Per-fold standardization
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_tr_n = sc_X.fit_transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
        y_tr_n = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
        X_va_n = sc_X.transform(X_va.reshape(-1, X_va.shape[-1])).reshape(X_va.shape)
        y_va_n = sc_y.transform(y_va.reshape(-1, 1)).ravel()
        X_te_n = sc_X.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)

        model = build_model(model_name, seq_length)
        if model is None:
            return float("inf"), 0.0, 0.0, []

        tr_loader = DataLoader(TSDataset(X_tr_n, y_tr_n), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(TSDataset(X_va_n, y_va_n), batch_size=batch_size, shuffle=False)

        model = train_model(model, tr_loader, va_loader, epochs=epochs, patience=patience)

        y_pred_n = predict(model, X_te_n)
        y_pred = sc_y.inverse_transform(y_pred_n.reshape(-1, 1)).ravel()

        rmse = np.sqrt(mean_squared_error(y_te_raw, y_pred))
        corr = np.corrcoef(y_te_raw, y_pred)[0, 1] if len(y_te_raw) > 1 else 0.0
        r2 = r2_score(y_te_raw, y_pred) if len(y_te_raw) > 1 else 0.0
        rmses.append(rmse)
        corrs.append(corr)
        r2s.append(r2)

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.mean(rmses), np.mean(corrs), np.mean(r2s), rmses


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

MODELS = ["lstm", "gru", "tcn", "tft", "patchtst"]
FILTER_CUTOFFS = [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2, None]
SEQ_LENGTHS = [30, 60, 90, 120, 180, 240, 360, 480, 600]


def run_stage_a(pidx, X_raw, y, participants, filter_cache, epochs, patience, out_dir):
    """Stage A: filter sweep at fixed seq_length=120."""
    print("\n" + "=" * 70)
    print("STAGE A: Filter Sweep (seq_length=120)")
    print("=" * 70)

    results = []
    seq_length = 120
    total = len(FILTER_CUTOFFS) * len(MODELS)
    count = 0

    for cutoff in FILTER_CUTOFFS:
        X_filt = filter_cache[cutoff]
        for model_name in MODELS:
            count += 1
            label = f"{cutoff}" if cutoff else "none"
            print(f"  [{count}/{total}] filter={label}, model={model_name}...", end=" ", flush=True)
            t0 = time.time()
            rmse, corr, r2, per_p = evaluate_config(
                model_name, X_filt, y, pidx, participants, seq_length, epochs, patience)
            elapsed = time.time() - t0
            print(f"RMSE={rmse:.4f} r={corr:.4f} ({elapsed:.0f}s)")
            results.append({
                "stage": "A", "model": model_name, "filter_hz": cutoff,
                "seq_length": seq_length, "mean_rmse": rmse, "mean_corr": corr,
                "mean_r2": r2, "std_rmse": np.std(per_p) if per_p else 0,
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "stage_a_results.csv"), index=False)

    # Find top 3 filters per model
    best_filters = {}
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("mean_rmse")
        top3 = sub.head(3)["filter_hz"].tolist()
        best_filters[m] = top3
        best = sub.iloc[0]
        print(f"  {m}: best filter={best['filter_hz']}, RMSE={best['mean_rmse']:.4f}")

    # Plot
    for m in MODELS:
        sub = df[df["model"] == m].copy()
        sub["filter_label"] = sub["filter_hz"].apply(lambda x: x if x else 0.6)
        sub = sub.sort_values("filter_label")
        plt.figure(figsize=(8, 4))
        plt.plot(sub["filter_label"], sub["mean_rmse"], "o-", markersize=5)
        plt.xscale("log")
        plt.xlabel("Filter Cutoff (Hz)")
        plt.ylabel("Mean RMSE")
        plt.title(f"Stage A: {m.upper()} — Filter Sweep")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"filter_curve_{m}.png"), dpi=150)
        plt.close()

    return best_filters


def run_stage_b(pidx, X_raw, y, participants, best_filters, epochs, patience, out_dir):
    """Stage B: seq_length sweep at each model's best filter."""
    print("\n" + "=" * 70)
    print("STAGE B: Sequence Length Sweep")
    print("=" * 70)

    results = []
    total = len(SEQ_LENGTHS) * len(MODELS)
    count = 0

    for model_name in MODELS:
        best_filter = best_filters[model_name][0]  # top 1 from stage A
        X_filt = apply_lowpass(X_raw, pidx, participants, best_filter)

        for seq_len in SEQ_LENGTHS:
            count += 1
            print(f"  [{count}/{total}] model={model_name}, filter={best_filter}, seq={seq_len}...",
                  end=" ", flush=True)
            t0 = time.time()
            rmse, corr, r2, per_p = evaluate_config(
                model_name, X_filt, y, pidx, participants, seq_len, epochs, patience)
            elapsed = time.time() - t0
            print(f"RMSE={rmse:.4f} r={corr:.4f} ({elapsed:.0f}s)")
            results.append({
                "stage": "B", "model": model_name, "filter_hz": best_filter,
                "seq_length": seq_len, "mean_rmse": rmse, "mean_corr": corr,
                "mean_r2": r2, "std_rmse": np.std(per_p) if per_p else 0,
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "stage_b_results.csv"), index=False)

    # Find top 3 seq_lengths per model
    best_seqs = {}
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("mean_rmse")
        top3 = sub.head(3)["seq_length"].tolist()
        best_seqs[m] = top3
        best = sub.iloc[0]
        print(f"  {m}: best seq={best['seq_length']}, RMSE={best['mean_rmse']:.4f}")

    # Plot
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("seq_length")
        plt.figure(figsize=(8, 4))
        plt.errorbar(sub["seq_length"], sub["mean_rmse"], yerr=sub["std_rmse"],
                     fmt="o-", markersize=5, capsize=3)
        plt.xlabel("Sequence Length (s)")
        plt.ylabel("Mean RMSE")
        plt.title(f"Stage B: {m.upper()} — Seq Length Sweep (filter={best_filters[m][0]})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"seqlen_curve_{m}.png"), dpi=150)
        plt.close()

    return best_seqs


def run_stage_c(pidx, X_raw, y, participants, best_filters, best_seqs,
                epochs, patience, out_dir):
    """Stage C: joint verification of top 3x3 combos per model."""
    print("\n" + "=" * 70)
    print("STAGE C: Joint Verification")
    print("=" * 70)

    results = []
    total = sum(len(best_filters[m]) * len(best_seqs[m]) for m in MODELS)
    count = 0

    for model_name in MODELS:
        for filt in best_filters[model_name]:
            X_filt = apply_lowpass(X_raw, pidx, participants, filt)
            for seq_len in best_seqs[model_name]:
                count += 1
                label_f = f"{filt}" if filt else "none"
                print(f"  [{count}/{total}] model={model_name}, filter={label_f}, seq={seq_len}...",
                      end=" ", flush=True)
                t0 = time.time()
                rmse, corr, r2, per_p = evaluate_config(
                    model_name, X_filt, y, pidx, participants, seq_len, epochs, patience)
                elapsed = time.time() - t0
                print(f"RMSE={rmse:.4f} r={corr:.4f} ({elapsed:.0f}s)")
                results.append({
                    "stage": "C", "model": model_name, "filter_hz": filt,
                    "seq_length": seq_len, "mean_rmse": rmse, "mean_corr": corr,
                    "mean_r2": r2, "std_rmse": np.std(per_p) if per_p else 0,
                })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "stage_c_results.csv"), index=False)

    # Determine best config per model + interaction check
    best_configs = {}
    print("\n" + "=" * 70)
    print("BEST CONFIGS PER MODEL")
    print("=" * 70)
    for m in MODELS:
        sub = df[df["model"] == m].sort_values("mean_rmse")
        best = sub.iloc[0]
        # Separable product: best filter from A (at fixed seq=120) + best seq from B (at fixed filter)
        sep_filter = best_filters[m][0]
        sep_seq = best_seqs[m][0]
        sep_row = sub[(sub["filter_hz"] == sep_filter) & (sub["seq_length"] == sep_seq)]

        if len(sep_row) > 0:
            sep_rmse = sep_row.iloc[0]["mean_rmse"]
            joint_rmse = best["mean_rmse"]
            threshold = 0.5 * best["std_rmse"]
            interaction = abs(joint_rmse - sep_rmse) > threshold

            if interaction and joint_rmse < sep_rmse:
                chosen_filter = best["filter_hz"]
                chosen_seq = int(best["seq_length"])
                note = "joint (interaction detected)"
            else:
                chosen_filter = sep_filter
                chosen_seq = int(sep_seq)
                note = "separable"
        else:
            chosen_filter = best["filter_hz"]
            chosen_seq = int(best["seq_length"])
            note = "joint (separable combo not in C)"

        best_configs[m] = {
            "filter_hz": chosen_filter if chosen_filter is not None else None,
            "seq_length": chosen_seq,
            "rmse": float(best["mean_rmse"]),
            "note": note,
        }
        print(f"  {m}: filter={chosen_filter}, seq={chosen_seq}, RMSE={best['mean_rmse']:.4f} ({note})")

    # Save
    with open(os.path.join(out_dir, "best_configs.json"), "w") as f:
        json.dump(best_configs, f, indent=2)

    # Heatmaps
    heatmap_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    for m in MODELS:
        sub = df[df["model"] == m]
        if len(sub) < 4:
            continue
        sub_copy = sub.copy()
        sub_copy["filter_label"] = sub_copy["filter_hz"].apply(lambda x: str(x) if x else "none")
        try:
            pivot = sub_copy.pivot_table(index="filter_label", columns="seq_length", values="mean_rmse")
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Seq Length")
            ax.set_ylabel("Filter (Hz)")
            ax.set_title(f"{m.upper()} — Joint RMSE (Stage C)")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, label="RMSE")
            plt.tight_layout()
            plt.savefig(os.path.join(heatmap_dir, f"{m}_heatmap.png"), dpi=150)
            plt.close()
        except Exception:
            pass

    return best_configs


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Filter + SeqLen Sweep")
    parser.add_argument("--input", type=str, default="../../data/TB-File-01.csv")
    parser.add_argument("--output", type=str, default="../../results/filter_seqlen_sweep")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading data from {args.input}...")
    pidx, X_raw, y, participants = load_raw(args.input)
    print(f"  {len(y)} samples, {len(participants)} participants")

    # Pre-cache filtered data for all cutoffs
    print("Pre-caching filtered data...")
    filter_cache = {}
    for cutoff in FILTER_CUTOFFS:
        filter_cache[cutoff] = apply_lowpass(X_raw, pidx, participants, cutoff)
    print(f"  Cached {len(filter_cache)} filter variants")

    # Stage A
    best_filters = run_stage_a(pidx, X_raw, y, participants, filter_cache,
                               args.epochs, args.patience, out_dir)

    # Stage B
    best_seqs = run_stage_b(pidx, X_raw, y, participants, best_filters,
                            args.epochs, args.patience, out_dir)

    # Stage C
    best_configs = run_stage_c(pidx, X_raw, y, participants, best_filters, best_seqs,
                               args.epochs, args.patience, out_dir)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"Results saved to {out_dir}/")
    print("Best configs:")
    for m, cfg in best_configs.items():
        print(f"  {m}: filter={cfg['filter_hz']}, seq={cfg['seq_length']}, RMSE={cfg['rmse']:.4f}")


if __name__ == "__main__":
    main()
