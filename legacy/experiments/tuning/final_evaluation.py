"""
Phase 3: Final Evaluation with Best Hyperparameters

Loads best HPs from Phase 2 (Optuna) and best preprocessing from Phase 1,
runs full 16-fold LOPO with 200 epochs, saves paper-ready results.

Usage:
    python final_evaluation.py --input ../../data/TB-File-01.csv --output ../../results/optuna_temporal
"""

import argparse
import copy
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["lstm", "gru", "tcn", "tft", "patchtst"]

# Import everything from the Optuna script (same model definitions)
# We inline the critical functions to keep this self-contained.
from optuna_temporal_tuning import (
    load_raw, apply_lowpass, create_sequences, TSDataset,
    build_model_from_params, train_model, predict
)


def run_final_lopo(model_type, params, filter_hz, seq_length, input_csv, out_dir):
    """Run full 16-fold LOPO with 200 epochs."""
    pidx, X_raw, y, participants = load_raw(input_csv)
    X_filt = apply_lowpass(X_raw, pidx, participants, filter_hz)
    X_seq, y_seq, pidx_seq = create_sequences(X_filt, y, pidx, seq_length)
    all_participants = np.unique(pidx_seq)

    batch_size = params.get("batch_size", 64)
    lr = params.get("lr", 0.001)
    wd = params.get("weight_decay", 0.0)

    per_participant = []
    all_y_true, all_y_pred, all_pids = [], [], []

    for fold_idx, test_pid in enumerate(all_participants):
        print(f"    Fold {fold_idx+1}/{len(all_participants)} (pid={test_pid})...", end=" ", flush=True)

        train_mask = pidx_seq != test_pid
        test_mask = ~train_mask
        train_pids = all_participants[all_participants != test_pid]
        val_pid = train_pids[fold_idx % len(train_pids)]
        val_mask = (pidx_seq == val_pid) & train_mask
        actual_train = train_mask & ~val_mask

        X_tr, y_tr = X_seq[actual_train], y_seq[actual_train]
        X_va, y_va = X_seq[val_mask], y_seq[val_mask]
        X_te, y_te = X_seq[test_mask], y_seq[test_mask]

        sc_X, sc_y = StandardScaler(), StandardScaler()
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
        corr = np.corrcoef(y_te, y_pred)[0, 1] if len(y_te) > 1 else 0.0
        r2 = r2_score(y_te, y_pred) if len(y_te) > 1 else 0.0

        per_participant.append({"participant": int(test_pid), "rmse": rmse,
                                "correlation": corr, "r2_score": r2})
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)
        all_pids.extend([int(test_pid)] * len(y_te))

        print(f"RMSE={rmse:.3f} r={corr:.3f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save
    model_dir = os.path.join(out_dir, model_type.upper())
    os.makedirs(model_dir, exist_ok=True)

    pd.DataFrame(per_participant).to_csv(
        os.path.join(model_dir, "metrics_per_participant.csv"), index=False)
    pd.DataFrame({"participant": all_pids, "y_true": all_y_true, "y_pred": all_y_pred}).to_csv(
        os.path.join(model_dir, "y_yhat.csv"), index=False)

    rmses = [p["rmse"] for p in per_participant]
    corrs = [p["correlation"] for p in per_participant]
    r2s = [p["r2_score"] for p in per_participant]

    return {
        "model": model_type, "mean_rmse": np.mean(rmses), "std_rmse": np.std(rmses),
        "mean_corr": np.mean(corrs), "mean_r2": np.mean(r2s),
        "per_participant_rmse": rmses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../data/TB-File-01.csv")
    parser.add_argument("--output", default="../../results/optuna_temporal")
    parser.add_argument("--best-configs", default="../../results/filter_seqlen_sweep/best_configs.json")
    args = parser.parse_args()

    with open(args.best_configs) as f:
        best_configs = json.load(f)

    all_results = []
    all_rmse_arrays = {}

    for model_type in MODELS:
        params_path = os.path.join(args.output, model_type.upper(), "best_params.json")
        if not os.path.exists(params_path):
            print(f"  Skipping {model_type} — no best_params.json found")
            continue

        with open(params_path) as f:
            hp_data = json.load(f)

        params = hp_data["params"]
        filter_hz = hp_data.get("filter_hz", best_configs[model_type]["filter_hz"])
        seq_length = hp_data.get("seq_length", best_configs[model_type]["seq_length"])

        print(f"\n{'='*70}")
        print(f"  FINAL EVALUATION: {model_type.upper()}")
        print(f"  filter={filter_hz}, seq={seq_length}")
        print(f"  params={params}")
        print(f"{'='*70}")

        result = run_final_lopo(model_type, params, filter_hz, seq_length, args.input, args.output)
        all_results.append(result)
        all_rmse_arrays[model_type] = result["per_participant_rmse"]

    # Comparison table
    print(f"\n{'='*70}")
    print("FINAL RANKING")
    print(f"{'='*70}")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_participant_rmse"} for r in all_results])
    df = df.sort_values("mean_rmse")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.output, "final_comparison.csv"), index=False)

    # Wilcoxon pairwise tests (deferred items, but compute now for convenience)
    if len(all_rmse_arrays) >= 2:
        models_tested = list(all_rmse_arrays.keys())
        pval_results = []
        for i in range(len(models_tested)):
            for j in range(i + 1, len(models_tested)):
                m1, m2 = models_tested[i], models_tested[j]
                try:
                    stat, pval = wilcoxon(all_rmse_arrays[m1], all_rmse_arrays[m2])
                    pval_results.append({"model_a": m1, "model_b": m2, "statistic": stat, "p_value": pval})
                except Exception:
                    pass
        if pval_results:
            pd.DataFrame(pval_results).to_csv(
                os.path.join(args.output, "significance_tests.csv"), index=False)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
