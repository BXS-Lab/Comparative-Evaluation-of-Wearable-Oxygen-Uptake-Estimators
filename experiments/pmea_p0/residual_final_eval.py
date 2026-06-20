"""
P0-2 (fair): does a WELL-TUNED temporal residual beat the direct temporal model?
For each model we use its Optuna best_params (same architecture / filter_hz / seq_length / training as the
tuned direct model from final_evaluation.py) and, per LOPO fold, train TWO models under identical conditions:
  DIRECT   : predict VO2/kg directly                       (reproduces optuna_temporal)
  RESIDUAL : predict VO2/kg - Ridge(point-wise) residual, then add Ridge back
Plus a leak-free blend ridge + k*residual (k fit on the training folds).
Paired per-participant RMSE + Wilcoxon. This isolates the residual idea with the tuning held fair.
Run from the repository root.  Outputs -> results/pmea_p0/residual_fair/
"""
import argparse, json, os, sys, copy
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import wilcoxon
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path("experiments/tuning").resolve()))
from optuna_temporal_tuning import (load_raw, apply_lowpass, create_sequences, TSDataset,
                                     build_model_from_params, train_model, predict)

np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed(42)
OUT = Path("results/pmea_p0/residual_fair"); OUT.mkdir(parents=True, exist_ok=True)
RIDGE_ALPHAS = np.logspace(-3, 6, 30)

def target_global_index(pidx, seq_length):
    """global point-index of each sequence's target, in create_sequences order."""
    gidx = []
    for pid in np.unique(pidx):
        idx = np.where(pidx == pid)[0]
        for i in range(seq_length, len(idx)):
            gidx.append(idx[i - 1])
    return np.array(gidx)

def run_model(model_type, input_csv, epochs, max_folds=None):
    hp = json.load(open(f"results/optuna_temporal/{model_type.upper()}/best_params.json"))
    params, filt, seqL = hp["params"], hp["filter_hz"], hp["seq_length"]
    pidx, X_raw, y, participants = load_raw(input_csv)
    X_temp = apply_lowpass(X_raw, pidx, participants, filt)      # temporal features
    X_ridge = apply_lowpass(X_raw, pidx, participants, 0.01)     # ridge baseline features
    X_seq, y_seq, pid_seq = create_sequences(X_temp, y, pidx, seqL)
    tgi = target_global_index(pidx, seqL)
    assert len(tgi) == len(y_seq) and np.allclose(y[tgi], y_seq), "target-index mismatch"
    parts = np.unique(pid_seq)
    bs, lr, wd = params.get("batch_size", 64), params.get("lr", 1e-3), params.get("weight_decay", 0.0)
    # reuse EXISTING direct results (same pipeline/seed/seq_length -> identical per-participant test sets)
    dm = pd.read_csv(f"results/optuna_temporal/{model_type.upper()}/metrics_per_participant.csv")
    direct_rmse = dict(zip(dm["participant"].astype(int), dm["rmse"]))
    rows = []
    folds = parts if max_folds is None else parts[:max_folds]
    for fi, test_pid in enumerate(folds):
        tr_pts = pidx != test_pid
        scX = StandardScaler().fit(X_ridge[tr_pts])
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(scX.transform(X_ridge[tr_pts]), y[tr_pts])
        ridge_pred = ridge.predict(scX.transform(X_ridge))                # all points (OOF for test)
        ridge_seq = ridge_pred[tgi]; resid_seq = y_seq - ridge_seq
        # sequence split (same val logic as final_evaluation)
        trm = pid_seq != test_pid; tem = ~trm
        tr_pids = parts[parts != test_pid]; val_pid = tr_pids[fi % len(tr_pids)]
        vlm = (pid_seq == val_pid) & trm; atm = trm & ~vlm
        scSX = StandardScaler().fit(X_seq[atm].reshape(-1, X_seq.shape[-1]))
        def sx(a): return scSX.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape)
        Xtr, Xva, Xte = sx(X_seq[atm]), sx(X_seq[vlm]), sx(X_seq[tem])
        out = {"participant": int(test_pid)}
        # ---- train ONLY the residual model; direct is reused from existing optuna_temporal results ----
        scY = StandardScaler().fit(resid_seq[atm].reshape(-1, 1))
        ytr = scY.transform(resid_seq[atm].reshape(-1, 1)).ravel()
        yva = scY.transform(resid_seq[vlm].reshape(-1, 1)).ravel()
        torch.manual_seed(42)
        model = build_model_from_params(model_type, params, seqL)
        if model is None: return None
        trl = DataLoader(TSDataset(Xtr, ytr), batch_size=bs, shuffle=True)
        val = DataLoader(TSDataset(Xva, yva), batch_size=bs, shuffle=False)
        model = train_model(model, trl, val, epochs=epochs, lr=lr, patience=15, wd=wd)
        resid_pred = scY.inverse_transform(predict(model, Xte).reshape(-1, 1)).ravel()
        del model; torch.cuda.empty_cache()
        yte = y_seq[tem]; ridge_te = ridge_seq[tem]
        out["rmse_direct"] = float(direct_rmse[int(test_pid)])   # existing tuned direct result
        out["rmse_ridge"] = np.sqrt(mean_squared_error(yte, ridge_te))
        out["rmse_resid_k1"] = np.sqrt(mean_squared_error(yte, ridge_te + resid_pred))
        out["_yte"] = yte; out["_ridge"] = ridge_te; out["_resid"] = resid_pred  # for leak-free k
        rows.append(out)
        print(f"  {model_type} fold {fi+1}/{len(folds)} pid={test_pid}: "
              f"direct={out['rmse_direct']:.3f} ridge={out['rmse_ridge']:.3f} resid_k1={out['rmse_resid_k1']:.3f}", flush=True)
    # leak-free blend coefficient k (fit on the OTHER folds' pooled residual/ridge), applied per held-out fold
    for i, r in enumerate(rows):
        num = den = 0.0
        for j, q in enumerate(rows):
            if j == i: continue
            num += np.sum(q["_resid"] * (q["_yte"] - q["_ridge"])); den += np.sum(q["_resid"] ** 2)
        k = num / den if den > 0 else 0.0
        r["k"] = k
        r["rmse_resid_kLF"] = np.sqrt(mean_squared_error(r["_yte"], r["_ridge"] + k * r["_resid"]))
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df.to_csv(OUT / f"{model_type}_folds.csv", index=False)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tcn,patchtst,gru,lstm")
    ap.add_argument("--input", default="data/TB-File-01.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-folds", type=int, default=None)
    a = ap.parse_args()
    summ = []
    for m in a.models.split(","):
        m = m.strip()
        print(f"\n===== {m.upper()} (epochs={a.epochs}, max_folds={a.max_folds}) =====", flush=True)
        df = run_model(m, a.input, a.epochs, a.max_folds)
        if df is None: print(f"  {m}: invalid config, skipped"); continue
        def wp(c):
            try: return wilcoxon(df["rmse_direct"], df[c]).pvalue
            except ValueError: return 1.0
        row = dict(model=m, n=len(df), k_mean=df["k"].mean(),
                   direct=df["rmse_direct"].mean(), ridge=df["rmse_ridge"].mean(),
                   resid_k1=df["rmse_resid_k1"].mean(), resid_kLF=df["rmse_resid_kLF"].mean(),
                   p_direct_vs_k1=wp("rmse_resid_k1"), p_direct_vs_kLF=wp("rmse_resid_kLF"),
                   kLF_better_n=int((df["rmse_resid_kLF"] < df["rmse_direct"]).sum()))
        summ.append(row)
        print(f"  SUMMARY {m}: direct={row['direct']:.3f} resid_k1={row['resid_k1']:.3f} "
              f"resid_kLF={row['resid_kLF']:.3f} (k={row['k_mean']:.2f}) "
              f"p(direct vs kLF)={row['p_direct_vs_kLF']:.3f} kLF<direct in {row['kLF_better_n']}/{len(df)}", flush=True)
    if summ:
        S = pd.DataFrame(summ); S.to_csv(OUT / "summary.csv", index=False)
        print("\n=== FAIR TUNED RESIDUAL vs DIRECT (per-participant mean RMSE) ===")
        print(S.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
