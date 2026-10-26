"""
R1 matched-objective re-tune: XGBoost, Optuna on LOPO-RMSE (single seed, reproducible).

XGBoost's objective was already RMSE; this run replaces the hard-coded / non-reproducible params with a
clean Optuna search and drops the disclosed "best-of-six-seeds" selection (single random_state=42).
Preprocessing identical to the shared tree-preprocessing routine.

Run from the repository root :
    python experiments/pmea_r1_matched_tuning/optuna_xgb.py --n-trials 200
    python experiments/pmea_r1_matched_tuning/optuna_xgb.py --n-trials 2 --smoke
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def butter_lowpass_filter(data, cutoff=0.01, fs=1.0, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, data, axis=0)


def load_and_prepare(csv_path):                      # identical to the shared tree preprocessing
    df = pd.read_csv(csv_path); pidx = df["ID"].values
    y = df["VO2"].values / df["Weight"].values
    N = len(df); X = np.zeros((N, 6))
    X[:, 0] = df["BF"]; X[:, 1] = df["VE"]; X[:, 2] = df["ACC"]; X[:, 3] = df["HR"]; X[:, 4] = df["CAD"]
    for pid in np.unique(pidx):
        idx = (pidx == pid); hr = X[idx, 3]; d = np.zeros(len(hr)); d[1:] = hr[1:] - hr[:-1]; X[idx, 5] = d
    return pidx, butter_lowpass_filter(X), y, np.unique(pidx)


def lopo_mean_rmse(params, pidx, X, y, parts):
    rmses = []
    for test_pid in parts:
        tr, te = pidx != test_pid, pidx == test_pid
        m = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist", **params)
        m.fit(X[tr], y[tr])
        rmses.append(np.sqrt(mean_squared_error(y[te], m.predict(X[te]))))
    return float(np.mean(rmses)), rmses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/TB-File-01.csv")
    ap.add_argument("--output", default="results/pmea_r1_matched_tuning/XGB")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pidx, X, y, parts = load_and_prepare(args.input)
    print(f"[data] N={len(y)}  participants={len(parts)}  features={X.shape[1]}")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0),
        )
        rmses = []
        for fold_idx, test_pid in enumerate(parts):
            tr, te = pidx != test_pid, pidx == test_pid
            m = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist", **params)
            m.fit(X[tr], y[tr])
            rmses.append(np.sqrt(mean_squared_error(y[te], m.predict(X[te]))))
            trial.report(float(np.mean(rmses)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses))

    db = out / "study.db"
    study = optuna.create_study(
        study_name="vo2_xgb_lopo_rmse", direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10, seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
        storage=f"sqlite:///{db}", load_if_exists=True,
    )
    n_trials = 2 if args.smoke else args.n_trials
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    dt = time.time() - t0

    best = study.best_trial
    best_mean, best_folds = lopo_mean_rmse(best.params, pidx, X, y, parts)
    print(f"\n[best] trial #{best.number}  LOPO-mean RMSE={best.value:.4f}  (recomputed {best_mean:.4f})")
    print(f"[best] params: {best.params}")
    print(f"[time] {n_trials} trials in {dt/60:.1f} min  ({dt/max(1,n_trials):.1f}s/trial)")

    json.dump(
        {"params": best.params, "rmse": best.value, "rmse_recomputed": best_mean,
         "per_participant_rmse": best_folds, "n_trials": n_trials,
         "objective": "lopo_mean_rmse", "preprocessing": "tree_global_0.01Hz", "seed": 42},
        open(out / "best_params.json", "w"), indent=2)
    study.trials_dataframe().to_csv(out / "all_trials.csv", index=False)
    print(f"[saved] {out}/best_params.json")


if __name__ == "__main__":
    main()
