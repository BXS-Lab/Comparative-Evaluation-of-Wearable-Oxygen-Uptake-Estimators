"""
R1 matched-objective re-tune: Random Forest, Optuna on LOPO-RMSE.

Replaces the manuscript's RF (which maximized Pearson r) with an RF selected on the SAME objective
the leaderboard reports: the mean 16-fold leave-one-participant-out RMSE. Preprocessing is identical
to the shared tree-preprocessing routine (the tree/linear convention) so RF stays comparable to
the linear family.

Run from the repository root :
    python experiments/pmea_r1_matched_tuning/optuna_rf.py --n-trials 150
    python experiments/pmea_r1_matched_tuning/optuna_rf.py --n-trials 2 --smoke   # quick validation
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
        m = RandomForestRegressor(n_jobs=-1, random_state=42, **params)
        m.fit(X[tr], y[tr])
        rmses.append(np.sqrt(mean_squared_error(y[te], m.predict(X[te]))))
    return float(np.mean(rmses)), rmses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/TB-File-01.csv")
    ap.add_argument("--output", default="results/pmea_r1_matched_tuning/RF")
    ap.add_argument("--n-trials", type=int, default=150)
    ap.add_argument("--smoke", action="store_true", help="tiny run for validation")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    pidx, X, y, parts = load_and_prepare(args.input)
    print(f"[data] N={len(y)}  participants={len(parts)}  features={X.shape[1]}")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        )
        rmses = []
        for fold_idx, test_pid in enumerate(parts):
            tr, te = pidx != test_pid, pidx == test_pid
            m = RandomForestRegressor(n_jobs=-1, random_state=42, **params)
            m.fit(X[tr], y[tr])
            rmses.append(np.sqrt(mean_squared_error(y[te], m.predict(X[te]))))
            trial.report(float(np.mean(rmses)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses))

    db = out / "study.db"
    study = optuna.create_study(
        study_name="vo2_rf_lopo_rmse", direction="minimize",
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
