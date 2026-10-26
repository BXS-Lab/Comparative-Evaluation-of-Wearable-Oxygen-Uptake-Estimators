"""Nested selection for the tree family: RF (60 trials) and XGBoost (100 trials).

One fresh Optuna study per (dataset, model, outer fold); the filter cutoff is a categorical hyperparameter;
inner grouped 5-fold CV on the outer-training participants; MedianPruner on the running mean of held-out
per-participant RMSE after each inner fold. Features and target are standardised on the training rows of each
fit. Refits use seed indices {0, 1, 2}."""
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from .cv import inner_folds, per_participant_rmse, run_seed, selection_score
from .preprocess import CUTOFF_GRID, FoldScaler

BUDGET = {"RF": 60, "XGB": 100}
CUTOFF_CHOICES = [str(c) for c in CUTOFF_GRID]  # Optuna categorical; "None" = unfiltered


def _cutoff(c):
    return None if c == "None" else float(c)


def suggest(trial, model):
    p = {"cutoff": trial.suggest_categorical("cutoff", CUTOFF_CHOICES)}
    if model == "RF":
        p.update(n_estimators=trial.suggest_int("n_estimators", 100, 800),
                 max_depth=trial.suggest_int("max_depth", 3, 30),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                 max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]))
    else:
        p.update(n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 max_depth=trial.suggest_int("max_depth", 2, 10),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 gamma=trial.suggest_float("gamma", 0.0, 5.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0))
    return p


def _make(model, params, seed, n_jobs):
    hp = {k: v for k, v in params.items() if k != "cutoff"}
    if model == "RF":
        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **hp)
    return xgb.XGBRegressor(tree_method="hist", random_state=seed, n_jobs=n_jobs, **hp)


def fit_predict(model, params, X, y, tr, te, seed, n_jobs):
    sx, sy = FoldScaler().fit(X[tr]), FoldScaler().fit(y[tr].reshape(-1, 1))
    m = _make(model, params, seed, n_jobs)
    m.fit(sx.transform(X[tr]), sy.transform(y[tr].reshape(-1, 1)).ravel())
    return sy.inverse(m.predict(sx.transform(X[te])).reshape(-1, 1)).ravel()


def nested_outer_fold(panel, model, outer_pid, filtered_cache, n_jobs=6, k_inner=5, n_trials=None,
                      seeds=(0, 1, 2)):
    """Returns (best_params, trials_df, best_so_far_df, eval_rows, {seed_index: y_pred})."""
    df = panel.df
    pid = df["pid"].to_numpy()
    y = df["y"].to_numpy(float)
    grid = panel.eval_mask()
    train_pids = np.array([p for p in panel.pids if p != outer_pid])
    folds = inner_folds(train_pids, k_inner)

    def objective(trial):
        params = suggest(trial, model)
        X = filtered_cache[_cutoff(params["cutoff"])]
        held = {}
        for fi, f in enumerate(folds):
            tr = np.isin(pid, train_pids) & ~np.isin(pid, f)
            te = np.isin(pid, f) & grid
            seed = run_seed(panel.name, model, outer_pid, fi, trial.number, 0, 0)
            yp = fit_predict(model, params, X, y, tr, te, seed, n_jobs)
            held.update(per_participant_rmse(y[te], yp, pid[te]).to_dict())
            trial.report(selection_score(held), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return selection_score(held)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10, seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials or BUDGET[model])
    trials = study.trials_dataframe()
    done = trials[trials["state"] == "COMPLETE"].sort_values("number")
    best_so_far = pd.DataFrame({"trial": done["number"], "best_so_far": done["value"].cummin()})
    best = dict(study.best_params)
    X = filtered_cache[_cutoff(best["cutoff"])]
    tr = np.isin(pid, train_pids)
    rows = np.flatnonzero((pid == outer_pid) & grid)
    preds = {s: fit_predict(model, best, X, y, tr, rows, run_seed(panel.name, model, outer_pid, "refit", -1, s, 0), n_jobs)
             for s in seeds}
    best["inner_score"] = float(study.best_value)
    best["n_complete"] = int((trials["state"] == "COMPLETE").sum())
    best["n_pruned"] = int((trials["state"] == "PRUNED").sum())
    return best, trials, best_so_far, rows, preds
