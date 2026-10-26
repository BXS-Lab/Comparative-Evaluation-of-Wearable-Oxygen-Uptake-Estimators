"""Nested selection for the linear family: OLS, Ridge, Lasso, ElasticNet.

Cutoff and penalty are chosen jointly by grouped inner CV on the outer-training participants; the outer
participant is never used. Training rows = all segment samples of the training participants; scoring rows
= evaluation-grid samples (first 240 s of each segment excluded)."""
import numpy as np
from sklearn.linear_model import enet_path

from .cv import inner_folds, per_participant_rmse, selection_score
from .preprocess import CUTOFF_GRID, FoldScaler, causal_filter

RIDGE_ALPHAS = np.logspace(-3, 6, 30)
LASSO_ALPHAS = np.logspace(-4, 3, 30)
ENET_ALPHAS = np.logspace(-4, 3, 20)
ENET_L1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
MODELS = ["OLS", "RIDGE", "LASSO", "ELASTICNET"]


def _candidates(model):
    if model == "OLS":
        return [{}]
    if model == "RIDGE":
        return [{"alpha": a} for a in RIDGE_ALPHAS]
    if model == "LASSO":
        return [{"alpha": a, "l1_ratio": 1.0} for a in LASSO_ALPHAS]
    return [{"alpha": a, "l1_ratio": r} for r in ENET_L1 for a in ENET_ALPHAS]


def _fit_all(model, Xs, ys):
    """Return coefficient vectors for every candidate on standardised data (train means are 0, so no intercept)."""
    if model == "OLS":
        return [np.linalg.lstsq(Xs, ys, rcond=None)[0]]
    if model == "RIDGE":
        U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        Uty = U.T @ ys
        return [Vt.T @ (s / (s ** 2 + a) * Uty) for a in RIDGE_ALPHAS]  # sklearn Ridge objective
    coefs = []
    ratios = [1.0] if model == "LASSO" else ENET_L1
    alphas = LASSO_ALPHAS if model == "LASSO" else ENET_ALPHAS
    for r in ratios:
        order = np.argsort(alphas)[::-1]
        _, path, _ = enet_path(Xs, ys, l1_ratio=r, alphas=alphas[order], max_iter=10000, tol=1e-6,
                               precompute=True)
        back = np.empty_like(path)
        back[:, order] = path
        coefs.extend(back[:, i] for i in range(len(alphas)))
    return coefs


def _fit_predict(model, X, y, tr, te):
    sx, sy = FoldScaler().fit(X[tr]), FoldScaler().fit(y[tr].reshape(-1, 1))
    Xs, ys = sx.transform(X[tr]), sy.transform(y[tr].reshape(-1, 1)).ravel()
    coefs = _fit_all(model, Xs, ys)
    Xte = sx.transform(X[te])
    return [sy.inverse((Xte @ w).reshape(-1, 1)).ravel() for w in coefs]


def nested_outer_fold(panel, model, outer_pid, k_inner=5, filtered_cache=None):
    """Select (cutoff, hyperparameters) on the outer-training participants; refit; predict the outer participant.
    Returns (selected, inner_table, eval_rows, y_pred)."""
    df = panel.df
    pid = df["pid"].to_numpy()
    y = df["y"].to_numpy(float)
    grid = panel.eval_mask()
    train_pids = np.array([p for p in panel.pids if p != outer_pid])
    folds = inner_folds(train_pids, k_inner)
    cands = _candidates(model)
    table = []
    for cutoff in CUTOFF_GRID:
        X = filtered_cache[cutoff] if filtered_cache is not None else causal_filter(df[panel.features], df["seg"], cutoff)
        held = [dict() for _ in cands]
        for f in folds:
            tr = np.isin(pid, train_pids) & ~np.isin(pid, f)
            te = np.isin(pid, f) & grid
            preds = _fit_predict(model, X, y, tr, te)
            for ci, yp in enumerate(preds):
                held[ci].update(per_participant_rmse(y[te], yp, pid[te]).to_dict())
        for ci, c in enumerate(cands):
            table.append({"cutoff": cutoff, **c, "score": selection_score(held[ci])})
    best = min(table, key=lambda r: r["score"])
    X = filtered_cache[best["cutoff"]] if filtered_cache is not None else causal_filter(df[panel.features], df["seg"], best["cutoff"])
    tr = np.isin(pid, train_pids)
    te_rows = np.flatnonzero((pid == outer_pid) & grid)
    ci = next(i for i, c in enumerate(cands) if all(np.isclose(c[k], best[k]) for k in c))
    y_pred = _fit_predict(model, X, y, tr, te_rows)[ci]
    return best, table, te_rows, y_pred
