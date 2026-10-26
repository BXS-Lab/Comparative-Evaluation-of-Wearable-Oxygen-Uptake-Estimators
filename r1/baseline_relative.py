"""Baseline-relative analysis: information added by the wearable sensor set beyond a target
history that is d seconds stale. Reference learners: Ridge (alpha and cutoff selected by inner grouped 5-fold CV)
and XGBoost with fixed library defaults (cutoff selected by inner CV). Outer LOPO."""
import numpy as np
import pandas as pd
import xgboost as xgb

from .cv import inner_folds, per_participant_rmse, run_seed, selection_score
from .linear import RIDGE_ALPHAS
from .preprocess import CUTOFF_GRID, FoldScaler

D_GRID = [5, 10, 15, 20, 30, 45, 60, 90, 120]
XGB_DEFAULTS = dict(n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
                    tree_method="hist")


def history_features(panel, d, window=30):
    """For each row r (time t): latest target value available at or before t-d, and mean / LS slope of the target
    values with time in (t-d-window, t-d] that are available at or before t-d (same segment). NaN if < 2 values."""
    df = panel.df
    y = df["y_hist"].to_numpy(float) if "y_hist" in df else df["y"].to_numpy(float)
    avail = df["avail"].to_numpy()
    seg = df["seg"].to_numpy()
    n = len(df)
    H = np.full((n, 3), np.nan)
    bounds = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        av = avail[s:e] - s          # positions within segment
        yy = y[s:e]
        idx = np.arange(e - s)
        cut = idx - d                 # latest admissible availability position
        J = np.searchsorted(av, cut, side="right") - 1   # av is non-decreasing
        lo = np.maximum(cut - window, -1)                # window (lo, J]
        c0 = np.r_[0, np.cumsum(np.ones_like(yy))]
        c1 = np.r_[0, np.cumsum(yy)]
        cj = np.r_[0, np.cumsum(idx.astype(float))]
        cjj = np.r_[0, np.cumsum(idx.astype(float) ** 2)]
        cjy = np.r_[0, np.cumsum(idx * yy)]
        ok = J >= 0
        a, b = np.clip(lo + 1, 0, None), np.clip(J + 1, 0, None)
        cnt = c0[b] - c0[a]
        sy, sj, sjj, sjy = c1[b] - c1[a], cj[b] - cj[a], cjj[b] - cjj[a], cjy[b] - cjy[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sy / cnt
            slope = (cnt * sjy - sj * sy) / (cnt * sjj - sj ** 2)
        valid = ok & (cnt >= 2) & (b > a)
        H[s:e, 0] = np.where(ok, yy[np.clip(J, 0, None)], np.nan)
        H[s:e, 1] = np.where(valid, mean, np.nan)
        H[s:e, 2] = np.where(valid, slope, np.nan)
    return H


def _ridge_fit_predict(X, y, tr, te):
    sx, sy = FoldScaler().fit(X[tr]), FoldScaler().fit(y[tr].reshape(-1, 1))
    Xs, ys = sx.transform(X[tr]), sy.transform(y[tr].reshape(-1, 1)).ravel()
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    Uty = U.T @ ys
    Xte = sx.transform(X[te])
    return [sy.inverse((Xte @ (Vt.T @ (s / (s ** 2 + a) * Uty))).reshape(-1, 1)).ravel() for a in RIDGE_ALPHAS]


def _xgb_fit_predict(X, y, tr, te, seed):
    sx, sy = FoldScaler().fit(X[tr]), FoldScaler().fit(y[tr].reshape(-1, 1))
    m = xgb.XGBRegressor(random_state=seed, n_jobs=4, **XGB_DEFAULTS)
    m.fit(sx.transform(X[tr]), sy.transform(y[tr].reshape(-1, 1)).ravel())
    return sy.inverse(m.predict(sx.transform(X[te])).reshape(-1, 1)).ravel()


def run_outer(panel, filtered_cache, H_by_d, rows_ok, outer_pid, learner):
    """Returns a long table of per-participant RMSE for the outer participant: (d, arm) for this learner."""
    df = panel.df
    pid = df["pid"].to_numpy()
    y = df["y"].to_numpy(float)
    train_pids = np.array([p for p in panel.pids if p != outer_pid])
    folds = inner_folds(train_pids, 5)
    train_mask = np.isin(pid, train_pids) & rows_ok
    te_rows = np.flatnonzero((pid == outer_pid) & rows_ok)
    out = []

    def design(arm, d, cutoff):
        parts = []
        if arm in ("history", "history+sensors"):
            parts.append(H_by_d[d])
        if arm in ("sensors", "history+sensors"):
            parts.append(filtered_cache[cutoff])
        return np.hstack(parts)

    arms = [("sensors", None)] + [(a, d) for d in D_GRID for a in ("history", "history+sensors")]
    for arm, d in arms:
        cutoffs = [None] if arm == "history" else CUTOFF_GRID
        best = None
        for c in cutoffs:
            X = design(arm, d, c)
            if learner == "ridge":
                held = [dict() for _ in RIDGE_ALPHAS]
                for f in folds:
                    tr = train_mask & ~np.isin(pid, f)
                    te = np.flatnonzero(np.isin(pid, f) & rows_ok)
                    for i, yp in enumerate(_ridge_fit_predict(X, y, tr, te)):
                        held[i].update(per_participant_rmse(y[te], yp, pid[te]).to_dict())
                for i, a in enumerate(RIDGE_ALPHAS):
                    sc = selection_score(held[i])
                    if best is None or sc < best[0]:
                        best = (sc, c, i)
            else:
                held = {}
                for fi, f in enumerate(folds):
                    tr = train_mask & ~np.isin(pid, f)
                    te = np.flatnonzero(np.isin(pid, f) & rows_ok)
                    yp = _xgb_fit_predict(X, y, tr, te, run_seed(panel.name, "W4b", outer_pid, fi, arm, d, c))
                    held.update(per_participant_rmse(y[te], yp, pid[te]).to_dict())
                sc = selection_score(held)
                if best is None or sc < best[0]:
                    best = (sc, c, 0)
        X = design(arm, d, best[1])
        if learner == "ridge":
            yp = _ridge_fit_predict(X, y, train_mask, te_rows)[best[2]]
        else:
            yp = _xgb_fit_predict(X, y, train_mask, te_rows, run_seed(panel.name, "W4b", outer_pid, "refit", arm, d, best[1]))
        rmse = float(np.sqrt(np.mean((yp - y[te_rows]) ** 2)))
        out.append({"pid": int(outer_pid), "learner": learner, "arm": arm, "d": d, "cutoff": best[1], "rmse": rmse})
    return pd.DataFrame(out)


def common_rows(panel, H_by_d):
    """Main evaluation grid intersected over all d (all history features defined)."""
    ok = panel.eval_mask().copy()
    for H in H_by_d.values():
        ok &= np.isfinite(H).all(axis=1)
    return ok
