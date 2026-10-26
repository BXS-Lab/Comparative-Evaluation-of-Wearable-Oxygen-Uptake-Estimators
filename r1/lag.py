"""Lag diagnostic: Ridge on wearable inputs at t predicting V̇O₂(t−h), h = −60…60 s.

Inputs are zero-phase filtered (2nd-order Butterworth, 0.05 Hz, filtfilt method='gust') per segment, or
unfiltered (sensitivity). This characterises signal timing and is not a deployable estimator. Outer LOPO; alpha
selected per lag by inner grouped 5-fold CV; computed with per-participant Gram matrices."""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .cv import inner_folds
from .linear import RIDGE_ALPHAS

H_GRID = np.arange(-60, 61)


def zero_phase(X, seg, cutoff=0.05):
    b, a = butter(2, cutoff / 0.5, btype="low")
    out = np.empty_like(X, dtype=float)
    bounds = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        out[s:e] = filtfilt(b, a, X[s:e], axis=0, method="gust")
    return out


def shifted_targets(panel):
    """Y[r, k] = y at row r - H_GRID[k] within the same segment (NaN otherwise); rows valid for every h."""
    df = panel.df
    y, seg = df["y"].to_numpy(float), df["seg"].to_numpy()
    n = len(y)
    Y = np.full((n, len(H_GRID)), np.nan)
    for k, h in enumerate(H_GRID):
        src = np.arange(n) - h
        ok = (src >= 0) & (src < n)
        ok[ok] &= seg[src[ok]] == seg[ok]
        Y[ok, k] = y[src[ok]]
    rows = panel.eval_mask() & np.isfinite(Y).all(axis=1)
    return Y, rows


def run(panel, X, rows, Y, outer_pids=None, pids=None):
    """X: (n, p) input matrix (already filtered or raw). Returns per-participant RMSE(h) table for outer participants.
    `pids` restricts the whole analysis (training and testing) to a participant subset."""
    df = panel.df
    pid = df["pid"].to_numpy()
    out = []
    P = sorted(int(q) for q in (panel.pids if pids is None else pids))
    rows = rows & np.isin(pid, P)
    for op in (outer_pids or P):
        train = [q for q in P if q != op]
        folds = inner_folds(train, 5)

        def fit(tr_pids):
            tr = np.isin(pid, tr_pids) & rows
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            Xs = (X[tr] - mu) / sd
            ym = Y[tr].mean(0)
            G = Xs.T @ Xs
            B = Xs.T @ (Y[tr] - ym)
            W = np.stack([np.linalg.solve(G + a * np.eye(G.shape[0]), B) for a in RIDGE_ALPHAS])  # (A, p, H)
            return mu, sd, ym, W

        def rmse_by_pid(model, te_pids):
            mu, sd, ym, W = model
            res = {}
            for q in te_pids:
                te = (pid == q) & rows
                P_ = np.einsum("np,aph->anh", (X[te] - mu) / sd, W) + ym      # (A, n, H)
                res[q] = np.sqrt(np.mean((P_ - Y[te][None]) ** 2, axis=1))    # (A, H)
            return res

        held = {}
        for f in folds:
            held.update(rmse_by_pid(fit([q for q in train if q not in set(f.tolist())]), f.tolist()))
        score = np.mean(np.stack(list(held.values())), axis=0)              # (A, H)
        best_a = score.argmin(axis=0)                                       # per h
        r = rmse_by_pid(fit(train), [op])[op]                               # (A, H)
        out.append(pd.DataFrame({"pid": op, "h": H_GRID, "rmse": r[best_a, np.arange(len(H_GRID))],
                                 "alpha": RIDGE_ALPHAS[best_a]}))
    return pd.concat(out, ignore_index=True)


def summarise(tab, edge=2, flat=0.01, n_boot=5000, seed=20260601):
    """Optimal h per participant (median, IQR, percentile bootstrap CI of the median) and of the pooled mean curve,
    with the edge/flat rule."""
    opt = tab.loc[tab.groupby("pid")["rmse"].idxmin()].set_index("pid")["h"]
    pooled = tab.groupby("h")["rmse"].mean()
    h_pool = int(pooled.idxmin())
    undetermined = (abs(h_pool) >= 60 - edge) or ((pooled.max() - pooled.min()) / pooled.min() < flat)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(opt.to_numpy(), len(opt))) for _ in range(n_boot)]
    return {"h_pooled": h_pool, "h_median": float(opt.median()), "h_q1": float(opt.quantile(0.25)),
            "h_q3": float(opt.quantile(0.75)), "h_median_ci_lo": float(np.percentile(meds, 2.5)),
            "h_median_ci_hi": float(np.percentile(meds, 97.5)), "undetermined": bool(undetermined)}
