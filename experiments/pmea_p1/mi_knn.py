"""
Reproducible mutual-information feature screening + dHR ablation for the wearable -> VO2 paper.

Individual MI : 10-bin histogram MI (nats) between each 0.01 Hz pooled low-pass-filtered feature and
                raw VO2 (the preprocessing the linear/tree models use). Deterministic.
dHR ablation  : change in Ridge leave-one-participant-out RMSE when delta-HR is removed (per-fold
                standardization + inner 5-fold alpha selection, matching the linear-model protocol).

These are the values reported in the Supplementary "Feature Information Content" results; the joint-
distribution figure (make_data_figures.py: fig_joint) recomputes the same individual MI in-script, so
the figure annotations and the text stay in sync. Conditional/unique MI was dropped: a rigorous k-NN
conditional-MI estimate did not support the earlier "dHR has the largest unique MI" claim and is
statistically unreliable with five conditioning variables.

Run from the repository root:
    python experiments/pmea_p1/mi_knn.py
"""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

FEATS = ["BF", "VE", "ACC", "HR", "CAD", "dHR"]


def load():
    df = pd.read_csv("data/TB-File-01.csv")
    df = df[~(df == -1).any(axis=1)].reset_index(drop=True)
    pid = df["ID"].to_numpy(int)
    vo2 = (df["VO2"] / df["Weight"]).to_numpy(float)
    base = {c: df[c].to_numpy(float) for c in ["BF", "VE", "ACC", "HR", "CAD"]}
    dhr = np.zeros(len(df))
    for p in np.unique(pid):
        m = pid == p
        dhr[m] = np.concatenate([[0.0], np.diff(base["HR"][m])])
    base["dHR"] = dhr
    return pid, np.column_stack([base[f] for f in FEATS]), vo2


def lowpass_pooled(A, cutoff=0.01, order=2):
    b, a = butter(order, cutoff / 0.5, btype="low")
    return filtfilt(b, a, A, axis=0)


def mi_10bin(x, y, bins=10):
    """10-bin histogram mutual information in nats."""
    c = np.histogram2d(x, y, bins=bins)[0]
    p = c / c.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True); nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (px * py)[nz])))


def ridge_lopo_rmse(pid, Xf, vo2, cols):
    parts = np.unique(pid); grid = {"alpha": np.logspace(-3, 6, 30)}
    rmses = []
    for tp in parts:
        tr, te = pid != tp, pid == tp
        scX, scY = StandardScaler(), StandardScaler()
        Xtr = scX.fit_transform(Xf[tr][:, cols]); ytr = scY.fit_transform(vo2[tr].reshape(-1, 1)).ravel()
        gs = GridSearchCV(Ridge(), grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1).fit(Xtr, ytr)
        yp = scY.inverse_transform(gs.predict(scX.transform(Xf[te][:, cols])).reshape(-1, 1)).ravel()
        rmses.append(np.sqrt(mean_squared_error(vo2[te], yp)))
    return float(np.mean(rmses))


def main():
    pid, X, vo2 = load()
    Xf = lowpass_pooled(X)
    mi = {f: round(mi_10bin(Xf[:, i], vo2), 3) for i, f in enumerate(FEATS)}

    all6 = list(range(len(FEATS)))
    no_dhr = [i for i, f in enumerate(FEATS) if f != "dHR"]
    r_all = ridge_lopo_rmse(pid, Xf, vo2, all6)
    r_no = ridge_lopo_rmse(pid, Xf, vo2, no_dhr)
    pct = round((r_no - r_all) / r_all * 100, 2)

    print("individual MI (10-bin nats, 0.01 Hz pooled low-pass):", mi)
    print(f"Ridge LOPO RMSE: all={r_all:.4f}  no_dHR={r_no:.4f}  removing dHR = {pct:+.2f}%")
    json.dump({"MI_individual": mi, "ridge_rmse_all": round(r_all, 4),
               "ridge_rmse_no_dHR": round(r_no, 4), "dHR_ablation_pct": pct,
               "method": "10-bin histogram MI (nats) on 0.01 Hz pooled low-pass features; Ridge LOPO ablation"},
              open("results/pmea_p1/mi_screening.json", "w"), indent=2)
    print("[saved] results/pmea_p1/mi_screening.json")


if __name__ == "__main__":
    main()
