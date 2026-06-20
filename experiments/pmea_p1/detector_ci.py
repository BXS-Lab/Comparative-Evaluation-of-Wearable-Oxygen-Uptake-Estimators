"""
Subject-cluster bootstrap CIs for the causal transition detector's reliability (panel request):
the manuscript reports Spearman rho=0.23 (causal detector inp_trans vs oracle dvo2_s) and AUC=0.47
(for 'PatchTST beats Ridge' locally) as bare point estimates. Here we quantify their uncertainty by
resampling the 16 participants with replacement.

Run from the repository root :  python experiments/pmea_p1/detector_ci.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

M = pd.read_parquet("results/pmea_p0/master_aligned.parquet")
M = M.dropna(subset=["inp_trans", "dvo2_s"]).copy()
parts = sorted(M["participant"].unique())
RNG = np.random.default_rng(42)

# point estimates on the full matched sample
rho_pt = spearmanr(M["inp_trans"], M["dvo2_s"]).statistic
event = (np.abs(M["PATCHTST"] - M["y_true"]) < np.abs(M["RIDGE"] - M["y_true"])).astype(int).values
auc_pt = roc_auc_score(event, M["inp_trans"].values)
print(f"point: Spearman rho(inp_trans, dvo2_s) = {rho_pt:.3f}   AUC(PatchTST beats Ridge | inp_trans) = {auc_pt:.3f}")

# subject-cluster bootstrap: resample participants, recompute on the pooled resampled rows
by_pid = {p: M[M.participant == p] for p in parts}
rhos, aucs = [], []
for _ in range(2000):
    idx = RNG.integers(0, len(parts), len(parts))
    sub = pd.concat([by_pid[parts[i]] for i in idx], ignore_index=True)
    try:
        r = spearmanr(sub["inp_trans"], sub["dvo2_s"]).statistic
        ev = (np.abs(sub["PATCHTST"] - sub["y_true"]) < np.abs(sub["RIDGE"] - sub["y_true"])).astype(int).values
        a = roc_auc_score(ev, sub["inp_trans"].values) if ev.min() != ev.max() else np.nan
    except Exception:
        r, a = np.nan, np.nan
    rhos.append(r); aucs.append(a)
rhos = np.array(rhos); aucs = np.array(aucs)
rlo, rhi = np.nanpercentile(rhos, [2.5, 97.5])
alo, ahi = np.nanpercentile(aucs, [2.5, 97.5])
print(f"Spearman rho : {rho_pt:.3f}  95% subject-cluster bootstrap CI [{rlo:.2f}, {rhi:.2f}]")
print(f"AUC          : {auc_pt:.3f}  95% subject-cluster bootstrap CI [{alo:.2f}, {ahi:.2f}]")
