"""
Reproduce the main model-comparison leaderboard (manuscript Table 2) from the matched master.
For each of the 11 models: Avg RMSE = mean of the 16 per-participant RMSEs; Corr. and R^2 are
POOLED over the matched sample (Pearson r and coefficient of determination across all 59,731 points).
This matches the column conventions stated in the Table 2 caption.

Run from the repository root:  python experiments/pmea_p1/make_leaderboard_table.py
Output -> results/pmea_p0/p04_table_ready.csv  (also consumed by make_nature_figures.py for family labels)
"""
import numpy as np
import pandas as pd

M = pd.read_parquet("results/pmea_p0/master_aligned.parquet")
parts = sorted(M["participant"].unique())
y = M["y_true"].to_numpy()
ss_tot = float(np.sum((y - y.mean()) ** 2))

FAM = {**{m: "Linear" for m in ["LINEAR", "RIDGE", "LASSO", "ELASTICNET"]},
       **{m: "Tree" for m in ["RF", "XGB"]},
       **{m: "Temporal" for m in ["GRU", "LSTM", "TCN", "TFT", "PATCHTST"]}}

rows = []
for m in FAM:
    yh = M[m].to_numpy()
    rmse_pp = float(np.mean([
        np.sqrt(np.mean((M.loc[M.participant == p, m].to_numpy()
                         - M.loc[M.participant == p, "y_true"].to_numpy()) ** 2))
        for p in parts]))                                   # per-participant mean RMSE
    corr_pooled = float(np.corrcoef(yh, y)[0, 1])           # pooled Pearson r
    r2_pooled = float(1.0 - np.sum((y - yh) ** 2) / ss_tot)  # pooled R^2
    rows.append(dict(model=m, family=FAM[m], rmse=rmse_pp, corr=corr_pooled, r2=r2_pooled))

df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
df.to_csv("results/pmea_p0/p04_table_ready.csv", index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n[saved] results/pmea_p0/p04_table_ready.csv")
