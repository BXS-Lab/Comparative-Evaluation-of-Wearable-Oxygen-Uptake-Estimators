"""
Predictability-horizon uncertainty (panel request): the ~21 s crossover is the single most load-bearing
positive result but was reported as a bare point estimate from one figure. Here we compute, PER PARTICIPANT,
the forecast horizon h at which naive persistence (copy-forward of the criterion VO2 by h seconds) first
exceeds a sensor-based (0-step) estimator's RMSE, and report the median + IQR across the 16 participants and a
subject-cluster bootstrap 95% CI on the median. Reuses the persistence logic of p0_master_analysis.py.

Run from the repository root :  python experiments/pmea_p1/horizon_uncertainty.py
"""
import numpy as np, pandas as pd
from pathlib import Path

RES = Path("results"); OUT = RES / "pmea_p1"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

# raw series (full per-participant VO2/kg on the within-participant index t)
raw = pd.read_csv("data/TB-File-01.csv").sort_values("ID", kind="stable").reset_index(drop=True)
raw["t"] = raw.groupby("ID").cumcount()
raw["vo2kg"] = raw["VO2"] / raw["Weight"]
full = raw.set_index(["ID", "t"])["vo2kg"]

# matched master (retuned RF/XGB already swapped in)
M = pd.read_parquet(RES / "pmea_p0" / "master_aligned.parquet")
parts = sorted(M["participant"].unique())

def model_pp_rmse(model):
    return (M.groupby("participant")
              .apply(lambda g: np.sqrt(np.mean((g[model].values - g["y_true"].values) ** 2)),
                     include_groups=False)
              .reindex(parts).values)

def persist_pp_rmse(h):
    """per-participant persistence-at-horizon-h RMSE on the matched rows."""
    out = {}
    for p, g in M.groupby("participant"):
        yt = g["y_true"].values; tt = g["t"].values
        prev = full.loc[p].reindex(tt - h).values
        ok = ~np.isnan(prev)
        out[p] = np.sqrt(np.mean((yt[ok] - prev[ok]) ** 2)) if ok.sum() else np.nan
    return pd.Series(out).reindex(parts).values

HGRID = np.arange(1, 61)                      # fine 1..60 s grid
persist = np.vstack([persist_pp_rmse(h) for h in HGRID])   # (len(HGRID), 16)

def crossing_per_participant(model):
    """per-participant horizon where persistence(h) first >= model RMSE, linearly interpolated."""
    mr = model_pp_rmse(model)                 # (16,)
    cross = np.full(len(parts), np.nan)
    for i in range(len(parts)):
        pc = persist[:, i]; target = mr[i]
        above = np.where(pc >= target)[0]
        if len(above) == 0:
            cross[i] = np.nan                 # persistence never exceeds model within 60 s (censored)
            continue
        k = above[0]
        if k == 0:
            cross[i] = HGRID[0]
        else:
            x0, x1 = HGRID[k - 1], HGRID[k]; y0, y1 = pc[k - 1], pc[k]
            cross[i] = x0 + (target - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    return cross, mr

def summarize(model):
    cross, mr = crossing_per_participant(model)
    valid = cross[~np.isnan(cross)]; ncens = int(np.isnan(cross).sum())
    med = np.median(valid); q1, q3 = np.percentile(valid, [25, 75])
    # subject-cluster bootstrap CI on the median (resample 16 participants)
    boots = []
    for _ in range(5000):
        idx = RNG.integers(0, len(parts), len(parts))
        cc = cross[idx]; cc = cc[~np.isnan(cc)]
        if len(cc): boots.append(np.median(cc))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(model=model, model_rmse=mr.mean(), median=med, q1=q1, q3=q3,
                ci_lo=lo, ci_hi=hi, n_censored=ncens,
                min=valid.min(), max=valid.max())

rows = [summarize(m) for m in ["TCN", "PATCHTST", "RF", "RIDGE", "GRU", "LSTM", "TFT", "XGB"]]
df = pd.DataFrame(rows)
df.to_csv(OUT / "horizon_crossing_per_participant.csv", index=False)

# raw per-participant crossing values (for the on-axis rug strip in the horizon figure)
cross_tcn, _ = crossing_per_participant("TCN")
pd.DataFrame({"participant": parts, "crossing_s": cross_tcn}).to_csv(
    OUT / "horizon_crossing_values_TCN.csv", index=False)


# pooled mean-curve crossing + subject-cluster bootstrap 95% CI (horizon-figure connectors & confidence zone).
# Separate RNG so the per-participant median CIs above stay byte-identical.
def pooled_cross(persist_cols, mrmse):
    pcv = persist_cols.mean(axis=1); above = np.where(pcv >= mrmse)[0]
    if len(above) == 0:
        return np.nan
    k = above[0]
    if k == 0:
        return float(HGRID[0])
    x0, x1 = HGRID[k - 1], HGRID[k]; y0, y1 = pcv[k - 1], pcv[k]
    return float(x0 + (mrmse - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)

RNG2 = np.random.default_rng(2024)
prows = []
for mdl in ["TCN", "RF", "XGB", "RIDGE"]:
    mr = model_pp_rmse(mdl)
    pt = pooled_cross(persist, float(np.nanmean(mr)))
    bs = []
    for _ in range(5000):
        idx = RNG2.integers(0, len(parts), len(parts))
        x = pooled_cross(persist[:, idx], float(np.nanmean(mr[idx])))
        if not np.isnan(x):
            bs.append(x)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    prows.append(dict(model=mdl, model_rmse=float(np.nanmean(mr)), pooled=pt, ci_lo=float(lo), ci_hi=float(hi)))
pdf = pd.DataFrame(prows)
pdf.to_csv(OUT / "horizon_pooled_crossing_ci.csv", index=False)
print("\nPooled crossing + subject-cluster bootstrap 95% CI (horizon figure):")
print(pdf.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("Per-participant predictability-horizon crossing (s): persistence(h) first exceeds model RMSE")
print("(median + IQR across 16 participants; 95% CI = subject-cluster bootstrap on the median)")
print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# pooled curve crossing (where MEAN persistence first exceeds MEAN model RMSE) for the headline models
print("\nPooled (mean-curve) crossing for reference:")
for m in ["TCN", "PATCHTST", "RF", "RIDGE"]:
    mr = model_pp_rmse(m).mean(); pc = persist.mean(axis=1)
    above = np.where(pc >= mr)[0]
    if len(above):
        k = above[0]
        x = HGRID[0] if k == 0 else HGRID[k-1] + (mr - pc[k-1])*(HGRID[k]-HGRID[k-1])/(pc[k]-pc[k-1])
        print(f"  {m:9s} model RMSE={mr:.3f}  pooled crossing={x:.1f}s")
    else:
        print(f"  {m:9s} model RMSE={mr:.3f}  pooled crossing>60s")
print(f"\n[saved] {OUT}/horizon_crossing_per_participant.csv")
