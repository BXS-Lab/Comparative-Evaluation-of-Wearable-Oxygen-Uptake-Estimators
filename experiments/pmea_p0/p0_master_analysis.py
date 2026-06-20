"""
PMEA P0 master analysis (reanalysis on saved predictions only).
Builds ONE matched 11-model dataframe (common time grid, common raw VO2/kg truth, both an
oracle |dVO2/dt| transition signal and a causal cadence-change detector), then computes:
  P0-4  matched-sample leaderboard + bootstrap CIs + Friedman + Holm-Wilcoxon (+family-level)
  P0-1  task-matched persistence skill score S=1-RMSE_model/RMSE_persist(h) + horizon curve
Run from the repository root :  python experiments/pmea_p0/p0_master_analysis.py
Outputs -> results/pmea_p0/
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results"); OUT = RES / "pmea_p0"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

raw = pd.read_csv("data/TB-File-01.csv").sort_values("ID", kind="stable").reset_index(drop=True)
raw["t"] = raw.groupby("ID").cumcount()
raw["vo2kg"] = raw["VO2"] / raw["Weight"]
rc = raw.groupby("ID").size().to_dict()
truth = raw.set_index(["ID", "t"])["vo2kg"]

MODEL_FILES = {
    "LINEAR":"linear_fixed/LINEAR/y_yhat.csv","RIDGE":"linear_fixed/RIDGE/y_yhat.csv",
    "LASSO":"linear_fixed/LASSO/y_yhat.csv","ELASTICNET":"linear_fixed/ELASTICNET/y_yhat.csv",
    "RF":"pmea_r1_matched_tuning/RF/y_yhat.csv","XGB":"pmea_r1_matched_tuning/XGB/y_yhat.csv",  # re-tuned trees (LOPO-RMSE objective)
    "GRU":"optuna_temporal/GRU/y_yhat.csv","LSTM":"optuna_temporal/LSTM/y_yhat.csv",
    "TCN":"optuna_temporal/TCN/y_yhat.csv","TFT":"optuna_temporal/TFT/y_yhat.csv",
    "PATCHTST":"optuna_temporal/PATCHTST/y_yhat.csv",
}
LINEAR=["LINEAR","RIDGE","LASSO","ELASTICNET"]; TREE=["RF","XGB"]; TEMP=["GRU","LSTM","TCN","TFT","PATCHTST"]
ALL=LINEAR+TREE+TEMP
FAMILY={**{m:"Linear" for m in LINEAR},**{m:"Tree" for m in TREE},**{m:"Temporal" for m in TEMP}}

def align(df):
    """within-participant time index via the integer shift best matching y_true to raw truth."""
    out=[]
    for p,g in df.groupby("participant",sort=True):
        base=rc[p]-len(g); best_s,best_e=0,np.inf
        for s in range(-3,4):
            t=np.arange(base+s,base+s+len(g)); ok=(t>=0)&(t<rc[p])
            tv=truth.loc[p].reindex(t[ok]).values
            e=np.nanmean(np.abs(g["y_true"].values[ok]-tv))
            if e<best_e: best_e,best_s=e,s
        g=g.copy(); g["t"]=np.arange(base+best_s,base+best_s+len(g)); out.append(g)
    res=pd.concat(out,ignore_index=True); return res[res.t>=0]

preds={m:align(pd.read_csv(RES/f)) for m,f in MODEL_FILES.items()}
M=raw[["ID","t","vo2kg"]].rename(columns={"ID":"participant","vo2kg":"y_true"}).copy()
for m in ALL:
    M=M.merge(preds[m][["participant","t","y_pred"]].rename(columns={"y_pred":m}),on=["participant","t"],how="inner")
print(f"matched common set: {len(M)} samples, {M.participant.nunique()} participants, {len(ALL)} models")

# transition signals
M["dvo2"]=M.groupby("participant")["y_true"].diff().abs()
M["dvo2_s"]=M.groupby("participant")["dvo2"].transform(lambda s:s.rolling(15,center=True,min_periods=1).mean())
r=raw.copy(); r["dCAD"]=r.groupby("ID")["CAD"].diff().abs()
r["inp_trans"]=r.groupby("ID")["dCAD"].transform(lambda s:s.diff().abs().rolling(90,min_periods=1).max())
M=M.merge(r[["ID","t","inp_trans"]].rename(columns={"ID":"participant"}),on=["participant","t"],how="left")
M.to_parquet(OUT/"master_aligned.parquet")

# transition masks: top tercile (oracle dvo2_s) and top tercile (causal inp_trans)
q67_o=M["dvo2_s"].quantile(0.67); q33_o=M["dvo2_s"].quantile(0.33)
trans_o=M["dvo2_s"]>=q67_o; steady_o=M["dvo2_s"]<=q33_o
q67_c=M["inp_trans"].quantile(0.67); trans_c=M["inp_trans"]>=q67_c

def pp_rmse(col,mask=None):
    """per-participant RMSE vector (len=16) on optional mask."""
    sub=M if mask is None else M[mask]
    return sub.groupby("participant").apply(lambda g:np.sqrt(np.mean((g[col]-g["y_true"])**2)),include_groups=False)

def boot_ci(vec,nb=2000):
    idx=np.arange(len(vec)); means=[np.mean(vec[RNG.integers(0,len(vec),len(vec))]) for _ in range(nb)]
    return np.percentile(means,[2.5,97.5])

# ---------- P0-4 matched leaderboard ----------
rows=[]
ppm={m:pp_rmse(m).reindex(sorted(M.participant.unique())).values for m in ALL}
for m in ALL:
    v=ppm[m]; lo,hi=boot_ci(v)
    rows.append(dict(model=m,family=FAMILY[m],
        rmse_mean=v.mean(),rmse_sd=v.std(ddof=1),ci_lo=lo,ci_hi=hi,
        rmse_pooled=np.sqrt(np.mean((M[m]-M["y_true"])**2)),
        rmse_steady=pp_rmse(m,steady_o).mean(),rmse_trans=pp_rmse(m,trans_o).mean(),
        trans_over_steady=pp_rmse(m,trans_o).mean()/pp_rmse(m,steady_o).mean()))
lb=pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
lb.to_csv(OUT/"p04_leaderboard_matched.csv",index=False)
print("\n=== P0-4 MATCHED-SAMPLE LEADERBOARD (per-participant mean RMSE) ===")
print(lb[["model","family","rmse_mean","ci_lo","ci_hi","rmse_steady","rmse_trans","trans_over_steady"]].to_string(index=False,float_format=lambda x:f"{x:.3f}"))

# Friedman across all models (per-participant RMSE blocks)
fried=friedmanchisquare(*[ppm[m] for m in ALL])
print(f"\nFriedman across {len(ALL)} models: chi2={fried.statistic:.2f}, p={fried.pvalue:.2e}")

# Holm-Wilcoxon pairwise
pairs=[]
for i in range(len(ALL)):
    for j in range(i+1,len(ALL)):
        a,b=ALL[i],ALL[j]; d=ppm[a]-ppm[b]
        try: w=wilcoxon(ppm[a],ppm[b]); p=w.pvalue
        except ValueError: p=1.0
        pairs.append(dict(a=a,b=b,median_diff=np.median(d),n_a_better=int((ppm[a]<ppm[b]).sum()),p_raw=p))
pf=pd.DataFrame(pairs).sort_values("p_raw").reset_index(drop=True)
mtot=len(pf)
# Holm step-down: sort ascending, multiply by (m-rank), enforce monotonicity, cap at 1
order=np.argsort(pf["p_raw"].values); adj=np.empty(mtot)
run=0.0
for rank,idx in enumerate(order):
    val=pf["p_raw"].values[idx]*(mtot-rank); run=max(run,val); adj[idx]=min(run,1.0)
pf["p_holm"]=adj
pf.to_csv(OUT/"p04_pairwise_holm_wilcoxon.csv",index=False)
nsurv=int((pf["p_holm"]<0.05).sum())
print(f"pairwise comparisons: {mtot}; surviving Holm (<0.05): {nsurv}")
print("top pairwise (by raw p):")
print(pf.head(6).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

# family-level: best-in-family per participant
def fam_best(fam):
    cols=[m for m in ALL if FAMILY[m]==fam]; return np.min(np.stack([ppm[m] for m in cols]),0)
famvec={f:fam_best(f) for f in ["Linear","Tree","Temporal"]}
fam_tests=[]
for f1,f2 in [("Temporal","Linear"),("Temporal","Tree"),("Tree","Linear")]:
    w=wilcoxon(famvec[f1],famvec[f2])
    fam_tests.append(dict(comparison=f"{f1} vs {f2}",median_diff=np.median(famvec[f1]-famvec[f2]),
        n_f1_better=int((famvec[f1]<famvec[f2]).sum()),p=w.pvalue))
ft=pd.DataFrame(fam_tests); ft.to_csv(OUT/"p04_family_wilcoxon.csv",index=False)
print("\nfamily-level (best-in-family per participant), Wilcoxon:")
print(ft.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

# ---------- P0-1 task-matched persistence skill + horizon ----------
# persistence(h): predict y_true(t-h) from the FULL raw series; evaluate on matched rows.
full=raw.set_index(["ID","t"])["vo2kg"]
def persist_rmse_pp(h):
    out={}
    for p,g in M.groupby("participant"):
        yt=g["y_true"].values; tt=g["t"].values
        prev=full.loc[p].reindex(tt-h).values
        ok=~np.isnan(prev)
        out[p]=np.sqrt(np.mean((yt[ok]-prev[ok])**2))
    return pd.Series(out)
def persist_rmse_pp_mask(h,mask):
    out={}; Mm=M[mask]
    for p,g in Mm.groupby("participant"):
        yt=g["y_true"].values; tt=g["t"].values; prev=full.loc[p].reindex(tt-h).values; ok=~np.isnan(prev)
        out[p]=np.sqrt(np.mean((yt[ok]-prev[ok])**2)) if ok.sum() else np.nan
    return pd.Series(out)

horizons=[1,2,3,5,8,10,15,20,30,45,60,90,120,180]
prh={h:persist_rmse_pp(h).reindex(sorted(M.participant.unique())).values for h in horizons}
persist_curve=pd.DataFrame({"horizon_s":horizons,
    "persist_rmse_mean":[prh[h].mean() for h in horizons],
    "persist_rmse_sd":[prh[h].std(ddof=1) for h in horizons]})
persist_curve.to_csv(OUT/"p01_persistence_curve.csv",index=False)

# skill of each model vs persistence-at-horizon; crossover = smallest h where model beats persistence
skill_rows=[]
focus=["RIDGE","RF","XGB","PATCHTST","TCN","GRU","LSTM","TFT","LINEAR"]
for m in ALL:
    mr=ppm[m].mean()
    cross=next((h for h in horizons if prh[h].mean()>=mr),None)
    # skill at a few horizons (overall) and in transitions at h=1
    s1=1-mr/prh[1].mean()
    skill_rows.append(dict(model=m,family=FAMILY[m],rmse=mr,
        persist_h1=prh[1].mean(),skill_vs_h1=s1,
        crossover_horizon_s=cross))
sk=pd.DataFrame(skill_rows).sort_values("rmse").reset_index(drop=True)
sk.to_csv(OUT/"p01_skill_scores.csv",index=False)
print("\n=== P0-1 PERSISTENCE & SKILL ===")
print("persistence RMSE by horizon (mean over participants):")
print(persist_curve.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print("\nper-model skill vs 1-step persistence + crossover horizon (s) where model beats persistence:")
print(sk[["model","family","rmse","skill_vs_h1","crossover_horizon_s"]].to_string(index=False,float_format=lambda x:f"{x:.3f}"))

# transition-restricted skill: model vs persistence(h=1) within top-tercile transition (oracle & causal)
tr_rows=[]
p1_trans_o=persist_rmse_pp_mask(1,trans_o).reindex(sorted(M.participant.unique())).values
p1_trans_c=persist_rmse_pp_mask(1,trans_c).reindex(sorted(M.participant.unique())).values
for m in ALL:
    mo=pp_rmse(m,trans_o).mean(); mc=pp_rmse(m,trans_c).mean()
    tr_rows.append(dict(model=m,family=FAMILY[m],
        trans_oracle_rmse=mo,persist_h1_trans_oracle=np.nanmean(p1_trans_o),skill_trans_oracle=1-mo/np.nanmean(p1_trans_o),
        trans_causal_rmse=mc,persist_h1_trans_causal=np.nanmean(p1_trans_c),skill_trans_causal=1-mc/np.nanmean(p1_trans_c)))
tr=pd.DataFrame(tr_rows).sort_values("trans_oracle_rmse").reset_index(drop=True)
tr.to_csv(OUT/"p01_transition_skill.csv",index=False)
print("\ntransition-window skill vs 1-step persistence (oracle & causal transition defs):")
print(tr[["model","family","trans_oracle_rmse","skill_trans_oracle","skill_trans_causal"]].to_string(index=False,float_format=lambda x:f"{x:.3f}"))

# plot horizon curve
fig,ax=plt.subplots(figsize=(7,4.5))
ax.plot(persist_curve.horizon_s,persist_curve.persist_rmse_mean,"-o",color="black",label="naive persistence (forecast horizon h)")
ax.fill_between(persist_curve.horizon_s,persist_curve.persist_rmse_mean-persist_curve.persist_rmse_sd,
                persist_curve.persist_rmse_mean+persist_curve.persist_rmse_sd,color="gray",alpha=.2)
for m,c in [("RIDGE","tab:blue"),("RF","tab:green"),("PATCHTST","tab:red")]:
    ax.axhline(ppm[m].mean(),ls="--",color=c,label=f"{m} (sensor estimate, RMSE {ppm[m].mean():.2f})")
ax.set_xlabel("persistence forecast horizon h (s)"); ax.set_ylabel("RMSE (mL/kg/min)")
ax.set_title("Predictability horizon: where sensor estimates beat naive persistence")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/"p01_horizon_curve.png",dpi=150)
print(f"\nsaved -> {OUT}/  (master_aligned.parquet, p04_*.csv, p01_*.csv, p01_horizon_curve.png)")
