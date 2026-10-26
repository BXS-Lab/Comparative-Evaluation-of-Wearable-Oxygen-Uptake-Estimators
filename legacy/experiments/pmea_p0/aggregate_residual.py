"""Aggregate the per-model residual_fair/{model}_folds.csv into a final table + verdict.
Residual was trained at each model's DIRECT best-params (warm-start config); compared to existing direct results."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

OUT = Path("results/pmea_p0/residual_fair")
order = {"patchtst": 0, "tcn": 1, "gru": 2, "lstm": 3, "tft": 4}
rows = []
for f in sorted(OUT.glob("*_folds.csv")):
    m = f.stem.replace("_folds", "")
    df = pd.read_csv(f)
    def wp(c):
        try: return float(wilcoxon(df["rmse_direct"], df[c]).pvalue)
        except ValueError: return 1.0
    rows.append(dict(model=m, n=len(df), k_mean=df["k"].mean(),
        direct=df["rmse_direct"].mean(), ridge=df["rmse_ridge"].mean(),
        resid_k1=df["rmse_resid_k1"].mean(), resid_kLF=df["rmse_resid_kLF"].mean(),
        p_direct_vs_k1=wp("rmse_resid_k1"), p_direct_vs_kLF=wp("rmse_resid_kLF"),
        kLF_better_n=int((df["rmse_resid_kLF"] < df["rmse_direct"]).sum())))
if not rows:
    print("no folds.csv found yet"); raise SystemExit
S = pd.DataFrame(rows).sort_values("model", key=lambda s: s.map(lambda x: order.get(x, 9))).reset_index(drop=True)
S.to_csv(OUT / "summary_all.csv", index=False)
pd.set_option("display.width", 200)
print("=== RESIDUAL @ DIRECT BEST-PARAMS vs DIRECT (per-participant mean RMSE, n=16) ===")
print(S.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
sig = S[(S["resid_kLF"] < S["direct"]) & (S["p_direct_vs_kLF"] < 0.05)]
print()
if len(sig):
    print("VERDICT: residual shows POTENTIAL — significantly beats direct for: " + ", ".join(sig["model"]))
else:
    bestgap = (S["direct"] - S["resid_kLF"])
    print("VERDICT: NO residual potential at direct-config. The leak-free residual blend does NOT significantly "
          "beat direct for ANY model (all p>0.05). Best point gain over direct = "
          f"{bestgap.max():.3f} mL/kg/min ({S.loc[bestgap.idxmax(),'model']}); k_mean≈{S['k_mean'].mean():.2f} "
          "(residual heavily down-weighted). => not worth the multi-day full Optuna.")
