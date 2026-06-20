"""
CAUSAL transition-window protocol (P1-2) for the wearable->VO2 benchmark.

Uses the deployable cadence-change detector `inp_trans` (NOT the oracle |dVO2/dt|)
to define transition vs steady regimes, then shows the transition penalty is robust
and agrees with the oracle (dvo2_s) labeling.

Outputs (to results/pmea_p1/):
  - causal_transition_per_model.csv      : per-model steady/transition RMSE + ratio (causal + oracle)
  - causal_transition_family.csv         : per-family penalty ratios + bootstrap CIs (causal + oracle)
  - causal_transition_summary.json       : detector-quality metrics + headline numbers
  - causal_transition_family.png         : grouped bar (steady vs transition RMSE by family)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DATA = r"results/pmea_p0/master_aligned.parquet"
OUTDIR = r"results/pmea_p1"
os.makedirs(OUTDIR, exist_ok=True)

MODELS = ["LINEAR", "RIDGE", "LASSO", "ELASTICNET", "RF", "XGB",
          "GRU", "LSTM", "TCN", "TFT", "PATCHTST"]
FAMILIES = {
    "Linear":   ["LINEAR", "RIDGE", "LASSO", "ELASTICNET"],
    "Tree":     ["RF", "XGB"],
    "Temporal": ["GRU", "LSTM", "TCN", "TFT", "PATCHTST"],
}
MODEL2FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}

RNG = np.random.default_rng(20260602)
N_BOOT = 5000


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def assign_regime(df, signal_col):
    """Top tercile of pooled signal = transition; bottom tercile = steady.
    Middle tercile dropped. Returns a Series with values {'steady','transition',NaN}."""
    lo, hi = df[signal_col].quantile([1 / 3, 2 / 3]).values
    regime = pd.Series(np.nan, index=df.index, dtype=object)
    regime[df[signal_col] <= lo] = "steady"
    regime[df[signal_col] >= hi] = "transition"
    return regime, lo, hi


def per_model_regime_rmse(df, regime_col):
    """Per-participant steady RMSE, transition RMSE, ratio for each model.
    Returns dict[model] -> dict with arrays (len 16) of steady, trans, ratio."""
    out = {}
    parts = sorted(df["participant"].unique())
    for m in MODELS:
        steady_v, trans_v, ratio_v = [], [], []
        for p in parts:
            g = df[df["participant"] == p]
            gs = g[g[regime_col] == "steady"]
            gt = g[g[regime_col] == "transition"]
            rs = rmse(gs[m].values, gs["y_true"].values)
            rt = rmse(gt[m].values, gt["y_true"].values)
            steady_v.append(rs)
            trans_v.append(rt)
            ratio_v.append(rt / rs)
        out[m] = {
            "steady": np.array(steady_v),
            "transition": np.array(trans_v),
            "ratio": np.array(ratio_v),
            "participants": np.array(parts),
        }
    return out


def family_arrays(per_model, family):
    """Stack per-participant values across the models in a family (participant x model),
    then average over models -> one value per participant for the family."""
    ms = FAMILIES[family]
    steady = np.mean([per_model[m]["steady"] for m in ms], axis=0)
    trans = np.mean([per_model[m]["transition"] for m in ms], axis=0)
    ratio = np.mean([per_model[m]["ratio"] for m in ms], axis=0)
    return steady, trans, ratio


def cluster_bootstrap_ci(per_participant_values, n_boot=N_BOOT, ci=0.95):
    """Subject-cluster bootstrap: resample participants with replacement,
    take the mean each time. per_participant_values is a 1-D array (len=16)."""
    n = len(per_participant_values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boots[b] = np.mean(per_participant_values[idx])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(np.mean(per_participant_values)), float(lo), float(hi)


def main():
    df = pd.read_parquet(DATA)
    assert df[MODELS + ["y_true", "inp_trans", "dvo2_s"]].isna().sum().sum() == 0

    # --- 1. Regime definitions ---
    df["regime_causal"], lo_c, hi_c = assign_regime(df, "inp_trans")
    df["regime_oracle"], lo_o, hi_o = assign_regime(df, "dvo2_s")

    n_steady_c = int((df["regime_causal"] == "steady").sum())
    n_trans_c = int((df["regime_causal"] == "transition").sum())
    n_steady_o = int((df["regime_oracle"] == "steady").sum())
    n_trans_o = int((df["regime_oracle"] == "transition").sum())

    # --- 2 & 3. Per-model regime RMSE (causal + oracle) ---
    pm_causal = per_model_regime_rmse(df, "regime_causal")
    pm_oracle = per_model_regime_rmse(df, "regime_oracle")

    # Per-model table
    rows = []
    for m in MODELS:
        rc, ro = pm_causal[m], pm_oracle[m]
        rows.append({
            "model": m,
            "family": MODEL2FAMILY[m],
            "causal_steady_rmse": float(np.mean(rc["steady"])),
            "causal_transition_rmse": float(np.mean(rc["transition"])),
            "causal_ratio": float(np.mean(rc["ratio"])),
            "oracle_steady_rmse": float(np.mean(ro["steady"])),
            "oracle_transition_rmse": float(np.mean(ro["transition"])),
            "oracle_ratio": float(np.mean(ro["ratio"])),
        })
    per_model_df = pd.DataFrame(rows)
    per_model_df.to_csv(os.path.join(OUTDIR, "causal_transition_per_model.csv"), index=False)

    # Per-family table with cluster bootstrap CIs
    fam_rows = []
    fam_plot = {}
    for fam in FAMILIES:
        s_c, t_c, r_c = family_arrays(pm_causal, fam)
        s_o, t_o, r_o = family_arrays(pm_oracle, fam)

        s_c_m, s_c_lo, s_c_hi = cluster_bootstrap_ci(s_c)
        t_c_m, t_c_lo, t_c_hi = cluster_bootstrap_ci(t_c)
        r_c_m, r_c_lo, r_c_hi = cluster_bootstrap_ci(r_c)

        s_o_m, s_o_lo, s_o_hi = cluster_bootstrap_ci(s_o)
        t_o_m, t_o_lo, t_o_hi = cluster_bootstrap_ci(t_o)
        r_o_m, r_o_lo, r_o_hi = cluster_bootstrap_ci(r_o)

        fam_rows.append({
            "family": fam,
            "causal_steady_rmse": s_c_m, "causal_steady_lo": s_c_lo, "causal_steady_hi": s_c_hi,
            "causal_transition_rmse": t_c_m, "causal_transition_lo": t_c_lo, "causal_transition_hi": t_c_hi,
            "causal_ratio": r_c_m, "causal_ratio_lo": r_c_lo, "causal_ratio_hi": r_c_hi,
            "oracle_steady_rmse": s_o_m, "oracle_steady_lo": s_o_lo, "oracle_steady_hi": s_o_hi,
            "oracle_transition_rmse": t_o_m, "oracle_transition_lo": t_o_lo, "oracle_transition_hi": t_o_hi,
            "oracle_ratio": r_o_m, "oracle_ratio_lo": r_o_lo, "oracle_ratio_hi": r_o_hi,
        })
        fam_plot[fam] = {
            "steady_m": s_c_m, "steady_lo": s_c_lo, "steady_hi": s_c_hi,
            "trans_m": t_c_m, "trans_lo": t_c_lo, "trans_hi": t_c_hi,
        }
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(os.path.join(OUTDIR, "causal_transition_family.csv"), index=False)

    # --- 4. Detector quality ---
    # AUC: event = "PatchTST beats Ridge" per sample (|PATCHTST-y| < |RIDGE-y|)
    err_patch = np.abs(df["PATCHTST"].values - df["y_true"].values)
    err_ridge = np.abs(df["RIDGE"].values - df["y_true"].values)
    event = (err_patch < err_ridge).astype(int)
    auc = float(roc_auc_score(event, df["inp_trans"].values))
    event_rate = float(event.mean())

    # Spearman corr(inp_trans, dvo2_s)
    rho, pval = spearmanr(df["inp_trans"].values, df["dvo2_s"].values)
    rho = float(rho)

    summary = {
        "n_rows": int(len(df)),
        "n_participants": int(df["participant"].nunique()),
        "regime_definition": {
            "causal_signal": "inp_trans",
            "causal_steady_threshold_le": float(lo_c),
            "causal_transition_threshold_ge": float(hi_c),
            "causal_n_steady": n_steady_c,
            "causal_n_transition": n_trans_c,
            "oracle_signal": "dvo2_s",
            "oracle_steady_threshold_le": float(lo_o),
            "oracle_transition_threshold_ge": float(hi_o),
            "oracle_n_steady": n_steady_o,
            "oracle_n_transition": n_trans_o,
        },
        "family_penalty_causal": {
            r["family"]: {"ratio": r["causal_ratio"],
                          "ci": [r["causal_ratio_lo"], r["causal_ratio_hi"]]}
            for r in fam_rows
        },
        "family_penalty_oracle": {
            r["family"]: {"ratio": r["oracle_ratio"],
                          "ci": [r["oracle_ratio_lo"], r["oracle_ratio_hi"]]}
            for r in fam_rows
        },
        "detector_quality": {
            "event": "|PATCHTST-y_true| < |RIDGE-y_true| (PatchTST beats Ridge)",
            "event_rate": event_rate,
            "auc_inp_trans": auc,
            "spearman_inp_trans_vs_dvo2_s": rho,
            "spearman_p": float(pval),
        },
    }
    with open(os.path.join(OUTDIR, "causal_transition_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- FIGURE: grouped bar steady vs transition RMSE by family (causal detector) ---
    fams = list(FAMILIES.keys())
    x = np.arange(len(fams))
    w = 0.38
    steady_m = [fam_plot[f]["steady_m"] for f in fams]
    trans_m = [fam_plot[f]["trans_m"] for f in fams]
    steady_err = [[fam_plot[f]["steady_m"] - fam_plot[f]["steady_lo"] for f in fams],
                  [fam_plot[f]["steady_hi"] - fam_plot[f]["steady_m"] for f in fams]]
    trans_err = [[fam_plot[f]["trans_m"] - fam_plot[f]["trans_lo"] for f in fams],
                 [fam_plot[f]["trans_hi"] - fam_plot[f]["trans_m"] for f in fams]]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    b1 = ax.bar(x - w / 2, steady_m, w, yerr=steady_err, capsize=4,
                color="#4C72B0", label="Steady (bottom tercile)", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, trans_m, w, yerr=trans_err, capsize=4,
                color="#C44E52", label="Transition (top tercile)", edgecolor="black", linewidth=0.5)
    for f, xi in zip(fams, x):
        r = fam_plot[f]["trans_m"] / fam_plot[f]["steady_m"]
        ytop = fam_plot[f]["trans_hi"]
        ax.text(xi + w / 2, ytop + 0.06, f"{r:.2f}x", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(fams)
    ax.set_ylabel(r"RMSE (mL kg$^{-1}$ min$^{-1}$)")
    ax.set_xlabel("Model family")
    ax.set_title("Error concentrates in transitions (causal cadence-change detector)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(trans_m) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "causal_transition_family.png"), dpi=150)
    plt.close(fig)

    # --- Console report ---
    print("=" * 70)
    print("CAUSAL TRANSITION-WINDOW PROTOCOL (P1-2)")
    print("=" * 70)
    print(f"Rows={len(df)}  participants={df['participant'].nunique()}")
    print(f"Causal (inp_trans): steady<= {lo_c:.3g}  transition>= {hi_c:.3g}  "
          f"(n_steady={n_steady_c}, n_trans={n_trans_c})")
    print(f"Oracle (dvo2_s):    steady<= {lo_o:.3g}  transition>= {hi_o:.3g}  "
          f"(n_steady={n_steady_o}, n_trans={n_trans_o})")
    print("-" * 70)
    print(f"{'Family':10s} | {'CAUSAL ratio [95% CI]':28s} | {'ORACLE ratio [95% CI]':28s}")
    for r in fam_rows:
        c = f"{r['causal_ratio']:.2f} [{r['causal_ratio_lo']:.2f},{r['causal_ratio_hi']:.2f}]"
        o = f"{r['oracle_ratio']:.2f} [{r['oracle_ratio_lo']:.2f},{r['oracle_ratio_hi']:.2f}]"
        print(f"{r['family']:10s} | {c:28s} | {o:28s}")
    print("-" * 70)
    print("Per-family steady -> transition RMSE (causal):")
    for r in fam_rows:
        print(f"  {r['family']:10s}: {r['causal_steady_rmse']:.3f} -> {r['causal_transition_rmse']:.3f}")
    print("-" * 70)
    print(f"Detector AUC (inp_trans -> PatchTST beats Ridge): {auc:.3f}  "
          f"(event rate={event_rate:.3f})")
    print(f"Spearman corr(inp_trans, dvo2_s): {rho:.3f}  (p={pval:.2e})")
    print("=" * 70)
    print("Wrote: causal_transition_per_model.csv, causal_transition_family.csv,")
    print("       causal_transition_summary.json, causal_transition_family.png")


if __name__ == "__main__":
    main()
