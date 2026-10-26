"""
P1-1 SIGNIFICANCE / statistics analysis for the evaluation-integrity wearable->VO2 benchmark.

Operates on the matched 16-fold-LOPO sample (results/pmea_p0/master_aligned.parquet).
Per-participant RMSE per model -> n=16 blocks.

Outputs (results/pmea_p1/):
  - p1_per_participant_rmse.csv      : 16 x 11 RMSE matrix
  - p1_pairwise_wilcoxon.csv         : all 55 pairs (W, raw p, Holm p, Cliff's delta)
  - p1_family_tests.csv              : family-level Wilcoxon (conservative + best-in-family)
  - p1_bootstrap_ci.csv              : subject-cluster bootstrap 95% CIs per model
  - p1_stats_table.csv               : compact per-model summary table
  - p1_headline.json                 : headline numbers
  - p1_bootstrap_ci.png              : forest plot of bootstrap CIs
"""

import os
import json
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
RNG_SEED = 20260601
DATA = "results/pmea_p0/master_aligned.parquet"
OUTDIR = "results/pmea_p1"
os.makedirs(OUTDIR, exist_ok=True)

MODELS = ["LINEAR", "RIDGE", "LASSO", "ELASTICNET", "RF", "XGB",
          "GRU", "LSTM", "TCN", "TFT", "PATCHTST"]

FAMILIES = {
    "Linear":   ["LINEAR", "RIDGE", "LASSO", "ELASTICNET"],
    "Tree":     ["RF", "XGB"],
    "Temporal": ["GRU", "LSTM", "TCN", "TFT", "PATCHTST"],
}


# ----------------------------------------------------------------------------
def cliffs_delta(a, b):
    """Cliff's delta for paired/independent samples a vs b.
    delta = P(a>b) - P(a<b). Here a,b are per-participant RMSE vectors.
    For RMSE, lower is better, so a positive delta means a has LARGER errors
    (i.e. a is worse). We compute the raw definition; interpretation noted in caller.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    n = len(a) * len(b)
    return (gt - lt) / n


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down correction. Returns corrected p-values
    (same order as input), clipped to 1.0 and enforced monotone non-decreasing."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)  # enforce monotonicity
        corrected[idx] = running_max
    return corrected


# ----------------------------------------------------------------------------
def main():
    df = pd.read_parquet(DATA)
    assert df["participant"].nunique() == 16, "expected 16 participants"

    # ---- per-participant RMSE (n=16 blocks) --------------------------------
    def per_block_rmse(g):
        return pd.Series({m: np.sqrt(np.mean((g[m].values - g["y_true"].values) ** 2))
                          for m in MODELS})

    rmse = (df.groupby("participant", sort=True)
              .apply(per_block_rmse, include_groups=False))
    rmse = rmse[MODELS]  # column order
    rmse.to_csv(os.path.join(OUTDIR, "p1_per_participant_rmse.csv"))
    n_blocks = rmse.shape[0]
    mean_rmse = rmse.mean()

    # ---- 1. Friedman omnibus ----------------------------------------------
    fr_stat, fr_p = stats.friedmanchisquare(*[rmse[m].values for m in MODELS])
    fr_df = len(MODELS) - 1

    # ---- 2. pairwise Wilcoxon (55 pairs) + Cliff's delta ------------------
    pairs = list(itertools.combinations(MODELS, 2))
    rows = []
    for a, b in pairs:
        va, vb = rmse[a].values, rmse[b].values
        # Wilcoxon signed-rank on paired per-participant RMSE differences.
        # wilcox='wilcox' default; handle the all-zero-diff edge gracefully.
        try:
            w_stat, w_p = stats.wilcoxon(va, vb, zero_method="wilcox",
                                         alternative="two-sided")
        except ValueError:
            w_stat, w_p = np.nan, 1.0
        delta = cliffs_delta(va, vb)  # >0 means 'a' has larger RMSE (a worse)
        rows.append({
            "model_a": a, "model_b": b,
            "mean_rmse_a": mean_rmse[a], "mean_rmse_b": mean_rmse[b],
            "W": w_stat, "p_raw": w_p,
            "cliffs_delta": delta, "abs_cliffs_delta": abs(delta),
        })
    pw = pd.DataFrame(rows)
    pw["p_holm"] = holm_bonferroni(pw["p_raw"].values)
    pw["survives_holm_0.05"] = pw["p_holm"] < 0.05

    # family membership lookup for delta interpretation
    fam_of = {}
    for fam, members in FAMILIES.items():
        for m in members:
            fam_of[m] = fam
    pw["fam_a"] = pw["model_a"].map(fam_of)
    pw["fam_b"] = pw["model_b"].map(fam_of)
    pw.sort_values("p_holm", inplace=True)
    pw.to_csv(os.path.join(OUTDIR, "p1_pairwise_wilcoxon.csv"), index=False)

    n_survive = int(pw["survives_holm_0.05"].sum())
    smallest = pw.iloc[0]
    smallest_pair = (smallest["model_a"], smallest["model_b"])
    smallest_holm = float(smallest["p_holm"])
    smallest_raw = float(smallest["p_raw"])

    # Cliff's delta range across all pairs
    delta_min = float(pw["cliffs_delta"].min())
    delta_max = float(pw["cliffs_delta"].max())
    abs_delta_max = float(pw["abs_cliffs_delta"].max())

    # largest |delta| among temporal-vs-linear pairs
    tl_mask = (((pw["fam_a"] == "Temporal") & (pw["fam_b"] == "Linear")) |
               ((pw["fam_a"] == "Linear") & (pw["fam_b"] == "Temporal")))
    tl = pw[tl_mask].copy()
    tl_row = tl.sort_values("abs_cliffs_delta", ascending=False).iloc[0]
    tl_max_abs_delta = float(tl_row["abs_cliffs_delta"])
    tl_max_pair = (tl_row["model_a"], tl_row["model_b"])
    tl_max_delta_signed = float(tl_row["cliffs_delta"])

    # ---- 3. family-level tests --------------------------------------------
    # Conservative: per-participant FAMILY-MEAN RMSE (mean across members).
    fam_mean = pd.DataFrame(
        {fam: rmse[members].mean(axis=1) for fam, members in FAMILIES.items()},
        index=rmse.index)
    # Best-in-family: per-participant min RMSE across members (best member,
    # selected per participant -> optimistic / favours larger families).
    fam_best = pd.DataFrame(
        {fam: rmse[members].min(axis=1) for fam, members in FAMILIES.items()},
        index=rmse.index)

    fam_pairs = [("Temporal", "Linear"), ("Temporal", "Tree"), ("Tree", "Linear")]
    fam_rows = []
    for f1, f2 in fam_pairs:
        # conservative (family-mean)
        w1, p1 = stats.wilcoxon(fam_mean[f1].values, fam_mean[f2].values,
                                alternative="two-sided")
        d1 = cliffs_delta(fam_mean[f1].values, fam_mean[f2].values)
        # best-in-family
        w2, p2 = stats.wilcoxon(fam_best[f1].values, fam_best[f2].values,
                                alternative="two-sided")
        d2 = cliffs_delta(fam_best[f1].values, fam_best[f2].values)
        fam_rows.append({
            "family_a": f1, "family_b": f2,
            "mean_rmse_a_conservative": fam_mean[f1].mean(),
            "mean_rmse_b_conservative": fam_mean[f2].mean(),
            "W_conservative": w1, "p_conservative": p1,
            "cliffs_delta_conservative": d1,
            "mean_rmse_a_bestinfam": fam_best[f1].mean(),
            "mean_rmse_b_bestinfam": fam_best[f2].mean(),
            "W_bestinfam": w2, "p_bestinfam": p2,
            "cliffs_delta_bestinfam": d2,
        })
    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(os.path.join(OUTDIR, "p1_family_tests.csv"), index=False)

    # ---- 4. subject-cluster bootstrap 95% CI -------------------------------
    rng = np.random.default_rng(RNG_SEED)
    n_boot = 2000
    rmse_mat = rmse.values  # (16, 11), rows = participants
    n_part = rmse_mat.shape[0]
    boot_means = np.empty((n_boot, len(MODELS)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_part, size=n_part)  # resample participants
        boot_means[b] = rmse_mat[idx].mean(axis=0)
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)
    boot_sd = boot_means.std(axis=0, ddof=1)

    boot_df = pd.DataFrame({
        "model": MODELS,
        "mean_rmse": mean_rmse.values,
        "boot_ci_lo": ci_lo,
        "boot_ci_hi": ci_hi,
        "boot_se": boot_sd,
    }).sort_values("mean_rmse").reset_index(drop=True)
    boot_df.to_csv(os.path.join(OUTDIR, "p1_bootstrap_ci.csv"), index=False)

    # ---- compact stats table ----------------------------------------------
    rank = mean_rmse.rank().astype(int)
    stats_tab = pd.DataFrame({
        "model": MODELS,
        "family": [fam_of[m] for m in MODELS],
        "mean_rmse": mean_rmse.values,
        "sd_rmse": rmse.std(axis=0, ddof=1).values,
        "boot_ci_lo": ci_lo,
        "boot_ci_hi": ci_hi,
        "rank": rank.values,
    }).sort_values("mean_rmse").reset_index(drop=True)
    stats_tab.to_csv(os.path.join(OUTDIR, "p1_stats_table.csv"), index=False)

    # ---- forest plot of bootstrap CIs -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    bd = boot_df.iloc[::-1].reset_index(drop=True)  # best at top
    ypos = np.arange(len(bd))
    fam_color = {"Linear": "#1f77b4", "Tree": "#2ca02c", "Temporal": "#d62728"}
    colors = [fam_color[fam_of[m]] for m in bd["model"]]
    ax.errorbar(bd["mean_rmse"], ypos,
                xerr=[bd["mean_rmse"] - bd["boot_ci_lo"],
                      bd["boot_ci_hi"] - bd["mean_rmse"]],
                fmt="none", ecolor="gray", elinewidth=1.4, capsize=3, zorder=1)
    ax.scatter(bd["mean_rmse"], ypos, c=colors, s=55, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(bd["model"])
    ax.set_xlabel(r"Per-participant mean RMSE (mL kg$^{-1}$ min$^{-1}$)")
    ax.set_title("Subject-cluster bootstrap 95% CIs (n=16 LOPO blocks, 2000 resamples)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=k,
                          markerfacecolor=v, markersize=9)
               for k, v in fam_color.items()]
    ax.legend(handles=handles, title="Family", loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "p1_bootstrap_ci.png"), dpi=150)
    plt.close(fig)

    # ---- headline JSON -----------------------------------------------------
    headline = {
        "n_lopo_blocks": int(n_blocks),
        "n_models": len(MODELS),
        "n_pairs": len(pairs),
        "leaderboard_mean_rmse": {m: float(mean_rmse[m]) for m in MODELS},
        "friedman": {
            "chi2": float(fr_stat),
            "df": int(fr_df),
            "p": float(fr_p),
            "significant_0.05": bool(fr_p < 0.05),
        },
        "pairwise_wilcoxon_holm": {
            "n_pairs": len(pairs),
            "n_surviving_holm_0.05": n_survive,
            "smallest_holm_p": smallest_holm,
            "smallest_holm_pair": list(smallest_pair),
            "smallest_raw_p_for_that_pair": smallest_raw,
        },
        "cliffs_delta": {
            "range_signed": [delta_min, delta_max],
            "max_abs_overall": abs_delta_max,
            "temporal_vs_linear_max_abs": tl_max_abs_delta,
            "temporal_vs_linear_max_abs_pair": list(tl_max_pair),
            "temporal_vs_linear_max_abs_signed": tl_max_delta_signed,
        },
        "family_tests": {},
        "bootstrap": {
            "n_resamples": n_boot,
            "best_model": str(boot_df.iloc[0]["model"]),
            "best_ci": [float(boot_df.iloc[0]["boot_ci_lo"]),
                        float(boot_df.iloc[0]["boot_ci_hi"])],
        },
    }
    for _, r in fam_df.iterrows():
        key = f"{r['family_a']}_vs_{r['family_b']}"
        headline["family_tests"][key] = {
            "p_conservative": float(r["p_conservative"]),
            "p_bestinfam": float(r["p_bestinfam"]),
            "cliffs_delta_conservative": float(r["cliffs_delta_conservative"]),
            "cliffs_delta_bestinfam": float(r["cliffs_delta_bestinfam"]),
        }

    with open(os.path.join(OUTDIR, "p1_headline.json"), "w") as f:
        json.dump(headline, f, indent=2)

    # ---- console summary ---------------------------------------------------
    print("=" * 70)
    print(f"n LOPO blocks = {n_blocks}, n models = {len(MODELS)}, n pairs = {len(pairs)}")
    print("-" * 70)
    print(f"FRIEDMAN: chi2 = {fr_stat:.3f}, df = {fr_df}, p = {fr_p:.3e} "
          f"(sig={fr_p < 0.05})")
    print("-" * 70)
    print(f"PAIRWISE Wilcoxon + Holm: {n_survive} / {len(pairs)} survive @0.05")
    print(f"  smallest Holm p = {smallest_holm:.4f} for "
          f"{smallest_pair[0]} vs {smallest_pair[1]} (raw p = {smallest_raw:.4f})")
    print(f"  Cliff's delta range = [{delta_min:.3f}, {delta_max:.3f}], "
          f"max |delta| = {abs_delta_max:.3f}")
    print(f"  Temporal-vs-Linear max |delta| = {tl_max_abs_delta:.3f} "
          f"({tl_max_pair[0]} vs {tl_max_pair[1]})")
    print("-" * 70)
    print("FAMILY tests (conservative family-mean | best-in-family):")
    for _, r in fam_df.iterrows():
        print(f"  {r['family_a']} vs {r['family_b']}: "
              f"p_cons = {r['p_conservative']:.4f} (d={r['cliffs_delta_conservative']:+.3f}) | "
              f"p_best = {r['p_bestinfam']:.4f} (d={r['cliffs_delta_bestinfam']:+.3f})")
    print("-" * 70)
    print("BOOTSTRAP 95% CIs (sorted best->worst):")
    for _, r in boot_df.iterrows():
        print(f"  {r['model']:<11} {r['mean_rmse']:.3f}  "
              f"[{r['boot_ci_lo']:.3f}, {r['boot_ci_hi']:.3f}]")
    print("=" * 70)
    print("OK")


if __name__ == "__main__":
    main()
