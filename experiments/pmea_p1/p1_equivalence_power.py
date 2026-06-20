"""
P1-5: SUBJECT-FOLD EQUIVALENCE / POWER analysis.

Goal: convert "no model is better" from "underpowered" into "informatively null".

Unit of analysis: the 16 LOPO held-out PARTICIPANTS (per-participant RMSE).
Justification vs the 1 Hz autocorrelation objection: per-second samples WITHIN a
participant are strongly autocorrelated and are NOT independent, but the 16 held-out
participants are mutually independent (each participant's RMSE is computed from a
disjoint set of rows; participants never share data across folds in LOPO). We therefore
treat the 16 per-participant RMSE values as the i.i.d. sampling unit and use paired tests
across the matched 16-fold sample.

Computes:
  1. Minimum Detectable Effect (MDE) at n=16, alpha=0.05, power=0.80 (paired, two-sided),
     using the OBSERVED SD of paired RMSE differences for the key contrasts.
  2. TOST equivalence tests at margins +/-0.25 and +/-0.5 mL/kg/min.
  3. Achieved (post-hoc) power to detect the observed best-vs-tree gap.

Outputs (results/pmea_p1/):
  - p1_equivalence_power.json
  - p1_per_participant_rmse.csv
  - p1_contrasts.csv
  - p1_equivalence_power.png
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "results/pmea_p0/master_aligned.parquet"
OUTDIR = "results/pmea_p1"
os.makedirs(OUTDIR, exist_ok=True)

MODELS = ["LINEAR", "RIDGE", "LASSO", "ELASTICNET", "RF", "XGB",
          "GRU", "LSTM", "TCN", "TFT", "PATCHTST"]
FAMILIES = {
    "Linear": ["LINEAR", "RIDGE", "LASSO", "ELASTICNET"],
    "Tree": ["RF", "XGB"],
    "Temporal": ["GRU", "LSTM", "TCN", "TFT", "PATCHTST"],
}

ALPHA = 0.05
POWER = 0.80
N = 16
MARGINS = [0.25, 0.50]  # equivalence bounds in mL/kg/min

# ---------------------------------------------------------------------------
# Load and build per-participant RMSE table (16 x 11)
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA)
assert df["participant"].nunique() == N, "expected 16 participants"


def per_participant_rmse(frame, model):
    err = frame[model].to_numpy() - frame["y_true"].to_numpy()
    return np.sqrt(np.mean(err ** 2))


parts = sorted(df["participant"].unique())
rmse_rows = []
for p in parts:
    g = df[df["participant"] == p]
    row = {"participant": p}
    for m in MODELS:
        row[m] = per_participant_rmse(g, m)
    rmse_rows.append(row)
R = pd.DataFrame(rmse_rows).set_index("participant")
R.to_csv(os.path.join(OUTDIR, "p1_per_participant_rmse.csv"))

# family means per participant (mean RMSE across that family's models, within participant)
Rfam = pd.DataFrame(index=R.index)
for fam, mlist in FAMILIES.items():
    Rfam[fam] = R[mlist].mean(axis=1)

leaderboard = R.mean().sort_values()
print("Per-participant mean RMSE leaderboard:")
print(leaderboard.round(3).to_string())
print()


# ---------------------------------------------------------------------------
# Helper statistics on paired differences
# ---------------------------------------------------------------------------
def cohen_dz_from_diff(d):
    """Paired effect size dz = mean(d)/sd(d), sd with ddof=1."""
    sd = np.std(d, ddof=1)
    return np.mean(d) / sd if sd > 0 else np.nan


def mde_paired(sd_diff, n=N, alpha=ALPHA, power=POWER):
    """
    Minimum detectable mean paired difference for a two-sided paired t-test,
    using the OBSERVED SD of paired differences (sd_diff, ddof=1).
    Solves for delta such that the noncentral-t power equals `power`.
    Returns MDE in the same units as sd_diff (mL/kg/min).
    """
    df_t = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df_t)
    se = sd_diff / np.sqrt(n)

    # Solve for the noncentrality parameter ncp such that
    # P(T' > t_crit) + P(T' < -t_crit) = power, with T' noncentral-t(df, ncp).
    def power_at_ncp(ncp):
        # two-sided power
        upper = stats.nct.sf(t_crit, df_t, ncp)      # P(T' > t_crit)
        lower = stats.nct.cdf(-t_crit, df_t, ncp)    # P(T' < -t_crit)
        return upper + lower

    lo, hi = 0.0, 50.0
    # ensure bracket
    while power_at_ncp(hi) < power and hi < 1e6:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power_at_ncp(mid) < power:
            lo = mid
        else:
            hi = mid
    ncp = 0.5 * (lo + hi)
    # ncp = delta / se  ->  delta = ncp * se
    mde = ncp * se
    return mde, ncp


def achieved_power_paired(mean_diff, sd_diff, n=N, alpha=ALPHA):
    """Post-hoc two-sided power for the observed effect (noncentral-t)."""
    df_t = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df_t)
    se = sd_diff / np.sqrt(n)
    ncp = mean_diff / se
    upper = stats.nct.sf(t_crit, df_t, ncp)
    lower = stats.nct.cdf(-t_crit, df_t, ncp)
    return upper + lower


def tost_paired(d, margin):
    """
    Two One-Sided Tests for equivalence on paired differences d, within +/- margin.
    H0a: mean >= +margin ; H0b: mean <= -margin.
    Reject both (equivalence) when both one-sided p < alpha.
    Returns dict with the two one-sided p-values, the binding (max) p, and verdict.
    """
    n = len(d)
    df_t = n - 1
    mean = np.mean(d)
    sd = np.std(d, ddof=1)
    se = sd / np.sqrt(n)
    # lower test: H0: mean <= -margin  vs Ha: mean > -margin
    t_lower = (mean - (-margin)) / se
    p_lower = stats.t.sf(t_lower, df_t)          # P(T > t_lower)
    # upper test: H0: mean >= +margin vs Ha: mean < +margin
    t_upper = (mean - margin) / se
    p_upper = stats.t.cdf(t_upper, df_t)         # P(T < t_upper)
    p_tost = max(p_lower, p_upper)
    return {
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "t_lower": t_lower,
        "p_lower": p_lower,
        "t_upper": t_upper,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < ALPHA),
        "margin": margin,
    }


def paired_ttest(d):
    """Standard two-sided paired t-test summary + 95% CI of the mean diff."""
    n = len(d)
    df_t = n - 1
    mean = np.mean(d)
    sd = np.std(d, ddof=1)
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df_t)
    tcrit = stats.t.ppf(1 - ALPHA / 2, df_t)
    ci = (mean - tcrit * se, mean + tcrit * se)
    return {"mean_diff": mean, "sd_diff": sd, "se_diff": se,
            "t": t, "p": p, "ci95_lo": ci[0], "ci95_hi": ci[1]}


# ---------------------------------------------------------------------------
# Define the key contrasts.
# Convention for the paired difference d = RMSE(A) - RMSE(B):
#   A = "candidate"/better model, B = "baseline". Negative mean => A lower error.
# ---------------------------------------------------------------------------
contrasts = {
    # best Temporal (TCN) vs best Tree (XGBoost; lower per-participant mean RMSE than RF)
    "TCN_vs_XGB": (R["TCN"], R["XGB"]),
    # best Temporal (TCN) vs RIDGE (best linear) -- positive control (temporal beats linear)
    "TCN_vs_RIDGE": (R["TCN"], R["RIDGE"]),
    # best Tree (XGBoost) vs RIDGE (best linear) -- positive control (tree beats linear)
    "XGB_vs_RIDGE": (R["XGB"], R["RIDGE"]),
    # Temporal family mean vs Tree family mean
    "Temporal_vs_Tree": (Rfam["Temporal"], Rfam["Tree"]),
}

results = {}
contrast_table = []
for name, (a, b) in contrasts.items():
    d = (a - b).to_numpy()          # A - B, paired across 16 participants
    mean_a, mean_b = float(a.mean()), float(b.mean())
    observed_gap = float(mean_a - mean_b)           # signed (A - B)
    observed_gap_abs = abs(observed_gap)

    tt = paired_ttest(d)
    sd_diff = tt["sd_diff"]
    dz = cohen_dz_from_diff(d)

    mde, ncp80 = mde_paired(sd_diff)
    ach_power = achieved_power_paired(abs(tt["mean_diff"]), sd_diff)

    tost_by_margin = {}
    for m in MARGINS:
        tost_by_margin[f"{m:.2f}"] = tost_paired(d, m)

    results[name] = {
        "A": name.split("_vs_")[0],
        "B": name.split("_vs_")[1],
        "mean_RMSE_A": mean_a,
        "mean_RMSE_B": mean_b,
        "observed_gap_signed_A_minus_B": observed_gap,
        "observed_gap_abs": observed_gap_abs,
        "sd_paired_diff": sd_diff,
        "se_paired_diff": tt["se_diff"],
        "cohen_dz": dz,
        "paired_t": tt["t"],
        "paired_p_twosided": tt["p"],
        "ci95_meandiff": [tt["ci95_lo"], tt["ci95_hi"]],
        "MDE_abs_mean_diff": mde,
        "MDE_ncp_at_power80": ncp80,
        "observed_gap_below_MDE": bool(observed_gap_abs < mde),
        "achieved_power_for_observed_gap": ach_power,
        "TOST": tost_by_margin,
    }

    contrast_table.append({
        "contrast": name,
        "mean_RMSE_A": round(mean_a, 4),
        "mean_RMSE_B": round(mean_b, 4),
        "observed_gap_abs": round(observed_gap_abs, 4),
        "sd_paired_diff": round(sd_diff, 4),
        "cohen_dz": round(dz, 4),
        "paired_p": round(tt["p"], 4),
        "MDE": round(mde, 4),
        "gap<MDE": observed_gap_abs < mde,
        "achieved_power": round(ach_power, 4),
        "TOST_p_0.25": round(tost_by_margin["0.25"]["p_tost"], 4),
        "equiv_0.25": tost_by_margin["0.25"]["equivalent"],
        "TOST_p_0.50": round(tost_by_margin["0.50"]["p_tost"], 4),
        "equiv_0.50": tost_by_margin["0.50"]["equivalent"],
    })

ct = pd.DataFrame(contrast_table)
ct.to_csv(os.path.join(OUTDIR, "p1_contrasts.csv"), index=False)

# ---------------------------------------------------------------------------
# Assemble JSON
# ---------------------------------------------------------------------------
out = {
    "meta": {
        "n_participants": N,
        "alpha": ALPHA,
        "target_power": POWER,
        "margins_mL_kg_min": MARGINS,
        "unit_of_analysis": "per-participant RMSE over 16 LOPO held-out participants",
        "iid_justification": (
            "Per-second (1 Hz) samples within a participant are strongly "
            "autocorrelated and are not independent; however the 16 held-out "
            "participants are mutually independent (LOPO folds use disjoint rows "
            "and no participant contributes to another's held-out RMSE). Each "
            "per-participant RMSE aggregates the within-participant autocorrelation "
            "into a single number, so the 16 RMSE values are treated as the i.i.d. "
            "sampling unit for paired inference."
        ),
        "data": DATA,
    },
    "leaderboard_per_participant_mean_RMSE": leaderboard.round(4).to_dict(),
    "contrasts": results,
}
with open(os.path.join(OUTDIR, "p1_equivalence_power.json"), "w") as f:
    json.dump(out, f, indent=2)

# ---------------------------------------------------------------------------
# Figure: observed gap vs MDE, plus TOST CI vs equivalence margins
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: observed |gap| vs MDE
names = list(results.keys())
labels = ["TCN vs XGBoost\n(Temporal best vs Tree best)",
          "TCN vs Ridge\n(Temporal best vs Linear)",
          "XGBoost vs Ridge\n(Tree best vs Linear)",
          "Temporal vs Tree\n(family means)"]
gaps = [results[n]["observed_gap_abs"] for n in names]
mdes = [results[n]["MDE_abs_mean_diff"] for n in names]
x = np.arange(len(names))
w = 0.38
axA = axes[0]
axA.bar(x - w / 2, gaps, w, label="Observed |gap|", color="#4C72B0")
axA.bar(x + w / 2, mdes, w, label="MDE (80% power)", color="#C44E52", alpha=0.85)
axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=7.5)
axA.set_ylabel(r"$\Delta$RMSE (mL kg$^{-1}$ min$^{-1}$)")
axA.set_title("A. Observed gap vs minimum detectable effect", fontsize=10)
axA.legend(fontsize=8, loc="upper left")
for xi, g, m in zip(x, gaps, mdes):
    axA.text(xi - w / 2, g + 0.005, f"{g:.3f}", ha="center", va="bottom", fontsize=7)
    axA.text(xi + w / 2, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=7)

# Panel B: mean diff with 95% CI vs equivalence margins (0.25, 0.50)
axB = axes[1]
for i, n in enumerate(names):
    md = results[n]["TOST"]["0.25"]["mean_diff"]
    ci = results[n]["ci95_meandiff"]
    axB.errorbar(md, i, xerr=[[md - ci[0]], [ci[1] - md]], fmt="o",
                 color="#4C72B0", capsize=4, markersize=6)
for marg, c, ls in zip(MARGINS, ["#C44E52", "#DD8452"], ["-", "--"]):
    axB.axvline(+marg, color=c, ls=ls, lw=1, label=f"$\\pm${marg:.2f} margin")
    axB.axvline(-marg, color=c, ls=ls, lw=1)
axB.axvline(0, color="grey", lw=0.8)
axB.set_yticks(range(len(names)))
axB.set_yticklabels([l.split("\n")[0] for l in labels], fontsize=8)
axB.set_xlabel(r"Mean paired $\Delta$RMSE, A$-$B (mL kg$^{-1}$ min$^{-1}$)")
axB.set_title("B. Mean difference (95% CI) vs equivalence margins", fontsize=10)
axB.legend(fontsize=7.5, loc="lower right")
axB.set_ylim(-0.6, len(names) - 0.4)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "p1_equivalence_power.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("=" * 78)
for n in names:
    r = results[n]
    print(f"\n[{n}]  A={r['A']} ({r['mean_RMSE_A']:.3f})  B={r['B']} ({r['mean_RMSE_B']:.3f})")
    print(f"  observed |gap|           = {r['observed_gap_abs']:.4f} mL/kg/min "
          f"(signed A-B = {r['observed_gap_signed_A_minus_B']:+.4f})")
    print(f"  SD of paired diff        = {r['sd_paired_diff']:.4f}  (dz = {r['cohen_dz']:+.3f})")
    print(f"  paired t-test p (2-sided)= {r['paired_p_twosided']:.4f}")
    print(f"  MDE (80% power)          = {r['MDE_abs_mean_diff']:.4f} mL/kg/min")
    print(f"  observed gap < MDE?      = {r['observed_gap_below_MDE']}")
    print(f"  achieved power (obs gap) = {r['achieved_power_for_observed_gap']:.4f}")
    for m in MARGINS:
        t = r["TOST"][f"{m:.2f}"]
        print(f"  TOST +/-{m:.2f}: p_lower={t['p_lower']:.4f} p_upper={t['p_upper']:.4f} "
              f"p_TOST={t['p_tost']:.4f} equivalent={t['equivalent']}")
print("\n" + "=" * 78)
print("Wrote:")
for f in ["p1_equivalence_power.json", "p1_per_participant_rmse.csv",
          "p1_contrasts.csv", "p1_equivalence_power.png"]:
    print("  ", os.path.join(OUTDIR, f))
print("DONE")
