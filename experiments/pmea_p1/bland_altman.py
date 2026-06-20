"""
P1-4: Bland-Altman agreement analysis vs the gold standard (indirect calorimetry).

For one representative model per family:
    Temporal = PATCHTST, Tree = RF, Linear = RIDGE.

For each model:
  1. d = predicted - measured (y_true) per sample; bias = mean(d).
  2. Limits of agreement (LoA) two ways:
       (a) NAIVE per-sample LoA = bias +/- 1.96 * SD(d) over all ~60k samples
           (understates uncertainty -- within-participant samples are autocorrelated).
       (b) VARIANCE-COMPONENT / repeated-measures LoA (Bland & Altman 1999/2007 for
           repeated measurements). total SD = sqrt(s_between^2 + s_within^2), where
             s_between^2 = variance of the 16 per-participant mean differences,
                           corrected so it estimates the true between-subject component
                           (subtracting the sampling variance of each subject mean);
             s_within^2  = pooled within-participant residual variance of d.
       LoA = bias +/- 1.96 * total SD.
  3. Per-participant bias range (min..max over the 16 subjects) -> heterogeneity.

Cross-check (b) with a linear mixed-effects model (random intercept per participant):
  total SD = sqrt(tau^2 + sigma^2). Reported alongside for robustness.

Figure: Bland-Altman plot for PATCHTST (difference vs mean of predicted & measured),
2D histogram to reduce overplotting, bias line + variance-component LoA bands.

Run from the repository root with `python`.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = r"results/pmea_p0/master_aligned.parquet"
OUTDIR = r"results/pmea_p1"
os.makedirs(OUTDIR, exist_ok=True)

REPS = {"Temporal": "PATCHTST", "Tree": "RF", "Linear": "RIDGE"}
Z = 1.959964  # 1.96 for 95% LoA


def variance_component_loa(df, model):
    """
    Bland-Altman LoA for repeated measurements (multiple obs per subject,
    one underlying 'true' value per subject is NOT assumed; here each subject
    contributes a cluster of paired measurements).

    Decompose the total variance of d = pred - measured into between-subject
    and within-subject components via a one-way random-effects ANOVA on subject.

    Following Bland & Altman (1999), Stat Methods Med Res 8:135-160, eq. for
    'replicate measurements / repeated measures' where the quantity differs
    within subject:
        s_w^2  = within-subject (residual) MS = SS_within / (N - k)
        between-subject true variance:
            s_b^2 = (MS_between - MS_within) / n0
        total SD of a single difference = sqrt(s_b^2 + s_w^2)
    where n0 corrects for unequal cluster sizes:
        n0 = (1/(k-1)) * (N - sum(n_i^2)/N)
    """
    g = df.groupby("participant")
    d_all = (df[model] - df["y_true"]).to_numpy()
    N = len(d_all)
    grand_mean = d_all.mean()

    subj_means = g.apply(lambda x: (x[model] - x["y_true"]).mean(), include_groups=False)
    n_i = g.size().to_numpy()
    k = len(n_i)

    # Between-subject sum of squares (weighted by cluster size)
    ss_between = float(np.sum(n_i * (subj_means.to_numpy() - grand_mean) ** 2))
    # Within-subject sum of squares
    ss_within = 0.0
    for pid, x in g:
        di = (x[model] - x["y_true"]).to_numpy()
        ss_within += float(np.sum((di - di.mean()) ** 2))

    df_between = k - 1
    df_within = N - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within  # = s_within^2 (pooled within-subject variance)

    # unequal-cluster-size correction
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)

    s_within2 = ms_within
    s_between2 = (ms_between - ms_within) / n0
    s_between2_clamped = max(s_between2, 0.0)  # variance cannot be negative

    total_sd = np.sqrt(s_between2_clamped + s_within2)

    return {
        "bias": float(grand_mean),
        "s_within2": float(s_within2),
        "s_within": float(np.sqrt(s_within2)),
        "s_between2": float(s_between2),
        "s_between2_clamped": float(s_between2_clamped),
        "s_between": float(np.sqrt(s_between2_clamped)),
        "total_sd": float(total_sd),
        "loa_lower": float(grand_mean - Z * total_sd),
        "loa_upper": float(grand_mean + Z * total_sd),
        "n0": float(n0),
        "ms_between": float(ms_between),
        "ms_within": float(ms_within),
        "subj_means": subj_means,
    }


def naive_loa(df, model):
    d = (df[model] - df["y_true"]).to_numpy()
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": float(bias - Z * sd),
        "loa_upper": float(bias + Z * sd),
    }


def mixed_effects_loa(df, model):
    """
    Random-intercept LMM cross-check: d ~ 1 + (1|participant).

    NOTE: with thousands of strongly autocorrelated 1-Hz observations per
    participant, statsmodels' REML optimizer frequently drives the random-
    intercept variance to the boundary (tau^2 -> 0, "singular covariance"),
    which would spuriously collapse the LMM total SD onto the within-subject
    SD. We detect this and flag it; in that situation the moment-based
    variance-component (ANOVA) estimate is the reliable one and is used as the
    primary reported LoA. We still report the LMM residual SD as an independent
    check on s_within.
    """
    try:
        import warnings
        import statsmodels.formula.api as smf
        tmp = df[["participant"]].copy()
        tmp["d"] = (df[model] - df["y_true"]).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm("d ~ 1", tmp, groups=tmp["participant"])
            mfit = md.fit(reml=True, method="lbfgs")
        tau2 = float(mfit.cov_re.iloc[0, 0])  # between-subject (random intercept) var
        sigma2 = float(mfit.scale)            # residual within-subject var
        singular = tau2 < 1e-6
        total_sd = np.sqrt(tau2 + sigma2)
        bias = float(mfit.params["Intercept"])
        return {
            "bias": bias,
            "tau2": tau2,
            "sigma2": sigma2,
            "tau": float(np.sqrt(max(tau2, 0.0))),
            "sigma": float(np.sqrt(sigma2)),
            "total_sd": float(total_sd),
            "loa_lower": float(bias - Z * total_sd),
            "loa_upper": float(bias + Z * total_sd),
            "singular_boundary": bool(singular),
            "ok": True,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def main():
    df = pd.read_parquet(DATA)

    results = {}
    rows = []
    for family, model in REPS.items():
        naive = naive_loa(df, model)
        vc = variance_component_loa(df, model)
        lmm = mixed_effects_loa(df, model)

        subj_means = vc["subj_means"]
        bias_min = float(subj_means.min())
        bias_max = float(subj_means.max())
        bias_min_pid = int(subj_means.idxmin())
        bias_max_pid = int(subj_means.idxmax())

        # ratio: how much wider VC LoA is vs naive LoA
        naive_width = naive["loa_upper"] - naive["loa_lower"]
        vc_width = vc["loa_upper"] - vc["loa_lower"]

        results[model] = {
            "family": family,
            "n_samples": int(len(df)),
            "n_participants": int(df["participant"].nunique()),
            "bias": vc["bias"],
            "naive": {
                "sd": naive["sd"],
                "loa_lower": naive["loa_lower"],
                "loa_upper": naive["loa_upper"],
                "width": float(naive_width),
            },
            "variance_component": {
                "s_within": vc["s_within"],
                "s_between": vc["s_between"],
                "s_between2_raw": vc["s_between2"],
                "total_sd": vc["total_sd"],
                "loa_lower": vc["loa_lower"],
                "loa_upper": vc["loa_upper"],
                "width": float(vc_width),
                "n0": vc["n0"],
            },
            "mixed_effects": lmm,
            "per_participant_bias": {
                "min": bias_min,
                "max": bias_max,
                "min_pid": bias_min_pid,
                "max_pid": bias_max_pid,
                "range": float(bias_max - bias_min),
            },
            "vc_to_naive_width_ratio": float(vc_width / naive_width),
        }

        rows.append({
            "family": family,
            "model": model,
            "bias": round(vc["bias"], 4),
            "naive_sd": round(naive["sd"], 4),
            "naive_loa_lower": round(naive["loa_lower"], 4),
            "naive_loa_upper": round(naive["loa_upper"], 4),
            "vc_total_sd": round(vc["total_sd"], 4),
            "vc_s_within": round(vc["s_within"], 4),
            "vc_s_between": round(vc["s_between"], 4),
            "vc_loa_lower": round(vc["loa_lower"], 4),
            "vc_loa_upper": round(vc["loa_upper"], 4),
            "vc_to_naive_width_ratio": round(vc_width / naive_width, 3),
            "lmm_tau": round(lmm["tau"], 4) if lmm.get("ok") else None,
            "lmm_sigma": round(lmm["sigma"], 4) if lmm.get("ok") else None,
            "lmm_total_sd": round(lmm["total_sd"], 4) if lmm.get("ok") else None,
            "lmm_singular_boundary": lmm.get("singular_boundary") if lmm.get("ok") else None,
            "ppt_bias_min": round(bias_min, 4),
            "ppt_bias_max": round(bias_max, 4),
            "ppt_bias_range": round(bias_max - bias_min, 4),
        })

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "bland_altman_summary.csv")
    summary.to_csv(csv_path, index=False)

    # JSON (drop the pandas Series before dumping)
    results_clean = {}
    for m, r in results.items():
        rr = dict(r)
        results_clean[m] = rr
    json_path = os.path.join(OUTDIR, "bland_altman_results.json")
    with open(json_path, "w") as f:
        json.dump(results_clean, f, indent=2)

    # Per-participant bias table (all 3 models)
    pp_rows = []
    for pid in sorted(df["participant"].unique()):
        sub = df[df["participant"] == pid]
        row = {"participant": int(pid), "n": int(len(sub))}
        for _, model in REPS.items():
            row[f"bias_{model}"] = round(float((sub[model] - sub["y_true"]).mean()), 4)
        pp_rows.append(row)
    pp_df = pd.DataFrame(pp_rows)
    pp_path = os.path.join(OUTDIR, "per_participant_bias.csv")
    pp_df.to_csv(pp_path, index=False)

    # ---------------- FIGURE: Bland-Altman for PATCHTST ----------------
    model = "PATCHTST"
    pred = df[model].to_numpy()
    meas = df["y_true"].to_numpy()
    mean_pm = (pred + meas) / 2.0
    diff = pred - meas

    vc = results[model]["variance_component"]
    bias = results[model]["bias"]
    lo = vc["loa_lower"]
    hi = vc["loa_upper"]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    hb = ax.hist2d(mean_pm, diff, bins=(120, 120), cmap="viridis", cmin=1)
    cb = fig.colorbar(hb[3], ax=ax)
    cb.set_label("samples per bin (1 Hz)")

    ax.axhline(bias, color="crimson", lw=1.8, ls="-",
               label=f"bias = {bias:+.2f}")
    ax.axhline(hi, color="crimson", lw=1.5, ls="--",
               label=f"95% LoA (repeated-meas.) = [{lo:+.2f}, {hi:+.2f}]")
    ax.axhline(lo, color="crimson", lw=1.5, ls="--")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")

    ax.set_xlabel(r"Mean of predicted and measured $\dot{V}O_2$"
                  r" (mL$\,$kg$^{-1}\,$min$^{-1}$)")
    ax.set_ylabel(r"Predicted $-$ measured $\dot{V}O_2$"
                  r" (mL$\,$kg$^{-1}\,$min$^{-1}$)")
    ax.set_title(f"Bland-Altman agreement: {model} vs indirect calorimetry")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig_path = os.path.join(OUTDIR, "bland_altman_patchtst.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---------------- console report ----------------
    print("=" * 78)
    print("BLAND-ALTMAN AGREEMENT vs INDIRECT CALORIMETRY (n=16, ~60k 1-Hz samples)")
    print("=" * 78)
    for family, model in REPS.items():
        r = results[model]
        print(f"\n[{family}] {model}")
        print(f"  bias                 = {r['bias']:+.3f}")
        print(f"  NAIVE per-sample SD  = {r['naive']['sd']:.3f}  "
              f"-> LoA [{r['naive']['loa_lower']:+.3f}, {r['naive']['loa_upper']:+.3f}]"
              f"  width {r['naive']['width']:.3f}")
        print(f"  VC  s_within={r['variance_component']['s_within']:.3f}  "
              f"s_between={r['variance_component']['s_between']:.3f}  "
              f"total_SD={r['variance_component']['total_sd']:.3f}")
        print(f"  VC repeated-meas LoA = [{r['variance_component']['loa_lower']:+.3f}, "
              f"{r['variance_component']['loa_upper']:+.3f}]  "
              f"width {r['variance_component']['width']:.3f}  "
              f"(x{r['vc_to_naive_width_ratio']:.2f} wider than naive)")
        if r["mixed_effects"].get("ok"):
            me = r["mixed_effects"]
            flag = "  [SINGULAR: tau^2->0 boundary; VC/ANOVA is primary]" \
                if me.get("singular_boundary") else ""
            print(f"  LMM cross-check      tau={me['tau']:.3f}  "
                  f"sigma(within)={me['sigma']:.3f}  total_SD={me['total_sd']:.3f}{flag}")
        pb = r["per_participant_bias"]
        print(f"  per-participant bias range = [{pb['min']:+.3f} (P{pb['min_pid']}) .. "
              f"{pb['max']:+.3f} (P{pb['max_pid']})]  range {pb['range']:.3f}")

    print("\nOutputs:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", pp_path)
    print(" ", fig_path)


if __name__ == "__main__":
    main()
