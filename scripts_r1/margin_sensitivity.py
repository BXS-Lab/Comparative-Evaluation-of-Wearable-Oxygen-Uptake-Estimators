"""Equivalence-margin sensitivity: for every dataset and family contrast, the two one-sided
tests p-value as a function of the margin (in units of the dataset's SD of VO2), the smallest margin at which
equivalence would be declared (Delta*/SD), and the class under the single rule at alternative margins.
Outputs: results_r1/K100/margin_sensitivity.csv, outputs/figures/figS3_margin_sensitivity.{pdf,png},
outputs/tables/supp_s10_margin.tex. Usage: python scripts_r1/margin_sensitivity.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_r1"))
import figstyle as F  # noqa: E402

K = ROOT / "results_r1" / "K100"
OV = ROOT / "outputs"
CONTRASTS = ["Temporal - Tree", "Temporal - Linear", "Tree - Linear"]
DATASETS = ["BEL", "WEEE", "ING"]
DS = {"BEL": "Beltrame", "WEEE": "WEEE", "ING": "Ingraham"}
MARGINS = [0.03, 0.05, 0.08, 0.10]
GRID = np.linspace(0, 0.15, 601)


def tost_p(d, m):
    """Paired two one-sided t-tests against +-m; returns the larger one-sided p."""
    n, mu, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    lo = stats.t.sf((mu + m) / se, n - 1)
    hi = stats.t.cdf((mu - m) / se, n - 1)
    return max(lo, hi)


def short(label):
    return label.replace(" within margin)", ")").replace("Equivalent (Temporal better)", "Equivalent; temporal better") \
        .replace("Equivalent (Tree better)", "Equivalent; tree better")


def main():
    t5 = pd.read_csv(K / "table5.csv").set_index("dataset")
    rows, curves = [], {}
    for d in DATASETS:
        sd = t5.loc[d, "sd_vo2"]
        fp = pd.read_csv(K / d / "analysis" / "family_per_participant_rmse.csv")
        fc = pd.read_csv(K / d / "analysis" / "family_contrasts.csv").set_index("contrast")
        for c in CONTRASTS:
            a, b = c.split(" - ")
            diff = (fp[a] - fp[b]).to_numpy()
            x = fc.loc[c]
            p_at_dstar = tost_p(diff, x.delta_star)
            assert abs(p_at_dstar - 0.05) < 1e-6, (d, c, p_at_dstar)
            curves[(d, c)] = np.array([tost_p(diff, g * sd) for g in GRID])
            r = {"dataset": d, "contrast": c, "n": len(diff), "sd_vo2": sd, "delta_star": x.delta_star,
                 "delta_star_over_sd": x.delta_star / sd, "p_holm": x.p_holm}
            for m in MARGINS:
                r[f"class_{m:.2f}sd"] = F.contrast_label(c, x, m * sd)
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(K / "margin_sensitivity.csv", index=False)

    # ---- figure ----
    F.apply()
    fig, axes = plt.subplots(1, 3, figsize=(F.FULL, 2.2), sharey=True)
    for k, (ax, c) in enumerate(zip(axes, CONTRASTS)):
        ax.axvline(0.05, color=F.INK2, lw=0.8)
        ax.axvline(0.10, color=F.INK2, lw=0.6, ls=(0, (3, 2)))
        ax.axhline(0.05, color=F.GRID, lw=0.8, zorder=0)
        for d in DATASETS:
            y = curves[(d, c)]
            ax.plot(GRID, y, color=F.DATASET_COLOR[d], lw=1.2, label=F.DATASET_LABEL[d])
            ds = out[(out.dataset == d) & (out.contrast == c)].delta_star_over_sd.iloc[0]
            if ds <= GRID[-1]:
                ax.plot([ds], [0.05], marker=F.DATASET_MARKER[d], color=F.DATASET_COLOR[d], ms=4.5,
                        mec="white", mew=0.6, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlim(0, 0.15)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
        ax.set_title(c.replace(" - ", " $-$ "))
        ax.set_xlabel("Equivalence margin / SD(" + F.VO2 + ")")
        F.panel(ax, "ABC"[k], x=-0.16 if k == 0 else -0.06)
    axes[0].set_ylabel("Two one-sided tests $p$")
    axes[2].legend(loc="lower left", handlelength=1.6)
    fig.tight_layout(w_pad=0.6)
    F.save(fig, OV / "figures", "figS3_margin_sensitivity")

    # ---- table ----
    body = [r"\begin{table}[ht!]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Equivalence-margin sensitivity. $\Delta^*$/SD is the smallest margin, in units of the dataset's SD of $\dot{V}$O$_2$, at which the two one-sided tests would declare equivalence; classes follow the single rule of the main analysis at each margin (the main analysis uses $0.05\times$SD). Equivalence at a margin below $\Delta^*$/SD is not possible with these data; any margin above it yields equivalence. See Supplementary Fig.~\ref{fig:s_margin} for the full curves.}",
            r"\label{tab:s_margin}", r"\begin{tabular}{llccccc}", r"\toprule",
            r"Dataset & Contrast & $\Delta^*$/SD & $0.03\times$SD & $0.05\times$SD (main) & $0.08\times$SD & $0.10\times$SD \\", r"\midrule"]
    for i, r in out.iterrows():
        first = DS[r.dataset] if r.contrast == CONTRASTS[0] else ""
        cells = " & ".join(short(r[f"class_{m:.2f}sd"]) for m in MARGINS)
        body.append(f"{first} & {r.contrast.replace(' - ', ' $-$ ')} & {r.delta_star_over_sd:.3f} & {cells} \\\\")
        if r.contrast == CONTRASTS[-1] and r.dataset != DATASETS[-1]:
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OV / "tables").mkdir(parents=True, exist_ok=True)
    (OV / "tables" / "supp_s10_margin.tex").write_text("\n".join(body), encoding="utf-8")
    print(out[["dataset", "contrast", "delta_star_over_sd"] + [f"class_{m:.2f}sd" for m in MARGINS]].to_string())


if __name__ == "__main__":
    main()
