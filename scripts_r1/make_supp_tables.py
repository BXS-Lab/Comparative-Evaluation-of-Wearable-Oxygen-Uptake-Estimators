"""Supplementary LaTeX tables (S1–S9), all read from saved results.
Output: outputs/tables/supp_<k>_<name>.tex. Usage: python scripts_r1/make_supp_tables.py"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_r1"))
import figstyle as F  # noqa: E402

RES = ROOT / "results_r1"
K = RES / "K100"
OUT = ROOT / "outputs" / "tables"
U = r"mL\,kg$^{-1}$\,min$^{-1}$"
DS = {"BEL": "Beltrame", "WEEE": "WEEE", "ING": "Ingraham", "WEEE_HRQ": "WEEE, HR-screened", "BEL_CORE": "Beltrame, shared sensors"}


def m(x, k=2):
    t = f"{x:.{k}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    return t.replace("-", "$-$")


def r1(x):
    """Round half away from zero to one decimal (avoids banker's rounding of quartiles such as -25.25)."""
    from decimal import Decimal, ROUND_HALF_UP
    return f"{Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):+}"


def tab(name, caption, label, colspec, header, rows, size="\\small", note=None, wide=False):
    body = [r"\begin{table}[ht!]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if wide:
        body.append(r"\resizebox{\linewidth}{!}{%")
    body += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    body += [r + r" \\" if not r.startswith("\\midrule") else r for r in rows]
    body += [r"\bottomrule", r"\end{tabular}" + ("}" if wide else "")]
    if note:
        body.append(r"\par\smallskip\raggedright\footnotesize " + note)
    body += [r"\end{table}", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(body), encoding="utf-8")


def s1_body_mass():
    t = pd.read_csv(K / "WEEE" / "analysis" / "weee_body_mass_evidence.csv").set_index("participant")
    rows = []
    for p in ["P04", "P08", "P15"]:
        r = t.loc[p]
        rows.append(f"{p} & {r.app_profile} & {r.profile_age} / {r.demographics_age} & {r.profile_mass_kg:.1f} & {r.app_implied_mass_kg:.1f} & "
                    f"{r.demographics_mass_kg:.1f} & {r.listed_BMI:.1f} & {r.BMI_from_demographics_mass:.1f} / {r.BMI_from_profile_mass:.1f} & "
                    f"{r.target_factor_corrected_over_app:.3f}")
    others = t.drop(["P04", "P08", "P15"]).target_factor_corrected_over_app
    tab("supp_s1_body_mass", "WEEE body-mass evidence. The analyser's per-kilogram export divides $\\dot{V}$O$_2$ by the mass of the selected athlete profile (the ratio of the per-minute to the per-kilogram export equals the profile mass in every sample). For three participants the profile did not belong to them (P04, P08) or carried a wrong mass (P15); the dataset's demographics file reproduces the listed BMI only with its own mass. The target was therefore computed as $\\dot{V}$O$_2$ (mL\\,min$^{-1}$) divided by the demographics mass for all participants.",
        "tab:s_bodymass", "lllcccccc",
        "Participant & Profile & \\shortstack{Age\\\\(profile / actual)} & \\shortstack{Profile\\\\mass (kg)} & \\shortstack{Implied\\\\mass (kg)} & "
        "\\shortstack{Demographics\\\\mass (kg)} & \\shortstack{Listed\\\\BMI} & \\shortstack{BMI (demographics /\\\\profile mass)} & \\shortstack{Target\\\\factor}",
        rows, size="\\scriptsize",
        note=f"Target factor = corrected / app-exported target. For the other participants the factor was {others.min():.3f}--{others.max():.3f}.")


def s2_activity():
    rows = []
    for d in ["WEEE", "ING"]:
        t = pd.read_csv(K / d / "analysis" / "per_activity_rmse.csv").set_index("activity")
        for i, (a, r) in enumerate(t.iterrows()):
            rows.append(f"{DS[d] if i == 0 else ''} & {a} & {int(r.n_participants)} & {int(r.n_samples):,} & {r.Linear:.2f} & {r.Tree:.2f} & {r.Temporal:.2f}".replace(",", "\\,"))
        rows.append("\\midrule")
    rows = rows[:-1]
    tab("supp_s2_activity", f"Error by activity on the external datasets: mean over participants of the family-mean per-participant RMSE ({U}); cells with less than 60~s of evaluation samples for a participant are omitted. Descriptive only.",
        "tab:s_activity", "llrrccc", "Dataset & Activity & Participants & Samples & Linear & Tree & Temporal", rows)


def s3_leaderboards():
    rows = []
    for d in ["WEEE", "ING"]:
        lb = pd.read_csv(K / d / "analysis" / "leaderboard.csv").sort_values("rmse")
        for i, r in enumerate(lb.itertuples()):
            seed = "--" if r.family == "Linear" else f"{r.seed_range:.3f}"
            rows.append(f"{DS[d] if i == 0 else ''} & {F.MODEL_LABEL[r.model]} & {r.family} & {r.rmse:.3f} [{r.rmse_bca_lo:.2f}, {r.rmse_bca_hi:.2f}] & {r.r_pooled:.3f} & {r.r2_pooled:.3f} & {seed}")
        rows.append("\\midrule")
    rows = rows[:-1]
    tab("supp_s3_leaderboards", f"Model comparison on the external datasets (nested leave-one-participant-out; WEEE $n=15$, Ingraham $n=10$). RMSE ({U}) is the mean of per-participant RMSE with BCa 95\\% CI; pooled $r$ and $R^2$; seed range as in Table~\\ref{{tab:model_comparison}}. No pairwise tests were performed (WEEE omnibus $p=0.25$; Ingraham $n<12$).",
        "tab:s_leaderboards", "lllcccc", "Dataset & Model & Family & RMSE [95\\% CI] & $r$ & $R^2$ & Seed range", rows, size="\\scriptsize")


def s4_ba():
    ba = pd.read_csv(K / "BEL" / "analysis" / "bland_altman.csv").set_index("model")
    rows = []
    for mm in F.MODEL_ORDER:
        r = ba.loc[mm]
        rows.append(f"{F.MODEL_LABEL[mm]} & {F.MODEL_FAMILY[mm]} & {m(r.bias)} & [{m(r.vc_loa_lo)}, {m(r.vc_loa_hi)}] & {r.s_within:.2f} & {r.s_between:.2f} & {m(r.ppt_bias_min)} to {m(r.ppt_bias_max)}")
    tab("supp_s4_ba", f"Bland--Altman agreement of all eleven models on the primary dataset ({U}); columns as in Table~\\ref{{tab:ba}}.",
        "tab:s_ba", "llccccc", "Model & Family & Bias & 95\\% LoA & $s_w$ & $s_b$ & Per-participant bias range", rows)


def s5_w4b():
    rows = []
    for d in ["BEL", "WEEE", "ING", "WEEE_HRQ"]:
        t = pd.read_csv(K / d / "analysis" / "w4b_summary.csv")
        for j, learner in enumerate(["ridge", "xgb"]):
            q = t[t.learner == learner].set_index("d")
            cells = [f"{m(q.loc[x, 'gain_mean'])} [{m(q.loc[x, 'gain_bca_lo'])}, {m(q.loc[x, 'gain_bca_hi'])}]" for x in [5.0, 10.0, 30.0, 60.0, 120.0]]
            rows.append(f"{DS[d] if j == 0 else ''} & {'Ridge' if learner == 'ridge' else 'XGBoost'} & " + " & ".join(cells))
        rows.append("\\midrule")
    rows = rows[:-1]
    tab("supp_s5_w4b", "Baseline-relative gain $1-\\mathrm{RMSE}_{h+s}/\\mathrm{RMSE}_{h}$ of the wearable sensor set over a $\\dot{V}$O$_2$ history available $d$~s late: mean over participants with BCa 95\\% CI. Ridge with nested selection (primary reference learner); XGBoost with library defaults.",
        "tab:s_w4b", "llccccc", "Dataset & Learner & $d=5$~s & 10~s & 30~s & 60~s & 120~s", rows, size="\\scriptsize", wide=True)


def s6_w4a():
    rows = []
    name = {"all": "All features", "HR": "HR", "dHR": "$\\Delta$HR", "ACC": "Acceleration", "CAD": "Cadence", "VE": "$\\dot{V}_\\mathrm{E}$",
            "BF": "BF", "BR": "Breathing rate", "BRAmplitude": "Breathing amplitude", "Activity": "Activity"}
    for d in ["BEL", "ING", "WEEE", "WEEE_HRQ"]:
        a = pd.read_csv(RES / d / "w4" / "w4a_summary.csv")
        t = a[a.filtering == "zero_phase_0.05Hz"]
        u = a[a.filtering == "unfiltered"].set_index("features")
        for i, r in enumerate(t.itertuples()):
            det = "undetermined" if r.undetermined else f"{r.h_pooled:+d}"
            ru = u.loc[r.features]
            unf = "undetermined" if ru.undetermined else f"{int(ru.h_pooled):+d}"
            rows.append(f"{DS[d] if i == 0 else ''} & {name.get(r.features, r.features)} & {det} & {r1(r.h_median)} & [{r1(r.h_q1)}, {r1(r.h_q3)}] & [{r1(r.h_median_ci_lo)}, {r1(r.h_median_ci_hi)}] & {unf}".replace("-", "$-$"))
        rows.append("\\midrule")
    rows = rows[:-1]
    tab("supp_s6_w4a", "Lag diagnostic: shift $h$ (s) minimising the RMSE of a ridge model predicting $\\dot{V}$O$_2(t-h)$ from the inputs at $t$ (zero-phase 0.05~Hz inputs; $h<0$: the signal leads $\\dot{V}$O$_2$). Pooled optimum of the mean curve (``undetermined'' if within 2~s of the $\\pm$60~s boundary or if the curve varies by less than 1\\%), per-participant median, interquartile range, and bootstrap 95\\% CI of the median. WEEE Zephyr features use only the four participants with a verified clock offset. The last column gives the pooled optimum with unfiltered inputs (sensitivity analysis). Values are effective response times of the processed signals, not physiological delays.",
        "tab:s_w4a", "llccccc", "Dataset & Input & Pooled optimum & Median & IQR & 95\\% CI of median & Unfiltered", rows, size="\\scriptsize")


def mstrict(d, root=K):
    return json.loads((root / d / "analysis" / "summary.json").read_text())["m_strict"]


def contrast_rows(d, label, root=K):
    fc = pd.read_csv(root / d / "analysis" / "family_contrasts.csv").set_index("contrast")
    ms = mstrict(d, root)
    out = []
    for i, c in enumerate(["Temporal - Tree", "Temporal - Linear", "Tree - Linear"]):
        x = fc.loc[c]
        out.append(f"{label if i == 0 else ''} & {c.replace(' - ', ' $-$ ')} & {m(x.mean_diff)} [{m(x.ci95_lo)}, {m(x.ci95_hi)}] & "
                   f"{x.delta_star:.3f} & {x.p_holm:.3f} & {F.contrast_label(c, x, ms)}")
    return out


def s7_sensitivity():
    rows = contrast_rows("BEL_CORE", "Beltrame, shared sensors (HR, $\\Delta$HR, acceleration; $n=16$)") + ["\\midrule"]
    rows += contrast_rows("WEEE_HRQ", "WEEE, four participants with implausible HR excluded ($n=11$)") + ["\\midrule"]
    wa = json.loads((K / "WEEE_ALL" / "tree_linear_sensitivity.json").read_text())
    wlab = "Tree better" if (wa["p_exact_wilcoxon"] < 0.05 and wa["ci95_hi"] < 0) else "Not resolved"
    rows += [f"WEEE, all participants ($n={wa['n']}$; linear and tree only) & Tree $-$ Linear & {m(wa['mean_diff'])} [{m(wa['ci95_lo'])}, {m(wa['ci95_hi'])}] & -- & {wa['p_exact_wilcoxon']:.3f}$^a$ & {wlab}", "\\midrule"]
    ing = pd.read_csv(K / "ING" / "analysis" / "ing_bad_breath_rescoring.csv")
    lab = {"exclude rows nearest a bad breath": "nearest samples excluded", "exclude rows in the interpolation span of a bad breath": "interpolation spans excluded",
           "re-score against cleaned target (bad breaths removed, re-interpolated)": "cleaned target"}
    for v, nm in lab.items():
        q = ing[(ing.variant == v)].set_index("c")
        for i, c in enumerate(["Temporal-Tree", "Temporal-Linear", "Tree-Linear"]):
            x = q.loc[c]
            xr = {"delta_star": x.dstar, "p_holm": x.p_holm, "ci95_lo": x.lo, "ci95_hi": x.hi, "mean_diff": x["diff"]}
            rows.append(f"{'Ingraham, implausible breaths: ' + nm if i == 0 else ''} & {c.replace('-', ' $-$ ')} & {m(x['diff'])} [{m(x.lo)}, {m(x.hi)}] & "
                        f"{x.dstar:.3f} & {x.p_holm:.3f} & {F.contrast_label(c.replace('-', ' - '), xr, x.m_strict)}")
        rows.append("\\midrule")
    rows = rows[:-1]
    tab("supp_s7_sensitivity", f"Sensitivity analyses. Family contrasts ({U}) with 95\\% CI, $\\Delta^*$, Holm-adjusted exact Wilcoxon $p$ over the three contrasts, and the class from the single rule of the main analysis (equivalent if $\\Delta^*<0.05\\times$SD; otherwise a resolved difference if $p_\\mathrm{{Holm}}<0.05$ and the 95\\% CI excludes zero; otherwise not resolved; equivalence with $\\Delta^*$ within 5\\% of the margin is marked borderline). Equivalence margins: Beltrame 0.266, WEEE 0.390, Ingraham 0.441~{U}. The WEEE heart-rate screen and the Ingraham re-scoring are post-hoc sensitivity analyses.",
        "tab:s_sensitivity", "p{0.3\\linewidth}lcccl", "Analysis & Contrast & Difference [95\\% CI] & $\\Delta^*$ & $p_\\mathrm{Holm}$ & Class", rows, size="\\scriptsize", wide=True,
        note="$^a$ Single contrast, unadjusted; $\\Delta^*$ not computed.")


def s8_hr_screen():
    t = pd.read_csv(K / "WEEE" / "analysis" / "weee_hr_screen.csv")
    rows = [f"P{int(r.pid):02d} & {r.exercise_minus_rest_bpm:.1f} & {r.wahoo_zephyr_gap_bpm:.1f} & {'yes' if r.passes_screen else 'no'}" for r in t.itertuples()]
    tab("supp_s8_hr_screen", "WEEE heart-rate plausibility screen (target-free; used only in a post-hoc sensitivity analysis). Exercise minus rest: highest per-condition median Wahoo HR during cycling or running minus the highest during sitting or standing (criterion $\\geq$15~beats\\,min$^{-1}$). Device gap: median over conditions of the absolute difference between the per-condition median Wahoo and Zephyr HR (criterion $\\leq$10~beats\\,min$^{-1}$).",
        "tab:s_hr_screen", "lccc", "Participant & Exercise $-$ rest (beats\\,min$^{-1}$) & Device gap (beats\\,min$^{-1}$) & Retained", rows)


def s9_budget():
    b = pd.read_csv(K / "budget_step_statistics.csv").pivot_table(index=["dataset", "model"], columns="step", values="median_improvement")
    rows = []
    for i, ((d, mm), r) in enumerate(b.iterrows()):
        first = i == 0 or b.index[i - 1][0] != d
        rows.append(f"{DS[d] if first else ''} & {F.MODEL_LABEL[mm]} & " + " & ".join(f"{r[s]:.3f}" for s in ["25->40", "40->60", "60->80", "80->100"]))
    tab("supp_s9_budget", f"Tuning-budget rule for the temporal models: median over outer folds of the improvement of the best inner-fold score ({U}) between successive trial counts, computed from the trial order of the final studies. The budget is adequate once every value is below 0.01, which holds from 80 trials; 100 trials were used throughout.",
        "tab:s_budget", "llcccc", "Dataset & Model & 25$\\to$40 & 40$\\to$60 & 60$\\to$80 & 80$\\to$100", rows, size="\\scriptsize")


if __name__ == "__main__":
    for f in (s1_body_mass, s2_activity, s3_leaderboards, s4_ba, s5_w4b, s6_w4a, s7_sensitivity, s8_hr_screen, s9_budget):
        f()
        print(f.__name__, "done", flush=True)
