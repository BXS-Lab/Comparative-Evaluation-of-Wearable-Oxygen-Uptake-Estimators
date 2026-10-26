"""LaTeX tables of the revised manuscript (Tables 1–5), all numbers read from saved results or code constants.

Output: outputs/tables/table<k>_<name>.tex (booktabs).
Usage: python scripts_r1/make_tables.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_r1"))
import figstyle as F  # noqa: E402
from r1 import linear, temporal, trees  # noqa: E402,F401
from r1.datasets import load  # noqa: E402
from r1.preprocess import CUTOFF_GRID  # noqa: E402

K100 = ROOT / "results_r1" / "K100"
OUT = ROOT / "outputs" / "tables"
VO2 = r"$\dot{V}\mathrm{O}_2$"
U = r"mL\,kg$^{-1}$\,min$^{-1}$"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")


def f2(x):
    return f"{x:.2f}".replace("-", "$-$")


def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


def pfmt(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def sci(p):
    if p >= 0.001:
        return f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


# ------------------------------------------------------------------ Table 1
def table1():
    st = {}
    for d in ["BEL", "WEEE", "ING"]:
        pnl = load(d)
        g = pnl.eval_mask()
        y = pnl.df.loc[g, "y"]
        n = pnl.df[g].groupby("pid").size()
        st[d] = (f"{y.mean():.1f} $\\pm$ {y.std():.1f} ({np.percentile(y, 1):.1f}--{np.percentile(y, 99):.1f})",
                 f"{n.median() / 60:.0f} min ({int(g.sum()):,} samples)".replace(",", "\\,"))
    rows = [
        ("Participants", "16 healthy, physically active men", "15 healthy adults (11 men, 4 women; 30.0 $\\pm$ 4.9 y); 2 of 17 excluded by data-quality criteria fixed before modelling",
         "10 healthy adults (8 men, 2 women; 24--37 y)"),
        ("Activities", "Over-ground walking at 75/105/135 steps\\,min$^{-1}$ (PRTS) and simulated activities of daily living",
         "Sitting, standing, cycling (2 intensities), treadmill running (2 speeds); continuous, $\\approx$30 min",
         "Walking, incline walking, backwards walking, running, cycling, stair climbing; 6-min conditions with standing and sitting periods"),
        ("Criterion", "COSMED K4b$^2$, breath-by-breath, interpolated to 1\\,Hz by the original authors",
         "VO2 Master, 1\\,Hz app output (held breath values); normalised by body mass from the demographics file",
         "Oxycon Mobile, breath-by-breath, linearly interpolated to 1\\,Hz"),
        ("Inputs (1\\,Hz)", "HR, $\\Delta$HR, $\\dot{V}_\\mathrm{E}$, BF (Hexoskin shirt); Hacc, CAD",
         "HR (Wahoo chest strap), $\\Delta$HR; BR, BR amplitude, activity (Zephyr chest strap; clock offset corrected where verifiable)",
         "HR (Polar strap), $\\Delta$HR; waist dynamic acceleration (APDM); BF from the criterion system$^{a}$"),
        ("Evaluation time per participant (median; total)", st["BEL"][1], st["WEEE"][1], st["ING"][1]),
        (f"{VO2} on the evaluation grid, mean $\\pm$ SD (1st--99th percentile), {U}", st["BEL"][0], st["WEEE"][0], st["ING"][0]),
        ("Transition marker", "Causal cadence-change detector (terciles)", "Protocol timing", "Protocol timing"),
    ]
    body = [r"\begin{table}[ht!]", r"\centering", r"\small",
            r"\caption{The three public datasets. The same pipeline (nested leave-one-participant-out evaluation, causal per-segment filtering, 240\,s burn-in per segment) and the same pre-declared criteria were applied to each.}",
            r"\label{tab:datasets}", r"\begin{tabular}{p{0.15\linewidth}p{0.235\linewidth}p{0.235\linewidth}p{0.235\linewidth}}", r"\toprule",
            r" & \citet{Main_paper_Beltrame2017} (primary) & WEEE \citep{Gashi2022} & \citet{Ingraham2019} \\", r"\midrule"]
    body += [" & ".join(r) + r" \\" for r in rows]
    body += [r"\bottomrule", r"\end{tabular}",
             r"\par\smallskip\raggedright\footnotesize $^{a}$ Used as a stand-in for a wearable respiration sensor (disclosed in Limitations).",
             r"\end{table}", ""]
    write("table1_datasets", "\n".join(body))


# ------------------------------------------------------------------ Table 2
def table2():
    cut = ", ".join(str(c) for c in CUTOFF_GRID if c is not None) + ", none"
    win = ", ".join(str(w) for w in temporal.WINDOWS)
    rows = [
        (r"\multicolumn{3}{p{0.95\linewidth}}{\emph{All models}: low-pass cutoff (Hz) $\in$ \{" + cut + r"\}; selected inside the inner loop} \\", None),
        ("OLS", "grid", "--"),
        ("Ridge", "grid", "$\\alpha$: 30 values, $10^{-3}$--$10^{6}$ (log)"),
        ("Lasso", "grid", "$\\alpha$: 30 values, $10^{-4}$--$10^{3}$ (log)"),
        ("Elastic net", "grid", "$\\alpha$: 20 values, $10^{-4}$--$10^{3}$ (log); $\\ell_1$ ratio $\\in$ \\{0.1, \\ldots, 0.9, 0.95, 0.99\\}"),
        ("RF", "60", "trees 100--800; depth 3--30; min.\\ leaf 1--8; max.\\ features \\{sqrt, log2\\}"),
        ("XGBoost", "100", "trees 100--1000; learning rate 0.01--0.3 (log); depth 2--10; subsample, column sample 0.5--1; $\\gamma$ 0--5; $\\alpha$, $\\lambda$ 0--2"),
        (r"\multicolumn{3}{p{0.95\linewidth}}{\emph{Temporal models}: input window $L$ (s) $\in$ \{" + win + r"\}; AdamW; batch \{32, 64, 128\} (PatchTST \{32, 64\})} \\", None),
        ("LSTM, GRU", "100", "hidden \\{32, 64, 96, 128\\}; layers 1--3; dropout 0.1--0.5; lr $10^{-4}$--$10^{-2}$ (log); weight decay $10^{-6}$--$10^{-2}$ (log)"),
        ("TCN", "100", "kernel \\{3, 5, 7\\}; levels 4--9 (raised until the receptive field covers $L$); channels \\{32, 48, 64\\}; dropout 0--0.5; lr $5\\times10^{-5}$--$10^{-3}$ (log)"),
        ("TFT$^{a}$", "100", "hidden \\{32, 48, 64, 96\\}; heads \\{2, 4\\}; dropout 0.1--0.5; lr $10^{-4}$--$10^{-2}$ (log); weight decay $10^{-6}$--$10^{-2}$ (log)"),
        ("PatchTST$^{a}$", "100", "patch \\{12, 16, 20, 24\\} ($\\geq$2 patches); $d_\\mathrm{model}$ \\{32, \\ldots, 128\\}; heads \\{4, 8\\}; layers 2--4; channel-independent or -dependent; dropout 0.1--0.3; lr $10^{-4}$--$10^{-3}$ (log); weight decay $10^{-6}$--$10^{-2}$ (log)"),
    ]
    body = [r"\begin{table}[ht!]", r"\centering", r"\small",
            r"\caption{Hyperparameter search spaces and budgets. Every selection was made inside the training participants of each outer fold (nested leave-one-participant-out): linear models by participant-grouped 5-fold inner cross-validation over the grid; tree and temporal models by Optuna (TPE sampler, median pruner) with participant-grouped 5-fold (trees) or 3-fold (temporal) inner cross-validation. Temporal training: stride 8, at most 60 epochs, early stopping (patience 10) on a dedicated outer-training participant, ReduceLROnPlateau, gradient clipping 1.0; three refit seeds per outer fold.}",
            r"\label{tab:search}", r"\begin{tabular}{p{0.13\linewidth}p{0.07\linewidth}p{0.68\linewidth}}", r"\toprule",
            r"Model & Trials & Search space \\", r"\midrule"]
    for r in rows:
        body.append(r[0] if r[1] is None else " & ".join(r) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\par\smallskip\raggedright\footnotesize $^{a}$ Adapted for single-step regression: a simplified TFT and a PatchTST-style patch Transformer without reversible instance normalisation.",
             r"\end{table}", ""]
    write("table2_search", "\n".join(body))


# ------------------------------------------------------------------ Table 3
def table3():
    lb = pd.read_csv(K100 / "BEL" / "analysis" / "leaderboard.csv").set_index("model")
    body = [r"\begin{table}[ht!]", r"\centering", r"\small",
            r"\caption{Model comparison on the primary dataset (Beltrame \emph{et al.}; $n=16$; 59\,731 matched 1\,Hz samples). RMSE is the mean of per-participant RMSE with a BCa bootstrap 95\% CI (participant as unit); for stochastic models each participant's RMSE is averaged over three training seeds, and the seed range is the spread of the three seed-wise means. Pooled $r$ and $R^2$ are computed over all samples. Nested leave-one-participant-out evaluation.}",
            r"\label{tab:model_comparison}", r"\begin{tabular}{llcccc}", r"\toprule",
            rf"Family & Model & RMSE ({U}) [95\% CI] & $r$ & $R^2$ & Seed range \\", r"\midrule"]
    for fam in ["Linear", "Tree", "Temporal"]:
        ms = [m for m in F.MODEL_ORDER if F.MODEL_FAMILY[m] == fam]
        ms = sorted(ms, key=lambda m: lb.loc[m, "rmse"])
        for i, m in enumerate(ms):
            r = lb.loc[m]
            seed = "--" if fam == "Linear" else f"{r.seed_range:.3f}"
            body.append(f"{fam if i == 0 else ''} & {F.MODEL_LABEL[m]} & {r.rmse:.3f} [{r.rmse_bca_lo:.2f}, {r.rmse_bca_hi:.2f}] & "
                        f"{r.r_pooled:.3f} & {r.r2_pooled:.3f} & {seed} \\\\")
        if fam != "Temporal":
            body.append(r"\midrule")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table3_leaderboard", "\n".join(body))


# ------------------------------------------------------------------ Table 4
def table4():
    ba = pd.read_csv(K100 / "BEL" / "analysis" / "bland_altman.csv").set_index("model")
    body = [r"\begin{table}[ht!]", r"\centering", r"\small",
            r"\caption{Agreement with indirect calorimetry on the primary dataset for the most accurate model of each family (all 11 models in the Supplement). Bias and repeated-measures (variance-component) 95\% limits of agreement (LoA); $s_w$ and $s_b$ are the within- and between-participant standard deviations of the differences; the per-participant bias range spans the 16 participants' mean differences. Units: " + U + ".}",
            r"\label{tab:ba}", r"\begin{tabular}{lccccc}", r"\toprule",
            r"Model (family) & Bias & 95\% LoA & $s_w$ & $s_b$ & Per-participant bias range \\", r"\midrule"]
    for m, fam in [("TCN", "temporal"), ("RF", "tree"), ("LASSO", "linear")]:
        r = ba.loc[m]
        body.append(f"{F.MODEL_LABEL[m]} ({fam}) & {f2(r.bias)} & [{f2(r.vc_loa_lo)}, {f2(r.vc_loa_hi)}] & {r.s_within:.2f} & {r.s_between:.2f} & "
                    f"{f2(r.ppt_bias_min)} to {f2(r.ppt_bias_max)} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table4_ba", "\n".join(body))


# ------------------------------------------------------------------ Table 5
def table5():
    T = pd.read_csv(K100 / "table5.csv").set_index("dataset")
    rep = pd.read_csv(K100 / "replication.csv")
    w = {d: pd.read_csv(K100 / d / "analysis" / "w4b_summary.csv").query("learner == 'ridge'").set_index("d") for d in ["BEL", "WEEE", "ING"]}
    fc = {d: pd.read_csv(K100 / d / "analysis" / "family_contrasts.csv").set_index("contrast") for d in ["BEL", "WEEE", "ING"]}
    D = ["BEL", "WEEE", "ING"]

    def gain(d, dd):
        r = w[d].loc[dd]
        return f"{f2(r.gain_mean)} [{f2(r.gain_bca_lo)}, {f2(r.gain_bca_hi)}]"

    def label(d, c):
        return F.contrast_label(c, fc[d].loc[c], T.loc[d, "m_strict"])

    def con(d, c):
        x = fc[d].loc[c]
        return (f"{f2(x.mean_diff)} [{f2(x.ci95_lo)}, {f2(x.ci95_hi)}]; $\\Delta^*$ {x.delta_star:.2f} ({x.delta_star / T.loc[d, 'sd_vo2']:.3f}\\,SD); "
                f"$p_\\mathrm{{H}}$ {pfmt(x.p_holm)}; {label(d, c).lower()}")

    def r2(d):
        if d == "BEL":
            return "reference"
        return F.replication_label(label("BEL", "Temporal - Tree"), label(d, "Temporal - Tree"))

    eq = {d: float(fc[d].loc["Temporal - Tree", "eq_mde"]) for d in D}

    rows = [
        ("", "$n$; SD of " + VO2, lambda d: f"{int(T.loc[d, 'n'])}; {T.loc[d, 'sd_vo2']:.2f}"),
        ("", "Family RMSE / SD (linear, tree, temporal)", lambda d: ", ".join(f"{T.loc[d, 'rmse_' + f] / T.loc[d, 'sd_vo2']:.2f}" for f in ["Linear", "Tree", "Temporal"])),
        ("1", "Ridge gain$(d)$, $d$ = 10 s", lambda d: gain(d, 10.0)),
        ("", "\\phantom{Ridge gain$(d)$,} $d$ = 30 s", lambda d: gain(d, 30.0)),
        ("", "\\phantom{Ridge gain$(d)$,} $d$ = 60 s", lambda d: gain(d, 60.0)),
        ("2", "Transition penalty (linear, tree, temporal)", lambda d: ", ".join(f"{T.loc[d, 'penalty_' + f]:.2f}" for f in ["Linear", "Tree", "Temporal"]) + (" (95\\% CIs $>$1)" if T.loc[d, "R3_Temporal"] == "robust" else " (CIs include 1)")),
        ("3", "Equivalence margin (0.05 SD)", lambda d: f"{T.loc[d, 'm_strict']:.2f}"),
        ("", "Omnibus $p$", lambda d: sci(T.loc[d, "omnibus_p"])),
        ("", "Temporal $-$ tree", lambda d: con(d, "Temporal - Tree")),
        ("", "Temporal $-$ linear", lambda d: con(d, "Temporal - Linear")),
        ("", "Tree $-$ linear", lambda d: con(d, "Tree - Linear")),
        ("4", "Per-participant bias range (all models)", lambda d: f"{f2(T.loc[d, 'ppt_bias_min'])} to {f2(T.loc[d, 'ppt_bias_max'])}"),
        ("", "Temporal-vs-tree class relative to Beltrame", r2),
    ]
    body = [r"\begin{table}[ht!]", r"\centering", r"\scriptsize",
            r"\caption{The four protocol items on the three datasets. Differences in " + U + r" with 95\% CI; $\Delta^*$ = largest difference compatible with the 90\% CI (in absolute units and as a fraction of the SD of " + VO2 + r"); $p_\mathrm{H}$ = Holm-adjusted exact Wilcoxon $p$ over the three family contrasts. Classification: equivalent if $\Delta^*$ is below the margin of 0.05 $\times$ SD; otherwise a resolved difference (``better'') if $p_\mathrm{H}<0.05$ and the 95\% CI excludes zero; otherwise not resolved. Item 1: Ridge reference learner, BCa 95\% CI. Item 2: geometric-mean ratio of transition- to steady-stratum RMSE (cadence detector on Beltrame; protocol timing on WEEE and Ingraham), not comparable in magnitude across datasets. Absolute RMSE is not comparable across datasets; RMSE/SD is shown instead. The t-based CI and the Holm-adjusted test can disagree (Beltrame tree $-$ linear; Ingraham temporal $-$ linear). Pairwise model tests were possible only on Beltrame (WEEE omnibus not significant; Ingraham $n<12$). Approximate equivalence MDE of the temporal $-$ tree contrast: Beltrame " + f"{eq['BEL']:.3f}, WEEE {eq['WEEE']:.3f}, Ingraham {eq['ING']:.3f}" + r". Sensitivity analyses (Supplement): WEEE excluding four participants with implausible heart rate, temporal $-$ tree not resolved (same direction); Beltrame with heart rate, $\Delta$HR and acceleration only, temporal better.}",
            r"\label{tab:datasets_protocol}", r"\begin{tabular}{p{0.03\linewidth}p{0.16\linewidth}p{0.22\linewidth}p{0.22\linewidth}p{0.22\linewidth}}", r"\toprule",
            r"Item & Quantity & Beltrame ($n=16$) & WEEE ($n=15$) & Ingraham ($n=10$) \\", r"\midrule"]
    for item, q, fn in rows:
        body.append(f"{item} & {q} & " + " & ".join(fn(d) for d in D) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table5_datasets", "\n".join(body))


if __name__ == "__main__":
    for f in (table1, table2, table3, table4, table5):
        f()
        print(f.__name__, "done", flush=True)
