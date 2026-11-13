"""Supplementary tables that extend three tables of scripts_r1/make_supp_tables.py, all read from saved results:

- supp_s5_w4b.tex: sensor gain at d = 5, 10, 15, 30, 60, 90, 120 s (adds 15 s and 90 s);
- supp_s11_ba_external.tex (new): Bland-Altman agreement of all eleven models on WEEE and Ingraham;
- supp_s9_budget.tex: tuning-budget step statistic for the temporal models plus the tree models (RF 60 trials,
  XGBoost 100 trials), computed with the same rule from the saved trial logs.

The make_supp_tables helpers are reused and the files in outputs/tables/ are overwritten, so run this script after
scripts_r1/make_supp_tables.py. Usage: python paper_analysis/supp_tables_extra.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts_r1"))
import make_supp_tables as T  # noqa: E402

F, K, DS, U, m = T.F, T.K, T.DS, T.U, T.m
RES = REPO / "results_r1"


def lo_hi(x):
    """CI bound: keep the sign of a bound that rounds to zero, so that an interval crossing zero shows as such."""
    return "$-$0.00" if -0.005 < x < 0 else m(x)


def s5_w4b():
    ds_ = [5.0, 10.0, 15.0, 30.0, 60.0, 90.0, 120.0]
    rows = []
    for d in ["BEL", "WEEE", "ING", "WEEE_HRQ"]:
        t = pd.read_csv(K / d / "analysis" / "w4b_summary.csv")
        for j, learner in enumerate(["ridge", "xgb"]):
            q = t[t.learner == learner].set_index("d")
            cells = [f"{m(q.loc[x, 'gain_mean'])} [{lo_hi(q.loc[x, 'gain_bca_lo'])}, {lo_hi(q.loc[x, 'gain_bca_hi'])}]" for x in ds_]
            rows.append(f"{DS[d] if j == 0 else ''} & {'Ridge' if learner == 'ridge' else 'XGBoost'} & " + " & ".join(cells))
        rows.append("\\midrule")
    T.tab("supp_s5_w4b", "Baseline-relative gain $1-\\mathrm{RMSE}_{h+s}/\\mathrm{RMSE}_{h}$ of the wearable sensor set over a $\\dot{V}$O$_2$ history available $d$~s late: mean over participants with BCa 95\\% CI. Ridge with nested selection (primary reference learner); XGBoost with library defaults. The analysis also used $d=20$ and 45~s (results in the code repository).",
          "tab:s_w4b", "llccccccc", "Dataset & Learner & $d=5$~s & 10~s & 15~s & 30~s & 60~s & 90~s & 120~s", rows[:-1],
          size="\\scriptsize", wide=True)


def s11_ba_external():
    rows = []
    for d in ["WEEE", "ING"]:
        ba = pd.read_csv(K / d / "analysis" / "bland_altman.csv").set_index("model")
        for i, mm in enumerate(F.MODEL_ORDER):
            r = ba.loc[mm]
            rows.append(f"{DS[d] if i == 0 else ''} & {F.MODEL_LABEL[mm]} & {F.MODEL_FAMILY[mm]} & {m(r.bias)} & [{m(r.vc_loa_lo)}, {m(r.vc_loa_hi)}] & "
                        f"{r.s_within:.2f} & {r.s_between:.2f} & {m(r.ppt_bias_min)} to {m(r.ppt_bias_max)}")
        rows.append("\\midrule")
    T.tab("supp_s11_ba_external", f"Bland--Altman agreement of all eleven models with indirect calorimetry on the external datasets ({U}; WEEE $n=15$, Ingraham $n=10$). Columns as in Supplementary Table~\\ref{{tab:s_ba}}.",
          "tab:s_ba_ext", "lllccccc", "Dataset & Model & Family & Bias & 95\\% LoA & $s_w$ & $s_b$ & Per-participant bias range", rows[:-1],
          size="\\scriptsize")


def tree_steps(d, model, steps):
    imp = {s: [] for s in steps}
    for f in sorted((RES / d / model).glob("outer_*/inner_trials.csv")):
        t = pd.read_csv(f)
        t = t[t.state.isin(["COMPLETE", "PRUNED"])].sort_values("number")
        best = lambda k: t.iloc[:k].query("state == 'COMPLETE'").value.min()  # noqa: E731
        for a, b in steps:
            imp[(a, b)].append(best(a) - best(b))
    return {f"{a}->{b}": float(np.median(v)) for (a, b), v in imp.items()}


def s9_budget():
    cols = ["25->40", "40->60", "60->80", "80->100"]
    b = pd.read_csv(K / "budget_step_statistics.csv").pivot_table(index=["dataset", "model"], columns="step", values="median_improvement")
    rows = []
    for i, ((d, mm), r) in enumerate(b.iterrows()):
        first = i == 0 or b.index[i - 1][0] != d
        rows.append(f"{DS[d] if first else ''} & {F.MODEL_LABEL[mm]} & " + " & ".join(f"{r[s]:.3f}" for s in cols))
    rows.append("\\midrule")
    for j, d in enumerate(["BEL", "BEL_CORE", "ING", "WEEE"]):
        for k, (mm, steps) in enumerate([("RF", [(25, 40), (40, 60)]), ("XGB", [(25, 40), (40, 60), (60, 80), (80, 100)])]):
            s = tree_steps(d, mm, steps)
            cells = [f"{s[c]:.3f}" if c in s else "--" for c in cols]
            rows.append(f"{DS[d] if k == 0 else ''} & {F.MODEL_LABEL[mm]} & " + " & ".join(cells))
    T.tab("supp_s9_budget", f"Tuning-budget rule: median over outer folds of the improvement of the best inner-fold score ({U}) between successive trial counts, computed from the trial order of the final studies. Upper block, temporal models: the budget is adequate once every value is below 0.01, which holds from 80 trials; 100 trials were used throughout. Lower block, tree models (60 trials for the random forest, 100 for XGBoost), computed in the same way: every value is below 0.01 (at most 0.006).",
          "tab:s_budget", "llcccc", "Dataset & Model & 25$\\to$40 & 40$\\to$60 & 60$\\to$80 & 80$\\to$100", rows, size="\\scriptsize")


for f in (s5_w4b, s11_ba_external, s9_budget):
    f()
    print(f.__name__, "done", flush=True)
