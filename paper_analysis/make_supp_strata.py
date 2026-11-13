"""Supplementary table: per-model stratum RMSE, transition penalty, and ranks (overall, steady, transition, transitions
re-weighted to 50% of samples). Writes outputs/tables/supp_s12_strata.tex from saved K = 100 predictions.
Usage: python paper_analysis/make_supp_strata.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'tables' / 'supp_s12_strata.tex'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts_r1'))
from r1 import evaluate as E  # noqa: E402
from r1 import stats as S  # noqa: E402
from r1.datasets import load, regime_labels  # noqa: E402
import make_figures as MF  # noqa: E402

LABEL = {'RF': 'RF', 'XGB': 'XGBoost', 'RIDGE': 'Ridge', 'ELASTICNET': 'Elastic net', 'LASSO': 'Lasso', 'OLS': 'OLS',
         'PATCHTST': 'PatchTST', 'GRU': 'GRU', 'TFT': 'TFT', 'LSTM': 'LSTM', 'TCN': 'TCN'}
ST = ['steady', 'middle', 'transition']
W50 = pd.Series([0.25, 0.25, 0.5], index=ST)

lines, notes, pen = [], {}, {}
for ds, part, title in [('BEL', 'detector', 'Beltrame, detector'), ('BEL', 'oracle', 'Beltrame, oracle'),
                        ('WEEE', 'protocol', 'WEEE, protocol timing'), ('ING', 'protocol', 'Ingraham, protocol timing')]:
    panel = load(ds)
    pred = E.load_predictions(ds, root=ROOT / 'results_r1' / 'K100')
    reg = MF.oracle_regime(panel) if part == 'oracle' else regime_labels(panel)
    p = pred.assign(stratum=reg.reindex(pred.row).to_numpy(), e2=(pred.y_pred - pred.y_true) ** 2)
    p = p[p.stratum.notna()]
    M = p.groupby(['model', 'seed', 'pid', 'stratum'])['e2'].mean().groupby(['model', 'pid', 'stratum']).mean().unstack()
    R = M ** 0.5
    LP = np.log(R.transition / R.steady).unstack('model')[E.MODELS].dropna()
    pen[ds, part] = int((pd.DataFrame(S.pairwise(LP.to_numpy(), list(LP.columns))).p_holm < 0.05).sum())
    lb = pd.read_csv(ROOT / 'results_r1' / 'K100' / ds / 'analysis' / 'leaderboard.csv').set_index('model')
    t = pd.DataFrame({'steady': R.steady.groupby('model').mean(), 'transition': R.transition.groupby('model').mean(),
                      'penalty': np.exp(np.log(R.transition / R.steady).groupby('model').mean()),
                      'overall': lb.rmse,
                      'w50': (M[ST].mul(W50, axis=1).sum(axis=1) ** 0.5).groupby('model').mean()})
    for c in ['overall', 'steady', 'transition', 'w50']:
        t['r_' + c] = t[c].rank().astype(int)
    t = t.sort_values('overall')
    if ds == 'BEL':
        cnt = []
        for s in ['steady', 'transition']:
            T = R[s].unstack('model')[E.MODELS].dropna()
            cnt.append(int((pd.DataFrame(S.pairwise(T.to_numpy(), list(T.columns))).p_holm < 0.05).sum()))
        notes[part] = cnt
    for i, (m, r) in enumerate(t.iterrows()):
        lines.append(f"{title if i == 0 else ''} & {LABEL[m]} & {E.FAMILY_OF[m]} & {r.steady:.2f} & {r.transition:.2f} & "
                     f"{r.penalty:.2f} & {int(r.r_overall)} & {int(r.r_steady)} & {int(r.r_transition)} & "
                     f"{int(r.r_w50)} \\\\")
    lines.append('\\midrule')
lines[-1] = '\\bottomrule'

assert not any(pen.values()), pen  # the caption states that no pair differed in penalty
cap = ("Error of every model by regime stratum. Steady and transition RMSE (mL\\,kg$^{-1}$\\,min$^{-1}$) are means of "
       "per-participant RMSE within the stratum; the penalty is the geometric mean of the per-participant "
       "transition-to-steady RMSE ratio. Ranks: overall (Table~\\ref{tab:s_leaderboards} and main-text "
       "Table~\\ref{tab:model_comparison}), within the steady and transition strata, and after re-weighting the three "
       "strata so that transitions make up 50\\% of the samples (steady and middle 25\\% each). On the primary dataset, "
       f"exact Wilcoxon tests with Holm correction over the 55 pairs resolved {notes['detector'][0]} pairs in the steady "
       f"and {notes['detector'][1]} in the transition stratum of the detector partition, and {notes['oracle'][0]} and "
       f"{notes['oracle'][1]} of the oracle partition; no pair differed in penalty on any dataset (exact Wilcoxon tests on the per-participant difference of the log "
       "penalties, Holm correction over the 55 pairs; on Ingraham, with $n=10$, none could). Stratum-level comparisons "
       "of RMSE are reported for the primary dataset only.")
tex = ("\\begin{table}[ht!]\n\\centering\n\\scriptsize\n\\renewcommand{\\arraystretch}{0.9}\n\\caption{" + cap + "}\n\\label{tab:s_strata}\n"
       "\\begin{tabular}{lllccccccc}\n\\toprule\n"
       " & & & \\multicolumn{3}{c}{RMSE and penalty} & \\multicolumn{4}{c}{Rank} \\\\\n"
       "\\cmidrule(lr){4-6}\\cmidrule(lr){7-10}\n"
       "Partition & Model & Family & Steady & Transition & Penalty & Overall & Steady & Transition & Trans.\\ 50\\% \\\\\n"
       "\\midrule\n" + "\n".join(lines) + "\n\\end{tabular}\n\\end{table}\n")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(tex, encoding='utf-8')
print(tex)
