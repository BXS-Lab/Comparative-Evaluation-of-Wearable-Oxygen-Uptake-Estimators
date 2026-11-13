"""Post-hoc check of the transition strata against VO2 level (read-only on the released results).

For the primary dataset (cadence-detector and oracle partitions) and the external datasets (protocol timing): mean VO2
per stratum, and the transition penalty (geometric mean over participants of transition/steady RMSE, seed-averaged MSE)
for the random forest, ridge and TCN, (i) as reported, and (ii) within pooled quintiles of measured VO2 (penalty from
the mean over participants and quintile bins of the log RMSE ratio), so that intensity is matched between strata.

Usage: python paper_analysis/strata_level.py
Writes paper_analysis/results/strata_level_out.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO), str(REPO / 'scripts_r1')]
from r1 import evaluate as E  # noqa: E402
from r1.datasets import load, regime_labels  # noqa: E402
import make_figures as MF  # noqa: E402

MODELS = ['RF', 'RIDGE', 'TCN']
out = []
for ds, part in [('BEL', 'detector'), ('BEL', 'oracle'), ('WEEE', 'protocol'), ('ING', 'protocol')]:
    panel = load(ds)
    reg = MF.oracle_regime(panel) if part == 'oracle' else regime_labels(panel)
    df = panel.df.assign(stratum=reg)
    df = df[df.stratum.notna()]
    lev = df.groupby('stratum').y.mean().round(2).to_dict()
    pred = E.load_predictions(ds, models=MODELS, root=REPO / 'results_r1' / 'K100')
    p = pred.assign(stratum=reg.reindex(pred.row).to_numpy(), e2=(pred.y_pred - pred.y_true) ** 2)
    p = p[p.stratum.notna()]
    M = p.groupby(['model', 'seed', 'pid', 'stratum']).e2.mean().groupby(['model', 'pid', 'stratum']).mean().unstack()
    pen = np.exp(0.5 * np.log(M.transition / M.steady).groupby('model').mean())
    p = p.assign(bin=pd.qcut(p.y_true, 5, labels=False))
    Q = p.groupby(['model', 'seed', 'pid', 'bin', 'stratum']).e2.mean().groupby(['model', 'pid', 'bin', 'stratum']).mean().unstack()
    penq = np.exp(0.5 * np.log(Q.transition / Q.steady).groupby('model').mean())
    out.append(f'== {ds} {part}: mean VO2 by stratum (mL/kg/min) {lev}')
    out.append(f'   penalty as reported: {pen.round(2).to_dict()}')
    out.append(f'   penalty within VO2 quintiles: {penq.round(2).to_dict()}')
txt = '\n'.join(out) + '\n'
(HERE / 'results' / 'strata_level_out.txt').write_text(txt, encoding='utf-8')
print(txt)
