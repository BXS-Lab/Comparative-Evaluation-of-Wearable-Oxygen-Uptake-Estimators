"""Post-hoc stratum analysis. The 55-pair classification rule of model_pairs.py applied to the
per-participant RMSE within the steady and the transition stratum, and to the transition - steady RMSE gap, for the
detector and oracle partitions of the primary dataset and the protocol-timing partitions of WEEE and Ingraham. Reads
the released K = 100 predictions.
Usage: python paper_analysis/strata_pairs.py > paper_analysis/results/strata_pairs_out.txt"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts_r1'), str(HERE)]
from r1 import evaluate as E
from r1.datasets import load, regime_labels
import make_figures as MF
import model_pairs as MP
out = []
for ds, part in [('BEL', 'detector'), ('BEL', 'oracle'), ('WEEE', 'protocol'), ('ING', 'protocol')]:
    panel = load(ds)
    pred = E.load_predictions(ds, root=ROOT / 'results_r1' / 'K100')
    reg = MF.oracle_regime(panel) if part == 'oracle' else regime_labels(panel)
    p = pred.assign(stratum=reg.reindex(pred.row).to_numpy(), e2=(pred.y_pred - pred.y_true) ** 2)
    p = p[p.stratum.notna()]
    M = p.groupby(['model', 'seed', 'pid', 'stratum'])['e2'].mean().groupby(['model', 'pid', 'stratum']).mean().unstack()
    R = M ** 0.5
    sd = json.load(open(ROOT / 'results_r1' / 'K100' / ds / 'analysis' / 'summary.json'))['sd_y']
    for s in ['steady', 'transition', 'gap']:
        T = (R.transition - R.steady) if s == 'gap' else R[s]
        T = T.unstack('model')[MP.MODELS].dropna()
        r = MP.pairs(T, sd)
        mean = T.mean().sort_values()
        top = mean.index[0]
        rt = r[(r.a == top) | (r.b == top)]
        res = r[r['class'] == 'resolved']
        print(f'== {ds} {part} {s}: n={len(T)}  resolved/equiv/not = '
              f"{(r['class']=='resolved').sum()}/{(r['class']=='equivalent').sum()}/{(r['class']=='not resolved').sum()}; "
              f'CI excl 0: {((r.lo95>0)|(r.hi95<0)).sum()}; min P_Holm {r.p_holm.min():.3f}')
        print('   ranking: ' + ', '.join(f'{m} {v:.2f}' for m, v in mean.items()))
        print(f'   top {top} vs 10: ' + str(rt['class'].value_counts().to_dict()))
        for _, x in res.iterrows():
            print(f'   resolved {x.a}-{x.b}: {x["diff"]:+.3f} [{x.lo95:.2f},{x.hi95:.2f}] P_Holm={x.p_holm:.3f}')
        x = r[(r.a == 'RF') & (r.b == 'TCN')].iloc[0]
        print(f'   RF-TCN: {x["diff"]:+.3f} [{x.lo95:.2f},{x.hi95:.2f}] P_Holm={x.p_holm:.3f} D*={x.dstar:.2f} {x["class"]}')
