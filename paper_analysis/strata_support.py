"""Post-hoc stratum analysis. Per partition and dataset: sample and squared-error shares of the
strata, per-model transition penalties and exact Wilcoxon tests on the per-participant difference of the log
penalties (Holm over the 55 pairs), the RMSE lead of TCN and GRU over the random forest, XGBoost and the linear mean
in each stratum with the geometric-mean penalty ratio and its t-based 95% CI, and the change in RMSE when the strata
are re-weighted (transitions 50% or 10%). Partitions: cadence detector and oracle
on the primary dataset, protocol timing on WEEE and Ingraham. The "act" ranks are a reconstruction; Supplementary
Table S14 takes the overall rank from the leaderboard.
Usage: python paper_analysis/strata_support.py > paper_analysis/results/strata_support_out.txt"""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts_r1'), str(HERE)]
from r1 import evaluate as E, stats as S
from r1.datasets import load, regime_labels
import make_figures as MF
import model_pairs as MP
LIN = ['OLS', 'RIDGE', 'LASSO', 'ELASTICNET']
ST = ['steady', 'middle', 'transition']
for ds, part in [('BEL', 'detector'), ('BEL', 'oracle'), ('WEEE', 'protocol'), ('ING', 'protocol')]:
    panel = load(ds)
    pred = E.load_predictions(ds, root=ROOT / 'results_r1' / 'K100')
    reg = MF.oracle_regime(panel) if part == 'oracle' else regime_labels(panel)
    p = pred.assign(stratum=reg.reindex(pred.row).to_numpy(), e2=(pred.y_pred - pred.y_true) ** 2)
    p = p[p.stratum.notna()]
    M = p.groupby(['model', 'seed', 'pid', 'stratum'])['e2'].mean().groupby(['model', 'pid', 'stratum']).mean().unstack()
    R = M ** 0.5
    # per-participant sample shares (one seed/model)
    one = p[(p.model == 'RF') & (p.seed == p.seed.min())]
    sh = one.groupby(['pid', 'stratum']).size().unstack().fillna(0)
    sh = sh.div(sh.sum(1), axis=0)
    print(f'== {ds} {part}: pooled shares ' + ', '.join(f'{s} {one.stratum.eq(s).mean():.0%}' for s in ST))
    # squared-error share of transition stratum (pooled, over all models)
    tot = p.groupby('stratum').e2.sum()
    print(f'   transition share of squared error (all models pooled): {tot["transition"]/tot.sum():.0%}; '
          f'per-model range: ' + '{:.0%}-{:.0%}'.format(*[f(p.groupby(["model","stratum"]).e2.sum().unstack().pipe(lambda d: d.transition/d.sum(1))) for f in (min, max)]))
    # log penalty per participant
    LP = np.log(R.transition / R.steady).unstack('model')[MP.MODELS]
    pen = np.exp(LP.mean())
    print('   penalty: ' + ', '.join(f'{m} {v:.2f}' for m, v in pen.sort_values().items()))
    rows = []
    for a, b in itertools.combinations(MP.MODELS, 2):
        d = (LP[a] - LP[b]).dropna().to_numpy()
        rows.append((a, b, S.wilcoxon_exact(d)[0], *S.t_ci(d, 0.95)))
    pr = pd.DataFrame(rows, columns=['a', 'b', 'p', 'lo', 'hi'])
    pr['ph'] = S.holm(pr.p.to_numpy())
    print(f'   penalty pairs Holm<0.05: {(pr.ph < 0.05).sum()}; CI excludes 0: {((pr.lo > 0) | (pr.hi < 0)).sum()}; min P_Holm {pr.ph.min():.3f}')
    # relative lead of the top models: ratio of mean RMSE, by stratum
    mean = R.groupby('model').mean()
    for top in ['TCN', 'GRU']:
        for ref in [['RF'], LIN, ['XGB']]:
            r_st = 1 - mean.loc[top, 'steady'] / mean.loc[ref, 'steady'].mean()
            r_tr = 1 - mean.loc[top, 'transition'] / mean.loc[ref, 'transition'].mean()
            # penalty ratio top/ref (per participant, geometric), CI
            d = (LP[top] - LP[ref].mean(1)).dropna().to_numpy()
            lo, hi = S.t_ci(d, 0.95)
            print(f'   {top} vs {"/".join(ref) if len(ref)==1 else "linear mean"}: lower RMSE by {r_st:.1%} steady, {r_tr:.1%} transition; '
                  f'penalty ratio {np.exp(d.mean()):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}]')
    # RMSE at evaluated mix vs re-weighted mixes
    def rw(w):
        return (M[ST].mul(pd.Series(w, index=ST), axis=1).sum(1) ** 0.5).groupby('model').mean()
    act = (M[ST].mul(sh.reindex(M.index.get_level_values('pid'))[ST].to_numpy()).sum(1) ** 0.5).groupby('model').mean()
    w50 = rw([0.25, 0.25, 0.5]); w10 = rw([0.45, 0.45, 0.10]); eq = rw([1/3, 1/3, 1/3])
    ch50 = (w50 / act - 1); ch10 = (w10 / act - 1)
    print(f'   RMSE at evaluated mix {act.min():.2f}-{act.max():.2f}; transitions 50%: change {ch50.min():+.1%} to {ch50.max():+.1%}; '
          f'transitions 10%: {ch10.min():+.1%} to {ch10.max():+.1%}')
    rk = lambda s: s.rank().astype(int)
    order = act.sort_values().index
    print('   ranks act/steady/trans/w50/w10: ' + '; '.join(f'{m} {rk(act)[m]}/{rk(mean.steady)[m]}/{rk(mean.transition)[m]}/{rk(w50)[m]}/{rk(w10)[m]}' for m in order))
    print(f'   range of model means: steady {mean.steady.max()-mean.steady.min():.2f}, transition {mean.transition.max()-mean.transition.min():.2f}; '
          f'transition-steady gap per model {(mean.transition-mean.steady).min():.2f}-{(mean.transition-mean.steady).max():.2f}')
