"""Model-level comparison of the eleven models: all 55 pairs on every dataset.

Uses only the released code (r1/stats.py) and released result files (results_r1/K100/<set>/analysis/
per_participant_rmse.csv and summary.json) of the archived pipeline (Zenodo v2.0.0). The only departure from that
pipeline is that the pairwise tests are not gated on the omnibus test and n >= 12 (r1/evaluate.py), and that no
family contrasts are tested.

For every pair of the 11 models on every set: paired difference of per-participant RMSE (a - b), t-based 95% and 90%
CI, Delta* (largest |bound| of the 90% CI), exact two-sided Wilcoxon (Pratt) with Holm over the 55 pairs of the set,
matched-pairs rank-biserial correlation, and t-based two one-sided tests (TOST) at the margin 0.05 x SD of the set's
VO2, Holm-adjusted over the 55 pairs. Class: equivalent if the Holm-adjusted TOST P < 0.05; otherwise resolved if the
Holm-adjusted Wilcoxon P < 0.05 and the 95% CI excludes 0; otherwise not resolved.

Usage: python paper_analysis/model_pairs.py
Writes model_pairs_all.csv, model_pairs_margin.csv and model_pairs_summary.txt to paper_analysis/results/.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / 'results'
sys.path.insert(0, str(REPO))
from r1 import stats as S  # noqa: E402

K = REPO / 'results_r1' / 'K100'
SETS = ['BEL', 'WEEE', 'ING', 'BEL_CORE', 'WEEE_HRQ']
MAIN = ['BEL', 'WEEE', 'ING']
MODELS = ['OLS', 'RIDGE', 'LASSO', 'ELASTICNET', 'RF', 'XGB', 'LSTM', 'GRU', 'TCN', 'TFT', 'PATCHTST']
FAMILY = {m: 'Linear' for m in MODELS[:4]} | {'RF': 'Tree', 'XGB': 'Tree'} | {m: 'Temporal' for m in MODELS[6:]}
MARGINS_SD = [0.03, 0.05, 0.08, 0.10]
ABS_REF = 0.25  # absolute TOST reference (primary dataset only)


def tag(m):
    return f'{round(m * 100):02d}sd'


def load(s):
    w = pd.read_csv(K / s / 'analysis' / 'per_participant_rmse.csv').set_index('pid')
    sd = json.load(open(K / s / 'analysis' / 'summary.json'))['sd_y']
    return w[[m for m in MODELS if m in w.columns]], sd


def classify(p_tost_holm, p_holm, lo95, hi95):
    res = (p_holm < 0.05) & ((lo95 > 0) | (hi95 < 0))
    return np.where(p_tost_holm < 0.05, 'equivalent', np.where(res, 'resolved', 'not resolved'))


def pairs(w, sd):
    rows = []
    for a, b in itertools.combinations(w.columns, 2):
        d = (w[a] - w[b]).to_numpy()
        lo95, hi95 = S.t_ci(d, 0.95)
        lo90, hi90 = S.t_ci(d, 0.90)
        rows.append(dict(a=a, b=b, family_a=FAMILY[a], family_b=FAMILY[b], n=len(d), mean_a=w[a].mean(), mean_b=w[b].mean(),
                         diff=d.mean(), lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90, dstar=S.delta_star(d),
                         p=S.wilcoxon_exact(d)[0], r_rb=S.rank_biserial(d),
                         **{f'p_tost_{tag(m)}': S.tost_p(d, m * sd) for m in MARGINS_SD},
                         p_tost_abs=S.tost_p(d, ABS_REF)))
    r = pd.DataFrame(rows)
    r['sd_y'] = sd
    r['margin'] = 0.05 * sd
    r['dstar_sd'] = r.dstar / sd
    r['p_holm'] = S.holm(r.p.to_numpy())
    for m in MARGINS_SD:
        r[f'p_tost_holm_{tag(m)}'] = S.holm(r[f'p_tost_{tag(m)}'].to_numpy())
        r[f'class_{tag(m)}'] = classify(r[f'p_tost_holm_{tag(m)}'], r.p_holm, r.lo95, r.hi95)
    r['p_tost_abs_holm'] = S.holm(r.p_tost_abs.to_numpy())
    r['p_tost_holm'] = r['p_tost_holm_05sd']
    r['class'] = r['class_05sd']
    return r


def variance_shares(w):
    """Two-way random-effects ANOVA (participant x model, one value per cell); negative components truncated at 0."""
    x = w.to_numpy()
    n, k = x.shape
    g = x.mean()
    ss_p = k * ((x.mean(1) - g) ** 2).sum()
    ss_m = n * ((x.mean(0) - g) ** 2).sum()
    ss_e = ((x - g) ** 2).sum() - ss_p - ss_m
    ms_p, ms_m, ms_e = ss_p / (n - 1), ss_m / (k - 1), ss_e / ((n - 1) * (k - 1))
    vp, vm = max(0.0, (ms_p - ms_e) / k), max(0.0, (ms_m - ms_e) / n)
    tot = vp + vm + ms_e
    return vp / tot, vm / tot, ms_e / tot


def main():
    out, marg, txt = [], [], []
    data = {s: load(s) for s in SETS}
    for s in SETS:
        w, sd = data[s]
        r = pairs(w, sd)
        r.insert(0, 'set', s)
        out.append(r)
        n = len(w)
        cnt = r['class'].value_counts().reindex(['resolved', 'equivalent', 'not resolved'], fill_value=0)
        rank = w.mean().sort_values()
        top = rank.index[0]
        t = r[(r.a == top) | (r.b == top)]['class'].value_counts().to_dict()
        txt.append(f'== {s}: n={n}, SD={sd:.2f}, margin={0.05 * sd:.3f}; min attainable P_Holm={min(1, 55 * 2 / 2 ** n):.3f}; '
                   f'min raw P={r.p.min():.2g}')
        txt.append('   ranking: ' + ', '.join(f'{m} {v:.2f}' for m, v in rank.items()))
        txt.append(f'   classes (resolved/equivalent/not resolved): {cnt["resolved"]}/{cnt["equivalent"]}/{cnt["not resolved"]}; '
                   f'95% CI excludes 0 but not resolved: {int((((r.lo95 > 0) | (r.hi95 < 0)) & (r["class"] != "resolved")).sum())}; '
                   f'Delta* < margin but not equivalent: {int(((r.dstar < r.margin) & (r["class"] != "equivalent")).sum())}')
        txt.append(f'   top model {top} vs other 10: {t}')
        for z in r[r['class'] == 'resolved'].itertuples():
            txt.append(f'   resolved {z.a}-{z.b}: {z.diff:+.3f} [{z.lo95:.3f}, {z.hi95:.3f}] P_Holm={z.p_holm:.3f} r_rb={z.r_rb:.2f} D*={z.dstar:.3f}')
        for z in r[r['class'] == 'equivalent'].itertuples():
            txt.append(f'   equivalent {z.a}-{z.b}: {z.diff:+.3f} D*={z.dstar:.3f} TOST P_Holm={z.p_tost_holm:.3g}')
        for z in r[(r.a == 'RF') | (r.b == 'RF')].itertuples():
            d = z.diff if z.b == 'RF' else -z.diff
            lo, hi = (z.lo95, z.hi95) if z.b == 'RF' else (-z.hi95, -z.lo95)
            other = z.a if z.b == 'RF' else z.b
            txt.append(f'   {other} - RF: {d:+.3f} [{lo:.3f}, {hi:.3f}] D*={z.dstar:.3f} ({z.dstar_sd:.3f} SD) P_Holm={z.p_holm:.3f} TOST P_Holm={z.p_tost_holm:.3f} {r.loc[z.Index, "class"]}')
        for m in MARGINS_SD:
            c = r[f'class_{tag(m)}'].value_counts()
            marg.append(dict(set=s, margin_sd=m, margin=m * sd, resolved=int(c.get('resolved', 0)),
                             equivalent=int(c.get('equivalent', 0)), not_resolved=int(c.get('not resolved', 0))))
        if s == 'BEL':
            txt.append(f'   TOST at +-{ABS_REF} (Holm over 55): {int((r.p_tost_abs_holm < 0.05).sum())} equivalent: '
                       + ', '.join(f'{a}-{b}' for a, b in r[r.p_tost_abs_holm < 0.05][['a', 'b']].itertuples(index=False)))
        if s in MAIN:
            vp, vm, ve = variance_shares(w)
            best = w.idxmin(axis=1).value_counts()
            psd = w.std(ddof=1)
            txt.append(f'   variance shares: participants {vp:.1%}, models {vm:.1%}, residual {ve:.1%}')
            txt.append(f'   best per participant: {best.to_dict()} ({len(best)} different models; top model {top} best for {best.get(top, 0)} of {n})')
            txt.append(f'   between-participant SD per model {psd.min():.2f}-{psd.max():.2f}; range of model means {rank.max() - rank.min():.2f}')
    allr = pd.concat(out, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    allr.to_csv(OUT / 'model_pairs_all.csv', index=False, float_format='%.6g')
    pd.DataFrame(marg).to_csv(OUT / 'model_pairs_margin.csv', index=False, float_format='%.4g')

    # cross-dataset comparison (main datasets)
    piv = allr[allr.set.isin(MAIN)].pivot_table(index=['a', 'b'], columns='set', values='class', aggfunc='first')[MAIN]
    dif = allr[allr.set.isin(MAIN)].pivot_table(index=['a', 'b'], columns='set', values='diff')[MAIN]
    txt.append('== Cross-dataset (BEL, WEEE, ING)')
    eq3 = piv[(piv == 'equivalent').all(axis=1)]
    txt.append(f'   equivalent on all three: {len(eq3)}: ' + ', '.join(f'{a}-{b}' for a, b in eq3.index))
    for (a, b), row in piv[piv.BEL == 'resolved'].iterrows():
        dd = dif.loc[(a, b)]
        txt.append(f'   BEL-resolved {a}-{b}: BEL {dd.BEL:+.2f}; WEEE {dd.WEEE:+.2f} ({row.WEEE}); ING {dd.ING:+.2f} ({row.ING})')
    same = (np.sign(dif.BEL) == np.sign(dif.WEEE)) & (np.sign(dif.BEL) == np.sign(dif.ING))
    txt.append(f'   pairs with the same direction on all three: {int(same.sum())} of {len(dif)}')
    (OUT / 'model_pairs_summary.txt').write_text('\n'.join(txt) + '\n', encoding='utf-8')
    print('\n'.join(txt))


if __name__ == '__main__':
    main()
