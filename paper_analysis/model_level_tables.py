"""Model-level tables, from model_pairs.py outputs (paper_analysis/results/model_pairs_all.csv, model_pairs_margin.csv)
and released family-level result files (WEEE_ALL tree_linear_sensitivity.json, Ingraham ing_bad_breath_rescoring.csv).
Run scripts_r1/make_tables.py first: the sensor-gain, transition-penalty and bias-range rows of Table 3 are taken from
the outputs/tables/table5_datasets.tex it writes. Writes into outputs/tables/:
  table5_datasets.tex    main-text Table 3
  supp_pairs.tex         Supplementary Table: all 55 pairs on each dataset
  supp_s7_sensitivity.tex  Supplementary Table: sensitivity analyses (model level + descriptive family rows)
  supp_s10_margin.tex    Supplementary Table: equivalence-margin sensitivity (pair counts)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TAB = REPO / 'outputs' / 'tables'
K = REPO / 'results_r1' / 'K100'
R = pd.read_csv(HERE / 'results' / 'model_pairs_all.csv')
MG = pd.read_csv(HERE / 'results' / 'model_pairs_margin.csv')
NAME = dict(OLS='OLS', RIDGE='Ridge', LASSO='Lasso', ELASTICNET='Elastic net', RF='RF', XGB='XGBoost', LSTM='LSTM',
            GRU='GRU', TCN='TCN', TFT='TFT', PATCHTST='PatchTST')
LABEL = dict(BEL='Beltrame', WEEE='WEEE', ING='Ingraham', BEL_CORE='Beltrame, shared sensors', WEEE_HRQ='WEEE, heart-rate screen')
CLS = {'resolved': 'resolved', 'equivalent': 'equivalent', 'not resolved': 'not resolved'}


def f(x, d=2):
    s = f'{x:.{d}f}'
    return s.replace('-', '$-$') if x < 0 and float(s) != 0 else s.lstrip('-')


def fp(p):
    if p >= 0.995:
        return '1.00'
    if p < 0.001:
        return '$<$0.001'
    return f'{p:.3f}'


def counts(s):
    c = R[R.set == s]['class'].value_counts()
    return int(c.get('resolved', 0)), int(c.get('equivalent', 0)), int(c.get('not resolved', 0))


def pair(s, a, b):
    """Row for a - b (flips sign if stored as b - a)."""
    q = R[(R.set == s) & (R.a == a) & (R.b == b)]
    if len(q):
        z = q.iloc[0]
        return z, z['diff'], z.lo95, z.hi95
    z = R[(R.set == s) & (R.a == b) & (R.b == a)].iloc[0]
    return z, -z['diff'], -z.hi95, -z.lo95


def direction(z, a, b, d):
    if z['class'] != 'resolved':
        return CLS[z['class']]
    return f'{NAME[a] if d < 0 else NAME[b]} better'


def cell(s, a, b):
    z, d, lo, hi = pair(s, a, b)
    return f'{f(d)} [{f(lo)}, {f(hi)}]; $\\Delta^*$ {z.dstar:.2f}; {direction(z, a, b, d)}'


def ranking(s):
    q = R[R.set == s]
    m = pd.concat([q[['a', 'mean_a']].set_axis(['m', 'v'], axis=1), q[['b', 'mean_b']].set_axis(['m', 'v'], axis=1)])
    return m.groupby('m').v.first().sort_values()


def top_vs_rest(s):
    top = ranking(s).index[0]
    c = R[(R.set == s) & ((R.a == top) | (R.b == top))]['class'].value_counts()
    return top, int(c.get('resolved', 0)), int(c.get('equivalent', 0)), int(c.get('not resolved', 0))


def variance_shares(s):
    w = pd.read_csv(K / s / 'analysis' / 'per_participant_rmse.csv').set_index('pid')
    x = w[[m for m in NAME if m in w.columns]].to_numpy()
    n, k = x.shape
    g = x.mean()
    ss_p, ss_m = k * ((x.mean(1) - g) ** 2).sum(), n * ((x.mean(0) - g) ** 2).sum()
    ms_e = (((x - g) ** 2).sum() - ss_p - ss_m) / ((n - 1) * (k - 1))
    vp, vm = max(0, (ss_p / (n - 1) - ms_e) / k), max(0, (ss_m / (k - 1) - ms_e) / n)
    return vp / (vp + vm + ms_e), vm / (vp + vm + ms_e)


def table3():
    # sensor-gain, penalty and bias rows: from the table written by scripts_r1/make_tables.py
    old = (TAB / 'table5_datasets.tex').read_text(encoding='utf-8').splitlines()
    keep = {key: [l for l in old if key in l] for key in ['Ridge gain', 'Transition penalty', 'Per-participant bias range']}
    sets = ['BEL', 'WEEE', 'ING']
    sd = {s: R[R.set == s].sd_y.iloc[0] for s in sets}
    n = {s: int(R[R.set == s].n.iloc[0]) for s in sets}
    rk = {s: ranking(s) for s in sets}
    rows = [r' & $n$; SD of $\dot{V}\mathrm{O}_2$ & ' + ' & '.join(f'{n[s]}; {sd[s]:.2f}' for s in sets) + r' \\',
            r' & RMSE / SD of the eleven models & ' + ' & '.join(f'{rk[s].min() / sd[s]:.2f}--{rk[s].max() / sd[s]:.2f}' for s in sets) + r' \\',
            r' & Top-ranked model (RMSE / SD) & ' + ' & '.join(f'{NAME[rk[s].index[0]]} ({rk[s].iloc[0] / sd[s]:.2f})' for s in sets) + r' \\']
    rows.append(r'1 & Equivalence margin (0.05 SD) & ' + ' & '.join(f'{0.05 * sd[s]:.2f}' for s in sets) + r' \\')
    rows.append(r' & Smallest attainable $P_\mathrm{Holm}$ & ' + ' & '.join(f'{min(1, 55 * 2 / 2 ** n[s]):.3f}' for s in sets) + r' \\')
    rows.append(r' & Pairs resolved / equivalent / not resolved (of 55) & ' + ' & '.join('{} / {} / {}'.format(*counts(s)) for s in sets) + r' \\')
    rows.append(r' & Top-ranked model against the other 10 (resolved / equivalent / not resolved) & '
                + ' & '.join('{1} / {2} / {3}'.format(*top_vs_rest(s)) for s in sets) + r' \\')
    rows.append(r' & TCN $-$ RF & ' + ' & '.join(cell(s, 'TCN', 'RF') for s in sets) + r' \\')
    rows.append(r' & XGBoost $-$ RF & ' + ' & '.join(cell(s, 'XGB', 'RF') for s in sets) + r' \\')
    # item order: 1 significance/equivalence, 2 sensor gain, 3 transition penalty, 4 per-participant
    rows += [l.replace('1 & Ridge gain', '2 & Ridge gain', 1) for l in keep['Ridge gain']]
    rows += [l.replace('2 & Transition penalty', '3 & Transition penalty', 1) for l in keep['Transition penalty']]
    rows += keep['Per-participant bias range']
    vs = {s: variance_shares(s) for s in sets}
    rows.append(r' & Share of variance in per-participant RMSE: participants / models & '
                + ' & '.join(f'{vs[s][0]:.0%} / {vs[s][1]:.0%}'.replace('%', r'\%') for s in sets) + r' \\')
    cap = (r'\caption{The four evaluation items of the recommended reporting protocol (Section~\ref{sec:protocol}) on the three datasets. Absolute RMSE is not comparable across datasets; RMSE/SD is shown instead. '
           r'Item 1: all 55 pairs of models on each dataset; exact Wilcoxon $P$ with Holm correction over the 55 pairs ($P_\mathrm{Holm}$). '
           r'Classification: equivalent if the Holm-adjusted two one-sided tests at the margin of 0.05 $\times$ SD are significant; '
           r'otherwise a resolved difference if $P_\mathrm{Holm}<0.05$ and the 95\% CI excludes zero; otherwise not resolved. '
           r'Differences in mL\,kg$^{-1}$\,min$^{-1}$ with 95\% CI; $\Delta^*$ = largest difference compatible with the 90\% CI (unadjusted, descriptive). '
           r'With $n=10$ no pair can reach $P_\mathrm{Holm}<0.05$. Item 2: ridge reference learner, BCa 95\% CI. '
           r'Item 3: geometric-mean ratio of transition- to steady-stratum RMSE for the linear, tree and temporal families '
           r'(cadence detector on Beltrame; protocol timing on WEEE and Ingraham), not comparable in magnitude across datasets. '
           r'Item 4: range over participants of the mean bias, across all models; variance shares from a two-way random-effects decomposition of per-participant RMSE. '
           r'All pairs: Supplementary Table~\ref{tab:s_pairs}; sensitivity analyses: Supplementary Table~\ref{tab:s_sensitivity}.}')
    tex = '\n'.join([r'\begin{table}[ht!]', r'\centering', r'\scriptsize', cap, r'\label{tab:datasets_protocol}',
                     r'\begin{tabular}{p{0.03\linewidth}p{0.16\linewidth}p{0.22\linewidth}p{0.22\linewidth}p{0.22\linewidth}}',
                     r'\toprule', r'Item & Quantity & Beltrame ($n=16$) & WEEE ($n=15$) & Ingraham ($n=10$) \\', r'\midrule',
                     *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (TAB / 'table5_datasets.tex').write_text(tex, encoding='utf-8')


def supp_pairs():
    head = (r'Pair & Families & Difference [95\% CI] & $\Delta^*$ (SD) & $P_\mathrm{Holm}$ & $r_\mathrm{rb}$ & TOST $P_\mathrm{Holm}$ & Class \\')
    out = [r'\begin{scriptsize}', r'\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{llccccll}',
           r'\caption{All 55 pairwise model comparisons on each dataset. Difference = first $-$ second model in per-participant RMSE '
           r'(mL\,kg$^{-1}$\,min$^{-1}$; negative values favour the first model) with $t$-based 95\% CI; $\Delta^*$ = largest absolute difference compatible '
           r'with the 90\% CI, with $\Delta^*$/SD in brackets (unadjusted, descriptive); $P_\mathrm{Holm}$ = exact two-sided Wilcoxon (Pratt) $P$, Holm-adjusted over the 55 pairs; '
           r'$r_\mathrm{rb}$ = matched-pairs rank-biserial correlation; TOST $P_\mathrm{Holm}$ = two one-sided $t$-tests at $\pm0.05\times$SD, Holm-adjusted over the 55 pairs. '
           r'Class: equivalent if TOST $P_\mathrm{Holm}<0.05$; otherwise resolved if $P_\mathrm{Holm}<0.05$ and the 95\% CI excludes zero; otherwise not resolved. '
           r'Families are descriptive labels. Margins: Beltrame 0.266, WEEE 0.390, Ingraham 0.441~mL\,kg$^{-1}$\,min$^{-1}$. '
           r'With $n=10$ (Ingraham) the smallest attainable $P_\mathrm{Holm}$ is 0.107.}\label{tab:s_pairs}\\',
           r'\toprule', head, r'\midrule', r'\endfirsthead', r'\toprule', head, r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
    fam = dict(Linear='L', Tree='Tr', Temporal='Te')
    for s in ['BEL', 'WEEE', 'ING']:
        q = R[R.set == s]
        out.append(r'\multicolumn{8}{l}{\textbf{' + f'{LABEL[s]} ($n={int(q.n.iloc[0])}$)' + r'}} \\')
        for z in q.itertuples():
            cls = q.loc[z.Index, 'class']
            if cls == 'resolved':
                cls = f'{NAME[z.a] if z.diff < 0 else NAME[z.b]} better'
            out.append(f'{NAME[z.a]} $-$ {NAME[z.b]} & {fam[z.family_a]}--{fam[z.family_b]} & {f(z.diff)} [{f(z.lo95)}, {f(z.hi95)}] & '
                       f'{z.dstar:.2f} ({z.dstar_sd:.3f}) & {fp(z.p_holm)} & {f(z.r_rb)} & {fp(z.p_tost_holm)} & {cls} \\\\')
        if s != 'ING':
            out.append(r'\midrule')
    out += [r'\end{longtable}', r'\par\noindent Families: L, linear; Tr, tree; Te, temporal.', r'\end{scriptsize}', '']
    (TAB / 'supp_pairs.tex').write_text('\n'.join(out), encoding='utf-8')


def supp_sensitivity():
    rows = []
    for s, label in [('BEL_CORE', r'Beltrame, shared sensors (HR, $\Delta$HR, acceleration; $n=16$)'),
                     ('WEEE_HRQ', r'WEEE, four participants with implausible HR excluded ($n=11$; post hoc)')]:
        r_, e_, u_ = counts(s)
        top = ranking(s).index[0]
        first = True
        for a in ['TCN', 'GRU', 'XGB']:
            z, d, lo, hi = pair(s, a, 'RF')
            rows.append(f'{label if first else ""} & {NAME[a]} $-$ RF & {f(d)} [{f(lo)}, {f(hi)}] & {z.dstar:.2f} & {fp(z.p_holm)} & '
                        f'{direction(z, a, "RF", d)} \\\\')
            first = False
        rows.append(f' & All 55 pairs & \\multicolumn{{4}}{{l}}{{{r_} resolved, {e_} equivalent, {u_} not resolved; top-ranked {NAME[top]}; '
                    f'margin {R[R.set == s].margin.iloc[0]:.3f}; smallest attainable $P_\\mathrm{{Holm}}$ {min(1, 55 * 2 / 2 ** int(R[R.set == s].n.iloc[0])):.3f}}} \\\\')
        # equivalent pairs that also met the resolution criterion: classified equivalent, direction reported here
        q = R[R.set == s]
        both = q[(q['class'] == 'equivalent') & (q.p_holm < 0.05) & ((q.lo95 > 0) | (q.hi95 < 0))]
        note = [f'RF ranked {list(ranking(s).index).index("RF") + 1}']
        note += [f'{NAME[z.a]}--{NAME[z.b]} equivalent, {NAME[z.a] if z.diff < 0 else NAME[z.b]} more accurate ($P_\\mathrm{{Holm}}={z.p_holm:.3f}$)'
                 for z in both.itertuples()]
        rows.append(f' & & \\multicolumn{{4}}{{l}}{{{"; ".join(note)}}} \\\\')
        rows.append(r'\midrule')
    wa = json.load(open(K / 'WEEE_ALL' / 'tree_linear_sensitivity.json'))
    rows.append(r'WEEE, all participants ($n=17$; linear and tree models only)$^a$ & Tree family $-$ linear family & '
                f'{f(wa["mean_diff"])} [{f(wa["ci95_lo"])}, {f(wa["ci95_hi"])}] & -- & -- & descriptive \\\\')
    rows.append(r'\midrule')
    ib = pd.read_csv(K / 'ING' / 'analysis' / 'ing_bad_breath_rescoring.csv')
    base = ib[ib.variant.str.startswith('primary')].set_index('c')['diff']
    var = {'exclude rows nearest a bad breath': 'nearest samples excluded', 'exclude rows in the interpolation span of a bad breath': 'interpolation spans excluded',
           're-score against cleaned target (bad breaths removed, re-interpolated)': 'cleaned target'}
    shift = max(abs(ib[ib.variant == v].set_index('c')['diff'] - base).max() for v in var)
    for v, lab in var.items():
        q = ib[ib.variant == v].set_index('c')
        rows.append(f'{"Ingraham, implausible breaths$^a$: " + lab} & Family means (temporal$-$tree; temporal$-$linear; tree$-$linear) & '
                    + '; '.join(f(q.loc[c, 'diff']) for c in ['Temporal-Tree', 'Temporal-Linear', 'Tree-Linear']) + r' & -- & -- & descriptive \\')
    cap = (r'\caption{Sensitivity analyses. Model-level rows: difference (mL\,kg$^{-1}$\,min$^{-1}$) with 95\% CI, $\Delta^*$, $P_\mathrm{Holm}$ over the 55 pairs of the analysed set, '
           r'and the class of the main rule, with the margin set to 0.05 $\times$ SD of the analysed data. '
           r'Family rows$^a$ are descriptive differences between family means (the released results for these analyses are family-level only); '
           r'the Ingraham re-scoring changed these differences by at most ' + f'{shift:.3f}' + r'~mL\,kg$^{-1}$\,min$^{-1}$ '
           r'(primary analysis: ' + '; '.join(f(base[c], 3) for c in ['Temporal-Tree', 'Temporal-Linear', 'Tree-Linear']) + r'). '
           r'The WEEE heart-rate screen and the Ingraham re-scoring are post-hoc sensitivity analyses.}')
    tex = '\n'.join([r'\begin{table}[ht!]', r'\centering', r'\scriptsize', cap, r'\label{tab:s_sensitivity}', r'\resizebox{\linewidth}{!}{%',
                     r'\begin{tabular}{p{0.3\linewidth}p{0.22\linewidth}cccl}', r'\toprule',
                     r'Analysis & Comparison & Difference [95\% CI] & $\Delta^*$ & $P_\mathrm{Holm}$ & Class \\', r'\midrule',
                     *rows, r'\bottomrule', r'\end{tabular}}', r'\end{table}', ''])
    (TAB / 'supp_s7_sensitivity.tex').write_text(tex, encoding='utf-8')
    return shift


def supp_margin():
    sets = ['BEL', 'WEEE', 'ING', 'BEL_CORE', 'WEEE_HRQ']
    ms = [0.03, 0.05, 0.08, 0.10]
    rows = []
    for s in sets:
        m = MG[MG.set == s].set_index('margin_sd')
        rows.append(f'{LABEL[s]} & ' + ' & '.join(f'{int(m.loc[x, "equivalent"])} ({int(m.loc[x, "resolved"])})' for x in ms) + r' \\')
    rows.append(r'\midrule')
    for a, b in [('TCN', 'RF'), ('GRU', 'RF'), ('XGB', 'RF'), ('GRU', 'TCN')]:
        for s in ['BEL', 'WEEE', 'ING']:
            z, d, lo, hi = pair(s, a, b)
            cl = []
            for x in ms:
                c = z[f'class_{round(x * 100):02d}sd']
                cl.append(direction(z, a, b, d) if c == 'resolved' else CLS[c])
            rows.append(f'{NAME[a]} $-$ {NAME[b]}, {LABEL[s]} ($\\Delta^*$/SD {z.dstar_sd:.3f}) & ' + ' & '.join(cl) + r' \\')
    cap = (r'\caption{Equivalence-margin sensitivity. Upper part: number of the 55 pairs classified as equivalent (in brackets: resolved) at each margin, '
           r'in units of the analysed data\textquotesingle{}s SD of $\dot{V}$O$_2$; the main analysis uses $0.05\times$SD. Lower part: class of selected pairs at each margin. '
           r'Equivalence: Holm-adjusted two one-sided tests over the 55 pairs. $\Delta^*$/SD, the unadjusted bound, is the smallest margin at which the pair alone '
           r'would be declared equivalent; with the Holm adjustment a larger margin is needed.}')
    tex = '\n'.join([r'\begin{table}[ht!]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}', cap, r'\label{tab:s_margin}',
                     r'\begin{tabular}{lcccc}', r'\toprule',
                     r'Dataset / pair & $0.03\times$SD & $0.05\times$SD (main) & $0.08\times$SD & $0.10\times$SD \\', r'\midrule',
                     *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (TAB / 'supp_s10_margin.tex').write_text(tex, encoding='utf-8')


if __name__ == '__main__':
    table3()
    supp_pairs()
    print('re-scoring shift', supp_sensitivity())
    supp_margin()
    print('written to', TAB)
