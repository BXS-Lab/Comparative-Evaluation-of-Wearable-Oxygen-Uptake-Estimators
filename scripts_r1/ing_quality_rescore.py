"""ING target quality: impact of breaths with VO2 <= 0 and spike breaths on scoring,
without retraining. Results saved to
results_r1/K100/ING/analysis/ing_bad_breath_rescoring.csv (family contrasts) and _transition.csv."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import r1.data_external as X
from r1.datasets import load
K = REPO / "results_r1/K100"
FAM = {"Linear": ["OLS", "RIDGE", "LASSO", "ELASTICNET"], "Tree": ["RF", "XGB"], "Temporal": ["LSTM", "GRU", "TCN", "TFT", "PATCHTST"]}
panel = load("ING"); df = panel.df; g = np.asarray(panel.eval_mask())
N = len(df)
direct = np.zeros(N, bool); span = np.zeros(N, bool); ycl = np.full(N, np.nan); hist_bad = np.zeros(N, bool)
row0 = 0; nb = dict(le0=0, spike=0, total=0); spikes = []
for subj in range(1, 11):
    mass = X.ING_MASS[subj]
    for blk, m, apdm in X._ing_blocks(subj):
        tb, code_b = m[0], m[1]; vo2_b = m[2] * 60.0 / mass; bf_b, hr_b = m[5], m[8]
        o = np.argsort(tb); tb, code_b, vo2_b, bf_b, hr_b = tb[o], code_b[o], vo2_b[o], bf_b[o], hr_b[o]
        grid = np.arange(int(np.ceil(tb.min())), int(np.floor(tb.max())) + 1)
        y = np.interp(grid, tb, vo2_b); lb = np.searchsorted(tb, grid, side="right") - 1
        hr = hr_b[lb]; bf = bf_b[lb]; gap_after = np.r_[np.diff(tb), 0.0]; bad_t = gap_after[lb] > X.GAP_BREATH_S
        ta = apdm[0]; mag = np.sqrt((apdm[2:5] ** 2).sum(axis=0)); ok = np.isfinite(ta) & np.isfinite(mag); ta, mag = ta[ok], mag[ok]
        lo = np.searchsorted(ta, grid - 1, side="right"); hi = np.searchsorted(ta, grid, side="right")
        c1 = np.r_[0, np.cumsum(mag)]; c2 = np.r_[0, np.cumsum(mag ** 2)]; cnt = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = (c1[hi] - c1[lo]) / cnt; acc = np.sqrt(np.maximum((c2[hi] - c2[lo]) / cnt - mu ** 2, 0) * cnt / np.maximum(cnt - 1, 1))
        acc[cnt < 64] = np.nan
        valid = ~bad_t & np.isfinite(acc) & np.isfinite(hr) & np.isfinite(bf) & np.isfinite(y)
        med = pd.Series(vo2_b).rolling(11, center=True, min_periods=3).median().to_numpy()
        isspike = (vo2_b > 2.5 * med) & (vo2_b - med > 8)
        bad = (vo2_b <= 0) | isspike
        nb["le0"] += int((vo2_b <= 0).sum()); nb["spike"] += int(isspike.sum()); nb["total"] += len(vo2_b)
        spikes += list(vo2_b[isspike])
        keep = ~bad
        ycl_blk = np.interp(grid, tb[keep], vo2_b[keep])
        nxt = np.minimum(lb + 1, len(tb) - 1)
        exact = np.isclose(grid, tb[lb])
        sp = bad[lb] | (bad[nxt] & ~exact)
        # nearest breath within 0.5 s
        near_idx = np.where(np.abs(grid - tb[lb]) <= np.abs(tb[nxt] - grid), lb, nxt)
        dr = bad[near_idx] & (np.abs(grid - tb[near_idx]) <= 0.5)
        for s, e in X._segments_from_valid(valid, X.GAP_1HZ_S):
            n = e - s
            assert np.allclose(df.y.to_numpy()[row0:row0 + n], y[s:e], atol=1e-12, rtol=0), (subj, blk)
            direct[row0:row0 + n] = dr[s:e]; span[row0:row0 + n] = sp[s:e]; ycl[row0:row0 + n] = ycl_blk[s:e]
            hist_bad[row0:row0 + n] = bad[lb[s:e]]
            row0 += n
assert row0 == N
print("breaths", nb, "spike max", np.round(sorted(spikes)[-5:], 1))
import json as _json
(K / "ING" / "analysis" / "ing_bad_breath_counts.json").write_text(_json.dumps({**nb, "bad_total": nb["le0"] + nb["spike"], "spike_max": float(max(spikes)), "grid_rows_affected": int((span & g).sum()), "grid_rows": int(g.sum()), "fraction": float((span & g).sum() / g.sum())}, indent=2))
print("rows: y<=0 all", int((df.y <= 0).sum()), "grid", int(((df.y <= 0) & g).sum()))
print("direct rows (grid)", int((direct & g).sum()), "span rows (grid)", int((span & g).sum()), "of", int(g.sum()), f"({(span & g).sum() / g.sum():.4%})",
      "y_hist from bad breath (grid)", int((hist_bad & g).sum()), "cleaned y differs (grid)", int(((np.abs(ycl - df.y.to_numpy()) > 1e-9) & g).sum()))
print("rows per pid (span, grid):", pd.Series(df.pid[span & g]).value_counts().sort_index().to_dict())
# predictions
fr = []
for m in sum(FAM.values(), []):
    for f in (K / "ING" / m).glob("outer_*/predictions_seed*.csv"):
        d = pd.read_csv(f, usecols=["row", "pid", "y_true", "y_pred"]); d["model"] = m; d["seed"] = int(f.stem[-1]); fr.append(d)
P = pd.concat(fr, ignore_index=True)
reg = df.regime.where(g).to_numpy()
OUT_C, OUT_T = [], []


def analyse(Q, label):
    Q = Q.assign(e2=(Q.y_pred - Q.y_true) ** 2)
    R = (Q.groupby(["model", "seed", "pid"]).e2.mean() ** .5).groupby(["model", "pid"]).mean().unstack("model")
    fam = pd.DataFrame({f: R[ms].mean(axis=1) for f, ms in FAM.items()})
    sd = Q.drop_duplicates("row").y_true.std(ddof=1)
    rows = []
    for a, b in [("Temporal", "Tree"), ("Temporal", "Linear"), ("Tree", "Linear")]:
        d = (fam[a] - fam[b]).to_numpy(); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
        h95 = stats.t.ppf(.975, n - 1) * se; h90 = stats.t.ppf(.95, n - 1) * se
        rows.append(dict(c=f"{a}-{b}", diff=d.mean(), lo=d.mean() - h95, hi=d.mean() + h95, dstar=abs(d.mean()) + h90, p=stats.wilcoxon(d, method="exact").pvalue))
    t = pd.DataFrame(rows); t["p_holm"] = multipletests(t.p, method="holm")[1]
    ms, ml = .05 * sd, .1 * sd
    def cls(r):
        if r.c != "Temporal-Tree": return ("ML better" if r["diff"] < 0 else "linear better") if r.p_holm < .05 else "unresolved"
        if r.dstar < ms: return "O1"
        if r.p_holm < .05 and (r.hi < 0 or r.lo > 0): return "O2-" + ("Temporal" if r["diff"] < 0 else "Tree")
        return "O1L" if r.dstar < ml else "O3"
    t["class"] = t.apply(cls, axis=1)
    # transition
    Q = Q.assign(st=reg[Q.row.to_numpy()]); Q = Q[pd.notna(Q.st)]
    mse = Q.groupby(["model", "seed", "pid", "st"]).e2.mean().groupby(["model", "pid", "st"]).mean()
    sse = Q.groupby(["model", "seed", "pid", "st"]).e2.sum().groupby(["model", "pid", "st"]).mean()
    ns = Q[(Q.model == "OLS") & (Q.seed == 0)].groupby("st").size(); ns = ns / ns.sum()
    tr = []
    for f, msl in FAM.items():
        fm = mse.loc[msl].groupby(["pid", "st"]).mean().unstack(); fs = sse.loc[msl].groupby(["pid", "st"]).mean().unstack()
        lr = 0.5 * np.log(fm.transition / fm.steady); n = len(lr); h = stats.t.ppf(.975, n - 1) * lr.std(ddof=1) / np.sqrt(n)
        share = fs.sum() / fs.sum().sum(); highest = fm.mean().idxmax()
        lo = np.exp(lr.mean() - h); r3 = "robust" if lo > 1 else ("point-only" if lr.mean() > 0 else "none")
        tr.append(dict(family=f, pen=np.exp(lr.mean()), lo=lo, hi=np.exp(lr.mean() + h), R3=r3,
                       claim=bool(highest == "transition" and share["transition"] > ns["transition"] and r3 == "robust")))
    OUT_C.append(t.assign(variant=label, sd=sd, m_strict=ms))
    OUT_T.append(pd.DataFrame(tr).assign(variant=label))
    print(f"--- {label}: rows/seed-model {Q[(Q.model=='OLS')].shape[0]} SD {sd:.4f} m_strict {ms:.4f}")
    print(t.round(4).to_string(index=False)); print(pd.DataFrame(tr).round(4).to_string(index=False))
    print("leaderboard top3:", R.mean().sort_values().round(3).head(3).to_dict())
analyse(P, "primary (all grid rows)")
analyse(P[~direct[P.row.to_numpy()]], "exclude rows nearest a bad breath")
analyse(P[~span[P.row.to_numpy()]], "exclude rows in the interpolation span of a bad breath")
analyse(P.assign(y_true=ycl[P.row.to_numpy()]), "re-score against cleaned target (bad breaths removed, re-interpolated)")
out = K / "ING" / "analysis"
pd.concat(OUT_C).to_csv(out / "ing_bad_breath_rescoring.csv", index=False)
pd.concat(OUT_T).to_csv(out / "ing_bad_breath_rescoring_transition.csv", index=False)
print("saved")
