"""Dataset-agnostic evaluation module.

Input: the per-model prediction files written by the runners (rows of the evaluation grid, one file per seed)
plus the panel (for regime markers). Output: tables under results_r1/<DATASET>/analysis/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as S
from .cv import RESULTS_ROOT

FAMILIES = {"Linear": ["OLS", "RIDGE", "LASSO", "ELASTICNET"], "Tree": ["RF", "XGB"],
            "Temporal": ["LSTM", "GRU", "TCN", "TFT", "PATCHTST"]}
MODELS = [m for ms in FAMILIES.values() for m in ms]
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}


def load_predictions(dataset, models=MODELS, root=RESULTS_ROOT):
    frames = []
    for m in models:
        for f in sorted((Path(root) / dataset / m).glob("outer_*/predictions_seed*.csv")):
            d = pd.read_csv(f)
            d["model"] = m
            d["seed"] = int(f.stem.replace("predictions_seed", ""))
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def per_participant(pred, value="rmse"):
    """(participant x model) matrix; per-seed metric then mean over seeds."""
    g = pred.assign(e2=(pred.y_pred - pred.y_true) ** 2).groupby(["model", "seed", "pid"])["e2"].mean()
    if value == "rmse":
        g = g ** 0.5
    return g.groupby(["model", "pid"]).mean().unstack("model")


def leaderboard(pred, R):
    rows = []
    sd_y = pred.drop_duplicates("row").y_true.std(ddof=1)
    for m in R.columns:
        sub = pred[pred.model == m]
        pooled = [(np.corrcoef(g.y_true, g.y_pred)[0, 1],
                   1 - ((g.y_pred - g.y_true) ** 2).sum() / ((g.y_true - g.y_true.mean()) ** 2).sum())
                  for _, g in sub.groupby("seed")]
        per_seed = sub.assign(e2=(sub.y_pred - sub.y_true) ** 2).groupby(["seed", "pid"])["e2"].mean() ** 0.5
        seed_rmse = per_seed.groupby("seed").mean()
        lo, hi = S.bca_mean(R[m].to_numpy())
        rows.append({"model": m, "family": FAMILY_OF[m], "rmse": R[m].mean(), "rmse_bca_lo": lo, "rmse_bca_hi": hi,
                     "nrmse": R[m].mean() / sd_y, "r_pooled": np.mean([p[0] for p in pooled]),
                     "r2_pooled": np.mean([p[1] for p in pooled]), "n_seeds": int(seed_rmse.size),
                     "seed_range": float(seed_rmse.max() - seed_rmse.min()) if seed_rmse.size > 1 else 0.0})
    return pd.DataFrame(rows).sort_values("rmse")


def family_contrasts(R, sd_y):
    fam = pd.DataFrame({f: R[ms].mean(axis=1) for f, ms in FAMILIES.items()})
    pairs = [("Temporal", "Tree"), ("Temporal", "Linear"), ("Tree", "Linear")]
    out = []
    for a, b in pairs:
        d = (fam[a] - fam[b]).to_numpy()
        p, _, _ = S.wilcoxon_exact(d)
        sd = d.std(ddof=1)
        lo95, hi95 = S.t_ci(d, 0.95)
        lo90, hi90 = S.t_ci(d, 0.90)
        ds = S.delta_star(d)
        out.append({"contrast": f"{a} - {b}", "mean_diff": d.mean(), "ci95_lo": lo95, "ci95_hi": hi95,
                    "ci90_lo": lo90, "ci90_hi": hi90, "delta_star": ds, "delta_star_over_sd": ds / sd_y,
                    "delta_star_over_linear_rmse": ds / fam["Linear"].mean(), "p": p,
                    "rank_biserial": S.rank_biserial(d), "mde": S.mde_difference(sd, len(d)),
                    "eq_mde": S.mde_equivalence(sd, len(d)),
                    "tost_p_strict": S.tost_p(d, 0.05 * sd_y), "tost_p_lenient": S.tost_p(d, 0.10 * sd_y),
                    "tost_p_025": S.tost_p(d, 0.25)})
    adj = S.holm([o["p"] for o in out])
    for o, q in zip(out, adj):
        o["p_holm"] = float(q)
    tt = next(o for o in out if o["contrast"] == "Temporal - Tree")
    d_tt = (fam["Temporal"] - fam["Tree"]).to_numpy()
    tt["class"], tt["sublabel"] = S.classify_temporal_tree(d_tt, tt["p_holm"], 0.05 * sd_y, 0.10 * sd_y)
    for o in out:
        if o["contrast"] in ("Temporal - Linear", "Tree - Linear"):
            o["class"] = S.r1_class((fam[o["contrast"].split(" - ")[0]] - fam["Linear"]).to_numpy(), o["p_holm"])
            o["sublabel"] = ""
    return pd.DataFrame(out), fam


def transition_analysis(pred, regime):
    """regime: Series indexed by panel row -> 'steady' | 'middle' | 'transition'. Family-level per-participant MSE per
    stratum, penalty (geometric mean ratio, t-CI) and additive squared-error shares."""
    p = pred.assign(stratum=regime.reindex(pred.row).to_numpy(), e2=(pred.y_pred - pred.y_true) ** 2)
    p = p[p.stratum.notna()]
    mse = p.groupby(["model", "seed", "pid", "stratum"])["e2"].mean().groupby(["model", "pid", "stratum"]).mean()
    sse = p.groupby(["model", "seed", "pid", "stratum"])["e2"].sum().groupby(["model", "pid", "stratum"]).mean()
    n_s = p[(p.model == p.model.iloc[0]) & (p.seed == p.seed.iloc[0])].groupby("stratum").size()
    rows = []
    for f, ms in FAMILIES.items():
        fm = mse.loc[ms].groupby(["pid", "stratum"]).mean().unstack("stratum")
        fs = sse.loc[ms].groupby(["pid", "stratum"]).mean().unstack("stratum")
        ok = fm[["steady", "transition"]].dropna()
        logr = 0.5 * np.log(ok["transition"] / ok["steady"])  # log RMSE ratio
        lo, hi = S.t_ci(logr.to_numpy(), 0.95)
        share = fs.sum() / fs.sum().sum()
        rec = {"family": f, "n_participants_penalty": len(ok), "penalty_gmean": float(np.exp(logr.mean())),
               "penalty_ci_lo": float(np.exp(lo)), "penalty_ci_hi": float(np.exp(hi))}
        for s in ["steady", "middle", "transition"]:
            if s in fm:
                rec[f"rmse_{s}"] = float((fm[s] ** 0.5).mean())
                rec[f"mse_{s}"] = float(fm[s].mean())
                rec[f"sse_share_{s}"] = float(share.get(s, np.nan))
                rec[f"sample_share_{s}"] = float(n_s.get(s, 0) / n_s.sum())
        rec["R3_class"] = "robust" if rec["penalty_ci_lo"] > 1 else ("point-only" if rec["penalty_gmean"] > 1 else "none")
        highest = max(["steady", "middle", "transition"], key=lambda s: rec.get(f"mse_{s}", -np.inf))
        rec["claim_concentrates_in_transitions"] = bool(highest == "transition" and
                                                        rec["sse_share_transition"] > rec["sample_share_transition"]
                                                        and rec["R3_class"] == "robust")
        rows.append(rec)
    return pd.DataFrame(rows)


def bland_altman(pred):
    rows = []
    for m, sub in pred.groupby("model"):
        recs = []
        for _, g in sub.groupby("seed"):
            d = g.y_pred - g.y_true
            pb = d.groupby(g.pid).mean()
            k = pb.size
            n_i = d.groupby(g.pid).size()
            N = n_i.sum()
            msb = (n_i * (pb - d.mean()) ** 2).sum() / (k - 1)
            msw = ((d - g.pid.map(pb)) ** 2).sum() / (N - k)
            n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
            sb2 = max(0.0, (msb - msw) / n0)
            sd_tot = np.sqrt(sb2 + msw)
            recs.append({"bias": d.mean(), "naive_sd": d.std(ddof=1), "s_within": np.sqrt(msw), "s_between": np.sqrt(sb2),
                         "vc_loa_lo": d.mean() - 1.96 * sd_tot, "vc_loa_hi": d.mean() + 1.96 * sd_tot,
                         "ppt_bias_min": pb.min(), "ppt_bias_max": pb.max()})
        r = pd.DataFrame(recs).mean().to_dict()
        r.update(model=m, family=FAMILY_OF[m])
        rows.append(r)
    return pd.DataFrame(rows)


def bel_regime(panel):
    """BEL: pooled tercile cut points of the cadence detector over the evaluation grid."""
    df = panel.df
    g = panel.eval_mask()
    sig = df["regime_signal"].where(g)
    q1, q2 = np.nanquantile(sig, [1 / 3, 2 / 3])
    lab = pd.Series(np.where(sig <= q1, "steady", np.where(sig > q2, "transition", "middle")), index=df.index)
    return lab.where(g)


def run(dataset, panel, root=RESULTS_ROOT, regime=None):
    out = Path(root) / dataset / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pred = load_predictions(dataset, root=root)
    R = per_participant(pred)[MODELS].dropna()  # complete blocks only
    sd_y = float(pred.drop_duplicates("row").y_true.std(ddof=1))
    lb = leaderboard(pred, R)
    chi2, F, p_om = S.iman_davenport(R.to_numpy())
    pw = pd.DataFrame(S.pairwise(R.to_numpy(), list(R.columns))) if len(R) >= 12 and p_om < 0.05 else pd.DataFrame()
    fc, fam = family_contrasts(R, sd_y)
    lb.to_csv(out / "leaderboard.csv", index=False)
    R.to_csv(out / "per_participant_rmse.csv")
    fam.to_csv(out / "family_per_participant_rmse.csv")
    fc.to_csv(out / "family_contrasts.csv", index=False)
    if len(pw):
        pw.to_csv(out / "pairwise.csv", index=False)
    summary = {"n_participants": int(len(R)), "sd_y": sd_y, "m_strict": 0.05 * sd_y, "m_lenient": 0.10 * sd_y,
               "friedman_chi2": chi2, "iman_davenport_F": F, "omnibus_p": p_om,
               "pairwise_tested": bool(len(pw)), "n_pairs_holm_sig": int((pw.p_holm < 0.05).sum()) if len(pw) else None}
    if regime is not None:
        tr = transition_analysis(pred, regime)
        tr.to_csv(out / "transition.csv", index=False)
    ba = bland_altman(pred)
    ba.to_csv(out / "bland_altman.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, lb, fc
