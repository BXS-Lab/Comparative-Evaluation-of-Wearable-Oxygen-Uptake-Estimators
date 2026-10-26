"""Supplementary outputs at K = 100.

Baseline-relative gain summary: gain(d) = 1 - RMSE(history+sensors)/RMSE(history) per participant; mean, BCa 95% CI, n positive,
    per dataset x learner x d                                   -> results_r1/K100/<DS>/analysis/w4b_summary.csv
Oracle partition, BEL only: oracle partition (centred 15 s rolling mean of |dV̇O₂| within segment, pooled terciles on the evaluation
    grid, same rule as the detector) -> transition analysis; detector-oracle Spearman rho (pooled over the grid) with a
    participant-cluster bootstrap 95% CI                      -> oracle_transition.csv, detector_oracle_rho.json
Error by activity, WEEE/ING: per-participant RMSE by activity (seed-averaged MSE), family mean      -> per_activity_rmse.csv
Also: BUDGET (budget-rule step statistics from the K = 100 studies) and WEEE_ALL (all-participants sensitivity analysis).
Usage: python scripts_r1/supplementary_k100.py BEL ING WEEE WEEE_HRQ BUDGET WEEE_ALL"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1 import evaluate as E  # noqa: E402
from r1 import stats as S  # noqa: E402
from r1.datasets import load  # noqa: E402

RES = ROOT / "results_r1"
K100 = RES / "K100"


def w4b_summary(dataset):
    t = pd.read_csv(RES / dataset / "w4" / "w4b_per_participant.csv")
    rows = []
    for learner, g in t.groupby("learner"):
        hist = g[g.arm == "history"].pivot_table(index="pid", columns="d", values="rmse")
        both = g[g.arm == "history+sensors"].pivot_table(index="pid", columns="d", values="rmse")
        for d in hist.columns:
            gain = (1 - both[d] / hist[d]).dropna().to_numpy()
            m, (lo, hi) = gain.mean(), S.bca_mean(gain)
            rows.append({"learner": learner, "d": d, "n": len(gain), "gain_mean": m, "gain_bca_lo": lo, "gain_bca_hi": hi,
                         "n_positive": int((gain > 0).sum()), "rmse_history": hist[d].mean(), "rmse_history_sensors": both[d].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(K100 / dataset / "analysis" / "w4b_summary.csv", index=False)
    return out


def oracle(panel, dataset, n_boot=2000, seed=20260601):
    df, g = panel.df, panel.eval_mask()
    dv = df.groupby("seg")["y"].diff().abs()
    orc = dv.groupby(df["seg"]).transform(lambda s: s.rolling(15, center=True, min_periods=1).mean())
    sig = orc.where(g)
    q1, q2 = np.nanquantile(sig, [1 / 3, 2 / 3])
    lab = pd.Series(np.where(sig <= q1, "steady", np.where(sig > q2, "transition", "middle")), index=df.index).where(g)
    pred = E.load_predictions(dataset, root=K100)
    tr = E.transition_analysis(pred, lab)
    tr.to_csv(K100 / dataset / "analysis" / "oracle_transition.csv", index=False)
    det = df["regime_signal"]
    ok = g & det.notna() & orc.notna()
    x, y, pid = det[ok].to_numpy(), orc[ok].to_numpy(), df.loc[ok, "pid"].to_numpy()
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    P = np.unique(pid)
    idx = {q: np.flatnonzero(pid == q) for q in P}
    boots = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[q] for q in rng.choice(P, len(P))])
        boots.append(spearmanr(x[sel], y[sel]).statistic)
    per_pid = [spearmanr(x[idx[q]], y[idx[q]]).statistic for q in P]
    res = {"rho_pooled": float(rho), "rho_cluster_boot_lo": float(np.percentile(boots, 2.5)),
           "rho_cluster_boot_hi": float(np.percentile(boots, 97.5)), "rho_per_participant_median": float(np.median(per_pid)),
           "rho_per_participant_min": float(np.min(per_pid)), "rho_per_participant_max": float(np.max(per_pid)),
           "n_samples": int(ok.sum()), "n_boot": n_boot,
           "agreement_same_stratum": float((E.bel_regime(panel)[g] == lab[g]).mean())}
    (K100 / dataset / "analysis" / "detector_oracle_rho.json").write_text(json.dumps(res, indent=2))
    return tr, res


ING_REST = {"22": "Standing", "23": "Sitting"}   # ING activity codes 22/23 = standing/sitting periods within blocks
MIN_CELL_S = 60                                    # per-participant activity cells with < 60 s of grid samples are not reported


def per_activity(panel, dataset):
    """Per-participant RMSE by activity. Model RMSE per participant from seed-averaged MSE; family value =
    mean of member models' per-participant RMSE (as in the family contrasts); summary = mean over participants."""
    df = panel.df
    if "block" in df:
        code = df["condition"].str.split(":").str[1]
        act = code.map(ING_REST).fillna(df["block"].str.replace(r"\d+$", "", regex=True))
    else:
        act = df["condition"]
    pred = E.load_predictions(dataset, root=K100)
    pred = pred.assign(activity=act.reindex(pred.row).to_numpy(), e2=(pred.y_pred - pred.y_true) ** 2)
    n = pred[(pred.model == E.MODELS[0]) & (pred.seed == 0)].groupby(["pid", "activity"]).size()
    rmse = pred.groupby(["model", "seed", "pid", "activity"])["e2"].mean().groupby(["model", "pid", "activity"]).mean() ** 0.5
    rmse = rmse[rmse.index.droplevel("model").map(lambda k: n.get(k, 0) >= MIN_CELL_S)]
    fam = pd.concat({f: rmse.loc[ms].groupby(["pid", "activity"]).mean() for f, ms in E.FAMILIES.items()}, axis=1)
    fam.to_csv(K100 / dataset / "analysis" / "per_activity_rmse_per_participant.csv")
    out = fam.groupby("activity").mean()
    out["n_participants"] = fam.groupby("activity").size()
    out["n_samples"] = n.groupby("activity").sum()
    out.to_csv(K100 / dataset / "analysis" / "per_activity_rmse.csv")
    return out


def budget_steps(datasets=("BEL", "BEL_CORE", "WEEE", "ING")):
    """Step statistic of the budget rule from the trial order of the persisted K = 100 studies."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    fin = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    rows = []
    for d in datasets:
        for m in E.FAMILIES["Temporal"]:
            steps = {s: [] for s in [(25, 40), (40, 60), (60, 80), (80, 100)]}
            for db in sorted((RES / f"{d}_T100" / m).glob("outer_*/study.db")):
                st = optuna.load_study(study_name="study", storage=f"sqlite:///{db.as_posix()}")
                tr = sorted((t for t in st.trials if t.state in fin), key=lambda t: t.number)
                best = lambda ts: min((t.value for t in ts if t.state == optuna.trial.TrialState.COMPLETE), default=np.inf)
                for a, b in steps:
                    steps[(a, b)].append(best(tr[:a]) - best(tr[:b]))
            for (a, b), v in steps.items():
                rows.append({"dataset": d, "model": m, "step": f"{a}->{b}", "median_improvement": float(np.median(v)),
                             "below_0.01": bool(np.median(v) < 0.01), "n_folds": len(v)})
    out = pd.DataFrame(rows)
    out.to_csv(K100 / "budget_step_statistics.csv", index=False)
    return out.pivot_table(index=["dataset", "model"], columns="step", values="median_improvement")


def weee_all():
    """WEEE all-participants sensitivity (linear and tree models only): Tree - Linear, exact Wilcoxon, t-CI."""
    ms = E.FAMILIES["Linear"] + E.FAMILIES["Tree"]
    R = E.per_participant(E.load_predictions("WEEE_ALL", models=ms, root=RES))[ms].dropna()
    d = (R[E.FAMILIES["Tree"]].mean(1) - R[E.FAMILIES["Linear"]].mean(1)).to_numpy()
    p, _, _ = S.wilcoxon_exact(d)
    lo, hi = S.t_ci(d, 0.95)
    res = {"dataset": "WEEE_ALL", "n": int(len(d)), "contrast": "Tree - Linear", "mean_diff": float(d.mean()),
           "ci95_lo": float(lo), "ci95_hi": float(hi), "p_exact_wilcoxon": float(p), "class": S.r1_class(d, p)}
    (K100 / "WEEE_ALL").mkdir(exist_ok=True)
    (K100 / "WEEE_ALL" / "tree_linear_sensitivity.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    for dataset in sys.argv[1:]:
        if dataset == "BUDGET":
            print(budget_steps().round(4).to_string(), flush=True)
            continue
        if dataset == "WEEE_ALL":
            print(weee_all(), flush=True)
            continue
        panel = load(dataset)
        print(f"== {dataset}\n", w4b_summary(dataset).round(3).to_string(index=False), flush=True)
        if dataset == "BEL":
            tr, res = oracle(panel, dataset)
            print(tr[["family", "rmse_steady", "rmse_middle", "rmse_transition", "penalty_gmean", "penalty_ci_lo",
                      "penalty_ci_hi", "R3_class"]].round(3).to_string(index=False))
            print(json.dumps(res, indent=1))
        else:
            print(per_activity(panel, dataset).round(2).to_string())


if __name__ == "__main__":
    main()
