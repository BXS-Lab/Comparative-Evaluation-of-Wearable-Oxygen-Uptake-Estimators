"""Main-text figures of the revised manuscript, all from saved K = 100 results, in one style (scripts_r1/figstyle.py).

Fig. 2 causal filtering example · Fig. 3 per-participant metrics · Fig. 4 BCa CIs · Fig. 5 baseline-relative gain ·
Fig. 6 transition strata (detector, oracle) · Fig. 7 Bland–Altman (TCN) · Fig. 8 BEL family contrasts ·
Fig. 9 time series · Fig. 10 family contrasts across datasets.
Output: outputs/figures/fig<k>_<name>.{pdf,png}. Usage: python scripts_r1/make_figures.py [2 3 ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_r1"))
import figstyle as F  # noqa: E402
from r1 import evaluate as E  # noqa: E402
from r1 import stats as S  # noqa: E402
from r1.datasets import load  # noqa: E402
from r1.preprocess import causal_filter  # noqa: E402

RES = ROOT / "results_r1"
K100 = RES / "K100"
OUT = ROOT / "outputs" / "figures"
CONTRASTS = ["Temporal - Tree", "Temporal - Linear", "Tree - Linear"]
CLABEL = {"Temporal - Tree": "Temporal − Tree", "Temporal - Linear": "Temporal − Linear", "Tree - Linear": "Tree − Linear"}
_cache = {}


def plain_label(c, x, m_strict):
    """Reader-facing label from the single classification rule (figstyle.contrast_label)."""
    return F.contrast_label(c, x, m_strict)


def pred(ds="BEL"):
    if ds not in _cache:
        _cache[ds] = E.load_predictions(ds, root=K100)
    return _cache[ds]


def seed_mean(p, model):
    return p[p.model == model].groupby("row").agg(pid=("pid", "first"), t=("t", "first"), y_true=("y_true", "first"),
                                                  y_pred=("y_pred", "mean"))


def tci(x):
    x = np.asarray(x, float)
    lo, hi = S.t_ci(x, 0.95)
    return x.mean(), lo, hi


# ---------------------------------------------------------------- Fig. 2
def fig2():
    panel = load("BEL")
    df = panel.df
    d = df[df.pid == 1]
    filt = causal_filter(d[panel.features], d["seg"], 0.01)
    names = {"HR": ("HR", "beats min$^{-1}$"), "dHR": (r"$\Delta$HR", "beats min$^{-1}$"),
             "VE": (r"$\dot{V}_\mathrm{E}$", "mL min$^{-1}$"), "BF": ("BF", "breaths min$^{-1}$"),
             "ACC": ("Hacc", "g"), "CAD": ("CAD", "steps min$^{-1}$")}
    order = ["HR", "dHR", "VE", "BF", "ACC", "CAD"]
    fig, axes = plt.subplots(3, 2, figsize=(F.FULL, 4.6), sharex=True)
    tmin = np.arange(len(d)) / 60
    for ax, f, k in zip(axes.flat, order, "ABCDEF"):
        ax.plot(tmin, d[f].to_numpy(), color="#b5b5b5", lw=0.5, label="Raw")
        ax.plot(tmin, np.asarray(filt)[:, panel.features.index(f)], color=F.INK, lw=1.0, label="Causal low-pass (0.01 Hz)")
        ax.set_ylabel(f"{names[f][0]} ({names[f][1]})")
        F.panel(ax, k, x=-0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time (min)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.tight_layout(h_pad=0.6, w_pad=1.5, rect=(0, 0, 1, 0.95))
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    F.save(fig, OUT, "fig2_causal_filter")


# ---------------------------------------------------------------- Fig. 3
def per_participant_metrics(p):
    rows = []
    for (m, s, q), g in p.groupby(["model", "seed", "pid"]):
        e = g.y_pred - g.y_true
        rmse = float(np.sqrt((e ** 2).mean()))
        r = float(np.corrcoef(g.y_true, g.y_pred)[0, 1])
        r2 = float(1 - (e ** 2).sum() / ((g.y_true - g.y_true.mean()) ** 2).sum())
        rows.append((m, s, q, rmse, r, r2))
    t = pd.DataFrame(rows, columns=["model", "seed", "pid", "RMSE", "r", "R2"])
    return t.groupby(["model", "pid"])[["RMSE", "r", "R2"]].mean().reset_index()


def fig3():
    t = per_participant_metrics(pred())
    order = t.groupby("model").RMSE.mean().sort_values().index.tolist()
    fig, axes = plt.subplots(3, 1, figsize=(F.FULL, 5.2), sharex=True)
    lab = {"RMSE": f"RMSE ({F.UNIT})", "r": "Pearson $r$", "R2": "$R^2$"}
    for ax, met, k in zip(axes, ["RMSE", "r", "R2"], "ABC"):
        data = [t.loc[t.model == m, met].to_numpy() for m in order]
        vp = ax.violinplot(data, positions=range(len(order)), widths=0.75, showextrema=False)
        for body, m in zip(vp["bodies"], order):
            body.set_facecolor(F.FAMILY_COLOR[F.MODEL_FAMILY[m]])
            body.set_edgecolor("none")
            body.set_alpha(0.28)
        rng = np.random.default_rng(1)
        for i, (m, v) in enumerate(zip(order, data)):
            c = F.FAMILY_COLOR[F.MODEL_FAMILY[m]]
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color=c, lw=0, alpha=0.9, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=F.INK, lw=1.0, zorder=4)
        ax.set_ylabel(lab[met])
        F.panel(ax, k, x=-0.08)
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels([F.MODEL_LABEL[m] for m in order], rotation=30, ha="right")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in F.FAMILY_COLOR.items()]
    axes[0].legend(handles=handles, loc="upper left", ncol=3)
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.5)
    F.save(fig, OUT, "fig3_metrics")


# ---------------------------------------------------------------- Fig. 4
def fig4():
    lb = pd.read_csv(K100 / "BEL" / "analysis" / "leaderboard.csv").sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(F.WIDE, 3.0))
    for i, r in enumerate(lb.itertuples()):
        c = F.FAMILY_COLOR[r.family]
        ax.plot([r.rmse_bca_lo, r.rmse_bca_hi], [i, i], color=c, lw=1.4, solid_capstyle="butt")
        ax.plot(r.rmse, i, "o", color=c, ms=4.5, mec="white", mew=0.6, zorder=3)
    best = lb.iloc[-1]
    ax.axvspan(best.rmse_bca_lo, best.rmse_bca_hi, color=F.GRID, zorder=0, lw=0)
    ax.set_yticks(range(len(lb)))
    ax.set_yticklabels([F.MODEL_LABEL[m] for m in lb.model])
    ax.set_xlabel(f"Mean per-participant RMSE ({F.UNIT}), BCa 95% CI")
    handles = [Line2D([], [], marker="o", color=c, label=f) for f, c in F.FAMILY_COLOR.items()]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    F.save(fig, OUT, "fig4_bca")


# ---------------------------------------------------------------- Fig. 5
def gains(ds, learner="ridge"):
    t = pd.read_csv(RES / ds / "w4" / "w4b_per_participant.csv")
    t = t[t.learner == learner]
    h = t[t.arm == "history"].pivot_table(index="pid", columns="d", values="rmse")
    b = t[t.arm == "history+sensors"].pivot_table(index="pid", columns="d", values="rmse")
    return 1 - b / h


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL, 2.7), sharey=True)
    g = gains("BEL")
    ax = axes[0]
    for _, row in g.iterrows():
        ax.plot(g.columns, row.values, color="#bdbdbd", lw=0.6, zorder=1)
    s = pd.read_csv(K100 / "BEL" / "analysis" / "w4b_summary.csv").query("learner == 'ridge'").sort_values("d")
    c = F.DATASET_COLOR["BEL"]
    ax.fill_between(s.d, s.gain_bca_lo, s.gain_bca_hi, color=c, alpha=0.18, lw=0)
    ax.plot(s.d, s.gain_mean, color=c, marker="o", ms=3.5, lw=1.4)
    F.panel(ax, "A")
    ax = axes[1]
    for ds in ["BEL", "ING", "WEEE"]:
        s = pd.read_csv(K100 / ds / "analysis" / "w4b_summary.csv").query("learner == 'ridge'").sort_values("d")
        c = F.DATASET_COLOR[ds]
        ax.fill_between(s.d, s.gain_bca_lo, s.gain_bca_hi, color=c, alpha=0.15, lw=0)
        ax.plot(s.d, s.gain_mean, color=c, marker=F.DATASET_MARKER[ds], ms=3.5, lw=1.3)
        ax.text(s.d.iloc[-1] + 3, s.gain_mean.iloc[-1], F.DATASET_LABEL[ds].split(" (")[0], va="center", fontsize=7, color=F.INK)
    F.panel(ax, "B", x=-0.06)
    for ax in axes:
        ax.axhline(0, color=F.INK2, lw=0.6)
        ax.set_xlabel(f"Staleness of the {F.VO2} history, $d$ (s)")
        ax.set_xlim(0, 140)
    axes[0].set_ylabel("Gain of the sensor set, $1-\\mathrm{RMSE}_{h+s}/\\mathrm{RMSE}_{h}$")
    fig.tight_layout(w_pad=1.0)
    F.save(fig, OUT, "fig5_gain")


# ---------------------------------------------------------------- Fig. 6
def family_stratum_rmse(p, regime):
    q = p.assign(stratum=regime.reindex(p.row).to_numpy(), e2=(p.y_pred - p.y_true) ** 2)
    q = q[q.stratum.notna()]
    mse = q.groupby(["model", "seed", "pid", "stratum"])["e2"].mean().groupby(["model", "pid", "stratum"]).mean()
    out = {}
    for f, ms in E.FAMILIES.items():
        out[f] = (mse.loc[ms].groupby(["pid", "stratum"]).mean() ** 0.5).unstack("stratum")
    return out


def oracle_regime(panel):
    df, g = panel.df, panel.eval_mask()
    dv = df.groupby("seg")["y"].diff().abs()
    orc = dv.groupby(df["seg"]).transform(lambda s: s.rolling(15, center=True, min_periods=1).mean()).where(g)
    q1, q2 = np.nanquantile(orc, [1 / 3, 2 / 3])
    return pd.Series(np.where(orc <= q1, "steady", np.where(orc > q2, "transition", "middle")), index=df.index).where(g)


def fig6():
    panel = load("BEL")
    p = pred()
    parts = [("Cadence detector (deployable)", E.bel_regime(panel), "transition.csv"),
             (f"Oracle ({F.VO2} rate of change)", oracle_regime(panel), "oracle_transition.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL, 2.8), sharey=True)
    strata = ["steady", "middle", "transition"]
    off = {"Linear": -0.18, "Tree": 0.0, "Temporal": 0.18}
    for ax, (title, reg, fname), k in zip(axes, parts, "AB"):
        fam = family_stratum_rmse(p, reg)
        tr = pd.read_csv(K100 / "BEL" / "analysis" / fname).set_index("family")
        for f, tab in fam.items():
            stats_ = [tci(tab[s].dropna()) for s in strata]
            x = np.arange(3) + off[f]
            c = F.FAMILY_COLOR[f]
            ax.errorbar(x, [m for m, _, _ in stats_], yerr=[[m - lo for m, lo, _ in stats_], [hi - m for m, _, hi in stats_]],
                        fmt="o-", color=c, ms=4, lw=1.1, capsize=0, elinewidth=1.0,
                        label=f"{f} (penalty {tr.loc[f, 'penalty_gmean']:.2f})")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Steady", "Middle", "Transition"])
        ax.set_title(title)
        ax.legend(loc="upper left")
        F.panel(ax, k, x=-0.14 if k == "A" else -0.06)
    axes[0].set_ylabel(f"Per-participant RMSE ({F.UNIT})")
    fig.tight_layout(w_pad=1.0)
    F.save(fig, OUT, "fig6_transition")


# ---------------------------------------------------------------- Fig. 7
def fig7(model="TCN"):
    s = seed_mean(pred(), model)
    m, d = (s.y_true + s.y_pred) / 2, s.y_pred - s.y_true
    ba = pd.read_csv(K100 / "BEL" / "analysis" / "bland_altman.csv").set_index("model").loc[model]
    fig, ax = plt.subplots(figsize=(F.WIDE, 3.0))
    h = ax.hist2d(m, d, bins=[120, 120], range=[[0, 35], [-12, 12]], cmap="Greys", cmin=1,
                  norm=__import__("matplotlib.colors", fromlist=["LogNorm"]).LogNorm())
    for v, ls, lab in [(ba.bias, "-", f"Bias {ba.bias:+.2f}"), (ba.vc_loa_lo, "--", f"LoA {ba.vc_loa_lo:.2f}"),
                       (ba.vc_loa_hi, "--", f"LoA {ba.vc_loa_hi:+.2f}")]:
        ax.axhline(v, color=F.FAMILY_COLOR["Temporal"], ls=ls, lw=1.0)
        ax.text(34.6, v + 0.35, lab.replace("-", "−"), va="bottom", ha="right", fontsize=7)
    ax.set_xlim(0, 35)
    ax.set_xlabel(f"Mean of measured and predicted {F.VO2} ({F.UNIT})")
    ax.set_ylabel(f"Predicted − measured ({F.UNIT})")
    cb = fig.colorbar(h[3], ax=ax, pad=0.12, fraction=0.04)
    cb.set_label("Samples per bin", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    F.save(fig, OUT, "fig7_bland_altman")


# ---------------------------------------------------------------- Fig. 8
def fig8():
    fc = pd.read_csv(K100 / "BEL" / "analysis" / "family_contrasts.csv").set_index("contrast")
    sm = json.loads((K100 / "BEL" / "analysis" / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(F.WIDE, 2.3))
    ax.axvspan(-sm["m_lenient"], sm["m_lenient"], color=F.BAND_LENIENT, lw=0, zorder=0)
    ax.axvspan(-sm["m_strict"], sm["m_strict"], color=F.BAND_STRICT, lw=0, zorder=0)
    for v in (-0.25, 0.25):
        ax.axvline(v, color=F.INK2, ls=":", lw=0.8)
    ax.axvline(0, color=F.INK2, lw=0.6)
    for i, c in enumerate(CONTRASTS[::-1]):
        x = fc.loc[c]
        ax.plot([x.ci95_lo, x.ci95_hi], [i, i], color=F.INK, lw=0.9)
        ax.plot([x.ci90_lo, x.ci90_hi], [i, i], color=F.INK, lw=3.2, solid_capstyle="butt")
        ax.plot(x.mean_diff, i, "o", color="white", mec=F.INK, ms=5, zorder=3)
        ax.text(0.62, i, f"{plain_label(c, x, sm['m_strict'])}\nΔ* {x.delta_star:.2f} · MDE {x.mde:.2f} · $p_\\mathrm{{Holm}}$ = {x.p_holm:.3f}",
                va="center", fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels([CLABEL[c] for c in CONTRASTS[::-1]])
    ax.set_xlim(-0.6, 1.15)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel(f"Paired difference in per-participant RMSE ({F.UNIT})")
    fig.tight_layout()
    F.save(fig, OUT, "fig8_equivalence")


# ---------------------------------------------------------------- Fig. 9
def fig9(pid=9, models=("LASSO", "RF", "TCN")):
    p = pred()
    fig, ax = plt.subplots(figsize=(F.FULL, 2.5))
    base = seed_mean(p, models[0])
    base = base[base.pid == pid].sort_index()
    tmin = np.arange(len(base)) / 60
    ax.plot(tmin, base.y_true.to_numpy(), color=F.INK, lw=1.1, label=f"Measured {F.VO2}")
    for m in models:
        s = seed_mean(p, m)
        s = s[s.pid == pid].sort_index()
        ax.plot(tmin, s.y_pred.to_numpy(), color=F.FAMILY_COLOR[F.MODEL_FAMILY[m]], lw=0.9,
                label=f"{F.MODEL_LABEL[m]} ({F.MODEL_FAMILY[m].lower()})")
    ax.set_xlabel("Time on the evaluation grid (min)")
    ax.set_ylabel(f"{F.VO2} ({F.UNIT})")
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(0, tmin[-1])
    fig.tight_layout()
    F.save(fig, OUT, "fig9_timeseries")


# ---------------------------------------------------------------- Fig. 10
def fig10():
    dss = ["BEL", "WEEE", "ING"]
    R = {d: (pd.read_csv(K100 / d / "analysis" / "family_contrasts.csv").set_index("contrast"),
             json.loads((K100 / d / "analysis" / "summary.json").read_text())) for d in dss}
    fig, axes = plt.subplots(1, 3, figsize=(F.FULL, 2.2), sharey=True)
    order = dss[::-1]
    for k, (ax, c) in enumerate(zip(axes, CONTRASTS)):
        ax.axvspan(-0.10, 0.10, color=F.BAND_LENIENT, lw=0, zorder=0)
        ax.axvspan(-0.05, 0.05, color=F.BAND_STRICT, lw=0, zorder=0)
        ax.axvline(0, color=F.INK2, lw=0.6)
        for i, d in enumerate(order):
            x, sd = R[d][0].loc[c], R[d][1]["sd_y"]
            col, mk = F.DATASET_COLOR[d], F.DATASET_MARKER[d]
            ax.plot([x.ci95_lo / sd, x.ci95_hi / sd], [i, i], color=col, lw=0.9)
            ax.plot([x.ci90_lo / sd, x.ci90_hi / sd], [i, i], color=col, lw=3.2, solid_capstyle="butt")
            ax.plot(x.mean_diff / sd, i, mk, color="white", mec=col, mew=1.2, ms=5, zorder=3)
            lab = plain_label(c, x, R[d][1]["m_strict"]).replace(" (within lenient margin)", "\n(within lenient margin)")
            ax.text(0.155, i, lab, va="center", fontsize=6.5)
        ax.set_xlim(-0.14, 0.36)
        ax.set_xticks([-0.1, 0, 0.1, 0.2])
        ax.set_ylim(-0.6, 2.6)
        ax.set_title(CLABEL[c])
        ax.set_xlabel(f"Difference / SD({F.VO2})")
        F.panel(ax, "ABC"[k], x=-0.42 if k == 0 else -0.08)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels([F.DATASET_LABEL[d] for d in order])
    fig.tight_layout(w_pad=0.6)
    F.save(fig, OUT, "fig10_datasets")


FIGS = {"2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8, "9": fig9, "10": fig10}

if __name__ == "__main__":
    F.apply()
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
        print("fig", k, "done", flush=True)
