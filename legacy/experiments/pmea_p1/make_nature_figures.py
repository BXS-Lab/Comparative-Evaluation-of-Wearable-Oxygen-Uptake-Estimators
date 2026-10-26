"""
Nature-grade re-render of the five PMEA analysis figures, from already-saved result data.
Unified style: Arial (editable text), restrained cool palette with consistent family colors,
no top/right spines, sized to display width. Exports 600-dpi PNG (replacing the paper's PNGs)
+ vector PDF + editable SVG. Run from the repository root:  python experiments/pmea_p1/make_nature_figures.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from scipy.interpolate import PchipInterpolator

# ---------- unified Nature style ----------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.9, "legend.frameon": False,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "figure.dpi": 150, "savefig.dpi": 600,
})
C = dict(Linear="#767676", Tree="#E1812C", Temporal="#0F4D92",
         persist="#4D4D4D", steady="#CFCECE", transition="#B64342",
         accent="#0F4D92", warn="#B64342", grid="#E6E6E6", equiv="#4C9A6E")
FAM = {**{m: "Linear" for m in ["LINEAR", "RIDGE", "LASSO", "ELASTICNET", "OLS"]},
       **{m: "Tree" for m in ["RF", "XGB"]},
       **{m: "Temporal" for m in ["GRU", "LSTM", "TCN", "TFT", "PATCHTST"]}}
NICE = {"LINEAR": "OLS", "PATCHTST": "PatchTST", "ELASTICNET": "ElasticNet",
        "RIDGE": "Ridge", "LASSO": "Lasso", "XGB": "XGBoost"}
nice = lambda m: NICE.get(m, m)
UNIT = r"RMSE (mL kg$^{-1}$ min$^{-1}$)"

R0 = Path("results/pmea_p0"); R1 = Path("results/pmea_p1")
FIG = Path("figures"); FIG.mkdir(parents=True, exist_ok=True)
PAPER = FIG  # self-contained deposit: all figures written under ./figures
ARCH = R1 / "figures_nature"; ARCH.mkdir(parents=True, exist_ok=True)

def save(fig, name, where=FIG):
    fig.savefig(where / f"{name}.png", bbox_inches="tight")        # replaces paper PNG (600 dpi)
    fig.savefig(where / f"{name}.pdf", bbox_inches="tight")        # vector for submission
    fig.savefig(ARCH / f"{name}.svg", bbox_inches="tight")        # editable archive
    plt.close(fig); print(f"  saved {name} (.png/.pdf/.svg)")

def panel(ax, lab):
    ax.text(-0.13, 1.04, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")

# ---------- Fig 1: predictability horizon ----------
def fig_horizon():
    pc = pd.read_csv(R0 / "p01_persistence_curve.csv")
    pci = pd.read_csv(R1 / "horizon_pooled_crossing_ci.csv").set_index("model")
    h, m, s = pc.horizon_s.values, pc.persist_rmse_mean.values, pc.persist_rmse_sd.values
    hf = np.geomspace(h.min(), h.max(), 400)
    mf, sf = PchipInterpolator(h, m)(hf), PchipInterpolator(h, s)(hf)

    models = [("TCN", C["Temporal"]), ("XGB", C["Tree"]), ("RIDGE", C["Linear"])]  # best of each family
    va = {"TCN": "top", "XGB": "center", "RIDGE": "bottom"}
    yrow = {"TCN": 1.05, "XGB": 0.70, "RIDGE": 0.35}
    XHI, YHI = 195, 7.2

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    # 95% bootstrap confidence zone of the leading model's crossover (same convention as Fig. bootstrap)
    ax.axvspan(float(pci.loc["TCN", "ci_lo"]), float(pci.loc["TCN", "ci_hi"]),
               color=C["Temporal"], alpha=0.07, lw=0, zorder=0)
    # naive-persistence curve (error grows with horizon) + between-participant SD ribbon
    ax.fill_between(hf, mf - sf, mf + sf, color="#111111", alpha=0.12, lw=0, zorder=1)
    ax.plot(hf, mf, color="#111111", lw=2.0, zorder=4)
    ax.plot(h[h <= 60], m[h <= 60], "o", color="#111111", ms=3.2, zorder=4)
    ax.text(46, 4.9, "naive persistence\n(copy VO$_2$ forward)", fontsize=8, color="#111111",
            ha="center", va="center")

    # sensor-model 0-step nowcast levels (dashed; horizon-independent), each crossed by the rising persistence curve,
    # with a dashed connector down to that model's crossover-time error bar
    for mdl, col in models:
        y, xc = float(pci.loc[mdl, "model_rmse"]), float(pci.loc[mdl, "pooled"])
        ax.plot([1, XHI], [y, y], color=col, ls="--", lw=1.4, zorder=3)
        ax.text(1.012, y, f"{nice(mdl)} {y:.2f}", color=col, fontsize=7.1, va=va[mdl], ha="left",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([xc, xc], [yrow[mdl], y], color=col, ls="--", lw=0.8, alpha=0.8, zorder=2)
        ax.plot([xc], [y], "o", color=col, ms=5, zorder=6)

    # crossover horizon per model: pooled mean-curve crossing (point) + subject-cluster bootstrap 95% CI (bar)
    for mdl, col in models:
        lo, hi, xc = (float(pci.loc[mdl, k]) for k in ("ci_lo", "ci_hi", "pooled"))
        yy = yrow[mdl]
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.8, solid_capstyle="round", zorder=5)
        ax.plot([xc], [yy], "o", color=col, ms=5, zorder=6)
        ax.text(lo * 0.93, yy, f"{nice(mdl)}  {xc:.0f} s", color=col, fontsize=7.1, ha="right", va="center")
    ax.text(1.4, 0.12, "crossover horizon: pooled mean-curve crossing (point), 95% bootstrap CI (bar)",
            fontsize=7.0, color="#33414B", ha="left", va="center", style="italic")
    # axes
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 30, 60, 120, 180])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlim(1, XHI); ax.set_ylim(0, YHI)
    ax.set_xlabel(r"Forecast horizon $h$ (s)"); ax.set_ylabel(UNIT)
    ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    save(fig, "horizon_curve")

# ---------- Fig 2: bootstrap CI forest ----------
def fig_bootstrap():
    d = pd.read_csv(R1 / "p1_bootstrap_ci.csv")
    fam = pd.read_csv(R0 / "p04_table_ready.csv").set_index("model")["family"]
    d["family"] = d.model.map(lambda m: fam.get(m, FAM.get(m, "Linear")))
    d = d.sort_values("mean_rmse", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    best = d.iloc[0]
    ax.axvspan(best.boot_ci_lo, best.boot_ci_hi, color=C["Temporal"], alpha=0.07, lw=0)
    for yi, (_, r) in zip(y, d.iterrows()):
        col = C[r.family]
        ax.plot([r.boot_ci_lo, r.boot_ci_hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(r.mean_rmse, yi, "o", color=col, ms=4.5)
    ax.set_yticks(y); ax.set_yticklabels([nice(m) for m in d.model])
    ax.set_xlabel(UNIT)
    ax.set_title("Per-participant mean RMSE with subject-cluster bootstrap 95% CI", fontsize=8.5)
    ax.legend(handles=[Patch(color=C[f], label=f) for f in ["Temporal", "Tree", "Linear"]],
              loc="upper right", title=None)
    ax.margins(y=0.04)
    save(fig, "bootstrap_ci")

# ---------- Fig 3: equivalence & power (2-panel) ----------
def fig_equiv():
    d = pd.read_csv(R1 / "p1_contrasts.csv")
    lab = {"TCN_vs_XGB": "TCN vs XGBoost", "TCN_vs_RIDGE": "TCN vs Ridge",
           "XGB_vs_RIDGE": "XGBoost vs Ridge",
           "Temporal_vs_Tree": "Temporal vs Tree", "TEMPORAL_vs_TREE": "Temporal vs Tree"}
    d["lbl"] = d.contrast.map(lambda c: lab.get(c, c.replace("_", " ")))
    order = [c for c in ["TCN_vs_XGB", "Temporal_vs_Tree", "TCN_vs_RIDGE", "XGB_vs_RIDGE"] if c in set(d.contrast)]
    d = d.set_index("contrast").loc[order].reset_index()
    y = np.arange(len(d))[::-1]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.4))
    # A: observed gap vs MDE
    for yi, (_, r) in zip(y, d.iterrows()):
        resolved = r.observed_gap_abs > r.MDE
        axA.plot([0, r.MDE], [yi, yi], color="#BBBBBB", lw=5, solid_capstyle="butt", alpha=0.8)
        axA.plot(r.observed_gap_abs, yi, "o", ms=6,
                 color=C["warn"] if resolved else C["accent"], zorder=3)
    axA.set_yticks(y); axA.set_yticklabels(d.lbl)
    axA.set_xlabel(r"Effect size (mL kg$^{-1}$ min$^{-1}$)")
    axA.set_xlim(0, max(d.MDE.max(), d.observed_gap_abs.max()) * 1.15)
    axA.legend(handles=[Patch(color="#BBBBBB", label="MDE (n=16, 80% power)"),
                        plt.Line2D([], [], marker="o", ls="", color=C["accent"], label="gap < MDE (unresolvable)"),
                        plt.Line2D([], [], marker="o", ls="", color=C["warn"], label="gap > MDE (resolved)")],
               loc="upper right", fontsize=6.6)
    panel(axA, "a")
    # B: TOST — mean diff with 90% CI vs margins
    tcrit = stats.t.ppf(0.95, 15)
    for yi, (_, r) in zip(y, d.iterrows()):
        diff = r.mean_RMSE_A - r.mean_RMSE_B
        half = tcrit * r.sd_paired_diff / np.sqrt(16)
        eq = bool(r["equiv_0.25"])
        col = C["equiv"] if eq else C["warn"]
        axB.plot([diff - half, diff + half], [yi, yi], color=col, lw=1.8, solid_capstyle="round")
        axB.plot(diff, yi, "o", ms=4.5, color=col)
    for mgn in (0.25, 0.5):
        axB.axvspan(-mgn, mgn, color="#42949E" if mgn == 0.25 else "#42949E", alpha=0.06, lw=0)
        axB.axvline(mgn, color="#9FB7BA", ls="--", lw=0.9); axB.axvline(-mgn, color="#9FB7BA", ls="--", lw=0.9)
    axB.axvline(0, color="#CCCCCC", lw=0.8)
    axB.text(0.25, len(d) - 0.5, "±0.25", color="#5E8186", fontsize=7, ha="center", va="bottom")
    axB.text(0.5, len(d) - 0.5, "±0.50", color="#5E8186", fontsize=7, ha="center", va="bottom")
    axB.set_yticks(y); axB.set_yticklabels([])
    axB.set_xlabel(r"Mean RMSE difference (90% CI)")
    axB.legend(handles=[plt.Line2D([], [], color=C["equiv"], label="equivalent (±0.25)"),
                        plt.Line2D([], [], color=C["warn"], label="not equivalent")],
               loc="upper right", fontsize=6.6, frameon=True, facecolor="white",
               framealpha=0.9, edgecolor="none")
    panel(axB, "b")
    for a in (axA, axB):
        a.set_ylim(-0.55, len(d) + 0.95)
    fig.tight_layout(w_pad=1.4)
    save(fig, "equivalence_power")

# ---------- Fig 4: causal transition penalty ----------
def fig_transition():
    d = pd.read_csv(R1 / "causal_transition_family.csv").set_index("family").loc[["Linear", "Tree", "Temporal"]]
    fams = list(d.index); x = np.arange(len(fams)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    st, tr = d.causal_steady_rmse.values, d.causal_transition_rmse.values
    st_err = [st - d.causal_steady_lo, d.causal_steady_hi - st]
    tr_err = [tr - d.causal_transition_lo, d.causal_transition_hi - tr]
    ax.bar(x - w/2, st, w, yerr=st_err, color=C["steady"], edgecolor="#5A5A5A", lw=0.8,
           error_kw=dict(elinewidth=0.9, capsize=2.5, capthick=0.9), label="steady")
    ax.bar(x + w/2, tr, w, yerr=tr_err, color=C["transition"], edgecolor="#7A2A2A", lw=0.8,
           error_kw=dict(elinewidth=0.9, capsize=2.5, capthick=0.9), label="transition")
    for xi, r in zip(x, d.causal_ratio.values):
        ax.text(xi, max(tr) * 1.06, f"×{r:.2f}", ha="center", fontsize=8, color="#7A2A2A", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylabel(UNIT); ax.set_ylim(0, max(d.causal_transition_hi) * 1.16)
    ax.set_title("Causal-detector transition penalty by model family", fontsize=8.5)
    ax.legend(loc="upper left")
    save(fig, "causal_transition")

# ---------- Fig 5: Bland-Altman (PatchTST) ----------
def fig_bland():
    M = pd.read_parquet(R0 / "master_aligned.parquet")
    s = pd.read_csv(R1 / "bland_altman_summary.csv").set_index("model").loc["PATCHTST"]
    pb = pd.read_csv(R1 / "per_participant_bias.csv")["bias_PATCHTST"].values
    diff = (M["PATCHTST"] - M["y_true"]).values
    mean = ((M["PATCHTST"] + M["y_true"]) / 2).values
    pb = np.asarray(pb, float)
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    hb = ax.hexbin(mean, diff, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    x0, x1 = ax.get_xlim()
    ax.axhline(s.bias, color="#272727", lw=1.3)
    ax.axhline(s.vc_loa_upper, color=C["warn"], ls="--", lw=1.1)
    ax.axhline(s.vc_loa_lower, color=C["warn"], ls="--", lw=1.1)
    lx = x0 + 0.4
    ax.text(lx, s.bias + 0.25, f"bias {s.bias:+.2f}", fontsize=7, color="#272727", va="bottom")
    ax.text(lx, s.vc_loa_upper + 0.25, f"Upper LoA  {s.vc_loa_upper:+.1f}", fontsize=7, color=C["warn"], va="bottom")
    ax.text(lx, s.vc_loa_lower - 0.25, f"Lower LoA  {s.vc_loa_lower:+.1f}", fontsize=7, color=C["warn"], va="top")
    # per-subject mean bias swarm placed in the sparse high-VO2 region (no collision with the cloud or colorbar)
    xs = 0.80 * x1
    rng = np.random.default_rng(0); xj = xs + rng.uniform(-1.0, 1.0, len(pb))
    ax.plot([xs, xs], [pb.min(), pb.max()], color=C["Temporal"], lw=0.8, alpha=0.5, zorder=3)
    ax.scatter(xj, pb, s=16, color=C["Temporal"], alpha=0.9, edgecolor="white", lw=0.4, zorder=4)
    ax.annotate(f"per-subject mean bias\n(range {s.ppt_bias_range:.1f})", xy=(xs, pb.max()),
                xytext=(xs, s.vc_loa_upper + 0.9), fontsize=7, color=C["Temporal"], ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=C["Temporal"], lw=0.7))
    cb = fig.colorbar(hb, ax=ax, pad=0.015, fraction=0.046); cb.set_label("samples / bin", fontsize=7); cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(r"Mean of predicted & measured $\dot{V}$O$_2$ (mL kg$^{-1}$ min$^{-1}$)")
    ax.set_ylabel(r"Predicted $-$ measured (mL kg$^{-1}$ min$^{-1}$)")
    save(fig, "bland_altman")

# ---------- Restyled pre-existing results figures (regenerated from the matched data) ----------
def fig_violins():
    from sklearn.metrics import r2_score
    M = pd.read_parquet(R0 / "master_aligned.parquet")
    models = ["TCN", "PATCHTST", "GRU", "TFT", "RF", "LSTM", "XGB", "RIDGE", "ELASTICNET", "LASSO", "LINEAR"]
    rows = []
    for m in models:
        for _, g in M.groupby("participant"):
            e = (g[m] - g["y_true"]).values
            rows.append(dict(model=m, rmse=np.sqrt(np.mean(e**2)),
                             r=np.corrcoef(g[m], g["y_true"])[0, 1], r2=r2_score(g["y_true"], g[m])))
    d = pd.DataFrame(rows)
    order = d.groupby("model")["rmse"].mean().sort_values().index.tolist()   # x-axis: best -> worst mean RMSE
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    specs = [("rmse", r"RMSE (mL kg$^{-1}$ min$^{-1}$)"), ("r", r"Pearson $r$"), ("r2", r"$R^2$")]
    for ax, (key, title) in zip(axes, specs):
        vals = [d[d.model == m][key].values for m in order]
        vp = ax.violinplot(vals, showmeans=True, widths=0.85)
        for body, m in zip(vp["bodies"], order):
            body.set_facecolor(C[FAM[m]]); body.set_edgecolor("#3A3A3A"); body.set_alpha(0.62); body.set_linewidth(0.6)
        for k in ("cmeans", "cmaxes", "cmins", "cbars"):
            if k in vp:
                vp[k].set_color("#3A3A3A"); vp[k].set_linewidth(0.8)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([nice(m) for m in order], rotation=45, ha="right", fontsize=6.5)
        ax.set_title(title, fontsize=9); ax.grid(axis="y", color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=C[f], label=f) for f in ["Temporal", "Tree", "Linear"]],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "model_metric_violins", where=PAPER)

def fig_timeseries():
    # keep the original's completeness (full series, all models per family); add house-style polish only
    M = pd.read_parquet(R0 / "master_aligned.parquet")
    g = M[M.participant == 9].sort_values("t")
    t = g.t.values
    OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]  # Okabe-Ito, colorblind-safe
    groups = [("Linear regression models", ["LINEAR", "RIDGE", "LASSO", "ELASTICNET"]),
              ("Decision-tree models", ["XGB", "RF"]),
              ("Temporal models", ["TCN", "PATCHTST", "LSTM"])]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    for ax, (title, mods), lab in zip(axes, groups, "abc"):
        ax.plot(t, g["y_true"].values, color="black", lw=0.7, label="Ground truth", zorder=6)
        for m, col in zip(mods, OI):
            ax.plot(t, g[m].values, color=col, lw=0.9, alpha=0.9, label=nice(m))
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        ax.legend(loc="upper left", fontsize=6.8, ncol=2, handlelength=1.4, columnspacing=1.0)
        ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
        ax.margins(x=0.005)
        panel(ax, lab)
    axes[1].set_ylabel(r"$\dot{V}$O$_2$ (mL kg$^{-1}$ min$^{-1}$)")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(h_pad=0.7)
    save(fig, "all_models_timeseries_from_row122")

if __name__ == "__main__":
    print("Rendering Nature-grade figures ->", FIG)
    fig_horizon(); fig_bootstrap(); fig_equiv(); fig_transition(); fig_bland()
    fig_violins(); fig_timeseries()
    print("done.")
