"""Model-level main-text figures (Figs 4 and 5 of the paper).

Input: paper_analysis/results/model_pairs_all.csv (55 model pairs per dataset; diff = a - b in per-participant RMSE; Holm over the
55 pairs; class at the 0.05 SD equivalence margin). Only the three main sets BEL, WEEE, ING are used.

fig8_pairs  : pairwise class matrix, one lower triangle per dataset, cell text = Delta*/SD.
fig10_vs_rf : forest plot of every model minus RF, in SD units, three datasets per row.

Style from scripts_r1/figstyle.py; panel letters bold lower case. Output: outputs/figures/<name>.{pdf,png}, PNG at
300 dpi.
Usage: python paper_analysis/model_level_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import offset_copy

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "outputs" / "figures"
sys.path.insert(0, str(REPO / "scripts_r1"))
import figstyle as F  # noqa: E402

SETS = ["BEL", "WEEE", "ING"]
W180 = 180 / 25.4                                             # double-column width, inches
CLASS_COLOR = {"resolved": "#882255", "equivalent": "#88CCEE", "not resolved": "#EEEEEE"}   # dark / mid / light
MS, DOT = 5.8, 2.4                                            # Fig. 10 marker and equivalence-dot sizes
CLASS_TEXT = {"resolved": "white", "equivalent": F.INK, "not resolved": F.INK2}


def load():
    d = pd.read_csv(HERE / "results" / "model_pairs_all.csv")
    d = d[d.set.isin(SETS)].rename(columns={"class": "cls"}).copy()
    assert d.groupby("set").size().eq(55).all()
    return d


def model_order(d):
    """Models by mean per-participant RMSE on Beltrame, most accurate first."""
    b = d[d.set == "BEL"]
    m = pd.concat([b[["a", "mean_a"]].set_axis(["model", "rmse"], axis=1),
                   b[["b", "mean_b"]].set_axis(["model", "rmse"], axis=1)]).groupby("model").rmse.first()
    return m.sort_values().index.tolist()


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def panel(ax, letter, x, y=1.0):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=F.INK)


# ---------------------------------------------------------------- Fig. 8: pairwise class matrix
def fig_pairs(d, order):
    k = len(order)
    rank = {m: i for i, m in enumerate(order)}
    fam_col = lambda m: F.FAMILY_COLOR[F.MODEL_FAMILY[m]]
    # direction of every resolved pair relative to the matrix layout (row = lower-ranked model on Beltrame)
    flagged = []
    for z in d[d.cls == "resolved"].itertuples():
        better = z.a if z.diff < 0 else z.b
        col_model = z.a if rank[z.a] < rank[z.b] else z.b
        if better != col_model:
            flagged.append((z.set, z.a, z.b))
    all_column = not flagged

    fig, axes = plt.subplots(1, 3, figsize=(W180, 3.35))
    for p, (ax, s) in enumerate(zip(axes, SETS)):
        t = d[d.set == s]
        cell = {}
        for z in t.itertuples():
            i, j = sorted((rank[z.a], rank[z.b]), reverse=True)    # i = row (lower ranked), j = column
            cell[(i, j)] = z
        for (i, j), z in cell.items():
            x, y = j, i - 1
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=CLASS_COLOR[z.cls], edgecolor="white", lw=0.8))
        for (i, j), z in cell.items():
            cls = z.cls
            lab = f"{z.dstar_sd:.2f}"
            if (s, z.a, z.b) in flagged:
                lab += "†"
            ax.text(j, i - 1, lab, ha="center", va="center", fontsize=6.0, color=CLASS_TEXT[cls],
                    fontweight="bold" if cls != "not resolved" else "normal")
        # row labels on the diagonal (right of each row's last cell), column labels below
        for i in range(1, k):
            ax.text(i - 0.4, i - 1, F.MODEL_LABEL[order[i]], ha="left", va="center", fontsize=6.5,
                    color=fam_col(order[i]))
        ax.set_xticks(range(k - 1))
        ax.set_xticklabels([F.MODEL_LABEL[m] for m in order[:-1]], rotation=90, fontsize=6.5)
        for tl, m in zip(ax.get_xticklabels(), order[:-1]):
            tl.set_color(fam_col(m))
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(-0.55, k - 1.45)
        ax.set_ylim(k - 1.45, -0.55)
        ax.set_aspect("equal")
        c = t.cls.value_counts().reindex(["resolved", "equivalent", "not resolved"], fill_value=0)
        ax.set_title(f"{F.DATASET_LABEL[s]}\nresolved {c['resolved']} / equivalent {c['equivalent']} / "
                     f"not resolved {c['not resolved']} of 55", fontsize=7.5, pad=3)
        if s == "ING":
            ax.text(k - 1.5, -0.35, "No pair can reach\n$P_\\mathrm{Holm}$ < 0.05\nat $n$ = 10", ha="right", va="top",
                    fontsize=6.5, color=F.INK2, linespacing=1.15)
        ax.annotate("abc"[p], xy=(-0.03, 1.0), xycoords=("axes fraction", ax.title), fontsize=10, fontweight="bold",
                    ha="left", va="top", color=F.INK)

    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="none", label=c.capitalize()) for c in CLASS_COLOR]
    fam = [Line2D([], [], ls="", marker="s", ms=5, color=col, label=f"{f} family") for f, col in F.FAMILY_COLOR.items()]
    fig.tight_layout(w_pad=0.4, rect=(0, 0.07, 1, 1))
    leg1 = fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.03, 0.0), handlelength=1.2,
                      columnspacing=1.0, title="Class (cell colour)", title_fontsize=7)
    fig.legend(handles=fam, loc="lower right", ncol=3, bbox_to_anchor=(0.99, 0.0), handlelength=0.8,
               columnspacing=1.0, title="Label colour (descriptive)", title_fontsize=7)
    fig.add_artist(leg1)
    note = ("Cell value: Δ*/SD. Rows and columns ordered by mean RMSE on Beltrame (most accurate first). "
            + ("In every resolved pair the column model is the more accurate." if all_column else
               "† resolved pair in which the row model is the more accurate; otherwise the column model is."))
    fig.text(0.5, 0.115, note, ha="center", va="bottom", fontsize=6.5, color=F.INK2)
    save(fig, "fig8_pairs")
    return all_column, flagged


# ---------------------------------------------------------------- Fig. 10: every model minus RF
def vs_rf(d, ref="RF"):
    rows = []
    for z in d[(d.a == ref) | (d.b == ref)].itertuples():
        sgn = 1 if z.b == ref else -1                            # diff = a - b; want model - RF
        other = z.a if z.b == ref else z.b
        lo95, hi95 = sorted((sgn * z.lo95, sgn * z.hi95))
        lo90, hi90 = sorted((sgn * z.lo90, sgn * z.hi90))
        rows.append(dict(set=z.set, model=other, x=sgn * z.diff / z.sd_y, lo95=lo95 / z.sd_y, hi95=hi95 / z.sd_y,
                         lo90=lo90 / z.sd_y, hi90=hi90 / z.sd_y, cls=z.cls))
    return pd.DataFrame(rows)


def fig_vs_rf(d, order):
    r = vs_rf(d)
    models = [m for m in order if m != "RF"]
    assert len(r) == 30 and set(r.model) == set(models)
    off = {"BEL": 0.24, "WEEE": 0.0, "ING": -0.24}
    fig, ax = plt.subplots(figsize=(W180, 4.4))
    ax.axvspan(-0.10, 0.10, color=F.BAND_LENIENT, lw=0, zorder=0)
    ax.axvspan(-0.05, 0.05, color=F.BAND_STRICT, lw=0, zorder=0)
    ax.axvline(0, color=F.INK2, lw=0.6, zorder=1)
    nrow = len(models)
    for i, m in enumerate(models):
        yb = nrow - 1 - i
        for s in SETS:
            z = r[(r.model == m) & (r.set == s)].iloc[0]
            y, col, mk = yb + off[s], F.DATASET_COLOR[s], F.DATASET_MARKER[s]
            ms = MS * (1.25 if mk == "^" else 1.0)                   # triangles look smaller at equal size
            ax.plot([z.lo95, z.hi95], [y, y], color=col, lw=0.8, zorder=2, solid_capstyle="butt")
            ax.plot([z.lo90, z.hi90], [y, y], color=col, lw=2.6, zorder=2, solid_capstyle="butt")
            if z.cls == "resolved":
                ax.plot(z.x, y, mk, color=col, mec=col, mew=1.2, ms=ms, zorder=3)
            else:
                ax.plot(z.x, y, mk, color="white", mec=col, mew=1.2, ms=ms, zorder=3)
                if z.cls == "equivalent":
                    tr = ax.transData if mk != "^" else offset_copy(ax.transData, fig=fig, y=-ms / 6, units="points")
                    ax.plot(z.x, y, "o", color=F.INK, mec=F.INK, ms=DOT, zorder=4, transform=tr)   # dark dot at the centroid
    for i in range(nrow - 1):
        ax.axhline(i + 0.5, color=F.GRID, lw=0.5, zorder=0.5)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([F.MODEL_LABEL[m] for m in models[::-1]])
    for tl, m in zip(ax.get_yticklabels(), models[::-1]):
        tl.set_color(F.FAMILY_COLOR[F.MODEL_FAMILY[m]])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.55, nrow - 0.45)
    lo, hi = r.lo95.min(), r.hi95.max()
    ax.set_xlim(min(-0.16, np.floor(lo * 20) / 20 - 0.01), max(0.16, np.ceil(hi * 20) / 20 + 0.01))
    ax.set_xlabel(f"Model − RF, paired difference in per-participant RMSE / SD({F.VO2})")
    ax.text(0.01, 1.005, "← more accurate than RF", transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
            color=F.INK2)
    ax.text(0.99, 1.005, "less accurate than RF →", transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=F.INK2)

    n_res = int((r.cls == "resolved").sum())
    ds_h = [Line2D([], [], marker=F.DATASET_MARKER[s], ls="-", lw=2.6, color=F.DATASET_COLOR[s], mfc="white",
                   mec=F.DATASET_COLOR[s], ms=MS * (1.25 if F.DATASET_MARKER[s] == "^" else 1.0), label=F.DATASET_LABEL[s]) for s in SETS]
    g = F.INK2
    cls_h = [Line2D([], [], marker="o", ls="", color=g, mec=g, ms=MS,
                    label="Resolved (filled)" + ("; none vs RF" if n_res == 0 else "")),
             (Line2D([], [], marker="o", ls="", mfc="white", mec=g, mew=1.2, ms=MS),
              Line2D([], [], marker="o", ls="", color=F.INK, mec=F.INK, ms=DOT)),
             Line2D([], [], marker="o", ls="", mfc="white", mec=g, mew=1.2, ms=MS, label="Not resolved (open)")]
    ci_h = [Line2D([], [], color=g, lw=2.6, label="90% CI"), Line2D([], [], color=g, lw=0.8, label="95% CI"),
            Patch(facecolor=F.BAND_STRICT, label="±0.05 SD (equivalence margin)"),
            Patch(facecolor=F.BAND_LENIENT, label="±0.10 SD (lenient reference)")]
    fam_h = [Line2D([], [], ls="", marker="s", ms=4.6, color=c, label=f"{f} family (label colour)")
             for f, c in F.FAMILY_COLOR.items()]
    labels = [h.get_label() for h in ds_h] + [cls_h[0].get_label(), "Equivalent (open, centre dot)",
                                              cls_h[2].get_label()] + [h.get_label() for h in ci_h]         + [h.get_label() for h in fam_h]
    ax.legend(ds_h + cls_h + ci_h + fam_h, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=2.0,
              borderaxespad=0.0, labelspacing=0.6)
    fig.tight_layout()
    save(fig, "fig10_vs_rf")
    return r


def main():
    F.apply()
    d = load()
    order = model_order(d)
    print("order:", [F.MODEL_LABEL[m] for m in order])
    all_col, flagged = fig_pairs(d, order)
    print("fig8_pairs done; every resolved pair favours the column model:", all_col, flagged)
    r = fig_vs_rf(d, order)
    print("fig10_vs_rf done")
    print(r.assign(**{c: r[c].round(3) for c in ["x", "lo90", "hi90", "lo95", "hi95"]}).to_string(index=False))


if __name__ == "__main__":
    main()
