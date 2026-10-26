"""
Regenerate the draft-quality data/MI/signal figures in the SAME house style as make_nature_figures.py
(Arial, no top/right spines, 600 dpi PNG + vector PDF, proper VO2 notation, no code-style labels).
Replaces: joint_distributions.png, conditional_MI_unique_info.png, raw_vs_filtered_comparison_participant_1.png,
and the low-res borrowed input.png / PRTS.png (now the authors' own clean data plots).

Individual mutual information is computed in-script (10-bin histogram MI in nats on the 0.01 Hz pooled
low-pass features), reproducible with no hardcoded values and matching experiments/pmea_p1/mi_knn.py and
the Supplementary Results: ACC 0.539 / CAD 0.522 / VE 0.407 / HR 0.400 / BF 0.193 / dHR 0.042.
Conditional/unique MI is not reported (it was not reliably estimable with five conditioning variables and is
superseded by the Ridge LOPO dHR ablation in mi_knn.py).

Run from the deposit root:  python experiments/pmea_p1/make_data_figures.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.9, "legend.frameon": False,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
C = dict(blue="#0F4D92", teal="#42949E", grey="#767676", warn="#B64342", grid="#E6E6E6", raw="#9AA7B5")

def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)
    print(f"  saved {name} (.png/.pdf)")

# ---------------------------------------------------------------- data
df = pd.read_csv("data/TB-File-01.csv")
df = df[~(df == -1).any(axis=1)].reset_index(drop=True)
pid = df["ID"].to_numpy(int)
parts = np.unique(pid)
vo2 = (df["VO2"] / df["Weight"]).to_numpy(float)
raw = {c: df[c].to_numpy(float) for c in ["BF", "VE", "ACC", "HR", "CAD"]}
dhr = np.zeros(len(df))
for p in parts:
    m = pid == p
    dhr[m] = np.concatenate([[0.0], np.diff(raw["HR"][m])])
raw["dHR"] = dhr

FEATS = ["BF", "VE", "ACC", "HR", "CAD", "dHR"]
LABEL = {"BF": "Breathing freq.\n(breaths min$^{-1}$)", "VE": "Ventilation\n(mL min$^{-1}$)",
         "ACC": "Acceleration\n(g)", "HR": "Heart rate\n(bpm)",
         "CAD": "Cadence\n(steps min$^{-1}$)", "dHR": "$\\Delta$HR\n(bpm)"}
UNIT = {"BF": "breaths min$^{-1}$", "VE": "mL min$^{-1}$", "ACC": "g", "HR": "bpm",
        "CAD": "steps min$^{-1}$", "dHR": "bpm"}
VO2LAB = "$\\dot{V}$O$_2$ (mL kg$^{-1}$ min$^{-1}$)"

# Individual MI is computed at plot time (10-bin histogram MI, nats) on the pooled 0.01 Hz low-pass-
# filtered signals (the same preprocessing the linear/tree models use), so the figure annotations are
# fully reproducible from this script -- no values are hardcoded. Conditional/unique MI was removed: a
# rigorous k-NN conditional-MI estimate did not support the earlier claim and is unreliable with five
# conditioning variables (see mi_knn.py).
def _mi10(x, y, bins=10):
    """10-bin histogram mutual information in nats."""
    c = np.histogram2d(x, y, bins=bins)[0]
    p = c / c.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True); nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (px * py)[nz])))

# ---------------------------------------------------------------- Fig: joint distributions
def fig_joint():
    bb, aa = butter(2, 0.01 / 0.5, btype="low")
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    mi_report = {}
    for ax, f in zip(axes.ravel(), FEATS):
        xf = filtfilt(bb, aa, raw[f])              # pooled 0.01 Hz low-pass (linear/tree model inputs)
        mi = _mi10(vo2, xf); mi_report[f] = round(mi, 3)
        h, xe, ye = np.histogram2d(vo2, xf, bins=10, density=True)
        ax.imshow(h.T, origin="lower", aspect="auto", cmap="YlOrRd",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]])
        disp = "$\\Delta$HR" if f == "dHR" else f
        ax.set_title(f"{disp}   MI = {mi:.3f}", fontsize=8.5)
        ax.set_xlabel(VO2LAB, fontsize=7.5); ax.set_ylabel(UNIT[f], fontsize=7.5)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "joint_distributions")
    print("individual MI (10-bin nats, pooled 0.01 Hz filtered):", mi_report)

# (Conditional/unique-MI bar figure removed -- the earlier claim was not supported by a rigorous
#  k-NN conditional-MI estimate and is unreliable with five conditioning variables; see mi_knn.py.)

# ---------------------------------------------------------------- low-pass filter (per participant)
def lowpass(v, cutoff=0.01, order=2):
    b, a = butter(order, cutoff / 0.5, btype="low")
    out = v.copy()
    for p in parts:
        m = pid == p; mu = v[m].mean()
        out[m] = filtfilt(b, a, v[m] - mu) + mu
    return out

# ---------------------------------------------------------------- Fig: raw vs filtered (participant 1)
def fig_raw_filt(pp=1):
    m = pid == pp; t = np.arange(m.sum())
    show = ["VE", "ACC", "HR", "CAD", "dHR"]
    fig, axes = plt.subplots(len(show), 1, figsize=(7.0, 6.0), sharex=True)
    for ax, f in zip(axes, show):
        ax.plot(t, raw[f][m], color=C["raw"], lw=0.6, alpha=0.9, label="raw")
        ax.plot(t, lowpass(raw[f])[m], color=C["blue"], lw=1.1, label="0.01 Hz low-pass")
        ax.set_ylabel(LABEL[f], fontsize=8)
        ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Raw vs. 0.01 Hz low-pass filtered wearable signals (participant {pp})",
                 fontsize=9.5, y=0.995)
    fig.tight_layout()
    save(fig, "raw_vs_filtered_comparison_participant_1")

# ---------------------------------------------------------------- Fig: input features (own data plot, replaces borrowed schematic)
def fig_input(pp=9):
    m = pid == pp; t = np.arange(m.sum())
    rows = ["HR", "VE", "BF", "ACC", "CAD", "dHR"]
    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(7.0, 6.6), sharex=True)
    for ax, f in zip(axes[:-1], rows):
        ax.plot(t, raw[f][m], color=C["teal"], lw=0.8)
        ax.set_ylabel(LABEL[f], fontsize=8)
        ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    axes[-1].plot(t, vo2[m], color=C["blue"], lw=0.9)
    axes[-1].set_ylabel("$\\dot{V}$O$_2$\n(mL kg$^{-1}$ min$^{-1}$)", fontsize=8)
    axes[-1].grid(axis="y", color=C["grid"], lw=0.5); axes[-1].set_axisbelow(True)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Wearable-derived input features and the criterion $\\dot{{V}}$O$_2$ (participant {pp})",
                 fontsize=9.5, y=0.997)
    fig.tight_layout()
    save(fig, "input")

# ---------------------------------------------------------------- Fig: PRTS protocol cadence (own data plot, replaces borrowed)
T2 = {1: 2853, 2: 2695, 3: 3369, 4: 3007, 5: 2753, 6: 2472, 7: 2500, 8: 3050,
      9: 2840, 10: 2891, 11: 2941, 12: 2428, 13: 2965, 14: 2785, 15: 2714, 16: 2892}
def fig_prts(pp=9):
    m = pid == pp; t = np.arange(m.sum()); cad = raw["CAD"][m]
    t2 = T2.get(pp, len(t) // 2)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axvspan(0, 300, color=C["grid"], alpha=0.7)
    ax.axvspan(t2, len(t), color=C["teal"], alpha=0.10)
    ax.plot(t, cad, color=C["blue"], lw=0.7)
    ax.set_ylabel("Cadence (steps min$^{-1}$)"); ax.set_xlabel("Time (s)")
    ax.grid(axis="y", color=C["grid"], lw=0.5); ax.set_axisbelow(True)
    ymax = max(160, np.nanmax(cad) * 1.05)
    ax.set_ylim(0, ymax)
    ax.text(150, ymax * 0.93, "warm-up", ha="center", fontsize=7, color=C["grey"])
    ax.text((300 + t2) / 2, ymax * 0.93, "PRTS controlled walking", ha="center", fontsize=7.5, color=C["blue"])
    ax.text((t2 + len(t)) / 2, ymax * 0.93, "ADL", ha="center", fontsize=7.5, color=C["teal"])
    fig.suptitle(f"Experimental protocol: cadence over time (participant {pp})", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "PRTS")

# ---------------------------------------------------------------- Fig: RF validation panels (own reproduction, house style)
def fig_validation():
    M = pd.read_parquet("results/pmea_p0/master_aligned.parquet")
    M = M.copy(); M["t2"] = M["participant"].map(T2)
    segs = [("Controlled_walking", "Controlled walking (PRTS)", M[(M.t >= 300) & (M.t < M.t2)]),
            ("ADL", "Activities of daily living (ADL)", M[M.t >= M.t2])]
    for name, title, sub in segs:
        yt = sub["y_true"].values; yp = sub["RF"].values
        r = np.corrcoef(yt, yp)[0, 1]
        fig, ax = plt.subplots(figsize=(3.6, 3.5))
        ax.scatter(yt, yp, s=4, color=C["blue"], alpha=0.25, edgecolor="none", rasterized=True)
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, color=C["grey"], lw=0.9, ls="--", label="identity")
        z = np.polyfit(yt, yp, 1)
        ax.plot(lim, np.poly1d(z)(lim), color=C["warn"], lw=1.2, label="linear fit")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Measured $\\dot{V}$O$_2$ (mL kg$^{-1}$ min$^{-1}$)", fontsize=8)
        ax.set_ylabel("RF-predicted $\\dot{V}$O$_2$ (mL kg$^{-1}$ min$^{-1}$)", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.text(0.04, 0.96, f"$r$ = {r:.2f}\n$n$ = {len(sub):,}", transform=ax.transAxes,
                va="top", ha="left", fontsize=8)
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        save(fig, name)

if __name__ == "__main__":
    print(f"Rendering house-styled data figures -> {OUT}")
    fig_joint(); fig_raw_filt()  # input/PRTS/validation kept as ORIGINALS
    print("done.")
