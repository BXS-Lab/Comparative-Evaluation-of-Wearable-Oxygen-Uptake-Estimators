"""Shared figure style for the revised manuscript (one look for every figure).

- Typeface: Times New Roman (matches the manuscript body), STIX maths; 8 pt labels, 7 pt ticks, 10 pt panel letters.
- Widths: full text width 6.5 in; 0.75 width 4.9 in.
- Colour = identity, fixed order, never by rank. Model families use the Okabe–Ito triple (validated with the dataviz
  palette checks: lightness band, chroma, CVD and normal-vision separation all PASS). Datasets use a second, disjoint
  triple (PASS; contrast WARN for two hues, so every dataset series also carries its own marker and a direct label).
- Recessive axes: left/bottom spines only, 0.6 pt, outward ticks, no grid unless a reference line is needed.
- Export: vector PDF (for LaTeX) + 600-dpi PNG (for review)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FULL, WIDE = 6.5, 4.9                                          # inches

FAMILY_COLOR = {"Linear": "#0072B2", "Tree": "#D55E00", "Temporal": "#009E73"}
DATASET_COLOR = {"BEL": "#6A3D9A", "WEEE": "#CC79A7", "ING": "#56B4E9"}
DATASET_MARKER = {"BEL": "o", "WEEE": "s", "ING": "^"}
DATASET_LABEL = {"BEL": "Beltrame (n = 16)", "WEEE": "WEEE (n = 15)", "ING": "Ingraham (n = 10)"}
INK, INK2, GRID = "#1a1a1a", "#5a5a5a", "#e3e3e3"
BAND_STRICT, BAND_LENIENT = "#cfcfcf", "#ececec"

MODEL_ORDER = ["OLS", "RIDGE", "LASSO", "ELASTICNET", "RF", "XGB", "LSTM", "GRU", "TCN", "TFT", "PATCHTST"]
MODEL_LABEL = {"OLS": "OLS", "RIDGE": "Ridge", "LASSO": "Lasso", "ELASTICNET": "Elastic net", "RF": "RF",
               "XGB": "XGBoost", "LSTM": "LSTM", "GRU": "GRU", "TCN": "TCN", "TFT": "TFT", "PATCHTST": "PatchTST"}
MODEL_FAMILY = {m: ("Linear" if i < 4 else "Tree" if i < 6 else "Temporal") for i, m in enumerate(MODEL_ORDER)}
VO2 = r"$\dot{V}\mathrm{O}_2$"
UNIT = r"mL kg$^{-1}$ min$^{-1}$"


def contrast_label(contrast, x, m_strict, borderline=0.95):
    """One classification rule for every family contrast (manuscript Section 2.4.5):
    equivalent if Delta* < equivalence margin (0.05 SD); otherwise a resolved difference if the Holm-adjusted exact
    test is significant and the 95% CI excludes zero; otherwise not resolved. The 0.10 SD value is a reference only.
    Equivalence within 5% of the margin is flagged borderline; an equivalent contrast whose difference is also resolved
    is flagged as such. `x` is a row of family_contrasts.csv; `contrast` is e.g. "Temporal - Tree"."""
    first, second = contrast.split(" - ")
    resolved = x["p_holm"] < 0.05 and (x["ci95_hi"] < 0 or x["ci95_lo"] > 0)
    if x["delta_star"] < m_strict:
        if resolved:
            return f"Equivalent ({first if x['mean_diff'] < 0 else second} better within margin)"
        return "Equivalent (borderline)" if x["delta_star"] > borderline * m_strict else "Equivalent"
    if resolved:
        return f"{first if x['mean_diff'] < 0 else second} better"
    return "Not resolved"


def replication_label(primary, external):
    """Primary-vs-external status from the two contrast labels."""
    if primary == "Not resolved":
        return "not resolved on the primary dataset"
    if external == "Not resolved":
        return "not resolved externally"
    eq = lambda s: s.startswith("Equivalent")
    if eq(primary) and eq(external) or primary == external:
        return "replicated"
    return "not replicated"


def apply():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.2, "lines.markersize": 4, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "normal",
        "figure.dpi": 150,
    })


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
