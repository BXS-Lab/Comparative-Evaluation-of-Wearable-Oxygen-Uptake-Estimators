"""Dataset loaders returning a common long-format panel.

Panel columns:
    pid      participant id (int)
    seg      segment id (int, unique within the dataset)
    t        seconds since segment start (int, 1 Hz)
    y        target V̇O₂ in mL kg^-1 min^-1 (scoring target)
    avail    index (row position in the panel) of the earliest sample at which this target value is
             known causally; used only when the target is an input (baseline-relative analysis)
    <features>
    regime_signal  dataset regime marker input (BEL: cadence detector), may be NaN
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BURN_IN_S = 240
@dataclass
class Panel:
    name: str
    df: pd.DataFrame
    features: list
    meta: dict = field(default_factory=dict)

    @property
    def pids(self):
        return np.unique(self.df["pid"].to_numpy())

    def eval_mask(self):
        """Common evaluation grid: drop the first BURN_IN_S seconds of every segment."""
        return (self.df["t"].to_numpy() >= BURN_IN_S)


def _cadence_detector(cad, window=90):
    """|Δ|ΔCAD||, trailing 90 s maximum (causal)."""
    s = pd.Series(cad, dtype=float)
    d = s.diff().abs().diff().abs()
    return d.rolling(window, min_periods=1).max().to_numpy()


def _bel_availability(vo2_raw, tol=1e-3):
    """(BEL): knots = |second difference| > tol on V̇O₂ as stored (mL/min); first and last
    samples are knots; availability of a sample = next knot at or after it (position within participant)."""
    n = len(vo2_raw)
    d2 = np.abs(np.diff(vo2_raw, 2))
    knot = np.zeros(n, dtype=bool)
    knot[1:-1] = d2 > tol
    knot[0] = knot[-1] = True
    knot_idx = np.flatnonzero(knot)
    pos = np.searchsorted(knot_idx, np.arange(n), side="left")
    return knot_idx[np.minimum(pos, len(knot_idx) - 1)]


def load_bel(csv_path):
    """Beltrame et al. 2017 (TB-File-01.csv). One continuous segment per participant (no time column)."""
    raw = pd.read_csv(csv_path)
    parts = []
    offset = 0
    for seg, (pid, g) in enumerate(raw.groupby("ID", sort=True)):
        g = g.reset_index(drop=True)
        n = len(g)
        d = pd.DataFrame({
            "pid": int(pid),
            "seg": seg,
            "t": np.arange(n),
            "y": g["VO2"].to_numpy(float) / g["Weight"].to_numpy(float),
            "BF": g["BF"].to_numpy(float),
            "VE": g["VE"].to_numpy(float),
            "ACC": g["ACC"].to_numpy(float),
            "HR": g["HR"].to_numpy(float),
            "CAD": g["CAD"].to_numpy(float),
        })
        d["dHR"] = np.r_[0.0, np.diff(d["HR"].to_numpy())]
        d["avail"] = offset + _bel_availability(g["VO2"].to_numpy(float))
        d["regime_signal"] = _cadence_detector(d["CAD"].to_numpy())
        parts.append(d)
        offset += n
    df = pd.concat(parts, ignore_index=True)
    return Panel("BEL", df, ["BF", "VE", "ACC", "HR", "CAD", "dHR"],
                 meta={"weight": raw.groupby("ID")["Weight"].first().to_dict()})


def default_bel_path():
    return Path(__file__).resolve().parents[1] / "data" / "TB-File-01.csv"
