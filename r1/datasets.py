"""Dataset registry used by all runners."""
import numpy as np

from .data import default_bel_path, load_bel


def load(name):
    if name == "BEL":
        return load_bel(default_bel_path())
    if name == "BEL_CORE":       # conditional core-set re-run: HR, dHR, ACC only
        p = load_bel(default_bel_path())
        p.name, p.features = "BEL_CORE", ["HR", "dHR", "ACC"]
        return p
    from . import data_external as X
    if name == "WEEE":
        return X.load_weee("protocol")
    if name == "WEEE_ALL":       # sensitivity (linear and tree models only)
        return X.load_weee("all")
    if name == "WEEE_HRQ":       # sensitivity: target-free HR plausibility screen
        p = X.load_weee("protocol")
        keep = [q for q, ok in hr_plausible(p.df).items() if ok]
        p.name, p.df = "WEEE_HRQ", p.df[p.df["pid"].isin(keep)].reset_index(drop=True)
        p.df["avail"] = np.arange(len(p.df))  # WEEE: available at t; positions must follow the subset rows
        return p
    if name == "ING":
        return X.load_ing()
    raise ValueError(name)


def hr_plausible(df, min_rise=15.0, max_device_gap=10.0):
    """WEEE HR plausibility (no target used). A participant passes if (i) the highest
    per-condition median Wahoo HR among exercise conditions (Cycle*, Run*) exceeds the highest among rest conditions
    (Sit, Stand) by at least `min_rise` bpm, and (ii) the median over conditions of |median Wahoo HR - median Zephyr HR|
    is at most `max_device_gap` bpm (agreement of two independent chest straps)."""
    med = df.groupby(["pid", "condition"])["HR"].median().unstack()
    ex = med[[c for c in med.columns if c.startswith(("Cycle", "Run"))]].max(axis=1)
    rest = med[[c for c in med.columns if c in ("Sit", "Stand")]].max(axis=1)
    medz = df.groupby(["pid", "condition"])["HR_zephyr"].median().unstack()
    gap = (med - medz).abs().median(axis=1)
    return (((ex - rest) >= min_rise) & (gap <= max_device_gap)).to_dict()


def regime_labels(panel):
    """Regime stratum per row on the evaluation grid (BEL: cadence detector terciles; others: protocol timing)."""
    import pandas as pd
    if panel.name in ("BEL", "BEL_CORE"):
        from .evaluate import bel_regime
        return bel_regime(panel)
    lab = panel.df["regime"].where(panel.eval_mask())
    return pd.Series(lab.to_numpy(), index=panel.df.index)
