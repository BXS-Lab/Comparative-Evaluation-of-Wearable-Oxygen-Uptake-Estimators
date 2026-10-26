"""Lag diagnostic and baseline-relative analysis.
Usage: python scripts_r1/run_w4.py BEL [--part w4a|w4b|both] [--workers 8] [--root DIR]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1 import baseline_relative as B  # noqa: E402
from r1 import data_external as X  # noqa: E402
from r1 import lag  # noqa: E402
from r1.cv import RESULTS_ROOT  # noqa: E402
from r1.datasets import load  # noqa: E402
from r1.preprocess import CUTOFF_GRID, causal_filter  # noqa: E402


ZEPHYR = ("BR", "BRAmplitude", "Activity")


def zephyr_eligible(panel):
    """flagged WEEE participants are excluded from the lag diagnostic of the Zephyr features."""
    off = pd.read_csv(X.EXT_ROOT / "weee_clock_offsets.csv")
    ok = {int(p[1:]) for p in off.loc[~off["flag"].astype(bool), "participant"]}
    return sorted(q for q in (int(x) for x in panel.pids) if q in ok)


def w4a(panel, out):
    df = panel.df
    Y, rows = lag.shifted_targets(panel)
    raw = df[panel.features].to_numpy(float)
    summaries = []
    for filt in ["zero_phase_0.05Hz", "unfiltered"]:
        X = lag.zero_phase(raw, df["seg"].to_numpy()) if filt != "unfiltered" else raw
        sets = {"all": list(range(raw.shape[1]))} | {f: [j] for j, f in enumerate(panel.features)}
        for name, cols in sets.items():
            sub = zephyr_eligible(panel) if panel.name.startswith("WEEE") and name in ZEPHYR else None
            tab = lag.run(panel, X[:, cols], rows, Y, pids=sub)
            tab.assign(features=name, filtering=filt).to_csv(out / f"w4a_{filt}_{name}.csv", index=False)
            summaries.append({"features": name, "filtering": filt, **lag.summarise(tab)})
    pd.DataFrame(summaries).to_csv(out / "w4a_summary.csv", index=False)


def w4b(panel, out, workers):
    df = panel.df
    H = {d: B.history_features(panel, d) for d in B.D_GRID}
    ok = B.common_rows(panel, H)
    cache = {c: causal_filter(df[panel.features], df["seg"], c) for c in CUTOFF_GRID}
    jobs = [(p, l) for l in ("ridge", "xgb") for p in panel.pids]
    res = Parallel(n_jobs=workers, backend="loky")(delayed(B.run_outer)(panel, cache, H, ok, p, l) for p, l in jobs)
    tab = pd.concat(res, ignore_index=True)
    tab.to_csv(out / "w4b_per_participant.csv", index=False)
    json.dump({"n_rows": int(ok.sum())}, open(out / "w4b_meta.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--part", default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=str(RESULTS_ROOT))
    a = ap.parse_args()
    panel = load(a.dataset)
    out = Path(a.root) / a.dataset / "w4"
    out.mkdir(parents=True, exist_ok=True)
    if a.part in ("w4a", "both"):
        w4a(panel, out)
    if a.part in ("w4b", "both"):
        w4b(panel, out, a.workers)
    print("done", a.dataset, a.part)  # results are not printed (blinding until the analysis stage)


if __name__ == "__main__":
    main()
