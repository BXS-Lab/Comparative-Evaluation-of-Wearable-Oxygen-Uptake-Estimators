"""WEEE data-quality inclusion criteria (fixed before any model was trained), recomputed from the raw data.

A participant is included if (1) at least 900 s of valid 1 Hz samples (V̇O₂ > 0 and Wahoo HR > 0) lie in segments of
at least 300 s (gaps up to 5 s bridged); (2) the condition start times are strictly increasing and at least one
exercise condition (Cycle1, Cycle2, Run1, Run2) contains at least 180 valid samples; and (3) the largest median V̇O₂
over the last 180 valid samples of an exercise condition is at least 10 mL kg^-1 min^-1.

The analysis uses the stored list data/external/weee_inclusion_v04.csv. This script reproduces its inclusion decisions
(P06 and P14 excluded) and the valid-sample counts; the medians in criterion (3) can differ slightly from the stored
values, without changing any decision. Usage: python scripts_r1/make_weee_inclusion.py [--out FILE]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1 import data_external as X  # noqa: E402

EXERCISE = ["Cycle1", "Cycle2", "Run1", "Run2"]
STARTS = ["Start_Sit", "Start_Stand", "Start_Cycle1", "Start_Cycle2", "Start_Run1", "Start_Run2"]


def criteria(p, info, mass):
    a = pd.read_csv(X.WEEE_ROOT / p / "VO2" / "DataAverage.csv", usecols=["Time", "VO2[mL/min]", "HR[bpm]"])
    a["ts"] = pd.to_datetime(a["Time"])
    a = a.drop_duplicates("ts").set_index("ts").sort_index()
    grid = pd.date_range(a.index.min(), a.index.max(), freq="1s")
    a = a.reindex(grid)
    ok = ((a["VO2[mL/min]"] > 0) & (a["HR[bpm]"] > 0)).to_numpy()
    seg = np.zeros(len(grid), bool)
    for s, e in X._segments_from_valid(ok, X.GAP_1HZ_S):
        seg[s:e] = True
    starts = pd.to_datetime(info.loc[p, STARTS])
    boundaries = bool(np.all(np.diff(starts.to_numpy().astype("int64")) > 0))
    cond = X._weee_conditions(info.loc[p], grid, a.index[a["VO2[mL/min]"] > 0].max()).to_numpy()
    y = (a["VO2[mL/min]"] / mass[p]).to_numpy()
    conds, meds = [], []
    for c in EXERCISE:
        m = (cond == c) & seg
        if m.sum() >= 180:
            conds.append(c)
            meds.append(float(np.median(y[m][-180:])))
    r = {"participant": p, "valid_s_in_segments": int(seg.sum()), "boundaries_possible": boundaries,
         "exercise_conds_ge180s": ",".join(conds), "max_last180_median": round(max(meds), 2) if meds else np.nan}
    r["crit1"] = r["valid_s_in_segments"] >= 900
    r["crit2"] = boundaries and len(conds) > 0
    r["crit3"] = bool(meds) and max(meds) >= 10
    r["included"] = r["crit1"] and r["crit2"] and r["crit3"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the recomputed table to this CSV file")
    args = ap.parse_args()
    info = pd.read_csv(X.WEEE_ROOT / "Study_Information.csv").set_index("Participant")
    mass = pd.read_csv(X.WEEE_ROOT / "Demographics.csv").set_index("Participant")["Weight"].astype(float).to_dict()
    ref = pd.read_csv(X.EXT_ROOT / "weee_inclusion_v04.csv").set_index("participant")
    out = pd.DataFrame([criteria(p, info, mass) for p in ref.index]).set_index("participant")
    print(out.to_string())
    same = (out["included"] == ref["included"]).all() and (out["valid_s_in_segments"] == ref["valid_s_in_segments"]).all()
    print("inclusion decisions and valid-sample counts match the stored list:", bool(same))
    if args.out:
        out.reset_index().to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
