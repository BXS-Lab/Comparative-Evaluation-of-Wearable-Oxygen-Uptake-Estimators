"""Estimate the WEEE Zephyr-to-VO2-Master clock offsets for all participants (r1/weee_offsets.py) and write
data/external/weee_clock_offsets.csv. Usage: python scripts_r1/make_weee_offsets.py [--out FILE]"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1 import data_external as X  # noqa: E402
from r1.weee_offsets import estimate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(X.EXT_ROOT / "weee_clock_offsets.csv"))
    a = ap.parse_args()
    inc = pd.read_csv(X.EXT_ROOT / "weee_inclusion_v04.csv").set_index("participant")["included"]
    rows = []
    for p in inc.index:
        r = estimate(X.WEEE_ROOT / p)
        rows.append({"participant": p, "included": bool(inc[p]), **r})
        print(p, r)
    pd.DataFrame(rows)[["participant", "included", "method", "offset_s", "n_windows_ok", "peak_r", "flag"]].to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
