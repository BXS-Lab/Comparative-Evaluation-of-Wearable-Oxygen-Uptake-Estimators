"""Run the nested linear family on one dataset. Usage: python scripts_r1/run_linear.py BEL [--outer 1,2]"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1 import linear  # noqa: E402
from r1.cv import outer_dir, save_json, save_predictions  # noqa: E402
from r1.datasets import load  # noqa: E402
from r1.preprocess import CUTOFF_GRID, causal_filter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--outer", default=None, help="comma-separated outer pids (default: all)")
    ap.add_argument("--models", default=",".join(linear.MODELS))
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    panel = load(a.dataset)
    cache = {c: causal_filter(panel.df[panel.features], panel.df["seg"], c) for c in CUTOFF_GRID}
    outers = [int(x) for x in a.outer.split(",")] if a.outer else list(panel.pids)
    kw = {"root": a.root} if a.root else {}
    for model in a.models.split(","):
        for p in outers:
            t0 = time.time()
            best, table, rows, yp = linear.nested_outer_fold(panel, model, p, filtered_cache=cache)
            d = outer_dir(a.dataset, model, p, **kw)
            save_json({k: v for k, v in best.items()}, d / "selected_params.json")
            pd.DataFrame(table).to_csv(d / "inner_trials.csv", index=False)
            save_predictions(d / "predictions_seed0.csv", panel.df, rows, yp)
            # Outer-test performance is deliberately not printed (blinding until the analysis stage).
            print(f"{a.dataset} {model} outer {p:2d}: cutoff={best['cutoff']} "
                  f"{ {k: v for k, v in best.items() if k not in ('cutoff', 'score')} } inner={best['score']:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
