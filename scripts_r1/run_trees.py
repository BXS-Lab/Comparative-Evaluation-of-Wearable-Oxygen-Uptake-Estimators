"""Run the nested tree family. Outer folds run in parallel worker processes.
Usage: python scripts_r1/run_trees.py BEL --models RF,XGB [--outer 1,2] [--workers 4 --threads 6] [--trials N]"""
import argparse
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _one(dataset, model, p, threads, trials, root):
    sys.path.insert(0, str(ROOT))
    from r1 import trees
    from r1.cv import outer_dir, save_json, save_predictions
    from r1.datasets import load
    from r1.preprocess import CUTOFF_GRID, causal_filter
    panel = load(dataset)
    cache = {c: causal_filter(panel.df[panel.features], panel.df["seg"], c) for c in CUTOFF_GRID}
    t0 = time.time()
    best, trials_df, bsf, rows, preds = trees.nested_outer_fold(panel, model, p, cache, n_jobs=threads, n_trials=trials)
    d = outer_dir(dataset, model, p, **({"root": root} if root else {}))
    save_json(best, d / "selected_params.json")
    trials_df.to_csv(d / "inner_trials.csv", index=False)
    bsf.to_csv(d / "inner_best_so_far.csv", index=False)
    for s, yp in preds.items():
        save_predictions(d / f"predictions_seed{s}.csv", panel.df, rows, yp)
    # Outer-test performance is deliberately not printed (blinding until the analysis stage).
    return f"{dataset} {model} outer {p:2d}: inner={best['inner_score']:.3f} cutoff={best['cutoff']} " \
           f"complete={best['n_complete']} pruned={best['n_pruned']} ({time.time() - t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--models", default="RF,XGB")
    ap.add_argument("--outer", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--trials", type=int, default=None, help="override budget (smoke tests only)")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from r1.datasets import load
    pids = [int(q) for q in load(a.dataset).pids]
    outers = [int(x) for x in a.outer.split(",")] if a.outer else pids
    from r1.cv import RESULTS_ROOT
    base = Path(a.root) if a.root else RESULTS_ROOT
    # Resume: skip (model, outer) pairs whose three seed predictions already exist (orchestration only).
    jobs = [(m, p) for m in a.models.split(",") for p in outers
            if not all((base / a.dataset / m / f"outer_{int(p):02d}" / f"predictions_seed{s}.csv").exists() for s in (0, 1, 2))]
    print(f"{len(jobs)} jobs to run", flush=True)
    for msg in Parallel(n_jobs=a.workers, backend="loky")(
            delayed(_one)(a.dataset, m, p, a.threads, a.trials, a.root) for m, p in jobs):
        print(msg, flush=True)


if __name__ == "__main__":
    main()
