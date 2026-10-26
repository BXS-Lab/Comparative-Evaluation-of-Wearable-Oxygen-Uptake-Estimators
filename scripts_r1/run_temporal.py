"""Run the nested temporal family. Each (model, outer fold) is one fresh Optuna study run
sequentially inside one worker process; several workers share the GPU (concurrency across outer folds).
Usage: python scripts_r1/run_temporal.py BEL --models TCN,PATCHTST [--outer 1,2] [--workers 3]
       smoke tests only: --trials N --epochs N"""
import argparse
import json
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _one(dataset, model, p, trials, epochs, stride, root, out_name=None):
    """Isolate failures: a crashed job (e.g. GPU driver reset) is reported, not propagated; its Optuna study
    persists on disk and resumes when the runner is restarted (scripts_r1/supervise_temporal.py)."""
    try:
        return _one_inner(dataset, model, p, trials, epochs, stride, root, out_name)
    except Exception as e:  # noqa: BLE001
        return f"FAILED {dataset} {model} outer {p:2d}: {type(e).__name__}: {str(e)[:160]}"


def _one_inner(dataset, model, p, trials, epochs, stride, root, out_name=None):
    sys.path.insert(0, str(ROOT))
    from r1 import temporal
    from r1.cv import outer_dir, save_json, save_predictions
    from r1.datasets import load
    from r1.preprocess import CUTOFF_GRID, causal_filter
    if epochs:
        temporal.SETTINGS["max_epochs"] = epochs
    if stride:
        temporal.SETTINGS["stride"] = stride
    panel = load(dataset)
    cache = {c: causal_filter(panel.df[panel.features], panel.df["seg"], c) for c in CUTOFF_GRID}
    t0 = time.time()
    d = outer_dir(out_name or dataset, model, p, **({"root": root} if root else {}))
    chosen, trials_df, bsf, events, rows, preds = temporal.nested_outer_fold(panel, model, p, cache, n_trials=trials,
                                                                             study_dir=d)
    save_json(chosen, d / "selected_params.json")
    save_json(dict(temporal.SETTINGS, trials=trials or temporal.SETTINGS["trials"]), d / "settings.json")
    trials_df.to_csv(d / "inner_trials.csv", index=False)
    bsf.to_csv(d / "inner_best_so_far.csv", index=False)
    for s, yp in preds.items():
        save_predictions(d / f"predictions_seed{s}.csv", panel.df, rows, yp)
    # Outer-test performance is deliberately not printed (blinding until the analysis stage).
    return (f"{dataset} {model} outer {p:2d}: inner={chosen['inner_score']:.3f} L={chosen['L']} cutoff={chosen['cutoff']} "
            f"complete={chosen['n_complete']} pruned={chosen['n_pruned']} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--models", default="LSTM,GRU,TCN,TFT,PATCHTST")
    ap.add_argument("--outer", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-name", default=None, help="results folder name (default: dataset); used for budget extensions")
    a = ap.parse_args()
    out_name = a.out_name or a.dataset
    from r1.datasets import load
    pids = [int(q) for q in load(a.dataset).pids]
    outers = [int(x) for x in a.outer.split(",")] if a.outer else pids
    from r1.cv import RESULTS_ROOT
    base = Path(a.root) if a.root else RESULTS_ROOT
    # Resume: skip (model, outer fold) pairs whose three seed predictions already exist (orchestration only).
    jobs = [(m, p) for m in a.models.split(",") for p in outers
            if not all((base / out_name / m / f"outer_{int(p):02d}" / f"predictions_seed{s}.csv").exists() for s in (0, 1, 2))]
    print(f"{len(jobs)} jobs to run", flush=True)
    failed = False
    for msg in Parallel(n_jobs=a.workers, backend="loky", return_as="generator_unordered")(
            delayed(_one)(a.dataset, m, p, a.trials, a.epochs, a.stride, a.root, out_name) for m, p in jobs):
        print(time.strftime("%Y-%m-%d %H:%M "), msg, flush=True)
        if msg.startswith("FAILED"):
            failed = True
    if failed:
        sys.exit(3)  # supervisor restarts in a fresh process (clean CUDA context)


if __name__ == "__main__":
    main()
