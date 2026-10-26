"""Uniform, convergence-driven tuning budget for the temporal models.

Checkpoints 40 -> 60 -> 80 -> 100 trials (cap). Step statistic for (dataset, model) at checkpoint k: the median over
outer folds of the improvement in the inner-CV best score from the first k_prev finished trials to the first k
(k_prev = previous checkpoint). The adopted budget K is the first checkpoint at which the statistic is < 0.01
mL kg^-1 min^-1 for every dataset x model (WEEE, ING, BEL, BEL_CORE x 5 temporal models), or 100. All temporal
models on all datasets are run to K. Because K cannot be below a checkpoint at which WEEE or ING fail, BEL and
BEL_CORE are extended to a checkpoint only once WEEE and ING pass it (or at the cap). Inner-CV scores only;
outer-test performance is never read here. Persisted Optuna studies are copied forward from the latest folder
(<DATASET> = 25 trials, <DATASET>_T<k>); earlier folders are left untouched. Decisions: results_r1/budget_decisions.json.
Usage: python scripts_r1/budget_ladder.py"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results_r1"
MODELS = ["TFT", "TCN", "PATCHTST", "LSTM", "GRU"]
FAST, SLOW = ["WEEE", "ING"], ["BEL", "BEL_CORE"]
CHECKPOINTS = [25, 40, 60, 80, 100]
THRESHOLD = 0.01
LOG = RES / "budget_decisions.json"
FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def folder(dataset, k):
    return RES / (dataset if k == 25 else f"{dataset}_T{k}")


def stamp():
    return time.strftime("%Y-%m-%d %H:%M")


def step_statistic(dataset, model, k_prev, k):
    imp = []
    for db in sorted((folder(dataset, k) / model).glob("outer_*/study.db")):
        s = optuna.load_study(study_name="study", storage=f"sqlite:///{db.as_posix()}")
        fin = sorted((t for t in s.trials if t.state in FINISHED), key=lambda t: t.number)
        if len(fin) < k:
            raise SystemExit(f"{db}: {len(fin)} finished trials < {k}")
        best = lambda ts: min((t.value for t in ts if t.state == optuna.trial.TrialState.COMPLETE), default=np.inf)
        imp.append(best(fin[:k_prev]) - best(fin[:k]))
    return float(np.median(imp)), len(imp)


def copy_forward(dataset, k):
    for m in MODELS:
        src_k = max(c for c in CHECKPOINTS if c < k and (folder(dataset, c) / m).is_dir())
        for src in sorted((folder(dataset, src_k) / m).glob("outer_*")):
            dst = folder(dataset, k) / m / src.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("study.db", "training_events.csv"):
                if (src / f).exists() and not (dst / f).exists():
                    shutil.copy2(src / f, dst / f)


def extend(dataset, k):
    copy_forward(dataset, k)
    cmd = [sys.executable, str(ROOT / "scripts_r1" / "supervise_temporal.py"), dataset, "--workers", "4", "--trials", str(k),
           "--out-name", folder(dataset, k).name, "--models", ",".join(MODELS)]
    print(stamp(), "ladder: running", " ".join(cmd[2:]), flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit(f"runner failed for {dataset} at {k}")


def evaluate(dataset, k_prev, k):
    ok = True
    for m in MODELS:
        med, n = step_statistic(dataset, m, k_prev, k)
        ok &= med < THRESHOLD
        entry = {"time": stamp(), "dataset": dataset, "model": m, "step": f"{k_prev}->{k}", "median_improvement": med,
                 "n_folds": n, "below_threshold": med < THRESHOLD}
        data = json.loads(LOG.read_text()) if LOG.exists() else []
        LOG.write_text(json.dumps(data + [entry], indent=2))
        print(stamp(), f"ladder: {dataset} {m} {k_prev}->{k} median {med:.4f} {'ok' if med < THRESHOLD else 'NOT converged'}", flush=True)
    return ok


def decide(k):
    data = json.loads(LOG.read_text()) if LOG.exists() else []
    LOG.write_text(json.dumps(data + [{"time": stamp(), "adopted_budget": k}], indent=2))
    print(stamp(), f"ladder: ADOPTED BUDGET K = {k}", flush=True)


def main():
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    for i in range(2, len(CHECKPOINTS)):          # 40 already evaluated for WEEE/ING (2026-10-17: not converged)
        k_prev, k = CHECKPOINTS[i - 1], CHECKPOINTS[i]
        cap = k == CHECKPOINTS[-1]
        fast_ok = True
        for d in FAST:
            extend(d, k)
            fast_ok &= evaluate(d, k_prev, k)
        if not (fast_ok or cap):
            continue
        slow_ok = True
        for d in SLOW:
            extend(d, k)
            slow_ok &= evaluate(d, k_prev, k)
        if slow_ok or cap:
            decide(k)
            return
    raise SystemExit("unreachable")


if __name__ == "__main__":
    main()
