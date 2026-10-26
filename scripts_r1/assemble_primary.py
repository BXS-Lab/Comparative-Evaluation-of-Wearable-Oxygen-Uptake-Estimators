"""Assemble and evaluate the primary analysis at the adopted temporal budget K.

Copies outer-fold prediction files into results_r1/K100/<DATASET>/<MODEL>/: linear and tree models from
results_r1/<DATASET>/ (their budgets are unchanged), temporal models from results_r1/<DATASET>_T100/. Then runs
the evaluation module on the assembled folder. Usage: python scripts_r1/assemble_primary.py WEEE ING"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r1 import evaluate as E  # noqa: E402
from r1.datasets import load, regime_labels  # noqa: E402

RES = ROOT / "results_r1"
DST = RES / "K100"
KEEP = ("predictions_seed0.csv", "predictions_seed1.csv", "predictions_seed2.csv", "selected_params.json")


def assemble(dataset):
    for m in E.MODELS:
        src = RES / (f"{dataset}_T100" if E.FAMILY_OF[m] == "Temporal" else dataset) / m
        folds = sorted(src.glob("outer_*"))
        need = KEEP[:1] if E.FAMILY_OF[m] == "Linear" else KEEP[:3]   # linear models are deterministic (seed 0 only)
        done = [f for f in folds if all((f / k).exists() for k in need)]
        if not folds or len(done) != len(folds):
            raise SystemExit(f"{src}: {len(done)}/{len(folds)} outer folds complete")
        for f in done:
            d = DST / dataset / m / f.name
            d.mkdir(parents=True, exist_ok=True)
            for k in KEEP:
                if (f / k).exists():
                    shutil.copy2(f / k, d / k)


def main():
    for dataset in sys.argv[1:]:
        assemble(dataset)
        panel = load(dataset)
        summary, lb, fc = E.run(dataset, panel, root=DST, regime=regime_labels(panel))
        print(dataset, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
