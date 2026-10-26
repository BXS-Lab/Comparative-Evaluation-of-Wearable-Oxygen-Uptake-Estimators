"""Restart the temporal runner until no job remains (orchestration only).
Each restart is a fresh process (clean CUDA context); completed (model, outer) pairs are skipped and interrupted
Optuna studies resume from their SQLite storage.
Usage: python scripts_r1/supervise_temporal.py BEL --workers 4 --models TFT,TCN,PATCHTST,LSTM,GRU [--max-restarts 30]"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    max_restarts = 30
    if "--max-restarts" in args:
        i = args.index("--max-restarts")
        max_restarts = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    for attempt in range(max_restarts + 1):
        print(time.strftime("%Y-%m-%d %H:%M"), f"supervisor: start attempt {attempt}", flush=True)
        proc = subprocess.run([sys.executable, str(ROOT / "scripts_r1" / "run_temporal.py")] + args, cwd=ROOT)
        if proc.returncode == 0:  # every job finished without failure
            print(time.strftime("%Y-%m-%d %H:%M"), "supervisor: all jobs complete", flush=True)
            return 0
        print(time.strftime("%Y-%m-%d %H:%M"), f"supervisor: runner exited with code {proc.returncode}; restarting in 60 s", flush=True)
        time.sleep(60)
    print("supervisor: max restarts reached", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
