"""Run scripts_r1 steps sequentially from a Python parent (orchestration only). A cmd.exe chain was observed to exit
after a joblib/loky step finished, dropping the remaining steps; a Python parent (as in budget_ladder.py) is robust.
Each step's exit code is logged; a failing step does not stop the chain.
Usage: python scripts_r1/run_chain.py "run_linear.py WEEE_HRQ" "run_trees.py WEEE_HRQ --workers 3 --threads 6" ..."""
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for step in sys.argv[1:]:
    args = shlex.split(step)
    print(time.strftime("%Y-%m-%d %H:%M"), "chain: start", step, flush=True)
    rc = subprocess.run([sys.executable, "-u", str(ROOT / "scripts_r1" / args[0]), *args[1:]], cwd=ROOT).returncode
    print(time.strftime("%Y-%m-%d %H:%M"), f"chain: EXIT {rc}", step, flush=True)
