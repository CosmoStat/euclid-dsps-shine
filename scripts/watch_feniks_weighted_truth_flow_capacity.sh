#!/bin/bash
set -Eeuo pipefail

ROOT=$(realpath "${1:?weighted truth flow capacity root}")
ONCE=${2:-}

while true; do
  if [[ -t 1 && "$ONCE" != --once ]]; then printf '\033[H\033[2J'; fi
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "$ALL_JOBS" -o '%.22i %.12T %.12M %R' || true
    FAILURES=$(sacct -X -n -j "$ALL_JOBS" \
      -s FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED \
      --format=JobID,State,ExitCode) || FAILURES='sacct unavailable'
    if [[ -n "${FAILURES//[[:space:]]/}" ]]; then
      printf '\nFAILURES\n%s\n' "$FAILURES"
    fi
  fi
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

print("\nFLOW       STATE       EPOCH      VAL NLL     BEST NLL")
settings = load(root / "MANIFEST.json").get("settings", {})
epochs = settings.get("flow", {}).get("epochs", "?")
for replica in range(2):
    directory = root / f"replica_{replica}"
    final = load(directory / "FINAL.json")
    progress = load(directory / "PROGRESS.json")
    state = "DONE" if final else ("RUNNING" if progress else "WAITING")
    epoch = f"{progress.get('epoch', '-')}/{epochs}"
    validation = progress.get("validation_nll")
    best = progress.get("best_nll")
    number = lambda value: f"{value:10.4f}" if isinstance(value, (int, float)) else f"{'-':>10}"
    print(f"Flow {replica + 1:<4} {state:11} {epoch:9} {number(validation)} {number(best)}")
report = load(root / "report/FINAL.json")
print("\nReport:", "DONE" if report else "WAITING")
if report:
    print("max physical sliced-Wasserstein:", report["max_physical_sliced_wasserstein"])
PY
  [[ "$ONCE" == --once ]] && break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
