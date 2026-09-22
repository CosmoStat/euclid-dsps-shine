#!/bin/bash
set -Eeuo pipefail

ROOT=$(realpath "${1:?failure-mode audit root}")
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
tasks = (
    ("Decoder + observation", "decoder_observation"),
    ("Nuisance conditional", "nuisance_sensitivity"),
    ("Posterior original", "posterior_original"),
    ("Posterior continued", "posterior_continued"),
    ("Population + classifier", "population_identifiability"),
)

def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

print("\nSTAGE                       STATE")
for label, directory in tasks:
    path = root / directory
    state = "DONE" if load(path / "FINAL.json") else (
        "RUNNING" if load(path / "PROGRESS.json") else "WAITING"
    )
    print(f"{label:27} {state}")

decision = load(root / "DECISION.json")
if decision:
    print("\nDECISION")
    for name, passed in decision["checks"].items():
        print(f"  {name:34} {'PASS' if passed else 'FAIL'}")
    modes = decision.get("failure_modes", [])
    print("failure modes:", ", ".join(modes) if modes else "none")
    print("ready for production:", decision["ready_for_production"])
else:
    print("\nReport: WAITING")
PY
  [[ "$ONCE" == --once ]] && break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
