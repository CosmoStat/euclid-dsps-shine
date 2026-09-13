#!/bin/bash
# Usage: watch_feniks_avi_next_validation.sh VALIDATION_ROOT [seconds]
set -Eeuo pipefail
ROOT="$(realpath "${1:?validation root}")"
INTERVAL="${2:-30}"
source "$ROOT/JOBS.env"

while true; do
  clear || true
  date
  echo "===== SLURM ====="
  squeue -r -j "$PREFLIGHT_JOB,$INFERENCE_JOB,$AUDIT_JOB,$REPORT_JOB" \
    -o "%.22i %.18T %.12M %R" || true
  sacct -n -X -j "$PREFLIGHT_JOB,$INFERENCE_JOB,$AUDIT_JOB,$REPORT_JOB" \
    --format=JobID%22,State%18,Elapsed,ExitCode | sed '/^[[:space:]]*$/d' || true
  echo
  echo "===== VALIDATION ====="
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stages = (
    ("selection", "selection"),
    ("preflight", "preflight"),
    ("inference", "inference"),
    ("SFH audit", "audit"),
    ("report", "report"),
)
for label, folder in stages:
    base = root / folder
    final = base / "FINAL.json"
    progress = base / "PROGRESS.json"
    if final.exists():
        print(label, json.loads(final.read_text()).get("status"))
    elif progress.exists():
        data = json.loads(progress.read_text())
        print(label, data.get("stage"), f"{data.get('percent', 0):.2f}%", data.get("case", ""))
    else:
        print(label, "waiting")
PY
  echo "Ctrl-C stops only this display. Refresh in $INTERVAL seconds."
  sleep "$INTERVAL"
done
