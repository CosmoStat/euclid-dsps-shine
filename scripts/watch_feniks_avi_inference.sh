#!/bin/bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?inference root}")
JOB=$(cat "$ROOT/inference_job.txt")
MIRA=$(cat "$ROOT/mira_job.txt")
while true; do
  date
  squeue -r -j "$JOB,$MIRA" -o '%.20i %.12T %.12M %R' || true
  sacct -X -j "$JOB,$MIRA" --format=JobID%22,State,Elapsed,ExitCode
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for arm in sorted((root / 'arms').glob('*')):
    path = arm / 'FINAL.json'
    if not path.exists():
        path = arm / 'PROGRESS.json'
    if path.exists():
        print(arm.name, json.loads(path.read_text()))
print('MIRA complete:', (root / 'FINAL.json').exists())
PY
  echo 'Ctrl-C stops only this display. Refresh in 30 seconds.'
  sleep 30
done
