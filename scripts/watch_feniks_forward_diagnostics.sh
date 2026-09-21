#!/bin/bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?diagnostic root}")
while true; do
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "$ALL_JOBS" -o '%.22i %.16T %.12M %R'
    sacct -X -j "$ALL_JOBS" -s FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED --format=JobID,State,ExitCode
  fi
  python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
for relative in ('baseline/report','classifier/population','posterior/posterior','posterior/report','capacity_analytic/report','capacity_truth/report','.'):
    p=r/relative
    file=next((p/n for n in ('FINAL.json','PROGRESS.json') if (p/n).exists()),None)
    print(relative, json.loads(file.read_text()) if file else 'waiting')
PY
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
