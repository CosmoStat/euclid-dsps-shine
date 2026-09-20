#!/bin/bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?campaign root}")
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
for stage in ('preflight','reference','population','posterior_bank','posterior','report'):
    p=r/stage
    paths=sorted(p.glob('shard_*')) if stage in ('reference','posterior_bank') else [p]
    for path in paths:
        final=path/'FINAL.json'; progress=path/'PROGRESS.json'
        state=json.loads(final.read_text()).get('status') if final.exists() else (
            json.loads(progress.read_text()) if progress.exists() else 'waiting')
        print(str(path.relative_to(r)),state)
PY
  echo 'Ctrl-C stops only the watcher. Refresh in 30 seconds.'
  sleep 30
done
