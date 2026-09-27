#!/bin/bash
set -uo pipefail
ROOT=${1:?audit root}
test -d "$ROOT" || exit 1
while true; do
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "$AUDIT_JOB" -o '%.18i %.12T %.10M %R' 2>/dev/null || true
    sacct -X -j "$AUDIT_JOB" --format=JobID,State,ExitCode --noheader 2>/dev/null || true
  fi
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
r = Path(sys.argv[1])
print('\nSTAGE       SAVED STATE')
for stage in ('capacity', 'tails', 'report'):
    p = r / stage
    if (p / 'FINAL.json').exists():
        state = 'DONE'
    elif (p / 'PROGRESS.json').exists():
        state = json.loads((p / 'PROGRESS.json').read_text()).get('stage', 'started')
    else:
        state = 'waiting'
    print(f'{stage:12s}{state}')
print('\nRead-only diagnosis. Saved state is not live SLURM state; no production promotion.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
