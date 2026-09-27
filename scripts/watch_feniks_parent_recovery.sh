#!/bin/bash
set -uo pipefail
ROOT=${1:?diagnostic root}
test -d "$ROOT" || exit 1
while true; do
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "$CURRENT_JOBS" -o '%.18i %.12T %.10M %R' 2>/dev/null || true
    sacct -X -j "$CURRENT_JOBS" --format=JobID,State,ExitCode --noheader 2>/dev/null || true
  fi
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
def read(p):
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except json.JSONDecodeError:
        return {}
print('\nSTAGE              SAVED STATE / DETAIL')
for name in ('cache', 'learned_parent', 'capacity_control', 'report'):
    out = root / name
    final, failed = read(out/'FINAL.json'), read(out/'FAILED.json')
    progress, blocked = read(out/'PROGRESS.json'), read(out/'BLOCKED.json')
    if final:
        state = 'DONE'
        if 'classifier_parent_sw' in final:
            gates = final['gates']
            state += f"  parent SW={final['classifier_parent_sw']:.4f}; flux-fit={'PASS' if gates['classifier'] else 'FAIL'}; labels={'PASS' if gates['label_penalized'] else 'FAIL'}"
    elif failed:
        state = 'FAILED  ' + failed['message'][:130]
    elif blocked:
        state = 'BLOCKED  ' + ', '.join(blocked['missing'])
    else:
        state = progress.get('stage', 'waiting')
        if 'total' in progress:
            state += f" {progress['done']}/{progress['total']}"
    print(f'{name:19s}{state}')
decision = read(root/'report/DECISION.json')
if decision: print('NEXT: ', decision['next'])
print('\nNo training or DSPS. Saved state is not live SLURM state. Diagnostic only.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
