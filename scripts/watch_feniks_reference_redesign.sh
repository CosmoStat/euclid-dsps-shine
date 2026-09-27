#!/bin/bash
set -uo pipefail
ROOT=${1:?redesign root}
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
r = Path(sys.argv[1])
def read(p):
    return json.loads(p.read_text()) if p.exists() else {}
def state(p):
    if read(p/'FINAL.json').get('status') == 'COMPLETE':
        return 'DONE'
    return read(p/'PROGRESS.json').get('stage', 'waiting')
print('\nPreparation:', state(r/'prepare'))
print('CANDIDATE      SAVED STATE             VAL CDF    PHYS SW  CAPACITY')
for name in ('legacy', 'affine_mass', 'local_128', 'local_256'):
    p = r/name/'capacity'
    f = read(p/'FINAL.json')
    values = f"{f['validation_cdf_max']:9.4f} {f['physical_sw']:10.4f}  {'PASS' if f['capacity_qualified'] else 'FAIL'}" if f else ''
    print(f'{name:15s}{state(p):24s}{values}')
print('Paired replay:', state(r/'replay'))
f = read(r/'replay/FINAL.json')
if f:
    print('  Saved-bank reproduction:', 'PASS' if f['replay_identity_pass'] else 'FAIL')
print('Report:', 'BLOCKED' if (r/'report/BLOCKED.json').exists() else state(r/'report'))
decision = read(r/'report/DECISION.json')
if decision:
    print('Candidate:', decision['selected_candidate'], '; next:', decision['next'])
print('\nSaved state is not live SLURM state. Qualification only, not production approval.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
