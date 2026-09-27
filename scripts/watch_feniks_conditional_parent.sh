#!/bin/bash
set -uo pipefail
ROOT=${1:?conditional parent root}
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
def read(path):
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}
cfg = read(root/'MANIFEST.json')['settings']
print('\nSTAGE                  SAVED STATE / DETAIL')
ref = read(root/'reference/FINAL.json')
print('Nested SFH reference  ', 'DONE' if ref else 'waiting')
print('Recycled bank         ', len(list((root/'banks').glob('shard_*/FINAL.json'))), '/', cfg['bank']['shards'])
for stage in ('population', 'tied', 'report'):
    out = root/stage
    state = 'DONE' if read(out/'FINAL.json') else 'waiting'
    if read(out/'BLOCKED.json'): state = 'BLOCKED'
    progress = read(out/'PROGRESS.json')
    details = ''
    if 'epoch' in progress:
        details = f"epoch={progress['epoch']} best NLL={progress['best_nll']:.4f}"
    elif progress:
        details = str(progress.get('stage', ''))
    stop = read(out/'STOP.json')
    if stop: details += ' '+stop['reason']
    print(f'{stage:23s}{state:10s}{details}')
d = read(root/'report/DECISION.json')
if d:
    print('Parent SW:', d['parent_physical_sw'])
    print('FAIL:', ', '.join(k for k,v in d['gates'].items() if not v) or 'none')
    print('Validation flux loglik gain vs tied:', d['comparison']['validation_loglik_gain_expanded_minus_tied'])
print('Blind parent fitting; no target theta, no q feedback, no new DSPS. Saved state is not live SLURM.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
