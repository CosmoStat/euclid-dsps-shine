#!/bin/bash
set -uo pipefail
ROOT=${1:?reference-to-parent root}
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
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except json.JSONDecodeError:
        return {}
q = read(r/'qualification/FINAL.json')
p = read(r/'qualification/PROGRESS.json')
print('\nSTAGE              SAVED STATE / DETAIL')
print('Capacity           ', ('PASS' if q['passed'] else 'FAIL') if q else p.get('stage', 'waiting'))
for row in q.get('results', []):
    print(f"  {row['objective']:16s} CDF={row['validation_cdf_max']:.4f} SW={row['physical_sw']:.4f}")
cfg = read(r/'MANIFEST.json').get('settings', {})
n = cfg.get('bank', {}).get('shards', 0)
done = sum(read(r/'banks'/f'shard_{i:03d}'/'FINAL.json').get('status') == 'COMPLETE' for i in range(n))
print(f'Reference banks     {done}/{n}')
f = read(r/'population/FINAL.json'); p = read(r/'population/PROGRESS.json')
print('Blind parent       ', 'DONE' if f else 'BLOCKED' if (r/'population/BLOCKED.json').exists() else f"epoch={p['epoch']} best NLL={p['best_nll']:.4f}" if 'epoch' in p else 'waiting')
s = read(r/'population/STOP.json')
if s: print('  Classifier stop: ', s['reason'])
decision = read(r/'report/DECISION.json')
blocked = read(r/'report/BLOCKED.json')
print('Report             ', 'DONE' if (r/'report/FINAL.json').exists() else 'BLOCKED' if blocked else 'waiting')
if blocked: print('Missing/failed:    ', ', '.join(blocked['missing']))
if decision:
    print('NEXT:              ', decision['next'])
    print('FAIL:              ', ', '.join(k for k,v in decision['gates'].items() if not v) or 'none')
print('\nSaved state is not live SLURM state. No posterior training or production approval.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
