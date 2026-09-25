#!/bin/bash
set -eu
ROOT=${1:?precision audit root}
while true; do
  date
  SLURM_STATES=''
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "$ALL_JOBS" -o '%.20i %.12T %.10M %R' || true
    SLURM_STATES=$(sacct -X -n -P -j "$ALL_JOBS" --format=JobID,State 2>/dev/null || true)
  fi
  export SLURM_STATES
  python - "$ROOT" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
def read(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
def state(job, path):
    if read(path/'FINAL.json').get('status') == 'COMPLETE':
        return 'DONE'
    rows = dict(line.split('|')[:2] for line in os.environ['SLURM_STATES'].splitlines() if '|' in line)
    live = rows.get(job, '')
    if live:
        return live if live != 'COMPLETED' else 'MISSING_FINAL'
    return 'PARTIAL' if (path/'PROGRESS.json').exists() else 'WAITING'
print('\nSTAGE           STATE             DETAIL')
for label, var, rel in [('Cache ratios','CACHE_JOB','cache'), ('Decoder','DECODER_JOB','decoder')]:
    p = root/rel
    progress = read(p/'PROGRESS.json')
    print(f"{label:15} {state(os.environ.get(var,''),p):17} {progress.get('stage','-')} {progress.get('complete','')}")
for i, arm in enumerate(('noiseless_photometry','noisy_photometry')):
    p=root/'metric'/arm
    job=os.environ.get('METRIC_JOB','')
    print(f"{'Metric '+arm.split('_')[0]:15} {state(job+'_'+str(i),p):17}")
total=read(root/'MANIFEST.json').get('settings',{}).get('bootstraps',8)*4
done=sum(read(p).get('status')=='COMPLETE' for p in (root/'bootstrap').glob('repeat_*/*/FINAL.json'))
print(f'Bootstrap cells {done}/{total}')
failed=[line for line in os.environ['SLURM_STATES'].splitlines() if any(s in line for s in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY'))]
for line in failed: print('SLURM:',line)
final=read(root/'report/FINAL.json')
print('Report:',final.get('status','WAITING'))
print('Roadmap:',root/'ROADMAP_STATUS.md')
PY
  [[ "${2:-}" == --once ]] && break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
