#!/bin/bash
set -eu
ROOT=${1:?coherent parent root}
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
root=Path(sys.argv[1])
def read(p):
    try: return json.loads(p.read_text())
    except (OSError, ValueError): return {}
print('\nSTAGE                  STATE      DETAIL')
states=dict(line.split('|')[:2] for line in os.environ['SLURM_STATES'].splitlines() if '|' in line)
for i,s in enumerate(('train','validation','test')):
    p=root/'sampling'/s
    progress=read(p/'PROGRESS.json')
    live=states.get(os.environ.get('SAMPLE_JOB','')+'_'+str(i),'WAITING')
    state='DONE' if read(p/'FINAL.json').get('status')=='COMPLETE' else live
    print(f"{'Parent '+s:22} {state:10} {progress.get('stage','')} {progress.get('done','')}/{progress.get('total','')}")
tasks=read(root/'MANIFEST.json').get('tasks',[])
done=sum(read(root/'photometry'/f'task_{i:03d}'/'FINAL.json').get('status')=='COMPLETE' for i in range(len(tasks)))
print(f'Photometry              {done}/{len(tasks)} tasks')
for i in range(len(tasks)):
    p=root/'photometry'/f'task_{i:03d}'
    if read(p/'FINAL.json').get('status')=='COMPLETE': continue
    progress=read(p/'PROGRESS.json')
    live=states.get(os.environ.get('PHOTO_JOB','')+'_'+str(i),'WAITING')
    if progress or live not in ('WAITING','PENDING'):
        print(f"  task {i:02d} {live:12} {progress.get('done',0)}/{progress.get('total',tasks[i]['stop']-tasks[i]['start'])}")
summary=read(root/'report/SUMMARY.json')
print('Dataset:',summary.get('status','WAITING'))
for s,v in summary.get('splits',{}).items():
    print(f"  {s}: parent={v['rows']} selected={v['selected']} alpha={v['empirical_alpha']:.4f}")
for line in os.environ['SLURM_STATES'].splitlines():
    if any(x in line for x in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY')): print('SLURM:',line)
for failure in summary.get('failures',[]): print('FAIL:',failure)
print('No parent-prior or posterior training in this run.')
PY
  [[ "${2:-}" == --once ]] && break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
