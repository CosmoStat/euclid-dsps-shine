#!/bin/bash
set -eu
ROOT=${1:?coherent representation root}
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
r=Path(sys.argv[1])
def read(p):
    try: return json.loads(p.read_text())
    except (ValueError,OSError): return {}
states=dict(line.split('|')[:2] for line in os.environ['SLURM_STATES'].splitlines() if '|' in line)
epochs=read(r/'MANIFEST.json').get('settings',{}).get('flow',{}).get('epochs','?')
print('\nFACTOR           STATE          EPOCH    VAL NLL   BEST NLL')
for i,s in enumerate(('physical','sfh_conditional')):
    p=read(r/s/'PROGRESS.json'); f=read(r/s/'FINAL.json')
    state='DONE' if f.get('status')=='COMPLETE' else states.get(os.environ.get('FIT_JOB','')+'_'+str(i),'WAITING')
    def fmt(k): return f'{p[k]:.4f}' if isinstance(p.get(k),(float,int)) else '-'
    print(f"{s:16} {state:12} {str(p.get('epoch','-'))+'/'+str(epochs):>8} {fmt('validation_nll'):>10} {fmt('best_nll'):>10}")
z=read(r/'sfh_zeros/FINAL.json')
print('SFH zero replay:', 'DONE' if z.get('status')=='COMPLETE' else states.get(os.environ.get('ZEROS_JOB',''),'WAITING'))
if 'replay_pass' in z: print('  stored contrasts reproduced:', z['replay_pass'])
f=read(r/'report/FINAL.json')
print('Report:', 'DONE' if f.get('status')=='COMPLETE' else states.get(os.environ.get('REPORT_JOB',''),'WAITING'))
if 'max_physical_sw' in f: print('Physical SW:',f['max_physical_sw'])
b=read(r/'report/BLOCKED.json')
if b: print('Incomplete stages:',b.get('missing'))
print('Truth-trained capacity oracle, not blind population recovery.')
PY
  [[ ${2:-} == --once ]] && break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
