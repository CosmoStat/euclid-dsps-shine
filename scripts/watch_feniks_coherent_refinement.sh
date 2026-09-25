#!/bin/bash
set -eu
ROOT=${1:?coherent refinement root}
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
def state(stage, job):
    final=read(r/stage/'FINAL.json')
    if final.get('status')=='COMPLETE': return 'DONE'
    value=states.get(os.environ.get(job,''),'WAITING')
    return 'NO RECEIPT' if value=='COMPLETED' else value
def number(d,k):
    return f'{d[k]:.4f}' if isinstance(d.get(k),(int,float)) else '-'
p=read(r/'physical/PROGRESS.json'); d=read(r/'physical/DISTRIBUTION_PROGRESS.json')
epochs=read(r/'MANIFEST.json').get('settings',{}).get('epochs','?')
print('\nSTAGE                 STATE         DETAIL')
print(f"Physical continuation {state('physical','FIT_JOB'):13} {p.get('epoch',0)}/{epochs}; best NLL={number(p,'best_nll')}")
print(f"Validation draws      milestone {d.get('epoch','-')}; physical SW={number(d,'validation_physical_sw')}")
print(f"Tails + SFH precision {state('audit','AUDIT_JOB')}")
z=read(r/'audit/FINAL.json')
if z: print('Stored SFH replay:', 'PASS' if z.get('replay_pass') else 'FAIL')
print(f"Report                {state('report','REPORT_JOB')}")
f=read(r/'report/FINAL.json')
if f: print(f"Physical test SW: {number(f,'physical_sw_before')} -> {number(f,'physical_sw_after')}")
b=read(r/'report/BLOCKED.json')
if b: print('Missing:',b.get('missing'))
print('Truth-trained oracle; frozen SFH; no production promotion.')
PY
  [[ ${2:-} == --once ]] && break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
