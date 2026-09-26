#!/bin/bash
set -uo pipefail
ROOT=$(realpath "${1:?run root}")
while true; do
  date
  ALL_JOBS=''
  [[ ! -f "$ROOT/JOBS.env" ]] || source "$ROOT/JOBS.env"
  if [[ -n "$ALL_JOBS" ]]; then
    squeue -j "$ALL_JOBS" -o '%.22i %.12T %.10M %R' 2>/dev/null || true
    sacct -X -j "$ALL_JOBS" --noheader --format=JobID,State,ExitCode 2>/dev/null | \
      awk '/FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL/' || true
  fi
  python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
def read(p):
    try: return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return {}
print('\nSTAGE          SAVED STATE       PROGRESS')
for name in ('reference','banks','oracle','population','posterior','report'):
    if name=='banks':
        total=read(r/'MANIFEST.json')['settings']['bank']['shards']
        done=sum((r/'banks'/f'shard_{i:03d}'/'FINAL.json').exists() for i in range(total))
        print(f'{name:14} {done}/{total} complete'); continue
    final=read(r/name/'FINAL.json'); p=read(r/name/'PROGRESS.json'); stop=read(r/name/'STOP.json')
    status='DONE' if final else ('BLOCKED' if (r/name/'BLOCKED.json').exists() else 'checkpoint' if p else 'waiting')
    detail=''
    if p.get('epoch') is not None: detail=f"epoch={p['epoch']} best NLL={p.get('best_nll',0):.4f}"
    if stop: detail+=f" {stop['reason']}"
    print(f'{name:14} {status:17} {detail}')
print('\nSaved progress is not live state; use SLURM above. No automatic production promotion.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
