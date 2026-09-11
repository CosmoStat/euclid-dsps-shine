#!/bin/bash
set -u
ROOT="$(realpath "${1:?experiment root}")"
while true; do
  date
  JOBS=$(cat "$ROOT/preflight_job.txt" "$ROOT/train_jobs.txt" 2>/dev/null | paste -sd,)
  if [[ -n "$JOBS" ]]; then
    squeue -r -j "$JOBS" -o '%.22i %.12T %.12M %R'
    sacct -X -j "$JOBS" --format=JobID%22,State,Elapsed,ExitCode
  fi
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
r = Path(sys.argv[1])
for parent in ('preflight', 'arms'):
    for d in sorted((r / parent).glob('*')):
        try:
            f = d / 'FINAL.json'
            if f.exists():
                print(parent, d.name, json.loads(f.read_text())['status'])
                continue
            p = json.loads((d / 'PROGRESS.json').read_text())
            percent = p.get('progress_percent', 0)
            print(parent, d.name, p.get('stage'), f'{percent:.2f}%',
                  'step=', p.get('step'), '/', p.get('total_steps'),
                  'loss=', p.get('loss'), 'seconds/step=', p.get('seconds'))
            if (d / 'PAUSED.json').exists():
                print('  PAUSED: resume using submit_feniks_avi_experiments.sh resume ROOT')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
PY
  echo 'Ctrl-C stops only this display. Refresh in 30 seconds.'
  sleep 30
done
