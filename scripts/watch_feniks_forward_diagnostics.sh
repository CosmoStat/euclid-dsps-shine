#!/bin/bash
set -Eeuo pipefail
ROOT=$(realpath "${1:?diagnostic root}")
ONCE=${2:-}
while true; do
  if [[ -t 1 && "$ONCE" != --once ]]; then printf '\033[H\033[2J'; fi
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "$ALL_JOBS" -o '%.22i %.16T %.12M %R' || true
    FAILURES=$(sacct -X -n -j "$ALL_JOBS" -s FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED --format=JobID,State,ExitCode) || FAILURES='sacct unavailable'
    [[ -z "${FAILURES//[[:space:]]/}" ]] || printf '\nECHECS SLURM\n%s\n' "$FAILURES"
  fi
  python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
def number(value):
    return f'{value:.4f}' if isinstance(value, (int,float)) else '-'
settings=read(r/'MANIFEST.json').get('settings',{})
print('\nETAPE                 ETAT         EPOQUE       VAL NLL   BEST NLL')
for label,relative,section in (
    ('Evaluation initiale','baseline/report',None),
    ('Classifieur + parent','classifier/population','classifier'),
    ('Posterior','posterior/posterior','posterior'),
    ('Evaluation posterior','posterior/report',None),
    ('Capacite analytique','capacity_analytic/report','capacity'),
    ('Capacite verite','capacity_truth/report','capacity'),
    ('Synthese','.',None),
):
    p=r/relative
    final=read(p/'FINAL.json')
    progress=read(p/'PROGRESS.json')
    state='TERMINE' if final else ('PROGRES' if progress else 'EN ATTENTE')
    epoch=f"{progress['epoch']}/{settings.get(section,{}).get('epochs','?')}" if 'epoch' in progress else '-'
    val=progress.get('validation_nll', final.get('classifier_validation_nll'))
    best=progress.get('best_nll')
    print(f'{label:21} {state:12} {epoch:10} {number(val):>10} {number(best):>10}')
print('\nPROGRES = dernier checkpoint; etat actif dans SLURM ci-dessus.')
PY
  [[ "$ONCE" != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
