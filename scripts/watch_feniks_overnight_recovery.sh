#!/bin/bash
set -uo pipefail
ROOT=${1:?recovery root}
test -d "$ROOT" || exit 1
while true; do
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "$CURRENT_JOBS" -o '%.18i %.12T %.10M %R' 2>/dev/null || true
    sacct -X -j "$CURRENT_JOBS" --format=JobID,State,ExitCode --noheader 2>/dev/null || true
  fi
  python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def read(p):
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except json.JSONDecodeError:
        return {}
print('\nSTAGE             SAVED STATE / DETAIL')
for stage in ('parent', 'audit', 'posterior', 'evaluation', 'report'):
    out = root/stage
    final, failed, blocked = (read(out/n) for n in ('FINAL.json','FAILED.json','BLOCKED.json'))
    state = 'DONE' if final else 'WAITING'
    if blocked: state = 'BLOCKED'
    if failed: state = 'FAILED'
    detail = ''
    if stage == 'parent':
        p = read(out/'tied/PROGRESS.json')
        detail = str(p.get('stage',''))
        if p and not final and not failed: state = 'PROGRESS'
        if (out/'tied/FINAL.json').exists(): detail = 'tied fit saved'
    elif stage == 'audit':
        detail = read(out/'PROGRESS.json').get('stage','')
        if detail and not final and not failed: state = 'PROGRESS'
        d = read(out/'DECISION.json')
        if d:
            detail = f"numerics={'PASS' if d['safe_to_optimize'] else 'FAIL'}; tails NOT certified"
            rounding = sum(c.get('bounded_quantization_compatible_coordinates',0) for c in d.get('checks',{}).values())
            if rounding: detail += f'; bounded rounding={rounding}'
            if not d['safe_to_optimize']: state = 'BLOCKED'
    elif stage == 'posterior':
        p = read(out/'PROGRESS.json')
        if p:
            total = read(root/'MANIFEST.json')['settings']['continuation']['epochs']
            detail = f"+{p['epoch']}/{total} epochs; best NLL={p['best_nll']:.4f}"
            if not final and not failed: state = 'PROGRESS'
        stop = read(out/'STOP.json')
        if stop: detail += ' '+stop['reason']
    if failed: detail = failed.get('message','')[:110]
    print(f'{stage:18s}{state:10s} {detail}')
print('\nNo new DSPS. Same old parent for q. Saved state is not live SLURM state.')
print('Numerical audit permits optimization only; no production approval.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
