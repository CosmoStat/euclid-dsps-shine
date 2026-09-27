#!/bin/bash
set -uo pipefail
ROOT=${1:?posterior root}
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
def read(p):
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except json.JSONDecodeError:
        return {}
cfg = read(root/'MANIFEST.json')['settings']
print('\nSTAGE             SAVED STATE / DETAIL')
done = len(list((root/'banks').glob('shard_*/FINAL.json')))
rows = sum(read(p).get('done', 0) for p in (root/'banks').glob('shard_*/PROGRESS.json'))
print(f"Fresh parent bank {done}/{cfg['bank']['shards']} complete; {rows} parent simulations saved")
for name in ('posterior', 'evaluation/in_model', 'evaluation/coherent_target', 'report'):
    out = root/name
    final, progress, stop = (read(out/n) for n in ('FINAL.json', 'PROGRESS.json', 'STOP.json'))
    state = 'DONE' if final else 'waiting'
    if not final and read(out/'FAILED.json'): state = 'FAILED'
    if not final and read(out/'BLOCKED.json'): state = 'BLOCKED'
    if progress:
        state += f" epoch={progress.get('epoch')} best NLL={progress.get('best_nll', 0):.4f}"
    if stop: state += ' '+stop['reason']
    print(f'{name:28s}{state}')
decision = read(root/'report/DECISION.json')
for label, groups in decision.get('calibration_flags', {}).items():
    print(label, '; '.join(f"{g}: coverage={v['coverage']} PIT={v['pit']}" for g,v in groups.items()))
print('\nExploratory: frozen imperfect parent; dust/SFH retained; no production approval.')
print('Saved state is not live SLURM state. Parent-recovery diagnostic left untouched.')
PY
  [[ ${2:-} != --once ]] || break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
