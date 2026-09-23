#!/bin/bash
set -eu
ROOT=${1:?clean-parent experiment root}
while true; do
  date
  if [[ -f "$ROOT/JOBS.env" ]]; then
    source "$ROOT/JOBS.env"
    squeue -j "${ALL_JOBS:?}" -o '%.20i %.12T %.10M %R' || true
    sacct -X -j "$ALL_JOBS" --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED \
      --format=JobID,State,ExitCode || true
  fi
  python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
r = Path(sys.argv[1])
def read(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
print('\nSTAGE                        PROGRESS')
contract = read(r / 'contracts/FINAL.json')
if contract.get('status'):
    d = contract.get('decisions', {})
    text = 'DONE ' + ' '.join(f'{k}={"PASS" if v else "FAIL"}' for k, v in d.items())
else:
    p = read(r / 'contracts/PROGRESS.json')
    text = p.get('stage', 'waiting')
print(f'{"contracts":28} {text}')
for arm in ('joint', 'structured'):
    for replica in range(2):
        d = r / arm / f'replica_{replica}'
        if read(d / 'FINAL.json').get('status'):
            text = 'DONE'
        else:
            factors = ('joint',) if arm == 'joint' else ('physical', 'conditional')
            progress = []
            for f in factors:
                p = read(d / f / 'PROGRESS.json')
                progress.append(f"{f}: epoch {p['epoch']} best={p['best_nll']:.4f}" if 'epoch' in p else f'{f}: waiting')
            text = '; '.join(progress)
        print(f'{arm + " " + str(replica):28} {text}')
for stage in ('closure', 'report'):
    p = read(r / stage / 'PROGRESS.json')
    text = 'DONE' if read(r / stage / 'FINAL.json').get('status') else (
        f"{p.get('stage', 'waiting')} {p.get('done', '')}/{p.get('total', '')}")
    print(f'{stage:28} {text}')
PY
  [[ "${2:-}" == --once ]] && break
  echo 'Ctrl-C stops only this watcher. Refresh in 30 seconds.'
  sleep 30
done
