#!/bin/bash
set -Eeuo pipefail
source "${1:-outputs/logs/feniks_sc_drws_local_vi_diagnostic_latest.env}"
while true; do
  date
  sacct -X -j "$DIAGNOSTIC_JOB" --format=JobID,JobName%24,State,Elapsed,Timelimit,ExitCode
  python - "$DIAGNOSTIC_ROOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
print('root=', root)
for name in ('CONTRACT_AUDIT.json', 'COST_PREFLIGHT.json', 'PROGRESS.json', 'FAILED.json', 'FINAL.json'):
    path = root/name
    if path.is_file():
        value = json.loads(path.read_text())
        if name == 'CONTRACT_AUDIT.json':
            value = {'status': value['status']}
        if name == 'FINAL.json':
            value = {k: value[k] for k in ('status', 'cases_complete', 'reason', 'budget', 'scientific_promotion') if k in value}
        print(name, json.dumps(value, sort_keys=True))
mode = json.loads((root/'RUN_MANIFEST.json').read_text()).get('mode', 'local_vi')
if mode in ('full_decoder_qualification', 'mdf_precision_qualification'):
    partial = root/'FULL_DECODER_PARTIAL.json'
    cases = json.loads(partial.read_text())['cases'] if partial.is_file() else []
    variants = ('merged_mdf32', 'merged_mdf64') if mode == 'mdf_precision_qualification' else ('legacy', 'merged')
    for variant in variants:
        done = [c for c in cases if c['variant'] == variant]
        print(variant, 'completed points:', len(done), 'checks:', [c['numerical_checks'] for c in done])
    report = root/'FULL_DECODER_QUALIFICATION.json'
    if report.is_file():
        value = json.loads(report.read_text())
        print('candidate numerical checks:', value['candidate_numerical_checks'], 'next:', value['next_stage'])
    print('No bank reuse, local VI, NPE or population training authorized')
elif mode in ('redshift_decomposition', 'photometry_reference'):
    partial = root/('PHOTOMETRY_REFERENCE_PARTIAL.json' if mode == 'photometry_reference' else 'REDSHIFT_DECOMPOSITION_PARTIAL.json')
    completed = len(json.loads(partial.read_text())['points']) if partial.is_file() else 0
    print('Completed ' + mode + ' points:', str(completed) + '/3')
    print('No local VI or population training in this mode')
elif mode == 'local_vi':
    print('Completed observed cases:', len(list((root/'cases').glob('observed_*/COMPLETE.json'))))
    print('Completed simulated cases:', len(list((root/'cases').glob('simulated_*/COMPLETE.json'))))
PY
  if [[ -s "$DIAGNOSTIC_ROOT/FINAL.json" || -s "$DIAGNOSTIC_ROOT/FAILED.json" ]]; then
    break
  fi
  STATE="$(sacct -X -n -P -j "$DIAGNOSTIC_JOB" --format=State | head -n 1)"
  case "$STATE" in
    FAILED*|TIMEOUT*|OUT_OF_MEMORY*|CANCELLED*|NODE_FAIL*|COMPLETED*)
      echo "Slurm=$STATE without final receipt: inspect $DIAGNOSTIC_LOG_ROOT/diagnostic-$DIAGNOSTIC_JOB.err"
      break ;;
  esac
  echo "Ctrl-C stops only this monitor. Refresh in 30s."
  sleep 30
done
