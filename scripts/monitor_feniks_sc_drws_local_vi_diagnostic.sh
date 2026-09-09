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
for name in ('CONTRACT_AUDIT.json', 'COST_PREFLIGHT.json', 'PROGRESS.json', 'NIGHT_PROGRESS.json', 'FAILED.json', 'FINAL.json', 'NIGHT_FINAL.json'):
    path = root/name
    if path.is_file():
        value = json.loads(path.read_text())
        if name == 'CONTRACT_AUDIT.json':
            value = {'status': value['status']}
        if name in ('FINAL.json', 'NIGHT_FINAL.json'):
            value = {k: value[k] for k in ('status', 'cases_complete', 'reason', 'budget', 'scientific_promotion') if k in value}
        print(name, json.dumps(value, sort_keys=True))
mode = json.loads((root/'RUN_MANIFEST.json').read_text()).get('mode', 'local_vi')
if mode == 'precision_night':
    partial = root/'FULL_DECODER_PARTIAL.json'
    if partial.is_file():
        cases = json.loads(partial.read_text())['cases']
        print('Qualification:', len(cases), '/6', [c['numerical_checks'] for c in cases])
    for arm in ('smoke', 'B', 'C'):
        receipt = root/'arms'/arm/'ARM_COMPLETE.json'
        progress = root/'arms'/arm/'train/training_progress.json'
        if receipt.is_file():
            print('Train', arm, 'COMPLETE')
        elif progress.is_file():
            print('Train', arm, progress.read_text())
    for arm in 'ABC':
        metadata = root/'validation'/arm/'tracking_k256/inference/shard_metadata'
        print(arm, 'K256:', len(list(metadata.glob('batch_*.json'))), '/32')
    print('Sequential fixed-parent pilot; no population training or scientific promotion')
elif mode == 'redshift_precision_audit':
    partial = root/'REDSHIFT_PRECISION_PARTIAL.json'
    if partial.is_file():
        value = json.loads(partial.read_text())
        print('Branches:', value['completed_branches'], '/', value['expected_branches'])
        for b in value['branches']:
            c = next(c for c in b['checks'] if c['component'] == 'lsst_z')
            print(b['name'], 'lsst_z:', c['status'], 'all bands/density:', b['all_checks_passed'])
    print('No NPE, local VI or population training authorized')
elif mode == 'target_resolution_audit':
    partial = root/'TARGET_RESOLUTION_PARTIAL.json'
    if partial.is_file():
        value = json.loads(partial.read_text())
        for c in value['checks']:
            print('point', c['point_index'], c['coordinate'], c['component'], c['status'])
        print('next:', value['next_stage'])
    print('No NPE, local VI or population training authorized')
elif mode in ('full_decoder_qualification', 'mdf_precision_qualification'):
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
    manifest = json.loads((root/'RUN_MANIFEST.json').read_text())
    total = manifest['objects_per_group']
    print('Completed observed cases:', len(list((root/'cases').glob('observed_*/COMPLETE.json'))), '/', total)
    print('Completed simulated cases:', len(list((root/'cases').glob('simulated_*/COMPLETE.json'))), '/', total)
    if 'qualified_night' in manifest:
        print('Qualified source arm:', manifest['qualified_night']['arm'])
        if 'objective_pilot' in manifest and (root/'OBJECTIVE_AUDIT.json').is_file():
            audit = json.loads((root/'OBJECTIVE_AUDIT.json').read_text())
            print('Full VI objective audit:', audit['status'], len(audit['audits']), '/', 4*total)
        controlled = json.loads((root/'RUN_MANIFEST.json').read_text()).get('optimization_regimes')
        probe = json.loads((root/'RUN_MANIFEST.json').read_text()).get('support_probe')
        if (root/'NIGHT_GATE.json').is_file():
            print('Night resource gate:', json.loads((root/'NIGHT_GATE.json').read_text())['status'])
        print('Gated overnight local reverse/wake extension; not global NN or population training' if 'night_extension' in manifest else
              'Versioned transport64 reverse/wake pilot; frozen parent; no promotion' if 'transport_contract' in manifest else
              'Transport precision audit only; no optimization or promotion' if 'transport_precision_reference' in manifest else
              'Audit-gated reverse/wake pilot; frozen source and parent; no promotion' if 'objective_pilot' in manifest else
              'Final checkpoint replay K4096; no optimization or promotion' if manifest.get('method') == 'qualified_long_replay_v1' else
              'Fixed dispersion/mixture probe; no optimization or promotion' if probe else
              'Fixed regimes, two starts, saved trajectories; no promotion' if controlled else
              'Two nearby starts; final direct draws only; no global NPE or population training')
PY
  NIGHT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mode") == "precision_night")' "$DIAGNOSTIC_ROOT/RUN_MANIFEST.json")"
  if [[ -s "$DIAGNOSTIC_ROOT/FAILED.json" || -s "$DIAGNOSTIC_ROOT/NIGHT_FINAL.json" || ( "$NIGHT" != "True" && -s "$DIAGNOSTIC_ROOT/FINAL.json" ) ]]; then
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
