#!/bin/bash
set -u

REFRESH_SECONDS="${1:-60}"
: "${ALL_JOBS:?Source outputs/logs/feniks_sc_drws_epoch160_catalogue_calibration_latest.env first}"
: "${CALIBRATION_JOB:?CALIBRATION_JOB is required}"
: "${FINAL_JOB:?FINAL_JOB is required}"
: "${CALIBRATION_ROOT:?CALIBRATION_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

while true; do
  clear
  date
  echo
  echo "===== SLURM ====="
  squeue -r -j "$ALL_JOBS" -o "%.18i %.28j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$ALL_JOBS" \
    --format=JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,NodeList 2>/dev/null || true

  echo
  echo "===== CALIBRATION CATALOGUE TEST INDÉPENDANT ====="
  for spec in "common32/mira" "common32/tarp" "q256/mira" "q256/tarp"; do
    if [[ -e "$CALIBRATION_ROOT/$spec/DONE" ]]; then
      echo "$spec: terminé"
    else
      echo "$spec: attente/en cours"
    fi
  done
  grep -hE "\[catalogue-calibration\] (mode=|complete ->)" \
    "$LOG_ROOT"/calibration-${CALIBRATION_JOB}_*.out 2>/dev/null | tail -n 12 || true

  echo
  echo "===== ERREURS ====="
  grep -hE "Traceback|RESOURCE_EXHAUSTED|Out of memory|ValueError|RuntimeError|TypeError|SystemError" \
    "$LOG_ROOT"/calibration-${CALIBRATION_JOB}_*.err \
    "$LOG_ROOT/finalize-${FINAL_JOB}.err" 2>/dev/null | tail -n 20 || true

  echo
  echo "===== REÇU FINAL ====="
  RECEIPT="$CALIBRATION_ROOT/CATALOGUE_CALIBRATION_COMPLETE.json"
  if [[ -s "$RECEIPT" ]]; then
    python - "$RECEIPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(
    f"status={r['status']} objects={r['catalogue_objects_evaluated']} "
    "common=32 q-only=256"
)
ess=r["iw_support_warning"]["median_effective_samples"]
print(f"support IW held-out K1024: raw ESS={ess['raw']:.2f}, ema ESS={ess['ema']:.2f}")
for name, artifact in sorted(r["artifacts"].items()):
    if name.endswith("_plot"):
        print(f"{name}: {artifact['path']}")
PY
  else
    echo "non terminé"
  fi

  echo
  echo "Ctrl-C arrête seulement ce monitor. Rafraîchissement dans ${REFRESH_SECONDS}s."
  sleep "$REFRESH_SECONDS"
done
