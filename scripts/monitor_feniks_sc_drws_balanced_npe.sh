#!/bin/bash
set -Eeuo pipefail
source "${1:-outputs/logs/feniks_sc_drws_balanced_npe_latest.env}"
INTERVAL="${2:-30}"
while true; do
  date
  sacct -X -j "$ALL_JOBS" --format=JobID,JobName%25,State,Elapsed,Timelimit,ExitCode
  python scripts/run_feniks_sc_drws_balanced_npe.py monitor --root "$BALANCED_ROOT"
  if [[ -s "$BALANCED_ROOT/BALANCED_NPE_COMPLETE.json" ]]; then
    break
  fi
  echo "Ctrl-C stops only this monitor. Refresh in ${INTERVAL}s."
  sleep "$INTERVAL"
done
