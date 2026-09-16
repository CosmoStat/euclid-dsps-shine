#!/bin/bash
set -Eeuo pipefail

ENV_FILE="${1:-outputs/logs/feniks_sc_drws_topology_npe_pilot_latest.env}"
INTERVAL="${2:-30}"
test -s "$ENV_FILE" || { echo "missing environment: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"

while true; do
  clear 2>/dev/null || true
  date
  echo
  echo "===== TOPOLOGY-CORRECTED FROZEN-PARENT NPE ====="
  echo "root=$PILOT_ROOT"
  squeue -j "$ALL_JOBS" -o "%.18i %.28j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$ALL_JOBS" \
    --format=JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,NodeList || true
  echo
  echo "===== 1. TOPOLOGY REBUILD ====="
  if [[ -s "$PILOT_ROOT/arms/B/topology_initial/TOPOLOGY_CORRECTED_INITIAL_COMPLETE.json" ]]; then
    python - "$PILOT_ROOT/arms/B/topology_initial/TOPOLOGY_CORRECTED_INITIAL_COMPLETE.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print("source counts=", x["source_topology"]["transform_counts"])
print("target counts=", x["target_topology"]["transform_counts"])
print("target min=", x["target_topology"]["minimum_transform_count"], "prior unchanged=", x["prior_bitwise_unchanged"])
PY
  else
    echo "attente"
  fi
  echo
  echo "===== 2-3. B PURE SLEEP / C SLEEP+ELBO ====="
  for arm in B C; do
    receipt="$PILOT_ROOT/arms/$arm/ARM_COMPLETE.json"
    if [[ -s "$receipt" ]]; then
      python - "$receipt" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
b=x["decoder_budget"]
print(f"arm {x['arm']}: COMPLETE best={x['best_checkpoint_value']:.6f} decoder={b['total_evaluations']} prior_unchanged={x['prior_bitwise_unchanged']}")
PY
    else
      echo "arm $arm: attente/en cours"
    fi
  done
  echo
  echo "===== 4. MATCHED TRUTH-FREE VALIDATION ====="
  for arm in A B C; do
    receipt="$PILOT_ROOT/validation/$arm/VALIDATION_COMPLETE.json"
    if [[ -s "$receipt" ]]; then
      python - "$receipt" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
s=x["support"]["support"]
i=x["internal"]
print(f"arm {x['arm']}: support={x['support']['technical_gate']['status']} ESS/K={s['raw_ess']['fraction_median']:.4f} bad-k={s['pareto_k']['gt_0p7_or_nonfinite_fraction']:.3f} heldout={i['held_out_band']['status']} simSBC={i['model_generated_calibration']['status']}")
PY
    else
      echo "arm $arm: attente/en cours"
    fi
  done
  echo
  echo "===== 5. POPULATION GATE ====="
  if [[ -s "$PILOT_ROOT/TOPOLOGY_NPE_PILOT_COMPLETE.json" ]]; then
    python - "$PILOT_ROOT/TOPOLOGY_NPE_PILOT_COMPLETE.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print("status=", x["status"], "winner=", x["winner"])
print("population=", x["population_vi_gate"]["status"])
print("population training started=", x["population_training_started"])
PY
  else
    echo "non terminé"
  fi
  echo
  echo "===== RECENT ERRORS ====="
  grep -h -E "Traceback|Error:|ValueError|RuntimeError|Out of memory" \
    "$PILOT_LOG_ROOT"/*.err 2>/dev/null | tail -n 20 || true
  if [[ -s "$PILOT_ROOT/TOPOLOGY_NPE_PILOT_COMPLETE.json" ]]; then
    echo
    echo "Final receipt exists. Ctrl-C stops only this monitor."
  else
    echo
    echo "Ctrl-C stops only this monitor. Refresh in ${INTERVAL}s."
  fi
  sleep "$INTERVAL"
done
