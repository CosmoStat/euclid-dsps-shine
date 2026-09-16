#!/bin/bash
set -u

REFRESH_SECONDS="${REFRESH_SECONDS:-60}"
LATEST="${1:-outputs/logs/feniks_sc_drws_postfreeze_latest.env}"
if [[ ! -s "$LATEST" ]]; then
  echo "missing post-freeze environment: $LATEST" >&2
  exit 2
fi
source "$LATEST"

while true; do
  clear
  date
  echo
  echo "===== POST-FREEZE JOBS ====="
  squeue -r -j "$ALL_JOBS" -o "%.18i %.28j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$ALL_JOBS" \
    --format=JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,NodeList \
    2>/dev/null || true

  echo
  echo "===== PRIOR PARENT / SELECTED ====="
  for variant in raw ema; do
    receipt="$CLOSURE_ROOT/prior_${variant}/report_receipt.json"
    if [[ -s "$receipt" ]]; then
      python - "$variant" "$receipt" <<'PY'
import json, sys
value = json.load(open(sys.argv[2]))
selection = value.get("selection", {})
print(
    f"{sys.argv[1]}: complete "
    f"alpha={selection.get('alpha')} "
    f"relative_error={selection.get('alpha_mc_relative_error')}"
)
PY
    else
      echo "$variant: attente"
    fi
  done

  echo
  echo "===== MIRA / TARP ====="
  for diagnostic in mira tarp; do
    if [[ -f "$CLOSURE_ROOT/$diagnostic/DONE" ]]; then
      echo "$diagnostic: complete"
    elif [[ -d "$CLOSURE_ROOT/$diagnostic" ]]; then
      echo "$diagnostic: en cours"
    else
      echo "$diagnostic: attente"
    fi
  done

  echo
  echo "===== INFÉRENCE CATALOGUE, 4 SHARDS ====="
  for shard in 0 1 2 3; do
    if [[ -f "$INFERENCE_ROOT/shard_${shard}/DONE" ]]; then
      echo "shard $shard: complete"
    elif [[ -d "$INFERENCE_ROOT/shard_${shard}" ]]; then
      echo "shard $shard: en cours"
    else
      echo "shard $shard: attente"
    fi
  done

  echo
  echo "===== RECEIPT FINAL ====="
  if [[ -s "$RECOVERY_ROOT/SC_DRWS_POSTFREEZE_RECEIPT.json" ]]; then
    python - "$RECOVERY_ROOT/SC_DRWS_POSTFREEZE_RECEIPT.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print("status:", value.get("status"))
print("scientific_promotion:", value.get("scientific_promotion"))
PY
  else
    echo "évaluations non terminées"
  fi

  echo
  echo "Ctrl-C arrête seulement ce monitor. Rafraîchissement dans ${REFRESH_SECONDS} s."
  sleep "$REFRESH_SECONDS"
done
