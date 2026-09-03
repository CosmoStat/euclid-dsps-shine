#!/bin/bash
set -u

ENV_FILE="${1:-outputs/logs/feniks_sc_drws_population_vem_latest.env}"
test -s "$ENV_FILE" || { echo "missing environment: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
INTERVAL="${INTERVAL:-60}"

while true; do
  clear
  date
  echo
  echo "===== FIVE-STAGE POPULATION VEM ====="
  squeue -r -j "$ALL_JOBS" -o "%.18i %.27j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$ALL_JOBS" \
    --format=JobID,JobName%27,State,Elapsed,Timelimit,ExitCode,NodeList \
    2>/dev/null || true

  echo
  echo "===== 1. BANKS + BETA AUDIT ====="
  for item in "q_fit:16" "q_validation:4" "selection_reference:8" "selection_audit:8"; do
    name="${item%%:*}"
    expected="${item##*:}"
    count=$(find "$VEM_ROOT/banks/$name/shards" -mindepth 2 -maxdepth 2 \
      -name COMPLETE.json -type f 2>/dev/null | wc -l)
    printf '%-22s %2d/%-2d shards\n' "$name" "$count" "$expected"
  done
  if [[ -s "$VEM_ROOT/selection_audit/SELECTION_CALIBRATION.json" ]]; then
    python - "$VEM_ROOT/selection_audit/SELECTION_CALIBRATION.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(f"beta audit={x['status']} global={x['global_absolute_error']:.4f} "
      f"ECE={x['expected_calibration_error']:.4f} zmax={x['maximum_redshift_bin_error']:.4f}")
PY
  fi

  echo
  echo "===== 2. PRIOR M-STEP ====="
  if [[ -s "$VEM_ROOT/prior/PROGRESS.json" ]]; then
    python - "$VEM_ROOT/prior/PROGRESS.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); v=x['latest_validation']
print(f"passes={x['completed_passes']}/{x['requested_passes']} best={x['best_pass']} "
      f"objective={v['selected_validation_objective']:.5f} alpha={v['alpha']:.4f}")
print(f"refESS={v['reference_ess_fraction']:.3f} "
      f"q-selected z W1/IQR={v['selected_prior_vs_q_redshift_w1_over_q_iqr']:.4f} "
      f"median15D={v['selected_prior_vs_q_median_w1_over_q_iqr']:.4f}")
PY
  else
    echo "attente"
  fi

  echo
  echo "===== 3. PRIOR-FROZEN AVI REFRESH ====="
  if [[ -s "$VEM_ROOT/q_refresh/training_progress.json" ]]; then
    python - "$VEM_ROOT/q_refresh/training_progress.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(f"epoch={x.get('epoch','?')}/2 best={x.get('best_loss','?')}")
PY
  else
    latest=$(ls -1t "$VEM_LOG_ROOT"/refresh-*.out 2>/dev/null | head -n 1)
    [[ -n "$latest" ]] && tail -n 3 "$latest" || echo "attente"
  fi

  echo
  echo "===== 4. FINAL LOW-DRAW BANKS ====="
  for item in "q_evaluation:8" "prior_evaluation:8"; do
    name="${item%%:*}"
    expected="${item##*:}"
    count=$(find "$VEM_ROOT/banks/$name/shards" -mindepth 2 -maxdepth 2 \
      -name COMPLETE.json -type f 2>/dev/null | wc -l)
    printf '%-22s %2d/%-2d shards\n' "$name" "$count" "$expected"
  done

  echo
  echo "===== 5. FINAL CLOSURE ====="
  if [[ -s "$VEM_ROOT/POPULATION_VEM_COMPLETE.json" ]]; then
    python - "$VEM_ROOT/POPULATION_VEM_COMPLETE.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
print(f"status={x['status']} objects={x['test_objects']} q_draws={x['q_draws_per_object']}")
for name, values in x['metrics'].items():
    print(f"{name}: z W1/IQR={values['redshift_wasserstein_over_target_iqr']:.4f} "
          f"median15D={values['median_wasserstein_over_target_iqr']:.4f}")
print('plots:', x['artifacts']['selected_population_plot'])
PY
  else
    echo "non terminé"
  fi

  echo
  echo "===== RECENT ERRORS ====="
  grep -hE "Traceback|RESOURCE_EXHAUSTED|Out of memory|ValueError|RuntimeError|SystemError" \
    "$VEM_LOG_ROOT"/*.err 2>/dev/null | tail -n 12 || true
  echo
  echo "Ctrl-C stops only this monitor. Refresh in ${INTERVAL}s."
  sleep "$INTERVAL"
done
