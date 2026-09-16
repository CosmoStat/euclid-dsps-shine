#!/bin/bash
set -u

ENV_FILE="${1:-outputs/logs/feniks_sc_drws_frozen_parent_npe_latest.env}"
INTERVAL="${2:-30}"
test -s "$ENV_FILE" || { echo "missing environment: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH or CACHE_ROOT}/feniks_sc_drws_runtime}"
test -s "$BASELINE_ENV" || { echo "missing baseline environment: $BASELINE_ENV" >&2; exit 2; }
INITIAL_NPE_JOBS="$ALL_JOBS"
source "$BASELINE_ENV"
BASELINE_JOBS="$ALL_JOBS"
# Reload the NPE variables after the baseline environment reused ALL_JOBS.
source "$ENV_FILE"
INITIAL_NPE_JOBS="$ALL_JOBS"

while true; do
  clear
  date
  echo
  echo "===== FROZEN-PARENT NPE: FIVE STAGES ====="
  JOBS="$BASELINE_JOBS,$INITIAL_NPE_JOBS"
  if [[ -s "$NPE_ROOT/downstream.env" ]]; then
    source "$NPE_ROOT/downstream.env"
    JOBS="$JOBS,$ALL_DOWNSTREAM_JOBS"
  fi
  squeue -r -j "$JOBS" -o "%.18i %.28j %.2t %.10M %R" 2>/dev/null || true
  sacct -X -j "$JOBS" \
    --format=JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,NodeList \
    2>/dev/null || true

  echo
  echo "===== 1. PROVENANCE / SUPPORT PREFLIGHT ====="
  python - "$NPE_ROOT" <<'PY' 2>/dev/null || echo "attente"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'STAGE1_PASS.json'
x=json.load(open(p))
print(f"status={x['status']} train={x['train_objects']} validation={x['validation_objects']} truth_used={x['truth_used']}")
PY

  echo
  echo "===== 2. CURRENT-Q INDEPENDENT-TEST BASELINE ====="
  printf 'full K256:    %s/32 shards\n' "$(find "$BASELINE_FULL_ROOT/shards" -name SHARD_COMPLETE.json 2>/dev/null | wc -l)"
  printf 'support K1024: %s/32 shards\n' "$(find "$BASELINE_SUPPORT_ROOT/shards" -name SHARD_COMPLETE.json 2>/dev/null | wc -l)"
  python - "$BASELINE_FULL_ROOT" "$BASELINE_SUPPORT_ROOT" <<'PY' 2>/dev/null || true
import json,sys
from pathlib import Path
for label,raw in zip(('full','support'),sys.argv[1:]):
    p=Path(raw)/'INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json'
    if p.is_file():
        x=json.load(open(p)); q=x['redshift_calibration']['q']; s=x['projected_parent_support']
        print(f"{label}: q PIT={q['pit_ks_uniform']:.4f} ECE={q['coverage_ece']:.4f} ESS={s['median_raw_ess']:.2f} k>0.7={s['fraction_pareto_k_gt_0p7']:.3f}")
PY

  echo
  echo "===== 3. PARALLEL PURE-SLEEP NPE ARMS ====="
  for arm in warm_start scratch_encoder; do
    python - "$NPE_ROOT" "$arm" <<'PY' 2>/dev/null || echo "$arm: attente/en cours"
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); arm=sys.argv[2]
receipt=root/'arms'/arm/'ARM_COMPLETE.json'
summary=root/'arms'/arm/'train'/'training_summary.json'
progress=root/'arms'/arm/'train'/'training_progress.json'
if receipt.is_file():
    x=json.load(open(receipt)); print(f"{arm}: {x['status']} validation_sleep_nll={x['validation_sleep_nll']:.6f}")
elif summary.is_file():
    x=json.load(open(summary)); print(f"{arm}: complete, gate pending best={x['best_loss']:.6f}")
elif progress.is_file():
    x=json.load(open(progress)); print(f"{arm}: epoch={x.get('epoch','?')} running")
else:
    raise SystemExit(1)
PY
  done
  python - "$NPE_ROOT" <<'PY' 2>/dev/null || true
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'NPE_WINNER_FROZEN.json'
if p.is_file():
    x=json.load(open(p)); print(f"winner={x['selected_arm']} validation_sleep_nll={x['validation_sleep_nll']:.6f} prior_unchanged={x['prior_bitwise_unchanged']}")
PY

  echo
  echo "===== 4. MATCHED WINNER EVALUATION ====="
  if [[ -s "$NPE_ROOT/downstream.env" ]]; then
    printf 'full K256:     %s/32 shards\n' "$(find "$NPE_FULL_ROOT/shards" -name SHARD_COMPLETE.json 2>/dev/null | wc -l)"
    printf 'support K1024: %s/32 shards\n' "$(find "$NPE_SUPPORT_ROOT/shards" -name SHARD_COMPLETE.json 2>/dev/null | wc -l)"
  else
    echo "attente du gagnant NPE figé"
  fi

  echo
  echo "===== 5. FROZEN POSTERIOR / POPULATION CLOSURE ====="
  FINAL="$NPE_ROOT/FROZEN_NPE_EXPERIMENT_COMPLETE.json"
  if [[ -s "$FINAL" ]]; then
    python - "$FINAL" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); b=x['baseline']; n=x['sleep_npe']; d=x['delta_sleep_npe_minus_baseline']
print(f"status={x['status']} winner={x['winner']['arm']}")
print(f"q PIT: {b['redshift_q_calibration']['pit_ks_uniform']:.4f} -> {n['redshift_q_calibration']['pit_ks_uniform']:.4f} delta={d['pit_ks_uniform']:+.4f}")
print(f"q ECE: {b['redshift_q_calibration']['coverage_ece']:.4f} -> {n['redshift_q_calibration']['coverage_ece']:.4f} delta={d['coverage_ece']:+.4f}")
print(f"K1024 ESS: {b['k1024_projected_parent_support']['median_raw_ess']:.2f} -> {n['k1024_projected_parent_support']['median_raw_ess']:.2f} delta={d['median_raw_ess']:+.2f}")
print(f"K1024 k>0.7: {b['k1024_projected_parent_support']['fraction_pareto_k_gt_0p7']:.3f} -> {n['k1024_projected_parent_support']['fraction_pareto_k_gt_0p7']:.3f} delta={d['fraction_pareto_k_gt_0p7']:+.3f}")
print(f"plots: {x['artifacts']['comparison_plot']}")
PY
  else
    echo "non terminé"
  fi

  echo
  echo "===== RECENT ERRORS ====="
  grep -R -h -E 'Traceback \(most recent call last\)|CUDA out of memory|RESOURCE_EXHAUSTED|Killed' \
    "$NPE_LOG_ROOT" "$CACHE_ROOT/slurm_logs/$(basename "$BASELINE_ROOT")-full" \
    "$CACHE_ROOT/slurm_logs/$(basename "$BASELINE_ROOT")-support" \
    2>/dev/null | tail -n 12 || true

  if [[ -s "$FINAL" ]]; then
    echo
    echo "Final receipt exists. Ctrl-C stops only this monitor."
  else
    echo
    echo "Ctrl-C stops only this monitor. Refresh in ${INTERVAL}s."
  fi
  sleep "$INTERVAL"
done
