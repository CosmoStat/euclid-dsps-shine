#!/bin/bash
set -u

REFRESH_SECONDS="${1:-60}"
: "${ALL_JOBS:?Source outputs/logs/feniks_sc_drws_epoch160_evaluation_latest.env first}"
: "${EVAL_ROOT:?EVAL_ROOT is required}"
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
  echo "===== CHECKPOINT 160 ====="
  if [[ -s "$EVAL_ROOT/CHECKPOINT_FROZEN.json" ]]; then
    python - "$EVAL_ROOT/CHECKPOINT_FROZEN.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(f"status={r['status']} epoch={r['epoch']} truth_training={r['truth_used_for_training_or_checkpoint_selection']}")
for name, value in r["components"].items():
    print(f"{name}: {value['sha256'][:16]}... {value['path']}")
PY
  else
    echo "en attente du snapshot durable checkpoints/epoch_0160"
    tail -n 2 "$LOG_ROOT"/epoch160-wait-*.out 2>/dev/null || true
  fi

  echo
  echo "===== HELD-OUT K=1024 ====="
  for variant in raw ema; do
    done_count=$(find "$EVAL_ROOT/heldout/$variant" -mindepth 2 -maxdepth 2 \
      -name DONE -type f 2>/dev/null | wc -l)
    echo "$variant: $done_count/4 shards terminés"
    if [[ -s "$EVAL_ROOT/heldout/${variant}_support_summary.json" ]]; then
      python - "$EVAL_ROOT/heldout/${variant}_support_summary.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(
    f"  support={r['status']} ESS/K={r['median_raw_ess_fraction']:.5f} "
    f"ESS={r['median_raw_ess']:.2f} maxw_p90={r['p90_max_raw_weight']:.4f} "
    f"k>0.7={r['fraction_pareto_k_gt_0p7']:.3f}"
)
PY
    fi
  done

  echo
  echo "===== TEST INDÉPENDANT COMPLET APRÈS SÉLECTION, K=256 ====="
  for variant in raw ema; do
    done_count=$(find "$EVAL_ROOT/catalogue/$variant" -mindepth 2 -maxdepth 2 \
      -name DONE -type f 2>/dev/null | wc -l)
    echo "$variant: $done_count/8 shards terminés"
  done
  grep -hE "inference batch|complete variant" \
    "$LOG_ROOT"/epoch160-catalogue-*.out 2>/dev/null | tail -n 8 || true

  echo
  echo "===== ERREURS ====="
  grep -hE "Traceback|RESOURCE_EXHAUSTED|Out of memory|ValueError|RuntimeError|TypeError" \
    "$LOG_ROOT/epoch160-wait-${WAIT_JOB}.err" \
    "$LOG_ROOT"/epoch160-heldout-${HELDOUT_JOB}_*.err \
    "$LOG_ROOT"/epoch160-catalogue-${CATALOGUE_JOB}_*.err \
    "$LOG_ROOT/epoch160-finalize-${FINAL_JOB}.err" \
    2>/dev/null | tail -n 12 || true

  echo
  echo "===== FINAL ====="
  if [[ -s "$EVAL_ROOT/EPOCH160_EVALUATION_COMPLETE.json" ]]; then
    python - "$EVAL_ROOT/EPOCH160_EVALUATION_COMPLETE.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(
    f"status={r['status']} heldout={r['heldout_objects']}x{r['heldout_draws_per_object']} "
    f"catalogue={r['catalogue_objects']}x{r['catalogue_draws_per_object']}"
)
print("MIRA:", r["artifacts"]["mira"]["path"])
print("population:", r["artifacts"]["population_recovery"]["path"])
print("individuals:", r["artifacts"]["individual_panel_manifest"]["path"])
PY
  else
    echo "agrégation population, MIRA/TARP et panneau individuel non terminés"
  fi

  echo
  echo "Ctrl-C arrête seulement ce monitor. Rafraîchissement dans ${REFRESH_SECONDS}s."
  sleep "$REFRESH_SECONDS"
done
