#!/bin/bash
set -u

REPO_DIR="${REPO_DIR:-$PWD}"
LATEST="${LATEST:-$REPO_DIR/outputs/logs/feniks_rws_recovery_latest.env}"
test -s "$LATEST" || { echo "missing recovery environment: $LATEST" >&2; exit 2; }
# shellcheck disable=SC1090
source "$LATEST"

INTERVAL="${INTERVAL:-60}"
JOBS="$SMOKE_JOB,$PILOT_JOB,$PILOT_GATE_JOB,$CONFIRM_JOB,$FINAL_JOB"
CANDIDATES=(historical_4x128 current_residual_6x256)
SEEDS=(260826 260827)

while true; do
  clear
  date
  echo
  echo "=== SLURM ==="
  squeue -r -j "$JOBS" -o '%.22i %.20j %.2t %.10M %R' 2>/dev/null || true
  echo
  echo "=== PILOT ==="
  for seed_index in 0 1; do
    for candidate_index in 0 1; do
      task=$((seed_index * 2 + candidate_index))
      candidate="${CANDIDATES[$candidate_index]}"
      seed="${SEEDS[$seed_index]}"
      out="$RECOVERY_ROOT/$candidate/seed_$seed"
      summary="$out/pilot_summary.json"
      train_summary="$out/train/training_summary.json"
      if [[ -s "$summary" ]]; then
        python - "$task" "$summary" <<'PY'
import json, sys
p = json.load(open(sys.argv[2]))
iw = p["exact_gaussian_ordinary_iw"]
ppc = p["exact_gaussian_posterior_predictive"]
print(f"task {sys.argv[1]} {p['candidate']} seed={p['seed']}: {p['status']} "
      f"ESS={iw['median_raw_ess_fraction']:.3f} "
      f"k>0.7={iw['fraction_pareto_k_gt_0p7']:.3f} "
      f"PPC_RMS={ppc['median_band_rms']:.2f}")
PY
      elif [[ -s "$train_summary" ]]; then
        echo "task $task $candidate seed=$seed: entraînement fini, évaluation IW/PPC"
      elif [[ -s "$out/train/training_log.csv" ]]; then
        python - "$task" "$candidate" "$seed" "$out/train/training_log.csv" <<'PY'
import pandas as pd, sys
d = pd.read_csv(sys.argv[4])
train = d[d.get("split", "") == "train"] if "split" in d else d
epoch = int(pd.to_numeric(train.get("epoch"), errors="coerce").max())
print(f"task {sys.argv[1]} {sys.argv[2]} seed={sys.argv[3]}: epoch {epoch}/180")
PY
      else
        echo "task $task $candidate seed=$seed: attente/démarrage"
      fi
    done
  done
  [[ -s "$RECOVERY_ROOT/PILOT_PASS.json" ]] && \
    python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("PILOT PASS ->", p["selected_candidate"])' "$RECOVERY_ROOT/PILOT_PASS.json"
  [[ -s "$RECOVERY_ROOT/PILOT_FAIL.json" ]] && echo "PILOT FAIL: arrêt automatique"
  echo
  echo "=== CONFIRMATION ==="
  if [[ -s "$RECOVERY_ROOT/PILOT_PASS.json" ]]; then
    selected=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$RECOVERY_ROOT/PILOT_PASS.json")
    for seed in "${SEEDS[@]}"; do
      summary="$RECOVERY_ROOT/$selected/seed_$seed/confirmation_summary.json"
      if [[ -s "$summary" ]]; then
        python - "$summary" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); iw=p["exact_gaussian_ordinary_iw"]
print(f"seed={p['seed']}: {p['status']} ESS={iw['median_raw_ess_fraction']:.3f} "
      f"k>0.7={iw['fraction_pareto_k_gt_0p7']:.3f}")
PY
      else
        count=$(find "$RECOVERY_ROOT/$selected/seed_$seed/confirmation_exact_gaussian_k2048/posterior_samples" -name '*.parquet' 2>/dev/null | wc -l)
        echo "seed=$seed: $count shards IW écrits"
      fi
    done
  else
    echo "en attente du gate pilote"
  fi
  echo
  echo "=== DECISION ==="
  [[ -s "$RECOVERY_ROOT/RWS_RECOVERY_PASS.json" ]] && echo "RWS RECOVERY PASS: prêt pour benchmark SMC diversité"
  [[ -s "$RECOVERY_ROOT/RWS_RECOVERY_FAIL.json" ]] && echo "RWS RECOVERY FAIL: ne pas lancer SMC/EM"
  echo
  echo "Ctrl-C pour quitter. Rafraîchissement dans ${INTERVAL}s."
  sleep "$INTERVAL"
done
