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
SMOKE_ROOT="${SMOKE_ROOT:-${RECOVERY_ROOT}_smoke}"

show_candidate_tasks() {
  local root="$1"
  local total_epochs="$2"
  for seed_index in 0 1; do
    for candidate_index in 0 1; do
      task=$((seed_index * 2 + candidate_index))
      candidate="${CANDIDATES[$candidate_index]}"
      seed="${SEEDS[$seed_index]}"
      out="$root/$candidate/seed_$seed"
      summary="$out/pilot_summary.json"
      receipt="$out/train/training_receipt.json"
      log="$out/train/sc_drws_training_log.csv"
      if [[ -s "$summary" ]]; then
        python - "$task" "$summary" <<'PY'
import json, sys
p = json.load(open(sys.argv[2]))
iw = p["exact_gaussian_ordinary_iw"]
print(f"task {sys.argv[1]} {p['candidate']} seed={p['seed']}: {p['status']} "
      f"ESS={iw['median_raw_ess_fraction']:.3f} "
      f"k>0.7={iw['fraction_pareto_k_gt_0p7']:.3f}")
PY
      elif [[ -s "$receipt" ]]; then
        echo "task $task $candidate seed=$seed: entraînement fini, évaluation IW/PPC"
      elif [[ -s "$log" ]]; then
        python - "$task" "$candidate" "$seed" "$log" "$total_epochs" <<'PY'
import pandas as pd, sys
d = pd.read_csv(sys.argv[4])
epoch = int(pd.to_numeric(d["epoch"], errors="coerce").max())
last = d.iloc[-1]
kind = last.get("update_kind", last.get("phase", "unknown"))
print(f"task {sys.argv[1]} {sys.argv[2]} seed={sys.argv[3]}: "
      f"epoch {epoch}/{sys.argv[5]} update={kind}")
PY
      else
        echo "task $task $candidate seed=$seed: attente/démarrage"
      fi
    done
  done
}

while true; do
  clear
  date
  echo
  echo "=== SLURM ==="
  squeue -r -j "$JOBS" -o '%.22i %.20j %.2t %.10M %R' 2>/dev/null || true
  echo
  echo "=== SMOKE ==="
  show_candidate_tasks "$SMOKE_ROOT" 8
  echo
  echo "=== PILOT ==="
  show_candidate_tasks "$RECOVERY_ROOT" 180
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
  [[ -s "$RECOVERY_ROOT/RWS_RECOVERY_PASS.json" ]] && echo "SC-DRWS CONFIRMATION PASS: full catalogue autorisé mais non soumis"
  [[ -s "$RECOVERY_ROOT/RWS_RECOVERY_FAIL.json" ]] && echo "SC-DRWS CONFIRMATION FAIL: ne pas lancer le full"
  echo
  echo "Ctrl-C pour quitter. Rafraîchissement dans ${INTERVAL}s."
  sleep "$INTERVAL"
done
