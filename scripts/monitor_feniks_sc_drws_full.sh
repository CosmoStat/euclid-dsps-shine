#!/bin/bash
set -u

REPO_DIR="${REPO_DIR:-$PWD}"
LATEST="${LATEST:-$REPO_DIR/outputs/logs/feniks_sc_drws_full_latest.env}"
test -s "$LATEST" || { echo "missing full-run environment: $LATEST" >&2; exit 2; }
# shellcheck disable=SC1090
source "$LATEST"

INTERVAL="${INTERVAL:-60}"
SELECTED=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' \
  "$RECOVERY_ROOT/RWS_RECOVERY_PASS.json")
SEEDS=(260826)

while true; do
  clear
  date
  echo
  echo "=== SLURM ==="
  squeue -r -j "$FULL_JOB,$FULL_GATE_JOB" \
    -o '%.22i %.24j %.2t %.10M %R' 2>/dev/null || true
  sacct -X -j "$FULL_JOB,$FULL_GATE_JOB" \
    --format=JobID,State,Elapsed,Timelimit,ExitCode,NodeList 2>/dev/null || true
  echo
  echo "=== SC-DRWS FULL ==="
  for seed in "${SEEDS[@]}"; do
    train="$RECOVERY_ROOT/full/$SELECTED/seed_$seed/train"
    log="$train/sc_drws_training_log.csv"
    support="$train/sc_drws_support_probe_log.csv"
    receipt="$train/training_receipt.json"
    echo "--- $SELECTED seed=$seed ---"
    if [[ -s "$receipt" ]]; then
      python - "$receipt" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
safety = r.get("checkpoint_safety", {})
print(
    "training complete",
    f"prior_updates={r['prior_updates']}",
    f"restored_epoch={safety.get('restored_best_support_epoch')}",
)
PY
    elif [[ -s "$log" ]]; then
      python - "$log" "$support" <<'PY'
import sys
from pathlib import Path
import pandas as pd

d = pd.read_csv(sys.argv[1])
last = d.iloc[-1]
wake = d[d["update_kind"] == "wake"]
print(
    f"epoch={int(last['epoch'])}/180 kind={last['update_kind']}",
    f"lr={last.get('q_learning_rate', float('nan')):.3g}",
    f"flow_mult={last.get('flow_gradient_multiplier', float('nan')):.3f}",
)
if len(wake):
    row = wake.iloc[-1]
    print(
        f"wake ESS1={row['first_pass_ess_fraction']:.5f}",
        f"ESSx={row['expanded_ess_fraction']:.5f}",
        f"maxw={row['max_weight']:.5f}",
        f"tau={row['q_weight_temperature']:.3f}",
        f"expanded={row['expansion_fraction']:.3f}",
        f"unresolved={row['unresolved_fraction']:.3f}",
    )
support = Path(sys.argv[2])
if support.is_file():
    probe = pd.read_csv(support).iloc[-1]
    print(
        f"probe epoch={int(probe['epoch'])}",
        f"ESS={probe['median_ess_fraction']:.5f}",
        f"maxw={probe['median_max_weight']:.5f}",
        f"action={probe['action']}",
    )
PY
    else
      echo "attente/demarrage"
    fi
  done
  echo
  echo "Ctrl-C pour quitter. Rafraichissement dans ${INTERVAL}s."
  sleep "$INTERVAL"
done
