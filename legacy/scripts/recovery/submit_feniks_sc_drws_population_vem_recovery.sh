#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
LATEST="${VEM_ENV:-$REPO_DIR/outputs/logs/feniks_sc_drws_population_vem_latest.env}"
test -s "$LATEST" || { echo "[population-vem-recovery][error] missing: $LATEST" >&2; exit 2; }
# shellcheck disable=SC1090
source "$LATEST"

MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
test "${RECOVER_FAILED_CHAIN:-0}" = 1 || {
  echo "[population-vem-recovery][error] set RECOVER_FAILED_CHAIN=1" >&2
  exit 2
}

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
RECOVERY_CODE_COMMIT="$(git rev-parse HEAD)"
test -d "$JOB_REPO_DIR" || { echo "[population-vem-recovery][error] missing code snapshot" >&2; exit 2; }
ACTUAL_COMMIT="$(git -C "$JOB_REPO_DIR" rev-parse HEAD)"
test "$ACTUAL_COMMIT" = "$CODE_COMMIT" || {
  echo "[population-vem-recovery][error] frozen code commit mismatch" >&2
  exit 2
}
test ! -e "$VEM_ROOT/STAGE1_PASS.json" || {
  echo "[population-vem-recovery][error] stage 1 already has a receipt" >&2
  exit 2
}
if [[ -e "$VEM_ROOT/RECOVERY_SUBMISSION.json" ]]; then
  RECOVERY_HISTORY="$VEM_ROOT/recovery_history"
  mkdir -p "$RECOVERY_HISTORY"
  cp "$VEM_ROOT/RECOVERY_SUBMISSION.json" \
    "$RECOVERY_HISTORY/failed-gate-${BANK_GATE_JOB}.json"
fi

for specification in q_fit:16 q_validation:4 selection_reference:8 selection_audit:8; do
  bank="${specification%%:*}"
  expected="${specification##*:}"
  actual="$(find "$VEM_ROOT/banks/$bank/shards" -mindepth 2 -maxdepth 2 -name COMPLETE.json -type f | wc -l)"
  test "$actual" -eq "$expected" || {
    echo "[population-vem-recovery][error] $bank markers: expected=$expected actual=$actual" >&2
    exit 2
  }
done

scancel "$PRIOR_JOB" "$REFRESH_JOB" "$EVAL_JOB" "$FINAL_JOB" 2>/dev/null || true
EXPORTS="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,VEM_ROOT=$VEM_ROOT,CACHE_ROOT=$CACHE_ROOT"
GATE_EXPORTS="$EXPORTS,VEM_RUNTIME_PYTHONPATH=$REPO_DIR"

GATE_RAW=$(sbatch --parsable \
  --output="$VEM_LOG_ROOT/bank-gate-recovery-%j.out" \
  --error="$VEM_LOG_ROOT/bank-gate-recovery-%j.err" --export="$GATE_EXPORTS" \
  scripts/feniks_sc_drws_population_vem_bank_finalize.slurm)
NEW_BANK_GATE_JOB="${GATE_RAW%%;*}"
PRIOR_RAW=$(sbatch --parsable --dependency="afterok:$NEW_BANK_GATE_JOB" \
  --output="$VEM_LOG_ROOT/prior-recovery-%j.out" \
  --error="$VEM_LOG_ROOT/prior-recovery-%j.err" --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_vem_prior_h100.slurm")
NEW_PRIOR_JOB="${PRIOR_RAW%%;*}"
REFRESH_RAW=$(sbatch --parsable --dependency="afterok:$NEW_PRIOR_JOB" \
  --output="$VEM_LOG_ROOT/refresh-recovery-%j.out" \
  --error="$VEM_LOG_ROOT/refresh-recovery-%j.err" --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_vem_refresh_h100.slurm")
NEW_REFRESH_JOB="${REFRESH_RAW%%;*}"
EVAL_RAW=$(sbatch --parsable --dependency="afterok:$NEW_REFRESH_JOB" \
  --array=0-15%16 --output="$VEM_LOG_ROOT/eval-recovery-%A_%a.out" \
  --error="$VEM_LOG_ROOT/eval-recovery-%A_%a.err" \
  --export="$EXPORTS,VEM_STAGE=final" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_vem_bank_h100.slurm")
NEW_EVAL_JOB="${EVAL_RAW%%;*}"
FINAL_RAW=$(sbatch --parsable --dependency="afterok:$NEW_EVAL_JOB" \
  --output="$VEM_LOG_ROOT/final-recovery-%j.out" \
  --error="$VEM_LOG_ROOT/final-recovery-%j.err" --export="$EXPORTS" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_population_vem_finalize_h100.slurm")
NEW_FINAL_JOB="${FINAL_RAW%%;*}"
NEW_ALL_JOBS="$NEW_BANK_GATE_JOB,$NEW_PRIOR_JOB,$NEW_REFRESH_JOB,$NEW_EVAL_JOB,$NEW_FINAL_JOB"

RECOVERY_ENV="$REPO_DIR/outputs/logs/feniks_sc_drws_population_vem_recovery_latest.env"
printf 'export BANK_JOB=%q\nexport BANK_GATE_JOB=%q\nexport PRIOR_JOB=%q\nexport REFRESH_JOB=%q\nexport EVAL_JOB=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport VEM_ROOT=%q\nexport VEM_LOG_ROOT=%q\nexport RECOVERY_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$BANK_JOB" "$NEW_BANK_GATE_JOB" "$NEW_PRIOR_JOB" "$NEW_REFRESH_JOB" \
  "$NEW_EVAL_JOB" "$NEW_FINAL_JOB" "$NEW_ALL_JOBS" "$VEM_ROOT" \
  "$VEM_LOG_ROOT" "$RECOVERY_ROOT" "$JOB_REPO_DIR" "$CODE_COMMIT" > "$RECOVERY_ENV"
cp "$RECOVERY_ENV" "$LATEST"

python - "$VEM_ROOT/RECOVERY_SUBMISSION.json" "$BANK_GATE_JOB" \
  "$NEW_BANK_GATE_JOB" "$NEW_PRIOR_JOB" "$NEW_REFRESH_JOB" \
  "$NEW_EVAL_JOB" "$NEW_FINAL_JOB" "$NEW_ALL_JOBS" "$CODE_COMMIT" \
  "$RECOVERY_CODE_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": "SUBMITTED",
    "failed_bank_gate_job": sys.argv[2],
    "bank_gate_job": sys.argv[3],
    "prior_job": sys.argv[4],
    "refresh_job": sys.argv[5],
    "evaluation_job": sys.argv[6],
    "final_job": sys.argv[7],
    "all_jobs": sys.argv[8],
    "bank_code_commit": sys.argv[9],
    "recovery_code_commit": sys.argv[10],
    "reused_initial_banks": True,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "reused_bank_job=$BANK_JOB"
echo "bank_gate_job=$NEW_BANK_GATE_JOB"
echo "prior_job=$NEW_PRIOR_JOB"
echo "refresh_job=$NEW_REFRESH_JOB"
echo "evaluation_job=$NEW_EVAL_JOB"
echo "final_job=$NEW_FINAL_JOB"
echo "latest_env=$LATEST"
echo "recovery_env=$RECOVERY_ENV"
echo "monitor: bash scripts/monitor_feniks_sc_drws_population_vem.sh"
