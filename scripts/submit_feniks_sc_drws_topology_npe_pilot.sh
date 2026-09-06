#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
SOURCE_ENV="${1:-outputs/logs/feniks_sc_drws_frozen_parent_npe_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
PILOT_ENV="${PILOT_ENV:-outputs/logs/feniks_sc_drws_topology_npe_pilot_latest.env}"

cd "$REPO_DIR"
REPO_DIR="$(pwd -P)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
SOURCE_ENV="$(realpath "$SOURCE_ENV")"
test -s "$SOURCE_ENV" || {
  echo "[topology-npe][error] missing source environment: $SOURCE_ENV" >&2
  exit 2
}
source "$SOURCE_ENV"
SOURCE_NPE_ROOT="$NPE_ROOT"
test -s "$SOURCE_NPE_ROOT/NPE_WINNER_FROZEN.json"
test -s "$SOURCE_NPE_ROOT/RUN_MANIFEST.json"
if ! git diff --quiet --exit-code || ! git diff --cached --quiet --exit-code; then
  echo "[topology-npe][error] tracked source changes must be committed" >&2
  exit 2
fi
CODE_COMMIT="$(git rev-parse HEAD)"
PILOT_ROOT="${PILOT_ROOT:-$RECOVERY_ROOT/frozen_parent_topology_npe_pilot_v1}"
PILOT_LOG_ROOT="${PILOT_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$PILOT_ROOT")}"
mkdir -p "$PILOT_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

python scripts/prepare_feniks_sc_drws_topology_npe_pilot.py \
  --source-root "$SOURCE_NPE_ROOT" --out "$PILOT_ROOT" --repo "$REPO_DIR" \
  --validation-objects 256 --support-objects 128 --seed 260906
test ! -e "$PILOT_ROOT/SUBMISSION.json" || {
  echo "[topology-npe][error] pilot already submitted: $PILOT_ROOT" >&2
  exit 2
}

JOB_REPO_DIR="${TOPOLOGY_NPE_CODE_ROOT:-$CACHE_ROOT/code/topology-npe-${CODE_COMMIT:0:12}}"
mkdir -p "$(dirname "$JOB_REPO_DIR")"
if [[ -e "$JOB_REPO_DIR" ]]; then
  test "$(git -C "$JOB_REPO_DIR" rev-parse HEAD)" = "$CODE_COMMIT"
else
  git worktree add --detach "$JOB_REPO_DIR" "$CODE_COMMIT"
fi
if [[ ! -e "$JOB_REPO_DIR/Data/diffsky" ]]; then
  mkdir -p "$JOB_REPO_DIR/Data"
  ln -s "$REPO_DIR/Data/diffsky" "$JOB_REPO_DIR/Data/diffsky"
fi

COMMON="ALL,REPO_DIR=$JOB_REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,PILOT_ROOT=$PILOT_ROOT,CACHE_ROOT=$CACHE_ROOT"
B_RAW=$(sbatch --parsable \
  --output="$PILOT_LOG_ROOT/train-B-%j.out" \
  --error="$PILOT_LOG_ROOT/train-B-%j.err" \
  --export="$COMMON,NPE_STAGE=B" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_train_h100.slurm")
B_JOB="${B_RAW%%;*}"
VALIDATION_A_RAW=$(sbatch --parsable --array=0 \
  --output="$PILOT_LOG_ROOT/validation-%A_%a.out" \
  --error="$PILOT_LOG_ROOT/validation-%A_%a.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_validate_h100.slurm")
VALIDATION_A_JOB="${VALIDATION_A_RAW%%;*}"
C_RAW=$(sbatch --parsable --dependency="afterok:$B_JOB" \
  --output="$PILOT_LOG_ROOT/train-C-%j.out" \
  --error="$PILOT_LOG_ROOT/train-C-%j.err" \
  --export="$COMMON,NPE_STAGE=C" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_train_h100.slurm")
C_JOB="${C_RAW%%;*}"
VALIDATION_B_RAW=$(sbatch --parsable --dependency="afterok:$B_JOB" --array=1 \
  --output="$PILOT_LOG_ROOT/validation-%A_%a.out" \
  --error="$PILOT_LOG_ROOT/validation-%A_%a.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_validate_h100.slurm")
VALIDATION_B_JOB="${VALIDATION_B_RAW%%;*}"
VALIDATION_C_RAW=$(sbatch --parsable --dependency="afterok:$C_JOB" --array=2 \
  --output="$PILOT_LOG_ROOT/validation-%A_%a.out" \
  --error="$PILOT_LOG_ROOT/validation-%A_%a.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_validate_h100.slurm")
VALIDATION_C_JOB="${VALIDATION_C_RAW%%;*}"
VALIDATION_JOBS="$VALIDATION_A_JOB,$VALIDATION_B_JOB,$VALIDATION_C_JOB"
FINAL_RAW=$(sbatch --parsable \
  --dependency="afterok:$VALIDATION_A_JOB:$VALIDATION_B_JOB:$VALIDATION_C_JOB" \
  --output="$PILOT_LOG_ROOT/finalize-%j.out" \
  --error="$PILOT_LOG_ROOT/finalize-%j.err" \
  --export="$COMMON" \
  "$JOB_REPO_DIR/scripts/feniks_sc_drws_topology_npe_finalize.slurm")
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$B_JOB,$C_JOB,$VALIDATION_JOBS,$FINAL_JOB"

printf 'export B_JOB=%q\nexport C_JOB=%q\nexport VALIDATION_A_JOB=%q\nexport VALIDATION_B_JOB=%q\nexport VALIDATION_C_JOB=%q\nexport VALIDATION_JOBS=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport PILOT_ROOT=%q\nexport PILOT_LOG_ROOT=%q\nexport SOURCE_NPE_ROOT=%q\nexport CACHE_ROOT=%q\nexport JOB_REPO_DIR=%q\nexport CODE_COMMIT=%q\n' \
  "$B_JOB" "$C_JOB" "$VALIDATION_A_JOB" "$VALIDATION_B_JOB" \
  "$VALIDATION_C_JOB" "$VALIDATION_JOBS" "$FINAL_JOB" "$ALL_JOBS" \
  "$PILOT_ROOT" "$PILOT_LOG_ROOT" "$SOURCE_NPE_ROOT" "$CACHE_ROOT" \
  "$JOB_REPO_DIR" "$CODE_COMMIT" > "$PILOT_ENV"

python - "$PILOT_ROOT/SUBMISSION.json" "$ALL_JOBS" "$CODE_COMMIT" "$JOB_REPO_DIR" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "status": "SUBMITTED",
    "jobs": sys.argv[2],
    "runtime_code_commit": sys.argv[3],
    "code_snapshot": sys.argv[4],
    "population_vi_submitted": False,
    "truth_used": False,
    "scientific_promotion": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "arm_B_job=$B_JOB"
echo "arm_C_job=$C_JOB"
echo "validation_A_job=$VALIDATION_A_JOB (concurrent with arm B)"
echo "validation_B_job=$VALIDATION_B_JOB (concurrent with arm C)"
echo "validation_C_job=$VALIDATION_C_JOB"
echo "final_job=$FINAL_JOB"
echo "root=$PILOT_ROOT"
echo "latest_env=$PILOT_ENV"
echo "population VI is not submitted unless POPULATION_VI_READY.json is produced"
echo "monitor: bash scripts/monitor_feniks_sc_drws_topology_npe_pilot.sh $PILOT_ENV 30"
