#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Set RECOVERY_ROOT}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RECOVERY_ROOT/manifests}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
AFTER_JOB="${AFTER_JOB:?Set AFTER_JOB to the final full gate job}"
CLOSURE_ROOT="${CLOSURE_ROOT:-$RECOVERY_ROOT/postfreeze_closure}"
INFERENCE_ROOT="${INFERENCE_ROOT:-$RECOVERY_ROOT/final_inference}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$RECOVERY_ROOT")-postfreeze}"

cd "$REPO_DIR"
for path in "$MANIFEST_ROOT/manifest.json" \
  "$MANIFEST_ROOT/full_train_indices.npy" "$MANIFEST_ROOT/train_indices.npy" \
  "$MANIFEST_ROOT/validation_indices.npy" \
  "$MANIFEST_ROOT/final_validation_indices.npy" \
  "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet"; do
  test -s "$path" || { echo "missing post-freeze input: $path" >&2; exit 2; }
done
test ! -e "$CLOSURE_ROOT" || { echo "immutable closure output exists: $CLOSURE_ROOT" >&2; exit 2; }
test ! -e "$INFERENCE_ROOT" || { echo "immutable inference output exists: $INFERENCE_ROOT" >&2; exit 2; }
mkdir -p "$BASE_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

COMMON_EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,CACHE_ROOT=$CACHE_ROOT"
CLOSURE_RAW=$(sbatch --parsable --dependency="afterok:$AFTER_JOB" \
  --output="$BASE_LOG_ROOT/closure-%j.out" \
  --error="$BASE_LOG_ROOT/closure-%j.err" \
  --export="$COMMON_EXPORTS,CLOSURE_ROOT=$CLOSURE_ROOT" \
  scripts/feniks_sc_drws_postfreeze_h100.slurm)
CLOSURE_JOB="${CLOSURE_RAW%%;*}"

REPO_DIR="$REPO_DIR" MINICONDA_PATH="$MINICONDA_PATH" CONDA_ENV="$CONDA_ENV" \
CATALOG_DIR="$CATALOG_DIR" RECOVERY_ROOT="$RECOVERY_ROOT" \
MANIFEST_ROOT="$MANIFEST_ROOT" CACHE_ROOT="$CACHE_ROOT" \
AFTER_JOB="$AFTER_JOB" ALLOW_DIAGNOSTIC_FULL=1 \
INFERENCE_ROOT="$INFERENCE_ROOT" LOG_ROOT="$BASE_LOG_ROOT/inference" \
  bash scripts/submit_feniks_sc_drws_inference.sh
source outputs/logs/feniks_sc_drws_inference_latest.env

FINAL_RAW=$(sbatch --parsable \
  --dependency="afterok:$CLOSURE_JOB:$INFERENCE_GATE_JOB" \
  --output="$BASE_LOG_ROOT/postfreeze-gate-%j.out" \
  --error="$BASE_LOG_ROOT/postfreeze-gate-%j.err" \
  --export="$COMMON_EXPORTS,CLOSURE_ROOT=$CLOSURE_ROOT,INFERENCE_ROOT=$INFERENCE_ROOT" \
  scripts/feniks_sc_drws_postfreeze_finalize.slurm)
POSTFREEZE_GATE_JOB="${FINAL_RAW%%;*}"

LATEST=outputs/logs/feniks_sc_drws_postfreeze_latest.env
ALL_JOBS="$AFTER_JOB,$CLOSURE_JOB,$INFERENCE_JOB,$INFERENCE_GATE_JOB,$POSTFREEZE_GATE_JOB"
printf 'export UPSTREAM_FULL_GATE_JOB=%q\nexport CLOSURE_JOB=%q\nexport INFERENCE_JOB=%q\nexport INFERENCE_GATE_JOB=%q\nexport POSTFREEZE_GATE_JOB=%q\nexport ALL_JOBS=%q\nexport RECOVERY_ROOT=%q\nexport CLOSURE_ROOT=%q\nexport INFERENCE_ROOT=%q\nexport LOG_ROOT=%q\n' \
  "$AFTER_JOB" "$CLOSURE_JOB" "$INFERENCE_JOB" "$INFERENCE_GATE_JOB" \
  "$POSTFREEZE_GATE_JOB" "$ALL_JOBS" "$RECOVERY_ROOT" "$CLOSURE_ROOT" \
  "$INFERENCE_ROOT" "$BASE_LOG_ROOT" > "$LATEST"

echo "closure_job=$CLOSURE_JOB"
echo "inference_job=$INFERENCE_JOB"
echo "inference_gate_job=$INFERENCE_GATE_JOB"
echo "postfreeze_gate_job=$POSTFREEZE_GATE_JOB"
echo "latest_env=$LATEST"
