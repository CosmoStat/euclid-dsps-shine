#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
RECOVERY_ROOT="${RECOVERY_ROOT:?Set RECOVERY_ROOT}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$RECOVERY_ROOT/manifests}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"
FULL_LOG_ROOT="${LOG_ROOT:-$CACHE_ROOT/slurm_logs/$(basename "$RECOVERY_ROOT")-full}"
FULL_AUTHORIZATION_RECEIPT="${FULL_AUTHORIZATION_RECEIPT:?Set FULL_AUTHORIZATION_RECEIPT}"
AFTER_JOB="${AFTER_JOB:?Set AFTER_JOB to the last queued full worker}"
OLD_GATE_JOB="${OLD_GATE_JOB:-}"

cd "$REPO_DIR"
for path in "$FULL_AUTHORIZATION_RECEIPT" "$MANIFEST_ROOT/manifest.json" \
  "$MANIFEST_ROOT/full_train_indices.npy" \
  "$MANIFEST_ROOT/train_indices.npy" "$MANIFEST_ROOT/validation_indices.npy" \
  "$MANIFEST_ROOT/confirmation_indices.npy" \
  "$MANIFEST_ROOT/final_validation_indices.npy" \
  "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet"; do
  test -s "$path" || { echo "missing full-tail input: $path" >&2; exit 2; }
done
mkdir -p "$FULL_LOG_ROOT" "$CACHE_ROOT/jax" outputs/logs

EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,CACHE_ROOT=$CACHE_ROOT,FULL_AUTHORIZATION_RECEIPT=$FULL_AUTHORIZATION_RECEIPT"
TAIL_RAW=$(sbatch --parsable --dependency="afterany:$AFTER_JOB" --array=0 \
  --output="$FULL_LOG_ROOT/full-tail-%A_%a.out" \
  --error="$FULL_LOG_ROOT/full-tail-%A_%a.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_full_h100.slurm)
TAIL_JOB="${TAIL_RAW%%;*}"
GATE_RAW=$(sbatch --parsable --dependency="afterok:$TAIL_JOB" \
  --output="$FULL_LOG_ROOT/full-tail-gate-%j.out" \
  --error="$FULL_LOG_ROOT/full-tail-gate-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_full_finalize.slurm)
TAIL_GATE_JOB="${GATE_RAW%%;*}"

REPO_DIR="$REPO_DIR" MINICONDA_PATH="$MINICONDA_PATH" CONDA_ENV="$CONDA_ENV" \
CATALOG_DIR="$CATALOG_DIR" RECOVERY_ROOT="$RECOVERY_ROOT" \
MANIFEST_ROOT="$MANIFEST_ROOT" CACHE_ROOT="$CACHE_ROOT" \
AFTER_JOB="$TAIL_GATE_JOB" BASE_LOG_ROOT="$FULL_LOG_ROOT/postfreeze" \
  bash scripts/submit_feniks_sc_drws_postfreeze.sh
source outputs/logs/feniks_sc_drws_postfreeze_latest.env
POSTFREEZE_ALL_JOBS="$ALL_JOBS"

if [[ -n "$OLD_GATE_JOB" ]]; then
  scancel "$OLD_GATE_JOB" 2>/dev/null || true
fi

ALL_JOBS="$AFTER_JOB,$TAIL_JOB,$TAIL_GATE_JOB,$POSTFREEZE_ALL_JOBS"
LATEST=outputs/logs/feniks_sc_drws_full_tail_latest.env
printf 'export UPSTREAM_FULL_JOB=%q\nexport FULL_JOB=%q\nexport FULL_GATE_JOB=%q\nexport POSTFREEZE_ALL_JOBS=%q\nexport ALL_JOBS=%q\nexport RECOVERY_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport LOG_ROOT=%q\nexport FULL_AUTHORIZATION_RECEIPT=%q\n' \
  "$AFTER_JOB" "$TAIL_JOB" "$TAIL_GATE_JOB" "$POSTFREEZE_ALL_JOBS" \
  "$ALL_JOBS" "$RECOVERY_ROOT" "$MANIFEST_ROOT" "$FULL_LOG_ROOT" \
  "$FULL_AUTHORIZATION_RECEIPT" > "$LATEST"

echo "upstream_full_job=$AFTER_JOB"
echo "night_worker=$TAIL_JOB"
echo "night_gate=$TAIL_GATE_JOB"
echo "postfreeze_jobs=$POSTFREEZE_ALL_JOBS"
echo "latest_env=$LATEST"
echo "postfreeze_env=outputs/logs/feniks_sc_drws_postfreeze_latest.env"
