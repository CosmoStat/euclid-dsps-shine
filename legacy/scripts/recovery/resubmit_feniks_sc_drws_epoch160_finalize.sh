#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
LATEST="${LATEST:-outputs/logs/feniks_sc_drws_epoch160_evaluation_latest.env}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:?Set SCRATCH}/feniks_sc_drws_runtime}"

cd "$REPO_DIR"
test -s "$LATEST" || { echo "missing epoch-160 environment: $LATEST" >&2; exit 2; }
# shellcheck disable=SC1090
source "$LATEST"

test -s "$EVAL_ROOT/CHECKPOINT_FROZEN.json" || {
  echo "missing frozen epoch-160 checkpoint receipt" >&2
  exit 2
}
test ! -e "$EVAL_ROOT/EPOCH160_EVALUATION_COMPLETE.json" || {
  echo "epoch-160 evaluation is already complete" >&2
  exit 2
}
for variant in raw ema; do
  for shard in 0 1 2 3; do
    test -f "$EVAL_ROOT/heldout/$variant/shard_$shard/DONE" || {
      echo "missing held-out DONE: $variant shard $shard" >&2
      exit 2
    }
  done
  for shard in 0 1 2 3 4 5 6 7; do
    test -f "$EVAL_ROOT/catalogue/$variant/shard_$shard/DONE" || {
      echo "missing catalogue DONE: $variant shard $shard" >&2
      exit 2
    }
  done
done

OLD_FINAL_JOB="$FINAL_JOB"
EXPORTS="ALL,REPO_DIR=$REPO_DIR,MINICONDA_PATH=$MINICONDA_PATH,CONDA_ENV=$CONDA_ENV,CATALOG_DIR=$CATALOG_DIR,RECOVERY_ROOT=$RECOVERY_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,CACHE_ROOT=$CACHE_ROOT,EVAL_ROOT=$EVAL_ROOT"
FINAL_RAW=$(sbatch --parsable \
  --output="$LOG_ROOT/epoch160-finalize-%j.out" \
  --error="$LOG_ROOT/epoch160-finalize-%j.err" --export="$EXPORTS" \
  scripts/feniks_sc_drws_epoch160_finalize_h100.slurm)
FINAL_JOB="${FINAL_RAW%%;*}"
ALL_JOBS="$WAIT_JOB,$HELDOUT_JOB,$CATALOGUE_JOB,$FINAL_JOB"

TEMP="${LATEST}.tmp.$$"
printf 'export WAIT_JOB=%q\nexport HELDOUT_JOB=%q\nexport CATALOGUE_JOB=%q\nexport FINAL_JOB=%q\nexport ALL_JOBS=%q\nexport RECOVERY_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport EVAL_ROOT=%q\nexport LOG_ROOT=%q\n' \
  "$WAIT_JOB" "$HELDOUT_JOB" "$CATALOGUE_JOB" "$FINAL_JOB" "$ALL_JOBS" \
  "$RECOVERY_ROOT" "$MANIFEST_ROOT" "$EVAL_ROOT" "$LOG_ROOT" > "$TEMP"
mv "$TEMP" "$LATEST"

echo "old_final_job=$OLD_FINAL_JOB"
echo "final_retry_job=$FINAL_JOB"
echo "reused_heldout_job=$HELDOUT_JOB"
echo "reused_catalogue_job=$CATALOGUE_JOB"
echo "latest_env=$LATEST"
