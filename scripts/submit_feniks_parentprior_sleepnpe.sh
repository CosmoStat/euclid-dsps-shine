#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_parentprior_sleepnpe_defensivewake_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
RUN_TAG="${RUN_TAG:-feniks_parentprior_sleepnpe_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/$RUN_TAG}"
TRAIN_ROOT="$RUN_ROOT/train"
MANIFEST_ROOT="$RUN_ROOT/manifests"
SMOKE_ROOT="${RUN_ROOT}_smoke"
SMOKE_TRAIN_ROOT="$SMOKE_ROOT/train"
SMOKE_MANIFEST_ROOT="$SMOKE_ROOT/manifests"

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in "$CONFIG" "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$REFERENCE_CHECKPOINT" "${REFERENCE_CHECKPOINT}.json"; do
  test -s "$path" || { echo "[feniks-sleepnpe-submit][error] missing: $path" >&2; exit 2; }
done
test ! -e "$RUN_ROOT" || {
  echo "[feniks-sleepnpe-submit][error] output exists: $RUN_ROOT" >&2; exit 2;
}
test ! -e "$SMOKE_ROOT" || {
  echo "[feniks-sleepnpe-submit][error] smoke output exists: $SMOKE_ROOT" >&2; exit 2;
}

JAX_PLATFORMS=cpu python scripts/build_feniks_parentprior_r25_manifests.py \
  --train-catalog "$CATALOG_DIR/train.parquet" \
  --test-catalog "$CATALOG_DIR/test.parquet" \
  --out "$MANIFEST_ROOT" \
  --validation-fraction 0.10 \
  --n-exact 32 \
  --seed 260821
JAX_PLATFORMS=cpu python scripts/build_feniks_parentprior_r25_manifests.py \
  --train-catalog "$CATALOG_DIR/train.parquet" \
  --test-catalog "$CATALOG_DIR/test.parquet" \
  --out "$SMOKE_MANIFEST_ROOT" \
  --validation-fraction 0.25 \
  --n-exact 8 \
  --max-selected-train 256 \
  --seed 260821

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG CATALOG_DIR REFERENCE_CHECKPOINT
smoke_raw=$(sbatch --parsable --time=00:45:00 \
  --export="ALL,SMOKE=1,TRAIN_ROOT=$SMOKE_TRAIN_ROOT,MANIFEST_ROOT=$SMOKE_MANIFEST_ROOT" \
  scripts/feniks_parentprior_sleepnpe_h100.slurm)
SMOKE_JOB="${smoke_raw%%;*}"
train_raw=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" \
  --export="ALL,SMOKE=0,TRAIN_ROOT=$TRAIN_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT" \
  scripts/feniks_parentprior_sleepnpe_h100.slurm)
TRAIN_JOB="${train_raw%%;*}"

latest=outputs/logs/feniks_parentprior_sleepnpe_latest.env
printf 'export SMOKE_JOB=%q\nexport TRAIN_JOB=%q\nexport RUN_ROOT=%q\nexport TRAIN_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport CONFIG=%q\nexport CATALOG_DIR=%q\nexport REFERENCE_CHECKPOINT=%q\n' \
  "$SMOKE_JOB" "$TRAIN_JOB" "$RUN_ROOT" "$TRAIN_ROOT" "$MANIFEST_ROOT" \
  "$CONFIG" "$CATALOG_DIR" "$REFERENCE_CHECKPOINT" > "$latest"

echo "smoke_job=$SMOKE_JOB"
echo "train_job=$TRAIN_JOB"
echo "run_root=$RUN_ROOT"
echo "training=one_4xH100_job_after_smoke"
echo "exact_benchmark=not_submitted_run_only_after_training_receipt_passes"
echo "monitor: squeue -r -j $SMOKE_JOB,$TRAIN_JOB"
echo "latest_env=$latest"
