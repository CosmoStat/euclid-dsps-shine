#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
RUN_TAG="${RUN_TAG:-feniks_architecture20k_$(date +%Y%m%d_%H%M%S)}"
ARCHITECTURE_ROOT="${ARCHITECTURE_ROOT:-outputs/runs/$RUN_TAG}"
SMOKE_ROOT="${ARCHITECTURE_ROOT}_smoke"
MANIFEST_ROOT="$ARCHITECTURE_ROOT/manifests"
SMOKE_MANIFEST_ROOT="$SMOKE_ROOT/manifests"
TRAIN_CATALOG="$CATALOG_DIR/train.parquet"
TEST_CATALOG="$CATALOG_DIR/test.parquet"
CONFIGS=(
  configs/experiments/feniks_architecture_20k_current_realnvp.yaml
  configs/experiments/feniks_architecture_20k_set_realnvp.yaml
  configs/experiments/feniks_architecture_20k_set_autoregressive_spline.yaml
)

cd "$REPO_DIR"
mkdir -p outputs/logs
for path in "$TRAIN_CATALOG" "$TEST_CATALOG" "$REFERENCE_CHECKPOINT" \
  "${REFERENCE_CHECKPOINT}.json" "${CONFIGS[@]}"; do
  test -s "$path" || { echo "[feniks-architecture-submit][error] missing: $path" >&2; exit 2; }
done
test ! -e "$ARCHITECTURE_ROOT" || {
  echo "[feniks-architecture-submit][error] output exists: $ARCHITECTURE_ROOT" >&2; exit 2;
}
test ! -e "$SMOKE_ROOT" || {
  echo "[feniks-architecture-submit][error] smoke output exists: $SMOKE_ROOT" >&2; exit 2;
}

JAX_PLATFORMS=cpu python scripts/build_feniks_architecture_20k_manifests.py \
  --train-catalog "$TRAIN_CATALOG" --test-catalog "$TEST_CATALOG" \
  --out "$MANIFEST_ROOT" --n-train 18000 --n-validation 2000 \
  --n-probe 256 --seed 260820
JAX_PLATFORMS=cpu python scripts/build_feniks_architecture_20k_manifests.py \
  --train-catalog "$TRAIN_CATALOG" --test-catalog "$TEST_CATALOG" \
  --out "$SMOKE_MANIFEST_ROOT" --n-train 192 --n-validation 64 \
  --n-probe 8 --seed 260820

export REPO_DIR MINICONDA_PATH CONDA_ENV CATALOG_DIR REFERENCE_CHECKPOINT
smoke_raw=$(sbatch --parsable --array=0-2%3 --time=00:30:00 \
  --export="ALL,SMOKE=1,ARCHITECTURE_ROOT=$SMOKE_ROOT,MANIFEST_ROOT=$SMOKE_MANIFEST_ROOT" \
  scripts/feniks_architecture_20k_h100.slurm)
SMOKE_JOB="${smoke_raw%%;*}"
full_raw=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" \
  --export="ALL,SMOKE=0,ARCHITECTURE_ROOT=$ARCHITECTURE_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT" \
  scripts/feniks_architecture_20k_h100.slurm)
ARCHITECTURE_JOB="${full_raw%%;*}"
final_raw=$(sbatch --parsable --dependency="afterok:${ARCHITECTURE_JOB}" \
  --export="ALL,ARCHITECTURE_ROOT=$ARCHITECTURE_ROOT" \
  scripts/feniks_architecture_20k_finalize.slurm)
ARCHITECTURE_FINALIZER_JOB="${final_raw%%;*}"

latest=outputs/logs/feniks_architecture_20k_latest.env
printf 'export SMOKE_JOB=%q\nexport ARCHITECTURE_JOB=%q\nexport ARCHITECTURE_FINALIZER_JOB=%q\nexport ARCHITECTURE_ROOT=%q\nexport MANIFEST_ROOT=%q\nexport CATALOG_DIR=%q\nexport REFERENCE_CHECKPOINT=%q\n' \
  "$SMOKE_JOB" "$ARCHITECTURE_JOB" "$ARCHITECTURE_FINALIZER_JOB" \
  "$ARCHITECTURE_ROOT" "$MANIFEST_ROOT" "$CATALOG_DIR" \
  "$REFERENCE_CHECKPOINT" > "$latest"

echo "smoke_job=$SMOKE_JOB"
echo "architecture_job=$ARCHITECTURE_JOB"
echo "architecture_finalizer_job=$ARCHITECTURE_FINALIZER_JOB"
echo "architecture_root=$ARCHITECTURE_ROOT"
echo "train=18000 validation=2000 blind_iw_probe=256 seeds=260820,260821"
echo "gpus=6_tasks_x_4_h100 max_concurrent=24"
echo "monitor: squeue -r -j $SMOKE_JOB,$ARCHITECTURE_JOB,$ARCHITECTURE_FINALIZER_JOB"
echo "latest_env=$latest"
