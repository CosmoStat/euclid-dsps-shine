#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
SOURCE_SMOKE_ROOT="${SOURCE_SMOKE_ROOT:?Set SOURCE_SMOKE_ROOT to the immutable smoke root}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CURRICULUM_TAG="${CURRICULUM_TAG:-feniks_adaptive_smc_q_curriculum_$(date +%Y%m%d_%H%M%S)}"
CURRICULUM_ROOT="${CURRICULUM_ROOT:-outputs/runs/$CURRICULUM_TAG}"
MANIFEST_ROOT="$SOURCE_SMOKE_ROOT/manifests"
BOOTSTRAP_CHECKPOINT="$SOURCE_SMOKE_ROOT/train/checkpoints/bootstrap.eqx"

cd "$REPO_DIR"
for path in "$CONFIG" "$CATALOG_DIR/train.parquet" \
  "$MANIFEST_ROOT/train_indices.npy" "$MANIFEST_ROOT/validation_indices.npy" \
  "$BOOTSTRAP_CHECKPOINT"; do
  test -s "$path" || { echo "[feniks-qboot][error] missing: $path" >&2; exit 2; }
done
test ! -e "$CURRICULUM_ROOT" || { echo "output exists: $CURRICULUM_ROOT" >&2; exit 2; }
mkdir -p outputs/logs

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG CATALOG_DIR MANIFEST_ROOT
export BOOTSTRAP_CHECKPOINT CURRICULUM_ROOT
raw=$(sbatch --parsable scripts/feniks_adaptive_smc_q_curriculum_h100.slurm)
CURRICULUM_JOB="${raw%%;*}"
latest=outputs/logs/feniks_adaptive_smc_q_curriculum_latest.env
printf 'export CURRICULUM_JOB=%q\nexport CURRICULUM_ROOT=%q\nexport SOURCE_SMOKE_ROOT=%q\n' \
  "$CURRICULUM_JOB" "$CURRICULUM_ROOT" "$SOURCE_SMOKE_ROOT" > "$latest"
echo "curriculum_job=$CURRICULUM_JOB"
echo "curriculum_root=$CURRICULUM_ROOT"
echo "big_job_not_submitted=1"
echo "latest_env=$latest"
