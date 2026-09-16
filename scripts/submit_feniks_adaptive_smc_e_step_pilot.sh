#!/bin/bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
SOURCE_SMOKE_ROOT="${SOURCE_SMOKE_ROOT:?Set SOURCE_SMOKE_ROOT to the failed immutable smoke root}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
PILOT_TAG="${PILOT_TAG:-feniks_adaptive_smc_e_step_pilot_$(date +%Y%m%d_%H%M%S)}"
PILOT_ROOT="${PILOT_ROOT:-outputs/runs/$PILOT_TAG}"
MANIFEST_ROOT="$SOURCE_SMOKE_ROOT/manifests"
BOOTSTRAP_CHECKPOINT="$SOURCE_SMOKE_ROOT/train/checkpoints/bootstrap.eqx"

cd "$REPO_DIR"
for path in "$CONFIG" "$CATALOG_DIR/train.parquet" \
  "$MANIFEST_ROOT/train_indices.npy" "$MANIFEST_ROOT/validation_indices.npy" \
  "$BOOTSTRAP_CHECKPOINT"; do
  test -s "$path" || { echo "[feniks-asmc-pilot][error] missing: $path" >&2; exit 2; }
done
test ! -e "$PILOT_ROOT" || { echo "output exists: $PILOT_ROOT" >&2; exit 2; }
mkdir -p outputs/logs

export REPO_DIR MINICONDA_PATH CONDA_ENV CONFIG CATALOG_DIR MANIFEST_ROOT
export BOOTSTRAP_CHECKPOINT PILOT_ROOT
raw=$(sbatch --parsable scripts/feniks_adaptive_smc_e_step_pilot_h100.slurm)
PILOT_JOB="${raw%%;*}"
latest=outputs/logs/feniks_adaptive_smc_e_step_pilot_latest.env
printf 'export PILOT_JOB=%q\nexport PILOT_ROOT=%q\nexport SOURCE_SMOKE_ROOT=%q\n' \
  "$PILOT_JOB" "$PILOT_ROOT" "$SOURCE_SMOKE_ROOT" > "$latest"
echo "pilot_job=$PILOT_JOB"
echo "pilot_root=$PILOT_ROOT"
echo "big_job_not_submitted=1"
echo "latest_env=$latest"
