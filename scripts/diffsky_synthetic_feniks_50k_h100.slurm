#!/bin/bash
#SBATCH --job-name=feniks50k
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=16:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

# Generate and validate the 50k Diffsky/FENIKS DSPS-closure dataset on H100.
# The generator is verbose by default and production diagnostics keep plots on.
#
# Typical production launch:
#
#   GEN_JOB=$(sbatch --parsable --export=ALL,STAGE=generate,OVERWRITE=1,RESUME=0 \
#     scripts/diffsky_synthetic_feniks_50k_h100.slurm)
#   sbatch --dependency=afterok:${GEN_JOB} --export=ALL,STAGE=validate \
#     scripts/diffsky_synthetic_feniks_50k_h100.slurm
#
# To resume an interrupted full run after deleting/replacing no final parquets:
#
#   sbatch --export=ALL,STAGE=generate,OVERWRITE=0,RESUME=1 \
#     scripts/diffsky_synthetic_feniks_50k_h100.slurm

if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/euclid_dsps" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  elif [[ -n "${WORK:-}" ]]; then
    REPO_DIR="$WORK/euclid-dsps-shine"
  else
    REPO_DIR="$PWD"
  fi
fi
if [[ -z "${MINICONDA_PATH:-}" ]]; then
  if [[ -n "${WORK:-}" && -d "$WORK/miniconda3" ]]; then
    MINICONDA_PATH="$WORK/miniconda3"
  else
    MINICONDA_PATH="$(dirname "$REPO_DIR")/miniconda3"
  fi
fi

CONDA_ENV="${CONDA_ENV:-shine}"
if [[ -z "${DIFFSKY_REPO:-}" && -n "${WORK:-}" && -d "$WORK/diffsky/diffsky" ]]; then
  DIFFSKY_REPO="$WORK/diffsky"
fi
STAGE="${STAGE:-generate}"
CONFIG="${CONFIG:-configs/diffsky_synthetic_feniks_260617_50k.yaml}"
VALIDATION_CONFIG="${VALIDATION_CONFIG:-configs/diffsky_synthetic_feniks_260617_trueparam_closure.yaml}"
DATASET_DIR="${DATASET_DIR:-Data/diffsky/synthetic/feniks_260617_dsps_closure}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/diffsky_synthetic_feniks_50k_${STAGE}_${SLURM_JOB_ID:-manual}.log}"

SPLIT="${SPLIT:-all}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"
SMOKE="${SMOKE:-0}"
MAX_GALAXIES="${MAX_GALAXIES:-}"
SAMPLE_SIZE="${SAMPLE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VALIDATION_RUNTIME="${VALIDATION_RUNTIME:-gpu}"

RAW_ROOT="${RAW_ROOT:-Data/diffsky/raw/hltds_cosmos_260215_04_14_2026}"
FILTER_FILE="${FILTER_FILE:-${RAW_ROOT}/diffsky_hltds_cosmos_260215_04_14_2026_transmission_curves.hdf5}"
SSP_FILE="${SSP_FILE:-${RAW_ROOT}/diffsky_hltds_cosmos_260215_04_14_2026_ssp_data.hdf5}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/euclid_dsps_matplotlib_${SLURM_JOB_ID:-manual}}"
export REPO_DIR
if [[ -n "${DIFFSKY_REPO:-}" ]]; then
  if [[ ! -d "$DIFFSKY_REPO/diffsky" ]]; then
    echo "[job][error] DIFFSKY_REPO=$DIFFSKY_REPO does not contain a diffsky package"
    exit 2
  fi
  export PYTHONPATH="$REPO_DIR:$DIFFSKY_REPO:${PYTHONPATH:-}"
else
  export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
fi
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$MPLCONFIGDIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[job] start: $(date --iso-8601=seconds)"
echo "[job] host: $(hostname)"
echo "[job] repo: $REPO_DIR"
echo "[job] diffsky_repo: ${DIFFSKY_REPO:-<conda/site-packages>}"
echo "[job] stage: $STAGE"
echo "[job] config: $CONFIG"
echo "[job] validation_config: $VALIDATION_CONFIG"
echo "[job] dataset_dir: $DATASET_DIR"
echo "[job] log_file: $LOG_FILE"
echo "[job] split: $SPLIT"
echo "[job] overwrite: $OVERWRITE"
echo "[job] resume: $RESUME"
echo "[job] smoke: $SMOKE"
echo "[job] max_galaxies: ${MAX_GALAXIES:-<config>}"
echo "[job] validation sample_size: $SAMPLE_SIZE"
echo "[job] validation batch_size: $BATCH_SIZE"
echo "[job] validation runtime: $VALIDATION_RUNTIME"

if [[ "$OVERWRITE" =~ ^(1|true|TRUE|yes|YES|y|Y)$ ]] && [[ "$RESUME" =~ ^(1|true|TRUE|yes|YES|y|Y)$ ]]; then
  echo "[job][error] OVERWRITE=1 and RESUME=1 are mutually exclusive for this generator."
  echo "[job][hint] Use OVERWRITE=1,RESUME=0 for a fresh 50k run after smoke tests."
  echo "[job][hint] Use OVERWRITE=0,RESUME=1 for an interrupted full run."
  exit 2
fi

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[job] python: $(which python)"
python - <<'PY'
import os
from pathlib import Path

import diffmah
import diffsky
import diffstar
import euclid_dsps
import jax

repo = Path(os.environ["REPO_DIR"]).resolve()
pkg = Path(euclid_dsps.__file__).resolve()
if repo not in pkg.parents:
    raise SystemExit(
        f"[job][error] euclid_dsps imported from {pkg}, not from repo {repo}. "
        "Check PYTHONPATH or the conda environment."
    )
print("[job] euclid_dsps:", pkg)
print("[job] diffsky:", Path(diffsky.__file__).resolve())
print("[job] diffstar:", Path(diffstar.__file__).resolve())
print("[job] diffmah:", Path(diffmah.__file__).resolve())
print("[job] jax_backend:", jax.default_backend())
print("[job] jax_devices:", jax.devices())
PY
nvidia-smi || true
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD

required_files=("$CONFIG" "$VALIDATION_CONFIG" "$FILTER_FILE" "$SSP_FILE")
missing=0
echo "[job] preflight files"
for path in "${required_files[@]}"; do
  if [[ -s "$path" ]]; then
    ls -lh "$path"
  else
    echo "[job][missing] $path"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "[job][error] Missing required configuration or Diffsky assets."
  exit 2
fi

run_generate() {
  cmd=(
    python -m euclid_dsps.cli
    --config "$CONFIG"
    diffsky-generate-dsps-closure
    --split "$SPLIT"
  )
  if [[ "$OVERWRITE" =~ ^(1|true|TRUE|yes|YES|y|Y)$ ]]; then
    cmd+=(--overwrite)
  fi
  if [[ "$RESUME" =~ ^(1|true|TRUE|yes|YES|y|Y)$ ]]; then
    cmd+=(--resume)
  fi
  if [[ "$SMOKE" =~ ^(1|true|TRUE|yes|YES|y|Y)$ ]]; then
    cmd+=(--smoke)
  fi
  if [[ -n "$MAX_GALAXIES" ]]; then
    cmd+=(--max-galaxies "$MAX_GALAXIES")
  fi
  echo "[job] generation command:"
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

run_validate() {
  cmd=(
    python -m euclid_dsps.cli
    --config "$VALIDATION_CONFIG"
    diffsky-validate-dsps-closure
    --dataset-dir "$DATASET_DIR"
    --sample-size "$SAMPLE_SIZE"
    --batch-size "$BATCH_SIZE"
    --runtime "$VALIDATION_RUNTIME"
  )
  echo "[job] validation command:"
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

case "$STAGE" in
  generate)
    run_generate
    ;;
  validate)
    run_validate
    ;;
  both)
    run_generate
    run_validate
    ;;
  *)
    echo "[job][error] Unknown STAGE=$STAGE. Use generate, validate, or both."
    exit 2
    ;;
esac

echo "[job] dataset outputs:"
ls -lh "$DATASET_DIR" || true
find "$DATASET_DIR/diagnostics/population" -maxdepth 2 -type f | sort || true
echo "[job] end: $(date --iso-8601=seconds)"
