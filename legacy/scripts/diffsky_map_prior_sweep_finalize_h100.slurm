#!/bin/bash
#SBATCH --job-name=diffsky_map_finalize
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_tau2safe_h100.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs/runs/diffsky_map_prior_sweep_sharded}"
LOG_DIR="${LOG_DIR:-outputs/logs}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/diffsky_map_finalize_${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[map-finalize] start: $(date --iso-8601=seconds)"
echo "[map-finalize] out_root: $OUT_ROOT"

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -m euclid_dsps.cli \
  --config "$CONFIG" \
  diffsky-finalize-map-prior-sweep \
  --out "$OUT_ROOT"

echo "[map-finalize] end: $(date --iso-8601=seconds)"
