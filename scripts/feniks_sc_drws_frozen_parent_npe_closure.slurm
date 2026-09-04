#!/bin/bash
#SBATCH --job-name=feniks_sc_npe_close
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
NPE_ROOT="${NPE_ROOT:?NPE_ROOT is required}"
BASELINE_ROOT="${BASELINE_ROOT:?BASELINE_ROOT is required}"
FULL_ROOT="${FULL_ROOT:?FULL_ROOT is required}"
SUPPORT_ROOT="${SUPPORT_ROOT:?SUPPORT_ROOT is required}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1

cd "$REPO_DIR"
module purge
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$MPLCONFIGDIR"
python scripts/finalize_feniks_sc_drws_frozen_parent_npe_closure.py \
  --root "$NPE_ROOT" --baseline-root "$BASELINE_ROOT" \
  --full-root "$FULL_ROOT" --support-root "$SUPPORT_ROOT"
