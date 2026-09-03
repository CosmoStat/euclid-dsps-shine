#!/bin/bash
#SBATCH --job-name=feniks_sc_vem_bank_gate
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
VEM_ROOT="${VEM_ROOT:?VEM_ROOT is required}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1

cd "$REPO_DIR"
module purge
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python scripts/finalize_feniks_sc_drws_population_vem_banks.py --root "$VEM_ROOT"
