#!/bin/bash
#SBATCH --job-name=feniks_qarch_gate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ARCHITECTURE_ROOT="${ARCHITECTURE_ROOT:?ARCHITECTURE_ROOT is required}"
SEEDS_CSV="${SEEDS_CSV:-260820,260821}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
cd "$REPO_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python scripts/select_feniks_architecture_20k.py \
  --root "$ARCHITECTURE_ROOT" --seeds "$SEEDS_CSV"
