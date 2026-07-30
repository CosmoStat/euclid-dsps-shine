#!/bin/bash
#SBATCH --job-name=feniks_ngsum
#SBATCH --account=jrx@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
BATCH_SIZES="${BATCH_SIZES:-1:2:4:8}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs
module purge
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/benchmark_feniks_nuts_multigalaxy.py summarize \
  --out "$ROOT_DIR" \
  --batch-sizes "$BATCH_SIZES"
