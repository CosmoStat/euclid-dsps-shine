#!/bin/bash
#SBATCH --job-name=feniks_xgate
#SBATCH --account=jrx@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
PREPARED_ROOT="${PREPARED_ROOT:?PREPARED_ROOT is required}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[two-galaxy-big-gate] summarizing completed batched probe"
python scripts/summarize_feniks_nuts_batched_probe.py \
  --root "$PREPARED_ROOT"

echo "[two-galaxy-big-gate] probe passed; submitting production run"
bash scripts/submit_feniks_exact_posterior_two_galaxy_nuts_big.sh
