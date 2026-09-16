#!/bin/bash
#SBATCH --job-name=cosmos_qadapt_gate
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
ADAPTER_ROOT="${ADAPTER_ROOT:?Set ADAPTER_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"
module purge
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/select_popcosmos_proposal_adapter.py \
  --root "$ADAPTER_ROOT" \
  --seeds 260820,260821,260822
