#!/bin/bash
#SBATCH --job-name=feniks_sc_e160_cal_gate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SOURCE_EVAL_ROOT="${SOURCE_EVAL_ROOT:?SOURCE_EVAL_ROOT is required}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:?CALIBRATION_ROOT is required}"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" JAX_PLATFORMS=cpu
cd "$REPO_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python scripts/finalize_feniks_sc_drws_epoch160_catalogue_calibration.py \
  --source-evaluation-root "$SOURCE_EVAL_ROOT" \
  --root "$CALIBRATION_ROOT"
