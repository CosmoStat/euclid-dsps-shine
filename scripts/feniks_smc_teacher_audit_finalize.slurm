#!/bin/bash
#SBATCH --job-name=feniks_tgate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:?BOOTSTRAP_ROOT is required}"
DISTILLED_ROOT="${DISTILLED_ROOT:?DISTILLED_ROOT is required}"
AUDIT_SUMMARY="${AUDIT_SUMMARY:?AUDIT_SUMMARY is required}"

export JAX_PLATFORMS=cpu EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python scripts/summarize_feniks_smc_teacher_audit.py \
  --bootstrap-root "$BOOTSTRAP_ROOT" \
  --distilled-root "$DISTILLED_ROOT" \
  --out "$AUDIT_SUMMARY"
