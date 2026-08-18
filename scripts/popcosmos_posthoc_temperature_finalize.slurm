#!/bin/bash
#SBATCH --job-name=cosmos_qtemp_gate
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
SMC_REFRESH_OUT="${SMC_REFRESH_OUT:?Set SMC_REFRESH_OUT}"
TEMPERATURE_SCAN_ROOT="${TEMPERATURE_SCAN_ROOT:?Set TEMPERATURE_SCAN_ROOT}"
PROPOSAL_TEMPERATURES_CSV="${PROPOSAL_TEMPERATURES_CSV:?Set PROPOSAL_TEMPERATURES_CSV}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2048}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export JAX_PLATFORMS=cpu
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
module purge
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/select_popcosmos_proposal_temperature.py \
  --refresh-root "$SMC_REFRESH_OUT" \
  --scan-root "$TEMPERATURE_SCAN_ROOT" \
  --temperatures "$PROPOSAL_TEMPERATURES_CSV" \
  --probe-samples "$PROBE_SAMPLES"
