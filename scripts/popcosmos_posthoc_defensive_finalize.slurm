#!/bin/bash
#SBATCH --job-name=cosmos_qdef_gate
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
BASELINE_REFRESH_OUT="${BASELINE_REFRESH_OUT:?Set BASELINE_REFRESH_OUT}"
DEFENSIVE_SCAN_ROOT="${DEFENSIVE_SCAN_ROOT:?Set DEFENSIVE_SCAN_ROOT}"
TAIL_TEMPERATURES_CSV="${TAIL_TEMPERATURES_CSV:?Set TAIL_TEMPERATURES_CSV}"
TAIL_FRACTIONS_CSV="${TAIL_FRACTIONS_CSV:?Set TAIL_FRACTIONS_CSV}"
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

python scripts/select_popcosmos_defensive_proposal.py \
  --baseline-importance "$BASELINE_REFRESH_OUT/moderate_k${PROBE_SAMPLES}_importance/importance_summary.json" \
  --scan-root "$DEFENSIVE_SCAN_ROOT" \
  --tail-temperatures "$TAIL_TEMPERATURES_CSV" \
  --tail-fractions "$TAIL_FRACTIONS_CSV" \
  --probe-samples "$PROBE_SAMPLES"
