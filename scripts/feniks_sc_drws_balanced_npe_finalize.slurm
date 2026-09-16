#!/bin/bash
#SBATCH --job-name=feniks_sc_balclose
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail
export PYTHONPATH="${REPO_DIR:?}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu EUCLID_DSPS_JAX_PLATFORMS=cpu
export EUCLID_DSPS_REQUIRE_GPU=0
module purge
source "${MINICONDA_PATH:?}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-shine}"
cd "$REPO_DIR"
python scripts/run_feniks_sc_drws_balanced_npe.py "${ACTION:?}" --root "${BALANCED_ROOT:?}"
