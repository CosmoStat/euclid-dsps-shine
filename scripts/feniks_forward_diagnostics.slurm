#!/bin/bash
#SBATCH --job-name=feniks_diag
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
set -Eeuo pipefail
cd "${FORWARD_CODE:?}"
module purge
module load arch/h100
source "${MINICONDA_PATH:-${WORK:?}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-shine}"
export PYTHONPATH="$FORWARD_CODE:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_REQUIRE_GPU=1
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
python -m scripts.feniks_forward_diagnostics "$FORWARD_MODE" --root "$FORWARD_ROOT" --task "${SLURM_ARRAY_TASK_ID:-0}"
