#!/bin/bash
#SBATCH --job-name=feniks_truthflow
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
set -Eeuo pipefail

cd "${FENIKS_CAPACITY_CODE:?}"
module purge
module load arch/h100
source "${MINICONDA_PATH:-${WORK:?}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-shine}"

export PYTHONPATH="$FENIKS_CAPACITY_CODE:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda
export EUCLID_DSPS_REQUIRE_GPU=1
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

python -m scripts.feniks_weighted_truth_flow_capacity \
  "${FENIKS_CAPACITY_MODE:?}" \
  --root "${FENIKS_CAPACITY_ROOT:?}" \
  --replica "${SLURM_ARRAY_TASK_ID:-0}"
