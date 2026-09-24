#!/bin/bash
#SBATCH --job-name=feniks_population_low_rank
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
set -Eeuo pipefail

cd "${FENIKS_POPULATION_LOW_RANK_CODE:?}"
module purge
module load arch/h100
source "${MINICONDA_PATH:-${WORK:?}/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-shine}"

export PYTHONPATH="$FENIKS_POPULATION_LOW_RANK_CODE:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_REQUIRE_GPU=1
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

python -m scripts.feniks_population_low_rank_audit \
  "${FENIKS_POPULATION_LOW_RANK_MODE:?}" \
  --root "${FENIKS_POPULATION_LOW_RANK_ROOT:?}" \
  --task "${SLURM_ARRAY_TASK_ID:-0}"
