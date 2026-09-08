#!/bin/bash
#SBATCH --job-name=feniks_sc_localvi
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail
: "${REPO_DIR:?}" "${DIAGNOSTIC_ROOT:?}" "${CACHE_ROOT:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export JAX_PLATFORMS=cuda EUCLID_DSPS_JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 MPLBACKEND=Agg
export MPLCONFIGDIR="$CACHE_ROOT/mpl/${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
module purge
module load arch/h100
source "${MINICONDA_PATH:?}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-shine}"
cd "$REPO_DIR"
mkdir -p "$MPLCONFIGDIR" "$CACHE_ROOT/jax"
nvidia-smi
python scripts/run_feniks_sc_drws_local_vi_diagnostic.py run --root "$DIAGNOSTIC_ROOT"
