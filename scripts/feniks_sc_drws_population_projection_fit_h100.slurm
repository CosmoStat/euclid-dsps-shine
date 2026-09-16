#!/bin/bash
#SBATCH --job-name=feniks_sc_proj_fit
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
PROJECTION_ROOT="${PROJECTION_ROOT:?PROJECTION_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
FIT_PASSES="${FIT_PASSES:-16}"
FIT_PATIENCE="${FIT_PATIENCE:-4}"
FIT_PEAK_LEARNING_RATE="${FIT_PEAK_LEARNING_RATE:-5.0e-5}"
FIT_FINAL_LEARNING_RATE="${FIT_FINAL_LEARNING_RATE:-2.0e-6}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$CACHE_ROOT/jax"
python scripts/train_feniks_sc_drws_population_projection.py \
  --root "$PROJECTION_ROOT" \
  --passes "$FIT_PASSES" \
  --patience "$FIT_PATIENCE" \
  --peak-learning-rate "$FIT_PEAK_LEARNING_RATE" \
  --final-learning-rate "$FIT_FINAL_LEARNING_RATE"
