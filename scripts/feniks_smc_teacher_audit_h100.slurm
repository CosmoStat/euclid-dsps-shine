#!/bin/bash
#SBATCH --job-name=feniks_teacher
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail
INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:?CONFIG is required}"
DATASET="${DATASET:?DATASET is required}"
FEATURE_STATS="${FEATURE_STATS:?FEATURE_STATS is required}"
BOOTSTRAP_CHECKPOINT="${BOOTSTRAP_CHECKPOINT:?BOOTSTRAP_CHECKPOINT is required}"
DISTILLED_CHECKPOINT="${DISTILLED_CHECKPOINT:?DISTILLED_CHECKPOINT is required}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:?BOOTSTRAP_ROOT is required}"
DISTILLED_ROOT="${DISTILLED_ROOT:?DISTILLED_ROOT is required}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${INDEX}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-/tmp}/${USER}/jax-cache/euclid-dsps/feniks-teacher-audit"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMMON=(--config "$CONFIG" --dataset "$DATASET" --feature-stats "$FEATURE_STATS" \
  --galaxy-index "$INDEX" --mode smoke --seed 260824 --ignore-truth \
  --skip-defensive-is)

echo "[teacher-audit] galaxy=$INDEX phase=bootstrap+extended-smc"
python scripts/run_feniks_exact_posterior_benchmark.py prepare-galaxy \
  "${COMMON[@]}" --checkpoint "$BOOTSTRAP_CHECKPOINT" --out "$BOOTSTRAP_ROOT" \
  --encoder-samples 4096 --map-starts 8 --map-iterations 300 \
  --adaptive-smc-mode teacher

echo "[teacher-audit] galaxy=$INDEX phase=nuts"
python scripts/run_feniks_exact_posterior_benchmark.py sample-nuts-batched \
  "${COMMON[@]}" --checkpoint "$BOOTSTRAP_CHECKPOINT" --out "$BOOTSTRAP_ROOT" \
  --nuts-warmup 500 --nuts-max-doublings 6 --sample-chunks 1000
python scripts/run_feniks_exact_posterior_benchmark.py finalize-galaxy \
  "${COMMON[@]}" --checkpoint "$BOOTSTRAP_CHECKPOINT" --out "$BOOTSTRAP_ROOT" \
  --samplers nuts

echo "[teacher-audit] galaxy=$INDEX phase=distilled-q"
python scripts/run_feniks_exact_posterior_benchmark.py prepare-galaxy \
  "${COMMON[@]}" --checkpoint "$DISTILLED_CHECKPOINT" --out "$DISTILLED_ROOT" \
  --encoder-samples 4096 --map-starts 1 --map-iterations 1 \
  --adaptive-smc-mode skip

echo "[teacher-audit] galaxy=$INDEX complete"
