#!/bin/bash
#SBATCH --job-name=feniks_parent_x
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

GALAXY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as a galaxy array}"
N_GALAXIES="${N_GALAXIES:-32}"
(( GALAXY_INDEX < N_GALAXIES )) || {
  echo "[feniks-parent-exact][error] invalid task $GALAXY_INDEX" >&2; exit 2;
}
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:?CONFIG is required}"
DATASET="${DATASET:?DATASET is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
FEATURE_STATS="${FEATURE_STATS:?FEATURE_STATS is required}"
EXACT_ROOT="${EXACT_ROOT:?EXACT_ROOT is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${GALAXY_INDEX}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-/tmp}/${USER}/jax-cache/euclid-dsps/feniks-parent-exact"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMMON=(
  --config "$CONFIG"
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS"
  --out "$EXACT_ROOT"
  --galaxy-index "$GALAXY_INDEX"
  --mode "${DIAGNOSTIC_MODE:-full}"
)

if [[ "${DIAGNOSTIC_MODE:-full}" == "smoke" ]]; then
  ENCODER_SAMPLES=256
  MAP_STARTS=4
  MAP_ITERATIONS=20
  NUTS_WARMUP=10
  NUTS_MAX_DOUBLINGS=3
  SAMPLE_CHUNKS=10
else
  ENCODER_SAMPLES=8192
  MAP_STARTS=16
  MAP_ITERATIONS=2000
  NUTS_WARMUP=500
  NUTS_MAX_DOUBLINGS=6
  SAMPLE_CHUNKS=100,500,1000
fi

python scripts/run_feniks_exact_posterior_benchmark.py prepare-galaxy \
  "${COMMON[@]}" --encoder-samples "$ENCODER_SAMPLES" \
  --map-starts "$MAP_STARTS" --map-iterations "$MAP_ITERATIONS"

python scripts/run_feniks_exact_posterior_benchmark.py sample-nuts-batched \
  "${COMMON[@]}" --nuts-warmup "$NUTS_WARMUP" \
  --nuts-max-doublings "$NUTS_MAX_DOUBLINGS" --sample-chunks "$SAMPLE_CHUNKS"

python scripts/run_feniks_exact_posterior_benchmark.py finalize-galaxy \
  "${COMMON[@]}" --samplers nuts

if [[ "$GALAXY_INDEX" == "0" && "${ENCODER_DIAGNOSTIC_ONLY:-0}" != "1" ]]; then
  python scripts/evaluate_feniks_parent_population.py \
    --config "$CONFIG" --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
    --feature-stats "$FEATURE_STATS" \
    --selected-indices "$MANIFEST_ROOT/selected_test_indices.npy" \
    --out "$EXACT_ROOT/population" --n-prior-samples 8192 \
    --q-samples-per-object 16 --batch-size 256 \
    --prior-decoder-batch-size 128
fi

echo "[feniks-parent-exact] complete galaxy=$GALAXY_INDEX"
