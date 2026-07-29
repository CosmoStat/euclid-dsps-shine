#!/bin/bash
#SBATCH --job-name=feniks_xbnuts
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

GALAXY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit as a galaxy array}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
MODE="${MODE:-pilot}"
NUTS_WARMUP="${NUTS_WARMUP:-50}"
NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"
SAMPLE_CHUNKS="${SAMPLE_CHUNKS:-100}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-/tmp}/${USER}/jax-cache/euclid-dsps/feniks-exact"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

cd "$REPO_DIR"
mkdir -p outputs/logs
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

extra=()
[[ -n "${CHAIN_INDICES:-}" ]] && extra+=(--chain-indices "$CHAIN_INDICES")

heartbeat() {
  while sleep 60; do
    printf '[exact-nuts-batched] heartbeat galaxy=%s elapsed=%s\n' \
      "$GALAXY_INDEX" "${SECONDS}s"
  done
}
heartbeat &
heartbeat_pid=$!
trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT

python scripts/run_feniks_exact_posterior_benchmark.py sample-nuts-batched \
  --config "$CONFIG" --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --out "$ROOT_DIR" \
  --galaxy-index "$GALAXY_INDEX" --mode "$MODE" \
  --nuts-warmup "$NUTS_WARMUP" \
  --nuts-max-doublings "$NUTS_MAX_DOUBLINGS" \
  --sample-chunks "$SAMPLE_CHUNKS" "${extra[@]}"
