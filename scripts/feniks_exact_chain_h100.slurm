#!/bin/bash
#SBATCH --job-name=feniks_xchain
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

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
SAMPLER="${SAMPLER:?SAMPLER is required}"
CHAINS="${CHAINS:-4}"
N_GALAXIES="${N_GALAXIES:-7}"
GALAXY_OFFSET="${GALAXY_OFFSET:-0}"
MODE="${MODE:-full}"
galaxy_local=$(( TASK_ID / CHAINS ))
chain=$(( TASK_ID % CHAINS ))
(( galaxy_local < N_GALAXIES )) || { echo "invalid chain task $TASK_ID"; exit 2; }
galaxy=$(( galaxy_local + GALAXY_OFFSET ))

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"

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
[[ -n "${SAMPLER_LABEL:-}" ]] && extra+=(--sampler-label "$SAMPLER_LABEL")
[[ -n "${PILOT_SELECTION:-}" ]] && extra+=(--pilot-selection "$PILOT_SELECTION")
[[ -n "${THINNING:-}" ]] && extra+=(--thinning "$THINNING")
[[ -n "${FRAC_TUNE:-}" ]] && extra+=(--frac-tune "$FRAC_TUNE")
[[ -n "${DESIRED_ENERGY_VAR:-}" ]] && extra+=(--desired-energy-var "$DESIRED_ENERGY_VAR")
[[ -n "${SAMPLE_CHUNKS:-}" ]] && extra+=(--sample-chunks "$SAMPLE_CHUNKS")
[[ -n "${NUTS_WARMUP:-}" ]] && extra+=(--nuts-warmup "$NUTS_WARMUP")
[[ -n "${MCLMC_TUNE:-}" ]] && extra+=(--mclmc-tune "$MCLMC_TUNE")

heartbeat() {
  while sleep 60; do
    printf '[exact-chain] heartbeat sampler=%s galaxy=%s chain=%s elapsed=%s\n' \
      "$SAMPLER" "$galaxy" "$chain" "${SECONDS}s"
  done
}
heartbeat &
heartbeat_pid=$!
trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT

python scripts/run_feniks_exact_posterior_benchmark.py sample-chain \
  --config "$CONFIG" --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --out "$ROOT_DIR" \
  --galaxy-index "$galaxy" --chain-index "$chain" \
  --sampler "$SAMPLER" --mode "$MODE" "${extra[@]}"
