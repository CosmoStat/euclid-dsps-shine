#!/bin/bash
#SBATCH --job-name=feniks_ngcap
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_paper_rws_k8_t2_seed2.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized/test.parquet}"
MODEL_ROOT="${MODEL_ROOT:-outputs/runs/feniks_selfsup_paper_v1/rws_k8_t2_seed2}"
CHECKPOINT="${CHECKPOINT:-$MODEL_ROOT/train/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-$MODEL_ROOT/train/feature_stats.json}"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
BATCH_SIZES="${BATCH_SIZES:-1:2:4:8}"
CHAINS="${CHAINS:-4}"
NUTS_WARMUP="${NUTS_WARMUP:-10}"
NUTS_DRAWS="${NUTS_DRAWS:-10}"
NUTS_MAX_DOUBLINGS="${NUTS_MAX_DOUBLINGS:-4}"
SEED="${SEED:-260730}"

IFS=: read -r -a batch_sizes <<< "$BATCH_SIZES"
task_index="${SLURM_ARRAY_TASK_ID:?Submit as a batch-size array}"
test "$task_index" -lt "${#batch_sizes[@]}" || {
  echo "[nuts-multigalaxy][error] task $task_index exceeds BATCH_SIZES"
  exit 2
}
batch_size="${batch_sizes[$task_index]}"
batch_dir="$ROOT_DIR/batch_g$(printf '%04d' "$batch_size")"
gpu_metrics="$batch_dir/gpu_metrics.csv"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH:-/tmp}/${USER}/jax-cache/euclid-dsps/feniks-nuts-capacity"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$batch_dir" "$JAX_COMPILATION_CACHE_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

printf '%s\n' \
  'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_w' \
  > "$gpu_metrics"
gpu_monitor() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits >> "$gpu_metrics" || true
    sleep 5
  done
}
gpu_monitor &
gpu_monitor_pid=$!

heartbeat() {
  while sleep 60; do
    printf '[nuts-multigalaxy] heartbeat batch_galaxies=%s elapsed=%ss\n' \
      "$batch_size" "$SECONDS"
  done
}
heartbeat &
heartbeat_pid=$!

cleanup() {
  kill "$gpu_monitor_pid" "$heartbeat_pid" 2>/dev/null || true
  wait "$gpu_monitor_pid" "$heartbeat_pid" 2>/dev/null || true
}
trap cleanup EXIT

printf '[nuts-multigalaxy] batch_galaxies=%s chains=%s warmup=%s draws=%s max_doublings=%s\n' \
  "$batch_size" "$CHAINS" "$NUTS_WARMUP" "$NUTS_DRAWS" "$NUTS_MAX_DOUBLINGS"

set +e
python scripts/benchmark_feniks_nuts_multigalaxy.py run \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --out "$ROOT_DIR" \
  --batch-size "$batch_size" \
  --chains "$CHAINS" \
  --warmup "$NUTS_WARMUP" \
  --draws "$NUTS_DRAWS" \
  --max-num-doublings "$NUTS_MAX_DOUBLINGS" \
  --seed "$SEED" \
  --gpu-metrics "$gpu_metrics"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  printf 'exit_code=%s\n' "$status" > "$batch_dir/FAILED"
fi
exit "$status"
