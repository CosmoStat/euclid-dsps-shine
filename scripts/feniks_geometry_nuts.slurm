#!/bin/bash
#SBATCH --job-name=feniks_geometry
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
set -Eeuo pipefail
: "${EXPERIMENT_ROOT:?}" "${REPO_DIR:?}" "${CACHE_ROOT:?}" "${EXPERIMENT_MODE:?}"
module purge
module load arch/h100
source "${WORK:?}/miniconda3/etc/profile.d/conda.sh"
conda activate shine
export PYTHONUNBUFFERED=1 JAX_ENABLE_X64=true JAX_PLATFORMS=cuda MPLBACKEND=Agg
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MPLCONFIGDIR="$CACHE_ROOT/mpl/${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
mkdir -p "$MPLCONFIGDIR" "$CACHE_ROOT/jax"
cd "$REPO_DIR"
ARGS=("$EXPERIMENT_MODE" --root "$EXPERIMENT_ROOT")
if [[ "$EXPERIMENT_MODE" == nuts ]]; then
  ARGS+=(--task "${SLURM_ARRAY_TASK_ID:?}")
fi
if [[ "$EXPERIMENT_MODE" == nuts ]]; then
  PARENT_PID=$$
  (
    while kill -0 "$PARENT_PID" 2>/dev/null; do
      sleep 300
      kill -0 "$PARENT_PID" 2>/dev/null || exit 0
      printf '[nuts-heartbeat] %s task=%s elapsed_s=%s state=alive warmup_progress=opaque\n' \
        "$(date --iso-8601=seconds)" "${SLURM_ARRAY_TASK_ID:?}" "$SECONDS"
    done
  ) &
  HEARTBEAT_PID=$!
  trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT
fi
python -m scripts.feniks_geometry_nuts "${ARGS[@]}"
