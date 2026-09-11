#!/bin/bash
#SBATCH --job-name=feniks_avi
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@600
set -Eeuo pipefail
module purge
module load arch/h100
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
cd "${AVI_CODE:?}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12
export MPLCONFIGDIR="$SCRATCH/feniks_sc_drws_runtime/mpl-${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="$SCRATCH/feniks_sc_drws_runtime/avi-jax-cache"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
python -m scripts.feniks_avi_experiments "${AVI_MODE:?}" --root "${AVI_ROOT:?}" \
  --task "${SLURM_ARRAY_TASK_ID:?}" --max-hours 18 &
PID=$!
trap 'kill -USR1 "$PID" 2>/dev/null || true' USR1
trap 'kill -TERM "$PID" 2>/dev/null || true' TERM
set +e
wait "$PID"
RC=$?
if kill -0 "$PID" 2>/dev/null; then
  wait "$PID"
  RC=$?
fi
exit "$RC"
