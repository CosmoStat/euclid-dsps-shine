#!/bin/bash
#SBATCH --job-name=feniks_avi_night
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
set -Eeuo pipefail

module purge
module load arch/h100
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
cd "${AVI_OVERNIGHT_CODE:?}"

export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MPLCONFIGDIR="$SCRATCH/feniks_sc_drws_runtime/mpl-${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="$SCRATCH/feniks_sc_drws_runtime/avi-overnight-cache"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"

case "${AVI_OVERNIGHT_MODE:?}" in
  preflight|train)
    python -m scripts.feniks_avi_overnight "$AVI_OVERNIGHT_MODE" \
      --root "${AVI_TRAIN_ROOT:?}" \
      --task "${SLURM_ARRAY_TASK_ID:?}" --max-hours 10
    ;;
  prepare-inference)
    JAX_PLATFORMS=cpu python -m scripts.feniks_avi_overnight prepare-inference \
      --root "${AVI_INFERENCE_ROOT:?}" --training "${AVI_TRAIN_ROOT:?}" \
      --source "${AVI_SOURCE_ROOT:?}" \
      --prior-followup "${AVI_PRIOR_ROOT:?}" --particles 4096
    ;;
  infer)
    python -m scripts.feniks_avi_overnight infer \
      --root "${AVI_INFERENCE_ROOT:?}" --task "${SLURM_ARRAY_TASK_ID:?}"
    ;;
  report)
    python -m scripts.feniks_avi_overnight report \
      --root "${AVI_INFERENCE_ROOT:?}"
    ;;
  *)
    echo "Unknown AVI_OVERNIGHT_MODE=$AVI_OVERNIGHT_MODE" >&2
    exit 2
    ;;
esac
