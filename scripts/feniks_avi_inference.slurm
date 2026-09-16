#!/bin/bash
#SBATCH --job-name=avi_infer
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
set -Eeuo pipefail
module purge
module load arch/h100
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate shine
cd "${AVI_CODE:?}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
export MPLCONFIGDIR="$SCRATCH/feniks_sc_drws_runtime/mpl-${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="$SCRATCH/feniks_sc_drws_runtime/avi-inference-cache"
mkdir -p "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
python -m scripts.feniks_avi_inference "${AVI_MODE:?}" --root "${INFERENCE_ROOT:?}" --task "${SLURM_ARRAY_TASK_ID:-0}"
