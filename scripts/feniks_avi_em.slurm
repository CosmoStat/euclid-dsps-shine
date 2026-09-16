#!/bin/bash
#SBATCH --job-name=feniks_avi_em
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${AVI_EM_CODE:?AVI_EM_CODE is required}"
ROOT="${AVI_EM_ROOT:?AVI_EM_ROOT is required}"
INFERENCE_ROOT="${AVI_EM_INFERENCE_ROOT:?AVI_EM_INFERENCE_ROOT is required}"
MODE="${AVI_EM_MODE:?AVI_EM_MODE is required}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export EUCLID_DSPS_JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1
export EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export MPLBACKEND=Agg

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

case "$MODE" in
  cycle)
    CYCLE="${AVI_EM_CYCLE:?AVI_EM_CYCLE is required for cycle mode}"
    CYCLE_TAG=$(printf '%02d' "$CYCLE")
    MSTEP="$ROOT/cycles/cycle_${CYCLE_TAG}/mstep"
    QSTEP="$ROOT/cycles/cycle_${CYCLE_TAG}/qstep"

    python -m scripts.feniks_avi_em prepare-mstep \
      --root "$ROOT" --cycle "$CYCLE"
    python -m scripts.feniks_avi_next_experiments preflight \
      --root "$MSTEP" --task 0
    python -m scripts.feniks_avi_next_experiments train \
      --root "$MSTEP" --task 0 --max-hours 4

    python -m scripts.feniks_avi_em prepare-qstep \
      --root "$ROOT" --cycle "$CYCLE"
    python -m scripts.feniks_avi_experiments preflight \
      --root "$QSTEP" --task 0
    python -m scripts.feniks_avi_experiments train \
      --root "$QSTEP" --task 0 --max-hours 4
    python -m scripts.feniks_avi_em finalize-cycle \
      --root "$ROOT" --cycle "$CYCLE"
    ;;
  prepare-inference)
    python -m scripts.feniks_avi_em prepare-inference \
      --root "$ROOT" --inference-root "$INFERENCE_ROOT" \
      --particles "${AVI_EM_PARTICLES:-4096}"
    ;;
  infer)
    python -m scripts.feniks_avi_em infer \
      --root "$INFERENCE_ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  report)
    python -m scripts.feniks_avi_em report --root "$INFERENCE_ROOT"
    ;;
  *)
    echo "Unknown AVI_EM_MODE=$MODE" >&2
    exit 2
    ;;
esac
