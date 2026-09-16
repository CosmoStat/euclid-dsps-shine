#!/bin/bash
#SBATCH --job-name=feniks_sbeb
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${FENIKS_SBEB_CODE:?FENIKS_SBEB_CODE is required}"
ROOT="${FENIKS_SBEB_ROOT:?FENIKS_SBEB_ROOT is required}"
MODE="${FENIKS_SBEB_MODE:?FENIKS_SBEB_MODE is required}"
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
  bootstrap)
    python -m scripts.feniks_sbeb_benchmark bootstrap \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}" \
      --max-hours "${FENIKS_SBEB_MAX_HOURS:-19}"
    ;;
  factor)
    python -m scripts.feniks_sbeb_benchmark factor \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  prepare-factor-inference)
    python -m scripts.feniks_sbeb_benchmark prepare-factor-inference --root "$ROOT"
    ;;
  factor-infer)
    python -m scripts.feniks_sbeb_benchmark factor-infer \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  factor-report)
    python -m scripts.feniks_sbeb_benchmark factor-report --root "$ROOT"
    ;;
  prepare-trajectories)
    python -m scripts.feniks_sbeb_benchmark prepare-trajectories --root "$ROOT"
    ;;
  em-cycle)
    python -m scripts.feniks_sbeb_benchmark em-cycle \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}" \
      --cycle "${FENIKS_SBEB_CYCLE:?cycle required}" \
      --max-hours "${FENIKS_SBEB_MAX_HOURS:-9}"
    ;;
  prepare-endpoint-factorials)
    python -m scripts.feniks_sbeb_benchmark prepare-endpoint-factorials --root "$ROOT"
    ;;
  endpoint-factorial-infer)
    python -m scripts.feniks_sbeb_benchmark endpoint-factorial-infer \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  endpoint-factorial-report)
    python -m scripts.feniks_sbeb_benchmark endpoint-factorial-report \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  prepare-em-inference)
    python -m scripts.feniks_sbeb_benchmark prepare-em-inference --root "$ROOT"
    ;;
  em-infer)
    python -m scripts.feniks_sbeb_benchmark em-infer \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  em-report)
    python -m scripts.feniks_sbeb_benchmark em-report \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  nuts-infer)
    python -m scripts.feniks_sbeb_benchmark nuts-infer \
      --root "$ROOT" --task "${SLURM_ARRAY_TASK_ID:?array task required}"
    ;;
  report)
    python -m scripts.feniks_sbeb_benchmark report --root "$ROOT"
    ;;
  *)
    echo "Unknown FENIKS_SBEB_MODE=$MODE" >&2
    exit 2
    ;;
esac
