#!/bin/bash
#SBATCH --job-name=feniks_asmc_qboot
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:?CONFIG is required}"
CATALOG_DIR="${CATALOG_DIR:?CATALOG_DIR is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
BOOTSTRAP_CHECKPOINT="${BOOTSTRAP_CHECKPOINT:?BOOTSTRAP_CHECKPOINT is required}"
CURRICULUM_ROOT="${CURRICULUM_ROOT:?CURRICULUM_ROOT is required}"
EXACT_OBJECTS="${EXACT_OBJECTS:-32}"
EXTENDED_MAX_STAGES="${EXTENDED_MAX_STAGES:-48}"
DISTILLATION_STEPS="${DISTILLATION_STEPS:-4}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0 MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$CURRICULUM_ROOT" "$MPLCONFIGDIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/run_feniks_adaptive_smc_q_curriculum.py \
  --config "$CONFIG" \
  --catalog "$CATALOG_DIR/train.parquet" \
  --train-indices-file "$MANIFEST_ROOT/train_indices.npy" \
  --validation-indices-file "$MANIFEST_ROOT/validation_indices.npy" \
  --checkpoint "$BOOTSTRAP_CHECKPOINT" \
  --out "$CURRICULUM_ROOT" \
  --exact-objects "$EXACT_OBJECTS" \
  --extended-max-stages "$EXTENDED_MAX_STAGES" \
  --distillation-steps "$DISTILLATION_STEPS" \
  --expected-devices 4

python scripts/validate_feniks_adaptive_smc_q_curriculum.py \
  --curriculum "$CURRICULUM_ROOT"
