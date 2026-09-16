#!/bin/bash
#SBATCH --job-name=feniks_qdiag_prep
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
: "${CONFIG:?CONFIG is required}"
: "${SOURCE_DATASET:?SOURCE_DATASET is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${FEATURE_STATS:?FEATURE_STATS is required}"
: "${OBSERVED_COHORT:?OBSERVED_COHORT is required}"
: "${INPUT_ROOT:?INPUT_ROOT is required}"
: "${EXACT_ROOT:?EXACT_ROOT is required}"

export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/build_feniks_encoder_diagnostic_dataset.py \
  --config "$CONFIG" --dataset "$SOURCE_DATASET" \
  --checkpoint "$CHECKPOINT" --feature-stats "$FEATURE_STATS" \
  --observed-cohort "$OBSERVED_COHORT" --out "$INPUT_ROOT" \
  --n-observed "${N_OBSERVED:-16}" --n-sleep "${N_SLEEP:-16}" \
  --seed "${DIAGNOSTIC_SEED:-260821}"

python scripts/run_feniks_exact_posterior_benchmark.py prepare-cohort \
  --config "$CONFIG" --dataset "$INPUT_ROOT/diagnostic_dataset.parquet" \
  --checkpoint "$CHECKPOINT" --feature-stats "$FEATURE_STATS" \
  --out "$EXACT_ROOT" --mode pilot \
  --cohort-file "$INPUT_ROOT/diagnostic_cohort.csv"

echo "[feniks-qdiag-prep] complete -> $EXACT_ROOT"
