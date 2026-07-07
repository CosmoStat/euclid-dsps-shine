#!/bin/bash
#SBATCH --job-name=diffsky_inferred_prior
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

CONFIG="${CONFIG:-configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml}"
INFERRED_INPUTS="${INFERRED_INPUTS:-outputs/runs/map_diffsky_synthetic_feniks_under_prior/map_estimates.parquet}"
OUT_DIR="${OUT_DIR:-outputs/runs/prior_diffsky_synthetic_feniks_from_inferred}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EPOCHS="${EPOCHS:-80}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-260617}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/diffsky_inferred_prior_${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[inferred-prior] start: $(date --iso-8601=seconds)"
echo "[inferred-prior] config: $CONFIG"
echo "[inferred-prior] inputs: $INFERRED_INPUTS"
echo "[inferred-prior] out: $OUT_DIR"

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cmd=(python -m euclid_dsps.cli --config "$CONFIG" diffsky-train-inferred-prior --out "$OUT_DIR" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --seed "$SEED" --validation-fraction "$VALIDATION_FRACTION")
IFS=',' read -r -a inputs <<< "$INFERRED_INPUTS"
for input in "${inputs[@]}"; do
  if [[ ! -s "$input" ]]; then
    echo "[inferred-prior][missing] $input"
    exit 2
  fi
  cmd+=(--input "$input")
done
if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[inferred-prior] command:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "[inferred-prior] checkpoint: $OUT_DIR/checkpoints/best.eqx"
echo "[inferred-prior] end: $(date --iso-8601=seconds)"
