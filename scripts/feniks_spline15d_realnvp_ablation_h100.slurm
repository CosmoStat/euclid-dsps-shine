#!/bin/bash
#SBATCH --job-name=feniks_rnvp15d_v2
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=16:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/euclid_dsps" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  else
    REPO_DIR="${WORK:?Set WORK or REPO_DIR}/euclid-dsps-shine"
  fi
fi
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
SPLINE_DATASET_DIR="${SPLINE_DATASET_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d}"
RUN_PREFIX="${RUN_PREFIX:-feniks_spline15d_realnvp_v2}"
SMOKE="${SMOKE:-0}"

configs=(
  configs/experiments/feniks_spline15d_realnvp_v2_a_control.yaml
  configs/experiments/feniks_spline15d_realnvp_v2_b_permutation.yaml
  configs/experiments/feniks_spline15d_realnvp_v2_c_conservative.yaml
  configs/experiments/feniks_spline15d_realnvp_v2_d_regularized.yaml
)
labels=(a_control b_permutation c_conservative d_regularized)
index="${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"
CONFIG="${configs[$index]}"
OUT_DIR="${OUT_DIR:-outputs/runs/${RUN_PREFIX}_${labels[$index]}}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}-${SLURM_JOB_ID:-manual}}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ -e "$OUT_DIR" ]]; then
  echo "[job][error] output already exists: $OUT_DIR"; exit 2
fi
for split in train validation test; do
  for suffix in .parquet _exact.parquet; do
    test -s "$SPLINE_DATASET_DIR/${split}${suffix}" || {
      echo "[job][error] missing $SPLINE_DATASET_DIR/${split}${suffix}"; exit 2;
    }
  done
done

cmd=(python scripts/train_feniks_spline15d_realnvp.py
  --config "$CONFIG" --dataset-dir "$SPLINE_DATASET_DIR" --out "$OUT_DIR"
  --data-parallel "${DATA_PARALLEL:-auto}")
if [[ "$SMOKE" == "1" ]]; then
  cmd+=(--limit "${LIMIT:-1000}" --epochs "${EPOCHS:-3}"
    --batch-size "${BATCH_SIZE:-256}" --n-layers "${N_LAYERS:-2}"
    --hidden-size "${HIDDEN_SIZE:-32}" --prior-samples "${PRIOR_SAMPLES:-1000}"
    --evaluation-every 1 --evaluation-samples "${EVALUATION_SAMPLES:-512}")
fi
printf '[job] config=%s output=%s command:' "$CONFIG" "$OUT_DIR"
printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
