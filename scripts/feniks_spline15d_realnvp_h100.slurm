#!/bin/bash
#SBATCH --job-name=feniks_realnvp15d
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=16:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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
CONFIG="${CONFIG:-configs/prior_feniks_spline15d_realnvp.yaml}"
SPLINE_DATASET_DIR="${SPLINE_DATASET_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d}"
OUT_DIR="${OUT_DIR:-outputs/runs/feniks_spline15d_realnvp_v1}"
SMOKE="${SMOKE:-0}"

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
for path in spline15d_contract.json train.parquet validation.parquet test.parquet; do
  test -s "$SPLINE_DATASET_DIR/$path" || {
    echo "[job][error] missing $SPLINE_DATASET_DIR/$path"; exit 2;
  }
done
python - <<'PY'
import equinox, euclid_dsps, jax, optax
print("[job] packages:", equinox.__version__, optax.__version__, euclid_dsps.__file__)
print("[job] backend/devices:", jax.default_backend(), jax.devices())
PY

cmd=(python scripts/train_feniks_spline15d_realnvp.py
  --config "$CONFIG" --dataset-dir "$SPLINE_DATASET_DIR" --out "$OUT_DIR"
  --data-parallel "${DATA_PARALLEL:-auto}")
if [[ "$SMOKE" == "1" ]]; then
  cmd+=(--limit "${LIMIT:-1000}" --epochs "${EPOCHS:-3}"
    --batch-size "${BATCH_SIZE:-256}" --n-layers "${N_LAYERS:-2}"
    --hidden-size "${HIDDEN_SIZE:-32}" --prior-samples "${PRIOR_SAMPLES:-1000}")
else
  [[ -n "${LIMIT:-}" ]] && cmd+=(--limit "$LIMIT")
  [[ -n "${EPOCHS:-}" ]] && cmd+=(--epochs "$EPOCHS")
  [[ -n "${BATCH_SIZE:-}" ]] && cmd+=(--batch-size "$BATCH_SIZE")
  [[ -n "${PRIOR_SAMPLES:-}" ]] && cmd+=(--prior-samples "$PRIOR_SAMPLES")
fi
printf '[job] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
echo "[job] done=$(date --iso-8601=seconds)"
