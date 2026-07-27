#!/bin/bash
#SBATCH --job-name=feniks_rnvp15d_v4
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
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
SPLINE_DATASET_DIR="${SPLINE_DATASET_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_v3}"
OUT_DIR="${OUT_DIR:-outputs/runs/feniks_spline15d_realnvp_shifted_v4}"
PRIOR_CONFIG="${PRIOR_CONFIG:-configs/prior_feniks_spline15d_realnvp_shifted_v4.yaml}"
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

for split in train validation test; do
  test -s "$SPLINE_DATASET_DIR/$split.parquet" || {
    echo "[job][error] missing $SPLINE_DATASET_DIR/$split.parquet"; exit 2;
  }
  test -s "$SPLINE_DATASET_DIR/${split}_exact.parquet" || {
    echo "[job][error] missing $SPLINE_DATASET_DIR/${split}_exact.parquet"; exit 2;
  }
done
test -s "$SPLINE_DATASET_DIR/spline15d_contract.json" || {
  echo "[job][error] missing spline15d contract"; exit 2;
}
if [[ -e "$OUT_DIR" ]]; then
  echo "[job][error] output already exists: $OUT_DIR"; exit 2
fi

python - <<'PY'
import equinox, euclid_dsps, jax, optax
print("[job] packages:", equinox.__version__, optax.__version__, euclid_dsps.__file__)
print("[job] backend/devices:", jax.default_backend(), jax.devices())
PY

training_args=(--data-parallel "${DATA_PARALLEL:-auto}")
if [[ "$SMOKE" == "1" ]]; then
  training_args+=(--limit "${LIMIT:-1000}" --epochs "${EPOCHS:-2}"
    --batch-size "${BATCH_SIZE:-256}" --n-layers "${N_LAYERS:-2}"
    --hidden-size "${HIDDEN_SIZE:-32}" --prior-samples "${PRIOR_SAMPLES:-1000}"
    --evaluation-samples "${EVALUATION_SAMPLES:-1000}"
    --snapshot-samples "${SNAPSHOT_SAMPLES:-1000}")
fi

echo "[job] step=realnvp_v4 start=$(date --iso-8601=seconds)"
python scripts/train_feniks_spline15d_realnvp.py \
  --config "$PRIOR_CONFIG" \
  --dataset-dir "$SPLINE_DATASET_DIR" \
  --out "$OUT_DIR" \
  "${training_args[@]}"
echo "[job] done=$(date --iso-8601=seconds)"
