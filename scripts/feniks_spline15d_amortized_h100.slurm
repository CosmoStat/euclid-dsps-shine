#!/bin/bash
#SBATCH --job-name=feniks_spline15d_e2e
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
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
CONFIG="${CONFIG:-configs/amortized_feniks_spline15d_18band_gpu.yaml}"
SOURCE_DIR="${SOURCE_DIR:-Data/diffsky/synthetic/feniks_260617_dsps_closure_18band_grouped}"
SPLINE_DIR="${SPLINE_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_v3}"
CATALOG_DIR="${CATALOG_DIR:-${SPLINE_DIR}/amortized}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-outputs/runs/feniks_spline15d_v6_positive_support_resume400_to800/checkpoints/best.eqx}"
OUT_DIR="${OUT_DIR:-outputs/runs/feniks_spline15d_amortized_frozen_prior_v1}"
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-120}"
N_SAMPLES="${N_SAMPLES:-2}"
DATA_PARALLEL="${DATA_PARALLEL:-single}"
VALIDATION_EVERY="${VALIDATION_EVERY:-1}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
START_EPOCH="${START_EPOCH:-1}"
NO_PROGRESS="${NO_PROGRESS:-1}"
LIMIT="${LIMIT:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}-${SLURM_JOB_ID:-manual}}"
export PIP_NO_INDEX=1
export DATA_PARALLEL

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -s "$PRIOR_CHECKPOINT" || {
  echo "[job][error] missing prior checkpoint: $PRIOR_CHECKPOINT"; exit 2;
}
for split in train validation test; do
  test -s "$SOURCE_DIR/$split.parquet" || {
    echo "[job][error] missing source split: $SOURCE_DIR/$split.parquet"; exit 2;
  }
  test -s "$SPLINE_DIR/${split}_exact.parquet" || {
    echo "[job][error] missing spline truth: $SPLINE_DIR/${split}_exact.parquet"; exit 2;
  }
done

if [[ ! -s "$CATALOG_DIR/amortized_catalog_contract.json" ]]; then
  echo "[job] joining existing photometry and exact spline truth"
  python scripts/build_feniks_spline15d_amortized_catalog.py \
    --source-dir "$SOURCE_DIR" \
    --spline-dir "$SPLINE_DIR" \
    --out "$CATALOG_DIR"
fi

python - <<'PY'
import os

import equinox, euclid_dsps, jax, optax

print("[job] packages:", equinox.__version__, optax.__version__, euclid_dsps.__file__)
print("[job] backend/devices:", jax.default_backend(), jax.devices())
if os.environ["DATA_PARALLEL"] == "pmap" and len(jax.local_devices()) < 2:
    raise SystemExit("[job][error] DATA_PARALLEL=pmap requires at least two GPUs")
PY

if [[ "$SMOKE" == "1" ]]; then
  OUT_DIR="${OUT_DIR}_smoke"
  LIMIT="${LIMIT:-128}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-16}"
  JAX_BATCH_SIZE="${SMOKE_JAX_BATCH_SIZE:-8}"
  EPOCHS="${SMOKE_EPOCHS:-2}"
  N_SAMPLES="${SMOKE_N_SAMPLES:-1}"
fi
test ! -e "$OUT_DIR" || {
  echo "[job][error] output already exists: $OUT_DIR"; exit 2;
}

cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-train-diffsky
  --dataset "$CATALOG_DIR/train.parquet"
  --prior-checkpoint "$PRIOR_CHECKPOINT"
  --out "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples "$N_SAMPLES"
  --data-parallel "$DATA_PARALLEL"
  --validation-every "$VALIDATION_EVERY"
  --start-epoch "$START_EPOCH"
)
if [[ -n "$INITIAL_CHECKPOINT" ]]; then
  test -s "$INITIAL_CHECKPOINT" || {
    echo "[job][error] missing initial checkpoint: $INITIAL_CHECKPOINT"; exit 2;
  }
  cmd+=(--initial-checkpoint "$INITIAL_CHECKPOINT")
fi
if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ "$NO_PROGRESS" == "1" ]]; then
  cmd+=(--no-progress)
fi

echo "[job] start=$(date --iso-8601=seconds)"
echo "[job] prior=$PRIOR_CHECKPOINT catalog=$CATALOG_DIR out=$OUT_DIR"
echo "[job] batch=$BATCH_SIZE jax_batch=$JAX_BATCH_SIZE samples=$N_SAMPLES data_parallel=$DATA_PARALLEL validation_every=$VALIDATION_EVERY no_progress=$NO_PROGRESS"
echo "[job] initial_checkpoint=${INITIAL_CHECKPOINT:-none} start_epoch=$START_EPOCH optimizer_state_resumed=0"
printf '[job] command:'; printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "[job] done=$(date --iso-8601=seconds)"
