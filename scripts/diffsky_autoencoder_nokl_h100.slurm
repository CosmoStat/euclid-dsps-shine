#!/bin/bash
#SBATCH --job-name=diffsky_nokl_h100
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_autoencoder_nokl_h100.yaml}"
RUN_NAME="${RUN_NAME:-diffsky_autoencoder_nokl_z035_$(date +%Y%m%d_%H%M%S)}"
TRAIN_OUT="${TRAIN_OUT:-outputs/runs/${RUN_NAME}}"
INFER_OUT="${INFER_OUT:-outputs/runs/${RUN_NAME}_infer}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-manual}.log}"

LIMIT="${LIMIT:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-12}"
N_SAMPLES="${N_SAMPLES:-1}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-64}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-512}"
SEED="${SEED:-42}"
INFER_LIMIT="${INFER_LIMIT:-${LIMIT}}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$(dirname "$TRAIN_OUT")" "$(dirname "$INFER_OUT")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[job] start: $(date --iso-8601=seconds)"
echo "[job] host: $(hostname)"
echo "[job] repo: $REPO_DIR"
echo "[job] config: $CONFIG"
echo "[job] train_out: $TRAIN_OUT"
echo "[job] infer_out: $INFER_OUT"
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] infer_limit: ${INFER_LIMIT:-<full dataset>}"
echo "[job] epochs: $EPOCHS"
echo "[job] batch_size: $BATCH_SIZE"
echo "[job] jax_batch_size: $JAX_BATCH_SIZE"
echo "[job] posterior_samples: $POSTERIOR_SAMPLES"

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - <<'PY'
import os
from pathlib import Path

import euclid_dsps
import jax

repo = Path(os.environ["REPO_DIR"]).resolve()
pkg = Path(euclid_dsps.__file__).resolve()
if repo not in pkg.parents:
    raise SystemExit(f"[job][error] euclid_dsps imported from {pkg}, not {repo}")
print("[job] euclid_dsps:", pkg)
print("[job] jax_backend:", jax.default_backend())
print("[job] jax_devices:", jax.devices())
PY
nvidia-smi || true
git status --short --branch

train_cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-train-diffsky
  --out "$TRAIN_OUT"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples "$N_SAMPLES"
  --kl-annealing-epochs 1
  --kl-weight-max 0.0
  --seed "$SEED"
)
if [[ -n "$LIMIT" ]]; then
  train_cmd+=(--limit "$LIMIT")
fi

echo "[job] train command: ${train_cmd[*]}"
"${train_cmd[@]}"

CHECKPOINT="${CHECKPOINT:-${TRAIN_OUT}/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-${TRAIN_OUT}/feature_stats.json}"
for path in "$CHECKPOINT" "$FEATURE_STATS"; do
  if [[ ! -s "$path" ]]; then
    echo "[job][missing] $path"
    exit 2
  fi
done

infer_cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-infer-diffsky
  --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS"
  --out "$INFER_OUT"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --posterior-samples "$POSTERIOR_SAMPLES"
  --prior-samples "$PRIOR_SAMPLES"
  --seed "$SEED"
)
if [[ -n "$INFER_LIMIT" ]]; then
  infer_cmd+=(--limit "$INFER_LIMIT")
fi

echo "[job] infer command: ${infer_cmd[*]}"
"${infer_cmd[@]}"

echo "[job] done: $(date --iso-8601=seconds)"
