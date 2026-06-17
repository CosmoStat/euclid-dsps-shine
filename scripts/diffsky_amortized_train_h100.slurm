#!/bin/bash
#SBATCH --job-name=diffsky_train_h100
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

# Train-only H100 job for Diffsky amortized photo-z experiments.
# It intentionally does not run inference or full-validation reports; use
# scripts/diffsky_amortized_infer_h100.slurm afterwards on the chosen checkpoint.

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_joint_realnvp_unsup_stable_h100.yaml}"
RUN_NAME="${RUN_NAME:-diffsky_realnvp_unsup_stable_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-manual}.log}"

BATCH_SIZE="${BATCH_SIZE:-128}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-20}"
N_SAMPLES="${N_SAMPLES:-4}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
KL_ANNEALING_EPOCHS="${KL_ANNEALING_EPOCHS:-15}"
KL_WEIGHT_MAX="${KL_WEIGHT_MAX:-0.02}"
VALIDATION_EVERY="${VALIDATION_EVERY:-1}"
BEST_CHECKPOINT_MIN_EPOCH="${BEST_CHECKPOINT_MIN_EPOCH:-}"
SELECTION_MODE="${SELECTION_MODE:-}"
STRATIFIED_STRATEGY="${STRATIFIED_STRATEGY:-}"
PRIOR_FREEZE_EPOCHS="${PRIOR_FREEZE_EPOCHS:-3}"
PRIOR_UPDATE_SCHEDULE="${PRIOR_UPDATE_SCHEDULE:-alternating}"
LIKELIHOOD_TEMPERATURE_INITIAL="${LIKELIHOOD_TEMPERATURE_INITIAL:-4.0}"
LIKELIHOOD_TEMPERATURE_FINAL="${LIKELIHOOD_TEMPERATURE_FINAL:-1.0}"
LIKELIHOOD_TEMPERATURE_ANNEALING_EPOCHS="${LIKELIHOOD_TEMPERATURE_ANNEALING_EPOCHS:-10}"
ENTROPY_FLOOR_WEIGHT="${ENTROPY_FLOOR_WEIGHT:-0.10}"
ENTROPY_FLOOR_MIN_LOG_STD="${ENTROPY_FLOOR_MIN_LOG_STD:--3.2}"

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
mkdir -p "$LOG_DIR" "$(dirname "$OUT_DIR")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[job] start: $(date --iso-8601=seconds)"
echo "[job] host: $(hostname)"
echo "[job] repo: $REPO_DIR"
echo "[job] run_name: $RUN_NAME"
echo "[job] out_dir: $OUT_DIR"
echo "[job] log_file: $LOG_FILE"
echo "[job] config: $CONFIG"
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] batch_size: $BATCH_SIZE"
echo "[job] jax_batch_size: $JAX_BATCH_SIZE"
echo "[job] epochs: $EPOCHS"
echo "[job] n_samples: $N_SAMPLES"
echo "[job] kl_annealing_epochs: $KL_ANNEALING_EPOCHS"
echo "[job] kl_weight_max: $KL_WEIGHT_MAX"
echo "[job] best_checkpoint_min_epoch: ${BEST_CHECKPOINT_MIN_EPOCH:-<config>}"
echo "[job] selection_mode: ${SELECTION_MODE:-<config>}"
echo "[job] stratified_strategy: ${STRATIFIED_STRATEGY:-<config>}"
echo "[job] prior_freeze_epochs: $PRIOR_FREEZE_EPOCHS"
echo "[job] prior_update_schedule: $PRIOR_UPDATE_SCHEDULE"
echo "[job] likelihood_temperature: $LIKELIHOOD_TEMPERATURE_INITIAL -> $LIKELIHOOD_TEMPERATURE_FINAL over $LIKELIHOOD_TEMPERATURE_ANNEALING_EPOCHS epochs"
echo "[job] entropy_floor_weight: $ENTROPY_FLOOR_WEIGHT"
echo "[job] entropy_floor_min_log_std: $ENTROPY_FLOOR_MIN_LOG_STD"
echo "[job] seed: $SEED"

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[job] python: $(which python)"
python - <<'PY'
import os
from pathlib import Path

import euclid_dsps
import jax

repo = Path(os.environ["REPO_DIR"]).resolve()
pkg = Path(euclid_dsps.__file__).resolve()
if repo not in pkg.parents:
    raise SystemExit(
        f"[job][error] euclid_dsps imported from {pkg}, not from repo {repo}. "
        "Check PYTHONPATH or the conda environment."
    )
print("[job] euclid_dsps:", pkg)
print("[job] jax_backend:", jax.default_backend())
print("[job] jax_devices:", jax.devices())
PY
nvidia-smi || true
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD

cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-train-diffsky
  --out "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --epochs "$EPOCHS"
  --n-samples "$N_SAMPLES"
  --kl-annealing-epochs "$KL_ANNEALING_EPOCHS"
  --kl-weight-max "$KL_WEIGHT_MAX"
  --validation-every "$VALIDATION_EVERY"
  --prior-freeze-epochs "$PRIOR_FREEZE_EPOCHS"
  --prior-update-schedule "$PRIOR_UPDATE_SCHEDULE"
  --likelihood-temperature-initial "$LIKELIHOOD_TEMPERATURE_INITIAL"
  --likelihood-temperature-final "$LIKELIHOOD_TEMPERATURE_FINAL"
  --likelihood-temperature-annealing-epochs "$LIKELIHOOD_TEMPERATURE_ANNEALING_EPOCHS"
  --entropy-floor-weight "$ENTROPY_FLOOR_WEIGHT"
  --entropy-floor-min-log-std "$ENTROPY_FLOOR_MIN_LOG_STD"
  --seed "$SEED"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ -n "$BEST_CHECKPOINT_MIN_EPOCH" ]]; then
  cmd+=(--best-checkpoint-min-epoch "$BEST_CHECKPOINT_MIN_EPOCH")
fi
if [[ -n "$SELECTION_MODE" ]]; then
  cmd+=(--selection-mode "$SELECTION_MODE")
fi
if [[ -n "$STRATIFIED_STRATEGY" ]]; then
  cmd+=(--stratified-strategy "$STRATIFIED_STRATEGY")
fi

echo "[job] command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "[job] training summary: $OUT_DIR/training_summary.json"
echo "[job] best checkpoint: $OUT_DIR/checkpoints/best.eqx"
echo "[job] last checkpoint: $OUT_DIR/checkpoints/last.eqx"
echo "[job] end: $(date --iso-8601=seconds)"
