#!/bin/bash
#SBATCH --job-name=feniks_prior_ladder
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

# Stage-based H100 launcher for the FENIKS prior-learning ladder.
# Outputs are never deleted here. Set OVERWRITE=1 only for downstream commands
# that explicitly support overwrite semantics.

if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/euclid_dsps" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  elif [[ -n "${WORK:-}" ]]; then
    REPO_DIR="$WORK/euclid-dsps-shine"
  else
    REPO_DIR="$PWD"
  fi
fi
if [[ -z "${MINICONDA_PATH:-}" ]]; then
  if [[ -n "${WORK:-}" && -d "$WORK/miniconda3" ]]; then
    MINICONDA_PATH="$WORK/miniconda3"
  else
    MINICONDA_PATH="$(dirname "$REPO_DIR")/miniconda3"
  fi
fi

CONDA_ENV="${CONDA_ENV:-shine}"
STAGE="${STAGE:-ae_train}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_dsps_closure/train.parquet}"
TEST_DATASET="${TEST_DATASET:-Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -z "${LIMIT+x}" ]]; then LIMIT=20000; fi
if [[ -z "${INFER_LIMIT+x}" ]]; then INFER_LIMIT=5000; fi
if [[ -z "${JLENS_LIMIT+x}" ]]; then JLENS_LIMIT=1024; fi

BATCH_SIZE="${BATCH_SIZE:-512}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-512}"
DATA_PARALLEL="${DATA_PARALLEL:-auto}"
EPOCHS="${EPOCHS:-80}"
N_SAMPLES="${N_SAMPLES:-4}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-256}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-8192}"
KL_WEIGHT_MAX="${KL_WEIGHT_MAX:-}"
KL_ANNEALING_EPOCHS="${KL_ANNEALING_EPOCHS:-}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-outputs/runs/prior_diffsky_synthetic_feniks_full_realnvp/checkpoints/best.eqx}"
JLENS_BATCH_SIZE="${JLENS_BATCH_SIZE:-8}"
JLENS_NUM_SHARDS="${JLENS_NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
JLENS_SHARD_INDEX="${JLENS_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
JLENS_FINALIZE="${JLENS_FINALIZE:-0}"
SEED="${SEED:-260617}"

case "$STAGE" in
  ae_train|ae_infer|ae_jlens)
    CONFIG="${CONFIG:-configs/experiments/feniks_autoencoder_reconstruction_18d_h100.yaml}"
    RUN_NAME="${RUN_NAME:-feniks_autoencoder_reconstruction_18d_h100}"
    ;;
  joint_fixed_train|joint_fixed_infer|joint_fixed_jlens)
    CONFIG="${CONFIG:-configs/experiments/feniks_joint_realnvp_kl_fixed_18d_h100.yaml}"
    RUN_NAME="${RUN_NAME:-feniks_joint_realnvp_kl_fixed_18d_h100}"
    KL_WEIGHT_MAX="${KL_WEIGHT_MAX:-0.1}"
    KL_ANNEALING_EPOCHS="${KL_ANNEALING_EPOCHS:-0}"
    ;;
  joint_annealed_train|joint_annealed_infer|joint_annealed_jlens)
    CONFIG="${CONFIG:-configs/experiments/feniks_joint_realnvp_kl_annealed_18d_h100.yaml}"
    RUN_NAME="${RUN_NAME:-feniks_joint_realnvp_kl_annealed_18d_h100}"
    KL_WEIGHT_MAX="${KL_WEIGHT_MAX:-1.0}"
    KL_ANNEALING_EPOCHS="${KL_ANNEALING_EPOCHS:-30}"
    ;;
  supervised_prior_train|supervised_prior_report)
    CONFIG="${CONFIG:-configs/experiments/feniks_supervised_truth_realnvp_18d_h100.yaml}"
    RUN_NAME="${RUN_NAME:-prior_diffsky_synthetic_feniks_full_realnvp}"
    ;;
  supervised_frozen_train|supervised_frozen_infer|supervised_frozen_jlens)
    CONFIG="${CONFIG:-configs/experiments/feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100.yaml}"
    RUN_NAME="${RUN_NAME:-feniks_amortized_supervised_prior_frozen_kl_annealed_18d_h100}"
    KL_WEIGHT_MAX="${KL_WEIGHT_MAX:-1.0}"
    KL_ANNEALING_EPOCHS="${KL_ANNEALING_EPOCHS:-30}"
    ;;
  supervised_finetune_train|supervised_finetune_infer|supervised_finetune_jlens)
    CONFIG="${CONFIG:-configs/experiments/feniks_amortized_supervised_prior_finetune_18d_h100.yaml}"
    RUN_NAME="${RUN_NAME:-feniks_amortized_supervised_prior_finetune_18d_h100}"
    KL_WEIGHT_MAX="${KL_WEIGHT_MAX:-1.0}"
    KL_ANNEALING_EPOCHS="${KL_ANNEALING_EPOCHS:-30}"
    ;;
  *)
    echo "[job][error] unknown STAGE=$STAGE"
    exit 2
    ;;
esac

OUT_DIR="${OUT_DIR:-outputs/runs/${RUN_NAME}}"
TRAIN_RUN="${TRAIN_RUN:-$OUT_DIR}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_RUN}/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-${TRAIN_RUN}/feature_stats.json}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${STAGE}_${SLURM_JOB_ID:-manual}_${JLENS_SHARD_INDEX}.log}"

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
echo "[job] stage: $STAGE"
echo "[job] repo: $REPO_DIR"
echo "[job] config: $CONFIG"
echo "[job] dataset: $DATASET"
echo "[job] test_dataset: $TEST_DATASET"
echo "[job] out_dir: $OUT_DIR"
echo "[job] train_run: $TRAIN_RUN"
echo "[job] checkpoint: $CHECKPOINT"
echo "[job] feature_stats: $FEATURE_STATS"
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] infer_limit: ${INFER_LIMIT:-<full dataset>}"
echo "[job] jlens_limit: ${JLENS_LIMIT:-<full dataset>}"
echo "[job] jlens_shard: ${JLENS_SHARD_INDEX}/${JLENS_NUM_SHARDS}"
echo "[job] data_parallel: $DATA_PARALLEL"
echo "[job] overwrite: $OVERWRITE"
echo "[job] log_file: $LOG_FILE"

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
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "[job][missing] $label: $path"
    exit 2
  fi
  ls -lh "$path"
}

append_limit() {
  local -n arr_ref=$1
  local value="$2"
  if [[ -n "$value" ]]; then
    arr_ref+=(--limit "$value")
  fi
}

run_cmd() {
  echo "[job] command:"
  printf ' %q' "$@"
  echo
  "$@"
}

train_amortized() {
  require_file "$DATASET" "training dataset"
  local cmd=(
    python -m euclid_dsps.cli
    --config "$CONFIG"
    amortized-train-diffsky
    --dataset "$DATASET"
    --out "$OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --jax-batch-size "$JAX_BATCH_SIZE"
    --data-parallel "$DATA_PARALLEL"
    --epochs "$EPOCHS"
    --n-samples "$N_SAMPLES"
    --seed "$SEED"
  )
  append_limit cmd "$LIMIT"
  if [[ -n "$KL_WEIGHT_MAX" ]]; then
    cmd+=(--kl-weight-max "$KL_WEIGHT_MAX")
  fi
  if [[ -n "$KL_ANNEALING_EPOCHS" ]]; then
    cmd+=(--kl-annealing-epochs "$KL_ANNEALING_EPOCHS")
  fi
  case "$STAGE" in
    supervised_frozen_train|supervised_finetune_train)
      require_file "$PRIOR_CHECKPOINT" "supervised prior checkpoint"
      cmd+=(--prior-checkpoint "$PRIOR_CHECKPOINT")
      ;;
  esac
  run_cmd "${cmd[@]}"
}

infer_amortized() {
  require_file "$TEST_DATASET" "inference dataset"
  require_file "$CHECKPOINT" "amortized checkpoint"
  require_file "$FEATURE_STATS" "feature stats"
  local cmd=(
    python -m euclid_dsps.cli
    --config "$CONFIG"
    amortized-infer-diffsky
    --dataset "$TEST_DATASET"
    --checkpoint "$CHECKPOINT"
    --feature-stats "$FEATURE_STATS"
    --out "${OUT_DIR}_infer"
    --batch-size "$BATCH_SIZE"
    --jax-batch-size "$JAX_BATCH_SIZE"
    --posterior-samples "$POSTERIOR_SAMPLES"
    --prior-samples "$PRIOR_SAMPLES"
  )
  append_limit cmd "$INFER_LIMIT"
  run_cmd "${cmd[@]}"
}

run_jlens() {
  require_file "$TEST_DATASET" "J-lens dataset"
  require_file "$CHECKPOINT" "amortized checkpoint"
  require_file "$FEATURE_STATS" "feature stats"
  local cmd=(
    python -m euclid_dsps.cli
    --config "$CONFIG"
    amortized-jacobian-lens-diffsky
    --dataset "$TEST_DATASET"
    --checkpoint "$CHECKPOINT"
    --feature-stats "$FEATURE_STATS"
    --out "${OUT_DIR}_jlens"
    --batch-size "$JLENS_BATCH_SIZE"
    --selection-mode random
    --selection-seed "$SEED"
    --mode both
    --include-prior-score
    --include-ae-lens
    --num-shards "$JLENS_NUM_SHARDS"
    --shard-index "$JLENS_SHARD_INDEX"
    --resume
  )
  append_limit cmd "$JLENS_LIMIT"
  run_cmd "${cmd[@]}"
  if [[ "$JLENS_FINALIZE" == "1" && "$JLENS_SHARD_INDEX" == "0" ]]; then
    run_cmd \
      python -m euclid_dsps.cli \
      --config "$CONFIG" \
      amortized-finalize-jacobian-lens \
      --out "${OUT_DIR}_jlens"
  fi
}

train_supervised_prior() {
  require_file "$DATASET" "truth-prior dataset"
  local cmd=(
    python -m euclid_dsps.cli
    --config "$CONFIG"
    diffsky-train-supervised-prior
    --dataset "$DATASET"
    --out "$OUT_DIR"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --seed "$SEED"
    --data-parallel "$DATA_PARALLEL"
  )
  append_limit cmd "$LIMIT"
  run_cmd "${cmd[@]}"
}

report_supervised_prior() {
  require_file "$DATASET" "truth-prior report dataset"
  require_file "${OUT_DIR}/checkpoints/best.eqx" "supervised prior checkpoint"
  run_cmd \
    python -m euclid_dsps.cli \
    --config "$CONFIG" \
    diffsky-supervised-prior-report \
    --run "$OUT_DIR" \
    --dataset "$DATASET" \
    --out "${OUT_DIR}/report"
}

case "$STAGE" in
  ae_train|joint_fixed_train|joint_annealed_train|supervised_frozen_train|supervised_finetune_train)
    train_amortized
    ;;
  ae_infer|joint_fixed_infer|joint_annealed_infer|supervised_frozen_infer|supervised_finetune_infer)
    infer_amortized
    ;;
  ae_jlens|joint_fixed_jlens|joint_annealed_jlens|supervised_frozen_jlens|supervised_finetune_jlens)
    run_jlens
    ;;
  supervised_prior_train)
    train_supervised_prior
    ;;
  supervised_prior_report)
    report_supervised_prior
    ;;
esac

echo "[job] done: $(date --iso-8601=seconds)"
