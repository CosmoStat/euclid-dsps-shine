#!/bin/bash
#SBATCH --job-name=diffsky_nokl_h100
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

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_autoencoder_nokl_h100.yaml}"
DATASET="${DATASET:-Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_continuous_lowz_fluxerr.parquet}"
SOURCE_DATASET="${SOURCE_DATASET:-Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_m5depth.parquet}"
REDSHIFT_MIN="${REDSHIFT_MIN:-0.0}"
REDSHIFT_MAX="${REDSHIFT_MAX:-0.35}"
ERROR_MODEL="${ERROR_MODEL:-m5_depth}"
ERROR_SNR="${ERROR_SNR:-50}"
SIGMA_SYS_MAG="${SIGMA_SYS_MAG:-0.005}"
FORCE_REBUILD_DATASET="${FORCE_REBUILD_DATASET:-0}"
RUN_NAME="${RUN_NAME:-diffsky_autoencoder_nokl_z035_$(date +%Y%m%d_%H%M%S)}"
TRAIN_OUT="${TRAIN_OUT:-outputs/runs/${RUN_NAME}}"
INFER_OUT="${INFER_OUT:-outputs/runs/${RUN_NAME}_infer}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-manual}.log}"

LIMIT="${LIMIT-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-20}"
N_SAMPLES="${N_SAMPLES:-1}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-64}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-512}"
DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}"
PRIOR_PREDICTIVE_BATCH_SIZE="${PRIOR_PREDICTIVE_BATCH_SIZE:-256}"
SEED="${SEED:-42}"
INFER_LIMIT="${INFER_LIMIT-${LIMIT}}"
SELECTION_MODE="${SELECTION_MODE:-}"
SELECTION_SEED="${SELECTION_SEED:-$SEED}"

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
echo "[job] dataset: $DATASET"
echo "[job] source_dataset: $SOURCE_DATASET"
echo "[job] redshift_range: [$REDSHIFT_MIN, $REDSHIFT_MAX]"
echo "[job] error_model: $ERROR_MODEL snr_if_fractional=$ERROR_SNR sigma_sys_mag=$SIGMA_SYS_MAG"
echo "[job] force_rebuild_dataset: $FORCE_REBUILD_DATASET"
echo "[job] train_out: $TRAIN_OUT"
echo "[job] infer_out: $INFER_OUT"
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] infer_limit: ${INFER_LIMIT:-<full dataset>}"
echo "[job] epochs: $EPOCHS"
echo "[job] batch_size: $BATCH_SIZE"
echo "[job] jax_batch_size: $JAX_BATCH_SIZE"
echo "[job] posterior_samples: $POSTERIOR_SAMPLES"
echo "[job] prior_samples: $PRIOR_SAMPLES"
echo "[job] decoder_sample_chunk_size: $DECODER_SAMPLE_CHUNK_SIZE"
echo "[job] prior_predictive_batch_size: $PRIOR_PREDICTIVE_BATCH_SIZE"
echo "[job] selection_mode: ${SELECTION_MODE:-<config>}"
echo "[job] selection_seed: $SELECTION_SEED"

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

ensure_diffsky_dataset() {
  case "$FORCE_REBUILD_DATASET" in
    1|true|TRUE|yes|YES|y|Y)
      echo "[job] force rebuild requested"
      ;;
    *)
      if [[ -s "$DATASET" ]]; then
        echo "[job] dataset exists"
        ls -lh "$DATASET"
        return
      fi
      ;;
  esac
  if [[ -s "$DATASET" ]]; then
    echo "[job] rebuilding existing dataset from source subset"
  else
    echo "[job] dataset missing, rebuilding from source subset"
  fi
  if [[ ! -s "$SOURCE_DATASET" ]]; then
    cat <<MSG
[job][error] Missing SOURCE_DATASET.
[job][error] DATASET=$DATASET
[job][error] SOURCE_DATASET=$SOURCE_DATASET
[job][hint] Either copy the local subset parquet to Jean-Zay or build the
[job][hint] full source parquet first with diffsky-prepare-dataset.
MSG
    exit 2
  fi
  mkdir -p "$(dirname "$DATASET")"
  subset_cmd=(
    python -m euclid_dsps.cli diffsky-redshift-subset
    --dataset "$SOURCE_DATASET"
    --out "$DATASET"
    --redshift-min "$REDSHIFT_MIN"
    --redshift-max "$REDSHIFT_MAX"
    --error-model "$ERROR_MODEL"
    --snr "$ERROR_SNR"
  )
  if [[ "$ERROR_MODEL" == "m5_depth" ]]; then
    subset_cmd+=(--sigma-sys-mag "$SIGMA_SYS_MAG")
  fi
  echo "[job] subset command: ${subset_cmd[*]}"
  "${subset_cmd[@]}"
  ls -lh "$DATASET"
}

ensure_diffsky_dataset

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
if [[ -n "$SELECTION_MODE" ]]; then
  train_cmd+=(--selection-mode "$SELECTION_MODE")
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
  --decoder-sample-chunk-size "$DECODER_SAMPLE_CHUNK_SIZE"
  --prior-predictive-batch-size "$PRIOR_PREDICTIVE_BATCH_SIZE"
  --seed "$SEED"
)
if [[ -n "$INFER_LIMIT" ]]; then
  infer_cmd+=(--limit "$INFER_LIMIT")
fi
if [[ -n "$SELECTION_MODE" ]]; then
  infer_cmd+=(--selection-mode "$SELECTION_MODE" --selection-seed "$SELECTION_SEED")
fi

echo "[job] infer command: ${infer_cmd[*]}"
"${infer_cmd[@]}"

echo "[job] done: $(date --iso-8601=seconds)"
