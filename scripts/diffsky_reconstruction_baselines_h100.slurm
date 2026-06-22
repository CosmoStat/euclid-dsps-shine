#!/bin/bash
#SBATCH --job-name=diffsky_recon_h100
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
TASK="${TASK:-rowsets}"

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_autoencoder_nokl_h100.yaml}"
MAP_CONFIG="${MAP_CONFIG:-configs/diffsky_hltds_04_14_simple_gpu.yaml}"
MCLMC_CONFIG="${MCLMC_CONFIG:-$MAP_CONFIG}"
PRIOR_CONFIG="${PRIOR_CONFIG:-configs/prior_diffsky_hltds_supervised_basic_realnvp.yaml}"
SUPERVISED_AMORTIZED_CONFIG="${SUPERVISED_AMORTIZED_CONFIG:-configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml}"

REF_TRAIN_RUN="${REF_TRAIN_RUN:-outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128}"
REF_INFER_RUN="${REF_INFER_RUN:-outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128_infer}"
ROWSET_DIR="${ROWSET_DIR:-outputs/rowsets/diffsky_autoencoder_nokl_m5sys_z035_rand20k}"
ROWSET="${ROWSET:-worst_1000}"
ROW_INDICES_FILE="${ROW_INDICES_FILE:-${ROWSET_DIR}/${ROWSET}.txt}"

CHECKPOINT="${CHECKPOINT:-${REF_TRAIN_RUN}/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-${REF_TRAIN_RUN}/feature_stats.json}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/diffsky_reconstruction}"
NN_OUT="${NN_OUT:-${RUN_ROOT}_nn_${ROWSET}}"
MAP_OUT="${MAP_OUT:-${RUN_ROOT}_map_${ROWSET}}"
MCLMC_OUT="${MCLMC_OUT:-${RUN_ROOT}_mclmc_${ROWSET}}"
COMPARE_OUT="${COMPARE_OUT:-${RUN_ROOT}_compare_${ROWSET}}"
TRAIN_OUT="${TRAIN_OUT:-${RUN_ROOT}_${TASK}_${ROWSET}}"
PRIOR_OUT="${PRIOR_OUT:-outputs/runs/diffsky_supervised_prior_basic}"

BATCH_SIZE="${BATCH_SIZE:-128}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-128}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-64}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-512}"
DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}"
PRIOR_PREDICTIVE_BATCH_SIZE="${PRIOR_PREDICTIVE_BATCH_SIZE:-256}"
SEED="${SEED:-42}"

MAP_BATCH_SIZE="${MAP_BATCH_SIZE:-128}"
MAP_MAXITER="${MAP_MAXITER:-200}"

MCLMC_BATCH_SIZE="${MCLMC_BATCH_SIZE:-1}"
MCLMC_NUM_WARMUP="${MCLMC_NUM_WARMUP:-64}"
MCLMC_NUM_SAMPLES="${MCLMC_NUM_SAMPLES:-128}"
MCLMC_POSTERIOR_PREDICTIVE_BATCH_SIZE="${MCLMC_POSTERIOR_PREDICTIVE_BATCH_SIZE:-32}"

EPOCHS="${EPOCHS:-30}"
N_SAMPLES="${N_SAMPLES:-1}"
KL_WEIGHT_MAX="${KL_WEIGHT_MAX:-0.05}"
KL_ANNEALING_EPOCHS="${KL_ANNEALING_EPOCHS:-5}"
INPUT_NOISE_SIGMA="${INPUT_NOISE_SIGMA:-1.0}"

LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/diffsky_reconstruction_${TASK}_${ROWSET}_${SLURM_JOB_ID:-manual}.log}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export REPO_DIR

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$(dirname "$NN_OUT")" "$(dirname "$ROWSET_DIR")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[job] start: $(date --iso-8601=seconds)"
echo "[job] task: $TASK"
echo "[job] repo: $REPO_DIR"
echo "[job] rowset: $ROWSET"
echo "[job] row_indices_file: $ROW_INDICES_FILE"
echo "[job] ref_train_run: $REF_TRAIN_RUN"
echo "[job] ref_infer_run: $REF_INFER_RUN"

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

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "[job][missing] $path"
    exit 2
  fi
}

require_rowset() {
  require_file "$ROW_INDICES_FILE"
}

case "$TASK" in
  rowsets)
    python -m euclid_dsps.cli \
      --config "$CONFIG" \
      diffsky-build-reconstruction-rowsets \
      --train-run "$REF_TRAIN_RUN" \
      --infer-run "$REF_INFER_RUN" \
      --out "$ROWSET_DIR" \
      --worst-size 500 \
      --worst-size 1000
    ;;

  nn-infer)
    require_rowset
    require_file "$CHECKPOINT"
    require_file "$FEATURE_STATS"
    python -m euclid_dsps.cli \
      --config "$CONFIG" \
      amortized-infer-diffsky \
      --checkpoint "$CHECKPOINT" \
      --feature-stats "$FEATURE_STATS" \
      --out "$NN_OUT" \
      --row-indices-file "$ROW_INDICES_FILE" \
      --batch-size "$BATCH_SIZE" \
      --jax-batch-size "$JAX_BATCH_SIZE" \
      --posterior-samples "$POSTERIOR_SAMPLES" \
      --prior-samples "$PRIOR_SAMPLES" \
      --decoder-sample-chunk-size "$DECODER_SAMPLE_CHUNK_SIZE" \
      --prior-predictive-batch-size "$PRIOR_PREDICTIVE_BATCH_SIZE" \
      --seed "$SEED"
    ;;

  map)
    require_rowset
    python -m euclid_dsps.cli \
      --config "$MAP_CONFIG" \
      fit \
      --out "$MAP_OUT" \
      --row-indices-file "$ROW_INDICES_FILE" \
      --batch-size "$MAP_BATCH_SIZE" \
      --fit-maxiter "$MAP_MAXITER" \
      --sed-samples 0 \
      --reporting-level light
    ;;

  mclmc)
    require_rowset
    python -m euclid_dsps.cli \
      --config "$MCLMC_CONFIG" \
      posterior \
      --out "$MCLMC_OUT" \
      --row-indices-file "$ROW_INDICES_FILE" \
      --batch-size "$MCLMC_BATCH_SIZE" \
      --sampler mclmc \
      --num-warmup "$MCLMC_NUM_WARMUP" \
      --num-samples "$MCLMC_NUM_SAMPLES" \
      --posterior-predictive-batch-size "$MCLMC_POSTERIOR_PREDICTIVE_BATCH_SIZE" \
      --seed "$SEED" \
      --no-progress
    ;;

  compare)
    require_rowset
    python -m euclid_dsps.cli \
      --config "$CONFIG" \
      diffsky-compare-reconstruction \
      --rowset "$ROW_INDICES_FILE" \
      --run "nn=${NN_OUT}" \
      --run "map=${MAP_OUT}" \
      --run "mclmc=${MCLMC_OUT}" \
      --out "$COMPARE_OUT"
    ;;

  train-noise)
    require_file "${ROWSET_DIR}/reference_train.txt"
    require_file "${ROWSET_DIR}/reference_validation.txt"
    python -m euclid_dsps.cli \
      --config "$CONFIG" \
      amortized-train-diffsky \
      --out "$TRAIN_OUT" \
      --train-indices-file "${ROWSET_DIR}/reference_train.txt" \
      --validation-indices-file "${ROWSET_DIR}/reference_validation.txt" \
      --batch-size "$BATCH_SIZE" \
      --jax-batch-size "$JAX_BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --n-samples "$N_SAMPLES" \
      --kl-weight-max 0.0 \
      --kl-annealing-epochs 1 \
      --input-noise-sigma-scale "$INPUT_NOISE_SIGMA" \
      --seed "$SEED"
    ;;

  train-kl)
    require_file "${ROWSET_DIR}/reference_train.txt"
    require_file "${ROWSET_DIR}/reference_validation.txt"
    python -m euclid_dsps.cli \
      --config "$CONFIG" \
      amortized-train-diffsky \
      --out "$TRAIN_OUT" \
      --train-indices-file "${ROWSET_DIR}/reference_train.txt" \
      --validation-indices-file "${ROWSET_DIR}/reference_validation.txt" \
      --batch-size "$BATCH_SIZE" \
      --jax-batch-size "$JAX_BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --n-samples "$N_SAMPLES" \
      --kl-weight-max "$KL_WEIGHT_MAX" \
      --kl-annealing-epochs "$KL_ANNEALING_EPOCHS" \
      --seed "$SEED"
    ;;

  train-supervised-prior)
    require_file "${ROWSET_DIR}/reference_20k.txt"
    python -m euclid_dsps.cli \
      --config "$PRIOR_CONFIG" \
      diffsky-train-supervised-prior \
      --out "$PRIOR_OUT" \
      --row-indices-file "${ROWSET_DIR}/reference_20k.txt" \
      --epochs "$EPOCHS" \
      --batch-size "${PRIOR_BATCH_SIZE:-512}" \
      --seed "$SEED"
    ;;

  train-supervised-prior-nn)
    require_file "${ROWSET_DIR}/reference_train.txt"
    require_file "${ROWSET_DIR}/reference_validation.txt"
    require_file "${PRIOR_OUT}/checkpoints/best.eqx"
    python -m euclid_dsps.cli \
      --config "$SUPERVISED_AMORTIZED_CONFIG" \
      amortized-train-diffsky \
      --out "$TRAIN_OUT" \
      --train-indices-file "${ROWSET_DIR}/reference_train.txt" \
      --validation-indices-file "${ROWSET_DIR}/reference_validation.txt" \
      --prior-checkpoint "${PRIOR_OUT}/checkpoints/best.eqx" \
      --batch-size "$BATCH_SIZE" \
      --jax-batch-size "$JAX_BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --n-samples "$N_SAMPLES" \
      --seed "$SEED"
    ;;

  *)
    echo "[job][error] unknown TASK=$TASK"
    echo "[job][hint] TASK must be rowsets, nn-infer, map, mclmc, compare, train-noise, train-kl, train-supervised-prior, or train-supervised-prior-nn"
    exit 2
    ;;
esac

echo "[job] done: $(date --iso-8601=seconds)"
