#!/bin/bash
#SBATCH --job-name=diffsky_infer_h100
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

# Sharded checkpoint-only amortized inference and posterior-predictive
# normalized residual diagnostics for the held-out FENIKS closure split.
#
# Default use case: inspect the completed FENIKS NN+DSPS+NF training run.
# Redshift is inferred; closure truth columns are used only for diagnostics.
#
# Submit a 5k-object diagnostic:
#
#   sbatch scripts/diffsky_amortized_infer_h100.slurm
#
# Full-dataset inference after diagnostics look sane:
#
#   sbatch --export=ALL,LIMIT=,RUN_NAME=amortized_diffsky_synthetic_feniks_full_test_infer \
#     scripts/diffsky_amortized_infer_h100.slurm
#
# Switch to a specific epoch checkpoint:
#
#   sbatch --export=ALL,CHECKPOINT=outputs/runs/.../checkpoints/epoch_0015.eqx \
#     scripts/diffsky_amortized_infer_h100.slurm

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

CONFIG="${CONFIG:-configs/amortized_diffsky_synthetic_feniks_full_gpu.yaml}"
DATASET="${DATASET:-Data/diffsky/synthetic/feniks_260617_dsps_closure/test.parquet}"
TRAIN_RUN="${TRAIN_RUN:-outputs/runs/amortized_diffsky_synthetic_feniks_full}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_RUN}/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-${TRAIN_RUN}/feature_stats.json}"

RUN_NAME="${RUN_NAME:-amortized_diffsky_synthetic_feniks_full_test_infer}"
OUT_DIR="${OUT_DIR:-outputs/runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-manual}.log}"

BATCH_SIZE="${BATCH_SIZE:-256}"
JAX_BATCH_SIZE="${JAX_BATCH_SIZE:-256}"
DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}"
PRIOR_PREDICTIVE_BATCH_SIZE="${PRIOR_PREDICTIVE_BATCH_SIZE:-256}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-256}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-8192}"
SEED="${SEED:-260717}"
LIMIT="${LIMIT:-}"
SELECTION_MODE="${SELECTION_MODE:-sequential}"
STRATIFIED_STRATEGY="${STRATIFIED_STRATEGY:-balanced}"
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
mkdir -p "$LOG_DIR" "$(dirname "$OUT_DIR")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[job] start: $(date --iso-8601=seconds)"
echo "[job] host: $(hostname)"
echo "[job] repo: $REPO_DIR"
echo "[job] run_name: $RUN_NAME"
echo "[job] out_dir: $OUT_DIR"
echo "[job] log_file: $LOG_FILE"
echo "[job] config: $CONFIG"
echo "[job] dataset: $DATASET"
echo "[job] checkpoint: $CHECKPOINT"
echo "[job] feature_stats: $FEATURE_STATS"
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] batch_size: $BATCH_SIZE"
echo "[job] jax_batch_size: $JAX_BATCH_SIZE"
echo "[job] decoder_sample_chunk_size: $DECODER_SAMPLE_CHUNK_SIZE"
echo "[job] prior_predictive_batch_size: $PRIOR_PREDICTIVE_BATCH_SIZE"
echo "[job] posterior_samples: $POSTERIOR_SAMPLES"
echo "[job] prior_samples: $PRIOR_SAMPLES"
echo "[job] seed: $SEED"
echo "[job] selection_mode: $SELECTION_MODE"
echo "[job] stratified_strategy: $STRATIFIED_STRATEGY"
echo "[job] selection_seed: $SELECTION_SEED"
echo "[job] outputs: sharded, resumable; compact posterior-predictive residual summaries enabled"
echo "[job] large posterior predictive flux and per-sample residual parquet outputs disabled"

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
    raise SystemExit(
        f"[job][error] euclid_dsps imported from {pkg}, not from repo {repo}. "
        "Check PYTHONPATH or the conda environment."
    )
print("[job] euclid_dsps:", pkg)
print("[job] jax_backend:", jax.default_backend())
print("[job] jax_devices:", jax.devices())
PY
nvidia-smi || true

for path in "$DATASET" "$CHECKPOINT" "$FEATURE_STATS"; do
  if [[ -s "$path" ]]; then
    ls -lh "$path"
  else
    echo "[job][missing] $path"
    exit 2
  fi
done

cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-infer-diffsky
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS"
  --out "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --jax-batch-size "$JAX_BATCH_SIZE"
  --posterior-samples "$POSTERIOR_SAMPLES"
  --prior-samples "$PRIOR_SAMPLES"
  --decoder-sample-chunk-size "$DECODER_SAMPLE_CHUNK_SIZE"
  --prior-predictive-batch-size "$PRIOR_PREDICTIVE_BATCH_SIZE"
  --seed "$SEED"
  --selection-mode "$SELECTION_MODE"
  --stratified-strategy "$STRATIFIED_STRATEGY"
  --selection-seed "$SELECTION_SEED"
  --shard-outputs
  --no-posterior-predictive
  --no-residual-samples
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[job] command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

finalize_cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  amortized-finalize-inference
  --out "$OUT_DIR"
)
if [[ -n "$LIMIT" ]]; then
  finalize_cmd+=(--limit "$LIMIT")
fi

echo "[job] finalize command:"
printf ' %q' "${finalize_cmd[@]}"
echo
"${finalize_cmd[@]}"

echo "[job] inference summary: $OUT_DIR/inference_summary.json"
echo "[job] shard manifest: $OUT_DIR/posterior_shards_manifest.json"
echo "[job] end: $(date --iso-8601=seconds)"
