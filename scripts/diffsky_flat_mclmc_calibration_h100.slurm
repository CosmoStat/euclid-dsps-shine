#!/bin/bash
#SBATCH --job-name=diffsky_flat_mclmc
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_autoencoder_nokl_deterministic_h100.yaml}"
ROWSET_FILE="${ROWSET_FILE:-outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt}"
OUT_ROOT="${OUT_ROOT:-outputs/runs/diffsky_flat_mclmc_calibration}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-4}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WARMUP="${NUM_WARMUP:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
NUM_CHAINS="${NUM_CHAINS:-1}"
MCLMC_L="${MCLMC_L:-}"
MCLMC_STEP_SIZE="${MCLMC_STEP_SIZE:-}"
MCLMC_PROGRESS_CHUNK_SIZE="${MCLMC_PROGRESS_CHUNK_SIZE:-64}"
SEED="${SEED:-42}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$OUT_ROOT/shards" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/diffsky_flat_mclmc_${SLURM_JOB_ID:-manual}_${TASK_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[mclmc] start: $(date --iso-8601=seconds)"
echo "[mclmc] task: $TASK_ID / $SHARD_COUNT"
echo "[mclmc] rowset: $ROWSET_FILE"
echo "[mclmc] out_root: $OUT_ROOT"

if [[ ! -s "$ROWSET_FILE" ]]; then
  echo "[mclmc][missing] $ROWSET_FILE"
  exit 2
fi

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - <<PY
from pathlib import Path

import numpy as np

rowset = Path("$ROWSET_FILE")
out = Path("$OUT_ROOT/shards")
indices = np.loadtxt(rowset, dtype=np.int64, ndmin=1)
chunks = np.array_split(indices, int("$SHARD_COUNT"))
for idx, chunk in enumerate(chunks):
    path = out / f"rowset_shard_{idx:03d}.txt"
    np.savetxt(path, chunk.astype(np.int64), fmt="%d")
    print(f"[mclmc] shard {idx}: rows={len(chunk)} -> {path}")
PY

SHARD_FILE="$OUT_ROOT/shards/rowset_shard_$(printf '%03d' "$TASK_ID").txt"
OUT_DIR="$OUT_ROOT/shard_$(printf '%03d' "$TASK_ID")"

cmd=(python -m euclid_dsps.cli --config "$CONFIG" posterior --all --row-indices-file "$SHARD_FILE" --out "$OUT_DIR" --batch-size "$BATCH_SIZE" --sampler mclmc --num-warmup "$NUM_WARMUP" --num-samples "$NUM_SAMPLES" --num-chains "$NUM_CHAINS" --mclmc-progress-chunk-size "$MCLMC_PROGRESS_CHUNK_SIZE" --mcmc-init-strategy map_jitter)
if [[ -n "$MCLMC_L" ]]; then
  cmd+=(--mclmc-l "$MCLMC_L")
fi
if [[ -n "$MCLMC_STEP_SIZE" ]]; then
  cmd+=(--mclmc-step-size "$MCLMC_STEP_SIZE")
fi

echo "[mclmc] command:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "[mclmc] end: $(date --iso-8601=seconds)"
