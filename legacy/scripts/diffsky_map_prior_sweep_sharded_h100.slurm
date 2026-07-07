#!/bin/bash
#SBATCH --job-name=diffsky_map_sweep
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-7
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_joint_realnvp_kl_annealed_zscale005_tau2safe_h100.yaml}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
FEATURE_STATS="${FEATURE_STATS:?FEATURE_STATS is required}"
ROWSET_FILE="${ROWSET_FILE:-outputs/rowsets/diffsky_hltds_nokl_reference/balanced20k.txt}"
OUT_ROOT="${OUT_ROOT:-outputs/runs/diffsky_map_prior_sweep_sharded}"
LOG_DIR="${LOG_DIR:-outputs/logs}"

WEIGHTS="${WEIGHTS:-0}"
SHARD_COUNT="${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-8}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_STARTS="${N_STARTS:-4}"
START_CHUNK_SIZE="${START_CHUNK_SIZE:-4}"
MAXITER="${MAXITER:-120}"
LEARNING_RATE="${LEARNING_RATE:-0.015}"
PRIOR_DENSITY_SPACE="${PRIOR_DENSITY_SPACE:-x}"
START_MODE="${START_MODE:-mixed}"
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
LOG_FILE="${LOG_DIR}/diffsky_map_sweep_${SLURM_JOB_ID:-manual}_${TASK_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[map-sweep] start: $(date --iso-8601=seconds)"
echo "[map-sweep] task: $TASK_ID"
echo "[map-sweep] config: $CONFIG"
echo "[map-sweep] checkpoint: $CHECKPOINT"
echo "[map-sweep] feature_stats: $FEATURE_STATS"
echo "[map-sweep] rowset: $ROWSET_FILE"
echo "[map-sweep] out_root: $OUT_ROOT"
echo "[map-sweep] weights: $WEIGHTS"
echo "[map-sweep] shard_count: $SHARD_COUNT"
echo "[map-sweep] batch_size: $BATCH_SIZE"
echo "[map-sweep] n_starts: $N_STARTS"
echo "[map-sweep] start_chunk_size: $START_CHUNK_SIZE"
echo "[map-sweep] maxiter: $MAXITER"
echo "[map-sweep] prior_density_space: $PRIOR_DENSITY_SPACE"

for path in "$CHECKPOINT" "$FEATURE_STATS" "$ROWSET_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "[map-sweep][missing] $path"
    exit 2
  fi
done

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

IFS=',' read -r -a WEIGHT_VALUES <<< "$WEIGHTS"
WEIGHT_COUNT="${#WEIGHT_VALUES[@]}"
TOTAL_TASKS=$((WEIGHT_COUNT * SHARD_COUNT))
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "[map-sweep] task $TASK_ID outside total task count $TOTAL_TASKS; exiting"
  exit 0
fi
WEIGHT_INDEX=$((TASK_ID / SHARD_COUNT))
SHARD_INDEX=$((TASK_ID % SHARD_COUNT))
PRIOR_WEIGHT="${WEIGHT_VALUES[$WEIGHT_INDEX]}"
WEIGHT_LABEL="$(python - <<PY
weight = float("$PRIOR_WEIGHT")
print(f"{weight:.4g}".replace("-", "m").replace(".", "p"))
PY
)"

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
    print(f"[map-sweep] shard {idx}: rows={len(chunk)} -> {path}")
PY

SHARD_FILE="$OUT_ROOT/shards/rowset_shard_$(printf '%03d' "$SHARD_INDEX").txt"
OUT_DIR="$OUT_ROOT/prior_weight_${WEIGHT_LABEL}/shard_$(printf '%03d' "$SHARD_INDEX")"

cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  diffsky-map-adam-prior
  --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS"
  --row-indices-file "$SHARD_FILE"
  --out "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --n-starts "$N_STARTS"
  --start-chunk-size "$START_CHUNK_SIZE"
  --maxiter "$MAXITER"
  --learning-rate "$LEARNING_RATE"
  --prior-weight "$PRIOR_WEIGHT"
  --prior-density-space "$PRIOR_DENSITY_SPACE"
  --start-mode "$START_MODE"
  --seed "$SEED"
)

echo "[map-sweep] command:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "[map-sweep] end: $(date --iso-8601=seconds)"
