#!/bin/bash
#SBATCH --job-name=diffsky_map_prior_h100
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

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_joint_realnvp_unsup_stable_h100.yaml}"
TRAIN_RUN="${TRAIN_RUN:-outputs/runs/diffsky_realnvp_unsup_stable_30k_e20}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_RUN}/checkpoints/best.eqx}"
FEATURE_STATS="${FEATURE_STATS:-${TRAIN_RUN}/feature_stats.json}"

RUN_NAME="${RUN_NAME:-$(basename "$TRAIN_RUN")_map_limit5000}"
OUT_DIR="${OUT_DIR:-outputs/runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-manual}.log}"

LIMIT="${LIMIT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_STARTS="${N_STARTS:-4}"
MAXITER="${MAXITER:-160}"
LEARNING_RATE="${LEARNING_RATE:-0.015}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.05}"
SEED="${SEED:-42}"
SELECTION_MODE="${SELECTION_MODE:-random}"
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
echo "[job] checkpoint: $CHECKPOINT"
echo "[job] feature_stats: $FEATURE_STATS"
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] batch_size: $BATCH_SIZE"
echo "[job] n_starts: $N_STARTS"
echo "[job] maxiter: $MAXITER"
echo "[job] learning_rate: $LEARNING_RATE"
echo "[job] prior_weight: $PRIOR_WEIGHT"
echo "[job] selection_mode: $SELECTION_MODE"
echo "[job] stratified_strategy: $STRATIFIED_STRATEGY"
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
    raise SystemExit(
        f"[job][error] euclid_dsps imported from {pkg}, not from repo {repo}. "
        "Check PYTHONPATH or the conda environment."
    )
print("[job] euclid_dsps:", pkg)
print("[job] jax_backend:", jax.default_backend())
print("[job] jax_devices:", jax.devices())
PY
nvidia-smi || true

for path in "$CHECKPOINT" "$FEATURE_STATS"; do
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
  diffsky-map-adam-prior
  --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS"
  --out "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --n-starts "$N_STARTS"
  --maxiter "$MAXITER"
  --learning-rate "$LEARNING_RATE"
  --prior-weight "$PRIOR_WEIGHT"
  --seed "$SEED"
  --selection-mode "$SELECTION_MODE"
  --stratified-strategy "$STRATIFIED_STRATEGY"
  --selection-seed "$SELECTION_SEED"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[job] command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "[job] map summary: $OUT_DIR/map_summary.json"
echo "[job] end: $(date --iso-8601=seconds)"
