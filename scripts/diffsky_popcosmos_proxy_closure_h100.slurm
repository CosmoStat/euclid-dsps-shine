#!/bin/bash
#SBATCH --job-name=diffsky_proxy_closure_h100
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${WORK:?WORK is not set}/euclid-dsps-shine}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?WORK is not set}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"

CONFIG="${CONFIG:-configs/experiments/diffsky_hltds_popcosmos_proxy_truth_closure_h100.yaml}"
DATASET="${DATASET:-Data/diffsky/processed/hltds_cosmos_260215_04_14_2026_photometry_truth_noerr.parquet}"
RUN_NAME="${RUN_NAME:-diffsky_popcosmos_proxy_truth_closure_limit1000}"
OUT_DIR="${OUT_DIR:-outputs/runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_${SLURM_JOB_ID:-manual}.log}"

LIMIT="${LIMIT:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"

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
echo "[job] limit: ${LIMIT:-<full dataset>}"
echo "[job] batch_size: $BATCH_SIZE"

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

cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  diffsky-popcosmos-proxy-closure
  --dataset "$DATASET"
  --out "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

echo "[job] command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "[job] report: $OUT_DIR/popcosmos_proxy_closure_report.md"
echo "[job] summary: $OUT_DIR/popcosmos_proxy_closure_summary.json"
echo "[job] end: $(date --iso-8601=seconds)"
