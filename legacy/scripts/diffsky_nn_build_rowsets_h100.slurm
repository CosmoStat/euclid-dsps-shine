#!/bin/bash
#SBATCH --job-name=diffsky_nn_rowsets
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
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
REF_TRAIN_RUN="${REF_TRAIN_RUN:-outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128}"
REF_INFER_RUN="${REF_INFER_RUN:-outputs/runs/diffsky_autoencoder_nokl_m5sys_z035_rand20k_e30_b128_infer}"
ROWSET_DIR="${ROWSET_DIR:-outputs/rowsets/diffsky_hltds_nokl_reference}"
BALANCED_SIZE="${BALANCED_SIZE:-20000}"
BALANCED_SEED="${BALANCED_SEED:-42}"
WORST_SIZES="${WORST_SIZES:-100 500 1000}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/diffsky_nn_rowsets_${SLURM_JOB_ID:-manual}.log}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$ROWSET_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[rowsets] start: $(date --iso-8601=seconds)"
echo "[rowsets] repo: $REPO_DIR"
echo "[rowsets] config: $CONFIG"
echo "[rowsets] train_run: $REF_TRAIN_RUN"
echo "[rowsets] infer_run: $REF_INFER_RUN"
echo "[rowsets] out: $ROWSET_DIR"
echo "[rowsets] balanced_size: $BALANCED_SIZE"
echo "[rowsets] balanced_seed: $BALANCED_SEED"
echo "[rowsets] worst_sizes: $WORST_SIZES"

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
    raise SystemExit(f"[rowsets][error] euclid_dsps imported from {pkg}, not {repo}")
print("[rowsets] euclid_dsps:", pkg)
print("[rowsets] jax_backend:", jax.default_backend())
print("[rowsets] jax_devices:", jax.devices())
PY
nvidia-smi || true
git status --short --branch

cmd=(
  python -m euclid_dsps.cli
  --config "$CONFIG"
  diffsky-build-reconstruction-rowsets
  --train-run "$REF_TRAIN_RUN"
  --infer-run "$REF_INFER_RUN"
  --out "$ROWSET_DIR"
  --balanced-size "$BALANCED_SIZE"
  --balanced-seed "$BALANCED_SEED"
)

for size in $WORST_SIZES; do
  cmd+=(--worst-size "$size")
done

echo "[rowsets] command:"
printf ' %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "[rowsets] manifest: $ROWSET_DIR/rowsets_manifest.json"
echo "[rowsets] balanced: $ROWSET_DIR/balanced20k.txt"
echo "[rowsets] end: $(date --iso-8601=seconds)"
