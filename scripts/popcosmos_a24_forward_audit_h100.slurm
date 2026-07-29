#!/bin/bash
#SBATCH --job-name=a24_dsps
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/popcosmos_a24_rws_joint.yaml}"
DATASET="${DATASET:-Data/cosmos2020/prepared/farmer_a24_n5000.parquet}"
RUN_DIR="${RUN_DIR:-outputs/runs/popcosmos_a24_rws_v2/n5k}"
MATCHED="${MATCHED:-$RUN_DIR/a24_comparison/matched_posteriors.parquet}"
CALIBRATION="${CALIBRATION:-$RUN_DIR/train/per_band_flux_calibration.csv}"
OUT="${OUT:-$RUN_DIR/a24_dsps_forward}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID:-local}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
test -s "$DATASET"
test -s "$MATCHED"
test -s "$CALIBRATION"
test ! -e "$OUT" || {
  echo "[a24-dsps-forward][error] refusing to overwrite $OUT"
  exit 2
}

module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - <<'PY'
import jax
print("[a24-dsps-forward] backend:", jax.default_backend())
print("[a24-dsps-forward] devices:", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit("[a24-dsps-forward][error] JAX did not select the H100")
PY

python scripts/audit_popcosmos_a24_dsps_forward.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --matched "$MATCHED" \
  --calibration "$CALIBRATION" \
  --out "$OUT" \
  --runtime gpu \
  --batch-size 64

test -s "$OUT/a24_dsps_forward_summary.json"
touch "$OUT/DONE"
