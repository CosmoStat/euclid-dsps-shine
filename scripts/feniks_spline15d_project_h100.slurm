#!/bin/bash
#SBATCH --job-name=feniks_spline15d
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

if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/euclid_dsps" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
  else
    REPO_DIR="${WORK:?Set WORK or REPO_DIR}/euclid-dsps-shine"
  fi
fi
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/feniks_spline15d_postprocess.yaml}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-Data/diffsky/synthetic/feniks_260617_dsps_closure_18band}"
SPLINE_DATASET_DIR="${SPLINE_DATASET_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d}"
SMOKE="${SMOKE:-0}"
OVERWRITE="${OVERWRITE:-0}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}-${SLURM_JOB_ID:-manual}}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "[job] start=$(date --iso-8601=seconds) repo=$REPO_DIR"
echo "[job] source=$SOURCE_DATASET_DIR output=$SPLINE_DATASET_DIR smoke=$SMOKE"
for split in train validation test; do
  test -s "$SOURCE_DATASET_DIR/$split.parquet" || {
    echo "[job][error] missing $SOURCE_DATASET_DIR/$split.parquet"; exit 2;
  }
done
python - <<'PY'
import diffmah, diffstar, euclid_dsps, jax
print("[job] packages:", diffmah.__file__, diffstar.__file__, euclid_dsps.__file__)
print("[job] backend/devices:", jax.default_backend(), jax.devices())
PY

cmd=(python scripts/build_feniks_spline15d_dataset.py
  --config "$CONFIG" --source-dir "$SOURCE_DATASET_DIR" --out "$SPLINE_DATASET_DIR")
if [[ "$OVERWRITE" =~ ^(1|true|TRUE|yes|YES)$ ]]; then cmd+=(--overwrite); fi
if [[ "$SMOKE" == "1" ]]; then
  cmd+=(--limit "${LIMIT:-256}" --batch-size "${BATCH_SIZE:-256}")
fi
printf '[job] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"
echo "[job] done=$(date --iso-8601=seconds)"
