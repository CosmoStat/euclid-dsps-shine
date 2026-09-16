#!/bin/bash
#SBATCH --job-name=feniks_asmc
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
TRAIN_ROOT="${TRAIN_ROOT:?TRAIN_ROOT is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
SMOKE="${SMOKE:-0}"
RESUME_STATE="${RESUME_STATE:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$MPLCONFIGDIR" "$TRAIN_ROOT"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

TRAIN_CATALOG="$CATALOG_DIR/train.parquet"
TRAIN_INDICES="$MANIFEST_ROOT/train_indices.npy"
VALIDATION_INDICES="$MANIFEST_ROOT/validation_indices.npy"
for path in "$CONFIG" "$TRAIN_CATALOG" "$MANIFEST_ROOT/manifest.json" \
  "$TRAIN_INDICES" "$VALIDATION_INDICES"; do
  test -s "$path" || { echo "[feniks-asmc][error] missing: $path" >&2; exit 2; }
done
test ! -e "$TRAIN_ROOT/DONE" || {
  echo "[feniks-asmc][error] already complete: $TRAIN_ROOT" >&2; exit 2;
}

python - <<'PY'
import jax
print("[feniks-asmc] backend:", jax.default_backend(), flush=True)
print("[feniks-asmc] devices:", jax.devices(), flush=True)
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("expected exactly four local H100 devices")
PY

ARGS=()
VALIDATION_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  ARGS+=(--smoke)
  VALIDATION_ARGS+=(--expect-smoke)
fi
if [[ -n "$RESUME_STATE" ]]; then
  test -s "$RESUME_STATE"
  test -s "${RESUME_STATE}.json"
  ARGS+=(--resume-state "$RESUME_STATE")
fi

echo "[feniks-asmc] smoke=$SMOKE out=$TRAIN_ROOT"
echo "[feniks-asmc] train_indices=$TRAIN_INDICES"
echo "[feniks-asmc] validation_indices=$VALIDATION_INDICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_feniks_adaptive_smc.py \
  --config "$CONFIG" \
  --catalog "$TRAIN_CATALOG" \
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --out "$TRAIN_ROOT" \
  --require-gpu \
  --expected-devices 4 \
  "${ARGS[@]}"

python scripts/validate_feniks_adaptive_smc_training.py \
  --train "$TRAIN_ROOT" \
  "${VALIDATION_ARGS[@]}"

touch "$TRAIN_ROOT/DONE"
echo "[feniks-asmc] complete -> $TRAIN_ROOT"
