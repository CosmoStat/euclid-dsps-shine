#!/bin/bash
#SBATCH --job-name=feniks_sleepnpe
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
CONFIG="${CONFIG:-configs/experiments/feniks_selfsup_parentprior_sleepnpe_defensivewake_selection_r25.yaml}"
CATALOG_DIR="${CATALOG_DIR:-Data/diffsky/synthetic/feniks_260617_spline15d_grouped_jaxcosmo_v1/amortized}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/runs/feniks_spline15d_jaxcosmo_prior_v1/checkpoints/best.eqx}"
TRAIN_ROOT="${TRAIN_ROOT:?TRAIN_ROOT is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
SMOKE="${SMOKE:-0}"

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
for path in "$CONFIG" "$TRAIN_CATALOG" "$REFERENCE_CHECKPOINT" \
  "${REFERENCE_CHECKPOINT}.json" "$MANIFEST_ROOT/manifest.json" \
  "$TRAIN_INDICES" "$VALIDATION_INDICES"; do
  test -s "$path" || { echo "[feniks-sleepnpe][error] missing: $path" >&2; exit 2; }
done
test ! -e "$TRAIN_ROOT/DONE" || {
  echo "[feniks-sleepnpe][error] already complete: $TRAIN_ROOT" >&2; exit 2;
}
test ! -e "$TRAIN_ROOT/training_summary.json" || {
  echo "[feniks-sleepnpe][error] partial output exists: $TRAIN_ROOT" >&2; exit 2;
}

python - <<'PY'
import jax
print("[feniks-sleepnpe] backend:", jax.default_backend(), flush=True)
print("[feniks-sleepnpe] devices:", jax.devices(), flush=True)
if jax.default_backend() != "gpu" or len(jax.local_devices()) != 4:
    raise SystemExit("expected exactly four local H100 devices")
PY

EPOCHS=80
BATCH_SIZE=1024
JAX_BATCH_SIZE=256
VALIDATION_EVERY=4
EXTRA_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=2
  BATCH_SIZE=256
  JAX_BATCH_SIZE=128
  VALIDATION_EVERY=1
  EXTRA_ARGS+=(
    --wake-start-encoder-epoch 2
    --wake-every-encoder-epochs 1
    --best-checkpoint-min-epoch 1
  )
fi

echo "[feniks-sleepnpe] smoke=$SMOKE epochs=$EPOCHS out=$TRAIN_ROOT"
echo "[feniks-sleepnpe] train_indices=$TRAIN_INDICES"
echo "[feniks-sleepnpe] validation_indices=$VALIDATION_INDICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m euclid_dsps.cli --config "$CONFIG" amortized-train-diffsky \
  --runtime gpu \
  --dataset "$TRAIN_CATALOG" \
  --prior-checkpoint "$REFERENCE_CHECKPOINT" \
  --train-indices-file "$TRAIN_INDICES" \
  --validation-indices-file "$VALIDATION_INDICES" \
  --out "$TRAIN_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --jax-batch-size "$JAX_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --n-samples 1 \
  --seed 260821 \
  --data-parallel pmap \
  --validation-every "$VALIDATION_EVERY" \
  --no-progress \
  "${EXTRA_ARGS[@]}"

VALIDATION_ARGS=()
[[ "$SMOKE" == "1" ]] && VALIDATION_ARGS+=(--smoke)
python scripts/validate_feniks_parentprior_training.py \
  --train "$TRAIN_ROOT" \
  --manifest "$MANIFEST_ROOT/manifest.json" \
  "${VALIDATION_ARGS[@]}"

touch "$TRAIN_ROOT/DONE"
echo "[feniks-sleepnpe] complete -> $TRAIN_ROOT"
