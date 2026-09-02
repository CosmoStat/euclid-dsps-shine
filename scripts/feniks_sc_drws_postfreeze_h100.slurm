#!/bin/bash
#SBATCH --job-name=feniks_sc_postfreeze
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:?CATALOG_DIR is required}"
RECOVERY_ROOT="${RECOVERY_ROOT:?RECOVERY_ROOT is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
CLOSURE_ROOT="${CLOSURE_ROOT:?CLOSURE_ROOT is required}"
AUTHORIZATION="$RECOVERY_ROOT/FULL_LAUNCH_AUTHORIZATION.json"
if [[ ! -s "$AUTHORIZATION" ]]; then
  AUTHORIZATION="$RECOVERY_ROOT/RWS_RECOVERY_PASS.json"
fi
SELECTED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$AUTHORIZATION")
case "$SELECTED" in
  historical_4x128) CONFIG=configs/experiments/feniks_sc_drws_r29_historical_production.yaml ;;
  current_residual_6x256) CONFIG=configs/experiments/feniks_sc_drws_r29_current_production.yaml ;;
  *) echo "[postfreeze][error] invalid candidate: $SELECTED" >&2; exit 2 ;;
esac
TRUTH_CONFIG=configs/experiments/feniks_sc_drws_r29_truth_closure.yaml
RUN_ROOT="$RECOVERY_ROOT/full/$SELECTED/seed_260826"
TRAIN="$RUN_ROOT/train"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1
export EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_JOB_ID}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$MPLCONFIGDIR" "$CACHE_ROOT/jax" "$CLOSURE_ROOT"

for path in "$AUTHORIZATION" "$CONFIG" "$TRUTH_CONFIG" \
  "$MANIFEST_ROOT/manifest.json" "$MANIFEST_ROOT/full_train_indices.npy" \
  "$MANIFEST_ROOT/train_indices.npy" "$MANIFEST_ROOT/validation_indices.npy" \
  "$MANIFEST_ROOT/final_validation_indices.npy" \
  "$CATALOG_DIR/train.parquet" "$CATALOG_DIR/test.parquet" \
  "$TRAIN/training_receipt.json" "$RUN_ROOT/full_summary.json"; do
  test -s "$path" || { echo "[postfreeze][error] missing: $path" >&2; exit 2; }
done

if [[ -s "$CLOSURE_ROOT/POSTFREEZE_COMPLETE.json" ]]; then
  echo "[postfreeze] already complete: $CLOSURE_ROOT"
  exit 0
fi

for VARIANT in raw ema; do
  VARIANT_OFFSET=0
  [[ "$VARIANT" == "ema" ]] && VARIANT_OFFSET=1
  PRIOR_OUT="$CLOSURE_ROOT/prior_${VARIANT}"
  if [[ -s "$PRIOR_OUT/report_receipt.json" ]]; then
    echo "[postfreeze] reusing ${VARIANT} prior report"
    continue
  fi
  if [[ -e "$PRIOR_OUT" ]]; then
    echo "[postfreeze][error] incomplete immutable prior report: $PRIOR_OUT" >&2
    exit 2
  fi
  python scripts/report_feniks_sc_drws.py \
    --config "$CONFIG" \
    --checkpoint "$TRAIN/checkpoints/final/${VARIANT}_model.eqx" \
    --train-indices-file "$MANIFEST_ROOT/train_indices.npy" \
    --validation-indices-file "$MANIFEST_ROOT/validation_indices.npy" \
    --out "$PRIOR_OUT" --prior-samples 8192 --selected-resamples 8192 \
    --decoder-batch-size 64 --seed "$((260902 + VARIANT_OFFSET))"
done

python scripts/evaluate_feniks_sc_drws_postfreeze.py \
  --training-config "$CONFIG" --truth-config "$TRUTH_CONFIG" \
  --test-catalog "$CATALOG_DIR/test.parquet" \
  --manifest "$MANIFEST_ROOT/manifest.json" --run-root "$RUN_ROOT" \
  --out "$CLOSURE_ROOT" --samples-per-object 128 \
  --num-regions 100 --num-bootstrap 1000

test -s "$CLOSURE_ROOT/POSTFREEZE_COMPLETE.json"
echo "[postfreeze] complete -> $CLOSURE_ROOT"
