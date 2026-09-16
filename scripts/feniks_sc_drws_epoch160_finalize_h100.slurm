#!/bin/bash
#SBATCH --job-name=feniks_sc_e160_final
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:?CATALOG_DIR is required}"
MANIFEST_ROOT="${MANIFEST_ROOT:?MANIFEST_ROOT is required}"
EVAL_ROOT="${EVAL_ROOT:?EVAL_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
CONFIG="configs/experiments/feniks_sc_drws_r29_current_production.yaml"
TRUTH_CONFIG="configs/experiments/feniks_sc_drws_r29_truth_closure.yaml"
RECEIPT="$EVAL_ROOT/CHECKPOINT_FROZEN.json"

readarray -t VALUES < <(python - "$RECEIPT" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
if receipt.get("status") != "FROZEN" or int(receipt.get("epoch", -1)) != 160:
    raise SystemExit("invalid epoch-160 freeze receipt")
print(receipt["components"]["raw_model"]["path"])
PY
)
CHECKPOINT="${VALUES[0]}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
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
mkdir -p "$EVAL_ROOT/population" "$MPLCONFIGDIR" "$CACHE_ROOT/jax"
for path in "$CONFIG" "$TRUTH_CONFIG" "$RECEIPT" "$CHECKPOINT" \
  "$MANIFEST_ROOT/manifest.json" "$MANIFEST_ROOT/train_indices.npy" \
  "$MANIFEST_ROOT/validation_indices.npy" "$MANIFEST_ROOT/full_test_indices.npy" \
  "$MANIFEST_ROOT/final_validation_indices.npy" "$CATALOG_DIR/train.parquet" \
  "$CATALOG_DIR/test.parquet"; do
  test -s "$path" || { echo "[epoch160-finalize][error] missing: $path" >&2; exit 2; }
done

if [[ ! -s "$EVAL_ROOT/population/prior/report_receipt.json" ]]; then
  if [[ -e "$EVAL_ROOT/population/prior" ]]; then
    QUARANTINE="$EVAL_ROOT/population/prior.incomplete-${SLURM_JOB_ID}"
    test ! -e "$QUARANTINE" || {
      echo "[epoch160-finalize][error] quarantine already exists: $QUARANTINE" >&2
      exit 2
    }
    echo "[epoch160-finalize] quarantining receipt-less prior report -> $QUARANTINE"
    mv "$EVAL_ROOT/population/prior" "$QUARANTINE"
  fi
  python scripts/report_feniks_sc_drws.py \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --train-indices-file "$MANIFEST_ROOT/train_indices.npy" \
    --validation-indices-file "$MANIFEST_ROOT/validation_indices.npy" \
    --out "$EVAL_ROOT/population/prior" --prior-samples 16384 \
    --selected-resamples 16384 --decoder-batch-size 64 --seed 261603
fi

python scripts/finalize_feniks_sc_drws_epoch160.py \
  --root "$EVAL_ROOT" --training-config "$CONFIG" \
  --truth-config "$TRUTH_CONFIG" --test-catalog "$CATALOG_DIR/test.parquet" \
  --manifest-root "$MANIFEST_ROOT" \
  --samples-per-object 128 --num-regions 100 --num-bootstrap 1000 \
  --aggregate-samples-per-object 32

test -s "$EVAL_ROOT/EPOCH160_EVALUATION_COMPLETE.json"
echo "[epoch160-finalize] complete -> $EVAL_ROOT"
