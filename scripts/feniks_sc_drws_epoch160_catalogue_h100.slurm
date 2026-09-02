#!/bin/bash
#SBATCH --job-name=feniks_sc_e160_cat
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:?CATALOG_DIR is required}"
EVAL_ROOT="${EVAL_ROOT:?EVAL_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
VARIANTS=(raw ema)
VARIANT="${VARIANTS[$((TASK / 8))]}"
SHARD="$((TASK % 8))"
CONFIG="configs/experiments/feniks_sc_drws_r29_current_production.yaml"
RECEIPT="$EVAL_ROOT/CHECKPOINT_FROZEN.json"
INDICES="$EVAL_ROOT/manifests/catalogue_shard_${SHARD}.npy"
OUT="$EVAL_ROOT/catalogue/$VARIANT/shard_${SHARD}"
INFERENCE="$OUT/exact_gaussian_k256"
IMPORTANCE="$OUT/ordinary_iw"

readarray -t VALUES < <(python - "$RECEIPT" "$VARIANT" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
if receipt.get("status") != "FROZEN" or int(receipt.get("epoch", -1)) != 160:
    raise SystemExit("invalid epoch-160 freeze receipt")
print(receipt["components"][f"{sys.argv[2]}_model"]["path"])
print(receipt["feature_stats"]["path"])
PY
)
CHECKPOINT="${VALUES[0]}"
FEATURE_STATS="${VALUES[1]}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1 EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export MPLBACKEND=Agg MPLCONFIGDIR="/tmp/mpl-${USER}-${SLURM_ARRAY_JOB_ID}-${TASK}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" PIP_NO_INDEX=1
export EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"

cd "$REPO_DIR"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$OUT" "$MPLCONFIGDIR" "$CACHE_ROOT/jax"
for path in "$CONFIG" "$RECEIPT" "$INDICES" "$CHECKPOINT" \
  "$FEATURE_STATS" "$CATALOG_DIR/test.parquet"; do
  test -s "$path" || { echo "[epoch160-catalogue][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/DONE" || { echo "already complete: $OUT"; exit 0; }

OBJECTS=$(python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "$INDICES")
python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
  --dataset "$CATALOG_DIR/test.parquet" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --row-indices-file "$INDICES" \
  --selection-mode sequential --limit "$OBJECTS" --batch-size 64 --jax-batch-size 64 \
  --posterior-samples 256 --prior-samples 64 --decoder-sample-chunk-size 2 \
  --prior-predictive-batch-size 64 --shard-outputs --no-posterior-predictive \
  --no-residual-samples --seed "$((261900 + TASK))" --out "$INFERENCE"
python -m euclid_dsps.cli --config "$CONFIG" amortized-finalize-inference \
  --out "$INFERENCE" --limit "$OBJECTS"
python scripts/importance_correct_posterior.py --posterior "$INFERENCE" \
  --out "$IMPORTANCE" --resample-count 32 --seed "$((262000 + TASK))" \
  --min-median-ess-fraction 0.01 --max-fraction-pareto-k-gt-0p7 0.50
touch "$OUT/DONE"
echo "[epoch160-catalogue] complete variant=$VARIANT shard=$SHARD objects=$OBJECTS"
