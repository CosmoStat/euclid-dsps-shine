#!/bin/bash
#SBATCH --job-name=feniks_sc_e160_hold
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
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
VARIANT="${VARIANTS[$((TASK / 4))]}"
SHARD="$((TASK % 4))"
CONFIG="configs/experiments/feniks_sc_drws_r29_current_production.yaml"
RECEIPT="$EVAL_ROOT/CHECKPOINT_FROZEN.json"
INDICES="$EVAL_ROOT/manifests/heldout_shard_${SHARD}.npy"
OUT="$EVAL_ROOT/heldout/$VARIANT/shard_${SHARD}"
SUPPORT="$OUT/exact_gaussian_k1024"
IMPORTANCE="$OUT/ordinary_iw"
PREDICTIVE="$OUT/exact_gaussian_predictive"

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
  test -s "$path" || { echo "[epoch160-heldout][error] missing: $path" >&2; exit 2; }
done
test ! -e "$OUT/DONE" || { echo "already complete: $OUT"; exit 0; }

OBJECTS=$(python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "$INDICES")
COMMON=(--dataset "$CATALOG_DIR/test.parquet" --checkpoint "$CHECKPOINT"
  --feature-stats "$FEATURE_STATS" --row-indices-file "$INDICES"
  --selection-mode sequential --limit "$OBJECTS" --batch-size 32 --jax-batch-size 32
  --prior-samples 64 --decoder-sample-chunk-size 1 --prior-predictive-batch-size 64
  --shard-outputs --no-residual-samples)

python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
  "${COMMON[@]}" --posterior-samples 1024 --no-posterior-predictive \
  --seed "$((261600 + TASK))" --out "$SUPPORT"
python -m euclid_dsps.cli --config "$CONFIG" amortized-finalize-inference \
  --out "$SUPPORT" --limit "$OBJECTS"
python scripts/importance_correct_posterior.py --posterior "$SUPPORT" \
  --out "$IMPORTANCE" --resample-count 128 --seed "$((261700 + TASK))" \
  --min-median-ess-fraction 0.05 --max-fraction-pareto-k-gt-0p7 0.20
python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
  "${COMMON[@]}" --posterior-samples 64 --seed "$((261800 + TASK))" \
  --out "$PREDICTIVE"
python -m euclid_dsps.cli --config "$CONFIG" amortized-finalize-inference \
  --out "$PREDICTIVE" --limit "$OBJECTS"
touch "$OUT/DONE"
echo "[epoch160-heldout] complete variant=$VARIANT shard=$SHARD objects=$OBJECTS"
