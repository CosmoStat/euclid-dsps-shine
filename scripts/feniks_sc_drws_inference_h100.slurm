#!/bin/bash
#SBATCH --job-name=feniks_sc_drws_infer
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-3%4

set -Eeuo pipefail
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
CATALOG_DIR="${CATALOG_DIR:?CATALOG_DIR is required}"
RECOVERY_ROOT="${RECOVERY_ROOT:?RECOVERY_ROOT is required}"
INFERENCE_ROOT="${INFERENCE_ROOT:?INFERENCE_ROOT is required}"
CACHE_ROOT="${CACHE_ROOT:?CACHE_ROOT is required}"
TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PASS="$RECOVERY_ROOT/FULL_TRAIN_PASS.json"
SELECTED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$PASS")
CHECKPOINT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$PASS")
FEATURE_STATS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_feature_stats"])' "$PASS")
case "$SELECTED" in
  historical_4x128) CONFIG=configs/experiments/feniks_sc_drws_r29_historical.yaml ;;
  current_residual_6x256) CONFIG=configs/experiments/feniks_sc_drws_r29_current.yaml ;;
  *) exit 2 ;;
esac
INDICES="$INFERENCE_ROOT/manifests/shard_${TASK}.npy"
OUT="$INFERENCE_ROOT/shard_${TASK}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-48}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export EUCLID_DSPS_JAX_PLATFORMS=cuda
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export EUCLID_DSPS_REQUIRE_GPU=1
export EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA
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
mkdir -p "$MPLCONFIGDIR" "$CACHE_ROOT/jax"
for path in "$PASS" "$CONFIG" "$CHECKPOINT" "$FEATURE_STATS" "$INDICES" \
  "$CATALOG_DIR/train.parquet"; do test -s "$path" || exit 2; done
OBJECTS=$(python -c 'import numpy as np,sys; print(len(np.load(sys.argv[1])))' "$INDICES")
python -m euclid_dsps.cli --config "$CONFIG" amortized-infer-diffsky \
  --dataset "$CATALOG_DIR/train.parquet" --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" --row-indices-file "$INDICES" \
  --selection-mode sequential --limit "$OBJECTS" --batch-size 128 \
  --jax-batch-size 64 --posterior-samples 256 --prior-samples 512 \
  --decoder-sample-chunk-size 2 --prior-predictive-batch-size 128 \
  --shard-outputs --no-residual-samples --seed "$((270827 + TASK))" --out "$OUT"
python -m euclid_dsps.cli --config "$CONFIG" amortized-finalize-inference \
  --out "$OUT" --limit "$OBJECTS"
touch "$OUT/DONE"
