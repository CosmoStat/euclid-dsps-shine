#!/bin/bash
#SBATCH --job-name=cosmos15_time
#SBATCH --account=jrx@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=outputs/logs/%x-%A_%a.out
#SBATCH --error=outputs/logs/%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MINICONDA_PATH="${MINICONDA_PATH:-${WORK:?Set WORK or MINICONDA_PATH}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-shine}"
ROOT_BASE="${ROOT_BASE:?Set ROOT_BASE to the completed native15d array root}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_BASE/publication_checks/timing_h100_v1}"
DATASET="${DATASET:-Data/cosmos2020/prepared/farmer_a24_full.parquet}"
EVAL_INDICES="${EVAL_INDICES:-Data/cosmos2020/prepared/evaluation_indices_n5000.npy}"
LIMIT="${LIMIT:-128}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-128}"
DECODER_SAMPLE_CHUNK_SIZE="${DECODER_SAMPLE_CHUNK_SIZE:-1}"
REPEATS="${REPEATS:-5}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK" in
  0)
    VARIANT="bands26"
    CONFIG="configs/experiments/popcosmos_native15d_rws.yaml"
    ;;
  1)
    VARIANT="bands24_no_irac"
    CONFIG="configs/experiments/popcosmos_native15d_rws_24band.yaml"
    ;;
  *)
    echo "[cosmos-timing][error] array task must be 0 or 1"
    exit 2
    ;;
esac

RUN="$ROOT_BASE/$VARIANT/full_cont120"
CHECKPOINT="$RUN/train/checkpoints/best.eqx"
FEATURE_STATS="$RUN/train/feature_stats.json"
OUT="$OUTPUT_ROOT/$VARIANT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PIP_NO_INDEX=1

cd "$REPO_DIR"
mkdir -p outputs/logs "$(dirname "$OUT")"
module purge
module load arch/h100
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

test -e "$RUN/DONE"
test -s "$CHECKPOINT"
test -s "${CHECKPOINT}.json"
test -s "$FEATURE_STATS"
test -s "$DATASET"
test -s "$EVAL_INDICES"
test ! -e "$OUT"

python scripts/benchmark_popcosmos_native15d_inference.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --feature-stats "$FEATURE_STATS" \
  --row-indices "$EVAL_INDICES" \
  --out "$OUT" \
  --limit "$LIMIT" \
  --posterior-samples "$POSTERIOR_SAMPLES" \
  --decoder-sample-chunk-size "$DECODER_SAMPLE_CHUNK_SIZE" \
  --repeats "$REPEATS" \
  --require-gpu

test -e "$OUT/DONE"
test -s "$OUT/timing_summary.json"
test -s "$OUT/timing_repeats.csv"
echo "[cosmos-timing] verified -> $OUT"
